# backend/utils/inference/belief_manager/journey/shm_service.py
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import asc

from backend.utils.inference.belief_manager.journey.shm_models import SHMEpisode
from backend.utils.crm_management.target_account_manager import (
    TargetAccount as TargetAccountORM,
)
from backend.utils.segment_utils import normalize_account_meta as normalize_meta_dict

from backend.super_models.shm.episode import (
    ShmEpisode,
    ShmEpisodeStep,
    StepBucket,
    EpisodeOutcome,
)

def _json_clean(value: Any) -> Any:
    """
    Coerce value into JSON-serializable primitives. Mirrors the serializer
    used upstream but kept local to avoid circular imports.
    """
    def _default(o: Any):
        if isinstance(o, datetime):
            return o.isoformat()
        try:
            from decimal import Decimal
            if isinstance(o, Decimal):
                return float(o)
        except Exception:
            pass
        try:
            import numpy as _np  # type: ignore
            if isinstance(o, _np.generic):
                return o.item()
        except Exception:
            pass
        if hasattr(o, "model_dump") and callable(getattr(o, "model_dump")):
            return o.model_dump()
        if hasattr(o, "dict") and callable(getattr(o, "dict")):
            return o.dict()
        try:
            import dataclasses
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
        except Exception:
            pass
        if isinstance(o, (set, tuple)):
            return list(o)
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="ignore")
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    import json

    try:
        return json.loads(json.dumps(value, default=_default))
    except TypeError:
        # best effort fallback: convert to string
        return json.loads(json.dumps(value, default=lambda obj: str(obj)))


META_BLOCKLIST = {
    "account_name",
    "accountid",
    "target_account_id",
    "deal_status",
    "status",
    "id",
    "_normalized_keys",
}


def _sanitize_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in meta.items():
        if value in (None, "", [], {}, ()):
            continue
        norm_key = str(key).strip()
        if not norm_key or norm_key.lower() in META_BLOCKLIST:
            continue
        cleaned[norm_key] = value
    return {
        key: value
        for key, value in cleaned.items()
    }


def resolve_episode_account_meta(
    db: Session,
    *,
    product_id: str,
    account_id: str,
    provided_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Merge any account metadata carried on the thesis with the authoritative
    TargetAccount record so that every SHM episode has segmentable meta.
    Returns (meta_dict, source_flag) where source_flag ∈ {"episode", "target_account", "none"}.
    """
    merged: Dict[str, Any] = {}
    meta_source = "none"

    thesis_meta = _sanitize_meta(provided_meta)
    if thesis_meta:
        merged.update(thesis_meta)
        meta_source = "episode"

    fallback_meta: Dict[str, Any] = {}
    try:
        account_row = (
            db.query(TargetAccountORM)
            .filter(
                TargetAccountORM.id == account_id,
                TargetAccountORM.product_id == product_id,
            )
            .first()
        )
    except Exception:
        account_row = None

    if account_row:
        fallback_meta = _sanitize_meta(
            {
                "account_name": account_row.account_name,
                "industry": account_row.industry,
                "revenue_range": account_row.revenue_range,
                "employee_range": account_row.employee_range,
                "geography": account_row.geography,
                "funding_stage": account_row.funding_stage,
            }
        )

    if fallback_meta:
        merged = {**fallback_meta, **merged} if merged else dict(fallback_meta)
        if meta_source != "episode":
            meta_source = "target_account"

    normalized_tokens = normalize_meta_dict(
        {
            key: value
            for key, value in merged.items()
            if key
            and str(key).strip().lower() not in {"account_name", "deal_status", "status", "_normalized_keys"}
        }
    )
    if normalized_tokens:
        merged["_normalized_keys"] = normalized_tokens

    return merged, meta_source

#--- Helpers----
def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            # allow "2025-11-21T10:15:00Z" and "2025-11-21T10:15:00"
            return datetime.fromisoformat(value.replace("Z", ""))
        except Exception:
            return datetime.utcnow()
    return datetime.utcnow()


# ---- write ----
def write_meta_episode_from_thesis(
    db: Session,
    *,
    product_id: str,
    account_id: str,
    thesis: Dict[str, Any],
) -> Optional[ShmEpisode]:
    """
    Persist a ShmEpisode + ShmEpisodeStep rows based on the belief thesis
    journey steps.

    This is what summarize_global_insights() reads when it does:

        db.query(ShmEpisode).filter(ShmEpisode.product_id == product_id)

    We keep it deliberately lossy: just enough structure for analytics and
    later replay.
    """
    journey = thesis.get("journey") or {}
    steps: List[Dict[str, Any]] = journey.get("steps") or []
    candidate_personas = journey.get("candidate_personas") or {}
    if not steps:
        return None

    # --- episode meta ---
    # Try to pick reasonable timestamps from the first / last step; fall back to now.
    first_ts = _parse_ts(steps[0].get("timestamp") or steps[0].get("t_timestamp"))
    last_ts = _parse_ts(steps[-1].get("timestamp") or steps[-1].get("t_timestamp"))

    # Model version if you have it in thesis; otherwise leave None
    model_version = (
        thesis.get("journey_model_version")
        or thesis.get("model_version")
        or None
    )

    provided_account_meta = thesis.get("account_meta") or thesis.get("account") or {}
    resolved_meta, meta_source = resolve_episode_account_meta(
        db,
        product_id=product_id,
        account_id=account_id,
        provided_meta=provided_account_meta if isinstance(provided_account_meta, dict) else {},
    )

    episode = ShmEpisode(
        product_id=product_id,
        account_id=account_id,
        journey_model_version=model_version,
        outcome=EpisodeOutcome.unknown,  # you can overwrite later from CRM
        is_censored=False,
        started_at=first_ts,
        ended_at=last_ts,
        account_meta=_json_clean(resolved_meta),
        metrics={**(journey.get("metrics") or {})},
        learning_summary=thesis.get("learning_summary"),
        num_steps=len(steps),
        candidate_personas=candidate_personas or None,
    )
    if candidate_personas:
        episode.metrics = {
            **(episode.metrics or {}),
            "candidate_personas": candidate_personas,
        }
    if meta_source and meta_source != "none":
        episode.metrics = {
            **(episode.metrics or {}),
            "account_meta_source": meta_source,
        }

    db.add(episode)
    db.flush()  # get episode.id

    # --- episode steps ---
    for idx, step in enumerate(steps):
        # t_index: use explicit t if present; else index
        t_index = step.get("t")
        if t_index is None:
            t_index = idx

        # bucket: map string to StepBucket enum if possible
        bucket_raw = (
            step.get("bucket")
            or step.get("step_bucket")
            or step.get("path_class")
        )
        bucket_enum: Optional[StepBucket] = None
        if bucket_raw:
            try:
                bucket_enum = StepBucket(bucket_raw)
            except ValueError:
                # unknown bucket label; leave as None
                bucket_enum = None

        # persona IDs
        observed_persona_id = (
            step.get("observed_persona_id")
            or step.get("persona_id")
            or step.get("best_persona_id")
            or step.get("observed_next")
        )
        predicted_top_persona_id = (
            step.get("predicted_top_persona_id")
            or step.get("expected_persona_id")
            or None
        )

        raw_topk = step.get("predicted_topK")

        # copies of predicted paths / metrics, if present
        predicted_topK = _json_clean(raw_topk)
        walk_paths = _json_clean(step.get("walk_paths"))
        metrics = step.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {"raw": metrics}
        if step.get("error"):
            metrics = {**metrics, "error": _json_clean(step.get("error"))}
        if step.get("local_adjustment"):
            metrics = {
                **metrics,
                "local_adjustment": _json_clean(step.get("local_adjustment")),
            }
        metrics = _json_clean(metrics)
        if isinstance(metrics, dict):
            metrics = dict(metrics)
        else:
            metrics = {}

        engagement_meta = step.get("engagement_meta") or step.get("engagement") or {}
        if step.get("asset_id"):
            engagement_meta = dict(engagement_meta or {})
            engagement_meta.setdefault("asset_id", step.get("asset_id"))
        if step.get("channel"):
            engagement_meta = dict(engagement_meta or {})
            engagement_meta.setdefault("channel", step.get("channel"))
        if engagement_meta:
            metrics.setdefault("engagement", engagement_meta)
            if engagement_meta.get("asset_id") is not None:
                metrics.setdefault("asset_id", engagement_meta.get("asset_id"))
            if engagement_meta.get("channel") is not None:
                metrics.setdefault("channel", engagement_meta.get("channel"))
        account_meta_payload = step.get("account_meta")
        if account_meta_payload:
            metrics.setdefault("account_meta", account_meta_payload)

        if not predicted_top_persona_id:
            try:
                if isinstance(raw_topk, list) and raw_topk:
                    head = raw_topk[0]
                    if isinstance(head, dict):
                        predicted_top_persona_id = (
                            head.get("persona")
                            or head.get("id")
                            or head.get("persona_id")
                        )
                    else:
                        predicted_top_persona_id = str(head)
            except Exception:
                predicted_top_persona_id = None

        hit_at_1 = step.get("hit_at_1")
        hit_at_3 = step.get("hit_at_3")

        try:
            db.add(
                ShmEpisodeStep(
                    episode_id=episode.id,
                    t_index=int(t_index),
                    bucket=bucket_enum,
                    observed_persona_id=observed_persona_id,
                    predicted_top_persona_id=predicted_top_persona_id,
                    predicted_topK=predicted_topK,
                    walk_paths=walk_paths,
                    metrics=metrics,
                    hit_at_1=hit_at_1,
                    hit_at_3=hit_at_3,
                )
            )
        except Exception as exc:
            print(
                "[shm_service] Failed to persist episode step",
                idx,
                "for",
                account_id,
                ":",
                repr(exc),
            )
            print(
                "[shm_service] step payload:",
                {
                    "predicted_topK": predicted_topK,
                    "walk_paths": walk_paths,
                    "metrics": metrics,
                },
            )
            raise

    return episode


def append_episode_steps(
    db: Session,
    *,
    product_id: str,
    account_id: str,
    episode_id: str,
    steps: List[Dict[str, Any]],
    account_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a batch of steps to an SHM episode.

    Each step dict should minimally contain:
      - step_index (int)
      - timestamp (datetime or iso str)
      - persona_id (str)
      - belief_state (str)
      - engagement_type (str)

    Optional:
      - belief_score (int)
      - channel (str)
      - asset_id (str)
      - effect_bucket (str)
      - meta (dict)
    """

    objs: List[SHMEpisode] = []
    for s in steps:
        ts = s.get("timestamp")
        if isinstance(ts, str):
            # naive parse, you can swap in dateutil if needed
            ts = datetime.fromisoformat(ts.replace("Z", ""))

        obj = SHMEpisode(
            product_id=product_id,
            account_id=account_id,
            episode_id=episode_id,
            step_index=int(s["step_index"]),
            timestamp=ts,
            persona_id=s["persona_id"],
            belief_state=s["belief_state"],
            belief_score=s.get("belief_score"),
            engagement_type=s["engagement_type"],
            channel=s.get("channel"),
            asset_id=s.get("asset_id"),
            effect_bucket=s.get("effect_bucket"),
            meta=s.get("meta") or {},
            account_meta=account_meta or s.get("account_meta") or {},
        )
        objs.append(obj)

    db.add_all(objs)


def log_single_step(
    db: Session,
    *,
    product_id: str,
    account_id: str,
    episode_id: str,
    step_index: int,
    persona_id: str,
    belief_state: str,
    engagement_type: str,
    timestamp=None,
    channel: Optional[str] = None,
    asset_id: Optional[str] = None,
    effect_bucket: Optional[str] = None,
    belief_score: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
    account_meta: Optional[Dict[str, Any]] = None,
) -> SHMEpisode:
    """
    Convenience helper when you’re writing one step at a time.
    """
    if timestamp is None:
        timestamp = datetime.utcnow()

    obj = SHMEpisode(
        product_id=product_id,
        account_id=account_id,
        episode_id=episode_id,
        step_index=step_index,
        timestamp=timestamp,
        persona_id=persona_id,
        belief_state=belief_state,
        belief_score=belief_score,
        engagement_type=engagement_type,
        channel=channel,
        asset_id=asset_id,
        effect_bucket=effect_bucket,
        meta=meta or {},
        account_meta=account_meta or (meta.get("account_meta") if meta else {}) or {},
    )
    db.add(obj)
    return obj

# ---- read ----

def load_episodes_for_product(
    db: Session,
    *,
    product_id: str,
) -> List[SHMEpisode]:
    return (
        db.query(SHMEpisode)
        .filter(SHMEpisode.product_id == product_id)
        .order_by(
            asc(SHMEpisode.account_id),
            asc(SHMEpisode.episode_id),
            asc(SHMEpisode.step_index),
        )
        .all()
    )


def load_episodes_for_account(
    db: Session,
    *,
    product_id: str,
    account_id: str,
) -> List[SHMEpisode]:
    return (
        db.query(SHMEpisode)
        .filter(
            SHMEpisode.product_id == product_id,
            SHMEpisode.account_id == account_id,
        )
        .order_by(
            asc(SHMEpisode.episode_id),
            asc(SHMEpisode.step_index),
        )
        .all()
    )
