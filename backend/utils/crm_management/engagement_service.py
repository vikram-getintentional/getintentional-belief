# backend/utils/crm_management/engagement_service.py
from typing import List, Dict, Any, Optional, Tuple, Set, Iterable
from sqlalchemy import text
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import re
from backend.database import get_db, engine
from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.crm_management.target_account_models_dto import TargetAccount
from backend.utils.crm_management.target_account_manager import get_account_by_id  # your ORM model
from backend.utils.knowledge_base.arsenal.db_models import (
    ApprovalStatus,
    ArsenalAsset,
    ArsenalChannel,
)
from backend.utils.knowledge_base.arsenal.channel_catalog import resolve_canonical_channel
from backend.utils.knowledge_base.arsenal.service import (
    get_or_create_asset,
    get_or_create_channel,
)
from backend.utils.knowledge_base.arsenal.parser import parse_engagement_activity
import traceback
from backend.utils.crm_management.person_service import match_persona_for_actor_in_graph
from backend.utils.graph_base.network_graph import build_product_graph, get_node_by_id
from backend.utils.segment_utils import segment_keys_from_meta, normalize_account_meta
from backend.utils.persona_normalization import normalize_persona_label

def _to_conf(x) -> int:
    try:
        v = float(x if x is not None else 1.0)
    except Exception:
        v = 1.0
    v = max(0.0, min(1.0, v))
    return int(round(v * 100))

def _to_dt_z(ts: str) -> datetime:
    # accepts "...Z" or "...+00:00"
    ts = (ts or "").strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)

def _now_z():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    return text or None

def _title_case(value: str) -> str:
    return " ".join(segment.capitalize() for segment in value.replace("_", " ").split() if segment)

def _persona_id_from_row(row: TargetAccountEngagement) -> Optional[str]:
    parts = [
        (row.actor_title or "").strip(),
        (row.actor_department or "").strip(),
        (row.actor_seniority or "").strip(),
    ]
    filtered = [p for p in parts if p]
    if not filtered:
        return None
    return "|".join(filtered)

def _persona_label(persona_id: Optional[str]) -> Optional[str]:
    if not persona_id:
        return None
    segments = [seg.strip() for seg in persona_id.split("|") if seg and seg.strip()]
    if not segments:
        return persona_id
    return " | ".join(_title_case(seg) for seg in segments)

BELIEF_STAGE_LABELS = {
    "problem": "Problem Realization",
    "pain": "Pain Realization",
    "resolution": "Resolution Discovery",
    "execution": "Execution Guidance",
}

_STAGE_NORMALIZATION = {
    "problem realization": "problem",
    "problem": "problem",
    "awareness": "problem",
    "pain realization": "pain",
    "pain": "pain",
    "mid funnel": "pain",
    "resolution discovery": "resolution",
    "resolution": "resolution",
    "evaluation": "resolution",
    "late funnel": "resolution",
    "execution guidance": "execution",
    "execution": "execution",
}

_PERSONA_STAGE_CACHE: Dict[Tuple[str, str], Tuple[Optional[str], Optional[str]]] = {}


def _canonical_stage_code(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    return _STAGE_NORMALIZATION.get(normalized, normalized)


def _persona_stage_for_id(product_id: str, persona_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not persona_id:
        return None, None
    cache_key = (product_id, persona_id)
    if cache_key in _PERSONA_STAGE_CACHE:
        return _PERSONA_STAGE_CACHE[cache_key]
    stage_code: Optional[str] = None
    stage_label: Optional[str] = None
    try:
        graph = build_product_graph(product_id)
        node = get_node_by_id(graph, persona_id) if graph else None
    except Exception:
        node = None
    if node:
        stage_raw = (
            node.get("journey_phase")
            or node.get("stage_label")
            or node.get("phase")
            or node.get("funnel_stage")
        )
        stage_code = _canonical_stage_code(stage_raw)
    if stage_code:
        stage_label = BELIEF_STAGE_LABELS.get(stage_code, stage_code.title())
    _PERSONA_STAGE_CACHE[cache_key] = (stage_code, stage_label)
    return stage_code, stage_label


def _normalized_actor(actor: Dict[str, Any]) -> Dict[str, str]:
    return {
        "name": (actor.get("name") or "").strip(),
        "title": (actor.get("title") or "").strip().lower(),
        "department": (actor.get("department") or "").strip().lower(),
        "seniority": (actor.get("seniority") or "").strip().lower(),
        "email": (actor.get("email") or "").strip().lower(),
    }


def _actor_resolver_key(actor: Dict[str, Any]) -> str:
    email = (actor.get("email") or "").strip().lower()
    if email:
        return f"email:{email}"
    name = (actor.get("name") or "").strip().lower()
    title = (actor.get("title") or "").strip().lower()
    dept = (actor.get("department") or "").strip().lower()
    sen = (actor.get("seniority") or "").strip().lower()
    return f"{name}|{title}|{dept}|{sen}"


def _account_meta_snapshot(account_meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not account_meta:
        return None
    keys = [
        "industry",
        "revenue_range",
        "employee_range",
        "funding_stage",
        "geography",
        "deal_status",
    ]
    snapshot = {key: account_meta.get(key) for key in keys if account_meta.get(key) not in (None, "", [])}
    if not snapshot:
        return None
    return snapshot


def _resolve_persona_context(
    product_id: str,
    actor: Dict[str, Any],
    resolver_states: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    normalized_actor = _normalized_actor(actor)
    actor_key = _actor_resolver_key(normalized_actor)
    resolver_state = resolver_states.get(actor_key)
    result = {}
    try:
        match = match_persona_for_actor_in_graph(
            product_id,
            normalized_actor,
            resolver_state=resolver_state,
        )
        resolution = match.get("resolution") or {}
        persona_id = (
            resolution.get("resolved_persona_id")
            or match.get("best")
        )
        persona_label = match.get("best_label") or _persona_label(persona_id)
        confidence = resolution.get("confidence")
        if resolution.get("state"):
            resolver_states[actor_key] = resolution["state"]
    except Exception:
        persona_id = None
        persona_label = None
        confidence = None
        match = {}
    stage_code, stage_label = _persona_stage_for_id(product_id, persona_id)
    result.update(
        {
            "persona_id": persona_id,
            "persona_label": persona_label,
            "persona_confidence": confidence,
            "belief_stage_code": stage_code,
            "belief_stage_label": stage_label,
        }
    )
    return result

def _segments_from_account_meta(account_meta: Optional[Dict[str, Any]]) -> List[str]:
    if not account_meta:
        return []
    mapping = [
        ("industry", "Industry"),
        ("revenue_range", "Revenue"),
        ("employee_range", "Employees"),
        ("geography", "Region"),
        ("deal_status", "Deal Stage"),
        ("funding_stage", "Funding"),
    ]
    segments: List[str] = []
    for key, label in mapping:
        value = account_meta.get(key)
        if value:
            segments.append(f"{label}: {value}")
    return segments

def _merge_string_lists(existing: Optional[Iterable[str]], additions: Iterable[str]) -> List[str]:
    merged: List[str] = []
    seen: Set[str] = set()
    for source in (existing or []):
        if not source:
            continue
        if source in seen:
            continue
        seen.add(source)
        merged.append(source)
    for item in additions:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged

def _resolve_asset_labels(
    payload: Dict[str, Any],
    inferred_asset_name: Optional[str] = None,
    inferred_slug_seed: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns a tuple of (display_name, slug_seed) for asset creation.
    """
    raw_label = (
        payload.get("asset_label")
        or payload.get("asset_name")
        or payload.get("asset_title")
        or payload.get("raw_activity")
        or payload.get("asset_id")
    )
    raw_label = raw_label.strip() if isinstance(raw_label, str) else raw_label
    slug_seed = payload.get("asset_id") or raw_label
    slug_seed = slug_seed.strip() if isinstance(slug_seed, str) else slug_seed
    if not raw_label and inferred_asset_name:
        raw_label = inferred_asset_name
    if not slug_seed and inferred_slug_seed:
        slug_seed = inferred_slug_seed
    return raw_label, slug_seed

def _resolve_channel_metadata(
    payload: Dict[str, Any],
    inferred_channel_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    raw_label = _normalize_text(payload.get("channel"))
    if not raw_label:
        # fallback to explicit source when it resembles a channel (e.g., "email")
        raw_label = _normalize_text(payload.get("source"))
    if not raw_label:
        raw_label = inferred_channel_label
    if not raw_label:
        return None
    slug_hint, canonical = resolve_canonical_channel(raw_label)
    return {
        "slug": slug_hint,
        "name": canonical.get("name") or raw_label,
        "channel_type_label": canonical.get("channel_type_label") or canonical.get("name") or raw_label,
        "channel_type": canonical.get("channel_type"),
        "delivery_mode": canonical.get("delivery_mode"),
        "raw_label": raw_label,
    }


def _resolve_engagement_verb(payload: Dict[str, Any]) -> Optional[str]:
    raw_activity = _normalize_text(payload.get("raw_activity"))
    action = _normalize_text(payload.get("action"))
    activity = _normalize_text(payload.get("activity"))
    for candidate in (payload.get("verb"), action, activity, raw_activity):
        normalized = _normalize_text(candidate)
        if not normalized:
            continue
        tokens = normalized.split()
        verb = tokens[0].lower()
        if verb.endswith("ed"):
            return verb
        mapped = {
            "download": "downloaded",
            "attend": "attended",
            "reply": "replied",
            "open": "opened",
            "view": "viewed",
            "click": "clicked",
        }.get(verb)
        if mapped:
            return mapped
    return None


def _ensure_reference_entry(
    existing: Optional[List[Dict[str, str]]],
    ref_id: str,
    label: str,
) -> List[Dict[str, str]]:
    records = list(existing or [])
    if any(entry.get("id") == ref_id for entry in records):
        return records
    records.append({"id": ref_id, "label": label})
    return records


def _ensure_engagement_columns() -> None:
    ddl = {
        "channel_id": "ALTER TABLE target_account_engagements ADD COLUMN channel_id VARCHAR",
        "engagement_verb": "ALTER TABLE target_account_engagements ADD COLUMN engagement_verb VARCHAR",
        "activity_label": "ALTER TABLE target_account_engagements ADD COLUMN activity_label VARCHAR",
        "asset_category": "ALTER TABLE target_account_engagements ADD COLUMN asset_category VARCHAR",
        "parser_version": "ALTER TABLE target_account_engagements ADD COLUMN parser_version VARCHAR",
        "parser_confidence": "ALTER TABLE target_account_engagements ADD COLUMN parser_confidence FLOAT",
        "persona_id": "ALTER TABLE target_account_engagements ADD COLUMN persona_id VARCHAR",
        "persona_label": "ALTER TABLE target_account_engagements ADD COLUMN persona_label VARCHAR",
        "persona_confidence": "ALTER TABLE target_account_engagements ADD COLUMN persona_confidence FLOAT",
        "belief_stage_code": "ALTER TABLE target_account_engagements ADD COLUMN belief_stage_code VARCHAR",
        "belief_stage_label": "ALTER TABLE target_account_engagements ADD COLUMN belief_stage_label VARCHAR",
        "account_meta": "ALTER TABLE target_account_engagements ADD COLUMN account_meta JSON",
        "segment_keys": "ALTER TABLE target_account_engagements ADD COLUMN segment_keys JSON",
        "derived_channel_id": "ALTER TABLE target_account_engagements ADD COLUMN derived_channel_id VARCHAR",
    }
    try:
        with engine.connect() as conn:
            existing = {
                row["name"]
                for row in conn.execute(text("PRAGMA table_info(target_account_engagements);"))
            }
        missing = {col: stmt for col, stmt in ddl.items() if col not in existing}
        if not missing:
            return
        with engine.begin() as conn:
            for statement in missing.values():
                try:
                    conn.execute(text(statement))
                except Exception:
                    # column may have been added concurrently; ignore
                    pass
    except Exception:
        # best-effort; if introspection fails we silently continue
        return


_ensure_engagement_columns()

def _attach_arsenal_references(
    db: Session,
    *,
    product_id: str,
    engagement_payload: Dict[str, Any],
    engagement_row: TargetAccountEngagement,
    asset_tracker: Dict[str, Set[str]],
    channel_tracker: Dict[str, Set[str]],
    account_meta: Optional[Dict[str, Any]] = None,
) -> None:
    persona_id = engagement_row.persona_id or _persona_id_from_row(engagement_row)
    persona_label = engagement_row.persona_label or _persona_label(persona_id)
    account_segments = _segments_from_account_meta(account_meta)
    engagement_row.engagement_verb = _resolve_engagement_verb(engagement_payload)

    parsed_activity = parse_engagement_activity(engagement_payload)
    engagement_row.activity_label = parsed_activity.activity_label.label
    engagement_row.asset_category = parsed_activity.asset_category.label
    engagement_row.parser_version = parsed_activity.parser_version
    engagement_row.parser_confidence = parsed_activity.asset_name.confidence

    asset_name, asset_slug_seed = _resolve_asset_labels(
        engagement_payload,
        parsed_activity.asset_name.label,
        parsed_activity.asset_slug_seed,
    )
    asset_identifier = engagement_payload.get("asset_id")
    asset: Optional[ArsenalAsset] = None
    if asset_name or asset_identifier:
        defaults: Dict[str, Any] = {}
        if asset_slug_seed:
            defaults["slug"] = ArsenalAsset.slug_for(str(asset_slug_seed))
        defaults["created_from_engagement_id"] = getattr(engagement_row, "id", None)
        defaults.setdefault("description", engagement_payload.get("raw_activity"))
        if persona_label:
            defaults["target_personas"] = [persona_label]
        if account_segments:
            defaults["target_account_segments"] = account_segments
        if parsed_activity.asset_category.label:
            defaults["category"] = parsed_activity.asset_category.label
        defaults["derived_metadata"] = parsed_activity.derived_metadata()
        defaults["auto_classification_confidence"] = parsed_activity.asset_name.confidence
        defaults["approval_status"] = ApprovalStatus.PENDING
        activity_labels: List[str] = []
        if parsed_activity.activity_label.label:
            activity_labels.append(parsed_activity.activity_label.label)
        defaults["activity_labels"] = activity_labels
        asset, created = get_or_create_asset(
            db,
            product_id=product_id,
            name=asset_name or asset_identifier or "Asset",
            defaults=defaults,
            explicit_id=asset_identifier.strip() if isinstance(asset_identifier, str) and asset_identifier.strip() else None,
        )
        if engagement_row.asset_id != asset.id:
            engagement_row.asset_id = asset.id
        if asset_name and asset.name in (None, "", asset.id):
            asset.name = asset_name
        if not asset.description and engagement_payload.get("raw_activity"):
            asset.description = engagement_payload.get("raw_activity")
        if not created:
            if persona_label:
                asset.target_personas = _merge_string_lists(
                    asset.target_personas, [persona_label]
                )
            if account_segments:
                asset.target_account_segments = _merge_string_lists(
                    asset.target_account_segments, account_segments
                )
        asset.update_metadata_status()
        if created:
            asset_tracker["created"].add(asset.id)
        if not asset.metadata_complete:
            asset_tracker["pending"].add(asset.id)
        else:
            asset_tracker["pending"].discard(asset.id)
        if not asset.created_from_engagement_id and getattr(engagement_row, "id", None):
            asset.created_from_engagement_id = engagement_row.id
        db.add(asset)

    channel_info = _resolve_channel_metadata(
        engagement_payload, parsed_activity.channel_label.label
    )
    channel: Optional[ArsenalChannel] = None
    if channel_info:
        canonical_slug = ArsenalChannel.slug_for(channel_info["slug"] or channel_info["name"])
        channel_identifier = f"channel:{canonical_slug}"
        defaults = {
            "created_from_engagement_id": getattr(engagement_row, "id", None),
            "channel_type": channel_info.get("channel_type"),
            "delivery_mode": channel_info.get("delivery_mode"),
            "derived_metadata": {
                "parser_version": parsed_activity.parser_version,
                "channel": parsed_activity.channel_label.as_dict(),
                "activity": parsed_activity.activity_label.as_dict(),
            },
            "auto_classification_confidence": parsed_activity.channel_label.confidence,
            "approval_status": ApprovalStatus.PENDING,
        }
        channel, created = get_or_create_channel(
            db,
            product_id=product_id,
            name=channel_info["name"],
            defaults=defaults,
            explicit_id=channel_identifier,
        )
        if created:
            channel_tracker["created"].add(channel.id)
        if not channel.metadata_complete:
            channel_tracker["pending"].add(channel.id)
        else:
            channel_tracker["pending"].discard(channel.id)
        if not channel.created_from_engagement_id and getattr(engagement_row, "id", None):
            channel.created_from_engagement_id = engagement_row.id
        if not channel.notes and engagement_payload.get("raw_activity"):
            channel.notes = engagement_payload.get("raw_activity")
        channel.update_metadata_status()
        db.add(channel)
        engagement_row.channel = channel_info.get("channel_type_label") or channel.name
        engagement_row.channel_id = channel.id

    if asset and channel:
        asset.typical_channels = _ensure_reference_entry(asset.typical_channels, channel.id, channel.name)
        channel.typical_assets = _ensure_reference_entry(channel.typical_assets, asset.id, asset.name)
        db.add(asset)
        db.add(channel)
        if not asset.primary_channel_id:
            asset.primary_channel_id = channel.id
    if asset and parsed_activity.activity_label.label:
        current = asset.activity_labels or []
        asset.activity_labels = _merge_string_lists(
            current,
            [parsed_activity.activity_label.label],
        )
        db.add(asset)
def save_and_update_account_engagements(product_id: str, engagements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Persist a batch of engagements (append-only).
    Updates per-account denormalized counters (optional) on TargetAccount if those fields exist.
    """
    print("Saving engagements for product_id:", product_id, " with engagements:", engagements)
    db: Session = next(get_db())
    created = 0
    touched: set[str] = set()
    asset_tracker = {"created": set(), "pending": set()}
    channel_tracker = {"created": set(), "pending": set()}

    resolver_states: Dict[str, Dict[str, Any]] = {}

    try:
        # preload accounts for this product
        
        for e in engagements or []:
            acc_id = e.get("account_id") or e.get("target_account_id")
            if not acc_id:
                print("Skipping engagement with missing account_id:", e)
                continue
            account = get_account_by_id(product_id, acc_id)
            if not account:
                print("Skipping engagement for unknown account_id:", acc_id)
                continue
            actor = e.get("actor") or {}
            account_snapshot = _account_meta_snapshot(account)
            ts_raw = (e.get("timestamp") or _now_z()).strip()
            row = TargetAccountEngagement(
                product_id=product_id,
                target_account_id=acc_id,
                timestamp=ts_raw,                 # raw string (optional)
                timestamp_dt=_to_dt_z(ts_raw),    # proper aware datetime
                source=(e.get("source") or "other"),
                channel=e.get("channel"),
                inferred=bool(e.get("inferred", False)),
                actor_name=(actor.get("name") or None),
                actor_title=(actor.get("title") or "").strip(),
                actor_department=(actor.get("department") or "").strip(),
                actor_seniority=(actor.get("seniority") or "").strip(),
                actor_confidence=_to_conf(actor.get("confidence")),
                raw_activity=(e.get("raw_activity") or ""),
                asset_id=e.get("asset_id"),
                payload=e,
            )
            row.candidate_persona_label = normalize_persona_label(
                actor.get("title"),
                actor.get("department"),
                actor.get("seniority"),
            )
            persona_context = _resolve_persona_context(
                product_id,
                actor,
                resolver_states,
            )
            row.persona_id = persona_context.get("persona_id")
            row.persona_label = persona_context.get("persona_label")
            row.persona_confidence = persona_context.get("persona_confidence")
            row.belief_stage_code = persona_context.get("belief_stage_code")
            row.belief_stage_label = persona_context.get("belief_stage_label")
            if account_snapshot:
                row.account_meta = account_snapshot
                normalized = normalize_account_meta(account_snapshot)
                segments = segment_keys_from_meta(normalized)
                if segments:
                    row.segment_keys = segments
            print("Adding engagement row to db:", row)
            db.add(row)
            db.flush()
            _attach_arsenal_references(
                db,
                product_id=product_id,
                engagement_payload=e,
                engagement_row=row,
                asset_tracker=asset_tracker,
                channel_tracker=channel_tracker,
                account_meta=account_snapshot,
            )
            created += 1
            touched.add(acc_id)
            print("Engagement added to db")

        """
        # Optional: update denormalized counters if you added them to TargetAccount
        if touched and hasattr(TargetAccount, "engagement_count"):
            
            for acc_id in touched:
                count = db.query(TAE).filter(TAE.target_account_id == acc_id).count()
                last_ts = db.query(TAE.timestamp).filter(TAE.target_account_id == acc_id)\
                           .order_by(TAE.timestamp.desc()).first()
                acct = get_account_by_id(acc_id)
                acct.engagement_count = count
                if hasattr(acct, "last_engagement_at"):
                    acct.last_engagement_at = last_ts[0] if last_ts else None
        """
        db.commit()
        print("Engagements committed to db")
        print("Created engagements:", created)
        print("Touched accounts:", list(touched))
        return {
            "created": created,
            "touched_accounts": list(touched),
            "arsenal_assets_created": sorted(asset_tracker["created"]),
            "arsenal_channels_created": sorted(channel_tracker["created"]),
            "arsenal_assets_missing_meta": sorted(asset_tracker["pending"]),
            "arsenal_channels_missing_meta": sorted(channel_tracker["pending"]),
        }
    except Exception:
        db.rollback()
        print("❌ Commit failed:", repr(e))
        print(traceback.format_exc())
        raise
    finally:
        db.close()

def get_account_engagements(product_id: str, account_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    account = get_account_by_id(product_id, account_id)
    if not account:
        print("Account not found for engagements fetch:", account_id)
        return []
    db: Session = next(get_db())
    try:
        q = (db.query(TargetAccountEngagement)
               .filter(TargetAccountEngagement.product_id == product_id,
                       TargetAccountEngagement.target_account_id == account_id)
               .order_by(TargetAccountEngagement.timestamp.desc())
               .limit(limit))
        rows = q.all()
        out = []
        for r in rows:
            out.append({
                "id": r.id,
                "account_id": r.target_account_id,
                "timestamp": r.timestamp,
                "source": r.source,
                "channel": r.channel,
                "channel_id": r.channel_id,
                "verb": r.engagement_verb,
                "inferred": r.inferred,
                "actor": {
                    "name": r.actor_name,
                    "title": r.actor_title,
                    "department": r.actor_department,
                    "seniority": r.actor_seniority,
                    "confidence": round((r.actor_confidence or 100) / 100, 2),
                },
                "candidate_persona_label": r.candidate_persona_label,
                "raw_activity": r.raw_activity,
                "asset_id": r.asset_id,
                "persona_id": r.persona_id,
                "persona_label": r.persona_label,
                "belief_stage": {
                    "code": r.belief_stage_code,
                    "label": r.belief_stage_label,
                } if r.belief_stage_code or r.belief_stage_label else None,
                "segment_keys": r.segment_keys or [],
                "account_meta": r.account_meta,
                "payload": r.payload,
            })
        return out
    finally:
        db.close()


def _infer_seniority_from_title(title: str) -> str:
    t = (title or "").lower()
    if any(k in t for k in ["chief", "cxo", "cfo", "ceo", "coo", "cto", "cmo", "vp", "vice president", "head", "director"]):
        return "Executive"
    if any(k in t for k in ["manager", "lead", "owner"]):
        return "Manager"
    if "senior" in t or "sr" in t:
        return "Senior"
    return "Operator"


def _parse_ts(ts: str, default_dt: datetime) -> datetime:
    try:
        # accept "...Z" or "+00:00"
        if ts and isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        pass
    return default_dt


def save_engagements_bulk(product_id: str, account_id: str, engagements: List[Dict[str, Any]]) -> int:
    """
    OVERWRITE MODE:
      - Deletes ALL existing rows for (product_id, account_id)
      - Inserts the provided engagements fresh
      - If engagements == [], the account is left with 0 rows

    Expected engagement shape (per item):
    {
      "account_id": "...",                  # ignored/overwritten by `account_id` arg
      "timestamp": "2025-01-01T10:00:00Z",
      "actor": {
        "name": "...",
        "title": "...",
        "department": "...",
        "seniority": "...",                 # optional; inferred if missing
        "confidence": 1.0                   # optional; 0..1
      },
      "channel": "webinar|email|...",
      "source": "marketing|sales|cs|product|other",
      "raw_activity": "...",
      "asset_id": "optional-id",
      "inferred": false
    }
    """
    print("Saving bulk engagements for product_id:", product_id, " account_id:", account_id, " engagements:", engagements)
    db: Session = next(get_db())
    now = datetime.now(timezone.utc)
    asset_tracker = {"created": set(), "pending": set()}
    channel_tracker = {"created": set(), "pending": set()}

    try:
        # Guard: target account must exist for this product
        account_row = get_account_by_id(product_id, account_id)
        if not account_row:
            print(f"[warn] account_id {account_id} not found for product {product_id}; skipping")
            return 0
        account_snapshot = _account_meta_snapshot(account_row)
        segments = None
        if account_snapshot:
            normalized = normalize_account_meta(account_snapshot)
            segments = segment_keys_from_meta(normalized)

        # 1) Wipe existing rows
        print("Wiping existing engagements for account:", account_id)
        (
            db.query(TargetAccountEngagement)
              .filter(
                  TargetAccountEngagement.product_id == product_id,
                  TargetAccountEngagement.target_account_id == account_id,
              )
              .delete(synchronize_session=False)
        )

        # 2) Insert provided payload
        inserted = 0
        seen_ids = set()  # dedupe within the payload
        print("cleared db rows, now inserting new engagements")

        for e in engagements or []:
            print("Inserting engagement:", e)

            ts_raw: str = (e.get("timestamp") or "").strip()
            actor = e.get("actor") or {}
            name = (actor.get("name") or "").strip() or None
            title = (actor.get("title") or actor.get("role") or "").strip()   # tolerate legacy 'role'
            dept  = (actor.get("department") or "").strip()
            snr   = (actor.get("seniority") or "").strip() or _infer_seniority_from_title(title)
            conf  = actor.get("confidence", 1.0)

            raw_activity = (e.get("raw_activity") or "").strip()
            if not title or not dept or not raw_activity or not ts_raw:
                # skip malformed entry
                print("[skip] missing one of required fields: title/department/raw_activity/timestamp")
                continue

            ts_dt = _parse_ts(ts_raw, now)

            # dedupe within payload (since we wiped the DB already)
            dedupe_key = f"{account_id}|{ts_dt.isoformat()}|{name}|{title}|{dept}|{raw_activity}"
            if dedupe_key in seen_ids:
                print("[dedupe] skipping duplicate payload row:", dedupe_key)
                continue
            seen_ids.add(dedupe_key)

            print("Data to insert:", {
                "product_id": product_id,
                "target_account_id": account_id,
                "timestamp": ts_raw,
                "timestamp_dt": ts_dt,
                "source": (e.get("source") or "other"),
                "channel": (e.get("channel") or None),
                "inferred": bool(e.get("inferred") or False),
                "actor_name": name,
                "actor_title": title,
                "actor_department": dept,
                "actor_seniority": snr,
                "actor_confidence": int(max(0.0, min(1.0, float(conf))) * 100),
                "raw_activity": raw_activity,
                "asset_id": (e.get("asset_id") or None),
                "payload": e,
            })

            row = TargetAccountEngagement(
                product_id=product_id,
                target_account_id=account_id,
                timestamp=ts_raw,
                timestamp_dt=ts_dt,
                source=(e.get("source") or "other"),
                channel=(e.get("channel") or None),
                inferred=bool(e.get("inferred") or False),

                actor_name=name,
                actor_title=title,
                actor_department=dept,
                actor_seniority=snr,
                actor_confidence=int(max(0.0, min(1.0, float(conf))) * 100),

                raw_activity=raw_activity,
                asset_id=(e.get("asset_id") or None),
                payload=e,
            )
            row.candidate_persona_label = normalize_persona_label(title, dept, snr)
            if account_snapshot:
                row.account_meta = account_snapshot
                if segments:
                    row.segment_keys = segments
            db.add(row)
            db.flush()
            _attach_arsenal_references(
                db,
                product_id=product_id,
                engagement_payload=e,
                engagement_row=row,
                asset_tracker=asset_tracker,
                channel_tracker=channel_tracker,
                account_meta=account_snapshot,
            )
            inserted += 1
            print("Inserted engagement row:", row.id if hasattr(row, "id") else "<pending>")

        db.commit()
        if asset_tracker["created"] or channel_tracker["created"]:
            print(
                "[arsenal] bulk load created assets:",
                list(asset_tracker["created"]),
                "channels:",
                list(channel_tracker["created"]),
            )
        if asset_tracker["pending"] or channel_tracker["pending"]:
            print(
                "[arsenal] bulk load pending metadata assets:",
                list(asset_tracker["pending"]),
                "channels:",
                list(channel_tracker["pending"]),
            )
        return inserted

    except Exception as ex:
        print("❌ Bulk save failed with exception:", repr(ex))
        try:
            cols = db.execute(text("PRAGMA table_info(target_account_engagements);")).fetchall()
            print("[debug] target_account_engagements columns:", cols)
        except Exception as ex2:
            print("[debug] failed to introspect columns:", repr(ex2))
        db.rollback()
        raise
    finally:
        db.close()
