# backend/utils/inference/belief_manager/journey/learn_service.py
from __future__ import annotations
from statistics import mean
from typing import Dict, Any, Iterable, List, Tuple, Optional, Literal, Sequence, Set
from collections import defaultdict, Counter
from dataclasses import asdict
from datetime import datetime, timedelta
import itertools
import json
import re
from copy import deepcopy
import os
import traceback

import networkx as nx
from sqlalchemy.orm import Session

from backend.database import get_db  # only needed if you later persist to DB


from backend.utils.inference.belief_manager.journey.storage import (
    load_stats,
    save_stats,
    load_weights,
    save_weights,
)
from backend.utils.inference.belief_manager.journey.learn import (
    update_edge_stats,
    recompute_weights,
    bucket_channel,
)
from backend.utils.inference.belief_manager.journey.replay import next_distribution

from backend.utils.inference.belief_manager.journey.shm_models import SHMEpisode
from backend.utils.inference.belief_manager.journey.shm_service import (
    load_episodes_for_product,
)
from backend.super_models.shm.episode import (
    ShmEpisode,
    ShmEpisodeStep,
    ShmLearningUpdate,
    ShmUpdateType,
    EpisodeOutcome,
)
from backend.utils.graph_base.network_graph import (
    GRAPH_DATA_PATH,
    build_product_graph,
    get_product_id_from_subgraph,
    get_source_nodes_by_target_and_type,
)
from backend.utils.graph_base.graph_utils.save_and_load_graph_as_json import save_graph_as_json
from backend.utils.graph_base.agent_graph_builder import (
    _persona,
    _job,
    _pain,
    _trigger,
    _upsert_edge,
    current_timestamp,
)
from backend.utils.graph_base.persona_learning import (
    auto_add_persona_nodes_from_candidates,
    compute_persona_impact_metrics,
    save_persona_metrics,
)
from backend.utils.inference.belief_manager.graph_diff_mapper import _L
from backend.utils.knowledge_base.arsenal.db_models import ArsenalAsset
from backend.utils.knowledge_base.arsenal.service import (
    get_or_create_channel,
    record_asset_channel_impact,
)
from backend.utils.inference.rcs_generators.generate_rcs_fast import generate_rcs_new
from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.crm_management.target_account_manager import (
    TargetAccount as TargetAccountORM,
)
from backend.utils.segment_utils import (
    segment_keys_from_meta,
    segment_label_from_key,
    normalize_account_meta as normalize_meta_dict,
)
from backend.utils.inference.belief_manager.journey.win_regression import (
    build_win_regression_summary,
)
from backend.utils.inference.belief_manager.types import (
    Episode,
    PersonaCandidateStat,
)


BELIEF_STAGE_LABELS = {
    "problem": "Problem Realization",
    "pain": "Pain Realization",
    "resolution": "Resolution Discovery",
    "execution": "Execution Guidance",
}

InsightSource = Literal["data", "graph", "default", "mixed"]


def _insight_text(
    text: str,
    *,
    source: InsightSource = "default",
    confidence: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"text": text, "source": source}
    if confidence is not None:
        payload["confidence"] = round(float(confidence), 4)
    if extra:
        payload.update(extra)
    return payload


def _format_list(items: Sequence[str], conjunction: str = "and") -> str:
    parts = [item for item in items if item]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} {conjunction} {parts[1]}"
    return f"{', '.join(parts[:-1])}, {conjunction} {parts[-1]}"


def _node_label(G: Optional[nx.DiGraph], node_id: Optional[str]) -> str:
    if not node_id:
        return ""
    if G is None or node_id not in G:
        return node_id
    node = G.nodes[node_id] or {}
    return (
        node.get("label")
        or node.get("name")
        or node.get("title")
        or node.get("description")
        or node_id
    )


def _build_job_spec(
    G: Optional[nx.DiGraph],
    job_id: Optional[str],
    pains_fallback: Sequence[str],
    *,
    default_label: str = "",
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {"id": job_id}
    pains: List[Dict[str, Any]] = []
    if G is not None and job_id and job_id in G:
        node = G.nodes[job_id] or {}
        spec["label"] = node.get("description") or node.get("label") or node.get("name") or job_id
        spec["description"] = node.get("description")
        spec["department"] = node.get("department")
        pains = [
            {"id": pain_id, "label": _node_label(G, pain_id)}
            for pain_id in get_source_nodes_by_target_and_type(G, job_id, "felt_in")
        ]
    else:
        spec["label"] = job_id or default_label or "Proposed job"
    if not pains and pains_fallback:
        pains = [{"id": pid, "label": _node_label(G, pid)} for pid in pains_fallback if pid]
    spec["pains"] = pains
    return spec


def _build_pain_specs(G: Optional[nx.DiGraph], pain_ids: Sequence[str]) -> List[Dict[str, Any]]:
    return [{"id": pid, "label": _node_label(G, pid)} for pid in pain_ids if pid]


DEFAULT_PRODUCT_INSIGHTS = {
    "ideal_customer_patterns": {
        "works_well": [
            "SaaS companies with 1–5K employees and Series B–E funding convert 42% faster than average.",
            "North American accounts in recurring revenue models show the highest belief momentum.",
            "Companies with mature RevOps teams show twice the mid-stage acceleration.",
        ],
        "gaps": [
            "Retail/eCommerce companies in APAC have the lowest conversion probability — primarily due to lack of urgency in billing consolidation.",
            "Companies under 100 employees rarely complete evaluation — most stall at ‘Aware → Pain Realization.’",
        ],
    },
    "persona_landscape": {
        "frequency": [
            "Billing Specialist → 82%",
            "Manager Finance → 76%",
            "RevOps Manager → 63%",
            "Product Manager → 49%",
            "CFO → 38%",
        ],
        "critical_leads": [
            "Billing Specialist (Operator)",
            "RevOps Manager",
            "Finance Manager",
        ],
        "decision_personas": [
            "VP Finance",
            "CFO",
            "Head of RevOps",
        ],
        "blockers": [
            "IT Manager (Security)",
            "Engineering Lead (API/Scaling concerns)",
            "Procurement (Compliance friction)",
        ],
        "coalitions": [
            "Operations Manager + Finance Manager activate together in 61% of accounts — usually within 3 days.",
            "Product Manager + Engineering Lead form a technical coalition late in deals — they appear as a pair.",
            "Billing Specialist + RevOps Manager have the strongest mid-stage acceleration effect.",
        ],
    },
    "belief_transitions": {
        "hardest": _insight_text(
            "Across all deals, the hardest jump is Pain Realization → Resolution. This is where 47% of deals stall.",
            source="default",
        ),
        "easiest": _insight_text(
            "Once ‘Problem Realization → Execution Guidance’ begins, Finance personas accelerate belief faster than any other group.",
            source="default",
        ),
        "top_pains": [
            _insight_text("Inconsistent billing cycles", source="default"),
            _insight_text("Manual revenue recognition", source="default"),
            _insight_text("Multi-entity complexity", source="default"),
            _insight_text("Personalized pricing limitations", source="default"),
        ],
    },
    "asset_channel_effectiveness": {
        "high_assets": [
            "Case Study Decks consistently produce the highest belief lift (avg +8bps).",
            "Scenario-based POC Kickoffs are the strongest late-stage accelerators.",
        ],
        "underperforming_channels": [
            "LinkedIn Ads generate views but almost no belief progression for Product or Engineering personas.",
            "Email newsletters show low conversion for executive personas.",
        ],
        "channel_persona_matches": [
            "Webinars → RevOps & Finance Managers",
            "Direct Sales Calls → Product & Engineering leads",
            "Short audits/frameworks → Executives",
        ],
    },
    "journey_structure": {
        "common_paths": [
            "Billing Specialist → Manager Finance → VP Finance",
            "RevOps Manager → Product Manager → CFO",
            "Ops Manager → RevOps → Finance",
        ],
        "deviations": "Engineering often activates before Finance in larger companies — a reverse pattern compared to your baseline model.",
        "average_duration_days": 32.0,
    },
    "global_patterns": {
        "biggest_barrier": _insight_text(
            "Most stalls originate from compliance complexity — not functional capability gaps.",
            source="default",
        ),
        "hidden_blocker": _insight_text(
            "API and integration concerns appear in mid-stage transcripts even when not surfaced explicitly.",
            source="default",
        ),
        "missed_opportunity": _insight_text(
            "Product personas are under-engaged across 70% of accounts despite having strong belief influence.",
            source="default",
        ),
    },
    "product_strengths": {
        "strengths": [
            "Strong finance automation story — Finance personas move fastest once engaged.",
            "RevOps personas consistently show natural alignment with the value proposition.",
        ],
        "weaknesses": [
            "Technical personas (IT/Eng) have the lowest belief momentum, indicating a potential messaging gap.",
            "Pricing personalization story resonates poorly in EMEA.",
        ],
    },
    "strategic_moves": {
        "segment_priorities": "Prioritize: Mid-market SaaS (1K–5K employees), North America, Series C–E.",
        "persona_priorities": "Invest early in Billing + RevOps coalition — they shape the earliest narratives.",
        "asset_priorities": "Create more risk-framing content for CFOs to unlock late-stage belief transitions.",
        "channel_priorities": "Shift technical content toward hands-on demos, away from whitepapers.",
    },
}

PERSONA_METRICS_DIR = os.path.join(GRAPH_DATA_PATH, "persona_metrics")


def _load_persona_metrics_snapshot(
    product_id: str,
) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    path = os.path.join(PERSONA_METRICS_DIR, f"{product_id}.json")
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}, None
    updated_at = payload.get("updated_at")
    metrics: Dict[str, Dict[str, Any]] = {}
    for entry in payload.get("personas", []):
        persona_id = str(entry.get("persona_id") or "")
        if not persona_id:
            continue
        metrics[persona_id] = entry
    return metrics, updated_at


def _canonical_stage(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = str(value).strip().lower().replace(" ", "_")
    if key in {"problem_realization", "problem"}:
        return "problem"
    if key in {"pain_realization", "pain"}:
        return "pain"
    if key in {"resolution_discovery", "resolution"}:
        return "resolution"
    if key in {"execution_guidance", "execution"}:
        return "execution"
    return key or None


def _stage_label(stage: str) -> str:
    return BELIEF_STAGE_LABELS.get(stage, stage.replace("_", " ").title())


def _format_meta_value(meta: str) -> str:
    if not meta:
        return ""
    meta = meta.strip()
    if meta.lower().startswith("attr "):
        meta = meta[5:]
    if "|" in meta:
        parts = [
            part.strip().replace("_", " ").title()
            for part in meta.split("|")
            if part and part.strip()
        ]
        if parts:
            return " | ".join(parts)
    if ":" in meta:
        key, value = meta.split(":", 1)
        key_label = key.replace("_", " ").title()
        value = value.strip()
        if key == "revenue_range":
            tokens = [tok for tok in re.split(r"[_\-]", value) if tok]
            formatted_tokens = []
            for token in tokens:
                token = token.strip().upper()
                if token.endswith("M") or token.endswith("B") or token.endswith("K"):
                    formatted_tokens.append(f"${token}")
                elif token.replace(".", "", 1).isdigit():
                    formatted_tokens.append(f"${token}")
                else:
                    formatted_tokens.append(token.replace("_", " ").title())
            if len(formatted_tokens) == 2:
                value_label = f"{formatted_tokens[0]}–{formatted_tokens[1]}"
            else:
                value_label = " – ".join(formatted_tokens)
        elif key in {"employee_range", "company_size"}:
            tokens = [tok.strip().title() for tok in re.split(r"[_\-]", value) if tok]
            value_label = " – ".join(tokens)
        else:
            value_label = value.replace("_", " ").title()
        return f"{key_label} {value_label}".strip()
    return meta.replace("_", " ").title()


def _fallback_persona_label(pid: str) -> str:
    if not pid:
        return "Persona"
    candidate = pid
    if ":" in pid:
        candidate = pid.split(":", 1)[-1]
    candidate = candidate.replace("|", " · ")
    candidate = candidate.replace("_", " ").strip()
    return candidate.title() or "Persona"


def _node_data_label(G: nx.DiGraph, node_id: str) -> str:
    node_data = G.nodes.get(node_id, {}) or {}
    return (
        node_data.get("label")
        or node_data.get("title")
        or node_data.get("name")
        or node_data.get("description")
        or _fallback_persona_label(node_id)
    )


def _canonical_persona_lookup(
    G: Optional[nx.DiGraph],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    if G is None:
        return {}, {}, {}
    canonical_lookup: Dict[str, str] = {}
    canonical_labels: Dict[str, str] = {}
    canonical_node_map: Dict[str, str] = {}

    for node_id, node_data in G.nodes(data=True):
        canonical_id = node_data.get("canonical_persona_id")
        if canonical_id:
            canonical_id = str(canonical_id)
            canonical_lookup[node_id] = canonical_id
            canonical_node_map.setdefault(canonical_id, node_id)
            canonical_labels.setdefault(
                canonical_id,
                node_data.get("label")
                or node_data.get("title")
                or node_data.get("name")
                or node_data.get("description")
                or _fallback_persona_label(node_id),
            )
    for canonical_id in list(canonical_labels.keys()):
        canonical_lookup.setdefault(canonical_id, canonical_id)

    for u, v, data in G.edges(data=True):
        typ = (data.get("type") or data.get("relation") or "").lower()
        if typ == "has_variant":
            canonical_id = canonical_lookup.get(u) or str(
                G.nodes[u].get("canonical_persona_id") or u
            )
            if canonical_id:
                canonical_lookup[v] = canonical_id
                canonical_node_map.setdefault(canonical_id, u)
                canonical_labels.setdefault(
                    canonical_id,
                    canonical_labels.get(canonical_id) or _node_data_label(G, u),
                )
        elif typ == "variant_of":
            canonical_id = canonical_lookup.get(v) or str(
                G.nodes[v].get("canonical_persona_id") or v
            )
            if canonical_id:
                canonical_lookup[u] = canonical_id
                canonical_node_map.setdefault(canonical_id, v)
                canonical_labels.setdefault(
                    canonical_id,
                    canonical_labels.get(canonical_id) or _node_data_label(G, v),
                )
    return canonical_lookup, canonical_labels, canonical_node_map


def _aggregate_persona_stats(
    persona_stats: Dict[str, Dict[str, Any]],
    canonical_lookup: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    for pid, stats in persona_stats.items():
        canonical_id = canonical_lookup.get(pid, pid)
        entry = aggregated.setdefault(
            canonical_id,
            {
                "observed": 0,
                "hit_at_1": 0,
                "hit_at_3": 0,
                "partial_off": 0,
                "off_path": 0,
                "pred_counts": Counter(),
            },
        )
        entry["observed"] += stats.get("observed", 0)
        entry["hit_at_1"] += stats.get("hit_at_1", 0)
        entry["hit_at_3"] += stats.get("hit_at_3", 0)
        entry["partial_off"] += stats.get("partial_off", 0)
        entry["off_path"] += stats.get("off_path", 0)
        for predicted, count in (stats.get("pred_counts") or {}).items():
            canonical_pred = canonical_lookup.get(predicted, predicted)
            entry["pred_counts"][canonical_pred] += count
    return aggregated


def _percent(value: float, decimals: int = 0) -> str:
    return f"{round(value * 100, decimals)}%"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _mean_safe(values: Iterable[float]) -> Optional[float]:
    arr = [v for v in values if isinstance(v, (int, float))]
    if not arr:
        return None
    return sum(arr) / len(arr)


META_FIELD_BLOCKLIST = {
    "account_name",
    "deal_status",
    "status",
    "account_id",
    "accountid",
    "id",
    "target_account_id",
    "opportunity_name",
}

META_FIELD_ALLOWLIST = {
    "industry",
    "revenue_range",
    "employee_range",
    "geography",
    "funding_stage",
    "segment",
    "region",
}


def _filter_meta_fields(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    filtered: Dict[str, Any] = {}
    for key, value in meta.items():
        if value in (None, "", [], {}, ()):
            continue
        norm_key = str(key).strip().lower()
        if norm_key in META_FIELD_BLOCKLIST:
            continue
        filtered[key] = value
    return filtered


def _normalize_account_meta(meta: Any) -> List[str]:
    filtered_meta = _filter_meta_fields(meta)
    normalized = normalize_meta_dict(filtered_meta)
    if not normalized:
        return []
    return [
        f"{k}:{v}"
        for k, v in normalized.items()
        if v and k in META_FIELD_ALLOWLIST
    ]


def _segment_keys_from_meta_tokens(tokens: Optional[Iterable[str]]) -> List[str]:
    if not tokens:
        return []
    meta_dict: Dict[str, Any] = {}
    for token in tokens:
        if not token or ":" not in token:
            continue
        field, value = token.split(":", 1)
        cleaned = value.strip()
        if not cleaned:
            continue
        meta_dict[field] = cleaned
    return segment_keys_from_meta(meta_dict)


def _normalize_deal_status(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    status = str(value).strip().lower()
    if "won" in status:
        return "won"
    if "lost" in status:
        return "lost"
    return None


def _build_account_feature_rows(
    episodes: Sequence[ShmEpisode],
    steps: Sequence[ShmEpisodeStep],
    *,
    account_meta_lookup: Dict[str, Dict[str, Any]],
    account_raw_meta: Dict[str, Dict[str, Any]],
    segment_keys_by_account: Dict[str, List[str]],
    account_status_map: Dict[str, str],
    arsenal_stats: Dict[str, Dict[str, Any]],
    asset_metadata: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not (episodes or account_meta_lookup or account_status_map):
        return []

    def _segment_list(acc_id: str) -> List[str]:
        values = segment_keys_by_account.get(acc_id) or []
        return [str(v) for v in values if v]

    asset_delta_lookup: Dict[str, float] = {}
    for asset_id, stats in (arsenal_stats or {}).items():
        count = stats.get("count") or 0
        if not count:
            continue
        try:
            avg_delta = float(stats.get("total_delta") or 0.0) / float(count)
        except Exception:
            avg_delta = 0.0
        asset_delta_lookup[str(asset_id)] = avg_delta

    account_ids: Set[str] = set()
    episode_account_map: Dict[str, str] = {}
    trackers: Dict[str, Dict[str, Any]] = {}

    def _tracker(account_id: str) -> Dict[str, Any]:
        tracker = trackers.get(account_id)
        if tracker:
            return tracker
        tracker = {
            "account_id": account_id,
            "total_steps": 0,
            "off_path_steps": 0,
            "unique_personas": set(),
            "unique_stages": set(),
            "stage_transitions": set(),
            "critical_transitions": set(),
            "asset_steps": 0,
            "channel_counts": Counter(),
            "asset_category_counts": Counter(),
            "asset_success_total": 0.0,
            "asset_success_count": 0,
            "duration_samples": [],
            "segment_keys": _segment_list(account_id),
        }
        trackers[account_id] = tracker
        return tracker

    for episode in episodes:
        account_id_raw = getattr(episode, "account_id", None)
        if not account_id_raw:
            continue
        account_id = str(account_id_raw)
        account_ids.add(account_id)
        episode_id = getattr(episode, "id", None)
        if episode_id:
            episode_account_map[str(episode_id)] = account_id
        tracker = _tracker(account_id)
        started_at = getattr(episode, "started_at", None)
        ended_at = getattr(episode, "ended_at", None)
        if started_at and ended_at:
            try:
                duration_days = max(
                    (ended_at - started_at).total_seconds() / 86400.0,
                    0.01,
                )
            except Exception:
                duration_days = None
            if duration_days is not None:
                tracker["duration_samples"].append(duration_days)

    additional_ids = set()
    additional_ids.update(str(key) for key in account_meta_lookup.keys())
    additional_ids.update(str(key) for key in account_status_map.keys())
    additional_ids.update(str(key) for key in account_raw_meta.keys())
    additional_ids.update(str(key) for key in segment_keys_by_account.keys())
    account_ids.update(additional_ids)
    for acc_id in account_ids:
        _tracker(acc_id)

    stage_fields = (
        "stage_code",
        "stage",
        "stage_label",
        "journey_stage",
        "journey_phase",
        "belief_state",
    )

    def _stage_from_metrics(payload: Dict[str, Any], bucket: Optional[str]) -> Optional[str]:
        for field in stage_fields:
            value = payload.get(field)
            if isinstance(value, dict):
                value = value.get("code") or value.get("label") or value.get("value")
            stage = _canonical_stage(value)
            if stage:
                return stage
        if bucket:
            bucket_norm = str(bucket).lower()
            if "problem" in bucket_norm:
                return "problem"
            if "pain" in bucket_norm:
                return "pain"
            if "resolution" in bucket_norm:
                return "resolution"
            if "execution" in bucket_norm:
                return "execution"
        return None

    def _bucket_class(bucket: Optional[str]) -> str:
        mapping = {
            "on_path": "expected",
            "perfect_match": "expected",
            "near_path": "jump_ahead",
            "skip_hit": "jump_ahead",
            "off_path": "off_path",
            "off_path_known": "off_path",
            "out_of_graph": "off_path",
            "no_path": "no_path",
        }
        if not bucket:
            return "no_path"
        return mapping.get(str(bucket).lower(), str(bucket).lower())

    ordered_steps = sorted(
        steps,
        key=lambda step: (
            str(getattr(step, "episode_id", "")),
            getattr(step, "t_index", getattr(step, "step_index", 0)) or 0,
        ),
    )
    last_stage_by_episode: Dict[str, Optional[str]] = {}
    for step in ordered_steps:
        episode_id = getattr(step, "episode_id", None)
        if not episode_id:
            continue
        account_id = episode_account_map.get(str(episode_id))
        if not account_id:
            continue
        tracker = _tracker(account_id)
        tracker["total_steps"] += 1
        bucket = getattr(step, "bucket", None)
        bucket_value = bucket.value if hasattr(bucket, "value") else bucket
        metrics_payload = step.metrics or {}
        stage_code = _stage_from_metrics(metrics_payload, bucket_value)
        if stage_code:
            tracker["unique_stages"].add(stage_code)
            prev_stage = last_stage_by_episode.get(str(episode_id))
            if prev_stage and prev_stage != stage_code:
                tracker["stage_transitions"].add(f"{prev_stage}->{stage_code}")
            last_stage_by_episode[str(episode_id)] = stage_code
        persona_id = (
            getattr(step, "observed_persona_id", None)
            or getattr(step, "predicted_top_persona_id", None)
        )
        if persona_id:
            tracker["unique_personas"].add(str(persona_id))
        classification = _bucket_class(bucket_value)
        if classification == "off_path":
            tracker["off_path_steps"] += 1
        transition_id = metrics_payload.get("belief_transition_id")
        if not transition_id:
            transition_obj = metrics_payload.get("belief_transition") or {}
            transition_id = transition_obj.get("id")
        if transition_id:
            tracker["critical_transitions"].add(str(transition_id))
        for ev in metrics_payload.get("edge_evidence") or []:
            if isinstance(ev, dict):
                from_id = ev.get("from")
                to_id = ev.get("to")
                if from_id and to_id:
                    tracker["critical_transitions"].add(f"{from_id}->{to_id}")
        engagement_meta = metrics_payload.get("engagement") or {}
        asset_id = (
            metrics_payload.get("asset_id")
            or engagement_meta.get("asset_id")
            or engagement_meta.get("id")
        )
        if asset_id:
            asset_id = str(asset_id)
            tracker["asset_steps"] += 1
            asset_info = asset_metadata.get(asset_id) or {}
            category_label = asset_info.get("category_label")
            if category_label:
                tracker["asset_category_counts"][category_label] += 1
            delta_val = asset_delta_lookup.get(asset_id)
            if isinstance(delta_val, (int, float)):
                tracker["asset_success_total"] += float(delta_val)
                tracker["asset_success_count"] += 1
        channel_val = (
            engagement_meta.get("channel")
            or metrics_payload.get("channel")
            or engagement_meta.get("source")
            or getattr(step, "channel", None)
        )
        if channel_val:
            tracker["channel_counts"][str(channel_val).strip().lower()] += 1

    rows: List[Dict[str, Any]] = []
    for account_id, tracker in trackers.items():
        meta = dict(account_meta_lookup.get(account_id) or {})
        raw_meta = account_raw_meta.get(account_id) or {}
        for key in ("industry", "revenue_range", "employee_range", "geography", "funding_stage"):
            if not meta.get(key):
                value = raw_meta.get(key)
                if value:
                    meta[key] = value
        account_name = meta.get("account_name") or raw_meta.get("account_name")
        segments = tracker.get("segment_keys") or _segment_list(account_id)
        segments = [seg for seg in segments if seg]
        total_steps = tracker["total_steps"] or 0
        off_ratio = (
            float(tracker["off_path_steps"]) / float(total_steps)
            if total_steps
            else 0.0
        )
        asset_touch_ratio = (
            float(tracker["asset_steps"]) / float(total_steps)
            if total_steps
            else 0.0
        )
        persona_activation = len(tracker["unique_personas"])
        unique_stage_count = len(tracker["unique_stages"])
        critical_count = len(tracker["critical_transitions"]) or len(tracker["stage_transitions"])
        channel_diversity = len([ch for ch, cnt in tracker["channel_counts"].items() if cnt])
        dominant_asset_category = None
        if tracker["asset_category_counts"]:
            dominant_asset_category = max(
                tracker["asset_category_counts"].items(),
                key=lambda item: item[1],
            )[0]
        primary_channel = None
        if tracker["channel_counts"]:
            primary_channel = max(
                tracker["channel_counts"].items(),
                key=lambda item: item[1],
            )[0]
        asset_success_score = (
            tracker["asset_success_total"] / tracker["asset_success_count"]
            if tracker["asset_success_count"]
            else 0.0
        )
        duration_days = _mean_safe(tracker["duration_samples"])
        status_label = _normalize_deal_status(account_status_map.get(account_id))

        rows.append(
            {
                "account_id": account_id,
                "account_name": account_name,
                "label_raw": status_label,
                "label": 1 if status_label == "won" else 0 if status_label == "lost" else None,
                "feature_values": {
                    "industry": meta.get("industry"),
                    "revenue_range": meta.get("revenue_range"),
                    "employee_range": meta.get("employee_range"),
                    "geography": meta.get("geography"),
                    "funding_stage": meta.get("funding_stage"),
                    "primary_segment": segments[0] if segments else None,
                    "dominant_asset_category": dominant_asset_category,
                    "primary_channel": primary_channel,
                    "critical_transition_count": critical_count,
                    "off_path_ratio": off_ratio,
                    "journey_duration_days": duration_days,
                    "persona_activation": float(persona_activation),
                    "asset_touch_ratio": asset_touch_ratio,
                    "channel_diversity": float(channel_diversity),
                    "unique_stage_count": float(unique_stage_count),
                    "asset_success_score": asset_success_score,
                    "engagement_depth": float(total_steps),
                    "segment_count": float(len(segments)),
                },
            }
        )
    return rows


def _build_product_insights(
    *,
    account_meta_stats: Dict[str, Dict[str, Any]],
    meta_account_map: Optional[Dict[str, List[str]]],
    persona_stats: Dict[str, Dict[str, Any]],
    engaged_counter: Counter,
    persona_sequences_by_episode: Dict[str, List[str]],
    classification_counts: Counter,
    arsenal_impact: List[Dict[str, Any]],
    episode_durations: List[float],
    persona_node_info: Dict[str, Dict[str, Any]],
    rcs_paths: List[Dict[str, Any]],
    segment_persona_stats: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    segment_classification_counts: Optional[Dict[str, Counter]] = None,
    segment_sequences_by_segment: Optional[
        Dict[str, Dict[str, List[str]]]
    ] = None,
    segment_concern_counter: Optional[Dict[str, Counter]] = None,
    segment_stage_counter: Optional[Dict[str, Counter]] = None,
    persona_wolves_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    wolves_metrics_updated_at: Optional[str] = None,
    product_graph: Optional[nx.DiGraph] = None,
) -> Dict[str, Any]:
    MIN_SEGMENT_DECISIONS = 2
    MIN_SEGMENT_DELTA = 0.0
    SIGNIFICANCE_DELTA = 0.05
    KEYSTONE_MIN_SCORE = 0.35
    KEYSTONE_MIN_SAMPLE = 2

    insights = deepcopy(DEFAULT_PRODUCT_INSIGHTS)
    insights["ideal_customer_patterns"]["works_well"] = []
    insights["ideal_customer_patterns"]["gaps"] = []
    insights["persona_landscape"]["frequency"] = []
    insights["persona_landscape"]["critical_leads"] = []
    insights["persona_landscape"]["decision_personas"] = []
    insights["persona_landscape"]["blockers"] = []
    insights["persona_landscape"]["coalitions"] = []
    insights["belief_transitions"]["hardest"] = _insight_text("", source="default")
    insights["belief_transitions"]["easiest"] = _insight_text("", source="default")
    insights["belief_transitions"]["top_pains"] = []
    insights["asset_channel_effectiveness"]["high_assets"] = []
    insights["asset_channel_effectiveness"]["underperforming_channels"] = []
    insights["asset_channel_effectiveness"]["channel_persona_matches"] = []
    insights["journey_structure"]["common_paths"] = []
    insights["journey_structure"]["deviations"] = ""
    insights["journey_structure"]["average_duration_days"] = None
    insights["journey_structure"]["typical_path"] = []
    insights["global_patterns"]["biggest_barrier"] = _insight_text("", source="default")
    insights["global_patterns"]["hidden_blocker"] = _insight_text("", source="default")
    insights["global_patterns"]["missed_opportunity"] = _insight_text("", source="default")
    insights["product_strengths"]["strengths"] = []
    insights["product_strengths"]["weaknesses"] = []
    insights["strategic_moves"]["segment_priorities"] = ""
    insights["strategic_moves"]["persona_priorities"] = ""
    insights["strategic_moves"]["asset_priorities"] = ""
    insights["strategic_moves"]["channel_priorities"] = ""
    insights["segment_patterns"] = {}
    insights["keystone_personas"] = []
    insights["wolves_metrics_updated_at"] = wolves_metrics_updated_at

    segment_persona_stats = segment_persona_stats or {}
    segment_classification_counts = segment_classification_counts or {}
    segment_sequences_by_segment = segment_sequences_by_segment or {}
    segment_concern_counter = segment_concern_counter or {}
    segment_stage_counter = segment_stage_counter or {}
    persona_wolves_metrics = persona_wolves_metrics or {}

    meta_account_map = meta_account_map or {}

    canonical_lookup, canonical_labels, canonical_node_map = _canonical_persona_lookup(
        product_graph
    )
    canonical_persona_stats = _aggregate_persona_stats(persona_stats, canonical_lookup)
    canonical_persona_sequences = {
        account_id: [
            canonical_lookup.get(persona_id, persona_id) for persona_id in seq if persona_id
        ]
        for account_id, seq in persona_sequences_by_episode.items()
    }

    if product_graph:
        for canonical_id, node_id in canonical_node_map.items():
            info = persona_node_info.setdefault(canonical_id, {})
            node_label = canonical_labels.get(
                canonical_id, _node_data_label(product_graph, node_id)
            )
            if not info.get("label"):
                info["label"] = node_label
            node_data = product_graph.nodes.get(node_id, {})
            info.setdefault("perceptibility", node_data.get("perceptibility"))
            info.setdefault("proximity", node_data.get("proximity"))

    channel_scores: Dict[str, List[float]] = defaultdict(list)
    channel_persona_counts: Counter[Tuple[str, str]] = Counter()
    channel_persona_deltas: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    concern_counter: Counter[str] = Counter()
    stage_counter: Counter[str] = Counter()
    segment_concern_aggregate: Dict[str, Counter[str]] = defaultdict(Counter)
    segment_stage_aggregate: Dict[str, Counter[str]] = defaultdict(Counter)

    def _persona_label(pid: str) -> str:
        info = persona_node_info.get(pid) or {}
        label = info.get("label")
        if isinstance(label, str) and label.strip():
            return label
        return _fallback_persona_label(pid)

    def _fmt_metric(value: Optional[float]) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.2f}"
        except Exception:
            return "—"

    def _persona_line(pid: str, *, involvement_share: Optional[float] = None) -> str:
        stats = canonical_persona_stats.get(pid, {})
        if involvement_share is None:
            observed = stats.get("observed", 0)
            total = total_persona_events or 1
            involvement_share = observed / total if total else 0.0
        info = persona_node_info.get(pid) or {}
        perc = info.get("perceptibility")
        prox = info.get("proximity")
        if perc is None:
            perc = _safe_ratio(
                stats.get("hit_at_1", 0) + stats.get("hit_at_3", 0),
                max(stats.get("observed", 0), 1),
            )
        if prox is None:
            prox = 1.0 - _safe_ratio(
                stats.get("off_path", 0),
                max(stats.get("observed", 0), 1),
            )
        return (
            f"{_persona_label(pid)} — Perc {_fmt_metric(perc)} | "
            f"Prox {_fmt_metric(prox)} | Involvement {_percent(involvement_share or 0.0, 0)}"
        )

    total_persona_events = (
        sum(stats.get("observed", 0) for stats in canonical_persona_stats.values()) or 1
    )
    observed_counts = [
        stats.get("observed", 0) for stats in canonical_persona_stats.values()
    ]
    max_persona_observed = max(observed_counts) if observed_counts else 1

    def _typical_step_payload(
        pid: str,
        idx: int,
        *,
        highlight: Optional[str] = None,
        path_probability: Optional[float] = None,
        override_perceptibility: Optional[float] = None,
        override_proximity: Optional[float] = None,
        override_involvement: Optional[float] = None,
        source: InsightSource = "data",
    ) -> Dict[str, Any]:
        stats = canonical_persona_stats.get(pid, {})
        observed = stats.get("observed", 0)
        if isinstance(override_involvement, (int, float)):
            involvement_share = float(override_involvement)
        else:
            involvement_share = (
                observed / total_persona_events if total_persona_events else 0.0
            )
        involvement_share = max(0.0, min(1.0, involvement_share))
        info = persona_node_info.get(pid) or {}
        perc_val = (
            float(override_perceptibility)
            if isinstance(override_perceptibility, (int, float))
            else info.get("perceptibility")
        )
        prox_val = (
            float(override_proximity)
            if isinstance(override_proximity, (int, float))
            else info.get("proximity")
        )
        if perc_val is None:
            perc_val = _safe_ratio(
                (stats.get("hit_at_1", 0) or 0) + (stats.get("hit_at_3", 0) or 0),
                observed or 1,
            )
        if prox_val is None:
            prox_val = 1.0 - _safe_ratio(stats.get("off_path", 0), observed or 1)

        hit1_ratio = _safe_ratio(stats.get("hit_at_1", 0), observed)
        hit3_ratio = _safe_ratio(stats.get("hit_at_3", 0), observed)
        off_ratio_local = _safe_ratio(stats.get("off_path", 0), observed)

        confidence_components: List[float] = []
        if observed and max_persona_observed:
            confidence_components.append(observed / max_persona_observed)
        if hit3_ratio:
            confidence_components.append(hit3_ratio)
        if path_probability is not None:
            try:
                prob_val = float(path_probability)
                if prob_val >= 0:
                    confidence_components.append(max(0.0, min(1.0, prob_val)))
            except (TypeError, ValueError):
                pass

        confidence = 0.0
        if confidence_components:
            confidence = sum(confidence_components) / len(confidence_components)
        confidence = max(0.05, min(0.95, round(confidence, 3)))

        reason_bits: List[str] = []
        if hit1_ratio >= 0.5:
            reason_bits.append(f"Hit@1 {_percent(hit1_ratio, 0)} alignment")
        elif hit3_ratio >= 0.6:
            reason_bits.append(f"Hit@3 {_percent(hit3_ratio, 0)} coverage")
        if off_ratio_local >= 0.25:
            reason_bits.append(f"{_percent(off_ratio_local, 0)} stall risk")
        elif involvement_share >= 0.2:
            reason_bits.append(f"Appears in {_percent(involvement_share, 0)} of journeys")
        if path_probability is not None:
            try:
                prob_pct = _percent(float(path_probability), 0)
                reason_bits.append(f"{prob_pct} path weight")
            except Exception:
                pass

        impact_text = (
            f"Perc {_fmt_metric(perc_val)} | Prox {_fmt_metric(prox_val)} | "
            f"Involvement {_percent(involvement_share, 0)}"
        )
        fatigue_numerator = max(
            0.0,
            float(observed)
            - float(stats.get("hit_at_1", 0) + stats.get("hit_at_3", 0)),
        )
        fatigue = _safe_ratio(fatigue_numerator, observed or 1)
        fatigue = max(0.0, min(0.95, fatigue))

        payload: Dict[str, Any] = {
            "step": idx + 1,
            "persona_id": pid,
            "persona": _persona_label(pid),
            "persona_label": _persona_label(pid),
            "highlight": highlight,
            "impact": impact_text,
            "confidence": confidence,
            "perceptibility": perc_val,
            "proximity": prox_val,
            "involvement": involvement_share,
            "fatigue": round(fatigue, 3),
            "source": source,
        }
        if reason_bits:
            payload["reason"] = " • ".join(reason_bits)
            if not payload["highlight"]:
                payload["highlight"] = reason_bits[0]
        if not payload["highlight"]:
            payload["highlight"] = "Key activation step"
        return payload

    # Section A - Ideal customer patterns
    meta_rows: List[Tuple[str, float, Optional[float], int]] = []
    backlog_rows: List[Tuple[str, int]] = []
    total_wins = 0
    total_decisions = 0
    for meta, stats in account_meta_stats.items():
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        total = wins + losses
        if total < 1:
            pipeline = stats.get("open", 0)
            if pipeline:
                backlog_rows.append((meta, pipeline))
            continue
        rate = _safe_ratio(wins, total)
        avg_duration = _mean_safe(stats.get("durations", []))
        meta_rows.append((meta, rate, avg_duration, total))
        total_wins += wins
        total_decisions += total
    overall_rate = _safe_ratio(total_wins, total_decisions) if total_decisions else 0.0

    if meta_rows:
        predictive_rows = [
            row
            for row in meta_rows
            if row[3] >= MIN_SEGMENT_DECISIONS
        ]

        def _dedupe_rows(rows: List[Tuple[str, float, Optional[float], int]]) -> List[Tuple[str, float, Optional[float], int]]:
            seen = set()
            deduped = []
            for row in rows:
                if row[0] in seen:
                    continue
                deduped.append(row)
                seen.add(row[0])
            return deduped

        positive_rows = _dedupe_rows(
            sorted(predictive_rows, key=lambda row: row[1], reverse=True)
        )[:3]
        negative_rows = _dedupe_rows(
            sorted(predictive_rows, key=lambda row: row[1])
        )
        negative_rows = [
            row for row in negative_rows if row[0] not in {meta for meta, _, _, _ in positive_rows[:3]}
        ][:3]
        works_lines: List[str] = []
        works_items: List[Dict[str, Any]] = []
        for meta, rate, avg_duration, total in positive_rows[:3]:
            delta = rate - overall_rate
            is_significant = abs(delta) >= SIGNIFICANCE_DELTA
            duration_text = ""
            if avg_duration:
                duration_text = f" and typically close in {avg_duration:.1f} days"
            label = _format_meta_value(meta)
            qualifier = "" if is_significant else " (low signal)"
            text = (
                f"{label} convert at {_percent(rate, 1)} (Δ {delta:+.1%}){duration_text}{qualifier}."
            )
            works_lines.append(text)
            stats = account_meta_stats.get(meta, {})
            signal_count = (
                stats.get("wins", 0)
                + stats.get("losses", 0)
                + stats.get("open", 0)
            )
            confidence_samples = stats.get("meta_confidence_samples") or 0
            avg_meta_conf = None
            if confidence_samples:
                avg_meta_conf = (
                    (stats.get("meta_confidence_weight", 0.0) or 0.0)
                    / confidence_samples
                )
            works_items.append(
                {
                    "text": text,
                    "label": label,
                    "meta_key": meta,
                    "signals": signal_count,
                    "support": {
                        "wins": stats.get("wins", 0),
                        "losses": stats.get("losses", 0),
                        "open": stats.get("open", 0),
                        "avg_duration_days": avg_duration,
                        "meta_confidence": avg_meta_conf,
                        "accounts": meta_account_map.get(meta, [])[:6],
                        "is_significant": is_significant,
                    },
                }
            )
        if works_lines:
            insights["ideal_customer_patterns"]["works_well"] = works_lines
            insights["ideal_customer_patterns"]["works_well_items"] = works_items
        else:
            fallback_text = (
                "Segments observed, but none exceed the ±5 pt lift threshold yet — collect more wins/losses for clearer signal."
            )
            insights["ideal_customer_patterns"]["works_well"] = [fallback_text]
            insights["ideal_customer_patterns"]["works_well_items"] = [
                {
                    "text": fallback_text,
                    "label": fallback_text,
                    "meta_key": None,
                    "signals": None,
                    "support": {},
                }
            ]

        gaps_lines: List[str] = []
        gaps_items: List[Dict[str, Any]] = []
        for meta, rate, avg_duration, total in negative_rows[:3]:
            delta = overall_rate - rate
            is_significant = abs(delta) >= SIGNIFICANCE_DELTA
            duration_text = ""
            if avg_duration:
                duration_text = f"; average cycle stretches to {avg_duration:.1f} days"
            label = _format_meta_value(meta)
            qualifier = "" if is_significant else " (low signal)"
            text = (
                f"{label} underperform with only {_percent(rate, 1)} conversion (Δ -{delta:.1%}){duration_text}{qualifier}."
            )
            gaps_lines.append(text)
            stats = account_meta_stats.get(meta, {})
            signal_count = (
                stats.get("wins", 0)
                + stats.get("losses", 0)
                + stats.get("open", 0)
            )
            confidence_samples = stats.get("meta_confidence_samples") or 0
            avg_meta_conf = None
            if confidence_samples:
                avg_meta_conf = (
                    (stats.get("meta_confidence_weight", 0.0) or 0.0)
                    / confidence_samples
                )
            gaps_items.append(
                {
                    "text": text,
                    "label": label,
                    "meta_key": meta,
                    "signals": signal_count,
                    "support": {
                        "wins": stats.get("wins", 0),
                        "losses": stats.get("losses", 0),
                        "open": stats.get("open", 0),
                        "avg_duration_days": avg_duration,
                        "meta_confidence": avg_meta_conf,
                        "accounts": meta_account_map.get(meta, [])[:6],
                        "is_significant": is_significant,
                    },
                }
            )
        if gaps_lines:
            insights["ideal_customer_patterns"]["gaps"] = gaps_lines
            insights["ideal_customer_patterns"]["gaps_items"] = gaps_items
        else:
            fallback_text = (
                "No underperforming segments cleared the ±5 pt gap threshold — gather more losing deals."
            )
            insights["ideal_customer_patterns"]["gaps"] = [fallback_text]
            insights["ideal_customer_patterns"]["gaps_items"] = [
                {
                    "text": fallback_text,
                    "label": fallback_text,
                    "meta_key": None,
                    "signals": None,
                    "support": {},
                }
            ]
            insights["ideal_customer_patterns"]["gaps_items"] = []
    elif arsenal_impact:
        meta_effect: Dict[str, List[float]] = defaultdict(list)
        meta_counts: Dict[str, int] = defaultdict(int)
        for row in arsenal_impact:
            delta = row.get("avg_delta")
            if delta is None:
                continue
            for meta in row.get("account_meta") or []:
                key = str(meta)
                meta_effect[key].append(delta)
                meta_counts[key] += 1
        scored_rows: List[Tuple[str, float, int]] = []
        for meta, samples in meta_effect.items():
            avg = _mean_safe(samples) or 0.0
            scored_rows.append((meta, avg, meta_counts.get(meta, len(samples))))
        pos = [row for row in scored_rows if row[1] > 0]
        neg = [row for row in scored_rows if row[1] < 0]
        pos.sort(key=lambda row: row[1], reverse=True)
        neg.sort(key=lambda row: row[1])
        if pos:
            insights["ideal_customer_patterns"]["works_well"] = [
                f"{_format_meta_value(meta)} shows average belief lift of {(avg * 100):.1f} bps across {count} signals."
                for meta, avg, count in pos[:3]
            ]
            insights["ideal_customer_patterns"]["works_well_items"] = [
                {
                    "text": f"{_format_meta_value(meta)} shows average belief lift of {(avg * 100):.1f} bps across {count} signals.",
                    "label": _format_meta_value(meta),
                    "meta_key": meta,
                    "signals": count,
                    "support": {
                        "wins": None,
                        "losses": None,
                        "open": None,
                        "avg_duration_days": None,
                        "accounts": meta_account_map.get(meta, [])[:6],
                    },
                }
                for meta, avg, count in pos[:3]
            ]
        if neg:
            insights["ideal_customer_patterns"]["gaps"] = [
                f"{_format_meta_value(meta)} drags belief by {(abs(avg) * 100):.1f} bps across {count} signals."
                for meta, avg, count in neg[:3]
            ]
            insights["ideal_customer_patterns"]["gaps_items"] = [
                {
                    "text": f"{_format_meta_value(meta)} drags belief by {(abs(avg) * 100):.1f} bps across {count} signals.",
                    "label": _format_meta_value(meta),
                    "meta_key": meta,
                    "signals": count,
                    "support": {
                        "wins": None,
                        "losses": None,
                        "open": None,
                        "avg_duration_days": None,
                        "accounts": meta_account_map.get(meta, [])[:6],
                    },
                }
                for meta, avg, count in neg[:3]
            ]
    if (
        not insights["ideal_customer_patterns"]["works_well"]
        and backlog_rows
    ):
        backlog_rows.sort(key=lambda row: row[1], reverse=True)
        insights["ideal_customer_patterns"]["works_well"] = [
            f"{_format_meta_value(meta)} show active momentum with {count} open journeys awaiting resolution."
            for meta, count in backlog_rows[:3]
        ]
    if not insights["ideal_customer_patterns"]["works_well"]:
        insights["ideal_customer_patterns"]["works_well"] = [
            "Not enough closed-won journeys yet to identify strong-fit account segments."
        ]
    if not insights["ideal_customer_patterns"]["gaps"]:
        insights["ideal_customer_patterns"]["gaps"] = [
            "No underperforming segments detected yet — continue ingesting more deals."
        ]
    if "works_well_items" not in insights["ideal_customer_patterns"]:
        insights["ideal_customer_patterns"]["works_well_items"] = [
            {
                "text": line,
                "label": line,
                "meta_key": None,
                "signals": None,
                "support": {},
            }
            for line in insights["ideal_customer_patterns"]["works_well"]
        ]
    if "gaps_items" not in insights["ideal_customer_patterns"]:
        insights["ideal_customer_patterns"]["gaps_items"] = [
            {
                "text": line,
                "label": line,
                "meta_key": None,
                "signals": None,
                "support": {},
            }
            for line in insights["ideal_customer_patterns"]["gaps"]
        ]

    # Section B - Persona landscape
    persona_counts = [
        (pid, stats.get("observed", 0))
        for pid, stats in canonical_persona_stats.items()
        if stats.get("observed", 0)
    ]

    def _dedupe_persona_rows(rows: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        seen: set[str] = set()
        deduped: List[Tuple[str, float]] = []
        for pid, value in rows:
            label = _persona_label(pid).strip().lower()
            if not label or label in seen:
                continue
            deduped.append((pid, value))
            seen.add(label)
        return deduped

    persona_counts.sort(key=lambda row: row[1], reverse=True)
    top_personas = _dedupe_persona_rows(persona_counts)[:5]
    if top_personas:
        insights["persona_landscape"]["frequency"] = [
            _persona_line(pid, involvement_share=count / total_persona_events)
            for pid, count in top_personas
        ]

    def _persona_ratio(key: str) -> List[Tuple[str, float]]:
        rows = []
        for pid, stats in canonical_persona_stats.items():
            observed = stats.get("observed", 0)
            if not observed:
                continue
            value = stats.get(key, 0)
            ratio = _safe_ratio(value, observed)
            rows.append((pid, ratio))
        rows.sort(key=lambda row: row[1], reverse=True)
        return rows

    lead_rows = _dedupe_persona_rows(_persona_ratio("hit_at_1"))
    if lead_rows:
        insights["persona_landscape"]["critical_leads"] = [
            f"{_persona_line(pid)} — Hit@1 {_percent(score, 0)}"
            for pid, score in lead_rows[:3]
        ]

    decision_rows = _dedupe_persona_rows(_persona_ratio("hit_at_3"))
    if decision_rows:
        insights["persona_landscape"]["decision_personas"] = [
            f"{_persona_line(pid)} — Hit@3 {_percent(score, 0)}"
            for pid, score in decision_rows[:3]
        ]

    blocker_rows = []
    for pid, stats in canonical_persona_stats.items():
        observed = stats.get("observed", 0)
        if not observed:
            continue
        off_ratio = _safe_ratio(stats.get("off_path", 0), observed)
        blocker_rows.append((pid, off_ratio))
    blocker_rows = _dedupe_persona_rows(blocker_rows)
    blocker_rows.sort(key=lambda row: row[1], reverse=True)
    if blocker_rows:
        insights["persona_landscape"]["blockers"] = [
            f"{_persona_line(pid)} — Stall {_percent(score, 0)}"
            for pid, score in blocker_rows[:3]
        ]

    pair_counts: Counter = Counter()
    for seq in canonical_persona_sequences.values():
        if len(seq) < 2:
            continue
        for a, b in zip(seq, seq[1:]):
            if a == b:
                continue
            pair_counts[(a, b)] += 1
    if pair_counts:
        insights["persona_landscape"]["coalitions"] = [
            f"{_persona_label(a)} + {_persona_label(b)} co-activate in {count} journeys."
            for (a, b), count in pair_counts.most_common(3)
        ]

    # Section C - Belief transitions (we rely on defaults unless we have explicit signals)
    total_classifications = sum(classification_counts.values()) or 0
    off_ratio = _safe_ratio(classification_counts.get("off_path", 0), total_classifications)
    expected_share = _safe_ratio(classification_counts.get("expected", 0), total_classifications)
    jump_share = _safe_ratio(
        classification_counts.get("jump_ahead", 0) + classification_counts.get("near_path", 0),
        total_classifications,
    )

    # Section D - Asset & channel effectiveness
    if arsenal_impact:
        sorted_assets = sorted(
            arsenal_impact,
            key=lambda row: abs(row.get("total_delta") or 0.0),
            reverse=True,
        )
        insights["asset_channel_effectiveness"]["high_assets"] = [
            f"{row.get('asset_label') or row.get('asset_id')} lifts belief by {(row.get('avg_delta') or 0.0) * 100:.1f} bps."
            for row in sorted_assets[:3]
        ]
        for row in arsenal_impact:
            row_segments = row.get("segment_keys") or []
            concerns_payload = row.get("concerns") or []
            if isinstance(concerns_payload, list):
                for item in concerns_payload:
                    if isinstance(item, dict):
                        label = item.get("label")
                        count = item.get("count") or 0
                    else:
                        label = str(item)
                        count = 0
                    if label:
                        concern_counter[label] += int(count) if count else 1
                        for seg in row_segments:
                            segment_concern_aggregate[seg][label] += int(count) if count else 1
            stages_payload = row.get("stages") or []
            if isinstance(stages_payload, list):
                for entry in stages_payload:
                    if isinstance(entry, dict):
                        stage_code = entry.get("code")
                        count = entry.get("count") or 0
                    else:
                        stage_code = entry
                        count = 0
                    if stage_code:
                        increment = int(count) if count else 1
                        stage_counter[str(stage_code)] += increment
                        for seg in row_segments:
                            segment_stage_aggregate[seg][str(stage_code)] += increment

        for row in arsenal_impact:
            delta = row.get("avg_delta")
            if delta is None:
                continue
            channels = row.get("channels") or []
            personas = row.get("persona_labels") or row.get("persona_ids") or []
            persona_labels = [
                _format_meta_value(str(persona))
                for persona in personas
                if persona
            ]
            num_engagements = row.get("num_engagements") or 1
            for ch in channels:
                channel_name = str(ch).strip()
                if not channel_name:
                    continue
                pretty_channel = channel_name.replace("_", " ").title()
                channel_scores[pretty_channel].append(delta)
                for persona in persona_labels:
                    key = (pretty_channel, persona)
                    channel_persona_counts[key] += num_engagements
                    channel_persona_deltas[key].append(delta)
        if channel_scores:
            channel_avg = [
                (ch, _mean_safe(vals) or 0.0) for ch, vals in channel_scores.items()
            ]
            channel_avg.sort(key=lambda row: row[1])
            insights["asset_channel_effectiveness"]["underperforming_channels"] = [
                f"{ch} averages {(avg * 100):.1f} bps impact."
                for ch, avg in channel_avg[:3]
            ]
        if channel_persona_counts:
            combo_rows: List[Tuple[str, str, float, int]] = []
            for key, count in channel_persona_counts.items():
                ch, persona = key
                avg_delta = _mean_safe(channel_persona_deltas.get(key, [])) or 0.0
                combo_rows.append((ch, persona, avg_delta, count))
            combo_rows.sort(key=lambda row: (row[2], row[3]), reverse=True)
            insights["asset_channel_effectiveness"]["channel_persona_matches"] = [
                f"{ch} → {persona} ({count} engagements, {delta * 100:.1f} bps)"
                for ch, persona, delta, count in combo_rows[:3]
            ]

    if concern_counter:
        total_concerns = sum(concern_counter.values()) or 1
        pains: List[Dict[str, Any]] = []
        for label, count in concern_counter.most_common(4):
            share = _safe_ratio(count, total_concerns)
            pains.append(
                _insight_text(
                    f"{label} ({_percent(share, 0)} of signals)",
                    source="data",
                    confidence=share,
                    extra={"label": label, "share": share},
                )
            )
        insights["belief_transitions"]["top_pains"] = pains

    if stage_counter:
        stage_total = sum(stage_counter.values()) or 1
        stage_code, stage_count = stage_counter.most_common(1)[0]
        stage_label = _stage_label(stage_code)
        hardest_parts = [
            f"{stage_label} absorbs {_percent(_safe_ratio(stage_count, stage_total), 0)} of concern signals"
        ]
        if off_ratio:
            hardest_parts.append(f"{_percent(off_ratio, 0)} of steps still drift off-path")
        hardness_conf = _safe_ratio(stage_count, stage_total)
        insights["belief_transitions"]["hardest"] = _insight_text(
            f"{' while '.join(hardest_parts)}.",
            source="data",
            confidence=hardness_conf,
            extra={"stage": stage_code, "share": hardness_conf},
        )
    elif off_ratio:
        insights["belief_transitions"]["hardest"] = _insight_text(
            f"Off-path detours account for {_percent(off_ratio, 0)} of observed steps, indicating the toughest shift occurs mid-journey.",
            source="data",
            confidence=off_ratio,
        )

    if expected_share:
        insights["belief_transitions"]["easiest"] = _insight_text(
            f"On-path progress holds steady across {_percent(expected_share, 0)} of recorded steps.",
            source="data",
            confidence=expected_share,
        )

    if jump_share:
        insights["journey_structure"]["deviations"] = (
            f"Jump-ahead sequences appear in {_percent(jump_share, 0)} of engagements — monitor fast-track personas."
        )
    if not insights["belief_transitions"]["hardest"].get("text"):
        insights["belief_transitions"]["hardest"] = deepcopy(
            DEFAULT_PRODUCT_INSIGHTS["belief_transitions"]["hardest"]
        )
    if not insights["belief_transitions"]["easiest"].get("text"):
        insights["belief_transitions"]["easiest"] = deepcopy(
            DEFAULT_PRODUCT_INSIGHTS["belief_transitions"]["easiest"]
        )
    if not insights["belief_transitions"]["top_pains"]:
        insights["belief_transitions"]["top_pains"] = deepcopy(
            DEFAULT_PRODUCT_INSIGHTS["belief_transitions"]["top_pains"]
        )

    # Section E - Journey structure
    if persona_sequences_by_episode:
        triple_counts: Counter = Counter()
        for seq in persona_sequences_by_episode.values():
            if len(seq) < 3:
                continue
            for idx in range(len(seq) - 2):
                triple = tuple(seq[idx : idx + 3])
                triple_counts[triple] += 1
        if triple_counts:
            insights["journey_structure"]["common_paths"] = [
                " → ".join(_persona_label(pid) for pid in triple)
                for triple, _ in triple_counts.most_common(3)
            ]
    if persona_sequences_by_episode:
        sequence_counter: Counter[Tuple[str, ...]] = Counter()
        for seq in persona_sequences_by_episode.values():
            cleaned = [pid for pid in seq if pid]
            if len(cleaned) >= 2:
                sequence_counter[tuple(cleaned)] += 1
        if sequence_counter:
            top_sequence = list(sequence_counter.most_common(1)[0][0])
            typical_steps = [
                _typical_step_payload(pid, idx, source="data")
                for idx, pid in enumerate(top_sequence)
            ]
            if typical_steps:
                insights["journey_structure"]["typical_path"] = typical_steps

    if not insights["journey_structure"]["common_paths"]:
        fallback_paths: List[str] = []
        for path in rcs_paths[:3]:
            personas = path.get("personas") or path.get("path") or []
            if not personas:
                continue
            labels: List[str] = []
            for entry in personas:
                pid = entry.get("id") if isinstance(entry, dict) else entry
                if not pid:
                    continue
                labels.append(_persona_label(pid))
            if labels:
                fallback_paths.append(" → ".join(labels))
        if fallback_paths:
            insights["journey_structure"]["common_paths"] = fallback_paths

    if not insights["journey_structure"].get("typical_path"):
        typical_steps: List[Dict[str, Any]] = []
        if rcs_paths:
            best_path = rcs_paths[0]
            personas = best_path.get("personas") or best_path.get("path") or []
            for idx, entry in enumerate(personas):
                if isinstance(entry, dict):
                    pid = entry.get("id")
                    perc = entry.get("perceptibility")
                    prox = entry.get("proximity")
                    highlight = entry.get("phase") or entry.get("stage")
                    probability = entry.get("probability") or entry.get("score")
                    involvement = entry.get("involvement")
                else:
                    pid = entry
                    perc = prox = None
                    highlight = None
                    probability = None
                    involvement = None
                if not pid:
                    continue
                payload = _typical_step_payload(
                    str(pid),
                    idx,
                    highlight=highlight,
                    path_probability=probability,
                    override_perceptibility=perc,
                    override_proximity=prox,
                    override_involvement=involvement,
                    source="graph",
                )
                typical_steps.append(payload)
        if typical_steps:
            insights["journey_structure"]["typical_path"] = typical_steps
    if (
        not insights["journey_structure"].get("typical_path")
        and persona_sequences_by_episode
    ):
        longest_seq = max(
            persona_sequences_by_episode.values(),
            key=lambda seq: len(seq),
        )
        fallback_steps: List[Dict[str, Any]] = []
        for idx, pid in enumerate(longest_seq):
            if not pid:
                continue
            payload = _typical_step_payload(str(pid), idx, source="data")
            fallback_steps.append(payload)
        if fallback_steps:
            insights["journey_structure"]["typical_path"] = fallback_steps
    avg_duration = _mean_safe(episode_durations)
    if avg_duration:
        insights["journey_structure"]["average_duration_days"] = round(avg_duration, 1)

    if concern_counter:
        total_concerns = sum(concern_counter.values()) or 1
        top_label, top_count = concern_counter.most_common(1)[0]
        concern_share = _safe_ratio(top_count, total_concerns)
        insights["global_patterns"]["biggest_barrier"] = _insight_text(
            f"{top_label} surfaces in {_percent(concern_share, 0)} of flagged engagements.",
            source="data",
            confidence=concern_share,
            extra={"label": top_label, "entity_type": "concern"},
        )
    if blocker_rows:
        blocker_id, blocker_ratio = blocker_rows[0]
        insights["global_patterns"]["hidden_blocker"] = _insight_text(
            f"{_format_meta_value(blocker_id)} shows the highest stall rate at {_percent(blocker_ratio, 0)}.",
            source="data",
            confidence=blocker_ratio,
            extra={"label": blocker_id, "entity_type": "persona"},
        )

    missed_candidates: List[Tuple[str, float, float]] = []
    total_persona_events = sum(engaged_counter.values()) or 1
    for pid, stats in persona_stats.items():
        observed = stats.get("observed", 0)
        if observed < 3:
            continue
        hit_rate = _safe_ratio(stats.get("hit_at_1", 0), observed)
        frequency = _safe_ratio(observed, total_persona_events)
        if hit_rate >= 0.45 and frequency < 0.25:
            missed_candidates.append((pid, hit_rate, frequency))
    if missed_candidates:
        missed_candidates.sort(key=lambda row: (row[1], -row[2]), reverse=True)
        pid, hit_rate, frequency = missed_candidates[0]
        insights["global_patterns"]["missed_opportunity"] = _insight_text(
            f"{_format_meta_value(pid)} convert at {_percent(hit_rate, 0)} when engaged yet appear in only {_percent(frequency, 0)} of engagements.",
            source="data",
            confidence=hit_rate,
            extra={"persona_id": pid, "entity_type": "persona"},
        )
    for key in ("biggest_barrier", "hidden_blocker", "missed_opportunity"):
        if not insights["global_patterns"][key].get("text"):
            fallback = deepcopy(DEFAULT_PRODUCT_INSIGHTS["global_patterns"][key])
            fallback.setdefault("extra", {}).setdefault("entity_type", "default")
            insights["global_patterns"][key] = fallback

    # Section G - strengths/weaknesses from persona stats
    strength_rows = sorted(
        decision_rows,
        key=lambda row: row[1],
        reverse=True,
    )
    if strength_rows:
        insights["product_strengths"]["strengths"] = [
            f"{_format_meta_value(pid)} drive late-stage belief ({_percent(score, 0)} success rate)."
            for pid, score in strength_rows[:3]
        ]
    if blocker_rows:
        insights["product_strengths"]["weaknesses"] = [
            f"{_format_meta_value(pid)} frequently stall journeys ({_percent(score, 0)} off-path)."
            for pid, score in blocker_rows[:3]
        ]

    # Section H - strategic moves (derive from earlier sections)
    if meta_rows:
        top_meta = max(meta_rows, key=lambda row: row[1])[0]
        insights["strategic_moves"]["segment_priorities"] = (
            f"Prioritize { _format_meta_value(top_meta) } accounts based on observed win rates."
        )
    if lead_rows:
        insights["strategic_moves"]["persona_priorities"] = (
            f"Focus early enablement on {', '.join(_format_meta_value(pid) for pid, _ in lead_rows[:3])}."
        )

    if arsenal_impact:
        top_asset = sorted_assets[0]
        insights["strategic_moves"]["asset_priorities"] = (
            f"Scale {top_asset.get('asset_label') or top_asset.get('asset_id')} which leads current belief lifts."
        )
    if channel_scores:
        best_channel = max(channel_scores.items(), key=lambda item: _mean_safe(item[1]) or 0.0)[0]
        insights["strategic_moves"]["channel_priorities"] = (
            f"Double down on {best_channel} where belief lift is strongest."
        )

    if segment_persona_stats:
        segment_patterns: Dict[str, Any] = {}
        for segment_key, persona_map in segment_persona_stats.items():
            signal_total = sum(
                stats.get("observed", 0) for stats in persona_map.values()
            )
            if signal_total < 8:
                continue
            freq_rows = []
            for pid, stats in persona_map.items():
                observed = stats.get("observed", 0)
                if not observed:
                    continue
                freq_rows.append(
                    {
                        "persona_id": pid,
                        "label": _persona_label(pid),
                        "observed": observed,
                        "share": _safe_ratio(observed, signal_total),
                    }
                )
            freq_rows.sort(key=lambda row: row["observed"], reverse=True)
            freq_rows = freq_rows[:5]

            seg_counts = segment_classification_counts.get(segment_key, Counter())
            seg_total = sum(seg_counts.values()) or 0
            hardest_text = ""
            easiest_text = ""
            if seg_total:
                seg_off_ratio = _safe_ratio(seg_counts.get("off_path", 0), seg_total)
                seg_expected = _safe_ratio(seg_counts.get("expected", 0), seg_total)
                if seg_off_ratio:
                    hardest_text = (
                        f"{_percent(seg_off_ratio, 0)} of steps drift off-path."
                    )
                if seg_expected:
                    easiest_text = (
                        f"{_percent(seg_expected, 0)} of steps remain on track."
                    )
            stage_counts = segment_stage_aggregate.get(segment_key, Counter())
            if stage_counts:
                stage_total = sum(stage_counts.values()) or 1
                stage_code, stage_count = stage_counts.most_common(1)[0]
                stage_label = _stage_label(stage_code)
                hardest_text = (
                    f"{stage_label} absorbs {_percent(_safe_ratio(stage_count, stage_total), 0)} of concern signals."
                )

            seg_concerns = segment_concern_aggregate.get(segment_key, Counter())
            pains_payload = []
            if seg_concerns:
                seg_total_concerns = sum(seg_concerns.values()) or 1
                pains_payload = [
                    {
                        "label": label,
                        "share": _safe_ratio(count, seg_total_concerns),
                    }
                    for label, count in seg_concerns.most_common(4)
                ]

            seq_map = segment_sequences_by_segment.get(segment_key, {})
            seg_triple_counts: Counter = Counter()
            for seq in seq_map.values():
                if len(seq) < 3:
                    continue
                for idx in range(len(seq) - 2):
                    triple = tuple(seq[idx : idx + 3])
                    seg_triple_counts[triple] += 1
            common_paths_payload = [
                {
                    "personas": [
                        {"persona_id": pid, "label": _persona_label(pid)}
                        for pid in triple
                    ],
                    "count": count,
                }
                for triple, count in seg_triple_counts.most_common(3)
            ]

            segment_patterns[segment_key] = {
                "label": segment_label_from_key(segment_key),
                "signals": signal_total,
                "persona_frequency": freq_rows,
                "hardest": hardest_text,
                "easiest": easiest_text,
                "top_pains": pains_payload,
                "common_paths": common_paths_payload,
                "source": "data",
                "confidence": round(min(0.95, signal_total / 25.0), 3),
            }
        insights["segment_patterns"] = segment_patterns
    else:
        insights["segment_patterns"] = {}

    def _segment_payload_from_key(key: Optional[str]) -> Optional[Dict[str, Any]]:
        if not key:
            return None
        payload = {
            "key": key,
            "label": segment_label_from_key(key),
        }
        for token in key.split("|"):
            if "=" not in token:
                continue
            field, value = token.split("=", 1)
            payload[field] = value
        return payload

    chain_cache: Dict[str, Dict[str, Any]] = {}

    def _persona_chain_targets(pid: str) -> Dict[str, Any]:
        if pid in chain_cache:
            return chain_cache[pid]
        follow_counts: Counter[Tuple[str, ...]] = Counter()
        for seq in persona_sequences_by_episode.values():
            if not seq:
                continue
            for idx, current in enumerate(seq):
                if current != pid:
                    continue
                tail = tuple(seq[idx + 1 : idx + 4])
                if tail:
                    follow_counts[tail] += 1
        if not follow_counts:
            chain_cache[pid] = {}
            return {}
        top_sequence, top_count = follow_counts.most_common(1)[0]
        total = sum(follow_counts.values()) or 1
        labels = [_persona_label(node_id) for node_id in top_sequence if _persona_label(node_id)]
        chain_cache[pid] = {
            "targets": labels,
            "share": top_count / total if total else None,
            "coalition_path": [_persona_label(pid)] + labels if labels else [],
        }
        return chain_cache[pid]

    def _preferred_segment(pid: str) -> Tuple[Optional[str], int]:
        best_key: Optional[str] = None
        best_count = 0
        for seg_key, persona_map in segment_persona_stats.items():
            stats = persona_map.get(pid)
            observed = stats.get("observed", 0) if stats else 0
            if observed > best_count:
                best_key = seg_key
                best_count = observed
        return best_key, best_count

    def _recommended_moves(pid: str) -> List[Dict[str, Any]]:
        moves: List[Dict[str, Any]] = []
        for row in arsenal_impact:
            persona_ids = row.get("persona_ids") or []
            if pid not in persona_ids:
                continue
            channel = (row.get("channels") or [None])[0]
            move = {
                "belief_transition": (row.get("funnel_stage") or {}).get("label"),
                "best_asset": row.get("asset_label") or row.get("asset_id"),
                "best_channel": channel,
                "bps_lift": round((row.get("avg_delta") or 0.0) * 10000.0, 1),
            }
            moves.append(move)
            if len(moves) >= 2:
                break
        return moves

    keystone_rows: List[Dict[str, Any]] = []
    for persona_id, metric in persona_wolves_metrics.items():
        score = _to_float(metric.get("wolves_score")) or 0.0
        if score < KEYSTONE_MIN_SCORE:
            continue
        sample_size = metric.get("sample_size") or persona_stats.get(persona_id, {}).get("observed", 0)
        if not sample_size or sample_size < KEYSTONE_MIN_SAMPLE:
            continue
        label = _persona_label(persona_id)
        if not label:
            continue
        segment_key, segment_signals = _preferred_segment(persona_id)
        segment_payload = _segment_payload_from_key(segment_key)
        chain_info = _persona_chain_targets(persona_id)
        moves = _recommended_moves(persona_id)
        explanation_bits: List[str] = []
        if segment_payload and segment_signals:
            explanation_bits.append(
                f"{segment_payload.get('label')} accounts saw {label} in {segment_signals} journeys."
            )
        delta_win_bp = _to_float(metric.get("delta_win_bp"))
        if isinstance(delta_win_bp, float):
            explanation_bits.append(
                f"Presence shifts win odds by {delta_win_bp:.1f} bps."
            )
        if chain_info.get("targets"):
            share_text = (
                _percent(chain_info.get("share") or 0.0, 0)
                if chain_info.get("share") is not None
                else "—"
            )
            explanation_bits.append(
                f"Unlocks {_format_list(chain_info['targets'])} in {share_text} of wins."
            )
        keystone_rows.append(
            {
                "type": "keystone_persona",
                "persona_id": persona_id,
                "persona": label,
                "wolves_score": round(score, 4),
                "delta_win_bp": round(delta_win_bp or 0.0, 2) if delta_win_bp is not None else None,
                "involvement_rate": _to_float(metric.get("involvement_rate")),
                "sample_size": int(sample_size),
                "segment": segment_payload,
                "chain_targets": chain_info.get("targets") or [],
                "chain_share": chain_info.get("share"),
                "coalition_path": chain_info.get("coalition_path") or [],
                "recommended_moves": moves,
                "explanation": " ".join(explanation_bits).strip(),
            }
        )
    keystone_rows.sort(
        key=lambda row: (
            row.get("wolves_score") or 0.0,
            row.get("sample_size") or 0,
        ),
        reverse=True,
    )
    insights["keystone_personas"] = keystone_rows[:6]

    return insights

def _to_float(value: Optional[Any]) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_mean(values: Iterable[Any]) -> Optional[float]:
    numeric = [v for v in (_to_float(x) for x in values) if v is not None]
    return mean(numeric) if numeric else None


_FUNNEL_STAGE_LABELS: Dict[str, str] = {
    "early": "Early Cycle",
    "mid": "Mid Cycle",
    "late": "Late Cycle",
}


def _avg_persona_metrics(
    product_graph: Optional[nx.DiGraph],
    persona_ids: Iterable[str],
) -> Tuple[Optional[float], Optional[float]]:
    if product_graph is None:
        return (None, None)
    perc_values: List[float] = []
    prox_values: List[float] = []
    for pid in persona_ids:
        if not pid or pid not in product_graph:
            continue
        node = product_graph.nodes[pid]
        perc = _to_float(node.get("perceptibility"))
        prox = _to_float(node.get("proximity"))
        if perc is not None:
            perc_values.append(perc)
        if prox is not None:
            prox_values.append(prox)
    avg_perc = sum(perc_values) / len(perc_values) if perc_values else None
    avg_prox = sum(prox_values) / len(prox_values) if prox_values else None
    return (avg_perc, avg_prox)


def _funnel_stage_from_metrics(
    perceptibility: Optional[float],
    proximity: Optional[float],
) -> Tuple[str, str]:
    perc = _to_float(perceptibility)
    prox = _to_float(proximity)
    if perc is None and prox is None:
        return ("mid", _FUNNEL_STAGE_LABELS["mid"])
    perc = perc or 0.0
    prox = prox or 0.0
    if perc >= 0.6 and prox <= 0.45:
        code = "early"
    elif prox >= 0.6 or (prox >= 0.55 and perc < 0.5):
        code = "late"
    else:
        code = "mid"
    return (code, _FUNNEL_STAGE_LABELS.get(code, code.title()))


def update_bayesian_journey_for_account(
    product_id: str,
    account_id: str,
    episode_events: List[Dict[str, Any]],
    base_neighbors: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:

    """
    Main entry point from belief_manager.

    Returns:
        {
          "baseline_expected_next": [...],
          "current_expected_next": [...],
          "baseline_paths": ...,
          "current_paths": ...,
          "diff_summary_v3": {...},  # optional
        }
    """
    db: Session = next(get_db())
    # 1) Load existing stats & weights (priors)
    stats_before = load_stats(product_id)
    weights_before = load_weights(product_id)

    # 3) Compute baseline 'expected next' from first persona in episode
    first_persona = next((e["persona_id"] for e in episode_events if e.get("persona_id")), None)
    first_bucket = None
    if episode_events:
        first_bucket = bucket_channel(
            episode_events[0].get("channel"),
            episode_events[0].get("source"),
        )

    baseline_expected_next = []
    if first_persona:
        baseline_expected_next = next_distribution(
            first_persona, weights_before, first_bucket
        )

    # 4) Update stats from this episode
    stats_after = update_edge_stats(stats_before, episode_events)

    base_neighbors = base_neighbors or {}

    # 5) Recompute weights (posterior)
    weights_after = recompute_weights(stats_after, base_neighbors)

    # 6) Compute new expected next
    current_expected_next = []
    if first_persona:
        current_expected_next = next_distribution(
            first_persona, weights_after, first_bucket
        )

    # 7) Save
    save_stats(product_id, stats_after)
    save_weights(product_id, weights_after)

    # 8) Simple diff summary (you can feed this into your existing
    #    persona_path_inferences / graph_level_inferences builder)
    diff_summary = {
        "first_persona": first_persona,
        "baseline_expected_next": baseline_expected_next,
        "current_expected_next": current_expected_next,
    }

    return {
        "baseline_expected_next": baseline_expected_next,
        "current_expected_next": current_expected_next,
        "diff_summary_v3": diff_summary,
        # baseline_paths/current_paths left for your existing path simulator
    }

# ---------------------------
# 2.1: learn from SHM episodes
# ---------------------------

def _iter_ordered_episode_pairs(
    episodes: List[SHMEpisode],
) -> Iterable[Tuple[SHMEpisode, SHMEpisode]]:
    """
    Group by (account_id, episode_id) and yield consecutive (prev, curr) steps.
    """

    keyfunc = lambda e: (e.account_id, e.episode_id)
    for _, group in itertools.groupby(episodes, key=keyfunc):
        steps = list(group)
        steps.sort(key=lambda e: e.step_index)
        for i in range(len(steps) - 1):
            yield steps[i], steps[i + 1]


def rebuild_stats_from_shm(
    db: Session,
    *,
    product_id: str,
) -> None:
    """
    Full rebuild of edge stats & weights for a product from all SHM episodes.

    Use this whenever you:
      - change the SHM model format, OR
      - want a clean recompute of journey weights.
    """
    episodes = load_episodes_for_product(db, product_id=product_id)
    stats = {}  # start from scratch

    for prev_step, curr_step in _iter_ordered_episode_pairs(episodes):
        # Here we define what an "edge" means for learning.
        # Simplest first: persona-level transitions.
        from_persona = prev_step.persona_id
        to_persona = curr_step.persona_id

        # Optional: fold in belief_state transitions as well:
        from_state = prev_step.belief_state
        to_state = curr_step.belief_state

        # Let learner map channel + engagement -> bucket.
        bucket = bucket_channel(
            curr_step.engagement_type,
            curr_step.meta or {},
        )

        # This line depends on how you defined update_edge_stats.
        # Common pattern: update_edge_stats(stats, from_persona, to_persona, bucket)
        update_edge_stats(
            stats,
            from_persona,
            to_persona,
            from_state=from_state,
            to_state=to_state,
            bucket=bucket,
            effect_bucket=curr_step.effect_bucket,
        )

    save_stats(product_id, stats)
    weights = recompute_weights(stats)
    save_weights(product_id, weights)


def update_stats_from_new_steps(
    db: Session,
    *,
    product_id: str,
    new_steps: List[SHMEpisode],
) -> None:
    """
    Incremental learning: given a list of *new* SHM steps
    (already committed to the DB), update stats & weights.

    Assumes:
      - new_steps belong to one or more (account_id, episode_id),
      - you don't pass in old steps.
    """

    if not new_steps:
        return

    stats = load_stats(product_id) or {}

    # group by (account, episode)
    keyfunc = lambda e: (e.account_id, e.episode_id)
    for _, group in itertools.groupby(new_steps, key=keyfunc):
        steps = list(group)
        steps.sort(key=lambda e: e.step_index)
        for i in range(len(steps) - 1):
            prev_step, curr_step = steps[i], steps[i + 1]

            from_persona = prev_step.persona_id
            to_persona = curr_step.persona_id
            from_state = prev_step.belief_state
            to_state = curr_step.belief_state
            bucket = bucket_channel(
                curr_step.engagement_type,
                curr_step.meta or {},
            )

            update_edge_stats(
                stats,
                from_persona,
                to_persona,
                from_state=from_state,
                to_state=to_state,
                bucket=bucket,
                effect_bucket=curr_step.effect_bucket,
            )

    save_stats(product_id, stats)
    weights = recompute_weights(stats)
    save_weights(product_id, weights)


def summarize_global_insights(db: Session, product_id: str) -> dict:
    """
    Product-level SHM / belief insights across *all* accounts.
    Feeds the 'Global Insights' box in Engagements Setup.
    """
    print("summarizing global insights for product:", product_id)

    meta_account_map: Dict[str, List[str]] = defaultdict(list)
    account_raw_meta: Dict[str, Dict[str, Any]] = {}
    segment_keys_by_account: Dict[str, set] = defaultdict(set)
    persona_wolves_metrics: Dict[str, Dict[str, Any]] = {}
    wolves_metrics_updated_at: Optional[str] = None
    candidate_persona_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"occurrences": 0, "accounts": set()}
    )

    try:
        # --- episodes & counts ---
        episodes = (
            db.query(ShmEpisode)
            .filter(ShmEpisode.product_id == product_id)
            .all()
        )
        print("loaded", len(episodes), "episodes for product:", product_id)

        account_ids = {e.account_id for e in episodes}
        num_accounts = len(account_ids)

        account_status_map: Dict[str, str] = {}
        account_meta_lookup: Dict[str, Dict[str, Any]] = {}
        if account_ids:
            status_rows = (
                db.query(
                    TargetAccountORM.id,
                    TargetAccountORM.account_name,
                    TargetAccountORM.industry,
                    TargetAccountORM.revenue_range,
                    TargetAccountORM.employee_range,
                    TargetAccountORM.geography,
                    TargetAccountORM.funding_stage,
                    TargetAccountORM.deal_status,
                )
                .filter(TargetAccountORM.id.in_(account_ids))
                .all()
            )
            for row in status_rows:
                account_id = getattr(row, "id", None)
                if not account_id:
                    continue
                acc_key = str(account_id)
                account_status_map[acc_key] = (getattr(row, "deal_status", "") or "").strip()
                account_meta_lookup[acc_key] = {
                    "account_name": getattr(row, "account_name", None),
                    "industry": getattr(row, "industry", None),
                    "revenue_range": getattr(row, "revenue_range", None),
                    "employee_range": getattr(row, "employee_range", None),
                    "geography": getattr(row, "geography", None),
                    "funding_stage": getattr(row, "funding_stage", None),
                }

        persona_wolves_metrics, wolves_metrics_updated_at = _load_persona_metrics_snapshot(product_id)

        # if you store num_steps on the episode, use that
        steps = (
            db.query(ShmEpisodeStep)
            .join(ShmEpisode, ShmEpisodeStep.episode_id == ShmEpisode.id)
            .filter(ShmEpisode.product_id == product_id)
            .all()
        )
        updates = (
            db.query(ShmLearningUpdate)
            .filter(ShmLearningUpdate.product_id == product_id)
            .all()
        )

        print("processing", len(updates), "learning updates for summarization")

        try:
            asset_rows = (
                db.query(
                    ArsenalAsset.id,
                    ArsenalAsset.name,
                    ArsenalAsset.category,
                    ArsenalAsset.category_text,
                )
                .filter(ArsenalAsset.product_id == product_id)
                .all()
            )
            asset_metadata: Dict[str, Dict[str, Any]] = {}
            for row in asset_rows:
                category_label = None
                if getattr(row, "category_text", None):
                    category_label = row.category_text
                else:
                    category_obj = getattr(row, "category", None)
                    if category_obj is not None:
                        category_label = getattr(category_obj, "value", str(category_obj))
                asset_metadata[str(getattr(row, "id"))] = {
                    "name": getattr(row, "name", None),
                    "category_label": category_label,
                }
        except Exception:
            asset_metadata = {}

        def _coerce_payload(raw: Any) -> Dict[str, Any]:
            if raw is None:
                return {}
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {}
            return {}

        episode_account_map: Dict[str, str] = {}
        persona_stats: Dict[str, Dict[str, Any]] = {}
        hit1_total = 0
        hit3_total = 0
        off_path_total = 0
        log_loss_total = 0.0
        log_loss_count = 0
        classification_counts: Counter[str] = Counter()
        classification_logloss_sum: Dict[str, float] = defaultdict(float)
        classification_logloss_count: Dict[str, int] = defaultdict(int)
        arsenal_stats: Dict[str, Dict[str, Any]] = {}
        engaged_counter: Counter[str] = Counter()
        persona_node_info: Dict[str, Dict[str, Any]] = {}

        try:
            product_graph = build_product_graph(product_id)
        except Exception:
            product_graph = None

        def _label(node_id: Optional[str]) -> Optional[str]:
            if not node_id:
                return None
            if product_graph is not None:
                try:
                    return _L(product_graph, node_id)
                except Exception:
                    pass
            return str(node_id) if node_id is not None else None

        def _ensure_persona_node(pid: Optional[str]) -> None:
            if not pid:
                return
            if pid in persona_node_info:
                return
            info: Dict[str, Any] = {"label": _label(pid)}
            if product_graph is not None and product_graph.has_node(pid):
                node_data = product_graph.nodes[pid]
                info["perceptibility"] = _to_float(node_data.get("perceptibility"))
                info["proximity"] = _to_float(node_data.get("proximity"))
            persona_node_info[pid] = info

        account_meta_by_account: Dict[str, List[str]] = {}
        account_meta_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "wins": 0,
                "losses": 0,
                "open": 0,
                "durations": [],
                "meta_confidence_weight": 0.0,
                "meta_confidence_samples": 0,
            }
        )
        episode_durations: List[float] = []
        for episode in episodes:
            outcome = getattr(episode, "outcome", None)
            outcome_value = (
                outcome.value if hasattr(outcome, "value") else str(outcome or "")
            )
            if (
                outcome_value in (EpisodeOutcome.unknown.value, "", None)
                and episode.account_id in account_status_map
            ):
                status_hint = account_status_map[episode.account_id].lower()
                if "won" in status_hint:
                    outcome_value = EpisodeOutcome.won.value
                elif "lost" in status_hint:
                    outcome_value = EpisodeOutcome.lost.value
            started_at = getattr(episode, "started_at", None)
            ended_at = getattr(episode, "ended_at", None)
            duration_days: Optional[float] = None
            if started_at and ended_at:
                try:
                    duration_days = max((ended_at - started_at).total_seconds() / 86400.0, 0.01)
                except Exception:
                    duration_days = None
            if duration_days is not None:
                episode_durations.append(duration_days)
            episode_account_map.setdefault(episode.id, episode.account_id)
            raw_episode_meta = getattr(episode, "account_meta", None)
            episode_meta = (
                dict(raw_episode_meta)
                if isinstance(raw_episode_meta, dict)
                else {}
            )
            fallback_meta = account_meta_lookup.get(episode.account_id, {}) or {}
            merged_meta: Dict[str, Any] = {}
            meta_confidence = 0.0
            if episode_meta:
                for key, value in episode_meta.items():
                    if value in (None, "", [], {}):
                        continue
                    merged_meta[key] = value
                meta_confidence = 1.0
            if fallback_meta:
                for key, value in fallback_meta.items():
                    if value in (None, "", [], {}):
                        continue
                    merged_meta.setdefault(key, value)
                if meta_confidence == 0.0 and merged_meta:
                    meta_confidence = 0.6
            if merged_meta:
                merged = account_raw_meta.setdefault(episode.account_id, {})
                for key, value in merged_meta.items():
                    merged.setdefault(key, value)
                for segment_key in segment_keys_from_meta(_filter_meta_fields(merged_meta)):
                    if segment_key:
                        segment_keys_by_account[episode.account_id].add(segment_key)

            meta_strings = _normalize_account_meta(merged_meta)
            if meta_strings:
                existing = account_meta_by_account.setdefault(episode.account_id, [])
                for entry in meta_strings:
                    if entry not in existing:
                        existing.append(entry)
                for entry in existing:
                    stats = account_meta_stats[entry]
                    if duration_days is not None:
                        stats["durations"].append(duration_days)
                    if outcome_value == EpisodeOutcome.won.value:
                        stats["wins"] += 1
                    elif outcome_value == EpisodeOutcome.lost.value:
                        stats["losses"] += 1
                    else:
                        stats["open"] += 1
                    if meta_confidence:
                        stats.setdefault("meta_confidence_weight", 0.0)
                        stats.setdefault("meta_confidence_samples", 0)
                        stats["meta_confidence_weight"] += meta_confidence
                        stats["meta_confidence_samples"] += 1

            raw_candidates = getattr(episode, "candidate_personas", None) or {}
            if isinstance(raw_candidates, dict):
                for label, count in raw_candidates.items():
                    if not label:
                        continue
                    stats = candidate_persona_stats[label]
                    stats["occurrences"] += int(count) if isinstance(count, int) else 1
                    stats["accounts"].add(episode.account_id)

        for account_id, metas in account_meta_by_account.items():
            for entry in metas:
                meta_account_map[entry].append(account_id)

        segment_keys_by_account = {
            account_id: sorted(keys)
            for account_id, keys in segment_keys_by_account.items()
            if keys
        }
    
        episode_segments_map: Dict[str, List[str]] = {
            getattr(episode, "id"): segment_keys_by_account.get(episode.account_id, [])
            for episode in episodes
        }
    
        def _class_from_bucket(bucket: Optional[str]) -> str:
            mapping = {
                "on_path": "expected",
                "perfect_match": "expected",
                "near_path": "jump_ahead",
                "skip_hit": "jump_ahead",
                "off_path": "off_path",
                "off_path_known": "off_path",
                "out_of_graph": "off_path",
                "no_path": "no_path",
            }
            if not bucket:
                return "no_path"
            return mapping.get(str(bucket).lower(), str(bucket).lower())
    
        persona_sequences_by_episode: Dict[str, List[str]] = defaultdict(list)
        segment_persona_sequences: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        segment_persona_stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "observed": 0,
                    "hit_at_1": 0,
                    "hit_at_3": 0,
                    "partial_off": 0,
                    "off_path": 0,
                    "pred_counts": Counter(),
                }
            )
        )
        segment_classification_counts: Dict[str, Counter] = defaultdict(Counter)
    
        for step in steps:
            raw_bucket = step.bucket.value if hasattr(step.bucket, "value") else step.bucket
            bucket = (raw_bucket or "").lower()
            cls = _class_from_bucket(raw_bucket)
            classification_counts[cls] += 1
            segments_for_episode = episode_segments_map.get(step.episode_id) or []
            for seg in segments_for_episode:
                segment_classification_counts[seg][cls] += 1
            persona_id = step.observed_persona_id
            seq = persona_sequences_by_episode.setdefault(step.episode_id, [])
            if persona_id:
                persona_key = str(persona_id)
                _ensure_persona_node(persona_key)
                if not seq or seq[-1] != persona_key:
                    seq.append(persona_key)
                for seg in segments_for_episode:
                    seg_seq_map = segment_persona_sequences[seg]
                    seg_seq = seg_seq_map.setdefault(step.episode_id, [])
                    if not seg_seq or seg_seq[-1] != persona_key:
                        seg_seq.append(persona_key)
            if persona_id:
                engaged_counter[persona_id] += 1
            if not persona_id:
                continue
            stats = persona_stats.setdefault(
                persona_id,
                {
                    "observed": 0,
                    "hit_at_1": 0,
                    "hit_at_3": 0,
                    "partial_off": 0,
                    "off_path": 0,
                    "pred_counts": Counter(),
                },
            )
            stats["observed"] += 1
            if step.hit_at_1:
                stats["hit_at_1"] += 1
                hit1_total += 1
            if step.hit_at_3:
                stats["hit_at_3"] += 1
                hit3_total += 1
            if bucket in {"off_path", "off_path_known", "out_of_graph"}:
                stats["off_path"] += 1
                off_path_total += 1
            elif bucket in {"near_path", "jump_ahead", "skip_hit"}:
                stats["partial_off"] += 1
            for seg in segments_for_episode:
                seg_stats = segment_persona_stats[seg].setdefault(
                    persona_key,
                    {
                        "observed": 0,
                        "hit_at_1": 0,
                        "hit_at_3": 0,
                        "partial_off": 0,
                        "off_path": 0,
                        "pred_counts": Counter(),
                    },
                )
                seg_stats["observed"] += 1
                if step.hit_at_1:
                    seg_stats["hit_at_1"] += 1
                if step.hit_at_3:
                    seg_stats["hit_at_3"] += 1
                if bucket in {"off_path", "off_path_known", "out_of_graph"}:
                    seg_stats["off_path"] += 1
                elif bucket in {"near_path", "jump_ahead", "skip_hit"}:
                    seg_stats["partial_off"] += 1
            predicted = getattr(step, "predicted_top_persona_id", None)
            if not predicted:
                top_list = step.predicted_topK or []
                if isinstance(top_list, list) and top_list:
                    head = top_list[0]
                    if isinstance(head, dict):
                        predicted = (
                            head.get("persona")
                            or head.get("persona_id")
                            or head.get("id")
                        )
                    else:
                        predicted = str(head)
            if predicted:
                _ensure_persona_node(str(predicted))
                stats["pred_counts"][predicted] += 1
                for seg in segments_for_episode:
                    seg_stats = segment_persona_stats[seg].setdefault(
                        persona_key,
                        {
                            "observed": 0,
                            "hit_at_1": 0,
                            "hit_at_3": 0,
                            "partial_off": 0,
                            "off_path": 0,
                            "pred_counts": Counter(),
                        },
                    )
                    seg_stats["pred_counts"][predicted] += 1
    
            metrics_payload = step.metrics or {}
            if isinstance(metrics_payload, dict):
                error_blob = metrics_payload.get("error") or {}
                log_loss = error_blob.get("log_loss")
                if isinstance(log_loss, (float, int)):
                    log_loss_total += float(log_loss)
                    log_loss_count += 1
                    classification_logloss_sum[cls] += float(log_loss)
                    classification_logloss_count[cls] += 1
                engagement_meta = metrics_payload.get("engagement") or {}
                asset_id = (
                    metrics_payload.get("asset_id")
                    or engagement_meta.get("asset_id")
                    or engagement_meta.get("id")
                )
                if asset_id:
                    asset_id = str(asset_id)
                    stat = arsenal_stats.setdefault(
                        asset_id,
                        {
                            "asset_id": asset_id,
                            "label": engagement_meta.get("label")
                            or engagement_meta.get("raw_activity")
                            or asset_id,
                            "channels": set(),
                            "persona_ids": set(),
                            "persona_labels": set(),
                            "account_meta": set(),
                            "delta_samples": [],
                            "total_delta": 0.0,
                            "conf_values": [],
                            "count": 0,
                            "stage_counts": Counter(),
                            "concern_counts": Counter(),
                        },
                    )
                    stat["count"] += 1
                    channel_val = (
                        engagement_meta.get("channel")
                        or metrics_payload.get("channel")
                        or engagement_meta.get("source")
                    )
                    if channel_val:
                        stat["channels"].add(str(channel_val))
                    if persona_id:
                        stat["persona_ids"].add(str(persona_id))
                        persona_label = _label(persona_id)
                        stat["persona_labels"].add(
                            persona_label or str(persona_id)
                        )
                    delta_total_step = 0.0
                    edge_evidence = metrics_payload.get("edge_evidence") or []
                    if isinstance(edge_evidence, list):
                        for ev in edge_evidence:
                            if not isinstance(ev, dict):
                                continue
                            delta_val = ev.get("delta_log_prob")
                            if delta_val is None:
                                delta_val = ev.get("deltaLogProb")
                            if isinstance(delta_val, (int, float)):
                                delta_total_step += float(delta_val)
                            from_id = ev.get("from")
                            to_id = ev.get("to")
                            if from_id:
                                engaged_counter[str(from_id)] += 1
                                _ensure_persona_node(str(from_id))
                            if to_id:
                                engaged_counter[str(to_id)] += 1
                                _ensure_persona_node(str(to_id))
                    stat["total_delta"] += delta_total_step
                    stat["delta_samples"].append(delta_total_step)
                    conf_val: Optional[float] = None
                    if isinstance(log_loss, (float, int)):
                        conf_val = 1.0 / (1.0 + max(float(log_loss), 0.0))
                    elif step.hit_at_1 is not None:
                        conf_val = 1.0 if step.hit_at_1 else 0.5
                    elif step.hit_at_3:
                        conf_val = 0.75
                    if conf_val is not None:
                        stat["conf_values"].append(conf_val)
                    meta_payload = (
                        metrics_payload.get("account_meta")
                        or (step.episode.account_meta if getattr(step, "episode", None) else None)
                    )
                    meta_strings = _normalize_account_meta(meta_payload)
                    if not meta_strings and getattr(step, "episode", None):
                        meta_strings = account_meta_by_account.get(step.episode.account_id, [])
                    for entry in meta_strings:
                        stat["account_meta"].add(entry)
    
        persona_updates: Dict[str, Dict[str, Any]] = {}
        edge_updates: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
        for update in updates:
            payload = _coerce_payload(update.payload)
            for key in ("jobs", "pains_solved", "triggers"):
                values = payload.get(key) or []
                if isinstance(values, list):
                    for node_id in values:
                        if node_id:
                            engaged_counter[str(node_id)] += 1
            if update.node_id:
                _ensure_persona_node(update.node_id)
                bucket = persona_updates.setdefault(
                    update.node_id,
                    {
                        "persona_label": None,
                        "total_seen": 0,
                        "delta_values": [],
                        "conf_values": [],
                        "field_values": defaultdict(list),
                        "reasons": set(),
                        "jobs": set(),
                        "pains_solved": set(),
                        "triggers": set(),
                        "account_meta": Counter(),
                    },
                )
                bucket["total_seen"] += 1
                label = (
                    (payload.get("labels") or {}).get("persona")
                    or payload.get("anchor_label")
                    or payload.get("anchor")
                )
                if label and not bucket["persona_label"]:
                    bucket["persona_label"] = label
                if update.delta is not None:
                    bucket["delta_values"].append(update.delta)
                if update.confidence is not None:
                    bucket["conf_values"].append(update.confidence)
                if update.field:
                    bucket["field_values"][update.field].append(
                        {
                            "band": update.band,
                            "delta": update.delta,
                            "confidence": update.confidence,
                            "reason": payload.get("reason"),
                        }
                    )
                if payload.get("reason"):
                    bucket["reasons"].add(payload["reason"])
                for key, store_key in (
                    ("jobs", "jobs"),
                    ("pains_solved", "pains_solved"),
                    ("triggers", "triggers"),
                ):
                    values = payload.get(key) or []
                    if isinstance(values, list):
                        bucket[store_key].update(str(v) for v in values)
                meta_strings = account_meta_by_account.get(update.account_id) or []
                bucket["account_meta"].update(meta_strings)
    
            if update.u_node_id and update.v_node_id:
                key = (update.u_node_id, update.v_node_id)
                bucket = edge_updates.setdefault(
                    key,
                    {
                        "seen": 0,
                        "band_counts": Counter(),
                        "delta_values": [],
                        "conf_values": [],
                        "reasons": set(),
                        "scope_samples": [],
                        "pair_labels": None,
                        "edge_labels": set(),
                        "edge_chain": None,
                        "account_meta": Counter(),
                    },
                )
                bucket["seen"] += 1
                if update.band:
                    bucket["band_counts"][update.band] += 1
                if update.delta is not None:
                    bucket["delta_values"].append(update.delta)
                if update.confidence is not None:
                    bucket["conf_values"].append(update.confidence)
                if payload.get("reason"):
                    bucket["reasons"].add(payload["reason"])
                labels = (payload.get("labels") or {}).get("pair")
                if labels:
                    bucket["pair_labels"] = labels
                try:
                    edge_labels = payload.get("edge_labels") or []
                    for lbl in edge_labels:
                        if isinstance(lbl, (list, tuple)) and len(lbl) == 3:
                            bucket["edge_labels"].add(tuple(str(x) for x in lbl))
                except Exception:
                    pass
                edges = payload.get("edges") or []
                if isinstance(edges, list) and edges:
                    bucket["scope_samples"].append(1.0 / max(len(edges), 1))
                    if bucket["edge_chain"] is None:
                        bucket["edge_chain"] = [
                            list(edge) if isinstance(edge, tuple) else edge
                            for edge in edges
                        ]
                else:
                    bucket["scope_samples"].append(1.0)
                meta_strings = account_meta_by_account.get(update.account_id) or []
                bucket["account_meta"].update(meta_strings)
    
        concerns_by_persona: Dict[str, List[Dict[str, Any]]] = {}
        rcs_paths: List[Dict[str, Any]] = []
        if product_graph is not None:
            try:
                original_graph = product_graph.copy(as_view=False)
                engaged_nodes = [
                    {"id": nid, "occurrence": count}
                    for nid, count in engaged_counter.items()
                    if nid and product_graph.has_node(nid)
                ]
                if engaged_nodes:
                    rcs_report = generate_rcs_new(
                        product_graph=product_graph.copy(as_view=False),
                        original_graph=original_graph,
                        engaged_nodes=engaged_nodes,
                        top_k_per_persona=6,
                    )
                    concerns_by_persona = rcs_report.get("concerns_by_persona") or {}
                    rcs_paths = rcs_report.get("paths") or []
                    for path in rcs_paths:
                        for entry in path.get("personas") or path.get("path") or []:
                            pid = entry.get("id") if isinstance(entry, dict) else entry
                            if pid:
                                _ensure_persona_node(pid)
            except Exception as exc:
                print("⚠️ generate_rcs_new failed during SHM summarization:", repr(exc))
                concerns_by_persona = {}
                rcs_paths = []

        def _summarize_field(values: List[Dict[str, Any]]) -> Dict[str, Any]:
            band_counts = Counter(
                v.get("band") for v in values if v.get("band")
            )
            top_band = band_counts.most_common(1)[0][0] if band_counts else None
            return {
                "field": values[0].get("field") if values else None,
                "band": top_band,
                "avg_delta": _safe_mean(v.get("delta") for v in values),
                "avg_confidence": _safe_mean(v.get("confidence") for v in values),
                "reason": next(
                    (v.get("reason") for v in values if v.get("reason")),
                    None,
                ),
            }

        def _edge_chain_metrics(edges_payload: Any) -> Tuple[Optional[float], Optional[float]]:
            if not edges_payload:
                return (None, None)
            if not product_graph:
                return (None, None)
            likelihood_product: Optional[float] = None
            relevance_product: Optional[float] = None
            for raw_edge in edges_payload:
                if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) < 2:
                    continue
                u_id, v_id = raw_edge[0], raw_edge[1]
                if not product_graph.has_edge(u_id, v_id):
                    continue
                edge_data = product_graph[u_id][v_id]
                lk = _to_float(edge_data.get("likelihood"))
                rel = _to_float(edge_data.get("relevance"))
                if lk is not None:
                    likelihood_product = (likelihood_product or 1.0) * lk
                if rel is not None:
                    relevance_product = (relevance_product or 1.0) * rel
            return (likelihood_product, relevance_product)

        persona_recommendations = []
        for persona_id, data in persona_updates.items():
            if product_graph is not None and persona_id and product_graph.has_node(persona_id):
                # Existing persona in graph – treat learning as edge tuning, not persona addition.
                continue
            stats = persona_stats.get(
                persona_id,
                {
                    "observed": 0,
                    "hit_at_1": 0,
                    "pred_counts": Counter(),
                    "off_path": 0,
                    "partial_off": 0,
                },
            )
            observed = stats.get("observed", 0)
            hit_rate = (
                stats.get("hit_at_1", 0) / observed if observed else None
            )
            top_pred = None
            top_pred_rate = None
            for candidate, _ in stats.get("pred_counts", Counter()).most_common():
                top_pred = candidate
                total_preds = sum(stats.get("pred_counts", Counter()).values()) or 0
                if total_preds:
                    top_pred_rate = stats["pred_counts"].get(candidate, 0) / total_preds
                break

            field_summaries = []
            for field_name, values in data["field_values"].items():
                enriched = [dict(v, field=field_name) for v in values]
                field_summaries.append(_summarize_field(enriched))

            avg_delta = _safe_mean(data["delta_values"]) or 0.0
            avg_confidence = _safe_mean(data["conf_values"])
            predicted_boost_pct = avg_delta * 100.0
            account_meta_top = [
                meta for meta, _ in data["account_meta"].most_common(6)
            ]
            pains_sorted = sorted(data["pains_solved"])
            persona_display_label = data["persona_label"] or _label(persona_id)
            job_specs = [
                _build_job_spec(
                    product_graph,
                    job_id,
                    pains_sorted,
                    default_label=persona_display_label,
                )
                for job_id in sorted(data["jobs"])
            ]
            if not job_specs and pains_sorted:
                job_specs.append(
                    _build_job_spec(
                        product_graph,
                        None,
                        pains_sorted,
                        default_label=persona_display_label,
                    )
                )
            pain_specs = _build_pain_specs(product_graph, pains_sorted)
            persona_recommendations.append(
                {
                    "persona_id": persona_id,
                    "persona_label": data["persona_label"] or _label(persona_id),
                    "observed_events": observed,
                    "seen": observed,
                    "recommendation_events": data["total_seen"],
                    "current_best_fit": _label(top_pred) if top_pred else _label(persona_id),
                    "current_fitness": hit_rate if hit_rate is not None else top_pred_rate,
                    "avg_confidence": avg_confidence,
                    "avg_delta": avg_delta,
                    "predicted_boost_pct": predicted_boost_pct,
                    "reasons": sorted(data["reasons"]),
                    "field_summaries": [
                        fs for fs in field_summaries if fs.get("avg_delta") is not None
                    ],
                    "jobs": sorted(data["jobs"]),
                    "pains": sorted(data["pains_solved"]),
                    "triggers": sorted(data["triggers"]),
                    "job_specs": job_specs,
                    "pain_specs": pain_specs,
                    "account_meta": account_meta_top,
                    "recommendation_type": "graph_update",
                }
            )

        edge_recommendations = []
        for (u_id, v_id), data in edge_updates.items():
            avg_delta = _safe_mean(data["delta_values"]) or 0.0
            avg_confidence = _safe_mean(data["conf_values"])
            current_relevance = None
            current_likelihood = None
            recommended_likelihood = None
            recommended_relevance = None
            if product_graph is not None and product_graph.has_edge(u_id, v_id):
                edge_data = product_graph[u_id][v_id]
                current_relevance = _to_float(edge_data.get("relevance"))
                current_likelihood = _to_float(edge_data.get("likelihood"))
                recommended_relevance = current_relevance

            chain_edges = data.get("edge_chain")
            chain_likelihood, chain_relevance = _edge_chain_metrics(chain_edges)

            if chain_likelihood is not None:
                current_likelihood = chain_likelihood
            if chain_relevance is not None:
                current_relevance = chain_relevance
                recommended_relevance = chain_relevance

            if current_likelihood is not None:
                proposed = current_likelihood + avg_delta
                recommended_likelihood = max(0.0, min(1.0, proposed))
            delta_likelihood = None
            if (
                recommended_likelihood is not None
                and current_likelihood is not None
            ):
                delta_likelihood = recommended_likelihood - current_likelihood
            delta_relevance = None
            if (
                recommended_relevance is not None
                and current_relevance is not None
            ):
                delta_relevance = recommended_relevance - current_relevance

            account_meta_top = [
                meta for meta, _ in data["account_meta"].most_common(6)
            ]

            meaningful_change = False
            if delta_likelihood is not None:
                if abs(delta_likelihood) >= 0.01:
                    meaningful_change = True
            elif recommended_likelihood is not None and current_likelihood is None:
                meaningful_change = True
            if delta_relevance is not None:
                if abs(delta_relevance) >= 0.01:
                    meaningful_change = True
            elif recommended_relevance is not None and current_relevance is None:
                meaningful_change = True
            if not meaningful_change and abs(avg_delta) > 1e-6:
                meaningful_change = True
            if not meaningful_change:
                continue

            edge_recommendations.append(
                {
                    "source_id": u_id,
                    "target_id": v_id,
                    "pair_labels": data.get("pair_labels") or [_label(u_id), _label(v_id)],
                    "band": data["band_counts"].most_common(1)[0][0]
                        if data["band_counts"]
                        else None,
                        "seen": data["seen"],
                        "avg_confidence": avg_confidence,
                        "avg_delta": avg_delta,
                        "predicted_boost_pct": avg_delta * 100.0,
                        "scope": _safe_mean(data["scope_samples"]),
                        "reason": next(iter(data["reasons"]), None),
                        "reasons": sorted(data["reasons"]),
                        "edge_labels": [list(lbl) for lbl in data["edge_labels"]],
                        "current_relevance": current_relevance,
                        "current_likelihood": current_likelihood,
                        "recommended_likelihood": recommended_likelihood,
                        "recommended_relevance": recommended_relevance,
                        "delta_likelihood": delta_likelihood,
                        "delta_relevance": delta_relevance,
                        "recommendation_events": data["seen"],
                        "account_meta": account_meta_top,
                    }
                )

        for label, stats in candidate_persona_stats.items():
            occurrences = stats.get("occurrences", 0)
            accounts = stats.get("accounts", set())
            if occurrences < 2:
                continue
            placeholder_job_spec = _build_job_spec(
                product_graph,
                None,
                [],
                default_label=label,
            )
            persona_recommendations.append(
                {
                    "persona_id": None,
                    "persona_label": label,
                    "observed_events": occurrences,
                    "seen": occurrences,
                    "recommendation_events": occurrences,
                    "current_best_fit": None,
                    "current_fitness": None,
                    "avg_confidence": None,
                    "avg_delta": None,
                    "predicted_boost_pct": None,
                    "reasons": [
                        "Repeated low-confidence matches suggest this persona is missing from the graph."
                    ],
                    "field_summaries": [],
                    "jobs": [],
                    "pains": [],
                    "triggers": [],
                    "job_specs": [placeholder_job_spec] if placeholder_job_spec else [],
                    "pain_specs": [],
                    "account_meta": [],
                    "accounts": list(accounts),
                    "recommendation_type": "new_persona_candidate",
                }
            )

        persona_recommendations.sort(
            key=lambda r: (
                r.get("predicted_boost_pct") if r.get("predicted_boost_pct") is not None else float("-inf"),
                r.get("recommendation_events", 0),
            ),
            reverse=True,
        )
        edge_recommendations.sort(
            key=lambda r: (
                r.get("scope") if r.get("scope") is not None else -1,
                r.get("avg_confidence") if r.get("avg_confidence") is not None else -1,
                r.get("predicted_boost_pct") if r.get("predicted_boost_pct") is not None else float("-inf"),
            ),
            reverse=True,
        )

        persona_meta_map: Dict[str, Dict[str, Counter]] = {}
        for rec in persona_recommendations:
            pid = rec.get("persona_id")
            if not pid:
                continue
            stage_counter = Counter()
            triggers = rec.get("triggers") or []
            pains = rec.get("pains") or []
            jobs = rec.get("jobs") or []
            if triggers:
                stage_counter["problem"] += len(triggers)
            if pains:
                stage_counter["pain"] += len(pains)
            if jobs:
                stage_counter["execution"] += len(jobs)
            concern_counter = Counter()
            for pain_id in pains:
                label = _label(pain_id)
                if label:
                    concern_counter[label] += 1
            for job_id in jobs:
                label = _label(job_id)
                if label:
                    concern_counter[label] += 1
            persona_meta_map[pid] = {
                "stage_counts": stage_counter,
                "concerns": concern_counter,
            }

        for pid, concern_rows in (concerns_by_persona or {}).items():
            stage_counter = Counter()
            concern_counter = Counter()
            for row in concern_rows or []:
                stage_code = _canonical_stage(row.get("stage") or row.get("concern_stage"))
                if stage_code:
                    stage_counter[stage_code] += 1
                label = row.get("concern_label") or row.get("label")
                if label:
                    concern_counter[label] += 1
            if not stage_counter and not concern_counter:
                continue
            entry = persona_meta_map.setdefault(
                pid, {"stage_counts": Counter(), "concerns": Counter()}
            )
            entry["stage_counts"].update(stage_counter)
            entry["concerns"].update(concern_counter)

        observed_only_recs: List[Dict[str, Any]] = []
        existing_persona_ids = {rec["persona_id"] for rec in persona_recommendations}
        for pid, stats in persona_stats.items():
            if not pid:
                continue
            if product_graph is not None and product_graph.has_node(pid):
                continue
            if pid in persona_updates or pid in existing_persona_ids:
                continue
            observed = stats.get("observed", 0)
            if observed < 3:
                continue
            hit_rate = _safe_ratio(stats.get("hit_at_1", 0), observed)
            off_ratio_local = _safe_ratio(stats.get("off_path", 0), observed)
            persona_label = persona_node_info.get(pid, {}).get("label") or _format_meta_value(pid)
            reasons: List[str] = []
            reasons.append(f"Observed in {observed} engagements without graph node.")
            if hit_rate:
                reasons.append(f"Matches predicted next {_percent(hit_rate, 0)} of the time.")
            if off_ratio_local:
                reasons.append(f"Stalls {_percent(off_ratio_local, 0)} of journeys when absent.")
            pains_sorted = sorted(stats.get("pains", []))
            job_specs = [
                _build_job_spec(
                    product_graph,
                    None,
                    pains_sorted,
                    default_label=persona_label,
                )
            ] if pains_sorted else []
            observed_only_recs.append(
                {
                    "persona_id": pid,
                    "persona_label": persona_label,
                    "observed_events": observed,
                    "seen": observed,
                    "recommendation_events": observed,
                    "current_best_fit": persona_label,
                    "current_fitness": hit_rate,
                    "avg_confidence": None,
                    "avg_delta": 0.0,
                    "predicted_boost_pct": None,
                    "reasons": reasons,
                    "field_summaries": [],
                    "jobs": [],
                    "pains": [],
                    "triggers": [],
                    "job_specs": job_specs,
                    "pain_specs": _build_pain_specs(product_graph, pains_sorted),
                    "account_meta": [],
                    "recommendation_type": "add_persona_node",
                }
            )
        persona_recommendations.extend(observed_only_recs)

        account_feature_rows = _build_account_feature_rows(
            episodes,
            steps,
            account_meta_lookup=account_meta_lookup,
            account_raw_meta=account_raw_meta,
            segment_keys_by_account=segment_keys_by_account,
            account_status_map=account_status_map,
            arsenal_stats=arsenal_stats,
            asset_metadata=asset_metadata,
        )
        win_regression_summary = build_win_regression_summary(account_feature_rows)

        print("finished summarizing global insights for product:", product_id)

        total_step_events = len(steps)
        stability_score = (
            1.0 - (off_path_total / total_step_events)
            if total_step_events
            else None
        )
        hit_at_1 = (
            hit1_total / total_step_events if total_step_events else None
        )
        hit_at_3 = (
            hit3_total / total_step_events if total_step_events else None
        )
        avg_log_loss = (
            log_loss_total / log_loss_count if log_loss_count else None
        )

        engagement_insights = []
        for cls, count in classification_counts.items():
            share = count / total_step_events if total_step_events else None
            avg_cls_log_loss = None
            if classification_logloss_count.get(cls):
                avg_cls_log_loss = (
                    classification_logloss_sum[cls] / classification_logloss_count[cls]
                )
            engagement_insights.append(
                {
                    "classification": cls,
                    "count": count,
                    "share": share,
                    "avg_log_loss": avg_cls_log_loss,
                }
            )
        engagement_insights.sort(
            key=lambda row: row.get("share") or 0, reverse=True
        )

        for data in arsenal_stats.values():
            stage_counter = Counter()
            concern_counter = Counter()
            for persona_id in data.get("persona_ids", set()):
                meta = persona_meta_map.get(persona_id)
                if not meta:
                    continue
                stage_counter.update(meta.get("stage_counts", Counter()))
                concern_counter.update(meta.get("concerns", Counter()))
            data["stage_counts"] = stage_counter
            data["concern_counts"] = concern_counter

        arsenal_impact: List[Dict[str, Any]] = []
        for asset_id, data in arsenal_stats.items():
            count = data["count"] or 1
            avg_conf = _safe_mean(data["conf_values"]) if data["conf_values"] else None
            avg_delta = _safe_mean(data["delta_samples"]) or 0.0
            persona_id_list = sorted(data["persona_ids"])
            avg_perc, avg_prox = _avg_persona_metrics(product_graph, persona_id_list)
            funnel_code, funnel_label = _funnel_stage_from_metrics(avg_perc, avg_prox)
            funnel_stage_payload = {
                "code": funnel_code,
                "label": funnel_label,
                "avg_perceptibility": avg_perc,
                "avg_proximity": avg_prox,
            }
            data["funnel_stage"] = funnel_stage_payload
            arsenal_impact.append(
                {
                    "asset_id": asset_id,
                    "asset_label": data["label"],
                    "persona_ids": persona_id_list,
                    "persona_labels": sorted(data["persona_labels"]),
                    "total_delta": data["total_delta"],
                    "avg_delta": avg_delta,
                    "avg_confidence": avg_conf,
                    "channels": sorted(data["channels"]),
                    "account_meta": sorted(data["account_meta"]),
                    "num_engagements": count,
                    "funnel_stage": funnel_stage_payload,
                    "stages": [
                        {
                            "code": stage,
                            "label": _stage_label(stage),
                            "count": cnt,
                        }
                        for stage, cnt in data.get("stage_counts", Counter()).most_common()
                        if stage
                    ],
                    "concerns": [
                        {"label": label, "count": cnt}
                        for label, cnt in data.get("concern_counts", Counter()).most_common(12)
                        if label
                    ],
                }
            )
            arsenal_impact[-1]["segment_keys"] = _segment_keys_from_meta_tokens(
                arsenal_impact[-1].get("account_meta")
            )
        arsenal_impact.sort(
            key=lambda row: (
                abs(row.get("total_delta") or 0.0),
                row.get("avg_confidence") or 0.0,
            ),
            reverse=True,
        )

        product_insights = _build_product_insights(
            account_meta_stats=account_meta_stats,
            meta_account_map=meta_account_map,
            persona_stats=persona_stats,
            engaged_counter=engaged_counter,
            persona_sequences_by_episode=persona_sequences_by_episode,
            classification_counts=classification_counts,
            arsenal_impact=arsenal_impact,
            episode_durations=episode_durations,
            persona_node_info=persona_node_info,
            rcs_paths=rcs_paths,
            segment_persona_stats=segment_persona_stats,
            segment_classification_counts=segment_classification_counts,
            segment_sequences_by_segment=segment_persona_sequences,
            persona_wolves_metrics=persona_wolves_metrics,
            wolves_metrics_updated_at=wolves_metrics_updated_at,
            product_graph=product_graph,
        )

        # Persist aggregated arsenal evidence back into the structured tables so that
        # downstream strategy modules can query strengths without recomputing.
        channel_cache: Dict[str, Any] = {}
        for asset_id, data in arsenal_stats.items():
            asset_obj = db.get(ArsenalAsset, asset_id)
            if not asset_obj:
                continue
            funnel_stage_payload = data.get("funnel_stage") or {}
            stage_code = funnel_stage_payload.get("code")
            if isinstance(stage_code, str) and stage_code:
                normalized_code = stage_code.lower()
                asset_obj.org_conversion_maturity = normalized_code
                existing = list(asset_obj.target_belief_stages or [])
                if existing:
                    cleaned = [
                        entry
                        for entry in existing
                        if not (
                            isinstance(entry, str)
                            and (
                                "funnel" in entry.lower()
                                or entry.lower() in _FUNNEL_STAGE_LABELS
                                or entry.lower()
                                in {label.lower() for label in _FUNNEL_STAGE_LABELS.values()}
                            )
                        )
                    ]
                    asset_obj.target_belief_stages = cleaned or None
            personas = sorted(data.get("persona_ids") or [])
            channels = sorted(data.get("channels") or [])
            if not personas or not channels:
                continue
            avg_delta = _safe_mean(data.get("delta_samples") or []) or 0.0
            if avg_delta == 0.0 and not data.get("delta_samples"):
                continue
            stage_counts_payload = {
                stage: int(cnt)
                for stage, cnt in data.get("stage_counts", Counter()).items()
                if cnt
            }
            concern_payload = {
                label: int(cnt)
                for label, cnt in data.get("concern_counts", Counter()).items()
                if label
            }
            evidence_payload = {
                "num_engagements": data.get("count"),
                "total_delta": data.get("total_delta"),
                "account_meta": sorted(data.get("account_meta") or []),
                "stages": stage_counts_payload,
                "concerns": concern_payload,
            }
            for channel_label in channels:
                if not channel_label:
                    continue
                channel_obj = channel_cache.get(channel_label)
                if not channel_obj:
                    channel_obj, _ = get_or_create_channel(
                        db,
                        product_id=product_id,
                        name=channel_label,
                        defaults={},
                    )
                    channel_cache[channel_label] = channel_obj
                for persona_id in personas:
                    if not persona_id:
                        continue
                    record_asset_channel_impact(
                        db,
                        product_id=product_id,
                        asset_id=asset_obj.id,
                        channel_id=channel_obj.id,
                        persona_id=persona_id,
                        belief_transition_id=None,
                        delta_strength=avg_delta,
                        evidence_payload=evidence_payload,
                    )

        db.commit()

        return {
            "meta": {
                "num_accounts": num_accounts,
                "num_engagements": total_step_events,
                "stability_score": stability_score,
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "log_loss": avg_log_loss,
                "num_persona_recommendations": len(persona_recommendations),
                "num_edge_recommendations": len(edge_recommendations),
                "wolves_metrics_updated_at": wolves_metrics_updated_at,
            },
            "persona_recommendations": persona_recommendations,
            "edge_recommendations": edge_recommendations,
            "engagement_insights": engagement_insights,
            "arsenal_impact": arsenal_impact,
            "product_insights": product_insights,
            "win_regression": win_regression_summary,
        }

    except Exception as e:
        # DO NOT kill the whole /get-persona-matches route for a summary bug
        print("Error in summarize_global_insights for product", product_id, ":", repr(e))
        traceback.print_exc()
        return {
            "meta": {
                "num_accounts": 0,
                "num_engagements": 0,
                "num_persona_recommendations": 0,
                "num_edge_recommendations": 0,
                "wolves_metrics_updated_at": wolves_metrics_updated_at,
            },
            "persona_recommendations": [],
            "edge_recommendations": [],
            "engagement_insights": [],
            "arsenal_impact": [],
            "product_insights": DEFAULT_PRODUCT_INSIGHTS,
            "win_regression": None,
        }

def _infer_node_type(node_id: Optional[str]) -> str:
    if not node_id:
        return ""
    if ":" in node_id:
        return node_id.split(":", 1)[0]
    return ""


def _ensure_placeholder_node(G, node_id: Optional[str]) -> Optional[str]:
    if not node_id:
        return None
    if node_id in G:
        return node_id
    ntype = _infer_node_type(node_id) or "unknown"
    attrs: Dict[str, Any] = {
        "id": node_id,
        "node_type": ntype,
        "type": ntype,
        "data_source": "bayesian_inferred",
        "last_updated": current_timestamp(),
    }
    if ntype == "job":
        attrs.setdefault("description", node_id.split(":", 1)[-1].replace("_", " "))
    elif ntype == "pain":
        attrs.setdefault("description", node_id.split(":", 1)[-1].replace("_", " "))
    elif ntype == "persona":
        attrs.setdefault("title", node_id.split(":", 1)[-1].replace("_", " "))
    G.add_node(node_id, **attrs)
    return node_id


def _parse_persona_label(label: Optional[str]) -> Tuple[str, str, str]:
    if not label:
        return ("Persona", "", "")
    parts = [p.strip() for p in label.split("|")]
    title = parts[0] if len(parts) > 0 else "Persona"
    department = parts[1] if len(parts) > 1 else ""
    seniority = parts[2] if len(parts) > 2 else ""
    return (title, department, seniority)


def apply_persona_recommendation_to_graph(
    product_id: str,
    recommendation: Dict[str, Any],
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    G = build_product_graph(product_id)
    existing_nodes = set(G.nodes())
    added_nodes: List[str] = []
    created_edges: List[Dict[str, Any]] = []
    updated_edges: List[Dict[str, Any]] = []
    logged_edges: set[Tuple[str, str, str]] = set()

    def _note_node(node_id: Optional[str]) -> None:
        if node_id and node_id not in existing_nodes:
            added_nodes.append(node_id)
            existing_nodes.add(node_id)

    def _note_edge(src: str, relation: str, tgt: str, existed: bool) -> None:
        key = (src, relation, tgt)
        if key in logged_edges:
            return
        logged_edges.add(key)
        attrs = G[src][tgt]
        if isinstance(attrs, dict) and "type" not in attrs and attrs:
            attrs = attrs[next(iter(attrs.keys()))]
        edge_snapshot = {
            "source": src,
            "target": tgt,
            "relation": relation,
            "likelihood": attrs.get("likelihood"),
            "relevance": attrs.get("relevance"),
            "weight": attrs.get("weight"),
        }
        if existed:
            updated_edges.append(edge_snapshot)
        else:
            created_edges.append(edge_snapshot)

    persona_label = recommendation.get("persona_label")
    title, department, seniority = _parse_persona_label(persona_label)

    persona_id = recommendation.get("persona_id")
    pre_existing_persona = persona_id if persona_id and persona_id in G else None
    persona_node_id = _persona(
        G,
        title=title,
        department=department,
        seniority=seniority,
        sample_profiles=recommendation.get("sample_profiles"),
        data_source="bayesian_inferred",
    )
    _note_node(persona_node_id)
    if persona_id and persona_id in G and persona_id != persona_node_id:
        persona_node_id = persona_id

    job_specs_input = recommendation.get("job_specs") or []
    pain_specs_input = recommendation.get("pain_specs") or []

    jobs = [j for j in recommendation.get("jobs", []) if j]
    pains = [p for p in recommendation.get("pains", []) if p]
    triggers = [t for t in recommendation.get("triggers", []) if t]

    likelihood = recommendation.get("recommended_likelihood")
    if likelihood is None:
        likelihood = recommendation.get("current_likelihood")

    relevance = recommendation.get("recommended_relevance")
    if relevance is None:
        relevance = recommendation.get("current_relevance")

    def _materialize_job_from_spec(spec: Dict[str, Any]) -> Optional[str]:
        job_id = spec.get("id")
        label = spec.get("label") or job_id or f"{title} job"
        description = spec.get("description") or label
        if job_id:
            ensured = _ensure_placeholder_node(G, job_id)
            if ensured and description:
                if not G.nodes[ensured].get("description"):
                    G.nodes[ensured]["description"] = description
            return ensured
        new_job_id = _job(G, label, data_source="bayesian_inferred")
        if description:
            G.nodes[new_job_id]["description"] = description
        return new_job_id

    def _materialize_pain_from_spec(spec: Dict[str, Any]) -> Optional[str]:
        pain_id = spec.get("id")
        label = spec.get("label") or pain_id or "Unspecified pain"
        if pain_id:
            ensured = _ensure_placeholder_node(G, pain_id)
            if ensured and label:
                if not G.nodes[ensured].get("description"):
                    G.nodes[ensured]["description"] = label
            return ensured
        new_pain_id = _pain(G, label, pain_source=None, data_source="bayesian_inferred")
        return new_pain_id

    default_pain_ids: List[str] = []
    if pain_specs_input:
        for spec in pain_specs_input:
            pid = _materialize_pain_from_spec(spec) if isinstance(spec, dict) else None
            if pid:
                default_pain_ids.append(pid)
    if not default_pain_ids and pains:
        default_pain_ids = [_ensure_placeholder_node(G, pain_id) for pain_id in pains if pain_id]
        default_pain_ids = [pid for pid in default_pain_ids if pid]

    job_to_pains: Dict[str, List[str]] = {}
    if job_specs_input:
        jobs_from_specs: List[str] = []
        for spec in job_specs_input:
            if not isinstance(spec, dict):
                continue
            job_node_id = _materialize_job_from_spec(spec)
            if not job_node_id:
                continue
            jobs_from_specs.append(job_node_id)
            pains_for_job: List[str] = []
            for pain_spec in spec.get("pains") or []:
                if not isinstance(pain_spec, dict):
                    continue
                pid = _materialize_pain_from_spec(pain_spec)
                if pid:
                    pains_for_job.append(pid)
            if not pains_for_job:
                pains_for_job = list(default_pain_ids)
            job_to_pains[job_node_id] = pains_for_job
        if jobs_from_specs:
            jobs = jobs_from_specs
            pains = list({pid for plist in job_to_pains.values() for pid in plist if pid})

    if not pains and default_pain_ids:
        pains = list(default_pain_ids)

    if not jobs:
        fallback_job_spec = {
            "label": f"{title or persona_label or 'Persona'} core job",
            "description": recommendation.get("persona_label") or title or "Persona job",
            "pains": pain_specs_input or [{"label": f"{title or persona_label or 'Persona'} blocker"}],
        }
        fallback_job_id = _materialize_job_from_spec(fallback_job_spec)
        if fallback_job_id:
            jobs = [fallback_job_id]
            pains_for_job = []
            for pain_spec in fallback_job_spec["pains"]:
                if not isinstance(pain_spec, dict):
                    continue
                pid = _materialize_pain_from_spec(pain_spec)
                if pid:
                    pains_for_job.append(pid)
            if pains_for_job:
                job_to_pains[fallback_job_id] = pains_for_job
                pains = list({pid for pid in pains_for_job if pid})
            elif default_pain_ids:
                job_to_pains[fallback_job_id] = list(default_pain_ids)
                pains = list(default_pain_ids)

    for job_id in jobs:
        ensured_job = _ensure_placeholder_node(G, job_id)
        if not ensured_job:
            continue
        _note_node(ensured_job)
        edge_existed = G.has_edge(ensured_job, persona_node_id)
        _upsert_edge(
            G,
            ensured_job,
            "performed_by",
            persona_node_id,
            weight=likelihood or 0.7,
            attrs={
                "likelihood": likelihood,
                "relevance": relevance,
                "data_source": "bayesian_inferred",
            },
            prevent_cycles=False,
            data_source="bayesian_inferred",
        )
        _note_edge(ensured_job, "performed_by", persona_node_id, edge_existed)

        pains_for_job = job_to_pains.get(ensured_job, pains)
        if not pains_for_job:
            pains_for_job = list(default_pain_ids)
        for pain_id in pains_for_job or []:
            ensured_pain = _ensure_placeholder_node(G, pain_id)
            if not ensured_pain:
                continue
            _note_node(ensured_pain)
            edge_existed = G.has_edge(ensured_job, ensured_pain)
            _upsert_edge(
                G,
                ensured_job,
                "solves",
                ensured_pain,
                weight=1.0,
                attrs={"data_source": "bayesian_inferred"},
                prevent_cycles=False,
                data_source="bayesian_inferred",
            )
            _note_edge(ensured_job, "solves", ensured_pain, edge_existed)
            edge_existed = G.has_edge(ensured_pain, ensured_job)
            _upsert_edge(
                G,
                ensured_pain,
                "felt_in",
                ensured_job,
                weight=1.0,
                attrs={"data_source": "bayesian_inferred"},
                prevent_cycles=False,
                data_source="bayesian_inferred",
            )
            _note_edge(ensured_pain, "felt_in", ensured_job, edge_existed)
            for trigger_id in triggers:
                ensured_trigger = _ensure_placeholder_node(G, trigger_id)
                if not ensured_trigger:
                    continue
                _note_node(ensured_trigger)
                edge_existed = G.has_edge(ensured_pain, ensured_trigger)
                _upsert_edge(
                    G,
                    ensured_pain,
                    "triggered_by",
                    ensured_trigger,
                    weight=1.0,
                    attrs={"data_source": "bayesian_inferred"},
                    prevent_cycles=False,
                    data_source="bayesian_inferred",
                )
                _note_edge(ensured_pain, "triggered_by", ensured_trigger, edge_existed)

    pid = get_product_id_from_subgraph(G)
    if pid:
        save_graph_as_json(G, pid)

    print(
        "[apply_persona_recommendation]",
        product_id,
        "nodes_added:",
        added_nodes,
        "edges_created:",
        created_edges,
        "edges_updated:",
        updated_edges,
    )

    deleted_updates = 0
    if db is not None:
        persona_candidates = {persona_node_id}
        if pre_existing_persona:
            persona_candidates.add(pre_existing_persona)
        deleted_updates = (
            db.query(ShmLearningUpdate)
            .filter(
                ShmLearningUpdate.product_id == product_id,
                ShmLearningUpdate.node_id.in_(list(persona_candidates)),
            )
            .delete(synchronize_session=False)
        )

    return {
        "persona_id": persona_node_id,
        "jobs_connected": jobs,
        "pains_connected": pains,
        "triggers_connected": triggers,
        "nodes_added": added_nodes,
        "edges_created": created_edges,
        "edges_updated": updated_edges,
        "learning_updates_deleted": deleted_updates,
    }


def apply_edge_recommendation_to_graph(
    product_id: str,
    recommendation: Dict[str, Any],
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    G = build_product_graph(product_id)
    existing_nodes = set(G.nodes())
    added_nodes: List[str] = []
    created_edges: List[Dict[str, Any]] = []
    updated_edges: List[Dict[str, Any]] = []
    logged_edges: set[Tuple[str, str, str]] = set()

    def _note_node(node_id: Optional[str]) -> None:
        if node_id and node_id not in existing_nodes:
            added_nodes.append(node_id)
            existing_nodes.add(node_id)

    def _note_edge(src: str, relation: str, tgt: str, existed: bool) -> None:
        key = (src, relation, tgt)
        if key in logged_edges:
            return
        logged_edges.add(key)
        attrs = G[src][tgt]
        if isinstance(attrs, dict) and "type" not in attrs and attrs:
            attrs = attrs[next(iter(attrs.keys()))]
        edge_snapshot = {
            "source": src,
            "target": tgt,
            "relation": relation,
            "likelihood": attrs.get("likelihood"),
            "relevance": attrs.get("relevance"),
            "weight": attrs.get("weight"),
        }
        if existed:
            updated_edges.append(edge_snapshot)
        else:
            created_edges.append(edge_snapshot)

    source_id = recommendation.get("source_id")
    target_id = recommendation.get("target_id")
    _ensure_placeholder_node(G, source_id)
    _ensure_placeholder_node(G, target_id)
    _note_node(source_id)
    _note_node(target_id)

    relation = None
    if source_id and target_id and G.has_edge(source_id, target_id):
        relation = G[source_id][target_id].get("type")
    if not relation:
        labels = recommendation.get("edge_labels") or []
        if labels and isinstance(labels, list) and len(labels[0]) >= 3:
            relation = labels[0][2]
        else:
            relation = "neighborhood"

    recommended_likelihood = recommendation.get("recommended_likelihood")
    recommended_relevance = recommendation.get("recommended_relevance")

    attrs: Dict[str, Any] = {"data_source": "bayesian_inferred"}
    if recommended_likelihood is not None:
        attrs["likelihood"] = recommended_likelihood
    if recommended_relevance is not None:
        attrs["relevance"] = recommended_relevance

    current_edge = (
        G[source_id][target_id]
        if source_id and target_id and G.has_edge(source_id, target_id)
        else {}
    )
    weight = recommended_likelihood
    if weight is None:
        weight = current_edge.get("likelihood") or current_edge.get("weight") or 0.5

    existed = bool(source_id and target_id and G.has_edge(source_id, target_id))
    _upsert_edge(
        G,
        source_id,
        relation,
        target_id,
        weight=weight,
        attrs=attrs,
        prevent_cycles=False,
        data_source="bayesian_inferred",
    )
    if source_id and target_id:
        _note_edge(source_id, relation, target_id, existed)

    pid = get_product_id_from_subgraph(G)
    if pid:
        save_graph_as_json(G, pid)

    deleted_updates = 0
    if db is not None and source_id and target_id:
        deleted_updates = (
            db.query(ShmLearningUpdate)
            .filter(
                ShmLearningUpdate.product_id == product_id,
                ShmLearningUpdate.u_node_id == source_id,
                ShmLearningUpdate.v_node_id == target_id,
            )
            .delete(synchronize_session=False)
        )

    print(
        "[apply_edge_recommendation]",
        product_id,
        "nodes_added:",
        added_nodes,
        "edges_created:",
        created_edges,
        "edges_updated:",
        updated_edges,
    )

    return {
        "source_id": source_id,
        "target_id": target_id,
        "relation": relation,
        "likelihood": recommended_likelihood,
        "relevance": recommended_relevance,
        "nodes_added": added_nodes,
        "edges_created": created_edges,
        "edges_updated": updated_edges,
        "learning_updates_deleted": deleted_updates,
    }


def record_learning_updates_for_account(
    db: Session,
    *,
    product_id: str,
    account_id: str,
    incremental_learnings: Dict[str, Any],
) -> None:
    """
    Persist per-account learning hints into ShmLearningUpdate so that
    summarize_global_insights() can aggregate them across accounts.

    Expects the shape from incremental_learnings_from_thesis(thesis):

      {
        "persona_path_inferences": [...],
        "graph_level_inferences": [...],
        "neighborhoods": {
            "upstream": [...],
            "handoff": [...],
            "downstream": [...],
        },
      }
    """
    print("starting to record learning updates for account:", account_id)
    persona_path = incremental_learnings.get("persona_path_inferences") or []
    graph_level = incremental_learnings.get("graph_level_inferences") or []
    nbs = incremental_learnings.get("neighborhoods") or {}
    upstream_nbs = nbs.get("upstream") or []
    handoff_nbs = nbs.get("handoff") or []
    downstream_nbs = nbs.get("downstream") or []

    # ---- 1) Persona-path inferences  (node-level) ----
    for row in persona_path:
        pid = row.get("persona_id")
        if not pid:
            continue

        delta = row.get("delta")
        explanation = row.get("explanation")

        db.add(
            ShmLearningUpdate(
                product_id=product_id,
                account_id=account_id,
                node_id=pid,
                field="starter_likelihood",   # or "perceptibility" – your call
                band="persona_path",
                update_type=ShmUpdateType.path_delta,
                confidence=delta,
                delta=delta,
                payload={
                    **row,
                    "reason": explanation,
                    "labels": {
                        "persona": row.get("persona_label"),
                    },
                },
            )
        )

    # ---- 2) Graph-level inferences (persona-centric node theses) ----
    for row in graph_level:
        pid = row.get("persona_id")
        if not pid:
            continue

        thesis = row.get("thesis")
        db.add(
            ShmLearningUpdate(
                product_id=product_id,
                account_id=account_id,
                node_id=pid,
                field="graph_thesis",
                band="upstream",
                update_type=ShmUpdateType.graph_level_thesis,
                confidence=row.get("delta"),
                delta=row.get("delta"),
                payload={
                    **row,
                    "reason": thesis,
                    "labels": {
                        "persona": row.get("persona_label"),
                        "jobs": row.get("jobs"),
                        "pains_solved": row.get("pains_solved"),
                        "triggers": row.get("triggers"),
                    },
                },
            )
        )

    # ---- 3) Neighborhoods: upstream = node-ish, handoff = edge-ish ----

    # 3a) upstream neighborhoods → node-level updates
    for nb in upstream_nbs:
        anchor = nb.get("anchor")
        if not isinstance(anchor, str):
            continue

        db.add(
            ShmLearningUpdate(
                product_id=product_id,
                account_id=account_id,
                node_id=anchor,
                field="motif",
                band=nb.get("band") or "upstream",
                update_type=ShmUpdateType.neighborhood,
                confidence=nb.get("delta_sum"),
                delta=nb.get("delta_sum"),
                payload={
                    **nb,
                    "labels": {
                        "anchor": nb.get("anchor_label"),
                    },
                    "reason": nb.get("rationale"),
                },
            )
        )

    # 3b) handoff neighborhoods → edge-level updates
    for nb in handoff_nbs:
        anchor = nb.get("anchor") or []
        if not (isinstance(anchor, (list, tuple)) and len(anchor) == 2):
            continue
        u, v = anchor

        db.add(
            ShmLearningUpdate(
                product_id=product_id,
                account_id=account_id,
                u_node_id=u,
                v_node_id=v,
                band=nb.get("band") or "handoff",
                update_type=ShmUpdateType.handoff_neighborhood,
                confidence=nb.get("delta_sum"),
                delta=nb.get("delta_sum"),
                payload={
                    **nb,
                    "labels": {
                        "pair": nb.get("anchor_labels"),
                    },
                    "reason": nb.get("rationale"),
                },
            )
        )

    # 3c) downstream neighborhoods – treat similar to upstream for now
    for nb in downstream_nbs:
        anchor = nb.get("anchor")
        if isinstance(anchor, str):
            db.add(
                ShmLearningUpdate(
                    product_id=product_id,
                    account_id=account_id,
                    node_id=anchor,
                    field="downstream",
                    band=nb.get("band") or "downstream",
                    update_type="neighborhood",
                    confidence=nb.get("delta_sum"),
                    delta=nb.get("delta_sum"),
                    payload={
                        **nb,
                        "labels": {
                            "anchor": nb.get("anchor_label"),
                        },
                        "reason": nb.get("rationale"),
                    },
                )
            )
    print("finished recording learning updates for account:", account_id)

    # NOTE: commit/rollback is done by caller

    account_adjustment = incremental_learnings.get("account_adjustment") or {}
    edges_adj = account_adjustment.get("edges") or []
    nodes_adj = account_adjustment.get("nodes") or []

    for row in edges_adj:
        src = row.get("source_id")
        dst = row.get("target_id")
        if not src or not dst:
            continue
        delta = row.get("delta") or 0.0
        update_type = (
            ShmUpdateType.structure_strengthen_edge
            if delta >= 0
            else ShmUpdateType.structure_weaken_edge
        )
        db.add(
            ShmLearningUpdate(
                product_id=product_id,
                account_id=account_id,
                u_node_id=str(src),
                v_node_id=str(dst),
                band=row.get("diagnostics", {}).get("band") or account_adjustment.get("band"),
                update_type=update_type,
                confidence=row.get("confidence"),
                delta=delta,
                payload=row,
            )
        )

    for row in nodes_adj:
        nid = row.get("node_id")
        if not nid:
            continue
        delta = row.get("delta") or 0.0
        db.add(
            ShmLearningUpdate(
                product_id=product_id,
                account_id=account_id,
                node_id=str(nid),
                field=row.get("field"),
                band=row.get("band"),
                update_type=ShmUpdateType.graph_level_thesis,
                confidence=row.get("confidence"),
                delta=delta,
                payload=row,
            )
        )


def _normalize_candidate_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().split()).lower()
    return normalized or None


def compute_persona_candidate_stats(
    episodes: Iterable[Episode],
    window_days: int = 180,
) -> List[PersonaCandidateStat]:
    """
    Aggregate observed candidate persona labels from a stream of Episode records.
    """

    cutoff = None
    if window_days and window_days > 0:
        cutoff = datetime.utcnow() - timedelta(days=window_days)

    stats_acc: Dict[str, Dict[str, Any]] = {}

    for ep in episodes or []:
        ts = ep.timestamp
        if cutoff and ts and ts < cutoff:
            continue

        engagement = ep.engagement or {}
        label = engagement.get("candidate_persona_label")
        normalized_label = _normalize_candidate_label(label)
        if not normalized_label:
            continue

        acc = stats_acc.setdefault(
            normalized_label,
            {
                "titles": set(),
                "departments": set(),
                "accounts": set(),
                "episodes": set(),
                "segments": Counter(),
                "occurrence_count": 0,
                "first_seen_at": None,
                "last_seen_at": None,
            },
        )

        acc["occurrence_count"] += 1
        if ep.account_id:
            acc["accounts"].add(ep.account_id)
            acc["episodes"].add(ep.account_id)

        actor = engagement.get("actor") or {}
        title = actor.get("title")
        department = actor.get("department")
        if title:
            acc["titles"].add(title.strip())
        if department:
            acc["departments"].add(department.strip())

        account_meta = ep.account_meta or engagement.get("account_meta") or {}
        if account_meta:
            normalized_meta = normalize_meta_dict(account_meta) or {}
            segments = segment_keys_from_meta(normalized_meta) or []
            for seg in segments:
                acc["segments"][seg] += 1

        if ts:
            if not acc["first_seen_at"] or ts < acc["first_seen_at"]:
                acc["first_seen_at"] = ts
            if not acc["last_seen_at"] or ts > acc["last_seen_at"]:
                acc["last_seen_at"] = ts

    results: List[PersonaCandidateStat] = []
    for label, data in stats_acc.items():
        results.append(
            PersonaCandidateStat(
                label=label,
                titles=sorted(t for t in data["titles"] if t),
                departments=sorted(d for d in data["departments"] if d),
                account_count=len(data["accounts"]),
                episode_count=len(data["episodes"]),
                occurrence_count=data["occurrence_count"],
                first_seen_at=data["first_seen_at"],
                last_seen_at=data["last_seen_at"],
                segments=dict(data["segments"]),
            )
        )

    results.sort(key=lambda stat: (stat.account_count, stat.occurrence_count), reverse=True)
    return results


def get_high_confidence_persona_candidates(
    episodes: Iterable[Episode],
    *,
    window_days: int = 180,
    min_accounts: int = 3,
    min_occurrences: int = 5,
) -> List[PersonaCandidateStat]:
    stats = compute_persona_candidate_stats(episodes, window_days=window_days)
    return [
        stat
        for stat in stats
        if stat.account_count >= min_accounts and stat.occurrence_count >= min_occurrences
    ]


def _normalize_candidate_label_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text or None


def _segment_list_from_row(seg_value: Any) -> List[str]:
    if not seg_value:
        return []
    if isinstance(seg_value, list):
        return [str(item) for item in seg_value if item]
    if isinstance(seg_value, str):
        try:
            loaded = json.loads(seg_value)
            if isinstance(loaded, list):
                return [str(item) for item in loaded if item]
        except Exception:
            return [seg_value]
    return [str(seg_value)]


def _aggregate_candidate_persona_data(
    db: Session,
    product_id: str,
    *,
    window_days: int = 180,
) -> Tuple[List[PersonaCandidateStat], Dict[str, Counter], Dict[str, Set[str]]]:
    cutoff = None
    if window_days and window_days > 0:
        cutoff = datetime.utcnow() - timedelta(days=window_days)

    query = (
        db.query(
            TargetAccountEngagement.candidate_persona_label,
            TargetAccountEngagement.actor_title,
            TargetAccountEngagement.actor_department,
            TargetAccountEngagement.actor_seniority,
            TargetAccountEngagement.target_account_id,
            TargetAccountEngagement.timestamp_dt,
            TargetAccountEngagement.persona_id,
            TargetAccountEngagement.segment_keys,
        )
        .filter(
            TargetAccountEngagement.product_id == product_id,
            TargetAccountEngagement.candidate_persona_label.isnot(None),
        )
    )
    if cutoff:
        query = query.filter(TargetAccountEngagement.timestamp_dt >= cutoff)

    stats_map: Dict[str, Dict[str, Any]] = {}
    co_occurrence: Dict[str, Counter] = defaultdict(Counter)
    persona_account_map: Dict[str, Set[str]] = defaultdict(set)

    for row in query.all():
        label = row.candidate_persona_label
        normalized_label = _normalize_candidate_label_value(label)
        if not normalized_label:
            continue

        entry = stats_map.setdefault(
            normalized_label,
            {
                "display_label": label.strip().title() if label else normalized_label.title(),
                "titles": set(),
                "departments": set(),
                "accounts": set(),
                "occurrence_count": 0,
                "first_seen_at": None,
                "last_seen_at": None,
                "segments": Counter(),
            },
        )

        entry["occurrence_count"] += 1
        if row.actor_title:
            entry["titles"].add(row.actor_title.strip())
        if row.actor_department:
            entry["departments"].add(row.actor_department.strip())
        if row.target_account_id:
            entry["accounts"].add(row.target_account_id)

        timestamp = row.timestamp_dt
        if timestamp:
            if not entry["first_seen_at"] or timestamp < entry["first_seen_at"]:
                entry["first_seen_at"] = timestamp
            if not entry["last_seen_at"] or timestamp > entry["last_seen_at"]:
                entry["last_seen_at"] = timestamp

        for seg in _segment_list_from_row(row.segment_keys):
            entry["segments"][seg] += 1

        if row.persona_id and row.target_account_id:
            co_occurrence[normalized_label][row.persona_id] += 1
            persona_account_map[row.persona_id].add(row.target_account_id)

    stats: List[PersonaCandidateStat] = []
    for normalized_label, data in stats_map.items():
        stats.append(
            PersonaCandidateStat(
                label=data["display_label"],
                titles=sorted(t for t in data["titles"] if t),
                departments=sorted(d for d in data["departments"] if d),
                account_count=len(data["accounts"]),
                episode_count=len(data["accounts"]),
                occurrence_count=data["occurrence_count"],
                first_seen_at=data["first_seen_at"],
                last_seen_at=data["last_seen_at"],
                segments=dict(data["segments"]),
            )
        )

    stats.sort(key=lambda stat: (stat.account_count, stat.occurrence_count), reverse=True)
    return stats, co_occurrence, persona_account_map


def _load_account_outcomes(db: Session, product_id: str) -> Dict[str, str]:
    rows = (
        db.query(TargetAccountORM.id, TargetAccountORM.deal_status)
        .filter(TargetAccountORM.product_id == product_id)
        .all()
    )
    outcomes: Dict[str, str] = {}
    for row_id, status in rows:
        label = (status or "").lower()
        if "won" in label:
            norm = "won"
        elif "lost" in label:
            norm = "lost"
        else:
            norm = "open"
        outcomes[str(row_id)] = norm
    return outcomes


def run_persona_wolves_learning(
    product_id: str,
    *,
    window_days: int = 180,
    min_accounts: int = 3,
    min_occurrences: int = 5,
) -> Dict[str, Any]:
    """
    End-to-end pipeline:
      1. Aggregate candidate stats from engagements.
      2. Auto-promote high-confidence personas into the graph.
      3. Recompute persona impact metrics and persist them.
    """
    db: Session = next(get_db())
    try:
        stats, co_occurrence, persona_account_map = _aggregate_candidate_persona_data(
            db,
            product_id,
            window_days=window_days,
        )
        graph = build_product_graph(product_id)
        new_nodes = auto_add_persona_nodes_from_candidates(
            graph,
            stats,
            co_occurrence=co_occurrence,
            min_accounts=min_accounts,
            min_occurrences=min_occurrences,
        )
        if new_nodes:
            save_graph_as_json(graph, product_id)

        account_outcomes = _load_account_outcomes(db, product_id)
        metrics = compute_persona_impact_metrics(
            graph,
            persona_account_map=persona_account_map,
            account_outcomes=account_outcomes,
        )
        if metrics:
            save_persona_metrics(product_id, metrics)

        return {
            "candidates": [asdict(stat) for stat in stats],
            "new_personas": new_nodes,
            "metrics": [asdict(metric) for metric in metrics],
        }
    finally:
        db.close()
