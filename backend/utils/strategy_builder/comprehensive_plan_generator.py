from __future__ import annotations

import logging
import json
import math
import os
import random
import difflib
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from copy import deepcopy

import networkx as nx

from backend.database import SessionLocal
from backend.utils.crm_management.person_models import (
    AccountPersonaMatch,
    AccountPerson,
)
from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.crm_management.target_account_manager import (
    get_account_by_id,
    get_target_account_ids,
)
from backend.utils.graph_base.network_graph import (
    build_product_graph,
    get_node_by_id,
    get_product_id_from_subgraph,
)
from backend.utils.inference.belief_manager.belief_manager import (
    build_belief_thesis_for_account,
    get_shm_transition_metrics,
)
from backend.utils.inference.belief_manager.cache import (
    get_cached_belief_thesis,
)
from backend.utils.inference.belief_manager.journey.learn_service import (
    summarize_global_insights,
)
from backend.utils.inference.belief_manager.journey.win_regression import (
    WinProbabilityCalibrator,
)
from backend.utils.inference.belief_manager.journey.storage import load_weights
from backend.utils.inference.belief_manager.journey.activity_story import build_activity_story
from backend.utils.inference.belief_manager.journey.storyline import compose_storyline
from backend.utils.knowledge_base.arsenal import service as arsenal_service
from backend.utils.embedding.embed_utils import get_embedding
from backend.utils.segment_utils import segment_keys_from_meta, segment_label_from_key
from backend.utils.inference.strategy.intervention_mode import choose_intervention_mode

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & filesystem helpers
# ---------------------------------------------------------------------------

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(MODULE_DIR)
GRAPH_BASE_DIR = os.path.join(UTILS_DIR, "graph_base")
ARSENAL_DIR = os.path.join(GRAPH_BASE_DIR, "graph_data", "arsenal_json")
PERSONA_METRICS_DIR = os.path.join(GRAPH_BASE_DIR, "graph_data", "persona_metrics")
_NEW_PERSONA_WINDOW_DAYS = 45


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _load_persona_wolves_metrics(product_id: str) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    path = os.path.join(PERSONA_METRICS_DIR, f"{product_id}.json")
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}, None
    updated_at = payload.get("updated_at")
    metrics_map: Dict[str, Dict[str, Any]] = {}
    for entry in payload.get("personas", []):
        pid = str(entry.get("persona_id") or "")
        if not pid:
            continue
        metrics_map[pid] = entry
    return metrics_map, updated_at


def _build_wolves_label_index(
    metrics_map: Optional[Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    if not metrics_map:
        return {}
    index: Dict[str, Dict[str, Any]] = {}
    for entry in metrics_map.values():
        label = entry.get("persona_label") or entry.get("label")
        if isinstance(label, str) and label.strip():
            index[label.strip().lower()] = entry
    return index


def _lookup_wolves_metric(
    persona_id: Optional[str],
    persona_label: Optional[str],
    metrics_map: Optional[Dict[str, Dict[str, Any]]],
    label_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    if not metrics_map:
        return None
    if persona_id:
        payload = metrics_map.get(str(persona_id))
        if payload:
            return payload
    if persona_label and label_index:
        return label_index.get(persona_label.strip().lower())
    if persona_label:
        lowered = persona_label.strip().lower()
        for entry in metrics_map.values():
            label = entry.get("persona_label") or entry.get("label")
            if isinstance(label, str) and label.strip().lower() == lowered:
                return entry
    return None


def _is_recent_persona_node(node: Dict[str, Any], days: int = _NEW_PERSONA_WINDOW_DAYS) -> bool:
    source = (node.get("source") or node.get("data_source") or "").lower()
    if source not in {"data_auto", "enrich_user"}:
        return False
    created_at = node.get("created_at") or node.get("last_updated")
    if not created_at:
        return True
    created_dt = _parse_iso_datetime(created_at)
    if not created_dt:
        return False
    return datetime.utcnow() - created_dt <= timedelta(days=days)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_catalog(filename: str, product_id: str) -> List[Dict[str, Any]]:
    path = os.path.join(ARSENAL_DIR, filename)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except Exception:
        return []
    return data.get(product_id, [])


def _segment_priority_boost_map(
    pattern: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    if not pattern:
        return {}
    mapping: Dict[str, float] = {}
    freq_rows = pattern.get("persona_frequency") or []
    for idx, row in enumerate(freq_rows):
        pid = row.get("persona_id")
        if not pid:
            continue
        share = _safe_float(row.get("share"), None)
        boost = 1.0 + min(0.4, (share or 0.2) * 1.5)
        boost -= idx * 0.05
        mapping[pid] = max(mapping.get(pid, 1.0), boost)
    return mapping


def _build_segment_context(
    meta_detail: Dict[str, Any],
    segment_patterns: Dict[str, Any],
) -> Dict[str, Any]:
    segment_keys = segment_keys_from_meta(meta_detail)
    segment_labels = [segment_label_from_key(key) for key in segment_keys]
    primary_pattern = None
    for key in segment_keys:
        pattern = segment_patterns.get(key)
        if pattern:
            primary_pattern = deepcopy(pattern)
            primary_pattern["key"] = key
            primary_pattern.setdefault("label", segment_label_from_key(key))
            break
    return {
        "keys": segment_keys,
        "labels": segment_labels,
        "pattern": primary_pattern,
    }


_ARSENAL_IMPACT_INDEX: Dict[str, Dict[str, Dict[Tuple[str, ...], Dict[str, Any]]]] = {}

_ASSET_BASE_LIFT_DEFAULTS: Dict[str, float] = {
    "webinar": 55.0,
    "report": 42.0,
    "ebook": 36.0,
    "case_study": 34.0,
    "playbook": 33.0,
    "demo": 60.0,
    "sales_call": 68.0,
    "toolkit": 38.0,
    "email_sequence": 28.0,
}

_ASSET_TIME_TO_EFFECT_DEFAULTS: Dict[str, int] = {
    "webinar": 7,
    "report": 21,
    "ebook": 14,
    "case_study": 10,
    "playbook": 14,
    "demo": 5,
    "sales_call": 2,
    "toolkit": 10,
    "email_sequence": 3,
}

_CHANNEL_DEFAULTS: Dict[str, Tuple[float, float]] = {
    "email": (0.55, 0.6),
    "sales_outreach": (0.45, 0.35),
    "paid_social": (0.78, 0.85),
    "paid_search": (0.6, 0.75),
    "display": (0.5, 0.9),
    "webinar_platform": (0.58, 0.8),
    "community": (0.35, 0.55),
    "field_event": (0.4, 0.5),
    "in_app": (0.52, 0.65),
    "direct_mail": (0.32, 0.3),
    "partner": (0.48, 0.45),
    "website": (0.62, 0.7),
    "organic_social": (0.5, 0.75),
}

_POSITIVE_SIGNAL_HINTS: Tuple[str, ...] = (
    "booked",
    "scheduled",
    "registered",
    "attended",
    "joined",
    "completed",
    "downloaded",
    "submitted",
    "requested",
    "trial",
    "pilot",
    "poc",
    "demo",
    "meeting",
    "call",
    "workshop",
    "roundtable",
    "kickoff",
    "approval",
    "signed",
)

_FATIGUE_RECENCY_WINDOW_DAYS = 45
_FATIGUE_DENSITY_WINDOW_DAYS = 30

_ASSET_SCORE_NORMALIZER = 1.5
_CHANNEL_SCORE_NORMALIZER = 3.5

_STAGE_TOKEN_MAP: Dict[str, List[str]] = {
    "problem": ["problem_realization", "problem_awareness"],
    "problem_realization": ["problem_realization", "problem_awareness"],
    "problem_awareness": ["problem_realization", "problem_awareness"],
    "pain": ["problem_realization", "problem_awareness"],
    "pain_realization": ["problem_realization", "problem_awareness"],
    "solution": ["solution_exploration", "promised_land"],
    "solution_exploration": ["solution_exploration", "promised_land"],
    "resolution": ["solution_exploration", "promised_land"],
    "resolution_discovery": ["solution_exploration", "promised_land"],
    "promised_land": ["solution_exploration", "promised_land"],
    "evaluation": ["evaluation", "decision"],
    "decision": ["evaluation", "decision"],
    "execution": ["evaluation", "decision"],
    "execution_guidance": ["evaluation", "decision"],
}

BELIEF_STAGE_LABELS: Dict[str, str] = {
    "problem": "Problem Realization",
    "execution": "Execution Guidance",
    "pain": "Pain Realization",
    "resolution": "Resolution Discovery",
}

PORTFOLIO_STAGE_CANONICALS: Tuple[Tuple[str, str], ...] = (
    ("zero moment", "Zero Moment of Truth"),
    ("zmot", "Zero Moment of Truth"),
    ("problem realization", "Problem Realization"),
    ("problem", "Problem Realization"),
    ("pain realization", "Pain Realization"),
    ("pain", "Pain Realization"),
    ("resolution discovery", "Resolution Discovery"),
    ("resolution", "Resolution Discovery"),
    ("barrier-breaking", "Barrier-breaking"),
    ("execution guidance", "Execution Guidance"),
    ("execution", "Execution Guidance"),
)

PORTFOLIO_STAGE_ORDER: Dict[str, int] = {
    "zero moment of truth": 0,
    "problem realization": 1,
    "pain realization": 2,
    "resolution discovery": 3,
    "barrier-breaking": 4,
    "execution guidance": 5,
}

_FUNNEL_STAGE_LABELS: Dict[str, str] = {
    "early": "Early Cycle",
    "mid": "Mid Cycle",
    "late": "Late Cycle",
}

PEOPLE_STAGE_SEQUENCE: Tuple[str, ...] = (
    "Problem Realization",
    "Execution Guidance",
    "Pain Realization",
    "Resolution Discovery",
)

SEGMENT_ATTRIBUTE_KEYS = [
    "industry",
    "geography",
    "revenue_range",
    "employee_range",
    "funding_stage",
    "competitor_used",
    "other_tech_stack",
]

SEGMENT_ATTRIBUTE_LABELS = {
    "industry": "Industry",
    "geography": "Geography",
    "revenue_range": "Revenue Range",
    "employee_range": "Employee Range",
    "funding_stage": "Funding Stage",
    "competitor_used": "Competitor",
    "other_tech_stack": "Tech Stack",
}

SEGMENT_LABEL_TO_KEY = {
    **{label.lower(): key for key, label in SEGMENT_ATTRIBUTE_LABELS.items()},
    **{key.lower(): key for key in SEGMENT_ATTRIBUTE_KEYS},
}


def _iso_or_none(value: Any) -> Optional[str]:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return None
    return None


def _person_display_name(person: Optional[AccountPerson]) -> Optional[str]:
    if not person:
        return None
    parts = [
        (person.name or "").strip(),
        (person.title or "").strip(),
        (person.department or "").strip(),
        (person.seniority or "").strip(),
    ]
    parts = [part for part in parts if part]
    if parts:
        return " · ".join(parts)
    return (person.name or "").strip() or None


def _aggregate_segment_filters(
    segment_dicts: Iterable[Optional[Dict[str, Any]]],
) -> Dict[str, str]:
    aggregates: Dict[str, set] = {key: set() for key in SEGMENT_ATTRIBUTE_KEYS}
    for segment in segment_dicts:
        if not segment:
            continue
        for key in SEGMENT_ATTRIBUTE_KEYS:
            value = segment.get(key)
            if value in (None, "", []):
                continue
            if isinstance(value, (list, tuple, set)):
                items = value
            else:
                items = (value,)
            for item in items:
                if item in (None, "", []):
                    continue
                if isinstance(item, (list, dict, set)):
                    aggregates[key].add(str(item))
                else:
                    aggregates[key].add(item)

    result: Dict[str, str] = {}
    for key in SEGMENT_ATTRIBUTE_KEYS:
        values = aggregates[key]
        if not values:
            result[key] = "any"
        elif len(values) == 1:
            result[key] = next(iter(values))
        else:
            result[key] = "any"
    return result


def _segment_summary_text(segment_filters: Dict[str, str]) -> Optional[str]:
    parts: List[str] = []
    for key in SEGMENT_ATTRIBUTE_KEYS:
        value = segment_filters.get(key)
        if value and value != "any":
            label = SEGMENT_ATTRIBUTE_LABELS.get(key, _title_case_value(key) or key)
            parts.append(f"{label}: {value}")
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts)


def _account_segment_map(meta: Optional[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not meta:
        return mapping
    for key in SEGMENT_ATTRIBUTE_KEYS:
        value = meta.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                mapping[key] = cleaned
    return mapping


def _account_segment_strings(meta: Optional[Dict[str, Any]]) -> List[str]:
    segments: List[str] = []
    if not meta:
        return segments
    for key in SEGMENT_ATTRIBUTE_KEYS:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            label = SEGMENT_ATTRIBUTE_LABELS.get(key, _title_case_value(key) or key)
            segments.append(f"{label}: {value.strip()}")
    return segments


def _cluster_label_from_token(token: str) -> str:
    if not token or ":" not in token:
        return token or "Mixed cluster"
    key, value = token.split(":", 1)
    value = value.replace("|", " · ").strip()
    key = key.strip().lower()
    if key == "industry":
        return f"Industry · {value}"
    if key == "geography":
        return f"Region · {value}"
    if key == "revenue_range":
        return f"Revenue · {value}"
    if key == "employee_range":
        return f"Employee size · {value}"
    if key == "funding_stage":
        return f"Funding · {value}"
    if key == "deal_status":
        return f"Deal status · {value}"
    if key == "persona":
        return f"Persona focus · {value}"
    if key == "combo_industry_geography":
        parts = value.split(" · ")
        if len(parts) == 2:
            return f"{parts[0]} + {parts[1]}"
    return value or token


def _derive_account_clusters(
    entries: Sequence[Dict[str, Any]],
    accounts_by_id: Dict[str, Dict[str, Any]],
    *,
    total_accounts: int = 0,
    total_delta_bp: float = 0.0,
    max_clusters: int = 4,
    persona_wolves_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    persona_wolves_label_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if not entries:
        return []
    total_score = sum(entry.get("score", 0.0) or 0.0 for entry in entries)
    if total_score <= 0:
        total_score = float(len(entries)) or 1.0
    persona_wolves_metrics = persona_wolves_metrics or {}
    persona_wolves_label_index = persona_wolves_label_index or {}
    token_scores: Counter[str] = Counter()
    token_accounts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        tokens = entry.get("tokens") or []
        score = entry.get("score", 0.0) or 0.0
        if score <= 0:
            score = 1.0
        for token in tokens:
            token_scores[token] += score
            token_accounts[token].append(entry)

    def _cluster_account_refs(
        token: str,
    ) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
        refs: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
        seen_ids: set[str] = set()
        for entry in token_accounts.get(token) or []:
            raw_id = entry.get("id")
            key = str(raw_id) if raw_id else None
            if key and key in seen_ids:
                continue
            if key:
                seen_ids.add(key)
            refs.append((entry, accounts_by_id.get(key)))
        return refs

    def _persona_coalition_story(
        persona_label: Optional[str],
        coalitions: Sequence[Dict[str, Any]],
    ) -> Optional[str]:
        if not persona_label or not coalitions:
            return None
        for coalition in coalitions:
            sequence = coalition.get("sequence") or []
            if persona_label not in sequence:
                continue
            share = coalition.get("share")
            try:
                idx = sequence.index(persona_label)
            except ValueError:
                continue
            followers = sequence[idx + 1 : idx + 3]
            if followers:
                follower_text = " and ".join(followers)
                return (
                    f"Engaging here unlocks {follower_text} in {_percent_label(share)} of wins."
                    if share is not None
                    else f"Engaging here unlocks {follower_text} in most modeled wins."
                )
            if share is not None:
                return f"Anchors {_percent_label(share)} of modeled wins."
        return None

    def _build_cluster_keystone_details(
        persona_expectations: Sequence[Dict[str, Any]],
        belief_coalitions: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        if not persona_expectations:
            return [], None

        def _sort_key(entry: Dict[str, Any]) -> Tuple[int, float, float]:
            has_wolves = 1 if entry.get("wolves_score") is not None else 0
            return (
                has_wolves,
                entry.get("wolves_score") or 0.0,
                entry.get("share") or 0.0,
            )

        sorted_entries = sorted(
            persona_expectations,
            key=_sort_key,
            reverse=True,
        )
        keystones: List[Dict[str, Any]] = []
        for entry in sorted_entries[:3]:
            story = _persona_coalition_story(entry.get("persona"), belief_coalitions)
            keystones.append(
                {
                    "persona": entry.get("persona"),
                    "persona_id": entry.get("persona_id"),
                    "stage": entry.get("stage"),
                    "share": entry.get("share"),
                    "match_rate": entry.get("match_rate"),
                    "wolves_score": entry.get("wolves_score"),
                    "wolves_delta_bp": entry.get("wolves_delta_bp"),
                    "wolves_involvement_rate": entry.get("wolves_involvement_rate"),
                    "wolves_blocker_rate": entry.get("wolves_blocker_rate"),
                    "wolves_sample_size": entry.get("wolves_sample_size"),
                    "coalition_story": story,
                }
            )

        caption = None
        if keystones:
            lead = keystones[0]
            if lead.get("coalition_story"):
                caption = f"{lead['persona']}: {lead['coalition_story']}"
            elif lead.get("share") is not None:
                caption = (
                    f"{lead['persona']} anchors {_percent_label(lead['share'])} of journeys."
                )
        return keystones, caption

    def _cluster_persona_expectations(
        account_refs: Sequence[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        persona_profiles: Dict[str, Dict[str, Any]] = {}
        total_weight = 0.0
        for entry, account in account_refs:
            weight = float(entry.get("score") or 0.0) or 1.0
            total_weight += weight
            requirements = (
                ((account or {}).get("enrichment") or {}).get("persona_requirements") or []
            )
            if requirements:
                for req in requirements:
                    label = (
                        req.get("persona_label")
                        or req.get("persona_title")
                        or req.get("persona_id")
                        or "Target persona"
                    )
                    profile = persona_profiles.setdefault(
                        label,
                        {
                            "weight": 0.0,
                            "stage_counts": Counter(),
                            "required": 0,
                            "with_matches": 0,
                            "matches": [],
                        },
                    )
                    persona_id = (
                        req.get("persona_id")
                        or (req.get("persona") or {}).get("id")
                        or req.get("id")
                    )
                    if persona_id and not profile.get("persona_id"):
                        profile["persona_id"] = persona_id
                    wolves_info = _lookup_wolves_metric(
                        persona_id,
                        label,
                        persona_wolves_metrics,
                        persona_wolves_label_index,
                    )
                    if wolves_info:
                        profile["wolves_score"] = _safe_float(
                            wolves_info.get("wolves_score"), None
                        )
                        profile["wolves_delta_bp"] = _safe_float(
                            wolves_info.get("delta_win_bp"), None
                        )
                        profile["wolves_involvement_rate"] = _safe_float(
                            wolves_info.get("involvement_rate"), None
                        )
                        profile["wolves_blocker_rate"] = _safe_float(
                            wolves_info.get("blocker_rate"), None
                        )
                        profile["wolves_sample_size"] = wolves_info.get("sample_size")
                    profile["weight"] += weight
                    profile["required"] += 1
                    if req.get("has_match"):
                        profile["with_matches"] += 1
                    stage_label = req.get("stage_label")
                    if stage_label:
                        profile["stage_counts"][stage_label] += weight
                    for person in req.get("matched_people") or []:
                        profile["matches"].append(
                            {
                                "name": person.get("display_name")
                                or person.get("person_name")
                                or person.get("person_id"),
                                "title": person.get("person_title"),
                                "department": person.get("person_department"),
                                "seniority": person.get("person_seniority"),
                                "account_id": (account or {}).get("account_id") or entry.get("id"),
                                "account_name": (account or {}).get("account_name") or entry.get("name"),
                                "confidence": person.get("match_confidence"),
                                "source": person.get("source"),
                            }
                        )
            else:
                for label in entry.get("personas") or []:
                    profile = persona_profiles.setdefault(
                        label,
                        {
                            "weight": 0.0,
                            "stage_counts": Counter(),
                            "required": 0,
                            "with_matches": 0,
                            "matches": [],
                        },
                    )
                    wolves_info = _lookup_wolves_metric(
                        None,
                        label,
                        persona_wolves_metrics,
                        persona_wolves_label_index,
                    )
                    if wolves_info:
                        profile["wolves_score"] = _safe_float(
                            wolves_info.get("wolves_score"), None
                        )
                        profile["wolves_delta_bp"] = _safe_float(
                            wolves_info.get("delta_win_bp"), None
                        )
                        profile["wolves_involvement_rate"] = _safe_float(
                            wolves_info.get("involvement_rate"), None
                        )
                        profile["wolves_blocker_rate"] = _safe_float(
                            wolves_info.get("blocker_rate"), None
                        )
                        profile["wolves_sample_size"] = wolves_info.get("sample_size")
                    profile["weight"] += weight
                    profile["required"] += 1
        if not persona_profiles:
            return [], []
        if total_weight <= 0:
            total_weight = sum(profile["weight"] or 1.0 for profile in persona_profiles.values()) or 1.0
        persona_expectations: List[Dict[str, Any]] = []
        match_records: List[Dict[str, Any]] = []
        for label, profile in persona_profiles.items():
            stage_counter: Counter[str] = profile.get("stage_counts") or Counter()
            stage = stage_counter.most_common(1)[0][0] if stage_counter else None
            share = (profile["weight"] / total_weight) if total_weight else 0.0
            required = profile.get("required", 0)
            match_rate = (
                (profile.get("with_matches", 0) / required) if required else None
            )
            sample_people = profile.get("matches", [])[:5]
            for person in sample_people:
                record = dict(person)
                record["persona"] = label
                record["stage"] = stage
                match_records.append(record)
            persona_expectations.append(
                {
                    "persona": label,
                    "persona_id": profile.get("persona_id"),
                    "stage": stage,
                    "share": share,
                    "match_rate": match_rate,
                    "required_personas": required,
                    "sample_people": sample_people,
                    "wolves_score": profile.get("wolves_score"),
                    "wolves_delta_bp": profile.get("wolves_delta_bp"),
                    "wolves_involvement_rate": profile.get("wolves_involvement_rate"),
                    "wolves_blocker_rate": profile.get("wolves_blocker_rate"),
                    "wolves_sample_size": profile.get("wolves_sample_size"),
                }
            )
        persona_expectations.sort(key=lambda item: item.get("share", 0.0), reverse=True)
        return persona_expectations[:5], match_records[:8]

    def _cluster_belief_coalitions(
        account_refs: Sequence[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        coalition_counter: Counter[Tuple[str, ...]] = Counter()
        total = 0.0
        for entry, account in account_refs:
            weight = float(entry.get("score") or 0.0) or 1.0
            paths = ((account or {}).get("prediction") or {}).get("persona_paths") or []
            for path in paths[:3]:
                probability = float(path.get("probability") or 0.0)
                if probability <= 0:
                    continue
                labels = tuple(
                    persona.get("label")
                    for persona in path.get("personas") or []
                    if persona.get("label")
                )
                if len(labels) < 2:
                    continue
                contribution = probability * weight
                if contribution <= 0:
                    continue
                coalition_counter[labels] += contribution
                total += contribution
        if not coalition_counter:
            return []
        coalitions = []
        for seq, contribution in coalition_counter.most_common(4):
            coalitions.append(
                {
                    "sequence": list(seq),
                    "score": round(contribution, 3),
                    "share": contribution / total if total else 0.0,
                }
            )
        return coalitions

    def _cluster_belief_transitions(
        account_refs: Sequence[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        transitions: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
        for entry, account in account_refs:
            plays = ((account or {}).get("execution") or {}).get("plays") or []
            for play in plays:
                bt = play.get("belief_transition") or {}
                persona_label = (bt.get("persona") or {}).get("label") or play.get("persona_label")
                if not persona_label:
                    continue
                stage_label = bt.get("stage_label") or play.get("stage_label")
                narrative = bt.get("narrative")
                key = (persona_label, stage_label, narrative)
                bucket = transitions.setdefault(
                    key,
                    {
                        "persona": persona_label,
                        "stage": stage_label,
                        "narrative": narrative,
                        "pain": (bt.get("pain") or {}).get("label"),
                        "resolution": (bt.get("resolution") or {}).get("label"),
                        "delta_bp": 0.0,
                        "accounts": set(),
                    },
                )
                bucket["delta_bp"] += float(play.get("expected_delta_bp") or 0.0)
                bucket["accounts"].add(entry.get("id"))
        results: List[Dict[str, Any]] = []
        for bucket in transitions.values():
            results.append(
                {
                    "persona": bucket.get("persona"),
                    "stage": bucket.get("stage"),
                    "narrative": bucket.get("narrative"),
                    "pain": bucket.get("pain"),
                    "resolution": bucket.get("resolution"),
                    "delta_bp": round(bucket.get("delta_bp", 0.0), 2),
                    "account_count": len(bucket.get("accounts") or []),
                }
            )
        results.sort(key=lambda item: item.get("delta_bp", 0.0), reverse=True)
        return results[:5]

    def _cluster_primary_plays(
        account_refs: Sequence[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        play_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for entry, account in account_refs:
            account_id = entry.get("id")
            plays = ((account or {}).get("execution") or {}).get("plays") or []
            for play in plays:
                asset = play.get("asset") or {}
                channel = play.get("channel") or {}
                asset_label = asset.get("name") or asset.get("title") or asset.get("id") or "Asset"
                channel_label = (
                    channel.get("name")
                    or channel.get("channel_type_label")
                    or channel.get("type")
                    or play.get("channel_label")
                    or "Channel"
                )
                key = (asset_label, channel_label)
                bucket = play_map.setdefault(
                    key,
                    {
                        "asset": asset_label,
                        "asset_type": asset.get("format") or asset.get("type") or play.get("asset_type_label"),
                        "channel": channel_label,
                        "channel_type": channel.get("channel_type") or channel.get("type"),
                        "delta_bp": 0.0,
                        "personas": set(),
                        "stages": set(),
                        "accounts": set(),
                    },
                )
                bucket["delta_bp"] += float(play.get("expected_delta_bp") or 0.0)
                if play.get("persona_label"):
                    bucket["personas"].add(play["persona_label"])
                stage = play.get("stage_label") or (play.get("belief_transition") or {}).get("stage_label")
                if stage:
                    bucket["stages"].add(stage)
                bucket["accounts"].add(account_id)
        primary: List[Dict[str, Any]] = []
        for bucket in play_map.values():
            primary.append(
                {
                    "asset": bucket["asset"],
                    "asset_type": bucket["asset_type"],
                    "channel": bucket["channel"],
                    "channel_type": bucket["channel_type"],
                    "delta_bp": round(bucket["delta_bp"], 2),
                    "persona_labels": sorted(bucket["personas"]),
                    "stage_labels": sorted(bucket["stages"]),
                    "account_count": len(bucket["accounts"]),
                }
            )
        primary.sort(key=lambda item: item.get("delta_bp", 0.0), reverse=True)
        return primary[:5]

    def _cluster_fallback_plan(
        account_refs: Sequence[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        fallback_map: Dict[str, Dict[str, Any]] = {}
        for entry, account in account_refs:
            account_id = entry.get("id")
            account_name = (account or {}).get("account_name") or entry.get("name") or account_id
            conversion_sequence = ((account or {}).get("execution") or {}).get("conversion_sequence") or []
            for node in conversion_sequence:
                plays = node.get("plays") or []
                matched = node.get("matched_people") or []
                if plays and matched:
                    continue
                persona_label = (node.get("persona") or {}).get("label") or "Next persona"
                key = f"{persona_label}|{node.get('timeline_index')}"
                bucket = fallback_map.setdefault(
                    key,
                    {
                        "persona": persona_label,
                        "stage": (node.get("belief_transition_meta") or {}).get("stage_label"),
                        "reasons": [],
                        "accounts": set(),
                    },
                )
                summary_texts = [
                    node.get("expected_outcome_summary"),
                    node.get("segment_summary"),
                ]
                for text in summary_texts:
                    if text and text not in bucket["reasons"]:
                        bucket["reasons"].append(text)
                bucket["accounts"].add(account_name)
            unmatched = ((account or {}).get("enrichment") or {}).get("unmatched_personas") or []
            for persona in unmatched:
                label = (
                    persona.get("persona_label")
                    or persona.get("persona_title")
                    or persona.get("persona_id")
                    or "Persona"
                )
                key = f"unmatched:{label}"
                bucket = fallback_map.setdefault(
                    key,
                    {
                        "persona": label,
                        "stage": persona.get("stage_label"),
                        "reasons": [],
                        "accounts": set(),
                    },
                )
                reason = persona.get("reason") or "No mapped person yet"
                if reason not in bucket["reasons"]:
                    bucket["reasons"].append(reason)
                bucket["accounts"].add(account_name)
        fallback_items: List[Dict[str, Any]] = []
        for bucket in fallback_map.values():
            fallback_items.append(
                {
                    "persona": bucket.get("persona"),
                    "stage": bucket.get("stage"),
                    "reason": " · ".join(bucket.get("reasons") or []),
                    "accounts": sorted(bucket.get("accounts") or []),
                }
            )
        fallback_items.sort(key=lambda item: len(item.get("accounts") or []), reverse=True)
        return fallback_items[:4]

    clusters: List[Dict[str, Any]] = []
    for token, score in token_scores.most_common(max_clusters):
        account_refs = _cluster_account_refs(token)
        if not account_refs:
            continue
        persona_expectations, people_matches = _cluster_persona_expectations(account_refs)
        belief_coalitions = _cluster_belief_coalitions(account_refs)
        belief_transitions = _cluster_belief_transitions(account_refs)
        primary_plays = _cluster_primary_plays(account_refs)
        fallback_plan = _cluster_fallback_plan(account_refs)

        cluster_accounts: List[Dict[str, Any]] = []
        seen_pairs: set[Tuple[Any, Any]] = set()
        for entry, account in account_refs:
            acc_id = entry.get("id") or (account or {}).get("account_id")
            acc_name = (account or {}).get("account_name") or entry.get("name") or acc_id
            pair = (acc_id, acc_name)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            cluster_accounts.append({"id": acc_id, "name": acc_name})

        cluster_delta_bp = sum(
            max(float((entry or {}).get("score") or 0.0), 0.0) for entry, _ in account_refs
        )
        share = cluster_delta_bp / total_delta_bp if total_delta_bp else score / total_score
        account_count = len(cluster_accounts) or len(account_refs)
        avg_per_account = (total_delta_bp / total_accounts) if total_accounts else 0.0
        generic_allocation = avg_per_account * account_count if avg_per_account else 0.0
        incremental_vs_generic = cluster_delta_bp - generic_allocation
        per_account_delta = cluster_delta_bp / account_count if account_count else None
        remaining_accounts = max(total_accounts - account_count, 0)
        peer_per_account = (
            (total_delta_bp - cluster_delta_bp) / remaining_accounts
            if remaining_accounts and (total_delta_bp - cluster_delta_bp)
            else None
        )
        keystone_personas, keystone_caption = _build_cluster_keystone_details(
            persona_expectations,
            belief_coalitions,
        )

        clusters.append(
            {
                "label": _cluster_label_from_token(token),
                "token": token,
                "account_count": account_count,
                "share_of_expected_lift": share,
                "accounts": cluster_accounts,
                "persona_expectations": persona_expectations,
                "people_matches": people_matches,
                "belief_coalitions": belief_coalitions,
                "belief_transitions": belief_transitions,
                "primary_plays": primary_plays,
                "fallback_plan": fallback_plan,
                 "keystone_personas": keystone_personas,
                 "keystone_caption": keystone_caption,
                "lift_analysis": {
                    "cluster_delta_bp": round(cluster_delta_bp, 2),
                    "generic_allocation_bp": round(generic_allocation, 2),
                    "incremental_vs_generic_bp": round(incremental_vs_generic, 2),
                    "per_account_delta_bp": round(per_account_delta, 2) if per_account_delta is not None else None,
                    "peer_per_account_delta_bp": round(peer_per_account, 2)
                    if peer_per_account is not None
                    else None,
                },
            }
        )
    return clusters


def _segment_entry_to_key_value(entry: Any) -> Tuple[Optional[str], Optional[str]]:
    if entry is None:
        return None, None
    if isinstance(entry, dict):
        key = entry.get("key") or entry.get("attribute")
        value = entry.get("value")
        if key and value:
            normalized = SEGMENT_LABEL_TO_KEY.get(str(key).strip().lower())
            return normalized, str(value).strip()
        entry = entry.get("label") or entry.get("text")
        if entry is None:
            return None, None
    text = str(entry)
    if not text:
        return None, None
    if ":" in text:
        label, raw_value = text.split(":", 1)
    else:
        parts = text.split()
        if len(parts) >= 2:
            label, raw_value = parts[0], " ".join(parts[1:])
        else:
            return None, None
    key = SEGMENT_LABEL_TO_KEY.get(label.strip().lower())
    value = raw_value.strip()
    if not key or not value:
        return None, None
    return key, value


def _segment_fit_score(
    target_segments: Optional[Iterable[Any]],
    account_segments: Optional[Dict[str, str]],
) -> Tuple[float, List[str]]:
    if not target_segments:
        return 0.0, []
    if not account_segments:
        return 0.0, []
    matches: List[str] = []
    parsed_segments: List[Tuple[str, str, str]] = []
    for entry in target_segments:
        key, value = _segment_entry_to_key_value(entry)
        if key and value:
            parsed_segments.append((key, value.lower(), str(entry)))
    if not parsed_segments:
        return 0.0, []
    for key, value, raw in parsed_segments:
        account_value = account_segments.get(key)
        if not account_value:
            continue
        account_norm = account_value.lower()
        if value == "any" or account_norm == value:
            matches.append(raw)
        elif value in account_norm or account_norm in value:
            matches.append(raw)
    total = len(parsed_segments)
    match_ratio = len(matches) / total if total else 0.0
    if matches:
        score = 0.65 + 0.45 * match_ratio
    else:
        score = -0.5
    return round(score, 3), matches


_STAGE_TOKEN_HINTS: Tuple[Tuple[str, str], ...] = (
    ("problem", "problem"),
    ("aware", "problem"),
    ("pain", "pain"),
    ("need", "pain"),
    ("solution", "resolution"),
    ("resolution", "resolution"),
    ("evaluation", "resolution"),
    ("execute", "execution"),
    ("guidance", "execution"),
)


def _stage_code_from_tokens(tokens: Optional[Sequence[str]]) -> Optional[str]:
    if not tokens:
        return None
    for token in tokens:
        normalized = str(token or "").strip().lower()
        if not normalized:
            continue
        for hint, stage in _STAGE_TOKEN_HINTS:
            if hint in normalized:
                return stage
    return None


def _segment_candidates_from_map(account_segments: Optional[Dict[str, str]]) -> List[str]:
    if not account_segments:
        return []
    candidates: List[str] = []
    for key, value in account_segments.items():
        if not value:
            continue
        candidates.append(f"{key}={value}")
    return candidates


def _persona_descriptor_from_meta(
    meta: Dict[str, Any],
    fallback: Optional[str] = None,
) -> str:
    label = meta.get("label") or fallback or "Persona"
    title = meta.get("title")
    department = meta.get("department")
    seniority = meta.get("seniority")

    if seniority and department:
        return f"{_title_case_value(seniority) or seniority} {department}"
    if department and (label or "").lower() != department.lower():
        return f"{department} {label}"
    if seniority and (label or "").lower() != str(seniority).lower():
        return f"{_title_case_value(seniority) or seniority} {label}"
    if title and title.lower() not in (label or "").lower():
        return f"{title}"
    return label


def _format_bps(value: Optional[float]) -> str:
    if value is None:
        return "0bps"
    return f"{round(float(value)):,}bps"


def _build_expected_outcome_summary(
    persona_descriptor: Optional[str],
    stage_label: Optional[str],
    delta_bp: Optional[float],
    account_count: int,
    segment_filters: Dict[str, str],
) -> str:
    persona_text = persona_descriptor or "target personas"
    stage_text = stage_label or "the next belief milestone"
    delta_text = _format_bps(delta_bp)
    account_text = (
        f"across {account_count} account{'s' if account_count != 1 else ''}"
        if account_count
        else "across target accounts"
    )
    segment_text = _segment_summary_text(segment_filters)
    segment_clause = f" for {segment_text}" if segment_text else ""
    return (
        f"Get {persona_text}{segment_clause} to {stage_text} to increase conversion "
        f"likelihood by {delta_text} {account_text}."
    )



def _serialize_persona_match(
    match: AccountPersonaMatch,
    person: Optional[AccountPerson],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "match_id": match.id,
        "product_id": match.product_id,
        "account_id": match.account_id,
        "persona_id": match.persona_id,
        "persona_label": match.persona_label,
        "stage": match.stage,
        "person_id": match.person_id,
        "match_confidence": match.match_confidence,
        "source": match.source,
        "notes": match.notes,
        "created_at": _iso_or_none(match.created_at),
        "updated_at": _iso_or_none(match.updated_at),
    }
    if person:
        payload.update(
            {
                "person_name": person.name,
                "person_title": person.title,
                "person_department": person.department,
                "person_seniority": person.seniority,
                "engagement_count": person.engagement_count,
                "last_seen_at": _iso_or_none(person.last_seen_at),
            }
        )
    display = _person_display_name(person)
    if display:
        payload["display_name"] = display
    return payload


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


def _persona_fatigue_from_matches(
    people: Sequence[Dict[str, Any]]
) -> Tuple[float, str]:
    if not people:
        return 0.05, "No mapped people"
    now = datetime.utcnow()
    recency_components: List[float] = []
    volume_components: List[float] = []
    for entry in people:
        count = _safe_float(entry.get("engagement_count"), 0.0) or 0.0
        volume_components.append(min(count / 8.0, 1.0))
        dt = _parse_iso_datetime(entry.get("last_seen_at"))
        if dt:
            days = max(0.0, (now - dt).total_seconds() / 86400.0)
            recency_components.append(1.0 - min(days / 45.0, 1.0))
        else:
            recency_components.append(0.25)
    avg_recency = sum(recency_components) / len(recency_components) if recency_components else 0.0
    avg_volume = sum(volume_components) / len(volume_components) if volume_components else 0.0
    simultaneity = min(len(people) / 4.0, 1.0)
    fatigue = min(
        0.95,
        max(0.05, 0.45 * avg_recency + 0.35 * avg_volume + 0.2 * simultaneity),
    )
    reason_bits = []
    if avg_recency > 0.2:
        reason_bits.append(f"Recent activity {_percent_label(avg_recency)}")
    if avg_volume > 0.2:
        reason_bits.append(f"Touch volume {_percent_label(avg_volume)}")
    if simultaneity > 0.25:
        reason_bits.append("Multiple people engaged")
    if not reason_bits:
        reason_bits.append("Light engagement intensity")
    return round(fatigue, 3), " · ".join(reason_bits)


def _normalize_name_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip().lower()
    return cleaned or None


def _is_positive_signal(entry: Dict[str, Any]) -> bool:
    payload = entry.get("payload") or {}
    text_bits = [
        entry.get("raw_activity"),
        payload.get("action"),
        payload.get("activity"),
        payload.get("asset_label"),
        entry.get("channel"),
        payload.get("intent"),
    ]
    normalized = " ".join(bit for bit in text_bits if bit).lower()
    if normalized:
        if any(keyword in normalized for keyword in _POSITIVE_SIGNAL_HINTS):
            return True
    intent = str(payload.get("intent") or "").lower()
    return intent in {"high", "handraise", "positive", "responded"}


def _top_counter_entries(counter: Counter, limit: int = 3) -> List[Dict[str, Any]]:
    if not counter:
        return []
    rows: List[Dict[str, Any]] = []
    for label, count in counter.most_common(limit):
        if not count:
            continue
        rows.append({"label": label or "Unspecified", "count": int(count)})
    return rows


def _load_engagement_buckets(
    product_id: str,
    account_id: str,
) -> Dict[str, List[Dict[str, Any]]]:
    try:
        with SessionLocal() as db:
            rows = (
                db.query(TargetAccountEngagement)
                .filter(
                    TargetAccountEngagement.product_id == product_id,
                    TargetAccountEngagement.target_account_id == account_id,
                )
                .all()
            )
    except Exception:
        return {}

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _normalize_name_key(row.actor_name)
        if not key:
            continue
        ts = row.timestamp_dt
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = _parse_iso_datetime(row.timestamp)
            if ts and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        if not ts:
            ts = datetime.now(timezone.utc)
        record = {
            "timestamp": ts,
            "raw_activity": row.raw_activity or "",
            "channel": row.channel or "",
            "source": row.source or "",
            "payload": row.payload or {},
        }
        record["is_positive"] = _is_positive_signal(record)
        buckets[key].append(record)

    for entries in buckets.values():
        entries.sort(key=lambda item: item["timestamp"])
    return buckets


def _persona_engagement_snapshot(
    persona_label: Optional[str],
    people: Sequence[Dict[str, Any]],
    engagement_buckets: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if not people or not engagement_buckets:
        return None

    keys: List[str] = []
    for person in people:
        for attr in ("person_name", "display_name", "notes"):
            key = _normalize_name_key(person.get(attr))
            if key and key not in keys:
                keys.append(key)
    if not keys:
        return None

    events: List[Dict[str, Any]] = []
    for key in keys:
        events.extend(engagement_buckets.get(key, []))
    if not events:
        return None

    events.sort(key=lambda item: item["timestamp"])
    now_dt = datetime.now(timezone.utc)
    total = len(events)
    positive_count = sum(1 for event in events if event.get("is_positive"))
    last_ts = events[-1]["timestamp"]
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    days_since_last = max(0.0, (now_dt - last_ts).total_seconds() / 86400.0)

    last_positive_ts: Optional[datetime] = None
    for event in reversed(events):
        if event.get("is_positive"):
            ts = event["timestamp"]
            if isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            last_positive_ts = ts
            break

    if last_positive_ts is not None:
        days_since_last_positive = max(
            0.0, (now_dt - last_positive_ts).total_seconds() / 86400.0
        )
    else:
        days_since_last_positive = None

    density_window = timedelta(days=_FATIGUE_DENSITY_WINDOW_DAYS)
    touches_recent = [
        event
        for event in events
        if (now_dt - event["timestamp"]) <= density_window
    ]
    channel_counter: Counter = Counter(
        (event.get("channel") or "Unspecified") for event in events
    )
    source_counter: Counter = Counter(
        (event.get("source") or "Unspecified") for event in events
    )

    non_positive_ratio = (
        float(total - positive_count) / float(total)
        if total
        else 0.0
    )
    recency_intensity = 1.0 - min(
        days_since_last / float(_FATIGUE_RECENCY_WINDOW_DAYS),
        1.0,
    )
    stagnation = 1.0
    if positive_count:
        gap = days_since_last_positive if days_since_last_positive is not None else days_since_last
        stagnation = min(gap / float(_FATIGUE_RECENCY_WINDOW_DAYS), 1.0)
    density = min(
        len(touches_recent)
        / max(_FATIGUE_DENSITY_WINDOW_DAYS / 7.0, 1.0),
        1.2,
    )

    fatigue = (
        0.5 * non_positive_ratio
        + 0.2 * recency_intensity
        + 0.15 * density
        + 0.15 * stagnation
    )
    fatigue = max(0.05, min(0.95, fatigue))

    reason_bits: List[str] = []
    if positive_count:
        reason_bits.append(f"{positive_count}/{total} positive signals")
    else:
        reason_bits.append("No positive signal yet")
    if days_since_last is not None:
        reason_bits.append(f"Last touch {int(round(days_since_last))}d ago")
    if days_since_last_positive is not None:
        reason_bits.append(
            f"Positive {int(round(days_since_last_positive))}d ago"
        )
    reason_bits.append(
        f"{len(touches_recent)} touches in {_FATIGUE_DENSITY_WINDOW_DAYS}d"
    )

    positive_rate = (positive_count / total) if total else 0.0
    density_per_week = len(touches_recent) / max(
        _FATIGUE_DENSITY_WINDOW_DAYS / 7.0,
        1.0,
    )

    return {
        "persona_label": persona_label,
        "total_touches": total,
        "positive_signals": positive_count,
        "positive_rate": round(positive_rate, 4),
        "last_touch_days": round(days_since_last, 1),
        "last_positive_days": round(days_since_last_positive, 1)
        if days_since_last_positive is not None
        else None,
        "touch_density_per_week": round(density_per_week, 3),
        "top_channels": _top_counter_entries(channel_counter),
        "top_sources": _top_counter_entries(source_counter),
        "fatigue": round(fatigue, 3),
        "fatigue_reason": " · ".join(reason_bits),
    }


def _load_persona_matches_lookup(
    product_id: str,
    account_id: str,
) -> Dict[str, List[Dict[str, Any]]]:
    try:
        with SessionLocal() as db:
            rows = (
                db.query(AccountPersonaMatch, AccountPerson)
                .outerjoin(AccountPerson, AccountPersonaMatch.person_id == AccountPerson.id)
                .filter(
                    AccountPersonaMatch.product_id == product_id,
                    AccountPersonaMatch.account_id == account_id,
                )
                .all()
            )
    except Exception:
        return {}

    matches_by_persona: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for match, person in rows:
        payload = _serialize_persona_match(match, person)
        matches_by_persona[match.persona_id].append(payload)

    for payloads in matches_by_persona.values():
        payloads.sort(
            key=lambda item: (
                -(item.get("match_confidence") or 0.0),
                item.get("person_name") or item.get("display_name") or "",
            )
        )
    return matches_by_persona


def _stage_from_average_order(order_avg: float) -> Tuple[int, str]:
    if not isinstance(order_avg, (int, float)) or math.isnan(order_avg):
        idx = 0
    else:
        idx = int(round(order_avg))
    idx = max(0, min(idx, len(PEOPLE_STAGE_SEQUENCE) - 1))
    return idx, PEOPLE_STAGE_SEQUENCE[idx]


def _compute_person_enrichment_requirements(
    persona_paths: Sequence[Dict[str, Any]],
    matches_by_persona: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}

    for path in persona_paths:
        probability = float(path.get("probability") or 0.0)
        for persona in path.get("personas") or []:
            persona_id = persona.get("id")
            if not persona_id:
                continue
            bucket = stats.setdefault(
                persona_id,
                {
                    "count": 0,
                    "order_sum": 0.0,
                    "prob_sum": 0.0,
                    "label": None,
                    "title": None,
                    "department": None,
                    "seniority": None,
                },
            )
            bucket["count"] += 1
            bucket["order_sum"] += float(persona.get("order", 0))
            bucket["prob_sum"] += probability
            for key in ("label", "title", "department", "seniority"):
                if not bucket.get(key):
                    bucket[key] = persona.get(key)

    requirements: List[Dict[str, Any]] = []
    required = 0
    with_matches = 0
    total_people = 0

    for persona_id, bucket in stats.items():
        if not bucket["count"]:
            continue
        avg_order = bucket["order_sum"] / bucket["count"]
        stage_index, stage_label = _stage_from_average_order(avg_order)
        expected = min(1.0, bucket["prob_sum"])
        matched_people = matches_by_persona.get(persona_id, [])
        people_names = [
            entry.get("display_name")
            or entry.get("person_name")
            or entry.get("notes")
            or entry.get("person_id")
            for entry in matched_people
        ]
        people_names = [name for name in people_names if name]
        requirement = {
            "persona_id": persona_id,
            "persona_label": bucket.get("label"),
            "persona_title": bucket.get("title"),
            "persona_department": bucket.get("department"),
            "persona_seniority": bucket.get("seniority"),
            "expected_in_deal": expected,
            "expected_in_deal_pct": round(expected * 100, 1),
            "stage_index": stage_index,
            "stage_label": stage_label,
            "matched_people": matched_people,
            "people_names": people_names,
            "match_count": len(matched_people),
            "has_match": bool(matched_people),
        }
        requirements.append(requirement)
        required += 1
        total_people += len(matched_people)
        if matched_people:
            with_matches += 1

    requirements.sort(
        key=lambda row: (
            -float(row.get("expected_in_deal") or 0.0),
            row.get("persona_label") or "",
        )
    )

    coverage_ratio = (with_matches / required) if required else 0.0
    summary = {
        "required_personas": required,
        "personas_with_matches": with_matches,
        "matched_people": total_people,
        "coverage_ratio": coverage_ratio,
        "coverage_pct": round(coverage_ratio * 100, 1),
        "total_people": total_people,
    }
    return requirements, summary


def _node_label(node: Any) -> Optional[str]:
    if not node:
        return None
    if isinstance(node, str):
        value = node.strip()
        return value or None
    if isinstance(node, dict):
        for key in ("label", "name", "title", "text", "value"):
            raw = node.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return None


def _persona_focus_label(value: Optional[str]) -> str:
    if not value:
        return "Persona"
    parts = [part.strip() for part in str(value).split("|") if part and part.strip()]
    if not parts:
        return str(value).strip()
    primary = parts[0]
    secondary = parts[1] if len(parts) > 1 else None
    if secondary:
        return f"{primary} · {secondary}"
    return primary


def _people_focus_label(
    persona_label: Optional[str],
    people_names: Sequence[str],
) -> str:
    names = [name for name in people_names if name]
    if names:
        head = ", ".join(names[:2])
        if len(names) > 2:
            head = f"{head} +{len(names) - 2}"
        if persona_label:
            persona_focus = _persona_focus_label(persona_label)
            return f"{head} ({persona_focus})"
        return head
    return _persona_focus_label(persona_label)


def _focus_core_label(transition: Dict[str, Any]) -> Optional[str]:
    if not transition:
        return None
    for key in ("pain", "resolution", "problem"):
        label = _node_label(transition.get(key))
        if label:
            return label
    return _node_label(transition.get("narrative"))


def _conversion_focus_summary_label(
    stage_label: Optional[str],
    focus_core: Optional[str],
    fallback: Optional[str] = None,
) -> str:
    if stage_label and focus_core:
        return f"{stage_label}: {focus_core}"
    if stage_label:
        return stage_label
    if focus_core:
        return focus_core
    return fallback or "Belief shift"


def _stage_for_transition(
    belief_transition: Dict[str, Any],
    *,
    stage_index: int,
) -> str:
    if belief_transition.get("stage"):
        return belief_transition["stage"]
    if belief_transition.get("pain"):
        return "pain"
    if belief_transition.get("resolution"):
        return "resolution"
    if belief_transition.get("problem"):
        return "problem"
    if stage_index == 0:
        return "problem"
    if stage_index >= 3:
        return "resolution"
    if stage_index >= 2:
        return "pain"
    return "execution"


def _title_case_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    parts = str(value).replace("|", " ").replace("_", " ").split()
    return " ".join(part.capitalize() for part in parts) if parts else None


def _ensure_list(values: Any) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [segment.strip() for segment in values.split(",") if segment and segment.strip()]
    items: List[str] = []
    if isinstance(values, (list, tuple, set)):
        for entry in values:
            if entry is None:
                continue
            if isinstance(entry, str):
                candidate = entry.strip()
                if candidate:
                    items.append(candidate)
            elif isinstance(entry, dict):
                for key in ("label", "value", "name", "title"):
                    raw = entry.get(key)
                    if isinstance(raw, str):
                        candidate = raw.strip()
                        if candidate:
                            items.append(candidate)
                            break
            else:
                candidate = str(entry).strip()
                if candidate:
                    items.append(candidate)
    else:
        candidate = str(values).strip()
        if candidate:
            items.append(candidate)
    return items


def _extract_funnel_stage(candidate_values: Any) -> Optional[Dict[str, str]]:
    if candidate_values in (None, "", [], ()):
        return None

    def _stage_from_text(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        cleaned = str(text).strip().lower()
        if not cleaned:
            return None
        cleaned = cleaned.replace("-", " ").replace("_", " ").replace(":", " ")
        parts = [part for part in cleaned.split() if part]
        if not parts:
            return None
        if parts[0] == "funnel":
            for token in parts[1:]:
                if token in _FUNNEL_STAGE_LABELS:
                    return token
        for token in parts:
            if token in _FUNNEL_STAGE_LABELS:
                return token
        return None

    iterable: Iterable[Any]
    if isinstance(candidate_values, (list, tuple, set)):
        iterable = candidate_values
    else:
        iterable = [candidate_values]

    for entry in iterable:
        stage_code: Optional[str] = None
        if isinstance(entry, dict):
            code_value = entry.get("code")
            if isinstance(code_value, str):
                stage_code = _stage_from_text(code_value)
            if not stage_code:
                label_value = (
                    entry.get("label")
                    or entry.get("value")
                    or entry.get("name")
                    or entry.get("title")
                )
                if isinstance(label_value, str):
                    stage_code = _stage_from_text(label_value)
        elif isinstance(entry, str):
            stage_code = _stage_from_text(entry)
        else:
            stage_code = _stage_from_text(str(entry))

        if stage_code and stage_code in _FUNNEL_STAGE_LABELS:
            return {
                "code": stage_code,
                "label": _FUNNEL_STAGE_LABELS[stage_code],
            }
    return None


def _stage_tokens_for_label(label: Optional[str]) -> List[str]:
    if not label:
        return []
    key = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    tokens = _STAGE_TOKEN_MAP.get(key)
    if tokens:
        return tokens
    return [key]


def _stage_for_transition(
    belief_transition: Dict[str, Any],
    *,
    stage_index: int,
) -> str:
    if belief_transition.get("pain"):
        return "pain"
    if belief_transition.get("resolution"):
        return "resolution"
    if belief_transition.get("problem"):
        return "problem"
    if stage_index == 0:
        return "problem"
    if stage_index >= 3:
        return "resolution"
    if stage_index >= 2:
        return "pain"
    return "execution"


def _default_base_lift(category: Optional[str], format_value: Optional[str]) -> float:
    candidates = [
        (category or "").lower(),
        (format_value or "").lower(),
    ]
    for candidate in candidates:
        if candidate in _ASSET_BASE_LIFT_DEFAULTS:
            return _ASSET_BASE_LIFT_DEFAULTS[candidate]
    return 35.0


def _default_time_to_effect(category: Optional[str], format_value: Optional[str]) -> int:
    candidates = [
        (category or "").lower(),
        (format_value or "").lower(),
    ]
    for candidate in candidates:
        if candidate in _ASSET_TIME_TO_EFFECT_DEFAULTS:
            return _ASSET_TIME_TO_EFFECT_DEFAULTS[candidate]
    return 14


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _normalize_stage_label(label: Optional[str]) -> str:
    if not label:
        return "Next Stage"
    cleaned = str(label).strip()
    if not cleaned:
        return "Next Stage"
    lower = cleaned.lower()
    for needle, canonical in PORTFOLIO_STAGE_CANONICALS:
        if needle in lower:
            return canonical
    return cleaned


def _stage_order_value(label: Optional[str]) -> int:
    normalized = _normalize_stage_label(label).lower()
    return PORTFOLIO_STAGE_ORDER.get(normalized, len(PORTFOLIO_STAGE_ORDER) + 10)


def _quarter_sort_key(value: Optional[str]) -> int:
    if not value:
        return 0
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else 0


def _percent_label(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{int(round(value * 100))}%"


def _weeks_label(days: Optional[float]) -> str:
    if days is None:
        return "—"
    return f"{int(round(days / 7.0))}w"


def _portfolio_stage_distribution(accounts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stage_counter: Counter[str] = Counter()
    for account in accounts:
        for entry in account.get("execution", {}).get("conversion_sequence") or []:
            meta = entry.get("belief_transition_meta") or {}
            raw_stage = meta.get("stage_label") or entry.get("stage_label")
            stage_label = _normalize_stage_label(raw_stage)
            stage_counter[stage_label] += 1
    rows = [
        {"stage": stage, "count": count}
        for stage, count in stage_counter.items()
    ]
    rows.sort(
        key=lambda row: (
            _stage_order_value(row["stage"]),
            -row["count"],
        )
    )
    return rows


def _portfolio_summary_report(
    accounts: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    portfolio_plan: Dict[str, Any],
) -> Dict[str, Any]:
    total_accounts = len(accounts)
    coverage_values: List[float] = []
    belief_scores: List[float] = []
    maturity_scores: List[float] = []
    timeline_days: List[float] = []
    channel_ids: set[str] = set()
    persona_asset_counter: Counter[str] = Counter()
    total_assets = 0
    total_delta = 0.0
    fatigue_alerts = 0

    def _channel_identifier(play: Dict[str, Any]) -> Optional[str]:
        channel = play.get("channel") or {}
        for key in (
            "id",
            "name",
            "slug",
            "channel_type",
            "type",
        ):
            if channel.get(key):
                return str(channel[key])
        for candidate in (
            play.get("channel_id"),
            play.get("channel_label"),
            play.get("channel_type"),
            play.get("channel_type_label"),
            play.get("channel_name"),
        ):
            if candidate:
                return str(candidate)
        return None

    for account in accounts:
        enrichment_summary = (account.get("enrichment") or {}).get("summary") or {}
        coverage = enrichment_summary.get("coverage_ratio")
        if isinstance(coverage, (int, float)):
            coverage_values.append(float(coverage))
        paths = account.get("prediction", {}).get("persona_paths") or []
        if paths:
            primary_prob = paths[0].get("probability")
            if isinstance(primary_prob, (int, float)):
                belief_scores.append(float(primary_prob))
        journey = (account.get("prediction") or {}).get("journey") or {}
        total_steps = journey.get("total_steps") or 0
        observed_steps = len(journey.get("steps") or [])
        if total_steps:
            maturity_scores.append(min(1.0, observed_steps / total_steps))

        plays = account.get("execution", {}).get("plays") or []
        total_assets += len(plays)
        for play in plays:
            total_delta += float(play.get("expected_delta_bp") or 0.0)
            channel_id = _channel_identifier(play)
            if channel_id:
                channel_ids.add(channel_id)
            persona_label = (
                play.get("persona_label")
                or play.get("persona_descriptor")
                or play.get("persona_focus")
            )
            if persona_label:
                persona_asset_counter[persona_label] += 1
        conversion_sequence = account.get("execution", {}).get("conversion_sequence") or []
        if conversion_sequence:
            last_entry = conversion_sequence[-1]
            timeline_value = last_entry.get("timeline_days")
            if isinstance(timeline_value, (int, float)):
                timeline_days.append(float(timeline_value))
        fatigue_entries = account.get("execution", {}).get("persona_engagements") or []
        for entry in fatigue_entries:
            fatigue = _safe_float(entry.get("fatigue"))
            if fatigue is not None and fatigue >= 0.6:
                fatigue_alerts += 1

    stage_distribution = _portfolio_stage_distribution(accounts)
    avg_coverage = _safe_mean(coverage_values)
    avg_belief = _safe_mean(belief_scores)
    avg_maturity = _safe_mean(maturity_scores)
    avg_timeline = _safe_mean(timeline_days)
    expected_conversion_rate = (
        portfolio_plan.get("randomization", {}).get("avg_path_probability")
        if portfolio_plan.get("randomization")
        else None
    )
    if expected_conversion_rate is None:
        expected_conversion_rate = avg_belief

    unmatched_counts = [
        len((account.get("enrichment") or {}).get("unmatched_personas") or [])
        for account in accounts
    ]

    high_risk_accounts: List[Dict[str, Any]] = []
    high_opportunity_accounts: List[Dict[str, Any]] = []
    for account in accounts:
        account_id = account.get("account_id")
        account_name = account.get("account_name") or account_id
        enrichment_summary = (account.get("enrichment") or {}).get("summary") or {}
        coverage = _safe_float(enrichment_summary.get("coverage_ratio"), 0.0) or 0.0
        fatigue_entries = account.get("execution", {}).get("persona_engagements") or []
        fatigue_values = [
            _safe_float(entry.get("fatigue"), 0.0) or 0.0 for entry in fatigue_entries
        ]
        avg_fatigue = _safe_mean(fatigue_values) or 0.0
        blocker_count = len((account.get("enrichment") or {}).get("unmatched_personas") or [])
        risk_score = (1.0 - coverage) + avg_fatigue + (blocker_count * 0.2)
        risk_reasons: List[str] = []
        if coverage < 0.6:
            risk_reasons.append(f"Coverage {_percent_label(coverage)}")
        if avg_fatigue > 0.45:
            risk_reasons.append(f"Fatigue {_percent_label(avg_fatigue)}")
        if blocker_count:
            risk_reasons.append(
                f"{blocker_count} persona gap{'s' if blocker_count != 1 else ''}"
            )
        high_risk_accounts.append(
            {
                "id": account_id,
                "name": account_name,
                "reason": " · ".join(risk_reasons) or "Monitoring",
                "score": round(risk_score, 3),
            }
        )

        paths = account.get("prediction", {}).get("persona_paths") or []
        primary_prob = _safe_float(paths[0].get("probability")) if paths else None
        conversion_sequence = account.get("execution", {}).get("conversion_sequence") or []
        last_entry = conversion_sequence[-1] if conversion_sequence else {}
        eta_days = _safe_float(last_entry.get("timeline_days"), 90.0) or 90.0
        time_factor = 1.0 - min(1.0, eta_days / 120.0)
        opportunity_score = (
            (coverage * 0.4)
            + ((primary_prob or 0.0) * 0.4)
            + (time_factor * 0.2)
        )
        opportunity_reasons = [
            f"Coverage {_percent_label(coverage)}",
        ]
        if primary_prob is not None:
            opportunity_reasons.append(f"Path p={_percent_label(primary_prob)}")
        opportunity_reasons.append(f"ETA {_weeks_label(eta_days)}")
        high_opportunity_accounts.append(
            {
                "id": account_id,
                "name": account_name,
                "reason": " · ".join(opportunity_reasons),
                "score": round(opportunity_score, 3),
            }
        )

    high_risk_accounts.sort(key=lambda item: item["score"], reverse=True)
    high_opportunity_accounts.sort(key=lambda item: item["score"], reverse=True)

    assets_per_persona = [
        {"persona": persona, "count": count}
        for persona, count in persona_asset_counter.most_common(8)
    ]

    return {
        "total_accounts": total_accounts,
        "coverage_pct": avg_coverage,
        "enriched_accounts": len([value for value in coverage_values if value >= 0.65]),
        "belief_score": avg_belief,
        "maturity_score": avg_maturity,
        "assets_planned": total_assets,
        "channel_count": len(channel_ids),
        "total_delta_bp": round(total_delta, 2),
        "expected_conversion_rate": expected_conversion_rate,
        "expected_time_to_conversion_days": avg_timeline,
        "stage_distribution": stage_distribution,
        "high_risk_accounts": high_risk_accounts[:4],
        "high_opportunity_accounts": high_opportunity_accounts[:4],
        "persona_mix": summary.get("top_personas", []),
        "pain_mix": summary.get("top_pains", []),
        "capability_mix": summary.get("top_capabilities", []),
        "fatigue_alerts": fatigue_alerts,
        "assets_per_persona": assets_per_persona,
        "unmatched_persona_totals": sum(unmatched_counts),
        "account_clusters": summary.get("account_clusters") or [],
    }


def _portfolio_thesis_data(
    summary: Dict[str, Any],
    portfolio_plan: Dict[str, Any],
    canonical_persona_path: Sequence[Dict[str, Any]],
    portfolio_summary_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    stage_distribution = (
        (portfolio_summary_report or {}).get("stage_distribution")
        or []
    )
    persona_sequence_source: Sequence[Dict[str, Any]] = canonical_persona_path or []
    if not persona_sequence_source:
        persona_sequence_source = (
            portfolio_plan.get("typical_path")
            or summary.get("canonical_persona_path")
            or []
        )
    persona_sequence: List[str] = []
    for idx, entry in enumerate(persona_sequence_source):
        label = (
            entry.get("label")
            or entry.get("persona")
            or entry.get("title")
            or entry.get("name")
        )
        persona_sequence.append(f"{idx + 1}. {label or 'Persona'}")

    top_pains = [item.get("label") for item in (summary.get("top_pains") or [])[:3] if item.get("label")]
    top_capabilities = [
        item.get("label")
        for item in (summary.get("top_capabilities") or [])[:3]
        if item.get("label")
    ]
    top_personas = [
        item.get("label")
        for item in (summary.get("top_personas") or [])[:3]
        if item.get("label")
    ]
    blockers = (portfolio_summary_report or {}).get("high_risk_accounts", [])
    segments = []
    for entry in portfolio_plan.get("campaign_themes") or []:
        seg = entry.get("segment_summary")
        if seg and seg not in segments:
            segments.append(seg)
    quarter_outcomes_raw = (
        (portfolio_plan.get("expected_outcomes") or {}).get("by_quarter") or []
    )
    quarter_outcomes = [
        {
            "quarter": entry.get("quarter") or "Quarter",
            "summary": entry.get("expected_outcome_summary")
            or "Outcome pending",
            "delta_bp": entry.get("total_delta_bp"),
            "conversion_ready_count": entry.get("conversion_ready_accounts"),
            "persona_descriptors": entry.get("persona_descriptor_counts"),
            "segment_summary": entry.get("segment_summary"),
        }
        for entry in quarter_outcomes_raw
    ]

    def _highlight_or_default(values: Sequence[str], prefix: str, default_note: str) -> str:
        if values:
            return f"{prefix} {', '.join(values)}."
        return default_note

    def _tidy_highlight_text(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        cleaned = " ".join(str(text).strip().split())
        cleaned = cleaned.rstrip(".")
        if not cleaned:
            return None
        words = cleaned.split(" ")
        dedup: List[str] = []
        for word in words:
            if dedup and dedup[-1].lower() == word.lower():
                continue
            dedup.append(word)
        normalized = " ".join(dedup)
        if not normalized:
            return None
        return normalized[0].upper() + normalized[1:]

    strategy_highlights = [
        _highlight_or_default(
            top_pains,
            "Starting pains concentrate on",
            "Starting pains remain distributed; prioritize discovery interviews.",
        ),
        _highlight_or_default(
            top_capabilities,
            "Dominant capabilities/themes:",
            "Capabilities need definition across campaigns.",
        ),
        _highlight_or_default(
            top_personas,
            "Most engaged personas:",
            "Persona engagement data is still sparse.",
        ),
        (
            f"Belief load concentrates in {stage_distribution[0]['stage']} "
            f"with {stage_distribution[0]['count']} personas queued."
            if stage_distribution
            else "Belief progression is evenly distributed."
        ),
        (
            f"High-risk accounts ({len(blockers)}) show fatigue or coverage gaps—pre-wire blockers early."
            if blockers
            else "No portfolio-level blockers surfaced yet."
        ),
    ]
    strategy_highlights = [
        tidy for tidy in (_tidy_highlight_text(item) for item in strategy_highlights) if tidy
    ]

    cluster_briefs = summary.get("account_clusters") or []

    return {
        "strategy_highlights": strategy_highlights,
        "persona_sequence": persona_sequence,
        "blockers": blockers,
        "belief_targets": stage_distribution,
        "segments": segments,
        "quarter_outcomes": quarter_outcomes,
        "persona_cadence": summary.get("persona_engagements") or [],
        "cluster_briefs": cluster_briefs,
    }


def _portfolio_arsenal_rows(accounts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for account in accounts:
        account_name = account.get("account_name") or account.get("account_id")
        for campaign in account.get("campaigns") or []:
            theme = campaign.get("theme") or "Campaign"
            quarter = campaign.get("quarter") or "Q1"
            for play in campaign.get("plays") or []:
                asset = play.get("asset") or {}
                channel = play.get("channel") or {}
                asset_name = (
                    asset.get("name")
                    or play.get("asset_label")
                    or asset.get("title")
                    or "Asset"
                )
                channel_name = (
                    channel.get("name")
                    or play.get("channel_label")
                    or play.get("channel_type")
                    or channel.get("channel_type")
                    or "Channel"
                )
                persona_label = (
                    play.get("persona_label")
                    or play.get("persona_descriptor")
                    or campaign.get("persona_focus")
                    or "Target persona"
                )
                linked_accounts = [
                    acct.get("name") or acct.get("id")
                    for acct in play.get("accounts") or []
                    if acct
                ]
                account_list = [
                    name for name in [account_name] + linked_accounts if name
                ]
                row_accounts = sorted(set(account_list))
                belief_transition = play.get("belief_transition") or {}
                stage_label = belief_transition.get("stage_label") or _journey_phase_label(
                    play.get("stage_index", 0),
                    5,
                )
                segment_filters = play.get("segment_filters")
                if not segment_filters:
                    meta_filters = account.get("meta") or {}
                    if meta_filters:
                        segment_filters = {
                            key: value
                            for key, value in meta_filters.items()
                            if isinstance(value, str) and value
                        }
                segment_summary = play.get("segment_summary")
                if not segment_summary and segment_filters:
                    segment_summary = _segment_summary_text(segment_filters)
                rows.append(
                    {
                        "theme": theme,
                        "quarter": quarter,
                        "asset": asset_name,
                        "channel": channel_name,
                        "persona": persona_label,
                        "accounts": row_accounts,
                        "delta_bp": _safe_float(play.get("expected_delta_bp"), 0.0)
                        or 0.0,
                        "confidence": _safe_float(play.get("confidence")),
                        "belief_conversion": _safe_float(
                            play.get("belief_conversion_likelihood")
                            or play.get("avg_belief_conversion")
                        ),
                        "time_to_impact_days": play.get("duration_days"),
                        "mode": play.get("mode"),
                        "stage_index": play.get("stage_index"),
                        "stage_label": stage_label,
                        "persona_id": play.get("persona_id"),
                        "belief_transition": belief_transition,
                        "asset_detail": asset,
                        "channel_detail": channel,
                        "reasons": play.get("reasons"),
                        "rationale": play.get("rationale"),
                        "segment_filters": segment_filters,
                        "segment_summary": segment_summary,
                        "account_id": account.get("account_id"),
                        "account_name": account_name,
                    }
                )
    rows.sort(
        key=lambda row: (
            _quarter_sort_key(row["quarter"]),
            row["theme"],
            row["persona"],
        )
    )
    return rows[:500]


def _standardize_asset_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    usage = raw.get("usage") or {}
    target_personas = _ensure_list(raw.get("target_personas"))
    usage_personas = [
        entry.get("label") or entry.get("name") or entry.get("id")
        for entry in usage.get("personas", [])
        if isinstance(entry, dict)
    ]
    persona_fit = target_personas or [p for p in usage_personas if p]

    usage_stage_candidates = [
        entry.get("code") or entry.get("label")
        for entry in usage.get("stages", [])
        if isinstance(entry, dict)
    ]
    stage_labels = _ensure_list(raw.get("target_belief_stages"))
    if not stage_labels:
        stage_labels = usage_stage_candidates

    funnel_stage = raw.get("funnel_stage")
    if not funnel_stage:
        funnel_stage = _extract_funnel_stage(raw.get("target_belief_stages"))
    if not funnel_stage:
        funnel_stage = _extract_funnel_stage(usage_stage_candidates)

    def _is_funnel_label(value: Optional[str]) -> bool:
        if not value:
            return False
        normalized = str(value).strip().lower()
        if not normalized:
            return False
        if normalized in _FUNNEL_STAGE_LABELS:
            return True
        return normalized.startswith("funnel")

    belief_stage_labels = [
        label for label in stage_labels if not _is_funnel_label(label)
    ]
    if not belief_stage_labels and usage_stage_candidates:
        belief_stage_labels = [
            label for label in usage_stage_candidates if not _is_funnel_label(label)
        ]

    stage_tokens: List[str] = []
    for stage in belief_stage_labels:
        stage_tokens.extend(_stage_tokens_for_label(stage))
    if not stage_tokens:
        stage_tokens.extend(_ensure_list(raw.get("stage_fit")))
    stage_tokens = list(dict.fromkeys(token for token in stage_tokens if token))

    concern_tags = _ensure_list(raw.get("target_concerns"))
    if not concern_tags:
        concern_tags = [
            entry.get("label")
            for entry in usage.get("concerns", [])
            if isinstance(entry, dict) and entry.get("label")
        ]

    category = (raw.get("category") or "").lower() or None
    format_value = raw.get("format") or category or raw.get("content_type") or raw.get("name")

    record = dict(raw)
    record["format"] = format_value
    record["category"] = category
    record["category_label"] = _title_case_value(category) if category else None
    record["persona_fit"] = persona_fit
    record["stage_fit"] = stage_tokens
    record["concern_tags"] = concern_tags
    record["target_personas"] = target_personas
    record["target_belief_stages"] = stage_labels
    record["target_concerns"] = concern_tags
    record["base_lift_bp"] = raw.get("base_lift_bp") or _default_base_lift(category, format_value)
    record["time_to_effect_days"] = raw.get("time_to_effect_days") or _default_time_to_effect(
        category, format_value
    )
    record["metadata_complete"] = bool(raw.get("metadata_complete"))
    record["usage"] = usage
    record["funnel_stage"] = funnel_stage
    record["learned_effectiveness"] = raw.get("learned_effectiveness") or []
    record["learned_summary"] = raw.get("learned_summary")
    learned_summary = record["learned_summary"] or {}
    if learned_summary.get("lift_bps") is not None:
        record["expected_lift_bp"] = learned_summary.get("lift_bps")
    else:
        record["expected_lift_bp"] = record["base_lift_bp"]
    if learned_summary.get("confidence") is not None:
        record["expected_confidence"] = learned_summary.get("confidence")
    return record


def _standardize_channel_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    usage = raw.get("usage") or {}
    channel_type = (raw.get("channel_type") or raw.get("type") or "").lower() or None
    reach, breadth = _CHANNEL_DEFAULTS.get(channel_type or "", (0.5, 0.5))
    reach_score = raw.get("reach_score") or raw.get("reach_score_estimate")
    breadth_multiplier = raw.get("breadth_multiplier")

    target_personas = _ensure_list(raw.get("target_personas"))
    if not target_personas:
        target_personas = [
            entry.get("label")
            for entry in usage.get("personas", [])
            if isinstance(entry, dict) and entry.get("label")
        ]

    stage_labels = _ensure_list(raw.get("target_belief_stages"))
    if not stage_labels:
        stage_labels = [
            entry.get("code") or entry.get("label")
            for entry in usage.get("stages", [])
            if isinstance(entry, dict)
        ]
    stage_tokens: List[str] = []
    for stage in stage_labels:
        stage_tokens.extend(_stage_tokens_for_label(stage))

    concerns = _ensure_list(raw.get("target_concerns"))
    if not concerns:
        concerns = [
            entry.get("label")
            for entry in usage.get("concerns", [])
            if isinstance(entry, dict) and entry.get("label")
        ]

    record = dict(raw)
    record["type"] = channel_type or raw.get("type")
    record["channel_type"] = channel_type or raw.get("type")
    record["channel_type_label"] = _title_case_value(channel_type) if channel_type else None
    record["reach_score"] = float(reach_score) if isinstance(reach_score, (int, float)) else reach
    record["reach_score_estimate"] = record["reach_score"]
    record["breadth_multiplier"] = (
        float(breadth_multiplier) if isinstance(breadth_multiplier, (int, float)) else breadth
    )
    record["target_personas"] = target_personas
    record["target_belief_stages"] = stage_labels
    record["target_concerns"] = concerns
    record["stage_fit"] = stage_tokens
    record["usage"] = usage
    record["metadata_complete"] = bool(raw.get("metadata_complete"))
    record["learned_effectiveness"] = raw.get("learned_effectiveness") or []
    record["learned_summary"] = raw.get("learned_summary")
    return record


def _index_impacts(impacts: Iterable[Dict[str, Any]]) -> Dict[str, Dict[Tuple[str, ...], Dict[str, Any]]]:
    by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    by_persona: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for impact in impacts or []:
        asset_id = impact.get("asset_id")
        channel_id = impact.get("channel_id")
        if not asset_id or not channel_id:
            continue
        impact_strength = impact.get("impact_strength")
        record = {
            "impact_strength": float(impact_strength) if isinstance(impact_strength, (int, float)) else None,
            "evidence_count": int(impact.get("evidence_count") or 0),
            "stages": {},
        }
        details = impact.get("evidence_details") or []
        if not isinstance(details, list):
            details = [details]
        for detail in details:
            if not isinstance(detail, dict):
                continue
            stages = detail.get("stages") or detail.get("stage_counts")
            if isinstance(stages, dict):
                for stage_label, count in stages.items():
                    for token in _stage_tokens_for_label(stage_label):
                        record["stages"][token] = record["stages"].get(token, 0) + int(count or 1)
        pair_key = (asset_id, channel_id)
        existing = by_pair.get(pair_key)
        if not existing or (
            record["impact_strength"] or 0
        ) >= (existing.get("impact_strength") or 0):
            by_pair[pair_key] = dict(record)
        persona_id = impact.get("persona_id")
        if persona_id:
            by_persona[(asset_id, channel_id, persona_id)] = dict(record)

    return {"by_pair": by_pair, "by_persona": by_persona}


@lru_cache(maxsize=8)
def _get_arsenal_library(product_id: str) -> Dict[str, Any]:
    try:
        with SessionLocal() as db:
            raw = arsenal_service.serialize_arsenal_library(db, product_id=product_id)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Unable to load arsenal library for %s: %s", product_id, exc)
        raw = None

    if raw:
        assets = [_standardize_asset_record(asset) for asset in raw.get("assets", [])]
        channels = [_standardize_channel_record(channel) for channel in raw.get("channels", [])]
        impacts = _index_impacts(raw.get("impacts", []))
        return {
            "assets": assets,
            "channels": channels,
            "impacts": impacts,
        }

    legacy_assets = [
        _standardize_asset_record(asset)
        for asset in _load_catalog("assets.json", product_id)
    ]
    legacy_channels = [
        _standardize_channel_record(channel)
        for channel in _load_catalog("channels.json", product_id)
    ]
    return {
        "assets": legacy_assets,
        "channels": legacy_channels,
        "impacts": {"by_pair": {}, "by_persona": {}},
    }


def _load_assets(product_id: str) -> List[Dict[str, Any]]:
    library = _get_arsenal_library(product_id)
    _ARSENAL_IMPACT_INDEX[product_id] = library.get("impacts") or {"by_pair": {}, "by_persona": {}}
    return library.get("assets", [])


def _load_channels(product_id: str) -> List[Dict[str, Any]]:
    library = _get_arsenal_library(product_id)
    return library.get("channels", [])


# ---------------------------------------------------------------------------
# Label & token helpers
# ---------------------------------------------------------------------------

def _label_persona(node_id: str, data: Dict[str, Any]) -> str:
    parts = [
        data.get("title"),
        data.get("department"),
        data.get("seniority"),
    ]
    pretty = " | ".join([p for p in parts if p])
    return pretty or data.get("name") or node_id


def _label_job(node_id: str, data: Dict[str, Any]) -> str:
    return (
        data.get("title")
        or data.get("description")
        or data.get("name")
        or node_id
    )


def _label_pain(node_id: str, data: Dict[str, Any]) -> str:
    return (
        data.get("text")
        or data.get("name")
        or data.get("description")
        or node_id
    )


def _label_capability(node_id: str, data: Dict[str, Any]) -> str:
    return data.get("name") or data.get("title") or data.get("description") or node_id


def _norm_token(value: str) -> str:
    return (
        value.lower()
        .replace("|", " ")
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
    )


def _tokenize(*values: Optional[str]) -> List[str]:
    tokens: List[str] = []
    for value in values:
        if not value:
            continue
        for chunk in _norm_token(value).split():
            if chunk and chunk not in tokens:
                tokens.append(chunk)
    return tokens


@lru_cache(maxsize=2048)
def _embedding_for_text(text: str) -> Tuple[float, ...]:
    normalized = (text or "").strip().lower()
    if not normalized:
        return ()
    vec = get_embedding(normalized)
    if not vec:
        return ()
    return tuple(vec)


def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if not norm1 or not norm2:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


def _lexical_similarity(a: str, b: str) -> float:
    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))
    if tokens_a and tokens_b:
        overlap = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        if union:
            return overlap / union
    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def _semantic_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    emb_a = _embedding_for_text(a)
    emb_b = _embedding_for_text(b)
    embedding_score = _cosine_similarity(emb_a, emb_b) if emb_a and emb_b else 0.0
    lexical_score = _lexical_similarity(a, b)
    return max(embedding_score, lexical_score)


def _best_similarity(
    candidates: Iterable[str],
    targets: Iterable[str],
) -> Tuple[float, Optional[str], Optional[str]]:
    best_score = 0.0
    best_candidate = None
    best_target = None
    for candidate in candidates or []:
        if not candidate:
            continue
        for target in targets or []:
            if not target:
                continue
            score = _semantic_similarity(candidate, target)
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_target = target
    return best_score, best_candidate, best_target


# ---------------------------------------------------------------------------
# Belief thesis → persona path reduction
# ---------------------------------------------------------------------------

def _top_paths_from_thesis(
    thesis: Dict[str, Any],
    *,
    max_paths: int = 3,
) -> List[Dict[str, Any]]:
    paths = thesis.get("current_paths") or []
    candidates: List[Dict[str, Any]] = []

    for idx, entry in enumerate(paths):
        persona_ids = entry.get("personas") or entry.get("path") or []
        prob = _safe_float(entry.get("probability"), 0.0) or 0.0
        score = _safe_float(entry.get("score"), 0.0) or 0.0
        candidates.append(
            {
                "id": entry.get("path_id") or f"path:{idx}",
                "persona_ids": list(persona_ids),
                "probability": prob,
                "score": score,
                "raw": entry,
                "source": "data",
                "confidence": max(prob, score),
            }
        )

    if not candidates:
        fallback = thesis.get("walk_paths") or []
        for idx, entry in enumerate(fallback):
            persona_ids = entry.get("personas") or entry.get("path") or []
            prob = _safe_float(entry.get("probability"), 0.0) or 0.0
            score = _safe_float(entry.get("score"), 0.0) or 0.0
            candidates.append(
                {
                    "id": entry.get("path_id") or f"walk:{idx}",
                    "persona_ids": list(persona_ids),
                    "probability": prob,
                    "score": score,
                    "raw": entry,
                    "source": "data",
                    "confidence": max(prob, score),
                }
            )

    if not candidates:
        return []

    candidates.sort(
        key=lambda row: (row["probability"], row["score"], -len(row["persona_ids"])),
        reverse=True,
    )
    selected = candidates[:max_paths]

    prob_total = sum(max(row["probability"], 0.0) for row in selected)
    if prob_total <= 1e-6:
        score_total = sum(max(row["score"], 0.0) for row in selected)
        if score_total <= 1e-6:
            score_total = float(len(selected))
        for row in selected:
            base = max(row["score"], 0.0) or 1.0
            row["probability"] = base / score_total
    else:
        for row in selected:
            row["probability"] = max(row["probability"], 0.0) / prob_total

    return selected


def _metric_from_sources(
    primary: Optional[Dict[str, Any]],
    fallback: Optional[Dict[str, Any]],
    *keys: str,
) -> Optional[float]:
    for source in (primary, fallback):
        if not source or not isinstance(source, dict):
            continue
        for key in keys:
            if key in source and source[key] is not None:
                return _safe_float(source[key])
    return None


def _persona_summary(
    G: nx.DiGraph,
    persona_id: str,
    order: int,
    path_probability: float,
    belief_metrics: Optional[Dict[str, Any]] = None,
    expected_next_prob: Optional[float] = None,
    belief_overrides: Optional[Dict[str, Any]] = None,
    wolves_metrics: Optional[Dict[str, Any]] = None,
    account_wolf_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = get_node_by_id(G, persona_id) or {}
    summary: Dict[str, Any] = {
        "id": persona_id,
        "order": order,
        "label": _label_persona(persona_id, data),
        "title": data.get("title"),
        "department": data.get("department"),
        "seniority": data.get("seniority"),
        "perceptibility": _metric_from_sources(
            belief_metrics,
            data,
            "perceptibility",
            "perceptibility_score",
            "perc",
        ),
        "proximity": _metric_from_sources(
            belief_metrics,
            data,
            "proximity",
            "proximity_score",
            "prox",
        ),
        "involvement": _metric_from_sources(
            belief_metrics,
            data,
            "involvement",
            "involvement_score",
            "inv",
        ),
        "activation": _metric_from_sources(
            belief_metrics,
            data,
            "activation",
            "activation_score",
            "act",
        ),
        "path_probability": path_probability,
    }
    if expected_next_prob is not None:
        summary["expected_next_prob"] = round(
            _safe_float(expected_next_prob, 0.0) or 0.0, 6
        )
    summary["belief_level"] = _belief_level(summary)
    if belief_overrides:
        override_level = belief_overrides.get("belief_level")
        if override_level is not None:
            summary["belief_level"] = round(
                _safe_float(override_level, summary["belief_level"]),
                4,
            )
        summary["phase_probs"] = belief_overrides.get("phase_probs")
        summary["dominant_phase"] = belief_overrides.get("dominant_phase") or _dominant_phase_from_probs(
            belief_overrides.get("phase_probs")
        )
    else:
        summary["phase_probs"] = None
        summary["dominant_phase"] = None
    summary["persona_source"] = data.get("source") or data.get("data_source")
    summary["is_new_persona"] = _is_recent_persona_node(data)
    summary["wolves_score"] = _safe_float((wolves_metrics or {}).get("wolves_score"), None)
    summary["wolves_delta_bp"] = _safe_float((wolves_metrics or {}).get("delta_win_bp"), None)
    summary["wolves_centrality"] = _safe_float((wolves_metrics or {}).get("centrality"), None)
    summary["wolves_involvement_rate"] = _safe_float(
        (wolves_metrics or {}).get("involvement_rate"), None
    )
    summary["wolves_blocker_rate"] = _safe_float(
        (wolves_metrics or {}).get("blocker_rate"), None
    )
    summary["wolves_sample_size"] = (wolves_metrics or {}).get("sample_size")
    summary["wolves_source"] = (wolves_metrics or {}).get("source")
    if account_wolf_metrics:
        if account_wolf_metrics.get("base_wolf_score") is not None:
            summary["base_wolf_score"] = _safe_float(account_wolf_metrics.get("base_wolf_score"), None)
        if account_wolf_metrics.get("dynamic_wolf_score") is not None:
            summary["dynamic_wolf_score"] = _safe_float(account_wolf_metrics.get("dynamic_wolf_score"), None)
        if account_wolf_metrics.get("subsidy_relevance") is not None:
            summary["subsidy_relevance"] = _safe_float(account_wolf_metrics.get("subsidy_relevance"), None)
    else:
        summary["base_wolf_score"] = summary.get("wolves_score")
        summary["dynamic_wolf_score"] = summary.get("wolves_score")
        summary["subsidy_relevance"] = None
    return summary


def _belief_level(persona: Dict[str, Any]) -> float:
    perceptibility = _safe_float(persona.get("perceptibility"), 0.0) or 0.0
    involvement = _safe_float(persona.get("involvement"), 0.0) or 0.0
    activation = _safe_float(persona.get("activation"), 0.0) or 0.0
    return round(
        0.45 * perceptibility + 0.35 * involvement + 0.2 * activation,
        4,
    )


def _belief_band(score: Optional[float]) -> str:
    level = _safe_float(score, 0.0) or 0.0
    if level >= 0.65:
        return "high"
    if level >= 0.4:
        return "medium"
    return "low"


def _dominant_phase_from_probs(
    phase_probs: Optional[Dict[str, float]]
) -> Optional[str]:
    if not phase_probs:
        return None
    return max(phase_probs.items(), key=lambda kv: kv[1])[0]


def _persona_priority_score(persona: Dict[str, Any]) -> float:
    belief_level = _safe_float(persona.get("belief_level"), 0.0) or 0.0
    proximity = _safe_float(persona.get("proximity"), 0.0) or 0.0
    involvement = _safe_float(persona.get("involvement"), 0.0) or 0.0
    expected_next = _safe_float(persona.get("expected_next_prob"), 0.0) or 0.0
    fatigue = _safe_float(persona.get("fatigue"), 0.0) or 0.0
    score = (
        0.4 * belief_level
        + 0.25 * proximity
        + 0.2 * involvement
        + 0.15 * expected_next
    )
    score *= max(0.2, 1.0 - 0.6 * fatigue)
    multiplier = _safe_float(persona.get("segment_priority_multiplier"), 1.0) or 1.0
    score *= max(0.2, multiplier)
    return round(score, 4)


def _match_strength(persona: Dict[str, Any]) -> float:
    matches = persona.get("top_people") or persona.get("matched_people") or []
    best_match = 0.0
    for person in matches:
        prob = _safe_float(
            person.get("committee_probability")
            or person.get("person_involvement_score")
            or person.get("match_confidence"),
            0.0,
        ) or 0.0
        if prob > best_match:
            best_match = prob
    depth_boost = min(0.3, 0.05 * len(matches)) if matches else 0.0
    snapshot = persona.get("engagement_snapshot") or {}
    positive_rate = _safe_float(snapshot.get("positive_rate"), None)
    if best_match <= 0.0 and positive_rate is not None:
        best_match = 0.2 + 0.6 * positive_rate
    if best_match <= 0.0:
        best_match = _safe_float(persona.get("expected_in_deal_prob"), 0.0) or 0.08
    return max(0.05, min(1.0, best_match + depth_boost))


def _play_strength(plays: Sequence[Dict[str, Any]]) -> float:
    if not plays:
        return 0.05
    max_delta = 0.0
    confidence_sum = 0.0
    sample_size = min(5, len(plays))
    for idx, play in enumerate(plays):
        delta = _safe_float(play.get("expected_delta_bp"), 0.0) or 0.0
        if delta > max_delta:
            max_delta = delta
        if idx < sample_size:
            confidence_sum += _safe_float(play.get("confidence"), 0.0) or 0.0
    delta_score = min(1.0, max_delta / 0.15) if max_delta > 0 else 0.0
    confidence_score = (
        min(1.0, (confidence_sum / sample_size) if sample_size else 0.0)
        if sample_size
        else 0.0
    )
    coverage_score = min(1.0, len(plays) / 4.0)
    return max(
        0.05,
        min(
            1.0,
            0.6 * delta_score + 0.25 * coverage_score + 0.15 * confidence_score,
        ),
    )


def _summarize_entry_point_people(
    persona: Dict[str, Any],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    people = persona.get("top_people") or persona.get("matched_people") or []
    rows: List[Dict[str, Any]] = []
    for entry in people[:limit]:
        probability = _safe_float(
            entry.get("committee_probability")
            or entry.get("person_involvement_score")
            or entry.get("person_belief_level"),
            0.0,
        ) or 0.0
        rows.append(
            {
                "person_id": entry.get("person_id"),
                "display_name": entry.get("display_name")
                or entry.get("person_name")
                or entry.get("notes"),
                "role_band": entry.get("role_band")
                or _probability_band(probability),
                "committee_probability": round(min(max(probability, 0.0), 1.0), 4),
            }
        )
    return rows


def _summarize_entry_point_plays(
    plays: Sequence[Dict[str, Any]],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for play in plays[:limit]:
        if not play:
            continue
        belief_transition = play.get("belief_transition")
        pain_label = None
        if isinstance(belief_transition, dict):
            pain_entry = belief_transition.get("pain") or {}
            pain_label = pain_entry.get("label")
        rows.append(
            {
                "play_id": play.get("play_id") or play.get("asset_id") or play.get("channel_id"),
                "stage": play.get("stage_label"),
                "concern": pain_label or play.get("concern_label"),
                "asset": (play.get("asset") or {}).get("name")
                or play.get("asset_label")
                or play.get("asset_type"),
                "channel": (play.get("channel") or {}).get("name")
                or play.get("channel_label")
                or play.get("channel"),
                "mode": play.get("mode"),
                "expected_delta_bp": _safe_float(play.get("expected_delta_bp"), 0.0) or 0.0,
                "confidence": _safe_float(play.get("confidence"), 0.0) or 0.0,
            }
        )
    return rows


def _persona_entry_point(
    persona: Dict[str, Any],
    persona_plays: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    graph_perceptibility = max(0.0, _safe_float(persona.get("perceptibility"), 0.0) or 0.0)
    graph_proximity = max(0.0, _safe_float(persona.get("proximity"), 0.0) or 0.0)
    intervention_reach = max(
        0.05,
        min(
            1.0,
            0.55 * _match_strength(persona) + 0.45 * _play_strength(persona_plays),
        ),
    )
    combined_perceptibility = math.sqrt(graph_perceptibility * intervention_reach)
    combined_proximity = math.sqrt(graph_proximity * intervention_reach)
    entry_score = 0.65 * combined_perceptibility + 0.35 * combined_proximity

    persona["graph_perceptibility"] = round(graph_perceptibility, 4)
    persona["graph_proximity"] = round(graph_proximity, 4)
    persona["intervention_reach"] = round(intervention_reach, 4)
    persona["perceptibility"] = round(combined_perceptibility, 4)
    persona["proximity"] = round(combined_proximity, 4)
    persona["entry_score"] = round(entry_score, 4)
    persona["belief_level"] = _belief_level(persona)
    persona["belief_band"] = _belief_band(persona["belief_level"])
    persona["priority_score"] = _persona_priority_score(persona)

    return {
        "persona_id": persona.get("id"),
        "persona_label": persona.get("label"),
        "graph_perceptibility": persona["graph_perceptibility"],
        "graph_proximity": persona["graph_proximity"],
        "intervention_reach": persona["intervention_reach"],
        "combined_perceptibility": persona["perceptibility"],
        "combined_proximity": persona["proximity"],
        "entry_score": persona["entry_score"],
        "belief_level": persona["belief_level"],
        "top_people": _summarize_entry_point_people(persona),
        "top_plays": _summarize_entry_point_plays(persona_plays),
    }


def _apply_entry_point_scores(
    persona_lookup: Dict[str, Dict[str, Any]],
    plays_by_persona: Mapping[str, Sequence[Dict[str, Any]]],
    ordered_persona_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    entry_points: List[Dict[str, Any]] = []
    for persona_id in ordered_persona_ids:
        persona = persona_lookup.get(persona_id)
        if not persona:
            continue
        persona_plays = plays_by_persona.get(persona_id) or []
        entry = _persona_entry_point(persona, persona_plays)
        entry_points.append(entry)
    # include any personas that were not part of ordered list but exist in lookup
    for persona_id, persona in persona_lookup.items():
        if persona_id in ordered_persona_ids:
            continue
        persona_plays = plays_by_persona.get(persona_id) or []
        entry_points.append(_persona_entry_point(persona, persona_plays))
    entry_points.sort(key=lambda row: row.get("entry_score", 0.0), reverse=True)
    for idx, entry in enumerate(entry_points, start=1):
        entry["rank"] = idx
    return entry_points


def _probability_band(probability: float) -> str:
    if probability >= 0.65:
        return "anchor"
    if probability >= 0.35:
        return "supporting"
    return "edge"


def _persona_belief_probability(persona: Dict[str, Any], stage_key: str) -> float:
    phase_probs = persona.get("phase_probs") or {}
    if not isinstance(phase_probs, dict) or not phase_probs:
        return 0.35
    stage_label = BELIEF_STAGE_LABELS.get(stage_key, _title_case_value(stage_key) or stage_key)
    candidate_keys = {
        stage_label,
        stage_label.lower() if isinstance(stage_label, str) else stage_label,
        stage_key,
        stage_key.capitalize(),
    }
    for key in candidate_keys:
        if key in phase_probs:
            return max(0.05, min(1.0, _safe_float(phase_probs.get(key), 0.0) or 0.0))
    highest = max(_safe_float(value, 0.0) or 0.0 for value in phase_probs.values())
    return max(0.05, min(1.0, highest * 0.8))


def _journey_phase_label(idx: int, total: int) -> str:
    if total <= 1:
        return "Late Funnel"
    fraction = idx / max(total - 1, 1)
    if fraction <= 0.33:
        return "Early Funnel"
    if fraction <= 0.66:
        return "Mid Funnel"
    return "Late Funnel"


def _persona_path_signal(personas: Sequence[Dict[str, Any]]) -> float:
    if not personas:
        return 0.0
    signal = 0.0
    total = len(personas)
    denominator = max(total - 1, 1)
    for idx, persona in enumerate(personas):
        perc = _safe_float(persona.get("perceptibility"), 0.0) or 0.0
        prox = _safe_float(persona.get("proximity"), 0.0) or 0.0
        involvement = _safe_float(persona.get("involvement"), 0.0) or 0.0
        stage_fraction = idx / denominator if denominator else 0.0
        perc_bias = 1.2 - 0.6 * stage_fraction
        prox_bias = 0.6 + 0.6 * stage_fraction
        component = (
            (perc + 0.05) * perc_bias
            * (prox + 0.05) * prox_bias
            * (involvement + 0.05)
        )
        signal += component
    return round(signal, 6)


def _prioritize_personas_for_path(personas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(personas, key=lambda p: p.get("order", 0))
    total = len(ranked)
    for idx, persona in enumerate(ranked):
        persona["original_order"] = persona.get("order")
        persona["order"] = idx
        persona["priority_score"] = _persona_priority_score(persona)
        persona["priority_rank"] = idx
        persona["journey_phase"] = _journey_phase_label(idx, total)
    return ranked


def _collect_persona_metrics(
    G: nx.DiGraph,
    path: Dict[str, Any],
    metrics_lookup: Optional[Dict[str, Dict[str, float]]] = None,
    expected_next_probs: Optional[Dict[str, float]] = None,
    expected_next_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    belief_posteriors: Optional[Dict[str, Dict[str, Any]]] = None,
    wolves_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    account_wolf_scores: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    personas = []
    belief_states: List[Dict[str, Any]] = []
    raw = path.get("raw")
    if isinstance(raw, dict):
        for key in (
            "belief_states",
            "belief_states_norm",
            "belief_scores",
            "states",
        ):
            if isinstance(raw.get(key), list):
                belief_states = raw[key]
                break
    if not belief_states and isinstance(path.get("belief_states"), list):
        belief_states = path["belief_states"]

    meta_lookup = expected_next_meta or {}

    for order, persona_id in enumerate(path.get("persona_ids") or []):
        metrics: Dict[str, Any] = {}
        if order < len(belief_states) and isinstance(belief_states[order], dict):
            metrics.update(belief_states[order])
        if metrics_lookup:
            for field in ("perceptibility", "proximity", "involvement"):
                value = metrics_lookup.get(field, {}).get(persona_id)
                if value is not None:
                    metrics[field] = value
        pid_str = str(persona_id)
        expected_prob = (
            expected_next_probs.get(pid_str)
            if expected_next_probs is not None
            else None
        )
        personas.append(
            _persona_summary(
                G,
                persona_id,
                order,
                path.get("probability", 0.0),
                belief_metrics=metrics or None,
                expected_next_prob=expected_prob,
                belief_overrides=(belief_posteriors or {}).get(persona_id),
                wolves_metrics=(wolves_metrics or {}).get(str(persona_id)),
                account_wolf_metrics=(account_wolf_scores or {}).get(str(persona_id)),
            )
        )
        meta_entry = meta_lookup.get(pid_str) or {}
        personas[-1]["expected_next_prob_base"] = (
            _safe_float(meta_entry.get("prob_base"), None)
        )
        personas[-1]["subsidy_lift"] = _safe_float(meta_entry.get("subsidy_lift"), None)
    return personas


def _extract_keystone_personas(
    persona_lookup: Dict[str, Dict[str, Any]],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for persona in persona_lookup.values():
        persona_id = persona.get("id")
        score = _safe_float(persona.get("wolves_score"), None)
        if not persona_id or score is None:
            continue
        entries.append(
            {
                "persona_id": persona_id,
                "persona_label": persona.get("label") or persona_id,
                "wolves_score": score,
                "wolves_delta_bp": _safe_float(persona.get("wolves_delta_bp"), None),
                "wolves_involvement_rate": _safe_float(
                    persona.get("wolves_involvement_rate"), None
                ),
                "wolves_blocker_rate": _safe_float(
                    persona.get("wolves_blocker_rate"), None
                ),
                "wolves_sample_size": persona.get("wolves_sample_size"),
                "is_new_persona": bool(persona.get("is_new_persona")),
                "persona_source": persona.get("persona_source"),
                "priority_score": persona.get("priority_score"),
                "journey_phase": persona.get("journey_phase"),
                "matched_people_count": persona.get("matched_people_count")
                or len(persona.get("matched_people") or []),
                "top_people": persona.get("top_people") or [],
            }
        )
    entries.sort(
        key=lambda row: (
            row.get("wolves_score") or 0.0,
            row.get("wolves_delta_bp") or 0.0,
        ),
        reverse=True,
    )
    return entries[:limit]


def _observed_persona_sequence(thesis: Dict[str, Any]) -> List[str]:
    steps = (thesis.get("journey") or {}).get("steps") or []
    sequence: List[str] = []
    for step in steps:
        persona_id = (
            step.get("observed_next")
            or step.get("observed_persona_id")
            or step.get("persona_id")
            or step.get("persona")
        )
        if persona_id:
            sequence.append(str(persona_id))
    return sequence


def _flatten_transition_weights(weights: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not weights:
        return {}
    transition = weights.get("transition") or {}
    flattened: Dict[str, Dict[str, float]] = {}
    for from_pid, bucket_map in transition.items():
        combined: Dict[str, float] = defaultdict(float)
        for to_map in (bucket_map or {}).values():
            for to_pid, prob in (to_map or {}).items():
                try:
                    combined[str(to_pid)] += float(prob or 0.0)
                except Exception:
                    continue
        total = sum(combined.values())
        if total <= 0:
            continue
        flattened[str(from_pid)] = {pid: val / total for pid, val in combined.items() if val > 0}
    return flattened


def _expand_bgn_paths(
    path: List[str],
    probability: float,
    win_likelihood: float,
    transitions: Dict[str, Dict[str, float]],
    persona_stats: Dict[str, Dict[str, Any]],
    max_depth: int,
    results: List[Tuple[List[str], float, float]],
) -> None:
    last = path[-1]
    next_options = transitions.get(last)
    if not next_options or len(path) >= max_depth:
        results.append((path[:], probability, win_likelihood))
        return
    for next_persona, next_prob in next_options.items():
        if next_persona in path:
            continue
        combined_prob = probability * next_prob
        if combined_prob < 1e-4:
            results.append((path[:], probability, win_likelihood))
            continue
        stats = persona_stats.get(last, {}).get(next_persona) or {}
        success_rate = stats.get("success_rate")
        if success_rate is None:
            success_rate = 0.5
        success_rate = max(0.1, min(0.95, success_rate))
        _expand_bgn_paths(
            path + [next_persona],
            combined_prob,
            win_likelihood * success_rate,
            transitions,
            persona_stats,
            max_depth,
            results,
        )


def _generate_persona_paths_from_weights(
    weights: Optional[Dict[str, Any]],
    thesis: Dict[str, Any],
    *,
    persona_transition_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    max_depth: int = 4,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    transitions = _flatten_transition_weights(weights)
    if not transitions:
        return []
    observed = _observed_persona_sequence(thesis)
    start_candidates: List[Tuple[str, float]] = []
    if observed:
        start_candidates.append((observed[-1], 1.0))
        if len(observed) > 1:
            start_candidates.append((observed[-2], 0.6))
    else:
        committee = thesis.get("persona_committee_probs") or {}
        if isinstance(committee, dict):
            ranked = sorted(
                ((str(pid), _safe_float(prob, 0.0) or 0.0) for pid, prob in committee.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            start_candidates.extend(ranked[:3])
    if not start_candidates:
        return []
    persona_stats = persona_transition_stats or {}
    raw_results: List[Tuple[List[str], float, float]] = []
    for persona_id, prior in start_candidates:
        if not persona_id or persona_id not in transitions:
            continue
        prob = prior if prior and prior > 0 else 1.0
        _expand_bgn_paths(
            path=[persona_id],
            probability=prob,
            win_likelihood=1.0,
            transitions=transitions,
            persona_stats=persona_stats,
            max_depth=max_depth,
            results=raw_results,
        )
    if not raw_results:
        return []
    unique: Dict[str, Tuple[List[str], float, float]] = {}
    for seq, prob, win_prob in raw_results:
        if not seq:
            continue
        key = "->".join(seq)
        existing = unique.get(key)
        if existing is None or prob > existing[1]:
            unique[key] = (seq, prob, win_prob)
    ordered = sorted(unique.values(), key=lambda row: row[1], reverse=True)[:top_k]
    payload: List[Dict[str, Any]] = []
    for idx, (seq, prob, win_prob) in enumerate(ordered):
        if not seq:
            continue
        payload.append(
            {
                "id": f"shm_bgn_path_{idx}",
                "persona_ids": seq,
                "probability": round(prob, 6),
                "score": round(prob, 6),
                "win_likelihood": round(win_prob, 6),
                "raw": {
                    "source": "shm_bgn",
                    "probability": prob,
                    "win_likelihood": win_prob,
                },
                "source": "data",
                "confidence": round(min(1.0, max(prob, win_prob)), 6),
            }
        )
    return payload


def _canonical_persona_path_from_steps(
    G: nx.DiGraph,
    typical_steps: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if not typical_steps:
        return []
    summaries: List[Dict[str, Any]] = []
    for idx, step in enumerate(typical_steps):
        persona_id = step.get("persona_id") or step.get("persona")
        if not persona_id:
            continue
        summary = _persona_summary(
            G,
            str(persona_id),
            idx,
            _safe_float(step.get("confidence"), 0.0) or 0.0,
            belief_metrics=step,
        )
        summary["highlight"] = step.get("highlight")
        summary["impact"] = step.get("impact")
        summary["reason"] = step.get("reason")
        summary["source"] = "canonical"
        if step.get("fatigue") is not None:
            summary["fatigue"] = step.get("fatigue")
        summaries.append(summary)
    return _prioritize_personas_for_path(summaries)


# ---------------------------------------------------------------------------
# Persona ↔ belief transitions
# ---------------------------------------------------------------------------

def _job_nodes_for_persona(G: nx.DiGraph, persona_id: str) -> List[str]:
    job_ids = []
    for u, v, data in G.in_edges(persona_id, data=True):
        if data.get("type") == "performed_by":
            job_ids.append(u)
    return job_ids


def _pain_nodes_for_job(G: nx.DiGraph, job_id: str) -> List[str]:
    pain_ids = []
    for u, v, data in G.in_edges(job_id, data=True):
        if data.get("type") in {"felt_in", "expressed_as"}:
            pain_ids.append(u)
    return pain_ids


def _capability_nodes_for_pain(G: nx.DiGraph, pain_id: str) -> List[str]:
    capability_ids = []
    for u, v, data in G.in_edges(pain_id, data=True):
        if data.get("type") == "solves":
            capability_ids.append(u)
    return capability_ids


def _persona_transitions(
    G: nx.DiGraph,
    persona: Dict[str, Any],
    *,
    stage_index: int,
) -> List[Dict[str, Any]]:
    transitions: List[Dict[str, Any]] = []
    persona_id = persona["id"]
    persona_label = persona["label"]

    job_ids = _job_nodes_for_persona(G, persona_id)
    if not job_ids:
        transitions.append(
            {
                "persona": persona,
                "stage_index": stage_index,
                "job": None,
                "belief_transition": {
                    "persona": {
                        "type": "persona",
                        "id": persona_id,
                        "label": persona_label,
                    },
                    "problem": {
                        "type": "persona",
                        "id": persona_id,
                        "label": f"{persona_label} recognises an unmet need",
                    },
                    "execution": {
                        "type": "persona",
                        "id": persona_id,
                        "label": persona_label,
                    },
                    "pain": None,
                    "resolution": None,
                    "narrative": f"Activate {persona_label} with the right belief catalyst",
                    "from_to": [],
                    "pain_options": [],
                    "resolution_options": [],
                },
                "pains": [],
                "capabilities": [],
            }
        )
        return transitions

    for job_id in job_ids:
        job_node = get_node_by_id(G, job_id) or {}
        job_label = _label_job(job_id, job_node)
        pain_ids = _pain_nodes_for_job(G, job_id)
        pain_entries = []
        capability_entries = []

        for pain_id in pain_ids:
            pain_node = get_node_by_id(G, pain_id) or {}
            pain_label = _label_pain(pain_id, pain_node)
            capability_ids = _capability_nodes_for_pain(G, pain_id)
            caps = []
            for cap_id in capability_ids:
                cap_node = get_node_by_id(G, cap_id) or {}
                caps.append(
                    {
                        "id": cap_id,
                        "type": "capability",
                        "label": _label_capability(cap_id, cap_node),
                        "importance": cap_node.get("importance"),
                        "coreness": _safe_float(cap_node.get("coreness")),
                        "centrality": _safe_float(cap_node.get("centrality")),
                    }
                )
            pain_entries.append(
                {
                    "id": pain_id,
                    "type": "pain",
                    "label": pain_label,
                    "perceptibility": _safe_float(pain_node.get("perceptibility")),
                    "proximity": _safe_float(pain_node.get("proximity")),
                    "involvement": _safe_float(pain_node.get("involvement")),
                    "tokens": _tokenize(pain_label),
                }
            )
            capability_entries.extend(caps)

        belief_transition = {
            "persona": {
                "type": "persona",
                "id": persona_id,
                "label": persona_label,
            },
            "problem": {
                "type": "job",
                "id": job_id,
                "label": job_label,
            },
            "execution": {
                "type": "persona",
                "id": persona_id,
                "label": persona_label,
            },
            "pain": None,
            "resolution": None,
            "narrative": "",
            "from_to": [],
        }

        primary_pain = pain_entries[0] if pain_entries else None
        primary_capability = capability_entries[0] if capability_entries else None

        if primary_pain:
            belief_transition["pain"] = {
                "type": "pain",
                "id": primary_pain["id"],
                "label": primary_pain["label"],
            }
            belief_transition["from_to"].append(
                {
                    "from_type": "job",
                    "from_id": job_id,
                    "from_label": job_label,
                    "to_type": "pain",
                    "to_id": primary_pain["id"],
                    "to_label": primary_pain["label"],
                }
            )

        if primary_capability:
            belief_transition["resolution"] = {
                "type": "capability",
                "id": primary_capability["id"],
                "label": primary_capability["label"],
            }
            belief_transition["from_to"].append(
                {
                    "from_type": "pain",
                    "from_id": primary_pain["id"] if primary_pain else None,
                    "from_label": primary_pain["label"] if primary_pain else None,
                    "to_type": "capability",
                    "to_id": primary_capability["id"],
                    "to_label": primary_capability["label"],
                }
            )

        pain_label = (
            primary_pain["label"]
            if primary_pain
            else ", ".join(p["label"] for p in pain_entries[:2]) if pain_entries else None
        )
        resolution_label = (
            primary_capability["label"]
            if primary_capability
            else ", ".join(c["label"] for c in capability_entries[:2]) if capability_entries else None
        )

        narrative = f"Get {persona_label}"
        if pain_label:
            narrative += f" to realise {pain_label}"
        else:
            narrative += " to internalize the required belief"
        if job_label:
            narrative += f" when they {job_label}"
        if resolution_label:
            narrative += f" and show how {resolution_label} addresses it"
        belief_transition["narrative"] = narrative
        belief_transition["pain_options"] = pain_entries
        belief_transition["resolution_options"] = capability_entries
        stage_key = _stage_for_transition(
            belief_transition,
            stage_index=stage_index,
        )
        belief_transition["stage"] = stage_key
        belief_transition["stage_label"] = BELIEF_STAGE_LABELS.get(
            stage_key,
            _title_case_value(stage_key) or stage_key,
        )

        transitions.append(
            {
                "persona": persona,
                "stage_index": stage_index,
                "job": {
                    "id": job_id,
                    "type": "job",
                    "label": job_label,
                    "tokens": _tokenize(job_label),
                    "importance": _safe_float(job_node.get("importance")),
                },
                "pains": pain_entries,
                "capabilities": capability_entries,
                "belief_transition": belief_transition,
            }
        )

    return transitions


# ---------------------------------------------------------------------------
# ZMOT helpers
# ---------------------------------------------------------------------------

def _format_zmot_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    if not entry:
        return {}
    observable = entry.get("observable_moments") or []
    keywords = entry.get("keywords") or []
    return {
        "zmot_event_id": entry.get("zmot_event_id"),
        "zmot_label": entry.get("zmot_label"),
        "pain_trigger_id": entry.get("pain_trigger_id"),
        "pain_trigger_label": entry.get("pain_trigger_label"),
        "pain_id": entry.get("pain_id"),
        "pain_label": entry.get("pain_label"),
        "boost": _safe_float(entry.get("boost"), 0.0),
        "trigger_boost": _safe_float(entry.get("trigger_boost"), 0.0),
        "observable_moments": observable[:5],
        "keywords": keywords[:8],
        "already_observed": bool(entry.get("already_observed")),
    }


# ---------------------------------------------------------------------------
# Arsenal recommendation
# ---------------------------------------------------------------------------

def _asset_tokens(asset: Dict[str, Any]) -> set[str]:
    parts: List[str] = []
    for field in (
        "name",
        "format",
        "persona_fit",
        "concern_tags",
        "stage_fit",
        "target_concerns",
        "target_personas",
        "target_account_segments",
    ):
        value = asset.get(field)
        if isinstance(value, str):
            parts.extend(_tokenize(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    parts.extend(_tokenize(item))
    if isinstance(asset.get("description"), str):
        parts.extend(_tokenize(asset.get("description")))
    return set(parts)


def _belief_tokens_from_transition(transition: Dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    belief_transition = transition.get("belief_transition") or {}
    for field in ("narrative",):
        tokens.update(_tokenize(belief_transition.get(field)))
    job = transition.get("job") or {}
    tokens.update(_tokenize(job.get("label")))
    pains = transition.get("pains") or []
    for pain in pains:
        tokens.update(pain.get("tokens") or [])
        tokens.update(_tokenize(pain.get("label")))
    primary_pain = belief_transition.get("pain") or {}
    tokens.update(_tokenize(primary_pain.get("label")))
    primary_resolution = belief_transition.get("resolution") or {}
    tokens.update(_tokenize(primary_resolution.get("label")))
    capabilities = transition.get("capabilities") or []
    for capability in capabilities:
        tokens.update(_tokenize(capability.get("label")))
    return {token for token in tokens if token}


def _score_asset(
    asset: Dict[str, Any],
    persona: Dict[str, Any],
    transition: Dict[str, Any],
    *,
    account_segments: Optional[Dict[str, str]] = None,
) -> Tuple[
    float,
    Dict[str, Any],
    bool,
    bool,
    bool,
    float,
    bool,
    float,
]:
    tokens_persona = _tokenize(persona.get("label"), persona.get("title"))
    tokens_job = _tokenize(
        transition.get("job", {}).get("label") if transition.get("job") else ""
    )
    tokens_pain = []
    for pain in transition.get("pains") or []:
        tokens_pain.extend(pain.get("tokens") or [])

    score = 0.0
    reasons: Dict[str, Any] = {}
    asset_tokens = _asset_tokens(asset)

    persona_fit = list(asset.get("persona_fit") or [])
    persona_fit.extend(asset.get("target_personas") or [])
    persona_alignment = False
    if persona_fit and persona.get("label"):
        best_score, best_tag, _ = _best_similarity(
            persona_fit,
            [persona.get("label"), persona.get("title")],
        )
        if best_score >= PERSONA_SIMILARITY_THRESHOLD:
            score += 1.5 + best_score
            reasons["persona_fit"] = [best_tag]
            persona_alignment = True
    if not persona_alignment and persona_fit:
        persona_matches = [
            tag for tag in persona_fit if _norm_token(tag) in tokens_persona
        ]
        if persona_matches:
            score += 1.25
            reasons.setdefault("persona_fit_alt", persona_matches)
            persona_alignment = True

    concern_tags = list(asset.get("concern_tags") or [])
    concern_tags.extend(asset.get("target_concerns") or [])
    concern_alignment = False
    concern_strength = 0.0
    pain_labels = [
        pain.get("label") for pain in transition.get("pains") or [] if pain.get("label")
    ]
    primary_bt = transition.get("belief_transition") or {}
    primary_pain = primary_bt.get("pain") or {}
    if isinstance(primary_pain, dict) and primary_pain.get("label"):
        pain_labels.append(primary_pain["label"])
    if concern_tags and pain_labels:
        best_concern_score, best_tag, matched_pain = _best_similarity(
            concern_tags,
            pain_labels,
        )
        if best_concern_score >= CONCERN_SIMILARITY_THRESHOLD:
            score += 1.2 + 0.6 * best_concern_score
            reasons["concern_fit"] = [best_tag]
            reasons["concern_match"] = matched_pain
            concern_alignment = True
            concern_strength = best_concern_score
    if not concern_alignment and concern_tags and tokens_pain:
        concern_matches = [
            tag for tag in concern_tags if _norm_token(tag) in tokens_pain
        ]
        if concern_matches:
            score += 0.9
            reasons.setdefault("concern_fit_alt", concern_matches)
            concern_alignment = True

    stage_fit = asset.get("stage_fit") or []
    stage_alignment = False
    if transition.get("stage_index", 0) <= 1:
        desired_stage = {"problem_realization", "problem_awareness"}
    elif transition.get("stage_index", 0) == 2:
        desired_stage = {"solution_exploration", "promised_land"}
    else:
        desired_stage = {"evaluation", "decision"}

    stage_matches = [
        stage for stage in stage_fit if stage in desired_stage
    ]
    if stage_matches:
        score += 1.25
        reasons["stage_fit"] = stage_matches
        stage_alignment = True

    if tokens_job:
        job_matches = [
            tag for tag in asset_tokens if tag in tokens_job
        ]
        if job_matches:
            score += 0.75
            reasons["job_alignment"] = job_matches[:3]

    target_account_segments = asset.get("target_account_segments")
    segment_score = 0.0
    segment_matches: List[str] = []
    segment_applicable = True
    if account_segments:
        segment_score, segment_matches = _segment_fit_score(
            target_account_segments,
            account_segments,
        )
        if segment_matches:
            reasons["segment_fit"] = segment_matches
        if segment_score:
            reasons["segment_fit_score"] = round(segment_score, 3)
        if target_account_segments and not segment_matches:
            segment_applicable = False
        if segment_score:
            score += segment_score

    belief_tokens = _belief_tokens_from_transition(transition)
    belief_overlap = asset_tokens & belief_tokens
    if belief_overlap:
        score += 0.65
        reasons["belief_alignment"] = list(sorted(belief_overlap))[:4]

    evergreen_bonus = 0.25 if asset.get("evergreen") else 0.0
    score += evergreen_bonus

    return (
        score,
        reasons,
        stage_alignment,
        persona_alignment,
        concern_alignment,
        segment_score,
        segment_applicable,
        concern_strength,
    )


def _preferred_channel_types(asset: Dict[str, Any]) -> List[str]:
    format_name = (asset.get("format") or "").lower()
    mapping = {
        "webinar": ["webinar_platform", "email", "paid_social"],
        "ebook": ["email", "organic_social"],
        "report": ["email", "paid_social", "organic_social"],
        "email": ["email"],
        "case study": ["sales_outreach", "email"],
        "deck": ["sales_outreach"],
        "playbook": ["email", "organic_social"],
        "demo": ["sales_outreach"],
    }
    for key, target in mapping.items():
        if key in format_name:
            return target
    return ["email", "sales_outreach", "paid_social"]


def _score_channel(
    channel: Dict[str, Any],
    asset: Dict[str, Any],
    transition: Dict[str, Any],
    *,
    broad_bias: bool,
    account_segments: Optional[Dict[str, str]] = None,
    transition_stage_tokens: Optional[Sequence[str]] = None,
) -> Tuple[float, Dict[str, Any], bool]:
    score = 0.0
    reasons: Dict[str, Any] = {}
    preferred = _preferred_channel_types(asset)
    if channel.get("type") in preferred:
        score += 1.5
        reasons["channel_preferred"] = channel.get("type")
    reach_score = channel.get("reach_score", 0.5)
    breadth = channel.get("breadth_multiplier", 0.5)
    score += reach_score
    score += 0.5 * breadth
    if broad_bias and breadth >= 0.8:
        score += 0.4
        reasons["channel_breadth_bonus"] = round(breadth, 3)

    target_segments = channel.get("target_account_segments")
    segment_applicable = True
    if account_segments:
        segment_score, segment_matches = _segment_fit_score(
            target_segments,
            account_segments,
        )
        if segment_score:
            score += segment_score
            reasons["channel_segment_fit_score"] = round(segment_score, 3)
        if segment_matches:
            reasons["channel_segment_fit"] = segment_matches
        if target_segments and not segment_matches:
            segment_applicable = False

    target_stage_tokens = [
        str(token).lower()
        for token in (channel.get("target_belief_stages") or [])
        if token
    ]
    if transition_stage_tokens and target_stage_tokens:
        stage_matches = [
            token for token in target_stage_tokens if token in transition_stage_tokens
        ]
        if stage_matches:
            score += 0.35
            reasons["channel_stage_fit"] = stage_matches

    concern_targets = channel.get("target_concerns") or []
    if concern_targets:
        pain_labels = [
            pain.get("label")
            for pain in transition.get("pains") or []
            if pain.get("label")
        ]
        if pain_labels:
            best_score, best_tag, matched_pain = _best_similarity(
                concern_targets,
                pain_labels,
            )
            if best_score >= 0.4:
                score += 0.4 * best_score
                reasons["channel_concern_fit"] = [best_tag]
                reasons["channel_concern_match"] = matched_pain

    return score, reasons, segment_applicable


def _select_learned_effectiveness(
    asset: Dict[str, Any],
    channel: Dict[str, Any],
    persona_id: Optional[str],
    transition_stage_tokens: Optional[Sequence[str]],
    account_segments: Optional[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    slices = asset.get("learned_effectiveness") or []
    if not slices:
        return None
    stage_code = _stage_code_from_tokens(transition_stage_tokens)
    segment_candidates = _segment_candidates_from_map(account_segments)
    channel_type = (channel.get("type") or channel.get("channel_type") or "").lower()
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for entry in slices:
        score = entry.get("coverage_score", 0.0) or 0.0
        entry_persona = entry.get("persona_id")
        if entry_persona and persona_id:
            if entry_persona != persona_id:
                continue
            score += 2.5
        elif entry_persona and not persona_id:
            score += 0.5
        elif not entry_persona:
            score += 0.25
        entry_stage = entry.get("stage_code")
        if stage_code and entry_stage:
            if entry_stage == stage_code:
                score += 1.2
            else:
                score -= 0.2
        entry_segment = entry.get("segment_key")
        if entry_segment and segment_candidates:
            if entry_segment in segment_candidates:
                score += 0.8
            else:
                score -= 0.2
        if channel_type and entry.get("channel_type"):
            if entry["channel_type"] == channel_type:
                score += 0.6
            else:
                score -= 0.15
        if score > best_score:
            best = entry
            best_score = score
    return best


def _asset_signal_for_calibration(
    asset: Dict[str, Any],
    learned_effect: Optional[Dict[str, Any]],
) -> float:
    if learned_effect and learned_effect.get("lift_bps") is not None:
        base = _safe_float(learned_effect.get("lift_bps"), 0.0)
    else:
        base = _safe_float(asset.get("base_lift_bp"), 35.0)
    if base is None:
        base = 35.0
    return float(base) / 1000.0


def _expected_delta_bp(
    asset: Dict[str, Any],
    channel: Dict[str, Any],
    persona: Dict[str, Any],
    transition: Dict[str, Any],
    *,
    path_probability: float,
    asset_score: float,
    channel_score: float,
    stage_alignment: bool,
    persona_alignment: bool,
    concern_alignment: bool,
    segment_score: float,
    belief_probability: float,
    concern_strength: float,
    learned_effect: Optional[Dict[str, Any]] = None,
) -> float:
    learned_lift = None
    learned_confidence = None
    if learned_effect:
        learned_lift = learned_effect.get("lift_bps")
        learned_confidence = learned_effect.get("confidence")
    base_lift_bp = asset.get("expected_lift_bp")
    if base_lift_bp is None:
        base_lift_bp = asset.get("base_lift_bp")
    if base_lift_bp is None:
        base_lift_bp = 35
    base_lift_bp = float(base_lift_bp if learned_lift is None else learned_lift)
    stage_factor = math.exp(-0.35 * transition.get("stage_index", 0))
    path_factor = max(path_probability, 0.05)
    channel_factor = (
        channel.get("reach_score", 0.5) + channel.get("breadth_multiplier", 0.5)
    ) / 2.0
    involvement = persona.get("involvement") or 0.5
    fit_factor = 0.4
    fit_factor += 0.2 * max(0.0, min(asset_score, 3.0)) / 3.0
    fit_factor += 0.15 * max(0.0, min(channel_score, 3.0)) / 3.0

    belief_factor = max(0.35, 0.55 + 0.9 * max(0.0, min(belief_probability, 1.0)))
    concern_factor = 0.65
    if concern_alignment:
        concern_factor = 0.95 + 0.3 * concern_strength
    elif concern_strength > 0.1:
        concern_factor = 0.75 + 0.25 * concern_strength
    segment_factor = 0.65
    if segment_score > 0:
        segment_factor = min(1.35, 0.85 + segment_score)
    elif segment_score < 0:
        segment_factor = max(0.3, 0.55 + segment_score)

    if stage_alignment:
        fit_factor += 0.1
    else:
        fit_factor *= 0.55
    if persona_alignment or concern_alignment:
        fit_factor *= 1.1
    else:
        fit_factor *= 0.6

    learned_factor = 1.0
    if learned_confidence is not None:
        learned_factor = 0.7 + 0.3 * max(0.0, min(1.0, float(learned_confidence)))

    raw = (
        base_lift_bp
        * belief_factor
        * concern_factor
        * segment_factor
        * channel_factor
        * stage_factor
        * path_factor
        * (0.75 + 0.5 * involvement)
        * fit_factor
        * learned_factor
    )
    cap = base_lift_bp * (0.85 + 0.55 * channel_factor)
    return max(5.0, min(raw, cap))


def _belief_conversion_likelihood(
    asset: Dict[str, Any],
    channel: Dict[str, Any],
    *,
    asset_score: float,
    channel_score: float,
    stage_alignment: bool,
    persona_alignment: bool,
    concern_alignment: bool,
    impact_record: Optional[Dict[str, Any]],
    transition_stage_tokens: Sequence[str],
    belief_probability: float,
) -> float:
    usage = asset.get("usage") or {}
    engagements = usage.get("total_engagements") or 0
    usage_strength = 0.0
    if engagements:
        usage_strength = min(0.6, math.log1p(float(engagements)) / math.log1p(60.0))

    asset_effect = 0.35
    asset_effect += 0.18 * max(0.0, min(asset_score, 3.0)) / 3.0
    asset_effect += 0.12 * usage_strength
    if stage_alignment:
        asset_effect += 0.12
    if persona_alignment:
        asset_effect += 0.1
    if concern_alignment:
        asset_effect += 0.08
    asset_effect += 0.12 * max(0.0, min(belief_probability, 1.0))

    impact_strength = None
    impact_stage_match = 0.0
    if impact_record:
        impact_strength = impact_record.get("impact_strength")
        stage_counts = impact_record.get("stages") or {}
        if stage_counts:
            total_stage = sum(stage_counts.values())
            if total_stage:
                overlap = sum(stage_counts.get(token, 0) for token in transition_stage_tokens)
                impact_stage_match = overlap / total_stage
        if impact_stage_match:
            asset_effect += 0.1 * impact_stage_match

    asset_effect = max(0.18, min(0.95, asset_effect))

    channel_usage = (channel.get("usage") or {}).get("total_engagements") or 0
    channel_usage_strength = 0.0
    if channel_usage:
        channel_usage_strength = min(0.45, math.log1p(float(channel_usage)) / math.log1p(80.0))

    reach_score = channel.get("reach_score", 0.5) or 0.5
    breadth = channel.get("breadth_multiplier", 0.5) or 0.5

    channel_effect = 0.32
    channel_effect += 0.22 * reach_score
    channel_effect += 0.1 * breadth
    channel_effect += 0.12 * max(0.0, min(channel_score, 3.0)) / 3.0
    channel_effect += 0.08 * channel_usage_strength

    target_stage_tokens = channel.get("stage_fit") or channel.get("target_belief_stages") or []
    if target_stage_tokens:
        normalized_targets = [str(token).lower() for token in target_stage_tokens]
        if any(token in normalized_targets for token in transition_stage_tokens):
            channel_effect += 0.06

    channel_effect = max(0.15, min(0.9, channel_effect))

    belief_conversion = asset_effect * channel_effect
    if impact_strength is not None:
        strength_clamped = max(0.05, min(0.95, float(impact_strength)))
        belief_conversion = 0.5 * belief_conversion + 0.5 * strength_clamped

    return max(0.05, min(0.95, belief_conversion))


def _confidence_score(
    asset_score: float,
    channel_score: float,
    persona: Dict[str, Any],
    *,
    path_probability: float,
    learned_effect: Optional[Dict[str, Any]] = None,
) -> float:
    base = 0.45 + 0.15 * path_probability
    base += 0.1 * (persona.get("perceptibility") or 0.3)
    base += 0.12 * (persona.get("proximity") or 0.3)
    base += 0.08 * asset_score
    base += 0.08 * channel_score
    if learned_effect and learned_effect.get("confidence") is not None:
        learned_conf = max(0.0, min(1.0, float(learned_effect.get("confidence"))))
        base = max(base, 0.4 + 0.35 * learned_conf)
    return max(0.2, min(0.95, base))


def _broadness_score(
    asset: Dict[str, Any],
    channel: Dict[str, Any],
    transition: Dict[str, Any],
) -> float:
    score = 0.5
    channel_type = (channel.get("type") or "").lower()
    broad_channel_types = {
        "paid_social",
        "display",
        "organic_social",
        "webinar_platform",
        "events",
    }
    focus_channel_types = {
        "sales_outreach",
        "email",
        "demo_platform",
        "call",
        "direct_mail",
    }
    if channel_type in broad_channel_types:
        score += 0.25
    if channel_type in focus_channel_types:
        score -= 0.25
    breadth = channel.get("breadth_multiplier")
    if breadth is not None:
        score += 0.2 * (breadth - 0.5)

    persona_fit = asset.get("persona_fit") or []
    if persona_fit:
        if len(persona_fit) >= 3:
            score += 0.15
        elif len(persona_fit) <= 1:
            score -= 0.15

    format_name = (asset.get("format") or "").lower()
    if any(term in format_name for term in ["blog", "social", "webinar", "report", "ebook", "newsletter", "whitepaper"]):
        score += 0.1
    if any(term in format_name for term in ["demo", "call", "deck", "roi", "case study", "playbook"]):
        score -= 0.1

    stage_fit = set(asset.get("stage_fit") or [])
    if stage_fit & {"problem_realization", "problem_awareness"}:
        score += 0.08
    if stage_fit & {"evaluation", "decision"}:
        score -= 0.06

    transition_focus = transition.get("belief_transition") or {}
    if transition_focus.get("pain"):
        score -= 0.05
    else:
        score += 0.05

    pain_count = len(transition.get("pains") or [])
    if pain_count >= 2:
        score += 0.05
    elif pain_count == 1:
        score -= 0.03

    return max(0.0, min(1.0, score))


def _recommend_assets_for_transition(
    transition: Dict[str, Any],
    *,
    persona: Dict[str, Any],
    assets: Sequence[Dict[str, Any]],
    channels: Sequence[Dict[str, Any]],
    impacts: Optional[Dict[str, Dict[Tuple[str, ...], Dict[str, Any]]]],
    path_probability: float,
    exploration_weight: float,
    max_combos: int = 3,
    account_segments: Optional[Dict[str, str]] = None,
    account_segment_labels: Optional[Sequence[str]] = None,
    account_id: Optional[str] = None,
    win_calibrator: Optional[WinProbabilityCalibrator] = None,
) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    broad_bias = exploration_weight >= 0.15
    impact_by_pair = (impacts or {}).get("by_pair", {})
    impact_by_persona = (impacts or {}).get("by_persona", {})
    persona_expected_prob = _safe_float(persona.get("expected_next_prob"), None)
    persona_expected_prob_base = _safe_float(persona.get("expected_next_prob_base"), None)
    persona_subsidy_lift = _safe_float(persona.get("subsidy_lift"), None)
    subsidy_lift_prob = 0.0
    if (
        persona_expected_prob is not None
        and persona_expected_prob_base is not None
    ):
        subsidy_lift_prob = max(0.0, persona_expected_prob - persona_expected_prob_base)
    elif persona_subsidy_lift is not None:
        subsidy_lift_prob = max(0.0, float(persona_subsidy_lift))
    time_to_subsidy_peak_days = 0.0

    stage_index = transition.get("stage_index", 0)
    belief_transition = transition.get("belief_transition") or {}
    stage_key = _stage_for_transition(belief_transition, stage_index=stage_index)
    stage_label = BELIEF_STAGE_LABELS.get(
        stage_key,
        _title_case_value(stage_key) or stage_key,
    )
    target_concern_label = _node_label(belief_transition.get("pain"))
    if not target_concern_label:
        pains = transition.get("pains") or []
        if pains:
            target_concern_label = pains[0].get("label")
    belief_probability = _persona_belief_probability(persona, stage_key)
    if stage_index <= 1:
        transition_stage_tokens = ["problem_realization", "problem_awareness"]
    elif stage_index == 2:
        transition_stage_tokens = ["solution_exploration", "promised_land"]
    else:
        transition_stage_tokens = ["evaluation", "decision"]

    def _lookup_impact(
        asset_id: Optional[str],
        channel_id: Optional[str],
        persona_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not asset_id or not channel_id:
            return None
        if persona_id:
            record = impact_by_persona.get((asset_id, channel_id, persona_id))
            if record:
                return dict(record)
        record = impact_by_pair.get((asset_id, channel_id))
        return dict(record) if record else None

    def _channel_persona_match(targets: Iterable[str], persona_label: Optional[str]) -> bool:
        if not targets or not persona_label:
            return False
        score, _, _ = _best_similarity(targets, [persona_label])
        if score >= PERSONA_SIMILARITY_THRESHOLD * 0.9:
            return True
        persona_norm = persona_label.lower()
        for target in targets:
            if not target:
                continue
            if target.lower() in persona_norm:
                return True
        return False

    def _asset_payload(asset: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": asset.get("id"),
            "name": asset.get("name"),
            "format": asset.get("format"),
            "category": asset.get("category"),
            "category_label": asset.get("category_label"),
            "evergreen": bool(asset.get("evergreen")),
            "stage_fit": asset.get("stage_fit"),
            "concern_tags": asset.get("concern_tags"),
            "persona_fit": asset.get("persona_fit"),
            "time_to_effect_days": asset.get("time_to_effect_days"),
            "base_lift_bp": asset.get("base_lift_bp"),
            "cost_tier": asset.get("cost_tier"),
            "metadata_complete": bool(asset.get("metadata_complete")),
            "target_personas": asset.get("target_personas"),
            "target_belief_stages": asset.get("target_belief_stages"),
            "target_concerns": asset.get("target_concerns"),
            "usage": asset.get("usage"),
        }

    def _channel_payload(channel: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "id": channel.get("id"),
            "name": channel.get("name"),
            "type": channel.get("type"),
            "channel_type": channel.get("channel_type") or channel.get("type"),
            "channel_type_label": channel.get("channel_type_label")
            or _title_case_value(channel.get("channel_type") or channel.get("type")),
            "reach_score": channel.get("reach_score"),
            "breadth_multiplier": channel.get("breadth_multiplier"),
            "time_to_effect_days": channel.get("time_to_effect_days"),
            "cadence_hint": channel.get("cadence_hint"),
            "delivery_mode": channel.get("delivery_mode"),
            "target_personas": channel.get("target_personas"),
            "target_belief_stages": channel.get("target_belief_stages"),
            "target_concerns": channel.get("target_concerns"),
            "usage": channel.get("usage"),
            "metadata_complete": bool(channel.get("metadata_complete")),
        }
        if "reach_score_estimate" in channel and payload["reach_score"] is None:
            payload["reach_score"] = channel.get("reach_score_estimate")
        return payload

    def _combo_is_duplicate(existing: List[Dict[str, Any]], candidate: Dict[str, Any]) -> bool:
        asset_id = candidate.get("asset", {}).get("id")
        channel_id = candidate.get("channel", {}).get("id")
        asset_type = candidate.get("asset_type")
        channel_type = candidate.get("channel_type")
        for row in existing:
            if asset_id and channel_id:
                if (
                    row.get("asset", {}).get("id") == asset_id
                    and row.get("channel", {}).get("id") == channel_id
                ):
                    return True
            if asset_type and channel_type:
                if (
                    row.get("asset_type") == asset_type
                    and row.get("channel_type") == channel_type
                ):
                    return True
        return False

    for asset in assets:
        (
            asset_score,
            reasons,
            stage_alignment,
            persona_alignment,
            concern_alignment,
            asset_segment_score,
            asset_segment_applicable,
            concern_strength,
        ) = _score_asset(
            asset,
            persona,
            transition,
            account_segments=account_segments,
        )
        if not asset_segment_applicable:
            continue
        if asset_score <= 0.1 and not (
            stage_alignment or persona_alignment or concern_alignment
        ):
            continue

        # choose top channels
        channel_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any], float]] = []
        for channel in channels:
            channel_score, channel_reasons, channel_applicable = _score_channel(
                channel,
                asset,
                transition,
                broad_bias=broad_bias,
                account_segments=account_segments,
                transition_stage_tokens=transition_stage_tokens,
            )
            if not channel_applicable:
                continue
            channel_segment_score = _safe_float(
                channel_reasons.get("channel_segment_fit_score"),
                0.0,
            ) or 0.0
            channel_candidates.append(
                (channel_score, channel, channel_reasons, channel_segment_score)
            )
        channel_candidates.sort(key=lambda item: item[0], reverse=True)
        channel_candidates = channel_candidates[:2]

        for channel_score, channel, channel_reasons, channel_segment_score in channel_candidates:
            impact_record = _lookup_impact(asset.get("id"), channel.get("id"), persona.get("id"))
            learned_effect = _select_learned_effectiveness(
                asset,
                channel,
                persona.get("id"),
                transition_stage_tokens,
                account_segments,
            )
            expected_delta = _expected_delta_bp(
                asset,
                channel,
                persona,
                transition,
                path_probability=path_probability,
                asset_score=asset_score,
                channel_score=channel_score,
                stage_alignment=stage_alignment,
                persona_alignment=persona_alignment,
                concern_alignment=concern_alignment,
                segment_score=max(asset_segment_score, channel_segment_score),
                belief_probability=belief_probability,
                concern_strength=concern_strength,
                learned_effect=learned_effect,
            )
            asset_payload = _asset_payload(asset)
            channel_payload = _channel_payload(channel)

            calibration_meta = None
            expected_source = None
            if win_calibrator and account_id:
                calibration = win_calibrator.estimate_delta_bp(
                    account_id,
                    stage_index=stage_index,
                    stage_key=stage_key,
                    asset_signal=_asset_signal_for_calibration(asset_payload, learned_effect),
                    base_delta_bp=expected_delta,
                )
                if calibration:
                    expected_delta = calibration["delta_bp"]
                    calibration_meta = {
                        "win_baseline": calibration["baseline_probability"],
                        "win_simulated": calibration["simulated_probability"],
                    }
                    expected_source = "win_regression"
            if expected_source is None:
                expected_source = (
                    "data"
                    if (learned_effect or (impact_record and impact_record.get("evidence_count")))
                    else "graph"
                )
            channel_persona_alignment = _channel_persona_match(
                channel_payload.get("target_personas") or [], persona.get("label")
            )
            channel_payload["persona_aligned"] = channel_persona_alignment

            confidence = _confidence_score(
                asset_score,
                channel_score,
                persona,
                path_probability=path_probability,
                learned_effect=learned_effect,
            )
            if impact_record and impact_record.get("evidence_count"):
                confidence = min(
                    0.95,
                    confidence + 0.02 * min(int(impact_record.get("evidence_count", 0)), 5),
                )
            duration_days = max(
                asset.get("time_to_effect_days") or 14,
                channel.get("time_to_effect_days") or 7,
            )
            broadness = _broadness_score(asset, channel, transition)
            mode = "broad" if broadness >= 0.6 else "focused"
            belief_conversion = _belief_conversion_likelihood(
                asset,
                channel,
                asset_score=asset_score,
                channel_score=channel_score,
                stage_alignment=stage_alignment,
                persona_alignment=persona_alignment,
                concern_alignment=concern_alignment,
                impact_record=impact_record,
                transition_stage_tokens=transition_stage_tokens,
                belief_probability=belief_probability,
            )

            asset_type = asset.get("category") or (asset.get("format") or "").lower() or None
            channel_type = channel_payload.get("channel_type")

            combo_reasons = {
                **reasons,
                **channel_reasons,
                "stage_alignment": stage_alignment,
                "persona_alignment": persona_alignment,
                "concern_alignment": concern_alignment,
                "channel_persona_alignment": channel_persona_alignment,
            }
            combo_rationale = {
                "segment": {
                    "asset_matches": reasons.get("segment_fit"),
                    "channel_matches": channel_reasons.get("channel_segment_fit"),
                    "account_segments": list(account_segment_labels or []),
                },
                "belief": {
                    "belief_tokens": reasons.get("belief_alignment"),
                    "pain_match": reasons.get("concern_match"),
                    "channel_concern_match": channel_reasons.get("channel_concern_match"),
                    "stage_tokens": list(transition_stage_tokens or []),
                },
            }
            evidence_payload = {
                "segment_fit": reasons.get("segment_fit"),
                "belief_alignment": reasons.get("belief_alignment"),
                "historical_win_lift": (impact_record or {}).get("impact_strength"),
                "evidence_count": (impact_record or {}).get("evidence_count"),
                "channel_stage_fit": channel_reasons.get("channel_stage_fit"),
                "channel_concern_fit": channel_reasons.get("channel_concern_fit"),
                "concern_match": reasons.get("concern_match"),
            }
            evidence_payload = {
                key: value
                for key, value in evidence_payload.items()
                if value not in (None, [], {}, "")
            }

            delta_meta = {
                "source": expected_source,
                "confidence": round(confidence, 4),
                "learned": bool(learned_effect),
                "impact_evidence": int((impact_record or {}).get("evidence_count", 0)),
            }
            if calibration_meta:
                delta_meta.update(calibration_meta or {})

            lift_asset_prob = max(0.0, expected_delta / 10000.0)
            intervention_mode = choose_intervention_mode(
                lift_asset=lift_asset_prob,
                lift_subsidy=subsidy_lift_prob,
                time_to_subsidy_peak_days=time_to_subsidy_peak_days,
            )

            scored.append(
                (
                    expected_delta,
                    {
                        "persona_id": persona["id"],
                        "persona_label": persona["label"],
                        "stage_index": transition.get("stage_index", 0),
                        "belief_transition": transition.get("belief_transition"),
                        "asset": asset_payload,
                        "asset_label": asset_payload.get("name")
                        or asset_payload.get("format")
                        or asset_payload.get("category_label"),
                        "channel": channel_payload,
                        "channel_label": channel_payload.get("name")
                        or channel_payload.get("channel_type_label"),
                        "name": asset_payload.get("name")
                        or asset_payload.get("format")
                        or asset_payload.get("category_label")
                        or "Recommended play",
                        "asset_type": asset_type,
                        "asset_type_label": asset.get("category_label")
                        or _title_case_value(asset.get("category") or asset.get("format")),
                        "channel_type": channel_type,
                        "channel_type_label": channel_payload.get("channel_type_label"),
                        "expected_delta_bp": expected_delta,
                        "expected_delta_pct": expected_delta / 100.0,
                        "confidence": confidence,
                        "expected_delta_bp_meta": delta_meta,
                        "asset_score": asset_score,
                        "channel_score": channel_score,
                        "exploration_weight": exploration_weight,
                        "path_probability": path_probability,
                        "mode": mode,
                        "broadness_score": broadness,
                        "duration_days": int(duration_days),
                        "belief_conversion_likelihood": belief_conversion,
                        "evidence_count": int((impact_record or {}).get("evidence_count", 0)),
                        "impact_strength": (impact_record or {}).get("impact_strength"),
                        "impact_stages": (impact_record or {}).get("stages"),
                        "reasons": combo_reasons,
                        "rationale": combo_rationale,
                        "target_stage": stage_label,
                        "target_concern": target_concern_label,
                        "segment_fit_score": round(
                            max(asset_segment_score, channel_segment_score),
                            3,
                        )
                        if (asset_segment_score or channel_segment_score)
                        else None,
                        "belief_probability": round(belief_probability, 4),
                        "evidence": evidence_payload or None,
                        "lift_asset_prob": round(lift_asset_prob, 6),
                        "lift_asset_bp": round(lift_asset_prob * 10000.0, 2),
                        "lift_subsidy_prob": round(subsidy_lift_prob, 6)
                        if subsidy_lift_prob
                        else 0.0,
                        "lift_subsidy_bp": round(subsidy_lift_prob * 10000.0, 2)
                        if subsidy_lift_prob
                        else 0.0,
                        "intervention_mode": intervention_mode.value,
                        "subsidy_time_to_peak_days": time_to_subsidy_peak_days,
                    },
                )
            )

    scored.sort(
        key=lambda item: (
            item[0],
            item[1].get("belief_conversion_likelihood") or 0.0,
        ),
        reverse=True,
    )
    if not scored:
        fallback_assets = sorted(
            assets,
            key=lambda a: float(a.get("base_lift_bp") or 20.0),
            reverse=True,
        )[:max_combos]
        fallback_channels = sorted(
            channels,
            key=lambda c: float(c.get("reach_score") or 0.5),
            reverse=True,
        )[:2]
        for asset in fallback_assets:
            for channel in fallback_channels or [{}]:
                base_lift = float(asset.get("base_lift_bp") or 20.0)
                reach = float(channel.get("reach_score") or 0.5)
                expected_delta = max(5.0, base_lift * reach * 0.6)
                asset_payload = _asset_payload(asset)
                channel_payload = _channel_payload(channel if channel else {})
                channel_type = channel_payload.get("channel_type")
                belief_conversion = _belief_conversion_likelihood(
                    asset,
                    channel,
                    asset_score=0.1,
                    channel_score=reach,
                    stage_alignment=False,
                    persona_alignment=False,
                    concern_alignment=False,
                    impact_record=None,
                    transition_stage_tokens=transition_stage_tokens,
                    belief_probability=belief_probability,
                )
                fallback_rationale = {
                    "segment": {
                        "account_segments": list(account_segment_labels or []),
                    },
                    "belief": {
                        "stage_tokens": list(transition_stage_tokens or []),
                    },
                }
                lift_asset_prob = max(0.0, expected_delta / 10000.0)
                intervention_mode = choose_intervention_mode(
                    lift_asset=lift_asset_prob,
                    lift_subsidy=subsidy_lift_prob,
                    time_to_subsidy_peak_days=time_to_subsidy_peak_days,
                )
                scored.append(
                    (
                        expected_delta,
                        {
                            "persona_id": persona["id"],
                            "persona_label": persona["label"],
                            "stage_index": transition.get("stage_index", 0),
                            "belief_transition": transition.get("belief_transition"),
                            "asset": asset_payload,
                            "channel": channel_payload,
                            "asset_type": asset.get("category") or (asset.get("format") or "").lower(),
                            "asset_type_label": asset.get("category_label")
                            or _title_case_value(asset.get("category") or asset.get("format")),
                            "channel_type": channel_type,
                            "channel_type_label": channel_payload.get("channel_type_label"),
                            "expected_delta_bp": expected_delta,
                            "expected_delta_pct": expected_delta / 100.0,
                            "confidence": 0.45,
                            "asset_score": 0.1,
                            "channel_score": reach,
                            "exploration_weight": exploration_weight,
                            "path_probability": path_probability,
                            "mode": "broad" if reach >= 0.6 else "focused",
                            "broadness_score": reach,
                            "duration_days": int(asset.get("time_to_effect_days") or 14),
                            "belief_conversion_likelihood": belief_conversion,
                            "evidence_count": 0,
                            "impact_strength": None,
                            "impact_stages": None,
                            "reasons": {
                                "fallback": True,
                            },
                            "rationale": fallback_rationale,
                            "target_stage": stage_label,
                            "target_concern": target_concern_label,
                            "belief_probability": round(belief_probability, 4),
                            "lift_asset_prob": round(lift_asset_prob, 6),
                            "lift_asset_bp": round(lift_asset_prob * 10000.0, 2),
                            "lift_subsidy_prob": round(subsidy_lift_prob, 6)
                            if subsidy_lift_prob
                            else 0.0,
                            "lift_subsidy_bp": round(subsidy_lift_prob * 10000.0, 2)
                            if subsidy_lift_prob
                            else 0.0,
                            "intervention_mode": intervention_mode.value,
                            "subsidy_time_to_peak_days": time_to_subsidy_peak_days,
                        },
                    )
                )

        scored.sort(key=lambda item: item[0], reverse=True)

    broad_candidates = [play for _, play in scored if play.get("mode") == "broad"]
    focus_candidates = [play for _, play in scored if play.get("mode") == "focused"]

    selected: List[Dict[str, Any]] = []
    if focus_candidates:
        candidate = focus_candidates[0]
        if not _combo_is_duplicate(selected, candidate):
            selected.append(candidate)
    if broad_candidates:
        candidate = broad_candidates[0]
        if not _combo_is_duplicate(selected, candidate):
            selected.append(candidate)

    for _, play in scored:
        if len(selected) >= max_combos:
            break
        if not _combo_is_duplicate(selected, play):
            selected.append(play)

    return selected[:max_combos]


# ---------------------------------------------------------------------------
# Scheduling & campaign grouping
# ---------------------------------------------------------------------------

def _quarter_from_day(day: int) -> str:
    quarter_index = max(0, day) // 90 + 1
    return f"Q{quarter_index}"


def _schedule_plays(plays: List[Dict[str, Any]]) -> None:
    if not plays:
        return

    broad_cursor = 0
    focus_cursor = 0

    plays.sort(
        key=lambda p: (
            p.get("stage_index", 0),
            -p.get("expected_delta_bp", 0.0),
            -p.get("confidence", 0.0),
        )
    )

    for play in plays:
        stage_base = play.get("stage_index", 0) * 30
        duration = max(14, play.get("duration_days") or 14)
        if play.get("mode") == "broad":
            start = max(stage_base, broad_cursor)
            broad_cursor = start + max(10, duration // 2)
        else:
            start = max(stage_base + 7, focus_cursor)
            focus_cursor = start + duration
        play["start_day"] = int(start)
        play["end_day"] = int(start + duration)
        play["quarter"] = _quarter_from_day(play["start_day"])
        bt = play.get("belief_transition") or {}
        stage_index = play.get("stage_index", 0)
        stage_key = _stage_for_transition(bt, stage_index=stage_index)
        bt.setdefault("stage", stage_key)
        bt.setdefault(
            "stage_label",
            BELIEF_STAGE_LABELS.get(stage_key, _title_case_value(stage_key) or stage_key),
        )
        stage_label = bt.get("stage_label")
        focus_core = _focus_core_label(bt)
        people_names = play.get("people_names") or []
        persona_focus = _people_focus_label(play.get("persona_label"), people_names)
        play["persona_focus"] = persona_focus
        play["stage_key"] = stage_key
        play["stage_label"] = stage_label
        play["focus_label"] = focus_core
        focus_summary = _conversion_focus_summary_label(
            stage_label,
            focus_core,
            bt.get("narrative"),
        )
        play["conversion_focus_label"] = focus_summary
        play["theme"] = f"{persona_focus} · {focus_summary}"
        play["sequence_index"] = len(
            [p for p in plays if p.get("quarter") == play["quarter"]]
        )


def _annotate_execution_layers(
    plays: List[Dict[str, Any]],
    persona_lookup: Dict[str, Dict[str, Any]],
    account_meta: Dict[str, Any],
) -> None:
    if not plays:
        return

    max_delta = max(float(play.get("expected_delta_bp") or 0.0) for play in plays) or 1.0
    ordered_indices = sorted(
        range(len(plays)),
        key=lambda idx: (
            plays[idx].get("start_day", 0),
            plays[idx].get("mode") == "broad",
            -plays[idx].get("expected_delta_bp", 0.0),
        ),
    )

    for timeline_index, play_idx in enumerate(ordered_indices):
        play = plays[play_idx]
        persona = persona_lookup.get(play.get("persona_id")) or {}
        perceptibility = _safe_float(persona.get("perceptibility"), 0.5) or 0.5
        proximity = _safe_float(persona.get("proximity"), 0.5) or 0.5
        involvement = _safe_float(persona.get("involvement"), 0.5) or 0.5
        delta_factor = min(
            1.0, float(play.get("expected_delta_bp") or 0.0) / max_delta
        )
        exploration = _safe_float(play.get("exploration_weight"), 0.15) or 0.15
        chance_factor = 1.0 - min(1.0, exploration / 0.35)
        persona_pressure = 0.6 * (1.0 - min(1.0, perceptibility)) + 0.4 * (
            1.0 - min(1.0, proximity)
        )
        criticality = (
            0.55 * delta_factor
            + 0.25 * max(0.0, persona_pressure)
            + 0.2 * chance_factor
        )
        effort = min(0.95, max(0.1, criticality))

        play["timeline_index"] = timeline_index
        play["timeline_label"] = f"T+{timeline_index}"
        play["timeline_days"] = int(play.get("start_day", 0))
        play["effort_required"] = round(effort, 3)
        play["effort_pct"] = int(round(effort * 100))
        play["account_meta"] = account_meta
        play["persona_meta"] = {
            "id": persona.get("id") or play.get("persona_id"),
            "label": persona.get("label") or play.get("persona_label"),
            "perceptibility": persona.get("perceptibility"),
            "proximity": persona.get("proximity"),
            "involvement": persona.get("involvement"),
            "priority_rank": persona.get("priority_rank"),
            "priority_score": persona.get("priority_score"),
            "title": persona.get("title"),
            "department": persona.get("department"),
            "seniority": persona.get("seniority"),
        }
        play["persona_descriptor"] = _persona_descriptor_from_meta(
            play["persona_meta"],
            play.get("persona_label"),
        )
        play["segment_filters"] = _aggregate_segment_filters(
            [account_meta.get("attributes")]
        )


def _build_conversion_sequence(
    plays: Sequence[Dict[str, Any]],
    persona_lookup: Dict[str, Dict[str, Any]],
    account_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not plays:
        return []

    sequence: List[Dict[str, Any]] = []
    seen: Dict[
        Tuple[Optional[str], Optional[str], Optional[str], Optional[str]],
        Dict[str, Any],
    ] = {}

    ordered_plays = sorted(
        plays,
        key=lambda play: (
            play.get("timeline_index", 0),
            play.get("start_day", 0),
            -play.get("expected_delta_bp", 0.0),
        ),
    )

    for play in ordered_plays:
        bt = play.get("belief_transition") or {}
        key = (
            play.get("persona_id"),
            (bt.get("problem") or {}).get("id"),
            (bt.get("pain") or {}).get("id"),
            (bt.get("resolution") or {}).get("id"),
        )
        persona = persona_lookup.get(play.get("persona_id")) or {}
        entry = seen.get(key)
        if not entry:
            entry = {
                "timeline_index": play.get("timeline_index", 0),
                "timeline_label": play.get("timeline_label"),
                "timeline_days": play.get("timeline_days", 0),
                "persona": {
                    "id": persona.get("id") or play.get("persona_id"),
                    "label": persona.get("label") or play.get("persona_label"),
                    "perceptibility": persona.get("perceptibility"),
                    "proximity": persona.get("proximity"),
                    "involvement": persona.get("involvement"),
                    "priority_rank": persona.get("priority_rank"),
                    "priority_score": persona.get("priority_score"),
                    "title": persona.get("title"),
                    "department": persona.get("department"),
                    "seniority": persona.get("seniority"),
                },
                "belief_transition": bt,
                "expected_delta_bp": 0.0,
                "confidence_sum": 0.0,
                "effort_weighted": 0.0,
                "effort_denominator": 0.0,
                "plays": [],
                "account": account_meta,
                "persona_descriptor": _persona_descriptor_from_meta(
                    {
                        "label": persona.get("label") or play.get("persona_label"),
                        "title": persona.get("title"),
                        "department": persona.get("department"),
                        "seniority": persona.get("seniority"),
                    },
                    play.get("persona_label"),
                ),
                "segment_buffer": [],
                "people_names_set": set(),
                "people_records": [],
                "belief_conversion_sum": 0.0,
                "belief_conversion_count": 0,
                "duration_values": [],
            }
            seen[key] = entry
            sequence.append(entry)

        delta = float(play.get("expected_delta_bp") or 0.0)
        confidence = float(play.get("confidence") or 0.0)
        effort = float(play.get("effort_required") or 0.0)
        entry["expected_delta_bp"] += delta
        entry["confidence_sum"] += confidence
        entry["effort_weighted"] += effort * (delta or 1.0)
        entry["effort_denominator"] += delta or 1.0
        entry["plays"].append(play)
        segment_attrs = (play.get("account_meta") or {}).get("attributes")
        if segment_attrs:
            entry["segment_buffer"].append(segment_attrs)
        for name in play.get("people_names") or []:
            entry["people_names_set"].add(name)
        entry["people_records"].extend(play.get("people") or [])
        belief_conv = play.get("belief_conversion_likelihood")
        if belief_conv is not None:
            entry["belief_conversion_sum"] += float(belief_conv)
            entry["belief_conversion_count"] += 1
        duration = play.get("duration_days")
        if duration is not None:
            entry["duration_values"].append(float(duration))

    for entry in sequence:
        plays_for_entry = entry.get("plays") or []
        entry["asset_table"] = _aggregate_asset_rows(plays_for_entry)
        total_delta = entry.get("expected_delta_bp") or 0.0
        if total_delta:
            entry["expected_delta_bp"] = round(total_delta, 2)
        denom = entry.pop("effort_denominator", 0.0) or 1.0
        weighted_effort = entry.pop("effort_weighted", 0.0) / denom
        entry["effort_required"] = round(min(0.95, max(0.1, weighted_effort)), 3)
        entry["effort_pct"] = int(round(entry["effort_required"] * 100))
        confidence_sum = entry.pop("confidence_sum", 0.0)
        entry["avg_confidence"] = (
            confidence_sum / len(plays_for_entry) if plays_for_entry else 0.0
        )
        entry["segment_filters"] = _aggregate_segment_filters(
            entry.pop("segment_buffer", [])
        )
        people_names = sorted(entry.pop("people_names_set", set()))
        if people_names:
            entry["people"] = people_names
        else:
            entry["people"] = []
        records = entry.pop("people_records", [])
        seen_match_ids: set = set()
        unique_records: List[Dict[str, Any]] = []
        for rec in records:
            match_id = rec.get("match_id")
            if match_id and match_id in seen_match_ids:
                continue
            if match_id:
                seen_match_ids.add(match_id)
            unique_records.append(rec)
        entry["matched_people"] = unique_records
        belief_count = entry.pop("belief_conversion_count", 0)
        belief_sum = entry.pop("belief_conversion_sum", 0.0)
        entry["avg_belief_conversion"] = (
            belief_sum / belief_count if belief_count else None
        )
        durations = entry.pop("duration_values", [])
        entry["time_to_effect_days"] = int(
            round(_safe_mean(durations) or 14.0)
        ) if durations else 14
        bt = entry.get("belief_transition") or {}
        entry["belief_transition_meta"] = {
            "stage": bt.get("stage"),
            "stage_label": bt.get("stage_label"),
            "problem": (bt.get("problem") or {}).get("label"),
            "pain": (bt.get("pain") or {}).get("label"),
            "resolution": (bt.get("resolution") or {}).get("label"),
            "from_to": bt.get("from_to") or [],
        }
        entry["segment_summary"] = _segment_summary_text(entry["segment_filters"])
        entry["expected_outcome_summary"] = _build_expected_outcome_summary(
            entry.get("persona_descriptor"),
            entry["belief_transition_meta"].get("stage_label"),
            entry.get("expected_delta_bp"),
            1,
            entry["segment_filters"],
        )

    sequence.sort(key=lambda item: (item.get("timeline_index", 0), item.get("timeline_days", 0)))
    return sequence


def _group_campaigns(plays: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for play in plays:
        persona_focus = play.get("persona_focus") or _persona_focus_label(
            play.get("persona_label")
        )
        quarter = play.get("quarter") or "Q1"
        key = (quarter, persona_focus)
        group = groups.setdefault(
            key,
            {
                "quarter": quarter,
                "persona_focus": persona_focus,
                "plays": [],
                "start_day": play.get("start_day", 0),
                "end_day": play.get("end_day", 0),
                "focus_personas": set(),
                "focus_people": set(),
                "pains": set(),
                "total_delta_bp": 0.0,
                "confidence_sum": 0.0,
                "broad_count": 0,
                "focus_count": 0,
                "focus_counter": Counter(),
                "stage_counter": Counter(),
                "conversion_focus_map": {},
                "effort_sum": 0.0,
                "effort_count": 0,
                "segment_buffer": [],
                "persona_descriptors": Counter(),
            },
        )
        group["plays"].append(play)
        group["start_day"] = min(group["start_day"], play.get("start_day", 0))
        group["end_day"] = max(group["end_day"], play.get("end_day", 0))
        if play.get("persona_label"):
            group["focus_personas"].add(play["persona_label"])
        for person_name in play.get("people_names") or []:
            if person_name:
                group["focus_people"].add(person_name)
        transition = play.get("belief_transition") or {}
        focus_label = play.get("conversion_focus_label")
        if focus_label:
            group["focus_counter"][focus_label] += float(
                play.get("expected_delta_bp") or 0.0
            )
        stage_label = play.get("stage_label") or transition.get("stage_label")
        if stage_label:
            group["stage_counter"][stage_label] += float(
                play.get("expected_delta_bp") or 0.0
            )
        pain_entry = transition.get("pain")
        if isinstance(pain_entry, dict):
            pain_label = pain_entry.get("label") or pain_entry.get("id")
        else:
            pain_label = pain_entry if isinstance(pain_entry, str) else None
        if pain_label:
            group["pains"].add(pain_label)
        group["total_delta_bp"] += float(play.get("expected_delta_bp") or 0.0)
        group["confidence_sum"] += float(play.get("confidence") or 0.0)
        group["effort_sum"] += float(play.get("effort_required") or 0.0)
        group["effort_count"] += 1
        segment_attrs = (play.get("account_meta") or {}).get("attributes")
        if segment_attrs:
            group["segment_buffer"].append(segment_attrs)
        descriptor = play.get("persona_descriptor")
        if descriptor:
            group["persona_descriptors"][descriptor] += 1
        if play.get("mode") == "broad":
            group["broad_count"] += 1
        else:
            group["focus_count"] += 1

        focus_map = group["conversion_focus_map"]
        bt = play.get("belief_transition") or {}
        focus_key = (
            play.get("persona_id"),
            (bt.get("problem") or {}).get("id"),
            (bt.get("pain") or {}).get("id"),
            (bt.get("resolution") or {}).get("id"),
        )
        focus_entry = focus_map.setdefault(
            focus_key,
            {
                "persona_label": play.get("persona_label"),
                "persona_focus": persona_focus,
                "stage": bt.get("stage"),
                "stage_label": bt.get("stage_label"),
                "focus_label": play.get("focus_label") or _focus_core_label(bt),
                "label": play.get("conversion_focus_label"),
                "belief_transition": bt,
                "expected_delta_bp": 0.0,
                "confidence_sum": 0.0,
                "play_count": 0,
                "accounts": set(),
                "people": set(),
                "plays": [],
                "durations": [],
                "effort_sum": 0.0,
                "segment_buffer": [],
                "persona_meta": play.get("persona_meta") or {},
                "persona_descriptors": Counter(),
            },
        )
        focus_entry["expected_delta_bp"] += float(play.get("expected_delta_bp") or 0.0)
        focus_entry["confidence_sum"] += float(play.get("confidence") or 0.0)
        focus_entry["play_count"] += 1
        focus_entry["accounts"].add(
            (
                play.get("account_id"),
                play.get("account_name") or play.get("account_id"),
            )
        )
        focus_entry["people"].update(
            [name for name in (play.get("people_names") or []) if name]
        )
        focus_entry["plays"].append(play)
        focus_entry["durations"].append(play.get("duration_days") or 14)
        focus_entry["effort_sum"] += float(play.get("effort_required") or 0.0)
        segment_attrs = (play.get("account_meta") or {}).get("attributes")
        if segment_attrs:
            focus_entry["segment_buffer"].append(segment_attrs)
        descriptor = play.get("persona_descriptor")
        if descriptor:
            focus_entry["persona_descriptors"][descriptor] += 1
        timeline_idx = play.get("timeline_index")
        if timeline_idx is not None:
            focus_entry.setdefault("timeline_indices", []).append(timeline_idx)

    campaigns: List[Dict[str, Any]] = []
    for group in groups.values():
        sorted_plays = sorted(
            group["plays"],
            key=lambda p: (p.get("start_day", 0), p.get("mode") == "broad"),
        )
        total_plays = len(sorted_plays)
        confidence = group["confidence_sum"] / total_plays if total_plays else 0.0

        seen_beliefs: set[
            Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]
        ] = set()
        belief_entries: List[Dict[str, Any]] = []
        for play in sorted_plays:
            bt = play.get("belief_transition") or {}
            belief_key = (
                (bt.get("persona") or {}).get("id"),
                (bt.get("problem") or {}).get("id"),
                (bt.get("pain") or {}).get("id"),
                (bt.get("resolution") or {}).get("id"),
            )
            if belief_key not in seen_beliefs:
                seen_beliefs.add(belief_key)
                belief_entries.append(bt)

        conversion_focuses: List[Dict[str, Any]] = []
        for focus_entry in group["conversion_focus_map"].values():
            play_count = focus_entry.get("play_count") or 0
            avg_confidence = (
                focus_entry["confidence_sum"] / play_count if play_count else 0.0
            )
            avg_effort = (
                focus_entry["effort_sum"] / play_count if play_count else 0.0
            )
            belief_scores = [
                play.get("belief_conversion_likelihood")
                for play in focus_entry.get("plays") or []
                if play.get("belief_conversion_likelihood") is not None
            ]
            avg_belief_conversion = _safe_mean(belief_scores)
            accounts = sorted(
                [
                    {"id": acc_id, "name": acc_name}
                    for acc_id, acc_name in focus_entry.get("accounts", set())
                    if acc_id
                ],
                key=lambda item: item["name"],
            )
            focus_asset_rows = _aggregate_asset_rows(focus_entry.get("plays") or [])
            avg_duration = _safe_mean(focus_entry.get("durations") or []) or 14.0
            segment_filters = _aggregate_segment_filters(
                focus_entry.get("segment_buffer", [])
            )
            persona_meta = focus_entry.get("persona_meta") or {}
            if not persona_meta and focus_entry.get("plays"):
                persona_meta = (focus_entry["plays"][0].get("persona_meta") or {}).copy()
            persona_descriptors = focus_entry.get("persona_descriptors") or Counter()
            if not persona_descriptors:
                fallback_descriptor = _persona_descriptor_from_meta(
                    persona_meta,
                    focus_entry.get("persona_label"),
                )
                persona_descriptors = Counter({fallback_descriptor: max(1, play_count)})
            descriptor_counts = [
                {"descriptor": desc, "count": cnt}
                for desc, cnt in persona_descriptors.most_common()
            ]
            primary_descriptor = descriptor_counts[0]["descriptor"]
            timeline_indices = focus_entry.get("timeline_indices") or []
            focus_timeline_index = (
                min(timeline_indices) if timeline_indices else None
            )
            time_to_effect_days = int(round(avg_duration))
            expected_outcome_summary = _build_expected_outcome_summary(
                primary_descriptor,
                focus_entry.get("stage_label"),
                focus_entry.get("expected_delta_bp"),
                len(accounts),
                segment_filters,
            )
            conversion_focuses.append(
                {
                    "persona_label": focus_entry.get("persona_label"),
                    "persona_focus": focus_entry.get("persona_focus"),
                    "stage": focus_entry.get("stage"),
                    "stage_label": focus_entry.get("stage_label"),
                    "focus_label": focus_entry.get("focus_label"),
                    "label": focus_entry.get("label"),
                    "belief_transition": focus_entry.get("belief_transition"),
                    "expected_delta_bp": focus_entry.get("expected_delta_bp"),
                    "confidence": avg_confidence,
                    "accounts": accounts,
                    "account_count": len(accounts),
                    "people": sorted(focus_entry.get("people") or []),
                    "asset_table": focus_asset_rows,
                    "avg_duration_days": avg_duration,
                    "time_to_effect_days": time_to_effect_days,
                    "avg_effort_required": avg_effort,
                    "avg_effort_pct": int(round(min(0.95, max(0.1, avg_effort)) * 100)),
                    "avg_belief_conversion": avg_belief_conversion,
                    "segment_filters": segment_filters,
                    "segment_summary": _segment_summary_text(segment_filters),
                    "persona_meta": persona_meta,
                    "persona_descriptor": primary_descriptor,
                    "persona_descriptor_counts": descriptor_counts,
                    "expected_outcome_summary": expected_outcome_summary,
                    "timeline_index": focus_timeline_index,
                    "timeline_label": (
                        f"T+{focus_timeline_index}"
                        if focus_timeline_index is not None
                        else None
                    ),
                    "plays": focus_entry.get("plays"),
                }
            )
        conversion_focuses.sort(
            key=lambda item: item.get("expected_delta_bp", 0.0), reverse=True
        )

        persona_focus_label = group.get("persona_focus") or "Persona"
        focus_labels = [
            entry.get("focus_label") or entry.get("label")
            for entry in conversion_focuses
            if entry.get("focus_label") or entry.get("label")
        ]
        focus_labels = [label for label in focus_labels if label]
        theme_focus = " + ".join(focus_labels[:2])
        if not theme_focus:
            top_stage = group["stage_counter"].most_common(1)
            theme_focus = top_stage[0][0] if top_stage else None
        theme_label = (
            f"{persona_focus_label} · {theme_focus}"
            if theme_focus
            else persona_focus_label
        )
        conversion_summary = ", ".join(label for label in focus_labels[:3] if label)
        segment_filters = _aggregate_segment_filters(group.get("segment_buffer", []))
        segment_summary = _segment_summary_text(segment_filters)
        persona_descriptor_counts = [
            {"descriptor": desc, "count": cnt}
            for desc, cnt in group["persona_descriptors"].most_common()
        ]
        primary_descriptor = (
            persona_descriptor_counts[0]["descriptor"]
            if persona_descriptor_counts
            else persona_focus_label
        )
        primary_stage_label = None
        if group["stage_counter"]:
            primary_stage_label = group["stage_counter"].most_common(1)[0][0]
        expected_outcome_summary = _build_expected_outcome_summary(
            primary_descriptor,
            primary_stage_label,
            group["total_delta_bp"],
            len(accounts),
            segment_filters,
        )

        campaigns.append(
            {
                "quarter": group["quarter"],
                "theme": theme_label,
                "persona_focus": persona_focus_label,
                "conversion_summary": conversion_summary,
                "conversion_focuses": conversion_focuses,
                "focus_personas": sorted([p for p in group["focus_personas"] if p]),
                "focus_people": sorted([p for p in group["focus_people"] if p]),
                "focus_pains": sorted([p for p in group["pains"] if p]),
                "start_day": group["start_day"],
                "end_day": group["end_day"],
                "total_delta_bp": group["total_delta_bp"],
                "confidence": confidence,
                "plays": sorted_plays,
                "mode_mix": {
                    "broad": group["broad_count"],
                    "focused": group["focus_count"],
                },
                "belief_transitions": belief_entries,
                "asset_table": _aggregate_asset_rows(sorted_plays),
                "segment_filters": segment_filters,
                "segment_summary": segment_summary,
                "persona_descriptor_counts": persona_descriptor_counts,
                "expected_outcome_summary": expected_outcome_summary,
            }
        )

    campaigns.sort(
        key=lambda c: (
            int("".join(ch for ch in str(c["quarter"]) if ch.isdigit()) or "0"),
            c["start_day"],
        )
    )
    return campaigns


# ---------------------------------------------------------------------------
# Randomness / exploration policy helpers
# ---------------------------------------------------------------------------

def _exploration_weight(path_index: int, stage_index: int, total_paths: int) -> float:
    base = 0.28 * math.exp(-0.4 * stage_index)
    path_modifier = 1.0 + (path_index / max(1, total_paths - 1)) * 0.6
    if path_index == 0:
        path_modifier *= 0.7
    weight = base * path_modifier
    return round(min(0.35, max(0.05, weight)), 3)


def _randomization_policy(
    paths: Sequence[Dict[str, Any]],
    plays: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    path_allocations = []
    for idx, path in enumerate(paths):
        path_allocations.append(
            {
                "path_id": path.get("id") or f"path:{idx}",
                "probability": round(path.get("probability", 0.0), 4),
                "exploration_weight": round(
                    _exploration_weight(idx, 0, len(paths)), 4
                ),
                "persona_ids": path.get("persona_ids") or [],
            }
        )

    early = [
        play["exploration_weight"]
        for play in plays
        if play.get("stage_index", 0) <= 1
    ]
    late = [
        play["exploration_weight"]
        for play in plays
        if play.get("stage_index", 0) >= 3
    ]

    return {
        "path_allocation": path_allocations,
        "belief_spread": {
            "early_cycle_avg": round(sum(early) / len(early), 3) if early else 0.0,
            "late_cycle_avg": round(sum(late) / len(late), 3) if late else 0.0,
        },
        "notes": "Exploration weights favour early-cycle breadth and continue sampling secondary paths.",
    }


# ---------------------------------------------------------------------------
# Asset aggregation helpers
# ---------------------------------------------------------------------------

def _normalized_score(
    value: Optional[float],
    *,
    max_value: float,
    min_value: float = 0.0,
) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if max_value <= min_value:
        return None
    normalized = (numeric - min_value) / (max_value - min_value)
    return float(min(1.0, max(0.0, normalized)))


def _asset_fit_from_play(play: Dict[str, Any]) -> Optional[float]:
    score = play.get("asset_score")
    return _normalized_score(_safe_float(score), max_value=_ASSET_SCORE_NORMALIZER)


def _channel_engagement_from_play(play: Dict[str, Any]) -> Optional[float]:
    channel = play.get("channel") or {}
    usage = channel.get("usage") or {}
    candidates = [
        _safe_float(usage.get("engagement_score")),
        _safe_float(channel.get("engagement_score")),
        _safe_float(channel.get("reach_score")),
    ]
    for candidate in candidates:
        if candidate is not None:
            return float(min(1.0, max(0.0, candidate)))
    chan_score = _safe_float(play.get("channel_score"))
    if chan_score is None:
        return None
    normalized = _normalized_score(
        chan_score,
        max_value=_CHANNEL_SCORE_NORMALIZER,
        min_value=0.0,
    )
    return normalized


def _concern_label_from_transition(bt: Optional[Dict[str, Any]]) -> Optional[str]:
    if not bt:
        return None
    return (
        _node_label(bt.get("pain"))
        or _node_label(bt.get("problem"))
        or _node_label(bt.get("resolution"))
    )


def _aggregate_asset_rows(plays: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not plays:
        return []

    asset_pairs: Dict[
        Tuple[str, str],
        Dict[str, Any],
    ] = {}

    for play in plays:
        asset = play.get("asset") or {}
        channel = play.get("channel") or {}
        asset_id = asset.get("id") or asset.get("name") or asset.get("slug") or "asset"
        channel_id = (
            channel.get("id")
            or channel.get("name")
            or channel.get("channel_type")
            or channel.get("type")
            or "channel"
        )
        pair_key = (asset_id, channel_id)

        entry = asset_pairs.setdefault(
            pair_key,
            {
                "asset": asset,
                "asset_type": play.get("asset_type"),
                "asset_type_label": play.get("asset_type_label"),
                "channel": channel,
                "channel_type": play.get("channel_type")
                or channel.get("channel_type")
                or channel.get("type"),
                "channel_type_label": play.get("channel_type_label")
                or channel.get("channel_type_label")
                or _title_case_value(channel.get("type")),
                "expected_delta_bp_total": 0.0,
                "belief_conversion_scores": [],
                "confidence_scores": [],
                "engagement_scores": [],
                "asset_fit_values": [],
                "channel_engagement_values": [],
                "exploration_scores": [],
                "evidence_counts": [],
                "duration_days": [],
                "play_count": 0,
            },
        )
        entry["expected_delta_bp_total"] += float(play.get("expected_delta_bp") or 0.0)
        entry["belief_conversion_scores"].append(
            play.get("belief_conversion_likelihood")
        )
        entry["confidence_scores"].append(play.get("confidence"))
        entry["engagement_scores"].append(play.get("channel", {}).get("engagement_score"))
        entry["asset_fit_values"].append(_asset_fit_from_play(play))
        entry["channel_engagement_values"].append(_channel_engagement_from_play(play))
        entry["exploration_scores"].append(play.get("exploration_weight"))
        entry["evidence_counts"].append(play.get("evidence_count"))
        entry["duration_days"].append(play.get("duration_days") or 14)
        entry["play_count"] += 1

    asset_rows: List[Dict[str, Any]] = []
    for entry in asset_pairs.values():
        play_count = max(entry["play_count"], 1)
        avg_delta = entry["expected_delta_bp_total"] / play_count
        avg_confidence = _safe_mean(entry["confidence_scores"])
        avg_conversion = _safe_mean(entry["belief_conversion_scores"])
        avg_duration = _safe_mean(entry["duration_days"]) or 14.0
        engagement = _safe_mean(entry["engagement_scores"])
        exploration = _safe_mean(entry["exploration_scores"])
        asset_fit = _safe_mean([val for val in entry["asset_fit_values"] if val is not None])
        channel_engagement = _safe_mean(
            [val for val in entry["channel_engagement_values"] if val is not None]
        )
        evidence_total = sum(
            int(count or 0)
            for count in entry["evidence_counts"]
            if isinstance(count, (int, float))
        )

        asset_rows.append(
            {
                "asset": entry["asset"],
                "asset_type": entry["asset_type"],
                "asset_type_label": entry["asset_type_label"],
                "asset_fit_score": asset_fit,
                "asset_expected_delta_bp": avg_delta,
                "channel": entry["channel"],
                "channel_type": entry["channel_type"],
                "channel_type_label": entry["channel_type_label"],
                "expected_delta_bp": avg_delta,
                "belief_conversion_likelihood": avg_conversion,
                "confidence": avg_confidence,
                "evidence_count": evidence_total,
                "play_count": play_count,
                "engagement_score": channel_engagement or engagement,
                "exploration_weight": exploration,
                "avg_duration_days": avg_duration,
            }
        )

    asset_rows.sort(key=lambda row: row.get("expected_delta_bp", 0.0), reverse=True)
    return asset_rows


def _build_account_execution_interventions(
    conversion_sequence: Sequence[Dict[str, Any]],
    *,
    account_id: Optional[str],
    account_name: Optional[str],
    segment_context: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not conversion_sequence:
        return []

    cluster_keys = (segment_context or {}).get("keys") or []
    cluster_key = cluster_keys[0] if cluster_keys else None
    interventions: List[Dict[str, Any]] = []

    for entry in conversion_sequence:
        plays = entry.get("plays") or []
        if not plays:
            continue
        best_play = max(
            plays,
            key=lambda play: _safe_float(play.get("expected_delta_bp"), 0.0) or 0.0,
            default=None,
        )
        if not best_play:
            continue
        asset = best_play.get("asset") or {}
        channel = best_play.get("channel") or {}
        persona = entry.get("persona") or {}
        timeline_days = entry.get("timeline_days")
        quarter = best_play.get("quarter") or _quarter_from_day(timeline_days or 0)
        concern_label = (
            (entry.get("belief_transition_meta") or {}).get("pain")
            or best_play.get("target_concern")
            or (entry.get("belief_transition_meta") or {}).get("problem")
        )
        asset_type = (
            best_play.get("asset_type_label")
            or best_play.get("asset_type")
            or asset.get("category_label")
            or _title_case_value(asset.get("category"))
            or (asset.get("format") or "Asset")
        )
        channel_label = (
            best_play.get("channel_type_label")
            or channel.get("channel_type_label")
            or _title_case_value(channel.get("channel_type"))
            or _title_case_value(channel.get("type"))
            or channel.get("name")
            or "Channel"
        )
        asset_fit = _safe_float(best_play.get("asset_fit_score"), None)
        channel_engagement = _safe_float(best_play.get("engagement_score"), None)
        fitness = None
        if asset_fit is not None or channel_engagement is not None:
            fitness = (asset_fit or 0.0) * (channel_engagement or 0.0)

        wolves_score = _safe_float(persona.get("wolves_score"), None)
        wolves_delta = _safe_float(persona.get("wolves_delta_bp"), None)
        wolves_involvement = _safe_float(persona.get("wolves_involvement_rate"), None)
        wolves_blocker = _safe_float(persona.get("wolves_blocker_rate"), None)
        wolves_sample_size = persona.get("wolves_sample_size")
        is_new_persona = bool(persona.get("is_new_persona"))
        persona_source = persona.get("persona_source") or persona.get("persona_origin")

        interventions.append(
            {
                "id": best_play.get("id"),
                "account_id": account_id,
                "account_name": account_name,
                "cluster_key": cluster_key,
                "quarter": quarter,
                "time_label": entry.get("timeline_label") or best_play.get("timeline_label"),
                "timeline_index": entry.get("timeline_index"),
                "persona": persona.get("label")
                or best_play.get("persona_label")
                or "Target persona",
                "persona_id": persona.get("id") or best_play.get("persona_id"),
                "concern": concern_label,
                "asset_type": asset_type,
                "channel": channel_label,
                "recommended_asset_id": asset.get("id"),
                "recommended_asset_name": asset.get("name") or asset.get("title"),
                "fitness": fitness,
                "asset_fit_score": asset_fit,
                "channel_engagement": channel_engagement,
                "expected_delta_bp": _safe_float(best_play.get("expected_delta_bp"), 0.0)
                or 0.0,
                "accounts_impacted": 1,
                "stage_label": (entry.get("belief_transition_meta") or {}).get(
                    "stage_label"
                ),
                "wolves_persona_score": wolves_score,
                "wolves_delta_bp": wolves_delta,
                "wolves_involvement_rate": wolves_involvement,
                "wolves_blocker_rate": wolves_blocker,
                "wolves_sample_size": wolves_sample_size,
                "is_new_persona": is_new_persona,
                "persona_source": persona_source,
            }
        )

    return interventions


# ---------------------------------------------------------------------------
# Intervention & gap helpers
# ---------------------------------------------------------------------------

_MIN_COVERAGE_THRESHOLD = 0.3
_STEADY_COVERAGE_THRESHOLD = 0.4
_STRONG_COVERAGE_THRESHOLD = 0.7


def _normalize_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _aggregate_execution_interventions(
    accounts: Sequence[Dict[str, Any]],
    *,
    persona_chain_effects: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for account in accounts:
        exec_block = account.get("execution") or {}
        items = exec_block.get("interventions") or []
        if items:
            candidates.extend(items)

    if not candidates:
        return []

    groups: Dict[
        Tuple[str, str, str, str, str],
        Dict[str, Any],
    ] = {}

    for item in candidates:
        quarter = item.get("quarter") or "Q1"
        cluster_key = item.get("cluster_key") or ""
        persona_label = item.get("persona") or "Target persona"
        concern_label = item.get("concern") or "Priority concern"
        asset_type = item.get("asset_type") or "Asset"
        channel_label = item.get("channel") or "Channel"

        key = (
            quarter,
            _normalize_key(persona_label) or persona_label.lower(),
            _normalize_key(concern_label) or concern_label.lower(),
            _normalize_key(asset_type) or asset_type.lower(),
            _normalize_key(channel_label) or channel_label.lower(),
        )
        bucket = groups.setdefault(
            key,
            {
                "quarter": quarter,
                "persona": persona_label,
                "concern": concern_label,
                "asset_type": asset_type,
                "channel": channel_label,
                "time_label": item.get("time_label"),
                "expected_delta_bp": 0.0,
                "account_ids": set(),
                "account_names": set(),
                "best_fitness": None,
                "best_asset_id": None,
                "best_asset_name": None,
                "cluster_keys": set(),
                "wolves_persona_score": None,
                "wolves_delta_bp": None,
                "wolves_involvement_rate": None,
                "wolves_blocker_rate": None,
                "wolves_sample_size": None,
                "is_new_persona": False,
                "persona_source": item.get("persona_source"),
            },
        )
        if not bucket.get("time_label") and item.get("time_label"):
            bucket["time_label"] = item.get("time_label")
        bucket["expected_delta_bp"] += float(item.get("expected_delta_bp") or 0.0)
        account_id = item.get("account_id")
        if account_id:
            bucket["account_ids"].add(account_id)
        account_name = item.get("account_name")
        if account_name:
            bucket["account_names"].add(account_name)
        if cluster_key:
            bucket["cluster_keys"].add(cluster_key)
        fitness = item.get("fitness")
        if fitness is not None:
            current_best = bucket["best_fitness"]
            if current_best is None or fitness > current_best:
                bucket["best_fitness"] = fitness
                bucket["best_asset_id"] = item.get("recommended_asset_id")
                bucket["best_asset_name"] = item.get("recommended_asset_name")
        wolves_score = item.get("wolves_persona_score")
        if wolves_score is not None:
            current_wolves = bucket.get("wolves_persona_score")
            if current_wolves is None or wolves_score > current_wolves:
                bucket["wolves_persona_score"] = wolves_score
                bucket["wolves_delta_bp"] = item.get("wolves_delta_bp")
                bucket["wolves_involvement_rate"] = item.get("wolves_involvement_rate")
                bucket["wolves_blocker_rate"] = item.get("wolves_blocker_rate")
                bucket["wolves_sample_size"] = item.get("wolves_sample_size")
        if item.get("is_new_persona"):
            bucket["is_new_persona"] = True
        if item.get("persona_source"):
            bucket["persona_source"] = item.get("persona_source")

    def _quarter_key(value: str) -> Tuple[int, str]:
        try:
            if value.upper().startswith("Q"):
                return (int(value[1:]), value)
        except Exception:
            pass
        return (99, value)

    aggregated: List[Dict[str, Any]] = []
    for key, bucket in groups.items():
        cluster_keys = sorted(bucket.get("cluster_keys") or [])
        cluster_labels = [
            label
            for label in (_cluster_label_from_key(value) for value in cluster_keys)
            if label
        ]
        aggregated.append(
            {
                "id": "|".join(key),
                "quarter": bucket["quarter"],
                "cluster_key": cluster_keys[0] if cluster_keys else None,
                "cluster_keys": cluster_keys,
                "cluster_labels": cluster_labels,
                "persona": bucket["persona"],
                "concern": bucket["concern"],
                "asset_type": bucket["asset_type"],
                "channel": bucket["channel"],
                "time_label": bucket.get("time_label"),
                "recommended_asset_id": bucket["best_asset_id"],
                "recommended_asset_name": bucket["best_asset_name"],
                "fitness": bucket["best_fitness"],
                "expected_delta_bp": round(bucket["expected_delta_bp"], 2),
                "accounts_impacted": len(bucket["account_ids"]),
                "account_names": sorted(bucket["account_names"])[:5],
                "wolves_persona_score": bucket.get("wolves_persona_score"),
                "wolves_delta_bp": bucket.get("wolves_delta_bp"),
                "wolves_involvement_rate": bucket.get("wolves_involvement_rate"),
                "wolves_blocker_rate": bucket.get("wolves_blocker_rate"),
                "wolves_sample_size": bucket.get("wolves_sample_size"),
                "is_new_persona": bucket.get("is_new_persona"),
                "persona_source": bucket.get("persona_source"),
            }
        )

    effect_index = persona_chain_effects or {}
    for row in aggregated:
        persona_label = row.get("persona")
        if not persona_label:
            continue
        effect = effect_index.get(persona_label.strip().lower())
        if not effect:
            continue
        row["chain_effect"] = effect.get("description")
        row["chain_targets"] = effect.get("next_personas")
        row["chain_share"] = effect.get("share")

    aggregated.sort(
        key=lambda row: (
            _quarter_key(row.get("quarter") or "Q1"),
            -float(row.get("expected_delta_bp") or 0.0),
        )
    )
    return aggregated


def _build_persona_chain_effects(
    accounts: Sequence[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    follow_counter: Dict[str, Counter[str]] = defaultdict(Counter)
    total_occurrences: Counter[str] = Counter()
    sequence_samples: Dict[str, List[List[str]]] = defaultdict(list)

    for account in accounts:
        paths = (account.get("prediction") or {}).get("persona_paths") or []
        if not paths:
            continue
        primary = paths[0].get("personas") or []
        labels = [persona.get("label") for persona in primary if persona.get("label")]
        if not labels:
            continue
        for idx, label in enumerate(labels):
            if not label:
                continue
            total_occurrences[label] += 1
            trailing = labels[idx + 1 : idx + 3]
            if trailing:
                key = " → ".join(trailing)
                follow_counter[label][key] += 1
                sequence_samples[label].append(labels[idx : idx + 4])

    effect_index: Dict[str, Dict[str, Any]] = {}
    for label, total in total_occurrences.items():
        if total <= 0:
            continue
        normalized = label.strip().lower()
        next_personas: List[str] = []
        description = None
        share_value = None
        if follow_counter[label]:
            top_sequence, count = follow_counter[label].most_common(1)[0]
            share_value = count / total if total else None
            next_personas = top_sequence.split(" → ")
            description = (
                f"Unlocks {top_sequence} in {_percent_label(share_value)} of modeled wins."
                if share_value is not None
                else None
            )
        effect_index[normalized] = {
            "label": label,
            "share": share_value,
            "next_personas": next_personas,
            "description": description,
            "sample_sequence": (sequence_samples.get(label) or [None])[0],
        }
    return effect_index


def _cluster_label_from_key(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    parts = str(key).split(":")
    if len(parts) == 3:
        _, dimension, value = parts
    elif len(parts) == 2:
        dimension, value = parts
    else:
        return key.replace("_", " ").title()
    dim_label = dimension.replace("_", " ").title()
    value_label = value.replace("_", " ").title()
    return f"{dim_label}: {value_label}"


def _play_snapshot(play: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    asset = play.get("asset") or {}
    channel = play.get("channel") or {}
    format_label = (
        asset.get("category_label")
        or _title_case_value(asset.get("category"))
        or _title_case_value(asset.get("format"))
        or asset.get("name")
    )
    channel_label = (
        channel.get("channel_type_label")
        or _title_case_value(channel.get("channel_type"))
        or _title_case_value(channel.get("type"))
        or channel.get("name")
    )
    stage_label = play.get("stage_label") or play.get("target_stage")
    stage_key = _normalize_key(stage_label)
    if not format_label or not channel_label or not stage_key:
        return None
    persona_label = play.get("persona_label") or play.get("persona_descriptor")
    persona_key = _normalize_key(persona_label)
    belief_transition = play.get("belief_transition") or {}
    concern_label = (
        play.get("target_concern")
        or _concern_label_from_transition(belief_transition)
    )
    concern_key = _normalize_key(concern_label)
    format_fit = _asset_fit_from_play(play)
    channel_engagement = _channel_engagement_from_play(play)
    if format_fit is None and channel_engagement is None:
        return None
    coverage = (format_fit or 0.0) * (channel_engagement or 0.0)
    return {
        "format_label": format_label,
        "channel_label": channel_label,
        "asset_name": asset.get("name") or asset.get("title"),
        "asset_id": asset.get("id"),
        "channel_name": channel.get("name") or channel_label,
        "channel_id": channel.get("id"),
        "format_fitment": format_fit or 0.0,
        "channel_engagement": channel_engagement or 0.0,
        "coverage_score": coverage,
        "persona_label": persona_label,
        "persona_key": persona_key,
        "stage_label": stage_label,
        "stage_key": stage_key,
        "concern_label": concern_label,
        "concern_key": concern_key,
        "expected_delta_bp": _safe_float(play.get("expected_delta_bp"), 0.0) or 0.0,
        "confidence": _safe_float(play.get("confidence"), 0.0),
        "source": "existing",
    }


class _ModalityStats:
    def __init__(self) -> None:
        self.by_signature: Dict[
            Tuple[str, str, str], Dict[Tuple[str, str], Dict[str, Any]]
        ] = defaultdict(dict)
        self.by_persona_stage: Dict[
            Tuple[str, str], Dict[Tuple[str, str], Dict[str, Any]]
        ] = defaultdict(dict)
        self.by_stage: Dict[Tuple[str], Dict[Tuple[str, str], Dict[str, Any]]] = defaultdict(dict)
        self.global_combos: Dict[
            Tuple[str], Dict[Tuple[str, str], Dict[str, Any]]
        ] = defaultdict(dict)

    def observe_play(self, play: Dict[str, Any]) -> None:
        snapshot = _play_snapshot(play)
        if not snapshot:
            return
        stage_key = snapshot["stage_key"]
        persona_key = snapshot.get("persona_key")
        concern_key = snapshot.get("concern_key")
        if stage_key:
            self._record(self.by_stage, (stage_key,), snapshot)
        self._record(self.global_combos, ("global",), snapshot)
        if persona_key and stage_key:
            self._record(self.by_persona_stage, (persona_key, stage_key), snapshot)
            if concern_key:
                self._record(
                    self.by_signature,
                    (persona_key, stage_key, concern_key),
                    snapshot,
                )

    def _record(
        self,
        bucket: Dict[Tuple[str, ...], Dict[Tuple[str, str], Dict[str, Any]]],
        key: Tuple[str, ...],
        snapshot: Dict[str, Any],
    ) -> None:
        if any(part is None for part in key):
            return
        combo_key = (snapshot["format_label"], snapshot["channel_label"])
        combos = bucket.setdefault(key, {})
        entry = combos.setdefault(
            combo_key,
            {
                "count": 0,
                "coverage_sum": 0.0,
                "format_fit_sum": 0.0,
                "channel_engagement_sum": 0.0,
                "sample": snapshot,
            },
        )
        entry["count"] += 1
        entry["coverage_sum"] += snapshot["coverage_score"]
        entry["format_fit_sum"] += snapshot["format_fitment"]
        entry["channel_engagement_sum"] += snapshot["channel_engagement"]

    @staticmethod
    def _best_combo(
        combos: Optional[Dict[Tuple[str, str], Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        if not combos:
            return None
        items = []
        for combo_key, payload in combos.items():
            count = payload.get("count") or 1
            avg_coverage = (payload.get("coverage_sum") or 0.0) / count
            items.append((avg_coverage, combo_key, payload))
        if not items:
            return None
        items.sort(key=lambda item: item[0], reverse=True)
        _, combo_key, payload = items[0]
        count = payload.get("count") or 1
        sample = payload.get("sample") or {}
        return {
            "format_label": combo_key[0],
            "channel_label": combo_key[1],
            "coverage_score": (payload.get("coverage_sum") or 0.0) / count,
            "format_fitment": (payload.get("format_fit_sum") or 0.0) / count,
            "channel_engagement": (payload.get("channel_engagement_sum") or 0.0) / count,
            "asset_name": sample.get("asset_name"),
            "channel_name": sample.get("channel_name"),
            "asset_id": sample.get("asset_id"),
            "channel_id": sample.get("channel_id"),
            "source": "reference",
        }

    def recommend(
        self,
        persona_label: Optional[str],
        stage_label: Optional[str],
        concern_label: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        persona_key = _normalize_key(persona_label) or ""
        stage_key = _normalize_key(stage_label) or ""
        concern_key = _normalize_key(concern_label) or ""
        lookups = [
            (self.by_signature, (persona_key, stage_key, concern_key)),
            (self.by_persona_stage, (persona_key, stage_key)),
            (self.by_stage, (stage_key,)),
            (self.global_combos, ("global",)),
        ]
        for table, key in lookups:
            combo = self._best_combo(table.get(key))
            if combo:
                return combo
        return None


def _build_modality_stats(accounts: Sequence[Dict[str, Any]]) -> _ModalityStats:
    stats = _ModalityStats()
    for account in accounts:
        plays = (account.get("execution") or {}).get("plays") or []
        for play in plays:
            stats.observe_play(play)
    return stats


def _play_candidates(plays: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for play in plays:
        snapshot = _play_snapshot(play)
        if snapshot:
            candidates.append(snapshot)
    candidates.sort(
        key=lambda snap: (
            snap.get("coverage_score") or 0.0,
            snap.get("expected_delta_bp") or 0.0,
        ),
        reverse=True,
    )
    return candidates


def _build_intervention_from_seed(
    seed: Dict[str, Any],
    stats: _ModalityStats,
) -> Optional[Dict[str, Any]]:
    expected_delta = _safe_float(seed.get("expected_delta_bp"), 0.0) or 0.0
    if expected_delta <= 0.0:
        return None
    persona = seed.get("persona") or {}
    persona_label = seed.get("persona_label") or persona.get("label")
    stage_label = (
        seed.get("stage_label")
        or (seed.get("belief_transition_meta") or {}).get("stage_label")
    )
    stage_key = _normalize_key(stage_label)
    plays = seed.get("plays") or []
    candidates = _play_candidates(plays)
    best_current = candidates[0] if candidates else None
    coverage_score = best_current["coverage_score"] if best_current else 0.0
    coverage_score = float(min(1.0, max(0.0, coverage_score or 0.0)))
    concern_label = (
        seed.get("concern_label")
        or (seed.get("belief_transition_meta") or {}).get("pain")
        or (best_current or {}).get("concern_label")
    )
    fallback = stats.recommend(persona_label, stage_label, concern_label)
    needs_fallback = coverage_score < _MIN_COVERAGE_THRESHOLD or not best_current
    recommended_modality = fallback if needs_fallback else None
    current_modality = None
    if best_current:
        current_modality = {
            "format_label": best_current["format_label"],
            "channel_label": best_current["channel_label"],
            "format_fitment": best_current["format_fitment"],
            "channel_engagement": best_current["channel_engagement"],
            "coverage_score": best_current["coverage_score"],
            "asset_name": best_current.get("asset_name"),
            "channel_name": best_current.get("channel_name"),
            "asset_id": best_current.get("asset_id"),
            "channel_id": best_current.get("channel_id"),
            "source": "existing",
        }
    elif fallback:
        current_modality = {
            **fallback,
            "source": "reference",
        }
    gap_score = expected_delta * (1.0 - coverage_score)
    coverage_state = (
        "strong"
        if coverage_score >= _STRONG_COVERAGE_THRESHOLD
        else "steady"
        if coverage_score >= _STEADY_COVERAGE_THRESHOLD
        else "weak"
    )
    accounts = seed.get("accounts") or []
    unique_accounts = []
    seen_accounts: set = set()
    for acc in accounts:
        acc_id = acc.get("id") or acc.get("account_id")
        key = acc_id or acc.get("name")
        if key in seen_accounts:
            continue
        seen_accounts.add(key)
        unique_accounts.append(
            {
                "id": acc_id,
                "name": acc.get("name") or acc.get("account_name") or acc_id,
            }
        )
    belief_transition = seed.get("belief_transition")
    belief_meta = seed.get("belief_transition_meta") or {}
    descriptor = seed.get("persona_descriptor") or persona.get("title")
    intervention_id = "|".join(
        filter(
            None,
            [
                seed.get("scope") or "portfolio",
                seed.get("account_id"),
                _normalize_key(persona_label) or "persona",
                stage_key or "stage",
                _normalize_key(concern_label) or "concern",
            ],
        )
    )
    asset_options = [
        {
            "format_label": snap["format_label"],
            "channel_label": snap["channel_label"],
            "coverage_score": snap["coverage_score"],
            "asset_name": snap.get("asset_name"),
            "channel_name": snap.get("channel_name"),
        }
        for snap in candidates[:3]
    ]
    return {
        "id": intervention_id,
        "scope": seed.get("scope") or "portfolio",
        "account_id": seed.get("account_id"),
        "accounts": unique_accounts,
        "expected_account_count": len(unique_accounts),
        "persona_label": persona_label,
        "persona_descriptor": descriptor,
        "stage_label": stage_label,
        "stage_key": stage_key,
        "belief_transition": belief_transition,
        "belief_transition_meta": belief_meta,
        "belief_lift_bp": expected_delta,
        "confidence": seed.get("confidence"),
        "people": seed.get("people") or [],
        "matched_people": seed.get("matched_people") or [],
        "segment_summary": seed.get("segment_summary"),
        "segment_filters": seed.get("segment_filters"),
        "timeline_label": seed.get("timeline_label"),
        "timeline_index": seed.get("timeline_index"),
        "theme": seed.get("theme"),
        "focus_label": seed.get("focus_label"),
        "concern_theme": concern_label,
        "messaging_hint": (
            seed.get("messaging_hint")
            or (belief_transition or {}).get("narrative")
        ),
        "current_modality": current_modality,
        "recommended_modality": recommended_modality,
        "coverage_score": coverage_score,
        "coverage_state": coverage_state,
        "gap_score": gap_score,
        "needs_net_new": coverage_score < _MIN_COVERAGE_THRESHOLD,
        "has_strong_assets": coverage_score >= _STRONG_COVERAGE_THRESHOLD,
        "asset_options": asset_options,
        "plays_considered": len(plays),
        "no_play_data": not bool(plays),
    }


def _attach_account_interventions(
    accounts: Sequence[Dict[str, Any]],
    stats: _ModalityStats,
) -> None:
    for account in accounts:
        seeds: List[Dict[str, Any]] = []
        conversion_sequence = (account.get("execution") or {}).get("conversion_sequence") or []
        for entry in conversion_sequence:
            seeds.append(
                {
                    "scope": "account",
                    "account_id": account.get("account_id"),
                    "persona": entry.get("persona") or {},
                    "persona_label": (entry.get("persona") or {}).get("label"),
                    "persona_descriptor": entry.get("persona_descriptor"),
                    "stage_label": (
                        (entry.get("belief_transition_meta") or {}).get("stage_label")
                        or entry.get("stage_label")
                    ),
                    "belief_transition": entry.get("belief_transition"),
                    "belief_transition_meta": entry.get("belief_transition_meta"),
                    "expected_delta_bp": entry.get("expected_delta_bp"),
                    "confidence": entry.get("avg_confidence"),
                    "plays": entry.get("plays"),
                    "segment_summary": entry.get("segment_summary"),
                    "segment_filters": entry.get("segment_filters"),
                    "people": entry.get("people"),
                    "matched_people": entry.get("matched_people"),
                    "timeline_label": entry.get("timeline_label"),
                    "timeline_index": entry.get("timeline_index"),
                    "accounts": [
                        {
                            "id": account.get("account_id"),
                            "name": account.get("account_name") or account.get("account_id"),
                        }
                    ],
                    "concern_label": (entry.get("belief_transition_meta") or {}).get("pain"),
                }
            )
        interventions: List[Dict[str, Any]] = []
        for seed in seeds:
            payload = _build_intervention_from_seed(seed, stats)
            if payload:
                interventions.append(payload)
        interventions.sort(key=lambda item: item.get("gap_score", 0.0), reverse=True)
        account["interventions"] = interventions


def _build_portfolio_interventions(
    focuses: Sequence[Dict[str, Any]],
    stats: _ModalityStats,
) -> List[Dict[str, Any]]:
    interventions: List[Dict[str, Any]] = []
    for focus in focuses:
        seed = {
            "scope": "portfolio",
            "persona": focus.get("persona_meta") or {},
            "persona_label": focus.get("persona_label") or focus.get("persona_focus"),
            "persona_descriptor": focus.get("persona_descriptor"),
            "stage_label": focus.get("stage_label"),
            "belief_transition": focus.get("belief_transition"),
            "belief_transition_meta": {
                "stage_label": focus.get("stage_label"),
                "pain": (focus.get("belief_transition") or {}).get("pain", {}).get("label"),
            },
            "expected_delta_bp": focus.get("expected_delta_bp"),
            "confidence": focus.get("confidence"),
            "plays": focus.get("plays"),
            "segment_summary": focus.get("segment_summary"),
            "segment_filters": focus.get("segment_filters"),
            "people": focus.get("people"),
            "accounts": focus.get("accounts"),
            "timeline_label": focus.get("timeline_label"),
            "timeline_index": focus.get("timeline_index"),
            "theme": focus.get("campaign_theme") or focus.get("persona_focus"),
            "focus_label": focus.get("focus_label") or focus.get("label"),
            "concern_label": (
                (focus.get("belief_transition") or {}).get("pain", {}) or {}
            ).get("label"),
        }
        payload = _build_intervention_from_seed(seed, stats)
        if payload:
            interventions.append(payload)
    interventions.sort(key=lambda item: item.get("gap_score", 0.0), reverse=True)
    return interventions


# ---------------------------------------------------------------------------
# Account-level blueprint builder
# ---------------------------------------------------------------------------

def build_account_marketing_blueprint(
    product_id: str,
    account_id: str,
    *,
    product_graph: Optional[nx.DiGraph] = None,
    canonical_persona_path: Optional[List[Dict[str, Any]]] = None,
    segment_patterns: Optional[Dict[str, Any]] = None,
    shm_metrics: Optional[Dict[str, Any]] = None,
    journey_weights: Optional[Dict[str, Any]] = None,
    win_regression: Optional[Dict[str, Any]] = None,
    persona_wolves_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    persona_metrics_updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    if product_graph is None:
        product_graph = build_product_graph(product_id)
    segment_patterns = segment_patterns or {}
    shm_metrics = shm_metrics or {}
    persona_transition_stats = shm_metrics.get("persona") or {}
    journey_weights = journey_weights or {}
    win_calibrator = None
    if win_regression:
        try:
            win_calibrator = WinProbabilityCalibrator.from_summary(win_regression)
        except Exception as exc:  # pragma: no cover - calibration is best effort
            LOGGER.warning("Unable to build win regression calibrator: %s", exc)
            win_calibrator = None

    account = get_account_by_id(product_id, account_id) or {"account_name": account_id}
    thesis, _ = get_cached_belief_thesis(
        product_id=product_id,
        account_id=account_id,
        build_fn=lambda: build_belief_thesis_for_account(product_id, account_id),
    )
    if persona_wolves_metrics is None:
        persona_wolves_metrics, persona_metrics_updated_at = _load_persona_wolves_metrics(
            product_id
        )

    bgn_paths = _generate_persona_paths_from_weights(
        journey_weights,
        thesis,
        persona_transition_stats=persona_transition_stats,
    )
    if bgn_paths:
        paths = bgn_paths
    elif canonical_persona_path:
        paths = [
            {
                "id": "canonical",
                "probability": 1.0,
                "score": 1.0,
                "personas_prefab": [deepcopy(persona) for persona in canonical_persona_path],
                "raw": {"source": "canonical"},
                "source": "graph",
                "confidence": 0.6,
            }
        ]
    else:
        paths = _top_paths_from_thesis(thesis)
    if not paths:
        return {
            "account_id": account_id,
            "account_name": account.get("account_name") or account_id,
            "error": "No persona paths available",
        }

    persona_committee_probs = {
        str(pid): _safe_float(prob, 0.0)
        for pid, prob in (thesis.get("persona_committee_probs") or {}).items()
    }
    persona_posteriors_src = (
        thesis.get("persona_posteriors")
        or thesis.get("persona_committee_probs")
        or {}
    )
    persona_posteriors = {
        str(pid): _safe_float(prob, 0.0) or 0.0
        for pid, prob in persona_posteriors_src.items()
    }
    persona_belief_posteriors = thesis.get("persona_belief_posteriors") or {}
    person_committee_entries = thesis.get("person_committee_probs") or []
    person_prob_by_match: Dict[str, Dict[str, Any]] = {}
    person_prob_by_person: Dict[str, Dict[str, Any]] = {}
    for entry in person_committee_entries:
        match_id = entry.get("match_id")
        person_id = entry.get("person_id")
        if match_id:
            person_prob_by_match[str(match_id)] = entry
        if person_id:
            person_prob_by_person[str(person_id)] = entry
    person_likelihoods: List[Dict[str, Any]] = []
    for entry in person_committee_entries:
        probability = _safe_float(entry.get("probability"), 0.0) or 0.0
        persona_id = entry.get("persona_id")
        person_likelihoods.append(
            {
                "match_id": entry.get("match_id"),
                "person_id": entry.get("person_id"),
                "display_name": entry.get("person_name")
                or entry.get("display_name")
                or entry.get("notes")
                or entry.get("person_id"),
                "persona_id": persona_id,
                "persona_label": entry.get("persona_label"),
                "probability": round(probability, 6),
                "role_band": _probability_band(probability),
                "belief_level": entry.get("belief_level"),
                "phase_probs": entry.get("phase_probs"),
                "dominant_phase": entry.get("dominant_phase"),
            }
        )
    person_likelihoods.sort(key=lambda item: item.get("probability", 0.0), reverse=True)

    matches_by_persona = _load_persona_matches_lookup(product_id, account_id)
    engagement_buckets = _load_engagement_buckets(product_id, account_id)
    expected_next_probs: Dict[str, float] = {}
    expected_next_meta: Dict[str, Dict[str, Any]] = {}
    fit_score = _safe_float((thesis.get("fit") or {}).get("overall"), None)
    for entry in thesis.get("current_expected_next") or []:
        persona_key = entry.get("persona")
        if not persona_key:
            continue
        pid = str(persona_key)
        prob_val = float(entry.get("prob") or 0.0)
        expected_next_probs[pid] = prob_val
        expected_next_meta[pid] = {
            "prob": prob_val,
            "prob_base": _safe_float(entry.get("prob_base"), None),
            "subsidy_lift": _safe_float(entry.get("subsidy_lift"), None),
            "persona_label": entry.get("persona_label"),
            "journey_stage": entry.get("journey_stage"),
            "reason": entry.get("reason"),
        }

    assets = _load_assets(product_id)
    channels = _load_channels(product_id)
    impacts = _ARSENAL_IMPACT_INDEX.get(
        product_id, {"by_pair": {}, "by_persona": {}}
    )

    zmot_forecasts = thesis.get("zmot_forecasts") or []
    zmot_by_persona: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in zmot_forecasts:
        pid = str(entry.get("persona_id") or "")
        if pid:
            zmot_by_persona[pid].append(entry)

    persona_paths = []
    scatter_personas: List[Dict[str, Any]] = []
    transitions_all: List[Dict[str, Any]] = []
    plays: List[Dict[str, Any]] = []
    persona_lookup: Dict[str, Dict[str, Any]] = {}
    ordered_persona_ids: List[str] = []
    plays_by_persona: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    account_wolf_scores = thesis.get("persona_wolf_scores") or {}

    account_meta_detail = {
        key: account.get(key)
        for key in [
            "industry",
            "revenue_range",
            "employee_range",
            "funding_stage",
            "geography",
            "competitor_used",
            "other_tech_stack",
        ]
        if account.get(key) not in (None, [], "")
    }
    account_segment_map = _account_segment_map(account_meta_detail)
    account_segment_labels = _account_segment_strings(account_meta_detail)
    account_meta_payload = {
        "account_id": account_id,
        "account_name": account.get("account_name") or account_id,
        "deal_status": account.get("deal_status"),
        "attributes": account_meta_detail,
    }
    segment_context = _build_segment_context(account_meta_detail, segment_patterns)
    account_segment_pattern = segment_context.get("pattern")

    for path_index, path in enumerate(paths):
        prefab_personas = path.get("personas_prefab")
        if prefab_personas:
            persona_summaries = [deepcopy(persona) for persona in prefab_personas]
        else:
            persona_summaries = _collect_persona_metrics(
                product_graph,
                path,
                expected_next_probs=expected_next_probs,
                expected_next_meta=expected_next_meta,
                belief_posteriors=persona_belief_posteriors,
                wolves_metrics=persona_wolves_metrics,
                account_wolf_scores=account_wolf_scores,
            )
            persona_summaries = _prioritize_personas_for_path(persona_summaries)
            if account_segment_pattern:
                boost_map = _segment_priority_boost_map(account_segment_pattern)
                if boost_map:
                    for persona in persona_summaries:
                        multiplier = boost_map.get(persona.get("id"))
                        if multiplier:
                            persona["segment_priority_multiplier"] = multiplier
                            persona["priority_score"] = _persona_priority_score(persona)
        path_signal = _persona_path_signal(persona_summaries)
        for persona in persona_summaries:
            persona_id = persona.get("id")
            matched_people = matches_by_persona.get(persona_id, []) if persona_id else []
            persona["matched_people"] = matched_people
            persona["matched_people_count"] = len(matched_people)
            persona["has_person_match"] = bool(matched_people)
            people_names = [
                entry.get("display_name")
                or entry.get("person_name")
                or entry.get("notes")
                or entry.get("person_id")
                for entry in matched_people
            ]
            persona["people_names"] = [name for name in people_names if name]
            if persona_id:
                persona_lookup[persona_id] = persona
                if persona_id not in ordered_persona_ids:
                    ordered_persona_ids.append(persona_id)
                committee_prob = persona_committee_probs.get(persona_id)
                if committee_prob is not None:
                    persona["expected_in_deal_prob"] = round(committee_prob, 6)
                    persona["expected_in_deal_pct"] = round(
                        min(max(committee_prob, 0.0), 1.0) * 100.0, 2
                    )
                zmot_events = zmot_by_persona.get(persona_id, [])
                if zmot_events:
                    top_events = sorted(
                        zmot_events,
                        key=lambda row: (
                            float(row.get("boost") or 0.0),
                            float(row.get("trigger_boost") or 0.0),
                        ),
                        reverse=True,
                    )[:3]
                    persona["zmot_events"] = [
                        _format_zmot_entry(entry) for entry in top_events
                    ]
                else:
                    persona["zmot_events"] = []
                engagement_snapshot = _persona_engagement_snapshot(
                    persona.get("label"),
                    matched_people,
                    engagement_buckets,
                )
                if engagement_snapshot:
                    persona["engagement_snapshot"] = engagement_snapshot
                    snapshot_fatigue = _safe_float(
                        engagement_snapshot.get("fatigue"), None
                    )
                    if snapshot_fatigue is not None:
                        persona["fatigue"] = snapshot_fatigue
                    if engagement_snapshot.get("fatigue_reason"):
                        persona["fatigue_reason"] = engagement_snapshot["fatigue_reason"]
                else:
                    persona["engagement_snapshot"] = None
                belief_level = _safe_float(persona.get("belief_level"), 0.0) or 0.0
                priority_score = _safe_float(persona.get("priority_score"), 0.0) or 0.0
                expected_next = _safe_float(
                    persona.get("expected_next_prob"), 0.0
                ) or 0.0
                base_person_signal = max(
                    0.05, 0.5 * belief_level + 0.3 * priority_score + 0.2 * expected_next
                )
                for match in matched_people:
                    probability_entry = None
                    match_id = match.get("match_id")
                    person_id = match.get("person_id")
                    if match_id and match_id in person_prob_by_match:
                        probability_entry = person_prob_by_match[match_id]
                    elif person_id and person_id in person_prob_by_person:
                        probability_entry = person_prob_by_person[person_id]
                    if probability_entry:
                        person_prob = _safe_float(
                            probability_entry.get("probability"), 0.0
                        ) or 0.0
                        person_prob = max(0.0, min(1.0, person_prob))
                        belief_override = _safe_float(
                            probability_entry.get("belief_level"), person_prob
                        ) or person_prob
                        person_involvement = round(person_prob, 4)
                        match["person_involvement_score"] = person_involvement
                        match["committee_probability"] = person_involvement
                        match["person_belief_level"] = round(belief_override, 4)
                        match["belief_phase_probs"] = probability_entry.get(
                            "phase_probs"
                        )
                        match["dominant_phase"] = probability_entry.get(
                            "dominant_phase"
                        )
                        match["role_band"] = _probability_band(person_prob)
                    else:
                        confidence = _safe_float(match.get("match_confidence"), 0.5) or 0.5
                        engagement_factor = 1.0
                        engagement_count = match.get("engagement_count")
                        if isinstance(engagement_count, (int, float)):
                            engagement_factor += (
                                min(max(float(engagement_count), 0.0), 8.0) * 0.03
                            )
                        raw_signal = (
                            base_person_signal
                            * (0.5 + 0.5 * confidence)
                            * engagement_factor
                        )
                        bounded_signal = max(0.0, min(1.0, raw_signal))
                        person_involvement = round(bounded_signal, 4)
                        match["person_involvement_score"] = person_involvement
                        match["person_belief_level"] = round(
                            belief_level * (0.5 + 0.5 * confidence),
                            4,
                        )
                        match["committee_probability"] = person_involvement
                        match["belief_phase_probs"] = None
                        match["dominant_phase"] = None
                        match["role_band"] = _probability_band(bounded_signal)
                fallback_fatigue = None
                fallback_reason = None
                if matched_people:
                    fallback_fatigue, fallback_reason = _persona_fatigue_from_matches(
                        matched_people
                    )
                existing_fatigue = None
                if persona.get("fatigue") is not None:
                    existing_fatigue = _safe_float(persona.get("fatigue"), None)
                if fallback_fatigue is not None:
                    if existing_fatigue is None or fallback_fatigue > existing_fatigue:
                        persona["fatigue"] = fallback_fatigue
                        if fallback_reason:
                            persona["fatigue_reason"] = fallback_reason
                if persona.get("fatigue") is None:
                    persona["fatigue"] = 0.05
                if matched_people:
                    ranked_people = sorted(
                        matched_people,
                        key=lambda item: _safe_float(
                            item.get("committee_probability")
                            or item.get("person_involvement_score")
                            or item.get("person_belief_level"),
                            0.0,
                        )
                        or 0.0,
                        reverse=True,
                    )[:3]
                    persona["top_people"] = [
                        {
                            "person_id": item.get("person_id"),
                            "display_name": item.get("display_name")
                            or item.get("person_name")
                            or item.get("notes"),
                            "person_involvement_score": item.get(
                                "person_involvement_score"
                            ),
                            "person_belief_level": item.get("person_belief_level"),
                            "committee_probability": item.get("committee_probability"),
                            "belief_phase_probs": item.get("belief_phase_probs"),
                            "dominant_phase": item.get("dominant_phase"),
                            "role_band": item.get("role_band")
                            or _probability_band(
                                _safe_float(
                                    item.get("committee_probability")
                                    or item.get("person_involvement_score"),
                                    0.0,
                                )
                                or 0.0
                            ),
                        }
                        for item in ranked_people
                    ]
                else:
                    persona["top_people"] = []
                persona["priority_score"] = _persona_priority_score(persona)
            else:
                persona["zmot_events"] = []
                persona["engagement_snapshot"] = None
                persona["top_people"] = []
        scatter_personas.extend(persona_summaries)
        persona_paths.append(
            {
                "id": path.get("id") or f"path:{path_index}",
                "probability": path.get("probability"),
                "score": path.get("score"),
                "personas": persona_summaries,
                "raw": path.get("raw"),
                "is_primary": False,
                "path_signal": path_signal,
            }
        )

        for stage_index, persona in enumerate(persona_summaries):
            transition_items = _persona_transitions(
                product_graph,
                persona,
                stage_index=stage_index,
            )
            transitions_all.extend(transition_items)
            exploration = _exploration_weight(path_index, stage_index, len(paths))

            combos = _recommend_assets_for_transition(
                transition_items[0] if transition_items else {
                    "persona": persona,
                    "stage_index": stage_index,
                    "belief_transition": {
                        "problem": persona.get("label"),
                        "execution": persona.get("label"),
                        "pain": None,
                        "resolution": None,
                    },
                    "pains": [],
                    "capabilities": [],
                },
                persona=persona,
                assets=assets,
                channels=channels,
                impacts=impacts,
                path_probability=path.get("probability", 0.0),
                exploration_weight=exploration,
                account_segments=account_segment_map,
                account_segment_labels=account_segment_labels,
                account_id=account_id,
                win_calibrator=win_calibrator,
            )

            for combo in combos:
                combo["path_id"] = path.get("id") or f"path:{path_index}"
                combo["path_probability"] = path.get("probability", 0.0)
                combo["path_index"] = path_index
                combo["stage_index"] = stage_index
                combo["exploration_weight"] = exploration
                combo["account_id"] = account_id
                combo["account_name"] = account.get("account_name") or account_id
                matched_people = persona.get("matched_people") or []
                people_names = persona.get("people_names") or []
                combo["people"] = matched_people
                combo["people_names"] = people_names
                combo["has_person_match"] = bool(matched_people)
                combo["primary_person"] = people_names[0] if people_names else None
                plays.append(combo)
                pid = combo.get("persona_id")
                if pid:
                    plays_by_persona[pid].append(combo)

    entry_points = _apply_entry_point_scores(
        persona_lookup=persona_lookup,
        plays_by_persona=plays_by_persona,
        ordered_persona_ids=ordered_persona_ids,
    )
    thesis["entry_points"] = entry_points

    persona_paths.sort(
        key=lambda row: (
            _safe_float(row.get("path_signal"), 0.0),
            _safe_float(row.get("probability"), 0.0),
        ),
        reverse=True,
    )
    for idx, path in enumerate(persona_paths):
        path["is_primary"] = idx == 0

    primary_personas = (
        persona_paths[0].get("personas") if persona_paths else []
    )
    if primary_personas is None:
        primary_personas = []
    total_persona_stages = len(primary_personas) or max(
        (len(path.get("personas") or []) for path in persona_paths),
        default=1,
    )
    persona_engagements = [
        _persona_engagement_plan(
            persona,
            plays_by_persona,
            total_persona_stages,
            fit_score=fit_score,
        )
        for persona in primary_personas
    ]

    persona_likelihoods: List[Dict[str, Any]] = []
    for persona_id, probability in sorted(
        persona_posteriors.items(), key=lambda kv: kv[1], reverse=True
    ):
        persona_info = persona_lookup.get(persona_id, {})
        label = persona_info.get("label")
        if not label:
            label = _title_case_value(str(persona_id).replace("|", " ")) or persona_id
        persona_likelihoods.append(
            {
                "persona_id": persona_id,
                "persona_label": label,
                "probability": round(probability, 6),
                "band": _probability_band(probability),
                "belief_level": persona_info.get("belief_level"),
                "belief_band": _belief_band(persona_info.get("belief_level")),
                "phase_probs": persona_info.get("phase_probs"),
                "dominant_phase": persona_info.get("dominant_phase"),
                "journey_phase": persona_info.get("journey_phase"),
                "path_probability": persona_info.get("path_probability"),
                "committee_probability": persona_committee_probs.get(persona_id),
                "top_people": persona_info.get("top_people") or [],
            }
        )

    expected_next_personas: List[Dict[str, Any]] = []
    seen_expected: set[str] = set()
    for entry in thesis.get("current_expected_next") or []:
        persona_id = str(entry.get("persona") or "")
        if not persona_id:
            continue
        prob = _safe_float(entry.get("prob"), 0.0) or 0.0
        prob_base = _safe_float(entry.get("prob_base"), None)
        subsidy_lift = _safe_float(entry.get("subsidy_lift"), None)
        persona_info = persona_lookup.get(persona_id, {})
        label = (
            entry.get("persona_label")
            or persona_info.get("label")
            or _title_case_value(persona_id.replace("|", " "))
            or persona_id
        )
        expected_next_personas.append(
            {
                "persona_id": persona_id,
                "persona_label": label,
                "probability": round(prob, 6),
                "band": _probability_band(prob),
                "journey_stage": entry.get("journey_stage")
                or persona_info.get("journey_phase"),
                "reason": entry.get("reason") or entry.get("label"),
                "top_people": persona_info.get("top_people") or [],
                "prob_base": prob_base,
                "subsidy_lift": subsidy_lift,
            }
        )
        seen_expected.add(persona_id)

    if expected_next_probs:
        for persona_id, prob in sorted(
            expected_next_probs.items(), key=lambda kv: kv[1], reverse=True
        ):
            if persona_id in seen_expected:
                continue
            persona_info = persona_lookup.get(persona_id, {})
            label = persona_info.get("label") or _title_case_value(
                str(persona_id).replace("|", " ")
            ) or persona_id
            expected_next_personas.append(
                {
                    "persona_id": persona_id,
                    "persona_label": label,
                    "probability": round(prob, 6),
                    "band": _probability_band(prob),
                    "journey_stage": persona_info.get("journey_phase"),
                    "reason": "Predicted by persona journey model",
                    "prob_base": None,
                    "subsidy_lift": None,
                    "top_people": persona_info.get("top_people") or [],
                }
            )

    expected_next_people: List[Dict[str, Any]] = []
    for persona_entry in expected_next_personas:
        for person in persona_entry.get("top_people") or []:
            if not (person.get("person_id") or person.get("display_name")):
                continue
            expected_next_people.append(
                {
                    "persona_id": persona_entry["persona_id"],
                    "persona_label": persona_entry["persona_label"],
                    "person_id": person.get("person_id"),
                    "display_name": person.get("display_name"),
                    "committee_probability": person.get("committee_probability"),
                    "role_band": person.get("role_band")
                    or _probability_band(
                        _safe_float(person.get("committee_probability"), 0.0) or 0.0
                    ),
                }
            )
        if len(expected_next_people) >= 8:
            break
    expected_next_people = expected_next_people[:8]

    zmot_watchlist: List[Dict[str, Any]] = []
    unique_zmot_events: Dict[str, Dict[str, Any]] = {}
    for persona_id in ordered_persona_ids:
        events_raw = zmot_by_persona.get(persona_id, [])
        if not events_raw:
            continue
        ranked = sorted(
            events_raw,
            key=lambda row: (
                float(row.get("boost") or 0.0),
                float(row.get("trigger_boost") or 0.0),
            ),
            reverse=True,
        )
        formatted_events = [_format_zmot_entry(item) for item in ranked[:5] if item]
        if not formatted_events:
            continue
        persona_info = persona_lookup.get(persona_id, {})
        persona_label = (
            persona_info.get("label")
            or (events_raw[0].get("persona_label") if events_raw else None)
            or persona_id
        )
        zmot_watchlist.append(
            {
                "persona_id": persona_id,
                "persona_label": persona_label,
                "events": formatted_events,
            }
        )
        for event in formatted_events:
            zmot_id = event.get("zmot_event_id")
            if not zmot_id:
                continue
            aggregate = unique_zmot_events.setdefault(
                zmot_id,
                {
                    "zmot_event_id": zmot_id,
                    "zmot_label": event.get("zmot_label"),
                    "pain_trigger_id": event.get("pain_trigger_id"),
                    "pain_trigger_label": event.get("pain_trigger_label"),
                    "pain_id": event.get("pain_id"),
                    "pain_label": event.get("pain_label"),
                    "personas": [],
                    "observable_moments": event.get("observable_moments"),
                    "keywords": event.get("keywords"),
                    "boost": event.get("boost"),
                    "trigger_boost": event.get("trigger_boost"),
                },
            )
            persona_list = aggregate.setdefault("personas", [])
            if not any(p.get("persona_id") == persona_id for p in persona_list):
                persona_list.append(
                    {
                        "persona_id": persona_id,
                        "persona_label": persona_label,
                        "already_observed": event.get("already_observed", False),
                    }
                )
            aggregate["boost"] = max(
                _safe_float(event.get("boost"), 0.0),
                _safe_float(aggregate.get("boost"), 0.0),
            )
            aggregate["trigger_boost"] = max(
                _safe_float(event.get("trigger_boost"), 0.0),
                _safe_float(aggregate.get("trigger_boost"), 0.0),
            )

    zmot_event_portfolio = sorted(
        unique_zmot_events.values(),
        key=lambda row: (
            _safe_float(row.get("boost"), 0.0),
            _safe_float(row.get("trigger_boost"), 0.0),
        ),
        reverse=True,
    )

    enrichment_requirements, enrichment_summary = _compute_person_enrichment_requirements(
        persona_paths,
        matches_by_persona,
    )
    unmatched_personas = [
        requirement
        for requirement in enrichment_requirements
        if not requirement.get("has_match")
    ]

    _schedule_plays(plays)
    _annotate_execution_layers(plays, persona_lookup, account_meta_payload)
    campaigns = _group_campaigns(plays)
    randomization = _randomization_policy(paths, plays)
    conversion_sequence = _build_conversion_sequence(
        plays,
        persona_lookup,
        account_meta_payload,
    )
    if primary_personas:
        existing_persona_ids = {
            entry.get("persona", {}).get("id")
            for entry in conversion_sequence
            if entry.get("persona")
        }
        fallback_entries: List[Dict[str, Any]] = []
        next_index = (
            max((entry.get("timeline_index", 0) for entry in conversion_sequence), default=-1)
            + 1
        )
        for persona in primary_personas:
            pid = persona.get("id")
            if not pid or pid in existing_persona_ids:
                continue
            fallback_entries.append(
                {
                    "timeline_index": next_index,
                    "timeline_label": f"T+{next_index}",
                    "timeline_days": next_index * 3,
                    "persona": {
                        "id": pid,
                        "label": persona.get("label"),
                        "perceptibility": persona.get("perceptibility"),
                        "proximity": persona.get("proximity"),
                        "involvement": persona.get("involvement"),
                        "priority_rank": persona.get("priority_rank"),
                        "priority_score": persona.get("priority_score"),
                        "title": persona.get("title"),
                        "department": persona.get("department"),
                        "seniority": persona.get("seniority"),
                        "wolves_score": persona.get("wolves_score"),
                        "wolves_delta_bp": persona.get("wolves_delta_bp"),
                        "wolves_involvement_rate": persona.get("wolves_involvement_rate"),
                        "wolves_blocker_rate": persona.get("wolves_blocker_rate"),
                        "wolves_sample_size": persona.get("wolves_sample_size"),
                        "is_new_persona": persona.get("is_new_persona"),
                        "persona_source": persona.get("persona_source"),
                    },
                    "belief_transition": {
                        "persona": {"id": pid, "label": persona.get("label"), "type": "persona"},
                        "narrative": persona.get("highlight") or "Canonical persona step",
                    },
                    "expected_delta_bp": 0.0,
                    "confidence_sum": 0.0,
                    "effort_weighted": 0.0,
                    "effort_denominator": 0.0,
                    "plays": [],
                    "account": account_meta_payload,
                    "persona_descriptor": _persona_descriptor_from_meta(
                        {"label": persona.get("label"), "title": persona.get("title")},
                        persona.get("label"),
                    ),
                    "segment_summary": None,
                }
            )
            next_index += 1
        if fallback_entries:
            conversion_sequence.extend(fallback_entries)

    activity_story = build_activity_story(thesis)
    storyline = compose_storyline(
        account_name=account_meta_payload.get("account_name") or account_id,
        account_meta=account_meta_payload.get("attributes"),
        deal_status=account.get("deal_status"),
        journey_steps=(thesis.get("journey") or {}).get("steps") or [],
        activity_story=activity_story,
        conversion_sequence=conversion_sequence,
        zmot_event=zmot_event_portfolio[0] if zmot_event_portfolio else None,
        product_id=product_id,
    )

    thesis_payload = {
        "persona_likelihoods": persona_likelihoods,
        "person_likelihoods": person_likelihoods,
        "expected_next_personas": expected_next_personas,
        "expected_next_people": expected_next_people,
        "persona_posteriors": persona_posteriors,
        "persona_committee_probs": persona_committee_probs,
        "persona_belief_posteriors": persona_belief_posteriors,
        "entry_points": entry_points,
    }

    account_execution_interventions = _build_account_execution_interventions(
        conversion_sequence,
        account_id=account_id,
        account_name=account.get("account_name") or account_id,
        segment_context=segment_context,
    )

    return {
        "account_id": account_id,
        "account_name": account.get("account_name") or account_id,
        "deal_status": account.get("deal_status"),
        "meta": account_meta_detail,
        "segment": segment_context,
        "keystone_personas": _extract_keystone_personas(persona_lookup),
        "wolves_metrics_updated_at": persona_metrics_updated_at,
        "prediction": {
            "persona_paths": persona_paths,
            "fit": thesis.get("fit") or {},
            "expected_next": thesis.get("current_expected_next"),
            "expected_next_personas": expected_next_personas,
            "observed_personas": thesis.get("observed_persona_labels") or [],
            "journey": {
                "steps": [
                    {
                        "t": step.get("t"),
                        "bucket": step.get("bucket"),
                        "predicted": list(step.get("predicted_topK") or [])[:3],
                        "observed": step.get("observed_next"),
                        "win_likelihood": _safe_float(step.get("win_likelihood")),
                        "hit_at_1": bool(step.get("hit_at_1")),
                        "hit_at_3": bool(step.get("hit_at_3")),
                    }
                    for step in (thesis.get("journey") or {}).get("steps", [])[:16]
                ],
                "total_steps": len((thesis.get("journey") or {}).get("steps", [])),
            },
        },
        "entry_points": entry_points,
        "transitions": transitions_all,
        "scatter": {"personas": scatter_personas},
        "execution": {
            "plays": plays,
            "conversion_sequence": conversion_sequence,
            "persona_engagements": persona_engagements,
            "asset_cadence": _build_asset_cadence_table(
                campaigns,
                total_persona_stages,
            ),
            "interventions": account_execution_interventions,
        },
        "storyline": storyline,
        "campaigns": campaigns,
        "zmot": {
            "watchlist": zmot_watchlist,
            "events": zmot_event_portfolio,
        },
        "thesis": thesis_payload,
        "learning": thesis.get("learning_summary") or {},
        "randomization": randomization,
        "enrichment": {
            "persona_requirements": enrichment_requirements,
            "summary": enrichment_summary,
            "unmatched_personas": unmatched_personas,
        },
    }


# ---------------------------------------------------------------------------
# Portfolio aggregation
# ---------------------------------------------------------------------------

def _aggregate_portfolio(accounts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    theme_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    focus_groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    broad_total = 0
    focus_total = 0
    scatter: List[Dict[str, Any]] = []
    path_probabilities: List[float] = []
    account_names: Dict[str, str] = {}

    for account in accounts:
        account_id = account.get("account_id")
        account_name = account.get("account_name") or account_id
        if account_id:
            account_names[account_id] = account_name

        scatter.extend(account.get("scatter", {}).get("personas") or [])
        for path in account.get("prediction", {}).get("persona_paths", []) or []:
            if path.get("probability") is not None:
                path_probabilities.append(path["probability"])

        for play in account.get("execution", {}).get("plays", []) or []:
            if play.get("mode") == "broad":
                broad_total += 1
            else:
                focus_total += 1

        for campaign in account.get("campaigns") or []:
            key = (campaign.get("quarter") or "Q1", campaign.get("theme") or "")
            group = theme_groups.setdefault(
                key,
                {
                    "quarter": key[0],
                    "theme": key[1],
                    "accounts": set(),
                    "total_delta_bp": 0.0,
                    "confidence_sum": 0.0,
                    "plays": [],
                    "focus_personas": set(),
                    "focus_pains": set(),
                    "focus_people": set(),
                    "focus_keys": set(),
                    "effort_sum": 0.0,
                    "effort_count": 0,
                    "segment_buffer": [],
                    "persona_descriptors": Counter(),
                },
            )
            group["accounts"].add((account_id, account_name))
            group["total_delta_bp"] += campaign.get("total_delta_bp", 0.0)
            group["confidence_sum"] += campaign.get("confidence", 0.0)
            group["plays"].extend(campaign.get("plays") or [])
            group["focus_personas"].update(campaign.get("focus_personas") or [])
            group["focus_pains"].update(campaign.get("focus_pains") or [])
            group["focus_people"].update(campaign.get("focus_people") or [])
            effort_val = campaign.get("avg_effort_required")
            if effort_val is not None:
                group["effort_sum"] += float(effort_val)
                group["effort_count"] += 1
            segment_filters = campaign.get("segment_filters")
            if segment_filters:
                group["segment_buffer"].append(segment_filters)
            for item in campaign.get("persona_descriptor_counts") or []:
                descriptor = item.get("descriptor")
                count = int(item.get("count") or 0)
                if descriptor:
                    group["persona_descriptors"][descriptor] += max(1, count)

            for focus in campaign.get("conversion_focuses") or []:
                focus_label = focus.get("focus_label") or focus.get("label") or "Focus"
                stage_label = focus.get("stage_label") or ""
                persona_focus = focus.get("persona_focus") or ""
                focus_key = (focus_label, stage_label, persona_focus)
                group["focus_keys"].add(focus_key)

                focus_group = focus_groups.setdefault(
                    focus_key,
                    {
                        "label": focus.get("label"),
                        "focus_label": focus_label,
                        "persona_focus": persona_focus,
                        "stage_label": stage_label,
                        "expected_delta_bp": 0.0,
                        "confidence_sum": 0.0,
                        "entry_count": 0,
                        "accounts": set(),
                        "people": set(),
                        "plays": [],
                        "avg_duration_values": [],
                        "campaign_themes": Counter(),
                        "effort_sum": 0.0,
                        "timeline_indices": [],
                        "segment_buffer": [],
                        "persona_descriptors": Counter(),
                        "persona_meta": focus.get("persona_meta") or {},
                        "belief_conversion_values": [],
                        "time_to_effect_values": [],
                    },
                )
                focus_group["expected_delta_bp"] += float(
                    focus.get("expected_delta_bp") or 0.0
                )
                focus_group["confidence_sum"] += float(focus.get("confidence") or 0.0)
                focus_group["entry_count"] += 1
                focus_group["avg_duration_values"].append(
                    float(focus.get("avg_duration_days") or 14.0)
                )
                focus_group["effort_sum"] += float(
                    focus.get("avg_effort_required") or 0.0
                )
                focus_group["people"].update(focus.get("people") or [])
                for acc in focus.get("accounts") or []:
                    acc_id = acc.get("id")
                    acc_name = acc.get("name") or account_names.get(acc_id) or acc_id
                    if acc_id:
                        focus_group["accounts"].add((acc_id, acc_name))
                if account_id:
                    focus_group["accounts"].add((account_id, account_name))
                focus_plays = focus.get("plays") or []
                focus_group["plays"].extend(focus_plays)
                focus_group["timeline_indices"].extend(
                    [
                        play.get("timeline_index")
                        for play in focus_plays
                        if play.get("timeline_index") is not None
                    ]
                )
                focus_group["campaign_themes"][campaign.get("theme") or ""] += 1
                segment_filters = focus.get("segment_filters")
                if segment_filters:
                    focus_group["segment_buffer"].append(segment_filters)
                for item in focus.get("persona_descriptor_counts") or []:
                    descriptor = item.get("descriptor")
                    count = int(item.get("count") or 0)
                    if descriptor:
                        focus_group["persona_descriptors"][descriptor] += max(1, count)
                if not focus_group["persona_meta"] and focus.get("persona_meta"):
                    focus_group["persona_meta"] = focus.get("persona_meta")
                belief_conv = focus.get("avg_belief_conversion")
                if belief_conv is not None:
                    focus_group["belief_conversion_values"].append(float(belief_conv))
                time_to_effect = focus.get("time_to_effect_days")
                if time_to_effect is not None:
                    focus_group["time_to_effect_values"].append(float(time_to_effect))

    focus_key_to_id: Dict[Tuple[str, str, str], str] = {
        focus_key: f"focus-{idx}"
        for idx, focus_key in enumerate(focus_groups.keys())
    }

    portfolio_focuses: List[Dict[str, Any]] = []
    for focus_key, data in focus_groups.items():
        focus_id = focus_key_to_id[focus_key]
        accounts = sorted(
            [
                {"id": acc_id, "name": acc_name or account_names.get(acc_id) or acc_id}
                for acc_id, acc_name in data["accounts"]
                if acc_id
            ],
            key=lambda item: item["name"],
        )
        avg_confidence = (
            data["confidence_sum"] / data["entry_count"]
            if data["entry_count"]
            else 0.0
        )
        avg_duration = _safe_mean(data["avg_duration_values"]) or 14.0
        asset_rows = _aggregate_asset_rows(data["plays"]) if data["plays"] else []
        dominant_theme = None
        if data["campaign_themes"]:
            dominant_theme = data["campaign_themes"].most_common(1)[0][0]
        coverage_ratio = None
        if account_names:
            coverage_ratio = len(accounts) / len(account_names)
        avg_effort_required = (
            data["effort_sum"] / data["entry_count"] if data["entry_count"] else 0.0
        )
        timeline_indices = [
            idx for idx in data.get("timeline_indices", []) if idx is not None
        ]
        focus_timeline_index = min(timeline_indices) if timeline_indices else None

        segment_filters = _aggregate_segment_filters(
            data.get("segment_buffer", [])
        )
        segment_summary = _segment_summary_text(segment_filters)
        persona_descriptors_counter = data.get("persona_descriptors") or Counter()
        if not persona_descriptors_counter:
            fallback_descriptor = _persona_descriptor_from_meta(
                data.get("persona_meta") or {},
                data.get("focus_label"),
            )
            persona_descriptors_counter = Counter({fallback_descriptor: 1})
        persona_descriptor_counts = [
            {"descriptor": desc, "count": cnt}
            for desc, cnt in persona_descriptors_counter.most_common()
        ]
        primary_descriptor = persona_descriptor_counts[0]["descriptor"]
        avg_belief_conversion = _safe_mean(data.get("belief_conversion_values") or [])
        time_to_effect_days = _safe_mean(data.get("time_to_effect_values") or [])
        if time_to_effect_days is None:
            time_to_effect_days = avg_duration
        expected_outcome_summary = _build_expected_outcome_summary(
            primary_descriptor,
            data.get("stage_label"),
            data.get("expected_delta_bp"),
            len(accounts),
            segment_filters,
        )

        portfolio_focuses.append(
            {
                "key": focus_id,
                "label": data.get("label") or data.get("focus_label"),
                "focus_label": data.get("focus_label"),
                "persona_focus": data.get("persona_focus"),
                "stage_label": data.get("stage_label"),
                "campaign_theme": dominant_theme or data.get("persona_focus"),
                "expected_delta_bp": data.get("expected_delta_bp", 0.0),
                "confidence": avg_confidence,
                "accounts": accounts,
                "account_count": len(accounts),
                "people": sorted(data.get("people") or []),
                "asset_table": asset_rows,
                "avg_duration_days": avg_duration,
                "coverage_ratio": coverage_ratio,
                "avg_effort_required": avg_effort_required,
                "avg_effort_pct": int(
                    round(min(0.95, max(0.1, avg_effort_required)) * 100)
                ),
                "avg_belief_conversion": avg_belief_conversion,
                "time_to_effect_days": int(round(time_to_effect_days)),
                "timeline_index": focus_timeline_index,
                "timeline_label": (
                    f"T+{focus_timeline_index}"
                    if focus_timeline_index is not None
                    else None
                ),
                "segment_filters": segment_filters,
                "segment_summary": segment_summary,
                "persona_descriptor": primary_descriptor,
                "persona_descriptor_counts": persona_descriptor_counts,
                "expected_outcome_summary": expected_outcome_summary,
            }
        )

    portfolio_focuses.sort(
        key=lambda focus: focus.get("avg_duration_days") or 0.0
    )

    campaign_themes = []
    quarter_buckets: Dict[str, Dict[str, Any]] = {}
    for group in theme_groups.values():
        play_count = len(group["plays"])
        accounts = sorted(
            [
                {"id": acc_id, "name": acc_name or account_names.get(acc_id) or acc_id}
                for acc_id, acc_name in group["accounts"]
                if acc_id
                ],
                key=lambda item: item["name"],
            )
        quarter_bucket = quarter_buckets.setdefault(
            group["quarter"],
            {
                "delta": 0.0,
                "descriptors": Counter(),
                "segments": [],
                "stage_counter": Counter(),
                "accounts": set(),
            },
        )
        quarter_bucket["delta"] += group["total_delta_bp"]
        quarter_bucket["descriptors"].update(group.get("persona_descriptors", Counter()))
        quarter_bucket["segments"].extend(group.get("segment_buffer", []))
        for _, stage_label, _ in group.get("focus_keys", set()):
            if stage_label:
                quarter_bucket["stage_counter"][stage_label] += 1
        quarter_bucket["accounts"].update(group["accounts"])
        segment_filters = _aggregate_segment_filters(group.get("segment_buffer", []))
        segment_summary = _segment_summary_text(segment_filters)
        persona_descriptor_counts = [
            {"descriptor": desc, "count": cnt}
            for desc, cnt in group["persona_descriptors"].most_common()
        ]
        primary_descriptor = (
            persona_descriptor_counts[0]["descriptor"]
            if persona_descriptor_counts
            else (group.get("persona_focus") or "Target personas")
        )
        stage_counts = Counter()
        for _, stage_label, _ in group.get("focus_keys", set()):
            if stage_label:
                stage_counts[stage_label] += 1
        primary_stage_label = stage_counts.most_common(1)[0][0] if stage_counts else None
        expected_outcome_summary = _build_expected_outcome_summary(
            primary_descriptor,
            primary_stage_label,
            group["total_delta_bp"],
            len(accounts),
            segment_filters,
        )
        campaign_themes.append(
            {
                "quarter": group["quarter"],
                "theme": group["theme"],
                "accounts": accounts,
                "focus_personas": sorted([p for p in group["focus_personas"] if p]),
                "focus_pains": sorted([p for p in group["focus_pains"] if p]),
                "focus_people": sorted([p for p in group["focus_people"] if p]),
                "total_delta_bp": group["total_delta_bp"],
                "avg_confidence": group["confidence_sum"] / play_count if play_count else 0.0,
                "play_count": play_count,
                "avg_effort_required": (
                    group["effort_sum"] / group["effort_count"]
                    if group["effort_count"]
                    else 0.0
                ),
                "avg_effort_pct": int(
                    round(
                        min(
                            0.95,
                            max(
                                0.1,
                                (
                                    group["effort_sum"] / group["effort_count"]
                                    if group["effort_count"]
                                    else 0.0
                                ),
                            ),
                        )
                        * 100
                    )
                ),
                "segment_filters": segment_filters,
                "segment_summary": segment_summary,
                "persona_descriptor_counts": persona_descriptor_counts,
                "expected_outcome_summary": expected_outcome_summary,
                "focus_keys": [
                    focus_key_to_id.get(key)
                    for key in sorted(group["focus_keys"])
                    if focus_key_to_id.get(key)
                ],
            }
        )

    campaign_themes.sort(
        key=lambda c: (
            int("".join(ch for ch in str(c["quarter"]) if ch.isdigit()) or "0"),
            -c["total_delta_bp"],
        )
    )

    global_descriptor_counter: Counter[str] = Counter()
    global_stage_counter: Counter[str] = Counter()
    for group in theme_groups.values():
        global_descriptor_counter.update(group.get("persona_descriptors", Counter()))
        for _, stage_label, _ in group.get("focus_keys", set()):
            if stage_label:
                global_stage_counter[stage_label] += 1

    global_segment_filters = _aggregate_segment_filters(
        [account.get("meta") or {} for account in accounts]
    )
    global_persona_descriptor_counts = [
        {"descriptor": desc, "count": cnt}
        for desc, cnt in global_descriptor_counter.most_common()
    ]
    global_primary_descriptor = (
        global_persona_descriptor_counts[0]["descriptor"]
        if global_persona_descriptor_counts
        else "Target personas"
    )
    global_primary_stage = (
        global_stage_counter.most_common(1)[0][0] if global_stage_counter else None
    )
    total_delta_global = sum(group["total_delta_bp"] for group in theme_groups.values())
    global_expected_outcome_summary = _build_expected_outcome_summary(
        global_primary_descriptor,
        global_primary_stage,
        total_delta_global,
        len(account_names),
        global_segment_filters,
    )

    quarterly_expected_outcomes: List[Dict[str, Any]] = []
    for quarter, bucket in quarter_buckets.items():
        quarter_segment_filters = _aggregate_segment_filters(bucket["segments"])
        quarter_descriptor_counts = [
            {"descriptor": desc, "count": cnt}
            for desc, cnt in bucket["descriptors"].most_common()
        ]
        quarter_primary_descriptor = (
            quarter_descriptor_counts[0]["descriptor"]
            if quarter_descriptor_counts
            else "Target personas"
        )
        quarter_stage_label = (
            bucket["stage_counter"].most_common(1)[0][0]
            if bucket["stage_counter"]
            else None
        )
        quarter_expected_outcome_summary = _build_expected_outcome_summary(
            quarter_primary_descriptor,
            quarter_stage_label,
            bucket["delta"],
            len(bucket["accounts"]),
            quarter_segment_filters,
        )
        quarterly_expected_outcomes.append(
            {
                "quarter": quarter,
                "segment_filters": quarter_segment_filters,
                "segment_summary": _segment_summary_text(quarter_segment_filters),
                "persona_descriptor_counts": quarter_descriptor_counts,
                "expected_outcome_summary": quarter_expected_outcome_summary,
                "total_delta_bp": bucket["delta"],
            }
        )

    expected_outcomes = {
        "portfolio": {
            "segment_filters": global_segment_filters,
            "segment_summary": _segment_summary_text(global_segment_filters),
            "persona_descriptor_counts": global_persona_descriptor_counts,
            "expected_outcome_summary": global_expected_outcome_summary,
            "total_delta_bp": total_delta_global,
        },
        "by_quarter": sorted(
            quarterly_expected_outcomes,
            key=lambda item: int("".join(ch for ch in str(item["quarter"]) if ch.isdigit()) or "0"),
        ),
    }

    scatter_summary: List[Dict[str, Any]] = []
    seen_personas: set[str] = set()
    for entry in scatter:
        pid = entry.get("id") or entry.get("persona_id")
        if not pid or pid in seen_personas:
            continue
        seen_personas.add(pid)
        scatter_summary.append(
            {
                "id": pid,
                "label": entry.get("label"),
                "perceptibility": entry.get("perceptibility"),
                "proximity": entry.get("proximity"),
                "involvement": entry.get("involvement"),
                "path_probability": entry.get("path_probability"),
                "belief_level": entry.get("belief_level"),
                "expected_next_prob": entry.get("expected_next_prob"),
                "activation": entry.get("activation"),
            }
        )

    total_touch = broad_total + focus_total
    mix = {
        "broad": broad_total,
        "focused": focus_total,
        "broad_ratio": (broad_total / total_touch) if total_touch else None,
    }

    randomization = {
        "avg_path_probability": (
            sum(path_probabilities) / len(path_probabilities)
            if path_probabilities
            else None
        ),
        "path_count_sampled": len(path_probabilities),
    }

    return {
        "campaign_themes": campaign_themes,
        "conversion_focuses": portfolio_focuses,
        "scatter": scatter_summary[:200],
        "broad_focus_mix": mix,
        "randomization": randomization,
        "expected_outcomes": expected_outcomes,
    }


def _summary_from_accounts(
    accounts: Sequence[Dict[str, Any]],
    *,
    persona_wolves_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    persona_wolves_label_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    account_count = len(accounts)
    persona_wolves_metrics = persona_wolves_metrics or {}
    persona_wolves_label_index = persona_wolves_label_index or {}
    keystone_rollup: Dict[str, Dict[str, Any]] = {}
    prob_primary = []
    accuracies = []
    persona_counter: Counter[str] = Counter()
    pain_counter: Counter[str] = Counter()
    capability_counter: Counter[str] = Counter()
    total_delta_bp = 0.0
    coverage_ratios: List[float] = []
    unmatched_total = 0
    required_total = 0
    cluster_entries: List[Dict[str, Any]] = []
    accounts_by_id: Dict[str, Dict[str, Any]] = {
        str(account.get("account_id")): account
        for account in accounts
        if account.get("account_id")
    }

    def _init_keystone_entry(
        persona_label: str,
        persona_id: Optional[str],
    ) -> Dict[str, Any]:
        wolves_info = _lookup_wolves_metric(
            persona_id,
            persona_label,
            persona_wolves_metrics,
            persona_wolves_label_index,
        )
        return {
            "persona_id": persona_id,
            "persona_label": persona_label,
            "wolves_score": _safe_float((wolves_info or {}).get("wolves_score"), None),
            "wolves_delta_bp": _safe_float((wolves_info or {}).get("delta_win_bp"), None),
            "wolves_involvement_rate": _safe_float(
                (wolves_info or {}).get("involvement_rate"), None
            ),
            "wolves_blocker_rate": _safe_float(
                (wolves_info or {}).get("blocker_rate"), None
            ),
            "wolves_sample_size": (wolves_info or {}).get("sample_size"),
            "occurrences": 0,
            "accounts": set(),
            "clusters": set(),
            "stories": [],
        }

    for account in accounts:
        account_personas: set[str] = set()
        account_pains: set[str] = set()
        account_capabilities: set[str] = set()

        paths = account.get("prediction", {}).get("persona_paths") or []
        if paths:
            prob_primary.append(paths[0].get("probability") or 0.0)
        fit = account.get("prediction", {}).get("fit") or {}
        if fit.get("accuracy") is not None:
            accuracies.append(fit["accuracy"])
        account_delta_bp = 0.0
        for campaign in account.get("campaigns") or []:
            total_delta_bp += campaign.get("total_delta_bp", 0.0)
            account_delta_bp += campaign.get("total_delta_bp", 0.0)
            for persona in campaign.get("focus_personas") or []:
                if persona:
                    account_personas.add(persona)
            for pain in campaign.get("focus_pains") or []:
                if pain:
                    account_pains.add(pain)
        for transition in account.get("transitions") or []:
            persona_label = transition.get("persona", {}).get("label")
            if persona_label:
                account_personas.add(persona_label)
            for capability in transition.get("capabilities") or []:
                label = capability.get("label")
                if label:
                    account_capabilities.add(label)
            for pain in transition.get("pains") or []:
                label = pain.get("label")
                if label:
                    account_pains.add(label)

        for persona in account_personas:
            persona_counter[persona] += 1
        for pain in account_pains:
            pain_counter[pain] += 1
        for capability in account_capabilities:
            capability_counter[capability] += 1

        meta = account.get("meta") or {}
        account_tokens: set[str] = set()
        for key in ("industry", "geography", "revenue_range", "employee_range", "funding_stage"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                account_tokens.add(f"{key}:{value.strip()}")
        industry = meta.get("industry")
        geography = meta.get("geography")
        if industry and geography:
            account_tokens.add(f"combo_industry_geography:{industry.strip()}|{geography.strip()}")
        for persona_label in list(account_personas)[:3]:
            account_tokens.add(f"persona:{persona_label}")
        deal_status = account.get("deal_status")
        if isinstance(deal_status, str):
            normalized_status = deal_status.strip()
            lower_status = normalized_status.lower()
            if normalized_status and lower_status not in {
                "new",
                "not started",
                "in progress",
                "in-progress",
                "pipeline",
                "open",
                "unknown",
            }:
                account_tokens.add(f"deal_status:{normalized_status}")
        if not account_tokens:
            account_tokens.add("cluster:General focus")
        cluster_entries.append(
            {
                "id": account.get("account_id"),
                "name": account.get("account_name") or account.get("account_id"),
                "tokens": list(account_tokens),
                "score": account_delta_bp or 1.0,
                "personas": sorted(account_personas),
            }
        )

        enrichment = account.get("enrichment") or {}
        enrichment_summary = enrichment.get("summary") or {}
        coverage_ratio = enrichment_summary.get("coverage_ratio")
        if isinstance(coverage_ratio, (int, float)):
            coverage_ratios.append(float(coverage_ratio))
        required = enrichment_summary.get("required_personas")
        matched = enrichment_summary.get("personas_with_matches")
        if isinstance(required, (int, float)):
            required_total += int(required)
        if isinstance(required, (int, float)) and isinstance(matched, (int, float)):
            unmatched_total += max(0, int(required) - int(matched))

        for kp in account.get("keystone_personas") or []:
            label = kp.get("persona_label") or kp.get("persona") or kp.get("label")
            if not label:
                continue
            persona_id = kp.get("persona_id")
            key = (persona_id or label).strip()
            if not key:
                continue
            entry = keystone_rollup.get(key)
            if entry is None:
                entry = _init_keystone_entry(label, persona_id)
                keystone_rollup[key] = entry
            wolves_score = kp.get("wolves_score")
            if wolves_score is not None:
                current = entry.get("wolves_score")
                if current is None or wolves_score > current:
                    entry["wolves_score"] = wolves_score
            for field in (
                "wolves_delta_bp",
                "wolves_involvement_rate",
                "wolves_blocker_rate",
                "wolves_sample_size",
            ):
                value = kp.get(field)
                if value is not None and entry.get(field) is None:
                    entry[field] = value
            entry["occurrences"] += 1
            account_id = account.get("account_id")
            if account_id:
                entry["accounts"].add(account_id)

    def _top(counter: Counter[str]) -> List[Dict[str, Any]]:
        return [
            {"label": label, "count": count}
            for label, count in counter.most_common(6)
        ]

    account_clusters = _derive_account_clusters(
        cluster_entries,
        accounts_by_id,
        total_accounts=account_count,
        total_delta_bp=total_delta_bp,
        persona_wolves_metrics=persona_wolves_metrics,
        persona_wolves_label_index=persona_wolves_label_index,
    )

    for cluster in account_clusters:
        for kp in cluster.get("keystone_personas") or []:
            label = kp.get("persona") or kp.get("persona_label")
            if not label:
                continue
            persona_id = kp.get("persona_id")
            key = (persona_id or label).strip()
            if not key:
                continue
            entry = keystone_rollup.get(key)
            if entry is None:
                entry = _init_keystone_entry(label, persona_id)
                keystone_rollup[key] = entry
            for field in (
                "wolves_score",
                "wolves_delta_bp",
                "wolves_involvement_rate",
                "wolves_blocker_rate",
                "wolves_sample_size",
            ):
                value = kp.get(field)
                if value is not None and entry.get(field) is None:
                    entry[field] = value
            cluster_label = cluster.get("label")
            if cluster_label:
                entry["clusters"].add(cluster_label)
            story = kp.get("coalition_story")
            if story:
                entry["stories"].append({"cluster": cluster_label, "story": story})

    return {
        "account_count": account_count,
        "avg_best_path_probability": (
            sum(prob_primary) / len(prob_primary) if prob_primary else 0.0
        ),
        "avg_accuracy": (
            sum(accuracies) / len(accuracies) if accuracies else None
        ),
        "top_personas": _top(persona_counter),
        "top_pains": _top(pain_counter),
        "top_capabilities": _top(capability_counter),
        "total_expected_delta_bp": total_delta_bp,
        "avg_people_coverage": (
            sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else None
        ),
        "unmatched_persona_count": unmatched_total,
        "total_persona_requirements": required_total,
        "account_clusters": account_clusters,
        "keystone_personas": _summarize_keystone_rollup(keystone_rollup),
    }


def _summarize_keystone_rollup(
    rollup: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not rollup:
        return []
    total_occurrences = sum(entry.get("occurrences", 0) for entry in rollup.values())
    rows: List[Dict[str, Any]] = []
    for entry in rollup.values():
        label = entry.get("persona_label")
        if not label:
            continue
        share = (
            entry.get("occurrences", 0) / total_occurrences
            if total_occurrences
            else None
        )
        rows.append(
            {
                "persona_id": entry.get("persona_id"),
                "persona_label": label,
                "wolves_score": entry.get("wolves_score"),
                "wolves_delta_bp": entry.get("wolves_delta_bp"),
                "wolves_involvement_rate": entry.get("wolves_involvement_rate"),
                "wolves_blocker_rate": entry.get("wolves_blocker_rate"),
                "wolves_sample_size": entry.get("wolves_sample_size"),
                "accounts_covered": len(entry.get("accounts") or []),
                "cluster_labels": sorted(entry.get("clusters") or []),
                "share": share,
                "sample_story": (entry.get("stories") or [None])[0],
            }
        )
    rows.sort(
        key=lambda row: (
            1 if row.get("wolves_score") is not None else 0,
            row.get("wolves_score") or 0.0,
            row.get("share") or 0.0,
            row.get("accounts_covered") or 0,
        ),
        reverse=True,
    )
    return rows[:5]


# ---------------------------------------------------------------------------
# Public plan builder
# ---------------------------------------------------------------------------

def build_product_marketing_plan(
    product_id: str,
    *,
    account_id: Optional[str] = None,
) -> Dict[str, Any]:
    product_graph = build_product_graph(product_id)
    canonical_product_id = get_product_id_from_subgraph(product_graph) or product_id
    try:
        shm_metrics = get_shm_transition_metrics(canonical_product_id)
    except Exception as exc:  # pragma: no cover - diagnostic only
        LOGGER.warning("Unable to load SHM metrics for %s: %s", canonical_product_id, exc)
        shm_metrics = {}
    if shm_metrics:
        product_graph.graph["shm_metrics"] = shm_metrics
    journey_weights = load_weights(canonical_product_id) or {}
    persona_wolves_metrics, persona_metrics_updated_at = _load_persona_wolves_metrics(
        canonical_product_id
    )
    persona_wolves_label_index = _build_wolves_label_index(persona_wolves_metrics)

    global_insights: Optional[Dict[str, Any]] = None
    product_insights: Optional[Dict[str, Any]] = None
    canonical_journey: Dict[str, Any] = {}
    canonical_persona_path: List[Dict[str, Any]] = []
    try:
        with SessionLocal() as db:
            global_insights = summarize_global_insights(
                db,
                product_id=canonical_product_id,
            )
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning(
            "Unable to summarize product insights for %s: %s", canonical_product_id, exc
        )
        global_insights = None

    segment_patterns: Optional[Dict[str, Any]] = None
    win_regression_summary: Optional[Dict[str, Any]] = None
    if global_insights:
        product_insights = global_insights.get("product_insights")
        segment_patterns = (
            (product_insights or {}).get("segment_patterns") if product_insights else None
        )
        canonical_journey = dict(product_insights.get("journey_structure") or {})
        canonical_persona_path = _canonical_persona_path_from_steps(
            product_graph,
            canonical_journey.get("typical_path"),
        )
        win_regression_summary = global_insights.get("win_regression")

    if account_id:
        account_ids = [account_id]
    else:
        account_ids = get_target_account_ids(
            canonical_product_id,
            {"status": {"nin": ["Closed-won", "Closed-lost"]}},
        )

    accounts: List[Dict[str, Any]] = []
    for acc_id in account_ids:
        try:
            plan = build_account_marketing_blueprint(
                canonical_product_id,
                acc_id,
                product_graph=product_graph,
                canonical_persona_path=canonical_persona_path,
                segment_patterns=segment_patterns,
                shm_metrics=shm_metrics,
                journey_weights=journey_weights,
                win_regression=win_regression_summary,
                persona_wolves_metrics=persona_wolves_metrics,
                persona_metrics_updated_at=persona_metrics_updated_at,
            )
            accounts.append(plan)
        except Exception as exc:  # pragma: no cover - defensive
            accounts.append(
                {
                    "account_id": acc_id,
                    "account_name": acc_id,
                    "error": str(exc),
                }
            )

    summary = _summary_from_accounts(
        accounts,
        persona_wolves_metrics=persona_wolves_metrics,
        persona_wolves_label_index=persona_wolves_label_index,
    )
    summary["wolves_metrics_updated_at"] = persona_metrics_updated_at
    modality_stats = _build_modality_stats(accounts)
    _attach_account_interventions(accounts, modality_stats)
    persona_chain_effects = _build_persona_chain_effects(accounts)
    portfolio_plan = _aggregate_portfolio(accounts)
    portfolio_plan["wolves_metrics_updated_at"] = persona_metrics_updated_at
    portfolio_plan["execution_interventions"] = _aggregate_execution_interventions(
        accounts,
        persona_chain_effects=persona_chain_effects,
    )
    portfolio_plan["interventions"] = _build_portfolio_interventions(
        portfolio_plan.get("conversion_focuses") or [],
        modality_stats,
    )
    union_plays: List[Dict[str, Any]] = []
    portfolio_asset_cadence: List[Dict[str, Any]] = []
    for account in accounts:
        union_plays.extend(account.get("execution", {}).get("plays") or [])
        portfolio_asset_cadence.extend(
            account.get("execution", {}).get("asset_cadence") or []
        )
    portfolio_plan["asset_cadence"] = portfolio_asset_cadence
    portfolio_plan["keystone_personas"] = summary.get("keystone_personas") or []

    portfolio_summary_report = _portfolio_summary_report(
        accounts,
        summary,
        portfolio_plan,
    )
    portfolio_thesis = _portfolio_thesis_data(
        summary,
        portfolio_plan,
        canonical_persona_path,
        portfolio_summary_report,
    )
    summary["portfolio_summary_report"] = portfolio_summary_report
    summary["portfolio_thesis"] = portfolio_thesis
    portfolio_plan["arsenal_table"] = _portfolio_arsenal_rows(accounts)

    if product_insights:
        canonical_journey = dict(product_insights.get("journey_structure") or {})
        summary["typical_path"] = canonical_journey.get("typical_path") or []
        portfolio_plan["typical_path"] = canonical_journey.get("typical_path") or []
        if canonical_persona_path:
            summary["canonical_persona_path"] = canonical_persona_path
            plays_by_persona: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for play in union_plays:
                pid = play.get("persona_id")
                if pid:
                    plays_by_persona[pid].append(play)
            total_stages = len(canonical_persona_path) or 1
            summary["persona_engagements"] = [
                _persona_engagement_plan(
                    persona,
                    plays_by_persona,
                    total_stages,
                )
                for persona in canonical_persona_path
            ]

    return {
        "product_id": canonical_product_id,
        "generated_at": _now_iso(),
        "summary": summary,
        "accounts": accounts,
        "portfolio_plan": portfolio_plan,
        "canonical_journey": canonical_journey,
        "canonical_persona_path": canonical_persona_path,
        "product_insights": product_insights,
        "wolves_metrics_updated_at": persona_metrics_updated_at,
    }


# ---------------------------------------------------------------------------
# Backwards-compatible exports
# ---------------------------------------------------------------------------

def build_integrated_portfolio_plan(
    product_id: str,
    account_id: Optional[str] = None,
) -> Dict[str, Any]:
    return build_product_marketing_plan(product_id, account_id=account_id)


def build_integrated_portfolio_plan_v2(
    *,
    generated_at_iso: str,
    per_account_scaffolds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Legacy scaffold-based callers should move to build_product_marketing_plan.
    return {
        "product_id": "unknown",
        "generated_at": generated_at_iso,
        "summary": {},
        "accounts": per_account_scaffolds,
        "portfolio_plan": {"campaign_themes": []},
    }


def _filter_by_window(
    plan: Dict[str, Any],
    window_start: Optional[str],
    window_end: Optional[str],
) -> Dict[str, Any]:
    # No filtering necessary – kept for compatibility with older imports.
    return plan


def _save_rcs_json(*args: Any, **kwargs: Any) -> None:
    LOGGER.warning("[marketing_plan] _save_rcs_json is deprecated; skipping legacy write.")
CADENCE_TEMPLATE: Tuple[Dict[str, Any], ...] = (
    {"phase": "probe", "frequency_multiplier": 0.4, "duration_days": 7, "preferred_mode": "broad"},
    {"phase": "ramp", "frequency_multiplier": 0.8, "duration_days": 10, "preferred_mode": "focused"},
    {"phase": "sustain", "frequency_multiplier": 1.0, "duration_days": 14, "preferred_mode": "focused"},
    {"phase": "decay", "frequency_multiplier": 0.5, "duration_days": 10, "preferred_mode": "broad"},
    {"phase": "park", "frequency_multiplier": 0.25, "duration_days": 21, "preferred_mode": "broad"},
)

CADENCE_PHASE_HINTS: Dict[str, str] = {
    "probe": "Low-pressure discovery touches to surface intent signals.",
    "ramp": "Increase touch frequency once belief momentum builds.",
    "sustain": "Hold the lane with consistent, higher-value touches.",
    "decay": "Dial back while monitoring for fresh signals or path shifts.",
    "park": "Pause active outreach until a new belief update or trigger fires.",
}

CADENCE_STATE_METADATA: Dict[str, Dict[str, Any]] = {
    "probe": {
        "label": "Probe",
        "reason": "Belief is emerging—use light exploratory touches.",
        "state_multiplier": 0.85,
    },
    "ramp": {
        "label": "Ramp",
        "reason": "Belief is accelerating—lean in with focused engagements.",
        "state_multiplier": 1.1,
    },
    "sustain": {
        "label": "Sustain",
        "reason": "Persona is warm—maintain steady pressure.",
        "state_multiplier": 1.0,
    },
    "decay": {
        "label": "Decay",
        "reason": "Signals are cooling or fatigue is high—start tapering.",
        "state_multiplier": 0.6,
    },
    "park": {
        "label": "Park",
        "reason": "Pause outreach until belief or response improves.",
        "state_multiplier": 0.25,
    },
}

CADENCE_STATE_PHASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "probe": {"probe": 1.0, "ramp": 0.7, "sustain": 0.5, "decay": 0.35, "park": 0.25},
    "ramp": {"probe": 0.85, "ramp": 1.05, "sustain": 0.8, "decay": 0.45, "park": 0.25},
    "sustain": {"probe": 0.4, "ramp": 0.75, "sustain": 1.0, "decay": 0.55, "park": 0.35},
    "decay": {"probe": 0.35, "ramp": 0.55, "sustain": 0.65, "decay": 0.9, "park": 0.45},
    "park": {"probe": 0.2, "ramp": 0.3, "sustain": 0.4, "decay": 0.5, "park": 0.35},
}
PERSONA_SIMILARITY_THRESHOLD = 0.45
CONCERN_SIMILARITY_THRESHOLD = 0.4


def _cadence_phase_for_play(play: Dict[str, Any], total_stages: int) -> str:
    stage_idx = play.get("stage_index") or 0
    mode = play.get("mode") or "focused"
    if stage_idx <= 0:
        return "probe" if mode == "broad" else "ramp"
    if stage_idx <= max(1, total_stages // 3):
        return "ramp" if mode == "focused" else "probe"
    if stage_idx <= max(1, (2 * total_stages) // 3):
        return "sustain"
    if stage_idx >= total_stages - 1:
        return "park"
    return "decay" if mode == "broad" else "sustain"


def _plays_for_persona_phase(
    persona_plays: Sequence[Dict[str, Any]],
    phase: str,
    total_stages: int,
) -> List[Dict[str, Any]]:
    if not persona_plays:
        return []

    filtered: List[Dict[str, Any]] = []
    for play in persona_plays:
        cadence_phase = _cadence_phase_for_play(play, total_stages)
        if cadence_phase == phase:
            filtered.append(play)
    if not filtered:
        filtered = list(persona_plays)

    filtered.sort(
        key=lambda p: (
            -_safe_float(p.get("expected_delta_bp"), 0.0),
            p.get("mode") != "focused",
        )
    )

    rows: List[Dict[str, Any]] = []
    for play in filtered[:3]:
        rows.append(
            {
                "asset": (play.get("asset") or {}).get("name")
                or play.get("asset_label")
                or "Asset",
                "channel": (play.get("channel") or {}).get("name")
                or play.get("channel_label")
                or "Channel",
                "mode": play.get("mode"),
                "expected_delta_bp": play.get("expected_delta_bp"),
                "confidence": play.get("confidence"),
                "funnel_phase": _journey_phase_label(
                    play.get("stage_index", 0),
                    total_stages,
                ),
            }
    )
    return rows


def _cadence_state_for_persona(
    persona: Dict[str, Any],
    snapshot: Optional[Dict[str, Any]],
) -> Tuple[str, str, bool]:
    belief_level = _safe_float(persona.get("belief_level"), 0.0) or 0.0
    committee_prob = _safe_float(persona.get("expected_in_deal_prob"), 0.0) or 0.0
    fatigue = _safe_float(persona.get("fatigue"), 0.0) or 0.0
    positive_rate = _safe_float((snapshot or {}).get("positive_rate"), None)
    last_positive_days = _safe_float((snapshot or {}).get("last_positive_days"), None)

    stalled_positive = (
        last_positive_days is not None and last_positive_days >= 45
    )
    low_positive_signal = positive_rate is not None and positive_rate <= 0.1
    effective_low_positive = positive_rate is None or low_positive_signal

    if fatigue >= 0.75 and effective_low_positive and stalled_positive:
        return (
            "park",
            "High fatigue and no recent positive signal—pause until belief updates.",
            True,
        )
    if fatigue >= 0.65 or (positive_rate is not None and positive_rate <= 0.15):
        return (
            "decay",
            "Dial down touches while fatigue cools or new signals emerge.",
            False,
        )
    if belief_level >= 0.65 or committee_prob >= 0.5:
        return ("sustain", "Belief is strong—hold consistent engagement.", False)
    if belief_level >= 0.4 or committee_prob >= 0.3:
        return ("ramp", "Belief is rising—accelerate targeted plays.", False)
    return ("probe", "Early belief—use exploratory touches to test interest.", False)


def _persona_engagement_plan(
    persona: Dict[str, Any],
    plays_by_persona: Dict[str, List[Dict[str, Any]]],
    total_stages: int,
    *,
    fit_score: Optional[float] = None,
) -> Dict[str, Any]:
    pid = persona.get("id")
    persona_plays = plays_by_persona.get(pid) or []
    fatigue = _safe_float(persona.get("fatigue"), 0.0) or 0.0
    snapshot = persona.get("engagement_snapshot") or {}
    positive_rate = _safe_float(snapshot.get("positive_rate"), None)

    committee_prob = _safe_float(persona.get("expected_in_deal_prob"), 0.0) or 0.0
    priority_score = _safe_float(persona.get("priority_score"), 0.0) or 0.0
    belief_level = _safe_float(persona.get("belief_level"), 0.0) or 0.0
    cadence_state, cadence_reason, park_until_signal = _cadence_state_for_persona(
        persona,
        snapshot,
    )
    state_meta = CADENCE_STATE_METADATA.get(cadence_state, CADENCE_STATE_METADATA["probe"])

    engagement_signal = min(
        1.0,
        max(
            0.0,
            0.45 * priority_score + 0.35 * committee_prob + 0.2 * belief_level,
        ),
    )
    base_freq = 0.4 + 1.2 * engagement_signal
    state_multiplier = state_meta.get("state_multiplier", 1.0)
    fatigue_modifier = max(0.35, 1.0 - 0.55 * fatigue)
    base_freq_adjusted = base_freq * state_multiplier * fatigue_modifier
    if cadence_state == "park":
        base_freq_adjusted = min(base_freq_adjusted, 0.2)

    fit_multiplier = 1.0
    if fit_score is not None:
        fit_multiplier = max(0.55, min(1.35, 0.7 + 0.6 * fit_score))

    diversify = cadence_state in {"decay", "park"}
    diversification_reason = None
    if diversify:
        diversification_reason = "Reallocate effort to fresher personas while this one cools."
    elif fatigue >= 0.65 and (positive_rate is None or positive_rate <= 0.15):
        diversify = True
        diversification_reason = "High fatigue with limited positive response"

    cadence_rows: List[Dict[str, Any]] = []
    phase_weights = CADENCE_STATE_PHASE_WEIGHTS.get(
        cadence_state, CADENCE_STATE_PHASE_WEIGHTS["probe"]
    )
    for entry in CADENCE_TEMPLATE:
        phase_label = entry["phase"]
        phase_weight = phase_weights.get(phase_label, 0.4)
        freq = (
            base_freq_adjusted
            * entry["frequency_multiplier"]
            * phase_weight
            * fit_multiplier
        )
        cadence_rows.append(
            {
                "phase": phase_label,
                "duration_days": entry["duration_days"],
                "frequency_per_week": round(max(0.1, freq), 2),
                "policy_note": CADENCE_PHASE_HINTS.get(phase_label),
                "assets": _plays_for_persona_phase(
                    persona_plays,
                    phase_label,
                    total_stages,
                ),
            }
        )
    return {
        "persona_id": pid,
        "persona_label": persona.get("label"),
        "journey_phase": persona.get("journey_phase"),
        "belief_level": round(belief_level, 4),
        "belief_band": _belief_band(belief_level),
        "committee_probability": round(committee_prob, 4),
        "priority_score": round(priority_score, 4),
        "phase_probs": persona.get("phase_probs"),
        "dominant_phase": persona.get("dominant_phase"),
        "belief_metrics": {
            "perceptibility": persona.get("perceptibility"),
            "proximity": persona.get("proximity"),
            "involvement": persona.get("involvement"),
            "activation": persona.get("activation"),
        },
        "fatigue": round(fatigue, 3),
        "fatigue_reason": persona.get("fatigue_reason"),
        "diversify": diversify,
        "diversification_reason": diversification_reason,
        "engagement_state": cadence_state,
        "state_reason": cadence_reason or state_meta.get("reason"),
        "park_until_signal": park_until_signal,
        "base_frequency_per_week": round(max(0.1, base_freq_adjusted), 2),
        "engagement_snapshot": snapshot if snapshot else None,
        "target_people": persona.get("top_people") or [],
        "cadence": cadence_rows,
    }


def _build_asset_cadence_table(
    campaigns: Sequence[Dict[str, Any]],
    total_stages: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for campaign in campaigns or []:
        timeframe = campaign.get("quarter") or "Q1"
        theme = campaign.get("theme") or campaign.get("persona_focus")
        for play in campaign.get("plays") or []:
            rows.append(
                {
                    "timeframe": timeframe,
                    "campaign_theme": theme,
                    "persona_id": play.get("persona_id"),
                    "persona_label": play.get("persona_label"),
                    "funnel_phase": _journey_phase_label(
                        play.get("stage_index", 0),
                        total_stages,
                    ),
                    "cadence_phase": _cadence_phase_for_play(play, total_stages),
                    "asset": (play.get("asset") or {}).get("name")
                    or play.get("asset_label"),
                    "channel": (play.get("channel") or {}).get("name")
                    or play.get("channel_label"),
                    "mode": play.get("mode"),
                    "expected_delta_bp": play.get("expected_delta_bp"),
                    "confidence": play.get("confidence"),
                    "duration_days": play.get("duration_days"),
                    "target_stage": play.get("target_stage") or play.get("stage_label"),
                    "target_concern": play.get("target_concern"),
                    "segment_fit_score": play.get("segment_fit_score"),
                    "belief_probability": play.get("belief_probability"),
                    "evidence": play.get("evidence"),
                }
            )
    return rows
