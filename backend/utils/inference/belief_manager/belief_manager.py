# backend/utils/inference/belief_manager/belief_manager.py
from __future__ import annotations
from dataclasses import dataclass
import dataclasses
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
import typing as t
from uuid import UUID
from sqlalchemy.orm import Session


import math
import networkx as nx
import pydantic

from backend.utils.inference.rcs_generators.persona_map.persona_belief_engine import (
    PersonaGraph,
    predict_snapshot,
    _renormalize_outgoing,
)


from backend.utils.inference.belief_manager.graph_diff_mapper import (
    SMALL_SAMPLE_CFG,
    infer_graph_updates_from_diffs,
)
from backend.utils.inference.belief_manager.learn.materialize import (
    materialize_graphstore_from_networkx,
)
from backend.utils.inference.belief_manager import shm_graph_learner

try:
    import numpy as _np

    _NP_SCALAR = _np.generic
except Exception:
    _np = None

    class _NP_SCALAR:  # type: ignore
        pass

from backend.database import SessionLocal
from backend.utils.graph_base import network_graph
from backend.utils.graph_base.network_graph import (
    _set_node_label,
    build_product_graph,
    get_product_id_from_subgraph,
    get_target_nodes_by_source_and_type,
    get_source_nodes_by_target_and_type,
    get_edge_weight,
    get_edge_attribute,
)
from backend.utils.crm_management.target_account_manager import (
    get_account_by_id,
    get_target_account_ids,
    map_account_meta_to_stable_ids,
    TargetAccount as TargetAccountORM,
)
from backend.utils.crm_management.engagement_service import (
    get_account_engagements,
)
from backend.utils.crm_management.person_service import (
    match_persona_for_actor_in_graph,
)
from backend.utils.crm_management.person_models import (
    AccountPersonaMatch,
    AccountPerson,
)
from backend.utils.inference.subsidies.subsidy_engine import (
    apply_subsidies_to_edge,
    subsidy_relevance_for_persona,
    wolf_score_dynamic,
)
from backend.utils.segment_utils import segment_keys_from_meta

from backend.utils.inference.belief_manager.learn.graph_learning import (
    GraphStore,
    learn_and_summarize,
)
from backend.utils.inference.belief_manager.journey.storage import (
    load_weights,
    load_stats,
)
from backend.utils.inference.belief_manager.journey.replay import next_distribution
from backend.utils.inference.belief_manager.journey.learn_service import (
    update_bayesian_journey_for_account,
    rebuild_stats_from_shm,
)
from backend.utils.inference.belief_manager.journey.shm_models import SHMEpisode
from backend.utils.inference.belief_manager.types import (
    Episode,
    LocalAdjustment,
    PersonaPrediction,
    PredictionError,
    EdgeAdjustment,
    NodeAdjustment,
)



# ---------------------------------------------------------------------
# Models (minimal)
# ---------------------------------------------------------------------
@dataclass
class LikelyPath:
    path: List[str]  # ordered node ids: persona_id then per-step ids (optional)
    personas: List[str]  # persona-only list (usually [persona_id])
    score: float
    probability: float
    rationale: str


@dataclass
class LearningArtifacts:
    """
    Learning-layer outputs.

    - `graphstore` is the in-memory posterior state (after back-scoring)
    - `summary` is the human/FE summary from learn_and_summarize (or similar)
    - `diffs` / `neighborhoods` are optional outputs if you also use
      infer_graph_updates_from_diffs (v3 Bayesian learner)
    - `apply_errors` collects any errors applying/syncing learning to the graph
    """

    graphstore: Optional[GraphStore]
    summary: Optional[Dict[str, Any]]
    diffs: Optional[Dict[str, Any]]
    neighborhoods: Optional[Dict[str, Any]]
    apply_errors: List[str]


@dataclass
class BeliefThesisCore:
    """
    Core, learning-layer representation of a belief thesis for an account.

    This is deliberately graph- and UI-agnostic; it is what the "engine"
    computes and what the UI layer decorates with labels, JSON-safe coercion,
    etc.
    """

    account_id: str
    product_id: str

    # Persona resolution + timeline
    persons: List[Dict[str, Any]]
    observed_persona_ids: List[str]
    persona_resolution_stats: Dict[str, Any]

    # Account prior summary
    account_prior_size: int

    # PersonaGraph snapshots
    baseline_paths: List[Dict[str, Any]]
    baseline_expected_next: List[Dict[str, Any]]
    current_paths: List[Dict[str, Any]]
    current_expected_next: List[Dict[str, Any]]

    # Journey + simple walk paths
    journey: Dict[str, Any]
    walk_paths: List[Dict[str, Any]]

    # Fit metrics (e.g. hit@1, hit@3, best_persona)
    fit: Dict[str, Any]

    # Learning artifacts from GraphStore/Bayesian layer
    learning: LearningArtifacts

    # Persona/person involvement projections
    persona_committee_probs: Dict[str, float]
    persona_posteriors: Dict[str, float]
    person_committee_probs: List[Dict[str, Any]]
    persona_belief_posteriors: Dict[str, Dict[str, Any]]
    persona_wolf_scores: Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------
# JSON-safety helpers
# ---------------------------------------------------------------------
def _scan_nonstring_keys(obj: t.Any, path: str = "$") -> t.Optional[str]:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if not isinstance(k, str):
                return f"{path} (has non-string key {type(k).__name__}: {k})"
            bad = _scan_nonstring_keys(v, f"{path}.{k}")
            if bad:
                return bad
        return None
    if isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            bad = _scan_nonstring_keys(v, f"{path}[{i}]")
            if bad:
                return bad
        return None
    return None


def _json_key(k: t.Any) -> str:
    if isinstance(k, str):
        return k
    if isinstance(k, (list, set, tuple)):
        return "|".join(_json_key(x) for x in k)
    if isinstance(k, (datetime, date)):
        return k.isoformat()
    if isinstance(k, (UUID, Decimal)):
        return str(k)
    if isinstance(k, Enum):
        return str(k.value)
    return str(k)


def _json_safe(obj: t.Any) -> t.Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (Decimal, UUID)):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, _NP_SCALAR):
        return obj.item()

    if pydantic is not None:
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            return _json_safe(obj.model_dump())
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            return _json_safe(obj.dict())

    if dataclasses.is_dataclass(obj):
        return _json_safe(dataclasses.asdict(obj))

    if hasattr(obj, "_asdict") and callable(getattr(obj, "_asdict")):
        return _json_safe(obj._asdict())

    try:
        import networkx as _nx

        if isinstance(obj, (_nx.Graph, _nx.DiGraph)):
            return {
                "_graph": True,
                "nodes": obj.number_of_nodes(),
                "edges": obj.number_of_edges(),
            }
    except Exception:
        pass

    if isinstance(obj, Mapping):
        out: dict[str, t.Any] = {}
        for k, v in obj.items():
            out[_json_key(k)] = _json_safe(v)
        return out

    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]

    return str(obj)


def _engagement_attr(obj: t.Any, key: str) -> t.Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _engagement_meta_from_event(event: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not event:
        return None
    engagement = event.get("engagement") if isinstance(event, dict) else None
    if engagement is None:
        return None
    channel = _engagement_attr(engagement, "channel") or _engagement_attr(engagement, "source")
    asset_id = _engagement_attr(engagement, "asset_id")
    label = (
        _engagement_attr(engagement, "name")
        or _engagement_attr(engagement, "title")
        or _engagement_attr(engagement, "raw_activity")
    )
    return {
        "channel": channel,
        "asset_id": asset_id,
        "label": label,
        "engagement_type": _engagement_attr(engagement, "engagement_type")
        or _engagement_attr(engagement, "type"),
    }


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def _as_dt(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts
    s = str(ts or "").replace("Z", "")
    return datetime.fromisoformat(s)


def _sorted_engagements(engagements: Iterable[Any]) -> List[Any]:
    return sorted(
        list(engagements or []),
        key=lambda e: _as_dt(
            getattr(e, "timestamp", None) or e.get("timestamp")
        ),
    )


def _actor_from(e: Any) -> Dict[str, str]:
    a = e.get("actor") if isinstance(e, dict) else getattr(e, "actor", {})
    name = getattr(a, "name", None) or a.get("name")
    title = getattr(a, "title", None) or a.get("title")
    dept = getattr(a, "department", None) or a.get("department")
    sen = getattr(a, "seniority", None) or a.get("seniority")
    email = getattr(a, "email", None) or a.get("email")

    return {
        "name": name or "",
        "title": (title or "").strip().lower(),
        "department": (dept or "").strip().lower(),
        "seniority": (sen or "").strip().lower(),
        "email": (email or "").strip().lower(),
    }


def _actor_resolver_key(actor: Dict[str, Any]) -> str:
    """
    Stable-ish key for this actor within (product_id, account_id) for the resolver.

    - Prefer email if present.
    - Fall back to (name|title|department|seniority) fingerprint.
    This is *not* the same as the canonical persona key – that's fine; we just
    need a consistent bucket to reuse resolver_state for the same human.
    """
    email = (actor.get("email") or "").strip().lower()
    if email:
        return f"email:{email}"

    name = (actor.get("name") or "").strip().lower()
    title = (actor.get("title") or "").strip().lower()
    dept = (actor.get("department") or "").strip().lower()
    sen = (actor.get("seniority") or "").strip().lower()
    return f"{name}|{title}|{dept}|{sen}"


_SHM_METRICS_CACHE: Dict[str, Tuple[Optional[str], Dict[str, Any]]] = {}


def _normalize_belief_state(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _init_transition_entry() -> Dict[str, Any]:
    return {"count": 0, "episodes": 0, "win_episodes": 0, "loss_episodes": 0}


def _compute_shm_transition_metrics(product_id: str) -> Dict[str, Any]:
    with SessionLocal() as db:
        steps = (
            db.query(
                SHMEpisode.account_id,
                SHMEpisode.episode_id,
                SHMEpisode.step_index,
                SHMEpisode.persona_id,
                SHMEpisode.belief_state,
            )
            .filter(SHMEpisode.product_id == product_id)
            .order_by(
                SHMEpisode.account_id,
                SHMEpisode.episode_id,
                SHMEpisode.step_index,
            )
            .all()
        )
        status_rows = (
            db.query(TargetAccountORM.id, TargetAccountORM.deal_status)
            .filter(TargetAccountORM.product_id == product_id)
            .all()
        )
    status_lookup = {
        row[0]: (row[1] or "").strip().lower()
        for row in status_rows
    }

    persona_stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(_init_transition_entry)
    )
    belief_stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(_init_transition_entry)
    )
    persona_belief_stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(_init_transition_entry)
    )

    current_key = None
    sequence: List[Any] = []

    def _flush_episode(ep_steps: List[Any], account_id: Optional[str]) -> None:
        if not ep_steps:
            return
        ep_steps.sort(key=lambda s: s.step_index)
        status = status_lookup.get(account_id or "", "")
        win_flag = status in {"closed-won", "won"}
        loss_flag = status in {"closed-lost", "lost"}

        persona_episode_seen: Set[Tuple[str, str]] = set()
        persona_win_seen: Set[Tuple[str, str]] = set()
        persona_loss_seen: Set[Tuple[str, str]] = set()
        belief_episode_seen: Set[Tuple[str, str]] = set()
        belief_win_seen: Set[Tuple[str, str]] = set()
        belief_loss_seen: Set[Tuple[str, str]] = set()
        persona_belief_seen: Set[Tuple[str, str]] = set()

        for idx, step in enumerate(ep_steps):
            persona_id = step.persona_id
            belief_code = _normalize_belief_state(step.belief_state)
            if persona_id and belief_code:
                entry = persona_belief_stats[persona_id][belief_code]
                entry["count"] += 1
                key = (persona_id, belief_code)
                if key not in persona_belief_seen:
                    entry["episodes"] += 1
                    persona_belief_seen.add(key)
                    if win_flag:
                        entry["win_episodes"] += 1
                    elif loss_flag:
                        entry["loss_episodes"] += 1
            if idx == 0:
                continue
            prev = ep_steps[idx - 1]
            prev_persona = prev.persona_id
            prev_belief = _normalize_belief_state(prev.belief_state)
            if prev_persona and persona_id:
                edge_key = (prev_persona, persona_id)
                entry = persona_stats[prev_persona][persona_id]
                entry["count"] += 1
                if edge_key not in persona_episode_seen:
                    entry["episodes"] += 1
                    persona_episode_seen.add(edge_key)
                    if win_flag:
                        entry["win_episodes"] += 1
                        persona_win_seen.add(edge_key)
                    elif loss_flag:
                        entry["loss_episodes"] += 1
                        persona_loss_seen.add(edge_key)
            if prev_belief and belief_code and prev_belief != belief_code:
                belief_key = (prev_belief, belief_code)
                entry = belief_stats[prev_belief][belief_code]
                entry["count"] += 1
                if belief_key not in belief_episode_seen:
                    entry["episodes"] += 1
                    belief_episode_seen.add(belief_key)
                    if win_flag:
                        entry["win_episodes"] += 1
                        belief_win_seen.add(belief_key)
                    elif loss_flag:
                        entry["loss_episodes"] += 1
                        belief_loss_seen.add(belief_key)

    for step in steps:
        key = (step.account_id, step.episode_id)
        if key != current_key:
            _flush_episode(sequence, current_key[0] if current_key else None)
            current_key = key
            sequence = []
        sequence.append(step)
    _flush_episode(sequence, current_key[0] if current_key else None)

    def _finalize(table: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for src, dests in table.items():
            result[src] = {}
            for dst, entry in dests.items():
                win_episodes = entry["win_episodes"]
                loss_episodes = entry["loss_episodes"]
                denom = win_episodes + loss_episodes
                success_rate = win_episodes / denom if denom else 0.5
                result[src][dst] = {
                    "count": entry["count"],
                    "episodes": entry["episodes"],
                    "win_episodes": win_episodes,
                    "loss_episodes": loss_episodes,
                    "success_rate": round(success_rate, 4),
                }
        return result

    return {
        "persona": _finalize(persona_stats),
        "belief": _finalize(belief_stats),
        "persona_belief": _finalize(persona_belief_stats),
    }


def get_shm_transition_metrics(product_id: str) -> Dict[str, Any]:
    stats_meta = load_stats(product_id) or {}
    stats_updated = stats_meta.get("updated_at")
    cached = _SHM_METRICS_CACHE.get(product_id)
    if cached and cached[0] == stats_updated:
        return cached[1]
    metrics = _compute_shm_transition_metrics(product_id)
    _SHM_METRICS_CACHE[product_id] = (stats_updated, metrics)
    return metrics


def _apply_shm_metrics_to_persona_graph(
    PG: PersonaGraph,
    metrics: Optional[Dict[str, Any]],
) -> None:
    if not metrics:
        return
    persona_stats = metrics.get("persona") or {}
    updated = False
    for u, v, data in PG.G.edges(data=True):
        src_type = PG.G.nodes[u].get("node_type")
        dst_type = PG.G.nodes[v].get("node_type")
        if src_type != "persona" or dst_type != "persona":
            continue
        stats = persona_stats.get(u, {}).get(v)
        if not stats:
            continue
        freq = stats.get("count") or 0
        success = stats.get("success_rate")
        if success is None:
            success = 0.5
        base = 0.0
        try:
            base = float(data.get("weight") or data.get("likelihood") or 0.0)
        except Exception:
            base = 0.0
        boost = 1.0 + min(freq, 50) / 50.0 * 0.5
        alignment = 0.5 + 0.5 * success
        new_weight = base * alignment * boost if base else base
        if new_weight > 0:
            data["weight"] = new_weight
            data["likelihood"] = new_weight
        data["shm_frequency"] = freq
        data["shm_success_rate"] = success
        data["shm_win_episodes"] = stats.get("win_episodes")
        data["shm_loss_episodes"] = stats.get("loss_episodes")
        updated = True
    if updated:
        _renormalize_outgoing(PG.G)
    PG.G.graph["shm_metrics"] = metrics
def _softmax(vals: List[float]) -> List[float]:
    if not vals:
        return []
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    Z = sum(exps) or 1.0
    return [e / Z for e in exps]


def _estimate_offpath_rate(account_id: str, product_id: str) -> float:
    """
    Replace with real cohort stats:
      offpath_rate = (#steps with missing/unknown next persona) / (#total steps)
    For now, return a conservative small prior like 0.1 if you have sparse data,
    or 0.2 if you know off-path is common.
    """
    return 0.2


BELIEF_PHASE_SEQUENCE: Tuple[str, ...] = (
    "Unaware",
    "Problem Realization",
    "Pain Realization",
    "Resolution Discovery",
    "Execution Guidance",
)

BELIEF_PHASE_WEIGHTS: Dict[str, float] = {
    "Unaware": 0.1,
    "Problem Realization": 0.35,
    "Pain Realization": 0.55,
    "Resolution Discovery": 0.75,
    "Execution Guidance": 0.95,
}

PHASE_LABEL_NORMALIZATION: Dict[str, str] = {
    "zero moment": "Unaware",
    "zero moment of truth": "Unaware",
    "unaware": "Unaware",
    "problem realization": "Problem Realization",
    "problem": "Problem Realization",
    "early funnel": "Problem Realization",
    "pain realization": "Pain Realization",
    "mid funnel": "Pain Realization",
    "resolution discovery": "Resolution Discovery",
    "late funnel": "Resolution Discovery",
    "execution guidance": "Execution Guidance",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _canonical_phase(label: Optional[str]) -> str:
    if not label:
        return "Unaware"
    normalized = str(label).strip().lower()
    if not normalized:
        return "Unaware"
    return PHASE_LABEL_NORMALIZATION.get(normalized, "Unaware")


def _phase_from_position(order: int, total: int) -> str:
    if total <= 1:
        return "Problem Realization"
    ratio = order / max(total - 1, 1)
    if ratio <= 0.2:
        return "Problem Realization"
    if ratio <= 0.5:
        return "Pain Realization"
    if ratio <= 0.8:
        return "Resolution Discovery"
    return "Execution Guidance"


def _persona_phase_lookup(paths: List[Dict[str, Any]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for path in paths or []:
        personas = path.get("personas") or path.get("path") or []
        total = len(personas)
        for idx, persona in enumerate(personas):
            persona_id = None
            if isinstance(persona, dict):
                persona_id = persona.get("id") or persona.get("persona_id")
                base_phase = persona.get("journey_phase") or persona.get("stage_label")
            else:
                persona_id = str(persona)
                base_phase = None
            if not persona_id or persona_id in lookup:
                continue
            if base_phase:
                lookup[persona_id] = _canonical_phase(base_phase)
            else:
                lookup[persona_id] = _phase_from_position(idx, total)
    return lookup


def _dominant_phase_from_probs(probs: Dict[str, float]) -> str:
    if not probs:
        return "Unaware"
    phase, _ = max(probs.items(), key=lambda kv: kv[1])
    return phase


# ---Label Setters for Graph Nodes ----
def _L(G: nx.DiGraph, nid: str) -> str:
    try:
        return _set_node_label(G, nid) or str(nid)
    except Exception:
        return str(nid)


def _add_expected_next_labels(G: nx.DiGraph, rows: list[dict]) -> list[dict]:
    out = []
    for r in rows or []:
        pid = str(r.get("persona"))
        out.append({**r, "persona_label": _L(G, pid)})
    return out


def _add_path_labels(G: nx.DiGraph, paths: list[dict]) -> list[dict]:
    out = []
    for p in paths or []:
        personas = list(p.get("personas") or p.get("path") or [])
        out.append({**p, "persona_labels": [_L(G, x) for x in personas]})
    return out


def _add_journey_labels(G: nx.DiGraph, journey: dict) -> dict:
    steps = []
    for s in journey.get("steps", []):
        steps.append(
            {
                **s,
                "predicted_topK": _add_expected_next_labels(
                    G, s.get("predicted_topK", [])
                ),
                "walk_paths": _add_path_labels(G, s.get("walk_paths", [])),
            }
        )
    return {**journey, "steps": steps}


def _label_neighborhoods(G: nx.DiGraph, nbs: Any) -> Any:
    """
    Add labels to graph_diff_mapper-style neighborhoods (if present).
    Keeps structure but adds label variants for edges & anchors.
    """
    if not isinstance(nbs, dict):
        return nbs
    labeled: Dict[str, Any] = {}
    for band, bucket in nbs.items():
        lbucket: Dict[str, Any] = {}
        for k, v in (bucket or {}).items():
            edges = v.get("edges", [])
            anchor = v.get("anchor")
            if isinstance(anchor, str):
                anchor_label = _L(G, anchor)
            else:
                anchor_label = [_L(G, x) for x in (anchor or [])]
            lbucket[k] = {
                **v,
                "anchor_label": anchor_label,
                "edge_labels": [
                    (_L(G, u), _L(G, vv), rel) for (u, vv, rel) in edges
                ],
            }
        labeled[band] = lbucket
    return labeled


def _load_persona_wolves_scores(product_id: str) -> Dict[str, Dict[str, Any]]:
    metrics_dir = getattr(network_graph, "GRAPH_DATA_PATH", None)
    if not metrics_dir:
        metrics_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graph_data")
    metrics_path = os.path.join(metrics_dir, "persona_metrics", f"{product_id}.json")
    if not os.path.exists(metrics_path):
        return {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    scores: Dict[str, Dict[str, Any]] = {}
    for entry in payload.get("personas", []) or []:
        pid = entry.get("persona_id")
        if not pid:
            continue
        scores[str(pid)] = entry
    return scores


def _wolves_multiplier(
    persona_id: str,
    wolves_scores: Dict[str, Dict[str, Any]],
    *,
    account_id: Optional[str] = None,
    segment_keys: Optional[Sequence[str]] = None,
    persona_node: Optional[Dict[str, Any]] = None,
) -> float:
    info = wolves_scores.get(str(persona_id))
    if not info:
        return 1.0
    base_score = float(info.get("wolves_score") or 0.0)
    try:
        dynamic_score = (
            wolf_score_dynamic(
                persona_node or {"id": persona_id},
                base_score,
                account_id,
                segment_keys,
                datetime.utcnow(),
            )
            if account_id
            else base_score
        )
    except Exception:
        dynamic_score = base_score
    delta_bp = float(info.get("delta_win_bp") or 0.0)
    involvement = float(info.get("involvement_rate") or 0.0)
    blocker_rate = float(info.get("blocker_rate") or 0.0)
    delta_component = max(min(delta_bp / 500.0, 0.5), -0.5)
    weight = 1.0 + 0.6 * dynamic_score + 0.2 * delta_component + 0.15 * involvement - 0.25 * blocker_rate
    return max(0.3, weight)


def _derive_base_wolf_score(
    persona_node: Optional[Dict[str, Any]],
    wolves_entry: Optional[Dict[str, Any]],
) -> float:
    if wolves_entry and wolves_entry.get("wolves_score") is not None:
        return max(0.0, min(1.0, float(wolves_entry.get("wolves_score") or 0.0)))
    if not persona_node:
        return 0.0
    perceptibility = _safe_float(persona_node.get("perceptibility"), 0.0) or 0.0
    proximity = _safe_float(persona_node.get("proximity"), 0.0) or 0.0
    involvement = _safe_float(persona_node.get("involvement"), 0.0) or 0.0
    activation = _safe_float(persona_node.get("activation"), 0.0) or 0.0
    fallback = (
        0.35 * perceptibility
        + 0.30 * proximity
        + 0.20 * involvement
        + 0.15 * activation
    )
    return max(0.0, min(1.0, fallback))


def _boost_probability_map(
    prob_map: Dict[str, float],
    wolves_scores: Dict[str, Dict[str, Any]],
    *,
    account_id: Optional[str] = None,
    segment_keys: Optional[Sequence[str]] = None,
    product_graph: Optional[nx.DiGraph] = None,
) -> Dict[str, float]:
    if not prob_map or not wolves_scores:
        return prob_map
    adjusted: Dict[str, float] = {}
    for persona_id, prob in prob_map.items():
        persona_node = (
            product_graph.nodes.get(persona_id, {}) if product_graph and persona_id in product_graph else {}
        )
        weight = _wolves_multiplier(
            persona_id,
            wolves_scores,
            account_id=account_id,
            segment_keys=segment_keys,
            persona_node=persona_node,
        )
        adjusted[persona_id] = prob * weight
    total = sum(adjusted.values())
    if total <= 0:
        return prob_map
    return {persona_id: value / total for persona_id, value in adjusted.items()}


def _boost_expected_next_rows(
    rows: Optional[List[Dict[str, Any]]],
    wolves_scores: Dict[str, Dict[str, Any]],
    *,
    account_id: Optional[str] = None,
    segment_keys: Optional[Sequence[str]] = None,
    product_graph: Optional[nx.DiGraph] = None,
) -> List[Dict[str, Any]]:
    if rows is None:
        return []
    if not rows or not wolves_scores:
        return rows
    base: Dict[str, float] = {}
    for row in rows:
        persona_id = str(row.get("persona") or row.get("persona_id") or "")
        if not persona_id:
            continue
        value = row.get("prob")
        if value is None:
            value = row.get("probability")
        base[persona_id] = base.get(persona_id, 0.0) + float(value or 0.0)
    boosted = _boost_probability_map(
        base,
        wolves_scores,
        account_id=account_id,
        segment_keys=segment_keys,
        product_graph=product_graph,
    )
    if not boosted:
        return rows
    for row in rows:
        persona_id = str(row.get("persona") or row.get("persona_id") or "")
        if not persona_id or persona_id not in boosted:
            continue
        if "prob" in row:
            row["prob"] = boosted[persona_id]
        elif "probability" in row:
            row["probability"] = boosted[persona_id]
    rows.sort(
        key=lambda r: boosted.get(str(r.get("persona") or r.get("persona_id") or ""), 0.0),
        reverse=True,
    )
    return rows


def _apply_subsidy_boosts_to_expected_next(
    rows: Optional[List[Dict[str, Any]]],
    *,
    account_id: Optional[str],
    segment_keys: Optional[Sequence[str]],
    timestamp: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    if rows is None:
        return []
    if not rows or not account_id:
        return rows
    ts = timestamp or datetime.utcnow()
    adjusted: List[Dict[str, Any]] = []
    total = 0.0
    for row in rows:
        persona_id = str(row.get("persona") or row.get("persona_id") or "")
        if not persona_id:
            continue
        base_prob = float(row.get("prob") or row.get("probability") or 0.0)
        edge = {
            "persona_id": persona_id,
            "persona_canonical_id": persona_id,
            "base_transition_prob": base_prob,
            "tags": row.get("tags") or [],
        }
        boosted = apply_subsidies_to_edge(edge, account_id, segment_keys, ts)
        mutated = dict(row)
        mutated["prob_base"] = base_prob
        mutated["prob"] = boosted
        mutated["subsidy_lift"] = max(0.0, boosted - base_prob)
        adjusted.append(mutated)
        total += boosted
    if not adjusted:
        return rows
    if total > 0:
        for entry in adjusted:
            entry["prob"] = entry["prob"] / total
    adjusted.sort(key=lambda r: r.get("prob", 0.0), reverse=True)
    return adjusted


def _job_ids_for_persona(G: nx.DiGraph, persona_id: str) -> List[str]:
    return [
        node_id
        for node_id in get_source_nodes_by_target_and_type(G, persona_id, "performed_by")
        if node_id in G
    ]


def _pain_ids_for_job(G: nx.DiGraph, job_id: str) -> List[str]:
    pains: set[str] = set()
    pains.update(
        nid
        for nid in get_source_nodes_by_target_and_type(G, job_id, "felt_in")
        if nid in G
    )
    pains.update(
        nid
        for nid in get_target_nodes_by_source_and_type(G, job_id, "solves")
        if nid in G
    )
    return list(pains)


def _trigger_ids_for_pain(G: nx.DiGraph, pain_id: str) -> List[str]:
    return [
        nid
        for nid in get_target_nodes_by_source_and_type(G, pain_id, "triggered_by")
        if nid in G
    ]


def _zmot_links_for_trigger(G: nx.DiGraph, trigger_id: str) -> List[Tuple[str, str]]:
    """
    Returns a list of (zmot_id, source_id) pairs. The source_id indicates
    whether the ZMOT edge originates from the trigger itself or from an
    attribute_value node associated with the trigger.
    """
    pairs: List[Tuple[str, str]] = []
    for zmot_id in get_target_nodes_by_source_and_type(G, trigger_id, "leads_to_zmot"):
        if zmot_id in G:
            pairs.append((zmot_id, trigger_id))

    for attr_id in get_target_nodes_by_source_and_type(G, trigger_id, "prevalent_in"):
        if attr_id not in G:
            continue
        for zmot_id in get_target_nodes_by_source_and_type(G, attr_id, "associated_zmot"):
            if zmot_id in G:
                pairs.append((zmot_id, attr_id))
    return pairs


def _observable_payloads_for_zmot(G: nx.DiGraph, zmot_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    observables: List[Dict[str, Any]] = []
    for obs_id in get_target_nodes_by_source_and_type(G, zmot_id, "observed_in"):
        node = G.nodes.get(obs_id, {})
        observables.append(
            {
                "id": obs_id,
                "label": _L(G, obs_id),
                "channels": node.get("channels") or node.get("sources") or [],
                "description": node.get("description"),
            }
        )
    keywords: List[Dict[str, Any]] = []
    for kw_id in get_target_nodes_by_source_and_type(G, zmot_id, "keyword"):
        node = G.nodes.get(kw_id, {})
        keywords.append(
            {
                "id": kw_id,
                "label": _L(G, kw_id),
                "keyword": node.get("keyword"),
            }
        )
    return observables, keywords


def _collect_zmot_candidates_for_persona(
    G: nx.DiGraph,
    persona_id: str,
    *,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    if persona_id not in G:
        return []

    suggestions: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    for job_id in _job_ids_for_persona(G, persona_id):
        job_label = _L(G, job_id)
        for pain_id in _pain_ids_for_job(G, job_id):
            pain_label = _L(G, pain_id)
            for trigger_id in _trigger_ids_for_pain(G, pain_id):
                trigger_label = _L(G, trigger_id)
                trigger_boost = (
                    get_edge_attribute(G, pain_id, trigger_id, "boost")
                    or get_edge_attribute(G, pain_id, trigger_id, "likelihood")
                    or get_edge_weight(G, pain_id, trigger_id)
                )
                for zmot_id, source_id in _zmot_links_for_trigger(G, trigger_id):
                    key = (persona_id, trigger_id, zmot_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    zmot_label = _L(G, zmot_id)
                    zmot_boost = (
                        get_edge_attribute(G, source_id, zmot_id, "boost")
                        or get_edge_attribute(G, source_id, zmot_id, "likelihood")
                        or get_edge_weight(G, source_id, zmot_id)
                    )
                    observables, keywords = _observable_payloads_for_zmot(G, zmot_id)

                    suggestions.append(
                        {
                            "persona_id": persona_id,
                            "persona_label": _L(G, persona_id),
                            "job_id": job_id,
                            "job_label": job_label,
                            "pain_id": pain_id,
                            "pain_label": pain_label,
                            "pain_trigger_id": trigger_id,
                            "pain_trigger_label": trigger_label,
                            "zmot_event_id": zmot_id,
                            "zmot_label": zmot_label,
                            "boost": _safe_float(zmot_boost, 0.0),
                            "trigger_boost": _safe_float(trigger_boost, 0.0),
                            "observable_moments": observables[:5],
                            "keywords": keywords[:8],
                        }
                    )

    if not suggestions:
        return []

    suggestions.sort(
        key=lambda row: (
            _safe_float(row.get("boost"), 0.0),
            _safe_float(row.get("trigger_boost"), 0.0),
        ),
        reverse=True,
    )
    return suggestions[:limit]


def _ordered_persona_candidates(paths: List[Dict[str, Any]]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for path in paths or []:
        personas = path.get("personas") or path.get("persona_ids") or path.get("path") or []
        for persona_id in personas:
            pid = str(persona_id)
            if pid not in seen:
                ordered.append(pid)
                seen.add(pid)
    return ordered


def _forecast_zmot_events(
    G: nx.DiGraph,
    *,
    current_paths: List[Dict[str, Any]],
    observed_personas: Iterable[str],
    max_per_persona: int = 3,
) -> List[Dict[str, Any]]:
    persona_priority = _ordered_persona_candidates(current_paths)
    observed_set = {str(pid) for pid in observed_personas or []}

    forecasts: List[Dict[str, Any]] = []
    for persona_id in persona_priority:
        entries = _collect_zmot_candidates_for_persona(
            G,
            persona_id,
            limit=max_per_persona,
        )
        if not entries:
            continue
        for entry in entries:
            entry["already_observed"] = entry["persona_id"] in observed_set
            forecasts.append(entry)
    return forecasts


# ------------------
# Bucket Mapping - for SHM Context
# ------------------
def _map_step_bucket(raw: str) -> Optional[str]:
    """
    Map the replay_learn_persona_paths bucket strings into a coarse,
    SHM-friendly effect bucket label. Keep it simple and string-based.
    """
    if not raw:
        return None
    raw = str(raw).lower()

    if raw == "on_path":
        return "on_path"
    if raw == "near_path":
        return "near_path"
    if raw in {"off_path_known", "out_of_graph"}:
        return "off_path"
    if raw == "no_path":
        return "no_path"

    return "unknown"



# ---------------------------
# Adapters for GraphLearning
# ---------------------------


def _expected_topk_from_baseline(
    baseline_expected_next: list[dict],
) -> list[str]:
    # baseline_expected_next items look like {"persona": "persona:...", "prob": ...}
    return [
        row.get("persona")
        for row in (baseline_expected_next or [])
        if row.get("persona")
    ]


def _persona_match_score_for_run(
    pid: str, *, observed_persona_ids: list[str]
) -> float:
    # simple: 1.0 if observed in this run, else a small prior
    return 1.0 if pid in set(observed_persona_ids) else 0.4


# ---------------------------------------------------------------------
# Stage-wise persona predictor from RCS outputs
# ---------------------------------------------------------------------
def compute_stage_persona_distributions(
    *,
    persona_scores: Dict[str, Dict[str, Any]],
    concern_backlog: List[Dict[str, Any]],
    alpha: float = 1.0,  # likelihood weight (backlog)
    beta: float = 0.5,  # prior weight (persona_scores)
    w_perc: float = 0.6,
    w_prox: float = 0.4,
) -> Dict[str, List[Tuple[str, float]]]:
    # prior from persona_scores
    pri_rows, pri_vals = [], []
    for pid, row in (persona_scores or {}).items():
        s = float(row.get("strength", 0.0))
        if s == 0.0:
            s = 0.5 * float(row.get("activation", 0.0))
        pri_rows.append((pid, s))
        pri_vals.append(s)
    pri_probs = _softmax(pri_vals) if pri_vals else []
    prior = {pid: p for (pid, _), p in zip(pri_rows, pri_probs)}

    # likelihood by stage/persona from backlog
    stages = ("problem", "execution", "pain", "resolution")
    like: Dict[str, Dict[str, float]] = {s: {} for s in stages}
    for r in (concern_backlog or []):
        pid = r.get("persona_id") or r.get("pid")
        stage = (r.get("stage") or "problem").lower()
        if not pid or stage not in like:
            continue
        lift = float(r.get("lift_proxy", 0.0)) or 0.0
        key = float(r.get("keyness", 0.0)) or 0.0
        perc = float(r.get("perceptibility", 0.0)) or 0.0
        prox = float(r.get("proximity", 0.0)) or 0.0
        cs = lift * (0.5 + 0.5 * key) * (w_perc * perc + w_prox * prox)
        like[stage][pid] = like[stage].get(pid, 0.0) + cs

    # blend prior^beta and like^alpha per stage (log domain)
    out: Dict[str, List[Tuple[str, float]]] = {}
    for s in stages:
        items = []
        for pid, lval in like[s].items():
            p0 = prior.get(pid, 1e-6)
            score_log = beta * math.log(max(p0, 1e-12)) + alpha * math.log(
                max(lval, 1e-12)
            )
            items.append((pid, score_log))
        if not items:
            out[s] = []
            continue
        ids, logs = zip(*items)
        probs = _softmax(list(logs))
        out[s] = sorted(
            [(pid, p) for pid, p in zip(ids, probs)],
            key=lambda x: x[1],
            reverse=True,
        )
    return out


def _marginalize_over_stages(
    stage_dists: Dict[str, List[Tuple[str, float]]]
) -> List[Tuple[str, float]]:
    agg: Dict[str, float] = {}
    for _, dist in (stage_dists or {}).items():
        for pid, p in dist:
            agg[pid] = agg.get(pid, 0.0) + p
    return sorted(agg.items(), key=lambda kv: kv[1], reverse=True)


def _classify_outcome(
    pred_ranked: List[str], observed_pid: Optional[str], k_skip: int = 5
) -> str:
    if not observed_pid:
        return "no_observation"
    if not pred_ranked:
        return "off_path_known"
    if observed_pid == pred_ranked[0]:
        return "perfect_match"
    if observed_pid in pred_ranked[:k_skip]:
        return "skip_hit"
    return "off_path_known"


def _top_concerns_for_persona_from_backlog(
    backlog: List[Dict[str, Any]],
    persona_id: str,
    top_m: int = 3,
    w_perc: float = 0.6,
    w_prox: float = 0.4,
) -> List[Dict[str, Any]]:
    rows = []
    for r in (backlog or []):
        if (r.get("persona_id") or r.get("pid")) != persona_id:
            continue
        lift = float(r.get("lift_proxy", 0.0)) or 0.0
        key = float(r.get("keyness", 0.0)) or 0.0
        perc = float(r.get("perceptibility", 0.0)) or 0.0
        prox = float(r.get("proximity", 0.0)) or 0.0
        score = lift * (0.5 + 0.5 * key) * (w_perc * perc + w_prox * prox)
        rows.append((score, r))
    rows.sort(key=lambda t: t[0], reverse=True)
    return [r for _s, r in rows[:top_m]]


def _reinforce_edges_for_concerns(
    G: nx.DiGraph,
    concerns: List[Dict[str, Any]],
    *,
    lr: float = 0.02,
    like_key: str = "likelihood",
    like_cap: float = 1.0,
):
    for r in (concerns or []):
        cid = r.get("cid") or r.get("concern_id")
        if not cid or cid not in G:
            continue
        for u in list(G.predecessors(cid)):
            w = float(G[u][cid].get(like_key, 0.1))
            G[u][cid][like_key] = min(like_cap, w + lr * (1.0 - w))
        for v in list(G.successors(cid)):
            w = float(G[cid][v].get(like_key, 0.1))
            G[cid][v][like_key] = min(like_cap, w + lr * (1.0 - w))


def _light_bridge_to_observed(
    G: nx.DiGraph,
    prev_personas: List[str],
    observed_pid: str,
    *,
    like_key="likelihood",
    bridge_w=0.02,
):
    if observed_pid not in G:
        return
    for p in prev_personas[-2:]:
        if p in G and not G.has_edge(p, observed_pid):
            G.add_edge(p, observed_pid, **{like_key: bridge_w})


# ---------------------------------------------------------------------
# Fit metrics (UI tiles)
# ---------------------------------------------------------------------
def _fit_metrics_from_snapshot(snap: Dict[str, Any]) -> Dict[str, Any]:
    observed: List[str] = list(snap.get("observed_persona_ids") or [])
    top_paths: List[Dict[str, Any]] = list(snap.get("walk_paths") or [])

    if not top_paths:
        return {
            "hit_at_1": None,
            "hit_at_3": None,
            "best_persona": None,
        }

    ordered = sorted(
        top_paths, key=lambda r: r.get("probability", 0.0), reverse=True
    )
    best = ordered[0]
    best_personas = list(best.get("personas") or [])
    best_persona = best_personas[0] if best_personas else None

    candidate_personas: List[str] = []
    for w in ordered:
        for p in (w.get("personas") or []):
            if p not in candidate_personas:
                candidate_personas.append(p)

    hit1 = (best_persona in observed) if (best_persona and observed) else None
    hit3 = (
        any(p in observed for p in candidate_personas[:3])
        if (observed and candidate_personas)
        else None
    )
    return {
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "best_persona": best_persona,
    }


# -----------------------------------------------------
# Persona Resolution Stats
# ---------------------------------------------------
def _persona_resolution_stats(
    observed_matches: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Summarize how the persona resolver feels about this account:

      - Counts of each resolution status
      - Lists of persona ids that ended up stable / tentative / revised
      - Number of 'new_node_candidate' episodes

    This tells you whether the current graph is a good fit
    or if you're constantly struggling to explain observations.
    """
    total = len(observed_matches or [])
    status_counts: Dict[str, int] = {}

    stable_ids: set[str] = set()
    tentative_ids: set[str] = set()
    revised_ids: set[str] = set()

    for m in (observed_matches or []):
        # Prefer the flat 'resolution_status' we added in _observed_personas_from_engagements;
        # fall back to nested resolution.status if needed.
        status = (
            (m.get("resolution_status") or "")
            or ((m.get("resolution") or {}).get("status") or "")
        ).lower() or "unknown"

        status_counts[status] = status_counts.get(status, 0) + 1

        pid = m.get("effective_persona_id") or m.get("best")

        if not pid:
            continue

        if status == "stable":
            stable_ids.add(pid)
        elif status == "tentative":
            tentative_ids.add(pid)
        elif status == "revised":
            revised_ids.add(pid)

    return {
        "total_personas": total,
        "status_counts": status_counts,  # e.g. {"stable": 2, "tentative": 1, "new_node_candidate": 1, ...}
        "stable_persona_ids": sorted(stable_ids),
        "tentative_persona_ids": sorted(tentative_ids),
        "revised_persona_ids": sorted(revised_ids),
        "num_new_node_candidates": status_counts.get(
            "new_node_candidate", 0
        ),
    }


# ---------------------------------------------------------------------
# Helpers for belief thesis
# ---------------------------------------------------------------------
def _normalize_persona_component(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value).strip().lower()
    except Exception:
        return str(value)


def _candidate_persona_id_from_match(
    match: Dict[str, Any],
    actor: Dict[str, Any],
) -> Optional[str]:
    canonical = (match.get("canonical_meta") or {}) if isinstance(match, dict) else {}
    title = (
        actor.get("title")
        or actor.get("role")
        or canonical.get("title")
        or ""
    )
    department = actor.get("department") or canonical.get("department") or ""
    seniority = actor.get("seniority") or canonical.get("seniority") or ""

    title_norm = _normalize_persona_component(title)
    dept_norm = _normalize_persona_component(department)
    snr_norm = _normalize_persona_component(seniority)

    if not (title_norm or dept_norm or snr_norm):
        return None
    return f"{title_norm}|{dept_norm}|{snr_norm}"


def _observed_personas_from_engagements(
    product_id: str,
    account_id: str,
    engagements: Optional[Iterable[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Walk engagements in chronological order and:
      - For each actor, run them through the persona resolver with a
        persistent per-actor resolver_state within this account run.
      - Pass already-resolved personas for this account as a graph prior.
      - For each persona, record the FIRST time it becomes our best
        explanation (effective_persona_id) in timeline order.

    Notes:
      - We treat only POSITIVE evidence (observed engagements).
      - We avoid forcing 'new_node_candidate' personas into the persona
        journey; they may indicate a missing graph node that should be
        handled separately.

    `engagements` can be passed in (preferred) to avoid re-hitting the DB;
    if omitted, we will fetch them ourselves.
    """
    if engagements is None:
        raw = get_account_engagements(product_id, account_id) or []
    else:
        raw = list(engagements or [])

    engs = _sorted_engagements(raw)

    # Per-actor resolver states: our local “SHM lite” for this belief run.
    resolver_states: Dict[str, Dict[str, Any]] = {}

    # Personas already "in the room" for this account (for graph prior).
    committee_personas: List[str] = []
    committee_set: set[str] = set()

    # Final observed personas: one row per persona in first-seen order.
    out: List[Dict[str, Any]] = []
    seen_personas: set[str] = set()

    for e in engs:
        actor = _actor_from(e)
        actor_key = _actor_resolver_key(actor)
        prev_state = resolver_states.get(actor_key)

        # Run the graph-aware matcher + resolver.
        match = match_persona_for_actor_in_graph(
            product_id,
            actor,
            resolver_state=prev_state,
            account_persona_ids=list(committee_set) if committee_set else None,
        ) or {}

        resolution = match.get("resolution") or {}
        state = resolution.get("state")
        if state:
            # Persist updated resolver state for this actor within this run.
            resolver_states[actor_key] = state

        status = (resolution.get("status") or "").lower()
        resolved_id = resolution.get("resolved_persona_id")

        # Decide what persona (if any) we treat as "observed" at this step.
        # - If resolver says "new_node_candidate", we DON'T push into the journey
        #   because there is no reliable persona node in the graph yet.
        # - Otherwise, use resolved_id if present, else fall back to per-episode best.
        candidate_pid: Optional[str] = None
        if status == "new_node_candidate":
            candidate_pid = _candidate_persona_id_from_match(match, actor) or None
            effective_pid = candidate_pid
        else:
            effective_pid = resolved_id or match.get("best")

        # Update committee prior: all personas that have *ever* been effective.
        if effective_pid and effective_pid not in committee_set:
            committee_set.add(effective_pid)
            committee_personas.append(effective_pid)

        # Record FIRST time each persona appears in the timeline, for PersonaGraph.
        if effective_pid and effective_pid not in seen_personas:
            seen_personas.add(effective_pid)
            enriched = dict(match)
            enriched["effective_persona_id"] = effective_pid
            enriched["resolution_status"] = status
            if candidate_pid:
                enriched["candidate_persona_id"] = candidate_pid
            out.append(enriched)

    return out


def _engagement_timeline_with_personas(
    product_id: str,
    account_id: str,
    engagements: Optional[Iterable[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Chronologically ordered engagements enriched with persona resolver output.

    Unlike `_observed_personas_from_engagements`, this keeps every engagement
    (including repeated personas) so the sequential learner can evaluate each
    step independently.
    """

    if engagements is None:
        raw = get_account_engagements(product_id, account_id) or []
    else:
        raw = list(engagements or [])

    engs = _sorted_engagements(raw)

    resolver_states: Dict[str, Dict[str, Any]] = {}
    committee_set: set[str] = set()

    timeline: List[Dict[str, Any]] = []

    for engagement in engs:
        actor = _actor_from(engagement)
        actor_key = _actor_resolver_key(actor)
        prev_state = resolver_states.get(actor_key)

        match = match_persona_for_actor_in_graph(
            product_id,
            actor,
            resolver_state=prev_state,
            account_persona_ids=list(committee_set) if committee_set else None,
        ) or {}

        resolution = match.get("resolution") or {}
        status = (resolution.get("status") or "").lower() or "unknown"
        state = resolution.get("state")
        if state:
            resolver_states[actor_key] = state

        candidate_pid: Optional[str] = None
        if status == "new_node_candidate":
            candidate_pid = _candidate_persona_id_from_match(match, actor) or None
            effective_pid = candidate_pid
        else:
            effective_pid = resolution.get("resolved_persona_id") or match.get("best")

        if effective_pid:
            committee_set.add(effective_pid)

        raw_ts = getattr(engagement, "timestamp", None)
        if raw_ts is None and isinstance(engagement, dict):
            raw_ts = engagement.get("timestamp")
        try:
            ts = _as_dt(raw_ts)
        except Exception:
            ts = datetime.utcnow()

        timeline.append(
            {
                "engagement": engagement,
                "timestamp": ts,
                "actor": actor,
                "match": match,
                "resolution_status": status,
                "effective_persona_id": effective_pid,
                "candidate_persona_id": candidate_pid,
                "committee_personas": list(committee_set),
            }
        )

    return timeline


def _compute_persona_committee_probs(
    observed_persona_ids: List[str],
    timeline: List[Dict[str, Any]],
    current_paths: List[Dict[str, Any]],
    current_expected_next: List[Dict[str, Any]],
    wolves_scores: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    account_id: Optional[str] = None,
    segment_keys: Optional[Sequence[str]] = None,
    product_graph: Optional[nx.DiGraph] = None,
) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)

    if observed_persona_ids:
        counts = Counter(observed_persona_ids)
        total = sum(counts.values()) or 1.0
        for pid, count in counts.items():
            scores[pid] += 0.5 * (count / total)

    committee_seen: set[str] = set()
    for step in timeline or []:
        committee_seen.update(step.get("committee_personas") or [])
    for pid in committee_seen:
        scores[pid] = max(scores.get(pid, 0.0), 0.05)

    for path in current_paths or []:
        prob = _safe_float(path.get("probability"), 0.0)
        personas = list(path.get("personas") or [])
        if not personas and path.get("persona_ids"):
            personas = list(path.get("persona_ids") or [])
        for idx, persona in enumerate(personas):
            if isinstance(persona, str):
                pid = persona
            elif isinstance(persona, Mapping):
                pid = persona.get("id") or persona.get("persona_id")
            else:
                pid = None
            if not pid:
                continue
            decay = max(0.25, 1.0 - 0.15 * idx)
            scores[str(pid)] += 0.25 * prob * decay

    for row in current_expected_next or []:
        pid = str(row.get("persona"))
        if not pid:
            continue
        scores[pid] += 0.25 * _safe_float(row.get("prob"), 0.0)

    filtered = {
        pid: max(score, 0.0) for pid, score in scores.items() if score > 0
    }
    if not filtered:
        return {}
    if wolves_scores:
        boosted = _boost_probability_map(
            filtered,
            wolves_scores,
            account_id=account_id,
            segment_keys=segment_keys,
            product_graph=product_graph,
        )
        normalized = {pid: round(prob, 6) for pid, prob in boosted.items()}
    else:
        total_score = sum(filtered.values()) or 1.0
        normalized = {
            pid: round(score / total_score, 6)
            for pid, score in filtered.items()
        }
    return dict(
        sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)
    )


def _load_account_person_matches(
    product_id: str,
    account_id: str,
) -> List[Dict[str, Any]]:
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
        return []

    payloads: List[Dict[str, Any]] = []
    for match, person in rows:
        payloads.append(
            {
                "match_id": str(match.id),
                "persona_id": match.persona_id,
                "persona_label": match.persona_label,
                "stage": match.stage,
                "person_id": person.id if person else match.person_id,
                "person_name": (person.name if person else None) or None,
                "person_title": person.title if person else None,
                "person_department": person.department if person else None,
                "person_seniority": person.seniority if person else None,
                "match_confidence": _safe_float(match.match_confidence, 0.5),
                "source": match.source,
                "notes": match.notes,
                "engagement_count": person.engagement_count if person else None,
                "last_seen_at": (
                    person.last_seen_at.isoformat()
                    if person and person.last_seen_at
                    else None
                ),
            }
        )
    return payloads


def _project_person_committee_probs(
    persona_probs: Dict[str, float],
    match_rows: List[Dict[str, Any]],
    product_graph: nx.DiGraph,
    persona_belief_posteriors: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not match_rows:
        return []
    entries: List[Dict[str, Any]] = []
    for row in match_rows:
        persona_id = row.get("persona_id")
        if not persona_id:
            continue
        persona_prob = persona_probs.get(persona_id, 0.0)
        confidence = _safe_float(row.get("match_confidence"), 0.5)
        engagement_count = _safe_float(row.get("engagement_count"), 0.0)
        engagement_factor = 1.0 + min(engagement_count, 12.0) * 0.03
        raw_prob = persona_prob * (0.5 + 0.5 * confidence) * engagement_factor
        probability = min(1.0, max(raw_prob, 0.0))
        belief_data = persona_belief_posteriors.get(persona_id) or {}
        base_belief_level = _safe_float(belief_data.get("belief_level"), 0.5) or 0.5
        projected_belief = min(
            1.0,
            max(0.0, base_belief_level * (0.5 + 0.5 * confidence) * engagement_factor),
        )
        projected = dict(row)
        projected["persona_label"] = (
            projected.get("persona_label") or _L(product_graph, persona_id)
        )
        projected["probability"] = round(probability, 6)
        projected["phase_probs"] = belief_data.get("phase_probs")
        projected["belief_level"] = round(projected_belief, 6)
        projected["dominant_phase"] = belief_data.get("dominant_phase")
        entries.append(projected)
    entries.sort(key=lambda item: item.get("probability", 0.0), reverse=True)
    return entries


def _paths_from_persona_scores(
    ps: Dict[str, Dict[str, Any]], k_top: int = 8
) -> List[Dict[str, Any]]:
    if not ps:
        return []
    rows = []
    scores = []
    for pid, row in ps.items():
        s = float(row.get("strength", 0.0))
        if s == 0.0:
            s = 0.5 * float(row.get("activation", 0.0))
        scores.append(s)
        rows.append(
            {
                "path": [pid],
                "personas": [pid],
                "score": s,
                "probability": 0.0,
            }
        )
    if not rows:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    Z = sum(exps) or 1.0
    out = []
    for (s, r), e in zip(rows, exps):
        r["probability"] = float(e / Z)
        out.append(r)
    out.sort(key=lambda r: r["probability"], reverse=True)
    return out[: max(1, k_top)]


def _edge_adjustments_from_recommendations(
    rows: Optional[Iterable[Dict[str, Any]]]
) -> List[EdgeAdjustment]:
    adjustments: List[EdgeAdjustment] = []
    for row in rows or []:
        src = row.get("u") or row.get("source_id") or row.get("from")
        dst = row.get("v") or row.get("target_id") or row.get("to")
        if not src or not dst:
            continue
        delta = float(row.get("delta") or row.get("weight_delta") or 0.0)
        confidence = row.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = None
        adjustments.append(
            EdgeAdjustment(
                source_id=str(src),
                target_id=str(dst),
                delta=delta,
                rationale=row.get("rationale") or row.get("reason"),
                confidence=confidence,
                diagnostics={
                    k: v
                    for k, v in row.items()
                    if k
                    not in {
                        "u",
                        "v",
                        "from",
                        "to",
                        "source_id",
                        "target_id",
                        "delta",
                        "weight_delta",
                        "confidence",
                        "rationale",
                        "reason",
                    }
                },
            )
        )
    return adjustments


def _node_adjustments_from_recommendations(
    rows: Optional[Iterable[Dict[str, Any]]]
) -> List[NodeAdjustment]:
    adjustments: List[NodeAdjustment] = []
    for row in rows or []:
        node_id = row.get("node_id") or row.get("persona_id") or row.get("id")
        if not node_id:
            continue
        field = row.get("field") or row.get("attribute") or "likelihood"
        delta = float(row.get("delta") or row.get("weight_delta") or 0.0)
        confidence = row.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = None

        adjustments.append(
            NodeAdjustment(
                node_id=str(node_id),
                field=str(field),
                delta=delta,
                band=row.get("band"),
                rationale=row.get("rationale") or row.get("reason"),
                confidence=confidence,
                diagnostics={
                    k: v
                    for k, v in row.items()
                    if k
                    not in {
                        "node_id",
                        "persona_id",
                        "id",
                        "field",
                        "attribute",
                        "delta",
                        "weight_delta",
                        "confidence",
                        "band",
                        "rationale",
                        "reason",
                    }
                },
            )
        )
    return adjustments


# ---------------------------------------------------------------------
# Replay & learn persona paths (now records per-step paths/expected_next)
# ---------------------------------------------------------------------
def replay_learn_persona_paths(
    *,
    persona_graph: PersonaGraph,
    observed_persona_ids: List[str],  # chronological persona ids
    steps_top_k: int = 5,
    do_learning: bool = True,
    kappa: float = 0.2,
    engagement_events: Optional[List[Dict[str, Any]]] = None,
    product_id: Optional[str] = None,
    account_id: Optional[str] = None,
    collect_diagnostics: bool = False,
) -> Dict[str, Any]:
    """
    For each t:
      - Predict before observing t (expected_next, paths, metrics)
      - Compare to observed_next; bucket outcome
      - Optionally learn from the prefix [0..t] (online)
    """
    PG = persona_graph
    engaged: List[str] = []
    results: List[Dict[str, Any]] = []

    for t in range(len(observed_persona_ids)):
        # 1) predict BEFORE observing step t
        snap = predict_snapshot(
            PG, engaged_personas=engaged, k_next=steps_top_k
        )
        predicted = [row["persona"] for row in snap["expected_next"]]
        observed_next = observed_persona_ids[t]

        # 2) bucket
        if not predicted:
            bucket = "no_path"
        elif observed_next == predicted[0]:
            bucket = "on_path"
        elif observed_next in predicted:
            bucket = "near_path"
        else:
            bucket = (
                "off_path_known"
                if observed_next in PG._personas()
                else "out_of_graph"
            )

        engagement_meta = None
        if engagement_events and t < len(engagement_events):
            engagement_meta = _engagement_meta_from_event(engagement_events[t])

        results.append(
            {
                "t": t,
                "state_personas": list(engaged),
                "predicted_topK": snap["expected_next"],
                "observed_next": observed_next,
                "bucket": bucket,
                "walk_paths": snap["walk_paths"],
                "metrics": snap["metrics"],
                "hit_at_1": observed_next
                == (predicted[0] if predicted else None),
                "hit_at_3": observed_next
                in (predicted[:3] if predicted else []),
                "engagement_meta": engagement_meta,
            }
        )

        # 3) advance
        engaged.append(observed_next)

        # 4) optional learning (small, local)
        if do_learning and len(engaged) >= 2:
            PG.learn_from_sequence(
                observed_personas_in_order=engaged[-2:],
                observed_products=None,
                kappa=kappa,
            )

    # aggregates
    n = max(1, len(results))
    hit1 = sum(1 for r in results if r["hit_at_1"]) / n
    hit3 = sum(1 for r in results if r["hit_at_3"]) / n
    bucket_counts: Dict[str, int] = {}
    for r in results:
        bucket_counts[r["bucket"]] = bucket_counts.get(r["bucket"], 0) + 1

    learning_payload = None
    if collect_diagnostics and product_id and account_id:
        transitions = shm_graph_learner.build_transition_dataset(observed_persona_ids)
        learning_payload = shm_graph_learner.summarize_learning_signals(
            persona_graph=persona_graph,
            transitions=transitions,
            journey_steps=results,
        )

    return {
        "steps": results,
        "metrics": {
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "buckets": bucket_counts,
        },
        "learning_recommendations": learning_payload,
    }


def run_replay_learning_job(
    product_id: str,
    *,
    account_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Batch replay across accounts to produce diagnostics for user review.
    """
    product_graph = build_product_graph(product_id)
    accounts = account_ids or (get_target_account_ids(product_id) or [])
    if limit is not None:
        accounts = accounts[:limit]

    results: List[Dict[str, Any]] = []
    for account_id in accounts:
        try:
            engagements = get_account_engagements(product_id, account_id) or []
            timeline = _engagement_timeline_with_personas(
                product_id, account_id, engagements=engagements
            )
            observed = [
                step["effective_persona_id"]
                for step in timeline
                if step.get("effective_persona_id")
            ]
            if not observed:
                results.append(
                    {
                        "account_id": account_id,
                        "note": "No observed personas for this account",
                    }
                )
                continue

            persona_graph = PersonaGraph.from_product_graph(
                product_graph,
                combine="noisy_or",
                normalize_outgoing=True,
                prior_offpath_rate=_estimate_offpath_rate(account_id, product_id),
                dirichlet_kappa=1.0,
            )
            replay = replay_learn_persona_paths(
                persona_graph=persona_graph,
                observed_persona_ids=observed,
                engagement_events=timeline,
                product_id=product_id,
                account_id=account_id,
                collect_diagnostics=True,
            )
            results.append(
                {
                    "account_id": account_id,
                    "metrics": replay.get("metrics"),
                    "recommendations": replay.get("learning_recommendations"),
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            results.append(
                {
                    "account_id": account_id,
                    "error": str(exc),
                }
            )

    summary = {
        "product_id": product_id,
        "accounts_processed": len(results),
        "results": results,
    }
    if output_path:
        import json

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    return summary

#----- SHM Episode Events ------------------------------

def _episode_events_from_journey(
    *,
    product_id: str,
    account_id: str,
    journey: Dict[str, Any],
    source: str = "belief_thesis_v2",
) -> List[Dict[str, Any]]:
    """
    Turn the per-step journey replay into SHM-compatible 'episode events'
    that the Bayesian journey learner understands.

    Each event is a dict with:
      - persona_id
      - belief_state (coarse string)
      - belief_score (0..1; simple proxy for now)
      - engagement_type / source / channel
      - effect_bucket (on_path / near_path / off_path / no_path)
      - timestamp (best-effort; we let SHM default if missing)
      - meta (full step blob for debugging)
    """
    steps = list((journey or {}).get("steps", []) or [])
    events: List[Dict[str, Any]] = []

    for step in steps:
        persona_id = step.get("observed_next")
        if not persona_id:
            # nothing to learn from this step
            continue

        bucket_raw = step.get("bucket")
        effect_bucket = _map_step_bucket(bucket_raw) or "unknown"

        # Very simple belief_state for now – later we can map to your
        # actual belief state machine (ZMOT, Discovery, Evaluation, ...)
        belief_state = "journey_replay"

        # Lightweight proxy belief_score: 1.0 if hit@1, 0.7 if hit@3,
        # otherwise 0.3. You can refine this later.
        hit1 = bool(step.get("hit_at_1"))
        hit3 = bool(step.get("hit_at_3"))
        if hit1:
            belief_score = 1.0
        elif hit3:
            belief_score = 0.7
        else:
            belief_score = 0.3

        events.append(
            {
                "product_id": product_id,
                "account_id": account_id,
                "persona_id": persona_id,
                "belief_state": belief_state,
                "belief_score": belief_score,
                "engagement_type": source,  # treated as 'type' of episode
                "source": source,
                "channel": "inference",      # not a real channel; fine for stats
                "effect_bucket": effect_bucket,
                "timestamp": None,           # SHM service will fill utcnow()
                "meta": {
                    "t": step.get("t"),
                    "state_personas": step.get("state_personas"),
                    "predicted_topK": step.get("predicted_topK"),
                    "walk_paths": step.get("walk_paths"),
                    "metrics": step.get("metrics"),
                    "hit_at_1": hit1,
                    "hit_at_3": hit3,
                },
            }
        )

    return events


#----------------------------
# Incremental Learning
#---------------------------
def incremental_learnings_from_thesis(thesis: dict) -> dict:
    ls = thesis.get("learning_summary") or {}
    nbs = thesis.get("learning_neighborhoods") or {}

    return {
        "persona_path_inferences": ls.get("persona_graph_inferences", []),
        "graph_level_inferences": ls.get("full_graph_inferences", []),
        "neighborhoods": {
            "upstream": list((nbs.get("upstream") or {}).values()),
            "handoff": list((nbs.get("handoff") or {}).values()),
            "downstream": list((nbs.get("downstream") or {}).values()),
        },
        "account_adjustment": ls.get("account_adjustment")
        or (thesis.get("journey") or {}).get("account_adjustment"),
    }

# ---------------------------------------------------------------------
# SHM + Bayesian Updates
#-----------------------------------------------------------------------
def get_journey_prediction_for_account(
    db: Session,
    *,
    product_id: str,
    account_id: str,
    current_persona_id: str,
    current_state: str,
    k: int = 3,
) -> Dict[str, Any]:
    """
    Use the learned journey weights (from SHM episodes) to predict the
    next top-K personas / states for an account at its current position.

    If no weights exist yet for this product, we rebuild them from SHM.
    """

    # 1) Ensure weights exist; rebuild from SHM if needed
    weights = load_weights(product_id)
    if not weights:
        rebuild_stats_from_shm(db, product_id=product_id)
        weights = load_weights(product_id) or {}

    # 2) Build a node id / state representation to feed into next_distribution.
    #    For now: node_id == persona_id, and states are carried separately.
    start_node = current_persona_id

    dist = next_distribution(
        weights=weights,
        start_node=start_node,
        start_state=current_state,
        top_k=k,
    )

    return {
        "account_id": account_id,
        "product_id": product_id,
        "current_persona_id": current_persona_id,
        "current_state": current_state,
        "predicted_next": dist,
    }


# ---------------------------------------------------------------------
# ORCHESTRATOR (v2): core learning layer
# ---------------------------------------------------------------------
def compute_belief_thesis_core(
    *,
    product_graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    account_id: str,
    account_meta: Optional[Dict[str, Any]] = None,
    account_segment_keys: Optional[Sequence[str]] = None,
    past_engagements: Optional[List[Dict[str, Any]]] = None,
    alpha: float = 0.85,  # kept for parity; used via graphwin_runtime
    weight_key: str = "likelihood",
    debug: bool = False,
    enable_pg_online_learning: bool = True,
    enable_graphstore_learning: bool = True,
    learning_kappa: float = 0.2,
) -> BeliefThesisCore:
    """
    Core computation for a belief thesis + learning, independent of UI.

    This is the "learning layer":
      - persona resolution & observed personas
      - PersonaGraph snapshots/journey
      - GraphStore-based Bayesian learning
      - optional: syncing posterior means back into the product_graph
      - optional: recording SHM learning episodes

    It returns a `BeliefThesisCore` which can be re-used by different UIs.
    """
    if debug:
        print("Starting belief thesis core build")

    product_id = get_product_id_from_subgraph(product_graph)
    if not product_id:
        raise ValueError("No product_id found in product_graph")

    segment_keys: List[str] = list(account_segment_keys or [])
    wolves_scores = _load_persona_wolves_scores(product_id)

    shm_metrics: Dict[str, Any] = {}
    try:
        shm_metrics = get_shm_transition_metrics(product_id)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"⚠️ Unable to load SHM transition metrics for {product_id}: {exc}")
        shm_metrics = {}

    # 1) resolver aware observed personas (reuse existing engagements if given)
    observed_persona_matches = _observed_personas_from_engagements(
        product_id, account_id, engagements=past_engagements
    )

    unique_persona_ids: List[str] = []
    for p in (observed_persona_matches or []):
        pid = p.get("effective_persona_id") or p.get("best")
        if pid:
            unique_persona_ids.append(pid)

    if debug:
        print(
            f"Observed personas for account {account_id}: {unique_persona_ids}"
        )
    print("Compute Belief Thesis Core: completed 1a")
    # 1b) summarize how confident the resolver is about this account
    persona_resolution_stats = _persona_resolution_stats(
        observed_persona_matches
    )

    # Build full engagement timeline (per engagement, not just first sightings)
    timeline = _engagement_timeline_with_personas(
        product_id, account_id, engagements=past_engagements
    )

    observed_persona_ids: List[str] = [
        step["effective_persona_id"]
        for step in timeline
        if step.get("effective_persona_id")
    ]
    print("Compute Belief Thesis Core: completed 1b")
    # 2) build PersonaGraph (reduced graph)
    PG = PersonaGraph.from_product_graph(
        product_graph,
        combine="noisy_or",  # or "sum"
        normalize_outgoing=True,
        prior_offpath_rate=_estimate_offpath_rate(account_id, product_id),
        dirichlet_kappa=1.0,
    )
    if shm_metrics:
        _apply_shm_metrics_to_persona_graph(PG, shm_metrics)

    if debug:
        print(
            "Built PersonaGraph from product graph with Nodes:",
            PG.G.number_of_nodes(),
            "Edges:",
            PG.G.number_of_edges(),
        )
    print("Compute Belief Thesis Core: completed 3")
    # 3) baseline snapshot (zero evidence)
    baseline = predict_snapshot(PG, engaged_personas=[], k_next=5)
    def _annotate_paths_with_metrics(paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        annotated = []
        for entry in paths or []:
            persona_ids = list(entry.get("personas") or entry.get("path") or [])
            belief_states: List[Dict[str, Any]] = []
            for pid in persona_ids:
                node = PG.G.nodes.get(pid, {})
                belief_states.append(
                    {
                        "persona_id": pid,
                        "perceptibility": node.get("perceptibility"),
                        "proximity": node.get("proximity"),
                        "involvement": node.get("involvement"),
                        "activation": node.get("activation"),
                        "strength": node.get("strength"),
                    }
                )
            annotated.append({**entry, "belief_states": belief_states})
        return annotated

    baseline_paths = _annotate_paths_with_metrics(baseline["walk_paths"])
    persona_phase_lookup = _persona_phase_lookup(baseline_paths)
    persona_phase_mass: Dict[str, Counter[str]] = defaultdict(Counter)
    baseline_expected_next_raw = baseline.get("expected_next", []) or []
    baseline_expected_next = _apply_subsidy_boosts_to_expected_next(
        baseline_expected_next_raw,
        account_id=account_id,
        segment_keys=segment_keys,
    )
    baseline_expected_next = _boost_expected_next_rows(
        baseline_expected_next,
        wolves_scores,
        account_id=account_id,
        segment_keys=segment_keys,
        product_graph=product_graph,
    )

    if debug:
        print("Computed baseline snapshot for belief thesis")
    print("Compute Belief Thesis Core: completed 4")
    # 4) Per-engagement prediction → observation → learning loop
    journey_steps: List[Dict[str, Any]] = []
    episodes: List[Episode] = []
    account_adjustment: Optional[LocalAdjustment] = None
    hit1_count = 0
    hit3_count = 0
    bucket_counts: Dict[str, int] = {}
    engaged_sequence: List[str] = []
    log_likelihood_total = 0.0
    brier_total = 0.0
    transition_counts: Dict[Tuple[str, str], int] = {}
    candidate_persona_counts: Dict[str, int] = defaultdict(int)
    prev_observed_persona: Optional[str] = None

    for idx, event in enumerate(timeline):
        ts: datetime = event["timestamp"]
        engagement_obj = event["engagement"]
        actual_persona = event.get("effective_persona_id")

        snap = predict_snapshot(
            PG,
            engaged_personas=engaged_sequence,
            k_next=5,
        )
        predicted_topk_raw = snap.get("expected_next", []) or []
        predicted_topk = _apply_subsidy_boosts_to_expected_next(
            predicted_topk_raw,
            account_id=account_id,
            segment_keys=segment_keys,
            timestamp=ts,
        )
        predicted_topk = _boost_expected_next_rows(
            predicted_topk,
            wolves_scores,
            account_id=account_id,
            segment_keys=segment_keys,
            product_graph=product_graph,
        )
        predicted_distribution_base = _probability_map_from_rows(predicted_topk_raw)
        predicted_distribution_boosted = _probability_map_from_rows(predicted_topk)
        predicted_distribution = (
            predicted_distribution_boosted or predicted_distribution_base
        )
        hidden_state_prior_base = predicted_distribution_base or predicted_distribution
        hidden_state_prior_boosted = predicted_distribution_boosted or predicted_distribution
        predicted_personas = list(predicted_distribution.keys())
        walk_paths = snap.get("walk_paths", [])
        metrics = snap.get("metrics", {})

        if not predicted_personas:
            bucket = "no_path"
        elif actual_persona is None:
            bucket = "no_persona"
        elif actual_persona == predicted_personas[0]:
            bucket = "on_path"
        elif actual_persona in predicted_personas:
            bucket = "near_path"
        elif actual_persona and actual_persona in PG._personas():
            bucket = "off_path_known"
        else:
            bucket = "out_of_graph"

        hit_at_1 = bool(actual_persona and predicted_personas and actual_persona == predicted_personas[0])
        hit_at_3 = bool(actual_persona and actual_persona in predicted_personas[:3])
        if hit_at_1:
            hit1_count += 1
        if hit_at_3:
            hit3_count += 1
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        candidate_label = _engagement_attr(engagement_obj, "candidate_persona_label")
        if candidate_label:
            normalized_candidate = str(candidate_label).strip().lower()
            if normalized_candidate:
                candidate_persona_counts[normalized_candidate] += 1

        prob_actual = (predicted_distribution_base or predicted_distribution).get(actual_persona or "", 0.0)
        log_loss = None
        brier_score = None
        if actual_persona:
            prob = max(prob_actual, 1e-9)
            log_loss = -math.log(prob)
            brier = 0.0
            for persona, p_val in hidden_state_prior_base.items():
                target = 1.0 if persona == actual_persona else 0.0
                brier += (p_val - target) ** 2
            remainder_base = max(0.0, 1.0 - sum(hidden_state_prior_base.values()))
            if remainder_base > 0:
                target = 0.0 if actual_persona in hidden_state_prior_base else 1.0
                brier += (remainder_base - target) ** 2
            brier_score = brier

        prior_for_posterior = predicted_distribution_base or predicted_distribution_boosted
        if actual_persona:
            smoothing = 0.75
            posterior_distribution = {}
            for persona, prob in prior_for_posterior.items():
                if persona == actual_persona:
                    posterior_distribution[persona] = prob + (1.0 - prob) * smoothing
                else:
                    posterior_distribution[persona] = prob * (1.0 - smoothing)
            Z = sum(posterior_distribution.values()) or 1.0
            posterior_distribution = {
                persona: val / Z for persona, val in posterior_distribution.items()
            }
        else:
            posterior_distribution = dict(prior_for_posterior)
        hidden_state_prior = posterior_distribution
        hidden_state_prior_boosted = predicted_distribution_boosted or predicted_distribution

        for persona_id, prob in posterior_distribution.items():
            if not persona_id:
                continue
            phase_label = _canonical_phase(persona_phase_lookup.get(persona_id))
            persona_phase_mass[persona_id][phase_label] += float(prob)
            persona_phase_lookup.setdefault(persona_id, phase_label)

        prediction_record = PersonaPrediction(
            account_id=account_id,
            timestamp=ts,
            persona_sequence_before=list(engaged_sequence),
            predicted_distribution=predicted_distribution,
            predicted_topk=predicted_personas,
            rationale=None,
            candidate_paths=[
                list(path.get("personas") or path.get("path") or [])
                for path in walk_paths or []
            ],
        )

        log_likelihood_contrib = None
        if actual_persona:
            prob_obs = max(predicted_distribution.get(actual_persona, 0.0), 1e-9)
            log_likelihood_contrib = math.log(prob_obs)
            log_likelihood_total += log_likelihood_contrib

        error_record = PredictionError(
            account_id=account_id,
            timestamp=ts,
            persona_sequence_before=list(engaged_sequence),
            predicted_distribution=predicted_distribution,
            predicted_topk=predicted_personas,
            actual_persona=actual_persona,
            hit_at_1=hit_at_1,
            hit_at_3=hit_at_3,
            bucket=bucket,
            log_loss=log_loss,
            brier_score=brier_score,
            rationale=None,
            log_likelihood=log_likelihood_contrib,
        )

        if brier_score is not None:
            brier_total += brier_score

        edge_evidence: List[Dict[str, Any]] = []
        sorted_paths = sorted(
            walk_paths or [], key=lambda p: p.get("probability", 0.0), reverse=True
        )
        for path in sorted_paths[:3]:
            personas = list(path.get("personas") or path.get("path") or [])
            if len(personas) < 2:
                continue
            path_prob = float(path.get("probability") or path.get("score") or 0.0)
            per_edge_log = math.log(max(path_prob, 1e-9))
            for u, v in zip(personas[:-1], personas[1:]):
                supports = actual_persona == v
                edge_evidence.append(
                    {
                        "from": str(u),
                        "to": str(v),
                        "supports": supports,
                        "delta_log_prob": per_edge_log if supports else -per_edge_log,
                        "path_probability": path_prob,
                    }
                )
            if len(edge_evidence) >= 12:
                break

        if isinstance(metrics, dict):
            err_payload = metrics.get("error") if isinstance(metrics.get("error"), dict) else {}
            err_payload = {
                **(err_payload or {}),
                "log_loss": log_loss,
                "brier_score": brier_score,
                "log_likelihood": log_likelihood_contrib,
            }
            metrics["error"] = err_payload
            metrics["hidden_state_prior"] = hidden_state_prior
            metrics["hidden_state_prior_boosted"] = hidden_state_prior_boosted
            metrics["hidden_state_posterior"] = posterior_distribution
            metrics["edge_evidence"] = edge_evidence
        else:
            metrics = {
                "hidden_state_prior": hidden_state_prior,
                "hidden_state_prior_boosted": hidden_state_prior_boosted,
                "hidden_state_posterior": posterior_distribution,
                "edge_evidence": edge_evidence,
                "error": {
                    "log_loss": log_loss,
                    "brier_score": brier_score,
                    "log_likelihood": log_likelihood_contrib,
                },
            }

        local_adjustment = LocalAdjustment(
            account_id=account_id,
            timestamp=ts,
            band=None,
            edges=[],
            nodes=[],
            error=error_record,
            diagnostic={
                "metrics": metrics,
                "bucket": bucket,
                "prob_actual": prob_actual,
                "resolution_status": event.get("resolution_status"),
                "hidden_state_prior": hidden_state_prior,
                "hidden_state_posterior": posterior_distribution,
                "log_likelihood_contrib": log_likelihood_contrib,
                "edge_evidence": edge_evidence,
            },
            global_params_version=None,
        )

        if isinstance(engagement_obj, dict):
            engagement_payload = engagement_obj
        else:
            engagement_payload = {"raw": _json_safe(engagement_obj)}

        episode = Episode(
            account_id=account_id,
            timestamp=ts,
            meta={
                "sequence_index": idx,
                "resolution_status": event.get("resolution_status"),
            },
            account_meta=_json_safe(account_meta) if account_meta else None,
            engagement=engagement_payload,
            prediction=prediction_record,
            actual_persona=actual_persona,
            error=error_record,
            local_adjustment=local_adjustment,
            global_params_version=None,
            candidate_personas=dict(candidate_persona_counts),
        )
        episodes.append(episode)

        engagement_meta = {
            "asset_id": _engagement_attr(engagement_obj, "asset_id"),
            "channel": _engagement_attr(engagement_obj, "channel")
            or _engagement_attr(engagement_obj, "source"),
            "source": _engagement_attr(engagement_obj, "source"),
            "raw_activity": _engagement_attr(engagement_obj, "raw_activity"),
            "label": _engagement_attr(engagement_obj, "name")
            or _engagement_attr(engagement_obj, "title")
            or _engagement_attr(engagement_obj, "raw_activity"),
        }
        engagement_meta = {
            k: v for k, v in engagement_meta.items() if v is not None
        }

        journey_steps.append(
            {
                "t": idx,
                "timestamp": ts.isoformat(),
                "state_personas": list(engaged_sequence),
                "predicted_topK": predicted_topk,
                "predicted_topK_base": predicted_topk_raw,
                "observed_next": actual_persona,
                "bucket": bucket,
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "walk_paths": walk_paths,
                "metrics": metrics,
                "prediction": _json_safe(prediction_record),
                "error": _json_safe(error_record),
                "local_adjustment": _json_safe(local_adjustment),
                "hidden_state_prior": hidden_state_prior,
                "hidden_state_prior_boosted": hidden_state_prior_boosted,
                "hidden_state_posterior": posterior_distribution,
                "edge_evidence": edge_evidence,
                "log_likelihood_contrib": log_likelihood_contrib,
                "brier_contrib": brier_score,
                "engagement_meta": _json_safe(engagement_meta)
                if engagement_meta
                else None,
                "asset_id": engagement_meta.get("asset_id")
                if engagement_meta
                else None,
                "channel": engagement_meta.get("channel")
                if engagement_meta
                else None,
                "account_meta": _json_safe(account_meta) if account_meta else None,
            }
        )

        if actual_persona:
            engaged_sequence.append(actual_persona)
            phase_label = _canonical_phase(persona_phase_lookup.get(actual_persona))
            persona_phase_mass[actual_persona][phase_label] += 1.0
            persona_phase_lookup.setdefault(actual_persona, phase_label)
            if enable_pg_online_learning and len(engaged_sequence) >= 2:
                PG.learn_from_sequence(
                    observed_personas_in_order=engaged_sequence[-2:],
                    observed_products=None,
                    kappa=learning_kappa,
                )
            if prev_observed_persona:
                key = (prev_observed_persona, actual_persona)
                transition_counts[key] = transition_counts.get(key, 0) + 1
            prev_observed_persona = actual_persona

    candidate_personas_dict = dict(candidate_persona_counts)

    if journey_steps:
        total_steps = len(journey_steps)
        journey_metrics = {
            "hit_at_1": hit1_count / total_steps,
            "hit_at_3": hit3_count / total_steps,
            "buckets": bucket_counts,
            "avg_log_likelihood": log_likelihood_total / total_steps,
            "avg_brier": brier_total / total_steps if total_steps else None,
        }
    else:
        journey_metrics = {
            "hit_at_1": None,
            "hit_at_3": None,
            "buckets": {},
            "avg_log_likelihood": None,
            "avg_brier": None,
        }

    journey = {
        "steps": journey_steps,
        "metrics": journey_metrics,
        "episodes": episodes,
        "candidate_personas": candidate_personas_dict,
        "sufficient_stats": {
            "transition_counts": [
                {"from": src, "to": dst, "count": count}
                for (src, dst), count in sorted(
                    transition_counts.items(), key=lambda kv: kv[1], reverse=True
                )
            ]
        },
    }

    if account_adjustment is not None:
        journey["account_adjustment"] = _json_safe(account_adjustment)

    journey["episodes"] = [_json_safe(ep) for ep in episodes]

    observed_persona_ids = list(engaged_sequence)
    print("Compute Belief Thesis Core: completed 4")
    # 5) current snapshot (after all observed) – note this uses the (possibly
    #    updated) PersonaGraph if online learning was enabled.
    print("Compute Belief Thesis Core: Step 5: Starting Predict Snapshot")
    current = predict_snapshot(
        PG, engaged_personas=observed_persona_ids, k_next=5
    )
    print("Compute Belief Thesis Core: Step 5: Predicted Snapshot")
    current_paths = _annotate_paths_with_metrics(current["walk_paths"])
    for pid, phase in _persona_phase_lookup(current_paths).items():
        persona_phase_lookup.setdefault(pid, phase)
    current_expected_next_raw = current.get("expected_next", []) or []
    current_expected_next = _apply_subsidy_boosts_to_expected_next(
        current_expected_next_raw,
        account_id=account_id,
        segment_keys=segment_keys,
    )
    current_expected_next = _boost_expected_next_rows(
        current_expected_next,
        wolves_scores,
        account_id=account_id,
        segment_keys=segment_keys,
        product_graph=product_graph,
    )
    print("Compute Belief Thesis Core: completed 5")
    # 6) GraphStore-based learning (Bayesian) over the full product graph
    learning_summary: Optional[Dict[str, Any]] = None
    raw_diffs: Optional[Dict[str, Any]] = None
    neighborhoods: Optional[Dict[str, Any]] = None
    learning_apply_errors: List[str] = []
    learning_store: Optional[GraphStore] = None
    journey_learning: Optional[Dict[str, Any]] = None 
    try:
        # ---------------------------
        # 6a) GraphStore global learning
        # ---------------------------
        graphstore_summary: Optional[Dict[str, Any]] = None

        if enable_graphstore_learning:
            # materialize GraphStore from *current* product_graph
            learning_store = materialize_graphstore_from_networkx(product_graph)

            # expected starting personas from baseline snapshot
            expected_topk_start = _expected_topk_from_baseline(
                baseline_expected_next
            )

            if observed_persona_ids and expected_topk_start:
                # ---- Injection hooks for learn_and_summarize ------------------
                def _persona_match_score(pid: str) -> float:
                    return _persona_match_score_for_run(
                        pid, observed_persona_ids=observed_persona_ids
                    )

                def _pick_job_for_persona(pid: str) -> str:
                    best_job: Optional[str] = None
                    best_w: float = -1.0
                    if pid in product_graph:
                        for _, v, data in product_graph.out_edges(
                            pid, data=True
                        ):
                            v_str = str(v)
                            if not v_str.startswith("job:"):
                                continue
                            w = float(
                                data.get(weight_key, data.get("likelihood", 0.0))
                                or 0.0
                            )
                            if w > best_w:
                                best_w = w
                                best_job = v_str
                    return best_job or "job:unknown"

                def _dominant_jobs(pid: str) -> List[str]:
                    jobs: List[Tuple[str, float]] = []
                    if pid in product_graph:
                        for _, v, data in product_graph.out_edges(
                            pid, data=True
                        ):
                            v_str = str(v)
                            if not v_str.startswith("job:"):
                                continue
                            w = float(
                                data.get(weight_key, data.get("likelihood", 0.0))
                                or 0.0
                            )
                            jobs.append((v_str, w))
                    jobs.sort(key=lambda t: t[1], reverse=True)
                    return [j for j, _ in jobs[:3]]

                def _candidate_pains_into_job(
                    job_id: str,
                ) -> List[Tuple[str, float]]:
                    pains: List[Tuple[str, float]] = []
                    if job_id in product_graph:
                        for u, _, data in product_graph.in_edges(
                            job_id, data=True
                        ):
                            u_str = str(u)
                            if not u_str.startswith("pain:"):
                                continue
                            w = float(
                                data.get(weight_key, data.get("likelihood", 0.0))
                                or 0.0
                            )
                            pains.append((u_str, w))
                    pains.sort(key=lambda t: t[1], reverse=True)
                    return pains

                def _infer_pain_between(
                    job_from: str, p_from: str, p_to: str
                ) -> List[Tuple[str, float]]:
                    # For now, reuse pains attached to job_from as the bridge
                    return _candidate_pains_into_job(job_from)

                # Run graph-level Bayesian learner (mutates learning_store).
                # Note: we do NOT hard-code any "gatekeeper" personas here;
                # gatekeeper-ness is learned empirically from early appearances
                # inside the graph_learning layer.
                graphstore_summary = learn_and_summarize(
                    learning_store,
                    account_id=account_id,
                    observed_path=observed_persona_ids,
                    expected_topk_start=expected_topk_start,
                    persona_match_score=_persona_match_score,
                    pick_job_for_persona=_pick_job_for_persona,
                    dominant_jobs=_dominant_jobs,
                    infer_pain_between=_infer_pain_between,
                    candidate_pains_into_job=_candidate_pains_into_job,
                )


                # Sync posterior means back into NetworkX graph
                from backend.utils.inference.belief_manager.learn.materialize import (
                    sync_edge_means_to_networkx,
                )

                sync_edge_means_to_networkx(
                    learning_store,
                    product_graph,
                    write_key=weight_key,
                    also_write_alpha_beta=True,
                )
            else:
                graphstore_summary = {
                    "note": "insufficient_data_for_learning",
                    "account_id": account_id,
                    "product_id": product_id,
                }
        else:
            graphstore_summary = {
                "note": "learning_disabled",
                "account_id": account_id,
                "product_id": product_id,
            }
        print("Compute Belief Thesis Core: completed 6a")
        # ---------------------------
        # 6b) Local Bayesian diffs (v3) from persona-path mismatch
        # ---------------------------
        diff_result: Optional[Dict[str, Any]] = None
        if baseline_expected_next and observed_persona_ids:
            # baseline_expected_next is already [{persona, prob}] shape in core,
            # but we normalize / coerce just to be safe
            baseline_for_diffs = [
                {
                    "persona": str(r.get("persona")),
                    "prob": float(r.get("prob", 0.0)),
                }
                for r in (baseline_expected_next_raw or [])
                if r.get("persona")
            ]

            diff_result = infer_graph_updates_from_diffs(
                product_graph=product_graph,
                baseline_expected_next=baseline_for_diffs,
                observed_persona_ids=observed_persona_ids,
                weight_key=weight_key,
                allocation="prior_weighted",
                learn_cfg=SMALL_SAMPLE_CFG,
            )

            raw_diffs = (diff_result or {}).get("diffs") or {}
            neighborhoods = (diff_result or {}).get("neighborhoods") or {}
            diff_summary = (diff_result or {}).get("summary") or {}

            # --- normalize learning outputs for UI / FE consumers ---

            # GraphStore summary usually contains persona-level & global graph inferences.
            gs = graphstore_summary or {}
            persona_graph_inferences = (
                gs.get("persona_graph_inferences")
                or gs.get("persona_inferences")
                or []
            )
            full_graph_inferences = (
                gs.get("full_graph_inferences")
                or gs.get("global_graph_inferences")
                or []
            )

            # Diff learner summary usually contains edge updates
            ds = diff_summary or {}
            recs = (ds.get("recommendations") or {})
            learned = (ds.get("learned") or {})

            edge_updates_ranked = recs.get("edge_updates_ranked") or []
            node_updates_ranked = recs.get("node_updates_ranked") or []

            persona_graph_inferences = learned.get("A_persona_path_theses", [])
            full_graph_inferences = (
                (learned.get("B_latent_node_theses") or [])
                + (learned.get("C_persona_product_theses") or [])
            )

            learning_summary = {
                # raw blobs
                "graphstore_summary": graphstore_summary,
                "graphstore_summary_v3": graphstore_summary,
                "diff_summary_v3": diff_summary,
                # FE-friendly slices
                "persona_graph_inferences": persona_graph_inferences,
                "full_graph_inferences": full_graph_inferences,
                "top_edge_updates": edge_updates_ranked,       # <- now populated
                "node_updates_ranked": node_updates_ranked,
                # leave the nested block too
                "recommendations": recs,
            }

            edge_adjustments = _edge_adjustments_from_recommendations(
                edge_updates_ranked
            )
            node_adjustments = _node_adjustments_from_recommendations(
                node_updates_ranked
            )

            if edge_adjustments or node_adjustments:
                account_adjustment = LocalAdjustment(
                    account_id=account_id,
                    timestamp=datetime.utcnow(),
                    band=None,
                    edges=edge_adjustments,
                    nodes=node_adjustments,
                    error=None,
                    diagnostic={"source": "graph_diff_mapper"},
                    global_params_version=None,
                )

                if episodes:
                    episodes[-1].local_adjustment.edges.extend(edge_adjustments)
                    episodes[-1].local_adjustment.nodes.extend(node_adjustments)
                    # refresh snapshot stored in journey steps
                    journey_steps[-1]["local_adjustment"] = _json_safe(
                        episodes[-1].local_adjustment
                    )
                if learning_summary is not None:
                    learning_summary.setdefault(
                        "account_adjustment", _json_safe(account_adjustment)
                    )
                else:
                    learning_summary = {
                        "account_adjustment": _json_safe(account_adjustment)
                    }

        else:
            # No persona path to learn from; still surface graphstore meta
            gs = graphstore_summary or {}
            persona_graph_inferences = (
                gs.get("persona_graph_inferences")
                or gs.get("persona_inferences")
                or []
            )
            full_graph_inferences = (
                gs.get("full_graph_inferences")
                or gs.get("global_graph_inferences")
                or []
            )

            learning_summary = {
                "graphstore_summary": graphstore_summary,
                "graphstore_summary_v3": graphstore_summary,
                "diff_summary_v3": None,
                "persona_graph_inferences": persona_graph_inferences,
                "full_graph_inferences": full_graph_inferences,
                "top_edge_updates": [],
                "recommendations": {},
            }


    except Exception as e:
        learning_apply_errors.append(str(e))
    print("Compute Belief Thesis Core: completed 6b")
    # -----------------------------------------------------------------
    # 6c) SHM + Bayesian journey learner (episodes + weights)
    # -----------------------------------------------------------------
    try:
        # 1) Build SHM-compatible events from the persona journey
        episode_events = _episode_events_from_journey(
            product_id=product_id,
            account_id=account_id,
            journey=journey,
            source="belief_thesis_v2",
        )

        if episode_events:
            # 2) Update SHM episodes + edge stats + journey weights
            #    This:
            #      - appends SHM steps (SHMEpisode rows)
            #      - loads stats/weights from storage
            #      - calls update_edge_stats(...)
            #      - recomputes weights via recompute_weights(...)
            #      - saves stats + weights back to storage
            #    and returns a per-run summary you can surface if needed.
            journey_learning = update_bayesian_journey_for_account(
                product_id=product_id,
                account_id=account_id,
                episode_events=episode_events,
                base_neighbors=None,  # you can pass persona neighbors later
            )
    except Exception as e:
        learning_apply_errors.append(
            f"journey_learning_failed: {e}"
        )
    # Attach SHM / journey-learning summary if we have one
    if journey_learning is not None:
        if learning_summary is None:
            learning_summary = {}
        learning_summary["journey_learning"] = journey_learning

    # 7) simple “walk_paths” for top personas from current metrics (optional)
    # reuse the PG.beam_paths seeded by the last engaged persona
    simple_walk_paths = PG.beam_paths(
        start_personas=observed_persona_ids[-1:] or PG._personas()[:1]
    )

    fit = _fit_metrics_from_snapshot(
        {
            "observed_persona_ids": observed_persona_ids,
            "walk_paths": simple_walk_paths,
        }
    )

    learning = LearningArtifacts(
        graphstore=learning_store,
        summary=learning_summary,
        diffs=raw_diffs,
        neighborhoods=neighborhoods,
        apply_errors=learning_apply_errors,
    )

    persona_committee_probs = _compute_persona_committee_probs(
        observed_persona_ids,
        timeline,
        current_paths,
        current_expected_next,
        wolves_scores=wolves_scores,
        account_id=account_id,
        segment_keys=segment_keys,
        product_graph=product_graph,
    )
    for pid in persona_committee_probs.keys():
        persona_phase_lookup.setdefault(pid, "Problem Realization")

    persona_belief_posteriors: Dict[str, Dict[str, Any]] = {}
    for persona_id, default_phase in persona_phase_lookup.items():
        counter = persona_phase_mass.get(persona_id, Counter())
        if not counter:
            counter = Counter({default_phase: 1.0})
        total = sum(counter.values()) or 1.0
        normalized = {
            phase: round(value / total, 6)
            for phase, value in counter.items()
        }
        belief_level = round(
            sum(
                BELIEF_PHASE_WEIGHTS.get(phase, 0.5) * prob
                for phase, prob in normalized.items()
            ),
            6,
        )
        persona_belief_posteriors[persona_id] = {
            "phase_probs": normalized,
            "belief_level": belief_level,
            "dominant_phase": _dominant_phase_from_probs(normalized),
        }

    match_rows = _load_account_person_matches(product_id, account_id)
    person_committee_probs = _project_person_committee_probs(
        persona_committee_probs,
        match_rows,
        product_graph,
        persona_belief_posteriors,
    )

    persona_wolf_scores: Dict[str, Dict[str, Any]] = {}
    wolf_persona_ids: Set[str] = set(observed_persona_ids)
    wolf_persona_ids.update(persona_committee_probs.keys())
    for path in baseline_paths:
        for persona in path.get("personas") or path.get("path") or []:
            wolf_persona_ids.add(str(persona))
    for path in current_paths:
        for persona in path.get("personas") or path.get("path") or []:
            wolf_persona_ids.add(str(persona))
    for rows in (baseline_expected_next, current_expected_next):
        for row in rows or []:
            pid = str(row.get("persona") or row.get("persona_id") or "")
            if pid:
                wolf_persona_ids.add(pid)
    now_ts = datetime.utcnow().replace(tzinfo=timezone.utc)
    for persona_id in wolf_persona_ids:
        node_data = dict(PG.G.nodes.get(persona_id, {}))
        node_data.setdefault("id", persona_id)
        node_data.setdefault("canonical_persona_id", node_data.get("canonical_persona_id") or persona_id)
        wolves_entry = (wolves_scores or {}).get(str(persona_id))
        base_wolf = _derive_base_wolf_score(node_data, wolves_entry)
        if account_id:
            dynamic_wolf = wolf_score_dynamic(
                node_data,
                base_wolf,
                account_id,
                segment_keys,
                now_ts,
            )
            subsidy_rel = subsidy_relevance_for_persona(
                node_data,
                account_id,
                segment_keys,
                now_ts,
            )
        else:
            dynamic_wolf = base_wolf
            subsidy_rel = 0.0
        persona_wolf_scores[persona_id] = {
            "base_wolf_score": round(base_wolf, 6),
            "dynamic_wolf_score": round(dynamic_wolf, 6),
            "subsidy_relevance": round(subsidy_rel, 6),
        }

    core = BeliefThesisCore(
        account_id=account_id,
        product_id=product_id,
        persons=observed_persona_matches,
        observed_persona_ids=observed_persona_ids,
        persona_resolution_stats=persona_resolution_stats,
        account_prior_size=0 if account_meta is None else len(account_meta),
        baseline_paths=baseline_paths,
        baseline_expected_next=_serialize_expected_next(baseline_expected_next),
        current_paths=current_paths,
        current_expected_next=_serialize_expected_next(current_expected_next),
        journey=journey,
        walk_paths=simple_walk_paths,
        fit=fit,
        learning=learning,
        persona_committee_probs=persona_committee_probs,
        persona_posteriors=persona_committee_probs,
        person_committee_probs=person_committee_probs,
        persona_belief_posteriors=persona_belief_posteriors,
        persona_wolf_scores=persona_wolf_scores,
    )

    if debug:
        print("Completed belief thesis core build:", core)

    return core


# ---------------------------------------------------------------------
# ORCHESTRATOR (v2): UI / read-only layer
# ---------------------------------------------------------------------
def belief_thesis_to_ui_dict(
    core: BeliefThesisCore, product_graph: nx.DiGraph
) -> Dict[str, Any]:
    """
    Take a BeliefThesisCore (learning-layer) and decorate it for UI:

      - add labels wherever helpful
      - expose learning summary + artifacts with labels
      - keep the exact top-level shape similar to the old build_belief_thesis
    """
    out: Dict[str, Any] = {
        "account_id": core.account_id,
        "product_id": core.product_id,
        "persons": core.persons,
        "observed_persona_ids": core.observed_persona_ids,
        "account_prior_size": core.account_prior_size,
        "baseline_paths": core.baseline_paths,
        "baseline_expected_next": core.baseline_expected_next,
        "current_paths": core.current_paths,
        "current_expected_next": core.current_expected_next,
        "journey": core.journey,
        "walk_paths": core.walk_paths,
        "fit": core.fit,
        "learning_summary": core.learning.summary,
        # For now, we treat the same summary as the "human" view;
        # down the road you can specialize this.
        "human_readable_learning": core.learning.summary,
        "persona_resolution_stats": core.persona_resolution_stats,
        "persona_committee_probs": core.persona_committee_probs,
        "persona_posteriors": core.persona_posteriors,
        "person_committee_probs": core.person_committee_probs,
        "persona_belief_posteriors": core.persona_belief_posteriors,
        "persona_wolf_scores": core.persona_wolf_scores,
    }

    if core.learning.diffs:
        out["learning_raw_diffs"] = core.learning.diffs
    if core.learning.neighborhoods:
        out["learning_neighborhoods"] = core.learning.neighborhoods
    if core.learning.apply_errors:
        out["learning_apply_errors"] = core.learning.apply_errors

    # Add labels for observed personas
    out["observed_persona_labels"] = [
        _L(product_graph, pid) for pid in out["observed_persona_ids"]
    ]

    # Label expected_next + paths + journey
    out["baseline_expected_next"] = _add_expected_next_labels(
        product_graph, out["baseline_expected_next"]
    )
    out["current_expected_next"] = _add_expected_next_labels(
        product_graph, out["current_expected_next"]
    )

    out["baseline_paths"] = _add_path_labels(
        product_graph, out["baseline_paths"]
    )
    out["current_paths"] = _add_path_labels(
        product_graph, out["current_paths"]
    )
    out["walk_paths"] = _add_path_labels(product_graph, out["walk_paths"])

    out["journey"] = _add_journey_labels(
        product_graph, out.get("journey", {})
    )

    # Add labels for persona_resolution_stats persona lists
    prs = out.get("persona_resolution_stats", {})
    if prs:
        prs["stable_persona_labels"] = [
            _L(product_graph, pid) for pid in prs.get("stable_persona_ids", [])
        ]
        prs["tentative_persona_labels"] = [
            _L(product_graph, pid)
            for pid in prs.get("tentative_persona_ids", [])
        ]
        prs["revised_persona_labels"] = [
            _L(product_graph, pid) for pid in prs.get("revised_persona_ids", [])
        ]
        out["persona_resolution_stats"] = prs

    # Label neighborhoods if present (graph_diff_mapper style)
    if "learning_neighborhoods" in out:
        out["learning_neighborhoods"] = _label_neighborhoods(
            product_graph, out["learning_neighborhoods"]
        )

    out["zmot_forecasts"] = _forecast_zmot_events(
        product_graph,
        current_paths=out.get("current_paths") or [],
        observed_personas=out.get("observed_persona_ids") or [],
        max_per_persona=3,
    )

    return out


# ---------------------------------------------------------------------
# Backwards-compatible wrappers
# ---------------------------------------------------------------------
def build_belief_thesis_for_account(
    product_id: str, account_id: str
) -> Dict[str, Any]:
    """
    1) Load product graph.
    2) Load account meta + engagements.
    3) Build belief thesis (core + UI).
    """
    product_graph = build_product_graph(product_id)
    if not product_graph:
        return {
            "note": "no product graph",
            "product_id": product_id,
            "account_id": account_id,
        }

    account = get_account_by_id(product_id, account_id)
    account_meta = []
    account_segment_keys: List[str] = []
    if account:
        account_meta = map_account_meta_to_stable_ids(product_id, account)
        account_segment_keys = segment_keys_from_meta(account)

    engagements = get_account_engagements(product_id, account_id) or []

    thesis = build_belief_thesis(
        product_graph=product_graph,
        original_graph=product_graph,
        account_id=account_id,
        account_meta=account_meta,
        account_segment_keys=account_segment_keys,
        past_engagements=engagements,
        alpha=0.85,
        weight_key="likelihood",
        debug=False,
    )

    # build_belief_thesis already JSON-sanitizes; just return.
    return thesis


def build_belief_thesis(
    *,
    product_graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    account_id: str,
    account_meta: Optional[Dict[str, Any]] = None,
    account_segment_keys: Optional[Sequence[str]] = None,
    past_engagements: Optional[List[Dict[str, Any]]] = None,
    alpha: float = 0.85,  # kept for parity; used via graphwin_runtime
    weight_key: str = "likelihood",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Backwards-compatible wrapper that:

      - runs the core (learning) layer with learning enabled
      - decorates as a UI dict
      - JSON-sanitizes for API responses

    If you want a *pure read* / no-learning view for UI, call
    `compute_belief_thesis_core(..., enable_pg_online_learning=False,
     enable_graphstore_learning=False)` and then
    `belief_thesis_to_ui_dict(core, product_graph)` directly.
    """
    core = compute_belief_thesis_core(
        product_graph=product_graph,
        original_graph=original_graph,
        account_id=account_id,
        account_meta=account_meta,
        account_segment_keys=account_segment_keys,
        past_engagements=past_engagements,
        alpha=alpha,
        weight_key=weight_key,
        debug=debug,
        enable_pg_online_learning=True,
        enable_graphstore_learning=True,
        learning_kappa=0.2,
    )

    print("[belief_manager] Completed belief thesis core computation:")

    out = belief_thesis_to_ui_dict(core, product_graph)
    print()
    print("/n /n #####################################################################")

    print(
        "[belief_manager] Converted belief thesis core to UI dict."
    )

    if debug:
        print("Completed belief thesis build. UI dict:", out)

    bad_path = _scan_nonstring_keys(out)
    if bad_path:
        print(
            f"[belief_manager] WARNING: non-string mapping key detected at {bad_path}. Sanitizing…"
        )

    return _json_safe(out)
def _probability_map_from_rows(rows: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:
    if not rows:
        return {}
    accum: Dict[str, float] = {}
    for row in rows:
        persona_id = str(row.get("persona") or row.get("persona_id") or "")
        if not persona_id:
            continue
        value = row.get("prob")
        if value is None:
            value = row.get("probability")
        accum[persona_id] = accum.get(persona_id, 0.0) + float(value or 0.0)
    total = sum(accum.values())
    if total <= 0:
        return accum
    return {pid: val / total for pid, val in accum.items()}


def _serialize_expected_next(rows: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for row in rows or []:
        persona_id = str(row.get("persona") or row.get("persona_id") or "")
        if not persona_id:
            continue
        prob_val = row.get("prob")
        if prob_val is None:
            prob_val = row.get("probability")
        payload: Dict[str, Any] = {
            "persona": persona_id,
            "prob": float(prob_val or 0.0),
        }
        if "persona_label" in row and row.get("persona_label") is not None:
            payload["persona_label"] = row.get("persona_label")
        if row.get("prob_base") is not None:
            payload["prob_base"] = float(row.get("prob_base") or 0.0)
        if row.get("subsidy_lift") is not None:
            payload["subsidy_lift"] = float(row.get("subsidy_lift") or 0.0)
        if row.get("reason"):
            payload["reason"] = row.get("reason")
        if row.get("journey_stage"):
            payload["journey_stage"] = row.get("journey_stage")
        if row.get("tags"):
            payload["tags"] = row.get("tags")
        serialized.append(payload)
    return serialized
