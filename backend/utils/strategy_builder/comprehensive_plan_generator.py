from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

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
)


# ---------------------------------------------------------------------------
# Constants & filesystem helpers
# ---------------------------------------------------------------------------

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(MODULE_DIR)
GRAPH_BASE_DIR = os.path.join(UTILS_DIR, "graph_base")
ARSENAL_DIR = os.path.join(GRAPH_BASE_DIR, "graph_data", "arsenal_json")


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


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


def _load_assets(product_id: str) -> List[Dict[str, Any]]:
    return _load_catalog("assets.json", product_id)


def _load_channels(product_id: str) -> List[Dict[str, Any]]:
    return _load_catalog("channels.json", product_id)


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
) -> Dict[str, Any]:
    data = get_node_by_id(G, persona_id) or {}
    return {
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
        "path_probability": path_probability,
    }


def _collect_persona_metrics(
    G: nx.DiGraph,
    path: Dict[str, Any],
    metrics_lookup: Optional[Dict[str, Dict[str, float]]] = None,
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

    for order, persona_id in enumerate(path.get("persona_ids") or []):
        metrics: Dict[str, Any] = {}
        if order < len(belief_states) and isinstance(belief_states[order], dict):
            metrics.update(belief_states[order])
        if metrics_lookup:
            for field in ("perceptibility", "proximity", "involvement"):
                value = metrics_lookup.get(field, {}).get(persona_id)
                if value is not None:
                    metrics[field] = value
        personas.append(
            _persona_summary(
                G,
                persona_id,
                order,
                path.get("probability", 0.0),
                belief_metrics=metrics or None,
            )
        )
    return personas


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
# Arsenal recommendation
# ---------------------------------------------------------------------------

def _asset_tokens(asset: Dict[str, Any]) -> set[str]:
    parts: List[str] = []
    for field in ("name", "format", "persona_fit", "concern_tags", "stage_fit"):
        value = asset.get(field)
        if isinstance(value, str):
            parts.extend(_tokenize(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    parts.extend(_tokenize(item))
    return set(parts)


def _score_asset(
    asset: Dict[str, Any],
    persona: Dict[str, Any],
    transition: Dict[str, Any],
) -> Tuple[float, Dict[str, Any], bool, bool, bool]:
    tokens_persona = _tokenize(persona.get("label"), persona.get("title"))
    tokens_job = _tokenize(
        transition.get("job", {}).get("label") if transition.get("job") else ""
    )
    tokens_pain = []
    for pain in transition.get("pains") or []:
        tokens_pain.extend(pain.get("tokens") or [])

    score = 0.0
    reasons: Dict[str, Any] = {}

    persona_fit = asset.get("persona_fit") or []
    persona_alignment = False
    if persona_fit:
        persona_matches = [
            tag for tag in persona_fit if _norm_token(tag) in tokens_persona
        ]
        if persona_matches:
            score += 2.0
            reasons["persona_fit"] = persona_matches
            persona_alignment = True

    concern_tags = asset.get("concern_tags") or []
    concern_alignment = False
    if concern_tags and tokens_pain:
        concern_matches = [
            tag for tag in concern_tags if _norm_token(tag) in tokens_pain
        ]
        if concern_matches:
            score += 1.5
            reasons["concern_fit"] = concern_matches
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
            tag for tag in _asset_tokens(asset) if tag in tokens_job
        ]
        if job_matches:
            score += 0.75
            reasons["job_alignment"] = job_matches[:3]

    evergreen_bonus = 0.25 if asset.get("evergreen") else 0.0
    score += evergreen_bonus

    return score, reasons, stage_alignment, persona_alignment, concern_alignment


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
    *,
    broad_bias: bool,
) -> float:
    score = 0.0
    preferred = _preferred_channel_types(asset)
    if channel.get("type") in preferred:
        score += 1.5
    score += channel.get("reach_score", 0.5)
    score += 0.5 * channel.get("breadth_multiplier", 0.5)
    if broad_bias and channel.get("breadth_multiplier", 0) >= 0.8:
        score += 0.4
    return score


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
) -> float:
    base_lift_bp = asset.get("base_lift_bp")
    if base_lift_bp is None:
        base_lift_bp = 35
    stage_factor = math.exp(-0.35 * transition.get("stage_index", 0))
    path_factor = max(path_probability, 0.05)
    channel_factor = (
        channel.get("reach_score", 0.5) + channel.get("breadth_multiplier", 0.5)
    ) / 2.0
    involvement = persona.get("involvement") or 0.5
    fit_factor = 0.35
    fit_factor += 0.25 * max(0.0, min(asset_score, 3.0)) / 3.0
    fit_factor += 0.2 * max(0.0, min(channel_score, 3.0)) / 3.0
    if stage_alignment:
        fit_factor += 0.15
    else:
        fit_factor *= 0.5
    if persona_alignment or concern_alignment:
        fit_factor *= 1.15
    else:
        fit_factor *= 0.55

    raw = float(base_lift_bp) * channel_factor * stage_factor * path_factor * (
        0.75 + 0.5 * involvement
    ) * fit_factor
    cap = float(base_lift_bp) * (0.8 + 0.6 * channel_factor)
    return max(5.0, min(raw, cap))


def _confidence_score(
    asset_score: float,
    channel_score: float,
    persona: Dict[str, Any],
    *,
    path_probability: float,
) -> float:
    base = 0.45 + 0.15 * path_probability
    base += 0.1 * (persona.get("perceptibility") or 0.3)
    base += 0.12 * (persona.get("proximity") or 0.3)
    base += 0.08 * asset_score
    base += 0.08 * channel_score
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
    path_probability: float,
    exploration_weight: float,
    max_combos: int = 3,
) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    broad_bias = exploration_weight >= 0.15

    for asset in assets:
        asset_score, reasons, stage_alignment, persona_alignment, concern_alignment = _score_asset(
            asset, persona, transition
        )
        if asset_score <= 0.1 and not (
            stage_alignment or persona_alignment or concern_alignment
        ):
            continue

        # choose top channels
        channel_candidates = []
        for channel in channels:
            channel_score = _score_channel(channel, asset, broad_bias=broad_bias)
            channel_candidates.append((channel_score, channel))
        channel_candidates.sort(key=lambda item: item[0], reverse=True)
        channel_candidates = channel_candidates[:2]

        for channel_score, channel in channel_candidates:
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
            )
            confidence = _confidence_score(
                asset_score,
                channel_score,
                persona,
                path_probability=path_probability,
            )
            duration_days = max(
                asset.get("time_to_effect_days") or 14,
                channel.get("time_to_effect_days") or 7,
            )
            broadness = _broadness_score(asset, channel, transition)
            mode = "broad" if broadness >= 0.6 else "focused"
            scored.append(
                (
                    expected_delta,
                    {
                        "persona_id": persona["id"],
                        "persona_label": persona["label"],
                        "stage_index": transition.get("stage_index", 0),
                        "belief_transition": transition.get("belief_transition"),
                        "asset": {
                            "id": asset.get("id"),
                            "name": asset.get("name"),
                            "format": asset.get("format"),
                            "evergreen": bool(asset.get("evergreen")),
                            "stage_fit": asset.get("stage_fit"),
                            "concern_tags": asset.get("concern_tags"),
                            "persona_fit": asset.get("persona_fit"),
                            "time_to_effect_days": asset.get("time_to_effect_days"),
                            "base_lift_bp": asset.get("base_lift_bp"),
                            "cost_tier": asset.get("cost_tier"),
                        },
                        "channel": {
                            "id": channel.get("id"),
                            "name": channel.get("name"),
                            "type": channel.get("type"),
                            "reach_score": channel.get("reach_score"),
                            "breadth_multiplier": channel.get("breadth_multiplier"),
                            "time_to_effect_days": channel.get("time_to_effect_days"),
                            "cadence_hint": channel.get("cadence_hint"),
                        },
                        "expected_delta_bp": expected_delta,
                        "expected_delta_pct": expected_delta / 100.0,
                        "confidence": confidence,
                        "asset_score": asset_score,
                        "channel_score": channel_score,
                        "exploration_weight": exploration_weight,
                        "path_probability": path_probability,
                        "mode": mode,
                        "broadness_score": broadness,
                        "duration_days": int(duration_days),
                        "reasons": {
                            **reasons,
                            "stage_alignment": stage_alignment,
                            "persona_alignment": persona_alignment,
                            "concern_alignment": concern_alignment,
                        },
                    },
                )
            )

    scored.sort(key=lambda item: item[0], reverse=True)
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
                scored.append(
                    (
                        expected_delta,
                        {
                            "persona_id": persona["id"],
                            "persona_label": persona["label"],
                            "stage_index": transition.get("stage_index", 0),
                            "belief_transition": transition.get("belief_transition"),
                            "asset": asset,
                            "channel": channel or {},
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
                            "reasons": {
                                "fallback": True,
                            },
                        },
                    )
                )

        scored.sort(key=lambda item: item[0], reverse=True)

    broad_candidates = [play for _, play in scored if play.get("mode") == "broad"]
    focus_candidates = [play for _, play in scored if play.get("mode") == "focused"]

    selected: List[Dict[str, Any]] = []
    if focus_candidates:
        selected.append(focus_candidates[0])
    if broad_candidates:
        if broad_candidates[0] not in selected:
            selected.append(broad_candidates[0])

    for _, play in scored:
        if play not in selected:
            selected.append(play)
        if len(selected) >= max_combos:
            break

    return selected[:max_combos]


# ---------------------------------------------------------------------------
# Scheduling & campaign grouping
# ---------------------------------------------------------------------------

def _quarter_from_day(day: int) -> str:
    quarter_index = max(0, day) // 90 + 1
    return f"Q{quarter_index}"


def _theme_for_transition(play: Dict[str, Any]) -> str:
    persona = play.get("persona_label") or "Persona"
    transition = play.get("belief_transition") or {}
    focus = transition.get("pain") or transition.get("problem") or "Belief Shift"
    return f"{persona}: {focus}"


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
        play["theme"] = _theme_for_transition(play)
        play["sequence_index"] = len(
            [p for p in plays if p.get("quarter") == play["quarter"]]
        )


def _group_campaigns(plays: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for play in plays:
        key = (play.get("quarter") or "Q1", play.get("theme") or "Campaign")
        group = groups.setdefault(
            key,
            {
                "quarter": key[0],
                "theme": key[1],
                "plays": [],
                "start_day": play.get("start_day", 0),
                "end_day": play.get("end_day", 0),
                "focus_personas": set(),
                "pains": set(),
                "total_delta_bp": 0.0,
                "confidence_sum": 0.0,
                "broad_count": 0,
                "focus_count": 0,
            },
        )
        group["plays"].append(play)
        group["start_day"] = min(group["start_day"], play.get("start_day", 0))
        group["end_day"] = max(group["end_day"], play.get("end_day", 0))
        group["focus_personas"].add(play.get("persona_label"))
        transition = play.get("belief_transition") or {}
        pain_label = None
        pain_entry = transition.get("pain")
        if isinstance(pain_entry, dict):
            pain_label = pain_entry.get("label") or pain_entry.get("id")
        elif isinstance(pain_entry, str):
            pain_label = pain_entry
        if pain_label:
            group["pains"].add(pain_label)
        group["total_delta_bp"] += play.get("expected_delta_bp", 0.0)
        group["confidence_sum"] += play.get("confidence", 0.0)
        if play.get("mode") == "broad":
            group["broad_count"] += 1
        else:
            group["focus_count"] += 1

    campaigns: List[Dict[str, Any]] = []
    for group in groups.values():
        total_plays = len(group["plays"])
        confidence = (
            group["confidence_sum"] / total_plays if total_plays else 0.0
        )

        sorted_plays = sorted(
            group["plays"],
            key=lambda p: (p.get("start_day", 0), p.get("mode") == "broad"),
        )

        seen_beliefs: set[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]] = set()
        belief_entries: List[Dict[str, Any]] = []
        asset_map: Dict[str, Dict[str, Any]] = {}

        for play in sorted_plays:
            bt = play.get("belief_transition") or {}
            key = (
                ((bt.get("persona") or {}).get("id")),
                ((bt.get("problem") or {}).get("id")),
                ((bt.get("pain") or {}).get("id")),
                ((bt.get("resolution") or {}).get("id")),
            )
            if key not in seen_beliefs:
                seen_beliefs.add(key)
                belief_entries.append(bt)

            asset_info = play.get("asset") or {}
            asset_id = asset_info.get("id") or asset_info.get("name") or f"asset:{len(asset_map)}"
            asset_entry = asset_map.setdefault(
                asset_id,
                {
                    "asset": asset_info,
                    "asset_scores": [],
                    "expected_delta_bp": 0.0,
                    "channels": {},
                },
            )
            if play.get("asset_score") is not None:
                asset_entry["asset_scores"].append(float(play.get("asset_score") or 0.0))
            asset_entry["expected_delta_bp"] += float(play.get("expected_delta_bp") or 0.0)

            channel_info = play.get("channel") or {}
            channel_id = channel_info.get("id") or channel_info.get("name") or f"channel:{len(asset_entry['channels'])}"
            channel_entry = asset_entry["channels"].setdefault(
                channel_id,
                {
                    "channel": channel_info,
                    "engagement_scores": [],
                    "expected_delta_bp": 0.0,
                    "modes": set(),
                    "exploration": [],
                },
            )
            if play.get("channel_score") is not None:
                channel_entry["engagement_scores"].append(float(play.get("channel_score") or 0.0))
            channel_entry["expected_delta_bp"] += float(play.get("expected_delta_bp") or 0.0)
            if play.get("mode"):
                channel_entry["modes"].add(play.get("mode"))
            if play.get("exploration_weight") is not None:
                channel_entry["exploration"].append(float(play.get("exploration_weight") or 0.0))

        asset_rows: List[Dict[str, Any]] = []
        for asset_entry in asset_map.values():
            channels_list: List[Dict[str, Any]] = []
            for channel_entry in asset_entry["channels"].values():
                engagement = (
                    sum(channel_entry["engagement_scores"]) / len(channel_entry["engagement_scores"])
                    if channel_entry["engagement_scores"]
                    else None
                )
                exploration = (
                    sum(channel_entry["exploration"]) / len(channel_entry["exploration"])
                    if channel_entry["exploration"]
                    else None
                )
                modes = sorted(list(channel_entry["modes"]))
                channel_payload = {
                    **channel_entry["channel"],
                    "engagement_score": engagement,
                    "expected_delta_bp": channel_entry["expected_delta_bp"],
                    "modes": modes,
                    "mode": modes[0] if modes else None,
                    "exploration_weight": exploration,
                }
                channels_list.append(channel_payload)
            channels_list.sort(key=lambda c: c.get("expected_delta_bp", 0.0), reverse=True)
            asset_fit = (
                sum(asset_entry["asset_scores"]) / len(asset_entry["asset_scores"])
                if asset_entry["asset_scores"]
                else None
            )
            asset_rows.append(
                {
                    "asset": asset_entry["asset"],
                    "asset_fit_score": asset_fit,
                    "expected_delta_bp": asset_entry["expected_delta_bp"],
                    "channels": channels_list,
                }
            )
        asset_rows.sort(key=lambda r: r.get("expected_delta_bp", 0.0), reverse=True)

        campaigns.append(
            {
                "quarter": group["quarter"],
                "theme": group["theme"],
                "focus_personas": sorted(
                    [p for p in group["focus_personas"] if p]
                ),
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
                "asset_table": asset_rows,
            }
        )

    campaigns.sort(
        key=lambda c: (int(c["quarter"].replace("Q", "")), c["start_day"])
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
# Account-level blueprint builder
# ---------------------------------------------------------------------------

def build_account_marketing_blueprint(
    product_id: str,
    account_id: str,
    *,
    product_graph: Optional[nx.DiGraph] = None,
) -> Dict[str, Any]:
    if product_graph is None:
        product_graph = build_product_graph(product_id)

    account = get_account_by_id(product_id, account_id) or {"account_name": account_id}
    thesis = build_belief_thesis_for_account(product_id, account_id)

    paths = _top_paths_from_thesis(thesis)
    if not paths:
        return {
            "account_id": account_id,
            "account_name": account.get("account_name") or account_id,
            "error": "No persona paths available",
        }

    assets = _load_assets(product_id)
    channels = _load_channels(product_id)

    persona_paths = []
    scatter_personas: List[Dict[str, Any]] = []
    transitions_all: List[Dict[str, Any]] = []
    plays: List[Dict[str, Any]] = []

    for path_index, path in enumerate(paths):
        persona_summaries = _collect_persona_metrics(product_graph, path)
        scatter_personas.extend(persona_summaries)
        persona_paths.append(
            {
                "id": path.get("id") or f"path:{path_index}",
                "probability": path.get("probability"),
                "score": path.get("score"),
                "personas": persona_summaries,
                "raw": path.get("raw"),
                "is_primary": path_index == 0,
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
                path_probability=path.get("probability", 0.0),
                exploration_weight=exploration,
            )

            for combo in combos:
                combo["path_id"] = path.get("id") or f"path:{path_index}"
                combo["path_probability"] = path.get("probability", 0.0)
                combo["path_index"] = path_index
                combo["stage_index"] = stage_index
                combo["exploration_weight"] = exploration
                plays.append(combo)

    _schedule_plays(plays)
    campaigns = _group_campaigns(plays)
    randomization = _randomization_policy(paths, plays)

    return {
        "account_id": account_id,
        "account_name": account.get("account_name") or account_id,
        "deal_status": account.get("deal_status"),
        "meta": {
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
        },
        "prediction": {
            "persona_paths": persona_paths,
            "fit": thesis.get("fit") or {},
            "expected_next": thesis.get("current_expected_next"),
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
        "transitions": transitions_all,
        "scatter": {"personas": scatter_personas},
        "execution": {"plays": plays},
        "campaigns": campaigns,
        "learning": thesis.get("learning_summary") or {},
        "randomization": randomization,
    }


# ---------------------------------------------------------------------------
# Portfolio aggregation
# ---------------------------------------------------------------------------

def _aggregate_portfolio(accounts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    theme_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    broad_total = 0
    focus_total = 0
    scatter: List[Dict[str, Any]] = []
    path_probabilities: List[float] = []

    for account in accounts:
        scatter.extend(account.get("scatter", {}).get("personas") or [])
        for path in account.get("prediction", {}).get("persona_paths", []) or []:
            if path.get("probability") is not None:
                path_probabilities.append(path["probability"])

        for play in account.get("execution", {}).get("plays", []):
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
                },
            )
            group["accounts"].add(account.get("account_id"))
            group["total_delta_bp"] += campaign.get("total_delta_bp", 0.0)
            group["confidence_sum"] += campaign.get("confidence", 0.0)
            group["plays"].extend(campaign.get("plays") or [])
            group["focus_personas"].update(campaign.get("focus_personas") or [])
            group["focus_pains"].update(campaign.get("focus_pains") or [])

    campaign_themes = []
    for group in theme_groups.values():
        play_count = len(group["plays"])
        campaign_themes.append(
            {
                "quarter": group["quarter"],
                "theme": group["theme"],
                "accounts": sorted(group["accounts"]),
                "focus_personas": sorted(
                    [p for p in group["focus_personas"] if p]
                ),
                "focus_pains": sorted([p for p in group["focus_pains"] if p]),
                "total_delta_bp": group["total_delta_bp"],
                "avg_confidence": group["confidence_sum"] / play_count if play_count else 0.0,
                "play_count": play_count,
            }
        )

    campaign_themes.sort(
        key=lambda c: (int(c["quarter"].replace("Q", "")), -c["total_delta_bp"])
    )

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
        "scatter": scatter_summary[:200],
        "broad_focus_mix": mix,
        "randomization": randomization,
    }


def _summary_from_accounts(accounts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    account_count = len(accounts)
    prob_primary = []
    accuracies = []
    persona_counter: Counter[str] = Counter()
    pain_counter: Counter[str] = Counter()
    capability_counter: Counter[str] = Counter()
    total_delta_bp = 0.0

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
        for campaign in account.get("campaigns") or []:
            total_delta_bp += campaign.get("total_delta_bp", 0.0)
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

    def _top(counter: Counter[str]) -> List[Dict[str, Any]]:
        return [
            {"label": label, "count": count}
            for label, count in counter.most_common(6)
        ]

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
    }


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

    summary = _summary_from_accounts(accounts)
    portfolio_plan = _aggregate_portfolio(accounts)

    return {
        "product_id": canonical_product_id,
        "generated_at": _now_iso(),
        "summary": summary,
        "accounts": accounts,
        "portfolio_plan": portfolio_plan,
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
    print("[marketing_plan] _save_rcs_json is deprecated; skipping legacy write.")
