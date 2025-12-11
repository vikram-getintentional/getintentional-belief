"""Utilities for learning from SHM-style persona journeys.

This module takes replayed persona journeys (the same structure that
``replay_learn_persona_paths`` emits) and produces human-reviewable
recommendations:

- Persona edge updates (which transitions appear mis-weighted)
- Suggested new personas (based on repeated unknown personas)
- Intervention effect summaries (channel/asset level lift estimates)

All outputs are diagnostic only so that a user can accept/reject.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import math

try:  # Optional import for type checking / static analysis
    from backend.utils.inference.rcs_generators.persona_map.persona_belief_engine import (
        PersonaGraph,
    )
except Exception:  # pragma: no cover - safe fallback for runtime when module not available
    PersonaGraph = None  # type: ignore


@dataclass
class EdgeRecommendation:
    source: str
    target: str
    delta: float
    confidence: float
    support: int
    rationale: str


@dataclass
class CandidatePersona:
    persona_id: str
    support: int
    example_sources: List[str]
    preceding_personas: List[str]


@dataclass
class InterventionEffect:
    persona_id: Optional[str]
    channel: Optional[str]
    asset_id: Optional[str]
    sample_size: int
    mean_effect: float
    stddev: float


_MIN_SUPPORT = 2
_MIN_EDGE_DELTA = 0.15
_MIN_NEW_EDGE_PROB = 0.25


def _nt(graph: PersonaGraph, node_id: str) -> str:
    if graph is None:
        return ""
    data = graph.G.nodes.get(node_id) if node_id in graph.G else {}
    return (data.get("node_type") or data.get("type") or "").lower()


def _edge_prob(graph: PersonaGraph, u: str, v: str) -> float:
    if graph is None:
        return 0.0
    if graph.G.has_edge(u, v):
        data = graph.G[u][v]
        return float(data.get("prob")) if data.get("prob") is not None else float(
            data.get("likelihood", 0.0)
        )
    return 0.0


def build_transition_dataset(observed_sequence: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for prev, curr in zip(observed_sequence[:-1], observed_sequence[1:]):
        if not prev or not curr:
            continue
        rows.append({
            "from": prev,
            "to": curr,
            "weight": 1.0,
        })
    return rows


def update_persona_edges_from_transitions(
    persona_graph: PersonaGraph,
    transitions: List[Dict[str, Any]],
    *,
    dirichlet_kappa: float = 1.0,
    inconsistency_threshold: float = _MIN_EDGE_DELTA,
) -> Dict[str, Any]:
    if persona_graph is None or not transitions:
        return {"edge_recommendations": [], "suggested_new_edges": [], "diagnostics": {}}

    counts: Counter[Tuple[str, str]] = Counter(
        (row["from"], row["to"]) for row in transitions if row.get("from") and row.get("to")
    )
    totals: Counter[str] = Counter(row["from"] for row in transitions if row.get("from"))

    recs: List[EdgeRecommendation] = []
    new_edges: List[EdgeRecommendation] = []

    for (src, dst), support in counts.items():
        if support < _MIN_SUPPORT:
            continue
        total = totals[src]
        posterior = support / max(1, total)
        prior = _edge_prob(persona_graph, src, dst)
        delta = posterior - prior
        if (src, dst) in persona_graph.G.edges:
            if abs(delta) >= inconsistency_threshold:
                confidence = min(1.0, abs(delta) * 2)
                recs.append(
                    EdgeRecommendation(
                        source=src,
                        target=dst,
                        delta=delta,
                        confidence=confidence,
                        support=support,
                        rationale="Observed transitions differ from prior edge weight",
                    )
                )
        else:
            if posterior >= _MIN_NEW_EDGE_PROB:
                confidence = min(1.0, posterior)
                new_edges.append(
                    EdgeRecommendation(
                        source=src,
                        target=dst,
                        delta=posterior,
                        confidence=confidence,
                        support=support,
                        rationale="Repeated transition not present in persona graph",
                    )
                )

    return {
        "edge_recommendations": [rec.__dict__ for rec in recs],
        "suggested_new_edges": [rec.__dict__ for rec in new_edges],
        "diagnostics": {
            "total_observed_transitions": len(transitions),
            "unique_pairs": len(counts),
        },
    }


def discover_candidate_personas_from_transitions(
    *,
    transitions: List[Dict[str, Any]],
    persona_graph: PersonaGraph,
) -> List[Dict[str, Any]]:
    known_personas = set(persona_graph.G.nodes) if persona_graph is not None else set()
    unknown_counts: Counter[str] = Counter()
    followers: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in transitions:
        nxt = row.get("to")
        prev = row.get("from")
        if not nxt or nxt in known_personas:
            continue
        unknown_counts[nxt] += 1
        if prev:
            followers[nxt][prev] += 1

    candidates: List[CandidatePersona] = []
    for persona_id, support in unknown_counts.items():
        if support < _MIN_SUPPORT:
            continue
        preceding = [src for src, _ in followers[persona_id].most_common(3)]
        candidates.append(
            CandidatePersona(
                persona_id=persona_id,
                support=support,
                example_sources=preceding[:2],
                preceding_personas=preceding,
            )
        )

    return [cand.__dict__ for cand in sorted(candidates, key=lambda c: c["support"], reverse=True)]


def learn_intervention_effects(
    journey_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not journey_steps:
        return {"effects": {}, "diagnostics": {}}

    buckets: Dict[Tuple[Optional[str], Optional[str], Optional[str]], List[float]] = defaultdict(list)

    for step in journey_steps:
        persona = step.get("observed_next")
        meta = step.get("engagement_meta") or {}
        channel = meta.get("channel")
        asset_id = meta.get("asset_id")
        hit_score = 1.0 if step.get("hit_at_1") else 0.6 if step.get("hit_at_3") else 0.2
        bucket = step.get("bucket")
        if bucket == "off_path_known":
            hit_score *= 0.3
        elif bucket == "no_path":
            hit_score *= 0.2
        buckets[(persona, channel, asset_id)].append(hit_score)

    effects: Dict[str, InterventionEffect] = {}
    for key, scores in buckets.items():
        if not scores:
            continue
        persona, channel, asset = key
        n = len(scores)
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / max(1, n - 1)
        effects_key = f"{persona or 'unknown'}|{channel or 'channel'}|{asset or 'asset'}"
        effects[effects_key] = InterventionEffect(
            persona_id=persona,
            channel=channel,
            asset_id=asset,
            sample_size=n,
            mean_effect=mean,
            stddev=math.sqrt(variance),
        ).__dict__

    return {
        "effects": effects,
        "diagnostics": {"total_samples": sum(len(v) for v in buckets.values())},
    }


def summarize_learning_signals(
    *,
    persona_graph: PersonaGraph,
    transitions: List[Dict[str, Any]],
    journey_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    edge_updates = update_persona_edges_from_transitions(persona_graph, transitions)
    candidates = discover_candidate_personas_from_transitions(
        transitions=transitions,
        persona_graph=persona_graph,
    )
    interventions = learn_intervention_effects(journey_steps)

    return {
        "edge_updates": edge_updates,
        "candidate_personas": candidates,
        "intervention_effects": interventions,
    }
