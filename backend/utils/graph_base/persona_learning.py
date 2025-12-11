"""
Utilities for discovering and scoring high-impact ("wolves") personas from
engagement evidence.

This module focuses on:
  * Matching candidate persona labels against existing canonical personas.
  * Promoting high-confidence candidates into the product graph with
    provenance + confidence metadata.
  * Computing persona impact metrics (wolves score, delta win basis points,
    centrality, etc.) so downstream planners can prioritize them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import os

import networkx as nx

from backend.utils.graph_base.agent_graph_builder import _persona, _upsert_edge
from backend.utils.graph_base.graph_utils.json_store import save_json
from backend.utils.graph_base.graph_utils.save_and_load_graph_as_json import save_graph_as_json
from backend.utils.graph_base.network_graph import GRAPH_DATA_PATH, build_product_graph
from backend.utils.inference.belief_manager.types import (
    PersonaCandidateStat,
    PersonaImpactMetrics,
)
from backend.utils.persona_normalization import normalize_persona_label

_METRICS_DIR = os.path.join(GRAPH_DATA_PATH, "persona_metrics")


def _sequence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a.lower(), b=b.lower()).ratio()


def _persona_label(data: Dict[str, any]) -> str:
    return data.get("label") or data.get("title") or data.get("name") or ""


def match_candidate_to_canonical_persona(
    label: str,
    persona_nodes: Sequence[Tuple[str, Dict[str, any]]],
    *,
    threshold: float = 0.85,
) -> Tuple[Optional[str], float]:
    """
    Returns (persona_id, similarity_score) for the best match.
    score < threshold => treat as new persona.
    """
    if not label:
        return None, 0.0
    best_id: Optional[str] = None
    best_score = 0.0
    for node_id, node_data in persona_nodes:
        existing_label = _persona_label(node_data)
        if not existing_label:
            continue
        score = _sequence_similarity(label, existing_label)
        if score > best_score:
            best_score = score
            best_id = node_id
    if best_score < threshold:
        return None, best_score
    return best_id, best_score


def _derive_title_department_seniority(stat: PersonaCandidateStat) -> Tuple[str, str, str]:
    if stat.titles:
        title = stat.titles[0]
    else:
        title = stat.label.split("·")[0].strip().title()

    if stat.departments:
        department = stat.departments[0]
    else:
        chunks = [chunk.strip() for chunk in stat.label.split("·")]
        department = chunks[1] if len(chunks) > 1 else ""

    seniority = ""
    chunks = [chunk.strip() for chunk in stat.label.split("·")]
    if len(chunks) > 2:
        seniority = chunks[2]
    return title, department, seniority


def auto_add_persona_nodes_from_candidates(
    G: nx.DiGraph,
    candidates: Iterable[PersonaCandidateStat],
    *,
    co_occurrence: Dict[str, Counter],
    min_accounts: int = 3,
    min_occurrences: int = 5,
    similarity_threshold: float = 0.85,
) -> List[Dict[str, any]]:
    """
    Takes high-confidence candidate stats and promotes them into the graph.
    Returns metadata describing newly created nodes.
    """
    persona_nodes = [
        (node_id, data)
        for node_id, data in G.nodes(data=True)
        if data.get("node_type") == "persona"
    ]
    created: List[Dict[str, any]] = []

    for stat in candidates:
        normalized_label = normalize_persona_label(stat.label, None, None)
        if not normalized_label:
            continue
        match_id, match_score = match_candidate_to_canonical_persona(
            stat.label,
            persona_nodes,
            threshold=similarity_threshold,
        )
        if match_id:
            node_data = G.nodes[match_id]
            aliases = set(node_data.get("aliases") or [])
            if stat.label not in aliases:
                aliases.add(stat.label)
                node_data["aliases"] = sorted(aliases)
            node_data["last_seen_candidate_at"] = (
                stat.last_seen_at.isoformat() if stat.last_seen_at else datetime.utcnow().isoformat()
            )
            continue

        if stat.account_count < min_accounts or stat.occurrence_count < min_occurrences:
            continue

        title, department, seniority = _derive_title_department_seniority(stat)
        node_id = _persona(
            G,
            title=title or stat.label.title(),
            department=department or "",
            seniority=seniority or "",
            sample_profiles=None,
            data_source="data_auto",
        )
        G.nodes[node_id].update(
            {
                "label": stat.label,
                "source": "data_auto",
                "confidence": 0.6,
                "account_coverage": stat.account_count,
                "occurrence_count": stat.occurrence_count,
                "candidate_label": stat.label,
                "segments": stat.segments,
                "first_seen_at": stat.first_seen_at.isoformat() if stat.first_seen_at else None,
                "last_seen_at": stat.last_seen_at.isoformat() if stat.last_seen_at else None,
                "created_at": datetime.utcnow().isoformat(),
            }
        )

        co_counter = co_occurrence.get(normalized_label) or Counter()
        for persona_id, weight in co_counter.most_common(5):
            if not persona_id or persona_id == node_id or persona_id not in G:
                continue
            w = max(0.05, min(0.4, float(weight) / 10.0))
            attrs = {"confidence": 0.3, "source": "data_auto"}
            _upsert_edge(G, node_id, "co_occurs_with", persona_id, weight=w, attrs=attrs)
            _upsert_edge(G, persona_id, "co_occurs_with", node_id, weight=w, attrs=attrs)

        created.append(
            {
                "id": node_id,
                "label": stat.label,
                "account_count": stat.account_count,
                "occurrence_count": stat.occurrence_count,
            }
        )

    return created


def compute_persona_impact_metrics(
    G: nx.DiGraph,
    *,
    persona_account_map: Dict[str, Iterable[str]],
    account_outcomes: Dict[str, str],
) -> List[PersonaImpactMetrics]:
    """
    Compute wolves metrics for every persona node in the graph.
    """
    persona_nodes = [
        node_id for node_id, data in G.nodes(data=True) if data.get("node_type") == "persona"
    ]
    if not persona_nodes:
        return []

    centrality_scores = nx.degree_centrality(G)
    if persona_nodes:
        values = [centrality_scores.get(pid, 0.0) for pid in persona_nodes]
        c_min = min(values)
        c_max = max(values)
        c_span = (c_max - c_min) or 1.0
    else:
        c_min = 0.0
        c_span = 1.0

    total_accounts = len(account_outcomes) or 1
    global_win_rate = (
        sum(1 for status in account_outcomes.values() if status == "won") / total_accounts
    )

    metrics: List[PersonaImpactMetrics] = []
    for persona_id in persona_nodes:
        accounts = set(persona_account_map.get(persona_id) or [])
        sample_size = len(accounts)
        if sample_size == 0:
            continue

        wins_with = sum(1 for acc in accounts if account_outcomes.get(acc) == "won")
        losses_with = sum(1 for acc in accounts if account_outcomes.get(acc) == "lost")
        win_rate_with = wins_with / sample_size if sample_size else 0.0

        other_accounts = [acc for acc in account_outcomes.keys() if acc not in accounts]
        if other_accounts:
            wins_without = sum(1 for acc in other_accounts if account_outcomes.get(acc) == "won")
            win_rate_without = wins_without / len(other_accounts)
        else:
            win_rate_without = global_win_rate

        delta_win_bp = (win_rate_with - win_rate_without) * 10000.0
        involvement_rate = sample_size / total_accounts
        blocker_rate = losses_with / sample_size if sample_size else 0.0

        centrality_raw = centrality_scores.get(persona_id, 0.0)
        centrality_norm = (centrality_raw - c_min) / c_span

        wolves_score = max(
            0.0,
            min(
                1.0,
                0.4 * centrality_norm
                + 0.35 * involvement_rate
                + 0.15 * max(0.0, delta_win_bp) / 1000.0
                + 0.10 * (1.0 - blocker_rate),
            ),
        )

        node_data = G.nodes[persona_id]
        source = node_data.get("source") or node_data.get("data_source") or "initial_seed"

        metrics.append(
            PersonaImpactMetrics(
                persona_id=persona_id,
                wolves_score=wolves_score,
                delta_win_bp=delta_win_bp,
                centrality=centrality_norm,
                involvement_rate=involvement_rate,
                blocker_rate=blocker_rate,
                sample_size=sample_size,
                last_updated_at=datetime.utcnow(),
                source=source,
            )
        )

    metrics.sort(key=lambda m: m.wolves_score, reverse=True)
    return metrics


def save_persona_metrics(product_id: str, metrics: Iterable[PersonaImpactMetrics]) -> str:
    """
    Persist the computed persona metrics to disk so the planner/UX can load them.
    Returns the file path written.
    """
    payload = {
        "product_id": product_id,
        "updated_at": datetime.utcnow().isoformat(),
        "personas": [asdict(metric) for metric in metrics],
    }
    path = os.path.join(_METRICS_DIR, f"{product_id}.json")
    save_json(path, payload)
    return path


def add_persona_node_from_candidate(
    product_id: str,
    *,
    label: str,
    title: Optional[str],
    department: Optional[str],
    seniority: Optional[str],
    source: str = "enrich_user",
    confidence: float = 0.9,
    co_occurring_persona_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Promote a candidate label into the product graph as a new persona node.
    """
    graph = build_product_graph(product_id)
    node_id = _persona(
        graph,
        title or label,
        department or "",
        seniority or "",
        sample_profiles=None,
        data_source=source,
    )
    node = graph.nodes[node_id]
    node.update(
        {
            "label": label,
            "title": title,
            "department": department,
            "seniority": seniority,
            "confidence": confidence,
            "source": source,
            "data_source": source,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
        }
    )
    for peer_id in co_occurring_persona_ids or []:
        if peer_id and peer_id in graph:
            _upsert_edge(
                graph,
                node_id,
                "co_occurs_with",
                peer_id,
                weight=0.2,
                attrs={"source": source, "confidence": 0.4},
            )
            _upsert_edge(
                graph,
                peer_id,
                "co_occurs_with",
                node_id,
                weight=0.2,
                attrs={"source": source, "confidence": 0.4},
            )
    save_graph_as_json(graph, product_id)
    return {"persona_id": node_id, "label": node.get("label"), "title": node.get("title")}


def alias_persona_with_label(
    product_id: str,
    persona_id: str,
    alias: str,
) -> Dict[str, Any]:
    """
    Attach a candidate label as an alias to an existing persona node.
    """
    graph = build_product_graph(product_id)
    if persona_id not in graph:
        raise ValueError(f"Persona '{persona_id}' not found in graph.")
    node = graph.nodes[persona_id]
    aliases = set(node.get("aliases") or [])
    aliases.add(alias)
    node["aliases"] = sorted(aliases)
    node["last_updated"] = datetime.utcnow().isoformat()
    save_graph_as_json(graph, product_id)
    return {"persona_id": persona_id, "label": node.get("label"), "aliases": node["aliases"]}


__all__ = [
    "auto_add_persona_nodes_from_candidates",
    "compute_persona_impact_metrics",
    "match_candidate_to_canonical_persona",
    "save_persona_metrics",
    "add_persona_node_from_candidate",
    "alias_persona_with_label",
]
