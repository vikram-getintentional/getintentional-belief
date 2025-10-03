from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import networkx as nx
from typing import Dict, Tuple

# --- 1) Configs --------------------------------------------------------------

SOURCE_WEIGHTS: Dict[str, float] = {
    "llm": 0.35,           # default lowest
    "external_api": 0.65,  # e.g., Clearbit, BuiltWith, etc.
    "user": 0.80,          # manually curated by user
    "crm": 1.00,           # highest trust
}

HALF_LIFE_DAYS = 90  # confidence halves every 90 days of staleness
NODE_EDGE_BLEND = 0.45  # alpha for nodes; (1-alpha) for edges → edges slightly heavier

# --- 2) Utilities ------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _parse_ts(dt_str: str | None) -> datetime | None:
    if not dt_str:
        return None
    try:
        # Accept both naive and tz-aware ISO strings
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def _decay_factor(last_updated: str | None, now: datetime) -> float:
    """
    Exponential time decay in [0,1]. 1.0 if fresh; approaches 0 as staleness grows.
    """
    ts = _parse_ts(last_updated)
    if ts is None:
        return 0.6  # mildly pessimistic default if timestamp missing
    days_old = max(0.0, (now - ts).total_seconds() / 86400.0)
    if HALF_LIFE_DAYS <= 0:
        return 1.0
    # exp2(- age / half_life) == 0.5 at half-life
    return 2 ** (-days_old / HALF_LIFE_DAYS)

def _source_weight(source: str | None) -> float:
    return SOURCE_WEIGHTS.get((source or "llm").lower(), SOURCE_WEIGHTS["llm"])

def _item_confidence(source: str | None, last_updated: str | None, now: datetime) -> float:
    return _source_weight(source) * _decay_factor(last_updated, now)

# --- 3) Core computation -----------------------------------------------------

@dataclass
class ConfidenceReport:
    graph_confidence: float
    nodes_confidence: float
    edges_confidence: float
    coverage: Dict[str, float]            # e.g., {"nodes_with_meta": 0.92, "edges_with_meta": 0.88}
    staleness_days_avg: Dict[str, float]  # avg staleness for nodes/edges
    as_of_iso: str

def compute_graph_confidence(G: nx.DiGraph) -> ConfidenceReport:
    print("Computing graph confidence...")
    now = _now()

    # Nodes
    node_scores, node_stale_days = [], []
    with_meta_nodes = 0
    for n, data in G.nodes(data=True):
        src = data.get("data_source")
        ts = data.get("last_updated")
        if src or ts:
            with_meta_nodes += 1
        node_scores.append(_item_confidence(src, ts, now))
        ts_parsed = _parse_ts(ts)
        if ts_parsed:
            node_stale_days.append(max(0.0, (now - ts_parsed).total_seconds() / 86400.0))

    nodes_conf = sum(node_scores) / len(node_scores) if node_scores else 0.0
    nodes_meta_cov = (with_meta_nodes / len(G)) if len(G) else 0.0
    nodes_stale_avg = (sum(node_stale_days) / len(node_stale_days)) if node_stale_days else float("nan")

    # Edges
    edge_scores, edge_stale_days = [], []
    with_meta_edges = 0
    for u, v, data in G.edges(data=True):
        src = data.get("data_source")
        ts = data.get("last_updated")
        if src or ts:
            with_meta_edges += 1
        edge_scores.append(_item_confidence(src, ts, now))
        ts_parsed = _parse_ts(ts)
        if ts_parsed:
            edge_stale_days.append(max(0.0, (now - ts_parsed).total_seconds() / 86400.0))

    edges_conf = sum(edge_scores) / len(edge_scores) if edge_scores else 0.0
    edges_meta_cov = (with_meta_edges / G.number_of_edges()) if G.number_of_edges() else 0.0
    edges_stale_avg = (sum(edge_stale_days) / len(edge_stale_days)) if edge_stale_days else float("nan")

    # Blend
    alpha = NODE_EDGE_BLEND
    graph_conf = alpha * nodes_conf + (1 - alpha) * edges_conf

    # Optional: penalize low metadata coverage slightly (so you notice missing meta)
    coverage_penalty = 0.5 * (nodes_meta_cov + edges_meta_cov) if (len(G) or G.number_of_edges()) else 0.0
    # Scale: 80% weight to core confidence, 20% to coverage presence
    graph_conf = 0.8 * graph_conf + 0.2 * coverage_penalty * graph_conf

    report = ConfidenceReport(
        graph_confidence=round(graph_conf, 4),
        nodes_confidence=round(nodes_conf, 4),
        edges_confidence=round(edges_conf, 4),
        coverage={
            "nodes_with_meta": round(nodes_meta_cov, 4),
            "edges_with_meta": round(edges_meta_cov, 4),
        },
        staleness_days_avg={
            "nodes": (round(nodes_stale_avg, 1) if not math.isnan(nodes_stale_avg) else None),
            "edges": (round(edges_stale_avg, 1) if not math.isnan(edges_stale_avg) else None),
        },
        as_of_iso=now.isoformat(),
    )

    # Persist on the graph for UI to read
    G.graph["confidence_report"] = report.__dict__
    G.graph["graph_confidence"] = report.graph_confidence
    G.graph["confidence_as_of"] = report.as_of_iso
    return report

# --- 4) Helpers to auto-refresh on edits ------------------------------------

def touch_node(G: nx.DiGraph, node_id: str, data_source: str | None = None):
    data = G.nodes[node_id]
    if data_source:
        data["data_source"] = data_source
    data["last_updated"] = _now().isoformat()

def touch_edge(G: nx.DiGraph, u: str, v: str, data_source: str | None = None):
    data = G.edges[u, v]
    if data_source:
        data["data_source"] = data_source
    data["last_updated"] = _now().isoformat()
