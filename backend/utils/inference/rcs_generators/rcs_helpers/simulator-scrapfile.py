# ============================
# File: backend/utils/inference/rcs_generators/rcs_simulator.py
# ============================
from __future__ import annotations
from typing import Dict, List, Optional
import networkx as nx
import copy

from backend.utils.graph_base.network_graph import get_node_by_id
from backend.utils.inference.rcs_generators.rcs_helpers.strategy_orchestrators import (
    build_account_strategy,
    update_tactical_plan,
)

# -------------------------------------------------------
# Core simulation orchestrator
# -------------------------------------------------------
def simulate_rcs(
    product_subgraph: nx.DiGraph,
    *,
    engaged_nodes: Optional[List[str]] = None,
    attributes: Optional[List[str]] = None,
    mode: str = "tactical",           # "strategy" | "tactical"
    frozen_strategy: Optional[Dict] = None,
) -> Dict:
    """
    Unified simulation entrypoint:
      - "strategy" mode builds a frozen baseline and an initial tactical preview.
      - "tactical" mode applies new engagements to an existing frozen strategy.
    """
    engaged_nodes = engaged_nodes or []
    attributes = attributes or []
    zmots: List[str] = []
    occurred_nodes: List[str] = []

    # Normalize engaged payload (handle strings or dicts)
    engaged_payload = []
    for n in set(engaged_nodes + attributes):
        if not n:
            continue
        if isinstance(n, dict):
            engaged_payload.append({"id": n.get("id"), "occurrence": float(n.get("occurrence") or 1.0)})
        else:
            engaged_payload.append({"id": n, "occurrence": 1.0})

    # Partition engaged nodes by type
    for n in engaged_payload:
        nid = n.get("id")
        if nid not in product_subgraph.nodes:
            continue
        t = product_subgraph.nodes[nid].get("node_type", "")
        if t == "attribute_value":
            if nid not in attributes:
                attributes.append(nid)
        elif t == "zmot_event":
            zmots.append(nid)
        else:
            occurred_nodes.append(nid)


    # ---------------------------------------------------
    # STRATEGY MODE
    # ---------------------------------------------------
    if mode == "strategy":
        frozen_strategy, fs_pain_graph, report = build_account_strategy(
            product_subgraph,
            attributes=attributes,
            zmots=zmots,
            initial_engagements=occurred_nodes,
        )
        tactical_update, tu_pain_graph = update_tactical_plan(
            product_subgraph,
            frozen_strategy,
            attributes=attributes,
            zmots=zmots,
            engagements_to_date=occurred_nodes,
            stage="cold",
        )
        graph_for_ui = export_graph_for_d3(_set_node_labels_positions(cleaned_graph_for_ui(fs_pain_graph)))
        return {
            "output": tactical_update, 
            "frozen_strategy": frozen_strategy, 
            "graph": graph_for_ui
            }

    # ---------------------------------------------------
    # TACTICAL MODE
    # ---------------------------------------------------
    if not frozen_strategy:
        frozen_strategy, fs_pain_graph, report = build_account_strategy(
            product_subgraph,
            attributes=attributes,
            zmots=zmots,
            initial_engagements=occurred_nodes,
        )

    tactical_update, tu_pain_graph = update_tactical_plan(
        product_subgraph,
        frozen_strategy,
        attributes=attributes,
        zmots=zmots,
        engagements_to_date=occurred_nodes,
        stage="auto",
    )

    graph_for_ui = export_graph_for_d3(_set_node_labels_positions(cleaned_graph_for_ui(tu_pain_graph)))
    return {
        "tactical_update": tactical_update,
        "frozen_strategy": frozen_strategy,
        "graph": graph_for_ui,
    }


# -------------------------------------------------------
# Graph cleanup / visualization utilities
# -------------------------------------------------------
def export_graph_for_d3(G: nx.DiGraph):
    """Convert NetworkX graph to D3-compatible JSON format."""
    nodes = [{"id": nid, **data} for nid, data in G.nodes(data=True)]
    links = [{"source": u, "target": v, **data} for u, v, data in G.edges(data=True)]
    return {"nodes": nodes, "links": links}


def _set_node_labels_positions(G: nx.DiGraph):
    """Attach readable labels to nodes for UI display."""
    for node_id in G.nodes:
        node = get_node_by_id(G, node_id)
        node_type = node.get("node_type", node.get("type", "unknown"))
        if node_type == "persona":
            title = node.get("title", "")
            dept = node.get("department", "")
            seniority = node.get("seniority", "")
            node["label"] = " | ".join(filter(None, [title, dept, seniority]))
        elif node_type == "job":
            node["label"] = node.get("description", "") or f"Job {node_id}"
        elif node_type == "pain":
            node["label"] = node.get("description", "") or f"Pain {node_id}"
        elif node_type == "capability":
            node["label"] = node.get("name", "") or f"Capability {node_id}"
        elif node_type == "product":
            node["label"] = node.get("url", "") or f"Product {node_id}"
        elif node_type == "pain_trigger":
            node["label"] = node.get("attribute", "") or f"Pain Trigger {node_id}"
        elif node_type == "zmot_event":
            node["label"] = node.get("description", "") or f"ZMOT Event {node_id}"
        elif node_type == "observable_moment":
            node["label"] = node.get("description", "") or f"Observable Moment {node_id}"
        elif node_type == "keyword":
            node["label"] = node.get("keyword", "") or f"Keyword {node_id}"
        elif node_type == "perceived_metric":
            node["label"] = node.get("metric", "") or f"Metric {node_id}"
        else:
            node["label"] = node.get("description", "") or f"Node {node_id}"
    return G


def cleaned_graph_for_ui(G: nx.DiGraph):
    """Trim unused node types and isolated nodes for cleaner visualization."""
    H = G.copy()
    keep_types = {"persona", "job", "pain", "capability", "product", "zmot_event", "pain_trigger"}
    for n in list(H.nodes):
        if H.nodes[n].get("node_type") not in keep_types:
            H.remove_node(n)
    for n in list(H.nodes):
        if n not in H or n not in H.nodes:
            continue
        if H.degree(n) == 0 and H.nodes[n].get("node_type") != "product":
            H.remove_node(n)
    return H
