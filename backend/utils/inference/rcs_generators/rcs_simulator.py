# backend/utils/inference/rcs_generators/rcs_simulator.py
from __future__ import annotations
from typing import Dict, List, Optional, Iterable, Tuple
import networkx as nx

from backend.utils.graph_base.network_graph import get_node_by_id
from backend.utils.inference.rcs_generators.generate_rcs_fast import generate_rcs
from backend.utils.inference.rcs_generators.rcs_helpers.strategy_orchestrators import (
    build_account_strategy,
    update_tactical_plan,
)

# ---- helpers ----
def _dedupe_ids(items: Iterable) -> List[str]:
    """items can be strings or dicts {'id': ...}. Returns unique ids preserving order."""
    seen, out = set(), []
    for x in items:
        nid = x.get("id") if isinstance(x, dict) else x
        if not nid or nid in seen:
            continue
        seen.add(nid)
        out.append(nid)
    return out

def _bucket_engagements(G: nx.DiGraph, node_ids: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    """Return (attributes, zmots, occurred_nodes) based on node_type."""
    attributes, zmots, occurred = [], [], []
    for nid in node_ids:
        if nid not in G.nodes:
            print(f"[warn] engaged node {nid} not in product subgraph")
            continue
        t = (get_node_by_id(G, nid) or {}).get("node_type", "")
        if t == "attribute_value":
            if nid not in attributes:
                attributes.append(nid)
        elif t == "zmot_event":
            if nid not in zmots:
                zmots.append(nid)
        else:
            if nid not in occurred:
                occurred.append(nid)
    return attributes, zmots, occurred

def _to_engaged_payload(ids: Iterable[str], default_occ: float = 1.0) -> List[Dict]:
    return [{"id": nid, "occurrence": float(default_occ)} for nid in ids]

# ---- main entry ----
def simulate_rcs(
    product_subgraph: nx.DiGraph,
    *,
    engaged_nodes: Optional[List] = None,   # strings or {'id','occurrence'}
    attributes: Optional[List[str]] = None,
    mode: str = "tactical",                 # "strategy" | "tactical"
    frozen_strategy: Optional[Dict] = None,
) -> Dict:
    engaged_nodes = engaged_nodes or []
    attributes = attributes or []

    # unify + dedupe by id (don’t use set() on dicts)
    merged_ids = _dedupe_ids(list(engaged_nodes) + list(attributes))
    attr_ids, zmot_ids, occurred_ids = _bucket_engagements(product_subgraph, merged_ids)

    # we only use this payload to render a quick graph for the UI
    engaged_payload = _to_engaged_payload(merged_ids, default_occ=1.0)
    print(f"Simulating RCS with these engaged nodes: {engaged_payload}")

    if mode == "strategy":
        # Build frozen strategy once (seed with attributes/zmots; no prior engagements by default)
        strategy = build_account_strategy(
            product_subgraph,
            attributes=attr_ids,
            zmots=zmot_ids,
            initial_engagements=[],   # or pass occurred_ids if you’re seeding history
        )

        # Provide an initial tactical slice for ‘cold’ posture
        tactical = update_tactical_plan(
            product_subgraph,
            strategy,
            attributes=attr_ids,
            zmots=zmot_ids,
            engagements_to_date=occurred_ids,
            stage="cold",
        )

        # Prep graph for UI (optional preview)
        G_work, _ = generate_rcs(product_subgraph, engaged_nodes=engaged_payload)
        G_work = _set_node_labels_positions(G_work)
        out_graph = export_graph_for_d3(cleaned_graph_for_ui(G_work))

        return {"output": tactical, "frozen_strategy": strategy, "graph": out_graph}

    # ---- default: tactical path ----
    if not frozen_strategy:
        frozen_strategy = build_account_strategy(
            product_subgraph,
            attributes=attr_ids,
            zmots=zmot_ids,
            initial_engagements=[],  # or occurred_ids if you want to seed
        )

    tactical = update_tactical_plan(
        product_subgraph,
        frozen_strategy,
        attributes=attr_ids,
        zmots=zmot_ids,
        engagements_to_date=occurred_ids,
        stage="auto",
    )

    G_work, _ = generate_rcs(product_subgraph, engaged_nodes=engaged_payload)
    G_work = _set_node_labels_positions(G_work)
    out_graph = export_graph_for_d3(cleaned_graph_for_ui(G_work))

    # FIX: return the *frozen_strategy* here, not an undefined `strategy`
    return {"output": tactical, "frozen_strategy": frozen_strategy, "graph": out_graph}

# ---- viz helpers (unchanged) ----
def export_graph_for_d3(G: nx.DiGraph):
    nodes = []
    for node_id, data in G.nodes(data=True):
        node = {"id": node_id}
        node.update(data)
        nodes.append(node)
    links = []
    for source, target, data in G.edges(data=True):
        link = {"source": source, "target": target}
        link.update(data)
        links.append(link)
    return {"nodes": nodes, "links": links}

def _set_node_labels_positions(G: nx.DiGraph):
    for node_id in G.nodes:
        node = get_node_by_id(G, node_id)
        node_type = node.get("node_type", node.get("type", "unknown"))
        if node_type == "persona":
            title = node.get("title", "")
            department = node.get("department", "")
            seniority = node.get("seniority", "")
            node["label"] = f"{title} | {department} | {seniority}".strip(" |")
        elif node_type == "job":
            node["label"] = node.get("description", "") or f'Job {node_id}'
        elif node_type == "pain":
            node["label"] = node.get("description", "") or f'Pain {node_id}'
        elif node_type == "capability":
            node["label"] = node.get("name", "") or f'Capability {node_id}'
        elif node_type == "product":
            node["label"] = node.get("url", "") or f'Product {node_id}'
        elif node_type == "pain_trigger":
            node["label"] = node.get("attribute", "") or f'Pain Trigger {node_id}'
        elif node_type == "zmot_event":
            node["label"] = node.get("description", "") or f'ZMOT Event {node_id}'
        elif node_type == "observable_moment":
            node["label"] = node.get("description", "") or f'Observable Moment {node_id}'
        elif node_type == "keyword":
            node["label"] = node.get("keyword", "") or f'Keyword {node_id}'
        elif node_type == "perceived_metric":
            node["label"] = node.get("metric", "") or f'Perceived Metric {node_id}'
        else:
            node["label"] = node.get("description", "") or f'Node {node_id}'
    return G

def cleaned_graph_for_ui(G: nx.DiGraph):
    H = G.copy()
    keep = ["persona", "job", "pain", "capability", "product", "zmot_event", "pain_trigger"]
    for n in list(H.nodes):
        if H.nodes[n].get("node_type") not in keep:
            H.remove_node(n)
    for n in list(H.nodes):
        if n in H and H.degree(n) == 0 and H.nodes[n].get("node_type") != "product":
            H.remove_node(n)
    return H
