# ============================
# File: backend/utils/inference/rcs_generators/rcs_simulator.py
# ============================
from __future__ import annotations
from collections import defaultdict
import inspect
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np

from backend.utils.graph_base.network_graph import (
    get_node_by_id,
    get_product_id_from_subgraph,
)
from backend.utils.inference.rcs_generators.rcs_helpers.strategy_orchestrators import (
    build_account_strategy,
    update_tactical_plan,
)

# -------------------------------------------------------
# JSON sanitation helpers
# -------------------------------------------------------
def _export_if_graph(obj):
    if isinstance(obj, nx.DiGraph):
        return export_graph_for_d3(cleaned_graph_for_ui(obj))
    return obj

def _json_sanitize(obj):
    """Deep-sanitize to JSON-safe primitives."""
    obj = _export_if_graph(obj)

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)

    # sets/tuples/lists
    if isinstance(obj, (set, tuple, list)):
        return [_json_sanitize(x) for x in obj]

    # dict-like
    if isinstance(obj, (dict, defaultdict)):
        out = {}
        for k, v in obj.items():
            # keys must be strings for JSON
            key = str(k)
            out[key] = _json_sanitize(v)
        return out

    # Fallback: stringify unknowns
    try:
        return _export_if_graph(obj)
    except Exception:
        return str(obj)

# -------------------------------------------------------
# Introspection helpers
# -------------------------------------------------------
def _filter_kwargs(func, **kwargs) -> Dict[str, Any]:
    """Keep only kwargs the target function accepts."""
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}

def _expects(func, name: str) -> bool:
    return name in inspect.signature(func).parameters

def _infer_product_id(G: nx.DiGraph, fallback: Optional[str]) -> str:
    try:
        pid = get_product_id_from_subgraph(G)
        if pid:
            return pid
    except Exception:
        pass
    return fallback or "product:simulated"

def _infer_account_id(fallback: Optional[str]) -> str:
    return fallback or "acct:simulated"

def _as_occurrence_payload(ids: List[str]) -> List[Dict[str, Any]]:
    """Coerce a list of node-ids into the richer payload newer code expects."""
    pay = []
    seen = set()
    for nid in ids:
        if not nid or nid in seen:
            continue
        seen.add(nid)
        pay.append({"id": nid, "occurrence": 1.0})
    return pay

# -------------------------------------------------------
# Safe callers with auto-filled args and shape normalization
# -------------------------------------------------------
def _safe_call_build_account_strategy(
    G: nx.DiGraph,
    *,
    product_id: Optional[str],
    account_id: Optional[str],
    engaged_nodes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, nx.DiGraph, Dict[str, Any]]:
    """
    Calls build_account_strategy with only supported kwargs and normalizes return shape to:
        (frozen_strategy, graph, report)
    """
    engaged_nodes = engaged_nodes or []

    kw = {
        "G": G,
        "product_id": product_id,
        "account_id": account_id,
        # common variants seen in codebase
        "engaged_nodes": engaged_nodes,
        "initial_engagements": engaged_nodes,
    }

    if _expects(build_account_strategy, "product_id") and "product_id" not in kw:
        kw["product_id"] = _infer_product_id(G, product_id)
    if _expects(build_account_strategy, "account_id") and "account_id" not in kw:
        kw["account_id"] = _infer_account_id(account_id)

    call_kwargs = _filter_kwargs(build_account_strategy, **kw)
    result = build_account_strategy(**call_kwargs)

    # -------- Normalize return shape to 3-tuple --------
    if isinstance(result, tuple):
        if len(result) == 3:
            frozen_strategy, graph_out, report = result
            return frozen_strategy, (graph_out or G), (report or {})
        if len(result) == 2:
            frozen_strategy, graph_out = result
            return frozen_strategy, (graph_out or G), {}
        if len(result) == 1:
            return result[0], G, {}
        return result, G, {}
    if isinstance(result, dict):
        return result, G, {}
    if isinstance(result, list):
        return result, G, {}
    return [result], G, {}

def _safe_call_update_tactical_plan(
    G: nx.DiGraph,
    *,
    frozen_strategy: Optional[Dict] = None,
    product_id: Optional[str],
    account_id: Optional[str],
    engagements_to_date: Optional[List[Dict[str, Any]]] = None,
    stage: str = "auto",
) -> Tuple[Any, nx.DiGraph]:
    """
    Calls update_tactical_plan with only supported kwargs and normalizes return shape to:
        (tactical_update, graph)
    """
    engagements_to_date = engagements_to_date or []

    kw = {
        "G": G,
        "frozen_strategy": frozen_strategy,
        "product_id": product_id,
        "account_id": account_id,
        # support either name
        "engagements_to_date": engagements_to_date,
        "engaged_nodes": engagements_to_date,
        "stage": stage,
    }

    if _expects(update_tactical_plan, "product_id") and "product_id" not in kw:
        kw["product_id"] = _infer_product_id(G, product_id)
    if _expects(update_tactical_plan, "account_id") and "account_id" not in kw:
        kw["account_id"] = _infer_account_id(account_id)

    call_kwargs = _filter_kwargs(update_tactical_plan, **kw)
    result = update_tactical_plan(**call_kwargs)

    # -------- Normalize return shape to 2-tuple --------
    if isinstance(result, tuple):
        if len(result) == 2:
            tactical_update, graph_out = result
            return tactical_update, (graph_out or G)
        if len(result) == 1:
            return result[0], G
        return result, G
    if isinstance(result, dict):
        return result, G
    if isinstance(result, list):
        return result, G
    return [result], G

# -------------------------------------------------------
# Core simulation orchestrator
# -------------------------------------------------------
def simulate_rcs(
    product_subgraph: nx.DiGraph,
    *,
    engaged_nodes: Optional[List[str]] = None,
    attributes: Optional[List[str]] = None,
    mode: str = "tactical",            # "strategy" | "tactical"
    frozen_strategy: Optional[Dict] = None,
    product_id: Optional[str] = None,
    account_id: Optional[str] = None,
) -> Dict:
    """
    Unified simulation entrypoint:
      - "strategy" mode builds a frozen baseline using ATTRIBUTES + ZMOTs as engagements.
      - "tactical" mode applies ATTRIBUTES + ZMOTs + OCCURRED nodes.
    """
    engaged_nodes = [n for n in (engaged_nodes or []) if n]   # drop blanks
    attributes = [a for a in (attributes or []) if a]         # drop blanks
    zmots: List[str] = []
    occurred_nodes: List[str] = []

    # Infer canonical ids if not provided
    _product_id = _infer_product_id(product_subgraph, product_id)
    _account_id = _infer_account_id(account_id)

    # Normalize payload (strings or dicts with {"id":...})
    normalized_ids: List[str] = []
    for n in set(engaged_nodes + attributes):
        nid = n.get("id") if isinstance(n, dict) else n
        if nid:
            normalized_ids.append(nid)

    # Partition by node_type (if present)
    for nid in normalized_ids:
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

    # If absolutely nothing selected, seed with product so we don't return empties
    if not occurred_nodes and not zmots and not attributes:
        if _product_id in product_subgraph.nodes:
            occurred_nodes.append(_product_id)

    # === Build the exact engagement lists per your rule ===
    engaged_for_strategy_ids = list(dict.fromkeys(attributes + zmots))
    engaged_for_tactical_ids = list(dict.fromkeys(attributes + zmots + occurred_nodes))

    engaged_for_strategy = _as_occurrence_payload(engaged_for_strategy_ids)
    engaged_for_tactical = _as_occurrence_payload(engaged_for_tactical_ids)

    # ---------------------------------------------------
    # STRATEGY MODE
    # ---------------------------------------------------
    if mode == "strategy":
        frozen_strategy_out, fs_pain_graph, report = _safe_call_build_account_strategy(
            product_subgraph,
            product_id=_product_id,
            account_id=_account_id,
            engaged_nodes=engaged_for_strategy,
        )

        tactical_update, tu_pain_graph = _safe_call_update_tactical_plan(
            product_subgraph,
            frozen_strategy=frozen_strategy_out,
            product_id=_product_id,
            account_id=_account_id,
            engagements_to_date=engaged_for_tactical,  # includes occurred + attrs + zmots
            stage="cold",
        )

        graph_for_ui = export_graph_for_d3(
            _set_node_labels_positions(cleaned_graph_for_ui(fs_pain_graph or product_subgraph))
        )
        return _json_sanitize({
            "tactical_update": tactical_update,
            "frozen_strategy": frozen_strategy_out,
            "graph": graph_for_ui,
            "report": report,
        })

    # ---------------------------------------------------
    # TACTICAL MODE
    # ---------------------------------------------------
    report: Dict[str, Any] = {}
    frozen_strategy_work = frozen_strategy

    if not frozen_strategy_work:
        frozen_strategy_work, fs_pain_graph, report = _safe_call_build_account_strategy(
            product_subgraph,
            product_id=_product_id,
            account_id=_account_id,
            engaged_nodes=engaged_for_strategy,  # baseline = attrs + zmots
        )

    tactical_update, tu_pain_graph = _safe_call_update_tactical_plan(
        product_subgraph,
        frozen_strategy=frozen_strategy_work,
        product_id=_product_id,
        account_id=_account_id,
        engagements_to_date=engaged_for_tactical,  # attrs + zmots + occurred
        stage="auto",
    )

    graph_for_ui = export_graph_for_d3(
        _set_node_labels_positions(cleaned_graph_for_ui(tu_pain_graph or product_subgraph))
    )

    return _json_sanitize({
        "tactical_update": tactical_update,
        "frozen_strategy": frozen_strategy_work,
        "graph": graph_for_ui,
        "report": report,
    })

# -------------------------------------------------------
# Graph cleanup / visualization utilities
# -------------------------------------------------------
def export_graph_for_d3(G: Optional[nx.DiGraph]):
    """Convert NetworkX graph to D3-compatible JSON format."""
    if G is None:
        return {"nodes": [], "links": []}
    nodes = [{"id": nid, **data} for nid, data in G.nodes(data=True)]
    links = [{"source": u, "target": v, **data} for u, v, data in G.edges(data=True)]
    return {"nodes": nodes, "links": links}

def _set_node_labels_positions(G: nx.DiGraph):
    """Attach readable labels to nodes for UI display."""
    for node_id in list(G.nodes):
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
        if n in H and H.degree(n) == 0 and H.nodes[n].get("node_type") != "product":
            H.remove_node(n)
    return H
