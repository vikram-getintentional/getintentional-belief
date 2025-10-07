from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Iterable, Callable
import math
import numpy as np
import networkx as nx
import pandas as pd

from backend.utils.graph_base.network_graph import get_edge_attribute, get_node_by_id, get_source_nodes_by_target_and_type
from backend.utils.inference.crm_analysis.category_column_schema import CAT_COLS

# --------- Types ---------
GraphWinFn = Callable[[nx.DiGraph, List[Dict[str, Any]]], Dict[str, Any]]
GraphWinWithOverrideFn = Callable[
    [List[Dict[str, Any]], Dict[Tuple[str, str], float]], float
]
# override format: {(u_node_id, v_node_id): new_weight, ...}
def get_graphwin_override_factory(product_graph: nx.DiGraph,
                                  get_graphwin: GraphWinFn) -> GraphWinWithOverrideFn:
    """
    Returns a function (engaged_nodes, overrides)->p_win that temporarily mutates
    edge weights, calls get_graphwin(product_graph, engaged_nodes), then restores.
    """
    def fn(engaged_nodes: List[Dict[str, Any]],
           overrides: Dict[Tuple[str, str], float]) -> float:
        if not overrides:
            return float(get_graphwin(product_graph, engaged_nodes)["win_likelihood"])
        touched = []
        try:
            for (u, v), w in overrides.items():
                old = product_graph.get_edge_data(u, v)
                touched.append((u, v, old))
                ed = {} if old is None else old.copy()
                ed["weight"] = w
                product_graph.add_edge(u, v, **ed)
            return float(get_graphwin(product_graph, engaged_nodes)["win_likelihood"])
        finally:
            for (u, v, old) in touched:
                if old is None:
                    if product_graph.has_edge(u, v):
                        product_graph.remove_edge(u, v)
                else:
                    product_graph.add_edge(u, v, **old)
    return fn

@dataclass
class EdgeProposal:
    edge_from: str
    edge_to: str
    edge_type: str
    layer: str
    current_weight: float
    proposed_weight: float
    delta: float
    support_n: int
    confidence: float
    reason: str

# --------- Config ---------
DEFAULT_EPS = 1e-2         # finite-diff epsilon for sensitivity
DEFAULT_LR = 0.5           # learning rate for weight updates
MIN_SUPPORT = 3            # edges need at least this many rows supporting
MAX_STEP = 0.25            # clamp single-step change magnitude
WEIGHT_CLIP = (1e-3, 1.0-1e-3)

# --------- Helpers ---------
def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _mean(x: Iterable[float]) -> float:
    x = list(x)
    return float(np.mean(x)) if x else float("nan")

def _std(x: Iterable[float]) -> float:
    x = list(x)
    return float(np.std(x)) if x else float("nan")

def layer_spec(name: str) -> Tuple[str, str, str]:
    """
    Maps friendly layer name → (src_type, dst_type, edge_type)
    Adjust types/edge_type strings to match your graph schema.
    """
    if name in ("attr->pain_trigger", "attribute_value->pain_trigger"):
        return ("attribute_value", "pain_trigger", "prevalent_in")
    if name in ("pain_trigger->pain",):
        return ("pain_trigger", "pain", "triggered_by")
    if name in ("pain->job",):
        return ("pain", "job", "solves")
    if name in ("job->pain",):
        return ("job", "pain", "felt_in")
    if name in ("pain->product_capability", "pain->capability"):
        return ("pain", "capability", "solves")
    raise ValueError(f"Unknown layer name: {name}")

# --------- Dynamic traversal rules ---------
EDGE_RULES = {
    "attribute_value": "prevalent_in",
    "pain_trigger": "triggered_by",
    "pain": "solves",
    "job": "felt_in",
    "capability": "offered_by",
}

def infer_type(nid: str, G: nx.DiGraph) -> Optional[str]:
    t = G.nodes[nid].get("node_type") or G.nodes[nid].get("type")
    if t:
        return t
    # fallback to prefix inference
    return nid.split(":", 1)[0] if ":" in nid else None


def walk_upstream_dynamic(
    G: nx.DiGraph,
    seed_nodes: List[str],
    max_depth: int = 5,
) -> Dict[int, Dict[str, List[str]]]:
    """
    Generic upstream walker.
    Uses EDGE_RULES to decide what edge type to follow at each hop.
    Returns dict:
        { depth: {"nodes": [node_ids], "edges": [(src, dst)]} }
    """
    visited = set(seed_nodes)
    depth_nodes = {0: {"nodes": seed_nodes, "edges": []}}
    frontier = list(seed_nodes)
    depth = 0

    while frontier and depth < max_depth:
        next_frontier = []
        edge_type_map = {}
        for tgt in frontier:
            if tgt not in G:
                continue
            ttype = infer_type(tgt, G)
            if not ttype:
                continue
            edge_type = EDGE_RULES.get(ttype)
            if not edge_type:
                continue
            edge_type_map[tgt] = edge_type

        depth_edges = []
        for tgt, edge_type in edge_type_map.items():
            for src, _, ed in G.in_edges(tgt, data=True):
                if ed.get("edge_type") == edge_type or ed.get("type") == edge_type:
                    depth_edges.append((src, tgt))
                    if src not in visited:
                        visited.add(src)
                        next_frontier.append(src)
        depth += 1
        if not depth_edges:
            break
        depth_nodes[depth] = {
            "nodes": next_frontier,
            "edges": depth_edges
        }
        frontier = next_frontier

    return depth_nodes


def engaged_nodes_to_attr_src_ids(engaged_nodes: List[Dict[str, Any]]) -> List[str]:
    """
    Extract attribute_value node IDs from your engaged_nodes list.
    Your engaged_nodes come in order: industry, revenue, employees, geo, funding.
    We only return those that look like node IDs (contain ':').
    """
    src_ids = []
    for item in engaged_nodes:
        nid = item.get("id")
        if isinstance(nid, str) and ":" in nid:
            src_ids.append(nid)
    return src_ids

def account_core_attrs(row: pd.Series) -> List[Dict[str, Any]]:
    """
    Return engaged_nodes list built from *_id columns (preferred)
    or from graph_attrs if already present.
    """
    # 1. If the compare step already stored engaged_nodes list → use it
    ga = row.get("graph_attrs")
    if isinstance(ga, list) and len(ga) > 0:
        return ga

    # 2. Otherwise build it from *_id columns
    engaged_nodes = []
    for col in CAT_COLS:
        node_id = None
        id_col = f"{col}_id"
        if id_col in row and isinstance(row[id_col], str) and row[id_col]:
            node_id = row[id_col]
        elif col in row and isinstance(row[col], str) and row[col]:
            node_id = row[col]
        if node_id:
            engaged_nodes.append({
                "id": node_id,
                "type": "attribute_value",
                "dimension": col
            })
    return engaged_nodes

def edges_from_srcs(G: nx.DiGraph, src_nodes: List[str], layer: str) -> List[Tuple[str, str]]:
    src_t, dst_t, edge_type = layer_spec(layer)
    print("Layer Spec Output:", src_t, dst_t)
    out = []
    for s in src_nodes:
        # ensure the node exists and has the right type
        if s not in G.nodes:
            continue
        ntype = G.nodes[s].get("node_type") or G.nodes[s].get("type")
        if ntype != src_t:
            continue
        down_nodes = get_source_nodes_by_target_and_type(G, s, edge_type)
        if not down_nodes:
            print(f"No downstream nodes from {s} via {edge_type}")
            continue
        for v in down_nodes:
            v_node = get_node_by_id(G, v)
            if v_node.get("node_type") == dst_t:
                out.append((s,v))
    return out


def set_edge_weight(G: nx.DiGraph, u: str, v: str, w: float, key: str = "weight"):
    ed = G.get_edge_data(u, v) or {}
    ed[key] = w
    G.add_edge(u, v, **ed)


# --------- Sensitivity (finite difference) ---------
def estimate_edge_sensitivity(
    G: nx.DiGraph,
    get_graphwin_override: GraphWinWithOverrideFn,
    attrs: Dict[str, str],
    edge: Tuple[str, str],
    base_p: Optional[float] = None,
    eps: float = DEFAULT_EPS,
) -> float:
    """
    Sensitivity ≈ d p_win / d w_edge via one-sided finite difference.
    """
    u, v = edge
    w0 = get_edge_attribute(G, u, v, "likelihood")
    if base_p is None:
        base_p = float(get_graphwin_override(attrs, {}))
    w_upd = _clip(w0 + eps, *WEIGHT_CLIP)
    dp = float(get_graphwin_override(attrs, {(u, v): w_upd})) - base_p
    return dp / (w_upd - w0 + 1e-12)

# --------- MRCA selection ---------
def mrca_edges_for_group(
    G: nx.DiGraph,
    layer: str,
    group_rows: pd.DataFrame,
) -> List[Tuple[str, str]]:
    """
    Simplified MRCA notion for a layer:
    - collect all candidate edges in this layer that are reachable from the group's sources
      (sources = cat col nodes; for attr->pain_trigger, sources are the 4 attribute_value nodes)
    - pick edges that appear across the **majority** of rows in the group (intersection heuristic)
    """
    src_t, dst_t, _ = layer_spec(layer)
    # collect per-row edge sets
    per_row_edges: List[set[Tuple[str, str]]] = []
    for _, r in group_rows.iterrows():
        attrs = account_core_attrs(r)
        print("attrs in mrca: ", attrs)
        srcs = []
        for i, col in enumerate(CAT_COLS):
            if i < len(attrs) and isinstance(attrs[i], dict):
                srcs.append(attrs[i].get("id"))
            else:
                srcs.append(None)
        srcs = [s for s in srcs if isinstance(s, str)]
        print(f"Row sources for layer {layer}:", srcs)
        eds = set(edges_from_srcs(G, srcs, layer))
        print(f"Row edges for layer {layer}:", eds)
        per_row_edges.append(eds)

    if not per_row_edges:
        return []
    print("Per-row edges in mrca:", per_row_edges)
    # majority vote (MRCA-like): edge must appear in >= 60% of rows
    counts: Dict[Tuple[str, str], int] = {}
    for s in per_row_edges:
        for e in s:
            counts[e] = counts.get(e, 0) + 1
    thresh = max(1, int(0.6 * len(per_row_edges)))
    mrca = [e for e, c in counts.items() if c >= thresh]
    return mrca

# --------- Proposal generator for a single layer ---------
def propose_edges_for_layer(
    G: nx.DiGraph,
    df_align: pd.DataFrame,
    layer: str,
    get_graphwin_override: GraphWinWithOverrideFn,
    min_support: int = MIN_SUPPORT,
    eps: float = DEFAULT_EPS,
    lr: float = DEFAULT_LR,
    max_step: float = MAX_STEP,
) -> List[EdgeProposal]:
    """
    Use residuals to suggest edge tweaks in the given layer.
    - Group rows by archetype (industry,revenue,employees,geo,funding_stage)
    - For top-|residual| groups, pick MRCA edges in this layer
    - For each candidate edge, aggregate gradients across supporting rows:
         grad ≈ mean( - residual_i * sensitivity_i )
      and propose: w_new = clip( w_old - lr * grad, WEIGHT_CLIP ), clamped by max_step
    - Confidence ~ shrinkage by support and gradient stability
    """
    # Build archetype key
    key_cols = list(CAT_COLS)
    for c in key_cols:
        if c not in df_align.columns:
            raise ValueError(f"df_align missing column: {c}")
    d = df_align.dropna(subset=["p_crm_win","p_graph_win"]).copy()
    d["residual"] = d["p_crm_win"] - d["p_graph_win"]
    

    # rank groups by |mean residual|
    grp = (d.groupby(key_cols, dropna=False)
             .agg(count=("account_id","count"),
                  mean_resid=("residual","mean"))
             .reset_index())
    grp["abs_resid"] = grp["mean_resid"].abs()
    grp = grp.sort_values("abs_resid", ascending=False)

    proposals: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # iterate top groups (you can also cap at top K)
    for _, g in grp.iterrows():
        arche = {k:g[k] for k in key_cols}
        rows = d[(d[key_cols] == pd.Series(arche)).all(axis=1)]
        if rows.empty:
            continue

        # candidate edges from MRCA in this layer
        candidates = mrca_edges_for_group(G, layer, rows)
        print("MRCA Completed with candidates:", candidates)
        if not candidates:
            continue

        # per-edge accumulate gradients across rows
        for e in candidates:
            u, v = e
            supports = 0
            grads = []
            # evaluate only on rows where this edge exists in their reachable set
            for _, r in rows.iterrows():
                attrs = account_core_attrs(r)
                # quick membership test: do src attributes include this edge's source?
                if u not in {item.get("id") for item in attrs if isinstance(item, dict)}:
                    print("Candidate node not in attr list - skipping")
                    continue
                # base & sensitivity
                try:
                    base_p = float(get_graphwin_override(attrs, {}))
                    sens = estimate_edge_sensitivity(G, get_graphwin_override, attrs, e, base_p, eps)
                    # gradient of 1/2*(crm-graph)^2 wrt w ≈ -(crm-graph)*sens
                    grad = -(r["residual"]) * sens
                    grads.append(grad)
                    supports += 1
                except Exception:
                    continue

            if supports < min_support:
                continue

            gmean = _mean(grads)
            gstd  = _std(grads)
            w0 = get_edge_attribute(G, u, v, "likelihood")
            step = _clip(-lr * gmean, -max_step, max_step)  # clamp
            w1 = _clip(w0 + step, *WEIGHT_CLIP)
            # --- EDIT 2: Only keep proposals with significant delta and confidence ---
            if abs(w1 - w0) < 1e-3 or confidence < 0.5:
                continue  # skip weak proposals

            # simplistic confidence: grows with support, shrinks with gradient dispersion
            conf_support = 1.0 - math.exp(-supports / 10.0)  # ~ saturates after ~20
            conf_stability = 1.0 / (1.0 + gstd) if math.isfinite(gstd) else 0.5
            confidence = float(_clip(0.5 * conf_support + 0.5 * conf_stability, 0.0, 1.0))

            proposals[e] = {
                "edge_from": u,
                "edge_to": v,
                "edge_type": layer_spec(layer)[2],
                "layer": layer,
                "current_weight": w0,
                "proposed_weight": w1,
                "delta": w1 - w0,
                "support_n": supports,
                "confidence": confidence,
                "reason": f"MRCA group={arche}, mean_resid={g['mean_resid']:.3f}, sens_mean={gmean:.3f}, n={supports}"
            }

    # return as a list sorted by confidence * |delta|
    out = [EdgeProposal(**p) for p in proposals.values()]
    out.sort(key=lambda x: (x.confidence * abs(x.delta)), reverse=True)
    return out

# --------- Full run (layer-wise) ---------
def run_reconciliation(
    G: nx.DiGraph,
    df_align: pd.DataFrame,
    get_graphwin_override: GraphWinWithOverrideFn,
    eps: float = DEFAULT_EPS,
    lr: float = DEFAULT_LR,
    min_support: int = MIN_SUPPORT,
    max_depth: int = 5,
) -> Dict[str, Any]:
    """
    Dynamic reconciliation:
    - Computes residuals
    - Walks upstream using EDGE_RULES (no hard-coded layers)
    - Groups residuals by archetype
    - Finds MRCA-like common upstream edges
    - Proposes edge weight corrections
    """
    # compute residuals
    d = df_align.dropna(subset=["p_crm_win","p_graph_win"]).copy()
    d["residual"] = d["p_crm_win"] - d["p_graph_win"]

    key_cols = list(CAT_COLS)
    grp = (d.groupby(key_cols, dropna=False)
             .agg(count=("account_id","count"),
                  mean_resid=("residual","mean"))
             .reset_index())
    grp["abs_resid"] = grp["mean_resid"].abs()
    grp = grp.sort_values("abs_resid", ascending=False)

    proposals: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for _, g in grp.iterrows():
        arche = {k:g[k] for k in key_cols}
        rows = d[(d[key_cols] == pd.Series(arche)).all(axis=1)]
        if rows.empty:
            continue

        # ---- dynamic traversal & MRCA candidate discovery ----
        per_row_edges = []
        for _, r in rows.iterrows():
            attrs = account_core_attrs(r)
            attr_ids = [a["id"] for a in attrs if isinstance(a, dict)]
            walk = walk_upstream_dynamic(G, attr_ids, max_depth=max_depth)
            # flatten all edges traversed
            all_edges = set()
            for dmap in walk.values():
                all_edges |= set(dmap.get("edges", []))
            per_row_edges.append(all_edges)

        if not per_row_edges:
            continue

        counts = {}
        for s in per_row_edges:
            for e in s:
                counts[e] = counts.get(e, 0) + 1
        thresh = max(1, int(0.6 * len(per_row_edges)))
        candidates = [e for e, c in counts.items() if c >= thresh]

        # ---- gradient & proposal step ----
        for e in candidates:
            u, v = e
            supports, grads = 0, []
            for _, r in rows.iterrows():
                attrs = account_core_attrs(r)
                try:
                    base_p = float(get_graphwin_override(attrs, {}))
                    sens = estimate_edge_sensitivity(G, get_graphwin_override, attrs, e, base_p, eps)
                    grad = -(r["residual"]) * sens
                    grads.append(grad)
                    supports += 1
                except Exception:
                    continue
            if supports < min_support:
                continue

            gmean = _mean(grads)
            gstd  = _std(grads)
            w0 = get_edge_attribute(G, u, v, "likelihood")
            step = _clip(-lr * gmean, -MAX_STEP, MAX_STEP)
            w1 = _clip(w0 + step, *WEIGHT_CLIP)

            conf_support = 1.0 - math.exp(-supports / 10.0)
            conf_stability = 1.0 / (1.0 + gstd) if math.isfinite(gstd) else 0.5
            confidence = float(_clip(0.5 * conf_support + 0.5 * conf_stability, 0.0, 1.0))
            if abs(w1 - w0) < 1e-3 or confidence < 0.5:
                continue

            proposals[e] = {
                "edge_from": u,
                "edge_to": v,
                "edge_type": G[u][v].get("edge_type", ""),
                "layer": "auto",  # dynamic
                "current_weight": w0,
                "proposed_weight": w1,
                "delta": w1 - w0,
                "support_n": supports,
                "confidence": confidence,
                "reason": f"dynamic MRCA, mean_resid={g['mean_resid']:.3f}, sens_mean={gmean:.3f}, n={supports}"
            }

    out = [EdgeProposal(**p) for p in proposals.values()]
    out.sort(key=lambda x: (x.confidence * abs(x.delta)), reverse=True)
    print(f"[Dynamic reconciliation] Generated {len(out)} proposals.")
    return {"proposals": [p.__dict__ for p in out]}
