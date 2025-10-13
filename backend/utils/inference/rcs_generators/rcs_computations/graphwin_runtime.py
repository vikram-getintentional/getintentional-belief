# --- graphwin_runtime.py ----------------------------------------------------
from __future__ import annotations
from typing import Dict, List, Optional
import networkx as nx

from backend.utils.graph_base.network_graph import get_node_by_id

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _build_reversed_and_normalized(G: nx.DiGraph, weight: str = "likelihood") -> nx.DiGraph:
    """
    Reverse G so transitions go attribute/value -> ... -> product,
    then row-normalize outgoing weights to be a proper Markov chain.
    """
    Grev = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        w = float(d.get(weight, d.get("weight", 1.0)))
        Grev.add_edge(v, u, weight=max(0.0, w))

    for n in list(Grev.nodes()):
        out_edges = list(Grev.out_edges(n, data=True))
        if not out_edges:
            continue
        s = sum(e[2].get("weight", 0.0) for e in out_edges)
        if s <= 0.0:
            p = 1.0 / len(out_edges)
            for _, _, d in out_edges:
                d["prob"] = p
        else:
            for _, _, d in out_edges:
                d["prob"] = d.get("weight", 0.0) / s
    return Grev

def _ppr(
    Grev: nx.DiGraph,
    personalization: Dict[str, float],
    alpha: float = 0.85,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Dict[str, float]:
    pz = {n: 0.0 for n in Grev.nodes()}
    s = sum(v for k, v in personalization.items() if k in pz and v > 0)
    if s <= 0:
        return {n: 0.0 for n in Grev.nodes()}
    for k, v in personalization.items():
        if k in pz and v > 0:
            pz[k] = v / s
    return nx.pagerank(
        Grev,
        alpha=alpha,
        personalization=pz,
        weight="prob",
        max_iter=max_iter,
        tol=tol,
        dangling=pz,                  # <-- important
    )

def _mix_personalization_with_extra(
    pers_attr: Dict[str, float],
    extra_prior: Dict[str, float],
    *,
    beta: float,
    valid_nodes: Optional[set] = None,
) -> Dict[str, float]:
    """
    Blend attribute personalization with non-attribute prior:
       pers = (1 - beta) * norm(pers_attr)  +  beta * norm(extra_prior)
    If extra_prior is empty or beta=0, this reduces to norm(pers_attr).
    """
    valid_nodes = valid_nodes or set()

    # normalize attribute part
    a = {k: v for k, v in pers_attr.items() if (not valid_nodes or k in valid_nodes) and v > 0}
    sa = sum(a.values())
    if sa > 0:
        for k in list(a.keys()):
            a[k] /= sa
    else:
        a = {}

    # normalize extra part
    e = {k: v for k, v in extra_prior.items() if (not valid_nodes or k in valid_nodes) and v > 0}
    se = sum(e.values())
    if se > 0:
        for k in list(e.keys()):
            e[k] /= se
    else:
        e = {}

    if not e or beta <= 0.0:
        return a if a else {}

    out = {}
    one_minus = max(0.0, 1.0 - beta)
    for k, v in a.items():
        out[k] = out.get(k, 0.0) + one_minus * v
    for k, v in e.items():
        out[k] = out.get(k, 0.0) + beta * v
    return out

# --- NEW GraphWin (PPR-based) ------------------------------------------------

def graphwin_ppr_from_attributes(
    *,
    product_graph: nx.DiGraph,
    product_id: str,
    attribute_nodes: List[str],
    attribute_weights: Optional[Dict[str, float]] = None,  # defaults to 1.0 each
    zmot_nodes: Optional[List[str]] = None,
    zmot_weights: Optional[Dict[str, float]] = None,
    new_engaged_nodes: Optional[List[str]] = None,         # pains/jobs/personas/capabilities/etc
    new_engaged_weights: Optional[Dict[str, float]] = None,
    extra_prior_beta: float = 0.35,                        # tuning knob for non-attribute influence
    alpha: float = 0.85,
    weight_key: str = "likelihood",
) -> float:
    Grev = _build_reversed_and_normalized(product_graph, weight=weight_key)
    if product_id not in Grev:
        return 0.0

    # Attributes → main personalization (weighted)
    pers_attr: Dict[str, float] = {}
    for a in attribute_nodes or []:
        if a in Grev:
            w = 1.0 if attribute_weights is None else float(attribute_weights.get(a, 1.0))
            if w > 0:
                pers_attr[a] = pers_attr.get(a, 0.0) + w

    if not pers_attr:
        return 0.0  # strict: require at least one attribute

    # Non-attribute prior (ZМОТ + others) → extra prior
    extra_prior: Dict[str, float] = {}
    for z in zmot_nodes or []:
        if z in Grev:
            wz = 1.0 if zmot_weights is None else float(zmot_weights.get(z, 1.0))
            if wz > 0:
                extra_prior[z] = extra_prior.get(z, 0.0) + wz
    for n in new_engaged_nodes or []:
        if n in Grev:
            wn = 1.0 if new_engaged_weights is None else float(new_engaged_weights.get(n, 1.0))
            if wn > 0:
                extra_prior[n] = extra_prior.get(n, 0.0) + wn

    # Blend into one personalization vector
    pers = _mix_personalization_with_extra(
        pers_attr,
        extra_prior,
        beta=extra_prior_beta,
        valid_nodes=set(Grev.nodes()),
    )

    if not pers:
        return 0.0

    pr = _ppr(Grev, personalization=pers, alpha=alpha)
    return _clip01(float(pr.get(product_id, 0.0)))



def graphwin_ppr_union_by_attributes(
    *,
    product_graph: nx.DiGraph,
    product_id: str,
    attribute_nodes: List[str],
    zmot_nodes: Optional[List[str]] = None,
    new_engaged_nodes: Optional[List[str]] = None,
    zmot_weights: Optional[Dict[str, float]] = None,
    new_engaged_weights: Optional[Dict[str, float]] = None,
    extra_prior_beta: float = 0.35,   # same knob applies per single-attribute run
    alpha: float = 0.85,
    weight_key: str = "likelihood",
) -> float:
    """
    Monotonic aggregator:
      score = 1 - Π_attr (1 - score_attr)
    Where score_attr is a PPR run seeded at that attribute,
    blended with the same non-attribute extra prior.
    """
    Grev = _build_reversed_and_normalized(product_graph, weight=weight_key)
    if product_id not in Grev:
        return 0.0

    # Precompute extra prior once (ZМОТ + others)
    extra_prior: Dict[str, float] = {}
    for z in zmot_nodes or []:
        if z in Grev:
            wz = 1.0 if zmot_weights is None else float(zmot_weights.get(z, 1.0))
            if wz > 0:
                extra_prior[z] = extra_prior.get(z, 0.0) + wz
    for n in new_engaged_nodes or []:
        if n in Grev:
            wn = 1.0 if new_engaged_weights is None else float(new_engaged_weights.get(n, 1.0))
            if wn > 0:
                extra_prior[n] = extra_prior.get(n, 0.0) + wn

    terms: List[float] = []
    valid_nodes = set(Grev.nodes())

    for a in attribute_nodes or []:
        if a not in valid_nodes:
            continue
        # Single-attribute seed (weight=1), blended with extra prior
        pers_attr = {a: 1.0}
        pers = _mix_personalization_with_extra(
            pers_attr, extra_prior, beta=extra_prior_beta, valid_nodes=valid_nodes
        )
        if not pers:
            continue
        pr = _ppr(Grev, personalization=pers, alpha=alpha)
        terms.append(_clip01(float(pr.get(product_id, 0.0))))

    if not terms:
        return 0.0

    prod = 1.0
    for t in terms:
        prod *= (1.0 - t)
    return 1.0 - prod




def get_graphwin(
    G: nx.DiGraph,
    engaged_nodes: Optional[List[Dict]] = None,
    *,
    alpha: float = 0.85,
    weight_key: str = "likelihood",
    use_monotonic_union: bool = False,   # set False to use single PPR across all attributes
    debug: bool = False,
) -> Dict:
    engaged_nodes = engaged_nodes or []
    product_ids = [n for n, d in G.nodes(data=True) if d.get("node_type") == "product"]
    product_id = product_ids[0] if product_ids else None
    if not product_id:
        return {"win_likelihood": 0.0, "note": "no product"}

    # Collect attribute seeds (optionally weight by provided 'occurrence')
    attrs = []
    attr_weights: Dict[str, float] = {}
    zmots = []
    zmot_weights: Dict[str, float] = {}
    new_engaged_nodes = []
    new_engaged_weights: Dict[str, float] = {}
    for n in engaged_nodes:
        nid = n.get("id", "")
        if nid in G:
            node = get_node_by_id(G, nid)
            if not node:
                continue
            node_type = node.get("node_type")
            if node_type == "attribute_value":
                attrs.append(nid)
                attr_weights[nid] = float(n.get("occurrence", 1.0)) or 1.0
            elif node_type == "zmot_event":
                zmots.append(nid)
                zmot_weights[nid] = float(n.get("occurrence", 1.0)) or 1.0
            else:
                new_engaged_nodes.append(nid)
                new_engaged_weights[nid] = float(n.get("occurrence", 1.0)) or 1.0


    # Fallback: if no attributes were engaged, return 0 (or consider a default)
    if not attrs:
        return {"win_likelihood": 0.0, "note": "no attribute seeds"}

    if use_monotonic_union:
        win = graphwin_ppr_union_by_attributes(
            product_graph=G,
            product_id=product_id,
            attribute_nodes=attrs,
            zmot_nodes=zmots,
            new_engaged_nodes=new_engaged_nodes,
            alpha=alpha,
            weight_key=weight_key,
        )
    else:
        win = graphwin_ppr_from_attributes(
            product_graph=G,
            product_id=product_id,
            attribute_nodes=attrs,
            attribute_weights=attr_weights,
            zmot_nodes=zmots,
            zmot_weights=zmot_weights,
            new_engaged_nodes=new_engaged_nodes,
            new_engaged_weights=new_engaged_weights,

            alpha=alpha,
            weight_key=weight_key,
        )

    out = {"win_likelihood": float(win)}
    if debug:
        out["debug"] = {
            "product_id": product_id,
            "attributes": attrs,
            "attribute_weights": attr_weights,
            "mode": "union_monotonic" if use_monotonic_union else "single_ppr",
            "alpha": alpha,
        }
    return out
