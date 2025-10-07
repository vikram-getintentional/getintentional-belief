# --- graphwin_runtime.py ----------------------------------------------------
from __future__ import annotations
from typing import Dict, List, Optional
import networkx as nx

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

# --- NEW GraphWin (PPR-based) ------------------------------------------------

def graphwin_ppr_union_by_attributes(
    *,
    product_graph: nx.DiGraph,
    product_id: str,
    attribute_nodes: List[str],
    alpha: float = 0.85,
    weight_key: str = "likelihood",
) -> float:
    Grev = _build_reversed_and_normalized(product_graph, weight=weight_key)
    if product_id not in Grev:
        return 0.0

    terms = []
    for attr in attribute_nodes:
        if attr not in Grev:
            continue
        pr = _ppr(Grev, personalization={attr: 1.0}, alpha=alpha)
        terms.append(_clip01(float(pr.get(product_id, 0.0))))

    prod = 1.0
    for a in terms:
        prod *= (1.0 - a)
    return 1.0 - prod


def graphwin_ppr_from_attributes(
    *,
    product_graph: nx.DiGraph,
    product_id: str,
    attribute_nodes: List[str],
    attribute_weights: Optional[Dict[str, float]] = None,  # defaults to 1.0 each
    alpha: float = 0.85,
    weight_key: str = "likelihood",
) -> float:
    # Single Pure PPR - non Monotonic
    Grev = _build_reversed_and_normalized(product_graph, weight=weight_key)
    if product_id not in Grev:
        return 0.0

    pers: Dict[str, float] = {}
    for a in attribute_nodes:
        if a in Grev:
            pers[a] = (attribute_weights.get(a, 1.0) if attribute_weights else 1.0)

    if not pers:
        return 0.0

    pr = _ppr(Grev, personalization=pers, alpha=alpha)
    return _clip01(float(pr.get(product_id, 0.0)))



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
    for n in engaged_nodes:
        nid = n.get("id", "")
        if nid.startswith("attribute_value:") and nid in G:
            attrs.append(nid)
            attr_weights[nid] = float(n.get("occurrence", 1.0)) or 1.0

    # Fallback: if no attributes were engaged, return 0 (or consider a default)
    if not attrs:
        return {"win_likelihood": 0.0, "note": "no attribute seeds"}

    if use_monotonic_union:
        win = graphwin_ppr_union_by_attributes(
            product_graph=G,
            product_id=product_id,
            attribute_nodes=attrs,
            alpha=alpha,
            weight_key=weight_key,
        )
    else:
        win = graphwin_ppr_from_attributes(
            product_graph=G,
            product_id=product_id,
            attribute_nodes=attrs,
            attribute_weights=attr_weights,
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
