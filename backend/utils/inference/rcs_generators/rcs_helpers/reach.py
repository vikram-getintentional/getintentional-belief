# ============================
# File: backend/utils/inference/rcs_generators/reach.py
# ============================
from __future__ import annotations
from typing import Dict, List
import networkx as nx

def _safe_pagerank(G: nx.DiGraph, *, alpha: float, personalization: Dict[str, float], max_iter: int = 100, tol: float = 1e-8):
    try:
        return nx.pagerank(G, alpha=alpha, personalization=personalization, max_iter=max_iter, tol=tol)
    except Exception:
        # fallback: uniform
        n = max(1, G.number_of_nodes())
        return {node: 1.0 / n for node in G.nodes}

def reverse_reach_to_product(
    G_pruned: nx.DiGraph,
    *,
    product_id: str,
    pain_triggers: List[str],
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-08,
) -> Dict[str, float]:
    """
    For each pain_trigger t, run Personalized PageRank on the *reversed* graph,
    seeded at t, and return the stationary mass at product_id as the reach proxy R_t.
    """
    Grev = G_pruned.reverse(copy=False)
    out: Dict[str, float] = {}
    for t in pain_triggers:
        if t not in Grev or product_id not in Grev:
            out[t] = 0.0
            continue
        pers = {n: 0.0 for n in Grev.nodes}
        pers[t] = 1.0
        pr = _safe_pagerank(Grev, alpha=alpha, personalization=pers, max_iter=max_iter, tol=tol)
        out[t] = float(pr.get(product_id, 0.0))
    return out

def reverse_reach_from_nodes(
    G_pruned: nx.DiGraph,
    *,
    product_id: str,
    source_nodes: List[str],
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-08,
) -> Dict[str, float]:
    """
    For each observed downstream node (pain/job/trigger), compute its reverse-PPR reach to product.
    These are added with prior=1.0 in GraphWin.
    """
    if not source_nodes:
        return {}
    Grev = G_pruned.reverse(copy=False)
    out: Dict[str, float] = {}
    for s in source_nodes:
        if s not in Grev or product_id not in Grev:
            out[s] = 0.0
            continue
        pers = {n: 0.0 for n in Grev.nodes}
        pers[s] = 1.0
        pr = _safe_pagerank(Grev, alpha=alpha, personalization=pers, max_iter=max_iter, tol=tol)
        out[s] = float(pr.get(product_id, 0.0))
    return out
