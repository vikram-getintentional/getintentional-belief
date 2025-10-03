# ============================
# File: backend/utils/inference/rcs_generators/rcs_helpers/reach.py
# ============================
from __future__ import annotations
from typing import Dict, Iterable, List
import networkx as nx

def reverse_reach_to_product_bulk(
    G_pruned: nx.DiGraph,
    *,
    product_id: str,
    nodes: Iterable[str],
    alpha: float = 0.85,    # damping
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """
    For each node s in `nodes`, run Personalized PageRank on the *reversed* graph
    seeded at s, and return the stationary mass at product_id as reach(s -> product).

    Assumes forward edges are product -> capability -> job -> pain -> pain_trigger.
    We reverse so the walk is: pain_trigger -> pain -> job -> capability -> product.
    """
    Grev = G_pruned.reverse(copy=False)
    out: Dict[str, float] = {}
    if product_id not in Grev:
        # If product missing, nothing is reachable
        return {s: 0.0 for s in nodes}

    for s in nodes:
        if s not in Grev:
            out[s] = 0.0
            continue
        try:
            pers = {n: 0.0 for n in Grev.nodes}
            pers[s] = 1.0
            pr = nx.pagerank(Grev, alpha=alpha, personalization=pers, max_iter=max_iter, tol=tol)
            out[s] = float(pr.get(product_id, 0.0))
        except Exception:
            out[s] = 0.0
    return out

def reverse_reach_to_product_for_triggers(
    G_pruned: nx.DiGraph,
    *,
    product_id: str,
    pain_triggers: List[str],
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """
    Convenience wrapper when you only care about triggers.
    """
    return reverse_reach_to_product_bulk(
        G_pruned,
        product_id=product_id,
        nodes=pain_triggers,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
    )
