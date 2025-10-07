import math
from typing import Dict, Iterable, Optional, List
import networkx as nx

def _build_reversed_and_normalized(G: nx.DiGraph, weight: str = "likelihood") -> nx.DiGraph:
    """
    Reverse the graph so that transition goes attribute -> ... -> product,
    and row-normalize outgoing weights into probabilities.
    Falls back to edge["weight"] or 1.0 if `weight` doesn't exist.
    """
    Grev = nx.DiGraph()
    # Reverse edges
    for u, v, d in G.edges(data=True):
        w = d.get(weight, d.get("weight", 1.0))
        Grev.add_edge(v, u, weight=float(w))

    # Normalize out-weights to sum to 1.0 per node (Markov row-stochastic)
    for n in list(Grev.nodes()):
        out_edges = list(Grev.out_edges(n, data=True))
        if not out_edges:
            continue
        s = sum(max(0.0, e[2].get("weight", 0.0)) for e in out_edges)
        if s <= 0.0:
            # if all zero, make them uniform
            p = 1.0 / len(out_edges)
            for _, _, d in out_edges:
                d["prob"] = p
        else:
            for _, _, d in out_edges:
                d["prob"] = max(0.0, d.get("weight", 0.0)) / s
    return Grev


def ppr_win_score(
    G: nx.DiGraph,
    seed_nodes: Iterable[str],
    product_id: str,
    alpha: float = 0.85,
    weight: str = "likelihood",
    personalization_weights: Optional[Dict[str, float]] = None,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> float:
    """
    Personalized PageRank WIN score of product given seed nodes.

    - Reverses and normalizes the graph.
    - Builds personalization r over seed_nodes (uniform unless personalization_weights provided).
    - Returns ppr[product_id].

    This is a "reachability strength" measure (tempered by restart).
    """
    Grev = _build_reversed_and_normalized(G, weight=weight)

    # Build personalization vector r
    personalization: Dict[str, float] = {n: 0.0 for n in Grev.nodes()}
    if personalization_weights:
        s = sum(max(0.0, w) for w in personalization_weights.values())
        if s > 0:
            for n, w in personalization_weights.items():
                if n in personalization:
                    personalization[n] = max(0.0, w) / s
    else:
        seeds = [n for n in seed_nodes if n in personalization]
        if not seeds:
            return 0.0
        u = 1.0 / len(seeds)
        for n in seeds:
            personalization[n] = u

    # Run pagerank on reversed graph using normalized edge prob
    pr = nx.pagerank(
        Grev,
        alpha=alpha,
        personalization=personalization,
        weight="prob",
        max_iter=max_iter,
        tol=tol,
    )
    return float(pr.get(product_id, 0.0))


def absorbing_win_prob(
    G: nx.DiGraph,
    seed_nodes: Iterable[str],
    product_id: str,
    weight: str = "likelihood",
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> float:
    """
    True probability of eventual absorption at `product_id` starting from the seed set.

    - Reverses and normalizes the graph (row-stochastic by "prob").
    - Makes `product_id` absorbing (no outgoing edges; self-loop prob=1).
    - Solves h(v) = sum_{u} P(v->u) * h(u), with boundary h(product)=1
      via simple value iteration.

    Returns the average absorption probability starting uniformly over the given seeds.
    If you want non-uniform seeds, pass the seed multiple times or wrap this function.
    """
    Grev = _build_reversed_and_normalized(G, weight=weight)

    # Make product absorbing
    if product_id not in Grev:
        return 0.0
    for _, _, d in list(Grev.out_edges(product_id, data=True)):
        # remove all outgoing
        pass
    Grev.remove_edges_from(list(Grev.out_edges(product_id)))
    Grev.add_edge(product_id, product_id, prob=1.0)

    # Initialize h=0 except h(product)=1
    h: Dict[str, float] = {n: 0.0 for n in Grev.nodes()}
    h[product_id] = 1.0

    # Precache out-edges
    out_map = {n: list(Grev.out_edges(n, data=True)) for n in Grev.nodes()}

    # Value iteration
    for it in range(max_iter):
        delta = 0.0
        new_h = h.copy()
        for v in Grev.nodes():
            if v == product_id:
                continue
            s = 0.0
            for _, u, d in out_map[v]:
                p = d.get("prob", 0.0)
                s += p * h[u]
            delta = max(delta, abs(s - h[v]))
            new_h[v] = s
        h = new_h
        if delta < tol:
            break

    # Average over seeds that exist
    seeds = [s for s in seed_nodes if s in h]
    if not seeds:
        return 0.0
    return float(sum(h[s] for s in seeds) / len(seeds))
