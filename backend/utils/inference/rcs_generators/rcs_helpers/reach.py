# --- reach.py ---------------------------------------------------------------
from __future__ import annotations
from typing import Dict, List, Optional
import networkx as nx

from backend.utils.graph_base.network_graph import get_edge_attribute

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def reverse_reach_to_product(
    G: nx.DiGraph,
    *,
    product_id: str,
    pain_triggers: List[str],
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-08,
    weight: str = None,     # set to "weight" if you have weighted edges
) -> Dict[str, float]:
    Grev = G.reverse(copy=False)
    out: Dict[str, float] = {}
    if product_id not in Grev:
        return {t: 0.0 for t in pain_triggers}
    for t in pain_triggers:
        if t not in Grev:
            out[t] = 0.0
            continue
        pers = {n: 0.0 for n in Grev.nodes}
        pers[t] = 1.0
        try:
            pr = nx.pagerank(Grev, alpha=alpha, personalization=pers,
                             max_iter=max_iter, tol=tol, weight=weight)
            out[t] = float(pr.get(product_id, 0.0))
        except Exception:
            out[t] = 0.0
    return out

def reverse_reach_from_nodes(
    G: nx.DiGraph,
    *,
    product_id: str,
    source_nodes: List[str],
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-08,
    weight: str = None,
) -> Dict[str, float]:
    Grev = G.reverse(copy=False)
    out: Dict[str, float] = {}
    if product_id not in Grev:
        return {s: 0.0 for s in source_nodes}
    for s in source_nodes or []:
        if s not in Grev:
            out[s] = 0.0
            continue
        pers = {n: 0.0 for n in Grev.nodes}
        pers[s] = 1.0
        try:
            pr = nx.pagerank(Grev, alpha=alpha, personalization=pers,
                             max_iter=max_iter, tol=tol, weight=weight)
            out[s] = float(pr.get(product_id, 0.0))
        except Exception:
            out[s] = 0.0
    return out


def reach_absorb_to_product(
    G: nx.DiGraph,
    *,
    product_id: str,
    pain_triggers: List[str],
    weight: Optional[str] = None,  # if you have weighted edges, pass the attribute name (e.g. "likelihood")
) -> Dict[str, float]:
    """
    Compute TRUE hitting probability R(node->product) on the reversed graph with product as absorbing.
    Solves (I - Q) h = r once for all transient nodes; h[i] is the probability that a random walk
    starting at node i will eventually hit product_id.

    Returns: {pain_trigger_id: R_t}
    """
    Grev = G.reverse(copy=False)
    if product_id not in Grev:
        return {t: 0.0 for t in pain_triggers}

    # Build a compact index
    nodes = list(Grev.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    import numpy as np
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except Exception:
        # Fallback (slow) Monte Carlo if SciPy isn't available
        return reach_absorb_to_product_mc(G, product_id=product_id, pain_triggers=pain_triggers, weight=weight)

    # Row-stochastic transition matrix on Grev; make product absorbing
    rows, cols, data = [], [], []
    prod_i = idx[product_id]

    for u in Grev.nodes:
        ui = idx[u]
        if u == product_id:
            # absorbing
            rows.append(ui); cols.append(ui); data.append(1.0)
            continue
        # outgoing from u in Grev
        nbrs = list(Grev.successors(u))
        if not nbrs:
            # dead-end: teleport to self (absorbing-like)
            rows.append(ui); cols.append(ui); data.append(1.0)
            continue
        if weight is None:
            w = np.ones(len(nbrs), dtype=float)
        else:
            w = np.array([float(Grev[u][v].get(weight, 0.0)) for v in nbrs], dtype=float)
            if np.all(w <= 0):
                w[:] = 1.0
        w = w / w.sum()
        for v, p in zip(nbrs, w):
            rows.append(ui); cols.append(idx[v]); data.append(float(p))

    P = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

    # Partition: transient T = nodes \ {product}, absorbing A = {product}
    T = [n for n in nodes if n != product_id]
    if not T:
        return {t: 1.0 if t == product_id else 0.0 for t in pain_triggers}

    Ti = [idx[n] for n in T]
    # Q = P[T,T], r = P[T, product]
    Q = P[Ti, :][:, Ti]
    r = P[Ti, prod_i].toarray().reshape(-1)

    # Solve (I - Q) h = r
    I = sp.eye(Q.shape[0], format="csr")
    try:
        h = spla.spsolve(I - Q, r)  # shape |T|
    except Exception:
        # robust fallback
        h = spla.lsmr(I - Q, r)[0]

    # Map results back
    hit = {product_id: 1.0}
    for n, val in zip(T, h):
        hit[n] = float(max(0.0, min(1.0, val)))

    return {t: hit.get(t, 0.0) for t in pain_triggers}


def reach_absorb_to_product_mc(
    G: nx.DiGraph,
    *,
    product_id: str,
    pain_triggers: List[str],
    weight: Optional[str] = None,
    walks_per_trigger: int = 2000,
    max_steps: int = 50,
) -> Dict[str, float]:
    """
    Monte-Carlo fallback for hitting probability on reversed graph (no teleport).
    """
    import random
    Grev = G.reverse(copy=False)
    if product_id not in Grev:
        return {t: 0.0 for t in pain_triggers}

    def step_from(u):
        nbrs = list(Grev.successors(u))
        if not nbrs:
            return u
        if weight is None:
            return random.choice(nbrs)
        ws = [float(Grev[u][v].get(weight, 0.0)) for v in nbrs]
        if sum(ws) <= 0:
            return random.choice(nbrs)
        r = random.random() * sum(ws)
        c = 0.0
        for v, w in zip(nbrs, ws):
            c += w
            if c >= r:
                return v
        return nbrs[-1]

    out = {}
    for t in pain_triggers:
        if t not in Grev:
            out[t] = 0.0
            continue
        hits = 0
        for _ in range(walks_per_trigger):
            u = t
            for _ in range(max_steps):
                if u == product_id:
                    hits += 1
                    break
                u = step_from(u)
        out[t] = hits / max(1, walks_per_trigger)
    return out

def gold_fixed_point_to_product(
    G: nx.DiGraph,
    *,
    product_id: str,
    seeds: Dict[str, float],             # initial gold at nodes (e.g., priors at triggers, 1.0 at hard nodes)
    edge_attr: str = "likelihood",       # edge success probability
    default_edge_p: float = 0.40,        # conservative default for untyped edges
    max_iter: int = 100,
    tol: float = 1e-6,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Fixed point on the reversed graph:
      gold[v] = max(seeds[v], 1 - Π_{u->v} (1 - gold[u] * p(u,v)))
    with gold[product]=1.0 seed if you want product to be absorbing.

    - Branching duplicates gold (we propagate to all successors).
    - Merging uses noisy-OR to avoid double counting.
    - Works on DAGs in one pass (topo); with cycles we iterate to convergence.
    """
    Grev = G.reverse(copy=False)
    if product_id not in Grev:
        return {n: 0.0 for n in G.nodes}

    # initialize
    gold_prev: Dict[str, float] = {n: 0.0 for n in Grev.nodes}
    seeds = {n: _clip01(p) for n, p in (seeds or {}).items()}
    # product can be treated as absorbing 1.0 or left to be computed; choose 1.0 for “ever reach product”
    if product_id not in seeds:
        seeds[product_id] = 0.0  # not forcing absorption; set to 1.0 if you want
    # iterate
    for _ in range(max_iter):
        delta = 0.0
        gold_new = gold_prev.copy()
        for v in Grev.nodes:
            # union from all predecessors (in reversed graph, preds are upstream)
            preds = list(Grev.predecessors(v))
            if preds:
                prod = 1.0
                for u in preds:
                    p_uv = get_edge_attribute(Grev, u, v, edge_attr)
                    prod *= (1.0 - _clip01(gold_prev.get(u, 0.0)) * p_uv)
                merged = 1.0 - prod
            else:
                merged = 0.0
            # respect seed at v
            new_v = max(_clip01(seeds.get(v, 0.0)), _clip01(merged))
            delta = max(delta, abs(new_v - gold_prev.get(v, 0.0)))
            gold_new[v] = new_v
        gold_prev = gold_new
        if delta < tol:
            break

    if debug:
        tops = sorted(((n, gold_prev[n]) for n in gold_prev), key=lambda x: x[1], reverse=True)[:8]
        print("[gold] top nodes:", tops)

    return gold_prev

def reach_prob_trigger_to_product(
    G: nx.DiGraph,
    *,
    product_id: str,
    trigger_id: str,
    edge_attr: str = "likelihood",
    default_edge_p: float = 0.40,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    seeds = {trigger_id: 1.0}
    gold = gold_fixed_point_to_product(
        G, product_id=product_id, seeds=seeds,
        edge_attr=edge_attr, default_edge_p=default_edge_p,
        max_iter=max_iter, tol=tol
    )
    return float(_clip01(gold.get(product_id, 0.0)))

def reach_prob_to_product_per_trigger(
    G: nx.DiGraph,
    *,
    product_id: str,
    pain_triggers: List[str],
    edge_attr: str = "likelihood",
    default_edge_p: float = 0.40,
) -> Dict[str, float]:
    out = {}
    for t in pain_triggers:
        out[t] = reach_prob_trigger_to_product(
            G, product_id=product_id, trigger_id=t,
            edge_attr=edge_attr, default_edge_p=default_edge_p
        )
    return out