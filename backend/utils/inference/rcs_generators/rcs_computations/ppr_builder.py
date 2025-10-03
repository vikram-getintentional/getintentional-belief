# ============================
# File: backend/utils/graph_base/ppr_builder.py
# ============================
from __future__ import annotations
from typing import Dict, Tuple, List


import numpy as np
import scipy.sparse as sp
import networkx as nx

from backend.utils.inference.rcs_generators.rcs_computations.ppr_engine import PPREngine




def build_ppr_engine_from_graph(G_rev: nx.DiGraph) -> PPREngine:
    """
    Build a PPREngine from a *reversed* graph where edges carry 'likelihood' weights
    and rows should be normalized to be stochastic.


    G_rev: Directed graph in the orientation you will PageRank on (your code uses
    reversed graph for into-conversion flows).
    """
    nodes = list(G_rev.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    idx_node = nodes[:]


    rows, cols, data = [], [], []
    for u, v, d in G_rev.edges(data=True):
        w = float(d.get("likelihood", 0.0))
        if w <= 0:
            continue
        ui, vi = node_index[u], node_index[v]
        rows.append(ui)
        cols.append(vi)
        data.append(w)


    if not rows:
        # handle empty edge case: identity (self-loops) to keep stochastic
        n = len(nodes)
        P = sp.eye(n, format="csr")
    else:
        P = sp.csr_matrix((np.array(data, dtype=float), (np.array(rows, dtype=int), np.array(cols, dtype=int))), shape=(len(nodes), len(nodes)))
        # Row normalize
        rs = np.array(P.sum(axis=1)).ravel()
        rs[rs == 0] = 1.0
        inv = 1.0 / rs
        Dinv = sp.diags(inv)
        P = Dinv @ P


    return PPREngine(n=len(nodes), node_index=node_index, idx_node=idx_node, P_base=P)

