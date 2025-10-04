# ============================
# File: backend/utils/inference/rcs_generators/ppr_engine.py
# ============================
from __future__ import annotations
from typing import Dict
import networkx as nx

def personalized_pagerank(
    G: nx.DiGraph,
    *,
    alpha: float,
    personalization: Dict[str, float],
    max_iter: int = 100,
    tol: float = 1e-08,
) -> Dict[str, float]:
    try:
        return nx.pagerank(G, alpha=alpha, personalization=personalization, max_iter=max_iter, tol=tol)
    except Exception:
        n = max(1, G.number_of_nodes())
        return {node: 1.0 / n for node in G.nodes}
