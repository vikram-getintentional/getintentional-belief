# ============================
# File: backend/utils/inference/rcs_generators/ppr_builder.py
# ============================
from __future__ import annotations
from typing import Dict, List
import networkx as nx
from .ppr_engine import personalized_pagerank

def ppr_from_source(G: nx.DiGraph, source: str, *, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-8) -> Dict[str, float]:
    if source not in G:
        return {n: 0.0 for n in G.nodes}
    pers = {n: 0.0 for n in G.nodes}
    pers[source] = 1.0
    return personalized_pagerank(G, alpha=alpha, personalization=pers, max_iter=max_iter, tol=tol)

def ppr_from_sources(G: nx.DiGraph, sources: List[str], *, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-8) -> Dict[str, float]:
    if not sources:
        return {n: 0.0 for n in G.nodes}
    pers = {n: 0.0 for n in G.nodes}
    for s in sources:
        if s in G:
            pers[s] = pers.get(s, 0.0) + 1.0
    # normalize personalization
    ssum = sum(pers.values()) or 1.0
    for k in pers:
        pers[k] /= ssum
    return personalized_pagerank(G, alpha=alpha, personalization=pers, max_iter=max_iter, tol=tol)
