# --- graphwin.py ------------------------------------------------------------
from __future__ import annotations
from typing import Dict, List, Optional
import math
import networkx as nx
import numpy as np

from backend.utils.graph_base.network_graph import get_edge_attribute

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _noisy_or(vals):
    prod = 1.0
    for v in vals:
        prod *= (1.0 - _clip01(v))
    return 1.0 - prod

def _nt(G: nx.DiGraph, n: str) -> str:
    d = G.nodes.get(n, {})
    return d.get("node_type") or d.get("type") or "unknown"

def _collect_occurrences(engaged_nodes: List[dict]) -> Dict[str, float]:
    occ: Dict[str, float] = {}
    for it in engaged_nodes or []:
        nid = it.get("id")
        if nid:
            occ[nid] = float(it.get("occurrence", 1.0)) or 1.0
    return occ


def get_pain_trigger_likelihoods(
    G: nx.DiGraph,
    engaged_nodes: Optional[List[dict]] = None,
    *,
    default_b0: float = 0.0,
    default_attr_beta: float = 0.0,
    default_zmot_beta: float = 0.0,
    attr_edge_attr: str = "likelihood",   # attribute_value -> pain_trigger
    zmot_edge_attr: str = "boost",        # zmot_event     -> pain_trigger
    dim_budget_cap: float = 0.65,        # << total share that dimensions can own (rest left to b0 & noise)
    seed_cap: Optional[float] = None,    # << set e.g. 0.15 to normalize total seeds (optional)
    verbose: bool = False,
) -> Dict[str, float]:
    
    occ = _collect_occurrences(engaged_nodes or [])
    engaged_attrs = {n for n in occ if n in G and _nt(G, n) == "attribute_value"}
    engaged_zmots = {n for n in occ if n in G and _nt(G, n) == "zmot_event"}
    engaged_trigs = {n for n in occ if n in G and _nt(G, n) == "pain_trigger"}

    priors: Dict[str, float] = {}
    for t in (n for n in G if _nt(G, n) == "pain_trigger"):
        # hard trigger evidence => 1
        if t in engaged_trigs:
            priors[t] = 1.0
            if verbose: print(f"[priors] {t} hard=1.0")
            continue

        

        attr_terms = []
        for a in engaged_attrs:
            if not G.has_edge(t,a):
                continue
            w = get_edge_attribute(G, t, a, attr_edge_attr)
            if not w or w <= 0.0:
                w = default_attr_beta
            attr_terms.append(_clip01(w) * _clip01(occ.get(a, 1.0)))
        p_attr = 1 - np.prod([1 - w for w in attr_terms]) if attr_terms else 0.0


        zmot_terms = []
        for z in engaged_zmots:
            if not G.has_edge(t,z):
                continue
            w = get_edge_attribute(G, t, z, zmot_edge_attr)
            if not w or w <= 0.0:
                w = default_zmot_beta
            zmot_terms.append(_clip01(w) * _clip01(occ.get(z, 1.0)))
        p_zmot = 1 - np.prod([1 - w for w in zmot_terms]) if zmot_terms else 0.0
        b0 = _clip01(float(G.nodes.get(t, {}).get("base_prior", default_b0)))
        p_t = 1 - (1 - b0) * (1 - p_attr) * (1 - p_zmot)
        priors[t] = _clip01(p_t)

    return priors
