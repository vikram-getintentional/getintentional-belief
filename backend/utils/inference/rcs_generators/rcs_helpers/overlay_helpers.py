# ============================
# File: backend/utils/graph_base/overlay_helpers.py
# ============================
from __future__ import annotations
from typing import Dict, Tuple, Iterable, List
import networkx as nx

from backend.utils.graph_base.network_graph import get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
from backend.utils.inference.rcs_generators.rcs_computations.ppr_engine import PPREngine






def make_edge_scales_for_persona(
    G: nx.DiGraph, G_rev: nx.DiGraph, engine: PPREngine, persona_id: str, boost: float = 2.0
) -> Dict[Tuple[int, int], float]:
    """
    Returns overlay scales for edges around a persona, consistent with your previous
    _apply_persona_boost logic, but in the *reversed* orientation.
    In the reversed graph, an original edge (a -> b) becomes (b -> a).
    We will scale entries at (row=src_idx, col=dst_idx) in the reversed graph.
    """
    ov: Dict[Tuple[int, int], float] = {}
    if persona_id not in G:
        return ov


    # 1) job -> persona (orig) ==> persona -> job (reversed)
    for j in get_source_nodes_by_target_and_type(G, persona_id, "owned_by") + \
    get_source_nodes_by_target_and_type(G, persona_id, "performed_by"):
        if (persona_id in G_rev) and (j in G_rev) and G_rev.has_edge(persona_id, j):
            i = engine.node_index.get(persona_id)
            k = engine.node_index.get(j)
            if i is not None and k is not None:
                ov[(i, k)] = boost


        # 2a) pain -> job (orig felt_in) ==> job -> pain (reversed)
        for p in get_source_nodes_by_target_and_type(G, j, "felt_in"):
            if (j in G_rev) and (p in G_rev) and G_rev.has_edge(j, p):
                i = engine.node_index.get(j)
                k = engine.node_index.get(p)
                if i is not None and k is not None:
                    ov[(i, k)] = boost


        # 2b) job -> solve_pain (orig solves) ==> solve_pain -> job (reversed)
        for sp in get_target_nodes_by_source_and_type(G, j, "solves"):
            if (sp in G_rev) and (j in G_rev) and G_rev.has_edge(sp, j):
                i = engine.node_index.get(sp)
                k = engine.node_index.get(j)
                if i is not None and k is not None:
                    ov[(i, k)] = boost
    return ov


def make_edge_scales_for_concern(
    G: nx.DiGraph, G_rev: nx.DiGraph, engine: PPREngine, concern_id: str, conv_id: str, boost: float = 2.0
) -> Dict[Tuple[int, int], float]:
    """
    Concern boost: multiply out-edges of concern by boost, but only if those edges
    can reach conversion. Because we operate on the reversed graph, an original edge
    (concern -> x) becomes (x -> concern). That means boosting *incoming* edges to
    the concern in the reversed orientation.
    """
    ov: Dict[Tuple[int, int], float] = {}
    if (concern_id not in G) or (concern_id not in G_rev):
        return ov


    # Precompute which nodes in G can reach conversion.
    # In original orientation, we want successors that eventually reach conv.
    # We'll conservatively boost all orig out-edges from concern.
    for _, v in G.out_edges(concern_id):
    # reversed has edge v -> concern
        if (v in G_rev) and G_rev.has_edge(v, concern_id):
            i = engine.node_index.get(v)
            k = engine.node_index.get(concern_id)
            if i is not None and k is not None:
                ov[(i, k)] = boost

    return ov

def make_edge_scales_for_attribute(
    G: nx.DiGraph, G_rev: nx.DiGraph, engine: PPREngine, attribute_id: str, boost: float = 1.5
) -> Dict[Tuple[int, int], float]:
    """
    Boost edges from attribute -> pain_trigger in the reversed graph.
    """
    ov: Dict[Tuple[int, int], float] = {}
    if attribute_id not in G:
        return ov

    for _, trig in G.out_edges(attribute_id):
        if (trig in G_rev) and G_rev.has_edge(trig, attribute_id):
            i = engine.node_index.get(trig)
            k = engine.node_index.get(attribute_id)
            if i is not None and k is not None:
                ov[(i, k)] = boost
    return ov

