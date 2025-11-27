# backend/utils/inference/belief_manager/learn/graph_recos.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple

from backend.utils.inference.belief_manager.learn.graph_learning import GraphStore, NodeKey, PersonaMeta, _beta_incr_edge


def apply_edge_updates(
    G: GraphStore,
    edge_updates_ranked: List[Dict[str, Any]],
    scale: float = 1.0,
    rationale: str = "diff-mapper attributed"
) -> None:
    """
    edge_updates_ranked: [{u, v, delta, rel, ...}]
    We interpret positive delta as alpha++ and negative as beta++.
    """
    for r in edge_updates_ranked:
        u_id, v_id = r["u"], r["v"]
        delta = float(r.get("delta", 0.0)) * scale
        if abs(delta) < 1e-9:
            continue
        src: NodeKey = (_infer_type(G, u_id), u_id)
        dst: NodeKey = (_infer_type(G, v_id), v_id)
        if delta > 0:
            _beta_incr_edge(G, src, dst, delta, rationale=rationale)
        else:
            # negative deltas: treat as “penalize” via sibling beta bumps; here we directly add beta on (src,dst)
            e = G.ensure_edge(src, dst)
            e.beta += abs(delta)
            e.rationale = rationale

def apply_node_updates(
    G: GraphStore,
    node_updates_ranked: List[Dict[str, Any]],
    field_name: str = "perceptibility"
) -> None:
    """
    node_updates_ranked: [{"node_id": pid, "field": "perceptibility", "delta": +0.07, ...}]
    """
    for r in node_updates_ranked:
        nid = r.get("node_id")
        if not nid:
            continue
        if r.get("field") != field_name:
            continue
        delta = float(r.get("delta", 0.0))
        meta = G.personas.setdefault(nid, PersonaMeta())
        meta.perceptibility_prior += delta
        meta.perceptibility_n += 1.0

def _infer_type(G, node_id: str) -> str:
    # lightweight classifier from your id prefixes
    # persona:xxxx, job:xxxx, pain:xxxx, capability:xxxx, product:xxxx
    return (node_id.split(":")[0] if ":" in node_id else "node").lower()
