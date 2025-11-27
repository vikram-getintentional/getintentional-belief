# backend/utils/inference/belief_manager/learn/materialize.py
from __future__ import annotations
from typing import Any, Dict, Tuple
import math
import networkx as nx

from .graph_learning import (
    GraphStore, EdgeBayes, PersonaMeta, NodeKey
)

# ---------- helpers ----------

def _infer_type(node_id: str) -> str:
    # Accepts ids like "persona:xxxx", "job:xxxx", "pain:xxxx", "capability:xxxx", "product:xxxx"
    return (node_id.split(":", 1)[0] if isinstance(node_id, str) and ":" in node_id else "node").lower()

def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def _edge_prior_from_attrs(
    attrs: Dict[str, Any],
    *,
    weight_key: str,
    default_n0: float
) -> Tuple[float, float]:
    """
    Resolve (alpha, beta) for a Beta prior:
      1) If attrs already carry 'alpha' and 'beta', use them.
      2) Else derive from likelihood (or weight): mean=w, n0=pseudo-counts => a=w*n0, b=(1-w)*n0
    """
    a = attrs.get("alpha")
    b = attrs.get("beta")
    if a is not None and b is not None:
        try:
            a = float(a); b = float(b)
            # guard against degenerate (<=0) values
            if a <= 0 or b <= 0:
                raise ValueError
            return a, b
        except Exception:
            pass

    w = attrs.get(weight_key)
    if w is None:
        w = attrs.get("weight", 0.5)
    w = _clamp01(w)

    n0 = attrs.get("n0", default_n0)
    try:
        n0 = float(n0)
    except Exception:
        n0 = default_n0
    if n0 <= 0:
        n0 = default_n0

    # Avoid exact zeros which can freeze learning in some places
    alpha = max(1e-6, w * n0)
    beta  = max(1e-6, (1.0 - w) * n0)
    return alpha, beta

# ---------- main: NX → GraphStore ----------

def materialize_graphstore_from_networkx(
    G_nx: nx.DiGraph,
    *,
    weight_key: str = "likelihood",
    default_n0: float = 4.0,
    persona_label_key: str = "label",
    persona_perc_key: str = "perceptibility_prior",
    persona_perc_n_key: str = "perceptibility_n"
) -> GraphStore:
    """
    Build a GraphStore view over an NX product graph.
    - Edges:
        - Reads EdgeBayes (alpha/beta) from 'alpha'/'beta' if present, else from `weight_key` (or 'weight') with pseudo-counts.
        - Populates outgoing/incoming adjacency.
    - Personas:
        - Copies perceptibility priors and label (if present) into PersonaMeta.
    """
    GS = GraphStore()

    # Nodes: pre-scan to identify personas and copy persona meta
    for n, ndata in G_nx.nodes(data=True):
        n_type = _infer_type(n)
        if n_type == "persona":
            meta = GS.personas.setdefault(str(n), PersonaMeta())
            # label (optional)
            lbl = ndata.get(persona_label_key)
            if isinstance(lbl, str) and lbl.strip():
                meta.label = lbl
            # perceptibility (optional)
            perc = ndata.get(persona_perc_key, 0.0)
            try:
                meta.perceptibility_prior = float(perc)
            except Exception:
                meta.perceptibility_prior = 0.0
            # perceptibility_n (optional)
            perc_n = ndata.get(persona_perc_n_key, 1.0)
            try:
                meta.perceptibility_n = max(1.0, float(perc_n))
            except Exception:
                meta.perceptibility_n = 1.0

    # Edges: create Beta priors + adjacency
    for u, v, attrs in G_nx.edges(data=True):
        u_id, v_id = str(u), str(v)
        u_t, v_t = _infer_type(u_id), _infer_type(v_id)
        src: NodeKey = (u_t, u_id)
        dst: NodeKey = (v_t, v_id)

        # ensure adjacency + EdgeBayes record
        eb: EdgeBayes = GS.ensure_edge(src, dst)

        # initialize alpha/beta only if this is the first time we touch this edge
        # (GraphStore.defaultdict creates EdgeBayes(alpha=1,beta=1) by default)
        a, b = _edge_prior_from_attrs(attrs, weight_key=weight_key, default_n0=default_n0)

        # If caller wants to *merge* multiple parallel loads later, you can add instead of assign.
        # Here we assign once because each NX edge is unique.
        eb.alpha = float(a)
        eb.beta  = float(b)

        # (Optional) carry over any rationale / timestamps if you keep them on NX
        if "rationale" in attrs and isinstance(attrs["rationale"], str):
            eb.rationale = attrs["rationale"]

    return GS

# ---------- optional: GraphStore → NX sync (writeback of means) ----------

def sync_edge_means_to_networkx(
    GS: GraphStore,
    G_nx: nx.DiGraph,
    *,
    write_key: str = "likelihood",
    also_write_alpha_beta: bool = True
) -> None:
    """
    Push back current Beta means to NX edges:
      - sets edge[write_key] = alpha / (alpha + beta)
      - optionally persists alpha/beta
    """
    for (src, dst), eb in GS.edges.items():
        u_id = src[1]; v_id = dst[1]
        if G_nx.has_edge(u_id, v_id):
            mean = eb.mean if (eb.alpha + eb.beta) > 0 else 0.5
            G_nx[u_id][v_id][write_key] = float(mean)
            if also_write_alpha_beta:
                G_nx[u_id][v_id]["alpha"] = float(eb.alpha)
                G_nx[u_id][v_id]["beta"]  = float(eb.beta)
