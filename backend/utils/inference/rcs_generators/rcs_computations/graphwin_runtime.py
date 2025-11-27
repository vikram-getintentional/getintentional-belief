# backend/utils/inference/rcs_generators/rcs_computations/graphwin_runtime.py
# --- graphwin_runtime.py ----------------------------------------------------
from __future__ import annotations
from typing import Dict, List, Optional, Iterable, Tuple
import networkx as nx

# ----------------------------
# Small helpers
# ----------------------------
def _clip01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def _node_type(g: nx.DiGraph, nid: str) -> str:
    d = g.nodes.get(nid, {})
    return (d.get("node_type") or d.get("type") or "").strip().lower()

def _build_reversed_and_normalized(G: nx.DiGraph, weight: str = "likelihood") -> nx.DiGraph:
    Grev = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        w = float(d.get(weight, d.get("weight", 1.0)) or 0.0)
        if w < 0.0: w = 0.0
        Grev.add_edge(v, u, weight=w)
    # row-normalize -> 'prob'
    for n in list(Grev.nodes()):
        outs = list(Grev.out_edges(n, data=True))
        if not outs: 
            continue
        s = sum(e[2].get("weight", 0.0) for e in outs)
        if s <= 0.0:
            p = 1.0/len(outs)
            for _,_,d in outs: d["prob"] = p
        else:
            invs = 1.0/s
            for _,_,d in outs: d["prob"] = d.get("weight", 0.0)*invs
    return Grev

def _ppr(
    G: nx.DiGraph,
    personalization: Dict[str, float],
    *,
    alpha: float = 0.85,
    tol: float = 1e-12,
    max_iter: int = 200,
    weight_key: str = "prob",
) -> Dict[str, float]:
    if not personalization:
        return {n: 0.0 for n in G.nodes()}
    # normalize pers over graph nodes
    pers = {n: 0.0 for n in G.nodes()}
    s = 0.0
    for k, v in personalization.items():
        if k in pers and v > 0.0:
            s += v
    if s <= 0.0:
        return {n: 0.0 for n in G.nodes()}
    invs = 1.0/s
    for k, v in personalization.items():
        if k in pers and v > 0.0:
            pers[k] = v*invs

    return nx.pagerank(G, alpha=alpha, personalization=pers, weight=weight_key, tol=tol, max_iter=max_iter, dangling=pers)

# ----------------------------
# Priors (PJP only as base)
# ----------------------------
_PJP = {"pain","job","persona"}

def _uniform_PJP_prior(G: nx.DiGraph) -> Dict[str, float]:
    prod = next((n for n,d in G.nodes(data=True) if (_node_type(G,n)=="product")), None)
    support = [n for n in G.nodes() if n != prod and _node_type(G, n) in _PJP]
    if not support:
        # fallback: all except product
        support = [n for n in G.nodes() if n != prod]
    w = 1.0/len(support) if support else 0.0
    return {n: (w if n in support else 0.0) for n in G.nodes()}

def _norm_filter(d: Dict[str, float], valid: Iterable[str]) -> Dict[str, float]:
    vset = set(valid)
    x = {k: float(v) for k, v in d.items() if k in vset and float(v) > 0.0}
    s = sum(x.values())
    if s <= 0.0: return {}
    invs = 1.0/s
    for k in list(x.keys()): x[k] *= invs
    return x

def _blend(a: Dict[str,float], b: Dict[str,float], beta: float) -> Dict[str,float]:
    if not a and not b: return {}
    if not b or beta <= 0.0: return dict(a)
    if not a or beta >= 1.0: return dict(b)
    out = {}
    om = 1.0 - beta
    for k,v in a.items(): out[k] = out.get(k,0.0) + om*v
    for k,v in b.items(): out[k] = out.get(k,0.0) + beta*v
    return out

# ----------------------------
# Context (attributes / ZMOT) → PJP projector
# ----------------------------
def project_context_to_pjp(
    G: nx.DiGraph,
    *,
    attributes: Optional[Dict[str,float]] = None,
    zmots: Optional[Dict[str,float]] = None,
    attr_to_pain_weight: str = "likelihood",
    zmot_to_pain_weight: str = "likelihood",
) -> Dict[str, float]:
    """
    Convert non-org context (attributes, zmot_events) into a soft prior over PJP nodes.

    Default: graph-topology projection
      attribute_value  --(any edge)--> pain/job/persona
      zmot_event       --(any edge)--> pain/job/persona

    If your schema mainly connects attributes/ZMOT -> pain, that’s fine:
    the mass will then propagate to PJP via those edges.
    """
    scores: Dict[str,float] = {}
    def _acc(dst: str, w: float):
        if _node_type(G, dst) in _PJP:
            scores[dst] = scores.get(dst, 0.0) + max(0.0, float(w))

    # attributes → neighbors
    for a, aw in (attributes or {}).items():
        if a not in G: continue
        for _, v, d in G.out_edges(a, data=True):
            _acc(v, aw * float(d.get(attr_to_pain_weight, d.get("weight", 1.0)) or 0.0))

    # zmots → neighbors
    for z, zw in (zmots or {}).items():
        if z not in G: continue
        for _, v, d in G.out_edges(z, data=True):
            _acc(v, zw * float(d.get(zmot_to_pain_weight, d.get("weight", 1.0)) or 0.0))

    # normalize over graph nodes
    return _norm_filter(scores, G.nodes())

def _mix_pjp_prior(
    *,
    Gsupp: nx.DiGraph,
    observed_pjp: Optional[Dict[str,float]],
    account_prior: Optional[Dict[str,float]],
    context_prior: Optional[Dict[str,float]],
    beta_observed: float,
    beta_account: float,
    beta_context: float,
    epsilon_uniform: float,
) -> Dict[str,float]:
    """
    Final personalization π on the reversed chain:
      π0 (uniform over PJP)
      π_obs   (from engaged PJP)
      π_acc   (account tilt to PJP)
      π_ctx   (projection of attributes/ZMOT onto PJP)

      stage1 = blend(π0,   π_obs,   beta_observed)
      stage2 = blend(stage1, π_acc,  beta_account)
      stage3 = blend(stage2, π_ctx,  beta_context)
      stage4 = blend(stage3, π0,     epsilon_uniform)   # smoothing
    """
    valid = list(Gsupp.nodes())
    pi0   = _uniform_PJP_prior(Gsupp)
    p_obs = _norm_filter(observed_pjp or {}, valid)
    p_acc = _norm_filter(account_prior or {}, valid)
    p_ctx = _norm_filter(context_prior or {}, valid)

    stage1 = _blend(pi0,   p_obs, beta_observed)
    stage2 = _blend(stage1, p_acc, beta_account)
    stage3 = _blend(stage2, p_ctx, beta_context)
    if epsilon_uniform > 0.0:
        stage3 = _blend(stage3, pi0, epsilon_uniform)
    return stage3

# ----------------------------
# Public APIs
# ----------------------------
def graphwin_ppr_general(
    *,
    product_graph: nx.DiGraph,
    product_id: str,
    observed_pjp: Optional[Dict[str,float]] = None,   # pains/jobs/personas seen
    account_prior: Optional[Dict[str,float]] = None,  # soft hints to PJP nodes
    attributes: Optional[Dict[str,float]] = None,     # attribute_value nodes
    zmots: Optional[Dict[str,float]] = None,          # zmot_event / observable moments
    beta_observed: float = 0.35,
    beta_account: float  = 0.20,
    beta_context: float  = 0.25,   # <— NEW: how strongly context tilts the prior
    epsilon_uniform: float = 0.05,
    alpha: float = 0.85,
    weight_key: str = "likelihood",
) -> float:
    Grev = _build_reversed_and_normalized(product_graph, weight=weight_key)
    if product_id not in Grev: return 0.0

    # 1) Project context (attributes/ZMOT) → PJP soft prior
    ctx = project_context_to_pjp(product_graph, attributes=attributes, zmots=zmots)

    # 2) Mix into the PJP prior
    pers = _mix_pjp_prior(
        Gsupp=Grev,
        observed_pjp=observed_pjp,
        account_prior=account_prior,
        context_prior=ctx,
        beta_observed=beta_observed,
        beta_account=beta_account,
        beta_context=beta_context,
        epsilon_uniform=epsilon_uniform,
    )
    if not pers: return 0.0

    pr = _ppr(Grev, pers, alpha=alpha, weight_key="prob")
    return _clip01(float(pr.get(product_id, 0.0)))

def get_graphwin(
    G: nx.DiGraph,
    engaged_nodes: Optional[List[Dict]] = None,
    *,
    alpha: float = 0.85,
    weight_key: str = "likelihood",
    account_prior: Optional[Dict[str,float]] = None,
    beta_observed: float = 0.35,
    beta_account: float = 0.20,
    beta_context: float = 0.25,
    epsilon_uniform: float = 0.05,
    debug: bool = False,
) -> Dict:
    engaged_nodes = engaged_nodes or []
    product_ids = [n for n,d in G.nodes(data=True) if (_node_type(G,n)=="product")]
    product_id = product_ids[0] if product_ids else None
    if not product_id:
        return {"win_likelihood": 0.0, "note": "no product"}

    observed: Dict[str,float] = {}
    attrs: Dict[str,float] = {}
    zmots: Dict[str,float] = {}

    for e in engaged_nodes:
        nid = e.get("id","")
        if nid not in G: continue
        w = float(e.get("occurrence", 1.0)) or 1.0
        t = _node_type(G, nid)
        if t in _PJP:
            observed[nid] = observed.get(nid, 0.0) + w
        elif t == "attribute_value":
            attrs[nid] = attrs.get(nid, 0.0) + w
        elif t in {"zmot_event","observable_moment","keyword","pain_trigger"}:
            zmots[nid] = zmots.get(nid, 0.0) + w

    win = graphwin_ppr_general(
        product_graph=G,
        product_id=product_id,
        observed_pjp=observed or None,
        account_prior=account_prior or None,
        attributes=attrs or None,
        zmots=zmots or None,
        beta_observed=beta_observed,
        beta_account=beta_account,
        beta_context=beta_context,
        epsilon_uniform=epsilon_uniform,
        alpha=alpha,
        weight_key=weight_key,
    )

    out = {"win_likelihood": float(win)}
    if debug:
        out["debug"] = {
            "product_id": product_id,
            "observed_pjp_count": len(observed),
            "attributes_count": len(attrs),
            "zmots_count": len(zmots),
            "alpha": alpha,
            "beta_observed": beta_observed,
            "beta_account": beta_account,
            "beta_context": beta_context,
            "epsilon_uniform": epsilon_uniform,
        }
    return out
