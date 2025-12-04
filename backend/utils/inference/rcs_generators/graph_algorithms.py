# ============================
# File: backend/utils/inference/rcs_generators/graph_algorithms.py
# ============================
from __future__ import annotations

from collections import defaultdict
import logging
import math
import networkx as nx
import numpy as np
from typing import Dict, Iterable, List, Tuple, Any, Optional

from backend.utils.graph_base.network_graph import (
    _set_node_label,
    get_edge_attribute,
    get_node_by_id,
    get_nodes_list_ids,
    get_product_id_from_subgraph,
    get_source_nodes_by_target_and_type,
    get_target_nodes_by_source_and_type,
)
from backend.utils.inference.rcs_generators.rcs_computations.graphwin_runtime import _blend, _norm_filter, _uniform_PJP_prior, get_graphwin, project_context_to_pjp


LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------
# Phase bucketing from Perceptibility/Proximity
# ------------------------------------------------------------
def _phase_from_perc_prox(perc: float, prox: float) -> str:
    perc = float(perc or 0.0)
    prox = float(prox or 0.0)

    if perc >= 0.60 and prox < 0.40:
        return "zmot"
    if perc >= 0.40 and prox >= 0.30:
        return "discovery"
    if prox >= 0.60 and perc < 0.40:
        return "barriers"
    if prox >= 0.75:
        return "implementation"
    return "discovery"


# ============================================================
# Helpers
# ============================================================
def _node_type(G: nx.DiGraph, nid: str) -> str:
    d = G.nodes.get(nid, {})
    return (d.get("node_type") or d.get("type") or "").strip().lower()


def _build_reversed(G: nx.DiGraph) -> nx.DiGraph:
    return G.reverse(copy=True)


def _ensure_edge_weights(G: nx.DiGraph, weight_key: str = "likelihood") -> None:
    """
    Ensure every edge has a numeric 'weight' derived from weight_key (or safe tiny default).
    Operates in-place on G.
    """
    for _, _, d in G.edges(data=True):
        w = d.get(weight_key, d.get("weight", 1.0))
        try:
            d["weight"] = float(w)
        except Exception:
            d["weight"] = 1e-9
        if d["weight"] <= 0:
            d["weight"] = 1e-9


def _normalize_01(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    lo = min(d.values())
    hi = max(d.values())
    if hi <= lo:
        return {k: 0.0 for k in d}
    span = hi - lo
    return {k: (v - lo) / span for k, v in d.items()}


# ============================================================
# Personalized PageRank
# ============================================================
def _ppr(G: nx.DiGraph, seeds: Dict[str, float], alpha: float = 0.85, weight_key: str = "weight") -> Dict[str, float]:
    """
    PPR honoring dangling redistribution to personalization.
    """
    if not seeds:
        return {n: 0.0 for n in G.nodes()}

    # normalize personalization over G's nodes
    pers = {n: 0.0 for n in G.nodes()}
    s = 0.0
    for k, v in seeds.items():
        if k in pers and v > 0:
            s += v
    if s <= 0.0:
        return {n: 0.0 for n in G.nodes()}
    invs = 1.0 / s
    for k, v in seeds.items():
        if k in pers and v > 0:
            pers[k] = v * invs

    return nx.pagerank(
        G,
        alpha=alpha,
        personalization=pers,
        weight=weight_key,
        dangling=pers,   # IMPORTANT
    )


# ============================================================
# Stage inference helpers
# ============================================================
_STAGE_ORDER = {"problem": 0, "pain": 1, "solution": 2}
def _stage_rank(stage: str) -> int:
    return _STAGE_ORDER.get((stage or "").lower(), 1)


def infer_stage_for_concern(G: nx.DiGraph, node_id: str) -> str:
    """
    Infer a user-friendly 'stage' label for a node in the concern flow.
    """
    if node_id not in G:
        return "problem"

    # 1) explicit override
    stage = (G.nodes[node_id].get("stage") or "").strip().lower()
    if stage in _STAGE_ORDER:
        return stage

    # 2) type-based default
    t = _node_type(G, node_id)
    if t in {"attribute", "attribute_value", "zmot_event", "observable_moment", "keyword", "pain_trigger"}:
        return "problem"
    if t in {"pain"}:
        return "pain"
    if t in {"job", "capability", "product"}:
        return "solution"

    # 3) neighborhood heuristic
    for u, _ in G.in_edges(node_id):
        nt = _node_type(G, u)
        if nt == "pain":
            return "pain"
        if nt in {"job", "capability"}:
            return "solution"
    for _, v in G.out_edges(node_id):
        nt = _node_type(G, v)
        if nt == "pain":
            return "pain"
        if nt in {"job", "capability"}:
            return "solution"
    return "problem"


# ============================================================
# Odds/logit helpers
# ============================================================
def _odds(p: float, eps: float = 1e-12) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return p / (1.0 - p)


def _logit(p: float, eps: float = 1e-12) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def activation_from(p_with: float, logit_base: float, baseline: float) -> float:
    # linearized lift around baseline via sigmoid derivative
    dl = _logit(p_with) - logit_base
    slope = baseline * (1.0 - baseline)
    return max(0.0, slope * dl)


# ============================================================
# Core node scoring (Perceptibility/Proximity/Involvement + Activations)
# ============================================================

def compute_node_strengths(
    product_graph: nx.DiGraph,
    product_id: str,
    engaged_attributes: Dict[str, float],
    *,
    engaged_zmots: Optional[Dict[str, float]] = None,
    engaged_other: Optional[Dict[str, float]] = None,
    alpha_forward: float = 0.85,          # forward walk: product seed
    alpha_backward: float = 0.65,         # backward walk: observed seeds on reversed
    weight_key: str = "likelihood",
    score_kinds: Tuple[str, ...] = ("pain", "job", "persona"),
    exclude_from_involvement: Tuple[str, ...] = (),
    attr_prior: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Returns per-node scores:
      - activation(n): odds/prob lifts via GraphWin delta when n is added to engaged
      - involvement(n): share of seed→product edge-flow mass collected at n
      - strength(n): involvement(n) * delta_p(n)
      - perceptibility(n): PPR(n | engaged_any on reversed)   (Perc)   0..1 normalized
      - proximity(n):      PPR(n | product on forward)        (Prox)   0..1 normalized
    """
    LOGGER.debug("Computing node strengths for product %s", product_id)
    R = product_graph
    _ensure_edge_weights(R, weight_key=weight_key)
    F = _build_reversed(R)

    # ---- Proximity: forward from product on original orientation (R)
    forward_raw_R = _ppr(R, {product_id: 1.0}, alpha=alpha_forward, weight_key="weight")
    LOGGER.debug("Computed forward PPR for proximity.")
    # ---- Perceptibility: from ANY observed (attributes + zmots + other) on reversed orientation (F)
    engaged_attributes = engaged_attributes or {}
    engaged_zmots = engaged_zmots or {}
    engaged_other = engaged_other or {}

    engaged_any = {}
    engaged_any.update({k: float(v) for k, v in engaged_attributes.items() if k in R})
    engaged_any.update({k: float(v) for k, v in engaged_zmots.items() if k in R})
    engaged_any.update({k: float(v) for k, v in engaged_other.items() if k in R})

    backward_raw_F = _ppr(F, engaged_any, alpha=alpha_backward, weight_key="weight") if engaged_any else {n: 0.0 for n in F.nodes}
    LOGGER.debug("Computed backward PPR for perceptibility.")
    # ---- Normalize Perc/Prox to 0..1 over node set
    # Build observed PJP from engaged_other (already only PJP if you kept your last split)
    observed_pjp = {}
    for nid, w in (engaged_other or {}).items():
        if _node_type(R, nid) in {"pain","job","persona"}:
            observed_pjp[nid] = observed_pjp.get(nid, 0.0) + float(w)

    # Optional: if you captured attributes / zmots earlier, pass them; if not, keep empty:
    ctx_prior = project_context_to_pjp(R, attributes=engaged_attributes or {}, zmots=engaged_zmots or {})

    pi0 = _uniform_PJP_prior(F)
    p_obs = _norm_filter(observed_pjp, F.nodes())
    p_ctx = _norm_filter(ctx_prior, F.nodes())

    # small tilts so perceptibility reflects org & context—tune these if needed
    piF = _blend(pi0, p_obs, beta=0.25)
    piF = _blend(piF, p_ctx, beta=0.20)

    backward_raw_F = _ppr(F, piF, alpha=alpha_backward, weight_key="weight")
    perc_norm = _normalize_01(backward_raw_F)
    prox_norm = _normalize_01(forward_raw_R)   # proximity
    LOGGER.debug("Normalized perceptibility and proximity scores. Sample Perc: %s Sample Prox: %s", list(perc_norm.items())[:5], list(prox_norm.items())[:5])
    # ---- Edge-flow based involvement (forward × weight × backward on edge head)
    edge_contrib = {}
    total_flow = 0.0
    for u, v, d in R.edges(data=True):
        fu = float(forward_raw_R.get(u, 0.0))
        w  = float(d.get("weight", 1.0))
        bv = float(backward_raw_F.get(v, 0.0))
        c = fu * w * bv
        if c > 0:
            edge_contrib[(u, v)] = c
            total_flow += c

    inv_raw = defaultdict(float)
    if total_flow > 0.0:
        invs = 1.0 / total_flow
        for (u, v), c in edge_contrib.items():
            normc = c * invs
            inv_raw[u] += normc
            inv_raw[v] += normc
    else:
        inv_raw = {n: 0.0 for n in R.nodes}
    LOGGER.debug("Crossed random edge loop")
    # ---- Activation via GraphWin deltas
    # Build a full observed engaged list for the baseline call
    engaged_list_full = []
    for nid, w in engaged_attributes.items():
        engaged_list_full.append({"id": nid, "occurrence": float(w)})
    for nid, w in engaged_zmots.items():
        engaged_list_full.append({"id": nid, "occurrence": float(w)})
    for nid, w in engaged_other.items():
        engaged_list_full.append({"id": nid, "occurrence": float(w)})

    # Baseline
    base_out = get_graphwin(R, engaged_list_full, account_prior=attr_prior)["win_likelihood"]  # NOTE: pass account_prior
    p0 = float(base_out)
    o0 = _odds(p0)
    l0 = _logit(p0)

    activation_map = {}
    for n, d in R.nodes(data=True):
        ntype = d.get("node_type", "")
        if ntype not in score_kinds:
            continue

        # Build a *tilted* engaged list for the +n case.
        engaged_plus_list = list(engaged_list_full)

        if ntype in {"pain","job","persona"}:
            # PJP → add directly
            engaged_plus_list.append({"id": n, "occurrence": 1.0})
        else:
            # Non-PJP (e.g., capability): project to PJP neighbors as seeds
            # Use the same projector we use for attributes/ZMOT
            proj = project_context_to_pjp(
                R,
                attributes=None,
                zmots=None,
                # we piggyback the function by temporarily treating `n` like a context node:
            )
            # Manually project: take out-neighbors that are PJP and add them
            added = False
            for _, v, ed in R.out_edges(n, data=True):
                if _node_type(R, v) in {"pain","job","persona"}:
                    wv = float(ed.get("likelihood", ed.get("weight", 1.0)) or 0.0)
                    if wv > 0.0:
                        engaged_plus_list.append({"id": v, "occurrence": wv})
                        added = True
            if not added:
                # if no PJP neighbors, no lift is expected
                pass

        pw = float(get_graphwin(R, engaged_plus_list, account_prior=attr_prior)["win_likelihood"])
        ow = _odds(pw)
        dl = _logit(pw) - l0

        activation_map[n] = {
            "p_with": pw,
            "activation": max(0.0, ow - o0),        # odds lift
            "p_base": p0,
            "delta_p": max(0.0, pw - p0),           # probability lift
            "delta_logit": max(0.0, dl),            # logit lift
            "activation_linearized": activation_from(pw, l0, p0),
        }

    # ---- Assemble node-wise results (now including perc & prox)
    exclude_set = set(exclude_from_involvement)
    results = {}
    for n, d in R.nodes(data=True):
        if d.get("node_type") not in score_kinds:
            continue
        involvement = float(inv_raw.get(n, 0.0))
        if d.get("node_type") in exclude_set:
            involvement = 0.0

        am = activation_map.get(n, {})
        delta_p = float(am.get("delta_p", 0.0))
        results[n] = {
            "id": n,
            "label": _set_node_label(R, n),
            "activation": float(am.get("activation", 0.0)),
            "delta_p": delta_p,
            "involvement": involvement,
            "strength": involvement * delta_p,
            "activation_map": am,
            "perceptibility": float(perc_norm.get(n, 0.0)),
            "proximity": float(prox_norm.get(n, 0.0)),
        }

    return results


# ============================================================
# Persona aggregation (strength/activation at persona level)
# ============================================================
def noisy_or(terms: List[float]) -> float:
    p = 1.0
    for t in terms:
        t = max(0.0, min(1.0, float(t)))
        p *= (1.0 - t)
    return 1.0 - p


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_persona_scores_from_core(
    *,
    product_graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    core_scores: Dict[str, Dict[str, float]],
    mode: str = "prob",
    odds_relevance_scale: float = 1.0,
) -> Dict[str, Any]:
    def _extract_p_base(core_scores: Dict[str, Dict[str, float]]) -> float:
        for v in core_scores.values():
            am = v.get("activation_map")
            if isinstance(am, dict) and "p_base" in am:
                return float(am["p_base"])
        return 0.0

    def _perc_prox(nid: str) -> Tuple[float, float]:
        sc = core_scores.get(nid, {})
        return float(sc.get("perceptibility", 0.0)), float(sc.get("proximity", 0.0))

    p_base = _extract_p_base(core_scores)
    l_base = _logit(p_base)

    all_pruned_jobs = get_nodes_list_ids(product_graph, "job", {})
    persona_acc: Dict[str, Dict[str, Any]] = {}
    if not all_pruned_jobs:
        return {"persona_scores": {}, "persona_activation_breakdown": {}}

    def _acc_level(acc_lvl: Dict[str, float], perc: float, prox: float, w: float) -> None:
        if w <= 0.0:
            return
        acc_lvl["perc_sum"] += perc * w
        acc_lvl["prox_sum"] += prox * w
        acc_lvl["w_sum"] += w

    for job in all_pruned_jobs:
        js = core_scores.get(job, {"involvement": 0.0, "activation_map": {}})
        i = float(js.get("involvement", 0.0))
        jam = js.get("activation_map") or {}
        delta_p = float(jam.get("delta_p", 0.0))
        dlogit = float(jam.get("delta_logit", 0.0))

        if (mode == "odds" and dlogit == 0.0 and i == 0.0) or (mode == "prob" and delta_p == 0.0 and i == 0.0):
            continue

        personas_for_job = get_target_nodes_by_source_and_type(original_graph, job, "performed_by")
        if not personas_for_job:
            continue

        job_perc, job_prox = _perc_prox(job)

        for pid in personas_for_job:
            pj_rel = float(get_edge_attribute(original_graph, job, pid, "relevance"))
            acc = persona_acc.setdefault(
                pid,
                {
                    "act_terms": [], "inv_terms": [], "contrib": [], "sum_dlogit": 0.0, "act_terms_prob": [],
                    "perc_sum": 0.0, "prox_sum": 0.0, "w_sum": 0.0,
                    "levels": {
                        "execution":  {"perc_sum": 0.0, "prox_sum": 0.0, "w_sum": 0.0},
                        "problem":    {"perc_sum": 0.0, "prox_sum": 0.0, "w_sum": 0.0},
                        "pain":       {"perc_sum": 0.0, "prox_sum": 0.0, "w_sum": 0.0},
                        "resolution": {"perc_sum": 0.0, "prox_sum": 0.0, "w_sum": 0.0},
                    },
                }
            )

            acc["inv_terms"].append(i * pj_rel)
            if mode == "odds":
                acc["sum_dlogit"] += odds_relevance_scale * (pj_rel * dlogit)
            else:
                acc["act_terms_prob"].append(delta_p * pj_rel)

            # Execution (job)
            w_job = pj_rel * i
            if w_job > 0.0:
                acc["perc_sum"] += job_perc * w_job
                acc["prox_sum"] += job_prox * w_job
                acc["w_sum"] += w_job
                _acc_level(acc["levels"]["execution"], job_perc, job_prox, w_job)

            acc["contrib"].append({
                "job": job, "concern_level": "execution", "edge_weight": pj_rel,
                "job_activation_prob_lift": delta_p, "job_activation_logit_lift": dlogit,
                "job_involvement": i, "job_strength_prob": i * delta_p, "concern_id": job,
            })

            # Problem (pains solved by job)
            solved_pains_of_job = get_target_nodes_by_source_and_type(product_graph, job, "solves")
            for p in solved_pains_of_job:
                job_to_pain = float(get_edge_attribute(product_graph, job, p, "relevance"))
                ps = core_scores.get(p, {"involvement": 0.0, "activation_map": {}})
                pam = ps.get("activation_map") or {}
                delta_pp = float(pam.get("delta_p", 0.0))
                dlogit_pp = float(pam.get("delta_logit", 0.0))
                p_perc, p_prox = _perc_prox(p)
                p_inv = float(ps.get("involvement", 0.0))

                if mode == "odds":
                    acc["sum_dlogit"] += odds_relevance_scale * (pj_rel * job_to_pain * dlogit_pp)
                else:
                    acc["act_terms_prob"].append(delta_pp * pj_rel * job_to_pain)

                w_prob = pj_rel * job_to_pain * p_inv
                if w_prob > 0.0:
                    acc["perc_sum"] += p_perc * w_prob
                    acc["prox_sum"] += p_prox * w_prob
                    acc["w_sum"] += w_prob
                    _acc_level(acc["levels"]["problem"], p_perc, p_prox, w_prob)

                acc["contrib"].append({
                    "job": job, "concern_level": "problem", "edge_weight": pj_rel * job_to_pain,
                    "job_activation_prob_lift": delta_pp, "job_activation_logit_lift": dlogit_pp,
                    "job_involvement": p_inv, "job_strength_prob": p_inv * delta_pp, "concern_id": p,
                })

            # Pain (pains felt in job)
            felt_pains_of_job = get_source_nodes_by_target_and_type(product_graph, job, "felt_in")
            for pa in felt_pains_of_job:
                job_to_pain = float(get_edge_attribute(product_graph, pa, job, "likelihood"))
                pas = core_scores.get(pa, {"involvement": 0.0, "activation_map": {}})
                paam = pas.get("activation_map") or {}
                delta_pa = float(paam.get("delta_p", 0.0))
                dlogit_pa = float(paam.get("delta_logit", 0.0))
                pa_perc, pa_prox = _perc_prox(pa)
                pa_inv = float(pas.get("involvement", 0.0))

                if mode == "odds":
                    acc["sum_dlogit"] += odds_relevance_scale * (pj_rel * job_to_pain * dlogit_pa)
                else:
                    acc["act_terms_prob"].append(delta_pa * pj_rel * job_to_pain)

                w_pain = pj_rel * job_to_pain * pa_inv
                if w_pain > 0.0:
                    acc["perc_sum"] += pa_perc * w_pain
                    acc["prox_sum"] += pa_prox * w_pain
                    acc["w_sum"] += w_pain
                    _acc_level(acc["levels"]["pain"], pa_perc, pa_prox, w_pain)

                acc["contrib"].append({
                    "job": job, "concern_level": "pain", "edge_weight": pj_rel * job_to_pain,
                    "job_activation_prob_lift": delta_pa, "job_activation_logit_lift": dlogit_pa,
                    "job_involvement": pa_inv, "job_strength_prob": pa_inv * delta_pa, "concern_id": pa,
                })

                # Resolution (other resolvers)
                solving_nodes_of_pain = get_source_nodes_by_target_and_type(product_graph, pa, "solves")
                for rs in solving_nodes_of_pain:
                    if rs == job:
                        continue
                    job_relevance = float(get_edge_attribute(product_graph, rs, pa, "relevance"))
                    rs_s = core_scores.get(rs, {"involvement": 0.0, "activation_map": {}})
                    rsam = rs_s.get("activation_map") or {}
                    delta_rs = float(rsam.get("delta_p", 0.0))
                    dlogit_rs = float(rsam.get("delta_logit", 0.0))
                    rs_perc, rs_prox = _perc_prox(rs)
                    rs_inv = float(rs_s.get("involvement", 0.0))

                    if mode == "odds":
                        acc["sum_dlogit"] += odds_relevance_scale * (pj_rel * job_relevance * dlogit_rs)
                    else:
                        acc["act_terms_prob"].append(delta_rs * pj_rel * job_relevance)

                    w_res = pj_rel * job_relevance * rs_inv
                    if w_res > 0.0:
                        acc["perc_sum"] += rs_perc * w_res
                        acc["prox_sum"] += rs_prox * w_res
                        acc["w_sum"] += w_res
                        _acc_level(acc["levels"]["resolution"], rs_perc, rs_prox, w_res)

                    acc["contrib"].append({
                        "job": job, "concern_level": "resolution", "edge_weight": pj_rel * job_relevance,
                        "job_activation_prob_lift": delta_rs, "job_activation_logit_lift": dlogit_rs,
                        "job_involvement": rs_inv, "job_strength_prob": rs_inv * delta_rs, "concern_id": rs,
                    })

    # -------- Assemble outputs --------
    persona_scores_out: Dict[str, Dict[str, Any]] = {}
    activation_breakdown_list: List[Dict[str, Any]] = []

    for pid, acc in persona_acc.items():
        I = noisy_or(acc["inv_terms"]) if acc["inv_terms"] else 0.0
        if mode == "odds":
            l_persona = _logit(p_base) + acc["sum_dlogit"]
            p_persona = 1.0 / (1.0 + math.exp(-l_persona))
            A = max(0.0, p_persona - p_base)
        else:
            A = noisy_or(acc["act_terms_prob"]) if acc["act_terms_prob"] else 0.0
            p_persona = min(1.0, p_base + A)

        S = A * I
        label = _set_node_label(original_graph, pid)

        # persona-level perc/prox (aggregate)
        if acc["w_sum"] > 0.0:
            perc_persona = acc["perc_sum"] / acc["w_sum"]
            prox_persona = acc["prox_sum"] / acc["w_sum"]
        else:
            perc_persona, prox_persona = _perc_prox(pid)

        persona_scores_out[pid] = {
            "activation": float(A),
            "involvement": float(I),
            "strength": float(S),
            "label": label,
            "p_base": float(p_base),
            "p_persona": float(p_persona),
            "mode": mode,
            "perceptibility": float(perc_persona),
            "proximity": float(prox_persona),
        }

        # choose top contributor per level
        top_per_level = {"execution": None, "problem": None, "pain": None, "resolution": None}
        for row in acc["contrib"]:
            lvl = row.get("concern_level")
            if lvl not in top_per_level:
                continue
            cur = top_per_level[lvl]
            if (cur is None) or (row.get("job_strength_prob", 0.0) > (cur.get("job_strength_prob", 0.0))):
                top_per_level[lvl] = row

        def _mk_activation_entry(best, lvl_key: str):
            lvl_acc = acc["levels"][lvl_key]
            wsum = lvl_acc["w_sum"] or 0.0
            lvl_perc = (lvl_acc["perc_sum"] / wsum) if wsum > 0 else 0.0
            lvl_prox = (lvl_acc["prox_sum"] / wsum) if wsum > 0 else 0.0
            if not best:
                return {"activation": 0.0, "strength": 0.0, "prob_lift": 0.0, "logit_lift": 0.0,
                        "concern_id": None, "perceptibility": float(lvl_perc), "proximity": float(lvl_prox)}
            return {
                "activation": float(best.get("job_activation_prob_lift", 0.0) or 0.0),
                "strength": float(best.get("job_strength_prob", 0.0) or 0.0),
                "prob_lift": float(best.get("job_activation_prob_lift", 0.0) or 0.0),
                "logit_lift": float(best.get("job_activation_logit_lift", 0.0) or 0.0),
                "concern_id": best.get("concern_id"),
                "perceptibility": float(lvl_perc),
                "proximity": float(lvl_prox),
            }

        persona_node = get_node_by_id(original_graph, pid)
        activation_breakdown_list.append({
            "persona_id": pid,
            "label": label,
            "department": persona_node.get("department"),
            "seniority": persona_node.get("seniority"),
            "title": persona_node.get("title"),
            "involvement": float(I),
            "execution_activation":  _mk_activation_entry(top_per_level["execution"],  "execution"),
            "problem_activation":    _mk_activation_entry(top_per_level["problem"],    "problem"),
            "pain_activation":       _mk_activation_entry(top_per_level["pain"],       "pain"),
            "resolution_activation": _mk_activation_entry(top_per_level["resolution"], "resolution"),
        })
    

    return {
        "persona_scores": persona_scores_out,
        "persona_activation_breakdown": activation_breakdown_list,
    }


# ============================================================
# Final Combined Report (no early bail on attributes)
# ============================================================

def get_involvement_activation_report(
    G: nx.DiGraph,
    Original_G: nx.DiGraph,
    engaged_nodes: Optional[List[Dict]] = None,
    *,
    alpha: float = 0.85,
    weight_key: str = "likelihood",
    attr_prior: Optional[Dict[str, float]] = None, 
) -> Dict[str, Dict[str, Any]]:
    """
    Compute involvement & activation scores for both core and persona nodes.
    Works with or without attribute seeds. Observed nodes (personas/jobs/pains/capabilities/ZMOTs)
    will seed perceptibility and GraphWin baselines.
    """
    LOGGER.debug("Starting involvement & activation report computation...")
    engaged_nodes = engaged_nodes or []

    # Identify product node
    product_id = get_product_id_from_subgraph(Original_G)
    if not product_id:
        LOGGER.debug("No product node found in original graph; aborting report.")
        return {"core_scores": {}, "persona_scores": {}}
    product_node = get_node_by_id(Original_G, product_id)
    if not product_node:
        return {"core_scores": {}, "persona_scores": {}}
    
    # Split engagements by type
    attr_weights: Dict[str, float] = {}
    zmot_weights: Dict[str, float] = {}
    other_weights: Dict[str, float] = {}
    for e in engaged_nodes:
        nid = e.get("id", "")
        if nid not in G:
            continue
        w = float(e.get("occurrence", 1.0)) or 1.0
        t = _node_type(G, nid)
        if t == "attribute_value":
            attr_weights[nid] = attr_weights.get(nid, 0.0) + w
        elif t in {"zmot_event", "observable_moment", "keyword"}:
            zmot_weights[nid] = zmot_weights.get(nid, 0.0) + w
        else:
            other_weights[nid] = other_weights.get(nid, 0.0) + w
    
    # Core node scores (includes perc/prox/involvement/activation maps)

    core_scores = compute_node_strengths(
        product_graph=G,
        product_id=product_id,
        engaged_attributes=attr_weights,
        engaged_zmots=zmot_weights,
        engaged_other=other_weights,
        attr_prior=attr_prior,
    )
    
    # Persona aggregation
    persona_scores_ret = compute_persona_scores_from_core(
        product_graph=G,
        original_graph=Original_G,
        core_scores=core_scores,
    )
    

    def _extract_p_base_from_core(cs: Dict[str, Dict[str, Any]]) -> float:
        for v in cs.values():
            am = v.get("activation_map")
            if isinstance(am, dict) and "p_base" in am:
                return float(am["p_base"])
        return 0.0

    p_base = _extract_p_base_from_core(core_scores)
    G.graph["product_id"] = product_id
    # record all observed inputs (not just attributes)
    G.graph["engaged_attributes"] = dict(attr_weights)
    G.graph["engaged_zmots"] = dict(zmot_weights)
    G.graph["engaged_other"] = dict(other_weights)
    G.graph["p_base"] = p_base

    # --- annotate core nodes ---
    for nid, sc in core_scores.items():
        nd = G.nodes[nid]
        nd["perceptibility"] = float(sc.get("perceptibility", 0.0))
        nd["proximity"]      = float(sc.get("proximity", 0.0))
        nd["involvement"]    = float(sc.get("involvement", 0.0))
        nd["strength"]       = float(sc.get("strength", 0.0))
        nd["delta_p"]        = float(sc.get("activation_map", {}).get("delta_p", 0.0))

    # --- annotate personas (these may live only in Original_G, so guard existence) ---
    for pid, ps in persona_scores_ret["persona_scores"].items():
        if pid not in G:
            continue
        nd = G.nodes[pid]
        nd["persona_activation"]    = float(ps.get("activation", 0.0))
        nd["persona_involvement"]   = float(ps.get("involvement", 0.0))
        nd["persona_strength"]      = float(ps.get("strength", 0.0))
        nd["persona_perceptibility"]= float(ps.get("perceptibility", 0.0))
        nd["persona_proximity"]     = float(ps.get("proximity", 0.0))
    LOGGER.debug("Annotated graph nodes with core and persona scores.")
    LOGGER.debug("Returning %d core scores and %d persona scores.", len(core_scores), len(persona_scores_ret['persona_scores']))
    output = {
        "graph": G,  # the working graph annotated for this run
        "core_scores": core_scores,
        "persona_scores": persona_scores_ret["persona_scores"],
        "activation_breakdown": persona_scores_ret["persona_activation_breakdown"],
    }
    LOGGER.debug("Composed output report.")
    return output


# ============================================================
# Concern Backlog Construction (Derived from Persona Scores )
# ============================================================
def _keyness_from(pp: float, pr: float, inv: float, a=(1.2, 1.0, 1.0)) -> float:
    a1, a2, a3 = a
    x = a1 * pr + a2 * pp + a3 * inv
    return 1.0 / (1.0 + math.exp(-x))  # sigmoid


def build_concern_backlog_from_activation_breakdown(
    G: nx.DiGraph,
    activation_breakdown,
    stage_weights=None,
    top_k_per_persona: int = 5,
):
    if stage_weights is None:
        stage_weights = {"problem": 1.0, "pain": 0.9, "resolution": 0.85, "execution": 0.75}

    concern_backlog = []
    concerns_by_persona = defaultdict(list)

    for p in activation_breakdown:
        pid = p["persona_id"]
        I = float(p.get("involvement", 0.0))
        persona_label = _set_node_label(G, pid)

        for stage in ["execution", "problem", "pain", "resolution"]:
            a = p.get(f"{stage}_activation", {}) or {}
            cid = a.get("concern_id")
            act = float(a.get("activation", 0.0))
            if not cid or act <= 0.0:
                continue

            pp = float(a.get("perceptibility", 0.0))
            pr = float(a.get("proximity", 0.0))
            inv_c = float(G.nodes.get(cid, {}).get("involvement", 0.0))
            kn = _keyness_from(pp, pr, inv_c)

            lift_proxy = act * I * float(stage_weights.get(stage, 1.0))
            phase = _phase_from_perc_prox(pp, pr)
            concern_row = {
                "pid": pid,
                "persona_label": persona_label,
                "cid": cid,
                "concern_label": _set_node_label(G, cid),
                "stage": stage,
                "activation": act,
                "involvement": I,
                "lift_proxy": lift_proxy,
                "perceptibility": pp,
                "proximity": pr,
                "concern_involvement": inv_c,
                "keyness": kn,
                "phase": phase,
            }
            concerns_by_persona[pid].append(concern_row)
            concern_backlog.append(concern_row)

    for pid, rows in concerns_by_persona.items():
        rows.sort(key=lambda x: x["lift_proxy"], reverse=True)
        concerns_by_persona[pid] = rows[:top_k_per_persona]

    concern_backlog.sort(key=lambda x: x["lift_proxy"], reverse=True)
    return concern_backlog, concerns_by_persona
