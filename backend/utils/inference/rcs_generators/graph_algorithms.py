from collections import defaultdict
import math
import networkx as nx
import numpy as np
from typing import Dict, Iterable, List, Tuple, Any, Optional

from backend.utils.graph_base.network_graph import _set_node_label, get_edge_attribute, get_node_by_id, get_nodes_list_ids, get_product_id_from_subgraph, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
from backend.utils.inference.rcs_generators.rcs_computations.graphwin_runtime import get_graphwin

# ------------------------------------------------------------
# Phase bucketing from Perceptibility/Proximity
# ------------------------------------------------------------
def _phase_from_perc_prox(perc: float, prox: float) -> str:
    """
    Bucket a concern by where it likely sits in the belief journey
    using (perceptibility, proximity). Thresholds are MVP-tunable.
    """
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
# Helper: Build reversed + normalized graph
# ============================================================
def _build_reversed_and_normalized(G: nx.DiGraph, weight_key: str = "likelihood") -> nx.DiGraph:
    R = G.reverse(copy=True)
    for u, v, data in R.edges(data=True):
        w = data.get(weight_key, 1.0)
        data["weight"] = max(float(w), 1e-9)
    return R

# ============================================================
# Helper: Personalized PageRank calculation
# ============================================================
def _ppr(G: nx.DiGraph, seeds: Dict[str, float], alpha: float = 0.85) -> Dict[str, float]:
    if not seeds:
        return {}
    personalization = {n: 0 for n in G.nodes()}
    for s, w in seeds.items():
        if s in personalization:
            personalization[s] = w
    scores = nx.pagerank(G, alpha=alpha, personalization=personalization, weight="weight")
    return scores

# ============================================================
# Stage inference helpers
# ============================================================
_STAGE_ORDER = {"problem": 0, "pain": 1, "solution": 2}
def _stage_rank(stage: str) -> int:
    return _STAGE_ORDER.get((stage or "").lower(), 1)

def infer_stage_for_concern(G: nx.DiGraph, node_id: str) -> str:
    """
    Infer a user-friendly 'stage' label for a node in the concern flow.
      problem  ≈ upstream signals/causes (attributes, zmots, triggers)
      pain     ≈ the felt pains / issues
      solution ≈ jobs/capabilities that resolve pains (toward product)

    Priority:
      1) explicit node["stage"] if present
      2) node_type mapping
      3) neighborhood-based heuristic
    """
    if node_id not in G:
        return "problem"

    # 1) explicit override
    stage = (G.nodes[node_id].get("stage") or "").strip().lower()
    if stage in _STAGE_ORDER:
        return stage

    # 2) type-based default
    t = (G.nodes[node_id].get("node_type") or G.nodes[node_id].get("type") or "").strip().lower()
    if t in {"attribute", "attribute_value", "zmot_event", "observable_moment", "keyword", "pain_trigger"}:
        return "problem"
    if t in {"pain"}:
        return "pain"
    if t in {"job", "capability"}:
        return "solution"
    if t in {"product"}:
        return "solution"  # terminal solution

    # 3) neighborhood heuristic
    # If it touches any pain, call it pain; else if it touches any job/capability, call it solution; else problem.
    for u, v in G.in_edges(node_id):
        nt = (G.nodes[u].get("node_type") or "").lower()
        if nt == "pain":
            return "pain"
        if nt in {"job", "capability"}:
            return "solution"
    for u, v in G.out_edges(node_id):
        nt = (G.nodes[v].get("node_type") or "").lower()
        if nt == "pain":
            return "pain"
        if nt in {"job", "capability"}:
            return "solution"

    return "problem"


# ============================================================
# Node Strength Computation (core propagation)
# ============================================================

def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    lo = min(d.values())
    hi = max(d.values())
    if hi <= lo:
        return {k: 0.0 for k in d}
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in d.items()}

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _engaged_payload_from_attr_weights(attr_w: Dict[str, float]) -> List[Dict[str, float]]:
    # GraphWin accepts a list of {"id": nid, "occurrence": w}
    payload = []
    for nid, w in attr_w.items():
        payload.append({"id": nid, "occurrence": float(w or 1.0)})
    return payload


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

def _robust01(vals: Iterable[float], lo_q: float = 0.05, hi_q: float = 0.95) -> Dict[int, float]:
    """
    Robustly scale a dense list-like to [0,1] using percentiles (clipping outside).
    Returns an index->scaled dict so we can map back efficiently.
    """
    arr = np.asarray(list(vals), dtype=float)
    if arr.size == 0:
        return {}
    lo = np.quantile(arr, lo_q)
    hi = np.quantile(arr, hi_q)
    if hi <= lo:
        # degenerate: all same
        out = np.zeros_like(arr)
    else:
        out = (arr - lo) / (hi - lo)
        out = np.clip(out, 0.0, 1.0)
    return {i: float(x) for i, x in enumerate(out)}

def _logit(p, eps=1e-9):
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p/(1.0 - p))


def activation_from(p_with, logit_base: float, baseline: float) -> float:
    dl = _logit(p_with) - logit_base      # log-odds lift (can be tiny but nonzero)
    # keep a linearized activation in [0,1] scale for consistency:
    # map a logit delta back to probability delta around baseline via local slope:
    slope = baseline * (1 - baseline)     # derivative of sigmoid at baseline
    return max(0.0, slope * dl)   

def _odds(p: float, eps: float = 1e-12) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return p / (1.0 - p)


# --- main ----------------------------------------------------

def compute_node_strengths(
    product_graph: nx.DiGraph,
    product_id: str,
    engaged_attributes: Dict[str, float],
    *,
    alpha_forward: float = 0.85,          # forward walk: seeds -> graph (product seed)
    alpha_backward: float = 0.65,         # backward walk: engaged -> product (engaged seeds on reversed)
    weight_key: str = "likelihood",
    score_kinds: Tuple[str, ...] = ("attribute_value", "pain", "pain_trigger", "job", "capability", "persona"),
    exclude_from_involvement: Tuple[str, ...] = (),
) -> Dict[str, Dict[str, float]]:
    """
    Returns per-node scores for the requested types:
      - activation(n): exact odds / prob lift via GraphWin when n is added to engaged
      - involvement(n): share of seeds→product flow collected at n (edge-flow based)
      - strength(n): involvement(n) * delta_p(n)
      - perceptibility(n): PPR(n | engaged)   (Perc)   0..1 normalized (min-max across graph)
      - proximity(n):      PPR(n | product)   (Prox)   0..1 normalized (min-max across graph)
    """
    R = product_graph
    _ensure_edge_weights(R, weight_key=weight_key)
    F = _build_reversed_and_normalized(R, weight_key=weight_key)

    # ---- Proximity: forward from product on original orientation (R)
    forward_raw_R = _ppr(R, {product_id: 1.0}, alpha=alpha_forward)

    # ---- Perceptibility: from engaged attributes on reversed orientation (F)
    engaged_attributes = {nid: float(w) for nid, w in engaged_attributes.items() if nid in product_graph}
    backward_raw_F = _ppr(F, engaged_attributes, alpha=alpha_backward) if engaged_attributes else {n: 0.0 for n in F.nodes}

    # ---- Normalize Perc/Prox to 0..1 over the current node set
    perc_norm = _normalize(backward_raw_F)   # perceptibility
    prox_norm = _normalize(forward_raw_R)    # proximity

    # ---- Edge-flow based involvement
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
        for (u, v), c in edge_contrib.items():
            inv_raw[u] += c
            inv_raw[v] += c
        inv_raw = {n: (c / total_flow) for n, c in inv_raw.items()}
    else:
        inv_raw = {n: 0.0 for n in R.nodes}

    # ---- Activation via GraphWin deltas
    engaged_list = [{"id": nid, "occurrence": float(w)} for nid, w in engaged_attributes.items()]
    base_out = get_graphwin(R, engaged_list)["win_likelihood"]
    p0 = float(base_out)
    o0 = _odds(p0)
    l0 = _logit(p0)

    activation_map = {}
    for n, d in R.nodes(data=True):
        if d.get("node_type") not in score_kinds:
            continue
        engaged_plus = dict(engaged_attributes)
        engaged_plus[n] = 1.0
        engaged_plus_list = [{"id": nid, "occurrence": float(w)} for nid, w in engaged_plus.items()]
        pw = float(get_graphwin(R, engaged_plus_list)["win_likelihood"])
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
# Helper: Get all jobs linked to a persona
# ============================================================
def _jobs_for_persona(G: nx.DiGraph, persona_id: str) -> List[str]:
    """
    Returns all job nodes performed by a persona node.
    """
    jobs = set()
    job_list = get_source_nodes_by_target_and_type(G, persona_id, "performed_by")
    jobs.update(job_list)
    jobs = list(jobs)
    return jobs


# ============================================================
# Persona-Level Aggregation
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
        if w <= 0.0: return
        acc_lvl["perc_sum"] += perc * w
        acc_lvl["prox_sum"] += prox * w
        acc_lvl["w_sum"]    += w

    for job in all_pruned_jobs:
        js = core_scores.get(job, {"involvement": 0.0, "activation_map": {}})
        i = float(js.get("involvement", 0.0))
        jam = js.get("activation_map") or {}
        delta_p  = float(jam.get("delta_p", 0.0))
        dlogit   = float(jam.get("delta_logit", 0.0))

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

            # Execution (job itself)
            w_job = pj_rel * i
            if w_job > 0.0:
                acc["perc_sum"] += job_perc * w_job
                acc["prox_sum"] += job_prox * w_job
                acc["w_sum"]    += w_job
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
                delta_pp  = float(pam.get("delta_p", 0.0))
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
                    acc["w_sum"]    += w_prob
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
                delta_pa  = float(paam.get("delta_p", 0.0))
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
                    acc["w_sum"]    += w_pain
                    _acc_level(acc["levels"]["pain"], pa_perc, pa_prox, w_pain)

                acc["contrib"].append({
                    "job": job, "concern_level": "pain", "edge_weight": pj_rel * job_to_pain,
                    "job_activation_prob_lift": delta_pa, "job_activation_logit_lift": dlogit_pa,
                    "job_involvement": pa_inv, "job_strength_prob": pa_inv * delta_pa, "concern_id": pa,
                })

                # Resolution (resolvers of pain)
                solving_nodes_of_pain = get_source_nodes_by_target_and_type(product_graph, pa, "solves")
                for rs in solving_nodes_of_pain:
                    if rs == job: continue
                    job_relevance = float(get_edge_attribute(product_graph, rs, pa, "relevance"))
                    rs_s  = core_scores.get(rs, {"involvement": 0.0, "activation_map": {}})
                    rsam  = rs_s.get("activation_map") or {}
                    delta_rs  = float(rsam.get("delta_p", 0.0))
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
                        acc["w_sum"]    += w_res
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
            l_persona = l_base + acc["sum_dlogit"]
            p_persona = _sigmoid(l_persona)
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
            if lvl not in top_per_level: continue
            cur = top_per_level[lvl]
            if (cur is None) or (row.get("job_strength_prob", 0.0) > cur.get("job_strength_prob", 0.0)):
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
# Final Combined Report Generator
# ============================================================
def get_involvement_activation_report(
    G: nx.DiGraph,
    Original_G: nx.DiGraph,
    engaged_nodes: Optional[List[Dict]] = None,
    *,
    alpha: float = 0.85,
    weight_key: str = "likelihood",
) -> Dict[str, Dict[str, Any]]:
    """
    Compute involvement & activation scores for both core and persona nodes.

    Returns:
        {
          "core_scores": {node_id: {activation, involvement, strength}},
          "persona_scores": {persona_id: {activation, involvement, strength}},
        }
    """
    engaged_nodes = engaged_nodes or []

    # Identify product node
    product_nodes = [n for n, d in Original_G.nodes(data=True) if d.get("node_type") == "product"]
    if not product_nodes:
        return {"core_scores": {}, "persona_scores": {}}
    product_id = get_product_id_from_subgraph(Original_G)
    if not product_id:
        return {"core_scores": {}, "persona_scores": {}}

    # Collect engaged attributes
    attr_weights: Dict[str, float] = {}
    for e in engaged_nodes:
        nid = e.get("id", "")
        if nid.startswith("attribute_value:") and nid in G:
            attr_weights[nid] = float(e.get("occurrence", 1.0)) or 1.0

    if not attr_weights:
        print("[WARN] No engaged attribute_value nodes found.")
        # 🔁 Still return the graph so downstream has a consistent object
        return {
            "graph": G,
            "core_scores": {},
            "persona_scores": {},
            "activation_breakdown": []
        }

    # Core node scores (includes perc/prox/involvement/activation maps)
    core_scores = compute_node_strengths(
        product_graph=G,
        product_id=product_id,
        engaged_attributes=attr_weights,
        score_kinds=("attribute_value", "pain", "pain_trigger", "job", "capability")
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
    G.graph["engaged_attributes"] = dict(attr_weights)  # seed set for this run
    G.graph["p_base"] = p_base

    # --- annotate core nodes ---
    for nid, sc in core_scores.items():
        nd = G.nodes[nid]
        nd["perceptibility"] = float(sc.get("perceptibility", 0.0))
        nd["proximity"]      = float(sc.get("proximity", 0.0))
        nd["involvement"]    = float(sc.get("involvement", 0.0))
        nd["strength"]       = float(sc.get("strength", 0.0))
        # optional: a compact activation signal
        nd["delta_p"]        = float(sc.get("activation_map", {}).get("delta_p", 0.0))

    # --- annotate personas (these may live only in Original_G, so guard existence) ---
    for pid, ps in persona_scores_ret["persona_scores"].items():
        if pid not in G: 
            continue
        nd = G.nodes[pid]
        nd["persona_activation"]   = float(ps.get("activation", 0.0))
        nd["persona_involvement"]  = float(ps.get("involvement", 0.0))
        nd["persona_strength"]     = float(ps.get("strength", 0.0))
        nd["persona_perceptibility"]= float(ps.get("perceptibility", 0.0))
        nd["persona_proximity"]    = float(ps.get("proximity", 0.0))

    return {
        "graph": G,  # ← NEW: the seeded/working graph used for this run
        "core_scores": core_scores,
        "persona_scores": persona_scores_ret["persona_scores"],
        "activation_breakdown": persona_scores_ret["persona_activation_breakdown"]
    }
# ============================================================
# Concern Backlog Construction (Derived from Persona Scores )
# ============================================================
def _keyness_from(pp: float, pr: float, inv: float, a=(1.2, 1.0, 1.0)) -> float:
    a1, a2, a3 = a
    x = a1*pr + a2*pp + a3*inv
    return 1.0 / (1.0 + math.exp(-x))  # sigmoid

def build_concern_backlog_from_activation_breakdown(
    G: nx.DiGraph,
    activation_breakdown,
    stage_weights=None,
    top_k_per_persona: int = 5,
):
    """
    activation_breakdown: list of persona activation summaries (each level has perc/prox)
    stage_weights: optional dict to prioritize earlier concern types
    Returns:
      concern_backlog: flat ranked list of actionable concerns
      concerns_by_persona: mapping of persona_id -> list of concerns
    """
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

            # node-local perc/prox packed in the level dict
            pp = float(a.get("perceptibility", 0.0))
            pr = float(a.get("proximity", 0.0))

            # if you store concern-node involvement on G.nodes[cid], include it; else 0.0
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
                "involvement": I,                 # persona involvement
                "lift_proxy": lift_proxy,
                "perceptibility": pp,             # concern node perc
                "proximity": pr,                  # concern node prox
                "concern_involvement": inv_c,     # concern node inv (if available)
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
