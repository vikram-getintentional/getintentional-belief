from collections import defaultdict
import math
import networkx as nx
import numpy as np
from typing import Dict, Iterable, List, Tuple, Any, Optional

from backend.utils.graph_base.network_graph import _set_node_label, get_edge_attribute, get_node_by_id, get_nodes_list_ids, get_product_id_from_subgraph, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
from backend.utils.inference.rcs_generators.rcs_computations.graphwin_runtime import get_graphwin

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
    alpha_forward: float = 0.85,          # forward walk: seeds -> graph
    alpha_backward: float = 0.65,         # backward walk: product -> graph (different alpha to reduce correlation)
    weight_key: str = "likelihood",
    score_kinds: Tuple[str, ...] = ("attribute_value", "pain", "pain_trigger", "job", "capability", "persona"),
    exclude_from_involvement: Tuple[str, ...] = (),  # involvement usually excludes seeds/attributes
) -> Dict[str, Dict[str, float]]:
    """
    Decoupled definitions:
      - activation(n) := forward personalized PR from engaged seeds on the forward graph.
      - involvement(n) := node's share of the (seeds -> product) *flow*, where
            edge_contrib(u->v) = forward[u] * w(u->v) * backward[v]
        and node involvement is the normalized sum of incident edge contributions.

    Returns per-node {activation, involvement, strength=sqrt(activation*involvement)} for relevant types.
    """

    # Make sure forward graph has consistent numeric weights
    R = product_graph
    _ensure_edge_weights(R, weight_key=weight_key)
    F = _build_reversed_and_normalized(R, weight_key=weight_key)

    # FORWARD from PRODUCT on the original orientation (product → … → attribute)
    forward_raw_R = _ppr(R, {product_id: 1.0}, alpha=alpha_forward)

    # BACKWARD from engaged ATTRIBUTES on the reversed graph (attributes → … → product in F)
    engaged_attributes = {nid: float(w) for nid, w in engaged_attributes.items() if nid in product_graph}
    if not engaged_attributes:
        print("No engaged attributes detected in input graph")
    if engaged_attributes:
        backward_raw_F = _ppr(F, engaged_attributes, alpha=alpha_backward)
    else:
        backward_raw_F = {n: 0.0 for n in F.nodes}

    # Edge flow on R (not F)
    edge_contrib = {}
    total_flow = 0.0
    for u, v, d in R.edges(data=True):
        fu = float(forward_raw_R.get(u, 0.0))
        bv = float(backward_raw_F.get(v, 0.0))
        w  = float(d.get("weight", 1e-9))
        c = fu * w * bv
        if c > 0.0:
            edge_contrib[(u, v)] = c
            total_flow += c

    node_flow = {n: 0.0 for n in R.nodes}
    if total_flow > 0.0:
        for (u, v), c in edge_contrib.items():
            node_flow[u] += c
            node_flow[v] += c
        inv_raw = {n: node_flow[n] / (2.0 * total_flow) for n in R.nodes}
    else:
        inv_raw = {n: 0.0 for n in R.nodes}

    # Activation (exact)
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
        p_with = get_graphwin(R, engaged_plus_list)["win_likelihood"]
        pw = float(p_with)

        ow = _odds(pw)
        dl = _logit(pw) - l0

        activation_map[n] = {
            "p_with": p_with,
            "activation": max(0.0, ow - o0),   # odds lift
            "p_base": p0,
            "delta_p": max(0.0, pw - p0),  # probability lift
            "delta_logit": max(0.0, dl),  # logit lift
            "activation_linearized": activation_from(pw, l0, p0),  # linearized activation
        }

    # Results (no robust scaling in the actual product metric)
    exclude_set = set(exclude_from_involvement)  # consider ("product",) if you want
    results = {}
    for n, d in R.nodes(data=True):
        if d.get("node_type") not in score_kinds:
            continue
        involvement = float(inv_raw.get(n, 0.0))
        if d.get("node_type") in exclude_set:
            involvement = 0.0
        am = activation_map.get(n, {})
        activation = float(am.get("activation", 0.0))
        delta_p = float(am.get("delta_p", 0.0))
        raw_pw = float(am.get("p_with", 0.0))
        strength = involvement * delta_p
        results[n] = {"activation": activation, "involvement": involvement, "strength": strength, "activation_map": am, "id": n, "label": _set_node_label(R,n)}
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
    """
    Returns:
      {
        "persona_scores": { persona_id: {activation, involvement, strength, label, p_base, p_persona, mode, contributing_jobs: [...]}, ... },
        "persona_activation_breakdown": [
          {
            "persona_id": pid,
            "involvement": I,
            "problem_activation":    {activation, strength, prob_lift, logit_lift, concern_id},
            "execution_activation":  {...},
            "pain_activation":       {...},
            "resolution_activation": {...},
          }, ...
        ]
      }
    """
    def _extract_p_base(core_scores: Dict[str, Dict[str, float]]) -> float:
        for v in core_scores.values():
            am = v.get("activation_map")
            if isinstance(am, dict) and "p_base" in am:
                return float(am["p_base"])
        return 0.0

    p_base = _extract_p_base(core_scores)
    l_base = _logit(p_base)

    all_pruned_jobs = get_nodes_list_ids(product_graph, "job", {})
    persona_acc: Dict[str, Dict[str, Any]] = {}
    if not all_pruned_jobs:
        return {
            "persona_scores": {},
            "persona_activation_breakdown": {},
        }
    for job in all_pruned_jobs:
        js = core_scores.get(job, {"involvement": 0.0, "activation_map": {}})
        i = float(js.get("involvement", 0.0))
        jam = js.get("activation_map") or {}

        # Pull lifts ONCE from jam
        delta_p  = float(jam.get("delta_p", 0.0))
        dlogit   = float(jam.get("delta_logit", 0.0))

        # Skip dead jobs
        if (mode == "odds" and dlogit == 0.0 and i == 0.0) or (mode == "prob" and delta_p == 0.0 and i == 0.0):
            continue

        personas_for_job = get_target_nodes_by_source_and_type(original_graph, job, "performed_by")
        if not personas_for_job:
            continue

        for pid in personas_for_job:
            pj_rel = float(get_edge_attribute(original_graph, job, pid, "relevance"))
            acc = persona_acc.setdefault(
                pid, {"act_terms": [], "inv_terms": [], "contrib": [], "sum_dlogit": 0.0, "act_terms_prob": []}
            )

            # involvement always scales by relevance
            acc["inv_terms"].append(i * pj_rel)

            if mode == "odds":
                acc["sum_dlogit"] += odds_relevance_scale * (pj_rel * dlogit)
            else:  # "prob"
                acc["act_terms_prob"].append(delta_p * pj_rel)

            acc["contrib"].append({
                "job": job,
                "concern_level": "execution",
                "edge_weight": pj_rel,
                "job_activation_prob_lift": delta_p,
                "job_activation_logit_lift": dlogit,
                "job_involvement": i,
                "job_strength_prob": i * delta_p,
                "concern_id": job,
            })

            # (B) problems solved by job: job --solves--> pain
            solved_pains_of_job = get_target_nodes_by_source_and_type(product_graph, job, "solves")
            for p in solved_pains_of_job:
                job_to_pain = float(get_edge_attribute(product_graph, job, p, "relevance"))
                ps = core_scores.get(p, {"involvement": 0.0, "activation_map": {}})
                pam = ps.get("activation_map") or {}
                delta_pp  = float(pam.get("delta_p", 0.0))
                dlogit_pp = float(pam.get("delta_logit", 0.0))

                if mode == "odds":
                    acc["sum_dlogit"] += odds_relevance_scale * (pj_rel * job_to_pain * dlogit_pp)
                else:
                    acc["act_terms_prob"].append(delta_pp * pj_rel * job_to_pain)

                acc["contrib"].append({
                    "job": job,
                    "concern_level": "problem",
                    "edge_weight": pj_rel * job_to_pain,
                    "job_activation_prob_lift": delta_pp,
                    "job_activation_logit_lift": dlogit_pp,
                    "job_involvement": float(ps.get("involvement", 0.0)),
                    "job_strength_prob": float(ps.get("involvement", 0.0)) * delta_pp,
                    "concern_id": p,
                })

            # (C) felt pains for job: pain --felt_in--> job (pain is source)
            felt_pains_of_job = get_source_nodes_by_target_and_type(product_graph, job, "felt_in")
            for pa in felt_pains_of_job:
                job_to_pain = float(get_edge_attribute(product_graph, pa, job, "likelihood"))
                pas = core_scores.get(pa, {"involvement": 0.0, "activation_map": {}})
                paam = pas.get("activation_map") or {}
                delta_pa  = float(paam.get("delta_p", 0.0))
                dlogit_pa = float(paam.get("delta_logit", 0.0))

                if mode == "odds":
                    acc["sum_dlogit"] += odds_relevance_scale * (pj_rel * job_to_pain * dlogit_pa)
                else:
                    acc["act_terms_prob"].append(delta_pa * pj_rel * job_to_pain)

                acc["contrib"].append({
                    "job": job,
                    "concern_level": "pain",
                    "edge_weight": pj_rel * job_to_pain,
                    "job_activation_prob_lift": delta_pa,
                    "job_activation_logit_lift": dlogit_pa,
                    "job_involvement": float(pas.get("involvement", 0.0)),  # FIX: use pas, not ps
                    "job_strength_prob": float(pas.get("involvement", 0.0)) * delta_pa,  # FIX
                    "concern_id": pa,
                })

                # (D) resolving nodes for a felt pain: rs --solves--> pa
                solving_nodes_of_pain = get_source_nodes_by_target_and_type(product_graph, pa, "solves")
                for rs in solving_nodes_of_pain:
                    if rs == job:
                        continue
                    job_relevance = float(get_edge_attribute(product_graph, rs, pa, "relevance"))
                    rs_s  = core_scores.get(rs, {"involvement": 0.0, "activation_map": {}})
                    rsam  = rs_s.get("activation_map") or {}
                    delta_rs  = float(rsam.get("delta_p", 0.0))
                    dlogit_rs = float(rsam.get("delta_logit", 0.0))

                    if mode == "odds":
                        acc["sum_dlogit"] += odds_relevance_scale * (pj_rel * job_relevance * dlogit_rs)
                    else:
                        acc["act_terms_prob"].append(delta_rs * pj_rel * job_relevance)

                    acc["contrib"].append({
                        "job": job,
                        "concern_level": "resolution",
                        "edge_weight": pj_rel * job_relevance,
                        "job_activation_prob_lift": delta_rs,
                        "job_activation_logit_lift": dlogit_rs,
                        "job_involvement": float(rs_s.get("involvement", 0.0)),
                        "job_strength_prob": float(rs_s.get("involvement", 0.0)) * delta_rs,
                        "concern_id": rs,
                    })
    
    # -------- Assemble outputs --------
    persona_scores_out: Dict[str, Dict[str, Any]] = {}
    activation_breakdown_list: List[Dict[str, Any]] = []

    for pid, acc in persona_acc.items():
        I = noisy_or(acc["inv_terms"]) if acc["inv_terms"] else 0.0
        if mode == "odds":
            l_persona = l_base + acc["sum_dlogit"]   # additive in log-odds
            p_persona = _sigmoid(l_persona)
            A = max(0.0, p_persona - p_base)         # activation lift in prob-space
        else:
            A = noisy_or(acc["act_terms_prob"]) if acc["act_terms_prob"] else 0.0
            p_persona = min(1.0, p_base + A)

        S = A * I
        label = _set_node_label(original_graph, pid)

        # --- persona_scores map (unchanged shape) ---
        persona_scores_out[pid] = {
            "activation": float(A),
            "involvement": float(I),
            "strength": float(S),
            "label": label,
            "p_base": float(p_base),
            "p_persona": float(p_persona),
            "mode": mode,
            #"contributing_jobs": sorted(acc["contrib"], key=lambda r: r["job_strength_prob"], reverse=True),
        }

        

        # --- activation breakdown (pick top per concern_level) ---
        top_per_level = {
            "execution": None,
            "problem": None,
            "pain": None,
            "resolution": None,
        }
        for row in acc["contrib"]:
            lvl = row.get("concern_level")
            if lvl not in top_per_level:
                continue
            # choose the best by job_strength_prob (or edge-weighted lift)
            current = top_per_level[lvl]
            if (current is None) or (row.get("job_strength_prob", 0.0) > current.get("job_strength_prob", 0.0)):
                top_per_level[lvl] = row

        def _mk_activation_entry(best):
            if not best:
                return {"activation": 0.0, "strength": 0.0, "prob_lift": 0.0, "logit_lift": 0.0, "concern_id": None}
            return {
                "activation": float(best.get("job_activation_prob_lift", 0.0) or 0.0),
                "strength": float(best.get("job_strength_prob", 0.0) or 0.0),
                "prob_lift": float(best.get("job_activation_prob_lift", 0.0) or 0.0),
                "logit_lift": float(best.get("job_activation_logit_lift", 0.0) or 0.0),
                "concern_id": best.get("concern_id"),
            }
        persona_node = get_node_by_id(original_graph, pid)
        activation_breakdown_list.append({
            "persona_id": pid,
            "label": label,
            "department": persona_node.get("department"),
            "seniority": persona_node.get("seniority"),
            "title": persona_node.get("title"),
            "involvement": float(I),
            "execution_activation":  _mk_activation_entry(top_per_level["execution"]),
            "problem_activation":    _mk_activation_entry(top_per_level["problem"]),
            "pain_activation":       _mk_activation_entry(top_per_level["pain"]),
            "resolution_activation": _mk_activation_entry(top_per_level["resolution"]),
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
    
    
    # Gather engaged attributes
    attr_weights: Dict[str, float] = {}
    for e in engaged_nodes:
        nid = e.get("id", "")
        if nid.startswith("attribute_value:") and nid in G:
            attr_weights[nid] = float(e.get("occurrence", 1.0)) or 1.0

    if not attr_weights:
        print("[WARN] No engaged attribute_value nodes found.")
        return {"core_scores": {}, "persona_scores": {}}
    
    # Compute core scores (pain, job, trigger, etc.)
    core_scores = compute_node_strengths(
        product_graph=G,
        product_id=product_id,
        engaged_attributes=attr_weights,
        score_kinds=("attribute_value", "pain", "pain_trigger", "job", "capability")
    )
    

    # Compute persona scores from jobs
    persona_scores_ret = compute_persona_scores_from_core(
        product_graph=G,
        original_graph=Original_G,
        core_scores=core_scores,
    )



    #print("Core Scores:", core_scores)
    #print("Persona Scores:", persona_scores)

    return {
        "core_scores": core_scores,
        "persona_scores": persona_scores_ret["persona_scores"],
        "activation_breakdown": persona_scores_ret["persona_activation_breakdown"]
    }
# ============================================================
# Concern Backlog Construction (Derived from Persona Scores )
# ============================================================

def build_concern_backlog_from_activation_breakdown(G:nx.DiGraph, activation_breakdown, stage_weights=None, top_k_per_persona=5):
    """
    activation_breakdown: list of persona activation summaries
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
        I = p.get("involvement", 0.0)
        # iterate over each activation layer
        for stage in ["execution", "problem", "pain", "resolution"]:
            a = p.get(f"{stage}_activation", {})
            cid = a.get("concern_id")
            act = float(a.get("activation", 0.0))
            if not cid or act <= 0:
                continue
            lift_proxy = act * I * stage_weights.get(stage, 1.0)
            concern_row = {
                "pid": pid,
                "persona_label": _set_node_label(G, pid),
                "cid": cid,
                "concern_label": _set_node_label(G, cid),
                "stage": stage,
                "activation": act,
                "involvement": I,
                "lift_proxy": lift_proxy,
            }
            concerns_by_persona[pid].append(concern_row)
            concern_backlog.append(concern_row)

    # sort within persona and globally
    for pid, rows in concerns_by_persona.items():
        rows.sort(key=lambda x: x["lift_proxy"], reverse=True)
        concerns_by_persona[pid] = rows[:top_k_per_persona]

    concern_backlog.sort(key=lambda x: x["lift_proxy"], reverse=True)
    return concern_backlog, concerns_by_persona

"""


def _score_for_concern(
    G: nx.DiGraph,
    cid: str,
    node_scores: Dict[str, Dict[str, float]],
    *,
    job_weight_key: str = "weight"
) -> Tuple[float, float]:
    
    Return (activation, involvement) for a concern node.
    If not present in node_scores (common for pains/pain_triggers),
    backfill from neighboring JOB nodes using edge weights (weighted mean).
    
    s = node_scores.get(cid)
    if s is not None:
        return float(s.get("activation", 0.0) or 0.0), float(s.get("involvement", 0.0) or 0.0)

    # Backfill from adjacent JOBs (both directions to be safe with schema)
    total_w = 0.0
    act_sum = 0.0
    inv_sum = 0.0

    # In-neighbors
    for u, v, ed in G.in_edges(cid, data=True):
        if G.nodes[u].get("node_type") == "job":
            w = float(ed.get(job_weight_key, 1.0) or 1.0)
            js = node_scores.get(u) or {}
            act_sum += w * float(js.get("activation", 0.0) or 0.0)
            inv_sum += w * float(js.get("involvement", 0.0) or 0.0)
            total_w += w

    # Out-neighbors
    for u, v, ed in G.out_edges(cid, data=True):
        if G.nodes[v].get("node_type") == "job":
            w = float(ed.get(job_weight_key, 1.0) or 1.0)
            js = node_scores.get(v) or {}
            act_sum += w * float(js.get("activation", 0.0) or 0.0)
            inv_sum += w * float(js.get("involvement", 0.0) or 0.0)
            total_w += w

    if total_w > 0.0:
        return act_sum / total_w, inv_sum / total_w
    return 0.0, 0.0


def _concern_backlog(
    G: nx.DiGraph,
    persona_ids: List[str],
    node_scores: Dict[str, Dict[str, float]],
    *,
    top_k: int = 60,
) -> List[Dict[str, Any]]:
    
    Build a concern backlog for top personas using graph connectivity × node activation/involvement.

    For each persona, rank pains/jobs by:
        lift_proxy = connectivity(persona → concern via jobs) * activation(concern) * involvement(concern)

    Returns a *flat* list that is the union of top_k rows per persona, sorted by lift_proxy desc.
    

    # Candidate concerns
    concerns = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") in ("pain", "pain_trigger", "job")
    ]
    print(f"Found {len(concerns)} candidate concerns in graph")

    # Per-persona ranking bucket
    per_persona: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for pid in persona_ids:
        rows: List[Dict[str, Any]] = []
        for cid in concerns:
            if pid == cid:
                continue

            conn = _connectivity_persona_pain_via_jobs(G, pid, cid)
            if conn <= 0.0:
                continue

            print(f"Persona {pid} to Concern {cid} connectivity: {conn}")

            # Robust scoring (with job-based backfill)
            act, inv = _score_for_concern(G, cid, node_scores)
            if act <= 0.0 or inv <= 0.0:
                continue

            lift = conn * act * inv
            if lift <= 0.0:
                continue

            rows.append({
                "pid": pid,
                "cid": cid,
                "concern_label": G.nodes[cid].get("label") or G.nodes[cid].get("name") or cid,
                "stage": infer_stage_for_concern(G, cid),
                "connectivity": conn,
                "activation": act,
                "involvement": inv,
                "lift_proxy": lift,
            })

        # sort within persona and cap
        rows.sort(key=lambda x: x["lift_proxy"], reverse=True)
        per_persona[pid] = rows[:top_k]

    # Flatten and sort globally for convenience
    flat = [r for rows in per_persona.values() for r in rows]
    flat.sort(key=lambda x: x["lift_proxy"], reverse=True)
    return flat
    """