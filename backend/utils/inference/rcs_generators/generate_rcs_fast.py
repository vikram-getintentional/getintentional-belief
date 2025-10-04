# ============================
# File: backend/utils/rcs_generators/generate_rcs_fast.py
# ============================
"""
Fast, feature-complete RCS generator.

- Uses rcs_staged (PPREngine + overlays + warm starts) to avoid recomputing PageRank.
- Returns the SAME report schema as the legacy rcs_generator_engine.
- Archetype-free: conditions entirely on engaged_nodes.
- ZMOT/attribute boosts are handled in rcs_staged.rcs_prepare().
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import itertools
import heapq
import networkx as nx
import numpy as np
from collections import defaultdict

from backend.utils.graph_base.network_graph import (
    _set_node_label,
    get_edge_attribute,
    get_node_by_id,
    get_nodes_list_ids,
    get_product_id_from_subgraph,
    get_source_nodes_by_target_and_type,
    get_target_nodes_by_source_and_type,
)

from backend.utils.inference.rcs_generators.rcs_helpers.graphwin import compute_graphwin
from backend.utils.inference.rcs_generators.rcs_helpers.reach import reverse_reach_to_product_bulk
from backend.utils.knowledge_base.arsenal.execution_arsenal_repository import (
    get_best_plays_for_concern,
)

# ---- staged, overlay-based helpers (fast PR) ----
from backend.utils.inference.rcs_generators.rcs_helpers.rcs_staged import (
    RCSPPRContext,
    rcs_pair_effects,
    rcs_prepare,
    rcs_marginal_lifts,
    rcs_concern_potentials,
)

# ---- small math/graph helpers reused from your centralized algorithms ----
from backend.utils.inference.rcs_generators.graph_algorithms import (
    clip01 as _clip,
    compute_activation_from_uplift,
    sigmoid as _sig,
    noisy_or as _noisy_or,
)

# =========================================================
# ------------- Local helpers (lightweight) ---------------
# =========================================================
EPS = 1e-9          # numerical floor for “non-zero”
EPS_PRINT = 1e-12   # treat smaller as 0 for display


def _jobs_of_persona(G: nx.DiGraph, persona_id: str) -> List[str]:
    j = get_source_nodes_by_target_and_type(G, persona_id, "performed_by") or []
    return [j for j in set(j) if j in G]

def _felt_pains_of_job(G: nx.DiGraph, job_id: str) -> List[str]:
    return [p for p in (get_source_nodes_by_target_and_type(G, job_id, "felt_in") or []) if p in G]

def _solve_pains_of_job(G: nx.DiGraph, job_id: str) -> List[str]:
    return [p for p in (get_target_nodes_by_source_and_type(G, job_id, "solves") or []) if p in G]

def _rel_job_to_persona(G: nx.DiGraph, job_id: str, persona_id: str) -> float:
    rel = get_edge_attribute(G, job_id, persona_id, "relevance")
    if rel is None:
        rel = get_edge_attribute(G, job_id, persona_id, "likelihood")
    return _clip(rel or 0.0)

def _rel_job_to_felt(G: nx.DiGraph, job_id: str, pain_id: str) -> float:
    return _clip(
        get_edge_attribute(G, job_id, pain_id, "relevance")
        or get_edge_attribute(G, pain_id, job_id, "relevance")
        or 0.5
    )

def _persona_involvement_from_jobs(G: nx.DiGraph, pr: Dict[str, float], persona_id: str) -> float:
    """same as legacy: noisy-OR across job PageRank weighted by relevance job->persona"""
    jobs = _jobs_of_persona(G, persona_id)
    if not jobs:
        return 0.0
    pairs = []
    for j in jobs:
        w = _rel_job_to_persona(G, j, persona_id)
        pj = float(pr.get(j, 0.0))
        if w > 0 and pj > 0:
            pairs.append(_clip(w * pj))
    return _clip(1.0 - float(__import__("functools").reduce(lambda a, b: a * (1 - b), pairs, 1.0))) if pairs else 0.0

def _felt_salience(felt_lik: float, rel_j2p: float, severity: float, beta=(0.0,1.0,1.0,1.0)) -> float:
    b0, b1, b2, b3 = beta
    return _sig(b0 + b1*felt_lik + b2*rel_j2p + b3*severity)

def _felt_lik(G: nx.DiGraph, pain_id: str, job_id: str) -> float:
    return _clip(get_edge_attribute(G, pain_id, job_id, "likelihood") or 0.0)

def _felt_severity(G: nx.DiGraph, pain_id: str) -> float:
    return _clip(G.nodes[pain_id].get("severity", 0.5))

def _felt_bundle_for_job(G: nx.DiGraph, job_id: str, pains: List[str], beta=(0.0,1.0,1.0,1.0)) -> float:
    if not pains:
        return 0.0
    vals = [
        _felt_salience(_felt_lik(G, p, job_id), _rel_job_to_felt(G, job_id, p), _felt_severity(G, p), beta)
        for p in pains
    ]
    return _noisy_or(vals)

def _job_lik(G: nx.DiGraph, job_id: str) -> float:
    return _clip(G.nodes[job_id].get("likelihood", 0.0))

def _job_activation(G: nx.DiGraph, persona_id: str, job_id: str, FJ: float, alpha=(0.0,1.0,1.0,1.0)) -> float:
    a0, a1, a2, a3 = alpha
    jl = _job_lik(G, job_id)
    rel = _rel_job_to_persona(G, job_id, persona_id)
    return _sig(a0 + a1*jl + a2*rel + a3*FJ)

def _job_attention_from_pr(job_ids: List[str], pr: Dict[str, float]) -> Dict[str, float]:
    s = sum(pr.get(j, 0.0) for j in job_ids) or 1.0
    return {j: float(pr.get(j, 0.0))/s for j in job_ids}

def _persona_activation(G: nx.DiGraph, persona_id: str, pr_for_attention: Dict[str, float],
                        alpha=(0.0,1.0,1.0,1.0), beta=(0.0,1.0,1.0,1.0), corr_lambda: float = 1.0) -> Tuple[float, Dict]:
    jobs = _jobs_of_persona(G, persona_id)
    if not jobs:
        return 0.0, {"jobs": []}
    attn = _job_attention_from_pr(jobs, pr_for_attention)
    rows, weights = [], []
    for j in jobs:
        pains = _felt_pains_of_job(G, j)
        FJ = _felt_bundle_for_job(G, j, pains, beta)
        AJ = _job_activation(G, persona_id, j, FJ, alpha)
        wj = max(1e-6, _rel_job_to_persona(G, j, persona_id) * max(1e-6, attn.get(j, 0.0)))
        rows.append({"job": j, "FJ": FJ, "AJ": AJ, "attn": attn.get(j, 0.0), "rel": _rel_job_to_persona(G, j, persona_id), "pains": pains})
        weights.append(wj)
    Z = sum(weights) or 1.0
    norm_w = [w/Z for w in weights]
    prod = 1.0
    for row, w in zip(rows, norm_w):
        prod *= (1.0 - corr_lambda * row["AJ"]) ** w
    A_i = 1.0 - prod
    return _clip(A_i), {"jobs": rows, "weights": norm_w}

def _rel_persona_to_concern(G: nx.DiGraph, persona_id: str, concern_id: str) -> float:
    """max over persona's jobs of relevance(job->persona) * relevance(job<->concern)"""
    best = 0.0
    for j in _jobs_of_persona(G, persona_id):
        rpj = _rel_job_to_persona(G, j, persona_id)
        if concern_id in _felt_pains_of_job(G, j):
            best = max(best, rpj * _rel_job_to_felt(G, j, concern_id))
        if concern_id in _solve_pains_of_job(G, j):
            rjs = _clip(get_edge_attribute(G, j, concern_id, "relevance") or 0.5)
            best = max(best, rpj * rjs)
    return _clip(best)

def _infer_stage_for_concern(G: nx.DiGraph, persona_id: str, concern_id: str) -> str:
    # If there's a pain -> job edge, we treat the pain as a PROBLEM signal
    jobs = set(_jobs_of_persona(G, persona_id))
    if concern_id not in G or not jobs:
        return "problem"

    # pain -> job  ==> problem
    for j in jobs:
        upstream_pains = get_source_nodes_by_target_and_type(G, j, "felt_in") or []
        if concern_id in upstream_pains:
            return "problem"

    # job -> pain  ==> pain
    for j in jobs:
        downstream_pains = get_target_nodes_by_source_and_type(G, j, "solves") or []
        if concern_id in downstream_pains:
            return "pain"

    # pain -> (some job) -> solves -> solution  ==> solution
    # (We check if concern is a solution node connected via solves from any job the persona does.)
    for j in jobs:
        sols = get_target_nodes_by_source_and_type(G, j, "solves") or []
        if concern_id in sols:
            return "solution"

    return "problem"


def _stage_explainer(stage: str) -> str:
    s = (stage or "").strip().lower()
    if s == "problem":
        return "Has not realized the problem"
    if s == "pain":
        return "Feels the pain but not acting on it yet"
    if s == "solution":
        return "Needs a resolution but does not know how"
    return (stage or "").capitalize()

def _reachable(ctx: RCSPPRContext, pid: str) -> bool:
    """Quick reachability (ignoring weights). If not reachable, lift will be 0."""
    G_rev = ctx.G_rev
    return pid in G_rev and ctx.conv_id in G_rev and nx.has_path(G_rev, pid, ctx.conv_id)

def _baseline_seeds(ctx: RCSPPRContext, persona_pool_ids: List[str]) -> List[str]:
    """
    Choose stable seeds for baseline PR:
    - never seed with conv/product
    - prefer reachable personas (in engine index) from a stable pool
    - fall back to all personas (reachable only)
    """
    eng = ctx.engine
    conv = ctx.conv_id
    G_rev = ctx.G_rev

    # Start from ctx.seeds, but strip conv and unreachable
    seeds = [s for s in (ctx.seeds or []) if s in eng.node_index and s != conv]
    seeds = [s for s in seeds if s in G_rev and nx.has_path(G_rev, s, conv)]

    # If empty, use persona_pool (top I/A) but only reachable ones
    if not seeds:
        seeds = [p for p in persona_pool_ids if p in eng.node_index and p != conv]
        seeds = [s for s in seeds if s in G_rev and nx.has_path(G_rev, s, conv)]

    # If still empty, last-resort: all personas in pruned graph that can reach conv
    if not seeds:
        all_personas = get_nodes_list_ids(ctx.G_pruned, "persona", {})
        seeds = [p for p in all_personas
                 if p in eng.node_index and p != conv and p in G_rev and nx.has_path(G_rev, p, conv)]

    return seeds[:16]

# Graph Math helpers

# ---------- Win probability (evidence → product) ----------

FORWARD_EDGE_TYPES = (
    # pain trigger → pain
    ("pain_trigger", "pain", "triggered_by"),     # accept any edge attr; we'll read likelihood if present
    # pain ↔ job
    ("pain", "job", "felt_in"),         # pain -> job (felt_in)
    ("job", "pain", "solves"),          # job -> pain  (solves)
    # job → capability
    ("job", "capability", None),
    # capability → product
    ("capability", "product", None),
)

def _etype(u_type, v_type, etype):
    return (u_type or "", v_type or "", etype or None)

def _node_type(G, n):
    return G.nodes[n].get("node_type") or G.nodes[n].get("type")

def _edge_lik(G, u, v):
    # prefer explicit likelihood/weight on edge; fall back to node likelihood
    w = (
        get_edge_attribute(G, u, v, "likelihood")
        or get_edge_attribute(G, u, v, "weight")
        or 0.0
    )
    if not w:
        # fallback: use target node prior as a soft strength
        w = float(G.nodes[v].get("likelihood", 0.0)) or 0.25
    return _clip(float(w))

def _node_prior(G, n):
    return _clip(float(G.nodes[n].get("likelihood", 0.0)))

def _neighbors_forward(G, u):
    tu = _node_type(G, u)
    for v in G.successors(u):
        tv = _node_type(G, v)
        et = G.edges[(u, v)].get("type")
        if _etype(tu, tv, et) in FORWARD_EDGE_TYPES:
            yield v

def _parents_forward(G, v):
    tv = _node_type(G, v)
    for u in G.predecessors(v):
        tu = _node_type(G, u)
        et = G.edges[(u, v)].get("type")
        if _etype(tu, tv, et) in FORWARD_EDGE_TYPES:
            yield u

def _boost_jobs_from_persona(G: nx.DiGraph, persona_id: str, P: Dict[str, float]):
    """If persona engaged, boost its jobs by rel(job->persona)."""
    for j in _jobs_of_persona(G, persona_id):
        rel = _rel_job_to_persona(G, j, persona_id)
        if rel > 0:
            P[j] = 1.0 - (1.0 - P.get(j, 0.0)) * (1.0 - rel)

def _initial_evidence_probs(G: nx.DiGraph, engaged_nodes: List[Dict], pre: Dict) -> Dict[str, float]:
    """
    Build initial occurrence probabilities for pain_triggers, pains, jobs.
    Rules (your spec):
    - attribute_value: sets prob of its linked pain_triggers to their likelihood (boosted by ZMOT when present)
    - zmot_event: boosts those same pain_triggers (multiplicative to 1 via noisy-OR)
    - pain/job/pain_trigger explicitly engaged: set P=1.0
    - persona engaged: only boosts their jobs by rel(job->persona)
    """
    P0: Dict[str, float] = {}

    # 1) explicit occurrences
    for e in engaged_nodes or []:
        nid = e.get("id") if isinstance(e, dict) else e
        if nid not in G: 
            continue
        t = _node_type(G, nid)
        if t in ("pain_trigger", "pain", "job"):
            P0[nid] = 1.0

    # 2) attributes → pain_triggers via precomputed boosts if available
    # rcs_prepare prints "Trig Boosts now: [(pt_id, boost), ...]". Try to read it:
    trig_boosts = dict(pre.get("trig_boosts", [])) if isinstance(pre.get("trig_boosts"), list) else {}

    # If we have *any* attribute_value selected, use node likelihood for the *relevant* triggers.
    selected_attrs = { (e.get("id") if isinstance(e, dict) else e) 
                       for e in (engaged_nodes or []) 
                       if e and ( _node_type(G, (e.get("id") if isinstance(e, dict) else e)) == "attribute_value") }

    if selected_attrs:
        # Best-effort: if rcs_prepare provided concrete trigger ids, use those;
        # otherwise, fall back to ALL pain_triggers (conservative) but that’s capped by their priors.
        if trig_boosts:
            for pt_id, boost in trig_boosts.items():
                if pt_id in G and _node_type(G, pt_id) == "pain_trigger":
                    base = _node_prior(G, pt_id)
                    # Treat 'boost' as an additional chance the trigger fires.
                    p = 1.0 - (1.0 - base) * (1.0 - _clip(float(boost)))
                    P0[pt_id] = max(P0.get(pt_id, 0.0), p)
        else:
            # Fallback: allow triggers to fire at their prior
            for pt in [n for n, d in G.nodes(data=True) if _node_type(G, n) == "pain_trigger"]:
                P0[pt] = max(P0.get(pt, 0.0), _node_prior(G, pt))

    # 3) ZMOT events: if present, treat as extra boost on all triggers
    has_zmot = any((_node_type(G, (e.get("id") if isinstance(e, dict) else e)) == "zmot_event")
                   for e in (engaged_nodes or []))
    if has_zmot:
        for pt in [n for n, d in G.nodes(data=True) if _node_type(G, n) == "pain_trigger"]:
            # small uniform lift if we don't have explicit mapping
            P0[pt] = 1.0 - (1.0 - P0.get(pt, 0.0)) * (1.0 - 0.15)

    # 4) personas: boost their jobs *only*
    for e in (engaged_nodes or []):
        nid = e.get("id") if isinstance(e, dict) else e
        if nid in G and _node_type(G, nid) == "persona":
            _boost_jobs_from_persona(G, nid, P0)

    return P0

def _propagate_noisyor(G: nx.DiGraph, P0: Dict[str, float], conv_id: str,
                       max_iters: int = 40, tol: float = 1e-6) -> float:
    """
    Noisy-OR belief propagation along FORWARD_EDGE_TYPES.
    Update rule for each node v:
      P(v) = 1 - Π_{u in parents(v)} (1 - P(u) * w(u→v))
    with clamping to [0,1]. Initialize with P0; keep P for nodes not in subgraph at 0.
    """
    P = {n: 0.0 for n in G.nodes}
    for n, p in (P0 or {}).items():
        if n in P: P[n] = _clip(p)

    # iterate until converged
    for _ in range(max_iters):
        delta = 0.0
        P_new = dict(P)
        for v in G.nodes:
            # keep explicit evidence fixed at 1
            if P0.get(v, 0.0) >= 1.0 - 1e-12:
                continue
            parents = list(_parents_forward(G, v))
            if not parents:
                continue
            prod = 1.0
            for u in parents:
                pu = P.get(u, 0.0)
                if pu <= 0.0:
                    continue
                w = _edge_lik(G, u, v)
                prod *= (1.0 - pu * w)
            pv = 1.0 - prod
            pv = _clip(pv)
            delta = max(delta, abs(pv - P.get(v, 0.0)))
            P_new[v] = pv
        P = P_new
        if delta < tol:
            break

    return _clip(P.get(conv_id, 0.0))


# Coalition Helpers
# ---------------- Sequences / coalitions (overlay-native) ----------------



def _sequence_seeds(ctx: RCSPPRContext, candidate_personas: List[str]) -> Tuple[List[str], Optional[Tuple[str,str]]]:
    """
    Return (seeds, warm_from_key) for PR calls used by sequences/coalitions.
    Never seed with the conversion/product node. Prefer personas present in the engine.
    """
    eng = ctx.engine
    if ctx.seeds:
        # sanitize ctx.seeds as well
        seeds = [s for s in ctx.seeds if s in eng.node_index and s != ctx.conv_id]
        return seeds, ("", ctx.seeds_key)

    # fallback = a small, stable subset of candidate personas
    seeds_fb = [p for p in candidate_personas if p in eng.node_index and p != ctx.conv_id][:16]
    return seeds_fb, None



def _overlay_for_persona(engine, G_pruned: nx.DiGraph, persona_id: str, boost: float) -> Dict[Tuple[int,int], float]:
    """
    Lightweight persona overlay in *reversed* orientation, consistent with rcs_staged overlay builder.
    We keep it local to avoid importing overlay_helpers here.
    """
    ov: Dict[Tuple[int,int], float] = {}
    if persona_id not in G_pruned or persona_id not in engine.node_index:
        return ov

    i = engine.node_index[persona_id]
    for j in _jobs_of_persona(G_pruned, persona_id):
        if j in engine.node_index:
            k = engine.node_index[j]
            ov[(i, k)] = boost
    return ov
def _single_lift(ctx: RCSPPRContext, pid: str, *, boost: float,
                 candidate_personas: Optional[List[str]] = None) -> float:
    eng = ctx.engine
    conv_idx = eng.node_index[ctx.conv_id]
    seeds, _ = _sequence_seeds(ctx, candidate_personas or [pid])

    p0 = float(eng.pr_raw(seeds)[conv_idx])
    ov = _overlay_for_persona(eng, ctx.G_pruned, pid, boost)
    p1 = float(eng.pr_raw(seeds, boost_key=f"pers:{pid}", edge_scales=ov)[conv_idx])
    return max(0.0, p1 - p0)

def _simulate_sequence_lift(ctx: RCSPPRContext, sequence: List[str], boost: float = 2.0,
                            *, candidate_personas: Optional[List[str]] = None) -> Dict:
    eng = ctx.engine
    conv_idx = eng.node_index[ctx.conv_id]

    # choose seeds for this simulation
    seeds, warm_base = _sequence_seeds(ctx, candidate_personas or sequence)

    pr_base = eng.pr_raw(seeds)
    base_win = float(pr_base[conv_idx])

    steps = []
    prev_key = ""
    prev_win = base_win

    for i, pid in enumerate(sequence, 1):
        ov = _overlay_for_persona(eng, ctx.G_pruned, pid, boost)
        key = f"{prev_key}+pers:{pid}" if prev_key else f"pers:{pid}"
        pr_step = eng.pr_raw(
            seeds,
            boost_key=key,
            edge_scales=ov,
            warm_from=(prev_key, ctx.seeds_key) if warm_base and prev_key else None
        )
        win = float(pr_step[conv_idx])
        delta = win - prev_win
        if abs(delta) < EPS_PRINT: delta = 0.0
        if abs(win)   < EPS_PRINT: win   = 0.0

        steps.append({
            "step": i,
            "persona": pid,
            "delta_lift": max(0.0, float(delta)),
            "win_likelihood": float(win),
            "efforts": {}
        })
        prev_key = key
        prev_win = win

    total = prev_win - base_win
    if abs(total) < EPS_PRINT: total = 0.0
    return {
        "base_win": float(base_win),
        "final_win": float(prev_win),
        "total_lift": max(0.0, float(total)),
        "steps": steps
    }



def _coalition_synergy(ctx, personas: List[str], concerns_by_persona: Dict[str, List[Dict]]) -> float:
    """
    Compute synergy multiplier based on concern overlap + stage diversity.
    """
    if not concerns_by_persona:
        return 1.0

    concern_sets = [set(c["concern_id"] for c in concerns_by_persona.get(p, [])) for p in personas]
    overlap = len(set.intersection(*concern_sets)) if len(concern_sets) > 1 else 0
    diversity = len({c["stage"] for p in personas for c in concerns_by_persona.get(p, [])})

    overlap_factor = 1.0 + 0.1 * overlap         # +10% per shared concern
    diversity_factor = 1.0 + 0.05 * max(0, diversity - 1)  # +5% per extra unique stage
    return overlap_factor * diversity_factor

def _beam_search_sequences(ctx: RCSPPRContext, candidate_personas: List[str], *,
                           boost: float = 2.0, max_len: int = 5, beam_width: int = 5) -> List[Dict]:
    eng = ctx.engine
    conv_idx = eng.node_index[ctx.conv_id]

    # screen: reachable + non-zero *with* fallback seeds
    screened = []
    for pid in candidate_personas:
        if not _reachable(ctx, pid):
            continue
        L1 = _single_lift(ctx, pid, boost=boost, candidate_personas=candidate_personas)
        if L1 > EPS:
            screened.append((L1, pid))
    screened.sort(reverse=True)

    frontier = [[pid] for _, pid in screened[:beam_width]]
    best = [
        (float(_simulate_sequence_lift(ctx, [pid], boost=boost, candidate_personas=candidate_personas)["final_win"]), [pid])
        for _, pid in screened[:beam_width]
    ]

    for L in range(2, max_len + 1):
        new_frontier = []
        seen = set()
        # precompute seeds for this pool once
        seeds, warm_base = _sequence_seeds(ctx, candidate_personas)

        for seq in frontier:
            prev_key = ""
            for pid in seq:
                prev_key = f"{prev_key}+pers:{pid}" if prev_key else f"pers:{pid}"
            remaining = [p for _, p in screened if p not in seq]
            for nxt in remaining:
                cand = tuple(seq + [nxt])
                if cand in seen:
                    continue
                seen.add(cand)
                ov = _overlay_for_persona(eng, ctx.G_pruned, nxt, boost)
                key = f"{prev_key}+pers:{nxt}" if prev_key else f"pers:{nxt}"
                pr = eng.pr_raw(
                    seeds,
                    boost_key=key,
                    edge_scales=ov,
                    warm_from=(prev_key, ctx.seeds_key) if warm_base and prev_key else None
                )
                win = float(pr[conv_idx])
                if win <= EPS:
                    continue
                new_frontier.append((win, list(cand)))
        if not new_frontier:
            break
        new_frontier.sort(reverse=True, key=lambda x: x[0])
        frontier = [seq for _, seq in new_frontier[:beam_width]]
        best.extend(new_frontier[:beam_width])

    dedup = {}
    for win, seq in best:
        k = tuple(seq)
        sim = _simulate_sequence_lift(ctx, seq, boost=boost, candidate_personas=candidate_personas)
        rec = {
            "sequence": seq,
            "final_win": float(sim["final_win"]),
            "base_win": float(sim["base_win"]),
            "total_lift": float(sim["total_lift"]),
            "steps": sim["steps"],
        }
        if k not in dedup or rec["final_win"] > dedup[k]["final_win"]:
            dedup[k] = rec
    return sorted(dedup.values(), key=lambda r: r["final_win"], reverse=True)



def _coalition_lift(ctx: RCSPPRContext, coalition: Tuple[str, ...], boost: float = 2.0,
                    candidate_personas: Optional[List[str]] = None) -> float:
    eng = ctx.engine
    conv_idx = eng.node_index[ctx.conv_id]
    seeds, _ = _sequence_seeds(ctx, list(candidate_personas or list(coalition)))

    p0 = float(eng.pr_raw(seeds)[conv_idx])

    prev_key = ""
    for pid in coalition:
        ov = _overlay_for_persona(eng, ctx.G_pruned, pid, boost)
        key = f"{prev_key}+pers:{pid}" if prev_key else f"pers:{pid}"
        _ = eng.pr_raw(seeds, boost_key=key, edge_scales=ov,
                       warm_from=(prev_key, ctx.seeds_key) if prev_key else None
        )
        prev_key = key

    p1 = float(eng.pr_raw(seeds, boost_key=prev_key)[conv_idx])
    lift = p1 - p0
    return max(0.0, 0.0 if abs(lift) < EPS_PRINT else lift)


def _find_top_coalitions(ctx: RCSPPRContext, persona_pool: List[str], *, boost: float = 2.0,
                         max_coalition_size: int = 3, top_k: int = 8,
                         candidate_personas: Optional[List[str]] = None) -> List[Dict]:
    # Precompute individual lifts and screen
    indiv = {
        pid: _single_lift(ctx, pid, boost=boost, candidate_personas=candidate_personas or persona_pool)
        for pid in persona_pool if _reachable(ctx, pid)
    }
    candidates = [pid for pid, L in indiv.items() if L > EPS]
    rows: List[Tuple[float, Tuple[str,...]]] = []

    for r in range(2, min(max_coalition_size, len(candidates)) + 1):
        for subset in itertools.combinations(candidates, r):
            set_lift = _coalition_lift(ctx, subset, boost=boost,
                                       candidate_personas=candidate_personas or persona_pool)
            if set_lift <= EPS:
                continue
            # contribution test
            contributes = True
            for p in subset:
                rem = tuple(x for x in subset if x != p)
                rem_lift = _coalition_lift(ctx, rem, boost=boost,
                                           candidate_personas=candidate_personas or persona_pool) if len(rem) >= 1 else 0.0
                if (set_lift - rem_lift) <= EPS:
                    contributes = False
                    break
            if contributes:
                rows.append((set_lift, subset))

    best = heapq.nlargest(top_k, rows, key=lambda x: x[0])
    return [{"personas": list(sub), "lift": float(lft),
             "synergy": float(lft - sum(indiv[p] for p in sub))}
            for lft, sub in best]



def _sequence_coalitions(ctx: RCSPPRContext, sequence: List[str], *, boost: float = 2.0,
                         max_coalition_size: int = 3, min_synergy: float = 1e-6,
                         candidate_personas: Optional[List[str]] = None) -> List[Dict]:
    """Within a sequence, find subsets where joint lift > sum of individual lifts."""
    indiv = {}
    for pid in sequence:
        indiv[pid] = _single_lift(ctx, pid, boost=boost,
                                  candidate_personas=candidate_personas or sequence)

    coalitions = []
    for r in range(2, min(max_coalition_size, len(sequence)) + 1):
        for subset in itertools.combinations(sequence, r):
            set_lift = _coalition_lift(ctx, subset, boost=boost,
                                       candidate_personas=candidate_personas or sequence)
            sum_indiv = sum(indiv[p] for p in subset)
            synergy = max(0.0, set_lift - sum_indiv)
            if synergy >= min_synergy and set_lift > 0:
                coalitions.append({"personas": list(subset), "lift": float(set_lift), "synergy": float(synergy)})
    return sorted(coalitions, key=lambda c: (c["synergy"], c["lift"]), reverse=True)


#---- Concern Based Coalitions Logic

def _flat_concern_items(G: nx.DiGraph,
                        concerns_by_persona: Dict[str, List[Dict]],
                        persona_rows: Dict[str, Dict],
                        rel_map: Dict[Tuple[str,str], float]) -> List[Dict]:
    """
    Returns a flat list of concern items:
    { pid, cid, stage, label, rel, base_score, inv, act, care, lift_proxy }
    """
    items = []
    for pid, plist in concerns_by_persona.items():
        prow = persona_rows.get(pid, {})
        inv = float(prow.get("involvement", 0.0))
        act = float(prow.get("activation", 0.0))
        care = float(prow.get("care", 0.0))
        for c in plist:
            cid   = c["concern_id"]
            stage = c.get("stage", "problem")
            base  = float(c.get("potential_lift", c.get("score", 0.0)))
            rel   = float(rel_map.get((pid, cid), 0.0))
            # simple, tunable fuse: care dominates, then inv/act, then rel and base
            persona_factor = 0.5*care + 0.25*inv + 0.25*act
            concern_factor = 0.6*base + 0.4*rel
            lift_proxy = max(0.0, min(1.0, 1.0 - (1.0 - persona_factor)*(1.0 - concern_factor)))
            
            items.append({
                "pid": pid,
                "cid": cid,
                "stage": stage,
                "concern_label": c.get("concern_label"),
                "rel": rel,
                "base_score": base,
                "inv": inv, "act": act, "care": care,
                "lift_proxy": lift_proxy,
            })
    # sort high → low
    items.sort(key=lambda r: r["lift_proxy"], reverse=True)
    return items

def _jobs_of_concern(G: nx.DiGraph, cid: str) -> set:
    # concern can be a pain or solved-by job; grab adjacent jobs either way
    js = set(get_source_nodes_by_target_and_type(G, cid, "felt_in") or [])
    js.update(get_source_nodes_by_target_and_type(G, cid, "solves") or [])
    js.update(get_target_nodes_by_source_and_type(G, cid, "solves") or [])
    return {j for j in js if j in G}

def _concern_overlap(G: nx.DiGraph, c1: str, c2: str) -> float:
    J1, J2 = _jobs_of_concern(G, c1), _jobs_of_concern(G, c2)
    if not J1 or not J2: return 0.0
    inter = len(J1 & J2); uni = len(J1 | J2)
    return inter / max(1, uni)

def _stage_transfer(s_from: str, s_to: str) -> float:
    # heuristic: helping problem → boosts pain; pain → boosts solution; small otherwise
    s_from, s_to = (s_from or "problem"), (s_to or "problem")
    if s_from == "problem"  and s_to == "pain":     return 0.20
    if s_from == "pain"     and s_to == "solution": return 0.20
    if s_from == s_to:                                return 0.05
    return 0.02

def _concern_transfer(G: nx.DiGraph, i: Dict, j: Dict) -> float:
    # Combine structural overlap and stage pattern
    ov = _concern_overlap(G, i["cid"], j["cid"])
    st = _stage_transfer(i["stage"], j["stage"])
    # cross-persona gets a bit less transfer than same-persona
    persona_factor = 1.0 if i["pid"] == j["pid"] else 0.5
    return persona_factor * (0.5*ov + 0.5*st)


def _coalition_lift_concerns(items: List[Dict], base_win: float) -> float:
    # noisy-OR over lift_proxy, applied on top of base_win
    import math
    vals = [max(0.0, min(1.0, x["lift_proxy"])) for x in items]
    coalition = 1.0 - math.prod([1.0 - v for v in vals]) if vals else 0.0
    win = 1.0 - (1.0 - base_win) * (1.0 - coalition)
    return max(0.0, win - base_win)

def find_top_concern_coalitions(G: nx.DiGraph,
                                flat_items: List[Dict],
                                base_win: float,
                                max_size: int = 3,
                                top_k: int = 8) -> List[Dict]:
    candidates = flat_items[:40]
    rows = []
    for r in range(2, min(max_size, len(candidates)) + 1):
        for subset in itertools.combinations(candidates, r):
            L = _coalition_lift_concerns(list(subset), base_win)
            if L <= 1e-9:
                continue
            sum_solo = sum(_coalition_lift_concerns([x], base_win) for x in subset)
            bonus = 0.0
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    bonus += 0.1 * _concern_overlap(G, subset[i]["cid"], subset[j]["cid"])
                    bonus += 0.1 * _stage_transfer(subset[i]["stage"], subset[j]["stage"])

            # Use combined lift = L + bonus for both ranking AND reporting
            L_combined = max(0.0, min(1.0, L + bonus))
            synergy = bonus / len(subset)


            rows.append((L_combined, synergy, list(subset)))

    rows.sort(reverse=True, key=lambda x: (x[0], x[1]))
    out = []
    for Lc, Sy, subset in rows[:top_k]:
        out.append({
            "concerns": [
                {"persona": x["pid"], "concern_id": x["cid"], "stage": x["stage"],
                 "concern_label": x["concern_label"], "lift_proxy": float(x["lift_proxy"])}
                for x in subset
            ],
            "lift": float(Lc),
            "synergy": float(Sy),
        })
    return out



def simulate_concern_sequence(G: nx.DiGraph,
                              seq: List[Dict],
                              base_win: float) -> Dict:
    win = base_win
    steps = []
    seen = []
    for i, x in enumerate(seq, 1):
        # transfer from all prior steps
        t_boost = sum(_concern_transfer(G, p, x) for p in seen)
        eff = x["lift_proxy"] * (1.0 + t_boost)
        eff = max(0.0, min(1.0, eff))
        new_win = 1.0 - (1.0 - win) * (1.0 - eff)
        steps.append({
            "step": i,
            "persona": x["pid"],
            "concern_id": x["cid"],
            "stage": x["stage"],
            "concern_label": x["concern_label"],
            "delta_lift": float(max(0.0, new_win - win)),
            "win_likelihood": float(new_win),
            "efforts": {},
        })
        win = new_win
        seen.append(x)
    return {
        "base_win": float(base_win),
        "final_win": float(win),
        "total_lift": float(max(0.0, win - base_win)),
        "steps": steps
    }

def beam_concern_sequences(G: nx.DiGraph,
                           flat_items: List[Dict],
                           base_win: float,
                           max_len: int = 5,
                           beam_width: int = 5) -> List[Dict]:
    # seed by best singletons
    seeds = []
    for x in flat_items[:30]:
        sim = simulate_concern_sequence(G, [x], base_win)
        if sim["total_lift"] > 1e-9:
            seeds.append((sim["final_win"], [x]))
    seeds.sort(reverse=True, key=lambda t: t[0])
    frontier = [seq for _, seq in seeds[:beam_width]]
    best = seeds[:beam_width]

    import itertools
    for L in range(2, max_len + 1):
        new_frontier = []
        seen = set()
        for seq in frontier:
            cur = simulate_concern_sequence(G, seq, base_win)
            cur_win = cur["final_win"]
            remaining = [x for x in flat_items if x not in seq]
            for nxt in remaining:
                cand = tuple((y["pid"], y["cid"]) for y in (seq + [nxt]))
                if cand in seen:
                    continue
                seen.add(cand)
                # quick upper bound: apply transfer but don’t fully simulate
                t_boost = sum(_concern_transfer(G, y, nxt) for y in seq)
                eff = nxt["lift_proxy"] * (1.0 + t_boost)
                eff = max(0.0, min(1.0, eff))
                est_win = 1.0 - (1.0 - cur_win) * (1.0 - eff)
                if est_win <= cur_win + 1e-9:
                    continue
                new_frontier.append((est_win, seq + [nxt]))
        if not new_frontier:
            break
        new_frontier.sort(reverse=True, key=lambda x: x[0])
        frontier = [seq for _, seq in new_frontier[:beam_width]]
        best.extend(new_frontier[:beam_width])

    # materialize & dedup
    out = {}
    for _, seq in best:
        key = tuple((x["pid"], x["cid"]) for x in seq)
        sim = simulate_concern_sequence(G, seq, base_win)
        rec = {
            "sequence": [{"persona": x["pid"], "concern_id": x["cid"], "stage": x["stage"], "concern_label": x["concern_label"]} for x in seq],
            "final_win": sim["final_win"],
            "base_win": sim["base_win"],
            "total_lift": sim["total_lift"],
            "steps": sim["steps"],
        }
        if key not in out or rec["final_win"] > out[key]["final_win"]:
            out[key] = rec
    return sorted(out.values(), key=lambda r: r["final_win"], reverse=True)

# -----------------------------------------------
# Theme clustering helpers (stable strategy layer)
# -----------------------------------------------


def _theme_similarity(G: nx.DiGraph, c1: Dict, c2: Dict) -> float:
    """
    Soft similarity for concerns (used to cluster themes):
    - structural overlap across adjacent jobs
    - stage transfer adjacency (problem -> pain -> solution)
    """
    def jobs(cid: str) -> set:
        js = set(get_source_nodes_by_target_and_type(G, cid, "felt_in") or [])
        js.update(get_source_nodes_by_target_and_type(G, cid, "solves") or [])
        js.update(get_target_nodes_by_source_and_type(G, cid, "solves") or [])
        return {j for j in js if j in G}

    s_overlap = 0.0
    J1, J2 = jobs(c1["concern_id"]), jobs(c2["concern_id"])
    if J1 and J2:
        s_overlap = len(J1 & J2) / max(1, len(J1 | J2))

    def stage_adj(a: str, b: str) -> float:
        a = (a or "problem"); b = (b or "problem")
        if a == b: return 0.2
        if a == "problem" and b == "pain": return 0.6
        if a == "pain" and b == "solution": return 0.6
        return 0.1

    s_stage = stage_adj(c1.get("stage"), c2.get("stage"))
    # blend; tuned to prefer structural adjacency
    return 0.65 * s_overlap + 0.35 * s_stage

def _cluster_themes(G: nx.DiGraph, flat_items: List[Dict], sim_thresh: float = 0.35, max_themes: int = 5):
    """
    Very simple agglomerative clustering by greedy linking on _theme_similarity.
    Returns list of themes with representative label and seed concerns.
    """
    items = flat_items[:60]  # cap for speed
    # Start themes by picking top concerns as seeds, then attach similar concerns.
    seeds = []
    used = set()
    for x in items:
        if len(seeds) >= max_themes: break
        if (x["pid"], x["cid"]) in used: continue
        theme = [x]
        used.add((x["pid"], x["cid"]))
        for y in items:
            if (y["pid"], y["cid"]) in used: continue
            if _theme_similarity(G, {"concern_id": x["cid"], "stage": x["stage"]},
                                    {"concern_id": y["cid"], "stage": y["stage"]}) >= sim_thresh:
                theme.append(y)
                used.add((y["pid"], y["cid"]))
        seeds.append(theme)

    # Build output
    out = []
    for t in seeds:
        # representative = most lift_proxy
        rep = max(t, key=lambda z: z["lift_proxy"])
        out.append({
            "theme_id": f"theme:{rep['cid']}",
            "label": rep.get("concern_label") or rep["cid"],
            "seed_concerns": [
                {"persona": r["pid"], "concern_id": r["cid"], "label": r.get("concern_label"), "stage": r["stage"],
                 "lift_proxy": float(r["lift_proxy"])}
                for r in sorted(t, key=lambda z: z["lift_proxy"], reverse=True)[:8]
            ],
            "personas": sorted(list({r["pid"] for r in t})),
            "predicted_effectiveness": float(max(r["lift_proxy"] for r in t)),
        })
    return out

# -----------------------------------------------
# Persona buckets & portfolio templates
# -----------------------------------------------
def _bucket_persona(inv: float, act: float) -> str:
    # z-score not required for stable bins; use simple quadrants
    if inv >= 0.5 and act >= 0.5: return "Potential Champions"
    if inv >= 0.5 and act <  0.5: return "Operators"
    if inv <  0.5 and act >= 0.5: return "Blockers"
    return "Passive Influencers"

def _default_portfolio_for_state(state: str) -> Dict[str, float]:
    """
    Fixed mix templates you described:
    - COLD: Breadth 60–70%, Mutation 20–30%, Depth 10–20%
    - WARM: Breadth 30–40%, Mutation 10–20%, Depth 40–60%
    - HOT:  Breadth 15–25%, Mutation 5–10%,  Depth 65–80%
    """
    if state == "COLD": return {"breadth": 0.65, "mutation": 0.20, "depth": 0.15}
    if state == "WARM": return {"breadth": 0.35, "mutation": 0.15, "depth": 0.50}
    return {"breadth": 0.20, "mutation": 0.10, "depth": 0.70}  # HOT
# -----------------------------------------------
# Stable strategy: build_account_strategy(...)
# -----------------------------------------------
def build_account_strategy(
    G: nx.DiGraph,
    *,
    engaged_nodes: Optional[List[Dict]] = None,
    top_concerns_per_persona: int = 5,
    top_personas: int = 20,
    boost_factor: float = 2.0,
) -> Dict:
    """
    Run once at account onboarding (or explicit reset).
    Freezes the macro structure: themes, sequences, coalitions, persona buckets, baseline portfolio.
    """
    # Reuse your generator to compute everything once
    Gp, report = generate_rcs(
        G,
        engaged_nodes=engaged_nodes,
        top_concerns_per_persona=top_concerns_per_persona,
        top_personas=top_personas,
        plays_per_concern=3,
        boost_factor=boost_factor,
    )

    base_win = float(
        report.get("baseline", {}).get("win_likelihood", 0.0)
        or 0.0
    )

    # Persona buckets
    buckets = []
    for grp in ("by_involvement", "by_activation", "by_marginal_lift"):
        for r in report["top_personas"].get(grp, []):
            buckets.append({
                "id": r["id"],
                "involvement": float(r.get("involvement", 0.0)),
                "activation":  float(r.get("activation", 0.0)),
                "care":        float(r.get("care", 0.0)),
            })
    # dedup by id
    dedup = {}
    for b in buckets:
        i = b["id"]
        if i not in dedup or (b["involvement"] + b["activation"]) > (dedup[i]["involvement"] + dedup[i]["activation"]):
            dedup[i] = b
    persona_buckets = [
        {"id": i, **dedup[i], "bucket": _bucket_persona(dedup[i]["involvement"], dedup[i]["activation"])}
        for i in dedup
    ]

    # Freeze themes from concern backlog (already prioritized)
    flat_items = report.get("concern_backlog", [])  # already lift-weighted
    themes = _cluster_themes(Gp, flat_items, sim_thresh=0.35, max_themes=5)

    # Choose best concern sequences (stable arcs) from your “concern_sequences”
    arcs = report.get("concern_sequences", [])[:3]  # cap for stability in execution

    # Initial portfolio state = COLD (unless you want to infer from base_win)
    portfolio_policy = {
        "state": "COLD",
        "mix": _default_portfolio_for_state("COLD"),
        "notes": "Shift to WARM/HOT on engagement milestones; strategy stays fixed."
    }

    # Pack strategy object (this is what you persist)
    strategy = {
        "graph_meta": {
            "base_win": base_win,
            "product_id": get_product_id_from_subgraph(Gp),
            "nodes": len(Gp),
        },
        "themes": themes,
        "frozen_theme_ids": [t["theme_id"] for t in themes],
        "frozen_arcs": arcs,                     # ordered concern arcs you’ll keep
        "persona_buckets": persona_buckets,      # Champions/Operators/Blockers/Passive
        "portfolio_policy": portfolio_policy,    # Breadth/Depth/Mutation mix
        "frozen_at": "__now__",                  # stamp in your caller
    }
    return strategy
# -----------------------------------------------
# Tactical updates: update_tactical_plan(...)
# -----------------------------------------------
def _score_within_theme(item: Dict, I: float, A: float, Care: float) -> float:
    """
    Keeps your original logic spirit: persona factor blended with concern lift.
    """
    persona_factor = 0.5*Care + 0.25*I + 0.25*A
    concern_factor = 0.6*item.get("lift_proxy", 0.0) + 0.4*item.get("rel", 0.0)
    from backend.utils.inference.rcs_generators.graph_algorithms import clip01
    return clip01(1.0 - (1.0 - persona_factor)*(1.0 - concern_factor))

def update_tactical_plan(
    G: nx.DiGraph,
    strategy: Dict,
    *,
    engaged_nodes: Optional[List[Dict]] = None,
    plays_per_concern: int = 3,
    boost_factor: float = 2.0,
) -> Dict:
    """
    Recomputes persona I/A/Care and *re-ranks inside frozen themes* only.
    Does not add/remove themes.
    """
    # Recompute fast context with new engaged nodes
    ctx, pre = rcs_prepare(G, engaged_nodes=engaged_nodes)
    Gp = ctx.G_pruned
    prod_id = get_product_id_from_subgraph(Gp)

    # Persona rows (I/A/Care)
    pr_post = ctx.baseline_dbg.get("weighted_pr0") or {}
    pr_prior = ctx.baseline_dbg.get("unconditioned_pr0") or {}
    persona_ids_all = get_nodes_list_ids(Gp, "persona", {})
    rows = []
    for pid in persona_ids_all:
        jobs = _jobs_of_persona(Gp, pid)
        inv_terms, act_terms = [], []
        for j in jobs:
            job_lk = float(Gp.nodes[j].get("likelihood", pr_post.get(j, 0.0)))
            inv_j = job_lk
            act_j = compute_activation_from_uplift(Gp, pr_prior, pr_post, focus_types=("job",), mode="ratio_bounded").get(j, 0.0)
            rel_jp = _rel_job_to_persona(Gp, j, pid)
            inv_terms.append(inv_j * rel_jp)
            act_terms.append(act_j * rel_jp)
        inv_noisy = 1.0 - np.prod([1.0 - t for t in inv_terms]) if inv_terms else 0.0
        act_noisy = 1.0 - np.prod([1.0 - t for t in act_terms]) if act_terms else 0.0
        persona_lk = float(Gp.nodes[pid].get("likelihood", 0.0))
        inv_p = max(persona_lk, inv_noisy)
        act_p = max(persona_lk, act_noisy)
        job_FJs = [float(Gp.nodes[j].get("likelihood", pr_post.get(j, 0.0))) for j in jobs]
        S_felt = float(np.mean(sorted(job_FJs, reverse=True)[:2])) if job_FJs else 0.0
        care = (max(1e-9, act_p) * max(1e-9, S_felt)) ** 0.5
        rows.append({"id": pid, "involvement": _clip(inv_p), "activation": _clip(act_p), "care": _clip(care)})

    I_map = {r["id"]: r["involvement"] for r in rows}
    A_map = {r["id"]: r["activation"] for r in rows}
    Care_map = {r["id"]: r["care"] for r in rows}

    # Build a quick lookup from persona->concerns (from frozen themes)
    frozen_theme_ids = set(strategy.get("frozen_theme_ids", []))
    themes = [t for t in strategy.get("themes", []) if t["theme_id"] in frozen_theme_ids]

    # Rehydrate concern rows we need to score
    frozen_concern_keys = set()
    for t in themes:
        for sc in t["seed_concerns"]:
            frozen_concern_keys.add((sc["persona"], sc["concern_id"]))

    # For those pairs, gather rel, stage, plays (fresh)
    rel_map = {}
    def _rel_persona_concern(pid, cid):
        return _rel_persona_to_concern(Gp, pid, cid)
    concern_rows = []
    for (pid, cid) in frozen_concern_keys:
        rel = _rel_persona_concern(pid, cid)
        stage = _infer_stage_for_concern(Gp, pid, cid)
        concern_rows.append({
            "pid": pid,
            "cid": cid,
            "stage": stage,
            "rel": float(rel),
            "concern_label": _set_node_label(Gp, cid),
        })

    # Score inside themes with new I/A/Care
    scored_by_theme = []
    for t in themes:
        entries = []
        for sc in t["seed_concerns"]:
            pid, cid = sc["persona"], sc["concern_id"]
            r = next((x for x in concern_rows if x["pid"] == pid and x["cid"] == cid), None)
            if not r: continue
            I, A, C = I_map.get(pid, 0.0), A_map.get(pid, 0.0), Care_map.get(pid, 0.0)
            # Use previous lift_proxy as prior if available; else fallback to rel
            prior_lift = float(sc.get("lift_proxy", 0.0)) or 0.0
            item = {**r, "lift_proxy": prior_lift}
            score = _score_within_theme(item, I, A, C)
            # fresh plays
            plays = get_best_plays_for_concern(
                persona_alias={"title": get_node_by_id(Gp, pid).get("title", pid)},
                concern_stage=r["stage"],
                product_id=prod_id,
                archetype={},
                top_k=plays_per_concern,
            ) or []
            entries.append({
                "persona": pid,
                "persona_label": _set_node_label(Gp, pid),
                "concern_id": cid,
                "concern_label": r["concern_label"],
                "stage": r["stage"],
                "priority": float(score),
                "plays": [
                    {
                        "asset_id": p["asset"]["id"],
                        "asset_name": p["asset"]["name"],
                        "channel": p.get("channel"),
                        "fitness": p.get("fitness", 0.0),
                        "why": p.get("why", ""),
                    } for p in plays
                ] if plays else [{"note": "No existing plays match. Generate a new play."}],
            })
        entries.sort(key=lambda e: e["priority"], reverse=True)
        scored_by_theme.append({
            "theme_id": t["theme_id"],
            "label": t["label"],
            "entries": entries,
        })

    # Portfolio state transitions: adjust only the MIX, keep themes the same
    # Simple heuristic: use % of personas with activation >= 0.5 as engagement proxy.
    act_vals = [A_map.get(p["id"], 0.0) for p in rows]
    engaged_ratio = sum(1 for a in act_vals if a >= 0.5) / max(1, len(act_vals))
    if engaged_ratio < 0.15:
        state = "COLD"
    elif engaged_ratio < 0.45:
        state = "WARM"
    else:
        state = "HOT"

    tactical = {
        "portfolio_policy": {"state": state, "mix": _default_portfolio_for_state(state)},
        "themes_ranked": sorted(scored_by_theme, key=lambda t: (max((e["priority"] for e in t["entries"]), default=0.0)), reverse=True),
        "engagement": {"engaged_ratio": float(engaged_ratio), "personas_scored": len(rows)},
        "refreshed_at": "__now__",
    }
    return tactical




# =========================================================
# -------------------- Public: generate_rcs ---------------
# =========================================================

def generate_rcs(
    G: nx.DiGraph,
    *,
    engaged_nodes: Optional[List[Dict]] = None,
    rel_min: float = 0.0,
    max_nodes: int = 5000,
    # Execution knobs (kept for API parity)
    top_concerns_per_persona: int = 5,
    top_personas: int = 20,
    plays_per_concern: int = 3,
    alpha_act=(0.0, 1.0, 1.0, 1.0),
    beta_felt=(0.0, 1.0, 1.0, 1.0),
    boost_factor: float = 2.0,
) -> Tuple[nx.DiGraph, Dict]:
    """
    Fast, archetype-free RCS with full parity to legacy report keys.
    """
    engaged_nodes = engaged_nodes or []
    print("Running fast rcs with:", engaged_nodes)
    # Stage 0: prepare context (prune, apply engaged boosts, build PPR engine)
    ctx, pre = rcs_prepare(G, [])
    
    # 1) Gather sets
    pain_triggers = get_nodes_list_ids(G, "pain_trigger", {})
    hard_trigs = [n["id"] for n in engaged_nodes if G.nodes[n.get("id")].get("node_type") == "pain_trigger"]
    hard_pains  = [n["id"] for n in engaged_nodes if G.nodes[n.get("id")].get("node_type") == "pain"]
    hard_jobs   = [n["id"] for n in engaged_nodes if G.nodes[n.get("id")].get("node_type") == "job"]
    hard_pers   = [n["id"] for n in engaged_nodes if G.nodes[n.get("id")].get("node_type") == "persona"]


    # 2) Build one reach map for everything we might include as atoms
    seed_nodes_for_reach = set(pain_triggers) | set(hard_trigs) | set(hard_pains) | set(hard_jobs) | set(hard_pers)
    reach_map = reverse_reach_to_product_bulk(ctx.G_pruned, product_id=ctx.conv_id, nodes=seed_nodes_for_reach)

    # 3) Attribute features (already computed earlier)
    dim_weights = pre.get("baseline", {}).get("dimension_weights_learned", {}) or {}

    # 4) Final GraphWin
    base_win = compute_graphwin(
        conv_id=ctx.conv_id,
        pain_triggers=pain_triggers,
        dim_weights=dim_weights,
        beta0=0.0,               # will learn later from CRM
        betas={},                # per-trigger/per-feature betas later
        baseline_prior=0.0,      # strict “no data => 0”
        hard_on_triggers=hard_trigs,
        hard_on_pains=hard_pains,
        hard_on_jobs=hard_jobs,
        hard_on_personas=hard_pers,
        reach_to_product=reach_map,
    )


    baseline_block = {
        "win_likelihood": float(base_win),
        "note": "GraphWin = noisyOR(trigger_prior × reverse-PPR reach). Baseline=0.",
    }

    print("Baseline block built with win rate:", base_win)
    G_pruned = ctx.G_pruned
    print(f"Pruned graph: {len(G_pruned)} nodes")

    # Personas universe (post-prune)
    persona_ids_all = get_nodes_list_ids(G_pruned, "persona", {})

    # PR for involvement/attention (dict form)
    pr_post = ctx.baseline_dbg.get("weighted_pr0") or {}
    pr_prior = ctx.baseline_dbg.get("unconditioned_pr0") or {}


    # Involvement / activation / care
    rows = []
    for pid in persona_ids_all:
        jobs = _jobs_of_persona(G_pruned, pid)

        inv_terms, act_terms = [], []
        for j in jobs:
            # evidence-aware likelihood
            job_lk = float(G_pruned.nodes[j].get("likelihood", pr_post.get(j, 0.0)))
            inv_j = job_lk
            act_j = compute_activation_from_uplift(
                G_pruned, pr_prior, pr_post, focus_types=("job",), mode="ratio_bounded"
            ).get(j, 0.0)

            rel_jp = _rel_job_to_persona(G_pruned, j, pid)
            inv_terms.append(inv_j * rel_jp)
            act_terms.append(act_j * rel_jp)

        inv_noisy = 1.0 - np.prod([1.0 - t for t in inv_terms]) if inv_terms else 0.0
        act_noisy = 1.0 - np.prod([1.0 - t for t in act_terms]) if act_terms else 0.0

        # evidence override: persona itself
        persona_lk = float(G_pruned.nodes[pid].get("likelihood", 0.0))
        inv_p = max(persona_lk, inv_noisy)
        act_p = max(persona_lk, act_noisy)

        # care proxy = sqrt(act * felt top jobs)
        job_FJs = [float(G_pruned.nodes[j].get("likelihood", pr_post.get(j, 0.0))) for j in jobs]
        S_felt = float(np.mean(sorted(job_FJs, reverse=True)[:2])) if job_FJs else 0.0
        care = (max(1e-9, act_p) * max(1e-9, S_felt)) ** 0.5

        rows.append({
            "id": pid,
            "involvement": _clip(inv_p),
            "activation": _clip(act_p),
            "care": _clip(care),
            "breakdown": {"jobs": jobs},
        })



    # (4) Optional: Corridor-only involvement/activation
    """
    try:
        corr_post, corr_prior = ctx.baseline_dbg.get("corridor_post_pr"), ctx.baseline_dbg.get("corridor_prior_pr")
        if corr_post and corr_prior:
            for row in rows:
                pid = row["id"]
                inv_corr = _persona_involvement_from_jobs(G_pruned, corr_post, pid)
                act_corr = compute_activation_from_uplift(
                    G_pruned, corr_prior, corr_post,
                    focus_types=("persona",), mode="ratio_bounded"
                ).get(pid, 0.0)

                row["corridor_involvement"] = _clip(inv_corr)
                row["corridor_activation"] = _clip(act_corr)
    except Exception as e:
        print("[warn] corridor-only involvement skipped:", e)
    """

    topI_all = sorted(rows, key=lambda r: r["involvement"], reverse=True)
    topA_all = sorted(rows, key=lambda r: r["activation"],   reverse=True)
    topI = topI_all[:top_personas]
    topA = topA_all[:top_personas]

    # Stage 1: marginal lifts (fast)
    # Use the pool from top involvement to be consistent with legacy
    pool_for_lift = [r["id"] for r in topI]
    lifts = rcs_marginal_lifts(ctx, pool_for_lift, boost=boost_factor)
    # decorate with involvement/activation/care like legacy
    id2row = {r["id"]: r for r in rows}
    top_lift = [{
        "id": r["id"],
        "marginal_lift": float(r["marginal_lift"]),
        "involvement": float(id2row.get(r["id"], {}).get("involvement", 0.0)),
        "activation": float(id2row.get(r["id"], {}).get("activation", 0.0)),
        "care": float(id2row.get(r["id"], {}).get("care", 0.0)),
    } for r in lifts][:top_personas]

    top_personas_block = {
        "by_involvement": [{"id": r["id"], "involvement": float(r["involvement"]),
                            "activation": float(r["activation"]), "care": float(r["care"]), "label": _set_node_label(G_pruned, r["id"])} for r in topI],
        "by_activation":  [{"id": r["id"], "activation": float(r["activation"]),
                            "involvement": float(r["involvement"]), "care": float(r["care"]), "label": _set_node_label(G_pruned, r["id"])} for r in topA],
        "by_marginal_lift": top_lift,
    }

    # Stage 2: Concerns + assets (CF-based, fast overlays)
    concerns_by_persona: Dict[str, List[Dict]] = {}
    assets_by_persona: Dict[str, Dict[str, List[Dict]]] = {}
    concerns_flat: List[Dict] = []

    # Build persona->concerns and rel map
    persona_to_concerns: Dict[str, List[str]] = {}
    rel_map: Dict[Tuple[str, str], float] = {}
    for r in topI:
        pid = r["id"]
        pains = set()
        for j in _jobs_of_persona(G_pruned, pid):
            pains.update(_felt_pains_of_job(G_pruned, j))
            pains.update(_solve_pains_of_job(G_pruned, j))
        persona_to_concerns[pid] = list(pains)
        for cid in pains:
            rel_map[(pid, cid)] = _rel_persona_to_concern(G_pruned, pid, cid)

    I_map = {r["id"]: float(r["involvement"]) for r in rows}
    Care_map = {r["id"]: float(r["care"]) for r in rows}

    # compute potentials (fast)
    potentials = rcs_concern_potentials(
        ctx,
        persona_to_concerns=persona_to_concerns,
        rel_map=rel_map,
        I_map=I_map,
        Care_map=Care_map,
        boost=boost_factor,
        top_k=top_concerns_per_persona,
    )

    # map to report + assets
    prod_id = get_product_id_from_subgraph(G_pruned)
    for pid, cons in potentials.items():
        persona_node = get_node_by_id(G_pruned, pid) or {}
        persona_meta = {
            "title": persona_node.get("title") or persona_node.get("name") or pid,
            "department": persona_node.get("department") or "General",
            "seniority": persona_node.get("seniority") or "Manager",
        }
        concerns_by_persona[pid] = []
        assets_by_persona[pid] = {}

        for c in cons:
            cid = c["concern_id"]
            stage = _infer_stage_for_concern(G_pruned, pid, cid)

            plays = get_best_plays_for_concern(
                persona_alias=persona_meta,
                concern_stage=stage,
                product_id=prod_id,
                archetype={},       # archetypes removed
                top_k=plays_per_concern,
            )

            if plays:
                top_play = plays[0]
                asset_reco = {
                    "asset_id":   top_play["asset"]["id"],
                    "asset_name": top_play["asset"]["name"],
                    "channel":    top_play.get("channel"),
                    "fitness":    top_play.get("fitness", 0.0),
                    "why":        top_play.get("why", ""),
                }
            else:
                asset_reco = {"note": "No existing plays match. Generate a new play."}

            concerns_by_persona[pid].append({
                "concern_id": cid,
                "concern_label": _set_node_label(G_pruned, cid),
                "potential_lift": float(c.get("score", c.get("potential_lift", 0.0))),
                "cf_lift": float(c.get("potential_lift", 0.0)),
                "elasticity": float(c.get("elasticity_proxy", 0.0)),  # may be absent from staged calc; harmless
                "rel": float(c.get("rel", 0.0)),
                "stage": stage,
                "stage_explainer": _stage_explainer(stage),
            })

            assets_by_persona[pid][cid] = (
                [
                    {
                        "asset_id": p["asset"]["id"],
                        "asset_name": p["asset"]["name"],
                        "channel": p.get("channel"),
                        "fitness": p.get("fitness", 0.0),
                        "why": p.get("why", ""),
                    }
                    for p in plays
                ]
                if plays else
                [asset_reco]
            )

            concerns_flat.append({
                "persona_id": pid,
                "persona_label": _set_node_label(G_pruned, pid),
                "concern_id": cid,
                "concern_label": _set_node_label(G_pruned, cid),
                "stage": stage,
                "stage_explainer": _stage_explainer(stage),
                "potential_lift": float(c.get("potential_lift", 0.0)),
                "rel": float(c.get("rel", 0.0)),
                "asset_reco": asset_reco,
            })

    concerns_flat.sort(key=lambda r: r["potential_lift"], reverse=True)
    
        # ---------------- Concern-centric backlog, coalitions, sequences ----------------
    # Build persona_rows -> used by the concern helpers
    persona_rows = {
        r["id"]: {
            "involvement": float(r["involvement"]),
            "activation":  float(r["activation"]),
            "care":        float(r["care"]),
            "label":      _set_node_label(G_pruned, r["id"]),
        }
        for r in rows
    }


    # rel_map already computed above; concerns_by_persona is ready.
    # Flatten (persona, concern) items and compute lift proxies
    flat_items = _flat_concern_items(
        G_pruned,
        concerns_by_persona=concerns_by_persona,
        persona_rows=persona_rows,
        rel_map=rel_map,
    )

    # You may want to keep just the top backlog visible
    concern_backlog = flat_items[:30]

    # Concern coalitions (simultaneous)
    concern_coalitions = find_top_concern_coalitions(
        G_pruned,
        flat_items=concern_backlog,
        base_win=base_win,
        max_size=3,
        top_k=8,
    )

    # Concern sequences (ordered)
    concern_sequences = beam_concern_sequences(
        G_pruned,
        flat_items=concern_backlog,
        base_win=base_win,
        max_len=5,
        beam_width=5,
    )



    # Stage 3: Coalitions and causal flows (pairs & chains)
    print("Starting coalitions and causal flows logic")
    
    coalition_pool_ids = list({*(r["id"] for r in topI[:10]), *(r["id"] for r in topA[:10])})
    print("Coalition pool:", coalition_pool_ids)
    # coalitions
    coalitions = _find_top_coalitions(
        ctx,
        coalition_pool_ids,
        boost=boost_factor,
        max_coalition_size=3,
        top_k=8,
        candidate_personas=coalition_pool_ids,
    )

    coalitions_out = []
    for row in coalitions:
        personas = row["personas"]
        # merge concerns across personas (same policy as legacy)
        seen: Dict[str, Dict] = {}
        for pid in personas:
            for c in (concerns_by_persona.get(pid) or [])[:3]:
                cid = c["concern_id"]
                if cid not in seen or c.get("potential_lift", 0.0) > seen[cid].get("potential_lift", 0.0):
                    seen[cid] = c
        merged = sorted(seen.values(), key=lambda x: x.get("potential_lift", 0.0), reverse=True)[:8]
        coalitions_out.append({**row, "concerns": merged})

    # pair effects then chains
    pair_effects = rcs_pair_effects(ctx, coalition_pool_ids, boost=boost_factor)

    # build causal flows report (pairs)
    def _pair_score(meta):
        # If you add burden/effort later, blend here; staged currently returns lift_gain_for_v
        return float(meta.get("lift_gain_for_v", 0.0))

    pair_rows = []
    for (u, v), meta in pair_effects.items():
        s = _pair_score(meta)
        if s <= 1e-9:
            continue
        pair_rows.append((s, u, v, meta))
    top_pairs_rows = heapq.nlargest(8, pair_rows, key=lambda x: x[0])

    pairs_out = []
    for score, u, v, meta in top_pairs_rows:
        def _merge_for(personas: List[str], max_per_persona=3, max_total=3):
            seen: Dict[str, Dict] = {}
            for pid in personas:
                for c in (concerns_by_persona.get(pid) or [])[:max_per_persona]:
                    cid = c["concern_id"]
                    if cid not in seen or c.get("potential_lift", 0.0) > seen[cid].get("potential_lift", 0.0):
                        seen[cid] = c
            return sorted(seen.values(), key=lambda r: r.get("potential_lift", 0.0), reverse=True)[:max_total]

        pairs_out.append({
            "sequence": [u, v],
            "labeled_sequence": [_set_node_label(G_pruned,u), _set_node_label(G_pruned, v)],
            "score": float(score),
            "burden_reduction_for_v": float(meta.get("burden_reduction", 0.0)),  # 0 if not computed
            "lift_gain_for_v": float(meta.get("lift_gain_for_v", 0.0)),
            "effort_v_before": float(meta.get("effort_v_baseline", 0.0)),
            "effort_v_after_u": float(meta.get("effort_v_after_u", 0.0)),
            "concerns_to_address": {u: _merge_for([u]), v: _merge_for([v])},
        })

    # greedy 3-node chains based on pair scores (u->v and v->w)
    chain_rows = []
    used = set()
    for score_uv, u, v, meta_uv in top_pairs_rows:
        best_w = None
        best_s = 0.0
        for w in coalition_pool_ids:
            if w in (u, v):
                continue
            meta_vw = pair_effects.get((v, w))
            if not meta_vw:
                continue
            s = _pair_score(meta_vw)
            if s > best_s:
                best_s = s
                best_w = w
        if best_w and (u, v, best_w) not in used:
            used.add((u, v, best_w))
            chain_rows.append((score_uv + best_s, u, v, best_w))
    top_chain_rows = heapq.nlargest(5, chain_rows, key=lambda x: x[0])

    chains_out = []
    for score, u, v, w in top_chain_rows:
        # per-persona concerns (2 each) and merged view (up to 6)
        def _merge_for(pid_list: List[str], mpp=2, mt=6):
            seen: Dict[str, Dict] = {}
            for pid in pid_list:
                for c in (concerns_by_persona.get(pid) or [])[:mpp]:
                    cid = c["concern_id"]
                    if cid not in seen or c.get("potential_lift", 0.0) > seen[cid].get("potential_lift", 0.0):
                        seen[cid] = c
            return sorted(seen.values(), key=lambda r: r.get("potential_lift", 0.0), reverse=True)[:mt]
        chains_out.append({
            "sequence": [u, v, w],
            "score": float(score),
            "concerns_to_address": {
                u: _merge_for([u], mpp=2, mt=2),
                v: _merge_for([v], mpp=2, mt=2),
                w: _merge_for([w], mpp=2, mt=2),
            },
            "merged_concerns": _merge_for([u, v, w], mpp=2, mt=6),
        })

    causal_flows = {"pairs": pairs_out, "chains": chains_out}

    # Stage 4: Sequence design & campaign stitching
    seq_pool_ids = coalition_pool_ids
    seqs = _beam_search_sequences(ctx, seq_pool_ids, boost=boost_factor, max_len=5, beam_width=5)


    def _concern_gaps_for_sequence(sequence: List[str], max_concerns_per_persona: int = 3, fitness_min: float = 0.25):
        out = {}
        for pid in sequence:
            ranked = sorted((concerns_by_persona.get(pid) or []), key=lambda r: r.get("potential_lift", 0.0), reverse=True)
            topc = ranked[:max_concerns_per_persona]
            gaps, covered = [], []
            for c in topc:
                cid = c["concern_id"]
                c["concern_label"] = _set_node_label(cid)
                plays = (assets_by_persona.get(pid, {}).get(cid) or [])
                if (not plays) or all((p.get("fitness", 0.0) or 0.0) < fitness_min for p in plays) or \
                   (len(plays) == 1 and "note" in plays[0]):
                    gaps.append({**c})
                else:
                    covered.append({**c, "plays": plays})
            out[pid] = {"top_concerns": topc, "gaps": gaps, "covered": covered}
        return out

    sequences_and_campaigns = []
    for rec in seqs[:5]:
        sequence = rec["sequence"]
        sequence_labels = [{"id": pid, "label": _set_node_label(G_pruned, pid)} for pid in sequence]
        coals = _sequence_coalitions(
            ctx,
            sequence,
            boost=boost_factor,
            max_coalition_size=3,
            min_synergy=1e-6,
            candidate_personas=seq_pool_ids,
        )

        gaps = _concern_gaps_for_sequence(sequence, max_concerns_per_persona=3, fitness_min=0.25)

        # enrich labels for readability
        for pid, gap_info in gaps.items():
            gap_info["persona_label"] = _set_node_label(G_pruned, pid)
            for key in ("top_concerns", "gaps", "covered"):
                for c in gap_info.get(key, []):
                    c["concern_label"] = c.get("concern_label") or _set_node_label(G_pruned, c["concern_id"])
                    c["persona_label"] = _set_node_label(G_pruned, pid)
                    if key == "covered":
                        for play in c.get("plays", []):
                            play["asset_label"] = play.get("asset_name")
                            play["channel_label"] = play.get("channel", {}).get("name") if play.get("channel") else None

        def _stage_of_first_concern(pid: str) -> str:
            lst = concerns_by_persona.get(pid) or []
            return (lst[0].get("stage") if lst else "problem")

        narrative_bits = [f"{_set_node_label(G_pruned,pid)} at {_stage_of_first_concern(pid)} level" for pid in sequence]
        narrative = " → then ".join(narrative_bits)

        sequences_and_campaigns.append({
            "narrative": f"Design a campaign that converts {narrative}.",
            "sequence": sequence,
            "sequence_labels": sequence_labels,
            "final_win": float(rec["final_win"]),
            "base_win": float(rec["base_win"]),
            "total_lift": float(rec["total_lift"]),
            "steps": rec["steps"],
            "coalitions": coals,
            "concern_gaps": gaps,
            
        })
    
    # Compose final report — identical keys to legacy engine
    report = {
        "baseline": baseline_block,
        "top_personas": top_personas_block,
        "concerns_by_persona": [
            {"persona": pid,"persona_label": _set_node_label(G_pruned, pid), "concerns": clist, "assets": assets_by_persona.get(pid, {})}
            for pid, clist in concerns_by_persona.items()
        ],
        "concerns_flat": concerns_flat,
        "coalitions": coalitions_out,
        "causal_flows": causal_flows,
        "sequences_and_campaigns": sequences_and_campaigns,
        "concern_backlog": concern_backlog,          # flat, prioritized (p, c, stage, lift_proxy, etc.)
        "concern_coalitions": concern_coalitions,    # best simultaneous bundles of concerns
        "concern_sequences": concern_sequences,      # best ordered playbooks of concerns

    }

    print("Report generation complete.")
    print("Win rate now:", baseline_block["win_likelihood"])
    

    # Return pruned (zmot/attribute removed) graph for UI continuity
    return G_pruned, report
