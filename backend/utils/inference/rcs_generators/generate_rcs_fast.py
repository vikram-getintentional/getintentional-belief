# ============================
# backend/utils/inference/rcs_generators/generate_rcs_fast.py
# ============================
from __future__ import annotations
import math, random
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import networkx as nx

from backend.utils.graph_base.network_graph import (
    _set_node_label,
    build_product_graph,
    get_nodes_list_ids,
)
from backend.utils.inference.rcs_generators.graph_algorithms import (
    _phase_from_perc_prox,
    build_concern_backlog_from_activation_breakdown,
    get_involvement_activation_report,
)
from backend.utils.inference.rcs_generators.rcs_computations.graphwin_runtime import (
    get_graphwin,
)

# ---------- tiny utils ----------
def _safe_get(d: dict, k: str, default=0.0) -> float:
    try:
        return float(d.get(k, default) or 0.0)
    except Exception:
        return float(default)

def _node_type(G: nx.DiGraph, nid: str) -> str:
    d = G.nodes.get(nid, {})
    return (d.get("node_type") or d.get("type") or "").strip().lower()

# ============================
# Concern sequences / coalitions (unchanged logic)
# ============================
def build_concern_sequences(G: nx.DiGraph, concerns_by_persona: Dict[str, List[Dict[str, Any]]], *, max_steps_per_persona: int = 6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pid, rows in (concerns_by_persona or {}).items():
        seq_items: List[Dict[str, Any]] = []
        for r in (rows or [])[:max_steps_per_persona]:
            cid = r.get("cid") or r.get("concern_id")
            if not cid: continue
            perc = _safe_get(r, "perceptibility")
            prox = _safe_get(r, "proximity")
            label = r.get("concern_label") or r.get("label") or _set_node_label(G, cid)
            stage = r.get("stage") or r.get("concern_stage") or "problem"
            phase = _phase_from_perc_prox(perc, prox)
            seq_items.append({
                "cid": cid, "label": label, "stage": stage, "phase": phase,
                "perceptibility": perc, "proximity": prox,
                "keyness": _safe_get(r, "keyness"), "lift_proxy": _safe_get(r, "lift_proxy"),
                "involvement": _safe_get(r, "involvement"), "activation": _safe_get(r, "activation"),
                "concern_involvement": _safe_get(r, "concern_involvement"),
                "persona": _set_node_label(G, pid),
                "concern_id": cid, "concern_label": label, "concern_stage": stage,
            })
        out.append({"persona_id": pid, "persona_label": _set_node_label(G, pid), "sequence": seq_items})
    return out

def _are_graph_adjacent(G: nx.DiGraph, a: str, b: str) -> bool:
    return a == b or G.has_edge(a, b) or G.has_edge(b, a)

def _has_common_neighbor(G: nx.DiGraph, a: str, b: str) -> bool:
    Nu = set(G.predecessors(a)) | set(G.successors(a))
    Nv = set(G.predecessors(b)) | set(G.successors(b))
    return len(Nu & Nv) > 0

def _concern_similarity(G: nx.DiGraph, r1: Dict[str, Any], r2: Dict[str, Any]) -> float:
    sim = 0.0
    if r1.get("phase") and r1.get("phase") == r2.get("phase"): sim += 0.50
    if r1.get("stage") and r1.get("stage") == r2.get("stage"): sim += 0.25
    c1, c2 = r1.get("cid"), r2.get("cid")
    if c1 and c2:
        if _are_graph_adjacent(G, c1, c2): sim += 0.25
        elif _has_common_neighbor(G, c1, c2): sim += 0.15
    return min(sim, 1.0)

def _aggregate_coalition_stats(members: List[Dict[str, Any]]) -> Dict[str, float]:
    if not members: return {"avg_perc":0.0,"avg_prox":0.0,"avg_key":0.0,"sum_lift":0.0}
    n = float(len(members))
    return {
        "avg_perc": sum(_safe_get(m,"perceptibility") for m in members)/n,
        "avg_prox": sum(_safe_get(m,"proximity") for m in members)/n,
        "avg_key":  sum(_safe_get(m,"keyness") for m in members)/n,
        "sum_lift": sum(_safe_get(m,"lift_proxy") for m in members),
    }

def _dominant_phase(members: List[Dict[str, Any]]) -> str:
    if not members: return "discovery"
    counts = defaultdict(int)
    for m in members: counts[m.get("phase","discovery")] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]

def build_persona_coalitions(G: nx.DiGraph, concerns_by_persona: Dict[str, List[Dict[str, Any]]], *, sim_threshold: float = 0.60, max_group_size: int = 6) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for pid, rows in concerns_by_persona.items():
        enriched = []
        for r in rows:
            if "phase" not in r:
                rr = dict(r)
                rr["phase"] = _phase_from_perc_prox(_safe_get(r,"perceptibility"), _safe_get(r,"proximity"))
                enriched.append(rr)
            else:
                enriched.append(r)
        def _seed_rank(x: Dict[str, Any]) -> float:
            return _safe_get(x,"lift_proxy")*(0.5+_safe_get(x,"keyness")/2.0)
        enriched.sort(key=_seed_rank, reverse=True)
        cols: List[Dict[str, Any]] = []
        cid = 1
        for r in enriched:
            placed = False
            for col in cols:
                if len(col["members"]) >= max_group_size: continue
                sim = max((_concern_similarity(G,r,m) for m in col["members"]), default=0.0)
                if sim >= sim_threshold:
                    col["members"].append(r); placed = True; break
            if not placed:
                cols.append({
                    "coalition_id": f"col_{pid}_{cid}",
                    "persona_id": pid, "persona_label": _set_node_label(G, pid),
                    "members": [r], "stats": {}, "dominant_phase": r.get("phase","discovery"),
                }); cid += 1
        for c in cols:
            c["stats"] = _aggregate_coalition_stats(c["members"])
            c["dominant_phase"] = _dominant_phase(c["members"])
        out[pid] = cols
    return out

def aggregate_org_coalitions(coalitions_by_persona: Dict[str, List[Dict[str, Any]]], top_k: int = 20) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for cols in coalitions_by_persona.values():
        for c in cols:
            s = c.get("stats", {})
            score = _safe_get(s,"sum_lift")*(0.5+_safe_get(s,"avg_key")/2.0)
            flat.append({**c, "org_rank_score": float(score)})
    flat.sort(key=lambda r: (-_safe_get(r,"org_rank_score"), -_safe_get(r.get("stats", {}),"avg_prox"), -_safe_get(r.get("stats", {}),"avg_perc")))
    return flat[:top_k]

def get_best_org_theme(G: nx.DiGraph, core_scores: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
    pain_trigger_nodes = get_nodes_list_ids(G, "pain_trigger", {})
    best = None
    for p in pain_trigger_nodes:
        if p not in core_scores: continue
        s = core_scores[p]
        row = {"id": p, "label": _set_node_label(G,p),
               "involvement": _safe_get(s,"involvement"),
               "activation": _safe_get(s,"activation"),
               "strength": _safe_get(s,"strength")}
        if (best is None) or (row["involvement"], row["strength"]) > (best["involvement"], best["strength"]):
            best = row
    return best

# ============================
# Evidence normalization & seeding
# ============================
def _as_engaged_list(G: nx.DiGraph, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items or []:
        nid = it.get("id") or it.get("node_id") or it.get("nid")
        if not nid or nid not in G: continue
        try: occ = float(it.get("occurrence", 1.0))
        except Exception: occ = 1.0
        if occ <= 0: occ = 1.0
        out.append({"id": nid, "occurrence": occ})
    return out

def _dedupe_merge_occurrences(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    acc: Dict[str, float] = {}
    for it in items or []:
        nid = it.get("id"); 
        if not nid: continue
        acc[nid] = acc.get(nid, 0.0) + (float(it.get("occurrence", 1.0)) or 1.0)
    return [{"id": k, "occurrence": v} for k, v in acc.items()]

def _apply_engagement_evidence(G: nx.DiGraph, engaged: List[Dict[str, Any]]) -> None:
    for e in engaged or []:
        nid = e.get("id")
        if not nid or nid not in G: continue
        nd = G.nodes[nid]
        nd["occurred"] = True
        nd["occurrence"] = float(nd.get("occurrence", 0.0)) + float(e.get("occurrence", 1.0) or 1.0)
        if "perceptibility_original" not in nd and "perceptibility" in nd:
            nd["perceptibility_original"] = nd.get("perceptibility")
        nd["perceptibility_evidence"] = 1.0
        nd["perceptibility"] = 1.0  # observed ⇒ fully perceptible

# ============================
# Persona-path scoring (core of your thesis)
# ============================
def _persona_metrics(pid: str, persona_scores: Dict[str, Dict[str, float]], G: nx.DiGraph) -> Tuple[float,float,float]:
    """Return (perceptibility, proximity, involvement) with safe fallbacks."""
    ps = persona_scores.get(pid, {})
    nd = G.nodes.get(pid, {})
    perc = _safe_get(ps, "perceptibility") or _safe_get(nd, "perceptibility")
    prox = _safe_get(ps, "proximity")       or _safe_get(nd, "proximity")
    inv  = _safe_get(ps, "involvement")     or _safe_get(nd, "involvement")
    return perc, prox, inv

def _persona_step_score(pid: str, persona_scores: Dict[str, Dict[str, float]], G: nx.DiGraph) -> float:
    perc, prox, inv = _persona_metrics(pid, persona_scores, G)
    # clip to [0,1] to avoid runaway products
    perc = max(0.0, min(1.0, perc)); prox = max(0.0, min(1.0, prox)); inv = max(0.0, min(1.0, inv))
    return perc * prox * inv

def _score_state_personas(
    *,
    product_graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    engaged: List[Dict[str, Any]],
    attr_prior: Optional[Dict[str, float]],
    alpha: float,
    weight_key: str,
) -> Tuple[nx.DiGraph, Dict[str, Dict[str, float]]]:
    """Seed evidence, compute persona_scores for current state."""
    Gs = product_graph.copy(as_view=False)
    _apply_engagement_evidence(Gs, engaged)
    rep = get_involvement_activation_report(
        G=Gs, Original_G=original_graph, engaged_nodes=engaged,
        alpha=alpha, weight_key=weight_key, attr_prior=attr_prior,
    )
    Gs = rep.get("graph", Gs)
    persona_scores = rep.get("persona_scores", {}) or {}
    return Gs, persona_scores

def beam_search_persona_paths(
    *,
    product_graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    engaged_nodes: List[Dict[str, Any]],
    attr_prior: Optional[Dict[str, float]],
    alpha: float,
    weight_key: str,
    beam_width: int = 3,
    max_depth: int = 6,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns (paths, expected_next_list).
    Each path: {"personas":[...], "score": float, "probability": float}
    Score uses product of f = perceptibility × proximity × involvement (log-sum internally).
    expected_next_list is the top personas at depth=1 expansion from current state.
    """
    engaged = _dedupe_merge_occurrences(engaged_nodes or [])
    engaged_set = {e["id"] for e in engaged}

    # initial state
    G0, ps0 = _score_state_personas(
        product_graph=product_graph, original_graph=original_graph,
        engaged=engaged, attr_prior=attr_prior, alpha=alpha, weight_key=weight_key,
    )

    # candidates for "next": sort by single-step score, exclude already-engaged
    def top_candidates(ps: Dict[str, Dict[str, float]], exclude: set, k: int) -> List[str]:
        cands = [pid for pid in ps.keys() if pid not in exclude and _node_type(G0, pid) == "persona"]
        cands.sort(key=lambda pid: _persona_step_score(pid, ps, G0), reverse=True)
        return cands[:k]

    expected_next = top_candidates(ps0, engaged_set, beam_width)

    # beam holds tuples: (neg_log_score, personas_list, engaged_list_snapshot)
    beam: List[Tuple[float, List[str], List[Dict[str, Any]]]] = []
    for pid in expected_next:
        s = _persona_step_score(pid, ps0, G0)
        if s <= 0: continue
        beam.append((-math.log(s), [pid], engaged + [{"id": pid, "occurrence": 1.0}]))
    if not beam:
        return ([], expected_next)

    best_paths: List[Tuple[float, List[str]]] = []

    depth = 1
    while depth < max_depth and beam:
        # Expand each beam member
        new_beam: List[Tuple[float, List[str], List[Dict[str, Any]]]] = []
        for neg_log, path, eng in beam:
            Gd, psd = _score_state_personas(
                product_graph=product_graph, original_graph=original_graph,
                engaged=eng, attr_prior=attr_prior, alpha=alpha, weight_key=weight_key,
            )
            exclude = set(p for p in path) | {e["id"] for e in eng}
            nxts = top_candidates(psd, exclude, beam_width)
            if not nxts:
                best_paths.append((neg_log, path)); continue
            for pid in nxts:
                s = _persona_step_score(pid, psd, Gd)
                if s <= 0: 
                    best_paths.append((neg_log, path))
                    continue
                new_beam.append((neg_log + (-math.log(s)), path + [pid], _dedupe_merge_occurrences(eng + [{"id": pid, "occurrence":1.0}])))
        # prune
        new_beam.sort(key=lambda t: t[0])
        beam = new_beam[:beam_width]
        depth += 1

    for neg_log, path, _ in beam:
        best_paths.append((neg_log, path))

    # softmax over -neg_log to produce probabilities
    if not best_paths:
        return ([], expected_next)

    scores = [-nl for nl, _ in best_paths]  # log-scores
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    Z = sum(exps) or 1.0
    paths = [{"personas": p, "score": float(math.exp(s)), "probability": float(e/Z)} for (s, p), e in zip(best_paths, exps)]
    paths.sort(key=lambda r: r["probability"], reverse=True)
    return (paths, expected_next)

# ============================
# Legacy wrapper (kept)
# ============================
def generate_rcs(*, product_graph: nx.DiGraph, original_graph: nx.DiGraph, engaged_nodes: Optional[List[Dict[str, Any]]] = None, stage_weights: Optional[Dict[str, float]] = None, max_steps_per_persona: int = 6) -> Dict[str, Any]:
    engaged_nodes = engaged_nodes or []
    win_likelihood = get_graphwin(product_graph, engaged_nodes).get("win_likelihood")
    if win_likelihood is not None:
        cur_win = product_graph.graph.get("graphwin", 0.0)
        if win_likelihood > cur_win: product_graph.graph["graphwin"] = win_likelihood

    report = get_involvement_activation_report(G=product_graph, Original_G=original_graph, engaged_nodes=engaged_nodes)
    core_scores = report.get("core_scores", {})
    persona_scores = report.get("persona_scores", {})
    activation_breakdown = report.get("activation_breakdown", [])
    graph_seeded = report.get("graph") or product_graph

    concern_backlog, concerns_by_persona = build_concern_backlog_from_activation_breakdown(
        original_graph, activation_breakdown, stage_weights=stage_weights, top_k_per_persona=max_steps_per_persona,
    )
    sequences = build_concern_sequences(original_graph, concerns_by_persona, max_steps_per_persona=max_steps_per_persona)
    coalitions_by_persona = build_persona_coalitions(original_graph, concerns_by_persona, sim_threshold=0.60, max_group_size=6)
    org_coalitions = aggregate_org_coalitions(coalitions_by_persona, top_k=20)
    org_theme_seed = get_best_org_theme(original_graph, core_scores)

    return {
        "graph": graph_seeded, "graphwin": product_graph.graph.get("graphwin", 0.0),
        "core_scores": core_scores, "persona_scores": persona_scores,
        "activation_breakdown": activation_breakdown,
        "concern_backlog": concern_backlog, "concerns_by_persona": concerns_by_persona,
        "sequences": sequences, "concern_sequences": sequences,
        "coalitions_by_persona": coalitions_by_persona, "org_coalitions": org_coalitions,
        "org_theme_seed": org_theme_seed,
    }

# ============================
# Preferred entrypoint
# ============================
def generate_rcs_new(
    *,
    product_graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    engaged_nodes: Optional[List[Dict[str, Any]]] = None,
    attr_prior: Optional[Dict[str, float]] = None,
    alpha: float = 0.85,
    weight_key: str = "likelihood",
    stage_weights: Optional[Dict[str, float]] = None,
    top_k_per_persona: int = 5,
    debug: bool = False,
) -> Dict[str, Any]:
    engaged_nodes = _dedupe_merge_occurrences(_as_engaged_list(product_graph, engaged_nodes or []))

    # Seed evidence and compute base report
    G_seed = product_graph.copy(as_view=False)
    _apply_engagement_evidence(G_seed, engaged_nodes)

    report = get_involvement_activation_report(
        G=G_seed, Original_G=original_graph, engaged_nodes=engaged_nodes,
        alpha=alpha, weight_key=weight_key, attr_prior=attr_prior,
    )
    G = report.get("graph", G_seed)
    core_scores = report.get("core_scores", {})
    persona_scores = report.get("persona_scores", {})
    activation_breakdown = report.get("activation_breakdown", [])

    # Backlog
    concern_backlog, concerns_by_persona = build_concern_backlog_from_activation_breakdown(
        G, activation_breakdown, stage_weights=stage_weights, top_k_per_persona=top_k_per_persona,
    )

    # GraphWin
    base_out = get_graphwin(G, engaged_nodes=engaged_nodes, account_prior=attr_prior, debug=debug)
    win_likelihood = float(base_out.get("win_likelihood", 0.0))

    # *** NEW: persona paths (beam) + expected next ***
    paths, expected_next = beam_search_persona_paths(
        product_graph=product_graph, original_graph=original_graph,
        engaged_nodes=engaged_nodes, attr_prior=attr_prior,
        alpha=alpha, weight_key=weight_key, beam_width=3, max_depth=6,
    )

    out = {
        "graph": G,
        "win_likelihood": win_likelihood,
        "core_scores": core_scores,
        "persona_scores": persona_scores,
        "activation_breakdown": activation_breakdown,
        "concern_backlog": concern_backlog,
        "concerns_by_persona": concerns_by_persona,
        "paths": paths,                         # <- ranked persona sequences by Π(perceptibility×proximity×involvement)
        "expected_next": expected_next,         # <- top 'next' personas right now
    }
    if debug:
        out["debug"] = {"engaged_nodes": engaged_nodes, "attr_prior_size": 0 if attr_prior is None else len(attr_prior)}
    return out

# ============================
# Journey replay (for UI chips)
# ============================
def _rank_personas_for_next(persona_scores: Dict[str, Dict[str, float]], G: nx.DiGraph, engaged_set: set, top_k: int = 5) -> List[str]:
    cands = [pid for pid in persona_scores.keys() if pid not in engaged_set and _node_type(G, pid) == "persona"]
    cands.sort(key=lambda pid: _persona_step_score(pid, persona_scores, G), reverse=True)
    return cands[:max(1, top_k)]

def build_likely_paths(
    *,
    product_graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    observed_personas_in_time: List[str],
    attr_prior: Optional[Dict[str, float]] = None,
    alpha: float = 0.85,
    weight_key: str = "likelihood",
    stage_weights: Optional[Dict[str, float]] = None,
    top_k_prediction: int = 5,
    debug: bool = False,
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    engaged: List[Dict[str, Any]] = []
    engaged_set: set = set()

    for t, pid in enumerate(observed_personas_in_time):
        Gt, pscore = _score_state_personas(
            product_graph=product_graph, original_graph=original_graph,
            engaged=engaged, attr_prior=attr_prior, alpha=alpha, weight_key=weight_key,
        )
        predicted = _rank_personas_for_next(pscore, Gt, engaged_set, top_k=top_k_prediction)
        bucket = "no_observation"; hit1 = False; hit3 = False
        if pid:
            if pid in predicted:
                idx = predicted.index(pid)
                if idx == 0: bucket, hit1, hit3 = "perfect_match", True, True
                elif idx < 3: bucket, hit3 = "skip_hit", True
                else: bucket = "off_path_known"
            else:
                bucket = "out_of_graph" if pid not in pscore else "off_path_known"

        steps.append({
            "t": t, "state_personas": list(engaged_set),
            "predicted_topK": predicted, "observed_next": pid,
            "bucket": bucket, "hit_at_1": hit1, "hit_at_3": hit3,
        })
        if pid:
            engaged.append({"id": pid, "occurrence": 1.0})
            engaged = _dedupe_merge_occurrences(engaged)
            engaged_set.add(pid)

    total = max(1, len(steps))
    acc1 = sum(1 for s in steps if s.get("hit_at_1"))/total
    acc3 = sum(1 for s in steps if s.get("hit_at_3"))/total
    fit = {"best_path_probability": float(acc1), "hit_at_1": acc1==1.0, "hit_at_3": acc3==1.0,
           "surprisal_index": float(1.0-acc3), "variance": float(max(0.0, acc3-acc1))}
    return {"steps": steps, "fit": fit}
