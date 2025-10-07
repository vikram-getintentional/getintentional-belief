# ============================
# File: backend/utils/inference/rcs_generators/generate_rcs_fast.py
# ============================
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from itertools import combinations
import math
import networkx as nx


from backend.utils.graph_base.network_graph import (
    _set_node_label,
)
from backend.utils.inference.rcs_generators.graph_algorithms import infer_node_involvement_activation
from backend.utils.inference.rcs_generators.rcs_computations.graphwin_runtime import get_graphwin



# -----------------------------------------------------------------------------
# Helpers: safe getters / node type / labels
# -----------------------------------------------------------------------------
def _nt(G: nx.DiGraph, n: str) -> str:
    d = G.nodes.get(n, {})
    return d.get("node_type") or d.get("type") or "unknown"

# -----------------------------------------------------------------------------
# rcs_prepare: prune graph, compute dim_weights, collect ids, etc.
# (kept intentionally simple; plug back your previous pruner if you had one)
# -----------------------------------------------------------------------------
def rcs_prepare(
    G: nx.DiGraph,
    *,
    engaged_nodes: Optional[List[Dict]] = None,
) -> Tuple[object, Dict]:
    engaged_nodes = engaged_nodes or []

    # Very simple “prune”: keep core node_types you care about
    keep_types = {"product", "capability", "job", "pain", "pain_trigger", "persona", "zmot_event", "attribute_value"}
    Gp = G.copy()
    for n in list(Gp.nodes):
        if _nt(Gp, n) not in keep_types:
            Gp.remove_node(n)

    # Pick product (assume single product node exists)
    product_ids = [n for n in Gp.nodes if _nt(Gp, n) == "product"]
    conv_id = product_ids[0] if product_ids else None

    # Dimension weights (existing behavior you had): if engaged attributes present,
    # make a 1-hot or normalized dict by dimension name. For now we reduce to {dimension: 1.0}
    # if any attribute_value of that dimension is present.
    dim_weights: Dict[str, float] = {}
    for item in engaged_nodes:
        nid = item.get("id")
        if not nid or nid not in Gp:
            continue
        if _nt(Gp, nid) == "attribute_value":
            dim = Gp.nodes[nid].get("dimension") or Gp.nodes[nid].get("type") or "attribute"
            dim_weights[dim] = 1.0

    # Pack a small context object
    class Ctx:
        pass
    ctx = Ctx()
    ctx.G_pruned = Gp
    ctx.conv_id = conv_id

    pre = {
        "dim_weights": dim_weights,
        "baseline": {},
        "engaged_nodes": engaged_nodes,
    }
    return ctx, pre

# -----------------------------------------------------------------------------
# Involvement / Activation
# -----------------------------------------------------------------------------
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _jobs_of_persona(G: nx.DiGraph, pid: str) -> List[str]:
    js = set()
    for u, v, d in G.out_edges(pid, data=True):
        if _nt(G, v) == "job":
            js.add(v)
    for u, v, d in G.in_edges(pid, data=True):
        if _nt(G, u) == "job":
            js.add(u)
    return list(js)

def _rel_job_to_persona(G: nx.DiGraph, job: str, pid: str) -> float:
    # crude relation proxy: 1 if edge exists either way, else 0.5 if 2-hop, else small
    if G.has_edge(pid, job) or G.has_edge(job, pid):
        return 1.0
    # light 2-hop check
    for nbr in G.neighbors(pid):
        if G.has_edge(nbr, job) or G.has_edge(job, nbr):
            return 0.5
    return 0.2

def _compute_involvement_activation(
    G: nx.DiGraph,
    persona_ids: List[str],
) -> List[Dict]:
    """
    Heuristic:
      - Involvement ≈ 1 - Π(1 - job_lk) across persona's jobs (fallback 0.25 if unknown)
      - Activation ≈ involvement * mean(top-2 job_lk)
    Where job_lk is proxied from PageRank centrality (normalized).
    """
    if not persona_ids:
        return []
    # compute simple PR as a generic influence proxy
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-8)
    except Exception:
        pr = {n: 1.0 / max(1, G.number_of_nodes()) for n in G.nodes}

    # normalize to [0,1] in a robust way
    vals = [pr.get(n, 0.0) for n in G.nodes]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    def nrm(x): 
        if hi <= lo: return 0.0
        return (x - lo) / (hi - lo)

    out = []
    for pid in persona_ids:
        jobs = _jobs_of_persona(G, pid)
        job_scores = [nrm(pr.get(j, 0.0)) for j in jobs] or [0.25]
        inv_terms = [js for js in job_scores]
        inv = 1.0 - math.prod([1.0 - t for t in inv_terms])
        top2 = sorted(job_scores, reverse=True)[:2]
        felt = sum(top2) / max(1, len(top2))
        act = inv * felt
        out.append({
            "id": pid,
            "persona_label": _set_node_label(G, pid),
            "involvement": _clip01(inv),
            "activation": _clip01(act),
            "care": _clip01((inv * max(1e-9, felt)) ** 0.5),
        })
    return out

# -----------------------------------------------------------------------------
# Concern backlog (flat)
# -----------------------------------------------------------------------------
def _infer_stage(G: nx.DiGraph, cid: str) -> str:
    # light stage inference (adjust if you have explicit)
    # prefer node attr
    stage = G.nodes.get(cid, {}).get("stage")
    if stage:
        return stage
    # else infer from neighbors
    for u, v, d in G.in_edges(cid, data=True):
        if _nt(G, u) == "pain":
            return "pain"
    for u, v, d in G.out_edges(cid, data=True):
        if _nt(G, v) == "pain":
            return "pain"
    for u, v, d in G.in_edges(cid, data=True):
        if _nt(G, u) == "job":
            return "solution"
    return "problem"

def _concern_backlog(G: nx.DiGraph, top_personas: List[str], top_k: int = 30) -> List[Dict]:
    """
    Build flat (pid,cid,stage,lift_proxy) list:
      - Concerns are ‘pain’ nodes adjacent to jobs/personas.
      - lift_proxy heuristic: PR(pain) * best job PR around it.
    """
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-8)
    except Exception:
        pr = {n: 1.0 / max(1, G.number_of_nodes()) for n in G.nodes}

    pains = [n for n in G.nodes if _nt(G, n) == "pain"]
    rows = []
    for pid in top_personas:
        for p in pains:
            # persona relates to pain via job or direct
            rel = 0.0
            for u, v, d in G.out_edges(pid, data=True):
                if _nt(G, v) == "job":
                    if G.has_edge(v, p) or G.has_edge(p, v):
                        rel = max(rel, 1.0)
            if G.has_edge(pid, p) or G.has_edge(p, pid):
                rel = max(rel, 0.7)
            if rel <= 0.0:
                continue
            # lift proxy
            lift = float(pr.get(p, 0.0))
            rows.append({
                "pid": pid,
                "persona_label": _set_node_label(G, pid),
                "cid": p,
                "concern_label": _set_node_label(G, p),
                "stage": _infer_stage(G, p),
                "lift_proxy": lift,
            })
    rows.sort(key=lambda r: r["lift_proxy"], reverse=True)
    return rows[:top_k]

# -----------------------------------------------------------------------------
# Coalitions & Sequences (restored)
# -----------------------------------------------------------------------------
def _jobs_of_concern(G, cid):
    jobs = set()
    for u, v, d in G.in_edges(cid, data=True):
        if _nt(G, u) == "job":
            jobs.add(u)
    for u, v, d in G.out_edges(cid, data=True):
        if _nt(G, v) == "job":
            jobs.add(v)
    return jobs

def _pains_of_concern(G, cid):
    pains = set()
    for u, v, d in G.in_edges(cid, data=True):
        if _nt(G, u) == "pain":
            pains.add(u)
    for u, v, d in G.out_edges(cid, data=True):
        if _nt(G, v) == "pain":
            pains.add(v)
    return pains

def _concern_similarity(G, a, b):
    aj, bj = _jobs_of_concern(G, a["cid"]), _jobs_of_concern(G, b["cid"])
    ap, bp = _pains_of_concern(G, a["cid"]), _pains_of_concern(G, b["cid"])
    def jacc(s1, s2):
        if not s1 and not s2: return 0.0
        return len(s1 & s2) / max(1, len(s1 | s2))
    s = 0.6 * jacc(aj, bj) + 0.35 * jacc(ap, bp)
    if (a.get("stage") or "") == (b.get("stage") or ""):
        s += 0.05
    return s

def build_concern_coalitions(
    G,
    concern_backlog,
    *,
    sim_threshold: float = 0.35,
    max_coalitions: int = 6,
    max_items_per: int = 8,
):
    items = [
        {"pid": r.get("pid"), "cid": r.get("cid"), "stage": r.get("stage"),
         "lift_proxy": float(r.get("lift_proxy", 0.0)),
         "concern_label": r.get("concern_label") or r.get("cid")}
        for r in concern_backlog
        if r.get("cid") in G
    ][:60]
    used = set()
    coalitions = []
    for x in items:
        kx = (x["pid"], x["cid"])
        if kx in used:
            continue
        group = [x]
        used.add(kx)
        for y in items:
            ky = (y["pid"], y["cid"])
            if ky in used:
                continue
            if _concern_similarity(G, x, y) >= sim_threshold:
                group.append(y)
                used.add(ky)
        rep = max(group, key=lambda z: z["lift_proxy"])
        coalitions.append({
            "coalition_id": f"coco:{rep['cid']}",
            "label": rep.get("concern_label", rep["cid"]),
            "members": sorted(
                [{"persona": g["pid"], "concern_id": g["cid"], "stage": g["stage"],
                  "lift_proxy": float(g["lift_proxy"]),
                  "label": g.get("concern_label", g["cid"])} for g in group],
                key=lambda a: a["lift_proxy"], reverse=True
            )[:max_items_per],
        })
        if len(coalitions) >= max_coalitions:
            break
    return coalitions

_STAGE_ORDER = {"problem": 0, "pain": 1, "solution": 2}
def _stage_rank(stage): return _STAGE_ORDER.get((stage or "").lower(), 1)

def build_concern_sequences(
    G,
    concern_backlog,
    *,
    max_sequences: int = 3,
    max_len: int = 4,
):
    by_persona = defaultdict(list)
    for r in concern_backlog:
        if r.get("cid") not in G: 
            continue
        by_persona[r["pid"]].append({
            "pid": r["pid"], "cid": r["cid"], "stage": r.get("stage"),
            "lift_proxy": float(r.get("lift_proxy", 0.0)),
            "label": r.get("concern_label") or r.get("cid"),
        })
    sequences = []
    for pid, rows in by_persona.items():
        rows.sort(key=lambda z: (_stage_rank(z["stage"]), -z["lift_proxy"]))
        i = 0
        while i < len(rows) and len(sequences) < max_sequences:
            start = rows[i]
            seq = [start]
            last_stage = _stage_rank(start["stage"])
            for j in range(i + 1, len(rows)):
                if len(seq) >= max_len:
                    break
                sj = _stage_rank(rows[j]["stage"])
                if sj > last_stage:
                    seq.append(rows[j])
                    last_stage = sj
            if len(seq) >= 1:
                sequences.append({
                    "sequence_id": f"seq:{pid}:{seq[0]['cid']}",
                    "persona": pid,
                    "sequence": [
                        {"persona": s["pid"], "concern_id": s["cid"], "stage": s["stage"],
                         "lift_proxy": s["lift_proxy"], "label": s["label"]}
                        for s in seq
                    ],
                    "final_win": float(sum(s["lift_proxy"] for s in seq) / max(1, len(seq))),
                })
            i += 1
            if len(sequences) >= max_sequences:
                break
    sequences.sort(key=lambda s: s["final_win"], reverse=True)
    return sequences[:max_sequences]

def build_persona_coalitions(
    G,
    top_personas_list,
    *,
    max_pairs: int = 8,
    min_overlap: float = 0.2,
):
    ids = [p["id"] for p in top_personas_list if p.get("id") in G][:12]
    def jobs_of(pid):
        js = set()
        for u, v, d in G.out_edges(pid, data=True):
            if _nt(G, v) == "job":
                js.add(v)
        for u, v, d in G.in_edges(pid, data=True):
            if _nt(G, u) == "job":
                js.add(u)
        return js
    def pains_of(pid):
        ps = set()
        for u, v, d in G.out_edges(pid, data=True):
            if _nt(G, v) == "pain":
                ps.add(v)
        for u, v, d in G.in_edges(pid, data=True):
            if _nt(G, u) == "pain":
                ps.add(u)
        return ps
    def jacc(a, b):
        if not a and not b: return 0.0
        return len(a & b) / max(1, len(a | b))
    scored = []
    for a, b in combinations(ids, 2):
        s = 0.6 * jacc(jobs_of(a), jobs_of(b)) + 0.4 * jacc(pains_of(a), pains_of(b))
        if s >= min_overlap:
            scored.append((s, a, b))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, a, b in scored[:max_pairs]:
        out.append({
            "coalition_id": f"pcoal:{a}|{b}",
            "pair": [a, b],
            "compatibility": float(s),
        })
    return out

# -----------------------------------------------------------------------------
# MAIN: generate_rcs
# -----------------------------------------------------------------------------
def generate_rcs(
    G: nx.DiGraph,
    *,
    engaged_nodes: Optional[List[Dict]] = None,
    boost_factor: float = 2.0,
    top_concerns_per_persona: int = 5,
    top_personas: int = 20,
    plays_per_concern: int = 3,
) -> Tuple[nx.DiGraph, Dict]:
    engaged_nodes = engaged_nodes or []
    ctx, pre = rcs_prepare(G, engaged_nodes=engaged_nodes)
    Gp = ctx.G_pruned
    product_id = ctx.conv_id

    # ---------------- GraphWin (baseline) ----------------
    print("Computing baseline GraphWin with engaged nodes...", engaged_nodes)
    baseline_block = get_graphwin(Gp, engaged_nodes=engaged_nodes)
    Gp.graph["win_likelihood"] = float(baseline_block.get("win_likelihood", 0.0))
    

    # ---------------- Personas: Involvement / Activation ----------------
    pa_rows = []
    pa_rows = infer_node_involvement_activation(Gp, engaged_nodes=engaged_nodes)

    top_by_inv = sorted(pa_rows, key=lambda r: r["involvement"], reverse=True)[:top_personas]
    top_by_act = sorted(pa_rows, key=lambda r: r["activation"], reverse=True)[:top_personas]
    top_by_lift = sorted(pa_rows, key=lambda r: r["care"], reverse=True)[:top_personas]


    # ---------------- Concerns per persona + backlog ----------------
    concern_backlog = _concern_backlog(Gp, [r["id"] for r in top_by_inv], top_k=60)
    # concerns_by_persona shape for UI (grouped)
    concerns_by_persona: List[Dict] = []
    byp = defaultdict(list)
    for r in concern_backlog:
        byp[r["pid"]].append(r)
    for pid, rows in byp.items():
        concerns_by_persona.append({
            "persona": pid,
            "persona_label": _set_node_label(Gp, pid),
            "concerns": [
                {"concern_id": rr["cid"], "label": rr["concern_label"], "stage": rr["stage"], "lift_proxy": rr["lift_proxy"]}
                for rr in rows[:top_concerns_per_persona]
            ]
        })

    # ---------------- Coalitions & Sequences ----------------
    concern_coalitions = build_concern_coalitions(Gp, concern_backlog, sim_threshold=0.35, max_coalitions=6)
    concern_sequences  = build_concern_sequences(Gp, concern_backlog, max_sequences=3, max_len=4)
    persona_coalitions = build_persona_coalitions(Gp, top_by_inv[:10], max_pairs=8, min_overlap=0.2)

    # ---------------- Pack report ----------------
    report = {
        "baseline": baseline_block,
        "top_personas": {
            "by_involvement": top_by_inv,
            "by_activation": top_by_act,
            "by_marginal_lift": top_by_lift
        },
        "concerns_by_persona": concerns_by_persona,
        "concerns_flat": concern_backlog,
        "concern_backlog": concern_backlog,
        "concern_coalitions": concern_coalitions,
        "concern_sequences": concern_sequences,
        "coalitions": persona_coalitions,
        # (leave placeholders you already had if the UI needs them)
        "sequences_and_campaigns": [],
        "causal_flows": [],
    }
    return Gp, report
