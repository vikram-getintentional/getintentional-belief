# ============================
# File: backend/utils/inference/rcs_generators/generate_rcs_fast.py
# ============================
from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from itertools import combinations
import networkx as nx

from backend.utils.graph_base.network_graph import _set_node_label, get_product_id_from_subgraph, get_source_nodes_by_target_and_type
from backend.utils.inference.rcs_generators.rcs_computations.graphwin_runtime import get_graphwin

# >>> SINGLE SOURCE OF TRUTH (math) <<<
# These imports must point to the *PPR-based* implementations you restored.
from backend.utils.inference.rcs_generators.graph_algorithms import (
    get_involvement_activation_report,   # -> {"core_scores": {...}, "persona_scores": {...}}
    _concern_backlog,                    # PPR-based backlog using core_scores (activation×involvement of pains/jobs)
)

# -----------------------------------------------------------------------------
# Light helpers (UI packaging, no math)
# -----------------------------------------------------------------------------
def _nt(G: nx.DiGraph, n: str) -> str:
    d = G.nodes.get(n, {})
    return d.get("node_type") or d.get("type") or "unknown"

def rcs_prepare(
    G: nx.DiGraph,
    *,
    engaged_nodes: Optional[List[Dict]] = None,
) -> Tuple[nx.DiGraph, Optional[str], List[Dict]]:
    """
    Keep this thin: prune noisy node types (if desired), return pruned graph,
    product_id, and the engaged_nodes payload unchanged.
    """
    engaged_nodes = engaged_nodes or []

    keep_types = {
        "product", "capability", "job", "pain", "pain_trigger",
        "persona", "zmot_event", "attribute_value"
    }
    Gp = G.copy()
    attributes =[]
    for n in list(Gp.nodes):
        if _nt(Gp, n) not in keep_types:
            Gp.remove_node(n)
        if _nt(Gp, n) == "attribute_value":
            attributes.append(n)
    product_id = get_product_id_from_subgraph(Gp)
    Gp = _prune_to_attr_product_paths(Gp, product_id, attributes)  

    return Gp, product_id, engaged_nodes

def _prune_to_attr_product_paths(G: nx.DiGraph, product_id: Optional[str], engaged_attr_ids: List[str]) -> nx.DiGraph:
    if not product_id or not engaged_attr_ids:
        return G.copy()

    # nodes that can reach the product (ancestors)
    try:
        descendants_of_product = set(nx.descendants(G, product_id))
    except nx.NetworkXError:
        descendants_of_product = set()
    descendants_of_product.add(product_id)

    keep_nodes: Set[str] = set()
    for attr_id in engaged_attr_ids:
        if attr_id not in G:
            continue
        try:
            ancestors_of_attr = set(nx.ancestors(G, attr_id))
        except nx.NetworkXError:
            ancestors_of_attr = set()
        ancestors_of_attr.add(attr_id)

        # nodes that are both reachable from the attribute and can reach the product
        keep_nodes |= (ancestors_of_attr & descendants_of_product)

        # explicitly add zmot_events connected to this attribute
        linked_zmots = get_source_nodes_by_target_and_type(G, attr_id, "boosted_in")
        if linked_zmots:
            keep_nodes.update(linked_zmots)

    # always keep the product if present
    if product_id in G:
        keep_nodes.add(product_id)

    if not keep_nodes:
        return G.copy()
    return G.subgraph(keep_nodes).copy()
    

# -----------------------------------------------------------------------------
# Concern coalitions / sequences (presentation logic)
# -----------------------------------------------------------------------------
def _jobs_of_concern(G, cid):
    jobs = set()
    for u, v, _ in G.in_edges(cid, data=True):
        if _nt(G, u) == "job":
            jobs.add(u)
    for u, v, _ in G.out_edges(cid, data=True):
        if _nt(G, v) == "job":
            jobs.add(v)
    return jobs

def _pains_of_concern(G, cid):
    pains = set()
    for u, v, _ in G.in_edges(cid, data=True):
        if _nt(G, u) == "pain":
            pains.add(u)
    for u, v, _ in G.out_edges(cid, data=True):
        if _nt(G, v) == "pain":
            pains.add(v)
    return pains

def _concern_similarity(G, a, b):
    aj, bj = _jobs_of_concern(G, a["cid"]), _jobs_of_concern(G, b["cid"])
    ap, bp = _pains_of_concern(G, a["cid"]), _pains_of_concern(G, b["cid"])
    def jacc(s1, s2):
        if not s1 and not s2:
            return 0.0
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
            "concern_label": r.get("concern_label") or r.get("cid"),
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
                         "lift_proxy": s["lift_proxy"], "concern_label": s["concern_label"]}
                        for s in seq
                    ],
                    # Use monotonic union (noisy-OR) or keep avg; here keep avg for stability:
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
        for u, v, _ in G.out_edges(pid, data=True):
            if _nt(G, v) == "job":
                js.add(v)
        for u, v, _ in G.in_edges(pid, data=True):
            if _nt(G, u) == "job":
                js.add(u)
        return js
    def pains_of(pid):
        ps = set()
        for u, v, _ in G.out_edges(pid, data=True):
            if _nt(G, v) == "pain":
                ps.add(v)
        for u, v, _ in G.in_edges(pid, data=True):
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
            "labels": [_set_node_label(G, a), _set_node_label(G, b)],
            "compatibility": float(s),
        })
    return out

# -----------------------------------------------------------------------------
# MAIN: generate_rcs (deterministic; math delegated to graph_algorithms)
# -----------------------------------------------------------------------------
def generate_rcs(
    G: nx.DiGraph,
    *,
    engaged_nodes: Optional[List[Dict]] = None,
    boost_factor: float = 2.0,
    top_concerns_per_persona: int = 5,
    top_personas: int = 50,
    plays_per_concern: int = 3,
) -> Tuple[nx.DiGraph, Dict]:
    engaged_nodes = engaged_nodes or []

    # 0) Prepare/prune
    Gp, product_id, engaged_nodes = rcs_prepare(G, engaged_nodes=engaged_nodes)

    # 1) Baseline GraphWin (PPR-based, with union mode configurable inside runtime)
    baseline_block = get_graphwin(Gp, engaged_nodes=engaged_nodes)
    Gp.graph["win_likelihood"] = float(baseline_block.get("win_likelihood", 0.0))

    # 2) Core + Persona scores (all math in graph_algorithms)
    #    get_involvement_activation_report returns:
    #    {
    #      "core_scores":    { node_id: {"activation","involvement","strength","node_type",...}, ... },
    #      "persona_scores": { persona_id: {...}, ... }
    #    }
    scores = get_involvement_activation_report(Gp, G, engaged_nodes=engaged_nodes)
    core_scores    = scores.get("core_scores", {}) or {}
    persona_scores = scores.get("persona_scores", {}) or {}

    # Normalize persona list for “top-by-*”
    pa_rows = [{"id": pid, **vals} for pid, vals in persona_scores.items()]
    top_by_inv = sorted(pa_rows, key=lambda r: r.get("involvement", 0.0), reverse=True)[:top_personas]
    top_by_act = sorted(pa_rows, key=lambda r: r.get("activation", 0.0),  reverse=True)[:top_personas]
    top_by_str = sorted(pa_rows, key=lambda r: r.get("strength",   0.0),  reverse=True)[:top_personas]

    # 3) Concerns backlog (PPR-based) — uses *core_scores* (NOT persona scores)
    concern_backlog = _concern_backlog(
        Gp,
        [r["id"] for r in top_by_inv],   # persona ids
        node_scores=core_scores,         # pains/jobs activation×involvement
        top_k=60,
    )

    # 4) Group for UI
    concerns_by_persona: List[Dict] = []
    byp = defaultdict(list)
    for r in concern_backlog:
        byp[r["pid"]].append(r)
    for pid, rows in byp.items():
        concerns_by_persona.append({
            "persona": pid,
            "label": _set_node_label(Gp, pid),
            "concerns": [
                {
                    "concern_id": rr["cid"],
                    "label": rr.get("concern_label") or rr["cid"],
                    "stage": rr.get("stage"),
                    "lift_proxy": rr.get("lift_proxy", 0.0),
                }
                for rr in rows[:top_concerns_per_persona]
            ]
        })

    # 5) Coalitions & Sequences (presentation)
    concern_coalitions = build_concern_coalitions(Gp, concern_backlog, sim_threshold=0.35, max_coalitions=6)
    concern_sequences  = build_concern_sequences(Gp, concern_backlog, max_sequences=3, max_len=4)
    persona_coalitions = build_persona_coalitions(Gp, top_by_inv[:10], max_pairs=8, min_overlap=0.2)

    # 6) Final report (stable keys)
    report = {
        "baseline": baseline_block,
        "top_personas": {
            "by_involvement": top_by_inv,
            "by_activation":  top_by_act,
            "by_strength":    top_by_str,
        },
        "concerns_by_persona": concerns_by_persona,
        "concerns_flat":       concern_backlog,
        "concern_backlog":     concern_backlog,
        "concern_coalitions":  concern_coalitions,
        "concern_sequences":   concern_sequences,
        "coalitions":          persona_coalitions,
        # Pass-throughs for downstream UI & orchestrators:
        "core_scores":         core_scores,
        "persona_scores":      persona_scores,
        "sequences_and_campaigns": [],
        "causal_flows": [],
    }
    return Gp, report
