# ============================
# File: backend/utils/inference/rcs_generators/generate_rcs_fast.py
# ============================
from __future__ import annotations
from typing import Any, Dict, List, Optional
from collections import defaultdict
import networkx as nx

from backend.utils.graph_base.network_graph import (
    _set_node_label,
    get_nodes_list_ids,
)
from backend.utils.inference.rcs_generators.graph_algorithms import _phase_from_perc_prox, build_concern_backlog_from_activation_breakdown, get_involvement_activation_report
from backend.utils.inference.rcs_generators.rcs_computations.graphwin_runtime import get_graphwin





# ------------------------------------------------------------
# Sequences (persona-local, linearized top concerns)
# ------------------------------------------------------------
def build_concern_sequences(
    G: nx.DiGraph,
    concerns_by_persona: Dict[str, List[Dict[str, Any]]],
    *,
    max_steps_per_persona: int = 6,
) -> List[Dict[str, Any]]:
    """
    For each persona, turn their top concerns into a simple linear sequence.

    Schema (backward-compatible):
      sequence_item = {
        # canonical
        "cid": str,                    # canonical
        "label": str,                  # canonical
        "stage": str,                  # canonical (execution|problem|pain|resolution...)
        "phase": str,                  # derived from perc/prox: zmot|discovery|barriers|implementation
        "perceptibility": float,
        "proximity": float,
        "keyness": float,
        "lift_proxy": float,
        "involvement": float,          # persona-level involvement
        "activation": float,
        "concern_involvement": float,
        "persona": str,                # convenience (persona_label)

        # aliases (compat)
        "concern_id": str,
        "concern_label": str,
        "concern_stage": str,
      }

      sequence_block = {
        "persona_id": str,
        "persona_label": str,
        "sequence": [sequence_item, ...]   # canonical
      }
    """
    sequences: List[Dict[str, Any]] = []

    for pid, rows in (concerns_by_persona or {}).items():
        persona_label = _set_node_label(G, pid)
        seq_items: List[Dict[str, Any]] = []

        for r in (rows or [])[:max_steps_per_persona]:
            # tolerate both 'cid' and 'concern_id' coming in
            cid = r.get("cid") or r.get("concern_id")
            if not cid:
                # skip malformed rows; keeps downstream clean
                continue

            perc = float(r.get("perceptibility", 0.0) or 0.0)
            prox = float(r.get("proximity", 0.0) or 0.0)
            label = r.get("concern_label") or r.get("label") or _set_node_label(G, cid)
            stage = r.get("stage") or r.get("concern_stage") or "problem"
            phase = _phase_from_perc_prox(perc, prox)

            item = {
                # canonical
                "cid": cid,
                "label": label,
                "stage": stage,
                "phase": phase,
                "perceptibility": perc,
                "proximity": prox,
                "keyness": float(r.get("keyness", 0.0) or 0.0),
                "lift_proxy": float(r.get("lift_proxy", 0.0) or 0.0),
                "involvement": float(r.get("involvement", 0.0) or 0.0),
                "activation": float(r.get("activation", 0.0) or 0.0),
                "concern_involvement": float(r.get("concern_involvement", 0.0) or 0.0),
                "persona": persona_label,

                # aliases (compat)
                "concern_id": cid,
                "concern_label": label,
                "concern_stage": stage,
            }
            seq_items.append(item)

        sequences.append({
            "persona_id": pid,
            "persona_label": persona_label,
            # canonical key expected by downstream
            "sequence": seq_items,
            # no 'steps' — we’ve normalized to 'sequence'
        })

    return sequences



# ------------------------------------------------------------
# Coalitions (persona-local clusters of related concerns)
# ------------------------------------------------------------
def _safe_get(d: dict, k: str, default=0.0) -> float:
    try:
        return float(d.get(k, default) or 0.0)
    except Exception:
        return float(default)

def _are_graph_adjacent(G: nx.DiGraph, a: str, b: str) -> bool:
    if a == b:
        return True
    return G.has_edge(a, b) or G.has_edge(b, a)

def _has_common_neighbor(G: nx.DiGraph, a: str, b: str) -> bool:
    Nu = set(G.predecessors(a)) | set(G.successors(a))
    Nv = set(G.predecessors(b)) | set(G.successors(b))
    return len(Nu & Nv) > 0

def _concern_similarity(G: nx.DiGraph, r1: Dict[str, Any], r2: Dict[str, Any]) -> float:
    """
    Soft similarity:
      +0.50 same phase
      +0.25 same concern_stage
      +0.25 graph-adjacent (or +0.15 if share neighbor)
    """
    sim = 0.0
    if r1.get("phase") and r1.get("phase") == r2.get("phase"):
        sim += 0.50
    if r1.get("stage") and r1.get("stage") == r2.get("stage"):
        sim += 0.25
    c1, c2 = r1.get("cid"), r2.get("cid")
    if c1 and c2:
        if _are_graph_adjacent(G, c1, c2):
            sim += 0.25
        elif _has_common_neighbor(G, c1, c2):
            sim += 0.15
    return min(sim, 1.0)

def _aggregate_coalition_stats(members: List[Dict[str, Any]]) -> Dict[str, float]:
    if not members:
        return {"avg_perc": 0.0, "avg_prox": 0.0, "avg_key": 0.0, "sum_lift": 0.0}
    n = float(len(members))
    avg_perc = sum(_safe_get(m, "perceptibility") for m in members) / n
    avg_prox = sum(_safe_get(m, "proximity")      for m in members) / n
    avg_key  = sum(_safe_get(m, "keyness")        for m in members) / n
    sum_lift = sum(_safe_get(m, "lift_proxy")     for m in members)
    return {"avg_perc": avg_perc, "avg_prox": avg_prox, "avg_key": avg_key, "sum_lift": sum_lift}

def _dominant_phase(members: List[Dict[str, Any]]) -> str:
    if not members:
        return "discovery"
    counts = defaultdict(int)
    for m in members:
        counts[m.get("phase", "discovery")] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]

def build_persona_coalitions(
    G: nx.DiGraph,
    concerns_by_persona: Dict[str, List[Dict[str, Any]]],
    *,
    sim_threshold: float = 0.60,
    max_group_size: int = 6,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Greedy clustering of concerns per persona into coalitions using phase/stage/graph proximity.
    Returns: { persona_id: [ {coalition_id, persona_id, persona_label, members:[...], stats:{...}, dominant_phase}, ... ] }
    """
    out: Dict[str, List[Dict[str, Any]]] = {}

    for pid, rows in concerns_by_persona.items():
        # attach phase if missing (should already be present from backlog builder)
        enriched = []
        for r in rows:
            if "phase" not in r:
                perc = _safe_get(r, "perceptibility")
                prox = _safe_get(r, "proximity")
                rr = dict(r)
                rr["phase"] = _phase_from_perc_prox(perc, prox)
                enriched.append(rr)
            else:
                enriched.append(r)

        # seed sort by lift + keyness as rough priority
        def _seed_rank(x: Dict[str, Any]) -> float:
            return _safe_get(x, "lift_proxy") * (0.5 + _safe_get(x, "keyness") / 2.0)

        enriched.sort(key=_seed_rank, reverse=True)

        coalitions: List[Dict[str, Any]] = []
        col_id = 1

        for r in enriched:
            placed = False
            for col in coalitions:
                members = col["members"]
                if len(members) >= max_group_size:
                    continue
                sim = 0.0
                for m in members:
                    sim = max(sim, _concern_similarity(G, r, m))
                    if sim >= sim_threshold:
                        break
                if sim >= sim_threshold:
                    members.append(r)
                    placed = True
                    break

            if not placed:
                coalitions.append({
                    "coalition_id": f"col_{pid}_{col_id}",
                    "persona_id": pid,
                    "persona_label": _set_node_label(G, pid),
                    "members": [r],
                    "stats": {},
                    "dominant_phase": r.get("phase", "discovery"),
                })
                col_id += 1

        for col in coalitions:
            col["stats"] = _aggregate_coalition_stats(col["members"])
            col["dominant_phase"] = _dominant_phase(col["members"])

        out[pid] = coalitions

    return out


# ------------------------------------------------------------
# Org-level coalition view (flatten + rank)
# ------------------------------------------------------------
def aggregate_org_coalitions(
    coalitions_by_persona: Dict[str, List[Dict[str, Any]]],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Flatten persona coalitions and rank at org level.
    org_rank_score = sum_lift × (0.5 + avg_key/2) with tie-breakers on avg_prox then avg_perc.
    """
    flat: List[Dict[str, Any]] = []
    for cols in coalitions_by_persona.values():
        for c in cols:
            stats = c.get("stats", {})
            score = _safe_get(stats, "sum_lift") * (0.5 + _safe_get(stats, "avg_key") / 2.0)
            flat.append({
                **c,
                "org_rank_score": float(score),
            })

    flat.sort(key=lambda r: (
        -_safe_get(r, "org_rank_score"),
        -_safe_get(r.get("stats", {}), "avg_prox"),
        -_safe_get(r.get("stats", {}), "avg_perc"),
    ))
    return flat[:top_k]


# ------------------------------------------------------------
# Optional: Best org theme seed (top pain_trigger by involvement/strength)
# ------------------------------------------------------------
def get_best_org_theme(G: nx.DiGraph, core_scores: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
    pain_trigger_nodes = get_nodes_list_ids(G, "pain_trigger", {})
    best = None
    for p in pain_trigger_nodes:
        if p not in core_scores:
            continue
        s = core_scores[p]
        row = {
            "id": p,
            "label": _set_node_label(G, p),
            "involvement": float(s.get("involvement", 0.0)),
            "activation": float(s.get("activation", 0.0)),
            "strength": float(s.get("strength", 0.0)),
        }
        if (best is None) or (row["involvement"], row["strength"]) > (best["involvement"], best["strength"]):
            best = row
    return best


# ------------------------------------------------------------
# MAIN: generate_rcs (deterministic; math lives in graph_algorithms)
# ------------------------------------------------------------
def generate_rcs(
    *,
    product_graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    engaged_nodes: Optional[List[Dict[str, Any]]] = None,
    stage_weights: Optional[Dict[str, float]] = None,
    max_steps_per_persona: int = 6,
) -> Dict[str, Any]:
    """
    MVP orchestration:
      1) Compute core & persona scores (now includes perc/prox everywhere).
      2) Build a concern backlog + persona-local lists (includes perc/prox/keyness).
      3) Build simple sequences (per persona).
      4) Build coalitions (per persona) and an org-level coalition ranking.
    """
    engaged_nodes = engaged_nodes or []
    product_graph.graph.setdefault("graphwin", 0.0)
    win_likelihood = get_graphwin(product_graph, engaged_nodes).get("win_likelihood")
    print("Computed graphwin likelihood:", win_likelihood, "of type: ", type(win_likelihood))
    if win_likelihood is not None:
        cur_win = product_graph.graph.get("graphwin", 0.0)
        print("Current stored graphwin likelihood:", cur_win, "of type: ", type(cur_win))
        if win_likelihood > cur_win:
            product_graph.graph["graphwin"] = win_likelihood

    # (1) Core + Persona (Perc/Prox included via graph_algorithms)
    report = get_involvement_activation_report(
        G=product_graph,
        Original_G=original_graph,
        engaged_nodes=engaged_nodes,
    )
    core_scores = report.get("core_scores", {})
    persona_scores = report.get("persona_scores", {})
    activation_breakdown = report.get("activation_breakdown", [])
    graph_seeded = report.get("graph") or product_graph

    # (2) Backlog & persona-local concern lists
    concern_backlog, concerns_by_persona = build_concern_backlog_from_activation_breakdown(
        original_graph,
        activation_breakdown,
        stage_weights=stage_weights,
        top_k_per_persona=max_steps_per_persona,
    )

    # (3) Sequences (persona-linear)
    sequences = build_concern_sequences(
        original_graph,
        concerns_by_persona,
        max_steps_per_persona=max_steps_per_persona,
    )

    # (4) Coalitions (persona-local → org-level)
    coalitions_by_persona = build_persona_coalitions(
        original_graph,
        concerns_by_persona,
        sim_threshold=0.60,
        max_group_size=6,
    )
    org_coalitions = aggregate_org_coalitions(coalitions_by_persona, top_k=20)

    # Optional: a seed theme for org POV (useful for top-level narrative)
    org_theme_seed = get_best_org_theme(original_graph, core_scores)

    return {
        "graph": graph_seeded,
        "graphwin": product_graph.graph.get("graphwin", 0.0),
        "core_scores": core_scores,
        "persona_scores": persona_scores,
        "activation_breakdown": activation_breakdown,
        "concern_backlog": concern_backlog,
        "concerns_by_persona": concerns_by_persona,
        "sequences": sequences,                 # canonical
        "concern_sequences": sequences,         # alias for older callers
        "coalitions_by_persona": coalitions_by_persona,
        "org_coalitions": org_coalitions,
        "org_theme_seed": org_theme_seed,
    }

