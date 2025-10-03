# zmot_icp_generation.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import hashlib
import json
import itertools
import networkx as nx
from dataclasses import dataclass

from backend.utils.graph_base.network_graph import (
    _set_node_label, get_nodes_list_ids, get_node_by_id, get_product_id_from_subgraph
)
from backend.utils.graph_base.graph_data.rcs_utils.save_and_load_rcs import (
    load_rcs_from_json, save_rcs_as_json
)
from backend.utils.inference.rcs_generators.generate_rcs_fast import generate_rcs

# -------------------------
# Attribute mining settings
# -------------------------
ATTR_FAMILIES = [
    ("industry",        "icp_industry"),
    ("revenue_range",   "icp_revenue"),
    ("employee_range",  "icp_employees"),
    ("funding_stage",   "icp_funding_stage"),
    ("geography",       "icp_geography"),
]

MAX_COMBO_LEN = 3  # singletons, pairs, triples (safe default)
TOP_K = 50         # limit how many combos you return to the UI

# -------------------------
# Fingerprint & caching
# -------------------------
def _graph_fingerprint(G: nx.DiGraph) -> str:
    """
    Cheap fingerprint: counts + simple sums; good enough to know when to invalidate cache.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    # sum of edge likelihoods (bounded) + some node likelihoods
    sum_e = 0.0
    for u, v, d in G.edges(data=True):
        try:
            sum_e += float(d.get("likelihood", 0.0))
        except Exception:
            pass
    sum_n = 0.0
    for nid, nd in G.nodes(data=True):
        try:
            sum_n += float(nd.get("likelihood", 0.0))
        except Exception:
            pass
    raw = f"{n}|{m}|{round(sum_e,6)}|{round(sum_n,6)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def _cache_key(product_id: str, kind: str, combo_ids: Optional[List[str]] = None) -> str:
    """
    kind: 'baseline' or 'combo'
    combo_ids: list of node ids used as engaged_nodes when kind='combo'
    """
    if kind == "baseline":
        return f"icp_uplifts::{product_id}::baseline"
    combo_str = "|".join(sorted(combo_ids or []))
    return f"icp_uplifts::{product_id}::combo::{combo_str}"

def _load_cached_rcs(product_id: str, attribute_dict = {}, key: str = None) -> Optional[Dict]:
    return load_rcs_from_json(product_id=product_id, attribute_dict=attribute_dict, zmot_id=key)

def _save_cached_rcs(product_id: str, attribute_dict: dict, key: str, causal_graph: nx.DiGraph, rcs_report: Dict) -> None:
    save_rcs_as_json(product_id=product_id, attribute_dict=attribute_dict, causal_graph=causal_graph, rcs_report=rcs_report, zmot_id=key)

# -------------------------
# Attribute chip helpers
# -------------------------
@dataclass(frozen=True)
class Chip:
    family: str      # e.g., 'industry'
    node_id: str     # graph node id, e.g., 'icp_industry:saas'
    label: str       # human label, e.g., 'SaaS'

def _collect_attribute_chips(G: nx.DiGraph) -> Dict[str, List[Chip]]:
    """
    Return dict family -> list of Chip for each attribute family that exists in this graph.
    """
    out: Dict[str, List[Chip]] = {}
    for family, _ in ATTR_FAMILIES:
        chips: List[Chip] = []
        for nid, node in G.nodes(data=True):
            if node.get("type") == "attribute_value" and node.get("dimension") == family:
                label = (
                    node.get("label") or node.get("title") or node.get("name") or
                    node.get("industry") or node.get("revenue_range") or
                    node.get("employee_range") or node.get("funding_stage") or
                    node.get("geography") or str(nid)
                )
                chips.append(Chip(family=family, node_id=nid, label=str(label)))
        if chips:
            out[family] = chips
    print("Collected attribute chips by family:", {k: [c.label for c in v] for k, v in out.items()})
    return out

def _valid_combos(chips_by_family: Dict[str, List[Chip]],
                  max_len: int = MAX_COMBO_LEN) -> List[List[Chip]]:
    """
    Build combos up to max_len, selecting at most 1 chip from each family.
    """
    # Flatten: families that have at least one chip
    families = [f for f in chips_by_family.keys() if chips_by_family[f]]
    # Singletons
    candidates: List[List[Chip]] = [[c] for f in families for c in chips_by_family[f]]

    # Pairs / Triples ...
    for r in range(2, max_len + 1):
        # choose r distinct families, then 1 chip per chosen family
        for fam_subset in itertools.combinations(families, r):
            pools = [chips_by_family[f] for f in fam_subset]
            for pick in itertools.product(*pools):
                candidates.append(list(pick))
    return candidates

# -------------------------
# Core evaluators
# -------------------------
def _evaluate_win_for_combo(G: nx.DiGraph, combo: List[Chip]) -> Tuple[float, Dict, nx.DiGraph]:
    """
    Use generate_rcs with engaged_nodes = combo node ids, retrieve win likelihood.
    """
    engaged = [{"id": c.node_id, "occurrence": 1.0} for c in combo]
    causal_graph, report = generate_rcs(G, engaged_nodes=engaged)
    win = float(causal_graph.graph.get("win_likelihood", 0.0))
    return win, report, causal_graph

def _evaluate_or_load_cached(G: nx.DiGraph, product_id: str, combo: List[Chip]) -> Tuple[float, Dict]:
    """
    Cache each combo’s RCS by a stable key to avoid recomputation.
    """
    key = _cache_key(product_id, "combo", [c.node_id for c in combo])
    cached = _load_cached_rcs(product_id, attribute_dict={}, key=key)
    if cached:
        cg = cached.get("causal_graph")  # not strictly needed here
        rep = cached.get("rcs_report") or {}
        win = float(
            (cg.graph.get("win_likelihood") if cg else None)
            or rep.get("baseline", {}).get("baseline_win_conditioned", 0.0)
        )
        # The saved structure from your save_rcs_as_json may differ;
        # fall back to the report->baseline if graph meta isn’t present.
        return win, rep

    win, rep, cg = _evaluate_win_for_combo(G, combo)
    _save_cached_rcs(product_id, {}, key, cg, rep)
    return win, rep

def _evaluate_or_load_baseline(G: nx.DiGraph, product_id: str) -> Tuple[float, Dict]:
    key = _cache_key(product_id, "baseline")
    cached = _load_cached_rcs(product_id, {}, key)
    if cached:
        cg = cached.get("causal_graph")
        rep = cached.get("rcs_report") or {}
        win = float(
            (cg.graph.get("win_likelihood") if cg else None)
            or rep.get("baseline", {}).get("baseline_win_conditioned", 0.0)
        )
        return win, rep

    # Run once with full graph, no archetype and no engaged nodes
    cg, rep = generate_rcs(G, engaged_nodes=None)
    win = float(cg.graph.get("win_likelihood", 0.0))
    _save_cached_rcs(product_id, {}, key, cg, rep)
    return win, rep

# -------------------------
# Public API
# -------------------------
def mine_icp_attribute_uplifts(
    sub_graph: nx.DiGraph,
    *,
    max_combo_len: int = MAX_COMBO_LEN,
    top_k: int = TOP_K
) -> Dict:
    """
    1) Baseline win on the full graph
    2) Evaluate attribute combos (singletons..max_combo_len) as engaged_nodes
    3) Return sorted list by absolute lift, with clean chip labels for the UI.

    NOTE: This is graph-driven. If you pass empirical data later, you can extend
    the payload with coverage/CI without changing the UI schema.
    """
    if sub_graph.number_of_nodes() == 0:
        return {"error": "empty_graph"}

    product_id = get_product_id_from_subgraph(sub_graph)
    if not product_id:
        return {"error": "no_product_node"}

    # Invalidation check: if graph fingerprint changed, drop combo cache
    fp = _graph_fingerprint(sub_graph)

    # Baseline (cached)
    base_win, _ = _evaluate_or_load_baseline(sub_graph, product_id)

    # Attribute chips
    chips_by_family = _collect_attribute_chips(sub_graph)
    combos = _valid_combos(chips_by_family, max_combo_len)

    results = []
    for combo in combos:
        # Don’t allow multiple chips from same family (guard, even though generator avoids this)
        fams = {c.family for c in combo}
        if len(fams) != len(combo):
            continue

        win, _ = _evaluate_or_load_cached(sub_graph, product_id, combo)
        lift_abs = float(win) - float(base_win)
        if lift_abs <= 0.0:
            # keep only uplifts; if you want, retain all and let UI filter
            continue

        results.append({
            "chips": [{"family": c.family, "node_id": c.node_id, "label": c.label} for c in combo],
            "win_rate": round(win, 6),
            "lift_abs": round(lift_abs, 6),
            "lift_rel": round((win / base_win) if base_win > 0 else 0.0, 6),
        })
    print("ZMOT ICP Results:", results)
    # Sort by absolute lift, then relative lift, then by #chips (more specific later)
    results.sort(key=lambda r: (r["lift_abs"], r["lift_rel"], len(r["chips"])), reverse=True)
    results = results[:top_k]

    payload = {
        "graph_fingerprint": fp,
        "baseline": {
            "win_rate": round(float(base_win), 6)
        },
        "combos": results
    }
    print("ICP Payload:", json.dumps(payload, indent=2))
    return payload

# -------------------------
# Optional: explicit cache invalidation
# -------------------------
def invalidate_icp_uplifts_cache(sub_graph: nx.DiGraph) -> Dict[str, str]:
    """
    If you keep a dedicated key-value store for save_rcs_as_json, add a method to wipe
    keys prefixed with 'icp_uplifts::<product_id>::'.
    Here we just return the prefix to delete (so your store layer can act on it).
    """
    product_id = get_product_id_from_subgraph(sub_graph)
    if not product_id:
        return {"error": "no_product_node"}
    prefix = f"icp_uplifts::{product_id}::"
    return {"prefix_to_delete": prefix}
# ============================

# -------------------------
# ZMOT discovery for attribute combos
# -------------------------

def _noisy_or_merge(a: float, b: float) -> float:
    a = max(0.0, min(1.0, float(a)))
    b = max(0.0, min(1.0, float(b)))
    return 1.0 - (1.0 - a) * (1.0 - b)


def collect_zmots_for_attribute_combo(
    G: nx.DiGraph,
    combo: List[Chip],
    *,
    min_edge_score: float = 0.0,
    top_k_zmots: int = 20,
    top_k_moments: int = 8,
    top_k_keywords: int = 12
) -> List[Dict]:
    """
    For a set of attribute chips (e.g., industry=SaaS, geo=APAC), aggregate ZMOTs reachable from ANY chip
    via edges of type 'relevant_event'. Aggregate the chip->ZMOT edge scores with noisy-OR.
    For each ZMOT, also return its observable moments ('observed_in') and keywords ('associated_with').

    Returns a list sorted by aggregate 'score' desc:
      [{
        "zmot_event_id": "...",
        "zmot_label": "...",
        "score": 0.42,                     # noisy-OR over chip edges
        "from_chips": [
          {"node_id": chip_id, "label": "...", "edge_score": 0.3},
          ...
        ],
        "observable_moments": [{"id": "...", "label": "...", "fitness": 0.55}, ...],
        "trigger_keywords":  [{"id": "...", "label": "...", "fitness": 0.61}, ...]
      }, ...]
    """
    # Lazy imports to reuse your helpers without circulars
    from backend.utils.graph_base.network_graph import (
        get_edge_attribute, get_node_by_id, get_target_nodes_by_source_and_type
    )

    zmot_acc: Dict[str, Dict] = {}  # zmot_id -> {score, contribs:[], ...}

    # 1) Gather ZMOTs from each chip and aggregate edge scores
    for chip in combo:
        chip_id = chip.node_id
        if chip_id not in G:
            continue
        zmot_ids = get_target_nodes_by_source_and_type(G, chip_id, "relevant_event") or []
        for zmot_id in zmot_ids:
            # prefer 'likelihood', fallback to 'relevance'
            e = get_edge_attribute(G, chip_id, zmot_id, "likelihood")
            if e is None:
                e = get_edge_attribute(G, chip_id, zmot_id, "relevance")
            edge_score = float(e or 0.0)
            if edge_score < min_edge_score:
                continue
            # init bucket
            if zmot_id not in zmot_acc:
                zmot_acc[zmot_id] = {"score": 0.0, "from_chips": []}
            # aggregate via noisy-OR
            zmot_acc[zmot_id]["score"] = _noisy_or_merge(zmot_acc[zmot_id]["score"], edge_score)
            zmot_acc[zmot_id]["from_chips"].append({
                "node_id": chip_id,
                "label": chip.label,
                "edge_score": round(edge_score, 6)
            })

    if not zmot_acc:
        return []

    # 2) Attach labels, observable moments, and keywords for each ZMOT
    def _collect_children(zid: str, rel_type: str, label_field_hint: str) -> List[Dict]:
        out = []
        ids = get_target_nodes_by_source_and_type(G, zid, rel_type) or []
        for cid in ids:
            # prefer 'relevance', fallback to 'likelihood'
            f = get_edge_attribute(G, zid, cid, "relevance")
            if f is None:
                f = get_edge_attribute(G, zid, cid, "likelihood")
            fitness = float(f or 0.0)
            out.append({
                "id": cid,
                "label": _set_node_label(G, cid),
                "fitness": round(fitness, 6)
            })
        out.sort(key=lambda r: r["fitness"], reverse=True)
        return out

    rows: List[Dict] = []
    for zmot_id, meta in zmot_acc.items():
        if zmot_id not in G:
            # skip dangling ids (shouldn't happen but safe)
            continue

        zmot_node = get_node_by_id(G, zmot_id) or {}
        zmot_label = _set_node_label(G, zmot_id)

        observable_moments = _collect_children(zmot_id, "observed_in", "text")[:top_k_moments]
        trigger_keywords  = _collect_children(zmot_id, "associated_with", "keyword")[:top_k_keywords]

        rows.append({
            "zmot_event_id": zmot_id,
            "zmot_label": str(zmot_label),
            "score": round(float(meta["score"]), 6),
            "from_chips": meta["from_chips"],
            "observable_moments": observable_moments,
            "trigger_keywords": trigger_keywords
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_k_zmots]

def get_attribute_options(G: nx.DiGraph) -> dict:
    """
    Returns a dict of attribute family -> list of {id, label} for all attribute_value nodes in the graph.
    """
    families = ["industry", "revenue_range", "employee_range", "funding_stage", "geography"]
    out = {}
    for fam in families:
        nodes = [
            {
                "id": nid,
                "label": (
                    node.get("label") or node.get("title") or node.get("name") or
                    node.get("industry") or node.get("revenue_range") or
                    node.get("employee_range") or node.get("funding_stage") or
                    node.get("geography") or str(nid)
                )
            }
            for nid, node in G.nodes(data=True)
            if node.get("type") == "attribute_value" and node.get("dimension") == fam
        ]
        if nodes:
            out[fam] = nodes
    return out
