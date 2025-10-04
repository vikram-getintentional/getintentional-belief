from typing import Dict, List, Any, Optional
import json
from collections import defaultdict

from backend.utils.graph_base.network_graph import calculate_soft_or_relevance, get_node_by_id, get_nodes_list, get_product_id_from_subgraph, get_source_nodes_by_target_and_type
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data


from backend.utils.graph_base.graph_data.rcs_utils.save_and_load_rcs import load_rcs_from_json, save_rcs_as_json
from backend.utils.inference.rcs_generators.generate_rcs_fast import generate_rcs

import networkx as nx

# ---------------- Utils ----------------

def get_company_products(base_graph: nx.DiGraph, company_id: str) -> List[Dict[str, Any]]:
    return [
        {"id": node_id, "url": node_data.get("url")}
        for node_id, node_data in get_nodes_list(base_graph, "product", {"company_id": company_id})
    ]

def _persona_label(sub_graph: nx.DiGraph, persona_id: str) -> Dict[str, str]:
    n = get_node_by_id(sub_graph, persona_id) or {}
    return {
        "title": n.get("title") or n.get("name") or "",
        "department": n.get("department") or "",
        "seniority": n.get("seniority") or "",
    }

def _rcs_cache_key_for_personas(product_id: str) -> str:
    # we use the zmot_id slot for namespacing
    return f"persona_tab::baseline"

# ---------------- NEW: RCS aggregation ----------------

def aggregated_personas_rcs(
    cards: List[Dict[str, Any]],
    *,
    group_on: tuple = ("title", "department", "seniority"),
    min_priority: float = 0.0,
    top_k: int | None = None
) -> List[Dict[str, Any]]:
    """
    Aggregate per-node persona cards into one card per logical persona.
    Grouping key defaults to (title, department, seniority).

    For each group:
      - importance, activation, care, marginal_lift: max
      - priority_score: recomputed = max_importance * max_activation
      - jobs/pains: union by description with max relevance, sorted desc
      - persona_ids: union of ids
    Output shape mirrors your older aggregated format but with RCS metrics added.
    """
    buckets: Dict[tuple, Dict[str, Any]] = {}

    for c in cards or []:
        meta = c.get("persona", {}) or {}
        key = tuple(meta.get(k, "") for k in group_on)

        if key not in buckets:
            buckets[key] = {
                "persona_title": meta.get("title", ""),
                "persona_departments": set([meta.get("department", "")]) if meta.get("department") else set(),
                "persona_seniority": set([meta.get("seniority", "")]) if meta.get("seniority") else set(),
                "persona_ids": set(),
                "importance": 0.0,
                "activation": 0.0,
                "care": 0.0,
                "marginal_lift": 0.0,
                "priority_score": 0.0,
                "jobs": {},   # desc -> max relevance
                "pains": {},  # desc -> max relevance
            }

        agg = buckets[key]
        agg["persona_ids"].add(c.get("persona_id"))
        if meta.get("department"): agg["persona_departments"].add(meta["department"])
        if meta.get("seniority"):  agg["persona_seniority"].add(meta["seniority"])

        # max aggregation for metrics
        imp = float(c.get("importance", 0.0) or 0.0)
        act = float(c.get("activation", 0.0) or 0.0)
        care = float(c.get("care", 0.0) or 0.0)
        ml   = float(c.get("marginal_lift", 0.0) or 0.0)

        agg["importance"]   = max(agg["importance"], imp)
        agg["activation"]   = max(agg["activation"], act)
        agg["care"]         = max(agg["care"], care)
        agg["marginal_lift"] = max(agg["marginal_lift"], ml)

        # union jobs/pains by description, keeping max relevance
        for j in (c.get("jobs") or []):
            desc = (j.get("description") or "").strip()
            if not desc: continue
            r = float(j.get("relevance", 0.0) or 0.0)
            agg["jobs"][desc] = max(agg["jobs"].get(desc, 0.0), r)

        for p in (c.get("pains") or []):
            desc = (p.get("description") or "").strip()
            if not desc: continue
            r = float(p.get("relevance", 0.0) or 0.0)
            agg["pains"][desc] = max(agg["pains"].get(desc, 0.0), r)

    # materialize & finalize
    out: List[Dict[str, Any]] = []
    for agg in buckets.values():
        # recompute priority on max signals
        agg["priority_score"] = float(agg["importance"]) * float(agg["activation"])

        if agg["priority_score"] < min_priority:
            continue

        out.append({
            "persona_title": agg["persona_title"],
            "persona_departments": sorted(list(agg["persona_departments"])),
            "persona_seniority": sorted(list(agg["persona_seniority"])),
            "persona_ids": sorted(list(pid for pid in agg["persona_ids"] if pid)),

            "importance": float(agg["importance"]),
            "activation": float(agg["activation"]),
            "care": float(agg["care"]),
            "marginal_lift": float(agg["marginal_lift"]),
            "priority_score": float(agg["priority_score"]),

            "jobs": sorted(
                [{"description": d, "relevance": r} for d, r in agg["jobs"].items()],
                key=lambda x: x["relevance"], reverse=True
            ),
            "pains": sorted(
                [{"description": d, "relevance": r} for d, r in agg["pains"].items()],
                key=lambda x: x["relevance"], reverse=True
            ),
        })

    out.sort(key=lambda c: c["priority_score"], reverse=True)
    if top_k is not None:
        return out[:top_k]
    return out

# ---------------- RCS Personas (calls aggregation) ----------------

def get_personas_rcs_priority(sub_graph: nx.DiGraph, attribute_dict: Optional[dict] = None, relevance_threshold: float = 0.0, top_k: int = 50) -> list[dict]:
    """
    Ranks personas by Importance × Activation using the RCS engine:
      importance := involvement (from generate_rcs)
      activation := activation (from generate_rcs)
    Returns AGGREGATED persona cards (one per logical persona).
    """
    if sub_graph.number_of_nodes() == 0:
        return []

    product_id = get_product_id_from_subgraph(sub_graph)
    if not product_id:
        return []
    attribute_dict = attribute_dict or {}
    # ---- 1) Load or compute RCS once (baseline, no engaged nodes) ----
    cache_key = _rcs_cache_key_for_personas(product_id)
    cached = load_rcs_from_json(product_id=product_id, attribute_dict=attribute_dict, zmot_id=cache_key)

    if cached:
        causal_graph = cached["causal_graph"]
        rcs_report = cached["rcs_report"]
    else:
        causal_graph, rcs_report = generate_rcs(sub_graph, engaged_nodes=None)
        save_rcs_as_json(product_id=product_id, attribute_dict={},
                         causal_graph=causal_graph, rcs_report=rcs_report, zmot_id=cache_key)

    # ---- 2) Collect per-persona metrics from the report ----
    top_block = (rcs_report or {}).get("top_personas", {}) or {}
    by_involvement = {r["id"]: r for r in top_block.get("by_involvement", [])}
    by_activation  = {r["id"]: r for r in top_block.get("by_activation", [])}
    by_lift        = {r["id"]: r for r in top_block.get("by_marginal_lift", [])}

    metrics: Dict[str, Dict[str, float]] = {}
    for pid in set(list(by_involvement.keys()) + list(by_activation.keys()) + list(by_lift.keys())):
        inv = float(by_involvement.get(pid, {}).get("involvement", 0.0))
        act = float(by_activation.get(pid, {}).get("activation", by_involvement.get(pid, {}).get("activation", 0.0)))
        care = float(by_involvement.get(pid, {}).get("care", by_activation.get(pid, {}).get("care", 0.0)))
        mlift = float(by_lift.get(pid, {}).get("marginal_lift", 0.0))
        metrics[pid] = {
            "involvement": inv,
            "activation": act,
            "care": care,
            "marginal_lift": mlift,
            "priority_score": inv * act
        }

    # ---- 3) Build per-node persona cards (keep for aggregation) ----
    node_cards: List[Dict[str, Any]] = []
    for persona_id, _ in get_nodes_list(sub_graph, "persona", {}):
        m = metrics.get(persona_id, {"involvement": 0.0, "activation": 0.0, "care": 0.0, "marginal_lift": 0.0, "priority_score": 0.0})
        meta = _persona_label(sub_graph, persona_id)

        jobs, pains = [], []
        persona_jobs = get_source_nodes_by_target_and_type(sub_graph, persona_id, "performed_by")
        for job_id in persona_jobs:
            job_node = get_node_by_id(sub_graph, job_id) or {}
            job_text = job_node.get("description") or job_node.get("text") or ""
            job_rel = get_cumulative_relevance_data(product_id, job_id)
            jobs.append({"description": job_text, "relevance": job_rel})

            pain_ids = get_source_nodes_by_target_and_type(sub_graph, job_id, "felt_in")
            for pain_id in pain_ids:
                pain_node = get_node_by_id(sub_graph, pain_id) or {}
                pain_text = pain_node.get("description") or pain_node.get("text") or ""
                pain_rel = get_cumulative_relevance_data(product_id, pain_id)
                pains.append({"description": pain_text, "relevance": pain_rel})

        node_cards.append({
            "persona_id": persona_id,
            "persona": meta,
            "importance": m["involvement"],
            "activation": m["activation"],
            "care": m["care"],
            "marginal_lift": m["marginal_lift"],
            "priority_score": m["priority_score"],
            "jobs": sorted(jobs, key=lambda x: x["relevance"], reverse=True),
            "pains": sorted(pains, key=lambda x: x["relevance"], reverse=True),
        })

    # ---- 4) Aggregate to remove duplicates (title/department/seniority) ----
    aggregated = aggregated_personas_rcs(
        node_cards,
        group_on=("title", "department", "seniority"),
        min_priority=relevance_threshold,
        top_k=top_k
    )
    return aggregated

