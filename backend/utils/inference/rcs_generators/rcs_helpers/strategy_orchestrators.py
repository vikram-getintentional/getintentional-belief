# ============================
# File: backend/utils/inference/rcs_generators/rcs_helpers/strategy_orchestrators.py
# ============================
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import copy
import networkx as nx
from backend.utils.inference.rcs_generators.generate_rcs_fast import generate_rcs

def _to_engaged_payload(ids: List[str]) -> List[Dict]:
    return [{"id": nid, "occurrence": 1.0} for nid in sorted(set(ids))]

def build_account_strategy(
    product_subgraph: nx.DiGraph,
    *,
    attributes: Optional[List[str]] = None,
    zmots: Optional[List[str]] = None,
    initial_engagements: Optional[List[str]] = None,
    boost_factor: float = 2.0,
) -> Dict:
    """
    One-time, *frozen* strategy snapshot for an account/product.
    Freezes: themes, concern families, coalition pool, initial portfolio split.
    """
    engaged = (attributes or []) + (zmots or []) + (initial_engagements or [])
    print("Hitting generate_rcs from strategy builder with inputs:", engaged)
    G_work, report = generate_rcs(
        product_subgraph,
        engaged_nodes=_to_engaged_payload(engaged),
        boost_factor=boost_factor,
    )

    # Freeze the strategic pieces you want stable for the account lifecycle.
    frozen = {
        "graph_version": G_work.graph.get("version", "unknown"),
        "baseline": report.get("baseline", {}),
        "frozen_persona_pool": [p["id"] for p in report.get("top_personas", {}).get("by_involvement", [])[:20]],
        "themes": report.get("concern_sequences", []),          # ordered concern sequences (themes/plays)
        "coalitions": report.get("concern_coalitions", []),     # concern coalitions (simultaneous)
        "persona_coalitions": report.get("coalitions", []),     # persona coalitions
        "concern_backlog": report.get("concern_backlog", []),   # flat ranked (pid,cid,stage,proxy)
        # initial portfolio mix (cold accounts -> breadth heavy)
        "portfolio_policy": {"breadth": 0.6, "depth": 0.2, "mutation": 0.2},
        # store the initial engaged context so we can diff later
        "seed_context": {
            "attributes": sorted(set(attributes or [])),
            "zmots": sorted(set(zmots or [])),
            "initial_engagements": sorted(set(initial_engagements or [])),
        },
    }
    return frozen

def update_tactical_plan(
    product_subgraph: nx.DiGraph,
    frozen_strategy: Dict,
    *,
    attributes: Optional[List[str]] = None,
    zmots: Optional[List[str]] = None,
    engagements_to_date: Optional[List[str]] = None,
    stage: str = "auto",  # "cold" | "warm" | "hot" | "auto"
    boost_factor: float = 2.0,
) -> Dict:
    """
    Refresh *within* the frozen strategy. Does NOT change frozen themes; it re-ranks and
    picks next best concern/cards & campaign blocks given latest engagement.
    """
    engaged = (attributes or []) + (zmots or []) + (engagements_to_date or [])
    print("Hitting rcs_generate with engaged notes from tactical:", engaged)
    G_work, report = generate_rcs(
        product_subgraph,
        engaged_nodes=_to_engaged_payload(engaged),
        boost_factor=boost_factor,
    )

    # Stage-aware portfolio tilt (keeps overall policy shape but nudges weights)
    if stage == "auto":
        # naive auto: if we see any engagements, treat as warm; if conversion prob > threshold, hot
        has_eng = bool(engagements_to_date)
        win = report.get("baseline", {}).get("win_likelihood", 0.0) or 0.0
        stage = "hot" if win >= 0.35 else ("warm" if has_eng else "cold")

    base_policy = frozen_strategy.get("portfolio_policy", {"breadth": 0.6, "depth": 0.2, "mutation": 0.2})
    policy = copy.deepcopy(base_policy)
    if stage == "cold":
        policy.update({"breadth": 0.6, "depth": 0.2, "mutation": 0.2})
    elif stage == "warm":
        policy.update({"breadth": 0.35, "depth": 0.5, "mutation": 0.15})
    elif stage == "hot":
        policy.update({"breadth": 0.2, "depth": 0.7, "mutation": 0.1})

    # Restrict dynamic choices to frozen pools (keeps plan stable)
    frozen_pids = set(frozen_strategy.get("frozen_persona_pool", []))
    concern_backlog = [
        x for x in report.get("concern_backlog", [])
        if x.get("pid") in frozen_pids
    ][:30]

    # Pick “next best” sequence & concern set from frozen themes guided by current report
    frozen_themes = frozen_strategy.get("themes", [])
    live_sequences = report.get("concern_sequences", [])
    # align by (persona, concern) keys
    def keyify_step(s): return (s["persona"], s["concern_id"])
    frozen_keys = [{keyify_step(s) for s in t.get("sequence", [])} for t in frozen_themes]

    reranked_themes = []
    for idx, ftheme in enumerate(frozen_themes):
        fkeys = frozen_keys[idx]
        # score theme by overlap with current top concerns + current lift
        overlap = sum(1 for it in concern_backlog if (it["pid"], it["cid"]) in fkeys)
        score_now = ftheme.get("final_win", 0.0) + 0.05 * overlap
        reranked_themes.append((score_now, ftheme))
    reranked_themes.sort(key=lambda t: t[0], reverse=True)
    top_theme = reranked_themes[0][1] if reranked_themes else None

    # Tactical picks for the next sprint (cards the UI needs today)
    next_concerns = concern_backlog[:8]
    next_sequences = (live_sequences[:3] if live_sequences else [])  # keep it dynamic but bounded

    # Return a frontend-compatible payload (your simulator already expects these keys)
    out = {
        "graph_win_likelihood": report.get("baseline", {}).get("win_likelihood", 0.0),
        "top_N_personas": report.get("top_personas", {}).get("by_involvement", [])[:10],
        "prioritized_10_concerns": report.get("concerns_flat", [])[:10],
        "all_personas": report.get("top_personas", {}).get("by_involvement", []),
        "concerns_by_persona": report.get("concerns_by_persona", []),
        "coalitions": report.get("coalitions", []),
        "causal_flows": report.get("causal_flows", []),
        "sequences_and_campaigns": report.get("sequences_and_campaigns", []),
        "concern_backlog": concern_backlog,
        "concern_coalitions": report.get("concern_coalitions", []),
        "concern_sequences": next_sequences,   # bounded dynamic
        # Strategy-aware extras:
        "portfolio_policy": policy,
        "stage": stage,
        "top_theme": top_theme,
    }
    return out
