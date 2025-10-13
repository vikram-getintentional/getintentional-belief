# ============================
# File: backend/utils/inference/rcs_generators/rcs_helpers/strategy_orchestrators.py
# ============================
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import copy
import networkx as nx

from backend.utils.inference.rcs_generators.generate_rcs_fast import generate_rcs


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def _to_engaged_payload(ids: List[str]) -> List[Dict]:
    """Convert node IDs to the payload structure used by generate_rcs."""
    return [{"id": nid, "occurrence": 1.0} for nid in sorted(set(ids))]


# ---------------------------------------------------------------------
# BUILD STRATEGY  (Frozen)
# ---------------------------------------------------------------------
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
    Freezes:
      - Personas likely to be involved
      - Core concern families (coalitions)
      - Campaign sequences (themes/plays)
      - Portfolio policy for breadth/depth/mutation
    """
    engaged = (attributes or []) + (zmots or [])
    print("🎯 [build_account_strategy] calling generate_rcs with engaged nodes:", engaged)

    G_work, report = generate_rcs(
        product_subgraph,
        engaged_nodes=_to_engaged_payload(engaged),
        boost_factor=boost_factor,
    )

    frozen_strategy = {
        "graph_version": G_work.graph.get("version", "unknown"),
        "baseline": report.get("baseline", {}),
        "frozen_persona_pool": report["top_personas"]["by_involvement"] if "top_personas" in report else [],
        "themes": report.get("concern_sequences", []),
        "coalitions": report.get("concern_coalitions", []),
        "persona_coalitions": report.get("coalitions", []),
        "concern_backlog": report.get("concern_backlog", []),
        # initial portfolio bias (cold stage)
        "portfolio_policy": {"breadth": 0.6, "depth": 0.2, "mutation": 0.2},
        "seed_context": {
            "attributes": sorted(set(attributes or [])),
            "zmots": sorted(set(zmots or [])),
            "initial_engagements": sorted(set(initial_engagements or [])),
        },
    }

    return frozen_strategy, G_work


# ---------------------------------------------------------------------
# UPDATE STRATEGY  (Tactical)
# ---------------------------------------------------------------------
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
    Tactical refresh that updates priorities within a frozen strategy.

    Inputs:
      - frozen_strategy: the baseline snapshot returned by build_account_strategy()
      - attributes, zmots, engagements_to_date: cumulative engagement so far

    Outputs:
      - Updated portfolio tilt
      - Recommended next concerns/sequences
      - Current stage label ("cold", "warm", "hot")
    """
    engaged = (attributes or []) + (zmots or []) + (engagements_to_date or [])
    print("⚙️ [update_tactical_plan] calling generate_rcs with engaged nodes:", engaged)

    G_work, report = generate_rcs(
        product_subgraph,
        engaged_nodes=_to_engaged_payload(engaged),
        boost_factor=boost_factor,
    )

    # ----------------------------------------
    # Stage inference (auto if not specified)
    # ----------------------------------------
    if stage == "auto":
        has_eng = bool(engagements_to_date)
        win_prob = report.get("baseline", {}).get("win_likelihood", 0.0) or 0.0
        stage = "hot" if win_prob >= 0.35 else ("warm" if has_eng else "cold")

    # ----------------------------------------
    # Adjust portfolio mix
    # ----------------------------------------
    base_policy = frozen_strategy.get("portfolio_policy", {"breadth": 0.6, "depth": 0.2, "mutation": 0.2})
    policy = copy.deepcopy(base_policy)

    if stage == "cold":
        policy.update({"breadth": 0.6, "depth": 0.2, "mutation": 0.2})
    elif stage == "warm":
        policy.update({"breadth": 0.35, "depth": 0.5, "mutation": 0.15})
    elif stage == "hot":
        policy.update({"breadth": 0.2, "depth": 0.7, "mutation": 0.1})

    # ----------------------------------------
    # Restrict to frozen persona pool
    # ----------------------------------------
    frozen_pids = []
    if frozen_strategy.get("frozen_persona_pool"):
        for p in frozen_strategy["frozen_persona_pool"]:
            if isinstance(p, str):
                frozen_pids.append(p)
            elif isinstance(p, dict):
                pid = p.get("id")
                if pid:
                    frozen_pids.append(pid)
    concern_backlog = [
        x for x in report.get("concern_backlog", [])
        if x.get("pid") in frozen_pids
    ][:30]

    # ----------------------------------------
    # Theme re-ranking (dynamic theme lift)
    # ----------------------------------------
    frozen_themes = frozen_strategy.get("themes", [])
    live_sequences = report.get("concern_sequences", [])

    def keyify(s): return (s["persona"], s["concern_id"])
    frozen_keys = [{keyify(s) for s in t.get("sequence", [])} for t in frozen_themes]

    reranked_themes = []
    for idx, theme in enumerate(frozen_themes):
        fkeys = frozen_keys[idx]
        overlap = sum(1 for c in concern_backlog if (c["pid"], c["cid"]) in fkeys)
        lift_now = theme.get("final_win", 0.0) + 0.05 * overlap
        reranked_themes.append((lift_now, theme))

    reranked_themes.sort(key=lambda x: x[0], reverse=True)
    top_theme = reranked_themes[0][1] if reranked_themes else None

    # ----------------------------------------
    # Tactical picks for next sprint
    # ----------------------------------------
    next_concerns = concern_backlog[:8]
    next_sequences = live_sequences[:3] if live_sequences else []

    # ----------------------------------------
    # Final structured output
    # ----------------------------------------
    out = {
        "graph_win_likelihood": report.get("baseline", {}).get("win_likelihood", 0.0),
        "stage": stage,
        "portfolio_policy": policy,
        "top_theme": top_theme,
        # Strategic context (unchanged)
        "frozen_personas": frozen_strategy.get("frozen_persona_pool", []),
        # Tactical focus
        "next_concerns": next_concerns,
        "next_sequences": next_sequences,
        # Pass-through data for UI
        "top_N_personas": report.get("top_personas", {}).get("by_involvement", [])[:10],
        "concerns_by_persona": report.get("concerns_by_persona", []),
        "coalitions": report.get("coalitions", []),
        "concern_coalitions": report.get("concern_coalitions", []),
        "concern_sequences": next_sequences,
        "concern_backlog": concern_backlog,
        "causal_flows": report.get("causal_flows", []),
        "sequences_and_campaigns": report.get("sequences_and_campaigns", []),
    }

    return out, G_work
