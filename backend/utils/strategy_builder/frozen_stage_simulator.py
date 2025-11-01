# frozen_stage_simulator.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import networkx as nx

from .belief_math import org_belief, expected_belief_delta, stage_gain_coeff

def _mk_stage_block(label: str, personas: List[str], pains: List[str], why: str, before: float) -> Dict[str, Any]:
    gain = expected_belief_delta(asset_fit=0.55, channel_fit=0.55, stage_factor=stage_gain_coeff(label))
    after = min(1.0, before + gain)
    return {
        "stage": label,
        "trigger": "",
        "belief_before": round(before, 3),
        "belief_after": round(after, 3),
        "personas": personas,
        "pains": pains,
        "evidence": [],
        "recommended_assets": [],
        "why": why,
    }

def _infer_seed_belief(report: Dict[str, Any]) -> float:
    ps = report.get("persona_scores") or {}
    if not ps:
        return 0.25
    vals = []
    for row in ps.values():
        p = float(row.get("perceptibility", 0.4) or 0.4)
        r = float(row.get("proximity", 0.3) or 0.3)
        i = float(row.get("involvement", 0.4) or 0.4)
        vals.append(org_belief(p, r, i))
    return sum(vals) / max(1, len(vals))

def _top_personas(report: Dict[str, Any], topk: int = 3) -> List[str]:
    ps = report.get("persona_scores") or {}
    ranked = sorted(ps.items(), key=lambda kv: kv[1].get("importance", 0.5), reverse=True)
    return [k for k,_ in ranked[:topk]]

def _top_pains(report: Dict[str, Any], topk: int = 3) -> List[str]:
    bl = report.get("concern_backlog") or []
    ranked = sorted(bl, key=lambda r: r.get("priority", 0.5), reverse=True)
    return [r.get("concern_label") or r.get("label") or "Concern" for r in ranked[:topk]]

def build_frozen_strategy(
    *,
    product_subgraph: nx.DiGraph,
    report: Dict[str, Any],
    account_id: str,
    product_id: str,
    archetype: Optional[Dict[str, Any]] = None,
    limits: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a list of phases; each phase has campaigns (empty initially).
    Downstream `attach_arsenal_tables_to_phases` will populate `campaigns[*].arsenalTable`.
    """
    del product_subgraph, product_id, archetype, limits, account_id  # not used here yet
    seed_belief = _infer_seed_belief(report)

    personas = _top_personas(report)
    pains = _top_pains(report)

    phases: List[Dict[str, Any]] = []

    # Phase 1: ZMOT / early discovery
    p1 = _mk_stage_block(
        "zmot" if report.get("zmot_theme") else "pre_zmot",
        personas=personas,
        pains=pains,
        why="Surface a sharp moment that makes the status quo untenable for economic + tech buyers.",
        before=seed_belief,
    )
    phases.append({"name": "Phase 1", "stage": p1["stage"], "stage_block": p1, "campaigns": []})

    # Phase 2: Problem Realization / Discovery
    p2 = _mk_stage_block(
        "problem_realization",
        personas=personas,
        pains=pains[:2],
        why="Quantify the business pain and link it to a concrete, owned job-to-be-done.",
        before=p1["belief_after"],
    )
    phases.append({"name": "Phase 2", "stage": p2["stage"], "stage_block": p2, "campaigns": []})

    # Phase 3: Barriers (objections, risks)
    p3 = _mk_stage_block(
        "barriers",
        personas=personas,
        pains=pains[:1],
        why="Tackle perceived risk and inertia; give the champion assets to win internal debates.",
        before=p2["belief_after"],
    )
    phases.append({"name": "Phase 3", "stage": p3["stage"], "stage_block": p3, "campaigns": []})

    # Add a minimal campaign shell per phase (filled later)
    for ph in phases:
        ph["campaigns"] = [{
            "id": f"{ph['name'].lower().replace(' ', '_')}_camp_01",
            "description": f"Resolve prioritized concerns in {ph['stage'].title()}",
            "timeframe": {},       # filled by orchestrator or plan manager if needed
            "personas": personas,  # default; can be narrowed
            "arsenalTable": [],    # will be filled by attach_arsenal_tables_to_phases
        }]

    return phases
