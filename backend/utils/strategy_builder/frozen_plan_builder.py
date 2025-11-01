# frozen_plan_builder.py
from __future__ import annotations
from typing import Any, Dict, List

from backend.utils.strategy_builder.belief_math import clamp01

def summarize_phase_belief(phases: List[Dict[str, Any]]) -> float:
    if not phases:
        return 0.0
    vals = []
    for ph in phases:
        sb = ph.get("stage_block") or {}
        vals.append(float(sb.get("belief_after", 0.0)))
    return clamp01(sum(vals) / max(1, len(vals)))

def to_portfolio_themes(account_scaffolds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    themes: List[Dict[str, Any]] = []
    for i, sc in enumerate(account_scaffolds):
        acct = sc.get("account_name") or sc.get("account_id") or f"acct_{i+1}"
        zmot = sc.get("zmot_theme") or ""
        phases = sc.get("execution_plan", {}).get("campaigns") or []
        campaigns = sc.get("execution_plan", {}).get("campaigns") or []
        themes.append({
            "id": f"theme_{i+1:02d}",
            "name": zmot or f"Belief Lift for {acct}",
            "explanation": f"Progress belief for {acct} via prioritized concerns.",
            "objective": "Reach proximity with economic and technical buyers.",
            "targetAccounts": [acct],
            "campaigns": campaigns,
        })
    return themes
