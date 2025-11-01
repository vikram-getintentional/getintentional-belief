# plan_manager.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import date, datetime

from backend.utils.strategy_builder.belief_math import belief_summary_meta


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None

def filter_plan_by_window(plan: Dict[str, Any], window_start: Optional[str], window_end: Optional[str]) -> Dict[str, Any]:
    """Non-destructive: filter campaigns by timeframe in plan['themes'][*]['campaigns']."""
    if not (window_start or window_end):
        return plan
    s = _parse_date(window_start)
    e = _parse_date(window_end)
    if not (s or e):
        return plan

    def in_window(tf: Dict[str, str]) -> bool:
        st = _parse_date((tf or {}).get("startDate"))
        en = _parse_date((tf or {}).get("endDate"))
        if s and st and en and en < s:
            return False
        if e and st and st > e:
            return False
        return True

    out = {**plan}
    new_themes = []
    for th in plan.get("themes", []):
        camps = [c for c in (th.get("campaigns") or []) if in_window(c.get("timeframe") or {})]
        th2 = {**th, "campaigns": camps}
        new_themes.append(th2)
    out["themes"] = new_themes
    return out

# alias to match older route code if it imports this name
def _filter_by_window(plan: Dict[str, Any], window_start: Optional[str], window_end: Optional[str]) -> Dict[str, Any]:
    return filter_plan_by_window(plan, window_start, window_end)

def build_integrated_portfolio_plan_v2(
    *,
    generated_at_iso: str,
    per_account_scaffolds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Aggregate quick stats
    total_accounts = len(per_account_scaffolds)
    all_personas = set()
    for sc in per_account_scaffolds:
        for p in sc.get("execution_plan", {}).get("personas", []):
            nm = p.get("name")
            if nm:
                all_personas.add(nm)

    # crude belief average for portfolio
    belief_vals = []
    for sc in per_account_scaffolds:
        bs = sc.get("execution_plan", {}).get("belief_state_summary", {}).get("average_belief_score")
        if isinstance(bs, (int, float)):
            belief_vals.append(float(bs))
    avg_belief = round(sum(belief_vals) / max(1, len(belief_vals)), 3)

    key_stats = {
        "totalTargetAccounts": total_accounts,
        "totalPersonasToEngage": len(all_personas),
        "expectedWinsPct": 0.0,   # left 0.0 until CRM calibration is wired
        "averageAccountBelief": f"{avg_belief:.2f}",
        "timeToWinMonths": 3,
    }

    from .frozen_plan_builder import to_portfolio_themes
    themes = to_portfolio_themes(per_account_scaffolds)

    plan = {
        "meta": {
            "version": "2.0",
            "generatedAt": generated_at_iso,
            "beliefScale": ["Cold", "Warming", "Engaged", "Primed"],
            **belief_summary_meta(avg_belief),
        },
        "portfolio": {
            "keyStats": key_stats
        },
        "themes": themes,
    }
    return plan

# Back-compat wrapper name used by your route
def build_integrated_portfolio_plan(product_id: str, account_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads per-account RCS scaffolds for product_id (or just `account_id`), returns plan in V2 schema.
    """
    from ..strategy_builder.comprehensive_plan_generator import construct_or_load_account_scaffolds
    from datetime import datetime

    scaffolds = construct_or_load_account_scaffolds(product_id, account_id=account_id)
    return build_integrated_portfolio_plan_v2(
        generated_at_iso=datetime.utcnow().isoformat(),
        per_account_scaffolds=scaffolds,
    )
