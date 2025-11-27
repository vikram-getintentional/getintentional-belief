# plan_manager.py
from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import date


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

# Back-compat wrapper name used by older routes
def build_integrated_portfolio_plan(product_id: str, account_id: Optional[str] = None) -> Dict[str, Any]:
    from backend.utils.strategy_builder.comprehensive_plan_generator import build_product_marketing_plan

    return build_product_marketing_plan(product_id, account_id=account_id)
