# backend/utils/strategy_builder/comprehensive_plan_generator.py
from __future__ import annotations

import json
import os
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
import statistics as stats

# --- GI imports (existing in your repo) ---
from backend.utils.crm_management.target_account_manager import get_target_account_ids
from backend.utils.graph_base.network_graph import build_product_graph, get_product_id_from_subgraph
from backend.utils.inference.rcs_generators.rcs_helpers.strategy_orchestrators import (
    rcs_for_target_account,
)

# -------------------------
# Paths & small utilities
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_DATA_DIR = os.path.join(BASE_DIR, "graph_base", "graph_data")
RCS_JSON_DIR = os.path.join(GRAPH_DATA_DIR, "rcs_json")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))

# -------------------------------------------------
# Load / Save RCS (kept compatible with your logs)
# -------------------------------------------------
def _rcs_path(product_id: str, account_id: str) -> str:
    return os.path.join(RCS_JSON_DIR, product_id, f"{account_id}.json")

def _save_rcs_json(product_id: str, account_id: str, rcs: Dict[str, Any]) -> None:
    folder = os.path.join(RCS_JSON_DIR, product_id)
    _ensure_dir(folder)
    with open(os.path.join(folder, f"{account_id}.json"), "w") as f:
        json.dump(rcs, f, indent=2)

def load_account_rcs_from_disk(product_id: str, account_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    p = _rcs_path(product_id, account_id)
    if not os.path.exists(p):
        return None, p
    with open(p, "r") as f:
        return json.load(f), p

# ----------------------------------------
# Campaign synthesis (fallback) from RCS
# ----------------------------------------
def _to_timeframe(idx: int) -> Dict[str, str]:
    # compact horizon: start today + 2w * idx, 10w duration
    start = date.today()
    return {
        "startDate": start.isoformat(),
        "endDate": start.replace(day=min(28, start.day)) .isoformat()
    }

def _synthesize_campaigns_from_sequences(rcs: Dict[str, Any]) -> List[Dict[str, Any]]:
    seqs = (rcs.get("rcs_brief") or rcs.get("report") or {}).get("concern_sequences") or []
    if not seqs:
        return []
    out: List[Dict[str, Any]] = []
    for i, seq in enumerate(seqs):
        steps = seq.get("sequence", []) or []
        personas = []
        rows = []
        for step in steps:
            pname = step.get("persona")
            if pname:
                personas.append(pname)
            label = step.get("label") or step.get("concern_label") or step.get("cid") or "Concern"
            rows.append({
                "asset": f"[TBD] Resolve “{label}”",
                "channels": "Email Nurture, LinkedIn Ads, Sales Assist",
                "fitment": step.get("stage") or "Concern resolution",
                "engagement": "Medium",
                "expectedLift": "+10%",
            })
        personas = sorted({p for p in personas if p})
        out.append({
            "id": f"camp_seq_{i:02d}",
            "description": f"Resolve prioritized concerns across {', '.join(personas[:2]) or 'key personas'}",
            "timeframe": _to_timeframe(i),
            "personas": personas,
            "arsenalTable": rows,
        })
    return out

# ----------------------------------------
# Belief metrics (portfolio summary)
# ----------------------------------------
def _avg_belief_band(avg: float) -> str:
    # map 0..1 to bands; tolerate 0..100 strings
    if avg > 1.5:  # assume it was given in %
        avg = avg / 100.0
    avg = _clamp01(avg)
    if avg < 0.25:
        return "Unaware"
    if avg < 0.5:
        return "Problem-aware"
    if avg < 0.75:
        return "Solution-aware"
    return "Most-aware"

def _belief_from_rcs(rcs: Dict[str, Any]) -> Optional[float]:
    # try persona_scores.avg or execution_plan.personas belief avg
    ps = ((rcs.get("rcs_brief") or {}).get("persona_scores") or {}) if rcs else {}
    vals: List[float] = []
    for row in ps.values():
        b = row.get("belief")
        if isinstance(b, (int, float)):
            vals.append(_clamp01(float(b)))
    if not vals:
        ppl = ((rcs.get("execution_plan") or {}).get("personas") or []) if rcs else []
        for p in ppl:
            b = p.get("belief")
            if isinstance(b, (int, float)):
                vals.append(_clamp01(float(b)))
    if not vals:
        return None
    return float(stats.mean(vals))

# ----------------------------------------
# Theme builders
# ----------------------------------------
def _theme_from_account_rcs(account_id: str, rcs: Dict[str, Any]) -> Dict[str, Any]:
    acct_label = rcs.get("account_name") or rcs.get("account") or account_id
    # prefer already prepared campaigns
    campaigns = ((rcs.get("execution_plan") or {}).get("campaigns") or [])[:]
    if not campaigns:
        # synthesize from sequences if blank
        campaigns = _synthesize_campaigns_from_sequences(rcs)

    return {
        "id": f"theme:{account_id}",
        "name": f"Plan for {acct_label}",
        "explanation": "Auto-constructed from account RCS",
        "objective": rcs.get("objectives", {}).get("winHypothesis", ""),
        "targetAccounts": [acct_label],
        "campaigns": campaigns,
    }

# ----------------------------------------
# Public helpers for the route
# ----------------------------------------
def build_integrated_portfolio_plan(product_id: str, account_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns the **v2** plan:
      { meta, portfolio: {keyStats}, themes: [...] }
    If account_id is provided, returns a single-account plan with one theme.
    Otherwise, aggregates all open target accounts.
    """
    # Normalize product_id (use actual id from subgraph)
    G = build_product_graph(product_id)
    product_id = get_product_id_from_subgraph(G) or product_id

    # Determine which accounts to include
    if account_id:
        account_ids = [account_id]
    else:
        account_ids = get_target_account_ids(product_id, {"status": {"nin": ["Closed-won", "Closed-lost"]}})

    themes: List[Dict[str, Any]] = []
    belief_vals: List[float] = []
    personas_all: set[str] = set()

    for acc in account_ids:
        # Load or (re)build RCS
        rcs, p = load_account_rcs_from_disk(product_id, acc)
        if rcs is None:
            # generate + save
            rcs = rcs_for_target_account(product_id, acc)
            # ensure a sane minimal shape if generators return partials
            if "execution_plan" not in rcs:
                rcs["execution_plan"] = {}
            _save_rcs_json(product_id, acc, rcs)

        # Theme from this account
        theme = _theme_from_account_rcs(acc, rcs)
        # collect persona count for portfolio
        for c in theme.get("campaigns", []):
            for p in c.get("personas", []) or []:
                personas_all.add(str(p))

        themes.append(theme)

        # belief contribution
        b = _belief_from_rcs(rcs)
        if b is not None:
            belief_vals.append(b)

    # Portfolio metrics
    total_accounts = len(account_ids)
    avg_belief = float(stats.mean(belief_vals)) if belief_vals else 0.0
    plan: Dict[str, Any] = {
        "meta": {
            "version": "2.0",
            "generatedAt": _now_iso(),
            "beliefScale": ["Unaware", "Problem-aware", "Solution-aware", "Most-aware"],
            "avgBeliefBand": _avg_belief_band(avg_belief),
        },
        "portfolio": {
            "keyStats": {
                "totalTargetAccounts": total_accounts,
                "totalPersonasToEngage": len(personas_all),
                "expectedWinsPct": 0.0,          # you can wire this to graphwin later
                "averageAccountBelief": f"{avg_belief:.2f}",
                "timeToWinMonths": 3,            # default; tune later
            }
        },
        "themes": themes,
    }
    return plan

def _filter_by_window(plan: Dict[str, Any], window_start: Optional[str], window_end: Optional[str]) -> Dict[str, Any]:
    """
    Non-destructive filter: keeps only campaigns whose timeframe overlaps [start, end].
    If either bound is None, it behaves like an open interval on that side.
    """
    if not window_start and not window_end:
        return plan

    def overlaps(tf: Dict[str, str]) -> bool:
        s = tf.get("startDate")
        e = tf.get("endDate")
        if not (s and e):
            return True  # keep if undefined
        if window_start and e < window_start:
            return False
        if window_end and s > window_end:
            return False
        return True

    themes = []
    for t in plan.get("themes", []):
        camps = [c for c in (t.get("campaigns") or []) if overlaps(c.get("timeframe") or {})]
        if camps:
            themes.append({**t, "campaigns": camps})

    out = dict(plan)
    out["themes"] = themes
    return out
