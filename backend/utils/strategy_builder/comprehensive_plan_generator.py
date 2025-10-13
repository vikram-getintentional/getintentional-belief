from datetime import date, datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from numpy import mean

from backend.utils.crm_management.target_account_manager import get_account_by_id, get_target_account_ids


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "utils", "dev_environment", "static_jsons")
RCS_DIR = os.path.join(BASE_DIR, "utils", "dev_environment", "static_jsons", "account_rcs_jsons")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _rcs_file_path(product_id: str, account_id: str):
    return os.path.join(RCS_DIR, f"{product_id}_{account_id}.json")

def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()

def _overlaps(item_start: Optional[str], item_end: Optional[str], w_start: date, w_end: date) -> bool:
    """Return True if [item_start, item_end] overlaps [w_start, w_end]."""
    # Default open intervals if missing
    s = _parse_date(item_start) or date.min
    e = _parse_date(item_end) or date.max
    return not (e < w_start or s > w_end)

def _load_plan_from_disk(product_id: str) -> dict:
    """
    Attempt to load a product-specific plan, otherwise fall back to default.
    Raises FileNotFoundError if neither file exists.
    """
    print("Loading plan from disk for product_id:", product_id)
    product_file = os.path.join(DATA_DIR, f"{product_id}_execution_plan.json")
    default_file = os.path.join(DATA_DIR, "execution_plan.json")

    print("Checking for product-specific plan file at:", product_file)

    if os.path.exists(product_file):
        with open(product_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if os.path.exists(default_file):
        print("Falling back to default plan file at:", default_file)
        with open(default_file, "r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError(
        f"No plan JSON found. Checked: {product_file} and {default_file}"
    )

def _filter_by_window(plan: dict, window_start: str | None, window_end: str | None) -> dict:
    """
    Filters campaigns in each theme by timeframe boundaries.
    Expects schema v2.0 where campaigns live under plan['themes'][*]['campaigns'].
    """
    if not window_start and not window_end:
        return plan

    w_start = _parse_date(window_start) or date.min
    w_end   = _parse_date(window_end) or date.max
    if w_end < w_start:
        raise HTTPException(status_code=422, detail="window_end must be >= window_start")

    out = dict(plan)
    filtered_themes = []

    for theme in plan.get("themes", []):
        theme_copy = dict(theme)
        filtered_campaigns = []
        for camp in theme.get("campaigns", []):
            tf = camp.get("timeframe", {})
            if _overlaps(tf.get("startDate"), tf.get("endDate"), w_start, w_end):
                filtered_campaigns.append(camp)
        theme_copy["campaigns"] = filtered_campaigns
        filtered_themes.append(theme_copy)

    out["themes"] = filtered_themes
    return out


def _recompute_portfolio_expectations(plan: dict) -> dict:
    """
    Recalculate expectedWinsPct and averageAccountBelief based on visible campaigns.
    Works with v2.0 JSON only.
    """
    portfolio = plan.get("portfolio", {})
    key_stats = portfolio.get("keyStats", {})
    campaigns = [c for t in plan.get("themes", []) for c in t.get("campaigns", [])]

    if not campaigns:
        key_stats["expectedWinsPct"] = 0.0
        key_stats["averageAccountBelief"] = "Unaware"
        plan["portfolio"]["keyStats"] = key_stats
        return plan

    # --- Expected Wins % ---
    total = len(campaigns)
    active = sum(1 for c in campaigns if c.get("timeframe"))
    base = key_stats.get("expectedWinsPct", 0.6)
    key_stats["expectedWinsPct"] = round(base * (active / total), 2)

    # --- Average Belief Level ---
    belief_scale = plan.get("meta", {}).get("beliefScale", [])
    if not belief_scale:
        belief_scale = ["Unaware", "ZMOT", "Discovery", "Evaluation", "Pilot", "Pre-Close", "Customer"]
    belief_map = {b.lower(): i for i, b in enumerate(belief_scale)}

    def infer_belief_from_text(campaign):
        txt = (campaign.get("description", "") + " " + " ".join(campaign.get("personas", []))).lower()
        if any(x in txt for x in ["awareness", "launch", "press", "analyst", "zmot"]):
            return belief_map.get("zmot", 1)
        if any(x in txt for x in ["discovery", "interest", "learn", "education"]):
            return belief_map.get("discovery", 2)
        if any(x in txt for x in ["evaluation", "blueprint", "workshop"]):
            return belief_map.get("evaluation", 3)
        if any(x in txt for x in ["pilot", "proof", "poc", "demo", "adoption"]):
            return belief_map.get("pilot", 4)
        if any(x in txt for x in ["customer", "reference", "evangelism", "story"]):
            return belief_map.get("customer", 6)
        return belief_map.get("discovery", 2)

    scores = [infer_belief_from_text(c) for c in campaigns]
    avg_idx = round(mean(scores)) if scores else belief_map.get("discovery", 2)
    key_stats["averageAccountBelief"] = belief_scale[min(avg_idx, len(belief_scale) - 1)]

    plan["portfolio"]["keyStats"] = key_stats
    return plan

#---- Account Level RCS JSON Generation ----#

def load_account_rcs_from_disk(product_id: str, account_id: str) -> dict:
    """
    Load or create an RCS JSON scaffold for this account.
    If missing or corrupted, create a new enriched scaffold using TargetAccount.
    """
    p = _rcs_file_path(product_id, account_id)

    if not p.exists():
        acct = get_account_by_id(product_id, account_id)
        rcs = _rcs_header_from_account(product_id, account_id, acct)
        with p.open("w", encoding="utf-8") as f:
            json.dump(rcs, f, indent=2)
        return rcs

    try:
        with p.open("r", encoding="utf-8") as f:
            rcs = json.load(f)
    except json.JSONDecodeError:
        acct = get_account_by_id(product_id, account_id)
        rcs = _rcs_header_from_account(product_id, account_id, acct)
        with p.open("w", encoding="utf-8") as f:
            json.dump(rcs, f, indent=2)
        return rcs

    # If it's an empty/fresh file with only headers, enrich it once
    if not (rcs.get("plays") or rcs.get("evidence") or rcs.get("stakeholders")):
        acct = get_account_by_id(product_id, account_id)
        if acct:
            enriched = _rcs_header_from_account(product_id, account_id, acct)
            # Preserve any edits user might have made to top-level keys we don't want to clobber
            # (keep any non-empty arrays the user added)
            for k in ("plays", "evidence", "nextActions", "hypotheses", "objectives"):
                if rcs.get(k):
                    enriched[k] = rcs[k]
            rcs = enriched
            with p.open("w", encoding="utf-8") as f:
                json.dump(rcs, f, indent=2)

    return rcs

#-----RCS Header & Pre-data fillers

def _account_to_rcs_seed(acct: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a TargetAccount row into fields useful for an RCS scaffold.
    This function is defensive: it tolerates missing/optional fields.
    """
    # Common fields you likely have on TargetAccount (rename if your schema differs)
    name            = acct.get("name") or acct.get("company") or ""
    website         = acct.get("website") or ""
    region          = acct.get("region") or acct.get("geo") or ""
    size_segment    = acct.get("segment") or acct.get("employee_band") or ""
    revenue_band    = acct.get("revenue_band") or ""
    archetypes      = acct.get("archetypes") or acct.get("archetype") or []
    competitor      = acct.get("current_competitor") or acct.get("billing_competitor") or ""
    current_stack   = acct.get("current_stack") or {}
    billing_model   = acct.get("billing_model") or acct.get("pricing_model") or ""
    icp_fit         = acct.get("icp_fit") if acct.get("icp_fit") is not None else None
    tags            = acct.get("tags") or []
    intent_signals  = acct.get("intent_signals") or []
    notes           = acct.get("notes") or ""
    # If you store belief/stage on the account, seed from there
    belief_state    = acct.get("belief_state") or acct.get("funnel_stage") or "Unaware"
    belief_score    = acct.get("belief_score")
    belief_reason   = acct.get("belief_rationale") or ""

    # Optional people/roles
    primary_contacts = acct.get("contacts") or []  # expect list of dicts, if you have it

    return {
        "account": {
            "id": acct.get("id"),
            "product_id": acct.get("product_id"),
            "name": name,
            "website": website,
            "region": region,
            "segment": size_segment,
            "revenueBand": revenue_band,
            "archetypes": archetypes if isinstance(archetypes, list) else [archetypes] if archetypes else [],
            "currentCompetitor": competitor,
            "currentStack": current_stack,
            "billingModel": billing_model,
            "icpFit": icp_fit,
            "intentSignals": intent_signals,
            "tags": tags,
            "notes": notes,
        },
        "belief": {
            "state": belief_state,
            "score": belief_score if isinstance(belief_score, (int, float)) else 0,
            "rationale": belief_reason,
        },
        "stakeholders": [
            {
                "name": c.get("name"),
                "title": c.get("title"),
                "email": c.get("email"),
                "role": c.get("role") or c.get("persona"),
                "influence": c.get("influence"),  # e.g., 1–5 or "High/Med/Low"
                "notes": c.get("notes", ""),
            }
            for c in primary_contacts
            if isinstance(c, dict)
        ],
    }


def _rcs_header_from_account(product_id: str, account_id: str, acct: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Construct a smart default RCS header using an optional TargetAccount.
    If acct is None, falls back to a minimal header.
    """
    now = _now_iso()

    if acct:
        seed = _account_to_rcs_seed(acct)
        # Light, opinionated hints you can prune later
        initial_hypotheses = []
        if seed["account"].get("currentCompetitor"):
            initial_hypotheses.append(
                f"Displace {seed['account']['currentCompetitor']} by proving Revenue Agility + lower cost-to-serve."
            )
        if seed["account"].get("billingModel"):
            initial_hypotheses.append(
                f"Pilot in {seed['account']['billingModel']} flow with guardrails and rollback plan."
            )
        if not initial_hypotheses:
            initial_hypotheses.append("Validate business case with CFO and align RevOps on phased rollout.")

        return {
            "meta": {"version": "1.0", "createdAt": now, "updatedAt": now},
            **seed,
            "hypotheses": [{"text": h, "status": "Open"} for h in initial_hypotheses],
            "objectives": {
                "winHypothesis": initial_hypotheses[0],
                "kpis": [
                    {"name": "Belief Shift", "target": "+1 stage"},
                    {"name": "Executive Alignment", "target": "CFO + RevOps sign-off"},
                ],
            },
            "plays": [],
            "evidence": [],
            "nextActions": [],
        }

    # Fallback minimal header (no account in DB)
    return {
        "meta": {"version": "1.0", "createdAt": now, "updatedAt": now},
        "account": {"id": account_id, "product_id": product_id, "name": ""},
        "belief": {"state": "Unaware", "score": 0, "rationale": ""},
        "stakeholders": [],
        "hypotheses": [],
        "objectives": {"winHypothesis": "", "kpis": []},
        "plays": [],
        "evidence": [],
        "nextActions": [],
    }


def _save_rcs_json(product_id: str, account_id: str, rcs_json: dict):
    _ensure_dir(DATA_DIR)
    path = _rcs_file_path(product_id, account_id)
    with open(path, "w") as f:
        json.dump(rcs_json, f, indent=2)



def load_all_account_rcs_jsons(product_id: str):
    print("Loading all account ids from comprehensive generator")
    account_ids = get_target_account_ids(product_id, {"status": {"nin": ["Closed-won", "Closed-lost"]}})
    rcs_list = []
    created = 0
    normalized = 0

    """for acc_id in account_ids:
        p = _rcs_file_path(product_id, acc_id)
        existed = p.exists()
        rcs = load_account_rcs_from_disk(product_id, acc_id)
        if not existed:
            created += 1
        else:
            if not (rcs.get("plays") or rcs.get("evidence") or rcs.get("stakeholders")):
                normalized += 1
        rcs_list.append(rcs)"""

    debug = {
        "rcs_accounts_count": len(rcs_list),
        "new_rcs_headers_created": created,
        "rcs_headers_normalized": normalized,
    }
    return rcs_list, debug
