from datetime import date, datetime
import json
import os
from typing import Optional

from fastapi import HTTPException
from numpy import mean

from backend.utils.crm_management.target_account_manager import get_account_by_id, get_target_account_ids


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "utils", "dev_environment", "static_jsons")
RCS_DIR = os.path.join(BASE_DIR, "utils", "dev_environment", "static_jsons", "account_rcs_jsons")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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

def load_account_rcs_from_disc(product_id: str, account_id: str):
    """Loads account-level RCS JSON if exists, else returns None."""
    path = _rcs_file_path(product_id, account_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _build_rcs_header(account):
    """Builds a header-only JSON."""
    return {
        "account": account.account_name,
        "target_account_id": account.id,
        "product_id": account.product_id,
        "archetype": {
            "industry": account.industry or "Unknown",
            "revenue": account.revenue_range or "Unknown",
            "employees": account.employee_range or "Unknown",
            "funding_stage": account.funding_stage or "Unknown",
            "geography": account.geography or "Unknown",
        },
        "zmot_theme": None,
        "reverse_case_study": {"stages": []},
        "execution_plan": {"themes": [], "campaigns": [], "arsenal": []},
        "created_at": datetime.now(datetime.timezone.utc).isoformat(),
    }


def _save_rcs_json(product_id: str, account_id: str, rcs_json: dict):
    _ensure_dir(DATA_DIR)
    path = _rcs_file_path(product_id, account_id)
    with open(path, "w") as f:
        json.dump(rcs_json, f, indent=2)


def load_all_account_rcs_jsons(product_id: str):
    print("Loading all account ids from comprehensive generator")
    account_ids = get_target_account_ids(
        product_id, {"deal_status": ["!in", ["Closed-Won", "Closed-Lost"]]}
    )
    if not account_ids or len(account_ids) <= 0:
        print("No target accounts found ")
        rcs_list=[]

        return 
    
    
    print(f"Found {len(account_ids)} target accounts for product_id {product_id}")
    rcs_list = []
    for acc_id in account_ids:
        rcs = load_account_rcs_from_disc(product_id, acc_id)
        if not rcs or not rcs.get("reverse_case_study", {}).get("stages"):
            account = get_account_by_id(product_id, acc_id)
            if not account:
                continue
            rcs = _build_rcs_header(account)
            _save_rcs_json(product_id, acc_id, rcs)
        rcs_list.append(rcs)

    return rcs_list