from datetime import date, datetime, timezone
import json
import os
from typing import Any, Dict, Optional
from pathlib import Path

from fastapi import HTTPException
from numpy import mean

from backend.utils.crm_management.target_account_manager import get_account_by_id, get_target_account_ids
from backend.utils.graph_base.network_graph import build_product_graph, get_product_id_from_subgraph
from backend.utils.inference.rcs_generators.rcs_helpers.strategy_orchestrators import construct_all_account_rcs
from backend.utils.strategy_builder.plan_manager import build_portfolio_plan_from_rcs_sequences


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "utils", "dev_environment", "static_jsons")
RCS_DIR = os.path.join(BASE_DIR, "utils", "dev_environment", "static_jsons", "account_rcs_jsons")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _rcs_file_path(product_id: str, account_id: str):
    return Path(RCS_DIR) / f"{product_id}_{account_id}.json"

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
    
    
    product_id_indexed = product_id
    product_subgraph = build_product_graph(product_id)
    product_id_actual = get_product_id_from_subgraph(product_subgraph)

    print("Loading plan from disk for product_id:", product_id)
    product_file = os.path.join(DATA_DIR, f"{product_id}_execution_plan.json")
    default_file = os.path.join(DATA_DIR, "execution_plan.json")
    

    print("Checking for product-specific plan file at:", product_file)
    """
    if os.path.exists(product_file):
        with open(product_file, "r", encoding="utf-8") as f:
            return json.load(f)
    """    
    print("Building all RCS for indexed product_id:", product_id_indexed)
    rcs_filled = construct_all_account_rcs(product_id_indexed)
    if rcs_filled:
        print("RCS data constructed for product_id:", product_id_indexed)
        print("RCS data type:", type(rcs_filled))
        print("RCS items length:", len(rcs_filled))
        for rcs in rcs_filled:
            if rcs.get("execution_plan"):
                print("RCS data contains execution_plan")

    if rcs_filled:
        # Create unified product-specific plan file with comprehensive RCS data
        print("Attempting to create product-specific plan with RCS data" )
        integrated_plan = build_portfolio_plan_from_rcs_sequences(
            rcs_list=rcs_filled,
            G=product_subgraph,
            product_id=product_id_actual
            )
        return integrated_plan

    else:
        print("No RCS data found to create product-specific plan.")
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

def _save_rcs_json(product_id: str, account_id: str, rcs_json: dict):
    p = Path(_rcs_file_path(product_id, account_id))
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(rcs_json, f, indent=2)

        print("Saved updated json to:", p)

def load_account_rcs_from_disk(product_id: str, account_id: str) -> dict:
    """
    Load or create an RCS JSON scaffold for this account.
    If missing or corrupted, create a new enriched scaffold using TargetAccount.
    """
    p = Path(_rcs_file_path(product_id, account_id))
    print("Loading RCS JSON from path:", p)
    try:
        existed = p.exists()
    except Exception as e:
        print("[ERROR] checking path exists:", e)
        existed = False

    if not existed:
        print("RCS file not found, creating new scaffold.")
        acct = get_account_by_id(product_id, account_id)
        rcs = _rcs_header_from_account(product_id, account_id, acct)
        print("New RCS scaffold created for account_id:", account_id, " with headers: ", rcs)
        # ensure directory exists then write
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                json.dump(rcs, f, indent=2)
        except Exception as e:
            print("[ERROR] failed to write new RCS file:", e)
        return rcs, False

    # file exists -> try to load
    try:
        with p.open("r", encoding="utf-8") as f:
            rcs = json.load(f)
    except json.JSONDecodeError:
        print("RCS JSON corrupted, recreating scaffold.")
        acct = get_account_by_id(product_id, account_id)
        rcs = _rcs_header_from_account(product_id, account_id, acct)
        try:
            with p.open("w", encoding="utf-8") as f:
                json.dump(rcs, f, indent=2)
        except Exception as e:
            print("[ERROR] failed to write repaired RCS file:", e)
        return rcs, False
    except Exception as e:
        print("[ERROR] unexpected error reading RCS file:", e)
        # return a scaffold so caller can continue
        acct = get_account_by_id(product_id, account_id)
        rcs = _rcs_header_from_account(product_id, account_id, acct)
        return rcs, False

    # If it's an empty/fresh file with only headers, enrich it once
    if not (rcs.get("plays") or rcs.get("evidence") or rcs.get("stakeholders")):
        acct = get_account_by_id(product_id, account_id)
        if acct:
            enriched = _rcs_header_from_account(product_id, account_id, acct)
            for k in ("plays", "evidence", "nextActions", "hypotheses", "objectives"):
                if rcs.get(k):
                    enriched[k] = rcs[k]
            rcs = enriched
            try:
                with p.open("w", encoding="utf-8") as f:
                    json.dump(rcs, f, indent=2)
            except Exception as e:
                print("[ERROR] failed to write enriched RCS file:", e)

    return rcs, True

#-----RCS Header & Pre-data fillers

def _account_to_rcs_seed(acct: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a TargetAccount row into fields useful for an RCS scaffold.
    This function is defensive: it tolerates missing/optional fields and
    maps the DB column names used by target_account_manager to the RCS seed.
    """
    # Prefer the TargetAccount column names, fall back to older keys if present
    name            = acct.get("account_name") or acct.get("name") or acct.get("company") or ""
    website         = acct.get("website") or acct.get("site") or ""
    region          = acct.get("geography") or acct.get("region") or acct.get("geo") or ""
    size_segment    = acct.get("employee_range") or acct.get("employee_band") or acct.get("segment") or ""
    revenue_band    = acct.get("revenue_range") or acct.get("revenueBand") or acct.get("revenue") or ""
    # competitor_used in DB is JSON list — pick first as primary competitor string if present
    competitor      = ""
    cu = acct.get("competitor_used") or acct.get("current_competitor") or acct.get("billing_competitor") or []
    if isinstance(cu, (list, tuple)) and cu:
        competitor = cu[0]
    elif isinstance(cu, str):
        competitor = cu
    current_stack   = acct.get("other_tech_stack") or acct.get("current_stack") or {}
    billing_model   = acct.get("billing_model") or acct.get("pricing_model") or ""
    icp_fit         = acct.get("icp_fit") if acct.get("icp_fit") is not None else None
    tags            = acct.get("tags") or []
    intent_signals  = acct.get("intent_signals") or []
    notes           = acct.get("notes") or ""
    # If you store belief/stage on the account, seed from there (support both names)
    belief_state    = acct.get("deal_status") or acct.get("belief_state") or acct.get("funnel_stage") or "Unaware"
    belief_score    = acct.get("belief_score") or acct.get("win_likelihood")  # optional numeric
    belief_reason   = acct.get("belief_rationale") or acct.get("belief_reason") or ""

    # Optional people/roles (TargetAccount currently may not have contacts; keep defensive)
    primary_contacts = acct.get("contacts") or acct.get("primary_contacts") or []

    return {
        "account": {
            "id": acct.get("id"),
            "product_id": acct.get("product_id"),
            "name": name,
            "account_name": name,
            "website": website,
            "region": region,
            "segment": size_segment,
            "revenueBand": revenue_band,
            "revenue_range": revenue_band,
            "employee_range": size_segment,
            "funding_stage": acct.get("funding_stage") or acct.get("funding") or "",
            "industry": acct.get("industry") or "",
            "currentCompetitor": competitor,
            "currentStack": current_stack,
            "billingModel": billing_model,
            "icpFit": icp_fit,
            "intentSignals": intent_signals,
            "tags": tags,
            "notes": notes,
            # preserve raw DB status too
            "deal_status": acct.get("deal_status"),
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
                "influence": c.get("influence"),
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
        account = seed.get("account", {})
        belief = seed.get("belief", {"state": "Unaware", "score": 0.0, "rationale": ""})
        stakeholders = seed.get("stakeholders", [])

        # Archetype mapping (best-effort from seed)
        archetype = {
            "industry": account.get("industry") or (account.get("archetypes")[0] if account.get("archetypes") else "") or "",
            "revenue_range": account.get("revenueBand") or account.get("revenue") or "",
            "employee_range": account.get("employee_range") or account.get("segment") or "",
            "geography": account.get("region") or account.get("geography") or "",
            "funding_stage": account.get("funding_stage") or account.get("funding") or "",
        }

        # Simple persona scoring from stakeholder influence if present
        top_personas_by_involvement = []
        try:
            sorted_stakeholders = sorted(
                [s for s in stakeholders if isinstance(s, dict)],
                key=lambda x: (x.get("influence") or 0),
                reverse=True,
            )
            for s in sorted_stakeholders[:6]:
                top_personas_by_involvement.append({
                    "id": s.get("role") or s.get("title") or s.get("name"),
                    "involvement": float(s.get("influence") or 0),
                    "activation": round(float(s.get("influence") or 0) * 0.5, 2),
                    "strength": round(float(s.get("influence") or 0) * 0.75, 2),
                })
        except Exception:
            top_personas_by_involvement = []

        hot_nodes = [s.get("role") or s.get("title") or s.get("name") for s in stakeholders][:6]

        scaffold = {
            "meta": {"version": "1.0", "createdAt": now, "updatedAt": now},
            "account_id": account.get("id") or account_id,
            "account_name": account.get("name") or account.get("company") or "",
            "product_id": account.get("product_id") or product_id,
            "website": account.get("website") or "",
            "deal_status": account.get("deal_status") or belief.get("state") or "Unaware",
            "archetype": archetype,

            "zmot_theme": "",
            "rcs_brief": {
                "baseline": {
                    "win_likelihood": float(belief.get("score") or 0.0),
                    "walk_path": [],
                    "hot_nodes": hot_nodes,
                },
                "top_personas": {
                    "by_involvement": top_personas_by_involvement,
                    "by_activation": [], 
                    "by_strength": [],
                },
                "concerns_by_persona": {},
                "concern_backlog": [],
                "concern_coalitions": [],
                "concern_sequences": [],
                "coalitions": [],
                "core_scores": {},
                "persona_scores": {},
                "sequences_and_campaigns": [],
                "causal_flows": []
            },
            "reverse_case_study": {
                "stages": [
                    {
                        "stage": "",
                        "trigger": "",
                        "belief_before": "",
                        "belief_after": "",
                        "personas": [],
                        "pains": [],
                        "evidence": [],
                        "recommended_assets": []
                    }
                ]
            },
            "execution_plan": {
                "belief_state_summary": {
                    "average_belief_score": float(belief.get("score") or 0.0),
                    "dominant_stage": belief.get("state") or "",
                    "recommended_focus": "",
                },
                "personas": [],
                "campaigns": []
            },
            # keep legacy/utility keys for downstream code
            "plays": [],
            "evidence": [],
            "nextActions": [],
            "hypotheses": [{"text": f"Validate with {stakeholders[0].get('role')}" , "status": "Open"}] if stakeholders else [],
            "objectives": {"winHypothesis": "", "kpis": []},
        }

        return scaffold

    # Fallback minimal header (no account in DB)
    return {
        "meta": {"version": "1.0", "createdAt": now, "updatedAt": now},
        "account": {"id": account_id, "product_id": product_id, "name": ""},
        "belief": {"state": "Unaware", "score": 0.0, "rationale": ""},
        "stakeholders": [],
        "hypotheses": [],
        "objectives": {"winHypothesis": "", "kpis": []},
        "plays": [],
        "evidence": [],
        "nextActions": [],
        "rcs_brief": {
            "baseline": {"win_likelihood": 0.0, "walk_path": [], "hot_nodes": []},
            "top_personas": {"by_involvement": [], "by_activation": [], "by_strength": []},
            "concerns_by_persona": {},
            "concern_backlog": [],
            "concern_coalitions": [],
            "concern_sequences": [],
            "coalitions": [],
            "core_scores": {},
            "persona_scores": {},
            "sequences_and_campaigns": [],
            "causal_flows": []
        },
        "reverse_case_study": {"stages": []},
        "execution_plan": {
            "belief_state_summary": {"average_belief_score": 0.0, "dominant_stage": "", "recommended_focus": ""},
            "personas": [],
            "campaigns": []
        }
    }





def load_all_account_rcs_jsons(product_id: str):
    print("Loading all account ids from comprehensive generator")
    product_id_indexed = product_id
    product_subgraph = build_product_graph(product_id)
    product_id_actual = get_product_id_from_subgraph(product_subgraph)

    account_ids = get_target_account_ids(product_id_actual, {"status": {"nin": ["Closed-won", "Closed-lost"]}})
    print("Account IDs fetched for product_id", product_id_actual, ":", account_ids)
    rcs_list = []
    created = 0
    normalized = 0

    for acc_id in account_ids:
        print("Processing account_id:", acc_id)
        existed = "False"
        rcs, existed = load_account_rcs_from_disk(product_id_actual, acc_id)
        print("Loaded RCS for account_id", acc_id)
        if not existed:
            created += 1
        else:
            if not (rcs.get("plays") or rcs.get("evidence") or rcs.get("stakeholders")):
                normalized += 1
        rcs_list.append(rcs)

    debug = {
        "rcs_accounts_count": len(rcs_list),
        "new_rcs_headers_created": created,
        "rcs_headers_normalized": normalized,
    }
    return rcs_list, debug

