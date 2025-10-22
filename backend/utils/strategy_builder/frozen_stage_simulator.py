# frozen_stage_simulator.py

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import copy

from backend.utils.inference.rcs_generators.generate_rcs_fast import generate_rcs
from backend.utils.strategy_builder.stage_playbooks import build_stage_campaign_set, infer_account_stage, simulate_next_stage


STAGE_ORDER = [
    "pre_zmot",
    "zmot",
    "problem_realization",
    "discovery",
    "barriers",
    "implementation",
]

def _iso(date: datetime) -> str:
    return date.strftime("%Y-%m-%d")

def _phase_window(start_dt: datetime, days: int) -> Dict[str, str]:
    end_dt = start_dt + timedelta(days=days-1)
    return {"startDate": _iso(start_dt), "endDate": _iso(end_dt)}

def _clamp_stage(stage: str) -> str:
    return stage if stage in STAGE_ORDER else "pre_zmot"

def _next_stage(stage: str) -> str:
    try:
        i = STAGE_ORDER.index(_clamp_stage(stage))
        return STAGE_ORDER[min(i+1, len(STAGE_ORDER)-1)]
    except ValueError:
        return "zmot"

def _default_limits_for(stage: str) -> Dict[str, int]:
    # tuning knobs per stage (how many sequences and per-persona caps)
    return {
        "pre_zmot":            {"per_persona": 1, "max": 6},
        "zmot":                {"per_persona": 2, "max": 6},
        "problem_realization": {"per_persona": 2, "max": 6},
        "discovery":           {"per_persona": 2, "max": 6},
        "barriers":            {"per_persona": 2, "max": 6},
        "implementation":      {"per_persona": 2, "max": 6},
    }.get(stage, {"per_persona": 1, "max": 6})

def _personas_from_campaigns(campaigns: List[Dict[str, Any]]) -> List[str]:
    out = []
    for c in campaigns:
        for p in (c.get("personas") or []):
            if p not in out:
                out.append(p)
    return out

def _engagement_payload_from_personas(personas: List[str]) -> List[Dict[str, Any]]:
    # ultra simple: mark these personas “engaged” going into next phase
    return [{"id": pid, "type": "persona", "weight": 1.0} for pid in personas]

def build_multi_quarter_frozen_strategy(
    *,
    product_subgraph,                   # nx.DiGraph
    account_id: str,
    account_name: str,
    product_id: str,
    rcs_report: Dict[str, Any],         # the report from generate_rcs(...)
    archetype: Optional[Dict[str, Any]] = None,
    forced_start_stage: Optional[str] = None,
    num_phases: int = 4,                # 4 * 90d ≈ one year
    days_per_phase: int = 90,
    start_date: Optional[datetime] = None,
    simulate_carryover: bool = True,    # if True, we “engage” personas hit in prior phase
) -> Dict[str, Any]:
    """
    Returns a frozen, multi-quarter account strategy with time-boxed campaigns
    per belief stage. Keeps everything deterministic & swappable.
    """
    print("Building multi-quarter frozen strategy for product:", product_id)
    start_date = start_date or datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    # 1) pick starting stage
    inferred = forced_start_stage or _clamp_stage(infer_account_stage(rcs_report, default="pre_zmot"))
    phases: List[Dict[str, Any]] = []

    # 2) working state for (optional) carry-over simulation
    work_graph = product_subgraph
    work_report = rcs_report
    all_p = work_report.get("top_personas", {}) or {}
    all_personas = all_p.get("by_strength") or []

    # 3) build each phase in order, advancing stage stair-step
    stage = inferred
    phase_start = start_date
    for idx in range(num_phases):
        limits = _default_limits_for(stage)

        # build stage campaign set for this phase
        stage_plan = build_stage_campaign_set(
            G=work_graph,
            account_id=account_id,
            product_id=product_id,
            report=work_report,
            stage=stage,
            plays_per_step=1,
            archetype=archetype or {},
            limits=limits,
        )
        print("compelted stage_plan")
        print("stage_plan:", stage_plan)
        # stamp timeframe on campaigns
        tf = _phase_window(phase_start, days_per_phase)
        print("tf:", tf)
        for camp in (stage_plan.get("campaigns") or []):
            print("setting timeframe for campaign:", camp.get("name"))
            camp["timeframe"] = tf
        print("set timeframe")
        phases.append({
            "phaseIndex": idx + 1,
            "stage": stage_plan["stage"],
            "stageLabel": stage_plan["stageLabel"],
            "policy": stage_plan["policy"],
            "timeframe": tf,
            "campaigns": stage_plan["campaigns"],
            "personas": stage_plan["personas"],
        })
        print("appended phases")

        # (optional) carry-over: personas reached in this phase → engaged_nodes for next phase
        if simulate_carryover and idx < num_phases - 1:
            engaged_personas = _personas_from_campaigns(stage_plan["campaigns"])
            engaged_payload = _engagement_payload_from_personas(engaged_personas)
            # re-run generate_rcs to reflect momentum
            work_graph, work_report2 = generate_rcs(
                product_subgraph,
                engaged_nodes=engaged_payload,
                boost_factor=2.0,
            )
            work_report = work_report2
        print("Simulated carryover - moving to next stage:", stage, "for account:", account_name)
        # advance stage & time
        stage = _next_stage(stage)
        phase_start = phase_start + timedelta(days=days_per_phase)
    
    # 4) package as a frozen strategy object (per-account theme with multi-phase campaigns)
    frozen = {
        "account_id": account_id,
        "account_name": account_name,
        "personas": all_personas,

        "portfolio_policy": {"breadth": 0.6, "depth": 0.3, "mutation": 0.1},  # cosmetic default
        "belief_start_stage": inferred,
        "phases": phases,
        # For parity with the rest of your app:
        "themes": [{
            "id": f"theme_{account_id}_frozen",
            "name": f"{account_name} · Multi-Quarter Plan",
            "explanation": "Sequenced, belief-stage campaigns over rolling quarters.",
            "objective": "Progress belief, activate coalitions, unblock barriers, and implement.",
            "targetAccounts": [account_name],
            "campaigns": [c for ph in phases for c in (ph.get("campaigns") or [])],
        }],
    }
    return frozen
