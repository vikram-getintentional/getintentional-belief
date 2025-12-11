from __future__ import annotations
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

BELIEF_SCALE = [
    "Unaware",
    "ZMOT",
    "Problem",
    "InternalBarrier",
    "Evaluation",
    "Approval",
    "Implementation",
    "PromisedLand",
]


def _belief_stage_from_probability(prob: float) -> str:
    if prob is None:
        return "Unaware"
    if prob <= 0.1:
        return "ZMOT"
    if prob <= 0.3:
        return "Problem"
    if prob <= 0.5:
        return "InternalBarrier"
    if prob <= 0.7:
        return "Evaluation"
    if prob <= 0.9:
        return "Approval"
    return "Implementation"


def _avg_time_to_first_conversion(accounts: Sequence[Dict[str, Any]]) -> float:
    timelines: List[float] = []
    for account in accounts:
        conv = account.get("execution", {}).get("conversion_sequence") or []
        if not conv:
            continue
        first = conv[0]
        days = first.get("timeline_days")
        if isinstance(days, (int, float)):
            timelines.append(float(days))
    if not timelines:
        return 90.0
    return mean(timelines)


def build_portfolio_key_stats(
    summary: Dict[str, Any], accounts: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    total_accounts = summary.get("account_count") or len(accounts)
    total_personas = summary.get("total_persona_requirements") or 0
    win_pct = summary.get("avg_best_path_probability") or summary.get("avg_accuracy") or 0.0
    avg_belief_stage_label = _belief_stage_from_probability(win_pct)
    time_to_win_months = max(0.5, _avg_time_to_first_conversion(accounts) / 30.0)
    wolf_ids = [
        account.get("account_id")
        for account in accounts
        if (account.get("keystone_personas") or [])
    ]
    wolf_pct = (len(wolf_ids) / total_accounts) if total_accounts else 0.0
    coverage_values = []
    for account in accounts:
        coverage = (account.get("enrichment") or {}).get("summary", {}).get("coverage_ratio")
        if isinstance(coverage, (int, float)):
            coverage_values.append(float(coverage))
    avg_readiness = mean(coverage_values) if coverage_values else 0.0
    return {
        "totalTargetAccounts": total_accounts,
        "totalPersonasToEngage": total_personas,
        "expectedWinsPct": win_pct,
        "averageAccountBelief": avg_belief_stage_label,
        "timeToWinMonths": round(time_to_win_months, 1),
        "wolfIdentifiedPct": round(wolf_pct, 2),
        "avgEnablementReadiness": round(avg_readiness, 2),
    }


def build_decision_drivers(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    drivers = []
    keystones = summary.get("keystone_personas") or []
    top_pains = summary.get("top_pains") or []
    pain_labels = [pain.get("label") for pain in top_pains if pain.get("label")]
    for entry in keystones[:3]:
        driver = {
            "personaId": entry.get("persona_id") or entry.get("persona_label") or "persona_unknown",
            "personaName": entry.get("persona_label") or entry.get("persona") or "Unknown",
            "winGateSharePct": round(min(entry.get("wolves_score") or 0.0, 1.0), 2),
            "medianDaysBeforeClose": 20,
            "typicalBlockingBeliefs": pain_labels[:2] or ["Internal risks unknown"],
        }
        drivers.append(driver)
    return drivers


def build_subsidy_map(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    drivers = build_decision_drivers(summary)
    subsidy_personas = [
        {
            "personaId": "persona_revops",
            "personaName": "Revenue Operations Leader",
            "changeBurdenDescription": "Owns migration and reconciliation design.",
            "failureRisk": "HIGH",
        },
        {
            "personaId": "persona_it",
            "personaName": "IT / Security Lead",
            "changeBurdenDescription": "Must validate controls and SSO.",
            "failureRisk": "MEDIUM",
        },
    ]
    return [
        {
            "wolfPersonaId": driver["personaId"],
            "wolfPersonaName": driver["personaName"],
            "avgSubsidyPersonasCount": len(subsidy_personas),
            "downstreamSubsidies": subsidy_personas,
        }
        for driver in drivers
    ]


def build_pain_themes(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    pains = []
    for index, entry in enumerate(summary.get("top_pains") or []):
        label = entry.get("label") or "Operational pain"
        stage = "Problem" if index == 0 else "InternalBarrier"
        pains.append(
            {
                "id": f"pain_{index}",
                "label": label,
                "prevalenceInWinsPct": round(min(entry.get("count", 0) / (summary.get("total_delta_bp") or 1), 1.0), 2),
                "dominantStage": stage,
            }
        )
    if not pains:
        pains.append(
            {
                "id": "pain_generic",
                "label": "Conversion friction",
                "prevalenceInWinsPct": 0.4,
                "dominantStage": "Problem",
            }
        )
    return pains


def serialize_campaign_themes(portfolio_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    themes = []
    for entry in portfolio_plan.get("conversion_focuses") or []:
        theme_id = entry.get("campaign_theme_id") or entry.get("label") or "theme_unnamed"
        campaigns = [
            {
                "id": f"camp_{idx}",
                "description": entry.get("focus_label") or "No description",
                "timeframe": {
                    "startDate": entry.get("start_date") or "2026-01-01",
                    "endDate": entry.get("end_date") or "2026-03-31",
                },
                "personas": entry.get("people") or [],
                "beliefShifts": [],
                "arsenalTable": entry.get("plays") or [],
            }
            for idx in range(1)
        ]
        themes.append(
            {
                "id": theme_id,
                "name": entry.get("focus_label") or entry.get("label") or "Conversion focus",
                "explanation": entry.get("expected_outcome_summary") or "",
                "objective": entry.get("focus_label") or "Enable belief shifts",
                "targetAccounts": [
                    acct.get("account_id") for acct in entry.get("accounts") or []
                ],
                "campaigns": campaigns,
            }
        )
    return themes


def build_comprehensive_execution_plan_contract(
    plan: Dict[str, Any], product_id: str
) -> Dict[str, Any]:
    summary = plan.get("summary") or {}
    portfolio_plan = plan.get("portfolio_plan") or {}
    accounts = plan.get("accounts") or []
    return {
        "meta": {
            "version": "2.1",
            "generatedAt": plan.get("generated_at") or "",
            "beliefScale": BELIEF_SCALE,
        },
        "portfolio": {
            "id": product_id,
            "name": f"Product {product_id}",
            "keyStats": build_portfolio_key_stats(summary, accounts),
            "decisionDrivers": build_decision_drivers(summary),
            "subsidyMap": build_subsidy_map(summary),
            "beliefProgression": {
                "canonicalPath": [
                    "ZMOT",
                    "Problem",
                    "InternalBarrier",
                    "Champion",
                    "Approval",
                    "Implementation",
                ],
                "medianDaysBetweenStages": {
                    "ZMOT→Problem": 9,
                    "Problem→InternalBarrier": 14,
                    "InternalBarrier→Approval": 12,
                    "Approval→Implementation": 18,
                },
                "dropOffPctByStage": {
                    "ZMOT→Problem": 0.12,
                    "Problem→InternalBarrier": 0.19,
                    "InternalBarrier→Approval": 0.27,
                },
            },
            "painThemes": build_pain_themes(summary),
        },
        "themes": serialize_campaign_themes(portfolio_plan),
    }


def build_account_plan_contract(
    account_plan: Dict[str, Any], product_id: str
) -> Dict[str, Any]:
    prediction = account_plan.get("prediction") or {}
    meta = account_plan.get("meta") or {}
    execution = account_plan.get("execution") or {}
    momentum = prediction.get("journey", {}).get("steps")
    steps = momentum or []
    stage = "Problem"
    if steps and isinstance(steps[0], dict):
        stage = steps[-1].get("stage_label") or stage
    return {
        "meta": {
            "generatedAt": account_plan.get("generated_at") or "",
            "beliefScale": BELIEF_SCALE,
        },
        "account": {
            "id": account_plan.get("account_id"),
            "name": account_plan.get("account_name"),
            "segment": account_plan.get("meta", {}).get("segment") or "Segment",
            "industry": account_plan.get("meta", {}).get("industry") or "Industry",
            "currentBeliefStage": stage,
            "dealValueEstimate": meta.get("deal_value") or 0.0,
            "predictedWinPct": account_plan.get("prediction", {}).get("persona_paths", [{}])[0].get("probability") or 0.0,
            "championStrength": 0.5,
            "enablementReadiness": (account_plan.get("enrichment", {}).get("summary", {}).get("coverage_ratio") or 0.0),
            "momentum": "MEDIUM",
            "predictedStallInDays": 14,
        },
        "decisionMap": {
            "primaryGate": {
                "personaId": "persona_cfo",
                "canonicalPersonaId": "persona_cfo",
                "name": "CFO",
                "beliefStage": stage,
                "influenceScore": 0.9,
                "gateType": "FINANCIAL_RISK",
            },
            "secondaryGates": [],
            "champion": {
                "personaId": "persona_revops",
                "canonicalPersonaId": "persona_revops",
                "name": "Revenue Operations",
                "beliefStage": "Champion",
                "influenceScore": 0.7,
            },
            "supportingPersons": [],
        },
        "enablementQueue": {
            "overallReadiness": (
                account_plan.get("enrichment", {}).get("summary", {}).get("coverage_ratio") or 0.0
            ),
            "items": [
                {
                    "personaId": "persona_it",
                    "canonicalPersonaId": "persona_it",
                    "personaName": "IT / Security Lead",
                    "fear": "Compliance risk",
                    "neededEvidence": ["Security design review", "SSO & RBAC deck"],
                    "riskIfIgnored": "HIGH",
                    "status": "RED",
                }
            ],
        },
        "nextBestActions": [
            {
                "id": "nba_security",
                "priority": 1,
                "targetPersonaId": "persona_it",
                "canonicalPersonaId": "persona_it",
                "targetPersonaName": "IT / Security Lead",
                "fromBelief": "Problem",
                "toBelief": "SolutionViable",
                "corePainId": "pain_security",
                "corePainLabel": "Security gaps",
                "assetType": "Security deck",
                "assetId": None,
                "channel": "AE_direct_email",
                "whyNow": "IT concerns surfaced on last call.",
                "whatItEnables": "Allows CFO to progress.",
                "expectedLiftPct": 0.12,
                "confidence": 0.71,
                "recommendedWindowDays": 3,
            }
        ],
        "outcomeSimulator": {
            "current": {
                "predictedWinPct": account_plan.get("prediction", {}).get("persona_paths", [{}])[0].get("probability") or 0.0,
                "expectedCloseWindowDays": [21, 42],
            },
            "ifRecommendedActionsExecuted": {
                "predictedWinPct": 0.59,
                "expectedCloseWindowDays": [18, 33],
            },
            "ifIgnored": {
                "predictedWinPct": 0.24,
                "stallProbabilityPct": 0.6,
                "predictedStallInDays": 14,
            },
        },
    }


def build_insights_inbox_contract(
    insights_payload: Dict[str, Any]
) -> Dict[str, Any]:
    product_insights = insights_payload.get("product_insights") or {}
    global_patterns = product_insights.get("global_patterns") or {}
    cards = []
    generated_at = (
        insights_payload.get("meta", {}).get("date")
        or insights_payload.get("meta", {}).get("wolves_metrics_updated_at")
        or ""
    )
    if global_patterns.get("biggest_barrier", {}).get("text"):
        cards.append(
            {
                "id": "ins_global_barrier",
                "type": "DECISION_DRIVER_SHIFT",
                "severity": "MEDIUM",
                "title": "Biggest barrier shifts",
                "description": global_patterns["biggest_barrier"]["text"],
                "scope": {},
                "evidence": {
                    "sampleSize": insights_payload.get("meta", {}).get("num_accounts", 0),
                    "timeWindowDays": 90,
                },
                "implication": "Revisit the decision gate sequencing.",
                "recommendedAction": "Shift focus to new blockers.",
                "createdAt": generated_at,
                "tags": ["wolf"],
            }
        )
    return {
        "meta": {"generatedAt": generated_at},
        "insights": cards,
    }


def build_personas_atlas_contract(summary: Dict[str, Any]) -> Dict[str, Any]:
    personas = []
    for entry in summary.get("keystone_personas") or []:
        personas.append(
            {
                "id": entry.get("persona_id") or entry.get("persona") or "persona_unknown",
                "name": entry.get("persona") or entry.get("persona_label") or "Persona",
                "departments": ["Unknown"],
                "exampleTitles": [entry.get("persona") or "Leadership"],
                "seniority": "DIRECTOR",
                "coalitionRole": "WOLF",
                "entryBeliefStage": "Problem",
                "commonPains": [],
                "occurrenceInWinsPct": round(entry.get("share") or 0.0, 2),
                "avgTimeToActivationDays": 21,
                "typicalPositionInCoalition": 1,
                "exampleAccounts": [],
                "metaSignals": {},
            }
        )
    return {
        "meta": {"generatedAt": summary.get("generated_at") or ""},  # best effort
        "personas": personas,
    }


def build_icp_overview_contract(summary: Dict[str, Any]) -> Dict[str, Any]:
    icps = []
    for cluster in summary.get("account_clusters") or []:
        icps.append(
            {
                "id": cluster.get("id") or "icp_unknown",
                "label": cluster.get("label") or "ICP",
                "industry": cluster.get("tokens", [""])[0].split(":")[1] if cluster.get("tokens") else "Industry",
                "employeeRange": {"min": 200, "max": 1000},
                "geo": ["US"],
                "techStackSignals": cluster.get("tokens", []),
                "typicalWolfPersonaId": "persona_cfo",
                "typicalWolfPersonaName": "Chief Financial Officer",
                "medianWinRatePct": 0.26,
                "medianSalesCycleDays": 72,
                "avgDealSize": 320000.0,
                "avgSubsidyBurdenScore": 0.7,
                "dominantPains": ["pain_revenue_leakage"],
                "emergingTrends": ["Security concerns rising"],
                "status": "CORE",
            }
        )
    if not icps:
        icps.append(
            {
                "id": "icp_default",
                "label": "Default ICP",
                "industry": "SaaS",
                "employeeRange": {"min": 50, "max": 500},
                "geo": ["US"],
                "techStackSignals": ["QuickBooks"],
                "typicalWolfPersonaId": "persona_cfo",
                "typicalWolfPersonaName": "Chief Financial Officer",
                "medianWinRatePct": 0.2,
                "medianSalesCycleDays": 60,
                "avgDealSize": 250000.0,
                "avgSubsidyBurdenScore": 0.65,
                "dominantPains": ["pain_manual_reconciliation"],
                "emergingTrends": ["RevOps champion roles rising"],
                "status": "EMERGING",
            }
        )
    return {
        "meta": {"generatedAt": summary.get("generated_at") or ""},
        "icps": icps,
    }
