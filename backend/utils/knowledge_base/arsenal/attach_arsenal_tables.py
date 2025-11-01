# attach_arsenal_tables.py
from __future__ import annotations
from typing import Any, Dict, List

from backend.utils.knowledge_base.arsenal.suggester import suggest_asset_channel_combos
from backend.utils.strategy_builder.belief_math import expected_belief_delta, lift_label_from_belief_delta, stage_gain_coeff


def _mk_row(combo: Dict[str, Any], stage: str) -> Dict[str, Any]:
    asset_fit = float(combo.get("fitScore", 0.5))
    channel_fit = float(combo.get("engagementScore", 0.5))
    delta = expected_belief_delta(asset_fit, channel_fit, stage_factor=stage_gain_coeff(stage))
    return {
        "asset": combo.get("asset"),
        "channel": combo.get("channel"),
        "fitment": ", ".join(combo.get("concernsAddressed") or []) or "Concern resolution",
        "engagement": "High" if channel_fit >= 0.7 else ("Medium" if channel_fit >= 0.4 else "Low"),
        "expectedLift": f"{int(round(delta * 100))}%",
        "expectedLiftLabel": lift_label_from_belief_delta(delta),
        "duration": "2-4 weeks",
    }

def _collect_concerns(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    concerns = []
    concerns.extend(report.get("concern_backlog") or [])
    for _p, rows in (report.get("concerns_by_persona") or {}).items():
        concerns.extend(rows or [])
    # dedupe by label
    seen, out = set(), []
    for c in concerns:
        lab = c.get("concern_label") or c.get("label")
        if not lab or lab in seen:
            continue
        seen.add(lab)
        out.append(c)
    return out

def _campaign_personas(camp: Dict[str, Any], fallback: List[str]) -> List[str]:
    p = [x for x in (camp.get("personas") or []) if x]
    return p or fallback

def attach_arsenal_tables_to_phases(
    *,
    frozen_strategy: Dict[str, Any],
    product_id: str,
    persona_scores: Dict[str, Any],
) -> None:
    """In-place mutation: fills campaigns[*].arsenalTable based on concerns + personas."""
    phases = frozen_strategy.get("phases") or []
    report = frozen_strategy.get("report") or {}
    if not phases:
        return

    concerns = _collect_concerns(report)
    persona_list = sorted([p for p in persona_scores.keys()]) if persona_scores else []
    for ph in phases:
        stage = ph.get("stage") or ph.get("stage_block", {}).get("stage") or "zmot"
        for camp in (ph.get("campaigns") or []):
            personas = _campaign_personas(camp, persona_list)
            combos = suggest_asset_channel_combos(product_id, concerns, personas, limit_per_concern=2)
            rows = [_mk_row(c, stage=stage) for c in combos[:6]]  # cap rows for readability
            camp["arsenalTable"] = rows
