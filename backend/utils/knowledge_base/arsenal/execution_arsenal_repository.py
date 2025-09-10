# =============================
# ARSENAL: Assets & Channels
# =============================
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
from backend.utils.knowledge_base.arsenal.arsenal_models import Asset, Channel, Persona, Play, Stage, STAGE_TO_PURPOSE
from backend.utils.graph_base.graph_data.asset_utils.save_and_load_arsenal import load_assets_from_arsenal_json, load_channels_from_arsenal_json

#----- Helpers for persona matching-----
def _seniority_rank(seniority: str) -> int:
    # Lower is more junior
    ranks = {"Junior": 0, "Operator": 1, "Manager": 2, "Senior": 3, "Executive": 4}
    return ranks.get(seniority, 2)  # Default to Manager if unknown

def _persona_match(persona_alias, candidate_persona):
    # Exact match
    if all(persona_alias.get(k) == candidate_persona.get(k) for k in ("title", "department", "seniority")):
        return 3
    # Direct match: department & seniority
    if (persona_alias.get("department") == candidate_persona.get("department") and
        persona_alias.get("seniority") == candidate_persona.get("seniority")):
        return 2
    # Broad match: department & closest seniority
    if persona_alias.get("department") == candidate_persona.get("department"):
        # Compute seniority closeness
        pa_rank = _seniority_rank(persona_alias.get("seniority"))
        cp_rank = _seniority_rank(candidate_persona.get("seniority"))
        # Allow any level difference
        if abs(pa_rank - cp_rank) <= 4:
            return 1
    return 0

def _find_best_persona_match(persona_alias, candidate_personas):
    # Returns best match score and matching persona dict
    best_score = 0
    best_persona = None
    best_distance = float("inf")
    pa_rank = _seniority_rank(persona_alias.get("seniority"))
    for cp in candidate_personas:
        # Exact match
        if all(persona_alias.get(k) == cp.get(k) for k in ("title", "department", "seniority")):
            return 3, cp
        # Direct match: department & seniority
        if (persona_alias.get("department") == cp.get("department") and
            persona_alias.get("seniority") == cp.get("seniority")):
            if best_score < 2:
                best_score = 2
                best_persona = cp
                best_distance = 0
        # Broad match: department & closest seniority
        if persona_alias.get("department") == cp.get("department"):
            cp_rank = _seniority_rank(cp.get("seniority"))
            distance = abs(pa_rank - cp_rank)
            if distance < best_distance:
                best_score = 1
                best_persona = cp
                best_distance = distance
    return best_score, best_persona

# ---- Arsenal Scoring ----
def _stage_weight(stage: Stage) -> float:
    return {"problem":0.9, "pain":1.0, "solution":1.05}[stage]

def _rank_plays_for(
    persona_alias: dict, stage: str, *,
    product_id: str,
    max_lead_days: Optional[int] = None,
    budget: Optional[float] = None,
    top_k: int = 8
) -> List[Play]:
    ASSETS: List[Asset] = load_assets_from_arsenal_json(product_id)
    CHANNELS: List[Channel] = load_channels_from_arsenal_json(product_id)
    purpose = STAGE_TO_PURPOSE[stage]

    a_candidates = []
    for a in ASSETS:
        if stage not in a.stages or a.purpose != purpose:
            continue
        match_score, _ = _find_best_persona_match(persona_alias, a.personas)
        if match_score > 0:
            a_candidates.append((a, match_score))

    if max_lead_days is not None:
        a_candidates = [(a, s) for a, s in a_candidates if a.production_days <= max_lead_days]

    c_candidates = []
    for c in CHANNELS:
        match_score, _ = _find_best_persona_match(persona_alias, [pf["persona"] for pf in c.persona_fit])
        if match_score > 0:
            c_candidates.append((c, match_score))

    if max_lead_days is not None:
        c_candidates = [(c, s) for c, s in c_candidates if c.lead_days <= max_lead_days]

    plays: List[Play] = []
    for a, a_score in a_candidates:
        for c, c_score in c_candidates:
            # Find persona_fit score for channel
            pf_score = 0.0
            for pf in c.persona_fit:
                if _persona_match(persona_alias, pf["persona"]) > 0:
                    pf_score = pf["score"]
                    break
            s_fit = c.stage_fit.get(stage, 0.0)
            proof = a.evidence_strength
            speed = 1.0 / (1.0 + (a.production_days + c.lead_days)/14.0)
            cost  = 1.0 / (1.0 + (a.est_cost/8000.0 + c.cost_index))
            reach = c.reach_score
            base = (0.35*proof + 0.2*reach + 0.2*pf_score + 0.15*s_fit + 0.10*speed) * _stage_weight(stage)
            score = base * cost
            if budget is not None and a.est_cost > budget:
                score *= 0.5  # penalty
            plays.append(Play(
                asset_id=a.id, channel_id=c.id, score=score,
                rationale=f"asset={a.format}/{a.name}; channel={c.name}; persona_match={a_score}/{c_score}; base={score:.3f}"
            ))
    plays.sort(key=lambda p: p.score, reverse=True)
    return plays[:top_k]

def _materialize_playbook(
    persona: Persona, stage: Stage, *,
    product_id: str,
    max_lead_days: Optional[int] = None,
    budget: Optional[float] = None,
    top_k: int = 5,
    roi_multiplier: float = 1.0
) -> List[Dict]:
    top = _rank_plays_for(persona, stage, product_id=product_id, max_lead_days=max_lead_days, budget=budget, top_k=top_k)
    ASSETS: List[Asset] = load_assets_from_arsenal_json(product_id)
    CHANNELS: List[Channel] = load_channels_from_arsenal_json(product_id)
    a_map = {a.id:a for a in ASSETS}
    c_map = {c.id:c for c in CHANNELS}
    out: List[Dict] = []
    for p in top:
        a = a_map[p.asset_id]; c = c_map[p.channel_id]
        out.append({
            "persona": persona,
            "stage": stage,
            "asset": {
                "id": a.id, "name": a.name, "format": a.format,
                "production_days": a.production_days, "est_cost": a.est_cost,
                "evidence_strength": a.evidence_strength, "evergreen": a.evergreen
            },
            "channel": {
                "id": c.id, "name": c.name, "type": c.type,
                "reach_score": c.reach_score, "cost_index": c.cost_index, "lead_days": c.lead_days
            },
            "base_score": round(p.score, 3),
            "priority_score": round(p.score * roi_multiplier, 3),
            "why": p.rationale + f"; roi_x={roi_multiplier:.2f}"
        })
    return out
