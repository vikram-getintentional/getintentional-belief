# =============================
# ARSENAL: Assets & Channels
# =============================
from __future__ import annotations

import math
from typing import Dict, List, Optional

from backend.utils.knowledge_base.arsenal.arsenal_models import (
    Asset, Channel, Play, Stage, STAGE_TO_PURPOSE
)
from backend.utils.graph_base.graph_data.asset_utils.save_and_load_arsenal import (
    load_assets_from_arsenal_json, load_channels_from_arsenal_json
)

# ----------- basic helpers -----------
def _safe_lower(x) -> str:
    try:
        return str(x or "").strip().lower()
    except Exception:
        return ""

def _safe_list(xs):
    return xs if isinstance(xs, (list, tuple)) else []

def _stage_norm(s: Optional[str]) -> str:
    s = _safe_lower(s)
    if s in ("problem", "problems", "problem_awareness"):
        return "problem"
    if s in ("pain", "pains", "pain_awareness"):
        return "pain"
    if s in ("solution", "solutions", "resolution", "resolve", "consideration"):
        return "solution"
    return "problem"

def _same_stage_family(query_stage: str, asset_stages: List[str]) -> bool:
    return _stage_norm(query_stage) in [_stage_norm(s) for s in (asset_stages or [])]

def _index_in(order: List[str], val: str) -> int:
    val = _safe_lower(val)
    try:
        return order.index(val)
    except ValueError:
        return -1

# ----------- persona matching -----------
def _seniority_rank(seniority: str) -> int:
    ranks = {"junior":0, "operator":1, "manager":2, "senior":3, "executive":4}
    return ranks.get(_safe_lower(seniority), 2)

def _persona_match(persona_alias, candidate_persona):
    # 3: exact title+dept+seniority; 2: dept+seniority; 1: dept only; 0: none
    if all(_safe_lower(persona_alias.get(k)) == _safe_lower(candidate_persona.get(k))
           for k in ("title","department","seniority")):
        return 3
    if _safe_lower(persona_alias.get("department")) == _safe_lower(candidate_persona.get("department")) and \
       _safe_lower(persona_alias.get("seniority"))  == _safe_lower(candidate_persona.get("seniority")):
        return 2
    if _safe_lower(persona_alias.get("department")) == _safe_lower(candidate_persona.get("department")):
        return 1
    return 0

# ----------- org-type ladders -----------
REVENUE_ORDER  = ["<1m","1-10m","10-50m","50-100m","100-250m","250-500m","500m-1b",">1b"]
EMPLOYEE_ORDER = ["1-10","11-50","51-200","201-500","501-1000","1001-5000","5001-10000",">10000"]
FUNDING_ORDER  = ["bootstrapped","pre-seed","seed","series a","series b","series c","late","public","acquired"]

def _range_match_score(order: List[str], query_val: str, asset_vals: List[str]) -> float:
    asset_vals = [_safe_lower(v) for v in _safe_list(asset_vals)]
    if not asset_vals:
        return 0.5
    q_idx = _index_in(order, query_val)
    if q_idx < 0:
        return 0.7 if asset_vals else 0.5
    if _safe_lower(query_val) in asset_vals:
        return 1.0

    a_indices = [i for i in (order.index(v) for v in asset_vals if v in order)]
    if not a_indices:
        return 0.5
    higher = [i for i in a_indices if i >= q_idx]
    lower  = [i for i in a_indices if i <  q_idx]

    LAMBDA_UP = 0.7
    LAMBDA_DN = 0.9
    if higher:
        d = min(i - q_idx for i in higher)
        return float(0.9 * math.exp(-LAMBDA_UP * d))
    d = min(q_idx - i for i in lower)
    return float(0.7 * math.exp(-LAMBDA_DN * d))

def _geo_match_score(query_geo: str, asset_geos: List[str]) -> float:
    asset_geos = [_safe_lower(g) for g in _safe_list(asset_geos)]
    q = _safe_lower(query_geo)
    if not asset_geos:
        return 0.5
    if q and q in asset_geos:
        return 1.0
    if "global" in asset_geos:
        return 0.9
    return 0.7

# ----------- master fitness -----------
def _compute_match_score(persona_alias, asset, query_archetype, query_stage):
    # Persona score
    persona_scores = []
    asset_personas = getattr(asset, "personas", []) or []
    if not asset_personas:
        persona_scores.append(0.5)  # generic asset
    else:
        for cp in asset_personas:
            if all(_safe_lower(persona_alias.get(k)) == _safe_lower(cp.get(k))
                   for k in ("title","department","seniority")):
                persona_scores.append(1.0); continue
            if (_safe_lower(persona_alias.get("department")) == _safe_lower(cp.get("department")) and
                _safe_lower(persona_alias.get("seniority"))  == _safe_lower(cp.get("seniority"))):
                persona_scores.append(0.85); continue
            if _safe_lower(persona_alias.get("department")) == _safe_lower(cp.get("department")):
                persona_scores.append(0.6); continue
            pa = _seniority_rank(persona_alias.get("seniority"))
            ca = _seniority_rank(cp.get("seniority"))
            persona_scores.append(max(0.0, 0.4 - 0.05 * abs(pa - ca)))
    persona_score = max(persona_scores) if persona_scores else 0.0

    # Stage score (soft)
    q_stage = _stage_norm(query_stage)
    asset_stages = [_stage_norm(s) for s in (getattr(asset, "stages", []) or [])]
    if q_stage in asset_stages:
        stage_score = 1.0
    elif _same_stage_family(q_stage, asset_stages):
        stage_score = 0.6
    elif asset_stages:
        stage_score = 0.35
    else:
        stage_score = 0.25

    # Org tags (soft)
    q_arch = query_archetype or {}
    def tag_score(q, tags):
        tags = [_safe_lower(t) for t in (tags or [])]
        if not q:        return 0.0
        if not tags:     return 0.4
        if _safe_lower(q) in tags: return 1.0
        return 0.6

    industry_score = tag_score(q_arch.get("industry"),        getattr(asset, "industry_tags", []))
    geo_score      = tag_score(q_arch.get("geography"),       getattr(asset, "geographies", []))
    rev_score      = tag_score(q_arch.get("revenue_range"),   getattr(asset, "revenue_ranges", []))
    emp_score      = tag_score(q_arch.get("employee_range"),  getattr(asset, "employee_ranges", []))

    # Blend
    fitness = (
        0.45 * persona_score +
        0.25 * stage_score   +
        0.18 * industry_score +
        0.06 * geo_score + 0.03 * rev_score + 0.03 * emp_score
    )
    return max(0.0, min(1.0, float(fitness)))

def _stage_weight(stage: Stage) -> float:
    return {"problem":0.9, "pain":1.0, "solution":1.05}[stage]

def _rank_plays_for(
    persona_alias: dict, stage: str, *,
    product_id: str,
    archetype: dict,
    top_k: int = 8
) -> List[Play]:
    ASSETS: List[Asset] = load_assets_from_arsenal_json(product_id) or []
    CHANNELS: List[Channel] = load_channels_from_arsenal_json(product_id) or []

    desired_purpose = STAGE_TO_PURPOSE.get(_stage_norm(stage), STAGE_TO_PURPOSE["problem"])

    def score_assets(pool):
        out = []
        for a in pool:
            fitness = _compute_match_score(persona_alias, a, archetype, stage)
            if fitness > 0:
                out.append((a, fitness))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    strict_pool = [a for a in ASSETS if _safe_lower(a.purpose) == _safe_lower(desired_purpose)]
    ranked = score_assets(strict_pool)
    if not ranked:
        ranked = score_assets(ASSETS)  # relax purpose

    if not ranked:
        return []

    plays: List[Play] = []
    for a, fitness in ranked[:top_k]:
        best_channel = None
        best_channel_score = 0.0
        for c in CHANNELS:
            pf_score = 0.0
            for pf in (c.persona_fit or []):
                if _persona_match(persona_alias, pf.get("persona", {})) > 0:
                    pf_score = pf.get("score", 0.0)
                    break
            s_fit = (c.stage_fit or {}).get(_stage_norm(stage), 0.0)
            channel_score = 0.6 * pf_score + 0.4 * s_fit
            if channel_score > best_channel_score:
                best_channel_score = channel_score
                best_channel = c
        if best_channel:
            plays.append(Play(
                asset_id=a.id,
                channel_id=best_channel.id,
                score=round(fitness * (best_channel_score or 0.5), 3),
                rationale=f"fitness={round(fitness*100,1)}%; asset={a.name}; channel={best_channel.name}"
            ))

    if not plays and CHANNELS:
        a, fitness = ranked[0]
        c = CHANNELS[0]
        plays.append(Play(
            asset_id=a.id,
            channel_id=c.id,
            score=round(fitness * 0.5, 3),
            rationale=f"fitness={round(fitness*100,1)}%; asset={a.name}; channel={c.name} (fallback)"
        ))

    plays.sort(key=lambda p: p.score, reverse=True)
    return plays[:top_k]

# ---------- PUBLIC API ----------
def get_best_plays_for_concern(
    *,
    persona_alias: Dict,           # {"title","department","seniority"}
    concern_stage: str,           # "problem" | "pain" | "solution"
    product_id: str,
    archetype: Dict,              # {"industry","revenue_range","employee_range","geography","funding_stage"}
    top_k: int = 3
) -> List[Dict]:
    """
    Returns ranked plays (asset+channel+fitness+rationale) for a single concern.
    Always tries to return something if any assets exist.
    """
    stage = _stage_norm(concern_stage)
    assets = load_assets_from_arsenal_json(product_id) or []
    channels = load_channels_from_arsenal_json(product_id) or []
    plays = _rank_plays_for(persona_alias, stage, product_id=product_id, archetype=archetype, top_k=top_k)
    if not plays:
        return []

    a_map = {a.id: a for a in assets}
    c_map = {c.id: c for c in channels}

    out: List[Dict] = []
    for p in plays:
        a = a_map.get(p.asset_id)
        c = c_map.get(p.channel_id)
        if not a or not c:
            continue
        out.append({
            "asset":   {"id": a.id, "name": a.name, "format": a.format, "evergreen": getattr(a, "evergreen", False)},
            "channel": {"id": c.id, "name": c.name, "type": c.type, "reach_score": getattr(c, "reach_score", 0.0)},
            "fitness": float(p.score),
            "why":     p.rationale,
        })
    return out
