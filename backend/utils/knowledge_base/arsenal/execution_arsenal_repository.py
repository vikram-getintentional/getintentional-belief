# execution_arsenal_repository.py
# Drop-in replacement that ranks Asset × Channel plays against a "concern bundle".
# Public API: get_best_plays_for_concern(...)

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import math
import re

from backend.utils.graph_base.graph_data.asset_utils.save_and_load_arsenal import load_assets_from_arsenal_json, load_channels_from_arsenal_json

# -----------------------------
# Data models (lightweight)
# -----------------------------

@dataclass
class Asset:
    id: str
    name: str
    format: str = ""
    evergreen: bool = False
    tags: List[str] = None
    stages: List[str] = None          # ["problem","pain","execution","resolution"]
    personas: List[str] = None        # strings to match title hints (optional)
    departments: List[str] = None
    seniorities: List[str] = None
    industries: List[str] = None
    geographies: List[str] = None
    revenue_ranges: List[str] = None  # e.g., ["$10M-$50M"]
    employee_ranges: List[str] = None # e.g., ["51-200"]
    funding_stages: List[str] = None  # e.g., ["Series B", "Series C+"]
    text: str = ""                    # description/body for text match

@dataclass
class Channel:
    id: str
    name: str
    type: str = ""
    reach_score: float = 0.5           # 0..1
    personas: List[str] = None         # optional targeting
    industries: List[str] = None
    geographies: List[str] = None

# -----------------------------
# Safe JSON loaders
# -----------------------------

# --------------------------------------------
# Broad Awareness Play Suggestions (Repository)
# --------------------------------------------
from typing import List, Dict, Any, Optional, Tuple
import math
import re

# If you already have these utilities, reuse them:
# - load_assets_from_arsenal_json(product_id)
# - load_channels_from_arsenal_json(product_id)
# - _stage_norm, _persona_fit, _firmographic_fit, etc.
# Below we only add breadth-focused scorers.

_BROAD_CHANNEL_TYPES = {
    "linkedin_ads",
    "display",
    "search_ads",
    "pr",
    "event",
    "youtube",
    "x_twitter",
}


def _text_tokens(s: str) -> List[str]:
    s = (s or "").lower()
    return re.findall(r"[a-z0-9]+", s)


def _safe_list(v, default=None):
    if default is None:
        default = []
    if isinstance(v, list):
        return v
    if v is None:
        return default
    return [v]


def _coerce_asset(d: Dict[str, Any]) -> Asset:
    """
    Accept either an Asset instance or a dict-like payload and return a fully-populated
    Asset with safe defaults. Normalizes:
      - stages -> normalized stage tokens
      - personas -> list[str] (converts persona dicts to title strings)
      - tags/industries/geographies -> lists
      - text -> description/text fallback
    """
    
    def _persona_to_str(p):
        if isinstance(p, dict):
            # prefer title, fall back to role/name; include dept for disambiguation
            title = p.get("title") or p.get("role") or p.get("name") or ""
            dept = p.get("department") or p.get("dept") or ""
            sr = p.get("seniority") or p.get("seniority_level") or ""
            if title and dept:
                return f"{title} ({dept})"
            if title:
                return title
            if dept:
                return dept
            if sr:
                return sr
            return ""
        return str(p)

    # If already an Asset instance, read attributes safely
    if isinstance(d, Asset):
        src = d
        raw_stages = getattr(src, "stages", None) or []
        raw_personas = getattr(src, "personas", None) or []
        tags = getattr(src, "tags", None) or []
        industries = getattr(src, "industries", None) or []
        geographies = getattr(src, "geographies", None) or []
        revenue_ranges = getattr(src, "revenue_ranges", None) or []
        employee_ranges = getattr(src, "employee_ranges", None) or []
        funding_stages = getattr(src, "funding_stages", None) or []
        text = getattr(src, "text", "") or ""
        return Asset(
            id=str(getattr(src, "id", "") or "asset"),
            name=str(getattr(src, "name", "") or "Asset"),
            format=str(getattr(src, "format", "") or ""),
            evergreen=bool(getattr(src, "evergreen", False)),
            tags=_safe_list(tags),
            stages=[_stage_norm(s) for s in _safe_list(raw_stages)],
            personas=[p for p in (_persona_to_str(p) for p in _safe_list(raw_personas)) if p],
            departments=_safe_list(getattr(src, "departments", None)),
            seniorities=_safe_list(getattr(src, "seniorities", None)),
            industries=_safe_list(industries),
            geographies=_safe_list(geographies),
            revenue_ranges=_safe_list(revenue_ranges),
            employee_ranges=_safe_list(employee_ranges),
            funding_stages=_safe_list(funding_stages),
            text=str(text),
        )

    # Otherwise assume dict-like input (tolerate None)
    data = d or {}
    raw_personas = data.get("personas") or data.get("persona_tags") or []
    # support legacy persona list-of-dicts
    persona_list = []
    for p in _safe_list(raw_personas):
        s = _persona_to_str(p)
        if s:
            persona_list.append(s)

    # tags may be 'industry_tags' or 'tags'
    tags = data.get("tags") or data.get("industry_tags") or data.get("purpose") or []
    stages_raw = data.get("stages") or data.get("stage_tags") or []
    return Asset(
        id=str(data.get("id") or data.get("asset_id") or data.get("name") or "asset"),
        name=str(data.get("name") or data.get("title") or "Asset"),
        format=str(data.get("format") or ""),
        evergreen=bool(data.get("evergreen", False)),
        tags=_safe_list(tags),
        stages=[_stage_norm(s) for s in _safe_list(stages_raw)],
        personas=persona_list,
        departments=_safe_list(data.get("departments") or data.get("department_tags")),
        seniorities=_safe_list(data.get("seniorities") or data.get("seniority_tags")),
        industries=_safe_list(data.get("industries") or data.get("industry_tags")),
        geographies=_safe_list(data.get("geographies") or data.get("geo") or data.get("regions")),
        revenue_ranges=_safe_list(data.get("revenue_ranges") or data.get("revenue_range")),
        employee_ranges=_safe_list(data.get("employee_ranges") or data.get("employee_range")),
        funding_stages=_safe_list(data.get("funding_stages") or data.get("funding_stage")),
        text=str(data.get("text") or data.get("description") or data.get("notes") or ""),
    )

def _coerce_channel(d: Dict[str, Any]) -> Channel:
    return Channel(
        id=str(d.get("id") or d.get("channel_id") or d.get("name") or "channel"),
        name=str(d.get("name") or "Channel"),
        type=str(d.get("type") or ""),
        reach_score=float(d.get("reach_score", 0.5)),
        personas=_safe_list(d.get("personas")),
        industries=_safe_list(d.get("industries")),
        geographies=_safe_list(d.get("geographies")),
    )


# -----------------------------
# Normalization helpers
# -----------------------------

_STAGE_ALIASES = {
    "problem": "problem",
    "problems": "problem",
    "pain": "pain",
    "pains": "pain",
    "execution": "execution",
    "job": "execution",
    "jobs": "execution",
    "solution": "resolution",
    "solutions": "resolution",
    "resolve": "resolution",
    "resolution": "resolution",
    "capability": "resolution",
    "capabilities": "resolution",
}

FUNDING_ORDER = {
    "bootstrap": 0, "pre-seed": 1, "seed": 2,
    "series a": 3, "series b": 4, "series c": 5, "series c+": 5,
    "series d": 6, "series e": 7,
    "private": 8, "public": 9,
}

def _stage_norm(s: Optional[str]) -> str:
    if not s:
        return "execution"
    key = str(s).strip().lower()
    return _STAGE_ALIASES.get(key, key)

def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _norm(s))

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def _prefix_match(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return 1.0 if a in b or b in a else 0.0

def _range_score(value: str, allowed: List[str]) -> float:
    if not value or not allowed:
        return 0.0
    v = _norm(value)
    allowed_norm = [_norm(x) for x in allowed]
    return 1.0 if v in allowed_norm else 0.0

def _funding_score(value: str, allowed: List[str]) -> float:
    if not value or not allowed:
        return 0.0
    v = FUNDING_ORDER.get(_norm(value), None)
    if v is None:
        return 0.0
    best = 0.0
    for a in allowed:
        av = FUNDING_ORDER.get(_norm(a), None)
        if av is None:
            continue
        # distance-based decay (closer rounds get higher score)
        dist = abs(v - av)
        best = max(best, 1.0 / (1.0 + dist))
    return best

def _persona_fit(persona_alias: Dict[str, str], asset: Asset) -> Tuple[float, List[str]]:
    title = persona_alias.get("title") or ""
    dept  = persona_alias.get("department") or ""
    sr    = persona_alias.get("seniority") or ""

    reasons = []
    score = 0.0

    # Exact (title+dept+seniority)
    t_hit = any(_prefix_match(title, p) for p in (asset.personas or []))
    d_hit = any(_prefix_match(dept,  p) for p in (asset.departments or []))
    s_hit = any(_prefix_match(sr,    s) for s in (asset.seniorities or []))

    if t_hit and d_hit and s_hit:
        score = 1.0; reasons.append("exact title+dept+seniority")
    elif (d_hit and s_hit) or (t_hit and d_hit):
        score = 0.75; reasons.append("dept+seniority (or title+dept) match")
    elif d_hit:
        score = 0.5; reasons.append("department match")
    elif s_hit:
        score = 0.35; reasons.append("seniority match")
    elif t_hit:
        score = 0.35; reasons.append("title match")

    return score, reasons

def _stage_fit(stage: str, asset: Asset) -> Tuple[float, List[str]]:
    if not asset.stages:
        return 0.2, ["no stage tags"]  # neutral
    stage_n = _stage_norm(stage)
    if stage_n in asset.stages:
        return 1.0, [f"stage:{stage_n}"]
    return 0.0, []

def _concern_match(concern_label: str, asset: Asset) -> Tuple[float, List[str]]:
    # Safe guards: some assets may lack tags/text/name
    concern_tokens = _tokenize(concern_label or "")
    asset_name = getattr(asset, "name", "") or ""
    asset_tags = getattr(asset, "tags", []) or []
    asset_text = getattr(asset, "text", "") or ""

    asset_tokens = _tokenize(" ".join([asset_name] + asset_tags + [asset_text]))
    sim = _jaccard(concern_tokens, asset_tokens)
    reasons = []
    if sim >= 0.4:
        reasons.append("strong concern-text overlap")
    elif sim >= 0.2:
        reasons.append("moderate concern-text overlap")
    return sim, reasons

def _firmographic_fit(archetype: Dict[str, Any], asset: Asset, channel: Channel) -> Tuple[float, List[str]]:
    ind = _norm(archetype.get("industry", ""))
    geo = _norm(archetype.get("geography", ""))
    rev = _norm(archetype.get("revenue_range", ""))
    emp = _norm(archetype.get("employee_range", ""))
    fund = _norm(archetype.get("funding_stage", ""))

    reasons = []
    score = 0.0

    # Industry / Geography — accept match on asset OR channel targeting
    ind_hit = _range_score(ind, asset.industries) or _range_score(ind, channel.industries)
    geo_hit = _range_score(geo, asset.geographies) or _range_score(geo, channel.geographies)
    if ind_hit > 0:
        reasons.append("industry match")
    if geo_hit > 0:
        reasons.append("geo match")

    # Revenue / Employee ranges (assets only)
    rev_hit = _range_score(rev, asset.revenue_ranges)
    emp_hit = _range_score(emp, asset.employee_ranges)
    if rev_hit > 0:
        reasons.append("revenue band match")
    if emp_hit > 0:
        reasons.append("employee band match")

    # Funding — distance-based
    fund_hit = _funding_score(fund, asset.funding_stages)
    if fund_hit > 0:
        reasons.append("funding stage proximity")

    # Weighted blend
    score = 0.35 * ind_hit + 0.25 * geo_hit + 0.2 * rev_hit + 0.2 * emp_hit
    score = max(score, 0.2 * fund_hit)  # allow funding to lift weak fits

    return score, reasons

def _engagement_score(channel: Channel, persona_strength: float) -> float:
    # Simple proxy: channel reach × (0.6 + 0.4*persona_strength)
    # persona_strength is 0..1 from persona matching tier
    return max(0.0, min(1.0, channel.reach_score * (0.6 + 0.4 * persona_strength)))

def _compose_why(parts: List[str]) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return "Best overall fit by concern, stage, persona, and firmographics."
    # de-duplicate while preserving order
    seen, uniq = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return "; ".join(uniq)

# -----------------------------
# Ranking
# -----------------------------

def _score_play(
    *,
    asset: Asset,
    channel: Channel,
    persona_alias: Dict[str, str],
    concern_label: str,
    concern_stage: str,
    archetype: Dict[str, Any],
) -> Tuple[float, float, str]:
    stage = _stage_norm(concern_stage)

    # Individual components
    concern_s, c_reasons      = _concern_match(concern_label, asset)
    stage_s, s_reasons        = _stage_fit(stage, asset)
    persona_s, p_reasons      = _persona_fit(persona_alias, asset)
    firmo_s, f_reasons        = _firmographic_fit(archetype or {}, asset, channel)
    engage_s                  = _engagement_score(channel, persona_s)

    # Weighted total fitness (tune weights as you like)
    # Emphasize concern match + stage + persona; firmographics and channel modulate.
    fitness = (
        0.35 * concern_s +
        0.20 * stage_s +
        0.25 * persona_s +
        0.15 * firmo_s +
        0.05 * engage_s
    )

    why = _compose_why(
        c_reasons + s_reasons + p_reasons + f_reasons +
        ([f"channel reach:{channel.reach_score:.2f}"] if channel.reach_score else [])
    )
    return float(fitness), float(engage_s), why


# --- CHANGE: add exclude_pairs param ---
def _rank_plays_for_concern(
    persona_alias: Dict[str, str],
    concern_label: str,
    concern_stage: str,
    *,
    product_id: str,
    archetype: Dict[str, Any],
    top_k: int = 3,
    exclude_pairs: Optional[set[tuple[str, str]]] = None,   # <-- NEW
) -> List[Dict]:
    print("running rank plays for concern")
    exclude_pairs = exclude_pairs or set()
    

    assets_raw = load_assets_from_arsenal_json(product_id) or []
    channels_raw = load_channels_from_arsenal_json(product_id) or []

    assets = []
    for a in assets_raw:
        assets.append(_coerce_asset(a))
    channels = []
    for c in channels_raw:
        channels.append(_coerce_channel(c))

    

    scored: list[tuple[float, float, Asset, Channel, str]] = []

    # score all asset×channel
    for a in assets:
        for c in channels:
            # --- NEW: skip combos already used this quarter ---
            if (getattr(a, "id", None), getattr(c, "id", None)) in exclude_pairs:
                continue

            fitness, engagement, why = _score_play(
                asset=a, channel=c,
                persona_alias=persona_alias,
                concern_label=concern_label,
                concern_stage=concern_stage,
                archetype=archetype or {},
            )
            scored.append((fitness, engagement, a, c, why))
    print(f"scored {len(scored)} asset×channel plays")

    # --- RELAXED FALLBACK to avoid empty (prevents “Play: Resolve … email”) ---
    if not scored:
        relaxed = [(a, c) for a in assets for c in channels
                   if (getattr(a, "id", None), getattr(c, "id", None)) not in exclude_pairs]
        if not relaxed:
            relaxed = [(a, c) for a in assets for c in channels]
        tmp = []
        for a, c in relaxed:
            simple = float(getattr(a, "engagement_rate", 0.05)) + float(getattr(c, "reach_score", 0.4))
            tmp.append((simple, 0.0, a, c, "relaxed-stage fallback"))
        tmp.sort(key=lambda t: t[0], reverse=True)
        scored = tmp

    scored.sort(key=lambda x: x[0], reverse=True)

    out: list[Dict] = []
    picked = 0
    for fit, eng, a, c, why in scored:
        aid, cid = getattr(a, "id", None), getattr(c, "id", None)
        if (aid, cid) in exclude_pairs:
            continue
        out.append({
            "asset":   {"id": a.id, "name": a.name, "format": a.format, "evergreen": getattr(a, "evergreen", False)},
            "channel": {"id": c.id, "name": c.name, "type": c.type, "reach_score": getattr(c, "reach_score", 0.0)},
            "fitness": float(fit),
            "persona": persona_alias,
            "engagement_score": float(eng),
            "why":     why,
        })
        # --- NEW: remember selection for this quarter ---
        exclude_pairs.add((aid, cid))
        picked += 1
        if picked >= max(1, top_k):
            break

    return out



def _asset_breadth_score(asset) -> Tuple[float, List[str]]:
    """
    Heuristic: reward assets that can reach many roles and generic stages.
    - More supported stages → broader
    - More personas across depts/seniority → broader
    - Formats that naturally travel well (press_release, event, blog, video, research_report)
    - Evergreen gets a small bump
    """
    reasons = []
    score = 0.0

    # Stage spread
    stages = set(_safe_list(getattr(asset, "stages", []) or []))
    if stages:
        stage_spread = min(len(stages) / 3.0, 1.0)  # cap
        score += 0.30 * stage_spread
        reasons.append(f"stageSpread:{stage_spread:.2f}")

    # Persona diversity (departments and seniority spread)
    personas = _safe_list(getattr(asset, "personas", []) or [])
    depts = { (p.get("department") or "").lower() for p in personas if isinstance(p, dict) }
    sens  = { (p.get("seniority") or "").lower()  for p in personas if isinstance(p, dict) }
    if depts:
        dept_div = min(len(depts)/3.0, 1.0)
        score += 0.25 * dept_div
        reasons.append(f"deptDiversity:{dept_div:.2f}")
    if sens:
        sen_div = min(len(sens)/3.0, 1.0)
        score += 0.15 * sen_div
        reasons.append(f"seniorityDiversity:{sen_div:.2f}")

    # Format reachability
    fmt = (getattr(asset, "format", "") or "").lower()
    fmt_bonus_map = {
        "press_release": 0.15,
        "event": 0.15,
        "blog": 0.10,
        "video": 0.12,
        "research_report": 0.10,
        "webinar": 0.08,
    }
    fmt_bonus = fmt_bonus_map.get(fmt, 0.05)
    score += fmt_bonus
    reasons.append(f"formatBonus:{fmt}:{fmt_bonus:.2f}")

    # Evergreen bump
    if getattr(asset, "evergreen", False):
        score += 0.05
        reasons.append("evergreen:+0.05")

    # Light penalty if asset is highly niche by industry_tags
    tags = _safe_list(getattr(asset, "industry_tags", []) or [])
    if tags and len(tags) == 1:
        score -= 0.05
        reasons.append("nicheIndustry:-0.05")

    return max(0.0, min(1.0, score)), reasons

def _channel_breadth_multiplier(channel) -> Tuple[float, List[str]]:
    """
    Favor channels with broad reach and no 1:1 consent requirement.
    """
    reasons = []
    ctype = (getattr(channel, "type", "") or "").lower()
    reach = float(getattr(channel, "reach_score", 0.0) or 0.0)

    base = 0.0
    if ctype in _BROAD_CHANNEL_TYPES:
        base = 0.7 + 0.3 * min(1.0, reach)  # 0.7..1.0
        reasons.append(f"broadType:{ctype}")
    else:
        base = 0.4 + 0.4 * min(1.0, reach)  # 0.4..0.8 for narrower channels
        reasons.append(f"narrowType:{ctype}")

    reasons.append(f"reach:{reach:.2f}")
    return base, reasons




# -----------------------------
# Public API (backward-compatible)
# -----------------------------

# --- CHANGE SIGNATURE: add exclude_pairs and pass to ranker ---
def get_best_plays_for_concern(
    *,
    persona_alias: Dict,
    concern_label: str,
    concern_stage: str,
    product_id: str,
    archetype: Dict,
    top_k: int = 3,
    exclude_pairs: Optional[set[tuple[str, str]]] = None,
) -> List[Dict]:
    """
    Returns a list of best plays (raw asset+channel+fitness dicts).
    Does NOT build arsenal rows or attach concern info.
    """
    plays = _rank_plays_for_concern(
        persona_alias=persona_alias,
        concern_label=concern_label,
        concern_stage=concern_stage,
        product_id=product_id,
        archetype=archetype,
        top_k=top_k,
        exclude_pairs=exclude_pairs,
    )
    if not plays:
        return []
    if exclude_pairs is not None:
        for p in plays:
            aid = (p.get("asset") or {}).get("id")
            cid2 = (p.get("channel") or {}).get("id")
            if aid and cid2:
                exclude_pairs.add((aid, cid2))
    return plays


def suggest_broad_awareness_plays(
    *,
    product_id: str,
    top_k: int = 6,
    channels_whitelist: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Repository function: return top-K broad awareness asset×channel recommendations.
    - Loads/normalizes assets & channels
    - Scores by breadth (asset) × breadthMultiplier (channel)
    - Adds 'why' and 'breadthScore' for transparency
    """
    print("suggest_broad_awareness_plays called for product:", product_id or "unknown")
    assets_raw = load_assets_from_arsenal_json(product_id) or []
    channels_raw = load_channels_from_arsenal_json(product_id) or []

    assets = []
    for a in assets_raw:
        assets.append(_coerce_asset(a))
    channels = []
    for c in channels_raw:
        channels.append(_coerce_channel(c))

    print("Loaded raw assets:", len(assets), " and channels:", len(channels))
    # Optional channel filter by id or type
    if channels_whitelist:
        wl = {c.lower() for c in channels_whitelist}
        channels = [
            c for c in channels
            if (getattr(c, "id", "").lower() in wl) or (getattr(c, "type", "").lower() in wl)
        ]

    scored: List[Tuple[float, float, Any, Any, str]] = []
    for a in assets:
        a_breadth, a_reasons = _asset_breadth_score(a)
        print("  Breadth score:", a_breadth, "reasons:", a_reasons)
        for c in channels:
            c_mult, c_reasons = _channel_breadth_multiplier(c)
            fitness = a_breadth * c_mult  # simple composite
            why = "; ".join(a_reasons + c_reasons)
            scored.append((fitness, a_breadth, a, c, why))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for fit, a_breadth, a, c, why in scored[:max(1, top_k)]:
        out.append({
            "asset":   {"id": a.id, "name": a.name, "format": a.format, "evergreen": getattr(a, "evergreen", False)},
            "channel": {"id": c.id, "name": c.name, "type": c.type, "reach_score": getattr(c, "reach_score", 0.0)},
            "fitness": float(fit),
            "breadthScore": float(a_breadth),
            "why": why,
        })
    print("Outputting suggest_broad_awareness_plays with", len(out), "plays")
    return out

def _derive_broad_pairs(assets, channels, top_n=2):
    return []