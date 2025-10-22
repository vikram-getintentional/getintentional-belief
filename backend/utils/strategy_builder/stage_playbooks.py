# stage_playbooks.py
from datetime import date, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import copy
import math


from backend.utils.inference.rcs_generators.generate_rcs_fast import generate_rcs
from backend.utils.inference.rcs_generators.rcs_helpers.campaign_managers import _campaigns_from_sequences_using_arsenal, _slug
from backend.utils.knowledge_base.arsenal.execution_arsenal_repository import _derive_broad_pairs, suggest_broad_awareness_plays


# ---------------------------
# Broad awareness selectors
# ---------------------------

_BROAD_CHANNEL_TYPES = {
    "search", "display", "linkedin_ads", "social_paid", "social_organic",
    "pr_newswire", "content_syndication", "event", "programmatic_display"
}

_BROAD_PURPOSE = {
    "thought_leadership", "brand_presence", "community_event",
    "vendor_validation", "event_recap"
}






import math
from collections import Counter, defaultdict

_STAGE_WEIGHT = {"problem": 1.0, "pain": 0.9, "execution": 0.6, "resolution": 0.5}

def _normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("capability:", "").replace("job:", "").replace("pain:", "")
    return s

def _clusterish_key(label: str) -> str:
    # extremely light grouping: normalize and keep top 3 keywords
    toks = [t for t in _normalize_label(label).split() if t.isalpha() and len(t) > 3]
    return " ".join(sorted(toks)[:3]) or _normalize_label(label)[:40]

def _top_focused_concerns(sequences: list, max_groups: int = 3, max_members_per: int = 3):
    """
    Input: list of sequences (each has steps with concern_label + stage + lift_proxy)
    Output: list of groups: [{"primary": "...", "labels": [...], "score": float}]
    """
    # 1) pool all concerns with a weighted score
    pool = []
    for seq in sequences or []:
        for st in (seq.get("sequence") or []):
            label = st.get("concern_label") or st.get("cid")
            stage = (st.get("stage") or "").lower()
            lift  = float(st.get("lift_proxy") or 0.0)
            w     = _STAGE_WEIGHT.get(stage, 0.6)
            pool.append((label, w * lift))

    if not pool:
        return []

    # 2) light clustering by “clusterish key”
    clusters = defaultdict(list)
    for label, score in pool:
        key = _clusterish_key(label)
        clusters[key].append((label, score))

    # 3) score clusters by sum of member scores
    ranked = []
    for key, items in clusters.items():
        # collapse dup labels inside cluster
        by_label = defaultdict(float)
        for lbl, sc in items:
            by_label[lbl] += sc
        # sort within cluster
        members = sorted(by_label.items(), key=lambda x: x[1], reverse=True)
        cluster_score = sum(v for _, v in members)
        # keep top N labels per cluster
        labels = [lbl for lbl, _ in members[:max_members_per]]
        primary = labels[0]
        ranked.append({
            "primary": primary,
            "labels": labels,
            "score": cluster_score,
        })

    ranked.sort(key=lambda g: g["score"], reverse=True)
    return ranked[:max_groups]




# -----------------------
# Stage labels & policies
# -----------------------

STAGE_LABELS = {
    "pre_zmot": "Pre-ZMOT",
    "zmot": "ZMOT",
    "problem_realization": "Problem Realization",
    "discovery": "Discovery",
    "barriers": "Barriers",
    "implementation": "Implementation",
}

# “Portfolio policy” knobs you can surface in UI (breadth/depth/mutation)
STAGE_PORTFOLIO_POLICY = {
    "pre_zmot":            {"breadth": 0.65, "depth": 0.20, "mutation": 0.15},
    "zmot":                {"breadth": 0.50, "depth": 0.35, "mutation": 0.15},
    "problem_realization": {"breadth": 0.30, "depth": 0.55, "mutation": 0.15},
    "discovery":           {"breadth": 0.25, "depth": 0.65, "mutation": 0.10},
    "barriers":            {"breadth": 0.20, "depth": 0.70, "mutation": 0.10},
    "implementation":      {"breadth": 0.20, "depth": 0.70, "mutation": 0.10},
}

_STAGE_ORDER = {"problem": 0, "pain": 1, "execution": 2, "resolution": 3}

def _stage_norm(s: Optional[str]) -> str:
    if not s: return ""
    x = s.strip().lower()
    return "solution" if x == "resolution" else x
def _to_timeframe(idx: int) -> Dict[str, str]:
    start = date.today() + timedelta(days=14*idx)
    end   = start + timedelta(days=84)  # ~12 weeks
    return {"startDate": start.isoformat(), "endDate": end.isoformat()}


# ------------------------------------------------
# Public API (import this in build_account_strategy)
# ------------------------------------------------

def infer_account_stage(report: Dict[str, Any], default: str = "pre_zmot") -> str:
    """
    Heuristic: infer belief stage from report signals.
    You can replace with a learned classifier later.
    """
    baseline = (report.get("baseline") or {})
    win = float(baseline.get("win_likelihood") or 0.0)

    # Evidence of engagement (you can pass this into generate_rcs in your app)
    engaged = report.get("engaged_nodes") or []

    # Cheap heuristics (tune thresholds later)
    # If we have sequences that end with solution, assume ≥ discovery.
    seqs = report.get("concern_sequences") or []
    ends_in_solution = any((_stage_norm((s.get("sequence") or [])[-1].get("stage")) == "solution")
                           for s in seqs if s.get("sequence"))

    # Persona mix: high involvement + high activation → later stages
    personas = (report.get("top_personas") or {}).get("by_involvement") or []
    hi_involved = [p for p in personas if float(p.get("involvement") or 0) >= 0.02]
    hi_act = [p for p in personas if float(p.get("activation") or 0) >= 0.30]
    hi_hi = len({p["id"] for p in hi_involved}) and len({p["id"] for p in hi_act})

    # Rough rules
    if not engaged and win < 0.15:
        return "pre_zmot"
    if engaged and win < 0.22:
        return "zmot"
    if ends_in_solution and win < 0.30:
        return "problem_realization"
    if ends_in_solution and win >= 0.30 and not _has_blockers(report):
        return "discovery"
    if _has_blockers(report):
        return "barriers"
    if hi_hi and win >= 0.45:
        return "implementation"
    return default


def _sequence_theme_label(seq: dict) -> str:
    t = (seq.get("theme") or {}).get("title") or ""
    sub = (seq.get("theme") or {}).get("subtitle") or ""
    return f"{t} — {sub}" if (t and sub) else (t or sub or "Resolve prioritized concerns")

def _concerns_from_sequence(seq: dict, k: int = 6) -> list:
    out = []
    for st in (seq.get("sequence") or []):
        lbl = st.get("concern_label") or st.get("cid")
        if lbl:
            out.append(lbl)
    # stable, compact
    return list(dict.fromkeys(out))[:k]

def _top_concerns_from_report(report: dict, k: int = 6) -> list:
    # Use the concern_backlog if present, else concerns_flat
    backlog = (report or {}).get("concern_backlog") or (report or {}).get("concerns_flat") or []
    rows = []
    for r in backlog:
        label = r.get("concern_label") or r.get("cid")
        lift  = float(r.get("lift_proxy", 0.0))
        if label:
            rows.append((lift, label))
    rows.sort(key=lambda x: x[0], reverse=True)
    # dedupe labels preserving order
    seen, out = set(), []
    for _, lbl in rows:
        if lbl not in seen:
            out.append(lbl); seen.add(lbl)
        if len(out) >= k: break
    return out

def _beautify_campaigns_with_theme(campaigns: list, seqs: list, report: dict = None) -> list:
    """
    - Dedup campaigns by id
    - If campaign looks sequence-driven, overwrite description with theme label
    - Ensure each arsenal row has concernsAddressed; use sequence concerns if possible
      else fall back to top-concerns from the report (for broad campaigns)
    """
    if not isinstance(campaigns, list):
        campaigns = list(campaigns)

    # Index sequences by a rough key (we’ll match by campaign index order)
    seqs = seqs or []
    seq_by_idx = {i: s for i, s in enumerate(seqs)}

    # Dedup by campaign id
    seen_ids = set()
    out = []
    for i, camp in enumerate(campaigns):
        if not isinstance(camp, dict):
            continue
        cid = camp.get("id") or f"camp_{i:02d}"
        if cid in seen_ids:
            # skip duplicates (this is likely your “double broad”)
            continue
        seen_ids.add(cid)

        c = dict(camp)  # shallow clone

        # 1) Description polishing
        # If this is a sequence-aligned campaign (same index), prefer theme title/subtitle
        seq = seq_by_idx.get(i)
        if seq:
            c["description"] = _sequence_theme_label(seq)
        else:
            # leave broad awareness or pre-titled campaigns as-is
            c["description"] = c.get("description") or "Resolve prioritized concerns"

        # 2) Concerns enrichment for rows
        rows = c.get("arsenalTable") or []
        if seq:
            concerns = _concerns_from_sequence(seq)
        else:
            concerns = _top_concerns_from_report(report, k=6)

        filled_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            rr = dict(row)
            rr.setdefault("concernsAddressed", concerns)
            filled_rows.append(rr)
        c["arsenalTable"] = filled_rows

        out.append(c)

    return out



def _broad_awareness_campaign(account_id: str, seqs: list, *, product_id: str) -> Dict[str, Any]:
    """
    Portfolio-friendly ‘catch-all’ campaign when we need broad reach early.
    Uses repository heuristic scoring for breadth; no persona consent needed.
    """
    print("Adding broad awareness campaign for account:", account_id or "unknown")
    plays = suggest_broad_awareness_plays(
        product_id=product_id,
        top_k=6,
        # optional: limit channels to ad/pr/display types if you want
        # channels_whitelist=["linkedin_ads","display","search_ads","pr","event","youtube","x_twitter"],
    )
    seen_types = set()
    deduped = []
    for p in plays:
        ctype = p["channel"]["type"]
        if ctype in seen_types: 
            continue
        deduped.append(p)
        seen_types.add(ctype)
        if len(deduped) >= 6: break
    plays = deduped
    print("added broad plays")

    def _row(play):
        return {
            "asset": play["asset"],
            "channel": play["channel"],
            "fitment": play.get("fitment") or "General",
            "engagement": "Broad",
            "expectedLift": "+10%",  # cosmetic default, you can model this later
            "why": play.get("why", ""),
            "breadthScore": play.get("breadthScore", 0.0),
            "concernsAddressed": _concerns_from_sequences(seqs)[:6] if seqs else [],
        }

    return {
        "id": f"camp_{account_id}_broad_00",
        "description": "Broad Awareness & Warm-Up",
        "timeframe": _to_timeframe(0),
        "personas": [],
        "arsenalTable": [_row(p) for p in plays],
    }

# Variant - Tighten focus on broad awareness campaigns
def _tight_broad_awareness_campaign(
    account_id: str,
    sequences: list,
    *,
    assets: list,      # from repository
    channels: list,    # from repository
    k_groups: int = 3
) -> dict:
    focus_groups = _top_focused_concerns(sequences, max_groups=k_groups, max_members_per=3)
    # If nothing to focus, fall back to a single generic group
    if not focus_groups:
        focus_groups = [{"primary":"Awareness", "labels":[], "score":0.0}]

    # choose broad-friendly assets/channels (you already have this logic)
    broad_pairs = _derive_broad_pairs(assets, channels, top_n=2)  # [(asset, channel, why, breadthScore), ...]

    rows = []
    for grp in focus_groups:
        for (a, c, why_base, breadth) in broad_pairs:
            rows.append({
                "asset":   {"id": a.id, "name": a.name, "format": a.format, "evergreen": getattr(a, "evergreen", False)},
                "channel": {"id": c.id, "name": c.name, "type": c.type, "reach_score": getattr(c, "reach_score", 0.0)},
                "fitment": "Awareness",
                "engagement": "Broad",
                "expectedLift": "+10%",
                "primaryConcern": grp["primary"],
                "concernsAddressed": grp["labels"],    # <= small focused set
                "breadthScore": float(breadth),
                "why": f"{why_base}; broad focus: {grp['primary']}"
            })

    return {
        "id": f"camp_{_slug(account_id)}_broad_00",
        "description": "Broad Awareness & Warm-Up",
        "timeframe": _to_timeframe(0),
        "personas": [],
        "arsenalTable": rows
    }


def build_stage_campaign_set(
    *,
    G,
    account_id: Optional[str],
    product_id: Optional[str],
    report: Dict[str, Any],
    stage: Optional[str] = None,
    plays_per_step: int = 1,
    archetype: Optional[Dict] = None,
    limits: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    
    print("Building stage plan for:", product_id, " at stage:", stage or "inferred")
    stage = stage or infer_account_stage(report, default="pre_zmot")
    label = STAGE_LABELS.get(stage, stage)
    policy = copy.deepcopy(STAGE_PORTFOLIO_POLICY.get(stage, STAGE_PORTFOLIO_POLICY["pre_zmot"]))
    limits = limits or {}

    # 1) choose sequences for stage
    seqs = report.get("concern_sequences") or []
    print("Total sequences available:", len(seqs))
    print("Selecting sequences for stage:", stage)
    selected = _select_sequences_for_stage(seqs, report, stage, limits=limits)

    # 2) if too few, synthesize from backlog to avoid empty UI
    if not selected:
        selected = _fallback_sequences_from_backlog(report)
    # --- NEW: one exclusion set per quarter/stage plan ---
    seen_pairs: set[tuple[str, str]] = set()
    # 3) persona/concern-driven campaigns
    stage_report = {"concern_sequences": selected}
    campaigns = _campaigns_from_sequences_using_arsenal(
        G=G,
        account_id=account_id,
        product_id=product_id,
        report=stage_report,
        start_idx=0,
        plays_per_step=plays_per_step,
        archetype=archetype or {},
        exclude_pairs=seen_pairs,
    )


    print("Generated campaigns in stage_playbooks.. proceeding to broadmatch")

    # 4) ALWAYS add a broad awareness campaign in pre-ZMOT (optionally ZMOT too)
    if stage in {"pre_zmot", "zmot"}:  # or {"pre_zmot", "zmot"} if you prefer both
        campaigns.insert(0, _broad_awareness_campaign(account_id=account_id, seqs=selected, product_id=product_id))
        print("Inserted broad awareness campaign")
    
    if not campaigns:
        campaigns = [_broad_awareness_campaign(account_id, seqs, product_id=product_id)]

    # 5) attach concerns to each arsenal row for transparency/editing
    concerns = _concerns_from_sequences(selected)
    for camp in campaigns:
        for row in (camp.get("arsenalTable") or []):
            if isinstance(row, dict):
                row.setdefault("concernsAddressed", concerns)
    print("Attached concerns to campaigns")
    campaigns = _beautify_campaigns_with_theme(campaigns, selected, report)
    print("returning build_stage_campaign_set")
    personas =[]
    for s in selected:
        seqn = s.get("sequence", [])
        if not seqn or len(seqn)<=0:
            print("Selected sequence has no steps.")
            continue
        else:
            for sn in seqn:
                if not sn.get("persona_label"):
                    print("Step has no persona label.")
                    continue
                persona_label = sn.get("persona_label")
                print("Step persona label:", persona_label)
                personas.append(persona_label)
         
    return {
        "stage": stage,
        "stageLabel": label,
        "policy": policy,
        "campaigns": campaigns,
        "personas": personas,
        "debug": {
            "selectedSequenceCount": len(selected),
            "selectedSequences": [s.get("sequence_id") for s in selected],
        }
    }

def simulate_next_stage(
    *,
    product_subgraph,
    current_report: Dict[str, Any],
    engage_personas: List[str],
    boost_factor: float = 2.0,
) -> Dict[str, Any]:
    """
    Minimal sim: re-run generate_rcs with new 'engaged_nodes' to preview a stage shift.
    You decide how to translate persona ids → engaged nodes (job/pain/cap mapping if needed).
    """
    engaged_payload = [{"id": pid, "type": "persona", "weight": 1.0} for pid in engage_personas]
    G2, rpt2 = generate_rcs(
        product_subgraph,
        engaged_nodes=engaged_payload,
        boost_factor=boost_factor,
    )
    return {"graph": G2, "report": rpt2}

# -----------------------
# Selection implementations
# -----------------------

STAGE_ORDER = {"problem": 0, "pain": 1, "execution": 2, "resolution": 3}
def _norm(s: Optional[str]) -> str:
    s = (s or "").lower().strip()
    return "solution" if s == "resolution" else s

def _seq_stats(seq: dict) -> dict:
    steps = seq.get("sequence") or []
    if not steps:
        return {"first": None, "last": None, "hasPain": False, "hasExec": False, "hasRes": False}
    stages = [_norm(s.get("stage")) for s in steps]
    return {
        "first": stages[0],
        "last":  stages[-1],
        "hasPain": "pain" in stages,
        "hasExec": "execution" in stages,
        "hasRes":  ("solution" in stages) or ("resolution" in (s.get("stage") or "").lower() for s in steps),
    }

def _select_sequences_for_stage(
    sequences: list[dict],
    report: dict,
    stage: str,
    *,
    limits: Optional[dict] = None
) -> list[dict]:
    limits = limits or {}
    max_sequences = limits.get("max_sequences", 2)

    bucket: list[tuple[float, dict]] = []
    for seq in (sequences or []):
        st = _seq_stats(seq)
        score = 0.0

        if stage == "pre_zmot":
            # Start with problem/pain; earlier stage is better
            if st["first"] in {"problem", "pain"}:
                score = 1.0
                # prefer longer arc (toward execution/solution)
                if st["hasExec"]: score += 0.2
                if st["hasRes"]:  score += 0.2

        elif stage == "zmot":
            # Pain present and trending toward action (pain→execution)
            if st["hasPain"]:
                score = 1.0
                if st["hasExec"]: score += 0.3
                if st["hasRes"]:  score += 0.1
                if st["first"] == "pain": score += 0.1

        elif stage == "problem_realization":
            # Execution present, ideally ending in resolution
            if st["hasExec"]:
                score = 1.0
                if st["hasRes"]:  score += 0.3
                if st["first"] in {"problem","pain"}: score += 0.1

        elif stage == "discovery":
            # Ends in solution/resolution or includes it
            if st["hasRes"]:
                score = 1.0
                if st["hasExec"]: score += 0.2

        elif stage == "barriers":
            # Prefer sequences with many steps and resolution present
            if st["hasRes"]:
                score = 1.0
                # length boosts (more stakeholders/concerns to unblock)
                score += 0.1 * min(4, len(seq.get("sequence") or []))

        elif stage == "implementation":
            # Resolution anchored; short execution→resolution arcs
            if st["hasRes"]:
                score = 1.0
                if st["last"] in {"solution"}: score += 0.1

        # small tie-breaker by final_win
        score += float(seq.get("final_win") or 0.0)

        if score > 0.0:
            bucket.append((score, seq))

    bucket.sort(key=lambda t: t[0], reverse=True)
    return [seq for _, seq in bucket[:max_sequences]]


def _persona_scores(report: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    pinv, pact = {}, {}
    for p in ((report.get("top_personas") or {}).get("by_involvement") or []):
        pid = p.get("id")
        if not pid: continue
        pinv[pid] = float(p.get("involvement") or 0.0)
        pact[pid] = float(p.get("activation") or 0.0)
    return pinv, pact

def _cap_per_persona(seqs: List[Dict[str, Any]], *, per_persona: int, cap: int) -> List[Dict[str, Any]]:
    out, seen = [], defaultdict(int)
    for s in seqs:
        pid = s.get("persona")
        if seen[pid] >= per_persona:
            continue
        out.append(s)
        seen[pid] += 1
        if len(out) >= cap:
            break
    return out

def _has_execution_step(seq: Dict[str, Any]) -> bool:
    for st in (seq.get("sequence") or []):
        if _stage_norm(st.get("stage")) == "execution":
            return True
    return False

def _has_blockers(report: Dict[str, Any]) -> bool:
    return bool(_blocker_personas(report))

def _blocker_personas(report: Dict[str, Any]) -> List[str]:
    pinv, pact = _persona_scores(report)
    # “blockers”: high involvement but relatively low activation
    # thresholds are placeholders; tune per product
    blockers = []
    for pid, inv in pinv.items():
        act = pact.get(pid, 0.0)
        if inv >= 0.02 and act < 0.22:
            blockers.append(pid)
    return blockers

def _fallback_sequences_from_backlog(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build minimal sequences from top backlog rows if sequences are empty.
    """
    backlog = report.get("concern_backlog") or []
    if not backlog:
        return []
    by_pid = defaultdict(list)
    for r in backlog[:20]:
        by_pid[r.get("pid")].append({
            "persona": r.get("pid"),
            "persona_label": r.get("persona_label") or r.get("pid"),
            "cid": r.get("cid"),
            "stage": r.get("stage"),
            "lift_proxy": float(r.get("lift_proxy") or 0.0),
            "concern_label": r.get("concern_label") or r.get("cid"),
        })
    out = []
    for pid, rows in by_pid.items():
        rows = [x for x in rows if x.get("cid")]
        rows.sort(key=lambda z: (_STAGE_ORDER.get(_stage_norm(z["stage"]), 99), -z["lift_proxy"]))
        if not rows: 
            continue
        seq = rows[:min(3, len(rows))]
        out.append({
            "sequence_id": f"seq:{pid}:{'|'.join([s['cid'] for s in seq])}",
            "persona": pid,
            "persona_label": seq[0]["persona_label"],
            "sequence": seq,
            "final_win": float(sum(s["lift_proxy"] for s in seq) / max(1, len(seq))),
        })
    out.sort(key=lambda s: -float(s.get("final_win") or 0.0))
    return out[:6]

def _concerns_from_sequences(seqs: List[Dict[str, Any]]) -> List[str]:
    names, seen = [], set()
    for seq in seqs:
        for st in (seq.get("sequence") or []):
            lab = st.get("concern_label") or st.get("cid")
            if lab and lab not in seen:
                names.append(lab); seen.add(lab)
    return names


