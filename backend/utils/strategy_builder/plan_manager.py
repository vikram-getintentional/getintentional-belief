# portfolio_merger.py (or fold into plan_manager.py)

from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone
import math
import copy
from typing import Dict, Any, List, Tuple, Optional, Set

from backend.utils.inference.rcs_generators.rcs_helpers.campaign_managers import _campaigns_from_sequences_using_arsenal

# --- If these helpers already exist somewhere, import and remove these stubs ---
STAGE_ORDER = ["pre_zmot", "zmot", "problem_realization", "discovery", "barriers", "implementation"]
STAGE_LABELS = {
    "pre_zmot": "Pre-ZMOT",
    "zmot": "ZMOT",
    "problem_realization": "Problem Realization",
    "discovery": "Discovery",
    "barriers": "Barriers",
    "implementation": "Implementation",
}

def _to_iso_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _avg(vals, default=0.0):
    xs = [v for v in vals if isinstance(v, (int, float))]
    return sum(xs) / len(xs) if xs else default

def _parse_pct(s):
    if not s or not isinstance(s, str): return None
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None

def _phase_for_sequence(seq: dict) -> str:
    steps = seq.get("sequence", [])
    if not steps: return "pre_zmot"
    first = (steps[0].get("stage") or "").lower()
    last  = (steps[-1].get("stage") or "").lower()
    if first in {"problem","pain"}:
        return "pre_zmot"
    if last == "resolution":
        return "zmot"  # “activation” style path
    return "problem_realization"  # coarse fallback

def _sequence_theme_label(seq: dict) -> str:
    t = (seq.get("theme") or {}).get("title")
    if t: return t
    # fallback to first concern label(s)
    labels = [s.get("concern_label") for s in (seq.get("sequence") or []) if s.get("concern_label")]
    return " · ".join(labels[:2]) or "Portfolio Campaign"

def _concerns_from_sequence(seq: dict) -> List[str]:
    labels = [s.get("concern_label") for s in (seq.get("sequence") or []) if s.get("concern_label")]
    return sorted({l for l in labels if l})

def _collect_sequences_from_rcs(rcs: Dict[str, Any]) -> List[Dict[str, Any]]:
    rb = rcs.get("rcs_brief") or {}
    seqs = rb.get("concern_sequences") or []
    # normalize final_win for weighting
    for s in seqs:
        s["final_win"] = float(s.get("final_win") or 0.0)
    return seqs

def _collect_backlog_as_singleton_sequences(rcs: Dict[str, Any], limit=3) -> List[Dict[str, Any]]:
    rb = rcs.get("rcs_brief") or {}
    backlog = rb.get("concern_backlog") or rb.get("concerns_flat") or []
    out = []
    for row in backlog[:limit]:
        seq = {
            "sequence_id": f"seq:{row.get('pid','persona')}:{row.get('cid','concern')}",
            "persona": row.get("pid"),
            "persona_label": row.get("persona_label"),
            "theme": {"title": row.get("concern_label") or row.get("cid") or "Concern", "subtitle": ""},
            "sequence": [{
                "persona": row.get("pid"),
                "persona_label": row.get("persona_label"),
                "cid": row.get("cid"),
                "stage": row.get("stage"),
                "lift_proxy": float(row.get("lift_proxy", 0.0)),
                "concern_label": row.get("concern_label") or row.get("cid")
            }],
            "final_win": float(row.get("lift_proxy") or 0.0),
        }
        out.append(seq)
    return out

def _ensure_object_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # Convert any “TBD” / stringy rows into object shape once
    if "asset" not in row or not isinstance(row["asset"], dict):
        row = {
            "asset": {"id": "tbd_asset", "name": row.get("asset") or "TBD Asset", "format": "email", "evergreen": False},
            "channel": {"id": "ch_email", "name": "Email", "type": "email", "reach_score": 0.6},
            "fitment": row.get("fitment") or "General",
            "engagement": row.get("engagement") or "Medium",
            "expectedLift": row.get("expectedLift") or "0.02",
        }
    if "channel" not in row or not isinstance(row["channel"], dict):
        row["channel"] = {"id": "ch_email", "name": "Email", "type": "email", "reach_score": 0.6}
    return row

def _quarter_success_metrics(campaigns: List[Dict[str, Any]]) -> Dict[str, Any]:
    personas_engaged = sum(len(c.get("personas") or []) for c in campaigns)  # no dedupe
    lifts = []
    for c in campaigns:
        for r in (c.get("arsenalTable") or []):
            pct = _parse_pct(r.get("expectedLift"))
            if pct is not None:
                lifts.append(pct)
    expected_belief_shift = round((_avg(lifts, 0.10) or 0.10) * 1.0, 2)  # default ~10%
    return {"personasEngaged": personas_engaged, "expectedBeliefShift": expected_belief_shift}

# ---- NEW: portfolio synthesis by sequences ----

def _portfolio_weighted_sequences(rcs_list: List[Dict[str, Any]], min_support: int = 1) -> Dict[str, Dict[str, Any]]:
    """
    Group sequences by (theme title + stage trajectory), weight by:
      score = avg(final_win) * (1 + log(1 + support))
    Return dict theme_key -> aggregator { 'seqs': [...], 'score': float, 'support': int }
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in rcs_list:
        seqs = _collect_sequences_from_rcs(r)
        if not seqs:
            # fall back to backlog if sequences absent
            seqs = _collect_backlog_as_singleton_sequences(r, limit=2)
        for s in seqs:
            title = _sequence_theme_label(s)
            stages = "→".join([ (x.get("stage") or "").lower() for x in (s.get("sequence") or []) ])
            theme_key = f"{title}||{stages}"
            bucket = buckets.setdefault(theme_key, {"seqs": [], "support": 0})
            bucket["seqs"].append(s)
            bucket["support"] += 1

    # compute scores
    for k, b in list(buckets.items()):
        if b["support"] < min_support:
            continue
        avg_fw = _avg([x.get("final_win") for x in b["seqs"]], 0.0)
        b["score"] = float(avg_fw * (1.0 + math.log1p(b["support"])))
        # carry a canonical representative sequence (max final_win)
        b["rep"] = sorted(b["seqs"], key=lambda z: z.get("final_win", 0.0), reverse=True)[0]

    return buckets

def _select_top_sequences_by_phase(buckets: Dict[str, Dict[str, Any]], top_per_phase: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each phase, pick top N sequences (by score) using rep sequences.
    """
    per_phase: Dict[str, List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
    for _, b in buckets.items():
        seq = b.get("rep")
        if not seq: continue
        phase = _phase_for_sequence(seq)
        per_phase[phase].append((b.get("score", 0.0), seq))

    out: Dict[str, List[Dict[str, Any]]] = {}
    for phase, arr in per_phase.items():
        arr.sort(key=lambda x: x[0], reverse=True)
        out[phase] = [seq for _, seq in arr[:top_per_phase]]
    return out

def _attach_concerns(rows: List[Dict[str, Any]], seq: Dict[str, Any]) -> None:
    labels = _concerns_from_sequence(seq)
    for r in rows:
        rr = _ensure_object_row(r)
        rr.setdefault("concernsAddressed", labels)

def _mk_quarters(start: datetime, n: int = 4, span_days: int = 84):
    q = []
    cur = start
    for i in range(n):
        end = cur + timedelta(days=span_days-1)
        q.append({
            "label": f"Q{i+1}",
            "timeframe": {"startDate": cur.strftime("%Y-%m-%d"), "endDate": end.strftime("%Y-%m-%d")},
            "campaigns": []
        })
        cur = end + timedelta(days=1)
    return q

# ---- PUBLIC: portfolio plan made from weighted sequences ----

def build_portfolio_plan_from_rcs_sequences(
    *,
    rcs_list: List[Dict[str, Any]],
    G,                       # product graph (for arsenal)
    product_id: str,
    account_id_for_prefix: Optional[str] = None,  # just for nice IDs
    plays_per_step: int = 1,
    top_per_phase: int = 3,
    quarter_span_days: int = 84
) -> Dict[str, Any]:
    """
    1) Aggregate sequences portfolio-wide and weight by cross-account support
    2) Pick top sequences per belief phase and assign them to Q1..Q4
    3) Call _campaigns_from_sequences_using_arsenal to turn sequences into asset×channel rows
    4) De-duplicate asset×channel pairs per quarter; attach concerns to each row
    """
    print(f"Building portfolio plan from {len(rcs_list)} RCS outputs...")
    # Portfolio stats up front
    total_accounts = len(rcs_list)
    total_personas = sum(len(((r.get("rcs_brief") or {}).get("top_personas") or {}).get("by_strength") or []) for r in rcs_list)  # no dedupe
    avg_baseline = _avg([((r.get("rcs_brief") or {}).get("baseline") or {}).get("win_likelihood", 0.0) for r in rcs_list], 0.0)
    print("KPI done")
    plan = {
        "meta": {
            "version": "3.0",
            "generatedAt": _to_iso_now(),
            "beliefScale": ["Unaware", "ZMOT", "Discovery", "Evaluation", "Pilot", "Pre-Close", "Customer"]
        },
        "portfolio": {
            "keyStats": {
                "totalTargetAccounts": total_accounts,
                "totalPersonasToEngage": total_personas,
                "timeToWinMonths": 12,
                "expectedWinsPct": round(min(max(avg_baseline, 0.05), 0.9), 2),
                "averageAccountBelief": "Unaware"  # coarse default
            }
        },
        "quarters": []
    }
    print("Plan meta done")

    # 1) weight sequences
    buckets = _portfolio_weighted_sequences(rcs_list, min_support=1)
    by_phase = _select_top_sequences_by_phase(buckets, top_per_phase=top_per_phase)

    # 2) Q1..Q4 from phases in this order
    quarters = _mk_quarters(datetime.now(timezone.utc), n=4, span_days=quarter_span_days)
    phase_to_q = {
        "pre_zmot": 0,
        "zmot": 1,
        "problem_realization": 2,
        "discovery": 3,   # if you need more phases, expand quarters or fold phases
    }
    print("Quarters done")
    # 3) for each phase→quarter, build campaigns via arsenal
    for phase, seqs in by_phase.items():
        qi = phase_to_q.get(phase, 0)
        if not seqs:
            continue
        # make a synthetic report for the arsenal function
        synthetic_report = {"concern_sequences": seqs}
        seen_pairs: Set[Tuple[str, str]] = set()  # avoid duplicate asset×channel inside this quarter

        # (a) generate portfolio campaigns for these unified sequences
        campaigns = _campaigns_from_sequences_using_arsenal(
            G=G,
            account_id=account_id_for_prefix or "portfolio",
            product_id=product_id,
            report=synthetic_report,
            start_idx=len(quarters[qi]["campaigns"]),
            plays_per_step=plays_per_step,
            archetype={},        # portfolio-wide, leave empty or inject a blended archetype
            exclude_pairs=seen_pairs  # <-- wire this in your function signature
        )

        # (b) attach personas (from steps) & concerns to each row
        for seq, camp in zip(seqs, campaigns):
            # personas: all step personas (no dedupe)
            personas = [s.get("persona") for s in (seq.get("sequence") or []) if s.get("persona")]
            camp["personas"] = personas
                

        # (c) push into the quarter
        quarters[qi]["campaigns"].extend(campaigns)
        print(f"  - Phase '{phase}' added {len(campaigns)} campaigns to Q{qi+1}")
    # 4) success metrics per quarter
    for q in quarters:
        q["successMetrics"] = _quarter_success_metrics(q.get("campaigns") or [])

    print("Success metrics done")
    plan["quarters"] = quarters
    print("Final plan done")
    return plan


