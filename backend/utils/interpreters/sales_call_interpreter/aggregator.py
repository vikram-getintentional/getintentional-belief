from collections import defaultdict
from typing import Dict, List

from sqlalchemy.orm import Session

from backend.super_models.call_interpretation import CallSegment, SegmentInterpretation


def role_weight(role: str | None) -> float:
    if not role:
        return 0.8
    normalized = role.lower()
    if "founder" in normalized or "ceo" in normalized:
        return 1.2
    if "cfo" in normalized or "finance" in normalized:
        return 1.1
    return 1.0


def _normalize_score(total: float, weight: float) -> float:
    if weight <= 0:
        return min(1.0, total)
    return min(1.0, total / weight)


def aggregate_call_interpretations(db: Session, call_id: str) -> Dict:
    segments = (
        db.query(CallSegment)
        .filter(CallSegment.call_id == call_id)
        .order_by(CallSegment.idx)
        .all()
    )
    segment_map = {segment.id: segment for segment in segments}
    segment_ids = list(segment_map.keys())
    if not segment_ids:
        return {}

    interpretations = (
        db.query(SegmentInterpretation)
        .filter(SegmentInterpretation.segment_id.in_(segment_ids))
        .all()
    )

    pain_scores: Dict[str, float] = defaultdict(float)
    pain_weights: Dict[str, float] = defaultdict(float)
    job_scores: Dict[str, float] = defaultdict(float)
    job_weights: Dict[str, float] = defaultdict(float)
    zmot_scores: Dict[tuple, float] = defaultdict(float)
    zmot_weights: Dict[tuple, float] = defaultdict(float)
    asp_scores: Dict[tuple, float] = defaultdict(float)
    asp_weights: Dict[tuple, float] = defaultdict(float)
    belief_scores: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    belief_weights: Dict[tuple, float] = defaultdict(float)
    emotion_scores: Dict[str, float] = defaultdict(float)
    emotion_weights: Dict[str, float] = defaultdict(float)

    for interp in interpretations:
        seg = segment_map.get(interp.segment_id)
        seg_weight = role_weight(seg.speaker_role if seg else None) * interp.confidence
        payload = interp.payload or {}
        for pain in payload.get("pains", []):
            pid = pain.get("pain_id")
            intensity = float(pain.get("intensity", 0.0))
            pain_scores[pid] += intensity * seg_weight
            pain_weights[pid] += seg_weight
        for job in payload.get("jobs", []):
            jid = job.get("job_id")
            intensity = float(job.get("intensity", 0.0))
            job_scores[jid] += intensity * seg_weight
            job_weights[jid] += seg_weight
        for zmot in payload.get("zmots", []):
            pid = zmot.get("pain_id")
            zid = zmot.get("zmot_id")
            intensity = float(zmot.get("intensity", 0.0))
            key = (pid, zid)
            zmot_scores[key] += intensity * seg_weight
            zmot_weights[key] += seg_weight
        for asp in payload.get("aspirations", []):
            pid = asp.get("pain_id")
            aid = asp.get("aspiration_id")
            intensity = float(asp.get("intensity", 0.0))
            key = (pid, aid)
            asp_scores[key] += intensity * seg_weight
            asp_weights[key] += seg_weight
        for belief in payload.get("beliefs", []):
            pid = belief.get("pain_id")
            aid = belief.get("aspiration_id")
            level = belief.get("belief_level")
            intensity = float(belief.get("intensity", 0.0))
            key = (pid, aid)
            belief_scores[key][level] += intensity * seg_weight
            belief_weights[key] += seg_weight
        for emotion in payload.get("emotions", []):
            label = emotion.get("label")
            intensity = float(emotion.get("intensity", 0.0))
            emotion_scores[label] += intensity * seg_weight
            emotion_weights[label] += seg_weight

    canonical_summary = {
        "pains": [
            {"pain_id": pid, "intensity": _normalize_score(pain_scores[pid], pain_weights.get(pid, 1.0))}
            for pid in pain_scores
        ],
        "jobs": [
            {"job_id": jid, "intensity": _normalize_score(job_scores[jid], job_weights.get(jid, 1.0))}
            for jid in job_scores
        ],
        "zmots": [
            {
                "pain_id": pid,
                "zmot_id": zid,
                "intensity": _normalize_score(zmot_scores[(pid, zid)], zmot_weights.get((pid, zid), 1.0)),
            }
            for pid, zid in zmot_scores
        ],
        "aspirations": [
            {
                "pain_id": pid,
                "aspiration_id": aid,
                "intensity": _normalize_score(asp_scores[(pid, aid)], asp_weights.get((pid, aid), 1.0)),
            }
            for pid, aid in asp_scores
        ],
        "beliefs": [
            {
                "pain_id": pid,
                "aspiration_id": aid,
                "belief_level": level,
                "intensity": _normalize_score(
                    belief_scores[(pid, aid)].get(level, 0.0),
                    belief_weights.get((pid, aid), 1.0),
                ),
            }
            for (pid, aid), levels in belief_scores.items()
            for level in levels
        ],
        "emotions": [
            {"label": label, "intensity": _normalize_score(emotion_scores[label], emotion_weights.get(label, 1.0))}
            for label in emotion_scores
        ],
    }
    canonical_summary["pains"] = [entry for entry in canonical_summary["pains"] if entry["intensity"] > 0]
    canonical_summary["jobs"] = [entry for entry in canonical_summary["jobs"] if entry["intensity"] > 0]
    canonical_summary["zmots"] = [entry for entry in canonical_summary["zmots"] if entry["intensity"] > 0]
    canonical_summary["aspirations"] = [
        entry for entry in canonical_summary["aspirations"] if entry["intensity"] > 0
    ]
    canonical_summary["beliefs"] = [entry for entry in canonical_summary["beliefs"] if entry["intensity"] > 0]
    canonical_summary["emotions"] = [entry for entry in canonical_summary["emotions"] if entry["intensity"] > 0]
    return canonical_summary
