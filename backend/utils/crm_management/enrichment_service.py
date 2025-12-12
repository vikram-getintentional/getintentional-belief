from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.crm_management.person_models import AccountPerson, AccountPersonJob, AccountPersonaMatch
from backend.utils.strategy_builder.comprehensive_plan_generator import (
    build_account_marketing_blueprint,
)
from backend.utils.graph_base.network_graph import build_product_graph, get_node_by_id, GRAPH_DATA_PATH
from backend.utils.graph_base.persona_learning import (
    match_candidate_to_canonical_persona,
    add_persona_node_from_candidate,
    alias_persona_with_label,
)
from backend.utils.persona_normalization import normalize_persona_label


STAGE_LABELS = ["Problem Realization", "Execution Guidance", "Pain Realization", "Resolution Discovery"]
_PERSONA_METRICS_DIR = os.path.join(GRAPH_DATA_PATH, "persona_metrics")


def _load_wolves_metrics(product_id: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(_PERSONA_METRICS_DIR, f"{product_id}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    personas = payload.get("personas") or []
    return {entry.get("persona_id"): entry for entry in personas if entry.get("persona_id")}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _persona_parts(graph, persona_id: str) -> Tuple[str, str, str, str]:
    """
    Returns (label, title, department, seniority) for a persona node.
    """
    node = get_node_by_id(graph, persona_id) or {}
    label = (
        node.get("label")
        or node.get("name")
        or persona_id.replace("|", " ").replace("_", " ").title()
    )

    node_type = (node.get("node_type") or node.get("type") or "").strip().lower()

    title = (node.get("title") or "").strip().lower()
    department = (node.get("department") or "").strip().lower()
    seniority = (node.get("seniority") or "").strip().lower()

    if node_type == "canonical_persona":
        title = (node.get("label") or title or label).strip().lower()
        departments = node.get("typical_departments") or []
        if departments:
            department = (departments[0] or "").strip().lower()
        dist = node.get("typical_seniority_distribution") or {}
        if dist:
            seniority = max(dist.items(), key=lambda kv: kv[1])[0]
    elif node_type == "persona_variant":
        title = (node.get("title") or node.get("label") or label).strip().lower() or title
        department = (node.get("department") or department).strip().lower()
        seniority = (node.get("seniority") or seniority).strip().lower()

    if (not title or not department or not seniority) and "|" in persona_id:
        parts = [p.strip().lower() for p in persona_id.split("|")]
        if len(parts) >= 3:
            title = title or parts[0]
            department = department or parts[1]
            seniority = seniority or parts[2]

    if not seniority:
        seniority = "operator"

    return label, title, department, seniority


def _resolve_canonical_persona_id(graph, persona_id: str) -> str:
    """
    Map a persona node (variant or canonical) back to its canonical persona ID.
    """
    if not persona_id:
        return persona_id
    node = get_node_by_id(graph, persona_id) or {}
    canonical_id = (node.get("canonical_persona_id") or "").strip()
    if canonical_id:
        return canonical_id
    node_type = (node.get("node_type") or node.get("type") or "").strip().lower()
    if node_type == "canonical_persona":
        return persona_id
    return persona_id


def _split_candidate_label(label: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = [segment.strip() for segment in label.split("·")]
    title = parts[0] if len(parts) > 0 else None
    department = parts[1] if len(parts) > 1 else None
    seniority = parts[2] if len(parts) > 2 else None
    return title, department, seniority


def _prettify_candidate_label(label: str) -> str:
    return " · ".join(segment.strip().title() for segment in label.split("·"))


def _most_common(values: Iterable[str]) -> Optional[str]:
    counter = Counter([value for value in values if value])
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _average(values: Iterable[Optional[float]]) -> Optional[float]:
    floated = [float(v) for v in values if isinstance(v, (int, float))]
    if not floated:
        return None
    return sum(floated) / len(floated)


def _stage_from_order(order_avg: float) -> Tuple[int, str]:
    if math.isnan(order_avg):
        return 0, STAGE_LABELS[0]
    idx = int(round(order_avg))
    idx = max(0, min(idx, len(STAGE_LABELS) - 1))
    return idx, STAGE_LABELS[idx]


def _load_account_people(db: Session, product_id: str, account_id: str) -> List[AccountPerson]:
    return (
        db.query(AccountPerson)
        .filter(
            AccountPerson.product_id == product_id,
            AccountPerson.account_id == account_id,
        )
        .all()
    )


def _load_person_jobs(
    db: Session, product_id: str, account_id: str
) -> Dict[str, List[AccountPersonJob]]:
    jobs = (
        db.query(AccountPersonJob)
        .filter(
            AccountPersonJob.product_id == product_id,
            AccountPersonJob.account_id == account_id,
        )
        .all()
    )
    by_person: Dict[str, List[AccountPersonJob]] = defaultdict(list)
    for job in jobs:
        by_person[job.person_id].append(job)
    return by_person


def _compute_persona_requirements(
    product_id: str,
    account_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Any, Any]:
    blueprint = build_account_marketing_blueprint(product_id, account_id)
    persona_paths = list(blueprint.get("persona_paths") or [])
    if not persona_paths:
        prediction = blueprint.get("prediction") or {}
        persona_paths = list(prediction.get("persona_paths") or prediction.get("paths") or [])
    graph = build_product_graph(product_id)

    stats: Dict[str, Dict[str, Any]] = {}
    for path in persona_paths:
        path_prob = float(path.get("probability") or 0.0)
        personas = path.get("personas") or []
        if not personas and isinstance(path.get("path"), list):
            personas = [{"id": pid, "order": idx} for idx, pid in enumerate(path["path"])]
        for persona in personas:
            persona_id = persona.get("id")
            if not persona_id:
                continue
            persona_id = str(persona_id)
            persona_id = _resolve_canonical_persona_id(graph, persona_id)
            bucket = stats.setdefault(
                persona_id,
                {
                    "count": 0,
                    "order_sum": 0.0,
                    "prob_sum": 0.0,
                    "perceptibility": [],
                    "proximity": [],
                    "involvement": [],
                    "label": None,
                    "title": None,
                    "department": None,
                    "seniority": None,
                },
            )
            bucket["count"] += 1
            bucket["order_sum"] += float(persona.get("order", 0))
            bucket["prob_sum"] += path_prob
            for metric in ("perceptibility", "proximity", "involvement"):
                value = persona.get(metric)
                if isinstance(value, (int, float)):
                    bucket[metric].append(float(value))
            if not bucket["label"]:
                label, title, department, seniority = _persona_parts(graph, persona_id)
                bucket["label"] = label
                bucket["title"] = title
                bucket["department"] = department
                bucket["seniority"] = seniority

    requirements: List[Dict[str, Any]] = []
    for persona_id, bucket in stats.items():
        if not bucket["count"]:
            continue
        avg_order = bucket["order_sum"] / bucket["count"]
        stage_index, stage_label = _stage_from_order(avg_order)
        expected = min(1.0, bucket["prob_sum"])
        requirements.append(
            {
                "persona_id": persona_id,
                "persona_label": bucket["label"],
                "persona_title": bucket["title"],
                "persona_department": bucket["department"],
                "persona_seniority": bucket["seniority"],
                "expected_in_deal": expected,
                "expected_in_deal_pct": round(expected * 100, 1),
                "stage_index": stage_index,
                "stage_label": stage_label,
                "metrics": {
                    "perceptibility": _average(bucket["perceptibility"]),
                    "proximity": _average(bucket["proximity"]),
                    "involvement": _average(bucket["involvement"]),
                },
            }
        )

    if not requirements:
        fallback: Dict[str, Dict[str, Any]] = {}
        for transition in blueprint.get("transitions") or []:
            persona = (transition or {}).get("persona") or {}
            persona_id = persona.get("id")
            if not persona_id:
                continue
            bucket = fallback.setdefault(
                persona_id,
                {
                    "count": 0,
                    "prob_sum": 0.0,
                    "stage_sum": 0.0,
                    "label": persona.get("label"),
                    "title": persona.get("title"),
                    "department": persona.get("department"),
                    "seniority": persona.get("seniority"),
                },
            )
            bucket["count"] += 1
            bucket["prob_sum"] += float(
                transition.get("path_probability")
                or persona.get("path_probability")
                or 0.1
            )
            bucket["stage_sum"] += float(transition.get("stage_index", 0))

        for persona_id, bucket in fallback.items():
            persona_id = str(persona_id)
            persona_id = _resolve_canonical_persona_id(graph, persona_id)
            label, title, department, seniority = _persona_parts(graph, persona_id)
            count = max(bucket["count"], 1)
            stage_index = int(round(bucket["stage_sum"] / count))
            stage_index = max(0, min(stage_index, len(STAGE_LABELS) - 1))
            if not bucket.get("label"):
                bucket["label"] = label
            if not bucket.get("title"):
                bucket["title"] = title
            if not bucket.get("department"):
                bucket["department"] = department
            if not bucket.get("seniority"):
                bucket["seniority"] = seniority
            expected = min(1.0, bucket["prob_sum"])
            requirements.append(
                {
                    "persona_id": persona_id,
                    "persona_label": bucket.get("label"),
                    "persona_title": bucket.get("title"),
                    "persona_department": bucket.get("department"),
                    "persona_seniority": bucket.get("seniority"),
                    "expected_in_deal": expected,
                    "expected_in_deal_pct": round(expected * 100, 1),
                    "stage_index": stage_index,
                    "stage_label": STAGE_LABELS[stage_index],
                    "metrics": {},
                }
            )

    requirements.sort(key=lambda r: r["expected_in_deal"], reverse=True)
    return requirements, stats, blueprint, graph


def serialize_match(
    match: AccountPersonaMatch,
    person: Optional[AccountPerson] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "match_id": match.id,
        "product_id": match.product_id,
        "account_id": match.account_id,
        "persona_id": match.persona_id,
        "persona_label": match.persona_label,
        "stage": match.stage,
        "person_id": match.person_id,
        "match_confidence": match.match_confidence,
        "source": match.source,
        "notes": match.notes,
        "created_at": match.created_at.isoformat() if match.created_at else None,
        "updated_at": match.updated_at.isoformat() if match.updated_at else None,
    }
    if person:
        payload.update(
            {
                "person_name": person.name,
                "person_title": person.title,
                "person_department": person.department,
                "person_seniority": person.seniority,
                "engagement_count": person.engagement_count,
                "last_seen_at": person.last_seen_at.isoformat()
                if person.last_seen_at
                else None,
            }
        )
    return payload


def _existing_matches(db: Session, product_id: str, account_id: str) -> Dict[str, List[Dict[str, Any]]]:
    matches = (
        db.query(AccountPersonaMatch, AccountPerson)
        .outerjoin(AccountPerson, AccountPersonaMatch.person_id == AccountPerson.id)
        .filter(
            AccountPersonaMatch.product_id == product_id,
            AccountPersonaMatch.account_id == account_id,
        )
        .all()
    )
    by_persona: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for match, person in matches:
        by_persona[match.persona_id].append(serialize_match(match, person))
    return by_persona


def _merge_matches_by_canonical(
    graph,
    matches: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    if not graph or not matches:
        return matches
    normalized: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for persona_id, entries in matches.items():
        normalized[persona_id].extend(entries)
        canonical_id = _resolve_canonical_persona_id(graph, persona_id)
        if canonical_id and canonical_id != persona_id:
            normalized[canonical_id].extend(entries)
    return normalized


def _person_signature(person: AccountPerson) -> Tuple[str, str, str]:
    """
    Returns a normalized (title, department, seniority) signature for scoring.
    Prefers canonical fields, but gracefully falls back to historical formats.
    """
    if person.canonical_persona_id and "|" in person.canonical_persona_id:
        parts = [p.strip().lower() for p in person.canonical_persona_id.split("|")]
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]

    title = (person.title or "").strip().lower()
    dept = (
        person.canonical_department
        or person.department
        or ""
    ).strip().lower()
    seniority = (
        person.canonical_seniority
        or person.seniority
        or ""
    ).strip().lower() or "operator"
    return title, dept, seniority


def _score_person_candidate(
    persona_meta: Dict[str, Any],
    person: AccountPerson,
    jobs: Sequence[AccountPersonJob],
) -> Tuple[float, List[str]]:
    persona_title = persona_meta.get("persona_title")
    persona_dept = persona_meta.get("persona_department")
    persona_seniority = persona_meta.get("persona_seniority")

    person_title, person_dept, person_seniority = _person_signature(person)

    score = 0.0
    reasons: List[str] = []

    if persona_title and persona_title == person_title:
        score += 0.5
        reasons.append("Exact title match")

    if persona_dept and persona_dept == person_dept:
        score += 0.25
        reasons.append("Department match")

    if persona_seniority and persona_seniority == person_seniority:
        score += 0.15
        reasons.append("Seniority match")

    if score >= 0.75:
        score += 0.1  # high confidence when core attributes align

    if person.engagement_count and person.engagement_count > 0:
        score += 0.1
        reasons.append("Has engagement history")

    # Lightweight job-text heuristic
    persona_label = (persona_meta.get("persona_label") or "").lower()
    if jobs:
        for job in jobs:
            job_text = (job.job_text or "").lower()
            if not job_text:
                continue
            if persona_label and persona_label in job_text:
                score += 0.1
                reasons.append("Job text references persona label")
                break
            if persona_title and persona_title in job_text:
                score += 0.08
                reasons.append("Job text references title")
                break

    return min(score, 1.0), reasons


def _candidate_suggestions(
    persona: Dict[str, Any],
    people: List[AccountPerson],
    matches: List[Dict[str, Any]],
    jobs_by_person: Dict[str, List[AccountPersonJob]],
) -> List[Dict[str, Any]]:
    matched_person_ids = {match["person_id"] for match in matches if match.get("person_id")}
    candidates: List[Dict[str, Any]] = []
    for person in people:
        if person.id in matched_person_ids:
            continue
        score, reasons = _score_person_candidate(
            persona,
            person,
            jobs_by_person.get(person.id, []),
        )
        if score < 0.1:
            continue
        candidates.append(
            {
                "person_id": person.id,
                "person_name": person.name,
                "person_title": person.title,
                "person_department": person.department,
                "person_seniority": person.seniority,
                "confidence": round(score, 3),
                "reasons": reasons,
                "engagement_count": person.engagement_count,
                "last_seen_at": person.last_seen_at.isoformat()
                if person.last_seen_at
                else None,
            }
        )
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates[:5]


def _collect_account_candidate_personas(
    session: Session,
    product_id: str,
    account_id: str,
    *,
    window_days: int = 365,
) -> List[Dict[str, Any]]:
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    query = (
        session.query(
            TargetAccountEngagement.candidate_persona_label,
            TargetAccountEngagement.actor_title,
            TargetAccountEngagement.actor_department,
            TargetAccountEngagement.actor_seniority,
            TargetAccountEngagement.timestamp_dt,
            TargetAccountEngagement.persona_id,
        )
        .filter(
            TargetAccountEngagement.product_id == product_id,
            TargetAccountEngagement.target_account_id == account_id,
            TargetAccountEngagement.candidate_persona_label.isnot(None),
        )
    )
    if window_days > 0:
        query = query.filter(TargetAccountEngagement.timestamp_dt >= cutoff)
    rows = query.all()

    aggregates: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        normalized = (row.candidate_persona_label or "").strip()
        if not normalized:
            continue
        bucket = aggregates.setdefault(
            normalized,
            {
                "titles": [],
                "departments": [],
                "seniority": [],
                "occurrences": 0,
                "first_seen": None,
                "last_seen": None,
            },
        )
        bucket["titles"].append((row.actor_title or "").strip())
        bucket["departments"].append((row.actor_department or "").strip())
        bucket["seniority"].append((row.actor_seniority or "").strip())
        bucket["occurrences"] += 1
        ts = row.timestamp_dt
        if ts:
            if not bucket["first_seen"] or ts < bucket["first_seen"]:
                bucket["first_seen"] = ts
            if not bucket["last_seen"] or ts > bucket["last_seen"]:
                bucket["last_seen"] = ts

    if not aggregates:
        return []

    labels = list(aggregates.keys())
    global_rows = (
        session.query(
            TargetAccountEngagement.candidate_persona_label.label("label"),
            func.count(func.distinct(TargetAccountEngagement.target_account_id)).label("accounts"),
            func.count(TargetAccountEngagement.id).label("occurrences"),
        )
        .filter(
            TargetAccountEngagement.product_id == product_id,
            TargetAccountEngagement.candidate_persona_label.in_(labels),
        )
        .group_by(TargetAccountEngagement.candidate_persona_label)
        .all()
    )
    global_map = {row.label: {"accounts": row.accounts, "occurrences": row.occurrences} for row in global_rows}

    graph = build_product_graph(product_id)
    persona_nodes = [
        (node_id, data)
        for node_id, data in graph.nodes(data=True)
        if data.get("node_type") == "persona"
    ]
    wolves_metrics = _load_wolves_metrics(product_id)

    candidates: List[Dict[str, Any]] = []
    for normalized, meta in aggregates.items():
        pretty_label = _prettify_candidate_label(normalized)
        title_guess, dept_guess, seniority_guess = _split_candidate_label(pretty_label)
        match_id, similarity = match_candidate_to_canonical_persona(
            pretty_label,
            persona_nodes,
            threshold=0.8,
        )
        wolves_info = wolves_metrics.get(match_id) if match_id else None
        match_node = get_node_by_id(graph, match_id) if match_id else None
        candidates.append(
            {
                "label": pretty_label,
                "normalized_label": normalized,
                "persona_title": _most_common(meta["titles"]) or title_guess,
                "persona_department": _most_common(meta["departments"]) or dept_guess,
                "persona_seniority": _most_common(meta["seniority"]) or seniority_guess,
                "account_occurrences": meta["occurrences"],
                "account_first_seen": meta["first_seen"].isoformat() if meta["first_seen"] else None,
                "account_last_seen": meta["last_seen"].isoformat() if meta["last_seen"] else None,
                "global_account_count": (global_map.get(normalized) or {}).get("accounts", 0),
                "global_occurrences": (global_map.get(normalized) or {}).get("occurrences", meta["occurrences"]),
                "suggested_persona_id": match_id,
                "suggested_persona_label": (match_node or {}).get("label"),
                "similarity": similarity if match_id else None,
                "wolves_score": (wolves_info or {}).get("wolves_score"),
                "wolves_delta_bp": (wolves_info or {}).get("delta_win_bp"),
            }
        )

    candidates.sort(key=lambda row: row["account_occurrences"], reverse=True)
    return candidates


def add_candidate_persona(
    product_id: str,
    label: str,
    *,
    title: Optional[str] = None,
    department: Optional[str] = None,
    seniority: Optional[str] = None,
    co_occurring_persona_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    title_guess, dept_guess, seniority_guess = _split_candidate_label(label)
    return add_persona_node_from_candidate(
        product_id,
        label=label,
        title=title or title_guess,
        department=department or dept_guess,
        seniority=seniority or seniority_guess,
        co_occurring_persona_ids=co_occurring_persona_ids,
    )


def alias_candidate_persona(
    product_id: str,
    persona_id: str,
    label: str,
) -> Dict[str, Any]:
    return alias_persona_with_label(product_id, persona_id, label)


def build_account_enrichment(
    product_id: str,
    account_id: str,
    *,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Returns enrichment metadata for an account:
      - persona requirements (probability of appearing in deal)
      - matched people
      - suggested people to match
      - coverage summary
    """
    external_session = db is not None
    session = db or next(get_db())
    try:
        requirements, stats_map, blueprint, persona_graph = _compute_persona_requirements(
            product_id, account_id
        )
        people = _load_account_people(session, product_id, account_id)
        jobs_by_person = _load_person_jobs(session, product_id, account_id)
        matches_by_persona = _existing_matches(session, product_id, account_id)
        matches_by_persona = _merge_matches_by_canonical(persona_graph, matches_by_persona)

        rows: List[Dict[str, Any]] = []
        for persona in requirements:
            persona_id = persona["persona_id"]
            matched_people = matches_by_persona.get(persona_id, [])
            suggestions = _candidate_suggestions(
                persona,
                people,
                matched_people,
                jobs_by_person,
            )
            rows.append(
                {
                    **persona,
                    "matched_people": matched_people,
                    "suggested_people": suggestions,
                }
            )

        required = len(rows)
        matched = sum(1 for row in rows if row["matched_people"])
        coverage = (matched / required) if required else 0.0
        persona_candidates = _collect_account_candidate_personas(session, product_id, account_id)

        return {
            "account_id": account_id,
            "product_id": product_id,
            "persona_requirements": rows,
            "persona_candidates": persona_candidates,
            "summary": {
                "required_personas": required,
                "personas_with_matches": matched,
                "matched_people": sum(len(row["matched_people"]) for row in rows),
                "total_people": len(people),
                "coverage_pct": round(coverage * 100, 1),
                "coverage_ratio": coverage,
            },
            "blueprint_generated_at": blueprint.get("generated_at") if blueprint else None,
        }
    finally:
        if not external_session:
            session.close()


def upsert_persona_match(
    db: Session,
    *,
    product_id: str,
    account_id: str,
    persona_id: str,
    person_id: Optional[str],
    persona_label: Optional[str] = None,
    stage: Optional[str] = None,
    match_confidence: Optional[float] = None,
    source: Optional[str] = "user",
    notes: Optional[str] = None,
    match_id: Optional[str] = None,
    person_name: Optional[str] = None,
    person_title: Optional[str] = None,
    person_department: Optional[str] = None,
    person_seniority: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create or update a persona → person match.
    """
# NOTE: keep canonical matches consistent
    graph = None
    persona_label_from_graph = None
    persona_title = None
    persona_department = None
    persona_seniority = None
    resolved_persona_id = persona_id
    try:
        graph = build_product_graph(product_id)
        resolved_persona_id = _resolve_canonical_persona_id(graph, persona_id)
        persona_label_from_graph, persona_title, persona_department, persona_seniority = _persona_parts(
            graph, resolved_persona_id
        )
    except Exception:
        graph = None

    stored_persona_id = resolved_persona_id or persona_id

    if match_id:
        match = (
            db.query(AccountPersonaMatch)
            .filter(
                AccountPersonaMatch.id == match_id,
                AccountPersonaMatch.product_id == product_id,
                AccountPersonaMatch.account_id == account_id,
            )
            .first()
        )
        if not match:
            raise ValueError(f"Persona match '{match_id}' not found.")
        match.persona_id = stored_persona_id
    else:
        match = AccountPersonaMatch(
            product_id=product_id,
            account_id=account_id,
            persona_id=stored_persona_id,
        )

    created_person = None
    if not person_id and person_name:
        existing = (
            db.query(AccountPerson)
            .filter(
                AccountPerson.product_id == product_id,
                AccountPerson.account_id == account_id,
                AccountPerson.name == person_name.strip(),
            )
            .first()
        )
        if existing:
            person = existing
        else:
            person = AccountPerson(
                product_id=product_id,
                account_id=account_id,
                name=person_name.strip(),
                title=(person_title or "").strip() or None,
                department=(person_department or "").strip() or None,
                seniority=(person_seniority or "").strip() or None,
                canonical_persona_id=stored_persona_id if stored_persona_id else None,
                canonical_department=persona_department or None,
                canonical_seniority=persona_seniority or None,
            )
            db.add(person)
            db.flush()
            created_person = person
        match.person_id = person.id
    else:
        match.person_id = person_id

    if match.person_id:
        person_obj = db.get(AccountPerson, match.person_id)
        if person_obj and stored_persona_id:
            person_obj.canonical_persona_id = stored_persona_id
            if persona_department:
                person_obj.canonical_department = persona_department
            if persona_seniority:
                person_obj.canonical_seniority = persona_seniority

    if persona_label:
        match.persona_label = persona_label
    elif not match.persona_label and persona_label_from_graph:
        match.persona_label = persona_label_from_graph
    if stage:
        match.stage = stage
    match.match_confidence = match_confidence
    if source:
        match.source = source
    match.notes = notes
    now = _utcnow()
    if not match.created_at:
        match.created_at = now
    match.updated_at = now

    db.add(match)
    db.flush()

    person = (
        created_person
        if created_person is not None
        else db.get(AccountPerson, match.person_id)
        if match.person_id
        else None
    )
    return serialize_match(match, person)


def delete_persona_match(
    db: Session,
    *,
    product_id: str,
    account_id: str,
    match_id: str,
) -> None:
    match = (
        db.query(AccountPersonaMatch)
        .filter(
            AccountPersonaMatch.id == match_id,
            AccountPersonaMatch.product_id == product_id,
            AccountPersonaMatch.account_id == account_id,
        )
        .first()
    )
    if not match:
        raise ValueError(f"Persona match '{match_id}' not found.")
    db.delete(match)
    db.flush()
