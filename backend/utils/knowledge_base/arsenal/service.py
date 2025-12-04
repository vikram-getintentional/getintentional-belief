from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import func, inspect, text
from sqlalchemy.exc import IntegrityError
import uuid
from sqlalchemy.orm import Session, joinedload

from backend.database import engine
from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.crm_management.target_account_manager import (
    TargetAccount as TargetAccountORM,
)
from backend.utils.inference.rcs_generators.generate_rcs_fast import generate_rcs_new
from backend.utils.graph_base.network_graph import build_product_graph
from backend.utils.graph_base.network_graph import build_product_graph
from backend.utils.knowledge_base.arsenal.db_models import (
    ApprovalStatus,
    ArsenalAsset,
    ArsenalChannel,
    AssetChannelImpact,
    AssetCategory,
    AssetDepth,
    ChannelDelivery,
    ChannelType,
    ContentType,
    TimeToConsume,
)


def _ensure_schema() -> None:
    try:
        insp = inspect(engine)
    except Exception:
        return

    def _ensure_columns(table: str, required: Dict[str, str]) -> None:
        try:
            existing = {col["name"] for col in insp.get_columns(table)}
        except Exception:
            return
        missing = {name: ddl for name, ddl in required.items() if name not in existing}
        if not missing:
            return
        with engine.begin() as conn:
            for ddl in missing.values():
                try:
                    conn.execute(text(ddl))
                except Exception:
                    # best-effort; ignore if column already added in race
                    pass

    _ensure_columns(
        "arsenal_assets",
        {
            "call_stage": "ALTER TABLE arsenal_assets ADD COLUMN call_stage VARCHAR",
            "target_personas": "ALTER TABLE arsenal_assets ADD COLUMN target_personas JSON",
            "target_account_segments": "ALTER TABLE arsenal_assets ADD COLUMN target_account_segments JSON",
            "target_belief_stages": "ALTER TABLE arsenal_assets ADD COLUMN target_belief_stages JSON",
            "target_concerns": "ALTER TABLE arsenal_assets ADD COLUMN target_concerns JSON",
            "org_conversion_maturity": "ALTER TABLE arsenal_assets ADD COLUMN org_conversion_maturity VARCHAR",
            "category_text": "ALTER TABLE arsenal_assets ADD COLUMN category_text VARCHAR",
            "content_type_text": "ALTER TABLE arsenal_assets ADD COLUMN content_type_text VARCHAR",
            "time_to_consume_text": "ALTER TABLE arsenal_assets ADD COLUMN time_to_consume_text VARCHAR",
            "depth_text": "ALTER TABLE arsenal_assets ADD COLUMN depth_text VARCHAR",
            "typical_channels": "ALTER TABLE arsenal_assets ADD COLUMN typical_channels JSON",
            "approval_status": "ALTER TABLE arsenal_assets ADD COLUMN approval_status VARCHAR DEFAULT 'PENDING'",
            "derived_metadata": "ALTER TABLE arsenal_assets ADD COLUMN derived_metadata JSON",
            "auto_classification_confidence": "ALTER TABLE arsenal_assets ADD COLUMN auto_classification_confidence FLOAT",
        },
    )
    _ensure_columns(
        "arsenal_channels",
        {
            "target_personas": "ALTER TABLE arsenal_channels ADD COLUMN target_personas JSON",
            "target_account_segments": "ALTER TABLE arsenal_channels ADD COLUMN target_account_segments JSON",
            "target_belief_stages": "ALTER TABLE arsenal_channels ADD COLUMN target_belief_stages JSON",
            "target_concerns": "ALTER TABLE arsenal_channels ADD COLUMN target_concerns JSON",
            "org_conversion_maturity": "ALTER TABLE arsenal_channels ADD COLUMN org_conversion_maturity VARCHAR",
            "channel_type_text": "ALTER TABLE arsenal_channels ADD COLUMN channel_type_text VARCHAR",
            "delivery_mode_text": "ALTER TABLE arsenal_channels ADD COLUMN delivery_mode_text VARCHAR",
            "typical_assets": "ALTER TABLE arsenal_channels ADD COLUMN typical_assets JSON",
            "approval_status": "ALTER TABLE arsenal_channels ADD COLUMN approval_status VARCHAR DEFAULT 'PENDING'",
            "derived_metadata": "ALTER TABLE arsenal_channels ADD COLUMN derived_metadata JSON",
            "auto_classification_confidence": "ALTER TABLE arsenal_channels ADD COLUMN auto_classification_confidence FLOAT",
        },
    )

    # normalize approval status casing
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE arsenal_assets SET approval_status='PENDING' "
                    "WHERE approval_status IS NULL OR LOWER(approval_status)='pending'"
                )
            )
            conn.execute(
                text(
                    "UPDATE arsenal_channels SET approval_status='PENDING' "
                    "WHERE approval_status IS NULL OR LOWER(approval_status)='pending'"
                )
            )
    except Exception:
        pass


_ensure_schema()


def _ensure_engagement_columns() -> None:
    ddl = {
        "channel_id": "ALTER TABLE target_account_engagements ADD COLUMN channel_id VARCHAR",
        "engagement_verb": "ALTER TABLE target_account_engagements ADD COLUMN engagement_verb VARCHAR",
    }
    try:
        with engine.connect() as conn:
            existing = {
                row["name"]
                for row in conn.execute(text("PRAGMA table_info(target_account_engagements);"))
            }
        missing = {col: stmt for col, stmt in ddl.items() if col not in existing}
        if not missing:
            return
        with engine.begin() as conn:
            for statement in missing.values():
                try:
                    conn.execute(text(statement))
                except Exception:
                    pass
    except Exception:
        return


_ensure_engagement_columns()


ASSET_MUTABLE_FIELDS = {
    "name",
    "category",
    "content_type",
    "time_to_consume",
    "depth",
    "description",
    "notes",
    "call_stage",
    "target_personas",
    "target_account_segments",
    "target_belief_stages",
    "target_concerns",
    "org_conversion_maturity",
    "typical_channels",
}

CHANNEL_MUTABLE_FIELDS = {
    "name",
    "channel_type",
    "delivery_mode",
    "reach_score_estimate",
    "notes",
    "target_personas",
    "target_account_segments",
    "target_belief_stages",
    "target_concerns",
    "org_conversion_maturity",
    "typical_assets",
}


def _json_from_enum(value) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _title_case(value: str) -> str:
    pieces = []
    for raw in value.replace("|", " ").replace("_", " ").split():
        pieces.append(raw[:1].upper() + raw[1:].lower())
    return " ".join(pieces) if pieces else value.strip()


def _graph_label(graph, node_id: Optional[str]) -> Optional[str]:
    if not graph or not node_id:
        return None
    if node_id in graph:
        data = graph.nodes[node_id]
        for key in ("label", "name", "title"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _persona_label(persona_id: Optional[str], graph=None) -> Optional[str]:
    if not persona_id:
        return None
    label_from_graph = _graph_label(graph, persona_id)
    if label_from_graph:
        return label_from_graph
    parts = [p.strip() for p in persona_id.split("|") if p and p.strip()]
    if not parts:
        return persona_id
    return " | ".join(_title_case(part) for part in parts)


def _clean_metadata_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    try:
        cleaned = str(value).strip()
        return cleaned or None
    except Exception:
        return None


def _metadata_value(enum_value: Optional[Any], text_value: Optional[str]) -> Optional[str]:
    return text_value or _json_from_enum(enum_value)


def _serialize_reference_list(refs: Optional[Iterable[Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for ref in refs or []:
        if isinstance(ref, dict):
            rid = _clean_metadata_text(ref.get("id"))
            label = _clean_metadata_text(ref.get("label")) or rid
        else:
            rid = _clean_metadata_text(ref)
            label = rid
        if not rid or rid in seen:
            continue
        seen.add(rid)
        out.append({"id": rid, "label": label or rid})
    return out


def _top_reference_list(
    score_map: Dict[str, float],
    label_map: Dict[str, str],
    limit: int,
) -> List[Dict[str, str]]:
    ordered = sorted(
        score_map.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )
    result: List[Dict[str, str]] = []
    for node_id, _ in ordered[:limit]:
        result.append({"id": node_id, "label": label_map.get(node_id, node_id)})
    return result

BELIEF_STAGE_LABELS: Dict[str, str] = {
    "problem": "Problem Realization",
    "problem_realization": "Problem Realization",
    "problem discovery": "Problem Realization",
    "pain": "Pain Realization",
    "pain_realization": "Pain Realization",
    "resolution": "Resolution Discovery",
    "resolution_discovery": "Resolution Discovery",
    "execution": "Execution Guidance",
    "execution_guidance": "Execution Guidance",
}

def _normalize_stage(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = value.strip().lower().replace(" ", "_")
    if key in BELIEF_STAGE_LABELS:
        return key
    # allow already labelled names to map back
    for canonical, label in BELIEF_STAGE_LABELS.items():
        if value.strip().lower() == label.lower():
            return canonical
    return key

def _stage_label(canonical: str) -> str:
    return BELIEF_STAGE_LABELS.get(canonical, _title_case(canonical))


_FUNNEL_STAGE_LABELS: Dict[str, str] = {
    "early": "Early Cycle",
    "mid": "Mid Cycle",
    "late": "Late Cycle",
}


def _funnel_stage_from_iterable(values: Optional[Iterable[Any]]) -> Optional[Dict[str, Any]]:
    if not values:
        return None

    def _stage_from_text(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        cleaned = str(text).strip().lower()
        if not cleaned:
            return None
        cleaned = cleaned.replace("-", " ").replace("_", " ").replace(":", " ")
        parts = [p for p in cleaned.split() if p]
        if not parts:
            return None
        if parts[0] == "funnel":
            for token in parts[1:]:
                if token in _FUNNEL_STAGE_LABELS:
                    return token
        for token in parts:
            if token in _FUNNEL_STAGE_LABELS:
                return token
        return None

    iterable: Iterable[Any]
    if isinstance(values, (list, tuple, set)):
        iterable = values
    else:
        iterable = [values]

    for entry in iterable:
        candidate_stage: Optional[str] = None
        if isinstance(entry, dict):
            code = entry.get("code")
            if isinstance(code, str):
                candidate_stage = _stage_from_text(code)
            if not candidate_stage:
                label = entry.get("label") or entry.get("value") or entry.get("name")
                if isinstance(label, str):
                    candidate_stage = _stage_from_text(label)
        elif isinstance(entry, str):
            candidate_stage = _stage_from_text(entry)
        else:
            candidate_stage = _stage_from_text(str(entry))

        if candidate_stage and candidate_stage in _FUNNEL_STAGE_LABELS:
            return {
                "code": candidate_stage,
                "label": _FUNNEL_STAGE_LABELS[candidate_stage],
            }
    return None


def _normalize_funnel_code(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("code", "value", "label", "stage"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                value = candidate
                break
        else:
            value = str(value)
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    cleaned = cleaned.replace("-", " ").replace("_", " ")
    tokens = [tok for tok in cleaned.split() if tok]
    for token in tokens:
        if token in _FUNNEL_STAGE_LABELS:
            return token
    return None


def _split_belief_and_maturity(
    raw_values: Optional[Iterable[Any]],
) -> Tuple[List[str], Optional[str]]:
    if raw_values is None:
        return ([], None)

    if isinstance(raw_values, str):
        iterable: Iterable[Any] = [raw_values]
    else:
        iterable = raw_values

    belief_labels: List[str] = []
    detected_code: Optional[str] = None
    seen: set[str] = set()

    for entry in iterable:
        if entry is None:
            continue
        if isinstance(entry, dict):
            candidate = None
            for key in ("canonical", "code", "label", "value", "stage"):
                raw = entry.get(key)
                if isinstance(raw, str) and raw.strip():
                    candidate = raw
                    break
            entry_value = candidate if candidate is not None else str(entry)
        else:
            entry_value = str(entry)

        funnel_code = _normalize_funnel_code(entry_value)
        if funnel_code and detected_code is None:
            detected_code = funnel_code
            continue

        canonical = _normalize_stage(entry_value)
        if canonical and canonical in BELIEF_STAGE_LABELS:
            label = _stage_label(canonical)
        elif canonical:
            label = _title_case(canonical)
        else:
            label = _title_case(entry_value)
        if label and label not in seen:
            seen.add(label)
            belief_labels.append(label)

    return (belief_labels, detected_code)


def _normalize_reference_list(
    db: Session,
    model,
    values: Optional[Iterable[Any]],
) -> List[Dict[str, str]]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        iterable: Iterable[Any] = [values]
    else:
        iterable = values

    ordered_ids: List[str] = []
    seen_ids: set[str] = set()
    for raw in iterable:
        if isinstance(raw, dict):
            candidate = _clean_metadata_text(raw.get("id") or raw.get("value"))
        else:
            candidate = _clean_metadata_text(raw)
        if not candidate or candidate in seen_ids:
            continue
        seen_ids.add(candidate)
        ordered_ids.append(candidate)

    if not ordered_ids:
        return []

    name_col = getattr(model, "name", None)
    slug_col = getattr(model, "slug", None)
    query = db.query(model.id)
    if name_col is not None:
        query = query.add_columns(name_col)
    if slug_col is not None:
        query = query.add_columns(slug_col)
    rows = query.filter(model.id.in_(ordered_ids)).all()
    label_map: Dict[str, str] = {}
    for row in rows:
        row_id = row[0]
        row_name = row[1] if len(row) > 1 else None
        row_slug = row[2] if len(row) > 2 else None
        label_map[row_id] = row_name or row_slug or row_id

    normalized: List[Dict[str, str]] = []
    for value in ordered_ids:
        normalized.append({"id": value, "label": label_map.get(value, value)})
    return normalized


def _backfill_typical_links(
    db: Session,
    *,
    product_id: str,
    limit: int = 5,
) -> None:
    """
    Populate typical asset/channel associations for historical data based on
    observed AssetChannelImpact rows. Only fills records that do not already
    have associations.
    """
    rows = (
        db.query(
            AssetChannelImpact.asset_id,
            AssetChannelImpact.channel_id,
            AssetChannelImpact.evidence_count,
            AssetChannelImpact.impact_strength,
        )
        .filter(AssetChannelImpact.product_id == product_id)
        .all()
    )
    if not rows:
        return

    asset_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    channel_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def _score(evidence_count, impact_strength) -> float:
        count = float(evidence_count or 0.0)
        strength = abs(float(impact_strength or 0.0))
        return max(count, 0.0) + strength

    for asset_id, channel_id, evidence_count, impact_strength in rows:
        if not asset_id or not channel_id:
            continue
        s = _score(evidence_count, impact_strength)
        if s <= 0.0:
            s = 0.1
        asset_scores[str(asset_id)][str(channel_id)] += s
        channel_scores[str(channel_id)][str(asset_id)] += s

    if not asset_scores and not channel_scores:
        return

    asset_label_map = {
        row.id: (row.name or row.slug or row.id)
        for row in db.query(ArsenalAsset.id, ArsenalAsset.name, ArsenalAsset.slug)
        .filter(ArsenalAsset.product_id == product_id)
        .all()
    }
    channel_label_map = {
        row.id: (row.name or row.slug or row.id)
        for row in db.query(ArsenalChannel.id, ArsenalChannel.name, ArsenalChannel.slug)
        .filter(ArsenalChannel.product_id == product_id)
        .all()
    }

    asset_typicals = {
        asset_id: _top_reference_list(scores, channel_label_map, limit)
        for asset_id, scores in asset_scores.items()
        if scores
    }
    channel_typicals = {
        channel_id: _top_reference_list(scores, asset_label_map, limit)
        for channel_id, scores in channel_scores.items()
        if scores
    }

    changed = False

    if asset_typicals:
        for asset in (
            db.query(ArsenalAsset)
            .filter(ArsenalAsset.product_id == product_id)
            .all()
        ):
            if asset.typical_channels:
                continue
            refs = asset_typicals.get(asset.id)
            if refs:
                asset.typical_channels = refs
                changed = True

    if channel_typicals:
        for channel in (
            db.query(ArsenalChannel)
            .filter(ArsenalChannel.product_id == product_id)
            .all()
        ):
            if channel.typical_assets:
                continue
            refs = channel_typicals.get(channel.id)
            if refs:
                channel.typical_assets = refs
                changed = True

    if changed:
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise


def _normalize_string_list(values: Optional[Iterable[Any]]) -> Optional[List[str]]:
    if values is None:
        return None
    flattened: List[str] = []
    if isinstance(values, str):
        candidate_values = [seg.strip() for seg in values.split(",")]
    else:
        candidate_values = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                candidate_values.append(value.strip())
            elif isinstance(value, dict):
                for key in ("label", "value", "id", "name"):
                    raw = value.get(key)
                    if isinstance(raw, str):
                        candidate_values.append(raw.strip())
                        break
    seen = set()
    for entry in candidate_values:
        if not entry:
            continue
        if entry in seen:
            continue
        seen.add(entry)
        flattened.append(entry)
    return flattened or None


def _persona_id_from_actor(
    title: Optional[str],
    department: Optional[str],
    seniority: Optional[str],
) -> Optional[str]:
    parts = [
        (title or "").strip(),
        (department or "").strip(),
        (seniority or "").strip(),
    ]
    filtered = [p for p in parts if p]
    if not filtered:
        return None
    return "|".join(filtered)


def _format_counter(counter: Counter, *, limit: int = 6) -> List[Dict[str, Any]]:
    if not counter:
        return []
    return [
        {"label": label, "count": count}
        for label, count in counter.most_common(limit)
        if label
    ]


def _format_stage(label: str) -> str:
    core = (label or "").replace("_", " ").strip()
    return _title_case(core) if core else label


def _finalize_usage_bucket(bucket: Dict[str, Any], graph=None) -> Dict[str, Any]:
    if not bucket or bucket.get("total_engagements", 0) == 0:
        return {
            "total_engagements": 0,
            "accounts": [],
            "industries": [],
            "regions": [],
            "sizes": [],
            "stages": [],
            "concerns": [],
            "personas": [],
            "segments": [],
        }

    accounts = list(bucket["accounts"].values())
    accounts.sort(key=lambda a: (a.get("name") or "").lower())
    stage_counter = Counter(bucket.get("stages", Counter()))
    concern_counter = Counter(bucket.get("concerns", Counter()))

    segment_counter = Counter()
    for account in accounts:
        industry = account.get("industry")
        if industry:
            segment_counter[f"Industry: {industry}"] += 1
        revenue = account.get("revenue_range")
        if revenue:
            segment_counter[f"Revenue Range: {revenue}"] += 1
        employees = account.get("employee_range")
        if employees:
            segment_counter[f"Employee Range: {employees}"] += 1
        geography = account.get("region") or account.get("geography")
        if geography:
            segment_counter[f"Geography: {geography}"] += 1
        deal_status = account.get("deal_status")
        if deal_status:
            segment_counter[f"Deal Status: {deal_status}"] += 1
        funding = account.get("funding_stage")
        if funding:
            segment_counter[f"Funding Stage: {funding}"] += 1

    return {
        "total_engagements": bucket.get("total_engagements", 0),
        "accounts": accounts[:8],
        "industries": _format_counter(bucket.get("industries", Counter())),
        "regions": _format_counter(bucket.get("regions", Counter())),
        "sizes": _format_counter(bucket.get("sizes", Counter())),
        "stages": [
            {
                "code": canonical,
                "label": _stage_label(canonical),
                "count": count,
            }
            for stage_code, count in stage_counter.most_common()
            for canonical in [_normalize_stage(stage_code)]
            if canonical and canonical in BELIEF_STAGE_LABELS
        ],
        "concerns": [
            {"label": label, "count": count}
            for label, count in concern_counter.most_common(12)
            if label
        ],
        "personas": [
            {
                "id": persona_id,
                "label": bucket.get("persona_labels", {}).get(persona_id)
                or _persona_label(persona_id, graph),
                "count": count,
            }
            for persona_id, count in bucket.get("personas", Counter()).most_common(8)
            if persona_id
        ],
        "segments": _format_counter(segment_counter, limit=12),
    }


def _compute_usage_stats(
    db: Session,
    *,
    product_id: str,
) -> Dict[str, Dict[str, Any]]:
    _ensure_engagement_columns()
    try:
        product_graph = build_product_graph(product_id)
    except Exception:
        product_graph = None

    def _make_bucket() -> Dict[str, Any]:
        return {
            "total_engagements": 0,
            "accounts": {},
            "industries": Counter(),
            "regions": Counter(),
            "sizes": Counter(),
            "stages": Counter(),
            "concerns": Counter(),
            "personas": Counter(),
            "persona_labels": {},
        }

    asset_usage: Dict[str, Dict[str, Any]] = defaultdict(_make_bucket)
    channel_usage: Dict[str, Dict[str, Any]] = defaultdict(_make_bucket)

    asset_rows = (
        db.query(
            TargetAccountEngagement.asset_id,
            TargetAccountEngagement.source,
            TargetAccountEngagement.actor_title,
            TargetAccountEngagement.actor_department,
            TargetAccountEngagement.actor_seniority,
            TargetAccountEngagement.target_account_id,
            TargetAccountORM.account_name,
            TargetAccountORM.industry,
            TargetAccountORM.geography,
            TargetAccountORM.revenue_range,
            TargetAccountORM.employee_range,
            TargetAccountORM.funding_stage,
            TargetAccountORM.deal_status,
        )
        .outerjoin(
            TargetAccountORM,
            TargetAccountORM.id == TargetAccountEngagement.target_account_id,
        )
        .filter(
            TargetAccountEngagement.product_id == product_id,
            TargetAccountEngagement.asset_id.isnot(None),
            TargetAccountEngagement.asset_id != "",
        )
        .all()
    )

    for row in asset_rows:
        asset_id = row.asset_id
        if not asset_id:
            continue
        bucket = asset_usage[asset_id]
        bucket["total_engagements"] += 1
        if row.target_account_id:
            bucket["accounts"][row.target_account_id] = {
                "id": row.target_account_id,
                "name": row.account_name,
                "industry": row.industry,
                "revenue_range": getattr(row, "revenue_range", None),
                "employee_range": getattr(row, "employee_range", None),
                "geography": getattr(row, "geography", None),
                "funding_stage": getattr(row, "funding_stage", None),
                "deal_status": row.deal_status,
            }
        if row.industry:
            bucket["industries"][row.industry] += 1
        if row.geography:
            bucket["regions"][row.geography] += 1
        if row.employee_range:
            bucket["sizes"][row.employee_range] += 1
        persona_id = _persona_id_from_actor(
            row.actor_title,
            row.actor_department,
            row.actor_seniority,
        )
        if persona_id:
            bucket["personas"][persona_id] += 1
            bucket["persona_labels"][persona_id] = _persona_label(persona_id, product_graph)

    channel_rows = (
        db.query(
            TargetAccountEngagement.channel_id,
            TargetAccountEngagement.channel,
            TargetAccountEngagement.source,
            TargetAccountEngagement.actor_title,
            TargetAccountEngagement.actor_department,
            TargetAccountEngagement.actor_seniority,
            TargetAccountEngagement.target_account_id,
            TargetAccountORM.account_name,
            TargetAccountORM.industry,
            TargetAccountORM.geography,
            TargetAccountORM.revenue_range,
            TargetAccountORM.employee_range,
            TargetAccountORM.funding_stage,
            TargetAccountORM.deal_status,
            ArsenalChannel.name.label("canonical_channel_name"),
        )
        .outerjoin(
            TargetAccountORM,
            TargetAccountORM.id == TargetAccountEngagement.target_account_id,
        )
        .outerjoin(
            ArsenalChannel,
            ArsenalChannel.id == TargetAccountEngagement.channel_id,
        )
        .filter(
            TargetAccountEngagement.product_id == product_id,
            (TargetAccountEngagement.channel_id.isnot(None))
            | (TargetAccountEngagement.channel.isnot(None)),
        )
        .all()
    )

    for row in channel_rows:
        channel_name = (row.canonical_channel_name or row.channel or "").strip()
        if not (row.channel_id or channel_name):
            continue
        channel_key = row.channel_id or f"channel:{ArsenalChannel.slug_for(channel_name)}"
        bucket = channel_usage[channel_key]
        if row.channel_id:
            alias_key = f"channel:{ArsenalChannel.slug_for(channel_name or row.channel_id)}"
            channel_usage.setdefault(alias_key, bucket)
        bucket["total_engagements"] += 1
        if row.target_account_id:
            bucket["accounts"][row.target_account_id] = {
                "id": row.target_account_id,
                "name": row.account_name,
                "industry": row.industry,
                "revenue_range": getattr(row, "revenue_range", None),
                "employee_range": getattr(row, "employee_range", None),
                "geography": getattr(row, "geography", None),
                "funding_stage": getattr(row, "funding_stage", None),
                "deal_status": row.deal_status,
            }
        if row.industry:
            bucket["industries"][row.industry] += 1
        if row.geography:
            bucket["regions"][row.geography] += 1
        if row.employee_range:
            bucket["sizes"][row.employee_range] += 1
        persona_id = _persona_id_from_actor(
            row.actor_title,
            row.actor_department,
            row.actor_seniority,
        )
        if persona_id:
            bucket["personas"][persona_id] += 1
            bucket["persona_labels"][persona_id] = _persona_label(persona_id, product_graph)

    impact_rows = (
        db.query(
            AssetChannelImpact.asset_id,
            AssetChannelImpact.channel_id,
            AssetChannelImpact.persona_id,
            AssetChannelImpact.evidence_details,
        )
        .filter(AssetChannelImpact.product_id == product_id)
        .all()
    )

    for impact in impact_rows:
        asset_bucket = asset_usage[impact.asset_id]
        channel_bucket = channel_usage[impact.channel_id]
        persona_id = impact.persona_id
        target_buckets = [asset_bucket, channel_bucket]
        for bucket in target_buckets:
            if bucket is None:
                continue
            if persona_id:
                bucket["personas"][persona_id] += 0  # ensure persona present
                bucket["persona_labels"][persona_id] = _persona_label(persona_id, product_graph)
        details_iter = impact.evidence_details or []
        if not isinstance(details_iter, list):
            details_iter = [details_iter]
        for detail in details_iter:
            if not isinstance(detail, dict):
                continue
            stage_entries = detail.get("stages") or detail.get("stage_counts")
            if isinstance(stage_entries, dict):
                for raw_stage, count in stage_entries.items():
                    canonical = _normalize_stage(raw_stage)
                    if not canonical:
                        continue
                    for bucket in target_buckets:
                        if bucket is None:
                            continue
                        bucket["stages"][canonical] += int(count or 0) if count is not None else 1
            concerns = detail.get("concerns") or detail.get("concern_labels")
            if concerns:
                if isinstance(concerns, dict):
                    concern_iter = concerns.items()
                elif isinstance(concerns, list):
                    concern_iter = [(c, 1) for c in concerns]
                else:
                    concern_iter = [(concerns, 1)]
                for label, count in concern_iter:
                    if not label:
                        continue
                    for bucket in target_buckets:
                        if bucket is None:
                            continue
                        bucket["concerns"][label] += int(count or 0) if isinstance(count, (int, float)) else 1

    finalized_assets = {
        asset_id: _finalize_usage_bucket(bucket, product_graph)
        for asset_id, bucket in asset_usage.items()
    }
    finalized_channels: Dict[str, Dict[str, Any]] = {}
    for channel_id, bucket in channel_usage.items():
        finalized = _finalize_usage_bucket(bucket, product_graph)
        finalized_channels[channel_id] = finalized
        if channel_id.startswith("channel:"):
            slug = channel_id.split(":", 1)[1]
            finalized_channels.setdefault(slug, finalized)
    return {
        "assets": finalized_assets,
        "channels": finalized_channels,
    }


def _serialize_asset(
    asset: ArsenalAsset,
    *,
    include_impacts: bool = False,
    usage: Optional[Dict[str, Any]] = None,
    product_graph=None,
) -> Dict[str, Any]:
    usage_payload = usage or _finalize_usage_bucket({})
    belief_stages, detected_code = _split_belief_and_maturity(asset.target_belief_stages or [])
    maturity_code = _normalize_funnel_code(asset.org_conversion_maturity) or detected_code
    if not maturity_code:
        fallback_stage = _funnel_stage_from_iterable(usage_payload.get("stages"))
        maturity_code = fallback_stage.get("code") if fallback_stage else None
    maturity_payload = (
        {
            "code": maturity_code,
            "label": _FUNNEL_STAGE_LABELS.get(maturity_code, _title_case(maturity_code)),
        }
        if maturity_code
        else None
    )
    payload: Dict[str, Any] = {
        "id": asset.id,
        "product_id": asset.product_id,
        "name": asset.name,
        "slug": asset.slug,
        "category": _metadata_value(asset.category, asset.category_text),
        "content_type": _metadata_value(asset.content_type, asset.content_type_text),
        "time_to_consume": _metadata_value(asset.time_to_consume, asset.time_to_consume_text),
        "depth": _metadata_value(asset.depth, asset.depth_text),
        "description": asset.description,
        "metadata_complete": bool(
            asset.metadata_complete and asset.approval_status == ApprovalStatus.APPROVED
        ),
        "created_from_engagement_id": asset.created_from_engagement_id,
        "notes": asset.notes,
        "active": bool(asset.active),
        "created_at": asset.created_at.isoformat() if asset.created_at else None,
        "updated_at": asset.updated_at.isoformat() if asset.updated_at else None,
        "call_stage": asset.call_stage,
        "target_personas": list(asset.target_personas or []),
        "target_account_segments": list(asset.target_account_segments or []),
        "target_belief_stages": belief_stages,
        "target_concerns": list(asset.target_concerns or []),
        "usage": usage_payload,
        "org_conversion_maturity": maturity_payload,
        "typical_channels": _serialize_reference_list(asset.typical_channels),
        "approval_status": asset.approval_status.value if asset.approval_status else None,
        "derived_metadata": asset.derived_metadata,
        "auto_classification_confidence": asset.auto_classification_confidence,
    }
    if include_impacts:
        payload["impacts"] = [
            _serialize_impact(impact, include_links=False, product_graph=product_graph)
            for impact in asset.channel_impacts or []
        ]
    return payload


def _serialize_channel(
    channel: ArsenalChannel,
    *,
    include_impacts: bool = False,
    usage: Optional[Dict[str, Any]] = None,
    product_graph=None,
) -> Dict[str, Any]:
    usage_payload = usage or _finalize_usage_bucket({})
    belief_stages, detected_code = _split_belief_and_maturity(channel.target_belief_stages or [])
    maturity_code = _normalize_funnel_code(channel.org_conversion_maturity) or detected_code
    if not maturity_code:
        fallback_stage = _funnel_stage_from_iterable(usage_payload.get("stages"))
        maturity_code = fallback_stage.get("code") if fallback_stage else None
    maturity_payload = (
        {
            "code": maturity_code,
            "label": _FUNNEL_STAGE_LABELS.get(maturity_code, _title_case(maturity_code)),
        }
        if maturity_code
        else None
    )
    payload: Dict[str, Any] = {
        "id": channel.id,
        "product_id": channel.product_id,
        "name": channel.name,
        "slug": channel.slug,
        "channel_type": _metadata_value(channel.channel_type, channel.channel_type_text),
        "delivery_mode": _metadata_value(channel.delivery_mode, channel.delivery_mode_text),
        "reach_score_estimate": channel.reach_score_estimate,
        "metadata_complete": bool(
            channel.metadata_complete and channel.approval_status == ApprovalStatus.APPROVED
        ),
        "created_from_engagement_id": channel.created_from_engagement_id,
        "notes": channel.notes,
        "active": bool(channel.active),
        "created_at": channel.created_at.isoformat() if channel.created_at else None,
        "updated_at": channel.updated_at.isoformat() if channel.updated_at else None,
        "target_personas": list(channel.target_personas or []),
        "target_account_segments": list(channel.target_account_segments or []),
        "target_belief_stages": belief_stages,
        "target_concerns": list(channel.target_concerns or []),
        "usage": usage_payload,
        "org_conversion_maturity": maturity_payload,
        "typical_assets": _serialize_reference_list(channel.typical_assets),
        "approval_status": channel.approval_status.value if channel.approval_status else None,
        "derived_metadata": channel.derived_metadata,
        "auto_classification_confidence": channel.auto_classification_confidence,
    }
    if include_impacts:
        payload["impacts"] = [
            _serialize_impact(impact, include_links=False, product_graph=product_graph)
            for impact in channel.asset_impacts or []
        ]
    return payload


def _serialize_impact(
    impact: AssetChannelImpact,
    include_links: bool = True,
    product_graph=None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": impact.id,
        "product_id": impact.product_id,
        "asset_id": impact.asset_id,
        "channel_id": impact.channel_id,
        "persona_id": impact.persona_id,
        "belief_transition_id": impact.belief_transition_id,
        "impact_strength": impact.impact_strength,
        "evidence_count": impact.evidence_count,
        "last_evidence_at": impact.last_evidence_at.isoformat() if impact.last_evidence_at else None,
        "evidence_details": impact.evidence_details,
        "persona_label": _persona_label(impact.persona_id, product_graph),
    }
    if include_links:
        payload["asset"] = (
            _serialize_asset(impact.asset, include_impacts=False, product_graph=product_graph)
            if impact.asset
            else None
        )
        payload["channel"] = (
            _serialize_channel(impact.channel, include_impacts=False, product_graph=product_graph)
            if impact.channel
            else None
        )
    return payload


def list_assets(
    db: Session,
    *,
    product_id: str,
    include_impacts: bool = False,
    usage_map: Optional[Dict[str, Dict[str, Any]]] = None,
    product_graph=None,
) -> List[Dict[str, Any]]:
    query = (
        db.query(ArsenalAsset)
        .options(joinedload(ArsenalAsset.channel_impacts) if include_impacts else ())
        .filter(
            ArsenalAsset.product_id == product_id,
            ArsenalAsset.active.is_(True),
        )
        .order_by(func.lower(ArsenalAsset.name))
    )
    rows = query.all()
    usage_map = usage_map or {}
    return [
        _serialize_asset(
            row,
            include_impacts=include_impacts,
            usage=usage_map.get(row.id) or usage_map.get(row.slug),
            product_graph=product_graph,
        )
        for row in rows
    ]


def list_channels(
    db: Session,
    *,
    product_id: str,
    include_impacts: bool = False,
    usage_map: Optional[Dict[str, Dict[str, Any]]] = None,
    product_graph=None,
) -> List[Dict[str, Any]]:
    query = (
        db.query(ArsenalChannel)
        .options(joinedload(ArsenalChannel.asset_impacts) if include_impacts else ())
        .filter(
            ArsenalChannel.product_id == product_id,
            ArsenalChannel.active.is_(True),
        )
        .order_by(func.lower(ArsenalChannel.name))
    )
    rows = query.all()
    usage_map = usage_map or {}
    return [
        _serialize_channel(
            row,
            include_impacts=include_impacts,
            usage=usage_map.get(row.id)
            or usage_map.get(row.slug)
            or usage_map.get(f"channel:{row.slug}"),
            product_graph=product_graph,
        )
        for row in rows
    ]


def list_asset_channel_impacts(
    db: Session,
    *,
    product_id: str,
    persona_id: Optional[str] = None,
    product_graph=None,
) -> List[Dict[str, Any]]:
    query = (
        db.query(AssetChannelImpact)
        .options(
            joinedload(AssetChannelImpact.asset),
            joinedload(AssetChannelImpact.channel),
        )
        .filter(AssetChannelImpact.product_id == product_id)
    )
    if persona_id:
        query = query.filter(AssetChannelImpact.persona_id == persona_id)
    rows = query.all()
    return [_serialize_impact(row, product_graph=product_graph) for row in rows]


def _resolve_enum(enum_cls, value):
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for member in enum_cls:
            if member.value == normalized:
                return member
    return None


def _resolve_approval_status(value) -> Optional[ApprovalStatus]:
    if value is None:
        return None
    if isinstance(value, ApprovalStatus):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        for status in ApprovalStatus:
            if status.value == normalized:
                return status
    return None


def _apply_auto_classification(target, defaults: Dict[str, Any]) -> None:
    """
    Applies parser-derived defaults (approval status, derived metadata, confidence)
    without overwriting human-approved metadata.
    """
    if not defaults:
        return
    status = _resolve_approval_status(defaults.get("approval_status"))
    if status and getattr(target, "approval_status", None) in (None, ApprovalStatus.PENDING):
        target.approval_status = status
    auto_conf = defaults.get("auto_classification_confidence")
    if auto_conf is not None and (
        getattr(target, "approval_status", ApprovalStatus.PENDING) != ApprovalStatus.APPROVED
        or getattr(target, "auto_classification_confidence", None) is None
    ):
        target.auto_classification_confidence = auto_conf
    derived_meta = defaults.get("derived_metadata")
    if derived_meta and getattr(target, "approval_status", ApprovalStatus.PENDING) != ApprovalStatus.APPROVED:
        target.derived_metadata = derived_meta


def get_or_create_asset(
    db: Session,
    *,
    product_id: str,
    name: str,
    defaults: Optional[Dict[str, Any]] = None,
    explicit_id: Optional[str] = None,
) -> Tuple[ArsenalAsset, bool]:
    defaults = defaults or {}
    slug = defaults.get("slug") or ArsenalAsset.slug_for(name)

    if explicit_id:
        existing_by_id = db.get(ArsenalAsset, explicit_id)
        if existing_by_id:
            return existing_by_id, False

    asset = (
        db.query(ArsenalAsset)
        .filter(
            ArsenalAsset.product_id == product_id,
            ArsenalAsset.slug == slug,
        )
        .first()
    )
    if asset:
        _apply_auto_classification(asset, defaults)
        return asset, False

    slug = _dedupe_slug(db, ArsenalAsset, slug)

    asset = ArsenalAsset(
        id=explicit_id or str(uuid.uuid4()),
        product_id=product_id,
        name=name.strip(),
        slug=slug,
    )

    asset.category = _resolve_enum(AssetCategory, defaults.get("category"))
    asset.content_type = _resolve_enum(ContentType, defaults.get("content_type"))
    asset.time_to_consume = _resolve_enum(TimeToConsume, defaults.get("time_to_consume"))
    asset.depth = _resolve_enum(AssetDepth, defaults.get("depth"))
    asset.description = defaults.get("description")
    asset.notes = defaults.get("notes")
    asset.created_from_engagement_id = defaults.get("created_from_engagement_id")
    asset.call_stage = defaults.get("call_stage")
    tp = _normalize_string_list(defaults.get("target_personas"))
    if tp is not None:
        asset.target_personas = tp
    segs = _normalize_string_list(defaults.get("target_account_segments"))
    if segs is not None:
        asset.target_account_segments = segs
    belief_stages, detected_maturity = _split_belief_and_maturity(
        defaults.get("target_belief_stages")
    )
    if belief_stages:
        asset.target_belief_stages = belief_stages
    else:
        asset.target_belief_stages = None
    override_maturity = _normalize_funnel_code(defaults.get("org_conversion_maturity"))
    asset.org_conversion_maturity = override_maturity or detected_maturity
    concerns = _normalize_string_list(defaults.get("target_concerns"))
    if concerns is not None:
        asset.target_concerns = concerns
    _apply_auto_classification(asset, defaults)
    asset.update_metadata_status()

    db.add(asset)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(ArsenalAsset)
            .filter(
                ArsenalAsset.product_id == product_id,
                ArsenalAsset.slug == slug,
            )
            .first()
        )
        if existing:
            return existing, False
        raise
    return asset, True


def get_or_create_channel(
    db: Session,
    *,
    product_id: str,
    name: str,
    defaults: Optional[Dict[str, Any]] = None,
    explicit_id: Optional[str] = None,
) -> Tuple[ArsenalChannel, bool]:
    defaults = defaults or {}
    slug = defaults.get("slug") or ArsenalChannel.slug_for(name)

    if explicit_id:
        existing_by_id = db.get(ArsenalChannel, explicit_id)
        if existing_by_id:
            return existing_by_id, False

    channel = (
        db.query(ArsenalChannel)
        .filter(
            ArsenalChannel.product_id == product_id,
            ArsenalChannel.slug == slug,
        )
        .first()
    )
    if channel:
        _apply_auto_classification(channel, defaults)
        return channel, False

    slug = _dedupe_slug(db, ArsenalChannel, slug)

    channel = ArsenalChannel(
        id=explicit_id or str(uuid.uuid4()),
        product_id=product_id,
        name=name.strip(),
        slug=slug,
    )
    channel.channel_type = _resolve_enum(ChannelType, defaults.get("channel_type"))
    channel.delivery_mode = _resolve_enum(ChannelDelivery, defaults.get("delivery_mode"))
    channel.reach_score_estimate = defaults.get("reach_score_estimate")
    channel.notes = defaults.get("notes")
    channel.created_from_engagement_id = defaults.get("created_from_engagement_id")
    tp = _normalize_string_list(defaults.get("target_personas"))
    if tp is not None:
        channel.target_personas = tp
    segs = _normalize_string_list(defaults.get("target_account_segments"))
    if segs is not None:
        channel.target_account_segments = segs
    belief_stages, detected_maturity = _split_belief_and_maturity(
        defaults.get("target_belief_stages")
    )
    if belief_stages:
        channel.target_belief_stages = belief_stages
    else:
        channel.target_belief_stages = None
    override_maturity = _normalize_funnel_code(defaults.get("org_conversion_maturity"))
    channel.org_conversion_maturity = override_maturity or detected_maturity
    concerns = _normalize_string_list(defaults.get("target_concerns"))
    if concerns is not None:
        channel.target_concerns = concerns
    _apply_auto_classification(channel, defaults)
    channel.update_metadata_status()

    db.add(channel)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(ArsenalChannel)
            .filter(
                ArsenalChannel.product_id == product_id,
                ArsenalChannel.slug == slug,
            )
            .first()
        )
        if existing:
            return existing, False
        raise
    return channel, True


def deactivate_asset(db: Session, *, asset_id: str) -> None:
    asset = db.get(ArsenalAsset, asset_id)
    if not asset:
        return
    asset.active = False
    db.add(asset)


def deactivate_channel(db: Session, *, channel_id: str) -> None:
    channel = db.get(ArsenalChannel, channel_id)
    if not channel:
        return
    channel.active = False
    db.add(channel)


def record_asset_channel_impact(
    db: Session,
    *,
    product_id: str,
    asset_id: str,
    channel_id: str,
    persona_id: str,
    belief_transition_id: Optional[str],
    delta_strength: float,
    evidence_payload: Optional[Dict[str, Any]] = None,
) -> AssetChannelImpact:
    impact = (
        db.query(AssetChannelImpact)
        .filter(
            AssetChannelImpact.asset_id == asset_id,
            AssetChannelImpact.channel_id == channel_id,
            AssetChannelImpact.product_id == product_id,
            AssetChannelImpact.persona_id == persona_id,
            AssetChannelImpact.belief_transition_id == belief_transition_id,
        )
        .first()
    )
    if not impact:
        impact = AssetChannelImpact(
            asset_id=asset_id,
            channel_id=channel_id,
            product_id=product_id,
            persona_id=persona_id,
            belief_transition_id=belief_transition_id,
            impact_strength=delta_strength,
            evidence_count=0,
            evidence_details=[],
        )
        db.add(impact)
        db.flush()

    impact.register_evidence(delta_strength, metadata=evidence_payload)
    db.add(impact)
    return impact


def serialize_arsenal_library(
    db: Session,
    *,
    product_id: str,
) -> Dict[str, Any]:
    _backfill_typical_links(db, product_id=product_id)
    usage = _compute_usage_stats(db, product_id=product_id)
    try:
        product_graph = build_product_graph(product_id)
    except Exception:
        product_graph = None
    assets = list_assets(
        db,
        product_id=product_id,
        include_impacts=True,
        usage_map=usage.get("assets"),
        product_graph=product_graph,
    )
    channels = list_channels(
        db,
        product_id=product_id,
        include_impacts=True,
        usage_map=usage.get("channels"),
        product_graph=product_graph,
    )
    impacts = list_asset_channel_impacts(
        db,
        product_id=product_id,
        product_graph=product_graph,
    )
    return {
        "assets": assets,
        "channels": channels,
        "impacts": impacts,
    }


def _apply_enum(enum_cls, value: Optional[str]):
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        enum_value = _resolve_enum(enum_cls, normalized)
        if enum_value is None:
            raise ValueError(f"Invalid value '{value}' for {enum_cls.__name__}")
        return enum_value
    if isinstance(value, enum_cls):
        return value
    raise ValueError(f"Invalid value '{value}' for {enum_cls.__name__}")


def create_asset_record(
    db: Session,
    *,
    product_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    name = (payload.get("name") or "").strip()
    if not name:
        raise ValueError("Asset name is required.")
    defaults = {
        "description": payload.get("description"),
        "notes": payload.get("notes"),
    }
    asset, created = get_or_create_asset(
        db,
        product_id=product_id,
        name=name,
        defaults=defaults,
    )
    if not created:
        raise ValueError("An asset with this name already exists. Edit it instead.")
    patch = {key: payload.get(key) for key in ASSET_MUTABLE_FIELDS if key in payload}
    if "name" not in patch:
        patch["name"] = name
    return update_asset_metadata(db, asset_id=asset.id, patch=patch)


def create_channel_record(
    db: Session,
    *,
    product_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    name = (payload.get("name") or "").strip()
    if not name:
        raise ValueError("Channel name is required.")
    defaults = {
        "notes": payload.get("notes"),
    }
    channel, created = get_or_create_channel(
        db,
        product_id=product_id,
        name=name,
        defaults=defaults,
    )
    if not created:
        raise ValueError("A channel with this name already exists.")
    patch = {key: payload.get(key) for key in CHANNEL_MUTABLE_FIELDS if key in payload}
    if "name" not in patch:
        patch["name"] = name
    return update_channel_metadata(db, channel_id=channel.id, patch=patch)


def update_asset_metadata(
    db: Session,
    *,
    asset_id: str,
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    asset = db.get(ArsenalAsset, asset_id)
    if asset is None:
        raise LookupError(f"Asset '{asset_id}' not found")

    if "name" in patch and patch["name"]:
        asset.name = str(patch["name"]).strip()
    if "category" in patch:
        cleaned = _clean_metadata_text(patch.get("category"))
        if not cleaned:
            asset.category = None
            asset.category_text = None
        else:
            try:
                asset.category = _apply_enum(AssetCategory, cleaned)
                asset.category_text = None
            except ValueError:
                asset.category = None
                asset.category_text = cleaned
    if "content_type" in patch:
        cleaned = _clean_metadata_text(patch.get("content_type"))
        if not cleaned:
            asset.content_type = None
            asset.content_type_text = None
        else:
            try:
                asset.content_type = _apply_enum(ContentType, cleaned)
                asset.content_type_text = None
            except ValueError:
                asset.content_type = None
                asset.content_type_text = cleaned
    if "time_to_consume" in patch:
        cleaned = _clean_metadata_text(patch.get("time_to_consume"))
        if not cleaned:
            asset.time_to_consume = None
            asset.time_to_consume_text = None
        else:
            try:
                asset.time_to_consume = _apply_enum(TimeToConsume, cleaned)
                asset.time_to_consume_text = None
            except ValueError:
                asset.time_to_consume = None
                asset.time_to_consume_text = cleaned
    if "depth" in patch:
        cleaned = _clean_metadata_text(patch.get("depth"))
        if not cleaned:
            asset.depth = None
            asset.depth_text = None
        else:
            try:
                asset.depth = _apply_enum(AssetDepth, cleaned)
                asset.depth_text = None
            except ValueError:
                asset.depth = None
                asset.depth_text = cleaned
    if "description" in patch:
        desc = patch.get("description")
        asset.description = desc.strip() if isinstance(desc, str) and desc.strip() else None
    if "notes" in patch:
        notes = patch.get("notes")
        asset.notes = notes.strip() if isinstance(notes, str) and notes.strip() else None
    if "call_stage" in patch:
        stage = patch.get("call_stage")
        asset.call_stage = stage.strip() if isinstance(stage, str) and stage.strip() else None
    if "target_personas" in patch:
        asset.target_personas = _normalize_string_list(patch.get("target_personas"))
    if "target_account_segments" in patch:
        asset.target_account_segments = _normalize_string_list(
            patch.get("target_account_segments")
        )
    if "target_belief_stages" in patch:
        belief_stages, detected_maturity = _split_belief_and_maturity(
            patch.get("target_belief_stages")
        )
        asset.target_belief_stages = belief_stages or None
        if detected_maturity is not None:
            asset.org_conversion_maturity = detected_maturity
    if "org_conversion_maturity" in patch:
        asset.org_conversion_maturity = _normalize_funnel_code(
            patch.get("org_conversion_maturity")
        )
    if "target_concerns" in patch:
        asset.target_concerns = _normalize_string_list(patch.get("target_concerns"))
    if "typical_channels" in patch:
        refs = _normalize_reference_list(db, ArsenalChannel, patch.get("typical_channels"))
        asset.typical_channels = refs or None
    if "approval_status" in patch:
        status = _resolve_approval_status(patch.get("approval_status"))
        if status:
            asset.approval_status = status

    asset.update_metadata_status()
    db.add(asset)
    db.flush()
    db.refresh(asset)
    usage_map = _compute_usage_stats(db, product_id=asset.product_id)["assets"]
    return _serialize_asset(
        asset,
        include_impacts=True,
        usage=usage_map.get(asset.id),
    )


def update_channel_metadata(
    db: Session,
    *,
    channel_id: str,
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    channel = db.get(ArsenalChannel, channel_id)
    if channel is None:
        raise LookupError(f"Channel '{channel_id}' not found")

    if "name" in patch and patch["name"]:
        channel.name = str(patch["name"]).strip()
    if "channel_type" in patch:
        cleaned = _clean_metadata_text(patch.get("channel_type"))
        if not cleaned:
            channel.channel_type = None
            channel.channel_type_text = None
        else:
            try:
                channel.channel_type = _apply_enum(ChannelType, cleaned)
                channel.channel_type_text = None
            except ValueError:
                channel.channel_type = None
                channel.channel_type_text = cleaned
    if "delivery_mode" in patch:
        cleaned = _clean_metadata_text(patch.get("delivery_mode"))
        if not cleaned:
            channel.delivery_mode = None
            channel.delivery_mode_text = None
        else:
            try:
                channel.delivery_mode = _apply_enum(ChannelDelivery, cleaned)
                channel.delivery_mode_text = None
            except ValueError:
                channel.delivery_mode = None
                channel.delivery_mode_text = cleaned
    if "reach_score_estimate" in patch:
        value = patch.get("reach_score_estimate")
        if value in (None, ""):
            channel.reach_score_estimate = None
        else:
            try:
                channel.reach_score_estimate = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("reach_score_estimate must be a number") from exc
    if "notes" in patch:
        notes = patch.get("notes")
        channel.notes = notes.strip() if isinstance(notes, str) and notes.strip() else None
    if "target_personas" in patch:
        channel.target_personas = _normalize_string_list(patch.get("target_personas"))
    if "target_account_segments" in patch:
        channel.target_account_segments = _normalize_string_list(
            patch.get("target_account_segments")
        )
    if "target_belief_stages" in patch:
        belief_stages, detected_maturity = _split_belief_and_maturity(
            patch.get("target_belief_stages")
        )
        channel.target_belief_stages = belief_stages or None
        if detected_maturity is not None:
            channel.org_conversion_maturity = detected_maturity
    if "org_conversion_maturity" in patch:
        channel.org_conversion_maturity = _normalize_funnel_code(
            patch.get("org_conversion_maturity")
        )
    if "target_concerns" in patch:
        channel.target_concerns = _normalize_string_list(patch.get("target_concerns"))
    if "typical_assets" in patch:
        refs = _normalize_reference_list(db, ArsenalAsset, patch.get("typical_assets"))
        channel.typical_assets = refs or None
    if "approval_status" in patch:
        status = _resolve_approval_status(patch.get("approval_status"))
        if status:
            channel.approval_status = status

    channel.update_metadata_status()
    db.add(channel)
    db.flush()
    db.refresh(channel)
    usage_map = _compute_usage_stats(db, product_id=channel.product_id)["channels"]
    return _serialize_channel(
        channel,
        include_impacts=True,
        usage=usage_map.get(channel.id),
    )


def enum_options_payload(db: Session) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    def _options(enum_cls, extras: Optional[List[str]] = None):
        rows: List[Dict[str, str]] = [
            {"value": member.value, "label": member.value.replace("_", " ").title()}
            for member in enum_cls
        ]
        seen = {member.value.lower() for member in enum_cls}
        for value in sorted(extras or []):
            cleaned = _clean_metadata_text(value)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"value": cleaned, "label": _title_case(cleaned)})
        return rows

    def _distinct_text_values(column) -> List[str]:
        try:
            rows = (
                db.query(column)
                .filter(column.isnot(None))
                .distinct()
                .all()
            )
        except Exception:
            return []
        values: List[str] = []
        for (raw,) in rows:
            cleaned = _clean_metadata_text(raw)
            if cleaned:
                values.append(cleaned)
        return values

    sales_call_stage_options = [
        {"value": "discovery", "label": "Discovery"},
        {"value": "value_proposition", "label": "Value Proposition"},
        {"value": "demo", "label": "Demo"},
        {"value": "implementation", "label": "Implementation"},
        {"value": "pricing", "label": "Pricing"},
        {"value": "barriers", "label": "Barriers"},
        {"value": "negotiation", "label": "Negotiation"},
        {"value": "contracting", "label": "Contracting"},
    ]
    belief_stage_options = [
        {"value": "problem", "label": BELIEF_STAGE_LABELS["problem"]},
        {"value": "pain", "label": BELIEF_STAGE_LABELS["pain"]},
        {"value": "resolution", "label": BELIEF_STAGE_LABELS["resolution"]},
        {"value": "execution", "label": BELIEF_STAGE_LABELS["execution"]},
    ]

    return {
        "asset": {
            "category": _options(AssetCategory, _distinct_text_values(ArsenalAsset.category_text)),
            "content_type": _options(ContentType, _distinct_text_values(ArsenalAsset.content_type_text)),
            "time_to_consume": _options(TimeToConsume, _distinct_text_values(ArsenalAsset.time_to_consume_text)),
            "depth": _options(AssetDepth, _distinct_text_values(ArsenalAsset.depth_text)),
            "sales_call_stage": sales_call_stage_options,
            "belief_stage": belief_stage_options,
        },
        "channel": {
            "channel_type": _options(ChannelType, _distinct_text_values(ArsenalChannel.channel_type_text)),
            "delivery_mode": _options(ChannelDelivery, _distinct_text_values(ArsenalChannel.delivery_mode_text)),
        },
    }
_SLUG_MAX_LENGTH = 120


def _truncate_slug(base: str, suffix: str) -> str:
    base = base[: max(1, _SLUG_MAX_LENGTH - len(suffix))]
    if base.endswith("-"):
        base = base.rstrip("-")
    return f"{base}{suffix}"


def _dedupe_slug(
    db: Session,
    model,
    base_slug: str,
) -> str:
    slug = base_slug
    counter = 1
    while (
        db.query(model.id)
        .filter(model.slug == slug)
        .first()
    ):
        suffix = f"-{counter}"
        slug = _truncate_slug(base_slug, suffix)
        counter += 1
    return slug
