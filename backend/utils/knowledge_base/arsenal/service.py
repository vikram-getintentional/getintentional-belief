from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import func, inspect, text
from sqlalchemy.exc import IntegrityError
import uuid
from sqlalchemy.orm import Session, joinedload

from backend.database import engine
from backend.utils.knowledge_base.arsenal.db_models import (
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
        columns = {col["name"] for col in insp.get_columns("arsenal_assets")}
    except Exception:
        return
    if "call_stage" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE arsenal_assets ADD COLUMN call_stage VARCHAR"))


_ensure_schema()


def _json_from_enum(value) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _serialize_asset(asset: ArsenalAsset, include_impacts: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": asset.id,
        "product_id": asset.product_id,
        "name": asset.name,
        "slug": asset.slug,
        "category": _json_from_enum(asset.category),
        "content_type": _json_from_enum(asset.content_type),
        "time_to_consume": _json_from_enum(asset.time_to_consume),
        "depth": _json_from_enum(asset.depth),
        "description": asset.description,
        "metadata_complete": bool(asset.metadata_complete),
        "created_from_engagement_id": asset.created_from_engagement_id,
        "notes": asset.notes,
        "active": bool(asset.active),
        "created_at": asset.created_at.isoformat() if asset.created_at else None,
        "updated_at": asset.updated_at.isoformat() if asset.updated_at else None,
        "call_stage": asset.call_stage,
    }
    if include_impacts:
        payload["impacts"] = [
            _serialize_impact(impact, include_links=False)
            for impact in asset.channel_impacts or []
        ]
    return payload


def _serialize_channel(channel: ArsenalChannel, include_impacts: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": channel.id,
        "product_id": channel.product_id,
        "name": channel.name,
        "slug": channel.slug,
        "channel_type": _json_from_enum(channel.channel_type),
        "delivery_mode": _json_from_enum(channel.delivery_mode),
        "reach_score_estimate": channel.reach_score_estimate,
        "metadata_complete": bool(channel.metadata_complete),
        "created_from_engagement_id": channel.created_from_engagement_id,
        "notes": channel.notes,
        "active": bool(channel.active),
        "created_at": channel.created_at.isoformat() if channel.created_at else None,
        "updated_at": channel.updated_at.isoformat() if channel.updated_at else None,
    }
    if include_impacts:
        payload["impacts"] = [
            _serialize_impact(impact, include_links=False)
            for impact in channel.asset_impacts or []
        ]
    return payload


def _serialize_impact(
    impact: AssetChannelImpact,
    include_links: bool = True,
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
    }
    if include_links:
        payload["asset"] = _serialize_asset(impact.asset, include_impacts=False) if impact.asset else None
        payload["channel"] = _serialize_channel(impact.channel, include_impacts=False) if impact.channel else None
    return payload


def list_assets(db: Session, *, product_id: str, include_impacts: bool = False) -> List[Dict[str, Any]]:
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
    return [_serialize_asset(row, include_impacts=include_impacts) for row in rows]


def list_channels(db: Session, *, product_id: str, include_impacts: bool = False) -> List[Dict[str, Any]]:
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
    return [_serialize_channel(row, include_impacts=include_impacts) for row in rows]


def list_asset_channel_impacts(
    db: Session,
    *,
    product_id: str,
    persona_id: Optional[str] = None,
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
    return [_serialize_impact(row) for row in rows]


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
        return asset, False

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
        return channel, False

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
    assets = list_assets(db, product_id=product_id, include_impacts=True)
    channels = list_channels(db, product_id=product_id, include_impacts=True)
    impacts = list_asset_channel_impacts(db, product_id=product_id)
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
        asset.category = _apply_enum(AssetCategory, patch.get("category"))
    if "content_type" in patch:
        asset.content_type = _apply_enum(ContentType, patch.get("content_type"))
    if "time_to_consume" in patch:
        asset.time_to_consume = _apply_enum(TimeToConsume, patch.get("time_to_consume"))
    if "depth" in patch:
        asset.depth = _apply_enum(AssetDepth, patch.get("depth"))
    if "description" in patch:
        desc = patch.get("description")
        asset.description = desc.strip() if isinstance(desc, str) and desc.strip() else None
    if "notes" in patch:
        notes = patch.get("notes")
        asset.notes = notes.strip() if isinstance(notes, str) and notes.strip() else None
    if "call_stage" in patch:
        stage = patch.get("call_stage")
        asset.call_stage = stage.strip() if isinstance(stage, str) and stage.strip() else None

    asset.update_metadata_status()
    db.add(asset)
    db.flush()
    db.refresh(asset)
    return _serialize_asset(asset, include_impacts=True)


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
        channel.channel_type = _apply_enum(ChannelType, patch.get("channel_type"))
    if "delivery_mode" in patch:
        channel.delivery_mode = _apply_enum(ChannelDelivery, patch.get("delivery_mode"))
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

    channel.update_metadata_status()
    db.add(channel)
    db.flush()
    db.refresh(channel)
    return _serialize_channel(channel, include_impacts=True)


def enum_options_payload() -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    def _options(enum_cls):
        return [
            {"value": member.value, "label": member.value.replace("_", " ").title()}
            for member in enum_cls
        ]

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

    return {
        "asset": {
            "category": _options(AssetCategory),
            "content_type": _options(ContentType),
            "time_to_consume": _options(TimeToConsume),
            "depth": _options(AssetDepth),
            "sales_call_stage": sales_call_stage_options,
        },
        "channel": {
            "channel_type": _options(ChannelType),
            "delivery_mode": _options(ChannelDelivery),
        },
    }
