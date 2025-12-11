from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from sqlalchemy.orm import Session

from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.knowledge_base.arsenal.channel_catalog import resolve_canonical_channel


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _history_key(row: TargetAccountEngagement) -> str:
    account = row.target_account_id or "global"
    persona = row.persona_id or ""
    return f"{account}:{persona}" if persona else account


def _channel_candidates(row: TargetAccountEngagement) -> Iterable[str]:
    if row.channel_id:
        yield row.channel_id
    if row.channel:
        yield row.channel
    if row.source:
        yield row.source
    if row.raw_activity:
        yield row.raw_activity
    if row.engagement_verb:
        yield row.engagement_verb
    payload = _normalize_payload(row.payload)
    for key in (
        "channel",
        "source",
        "utm_medium",
        "utm_source",
        "utm_campaign",
        "campaign",
        "campaign_id",
        "referrer",
        "origin",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            yield value
        elif isinstance(value, Iterable):
            for element in value:
                if isinstance(element, str) and element.strip():
                    yield element


def _canonical_channel_label(row: TargetAccountEngagement) -> Optional[str]:
    candidates = _channel_candidates(row)
    for candidate in candidates:
        normalized = str(candidate).strip()
        if not normalized:
            continue
        if normalized.startswith("channel:"):
            slug = normalized.split(":", 1)[1]
            if slug:
                return slug
        slug, metadata = resolve_canonical_channel(normalized)
        if slug:
            return slug
    return None


def attribute_channels_for_engagements(
    db: Session,
    *,
    product_id: Optional[str] = None,
) -> int:
    query = (
        db.query(TargetAccountEngagement)
        .order_by(
            TargetAccountEngagement.target_account_id,
            TargetAccountEngagement.persona_id,
            TargetAccountEngagement.timestamp_dt,
        )
    )
    if product_id:
        query = query.filter(TargetAccountEngagement.product_id == product_id)
    last_channels: Dict[str, str] = {}
    updated = 0
    for row in query.yield_per(200):
        key = _history_key(row)
        canonical = _canonical_channel_label(row)
        derived: Optional[str] = canonical
        fallback = last_channels.get(key)
        if canonical == "webinar_platform" and fallback:
            derived = fallback
        elif not canonical and fallback:
            derived = fallback
        if derived and derived != row.derived_channel_id:
            row.derived_channel_id = derived
            updated += 1
        elif not derived and row.derived_channel_id:
            row.derived_channel_id = None
            updated += 1
        if canonical and canonical != "webinar_platform":
            last_channels[key] = canonical
        db.add(row)
    db.commit()
    return updated
