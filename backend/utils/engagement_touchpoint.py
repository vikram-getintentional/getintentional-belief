from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

_TOUCHPOINT_GAP_HOURS = 4


def _normalize_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _infer_channel(event: Dict[str, Any], last_channel: Optional[str]) -> Optional[str]:
    channel = event.get("channel") or event.get("derived_channel_id")
    if channel:
        return channel
    payload = event.get("payload") or {}
    channel_label = payload.get("channel") or payload.get("channel_label")
    if channel_label:
        return channel_label
    channel_type = payload.get("channel_type")
    if channel_type:
        return channel_type
    if event.get("source") and not last_channel:
        return event.get("source")
    return last_channel


def _create_touchpoint(event: Dict[str, Any], channel: Optional[str]) -> Dict[str, Any]:
    payload = event.get("payload") or {}
    asset_label = (
        payload.get("asset_label")
        or payload.get("asset")
        or payload.get("activity_label")
        or event.get("asset_label")
    )
    return {
        "timestamp": _normalize_timestamp(event.get("timestamp")),
        "channel": channel or "Unspecified",
        "source": event.get("source") or payload.get("source") or "unknown",
        "asset": asset_label or "asset",
        "confidence": float(payload.get("confidence") or event.get("confidence") or 0.5),
        "persona_id": event.get("persona_id") or payload.get("persona_id"),
        "is_positive": bool(event.get("is_positive") or payload.get("is_positive")),
    }


def stitch_touchpoints(
    events: Iterable[Dict[str, Any]],
    gap_threshold: Optional[timedelta] = None,
) -> List[List[Dict[str, Any]]]:
    if gap_threshold is None:
        gap_threshold = timedelta(hours=_TOUCHPOINT_GAP_HOURS)
    sorted_events: List[Dict[str, Any]] = sorted(
        events,
        key=lambda ev: _normalize_timestamp(ev.get("timestamp"))
        or datetime.min.replace(tzinfo=timezone.utc),
    )
    journeys: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    last_channel: Optional[str] = None
    last_ts: Optional[datetime] = None
    for event in sorted_events:
        ts = _normalize_timestamp(event.get("timestamp"))
        if not ts:
            ts = datetime.now(timezone.utc)
        if last_ts and (ts - last_ts) > gap_threshold:
            if current:
                journeys.append(current)
            current = []
            last_channel = None
        channel = _infer_channel(event, last_channel)
        if channel:
            last_channel = channel
        touchpoint = _create_touchpoint(event, channel or last_channel)
        touchpoint["timestamp"] = ts
        current.append(touchpoint)
        last_ts = ts
    if current:
        journeys.append(current)
    return journeys


def flatten_touchpoints(journeys: Iterable[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [touch for journey in journeys for touch in journey]
