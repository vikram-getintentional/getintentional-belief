# backend/utils/strategy_builder/execution_arsenal_repository.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import math

# Reuse your dev loaders verbatim
from backend.database import get_db
from backend.utils.knowledge_base.arsenal.service import (
    list_assets,
    list_channels,
)

__all__ = ["load_assets", "load_channels"]


def _norm_asset(a: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an asset record into the minimal shape the strategy layer expects.
    Input is whatever is in assets.json (per your dev saver).
    """
    format_hint = (
        a.get("format")
        or a.get("type")
        or a.get("category")
        or a.get("content_type")
        or ""
    )
    tags = list(a.get("tags") or [])
    for candidate in ("depth", "time_to_consume"):
        value = a.get(candidate)
        if value and value not in tags:
            tags.append(value)
    return {
        "id": a.get("id") or a.get("node_id") or a.get("name"),
        "name": a.get("name") or a.get("title") or a.get("id"),
        "format": format_hint,
        "tags": tags or list(a.get("qtags") or []),
        "evergreen": bool(a.get("evergreen", True)),
    }


def _heuristic_reach_score(reach: Optional[float]) -> float:
    """
    If only a raw 'reach' number exists, map it into ~[0..1] gently.
    log10 scale keeps large numbers from blowing up.
    """
    if reach is None:
        return 0.5
    try:
        r = float(reach)
    except Exception:
        return 0.5
    return max(0.0, min(1.0, math.log10(1.0 + max(0.0, r)) / 6.0))


def _norm_channel(c: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a channel record into the minimal shape the strategy layer expects.
    """
    reach_score = c.get("reach_score")
    if reach_score is None:
        reach_score = _heuristic_reach_score(c.get("reach"))

    channel_type = (
        c.get("type")
        or c.get("channel_type")
        or ""
    )
    tags = list(c.get("tags") or c.get("qtags") or [])
    delivery = c.get("delivery_mode")
    if delivery and delivery not in tags:
        tags.append(delivery)

    return {
        "id": c.get("id") or c.get("node_id") or c.get("name"),
        "name": c.get("name") or c.get("title") or c.get("id"),
        "type": channel_type or c.get("family") or "",
        "tags": tags,
        "reach": c.get("reach"),
        "reach_score": float(reach_score),
    }


def load_assets(product_id: str) -> List[Dict[str, Any]]:
    """
    Pull assets for a product from your dev JSON and normalize them.
    JSON shape (from your saver):
      { "<product_id>": [ {id,name,format,tags,evergreen,...}, ... ] }
    """
    db = next(get_db())
    try:
        raw = list_assets(db, product_id=product_id, include_impacts=False)
        return [_norm_asset(a) for a in raw]
    finally:
        db.close()


def load_channels(product_id: str) -> List[Dict[str, Any]]:
    """
    Pull channels for a product from your dev JSON and normalize them.
    JSON shape (from your saver):
      { "<product_id>": [ {id,name,type,tags,reach,reach_score,...}, ... ] }
    """
    db = next(get_db())
    try:
        raw = list_channels(db, product_id=product_id, include_impacts=False)
        return [_norm_channel(c) for c in raw]
    finally:
        db.close()
