# backend/utils/strategy_builder/execution_arsenal_repository.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import math

# Reuse your dev loaders verbatim
from backend.utils.graph_base.graph_data.asset_utils.save_and_load_arsenal import (
    load_assets_from_arsenal_json,
    load_channels_from_arsenal_json,
)

__all__ = ["load_assets", "load_channels"]


def _norm_asset(a: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an asset record into the minimal shape the strategy layer expects.
    Input is whatever is in assets.json (per your dev saver).
    """
    return {
        "id": a.get("id") or a.get("node_id") or a.get("name"),
        "name": a.get("name") or a.get("title") or a.get("id"),
        "format": a.get("format") or a.get("type") or "",
        "tags": list(a.get("tags") or a.get("qtags") or []),
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

    return {
        "id": c.get("id") or c.get("node_id") or c.get("name"),
        "name": c.get("name") or c.get("title") or c.get("id"),
        "type": c.get("type") or c.get("family") or "",
        "tags": list(c.get("tags") or c.get("qtags") or []),
        "reach": c.get("reach"),
        "reach_score": float(reach_score),
    }


def load_assets(product_id: str) -> List[Dict[str, Any]]:
    """
    Pull assets for a product from your dev JSON and normalize them.
    JSON shape (from your saver):
      { "<product_id>": [ {id,name,format,tags,evergreen,...}, ... ] }
    """
    raw = load_assets_from_arsenal_json(product_id) or []
    return [_norm_asset(a) for a in raw]


def load_channels(product_id: str) -> List[Dict[str, Any]]:
    """
    Pull channels for a product from your dev JSON and normalize them.
    JSON shape (from your saver):
      { "<product_id>": [ {id,name,type,tags,reach,reach_score,...}, ... ] }
    """
    raw = load_channels_from_arsenal_json(product_id) or []
    return [_norm_channel(c) for c in raw]
