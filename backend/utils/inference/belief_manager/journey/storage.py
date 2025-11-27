# backend/utils/inference/belief_manager/journey/storage.py

from __future__ import annotations
from typing import Dict, Any
import json
from pathlib import Path
from datetime import datetime

# Root: backend/utils/inference/belief_manager/journey/
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _stats_path(product_id: str) -> Path:
    return DATA_DIR / f"{product_id}_journey_stats.json"

def _weights_path(product_id: str) -> Path:
    return DATA_DIR / f"{product_id}_journey_weights.json"

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ------------ stats --------------

def _empty_stats() -> Dict[str, Any]:
    return {
        "version": 1,
        "updated_at": _now_iso(),
        # transition[from_persona][bucket][to_persona] = count
        "transition": {},
        # emission[persona][bucket] = count
        "emission": {},
    }

def load_stats(product_id: str) -> Dict[str, Any]:
    path = _stats_path(product_id)
    if not path.exists():
        return _empty_stats()
    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception:
        # corrupt -> reset
        data = _empty_stats()
    return data

def save_stats(product_id: str, stats: Dict[str, Any]) -> None:
    stats["updated_at"] = _now_iso()
    path = _stats_path(product_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(stats, f, indent=2)


# ------------ weights ------------

def _empty_weights() -> Dict[str, Any]:
    return {
        "version": 1,
        "updated_at": _now_iso(),
        # transition[from_persona][bucket][to_persona] = prob
        "transition": {},
    }

def load_weights(product_id: str) -> Dict[str, Any]:
    path = _weights_path(product_id)
    if not path.exists():
        return _empty_weights()
    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception:
        data = _empty_weights()
    return data

def save_weights(product_id: str, weights: Dict[str, Any]) -> None:
    weights["updated_at"] = _now_iso()
    path = _weights_path(product_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(weights, f, indent=2)


# ------------ global thesis ------------

def _global_thesis_path(product_id: str) -> Path:
    return DATA_DIR / f"{product_id}_global_thesis.json"


def load_global_thesis(product_id: str) -> Dict[str, Any]:
    path = _global_thesis_path(product_id)
    if not path.exists():
        return {
            "version": 1,
            "updated_at": _now_iso(),
            "meta": {},
            "metrics": {},
            "transitions": [],
            "statements": [],
        }
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return {
            "version": 1,
            "updated_at": _now_iso(),
            "meta": {},
            "metrics": {},
            "transitions": [],
            "statements": [],
        }


def save_global_thesis(product_id: str, thesis: Dict[str, Any]) -> None:
    thesis = dict(thesis)
    thesis["updated_at"] = _now_iso()
    thesis.setdefault("version", 1)
    path = _global_thesis_path(product_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(thesis, f, indent=2)
