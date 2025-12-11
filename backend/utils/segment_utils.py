from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


SEGMENT_FIELD_ALIASES: Dict[str, str] = {
    "employee_bucket": "employee_range",
    "employees": "employee_range",
    "employee_count": "employee_range",
    "employee_band": "employee_range",
    "revenue_bucket": "revenue_range",
    "geo": "geography",
    "region": "geography",
    "funding": "funding_stage",
}

SEGMENT_FIELD_LABELS: Dict[str, str] = {
    "industry": "",
    "employee_range": "employees",
    "revenue_range": "revenue",
    "geography": "",
    "funding_stage": "funding",
}

SEGMENT_RECIPES: Tuple[Tuple[str, ...], ...] = (
    ("industry", "employee_range"),
    ("industry", "revenue_range"),
    ("industry", "geography"),
    ("industry", "funding_stage"),
    ("geography", "employee_range"),
    ("industry",),
)


def _clean_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def normalize_account_meta(meta: Any) -> Dict[str, str]:
    """
    Normalizes account meta dictionaries into a canonical set of fields that
    downstream segmentation helpers can reason about.
    """
    if not isinstance(meta, dict):
        return {}
    normalized: Dict[str, str] = {}
    for key, value in meta.items():
        alias = SEGMENT_FIELD_ALIASES.get(key, key)
        cleaned = _clean_value(value)
        if cleaned:
            normalized[alias] = cleaned
    return normalized


def segment_keys_from_meta(meta: Any) -> List[str]:
    normalized = normalize_account_meta(meta)
    if not normalized:
        return []
    keys: List[str] = []
    for recipe in SEGMENT_RECIPES:
        parts: List[str] = []
        for field in recipe:
            value = normalized.get(field)
            if not value:
                parts = []
                break
            parts.append(f"{field}={value}")
        if parts:
            keys.append("|".join(parts))
    # fallback to single industry if nothing matched
    if not keys and normalized.get("industry"):
        keys.append(f"industry={normalized['industry']}")
    return keys


def segment_label_from_key(key: str) -> str:
    if not key:
        return "Global"
    parts = []
    for token in key.split("|"):
        if "=" not in token:
            parts.append(token.strip().title())
            continue
        field, value = token.split("=", 1)
        label_hint = SEGMENT_FIELD_LABELS.get(field, field.replace("_", " "))
        if label_hint:
            parts.append(f"{value} {label_hint}".strip())
        else:
            parts.append(value)
    return " · ".join(part for part in parts if part)


__all__ = ["segment_keys_from_meta", "segment_label_from_key", "normalize_account_meta"]
