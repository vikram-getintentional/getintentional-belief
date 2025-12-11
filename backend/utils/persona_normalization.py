"""
Utilities for normalizing persona candidate labels extracted from engagements.

We want a deterministic representation that combines title, department, and
seniority so downstream aggregation can treat repeated actors consistently.
"""

from __future__ import annotations

from typing import Optional


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = " ".join(str(value).strip().split())
    return stripped or None


def normalize_persona_label(
    title: Optional[str],
    department: Optional[str] = None,
    seniority: Optional[str] = None,
) -> Optional[str]:
    """
    Normalize the observed actor metadata into a canonical, lowercase label.

    Examples:
        ("Director of Data Engineering", "Data Platform", "Director")
            -> "director of data engineering · data platform · director"
        ("VP Sales", None, None)
            -> "vp sales"

    Returns None if all parts are missing/empty.
    """

    parts = [_clean(part) for part in (title, department, seniority)]
    normalized_parts = [part.lower() for part in parts if part]
    if not normalized_parts:
        return None
    return " · ".join(normalized_parts)


__all__ = ["normalize_persona_label"]
