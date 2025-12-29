#!/usr/bin/env python3
"""
One-off helper: rewrite every network graph JSON by canonicalizing all persona
nodes using the job->persona map we batch built.

Run from the repo root:

    python scripts/canonicalize_personas.py

Each persona node will retain its node id, but fields like title,
department, seniority, label, and canonical_persona_id/persona_variant_id will be
aligned with the canonicalization knowledge base we already ship.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, Tuple

CANONICAL_PERSONA_MAP = Path(
    "backend/utils/knowledge_base/canonical_maps/persona_to_canonical.json"
)
GRAPH_JSON_DIR = Path("backend/utils/graph_base/graph_data/network_json")


def _parse_legacy_key(raw_key: str) -> Dict[str, str] | None:
    """The legacy map keys are str(dict) (single quotes), so parse with ast."""
    try:
        parsed = ast.literal_eval(raw_key)  # type: ignore[arg-type]
        if isinstance(parsed, dict):
            return {
                "title": (parsed.get("title") or "").strip(),
                "department": (parsed.get("department") or "").strip(),
                "seniority": (parsed.get("seniority") or "").strip(),
            }
    except Exception:
        pass
    return None


def _normalize_triplet(entry: Dict[str, str]) -> Tuple[str, str, str]:
    """Lowercase and strip a persona triplet for fuzzy matching."""
    return (
        entry.get("title", "").strip().lower(),
        entry.get("department", "").strip().lower(),
        entry.get("seniority", "").strip().lower(),
    )


def build_lookup_tables(
    canonical_persona_map: Dict[str, Dict[str, str]]
) -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[Tuple[str, str, str], Dict[str, str]],
]:
    """Return (original_map, normalized_map) for fast lookups."""
    normalized: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for raw_key, entry in canonical_persona_map.items():
        parsed = _parse_legacy_key(raw_key)
        if not parsed:
            continue
        key = _normalize_triplet(parsed)
        if key not in normalized:
            normalized[key] = entry
    return canonical_persona_map, normalized


def canonicalize_persona_node(
    node: Dict[str, str],
    canonical_map: Dict[str, Dict[str, str]],
    normalized_map: Dict[Tuple[str, str, str], Dict[str, str]],
) -> bool:
    """
    Return True if we mutated the node.
    """
    title = (node.get("title") or node.get("label") or "").strip()
    department = (node.get("department") or "").strip()
    seniority = (node.get("seniority") or "").strip()

    legacy_key = str(
        {"title": title, "department": department, "seniority": seniority}
    )
    entry = canonical_map.get(legacy_key) or normalized_map.get(
        (title.lower(), department.lower(), seniority.lower())
    )
    if not entry:
        return False

    updated = False
    for attr in ("title", "department", "seniority"):
        canonical_value = (entry.get(attr) or "").strip()
        if canonical_value and canonical_value != node.get(attr):
            node[attr] = canonical_value
            updated = True

    canonical_label = (entry.get("title") or "").strip()
    if canonical_label and canonical_label != node.get("label"):
        node["label"] = canonical_label
        updated = True

    for attr in ("canonical_persona_id", "persona_variant_id"):
        canonical_value = entry.get(attr)
        if canonical_value and canonical_value != node.get(attr):
            node[attr] = canonical_value
            updated = True

    return updated


def canonicalize_graph_files() -> None:
    if not CANONICAL_PERSONA_MAP.exists():
        raise SystemExit(
            f"Missing canonical map file: {CANONICAL_PERSONA_MAP.absolute()}"
        )
    canonical_map_raw = json.loads(CANONICAL_PERSONA_MAP.read_text())
    canonical_map, normalized_map = build_lookup_tables(canonical_map_raw)

    if not GRAPH_JSON_DIR.exists():
        raise SystemExit(f"Graph directory not found: {GRAPH_JSON_DIR.absolute()}")

    total_nodes = 0
    total_updated = 0
    touched_files = 0

    for graph_file in sorted(GRAPH_JSON_DIR.glob("*.json")):
        payload = json.loads(graph_file.read_text())
        nodes = payload.get("nodes") or []
        updated_in_file = 0
        for node in nodes:
            node_type = (node.get("node_type") or node.get("type") or "").lower()
            if node_type != "persona":
                continue
            total_nodes += 1
            if canonicalize_persona_node(node, canonical_map, normalized_map):
                updated_in_file += 1
                total_updated += 1
        if updated_in_file:
            graph_file.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"Updated {updated_in_file} persona(s) in {graph_file.name}")
            touched_files += 1

    print(
        f"Finished canonicalizing personas: "
        f"{total_updated}/{total_nodes} nodes touched across {touched_files} files."
    )


if __name__ == "__main__":
    canonicalize_graph_files()
