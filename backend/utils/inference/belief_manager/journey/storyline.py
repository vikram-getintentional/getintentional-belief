from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.utils.graph_base.network_graph import (
    _set_node_label,
    build_product_graph,
)

PHASE_DEFAULTS = ("early", "mid", "late")

_GRAPH_CACHE: Dict[str, Any] = {}


def _get_product_graph(product_id: Optional[str]):
    if not product_id:
        return None
    if product_id in _GRAPH_CACHE:
        return _GRAPH_CACHE[product_id]
    try:
        graph = build_product_graph(product_id)
    except Exception:
        graph = None
    _GRAPH_CACHE[product_id] = graph
    return graph


def _make_labeler(product_id: Optional[str]):
    graph = _get_product_graph(product_id)
    cache: Dict[str, str] = {}

    def resolve(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        text = str(value)
        if text in cache:
            return cache[text]
        label: Optional[str] = None
        if graph is not None and text in graph:
            try:
                label = _set_node_label(graph, text)
            except Exception:
                label = None
        if not label and text.startswith("persona:"):
            tail = text.split(":", 1)[1]
            label = tail.replace("_", " ").title()
        if not label:
            label = _persona_label_from_id(text)
        cache[text] = label
        return label

    return resolve

def _titleize(value: Optional[str]) -> str:
    if not value:
        return ""
    tokens = [
        token.capitalize()
        for token in str(value).replace("_", " ").replace("|", " ").split()
        if token
    ]
    return " ".join(tokens)


def _format_segments(meta: Optional[Dict[str, Any]]) -> List[str]:
    if not meta:
        return []
    parts: List[str] = []
    for key in (
        "industry",
        "revenue_range",
        "employee_range",
        "funding_stage",
        "geography",
        "competitor_used",
        "other_tech_stack",
    ):
        value = meta.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            label = ", ".join(_titleize(v) for v in value if v)
        else:
            label = _titleize(value)
        parts.append(f"{_titleize(key)}: {label}")
    return parts


def _phase_from_stage(stage_label: Optional[str]) -> str:
    stage = (stage_label or "").lower()
    if any(token in stage for token in ("problem", "pain", "aware")):
        return "early"
    if any(token in stage for token in ("resolution", "solution", "exploration")):
        return "mid"
    if stage:
        return "late"
    return "mid"


def _phase_from_bucket(bucket: Optional[str]) -> str:
    bucket_key = (bucket or "").lower()
    if bucket_key in {"on_path", "perfect_match", "near_path"}:
        return "early"
    if bucket_key in {"skip_hit", "jump_ahead"}:
        return "mid"
    if bucket_key in {"off_path", "off_path_known", "out_of_graph"}:
        return "late"
    return "mid"


def _persona_label_from_id(persona_id: Optional[str]) -> Optional[str]:
    if not persona_id:
        return None
    parts = [part.strip() for part in str(persona_id).split("|") if part.strip()]
    if not parts:
        return persona_id
    return " · ".join(_titleize(part) for part in parts)


def _meta_lines(account_name: str, account_meta: Optional[Dict[str, Any]], deal_status: Optional[str]) -> Tuple[str, str]:
    segments = _format_segments(account_meta)
    summary_lines = [account_name]
    meta_line = ""
    if segments:
        meta_line = " · ".join(segments)
    if deal_status:
        summary_lines.append(f"Status: {_titleize(deal_status)}")
    summary_text = ". ".join(line for line in summary_lines if line)
    return summary_text, meta_line


def compose_storyline(
    *,
    account_name: str,
    account_meta: Optional[Dict[str, Any]],
    deal_status: Optional[str],
    journey_steps: Iterable[Dict[str, Any]],
    activity_story: Iterable[Dict[str, Any]],
    conversion_sequence: Optional[List[Dict[str, Any]]] = None,
    zmot_event: Optional[Dict[str, Any]] = None,
    product_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a human-readable storyline for an account using journey data.
    """

    summary_text, meta_line = _meta_lines(account_name, account_meta, deal_status)

    zmot_text = None
    if zmot_event:
        zmot_text = zmot_event.get("label") or zmot_event.get("description")
    elif account_meta and account_meta.get("zmot"):
        zmot_text = account_meta.get("zmot")

    label_for = _make_labeler(product_id)

    stakeholders: Dict[str, List[str]] = {"early": [], "mid": [], "late": []}
    stakeholder_counter: Dict[str, Counter] = {
        "early": Counter(),
        "mid": Counter(),
        "late": Counter(),
    }

    if conversion_sequence:
        for entry in conversion_sequence:
            persona_descriptor = entry.get("persona_descriptor") or (
                entry.get("persona", {}) or {}
            ).get("label")
            if not persona_descriptor:
                continue
            persona_descriptor = label_for(persona_descriptor) or persona_descriptor
            phase = _phase_from_stage(
                (entry.get("belief_transition_meta") or {}).get("stage_label")
            )
            stakeholder_counter[phase][persona_descriptor] += 1

    for phase in stakeholders:
        stakeholders[phase] = [
            label for label, _ in stakeholder_counter[phase].most_common(5)
        ]

    nodes: List[Dict[str, Any]] = []
    narrative_sentences: List[str] = []

    for idx, item in enumerate(activity_story or []):
        kind = item.get("kind") or "observed"
        node_id = item.get("id") or f"story-{idx}"
        persona_label = (
            item.get("persona_label")
            or label_for(item.get("persona_id"))
            or _persona_label_from_id(item.get("persona_id"))
        )
        phase = _phase_from_bucket(item.get("bucket"))
        order_key = idx * 10
        if kind == "latent":
            text = item.get("narrative") or item.get("reason") or ""
            if not text and persona_label:
                target_persona_id = item.get("target_persona_id")
                target_label = (
                    item.get("target_persona_label")
                    or label_for(target_persona_id)
                    or _persona_label_from_id(target_persona_id)
                )
                if target_label:
                    text = (
                        f"{persona_label} likely briefed {target_label} before the next step."
                    )
                else:
                    text = (
                        f"{persona_label} probably advanced the belief state behind the scenes."
                    )
            nodes.append(
                {
                    "id": node_id,
                    "text": text,
                    "phase": phase,
                    "source": "inferred",
                    "persona": persona_label,
                    "editable": True,
                    "sequence_order": order_key,
                }
            )
            narrative_sentences.append(text.rstrip(".") + ".")
            continue

        engagement_meta = item.get("engagement_meta") or {}
        raw_text = engagement_meta.get("raw_activity")
        if not raw_text:
            channel = engagement_meta.get("channel") or engagement_meta.get("source")
            if channel:
                raw_text = f"{persona_label or 'A persona'} engaged via {channel}."
            else:
                raw_text = f"{persona_label or 'A persona'} progressed the journey."

        nodes.append(
            {
                "id": node_id,
                "text": raw_text,
                "phase": phase,
                "source": "observed",
                "persona": persona_label,
                "editable": False,
                "sequence_order": order_key,
            }
        )
        sentence = raw_text.rstrip(".") if raw_text else "Advanced the journey"
        if persona_label:
            narrative_sentences.append(f"{persona_label}: {sentence}.")
        else:
            narrative_sentences.append(sentence + ".")

    seen_predicted = set(node["id"] for node in nodes)
    if conversion_sequence:
        for entry in conversion_sequence:
            node_id = entry.get("id") or f"predicted-{entry.get('timeline_index')}"
            if node_id in seen_predicted:
                continue
            persona_descriptor = entry.get("persona_descriptor") or (
                entry.get("persona", {}) or {}
            ).get("label")
            stage_label = (entry.get("belief_transition_meta") or {}).get("stage_label")
            persona_descriptor = label_for(persona_descriptor) or persona_descriptor
            highlight = entry.get("expected_outcome_summary") or (
                entry.get("belief_transition_meta") or {}
            ).get("narrative")
            if not highlight and stage_label and persona_descriptor:
                highlight = f"{persona_descriptor} expected to drive {stage_label.lower()}."
            if not highlight:
                highlight = "Belief progression expected from this persona."
            nodes.append(
                {
                    "id": node_id,
                    "text": highlight,
                    "phase": _phase_from_stage(stage_label),
                    "source": "predicted",
                    "persona": persona_descriptor,
                    "editable": True,
                    "sequence_order": (len(activity_story or []) + len(nodes)) * 10,
                }
            )
            narrative_sentences.append(highlight.rstrip(".") + ".")

    if not any(stakeholders.values()):
        phase_counts: Dict[str, Counter] = {
            "early": Counter(),
            "mid": Counter(),
            "late": Counter(),
        }
        for node in nodes:
            persona = node.get("persona")
            phase_counts[node.get("phase") or "mid"][persona or "Unknown persona"] += 1
        for phase in stakeholders:
            stakeholders[phase] = [label for label, _ in phase_counts[phase].most_common(5)]

    for phase, values in stakeholders.items():
        stakeholders[phase] = [
            label_for(value) or value
            for value in values
            if value
        ]

    nodes.sort(key=lambda node: node.get("sequence_order", 0))
    for node in nodes:
        node.pop("sequence_order", None)

    return {
        "summary": {
            "text": summary_text,
            "segments": _format_segments(account_meta),
            "meta_line": meta_line,
            "deal_status": deal_status,
            "zmot": zmot_text,
            "needs_zmot": not bool(zmot_text),
        },
        "stakeholders": stakeholders,
        "nodes": nodes,
        "narrative": [sentence for sentence in narrative_sentences if sentence.strip()],
    }


def compose_portfolio_story(
    accounts: List[Dict[str, Any]],
    filters: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Build a portfolio-level storyline by aggregating individual account narratives.
    """
    filters = {k: v for k, v in (filters or {}).items() if v}
    summary_bits = [
        f"{len(accounts)} account{'s' if len(accounts) != 1 else ''} in focus"
    ]
    if filters:
        summary_bits.append(
            "Filters: " + ", ".join(f"{_titleize(k)}={_titleize(v)}" for k, v in filters.items())
        )

    aggregated_nodes: List[Dict[str, Any]] = []
    aggregated_stakeholders: Dict[str, Counter] = {
        "early": Counter(),
        "mid": Counter(),
        "late": Counter(),
    }

    for account in accounts:
        story = account.get("storyline") or {}
        for phase in PHASE_DEFAULTS:
            for label in story.get("stakeholders", {}).get(phase, []) or []:
                if label:
                    aggregated_stakeholders[phase][label] += 1
        for node in story.get("nodes") or []:
            aggregated_nodes.append(
                {
                    "id": f"{account.get('account_id')}-{node.get('id')}",
                    "text": f"[{account.get('account_name')}] {node.get('text')}",
                    "phase": node.get("phase"),
                    "source": node.get("source"),
                    "persona": node.get("persona"),
                    "editable": False,
                }
            )

    if not aggregated_nodes and accounts:
        representative = accounts[0].get("storyline") or {}
        aggregated_nodes = representative.get("nodes", [])

    stakeholders = {
        phase: [label for label, _ in aggregated_stakeholders[phase].most_common(5)]
        for phase in PHASE_DEFAULTS
    }

    return {
        "summary": {
            "text": ". ".join(summary_bits),
            "segments": [],
            "deal_status": None,
            "zmot": None,
        },
        "stakeholders": stakeholders,
        "nodes": aggregated_nodes[:12],
        "match_count": len(accounts),
    }


def collect_filter_options(accounts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    options: Dict[str, set] = defaultdict(set)
    for account in accounts:
        meta = account.get("meta") or {}
        for key, value in meta.items():
            if not value:
                continue
            if isinstance(value, (list, tuple)):
                for item in value:
                    if item:
                        options[key].add(str(item))
            else:
                options[key].add(str(value))
    return {key: sorted(values) for key, values in options.items()}
