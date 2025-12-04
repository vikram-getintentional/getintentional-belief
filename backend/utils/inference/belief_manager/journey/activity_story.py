from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.utils.graph_base.network_graph import (
    _set_node_label,
    build_product_graph,
)


def _persona_label(pid: Optional[str]) -> Optional[str]:
    if not pid:
        return None
    pid_str = str(pid)
    parts = [seg.strip() for seg in pid_str.split("|") if seg and seg.strip()]
    if not parts:
        return pid_str.replace("_", " ").title()
    cleaned: List[str] = []
    for part in parts:
        tokens = part.replace("_", " ").replace("/", " ").split()
        cleaned.append(" ".join(w.capitalize() for w in tokens if w))
    return " | ".join(cleaned) if cleaned else pid_str.replace("_", " ").title()


def _as_str_list(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values or []:
        if value is None:
            continue
        out.append(str(value))
    return out


def _collect_persona_labels(step: Dict[str, Any]) -> Dict[str, str]:
    label_map: Dict[str, str] = {}

    walk_paths = step.get("walk_paths") or []
    for path in walk_paths:
        personas = _as_str_list(path.get("personas") or path.get("path") or [])
        labels = list(path.get("persona_labels") or [])
        for pid, label in zip(personas, labels):
            if label and isinstance(label, str):
                label_map.setdefault(pid, label)
            else:
                label_map.setdefault(pid, _persona_label(pid) or pid)

    predicted_topk = step.get("predicted_topK") or []
    for entry in predicted_topk:
        if not isinstance(entry, dict):
            continue
        pid = (
            entry.get("persona")
            or entry.get("id")
            or entry.get("persona_id")
            or entry.get("node")
        )
        if not pid:
            continue
        pid_str = str(pid)
        label = entry.get("persona_label") or entry.get("label")
        if label and isinstance(label, str):
            label_map.setdefault(pid_str, label)
        else:
            label_map.setdefault(pid_str, _persona_label(pid_str) or pid_str)

    observed = step.get("observed_next")
    if observed is not None:
        ob_str = str(observed)
        label_map.setdefault(ob_str, _persona_label(ob_str) or ob_str)

    for persona in _as_str_list(step.get("state_personas") or []):
        label_map.setdefault(persona, _persona_label(persona) or persona)

    return label_map


def _pick_best_path(walk_paths: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not walk_paths:
        return None

    def _score(path: Dict[str, Any]) -> float:
        if not isinstance(path, dict):
            return 0.0
        prob = path.get("probability")
        score = path.get("score")
        if isinstance(prob, (int, float)):
            return float(prob)
        if isinstance(score, (int, float)):
            return float(score)
        return 0.0

    ranked = sorted(
        (path for path in walk_paths if isinstance(path, dict)),
        key=_score,
        reverse=True,
    )
    return ranked[0] if ranked else None


def _edge_candidates(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    payloads = [
        step.get("edge_evidence"),
        step.get("metrics", {}).get("edge_evidence") if isinstance(step.get("metrics"), dict) else None,
    ]
    for payload in payloads:
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            edges.append(row)
    return edges


def _find_edge_hint(
    *,
    edges: List[Dict[str, Any]],
    src: Optional[str],
    dst: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not src or not dst:
        return None
    src_str, dst_str = str(src), str(dst)
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for edge in edges:
        raw_src = edge.get("from") or edge.get("u") or edge.get("source")
        raw_dst = edge.get("to") or edge.get("v") or edge.get("target")
        if raw_src is None or raw_dst is None:
            continue
        if str(raw_src) != src_str or str(raw_dst) != dst_str:
            continue
        delta = edge.get("delta_log_prob") or edge.get("deltaLogProb") or 0.0
        try:
            magnitude = abs(float(delta))
        except Exception:
            magnitude = 0.0
        if best is None or magnitude > best[0]:
            best = (magnitude, edge)
    return best[1] if best else None


def _sequence_confidence(path_prob: float, order: int) -> float:
    base = max(0.15, min(0.95, path_prob or 0.0))
    decay = 0.85 ** order
    return round(max(0.1, min(0.95, base * decay + 0.05)), 3)


def _compose_narrative(
    *,
    anchor_label: Optional[str],
    target_label: Optional[str],
    edge_hint: Optional[Dict[str, Any]],
    context_hint: Optional[str] = None,
) -> str:
    anchor = anchor_label or "Key persona"
    target = target_label or "the next persona"
    templates = [
        "{anchor} likely prepared {target} for the next evaluation step.",
        "{anchor} probably socialized the solution with {target} to keep momentum.",
        "{anchor} may have briefed {target} on gaps we surfaced in earlier conversations.",
        "{anchor} is inferred to have pushed context internally before {target} engaged.",
    ]
    seed = f"{anchor}|{target}|{edge_hint.get('path_probability') if edge_hint else ''}"
    idx = sum(ord(ch) for ch in seed) % len(templates)
    narrative = templates[idx].format(anchor=anchor, target=target)
    if edge_hint:
        topic = edge_hint.get("topic") or edge_hint.get("rel") or edge_hint.get("relationship")
        if topic:
            narrative += f" Focus was likely on {topic}."
    if context_hint:
        cleaned = context_hint.strip().rstrip(".")
        if cleaned:
            narrative += f" They were reacting to {cleaned}."
    return narrative


def _clean_edge_hint(
    edge: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not isinstance(edge, dict):
        return None
    out: Dict[str, Any] = {}
    for key in ("from", "to", "supports", "delta_log_prob", "path_probability", "rel"):
        if key in edge:
            out[key] = edge[key]
    if "delta_log_prob" not in out and "deltaLogProb" in edge:
        out["delta_log_prob"] = edge["deltaLogProb"]
    if "path_probability" not in out and "pathProbability" in edge:
        out["path_probability"] = edge["pathProbability"]
    return out or None


def _fallback_sequence(step: Dict[str, Any]) -> Tuple[List[str], float, str]:
    """Use expected_next / predicted_topK when walk_paths are missing."""

    expected = step.get("expected_next") or step.get("current_expected_next") or []
    fallback: List[str] = []
    for persona in expected:
        if persona is None:
            continue
        fallback.append(str(persona))
    if fallback:
        return fallback, 0.22, "Expected persona sequence before next observed step."

    predicted = step.get("predicted_topK") or []
    for entry in predicted:
        if not isinstance(entry, dict):
            continue
        pid = (
            entry.get("persona")
            or entry.get("persona_id")
            or entry.get("id")
            or entry.get("node")
        )
        if not pid:
            continue
        fallback.append(str(pid))
        if len(fallback) >= 2:
            break
    if fallback:
        prob = 0.18
        top_entry = predicted[0] if predicted else {}
        with_prob = top_entry.get("probability") or top_entry.get("score")
        try:
            if with_prob is not None:
                prob = float(with_prob)
        except Exception:
            pass
        return fallback, max(0.15, min(0.4, prob)), "Based on top predicted personas ahead of next step."

    return [], 0.0, ""


def _build_label_lookup(thesis: Dict[str, Any]) -> Callable[[Optional[str]], Optional[str]]:
    product_id = thesis.get("product_id") or thesis.get("product_lookup_id")
    cache: Dict[str, Optional[str]] = {}
    graph = None
    if product_id:
        try:
            graph = build_product_graph(product_id)
        except Exception:
            graph = None

    def resolve(pid: Optional[str]) -> Optional[str]:
        if not pid:
            return None
        pid_str = str(pid)
        if pid_str in cache:
            return cache[pid_str]
        label: Optional[str] = None
        if graph is not None and pid_str in graph:
            try:
                label = _set_node_label(graph, pid_str)
            except Exception:
                label = None
        if not label:
            if pid_str.startswith("persona:") and ":" in pid_str:
                label = pid_str.split(":", 1)[1].replace("_", " ").title()
            else:
                label = _persona_label(pid_str)
        cache[pid_str] = label
        return label

    return resolve


def infer_latent_activity(
    thesis: Dict[str, Any],
    label_lookup: Optional[Callable[[Optional[str]], Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    journey = thesis.get("journey") or {}
    steps: List[Dict[str, Any]] = journey.get("steps") or []
    if not steps:
        return []

    inferred: List[Dict[str, Any]] = []
    seen_pairs: set[Tuple[str, str, int]] = set()
    observed_personas_all = {
        str(step.get("observed_next"))
        for step in steps
        if step.get("observed_next") is not None
    }

    # Slightly relax thresholds so latent steps still surface in the timeline
    MIN_PATH_PROBABILITY = 0.12
    MIN_CONFIDENCE = 0.15
    UNOBSERVED_PATH_PROB_THRESHOLD = 0.2

    for idx, step in enumerate(steps):
        observed_persona = step.get("observed_next")
        observed_persona_str = str(observed_persona) if observed_persona else None
        label_map = _collect_persona_labels(step)
        walk_paths = step.get("walk_paths") or []
        best_path = _pick_best_path(walk_paths)

        path_personas: List[str] = []
        path_probability = 0.0
        reason = "Inferred from SHM best path prior to observed engagement."

        if best_path:
            path_personas = _as_str_list(
                best_path.get("personas") or best_path.get("path") or []
            )
            path_prob_raw = best_path.get("probability") or best_path.get("score") or 0.0
            try:
                path_probability = float(path_prob_raw)
            except Exception:
                path_probability = 0.0
        if not path_personas:
            fallback_seq, fallback_prob, fallback_reason = _fallback_sequence(step)
            path_personas = fallback_seq
            path_probability = fallback_prob
            if fallback_reason:
                reason = fallback_reason
        if not path_personas:
            continue

        # Stop collecting once we reach the observed persona (if present).
        missing_sequence: List[str] = []
        state_personas = set(_as_str_list(step.get("state_personas") or []))
        for persona in path_personas:
            if observed_persona_str and persona == observed_persona_str:
                break
            if persona in state_personas:
                continue
            missing_sequence.append(persona)
            state_personas.add(persona)

        if not missing_sequence:
            continue

        if path_probability < MIN_PATH_PROBABILITY:
            continue

        edges = _edge_candidates(step)
        graph_context = None
        graph_impact = step.get("graph_impact")
        if isinstance(graph_impact, list) and graph_impact:
            graph_context = str(graph_impact[0])

        for order, persona in enumerate(missing_sequence):
            target = (
                missing_sequence[order + 1]
                if order + 1 < len(missing_sequence)
                else observed_persona_str
            )
            pair_key = (persona, target or "", idx)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            edge_hint = _find_edge_hint(edges=edges, src=persona, dst=target)
            anchor_label = label_map.get(persona)
            if not anchor_label and label_lookup:
                anchor_label = label_lookup(persona)
            if not anchor_label:
                anchor_label = _persona_label(persona)

            target_label = label_map.get(target)
            if not target_label and target and label_lookup:
                target_label = label_lookup(target)
            if not target_label and target:
                target_label = _persona_label(target)
            if not anchor_label and persona not in observed_personas_all:
                continue
            confidence = _sequence_confidence(path_probability, order)
            if confidence < MIN_CONFIDENCE:
                continue
            if (
                persona not in observed_personas_all
                and best_path
                and path_probability < UNOBSERVED_PATH_PROB_THRESHOLD
            ):
                continue
            narrative = _compose_narrative(
                anchor_label=anchor_label,
                target_label=target_label,
                edge_hint=edge_hint,
                context_hint=graph_context,
            )

            inferred.append(
                {
                    "id": f"latent-{idx}-{order}",
                    "kind": "latent",
                    "insert_before_step_index": idx,
                    "step_t": step.get("t", idx),
                    "persona_id": persona,
                    "persona_label": anchor_label,
                    "target_persona_id": target,
                    "target_persona_label": target_label,
                    "confidence": confidence,
                    "path_probability": path_probability,
                    "narrative": narrative,
                    "reason": reason,
                    "edge_evidence": _clean_edge_hint(edge_hint),
                    "path_excerpt": path_personas,
                }
            )

    inferred.sort(
        key=lambda row: (
            row.get("insert_before_step_index", 0),
            row.get("step_t") if row.get("step_t") is not None else 0,
            row.get("confidence", 0.0) * -1.0,
        )
    )
    return inferred


def build_activity_story(thesis: Dict[str, Any]) -> List[Dict[str, Any]]:
    journey = thesis.get("journey") or {}
    steps: List[Dict[str, Any]] = journey.get("steps") or []
    if not steps:
        return []

    label_lookup = _build_label_lookup(thesis)
    latent = infer_latent_activity(thesis, label_lookup)
    latents_by_step: Dict[int, List[Dict[str, Any]]] = {}
    for row in latent:
        step_index = int(row.get("insert_before_step_index", 0))
        latents_by_step.setdefault(step_index, []).append(row)

    story: List[Dict[str, Any]] = []
    for idx, step in enumerate(steps):
        for inferred_row in latents_by_step.get(idx, []):
            persona_label = inferred_row.get("persona_label")
            persona_id = inferred_row.get("persona_id")
            if persona_label and persona_label.startswith("persona:"):
                persona_label = label_lookup(persona_id or persona_label) or persona_label
            elif not persona_label:
                persona_label = label_lookup(persona_id) or None
            if persona_label:
                inferred_row["persona_label"] = persona_label
            story.append(inferred_row)

        observed_persona = step.get("observed_next")
        observed_label = (
            step.get("observed_persona_label")
            or label_lookup(observed_persona)
            or (_persona_label(str(observed_persona)) if observed_persona else None)
        )

        story.append(
            {
                "id": f"observed-{idx}",
                "kind": "observed",
                "step_index": idx,
                "step_t": step.get("t", idx),
                "timestamp": step.get("timestamp"),
                "persona_id": str(observed_persona) if observed_persona else None,
                "persona_label": observed_label,
                "bucket": step.get("bucket"),
                "engagement_meta": step.get("engagement_meta"),
            }
        )

    return story
