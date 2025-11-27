# backend/utils/inference/belief_manager/journey/learn_service.py
from __future__ import annotations
from statistics import mean
from typing import Dict, Any, Iterable, List, Tuple, Optional
from collections import defaultdict, Counter
import itertools
import json

from sqlalchemy.orm import Session

from backend.database import get_db  # only needed if you later persist to DB


from backend.utils.inference.belief_manager.journey.storage import (
    load_stats,
    save_stats,
    load_weights,
    save_weights,
)
from backend.utils.inference.belief_manager.journey.learn import (
    update_edge_stats,
    recompute_weights,
    bucket_channel,
)
from backend.utils.inference.belief_manager.journey.replay import next_distribution

from backend.utils.inference.belief_manager.journey.shm_models import SHMEpisode
from backend.utils.inference.belief_manager.journey.shm_service import (
    load_episodes_for_product,
)
from backend.super_models.shm.episode import (
    ShmEpisode,
    ShmEpisodeStep,
    ShmLearningUpdate,
    ShmUpdateType,
)
from backend.utils.graph_base.network_graph import build_product_graph, get_product_id_from_subgraph
from backend.utils.graph_base.graph_utils.save_and_load_graph_as_json import save_graph_as_json
from backend.utils.graph_base.agent_graph_builder import (
    _persona,
    _job,
    _pain,
    _trigger,
    _upsert_edge,
    current_timestamp,
)
from backend.utils.inference.belief_manager.graph_diff_mapper import _L
from backend.utils.knowledge_base.arsenal.db_models import ArsenalAsset
from backend.utils.knowledge_base.arsenal.service import (
    get_or_create_channel,
    record_asset_channel_impact,
)


def _to_float(value: Optional[Any]) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_mean(values: Iterable[Any]) -> Optional[float]:
    numeric = [v for v in (_to_float(x) for x in values) if v is not None]
    return mean(numeric) if numeric else None


def update_bayesian_journey_for_account(
    product_id: str,
    account_id: str,
    episode_events: List[Dict[str, Any]],
    base_neighbors: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:

    """
    Main entry point from belief_manager.

    Returns:
        {
          "baseline_expected_next": [...],
          "current_expected_next": [...],
          "baseline_paths": ...,
          "current_paths": ...,
          "diff_summary_v3": {...},  # optional
        }
    """
    db: Session = next(get_db())
    # 1) Load existing stats & weights (priors)
    stats_before = load_stats(product_id)
    weights_before = load_weights(product_id)

    # 3) Compute baseline 'expected next' from first persona in episode
    first_persona = next((e["persona_id"] for e in episode_events if e.get("persona_id")), None)
    first_bucket = None
    if episode_events:
        first_bucket = bucket_channel(
            episode_events[0].get("channel"),
            episode_events[0].get("source"),
        )

    baseline_expected_next = []
    if first_persona:
        baseline_expected_next = next_distribution(
            first_persona, weights_before, first_bucket
        )

    # 4) Update stats from this episode
    stats_after = update_edge_stats(stats_before, episode_events)

    base_neighbors = base_neighbors or {}

    # 5) Recompute weights (posterior)
    weights_after = recompute_weights(stats_after, base_neighbors)

    # 6) Compute new expected next
    current_expected_next = []
    if first_persona:
        current_expected_next = next_distribution(
            first_persona, weights_after, first_bucket
        )

    # 7) Save
    save_stats(product_id, stats_after)
    save_weights(product_id, weights_after)

    # 8) Simple diff summary (you can feed this into your existing
    #    persona_path_inferences / graph_level_inferences builder)
    diff_summary = {
        "first_persona": first_persona,
        "baseline_expected_next": baseline_expected_next,
        "current_expected_next": current_expected_next,
    }

    return {
        "baseline_expected_next": baseline_expected_next,
        "current_expected_next": current_expected_next,
        "diff_summary_v3": diff_summary,
        # baseline_paths/current_paths left for your existing path simulator
    }

# ---------------------------
# 2.1: learn from SHM episodes
# ---------------------------

def _iter_ordered_episode_pairs(
    episodes: List[SHMEpisode],
) -> Iterable[Tuple[SHMEpisode, SHMEpisode]]:
    """
    Group by (account_id, episode_id) and yield consecutive (prev, curr) steps.
    """

    keyfunc = lambda e: (e.account_id, e.episode_id)
    for _, group in itertools.groupby(episodes, key=keyfunc):
        steps = list(group)
        steps.sort(key=lambda e: e.step_index)
        for i in range(len(steps) - 1):
            yield steps[i], steps[i + 1]


def rebuild_stats_from_shm(
    db: Session,
    *,
    product_id: str,
) -> None:
    """
    Full rebuild of edge stats & weights for a product from all SHM episodes.

    Use this whenever you:
      - change the SHM model format, OR
      - want a clean recompute of journey weights.
    """
    episodes = load_episodes_for_product(db, product_id=product_id)
    stats = {}  # start from scratch

    for prev_step, curr_step in _iter_ordered_episode_pairs(episodes):
        # Here we define what an "edge" means for learning.
        # Simplest first: persona-level transitions.
        from_persona = prev_step.persona_id
        to_persona = curr_step.persona_id

        # Optional: fold in belief_state transitions as well:
        from_state = prev_step.belief_state
        to_state = curr_step.belief_state

        # Let learner map channel + engagement -> bucket.
        bucket = bucket_channel(
            curr_step.engagement_type,
            curr_step.meta or {},
        )

        # This line depends on how you defined update_edge_stats.
        # Common pattern: update_edge_stats(stats, from_persona, to_persona, bucket)
        update_edge_stats(
            stats,
            from_persona,
            to_persona,
            from_state=from_state,
            to_state=to_state,
            bucket=bucket,
            effect_bucket=curr_step.effect_bucket,
        )

    save_stats(product_id, stats)
    weights = recompute_weights(stats)
    save_weights(product_id, weights)


def update_stats_from_new_steps(
    db: Session,
    *,
    product_id: str,
    new_steps: List[SHMEpisode],
) -> None:
    """
    Incremental learning: given a list of *new* SHM steps
    (already committed to the DB), update stats & weights.

    Assumes:
      - new_steps belong to one or more (account_id, episode_id),
      - you don't pass in old steps.
    """

    if not new_steps:
        return

    stats = load_stats(product_id) or {}

    # group by (account, episode)
    keyfunc = lambda e: (e.account_id, e.episode_id)
    for _, group in itertools.groupby(new_steps, key=keyfunc):
        steps = list(group)
        steps.sort(key=lambda e: e.step_index)
        for i in range(len(steps) - 1):
            prev_step, curr_step = steps[i], steps[i + 1]

            from_persona = prev_step.persona_id
            to_persona = curr_step.persona_id
            from_state = prev_step.belief_state
            to_state = curr_step.belief_state
            bucket = bucket_channel(
                curr_step.engagement_type,
                curr_step.meta or {},
            )

            update_edge_stats(
                stats,
                from_persona,
                to_persona,
                from_state=from_state,
                to_state=to_state,
                bucket=bucket,
                effect_bucket=curr_step.effect_bucket,
            )

    save_stats(product_id, stats)
    weights = recompute_weights(stats)
    save_weights(product_id, weights)


def summarize_global_insights(db: Session, product_id: str) -> dict:
    """
    Product-level SHM / belief insights across *all* accounts.
    Feeds the 'Global Insights' box in Engagements Setup.
    """
    print("summarizing global insights for product:", product_id)

    try:
        # --- episodes & counts ---
        episodes = (
            db.query(ShmEpisode)
            .filter(ShmEpisode.product_id == product_id)
            .all()
        )
        print("loaded", len(episodes), "episodes for product:", product_id)

        account_ids = {e.account_id for e in episodes}
        num_accounts = len(account_ids)

        # if you store num_steps on the episode, use that
        steps = (
            db.query(ShmEpisodeStep)
            .join(ShmEpisode, ShmEpisodeStep.episode_id == ShmEpisode.id)
            .filter(ShmEpisode.product_id == product_id)
            .all()
        )
        updates = (
            db.query(ShmLearningUpdate)
            .filter(ShmLearningUpdate.product_id == product_id)
            .all()
        )

        print("processing", len(updates), "learning updates for summarization")

        def _coerce_payload(raw: Any) -> Dict[str, Any]:
            if raw is None:
                return {}
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {}
            return {}

        persona_stats: Dict[str, Dict[str, Any]] = {}
        hit1_total = 0
        hit3_total = 0
        off_path_total = 0
        log_loss_total = 0.0
        log_loss_count = 0
        classification_counts: Counter[str] = Counter()
        classification_logloss_sum: Dict[str, float] = defaultdict(float)
        classification_logloss_count: Dict[str, int] = defaultdict(int)
        arsenal_stats: Dict[str, Dict[str, Any]] = {}

        try:
            product_graph = build_product_graph(product_id)
        except Exception:
            product_graph = None

        def _label(node_id: Optional[str]) -> Optional[str]:
            if not node_id:
                return None
            if product_graph is not None:
                try:
                    return _L(product_graph, node_id)
                except Exception:
                    pass
            return str(node_id) if node_id is not None else None

        def _normalize_account_meta(meta: Any) -> List[str]:
            if not meta:
                return []
            if isinstance(meta, dict):
                out: List[str] = []
                for k, v in meta.items():
                    if v is None:
                        continue
                    out.append(f"{k}:{v}")
                return out
            if isinstance(meta, list):
                out: List[str] = []
                for item in meta:
                    if not item:
                        continue
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        out.extend(_normalize_account_meta(item))
                    else:
                        out.append(str(item))
                return out
            return [str(meta)]

        account_meta_by_account: Dict[str, List[str]] = {}
        for episode in episodes:
            meta_strings = _normalize_account_meta(getattr(episode, "account_meta", None))
            if not meta_strings:
                continue
            existing = account_meta_by_account.setdefault(episode.account_id, [])
            for entry in meta_strings:
                if entry not in existing:
                    existing.append(entry)

        def _class_from_bucket(bucket: Optional[str]) -> str:
            mapping = {
                "on_path": "expected",
                "perfect_match": "expected",
                "near_path": "jump_ahead",
                "skip_hit": "jump_ahead",
                "off_path": "off_path",
                "off_path_known": "off_path",
                "out_of_graph": "off_path",
                "no_path": "no_path",
            }
            if not bucket:
                return "no_path"
            return mapping.get(bucket.lower(), bucket.lower())

        for step in steps:
            raw_bucket = step.bucket.value if hasattr(step.bucket, "value") else step.bucket
            bucket = (raw_bucket or "").lower()
            cls = _class_from_bucket(raw_bucket)
            classification_counts[cls] += 1
            persona_id = step.observed_persona_id
            if not persona_id:
                continue
            stats = persona_stats.setdefault(
                persona_id,
                {
                    "observed": 0,
                    "hit_at_1": 0,
                    "hit_at_3": 0,
                    "partial_off": 0,
                    "off_path": 0,
                    "pred_counts": Counter(),
                },
            )
            stats["observed"] += 1
            if step.hit_at_1:
                stats["hit_at_1"] += 1
                hit1_total += 1
            if step.hit_at_3:
                stats["hit_at_3"] += 1
                hit3_total += 1
            if bucket in {"off_path", "off_path_known", "out_of_graph"}:
                stats["off_path"] += 1
                off_path_total += 1
            elif bucket in {"near_path", "jump_ahead", "skip_hit"}:
                stats["partial_off"] += 1
            predicted = getattr(step, "predicted_top_persona_id", None)
            if not predicted:
                top_list = step.predicted_topK or []
                if isinstance(top_list, list) and top_list:
                    head = top_list[0]
                    if isinstance(head, dict):
                        predicted = (
                            head.get("persona")
                            or head.get("persona_id")
                            or head.get("id")
                        )
                    else:
                        predicted = str(head)
            if predicted:
                stats["pred_counts"][predicted] += 1

            metrics_payload = step.metrics or {}
            if isinstance(metrics_payload, dict):
                error_blob = metrics_payload.get("error") or {}
                log_loss = error_blob.get("log_loss")
                if isinstance(log_loss, (float, int)):
                    log_loss_total += float(log_loss)
                    log_loss_count += 1
                    classification_logloss_sum[cls] += float(log_loss)
                    classification_logloss_count[cls] += 1
                engagement_meta = metrics_payload.get("engagement") or {}
                asset_id = (
                    metrics_payload.get("asset_id")
                    or engagement_meta.get("asset_id")
                    or engagement_meta.get("id")
                )
                if asset_id:
                    asset_id = str(asset_id)
                    stat = arsenal_stats.setdefault(
                        asset_id,
                        {
                            "asset_id": asset_id,
                            "label": engagement_meta.get("label")
                            or engagement_meta.get("raw_activity")
                            or asset_id,
                            "channels": set(),
                            "persona_ids": set(),
                            "persona_labels": set(),
                            "account_meta": set(),
                            "delta_samples": [],
                            "total_delta": 0.0,
                            "conf_values": [],
                            "count": 0,
                        },
                    )
                    stat["count"] += 1
                    channel_val = (
                        engagement_meta.get("channel")
                        or metrics_payload.get("channel")
                        or engagement_meta.get("source")
                    )
                    if channel_val:
                        stat["channels"].add(str(channel_val))
                    if persona_id:
                        stat["persona_ids"].add(str(persona_id))
                        persona_label = _label(persona_id)
                        stat["persona_labels"].add(
                            persona_label or str(persona_id)
                        )
                    delta_total_step = 0.0
                    edge_evidence = metrics_payload.get("edge_evidence") or []
                    if isinstance(edge_evidence, list):
                        for ev in edge_evidence:
                            if not isinstance(ev, dict):
                                continue
                            delta_val = ev.get("delta_log_prob")
                            if delta_val is None:
                                delta_val = ev.get("deltaLogProb")
                            if isinstance(delta_val, (int, float)):
                                delta_total_step += float(delta_val)
                    stat["total_delta"] += delta_total_step
                    stat["delta_samples"].append(delta_total_step)
                    conf_val: Optional[float] = None
                    if isinstance(log_loss, (float, int)):
                        conf_val = 1.0 / (1.0 + max(float(log_loss), 0.0))
                    elif step.hit_at_1 is not None:
                        conf_val = 1.0 if step.hit_at_1 else 0.5
                    elif step.hit_at_3:
                        conf_val = 0.75
                    if conf_val is not None:
                        stat["conf_values"].append(conf_val)
                    meta_payload = (
                        metrics_payload.get("account_meta")
                        or (step.episode.account_meta if getattr(step, "episode", None) else None)
                    )
                    meta_strings = _normalize_account_meta(meta_payload)
                    if not meta_strings and getattr(step, "episode", None):
                        meta_strings = account_meta_by_account.get(step.episode.account_id, [])
                    for entry in meta_strings:
                        stat["account_meta"].add(entry)

        persona_updates: Dict[str, Dict[str, Any]] = {}
        edge_updates: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for update in updates:
            payload = _coerce_payload(update.payload)
            if update.node_id:
                bucket = persona_updates.setdefault(
                    update.node_id,
                    {
                        "persona_label": None,
                        "total_seen": 0,
                        "delta_values": [],
                        "conf_values": [],
                        "field_values": defaultdict(list),
                        "reasons": set(),
                        "jobs": set(),
                        "pains_solved": set(),
                        "triggers": set(),
                        "account_meta": Counter(),
                    },
                )
                bucket["total_seen"] += 1
                label = (
                    (payload.get("labels") or {}).get("persona")
                    or payload.get("anchor_label")
                    or payload.get("anchor")
                )
                if label and not bucket["persona_label"]:
                    bucket["persona_label"] = label
                if update.delta is not None:
                    bucket["delta_values"].append(update.delta)
                if update.confidence is not None:
                    bucket["conf_values"].append(update.confidence)
                if update.field:
                    bucket["field_values"][update.field].append(
                        {
                            "band": update.band,
                            "delta": update.delta,
                            "confidence": update.confidence,
                            "reason": payload.get("reason"),
                        }
                    )
                if payload.get("reason"):
                    bucket["reasons"].add(payload["reason"])
                for key, store_key in (
                    ("jobs", "jobs"),
                    ("pains_solved", "pains_solved"),
                    ("triggers", "triggers"),
                ):
                    values = payload.get(key) or []
                    if isinstance(values, list):
                        bucket[store_key].update(str(v) for v in values)
                meta_strings = account_meta_by_account.get(update.account_id) or []
                bucket["account_meta"].update(meta_strings)

            if update.u_node_id and update.v_node_id:
                key = (update.u_node_id, update.v_node_id)
                bucket = edge_updates.setdefault(
                    key,
                    {
                        "seen": 0,
                        "band_counts": Counter(),
                        "delta_values": [],
                        "conf_values": [],
                        "reasons": set(),
                        "scope_samples": [],
                        "pair_labels": None,
                        "edge_labels": set(),
                        "edge_chain": None,
                        "account_meta": Counter(),
                    },
                )
                bucket["seen"] += 1
                if update.band:
                    bucket["band_counts"][update.band] += 1
                if update.delta is not None:
                    bucket["delta_values"].append(update.delta)
                if update.confidence is not None:
                    bucket["conf_values"].append(update.confidence)
                if payload.get("reason"):
                    bucket["reasons"].add(payload["reason"])
                labels = (payload.get("labels") or {}).get("pair")
                if labels:
                    bucket["pair_labels"] = labels
                try:
                    edge_labels = payload.get("edge_labels") or []
                    for lbl in edge_labels:
                        if isinstance(lbl, (list, tuple)) and len(lbl) == 3:
                            bucket["edge_labels"].add(tuple(str(x) for x in lbl))
                except Exception:
                    pass
                edges = payload.get("edges") or []
                if isinstance(edges, list) and edges:
                    bucket["scope_samples"].append(1.0 / max(len(edges), 1))
                    if bucket["edge_chain"] is None:
                        bucket["edge_chain"] = [
                            list(edge) if isinstance(edge, tuple) else edge
                            for edge in edges
                        ]
                else:
                    bucket["scope_samples"].append(1.0)
                meta_strings = account_meta_by_account.get(update.account_id) or []
                bucket["account_meta"].update(meta_strings)

        def _summarize_field(values: List[Dict[str, Any]]) -> Dict[str, Any]:
            band_counts = Counter(
                v.get("band") for v in values if v.get("band")
            )
            top_band = band_counts.most_common(1)[0][0] if band_counts else None
            return {
                "field": values[0].get("field") if values else None,
                "band": top_band,
                "avg_delta": _safe_mean(v.get("delta") for v in values),
                "avg_confidence": _safe_mean(v.get("confidence") for v in values),
                "reason": next(
                    (v.get("reason") for v in values if v.get("reason")),
                    None,
                ),
            }

        def _edge_chain_metrics(edges_payload: Any) -> Tuple[Optional[float], Optional[float]]:
            if not edges_payload:
                return (None, None)
            if not product_graph:
                return (None, None)
            likelihood_product: Optional[float] = None
            relevance_product: Optional[float] = None
            for raw_edge in edges_payload:
                if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) < 2:
                    continue
                u_id, v_id = raw_edge[0], raw_edge[1]
                if not product_graph.has_edge(u_id, v_id):
                    continue
                edge_data = product_graph[u_id][v_id]
                lk = _to_float(edge_data.get("likelihood"))
                rel = _to_float(edge_data.get("relevance"))
                if lk is not None:
                    likelihood_product = (likelihood_product or 1.0) * lk
                if rel is not None:
                    relevance_product = (relevance_product or 1.0) * rel
            return (likelihood_product, relevance_product)

        persona_recommendations = []
        for persona_id, data in persona_updates.items():
            stats = persona_stats.get(
                persona_id,
                {
                    "observed": 0,
                    "hit_at_1": 0,
                    "pred_counts": Counter(),
                    "off_path": 0,
                    "partial_off": 0,
                },
            )
            observed = stats.get("observed", 0)
            hit_rate = (
                stats.get("hit_at_1", 0) / observed if observed else None
            )
            top_pred = None
            top_pred_rate = None
            for candidate, _ in stats.get("pred_counts", Counter()).most_common():
                top_pred = candidate
                total_preds = sum(stats.get("pred_counts", Counter()).values()) or 0
                if total_preds:
                    top_pred_rate = stats["pred_counts"].get(candidate, 0) / total_preds
                break

            field_summaries = []
            for field_name, values in data["field_values"].items():
                enriched = [dict(v, field=field_name) for v in values]
                field_summaries.append(_summarize_field(enriched))

            avg_delta = _safe_mean(data["delta_values"]) or 0.0
            avg_confidence = _safe_mean(data["conf_values"])
            predicted_boost_pct = avg_delta * 100.0
            account_meta_top = [
                meta for meta, _ in data["account_meta"].most_common(6)
            ]
            persona_recommendations.append(
                {
                    "persona_id": persona_id,
                    "persona_label": data["persona_label"] or _label(persona_id),
                    "observed_events": observed,
                    "seen": observed,
                    "recommendation_events": data["total_seen"],
                    "current_best_fit": _label(top_pred) if top_pred else _label(persona_id),
                    "current_fitness": hit_rate if hit_rate is not None else top_pred_rate,
                    "avg_confidence": avg_confidence,
                    "avg_delta": avg_delta,
                    "predicted_boost_pct": predicted_boost_pct,
                    "reasons": sorted(data["reasons"]),
                    "field_summaries": [
                        fs for fs in field_summaries if fs.get("avg_delta") is not None
                    ],
                    "jobs": sorted(data["jobs"]),
                    "pains": sorted(data["pains_solved"]),
                    "triggers": sorted(data["triggers"]),
                    "account_meta": account_meta_top,
                }
            )

        edge_recommendations = []
        for (u_id, v_id), data in edge_updates.items():
            avg_delta = _safe_mean(data["delta_values"]) or 0.0
            avg_confidence = _safe_mean(data["conf_values"])
            current_relevance = None
            current_likelihood = None
            recommended_likelihood = None
            recommended_relevance = None
            if product_graph is not None and product_graph.has_edge(u_id, v_id):
                edge_data = product_graph[u_id][v_id]
                current_relevance = _to_float(edge_data.get("relevance"))
                current_likelihood = _to_float(edge_data.get("likelihood"))
                recommended_relevance = current_relevance

            chain_edges = data.get("edge_chain")
            chain_likelihood, chain_relevance = _edge_chain_metrics(chain_edges)

            if chain_likelihood is not None:
                current_likelihood = chain_likelihood
            if chain_relevance is not None:
                current_relevance = chain_relevance
                recommended_relevance = chain_relevance

            if current_likelihood is not None:
                proposed = current_likelihood + avg_delta
                recommended_likelihood = max(0.0, min(1.0, proposed))

            account_meta_top = [
                meta for meta, _ in data["account_meta"].most_common(6)
            ]

            edge_recommendations.append(
                {
                    "source_id": u_id,
                    "target_id": v_id,
                    "pair_labels": data.get("pair_labels") or [_label(u_id), _label(v_id)],
                    "band": data["band_counts"].most_common(1)[0][0]
                    if data["band_counts"]
                    else None,
                    "seen": data["seen"],
                    "avg_confidence": avg_confidence,
                    "avg_delta": avg_delta,
                    "predicted_boost_pct": avg_delta * 100.0,
                    "scope": _safe_mean(data["scope_samples"]),
                    "reason": next(iter(data["reasons"]), None),
                    "reasons": sorted(data["reasons"]),
                    "edge_labels": [list(lbl) for lbl in data["edge_labels"]],
                    "current_relevance": current_relevance,
                    "current_likelihood": current_likelihood,
                    "recommended_likelihood": recommended_likelihood,
                    "recommended_relevance": recommended_relevance,
                    "recommendation_events": data["seen"],
                    "account_meta": account_meta_top,
                }
            )

        persona_recommendations.sort(
            key=lambda r: (r["predicted_boost_pct"], r["recommendation_events"]), reverse=True
        )
        edge_recommendations.sort(
            key=lambda r: (
                r.get("scope") if r.get("scope") is not None else -1,
                r.get("avg_confidence") if r.get("avg_confidence") is not None else -1,
                r.get("predicted_boost_pct") if r.get("predicted_boost_pct") is not None else float("-inf"),
            ),
            reverse=True,
        )

        print("finished summarizing global insights for product:", product_id)

        total_step_events = len(steps)
        stability_score = (
            1.0 - (off_path_total / total_step_events)
            if total_step_events
            else None
        )
        hit_at_1 = (
            hit1_total / total_step_events if total_step_events else None
        )
        hit_at_3 = (
            hit3_total / total_step_events if total_step_events else None
        )
        avg_log_loss = (
            log_loss_total / log_loss_count if log_loss_count else None
        )

        engagement_insights = []
        for cls, count in classification_counts.items():
            share = count / total_step_events if total_step_events else None
            avg_cls_log_loss = None
            if classification_logloss_count.get(cls):
                avg_cls_log_loss = (
                    classification_logloss_sum[cls] / classification_logloss_count[cls]
                )
            engagement_insights.append(
                {
                    "classification": cls,
                    "count": count,
                    "share": share,
                    "avg_log_loss": avg_cls_log_loss,
                }
            )
        engagement_insights.sort(
            key=lambda row: row.get("share") or 0, reverse=True
        )

        arsenal_impact: List[Dict[str, Any]] = []
        for asset_id, data in arsenal_stats.items():
            count = data["count"] or 1
            avg_conf = _safe_mean(data["conf_values"]) if data["conf_values"] else None
            avg_delta = _safe_mean(data["delta_samples"]) or 0.0
            arsenal_impact.append(
                {
                    "asset_id": asset_id,
                    "asset_label": data["label"],
                    "persona_ids": sorted(data["persona_ids"]),
                    "persona_labels": sorted(data["persona_labels"]),
                    "total_delta": data["total_delta"],
                    "avg_delta": avg_delta,
                    "avg_confidence": avg_conf,
                    "channels": sorted(data["channels"]),
                    "account_meta": sorted(data["account_meta"]),
                    "num_engagements": count,
                }
            )
        arsenal_impact.sort(
            key=lambda row: (
                abs(row.get("total_delta") or 0.0),
                row.get("avg_confidence") or 0.0,
            ),
            reverse=True,
        )

        # Persist aggregated arsenal evidence back into the structured tables so that
        # downstream strategy modules can query strengths without recomputing.
        channel_cache: Dict[str, Any] = {}
        for asset_id, data in arsenal_stats.items():
            asset_obj = db.get(ArsenalAsset, asset_id)
            if not asset_obj:
                continue
            personas = sorted(data.get("persona_ids") or [])
            channels = sorted(data.get("channels") or [])
            if not personas or not channels:
                continue
            avg_delta = _safe_mean(data.get("delta_samples") or []) or 0.0
            if avg_delta == 0.0 and not data.get("delta_samples"):
                continue
            evidence_payload = {
                "num_engagements": data.get("count"),
                "total_delta": data.get("total_delta"),
                "account_meta": sorted(data.get("account_meta") or []),
            }
            for channel_label in channels:
                if not channel_label:
                    continue
                channel_obj = channel_cache.get(channel_label)
                if not channel_obj:
                    channel_obj, _ = get_or_create_channel(
                        db,
                        product_id=product_id,
                        name=channel_label,
                        defaults={},
                    )
                    channel_cache[channel_label] = channel_obj
                for persona_id in personas:
                    if not persona_id:
                        continue
                    record_asset_channel_impact(
                        db,
                        product_id=product_id,
                        asset_id=asset_obj.id,
                        channel_id=channel_obj.id,
                        persona_id=persona_id,
                        belief_transition_id=None,
                        delta_strength=avg_delta,
                        evidence_payload=evidence_payload,
                    )

        db.commit()

        return {
            "meta": {
                "num_accounts": num_accounts,
                "num_engagements": total_step_events,
                "stability_score": stability_score,
                "hit_at_1": hit_at_1,
                "hit_at_3": hit_at_3,
                "log_loss": avg_log_loss,
                "num_persona_recommendations": len(persona_recommendations),
                "num_edge_recommendations": len(edge_recommendations),
            },
            "persona_recommendations": persona_recommendations,
            "edge_recommendations": edge_recommendations,
            "engagement_insights": engagement_insights,
            "arsenal_impact": arsenal_impact,
        }

    except Exception as e:
        # DO NOT kill the whole /get-persona-matches route for a summary bug
        print("Error in summarize_global_insights for product", product_id, ":", repr(e))
        return {
            "meta": {
                "num_accounts": 0,
                "num_engagements": 0,
                "num_persona_recommendations": 0,
                "num_edge_recommendations": 0,
            },
            "persona_recommendations": [],
            "edge_recommendations": [],
            "engagement_insights": [],
            "arsenal_impact": [],
        }

def _infer_node_type(node_id: Optional[str]) -> str:
    if not node_id:
        return ""
    if ":" in node_id:
        return node_id.split(":", 1)[0]
    return ""


def _ensure_placeholder_node(G, node_id: Optional[str]) -> Optional[str]:
    if not node_id:
        return None
    if node_id in G:
        return node_id
    ntype = _infer_node_type(node_id) or "unknown"
    attrs: Dict[str, Any] = {
        "id": node_id,
        "node_type": ntype,
        "type": ntype,
        "data_source": "belief_update",
        "last_updated": current_timestamp(),
    }
    if ntype == "job":
        attrs.setdefault("description", node_id.split(":", 1)[-1].replace("_", " "))
    elif ntype == "pain":
        attrs.setdefault("description", node_id.split(":", 1)[-1].replace("_", " "))
    elif ntype == "persona":
        attrs.setdefault("title", node_id.split(":", 1)[-1].replace("_", " "))
    G.add_node(node_id, **attrs)
    return node_id


def _parse_persona_label(label: Optional[str]) -> Tuple[str, str, str]:
    if not label:
        return ("Persona", "", "")
    parts = [p.strip() for p in label.split("|")]
    title = parts[0] if len(parts) > 0 else "Persona"
    department = parts[1] if len(parts) > 1 else ""
    seniority = parts[2] if len(parts) > 2 else ""
    return (title, department, seniority)


def apply_persona_recommendation_to_graph(
    product_id: str,
    recommendation: Dict[str, Any],
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    G = build_product_graph(product_id)
    existing_nodes = set(G.nodes())
    added_nodes: List[str] = []
    created_edges: List[Dict[str, Any]] = []
    updated_edges: List[Dict[str, Any]] = []
    logged_edges: set[Tuple[str, str, str]] = set()

    def _note_node(node_id: Optional[str]) -> None:
        if node_id and node_id not in existing_nodes:
            added_nodes.append(node_id)
            existing_nodes.add(node_id)

    def _note_edge(src: str, relation: str, tgt: str, existed: bool) -> None:
        key = (src, relation, tgt)
        if key in logged_edges:
            return
        logged_edges.add(key)
        attrs = G[src][tgt]
        if isinstance(attrs, dict) and "type" not in attrs and attrs:
            attrs = attrs[next(iter(attrs.keys()))]
        edge_snapshot = {
            "source": src,
            "target": tgt,
            "relation": relation,
            "likelihood": attrs.get("likelihood"),
            "relevance": attrs.get("relevance"),
            "weight": attrs.get("weight"),
        }
        if existed:
            updated_edges.append(edge_snapshot)
        else:
            created_edges.append(edge_snapshot)

    persona_label = recommendation.get("persona_label")
    title, department, seniority = _parse_persona_label(persona_label)

    persona_id = recommendation.get("persona_id")
    pre_existing_persona = persona_id if persona_id and persona_id in G else None
    persona_node_id = _persona(
        G,
        title=title,
        department=department,
        seniority=seniority,
        sample_profiles=recommendation.get("sample_profiles"),
        data_source="belief_update",
    )
    _note_node(persona_node_id)
    if persona_id and persona_id in G and persona_id != persona_node_id:
        persona_node_id = persona_id

    jobs = [j for j in recommendation.get("jobs", []) if j]
    pains = [p for p in recommendation.get("pains", []) if p]
    triggers = [t for t in recommendation.get("triggers", []) if t]

    likelihood = recommendation.get("recommended_likelihood")
    if likelihood is None:
        likelihood = recommendation.get("current_likelihood")

    relevance = recommendation.get("recommended_relevance")
    if relevance is None:
        relevance = recommendation.get("current_relevance")

    for job_id in jobs:
        ensured_job = _ensure_placeholder_node(G, job_id)
        if not ensured_job:
            continue
        _note_node(ensured_job)
        edge_existed = G.has_edge(ensured_job, persona_node_id)
        _upsert_edge(
            G,
            ensured_job,
            "performed_by",
            persona_node_id,
            weight=likelihood or 0.7,
            attrs={
                "likelihood": likelihood,
                "relevance": relevance,
                "data_source": "belief_update",
            },
            prevent_cycles=False,
            data_source="belief_update",
        )
        _note_edge(ensured_job, "performed_by", persona_node_id, edge_existed)

        for pain_id in pains:
            ensured_pain = _ensure_placeholder_node(G, pain_id)
            if not ensured_pain:
                continue
            _note_node(ensured_pain)
            edge_existed = G.has_edge(ensured_job, ensured_pain)
            _upsert_edge(
                G,
                ensured_job,
                "solves",
                ensured_pain,
                weight=1.0,
                attrs={"data_source": "belief_update"},
                prevent_cycles=False,
                data_source="belief_update",
            )
            _note_edge(ensured_job, "solves", ensured_pain, edge_existed)
            edge_existed = G.has_edge(ensured_pain, ensured_job)
            _upsert_edge(
                G,
                ensured_pain,
                "felt_in",
                ensured_job,
                weight=1.0,
                attrs={"data_source": "belief_update"},
                prevent_cycles=False,
                data_source="belief_update",
            )
            _note_edge(ensured_pain, "felt_in", ensured_job, edge_existed)
            for trigger_id in triggers:
                ensured_trigger = _ensure_placeholder_node(G, trigger_id)
                if not ensured_trigger:
                    continue
                _note_node(ensured_trigger)
                edge_existed = G.has_edge(ensured_pain, ensured_trigger)
                _upsert_edge(
                    G,
                    ensured_pain,
                    "triggered_by",
                    ensured_trigger,
                    weight=1.0,
                    attrs={"data_source": "belief_update"},
                    prevent_cycles=False,
                    data_source="belief_update",
                )
                _note_edge(ensured_pain, "triggered_by", ensured_trigger, edge_existed)

    pid = get_product_id_from_subgraph(G)
    if pid:
        save_graph_as_json(G, pid)

    print(
        "[apply_persona_recommendation]",
        product_id,
        "nodes_added:",
        added_nodes,
        "edges_created:",
        created_edges,
        "edges_updated:",
        updated_edges,
    )

    deleted_updates = 0
    if db is not None:
        persona_candidates = {persona_node_id}
        if pre_existing_persona:
            persona_candidates.add(pre_existing_persona)
        deleted_updates = (
            db.query(ShmLearningUpdate)
            .filter(
                ShmLearningUpdate.product_id == product_id,
                ShmLearningUpdate.node_id.in_(list(persona_candidates)),
            )
            .delete(synchronize_session=False)
        )

    return {
        "persona_id": persona_node_id,
        "jobs_connected": jobs,
        "pains_connected": pains,
        "triggers_connected": triggers,
        "nodes_added": added_nodes,
        "edges_created": created_edges,
        "edges_updated": updated_edges,
        "learning_updates_deleted": deleted_updates,
    }


def apply_edge_recommendation_to_graph(
    product_id: str,
    recommendation: Dict[str, Any],
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    G = build_product_graph(product_id)
    existing_nodes = set(G.nodes())
    added_nodes: List[str] = []
    created_edges: List[Dict[str, Any]] = []
    updated_edges: List[Dict[str, Any]] = []
    logged_edges: set[Tuple[str, str, str]] = set()

    def _note_node(node_id: Optional[str]) -> None:
        if node_id and node_id not in existing_nodes:
            added_nodes.append(node_id)
            existing_nodes.add(node_id)

    def _note_edge(src: str, relation: str, tgt: str, existed: bool) -> None:
        key = (src, relation, tgt)
        if key in logged_edges:
            return
        logged_edges.add(key)
        attrs = G[src][tgt]
        if isinstance(attrs, dict) and "type" not in attrs and attrs:
            attrs = attrs[next(iter(attrs.keys()))]
        edge_snapshot = {
            "source": src,
            "target": tgt,
            "relation": relation,
            "likelihood": attrs.get("likelihood"),
            "relevance": attrs.get("relevance"),
            "weight": attrs.get("weight"),
        }
        if existed:
            updated_edges.append(edge_snapshot)
        else:
            created_edges.append(edge_snapshot)

    source_id = recommendation.get("source_id")
    target_id = recommendation.get("target_id")
    _ensure_placeholder_node(G, source_id)
    _ensure_placeholder_node(G, target_id)
    _note_node(source_id)
    _note_node(target_id)

    relation = None
    if source_id and target_id and G.has_edge(source_id, target_id):
        relation = G[source_id][target_id].get("type")
    if not relation:
        labels = recommendation.get("edge_labels") or []
        if labels and isinstance(labels, list) and len(labels[0]) >= 3:
            relation = labels[0][2]
        else:
            relation = "neighborhood"

    recommended_likelihood = recommendation.get("recommended_likelihood")
    recommended_relevance = recommendation.get("recommended_relevance")

    attrs: Dict[str, Any] = {"data_source": "belief_update"}
    if recommended_likelihood is not None:
        attrs["likelihood"] = recommended_likelihood
    if recommended_relevance is not None:
        attrs["relevance"] = recommended_relevance

    current_edge = (
        G[source_id][target_id]
        if source_id and target_id and G.has_edge(source_id, target_id)
        else {}
    )
    weight = recommended_likelihood
    if weight is None:
        weight = current_edge.get("likelihood") or current_edge.get("weight") or 0.5

    existed = bool(source_id and target_id and G.has_edge(source_id, target_id))
    _upsert_edge(
        G,
        source_id,
        relation,
        target_id,
        weight=weight,
        attrs=attrs,
        prevent_cycles=False,
        data_source="belief_update",
    )
    if source_id and target_id:
        _note_edge(source_id, relation, target_id, existed)

    pid = get_product_id_from_subgraph(G)
    if pid:
        save_graph_as_json(G, pid)

    deleted_updates = 0
    if db is not None and source_id and target_id:
        deleted_updates = (
            db.query(ShmLearningUpdate)
            .filter(
                ShmLearningUpdate.product_id == product_id,
                ShmLearningUpdate.u_node_id == source_id,
                ShmLearningUpdate.v_node_id == target_id,
            )
            .delete(synchronize_session=False)
        )

    print(
        "[apply_edge_recommendation]",
        product_id,
        "nodes_added:",
        added_nodes,
        "edges_created:",
        created_edges,
        "edges_updated:",
        updated_edges,
    )

    return {
        "source_id": source_id,
        "target_id": target_id,
        "relation": relation,
        "likelihood": recommended_likelihood,
        "relevance": recommended_relevance,
        "nodes_added": added_nodes,
        "edges_created": created_edges,
        "edges_updated": updated_edges,
        "learning_updates_deleted": deleted_updates,
    }


def record_learning_updates_for_account(
    db: Session,
    *,
    product_id: str,
    account_id: str,
    incremental_learnings: Dict[str, Any],
) -> None:
    """
    Persist per-account learning hints into ShmLearningUpdate so that
    summarize_global_insights() can aggregate them across accounts.

    Expects the shape from incremental_learnings_from_thesis(thesis):

      {
        "persona_path_inferences": [...],
        "graph_level_inferences": [...],
        "neighborhoods": {
            "upstream": [...],
            "handoff": [...],
            "downstream": [...],
        },
      }
    """
    print("starting to record learning updates for account:", account_id)
    persona_path = incremental_learnings.get("persona_path_inferences") or []
    graph_level = incremental_learnings.get("graph_level_inferences") or []
    nbs = incremental_learnings.get("neighborhoods") or {}
    upstream_nbs = nbs.get("upstream") or []
    handoff_nbs = nbs.get("handoff") or []
    downstream_nbs = nbs.get("downstream") or []

    # ---- 1) Persona-path inferences  (node-level) ----
    for row in persona_path:
        pid = row.get("persona_id")
        if not pid:
            continue

        delta = row.get("delta")
        explanation = row.get("explanation")

        db.add(
            ShmLearningUpdate(
                product_id=product_id,
                account_id=account_id,
                node_id=pid,
                field="starter_likelihood",   # or "perceptibility" – your call
                band="persona_path",
                update_type=ShmUpdateType.path_delta,
                confidence=delta,
                delta=delta,
                payload={
                    **row,
                    "reason": explanation,
                    "labels": {
                        "persona": row.get("persona_label"),
                    },
                },
            )
        )

    # ---- 2) Graph-level inferences (persona-centric node theses) ----
    for row in graph_level:
        pid = row.get("persona_id")
        if not pid:
            continue

        thesis = row.get("thesis")
        db.add(
            ShmLearningUpdate(
                product_id=product_id,
                account_id=account_id,
                node_id=pid,
                field="graph_thesis",
                band="upstream",
                update_type=ShmUpdateType.graph_level_thesis,
                confidence=row.get("delta"),
                delta=row.get("delta"),
                payload={
                    **row,
                    "reason": thesis,
                    "labels": {
                        "persona": row.get("persona_label"),
                        "jobs": row.get("jobs"),
                        "pains_solved": row.get("pains_solved"),
                        "triggers": row.get("triggers"),
                    },
                },
            )
        )

    # ---- 3) Neighborhoods: upstream = node-ish, handoff = edge-ish ----

    # 3a) upstream neighborhoods → node-level updates
    for nb in upstream_nbs:
        anchor = nb.get("anchor")
        if not isinstance(anchor, str):
            continue

        db.add(
            ShmLearningUpdate(
                product_id=product_id,
                account_id=account_id,
                node_id=anchor,
                field="motif",
                band=nb.get("band") or "upstream",
                update_type=ShmUpdateType.neighborhood,
                confidence=nb.get("delta_sum"),
                delta=nb.get("delta_sum"),
                payload={
                    **nb,
                    "labels": {
                        "anchor": nb.get("anchor_label"),
                    },
                    "reason": nb.get("rationale"),
                },
            )
        )

    # 3b) handoff neighborhoods → edge-level updates
    for nb in handoff_nbs:
        anchor = nb.get("anchor") or []
        if not (isinstance(anchor, (list, tuple)) and len(anchor) == 2):
            continue
        u, v = anchor

        db.add(
            ShmLearningUpdate(
                product_id=product_id,
                account_id=account_id,
                u_node_id=u,
                v_node_id=v,
                band=nb.get("band") or "handoff",
                update_type=ShmUpdateType.handoff_neighborhood,
                confidence=nb.get("delta_sum"),
                delta=nb.get("delta_sum"),
                payload={
                    **nb,
                    "labels": {
                        "pair": nb.get("anchor_labels"),
                    },
                    "reason": nb.get("rationale"),
                },
            )
        )

    # 3c) downstream neighborhoods – treat similar to upstream for now
    for nb in downstream_nbs:
        anchor = nb.get("anchor")
        if isinstance(anchor, str):
            db.add(
                ShmLearningUpdate(
                    product_id=product_id,
                    account_id=account_id,
                    node_id=anchor,
                    field="downstream",
                    band=nb.get("band") or "downstream",
                    update_type="neighborhood",
                    confidence=nb.get("delta_sum"),
                    delta=nb.get("delta_sum"),
                    payload={
                        **nb,
                        "labels": {
                            "anchor": nb.get("anchor_label"),
                        },
                        "reason": nb.get("rationale"),
                    },
                )
            )
    print("finished recording learning updates for account:", account_id)

    # NOTE: commit/rollback is done by caller

    account_adjustment = incremental_learnings.get("account_adjustment") or {}
    edges_adj = account_adjustment.get("edges") or []
    nodes_adj = account_adjustment.get("nodes") or []

    for row in edges_adj:
        src = row.get("source_id")
        dst = row.get("target_id")
        if not src or not dst:
            continue
        delta = row.get("delta") or 0.0
        update_type = (
            ShmUpdateType.structure_strengthen_edge
            if delta >= 0
            else ShmUpdateType.structure_weaken_edge
        )
        db.add(
            ShmLearningUpdate(
                product_id=product_id,
                account_id=account_id,
                u_node_id=str(src),
                v_node_id=str(dst),
                band=row.get("diagnostics", {}).get("band") or account_adjustment.get("band"),
                update_type=update_type,
                confidence=row.get("confidence"),
                delta=delta,
                payload=row,
            )
        )

    for row in nodes_adj:
        nid = row.get("node_id")
        if not nid:
            continue
        delta = row.get("delta") or 0.0
        db.add(
            ShmLearningUpdate(
                product_id=product_id,
                account_id=account_id,
                node_id=str(nid),
                field=row.get("field"),
                band=row.get("band"),
                update_type=ShmUpdateType.graph_level_thesis,
                confidence=row.get("confidence"),
                delta=delta,
                payload=row,
            )
        )
