from __future__ import annotations

from collections import Counter, defaultdict, deque
from statistics import mean
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from sqlalchemy.orm import Session

from backend.super_models.shm.episode import ShmEpisode, ShmEpisodeStep, EpisodeOutcome
from backend.utils.inference.belief_manager.journey.storage import (
    save_global_thesis,
    load_global_thesis,
)


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return mean(vals) if vals else None


def _convert_outcome(outcome: Optional[EpisodeOutcome]) -> str:
    if outcome is None:
        return "unknown"
    try:
        return outcome.value
    except AttributeError:
        return str(outcome)


def _topk_personas(step: ShmEpisodeStep) -> List[str]:
    if step.predicted_topK is None:
        return []
    if isinstance(step.predicted_topK, list):
        out: List[str] = []
        for entry in step.predicted_topK:
            if isinstance(entry, dict):
                pid = entry.get("persona") or entry.get("id") or entry.get("persona_id")
                if pid:
                    out.append(str(pid))
            else:
                out.append(str(entry))
        return out
    return []


def _get_error_metrics(step: ShmEpisodeStep) -> Dict[str, Any]:
    metrics = step.metrics or {}
    if isinstance(metrics, dict):
        error = metrics.get("error")
        if isinstance(error, dict):
            return error
    return {}


def rebuild_global_thesis(
    db: Session,
    *,
    product_id: str,
    window: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Aggregate SHM episodes to produce a global Bayesian journey thesis.
    """
    episodes: List[ShmEpisode] = (
        db.query(ShmEpisode)
        .filter(ShmEpisode.product_id == product_id)
        .order_by(ShmEpisode.started_at.asc())
        .all()
    )

    transition_stats: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "win": 0, "loss": 0, "band": Counter()}
    )
    persona_counts: Counter = Counter()
    log_losses: List[float] = []
    brier_scores: List[float] = []
    log_likelihoods: List[float] = []
    top1_hits: List[int] = []
    top3_hits: List[int] = []

    window_log_loss: Deque[float] = deque(maxlen=window or 0)
    window_brier: Deque[float] = deque(maxlen=window or 0)
    window_ll: Deque[float] = deque(maxlen=window or 0)

    total_steps = 0

    for episode in episodes:
        outcome = _convert_outcome(episode.outcome)
        is_win = outcome == EpisodeOutcome.won.value
        is_loss = outcome == EpisodeOutcome.lost.value

        steps = sorted(episode.steps, key=lambda s: s.t_index)
        prev_persona: Optional[str] = None
        for step in steps:
            total_steps += 1
            persona = step.observed_persona_id
            if persona:
                persona_counts[persona] += 1
            metrics_error = _get_error_metrics(step)

            log_loss = metrics_error.get("log_loss")
            if isinstance(log_loss, (int, float)):
                log_losses.append(float(log_loss))
                if window:
                    window_log_loss.append(float(log_loss))

            brier = metrics_error.get("brier_score")
            if isinstance(brier, (int, float)):
                brier_scores.append(float(brier))
                if window:
                    window_brier.append(float(brier))

            log_like = metrics_error.get("log_likelihood")
            if isinstance(log_like, (int, float)):
                log_likelihoods.append(float(log_like))
                if window:
                    window_ll.append(float(log_like))

            if step.hit_at_1 is not None:
                top1_hits.append(1 if step.hit_at_1 else 0)
            if step.hit_at_3 is not None:
                top3_hits.append(1 if step.hit_at_3 else 0)

            if prev_persona and persona:
                stat = transition_stats[(prev_persona, persona)]
                stat["total"] += 1
                if is_win:
                    stat["win"] += 1
                elif is_loss:
                    stat["loss"] += 1

                topk = _topk_personas(step)
                if topk:
                    stat["band"][tuple(topk[:3])] += 1

            if persona:
                prev_persona = persona

    global_meta = {
        "num_episodes": len(episodes),
        "num_steps": total_steps,
        "window": window,
    }

    def _rate(win: int, loss: int, total: int) -> float:
        prior = 1.0
        return (win + prior) / (total + 2 * prior)

    transitions_summary: List[Dict[str, Any]] = []
    for (src, dst), stat in sorted(
        transition_stats.items(), key=lambda kv: kv[1]["total"], reverse=True
    ):
        total = stat["total"]
        if total == 0:
            continue
        win = stat["win"]
        loss = stat["loss"]
        win_rate = _rate(win, loss, total)
        loss_rate = _rate(loss, win, total)
        transitions_summary.append(
            {
                "from": src,
                "to": dst,
                "count": total,
                "win_count": win,
                "loss_count": loss,
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "win_lift": win_rate - loss_rate,
            }
        )

    statements: List[Dict[str, Any]] = []
    for item in transitions_summary:
        if item["count"] < 3:
            continue
        lift = item["win_lift"]
        if abs(lift) < 0.05:
            continue
        polarity = "supports" if lift > 0 else "contradicts"
        statements.append(
            {
                "statement": f"{item['from']} → {item['to']} {polarity} wins by {abs(lift)*100:.1f}%",
                "evidence": {
                    "count": item["count"],
                    "win_rate": item["win_rate"],
                    "loss_rate": item["loss_rate"],
                },
            }
        )

    metrics_summary = {
        "log_loss": {
            "mean": _safe_mean(log_losses),
            "recent_mean": _safe_mean(window_log_loss) if window else None,
        },
        "brier": {
            "mean": _safe_mean(brier_scores),
            "recent_mean": _safe_mean(window_brier) if window else None,
        },
        "log_likelihood": {
            "mean": _safe_mean(log_likelihoods),
            "recent_mean": _safe_mean(window_ll) if window else None,
        },
        "hit_at_1": _safe_mean(top1_hits),
        "hit_at_3": _safe_mean(top3_hits),
    }

    thesis = {
        "meta": global_meta,
        "metrics": metrics_summary,
        "persona_counts": [{"persona": p, "count": c} for p, c in persona_counts.most_common()],
        "transitions": transitions_summary,
        "statements": statements,
    }

    previous = load_global_thesis(product_id)
    thesis["previous"] = {
        "updated_at": previous.get("updated_at"),
        "metrics": previous.get("metrics"),
    }

    save_global_thesis(product_id, thesis)
    return thesis
