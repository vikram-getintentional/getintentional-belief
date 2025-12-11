"""
Simple heuristic for choosing how to act on a belief transition.
"""

from __future__ import annotations

from enum import Enum


class InterventionMode(str, Enum):
    DO_NOTHING = "do_nothing"
    WAIT_EXPLOIT = "wait_exploit"
    PUSH_ASSET = "push_asset"


def choose_intervention_mode(
    *,
    lift_asset: float,
    lift_subsidy: float,
    time_to_subsidy_peak_days: float,
    max_wait_days: int = 30,
) -> InterventionMode:
    lift_asset = max(0.0, float(lift_asset))
    lift_subsidy = max(0.0, float(lift_subsidy))
    time_to_subsidy_peak_days = float(time_to_subsidy_peak_days)

    if lift_asset <= 0.0 and lift_subsidy <= 0.0:
        return InterventionMode.DO_NOTHING

    if (
        lift_subsidy > 0.0
        and lift_asset > 0.0
        and lift_subsidy >= 0.8 * lift_asset
        and 0.0 <= time_to_subsidy_peak_days <= max_wait_days
    ):
        return InterventionMode.WAIT_EXPLOIT

    if lift_subsidy > 0.0 and lift_asset <= 0.0:
        # Subsidy alone is doing the work – wait unless it is far away.
        if 0.0 <= time_to_subsidy_peak_days <= max_wait_days:
            return InterventionMode.WAIT_EXPLOIT
        return InterventionMode.DO_NOTHING

    return InterventionMode.PUSH_ASSET


__all__ = ["InterventionMode", "choose_intervention_mode"]
