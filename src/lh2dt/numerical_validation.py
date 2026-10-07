"""Numerical time-step comparisons kept separate from plant-data agreement.

Two time grids can disprove that a coarse result is resolved.  Agreement of
two grids alone does not prove convergence, and the scale supplied here is a
numerical comparison scale, never an observed equipment-capacity denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StepRefinementMetrics:
    compared_time_count: int
    coarse_max_step_s: float
    refined_max_step_s: float
    reference_scale: float
    maximum_absolute_difference: float
    endpoint_absolute_difference: float
    maximum_difference_fraction_of_scale: float
    endpoint_difference_fraction_of_scale: float


def compare_step_refinement(
    coarse_time_s: np.ndarray,
    coarse_values: np.ndarray,
    refined_time_s: np.ndarray,
    refined_values: np.ndarray,
    *,
    reference_scale: float,
) -> StepRefinementMetrics:
    """Compare the same modeled quantity on two explicitly supplied grids.

    The refined series is interpolated only at coarse timestamps.  Both runs
    must cover the same interval, and the refined maximum step must be smaller.
    No measured series, acceptance criterion, or fitted parameter is used.
    """

    if isinstance(reference_scale, bool):
        raise ValueError("reference_scale must be finite and positive")
    try:
        scale = float(reference_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference_scale must be finite and positive") from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("reference_scale must be finite and positive")

    coarse_t = np.asarray(coarse_time_s, dtype=float)
    coarse_y = np.asarray(coarse_values, dtype=float)
    refined_t = np.asarray(refined_time_s, dtype=float)
    refined_y = np.asarray(refined_values, dtype=float)
    if (
        coarse_t.ndim != 1 or refined_t.ndim != 1
        or coarse_y.shape != coarse_t.shape
        or refined_y.shape != refined_t.shape
        or len(coarse_t) < 2 or len(refined_t) < 3
    ):
        raise ValueError("time and value series must be matching one-dimensional grids")
    if any(np.any(~np.isfinite(values)) for values in (coarse_t, coarse_y, refined_t, refined_y)):
        raise ValueError("time and value series must be finite")
    coarse_steps = np.diff(coarse_t)
    refined_steps = np.diff(refined_t)
    if np.any(coarse_steps <= 0.0) or np.any(refined_steps <= 0.0):
        raise ValueError("time grids must be strictly increasing")
    if not (
        math.isclose(coarse_t[0], refined_t[0], rel_tol=0.0, abs_tol=1.0e-9)
        and math.isclose(coarse_t[-1], refined_t[-1], rel_tol=0.0, abs_tol=1.0e-9)
    ):
        raise ValueError("time grids must have the same initial and final times")
    coarse_max_step = float(np.max(coarse_steps))
    refined_max_step = float(np.max(refined_steps))
    if refined_max_step >= coarse_max_step:
        raise ValueError("refined grid must have a smaller maximum time step")

    differences = coarse_y - np.interp(coarse_t, refined_t, refined_y)
    maximum = float(np.max(np.abs(differences)))
    endpoint = float(abs(differences[-1]))
    return StepRefinementMetrics(
        compared_time_count=len(coarse_t),
        coarse_max_step_s=coarse_max_step,
        refined_max_step_s=refined_max_step,
        reference_scale=scale,
        maximum_absolute_difference=maximum,
        endpoint_absolute_difference=endpoint,
        maximum_difference_fraction_of_scale=maximum / scale,
        endpoint_difference_fraction_of_scale=endpoint / scale,
    )
