"""Metrics for forward-prediction comparison; no parameter fitting is included."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SeriesMetrics:
    count: int
    mean_error: float
    mean_absolute_error: float
    root_mean_squared_error: float
    maximum_absolute_error: float
    normalized_rmse: float | None
    normalized_mae_by_capacity_percent: float | None
    within_uncertainty_fraction: float | None


@dataclass(frozen=True)
class SeriesAcceptance:
    """Explicit, fixed-parameter acceptance decision for one series.

    A missing metric needed by a caller-supplied criterion is ``blocked``;
    a measured violation is ``rejected``.  The function below never creates
    a criterion or estimates a missing metric from the observation series.
    """

    status: str
    reasons: tuple[str, ...]
    evaluated_criteria: tuple[str, ...]


def evaluate_series_metrics(
    metrics: SeriesMetrics,
    *,
    max_nmape_percent: float | None = None,
    max_rmse: float | None = None,
    min_within_uncertainty_fraction: float | None = None,
) -> SeriesAcceptance:
    """Evaluate explicitly supplied comparison criteria without fitting.

    At least one criterion must be supplied.  ``max_nmape_percent`` uses the
    nMAPE value calculated with an explicit capacity reference in
    :func:`compare_series`; capacity is never inferred here.  A criterion
    whose metric was not requested or cannot be calculated blocks the decision
    instead of being treated as a pass.
    """

    criteria = {
        "max_nmape_percent": max_nmape_percent,
        "max_rmse": max_rmse,
        "min_within_uncertainty_fraction": min_within_uncertainty_fraction,
    }
    if not any(value is not None for value in criteria.values()):
        raise ValueError("at least one explicit acceptance criterion is required")
    for name, value in criteria.items():
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if name in {"max_nmape_percent", "max_rmse"} and value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        if name == "min_within_uncertainty_fraction" and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    blocked: list[str] = []
    rejected: list[str] = []
    evaluated: list[str] = []

    if max_nmape_percent is not None:
        if metrics.normalized_mae_by_capacity_percent is None:
            blocked.append("max_nmape_percent requires an explicit capacity_reference")
        else:
            evaluated.append("max_nmape_percent")
            if metrics.normalized_mae_by_capacity_percent > float(max_nmape_percent):
                rejected.append(
                    "nMAPE %.6g%% exceeds %.6g%%"
                    % (metrics.normalized_mae_by_capacity_percent, float(max_nmape_percent))
                )

    if max_rmse is not None:
        evaluated.append("max_rmse")
        if metrics.root_mean_squared_error > float(max_rmse):
            rejected.append(
                "RMSE %.6g exceeds %.6g"
                % (metrics.root_mean_squared_error, float(max_rmse))
            )

    if min_within_uncertainty_fraction is not None:
        if metrics.within_uncertainty_fraction is None:
            blocked.append("min_within_uncertainty_fraction requires observed_uncertainty")
        else:
            evaluated.append("min_within_uncertainty_fraction")
            if metrics.within_uncertainty_fraction < float(min_within_uncertainty_fraction):
                rejected.append(
                    "within-uncertainty fraction %.6g is below %.6g"
                    % (metrics.within_uncertainty_fraction, float(min_within_uncertainty_fraction))
                )

    status = "blocked" if blocked else ("rejected" if rejected else "accepted")
    return SeriesAcceptance(status, tuple(blocked + rejected), tuple(evaluated))


def compare_series(
    model_time_s: np.ndarray,
    model_values: np.ndarray,
    observed_time_s: np.ndarray,
    observed_values: np.ndarray,
    observed_uncertainty: np.ndarray | float | None = None,
    *,
    require_full_observation_coverage: bool = True,
    capacity_reference: float | None = None,
) -> SeriesMetrics:
    """Compare model predictions at observation times by linear interpolation.

    The function calculates scores only.  It deliberately exposes no optimizer
    or parameter-estimation path. By default every finite observation must be
    inside the model time range; callers may opt into partial scoring explicitly
    when that policy is documented by the surrounding validation protocol.
    """

    mt = np.asarray(model_time_s, dtype=float)
    mv = np.asarray(model_values, dtype=float)
    ot = np.asarray(observed_time_s, dtype=float)
    ov = np.asarray(observed_values, dtype=float)
    if mt.ndim != 1 or mv.shape != mt.shape or ot.ndim != 1 or ov.shape != ot.shape:
        raise ValueError("Time and value arrays must be matching one-dimensional arrays")
    if len(mt) < 2 or len(ot) < 1 or np.any(np.diff(mt) <= 0.0):
        raise ValueError("Model time must be strictly increasing with at least two points")
    if np.any(~np.isfinite(mt)) or np.any(~np.isfinite(mv)):
        raise ValueError("Model time and values must be finite")
    if np.any(~np.isfinite(ot)) or np.any(np.diff(ot) <= 0.0):
        raise ValueError(
            "Observed time must be finite and strictly increasing; no automatic "
            "sorting or timestamp repair is permitted"
        )
    if capacity_reference is not None:
        capacity_reference = float(capacity_reference)
        if not math.isfinite(capacity_reference) or capacity_reference <= 0.0:
            raise ValueError("capacity_reference must be finite and positive")
    finite_observations = np.isfinite(ot) & np.isfinite(ov)
    outside_model_range = finite_observations & ((ot < mt[0]) | (ot > mt[-1]))
    if require_full_observation_coverage and np.any(outside_model_range):
        raise ValueError(
            "finite observations extend outside the model time range; no "
            "automatic truncation is permitted"
        )
    valid = finite_observations & (ot >= mt[0]) & (ot <= mt[-1])
    if not np.any(valid):
        raise ValueError("No finite observations overlap the model time range")
    predicted = np.interp(ot[valid], mt, mv)
    error = predicted - ov[valid]
    span = float(np.ptp(ov[valid]))
    rmse = float(np.sqrt(np.mean(error**2)))
    normalized_mae_by_capacity_percent = (
        float(np.mean(np.abs(error)) / capacity_reference * 100.0)
        if capacity_reference is not None
        else None
    )
    within = None
    if observed_uncertainty is not None:
        uncertainty = np.broadcast_to(np.asarray(observed_uncertainty, dtype=float), ov.shape)[valid]
        if np.any(~np.isfinite(uncertainty)) or np.any(uncertainty <= 0.0):
            raise ValueError("Observed uncertainty must be finite and positive")
        within = float(np.mean(np.abs(error) <= uncertainty))
    return SeriesMetrics(
        count=int(np.sum(valid)),
        mean_error=float(np.mean(error)),
        mean_absolute_error=float(np.mean(np.abs(error))),
        root_mean_squared_error=rmse,
        maximum_absolute_error=float(np.max(np.abs(error))),
        normalized_rmse=rmse / span if span > 0.0 and math.isfinite(span) else None,
        normalized_mae_by_capacity_percent=normalized_mae_by_capacity_percent,
        within_uncertainty_fraction=within,
    )
