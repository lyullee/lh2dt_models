"""Explicit accuracy-target definitions for forward validation.

The physical comparison layer reports an error metric.  This module keeps
the separate question of what an accuracy target means explicit.  A target
without a confirmed metric, scope, denominator, and evidence is blocked; it
is never silently converted into a pass/fail criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .validation import SeriesMetrics


_TARGET_STATUSES = {"unconfirmed", "candidate", "confirmed", "rejected"}


def _required_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class AccuracyTargetDefinition:
    """A reviewable accuracy target supplied by an external specification.

    ``target_accuracy_percent`` is expressed as ``100 - nMAPE`` only when
    the definition is confirmed.  No value is inferred from observed data.
    ``target_error_percent`` may be supplied to make that conversion
    auditable; when present it must agree with the accuracy target.
    """

    target_accuracy_percent: float | None
    metric_name: str | None
    scope: str | None
    denominator: str | None
    status: str
    target_error_percent: float | None = None
    source_file: str | None = None
    source_locator: str | None = None
    reviewer: str | None = None
    reviewed_at: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AccuracyTargetDefinition":
        if not isinstance(value, Mapping):
            raise TypeError("accuracy target definition must be a mapping")
        def _percent(raw: Any, field: str) -> float | None:
            if raw is None:
                return None
            result = float(raw)
            if not math.isfinite(result):
                raise ValueError(f"{field} must be finite")
            return result

        status = str(value.get("status", "")).strip()
        if status not in _TARGET_STATUSES:
            raise ValueError(f"unsupported accuracy target status: {status}")
        accuracy = _percent(value.get("target_accuracy_percent"), "target_accuracy_percent")
        error = _percent(value.get("target_error_percent"), "target_error_percent")
        for field, percent in (("target_accuracy_percent", accuracy), ("target_error_percent", error)):
            if percent is not None and not 0.0 <= percent <= 100.0:
                raise ValueError(f"{field} must be between 0 and 100")
        return cls(
            target_accuracy_percent=accuracy,
            metric_name=_required_text(value.get("metric_name"), "metric_name"),
            scope=_required_text(value.get("scope"), "scope"),
            denominator=_required_text(value.get("denominator"), "denominator"),
            status=status,
            target_error_percent=error,
            source_file=_required_text(value.get("source_file"), "source_file"),
            source_locator=_required_text(value.get("source_locator"), "source_locator"),
            reviewer=_required_text(value.get("reviewer"), "reviewer"),
            reviewed_at=_required_text(value.get("reviewed_at"), "reviewed_at"),
        )

    @property
    def blockers(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.status != "confirmed":
            reasons.append(f"status={self.status} is not confirmed")
        if self.target_accuracy_percent is None:
            reasons.append("target_accuracy_percent is unresolved")
        if self.metric_name is None:
            reasons.append("metric_name is unresolved")
        if self.scope is None:
            reasons.append("scope is unresolved")
        if self.denominator is None:
            reasons.append("denominator is unresolved")
        for field in ("source_file", "source_locator", "reviewer", "reviewed_at"):
            if getattr(self, field) is None:
                reasons.append(f"{field} evidence is unresolved")
        if self.target_accuracy_percent is not None and self.target_error_percent is not None:
            expected = 100.0 - self.target_accuracy_percent
            if not math.isclose(expected, self.target_error_percent, rel_tol=0.0, abs_tol=1.0e-12):
                reasons.append("target_accuracy_percent and target_error_percent are inconsistent")
        return tuple(reasons)

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def max_nmape_percent(self) -> float:
        if not self.ready:
            raise ValueError("accuracy target is not ready: " + "; ".join(self.blockers))
        assert self.target_accuracy_percent is not None
        return 100.0 - self.target_accuracy_percent


@dataclass(frozen=True)
class AccuracyVerificationResult:
    """Fail-closed result of one series accuracy assessment."""

    status: str
    accuracy_percent: float | None
    error_percent: float | None
    target_accuracy_percent: float | None
    metric_name: str | None
    blockers: tuple[str, ...]


def assess_accuracy_target(
    metrics: SeriesMetrics,
    target: AccuracyTargetDefinition,
) -> AccuracyVerificationResult:
    """Convert explicit nMAPE into accuracy and apply a confirmed target.

    The result is ``blocked`` for an unconfirmed target or missing explicit
    capacity reference.  It is never accepted merely because a target value
    happens to be present.
    """

    if not target.ready:
        return AccuracyVerificationResult(
            "blocked",
            None,
            None,
            target.target_accuracy_percent,
            target.metric_name,
            target.blockers,
        )
    nmape = metrics.normalized_mae_by_capacity_percent
    if nmape is None:
        return AccuracyVerificationResult(
            "blocked",
            None,
            None,
            target.target_accuracy_percent,
            target.metric_name,
            ("nMAPE requires an explicit capacity_reference",),
        )
    error = float(nmape)
    accuracy = 100.0 - error
    assert target.target_accuracy_percent is not None
    status = "accepted" if accuracy >= target.target_accuracy_percent else "rejected"
    return AccuracyVerificationResult(
        status,
        accuracy,
        error,
        target.target_accuracy_percent,
        target.metric_name,
        (),
    )


__all__ = ["AccuracyTargetDefinition", "AccuracyVerificationResult", "assess_accuracy_target"]
