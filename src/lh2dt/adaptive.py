"""Deterministic adaptive-step policy and diagnostics.

The policy changes only the integration step.  It does not alter physical
parameters, feed observations back into a state, or estimate a heat-transfer
coefficient.  Event times are supplied by the caller so valve and command
transitions can be hit exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


@dataclass(frozen=True)
class AdaptiveStepPolicy:
    minimum_step_s: float = 0.1
    maximum_step_s: float = 300.0
    max_relative_pressure_change: float = 0.01
    max_temperature_change_K: float = 0.5
    growth_factor: float = 1.5
    shrink_factor: float = 0.5
    event_times_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "minimum_step_s", "maximum_step_s", "max_relative_pressure_change",
            "max_temperature_change_K", "growth_factor", "shrink_factor",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.minimum_step_s <= 0.0 or self.maximum_step_s < self.minimum_step_s:
            raise ValueError("step bounds are invalid")
        if self.max_relative_pressure_change <= 0.0 or self.max_temperature_change_K <= 0.0:
            raise ValueError("change limits must be positive")
        if not 1.0 < self.growth_factor <= 4.0 or not 0.1 <= self.shrink_factor < 1.0:
            raise ValueError("growth_factor or shrink_factor is outside the supported range")
        events = tuple(sorted(float(item) for item in self.event_times_s))
        if any(not math.isfinite(item) or item < 0.0 for item in events):
            raise ValueError("event_times_s must contain finite non-negative values")
        if any(events[index] == events[index + 1] for index in range(len(events) - 1)):
            raise ValueError("event_times_s must be unique")
        object.__setattr__(self, "event_times_s", events)

    def limit_to_event(self, time_s: float, requested_step_s: float, end_time_s: float) -> float:
        """Return a positive step that does not cross the next event or end."""

        time = float(time_s)
        requested = min(float(requested_step_s), self.maximum_step_s)
        candidates = [requested, float(end_time_s) - time]
        future_events = [event - time for event in self.event_times_s if event > time + 1.0e-12]
        if future_events:
            candidates.append(min(future_events))
        step = min(candidates)
        if step <= 0.0:
            raise ValueError("requested step does not advance time")
        if step < self.minimum_step_s:
            reaches_end = math.isclose(time + step, float(end_time_s), rel_tol=0.0, abs_tol=1.0e-10)
            reaches_event = any(math.isclose(time + step, event, rel_tol=0.0, abs_tol=1.0e-10) for event in self.event_times_s)
            if not (reaches_end or reaches_event):
                return self.minimum_step_s
        return step

    def accepts(self, before: Any, after: Any) -> bool:
        pressure_before = float(before.pressure_Pa)
        pressure_after = float(after.pressure_Pa)
        temperature_before = float(before.temperature_K)
        temperature_after = float(after.temperature_K)
        if not all(math.isfinite(value) for value in (pressure_before, pressure_after, temperature_before, temperature_after)):
            return False
        relative_pressure = abs(pressure_after - pressure_before) / max(abs(pressure_before), 1.0)
        temperature_change = abs(temperature_after - temperature_before)
        return relative_pressure <= self.max_relative_pressure_change and temperature_change <= self.max_temperature_change_K

    def next_step(self, accepted_step_s: float, before: Any, after: Any) -> float:
        pressure_before = float(before.pressure_Pa)
        pressure_after = float(after.pressure_Pa)
        relative_pressure = abs(pressure_after - pressure_before) / max(abs(pressure_before), 1.0)
        temperature_change = abs(float(after.temperature_K) - float(before.temperature_K))
        pressure_fraction = relative_pressure / self.max_relative_pressure_change
        temperature_fraction = temperature_change / self.max_temperature_change_K
        if max(pressure_fraction, temperature_fraction) < 0.25:
            return min(self.maximum_step_s, accepted_step_s * self.growth_factor)
        if max(pressure_fraction, temperature_fraction) > 0.8:
            return max(self.minimum_step_s, accepted_step_s * self.shrink_factor)
        return min(self.maximum_step_s, max(self.minimum_step_s, accepted_step_s))


@dataclass(frozen=True)
class AdaptiveStepDiagnostic:
    time_s: float
    attempted_step_s: float
    minimum_step_s: float
    retries: int
    reason: str


class AdaptiveIntegrationError(RuntimeError):
    """Raised when an adaptive run cannot produce a valid physical state."""

    def __init__(self, diagnostic: AdaptiveStepDiagnostic):
        self.diagnostic = diagnostic
        super().__init__(
            f"adaptive integration failed at t={diagnostic.time_s:g}s after "
            f"{diagnostic.retries} retries: {diagnostic.reason}"
        )


__all__ = ["AdaptiveIntegrationError", "AdaptiveStepDiagnostic", "AdaptiveStepPolicy"]
