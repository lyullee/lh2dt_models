"""Discrete PID control with explicit limits and anti-windup state."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class PIDParameters:
    proportional_gain: float
    integral_gain_per_s: float
    derivative_gain_s: float
    output_minimum: float = 0.0
    output_maximum: float = 1.0
    output_bias: float = 0.0
    derivative_filter_time_s: float = 0.0
    reverse_acting: bool = False

    def __post_init__(self) -> None:
        values = {
            name: _finite(getattr(self, name), name)
            for name in (
                "proportional_gain",
                "integral_gain_per_s",
                "derivative_gain_s",
                "output_minimum",
                "output_maximum",
                "output_bias",
                "derivative_filter_time_s",
            )
        }
        for name in ("proportional_gain", "integral_gain_per_s", "derivative_gain_s"):
            if values[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if values["output_maximum"] <= values["output_minimum"]:
            raise ValueError("output_maximum must exceed output_minimum")
        if values["derivative_filter_time_s"] < 0.0:
            raise ValueError("derivative_filter_time_s must be non-negative")
        if not isinstance(self.reverse_acting, bool):
            raise ValueError("reverse_acting must be bool")


@dataclass(frozen=True)
class PIDState:
    integral_error_s: float = 0.0
    previous_measurement: float | None = None
    filtered_measurement_rate_per_s: float = 0.0


@dataclass(frozen=True)
class PIDResult:
    output: float
    unsaturated_output: float
    error: float
    proportional_term: float
    integral_term: float
    derivative_term: float
    saturated: bool
    state: PIDState


class PIDController:
    """Parallel-form PID with derivative-on-measurement and clamped integral.

    Conditional integration prevents further windup while an output limit is
    active.  The optional first-order derivative filter is updated with its
    exact zero-order-hold response.
    """

    def __init__(self, parameters: PIDParameters) -> None:
        self.parameters = parameters

    def step(
        self,
        state: PIDState,
        setpoint: float,
        measurement: float,
        time_step_s: float,
    ) -> PIDResult:
        target = _finite(setpoint, "setpoint")
        measured = _finite(measurement, "measurement")
        dt = _finite(time_step_s, "time_step_s")
        if dt <= 0.0:
            raise ValueError("time_step_s must be positive")
        p = self.parameters
        direction = -1.0 if p.reverse_acting else 1.0
        error = direction * (target - measured)
        proportional = p.proportional_gain * error

        if state.previous_measurement is None:
            raw_measurement_rate = 0.0
        else:
            raw_measurement_rate = (measured - state.previous_measurement) / dt
        if p.derivative_filter_time_s == 0.0:
            filtered_rate = raw_measurement_rate
        else:
            alpha = math.exp(-dt / p.derivative_filter_time_s)
            filtered_rate = (
                alpha * state.filtered_measurement_rate_per_s
                + (1.0 - alpha) * raw_measurement_rate
            )
        derivative = -direction * p.derivative_gain_s * filtered_rate

        candidate_integral = state.integral_error_s + error * dt
        candidate_output = (
            p.output_bias
            + proportional
            + p.integral_gain_per_s * candidate_integral
            + derivative
        )
        at_high = candidate_output > p.output_maximum
        at_low = candidate_output < p.output_minimum
        drives_further_into_limit = (at_high and error > 0.0) or (at_low and error < 0.0)
        integral = state.integral_error_s if drives_further_into_limit else candidate_integral
        integral_term = p.integral_gain_per_s * integral
        unsaturated = p.output_bias + proportional + integral_term + derivative
        output = min(max(unsaturated, p.output_minimum), p.output_maximum)
        next_state = PIDState(
            integral_error_s=integral,
            previous_measurement=measured,
            filtered_measurement_rate_per_s=filtered_rate,
        )
        return PIDResult(
            output=output,
            unsaturated_output=unsaturated,
            error=error,
            proportional_term=proportional,
            integral_term=integral_term,
            derivative_term=derivative,
            saturated=output != unsaturated,
            state=next_state,
        )
