"""Deterministic transmitter dynamics for model-to-instrument comparison.

Plant-state predictions and recorded historian values are different layers.
This module represents finite response time, range, scale/bias, saturation,
resolution, and stated percent-of-span accuracy without inventing calibration
constants or adding random noise.
"""

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
class TransmitterParameters:
    lower_range: float
    upper_range: float
    time_constant_s: float = 0.0
    scale_factor: float = 1.0
    bias: float = 0.0
    resolution: float = 0.0
    accuracy_fraction_of_span: float = 0.0

    def __post_init__(self) -> None:
        values = {
            name: _finite(getattr(self, name), name)
            for name in (
                "lower_range",
                "upper_range",
                "time_constant_s",
                "scale_factor",
                "bias",
                "resolution",
                "accuracy_fraction_of_span",
            )
        }
        if values["upper_range"] <= values["lower_range"]:
            raise ValueError("upper_range must exceed lower_range")
        if values["time_constant_s"] < 0.0:
            raise ValueError("time_constant_s must be non-negative")
        if values["scale_factor"] <= 0.0:
            raise ValueError("scale_factor must be positive")
        if values["resolution"] < 0.0:
            raise ValueError("resolution must be non-negative")
        if values["accuracy_fraction_of_span"] < 0.0:
            raise ValueError("accuracy_fraction_of_span must be non-negative")

    @property
    def span(self) -> float:
        return self.upper_range - self.lower_range


@dataclass(frozen=True)
class TransmitterState:
    filtered_process_value: float

    def __post_init__(self) -> None:
        _finite(self.filtered_process_value, "filtered_process_value")


@dataclass(frozen=True)
class TransmitterReading:
    indicated_value: float
    filtered_process_value: float
    unsaturated_indicated_value: float
    saturated: bool
    quantization_error_bound: float
    stated_accuracy_bound: float


class FirstOrderTransmitter:
    """First-order lag followed by deterministic indication effects.

    The lag update uses the exact zero-order-hold solution, so its result at a
    sample time does not acquire Euler time-step error.  This matters when a
    model trajectory is compared with a slower PLC or historian scan.
    """

    def __init__(self, parameters: TransmitterParameters) -> None:
        self.parameters = parameters

    def initialize(self, process_value: float) -> TransmitterState:
        return TransmitterState(_finite(process_value, "process_value"))

    def advance(
        self,
        state: TransmitterState,
        process_value: float,
        time_step_s: float,
    ) -> tuple[TransmitterState, TransmitterReading]:
        target = _finite(process_value, "process_value")
        dt = _finite(time_step_s, "time_step_s")
        if dt <= 0.0:
            raise ValueError("time_step_s must be positive")
        tau = self.parameters.time_constant_s
        if tau == 0.0:
            filtered = target
        else:
            filtered = target + (state.filtered_process_value - target) * math.exp(-dt / tau)
        next_state = TransmitterState(filtered)
        return next_state, self.read(next_state)

    def read(self, state: TransmitterState) -> TransmitterReading:
        p = self.parameters
        raw = p.scale_factor * state.filtered_process_value + p.bias
        clipped = min(max(raw, p.lower_range), p.upper_range)
        saturated = clipped != raw
        if p.resolution > 0.0:
            steps = round((clipped - p.lower_range) / p.resolution)
            indicated = p.lower_range + steps * p.resolution
            indicated = min(max(indicated, p.lower_range), p.upper_range)
        else:
            indicated = clipped
        return TransmitterReading(
            indicated_value=indicated,
            filtered_process_value=state.filtered_process_value,
            unsaturated_indicated_value=raw,
            saturated=saturated,
            quantization_error_bound=p.resolution / 2.0,
            stated_accuracy_bound=p.accuracy_fraction_of_span * p.span,
        )


def absolute_to_gauge_pressure_Pa(
    absolute_pressure_Pa: float,
    atmospheric_pressure_Pa: float = 101_325.0,
) -> float:
    absolute = _finite(absolute_pressure_Pa, "absolute_pressure_Pa")
    atmosphere = _finite(atmospheric_pressure_Pa, "atmospheric_pressure_Pa")
    if absolute <= 0.0 or atmosphere <= 0.0:
        raise ValueError("absolute and atmospheric pressures must be positive")
    return absolute - atmosphere


def gauge_to_absolute_pressure_Pa(
    gauge_pressure_Pa: float,
    atmospheric_pressure_Pa: float = 101_325.0,
) -> float:
    gauge = _finite(gauge_pressure_Pa, "gauge_pressure_Pa")
    atmosphere = _finite(atmospheric_pressure_Pa, "atmospheric_pressure_Pa")
    absolute = gauge + atmosphere
    if atmosphere <= 0.0 or absolute <= 0.0:
        raise ValueError("converted absolute pressure must be positive")
    return absolute
