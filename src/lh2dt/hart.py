"""Explicit HART observation transforms for a PMD75B-class DPT.

The HART variables are measurement-layer values, not additional plant states.
In particular, PV and QV are two paths through the same configured
differential-pressure cell.  This module therefore requires the caller to
provide both the raw sensor pressure (QV) and the position-adjusted pressure
used for the PV path; it never infers a range, unit, address, or calibration
constant from a tag or from a recorded series.
"""

from __future__ import annotations

from dataclasses import dataclass

from .instrumentation import FirstOrderTransmitter, TransmitterState, _finite


@dataclass(frozen=True)
class HartDynamicVariables:
    """One HART dynamic-variable observation in the device's configured units."""

    pv: float
    sv: float
    tv: float
    qv: float

    def __post_init__(self) -> None:
        for name in ("pv", "sv", "tv", "qv"):
            _finite(getattr(self, name), name)

    def as_dict(self) -> dict[str, float]:
        return {"PV": self.pv, "SV": self.sv, "TV": self.tv, "QV": self.qv}


@dataclass(frozen=True)
class HartObservationState:
    """State of the optional PV damping/indication path."""

    pv_transmitter_state: TransmitterState | None


class PMD75BHartObservation:
    """Map explicit DPT inputs to PMD75B HART PV/SV/TV/QV variables.

    ``pv_transmitter`` is deliberately optional.  With no transmitter object,
    PV is the supplied position-adjusted pressure and no range or damping is
    assumed.  A caller may provide a :class:`FirstOrderTransmitter` only when
    the installed range, damping and indication settings are known or when a
    synthetic instrument test is being run.  QV always remains the supplied
    pre-damping, pre-position-adjustment sensor pressure.
    """

    VARIABLE_MEANINGS = {
        "PV": "pressure after damping and position adjustment",
        "SV": "sensor temperature",
        "TV": "electronics temperature",
        "QV": "sensor pressure before damping and position adjustment",
    }

    def __init__(self, pv_transmitter: FirstOrderTransmitter | None = None) -> None:
        self.pv_transmitter = pv_transmitter

    def initialize(self, position_adjusted_pressure: float) -> HartObservationState:
        adjusted = _finite(position_adjusted_pressure, "position_adjusted_pressure")
        if self.pv_transmitter is None:
            return HartObservationState(None)
        return HartObservationState(self.pv_transmitter.initialize(adjusted))

    def advance(
        self,
        state: HartObservationState,
        *,
        raw_sensor_pressure: float,
        position_adjusted_pressure: float,
        sensor_temperature: float,
        electronics_temperature: float,
        time_step_s: float,
    ) -> tuple[HartObservationState, HartDynamicVariables]:
        qv = _finite(raw_sensor_pressure, "raw_sensor_pressure")
        adjusted = _finite(position_adjusted_pressure, "position_adjusted_pressure")
        sv = _finite(sensor_temperature, "sensor_temperature")
        tv = _finite(electronics_temperature, "electronics_temperature")
        dt = _finite(time_step_s, "time_step_s")
        if dt <= 0.0:
            raise ValueError("time_step_s must be positive")
        if not isinstance(state, HartObservationState):
            raise TypeError("state must be HartObservationState")

        if self.pv_transmitter is None:
            next_state = HartObservationState(None)
            pv = adjusted
        else:
            if state.pv_transmitter_state is None:
                raise ValueError("state was initialized without the PV transmitter")
            transmitter_state, reading = self.pv_transmitter.advance(
                state.pv_transmitter_state, adjusted, dt
            )
            next_state = HartObservationState(transmitter_state)
            pv = reading.indicated_value
        return next_state, HartDynamicVariables(pv=pv, sv=sv, tv=tv, qv=qv)


def pmd75b_hart_variable_meanings() -> dict[str, str]:
    """Return a copy of the manufacturer-family variable meaning contract."""

    return dict(PMD75BHartObservation.VARIABLE_MEANINGS)
