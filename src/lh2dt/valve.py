"""Adiabatic valve/orifice model with a homogeneous-equilibrium throat search."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .properties import HydrogenProperties, ThermoState


@dataclass(frozen=True)
class ValveResult:
    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float
    throat_pressure_Pa: float
    mass_flux_kg_m2_s: float
    choked: bool
    upstream_side: str | None


class Valve:
    """Two-way throttling element.

    The discharge coefficient and full-open area remain explicit engineering
    parameters.  No plant data are used to infer either value.
    """

    def __init__(
        self,
        full_open_area_m2: float,
        discharge_coefficient: float,
        properties: HydrogenProperties | None = None,
        pressure_samples: int = 60,
        allow_reverse: bool = True,
        closed_leakage_CdA_m2: float = 0.0,
    ) -> None:
        if isinstance(full_open_area_m2, bool) or isinstance(discharge_coefficient, bool):
            raise ValueError("valve geometry values must be numeric")
        try:
            full_open_area = float(full_open_area_m2)
            discharge = float(discharge_coefficient)
        except (TypeError, ValueError) as exc:
            raise ValueError("valve geometry values must be numeric") from exc
        if full_open_area <= 0.0 or not math.isfinite(full_open_area):
            raise ValueError("full_open_area_m2 must be finite and positive")
        if not math.isfinite(discharge) or not 0.0 < discharge <= 1.0:
            raise ValueError("discharge_coefficient must be in (0, 1]")
        if isinstance(pressure_samples, bool) or not isinstance(pressure_samples, int):
            raise ValueError("pressure_samples must be an integer")
        if pressure_samples < 12:
            raise ValueError("pressure_samples must be at least 12")
        if not isinstance(allow_reverse, bool):
            raise ValueError("allow_reverse must be bool")
        if isinstance(closed_leakage_CdA_m2, bool):
            raise ValueError("closed_leakage_CdA_m2 must be numeric")
        try:
            closed_leakage_CdA = float(closed_leakage_CdA_m2)
        except (TypeError, ValueError) as exc:
            raise ValueError("closed_leakage_CdA_m2 must be numeric") from exc
        full_open_CdA = discharge * full_open_area
        if (
            not math.isfinite(closed_leakage_CdA)
            or closed_leakage_CdA < 0.0
            or closed_leakage_CdA > full_open_CdA
        ):
            raise ValueError(
                "closed_leakage_CdA_m2 must be finite and lie in "
                "[0, discharge_coefficient * full_open_area_m2]"
            )
        self.area = full_open_area
        self.cd = discharge
        self.closed_leakage_CdA_m2 = closed_leakage_CdA
        self.properties = properties or HydrogenProperties()
        self.pressure_samples = int(pressure_samples)
        self.allow_reverse = allow_reverse

    @staticmethod
    def _opening_area(opening: float, area: float) -> float:
        if isinstance(opening, bool):
            raise ValueError("opening must be numeric")
        try:
            opening = float(opening)
        except (TypeError, ValueError) as exc:
            raise ValueError("opening must be numeric") from exc
        if not math.isfinite(opening):
            raise ValueError("opening must be finite")
        if not 0.0 <= opening <= 1.0:
            raise ValueError("opening must be in [0, 1]")
        return area * opening

    def _effective_CdA_m2(self, opening: float) -> float:
        """Blend finite closed-seat leakage with commanded opening."""

        commanded_area = self._opening_area(opening, self.area)
        opening_fraction = commanded_area / self.area
        full_open_CdA = self.cd * self.area
        return (
            self.closed_leakage_CdA_m2
            + (full_open_CdA - self.closed_leakage_CdA_m2) * opening_fraction
        )

    def _forward(self, upstream: ThermoState, downstream_pressure_Pa: float, opening: float) -> ValveResult:
        effective_CdA = self._effective_CdA_m2(opening)
        if effective_CdA == 0.0 or upstream.pressure_Pa <= downstream_pressure_Pa:
            return ValveResult(0.0, upstream.specific_enthalpy_J_kg, upstream.pressure_Pa, 0.0, False, None)
        pressure_low = max(float(downstream_pressure_Pa), 1.0e3)
        # Near pressure equilibrium the small offset from upstream pressure
        # can fall below the downstream pressure.  Keep every throat sample
        # within the actual expansion interval; otherwise a nearly closed
        # pressure drop produces a fictitious extra expansion and flow.
        pressure_high = max(pressure_low, upstream.pressure_Pa * (1.0 - 1.0e-9))
        pressures = np.geomspace(pressure_low, pressure_high, self.pressure_samples)
        flux = np.zeros_like(pressures)
        for i, pressure in enumerate(pressures):
            try:
                throat = self.properties.from_ps(pressure, upstream.specific_entropy_J_kgK)
                kinetic_specific_energy = max(
                    0.0,
                    2.0 * (upstream.specific_enthalpy_J_kg - throat.specific_enthalpy_J_kg),
                )
                flux[i] = throat.density_kg_m3 * math.sqrt(kinetic_specific_energy)
            except (ValueError, RuntimeError):
                flux[i] = 0.0
        index = int(np.argmax(flux))
        maximum_flux = float(flux[index])
        throat_pressure = float(pressures[index])
        return ValveResult(
            mass_flow_kg_s=effective_CdA * maximum_flux,
            outlet_specific_enthalpy_J_kg=upstream.specific_enthalpy_J_kg,
            throat_pressure_Pa=throat_pressure,
            mass_flux_kg_m2_s=maximum_flux,
            choked=throat_pressure > downstream_pressure_Pa * 1.002,
            upstream_side="left",
        )

    def evaluate(self, left: ThermoState, right: ThermoState, opening: float = 1.0) -> ValveResult:
        """Return positive mass flow for left-to-right flow."""

        if left.pressure_Pa >= right.pressure_Pa:
            return self._forward(left, right.pressure_Pa, opening)
        if not self.allow_reverse:
            return ValveResult(0.0, right.specific_enthalpy_J_kg, right.pressure_Pa, 0.0, False, None)
        reverse = self._forward(right, left.pressure_Pa, opening)
        return ValveResult(
            mass_flow_kg_s=-reverse.mass_flow_kg_s,
            outlet_specific_enthalpy_J_kg=reverse.outlet_specific_enthalpy_J_kg,
            throat_pressure_Pa=reverse.throat_pressure_Pa,
            mass_flux_kg_m2_s=reverse.mass_flux_kg_m2_s,
            choked=reverse.choked,
            upstream_side="right" if reverse.mass_flow_kg_s else None,
        )


class CheckValve(Valve):
    """Forward-only throttling element with an explicit cracking pressure.

    The opening criterion is a pressure-difference boundary condition.  Once
    the upstream-to-downstream pressure difference exceeds
    ``cracking_pressure_Pa``, the same isenthalpic/choked orifice closure as
    :class:`Valve` is evaluated with the cracking drop consumed before the
    downstream pressure.  Reverse flow is always blocked.  The cracking
    pressure is an explicit design input and is never inferred from plant
    series.
    """

    def __init__(
        self,
        full_open_area_m2: float,
        discharge_coefficient: float,
        cracking_pressure_Pa: float = 0.0,
        properties: HydrogenProperties | None = None,
        pressure_samples: int = 60,
    ) -> None:
        if isinstance(cracking_pressure_Pa, bool):
            raise ValueError("cracking_pressure_Pa must be numeric")
        try:
            cracking = float(cracking_pressure_Pa)
        except (TypeError, ValueError) as exc:
            raise ValueError("cracking_pressure_Pa must be numeric") from exc
        if not math.isfinite(cracking) or cracking < 0.0:
            raise ValueError("cracking_pressure_Pa must be finite and non-negative")
        super().__init__(
            full_open_area_m2,
            discharge_coefficient,
            properties,
            pressure_samples=pressure_samples,
            allow_reverse=False,
        )
        self.cracking_pressure_Pa = cracking

    def evaluate(self, left: ThermoState, right: ThermoState, opening: float = 1.0) -> ValveResult:
        """Return positive flow only after the explicit cracking drop is met."""

        pressure_drop = left.pressure_Pa - right.pressure_Pa
        if pressure_drop <= self.cracking_pressure_Pa:
            # Validate the command even while closed so bad control input is
            # not silently hidden behind the check-valve state.
            self._opening_area(opening, self.area)
            return ValveResult(
                0.0,
                left.specific_enthalpy_J_kg,
                left.pressure_Pa,
                0.0,
                False,
                None,
            )
        return self._forward(
            left,
            right.pressure_Pa + self.cracking_pressure_Pa,
            opening,
        )


class CommandedValve:
    """Network two-port wrapper with an explicit opening command."""

    def __init__(self, valve: Valve, initial_opening: float = 0.0) -> None:
        if not isinstance(valve, Valve):
            raise TypeError("valve must be a Valve instance")
        self.valve = valve
        self._opening = 0.0
        self.set_opening(initial_opening)

    @property
    def opening(self) -> float:
        return self._opening

    def set_opening(self, opening: float) -> None:
        if isinstance(opening, bool):
            raise ValueError("opening must be numeric")
        try:
            opening = float(opening)
        except (TypeError, ValueError) as exc:
            raise ValueError("opening must be numeric") from exc
        if not math.isfinite(opening) or not 0.0 <= opening <= 1.0:
            raise ValueError("opening must be finite and in [0, 1]")
        self._opening = opening

    def evaluate(self, left: ThermoState, right: ThermoState) -> ValveResult:
        return self.valve.evaluate(left, right, self._opening)

class StrokeLimitedCommandedValve:
    """Commanded valve with an explicit finite stroke time.

    ``CommandedValve`` remains the ideal, instantaneous actuator used by
    existing screening scenarios.  This wrapper is the replaceable dynamic
    alternative for a real valve whose measured/declared stroke time matters:
    ``set_target_opening`` changes the command and ``advance`` integrates the
    actuator position at a bounded rate.  The hydraulic closure is still the
    underlying :class:`Valve`, so no empirical flow law is introduced.

    The actuator state is intentionally separate from the fluid network state.
    A scenario callback advances it with the elapsed simulation time before
    applying commands at that time.  This makes command timing explicit while
    keeping the two-port and conservation contracts unchanged.
    """

    def __init__(
        self,
        valve: Valve,
        stroke_time_s: float,
        initial_opening: float = 0.0,
    ) -> None:
        if not isinstance(valve, Valve):
            raise TypeError("valve must be a Valve instance")
        if isinstance(stroke_time_s, bool):
            raise ValueError("stroke_time_s must be numeric")
        try:
            stroke_time = float(stroke_time_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("stroke_time_s must be numeric") from exc
        if not math.isfinite(stroke_time) or stroke_time <= 0.0:
            raise ValueError("stroke_time_s must be finite and positive")
        self.valve = valve
        self.stroke_time_s = stroke_time
        self._opening = 0.0
        self._target_opening = 0.0
        self.set_opening(initial_opening)

    @property
    def opening(self) -> float:
        """Current physical opening used by the hydraulic model."""

        return self._opening

    @property
    def target_opening(self) -> float:
        """Most recent commanded opening."""

        return self._target_opening

    @staticmethod
    def _validate_opening(opening: float) -> float:
        if isinstance(opening, bool):
            raise ValueError("opening must be numeric")
        try:
            value = float(opening)
        except (TypeError, ValueError) as exc:
            raise ValueError("opening must be numeric") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("opening must be finite and in [0, 1]")
        return value

    def set_target_opening(self, opening: float) -> None:
        """Apply a command without teleporting the physical valve position."""

        self._target_opening = self._validate_opening(opening)

    def set_opening(self, opening: float) -> None:
        """Set both physical and commanded position for initialization/reset."""

        value = self._validate_opening(opening)
        self._opening = value
        self._target_opening = value

    def advance(self, time_step_s: float) -> float:
        """Advance the actuator position by a non-negative elapsed time."""

        if isinstance(time_step_s, bool):
            raise ValueError("time_step_s must be numeric")
        try:
            dt = float(time_step_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("time_step_s must be numeric") from exc
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("time_step_s must be finite and non-negative")
        difference = self._target_opening - self._opening
        maximum_change = dt / self.stroke_time_s
        if abs(difference) <= maximum_change:
            self._opening = self._target_opening
        elif difference > 0.0:
            self._opening += maximum_change
        else:
            self._opening -= maximum_change
        return self._opening

    def evaluate(self, left: ThermoState, right: ThermoState) -> ValveResult:
        return self.valve.evaluate(left, right, self._opening)

    def export_accident_history(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Export a bounded gas outlet with an explicit upstream ledger.

        The valve has no inventory state of its own.  ``available_mass_kg``
        and the termination basis therefore must be supplied by the upstream
        equipment model; this method never infers inventory depletion or
        isolation from the actuator position.
        """
        from .accident_export import export_fixed_source_accident_history

        return export_fixed_source_accident_history(
            request,
            properties=self.valve.properties,
            provider_model=f"{type(self).__module__}.{type(self).__qualname__}",
            evaluate=self.valve.evaluate,
            advance_opening=self.advance,
            opening_area_m2=self.valve.area,
            area_provenance="native StrokeLimitedCommandedValve full-open bore scaled by stroke state",
        )


class PressureReliefValve:
    """One-way pressure relief wrapper with explicit set/reseat hysteresis.

    The underlying :class:`Valve` supplies the adiabatic orifice physics.  The
    relief state is a separate safety path: it opens when the upstream pressure
    reaches ``set_pressure_Pa`` and remains open until it falls to
    ``reseat_pressure_Pa``.  Both pressures are engineering inputs; no plant
    time series is used to infer them.  The wrapper is deliberately binary so
    a manufacturer lift curve can replace it without changing the two-port
    contract.
    """

    def __init__(
        self,
        valve: Valve,
        set_pressure_Pa: float,
        reseat_pressure_Pa: float,
    ) -> None:
        if not isinstance(valve, Valve):
            raise TypeError("valve must be a Valve instance")
        if isinstance(set_pressure_Pa, bool) or isinstance(reseat_pressure_Pa, bool):
            raise ValueError("relief pressures must be numeric")
        try:
            set_pressure = float(set_pressure_Pa)
            reseat = float(reseat_pressure_Pa)
        except (TypeError, ValueError) as exc:
            raise ValueError("relief pressures must be numeric") from exc
        if not math.isfinite(set_pressure) or set_pressure <= 0.0:
            raise ValueError("set_pressure_Pa must be finite and positive")
        if not math.isfinite(reseat) or not 0.0 < reseat <= set_pressure:
            raise ValueError("reseat_pressure_Pa must lie in (0, set_pressure_Pa]")
        self.valve = valve
        self.set_pressure_Pa = set_pressure
        self.reseat_pressure_Pa = reseat
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def reset(self) -> None:
        self._open = False

    def _update_state(self, upstream_pressure_Pa: float) -> float:
        if self._open:
            if upstream_pressure_Pa <= self.reseat_pressure_Pa:
                self._open = False
        elif upstream_pressure_Pa >= self.set_pressure_Pa:
            self._open = True
        return 1.0 if self._open else 0.0

    def evaluate(self, left: ThermoState, right: ThermoState) -> ValveResult:
        """Evaluate only left-to-right relief flow; reverse flow is blocked."""

        if left.pressure_Pa < right.pressure_Pa:
            return self.valve.evaluate(left, right, opening=0.0)
        opening = self._update_state(left.pressure_Pa)
        return self.valve.evaluate(left, right, opening=opening)


class TemperatureProtectionValve:
    """One-way temperature-trip wrapper around the valve orifice model.

    The device opens at ``trip_temperature_K`` and reseats at
    ``reset_temperature_K``.  The thresholds are explicit protection-device
    inputs; no temperature history is used to infer them.
    """

    def __init__(
        self,
        valve: Valve,
        trip_temperature_K: float,
        reset_temperature_K: float,
    ) -> None:
        if not isinstance(valve, Valve):
            raise TypeError("valve must be a Valve instance")
        if isinstance(trip_temperature_K, bool) or isinstance(reset_temperature_K, bool):
            raise ValueError("temperature thresholds must be numeric")
        try:
            trip_temperature = float(trip_temperature_K)
            reset_temperature = float(reset_temperature_K)
        except (TypeError, ValueError) as exc:
            raise ValueError("temperature thresholds must be numeric") from exc
        if not math.isfinite(trip_temperature) or trip_temperature <= 0.0:
            raise ValueError("trip_temperature_K must be finite and positive")
        if not math.isfinite(reset_temperature) or not 0.0 < reset_temperature <= trip_temperature:
            raise ValueError("reset_temperature_K must lie in (0, trip_temperature_K]")
        self.valve = valve
        self.trip_temperature_K = trip_temperature
        self.reset_temperature_K = reset_temperature
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def reset(self) -> None:
        self._open = False

    def _update_state(self, upstream_temperature_K: float) -> float:
        if self._open:
            if upstream_temperature_K <= self.reset_temperature_K:
                self._open = False
        elif upstream_temperature_K >= self.trip_temperature_K:
            self._open = True
        return 1.0 if self._open else 0.0

    def evaluate(self, left: ThermoState, right: ThermoState) -> ValveResult:
        """Evaluate only left-to-right protection flow; reverse flow is blocked."""

        if left.pressure_Pa < right.pressure_Pa:
            return self.valve.evaluate(left, right, opening=0.0)
        opening = self._update_state(left.temperature_K)
        return self.valve.evaluate(left, right, opening=opening)
