"""First-principles hydrogen-gas compressor energy balance."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .properties import HydrogenProperties, ThermoState


@dataclass(frozen=True)
class CompressorResult:
    """Thermodynamic outlet and shaft duty for one compression point."""

    outlet: ThermoState
    pressure_ratio: float
    pressure_rise_Pa: float
    shaft_power_W: float
    isentropic_power_W: float
    efficiency: float


class Compressor:
    """Adiabatic hydrogen compressor with an explicit pressure ratio.

    The outlet is calculated from an isentropic reference state followed by
    the declared isentropic efficiency.  No plant time-series coefficient is
    used.  This operating-point object is intentionally analogous to
    :class:`lh2dt.pump.Pump`, but rejects liquid inlet states.
    """

    def __init__(
        self,
        properties: HydrogenProperties | None = None,
        *,
        pressure_ratio: float,
        isentropic_efficiency: float,
    ) -> None:
        for name, value in (
            ("pressure_ratio", pressure_ratio),
            ("isentropic_efficiency", isentropic_efficiency),
        ):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
        if float(pressure_ratio) <= 1.0:
            raise ValueError("pressure_ratio must be greater than 1")
        if not 0.0 < float(isentropic_efficiency) <= 1.0:
            raise ValueError("isentropic_efficiency must be in (0, 1]")
        self.properties = properties or HydrogenProperties()
        self.pressure_ratio = float(pressure_ratio)
        self.isentropic_efficiency = float(isentropic_efficiency)

    def evaluate(self, inlet: ThermoState, mass_flow_kg_s: float) -> CompressorResult:
        if isinstance(mass_flow_kg_s, bool):
            raise ValueError("mass_flow_kg_s must be numeric")
        try:
            mass_flow = float(mass_flow_kg_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("mass_flow_kg_s must be numeric") from exc
        if not math.isfinite(mass_flow) or mass_flow <= 0.0:
            raise ValueError("mass_flow_kg_s must be finite and positive")
        phase = str(inlet.phase).strip().lower()
        if inlet.quality is not None and inlet.quality < 1.0 - 1.0e-7:
            raise ValueError("Compressor requires a vapor inlet state")
        if inlet.quality is None and "liquid" in phase and "gas" not in phase:
            raise ValueError("Compressor requires a vapor inlet state")

        outlet_pressure = inlet.pressure_Pa * self.pressure_ratio
        ideal_outlet = self.properties.from_ps(
            outlet_pressure, inlet.specific_entropy_J_kgK
        )
        ideal_dh = (
            ideal_outlet.specific_enthalpy_J_kg
            - inlet.specific_enthalpy_J_kg
        )
        actual_dh = ideal_dh / self.isentropic_efficiency
        outlet = self.properties.from_ph(
            outlet_pressure,
            inlet.specific_enthalpy_J_kg + actual_dh,
        )
        return CompressorResult(
            outlet=outlet,
            pressure_ratio=self.pressure_ratio,
            pressure_rise_Pa=outlet_pressure - inlet.pressure_Pa,
            shaft_power_W=mass_flow * actual_dh,
            isentropic_power_W=mass_flow * ideal_dh,
            efficiency=self.isentropic_efficiency,
        )
