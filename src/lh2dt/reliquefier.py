"""Reduced-order reliquefier with an explicit first-law boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .properties import HydrogenProperties, ThermoState


@dataclass(frozen=True)
class ReliquefierResult:
    outlet: ThermoState
    heat_removed_W: float
    requested_heat_removal_W: float
    liquefied_mass_fraction: float
    capacity_limited: bool


class Reliquefier:
    def __init__(self, cooling_capacity_W: float, properties: HydrogenProperties | None = None) -> None:
        if isinstance(cooling_capacity_W, bool):
            raise ValueError("cooling_capacity_W must be numeric")
        try:
            capacity = float(cooling_capacity_W)
        except (TypeError, ValueError) as exc:
            raise ValueError("cooling_capacity_W must be numeric") from exc
        if not math.isfinite(capacity):
            raise ValueError("cooling_capacity_W must be finite")
        if capacity <= 0.0:
            raise ValueError("cooling_capacity_W must be positive")
        self.cooling_capacity = capacity
        self.properties = properties or HydrogenProperties()

    def evaluate(
        self,
        inlet: ThermoState,
        outlet_pressure_Pa: float,
        mass_flow_kg_s: float,
        target_quality: float = 0.0,
    ) -> ReliquefierResult:
        if (
            isinstance(mass_flow_kg_s, bool)
            or isinstance(outlet_pressure_Pa, bool)
            or isinstance(target_quality, bool)
        ):
            raise ValueError("reliquefier operating values must be numeric")
        try:
            mass_flow = float(mass_flow_kg_s)
            outlet_pressure = float(outlet_pressure_Pa)
            quality = float(target_quality)
        except (TypeError, ValueError) as exc:
            raise ValueError("reliquefier operating values must be numeric") from exc
        if not math.isfinite(mass_flow) or mass_flow <= 0.0:
            raise ValueError("mass_flow_kg_s must be finite and positive")
        if not math.isfinite(outlet_pressure) or outlet_pressure <= 0.0:
            raise ValueError("outlet_pressure_Pa must be finite and positive")
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("target_quality must be in [0, 1]")
        liquid = self.properties.saturated_liquid(outlet_pressure)
        vapor = self.properties.saturated_vapor(outlet_pressure)
        target_h = liquid.specific_enthalpy_J_kg + quality * (
            vapor.specific_enthalpy_J_kg - liquid.specific_enthalpy_J_kg
        )
        requested = max(0.0, mass_flow * (inlet.specific_enthalpy_J_kg - target_h))
        removed = min(requested, self.cooling_capacity)
        outlet_h = inlet.specific_enthalpy_J_kg - removed / mass_flow
        outlet = self.properties.from_ph(outlet_pressure, outlet_h)
        liquid_fraction = 0.0
        if outlet.quality is not None:
            liquid_fraction = 1.0 - outlet.quality
        elif outlet.specific_enthalpy_J_kg <= liquid.specific_enthalpy_J_kg:
            liquid_fraction = 1.0
        return ReliquefierResult(
            outlet=outlet,
            heat_removed_W=removed,
            requested_heat_removal_W=requested,
            liquefied_mass_fraction=liquid_fraction,
            capacity_limited=removed < requested * (1.0 - 1e-12),
        )
