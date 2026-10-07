"""Rigid homogeneous-equilibrium tank with a separate lumped wall.

This is the minimum open-system tank model.  It conserves total hydrogen mass
and total energy exactly at the differential-equation level.  Stratification,
spray condensation and nozzle-height effects belong to later interchangeable
tank models and are intentionally not hidden in fitted coefficients here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

from .properties import HydrogenProperties, ThermoState
from .streams import MassEnergyFlow
from .geometry import HorizontalVesselGeometry


@dataclass(frozen=True)
class TankGeometry:
    volume_m3: float
    wall_heat_capacity_J_K: float
    ambient_UA_W_K: float
    fluid_wall_UA_W_K: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class TankState:
    hydrogen_mass_kg: float
    hydrogen_internal_energy_J: float
    wall_temperature_K: float

    def __post_init__(self) -> None:
        if self.hydrogen_mass_kg <= 0.0 or not math.isfinite(self.hydrogen_mass_kg):
            raise ValueError("hydrogen_mass_kg must be finite and positive")
        if not math.isfinite(self.hydrogen_internal_energy_J):
            raise ValueError("hydrogen_internal_energy_J must be finite")
        if self.wall_temperature_K <= 0.0 or not math.isfinite(self.wall_temperature_K):
            raise ValueError("wall_temperature_K must be finite and positive")


@dataclass(frozen=True)
class TankDerivative:
    mass_kg_s: float
    internal_energy_W: float
    wall_temperature_K_s: float
    ambient_heat_W: float
    wall_to_fluid_heat_W: float


@dataclass(frozen=True)
class TankPhaseInventory:
    liquid_mass_kg: float
    vapor_mass_kg: float
    liquid_volume_m3: float
    vapor_volume_m3: float
    equilibrium_quality: float


@dataclass(frozen=True)
class TankStepResult:
    time_s: float
    state: TankState
    thermo: ThermoState
    cumulative_boundary_mass_kg: float
    cumulative_boundary_energy_J: float
    total_energy_residual_J: float


class HomogeneousTank:
    def __init__(self, geometry: TankGeometry, properties: HydrogenProperties | None = None) -> None:
        self.geometry = geometry
        self.properties = properties or HydrogenProperties()

    def initialize_saturated(
        self,
        pressure_Pa: float,
        liquid_volume_fraction: float,
        wall_temperature_K: float | None = None,
    ) -> TankState:
        if not 0.0 < liquid_volume_fraction < 1.0:
            raise ValueError("liquid_volume_fraction must be between zero and one")
        liquid = self.properties.saturated_liquid(pressure_Pa)
        vapor = self.properties.saturated_vapor(pressure_Pa)
        liquid_volume = self.geometry.volume_m3 * liquid_volume_fraction
        vapor_volume = self.geometry.volume_m3 - liquid_volume
        liquid_mass = liquid.density_kg_m3 * liquid_volume
        vapor_mass = vapor.density_kg_m3 * vapor_volume
        mass = liquid_mass + vapor_mass
        energy = (
            liquid_mass * liquid.specific_internal_energy_J_kg
            + vapor_mass * vapor.specific_internal_energy_J_kg
        )
        wall_temperature = liquid.temperature_K if wall_temperature_K is None else wall_temperature_K
        return TankState(mass, energy, wall_temperature)

    def thermo(self, state: TankState) -> ThermoState:
        return self.properties.from_rho_u(
            state.hydrogen_mass_kg / self.geometry.volume_m3,
            state.hydrogen_internal_energy_J / state.hydrogen_mass_kg,
        )

    def phase_inventory(self, state: TankState) -> TankPhaseInventory:
        thermo = self.thermo(state)
        if thermo.quality is None:
            raise ValueError("Phase inventory is defined only for a two-phase equilibrium state")
        liquid_mass = state.hydrogen_mass_kg * (1.0 - thermo.quality)
        vapor_mass = state.hydrogen_mass_kg * thermo.quality
        saturated_liquid = self.properties.saturated_liquid(thermo.pressure_Pa)
        saturated_vapor = self.properties.saturated_vapor(thermo.pressure_Pa)
        return TankPhaseInventory(
            liquid_mass_kg=liquid_mass,
            vapor_mass_kg=vapor_mass,
            liquid_volume_m3=liquid_mass / saturated_liquid.density_kg_m3,
            vapor_volume_m3=vapor_mass / saturated_vapor.density_kg_m3,
            equilibrium_quality=thermo.quality,
        )

    def liquid_level_m(self, state: TankState, shape: HorizontalVesselGeometry) -> float:
        if abs(shape.volume_m3 - self.geometry.volume_m3) > 1e-8 * self.geometry.volume_m3:
            raise ValueError("Horizontal shape volume must match thermodynamic tank volume")
        return shape.height_from_liquid_volume(self.phase_inventory(state).liquid_volume_m3)

    def derivative(
        self,
        state: TankState,
        flows: Iterable[MassEnergyFlow],
        ambient_temperature_K: float,
        direct_wall_heat_W: float = 0.0,
    ) -> TankDerivative:
        if ambient_temperature_K <= 0.0 or not math.isfinite(ambient_temperature_K):
            raise ValueError("ambient_temperature_K must be finite and positive")
        thermo = self.thermo(state)
        mass_rate = 0.0
        enthalpy_rate = 0.0
        for flow in flows:
            mass_rate += flow.mass_flow_kg_s
            if flow.mass_flow_kg_s >= 0.0:
                enthalpy_rate += flow.enthalpy_flow_W
            else:
                # The tank owns the outflow state.  A caller cannot silently
                # remove mass with a foreign enthalpy and break conservation.
                enthalpy_rate += flow.mass_flow_kg_s * thermo.specific_enthalpy_J_kg
        if not math.isfinite(direct_wall_heat_W):
            raise ValueError("direct_wall_heat_W must be finite")
        q_ambient = self.geometry.ambient_UA_W_K * (
            ambient_temperature_K - state.wall_temperature_K
        ) + direct_wall_heat_W
        q_fluid = self.geometry.fluid_wall_UA_W_K * (
            state.wall_temperature_K - thermo.temperature_K
        )
        return TankDerivative(
            mass_kg_s=mass_rate,
            internal_energy_W=enthalpy_rate + q_fluid,
            wall_temperature_K_s=(q_ambient - q_fluid) / self.geometry.wall_heat_capacity_J_K,
            ambient_heat_W=q_ambient,
            wall_to_fluid_heat_W=q_fluid,
        )

    @staticmethod
    def _advance(state: TankState, derivative: TankDerivative, dt_s: float) -> TankState:
        return TankState(
            state.hydrogen_mass_kg + derivative.mass_kg_s * dt_s,
            state.hydrogen_internal_energy_J + derivative.internal_energy_W * dt_s,
            state.wall_temperature_K + derivative.wall_temperature_K_s * dt_s,
        )

    def simulate(
        self,
        initial: TankState,
        duration_s: float,
        time_step_s: float,
        ambient_temperature_K: float | Callable[[float], float],
        flow_callback: Callable[[float, ThermoState], Iterable[MassEnergyFlow]],
        wall_heat_callback: Callable[[float, ThermoState, float], float] | None = None,
    ) -> list[TankStepResult]:
        """Advance with RK4 while keeping an independent boundary ledger."""

        if duration_s <= 0.0 or time_step_s <= 0.0:
            raise ValueError("duration_s and time_step_s must be positive")
        state = initial
        initial_total_energy = (
            state.hydrogen_internal_energy_J
            + self.geometry.wall_heat_capacity_J_K * state.wall_temperature_K
        )
        cumulative_mass = 0.0
        cumulative_energy = 0.0
        results: list[TankStepResult] = []

        def evaluate(time: float, candidate: TankState) -> tuple[TankDerivative, float, float]:
            thermo = self.thermo(candidate)
            flows = tuple(flow_callback(time, thermo))
            ambient = ambient_temperature_K(time) if callable(ambient_temperature_K) else ambient_temperature_K
            direct_wall_heat = (
                wall_heat_callback(time, thermo, candidate.wall_temperature_K)
                if wall_heat_callback is not None
                else 0.0
            )
            derivative = self.derivative(candidate, flows, ambient, direct_wall_heat)
            boundary_mass = sum(flow.mass_flow_kg_s for flow in flows)
            boundary_energy = sum(
                flow.mass_flow_kg_s
                * (flow.specific_enthalpy_J_kg if flow.mass_flow_kg_s >= 0.0 else thermo.specific_enthalpy_J_kg)
                for flow in flows
            ) + derivative.ambient_heat_W
            return derivative, boundary_mass, boundary_energy

        time = 0.0
        while time < duration_s - 1e-12:
            dt = min(time_step_s, duration_s - time)
            k1, m1, e1 = evaluate(time, state)
            k2_state = self._advance(state, k1, dt / 2.0)
            k2, m2, e2 = evaluate(time + dt / 2.0, k2_state)
            k3_state = self._advance(state, k2, dt / 2.0)
            k3, m3, e3 = evaluate(time + dt / 2.0, k3_state)
            k4_state = self._advance(state, k3, dt)
            k4, m4, e4 = evaluate(time + dt, k4_state)
            state = TankState(
                state.hydrogen_mass_kg + dt * (k1.mass_kg_s + 2*k2.mass_kg_s + 2*k3.mass_kg_s + k4.mass_kg_s) / 6.0,
                state.hydrogen_internal_energy_J + dt * (k1.internal_energy_W + 2*k2.internal_energy_W + 2*k3.internal_energy_W + k4.internal_energy_W) / 6.0,
                state.wall_temperature_K + dt * (k1.wall_temperature_K_s + 2*k2.wall_temperature_K_s + 2*k3.wall_temperature_K_s + k4.wall_temperature_K_s) / 6.0,
            )
            cumulative_mass += dt * (m1 + 2*m2 + 2*m3 + m4) / 6.0
            cumulative_energy += dt * (e1 + 2*e2 + 2*e3 + e4) / 6.0
            time += dt
            current_total_energy = (
                state.hydrogen_internal_energy_J
                + self.geometry.wall_heat_capacity_J_K * state.wall_temperature_K
            )
            results.append(TankStepResult(
                time_s=time,
                state=state,
                thermo=self.thermo(state),
                cumulative_boundary_mass_kg=cumulative_mass,
                cumulative_boundary_energy_J=cumulative_energy,
                total_energy_residual_J=current_total_energy - initial_total_energy - cumulative_energy,
            ))
        return results
