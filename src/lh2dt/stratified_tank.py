"""Conservative liquid/vapor nonequilibrium model for a rigid LH2 vessel.

The liquid, vapor, lower wall and upper wall are separate control volumes.  A
single pressure is recovered from the rigid-volume constraint.  Phase-change
mass and energy are internal transfers, so they cancel exactly in the complete
tank energy balance.  Heat-transfer conductances are explicit closure inputs;
the model contains no hidden calibration against facility observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import brentq

from .geometry import HorizontalVesselGeometry
from .properties import HydrogenProperties, ThermoState
from .streams import MassEnergyFlow


@dataclass(frozen=True)
class StratifiedTankParameters:
    lower_wall_heat_capacity_J_K: float
    upper_wall_heat_capacity_J_K: float
    lower_ambient_UA_W_K: float
    upper_ambient_UA_W_K: float
    wall_liquid_U_W_m2K: float
    wall_vapor_U_W_m2K: float
    liquid_interface_UA_W_K: float
    vapor_interface_UA_W_K: float
    wall_axial_UA_W_K: float
    wall_zone_split_height_m: float
    minimum_pressure_Pa: float = 10_000.0
    maximum_pressure_Pa: float = 1_250_000.0

    def __post_init__(self) -> None:
        positive = (
            "lower_wall_heat_capacity_J_K",
            "upper_wall_heat_capacity_J_K",
            "minimum_pressure_Pa",
            "maximum_pressure_Pa",
        )
        nonnegative = (
            "lower_ambient_UA_W_K",
            "upper_ambient_UA_W_K",
            "wall_liquid_U_W_m2K",
            "wall_vapor_U_W_m2K",
            "liquid_interface_UA_W_K",
            "vapor_interface_UA_W_K",
            "wall_axial_UA_W_K",
            "wall_zone_split_height_m",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.maximum_pressure_Pa <= self.minimum_pressure_Pa:
            raise ValueError("maximum_pressure_Pa must exceed minimum_pressure_Pa")


@dataclass(frozen=True)
class StratifiedTankState:
    liquid_mass_kg: float
    liquid_internal_energy_J: float
    vapor_mass_kg: float
    vapor_internal_energy_J: float
    lower_wall_temperature_K: float
    upper_wall_temperature_K: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.liquid_mass_kg) or self.liquid_mass_kg <= 0.0:
            raise ValueError("liquid_mass_kg must be finite and positive")
        if not math.isfinite(self.vapor_mass_kg) or self.vapor_mass_kg <= 0.0:
            raise ValueError("vapor_mass_kg must be finite and positive")
        if not math.isfinite(self.liquid_internal_energy_J):
            raise ValueError("liquid_internal_energy_J must be finite")
        if not math.isfinite(self.vapor_internal_energy_J):
            raise ValueError("vapor_internal_energy_J must be finite")
        for name in ("lower_wall_temperature_K", "upper_wall_temperature_K"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class StratifiedThermoState:
    pressure_Pa: float
    liquid: ThermoState
    vapor: ThermoState
    liquid_volume_m3: float
    vapor_volume_m3: float
    liquid_height_m: float
    interface_area_m2: float
    lower_wet_area_m2: float
    lower_dry_area_m2: float
    upper_wet_area_m2: float
    upper_dry_area_m2: float


@dataclass(frozen=True)
class StratifiedTankDerivative:
    liquid_mass_kg_s: float
    liquid_internal_energy_W: float
    vapor_mass_kg_s: float
    vapor_internal_energy_W: float
    lower_wall_temperature_K_s: float
    upper_wall_temperature_K_s: float
    evaporation_rate_kg_s: float
    lower_ambient_heat_W: float
    upper_ambient_heat_W: float
    lower_wall_to_liquid_heat_W: float
    lower_wall_to_vapor_heat_W: float
    upper_wall_to_liquid_heat_W: float
    upper_wall_to_vapor_heat_W: float
    liquid_to_interface_heat_W: float
    vapor_to_interface_heat_W: float
    lower_to_upper_wall_heat_W: float


@dataclass(frozen=True)
class StratifiedTankStepResult:
    time_s: float
    state: StratifiedTankState
    thermo: StratifiedThermoState
    cumulative_boundary_mass_kg: float
    cumulative_boundary_energy_J: float
    total_mass_residual_kg: float
    total_energy_residual_J: float


FlowCallback = Callable[
    [float, StratifiedThermoState],
    tuple[Iterable[MassEnergyFlow], Iterable[MassEnergyFlow]],
]
WallHeatCallback = Callable[[float, StratifiedThermoState], tuple[float, float]]


class StratifiedTank:
    """Two-region hydrogen inventory coupled to two horizontal wall zones."""

    def __init__(
        self,
        shape: HorizontalVesselGeometry,
        parameters: StratifiedTankParameters,
        properties: HydrogenProperties | None = None,
    ) -> None:
        if parameters.wall_zone_split_height_m > shape.diameter:
            raise ValueError("wall_zone_split_height_m must lie within the vessel")
        self.shape = shape
        self.parameters = parameters
        self.properties = properties or HydrogenProperties()
        split = shape.at_height(parameters.wall_zone_split_height_m)
        self._lower_wall_area_m2 = split.wetted_inner_area_m2
        self._upper_wall_area_m2 = shape.inner_surface_area_m2 - self._lower_wall_area_m2

    def initialize_saturated(
        self,
        pressure_Pa: float,
        liquid_volume_fraction: float,
        lower_wall_temperature_K: float | None = None,
        upper_wall_temperature_K: float | None = None,
    ) -> StratifiedTankState:
        if not 0.0 < liquid_volume_fraction < 1.0:
            raise ValueError("liquid_volume_fraction must be between zero and one")
        liquid = self.properties.saturated_liquid(pressure_Pa)
        vapor = self.properties.saturated_vapor(pressure_Pa)
        liquid_volume = self.shape.volume_m3 * liquid_volume_fraction
        vapor_volume = self.shape.volume_m3 - liquid_volume
        liquid_mass = liquid.density_kg_m3 * liquid_volume
        vapor_mass = vapor.density_kg_m3 * vapor_volume
        wall_temperature = liquid.temperature_K
        return StratifiedTankState(
            liquid_mass,
            liquid_mass * liquid.specific_internal_energy_J_kg,
            vapor_mass,
            vapor_mass * vapor.specific_internal_energy_J_kg,
            wall_temperature if lower_wall_temperature_K is None else lower_wall_temperature_K,
            wall_temperature if upper_wall_temperature_K is None else upper_wall_temperature_K,
        )

    def _volume_residual(
        self,
        pressure_Pa: float,
        state: StratifiedTankState,
    ) -> tuple[float, ThermoState, ThermoState]:
        liquid = self.properties.from_pu(
            pressure_Pa, state.liquid_internal_energy_J / state.liquid_mass_kg
        )
        vapor = self.properties.from_pu(
            pressure_Pa, state.vapor_internal_energy_J / state.vapor_mass_kg
        )
        volume = (
            state.liquid_mass_kg / liquid.density_kg_m3
            + state.vapor_mass_kg / vapor.density_kg_m3
        )
        return volume - self.shape.volume_m3, liquid, vapor

    def thermo(
        self,
        state: StratifiedTankState,
        pressure_hint_Pa: float | None = None,
    ) -> StratifiedThermoState:
        """Recover common pressure from both phase energies and rigid volume."""

        pmin = self.parameters.minimum_pressure_Pa
        pmax = self.parameters.maximum_pressure_Pa
        def build(pressure: float) -> StratifiedThermoState:
            residual, liquid, vapor = self._volume_residual(pressure, state)
            if liquid.density_kg_m3 <= vapor.density_kg_m3:
                raise ValueError("Recovered liquid density must exceed vapor density")
            liquid_volume = state.liquid_mass_kg / liquid.density_kg_m3
            vapor_volume = state.vapor_mass_kg / vapor.density_kg_m3
            height = self.shape.height_from_liquid_volume(liquid_volume)
            geometry = self.shape.at_height(height)
            lower_wet = min(geometry.wetted_inner_area_m2, self._lower_wall_area_m2)
            upper_wet = max(0.0, geometry.wetted_inner_area_m2 - self._lower_wall_area_m2)
            return StratifiedThermoState(
                pressure_Pa=pressure,
                liquid=liquid,
                vapor=vapor,
                liquid_volume_m3=liquid_volume,
                vapor_volume_m3=vapor_volume,
                liquid_height_m=height,
                interface_area_m2=geometry.interface_area_m2,
                lower_wet_area_m2=lower_wet,
                lower_dry_area_m2=self._lower_wall_area_m2 - lower_wet,
                upper_wet_area_m2=upper_wet,
                upper_dry_area_m2=self._upper_wall_area_m2 - upper_wet,
            )

        target = pressure_hint_Pa
        if target is None:
            target = math.sqrt(pmin * pmax)
        target = min(max(float(target), pmin), pmax)
        try:
            target_residual = self._volume_residual(target, state)[0]
            if abs(target_residual) <= max(self.shape.volume_m3, 1.0) * 1e-11:
                return build(target)
        except (ValueError, RuntimeError, OverflowError):
            target_residual = math.nan

        # Dynamic integration normally supplies the preceding pressure. Expand
        # around that value first so a time step needs only a few EOS flashes.
        local: list[tuple[float, float]] = []
        if math.isfinite(target_residual):
            local.append((target, target_residual))
        lower = upper = target
        for _ in range(28):
            lower = max(pmin, lower / 1.25)
            upper = min(pmax, upper * 1.25)
            for pressure in (lower, upper):
                if any(existing[0] == pressure for existing in local):
                    continue
                try:
                    residual = self._volume_residual(pressure, state)[0]
                except (ValueError, RuntimeError, OverflowError):
                    continue
                if math.isfinite(residual):
                    local.append((pressure, residual))
            local.sort()
            for (pa, fa), (pb, fb) in zip(local, local[1:]):
                if fa * fb < 0.0:
                    pressure = float(brentq(
                        lambda p: self._volume_residual(p, state)[0],
                        pa, pb, xtol=1e-7, rtol=1e-11,
                    ))
                    return build(pressure)
            if lower == pmin and upper == pmax:
                break

        samples: list[tuple[float, float]] = []
        for pressure in np.geomspace(pmin, pmax, 120):
            try:
                residual, _, _ = self._volume_residual(float(pressure), state)
            except (ValueError, RuntimeError, OverflowError):
                continue
            if math.isfinite(residual):
                samples.append((float(pressure), residual))
        brackets: list[tuple[float, float]] = []
        for (pa, fa), (pb, fb) in zip(samples, samples[1:]):
            if fa == 0.0:
                brackets.append((pa, pa))
            elif fa * fb < 0.0:
                brackets.append((pa, pb))
        if samples and samples[-1][1] == 0.0:
            brackets.append((samples[-1][0], samples[-1][0]))
        if not brackets:
            raise ValueError(
                "No common-pressure solution satisfies the rigid tank volume within the configured pressure range"
            )
        bracket = min(brackets, key=lambda item: abs(math.log(math.sqrt(item[0] * item[1]) / target)))
        if bracket[0] == bracket[1]:
            pressure = bracket[0]
        else:
            pressure = float(brentq(
                lambda p: self._volume_residual(p, state)[0],
                bracket[0], bracket[1], xtol=1e-7, rtol=1e-11,
            ))
        return build(pressure)

    @staticmethod
    def _flow_balance(
        flows: Iterable[MassEnergyFlow], owner: ThermoState
    ) -> tuple[float, float]:
        mass_rate = 0.0
        enthalpy_rate = 0.0
        for flow in flows:
            if not math.isfinite(flow.mass_flow_kg_s) or not math.isfinite(flow.specific_enthalpy_J_kg):
                raise ValueError("Flows must contain finite mass flow and enthalpy")
            mass_rate += flow.mass_flow_kg_s
            enthalpy = (
                flow.specific_enthalpy_J_kg
                if flow.mass_flow_kg_s >= 0.0
                else owner.specific_enthalpy_J_kg
            )
            enthalpy_rate += flow.mass_flow_kg_s * enthalpy
        return mass_rate, enthalpy_rate

    def derivative(
        self,
        state: StratifiedTankState,
        liquid_flows: Iterable[MassEnergyFlow],
        vapor_flows: Iterable[MassEnergyFlow],
        ambient_temperature_K: float,
        direct_lower_wall_heat_W: float = 0.0,
        direct_upper_wall_heat_W: float = 0.0,
        pressure_hint_Pa: float | None = None,
        direct_liquid_heat_W: float = 0.0,
        direct_vapor_heat_W: float = 0.0,
    ) -> StratifiedTankDerivative:
        if not math.isfinite(ambient_temperature_K) or ambient_temperature_K <= 0.0:
            raise ValueError("ambient_temperature_K must be finite and positive")
        direct_heats = (
            direct_lower_wall_heat_W, direct_upper_wall_heat_W,
            direct_liquid_heat_W, direct_vapor_heat_W,
        )
        if any(not math.isfinite(value) for value in direct_heats):
            raise ValueError("Direct heat inputs must be finite")
        thermo = self.thermo(state, pressure_hint_Pa)
        return self._derivative_from_thermo(
            state, thermo, liquid_flows, vapor_flows, ambient_temperature_K,
            direct_lower_wall_heat_W, direct_upper_wall_heat_W,
            direct_liquid_heat_W, direct_vapor_heat_W,
        )

    def _derivative_from_thermo(
        self,
        state: StratifiedTankState,
        thermo: StratifiedThermoState,
        liquid_flows: Iterable[MassEnergyFlow],
        vapor_flows: Iterable[MassEnergyFlow],
        ambient_temperature_K: float,
        direct_lower_wall_heat_W: float,
        direct_upper_wall_heat_W: float,
        direct_liquid_heat_W: float,
        direct_vapor_heat_W: float,
    ) -> StratifiedTankDerivative:
        p = self.parameters
        liquid_mass_rate, liquid_enthalpy_rate = self._flow_balance(liquid_flows, thermo.liquid)
        vapor_mass_rate, vapor_enthalpy_rate = self._flow_balance(vapor_flows, thermo.vapor)

        q_lower_liquid = p.wall_liquid_U_W_m2K * thermo.lower_wet_area_m2 * (
            state.lower_wall_temperature_K - thermo.liquid.temperature_K
        )
        q_lower_vapor = p.wall_vapor_U_W_m2K * thermo.lower_dry_area_m2 * (
            state.lower_wall_temperature_K - thermo.vapor.temperature_K
        )
        q_upper_liquid = p.wall_liquid_U_W_m2K * thermo.upper_wet_area_m2 * (
            state.upper_wall_temperature_K - thermo.liquid.temperature_K
        )
        q_upper_vapor = p.wall_vapor_U_W_m2K * thermo.upper_dry_area_m2 * (
            state.upper_wall_temperature_K - thermo.vapor.temperature_K
        )

        saturated_liquid = self.properties.saturated_liquid(thermo.pressure_Pa)
        saturated_vapor = self.properties.saturated_vapor(thermo.pressure_Pa)
        latent_heat = (
            saturated_vapor.specific_enthalpy_J_kg
            - saturated_liquid.specific_enthalpy_J_kg
        )
        if latent_heat <= 0.0:
            raise ValueError("Positive latent heat is required for phase change")
        q_liquid_interface = p.liquid_interface_UA_W_K * (
            thermo.liquid.temperature_K - saturated_liquid.temperature_K
        )
        q_vapor_interface = p.vapor_interface_UA_W_K * (
            thermo.vapor.temperature_K - saturated_vapor.temperature_K
        )
        evaporation = (q_liquid_interface + q_vapor_interface) / latent_heat

        q_lower_ambient = p.lower_ambient_UA_W_K * (
            ambient_temperature_K - state.lower_wall_temperature_K
        )
        q_upper_ambient = p.upper_ambient_UA_W_K * (
            ambient_temperature_K - state.upper_wall_temperature_K
        )
        q_lower_to_upper = p.wall_axial_UA_W_K * (
            state.lower_wall_temperature_K - state.upper_wall_temperature_K
        )

        return StratifiedTankDerivative(
            liquid_mass_kg_s=liquid_mass_rate - evaporation,
            liquid_internal_energy_W=(
                liquid_enthalpy_rate + q_lower_liquid + q_upper_liquid + direct_liquid_heat_W
                - q_liquid_interface - evaporation * saturated_liquid.specific_enthalpy_J_kg
            ),
            vapor_mass_kg_s=vapor_mass_rate + evaporation,
            vapor_internal_energy_W=(
                vapor_enthalpy_rate + q_lower_vapor + q_upper_vapor + direct_vapor_heat_W
                - q_vapor_interface + evaporation * saturated_vapor.specific_enthalpy_J_kg
            ),
            lower_wall_temperature_K_s=(
                q_lower_ambient + direct_lower_wall_heat_W
                - q_lower_liquid - q_lower_vapor - q_lower_to_upper
            ) / p.lower_wall_heat_capacity_J_K,
            upper_wall_temperature_K_s=(
                q_upper_ambient + direct_upper_wall_heat_W
                - q_upper_liquid - q_upper_vapor + q_lower_to_upper
            ) / p.upper_wall_heat_capacity_J_K,
            evaporation_rate_kg_s=evaporation,
            lower_ambient_heat_W=q_lower_ambient,
            upper_ambient_heat_W=q_upper_ambient,
            lower_wall_to_liquid_heat_W=q_lower_liquid,
            lower_wall_to_vapor_heat_W=q_lower_vapor,
            upper_wall_to_liquid_heat_W=q_upper_liquid,
            upper_wall_to_vapor_heat_W=q_upper_vapor,
            liquid_to_interface_heat_W=q_liquid_interface,
            vapor_to_interface_heat_W=q_vapor_interface,
            lower_to_upper_wall_heat_W=q_lower_to_upper,
        )

    @staticmethod
    def _advance(
        state: StratifiedTankState,
        derivative: StratifiedTankDerivative,
        dt_s: float,
    ) -> StratifiedTankState:
        return StratifiedTankState(
            state.liquid_mass_kg + derivative.liquid_mass_kg_s * dt_s,
            state.liquid_internal_energy_J + derivative.liquid_internal_energy_W * dt_s,
            state.vapor_mass_kg + derivative.vapor_mass_kg_s * dt_s,
            state.vapor_internal_energy_J + derivative.vapor_internal_energy_W * dt_s,
            state.lower_wall_temperature_K + derivative.lower_wall_temperature_K_s * dt_s,
            state.upper_wall_temperature_K + derivative.upper_wall_temperature_K_s * dt_s,
        )

    def _total_energy_J(self, state: StratifiedTankState) -> float:
        p = self.parameters
        return (
            state.liquid_internal_energy_J + state.vapor_internal_energy_J
            + p.lower_wall_heat_capacity_J_K * state.lower_wall_temperature_K
            + p.upper_wall_heat_capacity_J_K * state.upper_wall_temperature_K
        )

    def simulate(
        self,
        initial: StratifiedTankState,
        duration_s: float,
        time_step_s: float,
        ambient_temperature_K: float | Callable[[float], float],
        flow_callback: FlowCallback,
        wall_heat_callback: WallHeatCallback | None = None,
        fluid_heat_callback: WallHeatCallback | None = None,
    ) -> list[StratifiedTankStepResult]:
        """Advance with RK4 and independent mass/energy boundary ledgers."""

        if duration_s <= 0.0 or time_step_s <= 0.0:
            raise ValueError("duration_s and time_step_s must be positive")
        state = initial
        initial_mass = initial.liquid_mass_kg + initial.vapor_mass_kg
        initial_energy = self._total_energy_J(initial)
        cumulative_mass = 0.0
        cumulative_energy = 0.0
        results: list[StratifiedTankStepResult] = []
        pressure_hint = self.thermo(initial).pressure_Pa

        def evaluate(
            time: float,
            candidate: StratifiedTankState,
            hint: float,
        ) -> tuple[StratifiedTankDerivative, float, float, float]:
            thermo = self.thermo(candidate, hint)
            liquid_flows_iter, vapor_flows_iter = flow_callback(time, thermo)
            liquid_flows = tuple(liquid_flows_iter)
            vapor_flows = tuple(vapor_flows_iter)
            ambient = ambient_temperature_K(time) if callable(ambient_temperature_K) else ambient_temperature_K
            lower_heat, upper_heat = (0.0, 0.0)
            if wall_heat_callback is not None:
                lower_heat, upper_heat = wall_heat_callback(time, thermo)
            liquid_heat, vapor_heat = (0.0, 0.0)
            if fluid_heat_callback is not None:
                liquid_heat, vapor_heat = fluid_heat_callback(time, thermo)
            derivative = self._derivative_from_thermo(
                candidate, thermo, liquid_flows, vapor_flows, ambient,
                lower_heat, upper_heat, liquid_heat, vapor_heat,
            )
            lm, le = self._flow_balance(liquid_flows, thermo.liquid)
            vm, ve = self._flow_balance(vapor_flows, thermo.vapor)
            boundary_energy = (
                le + ve + derivative.lower_ambient_heat_W + derivative.upper_ambient_heat_W
                + lower_heat + upper_heat + liquid_heat + vapor_heat
            )
            return derivative, lm + vm, boundary_energy, thermo.pressure_Pa

        time = 0.0
        while time < duration_s - 1e-12:
            dt = min(time_step_s, duration_s - time)
            k1, m1, e1, p1 = evaluate(time, state, pressure_hint)
            s2 = self._advance(state, k1, dt / 2.0)
            k2, m2, e2, p2 = evaluate(time + dt / 2.0, s2, p1)
            s3 = self._advance(state, k2, dt / 2.0)
            k3, m3, e3, p3 = evaluate(time + dt / 2.0, s3, p2)
            s4 = self._advance(state, k3, dt)
            k4, m4, e4, p4 = evaluate(time + dt, s4, p3)
            state = StratifiedTankState(
                state.liquid_mass_kg + dt * (k1.liquid_mass_kg_s + 2*k2.liquid_mass_kg_s + 2*k3.liquid_mass_kg_s + k4.liquid_mass_kg_s) / 6.0,
                state.liquid_internal_energy_J + dt * (k1.liquid_internal_energy_W + 2*k2.liquid_internal_energy_W + 2*k3.liquid_internal_energy_W + k4.liquid_internal_energy_W) / 6.0,
                state.vapor_mass_kg + dt * (k1.vapor_mass_kg_s + 2*k2.vapor_mass_kg_s + 2*k3.vapor_mass_kg_s + k4.vapor_mass_kg_s) / 6.0,
                state.vapor_internal_energy_J + dt * (k1.vapor_internal_energy_W + 2*k2.vapor_internal_energy_W + 2*k3.vapor_internal_energy_W + k4.vapor_internal_energy_W) / 6.0,
                state.lower_wall_temperature_K + dt * (k1.lower_wall_temperature_K_s + 2*k2.lower_wall_temperature_K_s + 2*k3.lower_wall_temperature_K_s + k4.lower_wall_temperature_K_s) / 6.0,
                state.upper_wall_temperature_K + dt * (k1.upper_wall_temperature_K_s + 2*k2.upper_wall_temperature_K_s + 2*k3.upper_wall_temperature_K_s + k4.upper_wall_temperature_K_s) / 6.0,
            )
            cumulative_mass += dt * (m1 + 2*m2 + 2*m3 + m4) / 6.0
            cumulative_energy += dt * (e1 + 2*e2 + 2*e3 + e4) / 6.0
            time += dt
            thermo = self.thermo(state, p4)
            pressure_hint = thermo.pressure_Pa
            total_mass = state.liquid_mass_kg + state.vapor_mass_kg
            results.append(StratifiedTankStepResult(
                time_s=time,
                state=state,
                thermo=thermo,
                cumulative_boundary_mass_kg=cumulative_mass,
                cumulative_boundary_energy_J=cumulative_energy,
                total_mass_residual_kg=total_mass - initial_mass - cumulative_mass,
                total_energy_residual_J=self._total_energy_J(state) - initial_energy - cumulative_energy,
            ))
        return results
