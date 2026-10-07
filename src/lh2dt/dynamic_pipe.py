"""Finite-volume transient homogeneous-equilibrium pipe model.

The steady :class:`~lh2dt.pipe.HEMPipe` is sufficient for a pressure-network
screening solve, but cooldown and flashing require a finite fluid inventory.
This module adds one explicit control volume with mass, internal energy, and
wall temperature states.  The two half-length HEM hydraulic sections provide
the inlet and outlet mass fluxes; no slip, phase-separation, or plant-data
fitting is hidden in the closure.

The component exposes its two hydraulic half-segments and a dynamic-node
adapter can attach them to the reusable pressure/enthalpy network without
changing the conservation equations here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .pipe import HEMPipe
from .properties import HydrogenProperties, ThermoState
from .streams import MassEnergyFlow


@dataclass(frozen=True)
class DynamicHEMPipeState:
    """Finite-volume fluid and wall state."""

    mass_kg: float
    internal_energy_J: float
    wall_temperature_K: float


@dataclass(frozen=True)
class DynamicHEMPipeDerivative:
    """Conservative state derivative and its explicit flux terms."""

    mass_kg_s: float
    internal_energy_W: float
    wall_temperature_K_s: float
    mass_flow_in_kg_s: float
    mass_flow_out_kg_s: float
    inlet_enthalpy_flow_W: float
    outlet_enthalpy_flow_W: float
    heat_from_wall_W: float
    heat_from_ambient_W: float

    @property
    def ambient_heat_W(self) -> float:
        """Expose the ambient source under the dynamic-node ledger contract."""

        return self.heat_from_ambient_W


@dataclass(frozen=True)
class DynamicHEMPipeStepResult:
    """One RK4 update and equation-closure diagnostics."""

    state: DynamicHEMPipeState
    fluid: ThermoState
    derivative: DynamicHEMPipeDerivative
    mass_balance_residual_kg_s: float
    energy_balance_residual_W: float
    wall_energy_balance_residual_W: float


class DynamicHEMPipe:
    """A transient finite-volume HEM pipe segment.

    ``fluid_volume_m3`` defaults to the cylindrical geometric volume.  When a
    confirmed effective volume is available it may be supplied explicitly;
    the model never derives it from a time series.  The wall exchanges
    heat with the fluid through ``fluid_wall_UA_W_K`` and with the ambient
    reservoir through ``ambient_UA_W_K``.  Both conductances are explicit
    design or literature parameters.
    """

    def __init__(
        self,
        length_m: float,
        inner_diameter_m: float,
        roughness_m: float,
        wall_heat_capacity_J_K: float,
        *,
        local_loss_coefficient: float = 0.0,
        fluid_wall_UA_W_K: float = 0.0,
        ambient_temperature_K: float = 300.0,
        ambient_UA_W_K: float = 0.0,
        fluid_volume_m3: float | None = None,
        properties: HydrogenProperties | None = None,
    ) -> None:
        values = {
            "length_m": length_m,
            "inner_diameter_m": inner_diameter_m,
            "wall_heat_capacity_J_K": wall_heat_capacity_J_K,
            "ambient_temperature_K": ambient_temperature_K,
        }
        for name, value in values.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric, not bool")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in {
            "roughness_m": roughness_m,
            "local_loss_coefficient": local_loss_coefficient,
            "fluid_wall_UA_W_K": fluid_wall_UA_W_K,
            "ambient_UA_W_K": ambient_UA_W_K,
        }.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric, not bool")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if fluid_volume_m3 is not None:
            if isinstance(fluid_volume_m3, bool):
                raise ValueError("fluid_volume_m3 must be numeric, not bool")
            try:
                fluid_volume = float(fluid_volume_m3)
            except (TypeError, ValueError) as exc:
                raise ValueError("fluid_volume_m3 must be numeric") from exc
            if not math.isfinite(fluid_volume) or fluid_volume <= 0.0:
                raise ValueError("fluid_volume_m3 must be finite and positive when supplied")

        self.properties = properties or HydrogenProperties()
        self.length = float(length_m)
        self.diameter = float(inner_diameter_m)
        self.roughness = float(roughness_m)
        self.local_k = float(local_loss_coefficient)
        self.wall_heat_capacity = float(wall_heat_capacity_J_K)
        self.fluid_wall_UA = float(fluid_wall_UA_W_K)
        self.ambient_temperature = float(ambient_temperature_K)
        self.ambient_UA = float(ambient_UA_W_K)
        area = math.pi * self.diameter**2 / 4.0
        self.fluid_volume = float(fluid_volume_m3) if fluid_volume_m3 is not None else area * self.length
        half_length = self.length / 2.0
        self._inlet_pipe = HEMPipe(
            half_length,
            self.diameter,
            self.roughness,
            local_loss_coefficient=self.local_k / 2.0,
            properties=self.properties,
        )
        self._outlet_pipe = HEMPipe(
            half_length,
            self.diameter,
            self.roughness,
            local_loss_coefficient=self.local_k / 2.0,
            properties=self.properties,
        )

    @property
    def inlet_hydraulic_component(self) -> HEMPipe:
        """Return the explicit hydraulic half-segment at the inlet port."""

        return self._inlet_pipe

    @property
    def outlet_hydraulic_component(self) -> HEMPipe:
        """Return the explicit hydraulic half-segment at the outlet port."""

        return self._outlet_pipe

    def initialize(
        self,
        fluid: ThermoState,
        *,
        wall_temperature_K: float | None = None,
    ) -> DynamicHEMPipeState:
        """Create a finite-volume state from an explicit thermodynamic state."""

        if wall_temperature_K is None:
            wall_temperature = fluid.temperature_K
        else:
            if isinstance(wall_temperature_K, bool):
                raise ValueError("wall_temperature_K must be numeric, not bool")
            try:
                wall_temperature = float(wall_temperature_K)
            except (TypeError, ValueError) as exc:
                raise ValueError("wall_temperature_K must be numeric") from exc
        if not math.isfinite(wall_temperature) or wall_temperature <= 0.0:
            raise ValueError("wall_temperature_K must be finite and positive")
        mass = self.fluid_volume * fluid.density_kg_m3
        return DynamicHEMPipeState(
            mass_kg=mass,
            internal_energy_J=mass * fluid.specific_internal_energy_J_kg,
            wall_temperature_K=wall_temperature,
        )

    def thermo(self, state: DynamicHEMPipeState) -> ThermoState:
        """Recover the EOS state from conserved mass and internal energy."""

        if not math.isfinite(state.mass_kg) or state.mass_kg <= 0.0:
            raise ValueError("dynamic pipe mass must be finite and positive")
        if not math.isfinite(state.internal_energy_J):
            raise ValueError("dynamic pipe internal energy must be finite")
        return self.properties.from_rho_u(
            state.mass_kg / self.fluid_volume,
            state.internal_energy_J / state.mass_kg,
        )

    @staticmethod
    def _same_thermo(left: ThermoState, right: ThermoState) -> bool:
        """Identify the same endpoint state within EOS round-off precision.

        CoolProp can return a sub-nanobar pressure difference when the same
        two-phase state is reconstructed from ``rho`` and ``u``.  Passing that
        numerical noise into a square-root friction law creates a spurious
        flow and can destabilise an otherwise stationary explicit step.  The
        tolerance is a numerical precision guard, not a fitted flow threshold.
        """

        return (
            abs(left.pressure_Pa - right.pressure_Pa)
            <= 1.0e-10 * max(1.0, abs(left.pressure_Pa), abs(right.pressure_Pa))
            and abs(left.specific_enthalpy_J_kg - right.specific_enthalpy_J_kg)
            <= 1.0e-10
            * max(1.0, abs(left.specific_enthalpy_J_kg), abs(right.specific_enthalpy_J_kg))
        )

    @staticmethod
    def _advance(
        state: DynamicHEMPipeState,
        derivative: DynamicHEMPipeDerivative,
        time_step_s: float,
    ) -> DynamicHEMPipeState:
        result = DynamicHEMPipeState(
            mass_kg=state.mass_kg + derivative.mass_kg_s * time_step_s,
            internal_energy_J=state.internal_energy_J + derivative.internal_energy_W * time_step_s,
            wall_temperature_K=state.wall_temperature_K + derivative.wall_temperature_K_s * time_step_s,
        )
        if result.mass_kg <= 0.0 or not math.isfinite(result.mass_kg):
            raise ValueError("dynamic pipe step would produce non-positive mass")
        if not math.isfinite(result.internal_energy_J) or result.wall_temperature_K <= 0.0:
            raise ValueError("dynamic pipe step produced an invalid state")
        return result

    def derivative(
        self,
        state: DynamicHEMPipeState,
        inlet: ThermoState,
        outlet: ThermoState,
    ) -> DynamicHEMPipeDerivative:
        """Evaluate conservative fluxes for explicit endpoint states."""

        fluid = self.thermo(state)
        inlet_link = None if self._same_thermo(inlet, fluid) else self._inlet_pipe.evaluate(inlet, fluid)
        outlet_link = None if self._same_thermo(fluid, outlet) else self._outlet_pipe.evaluate(fluid, outlet)
        mass_in = 0.0 if inlet_link is None else float(inlet_link.mass_flow_kg_s)
        mass_out = 0.0 if outlet_link is None else float(outlet_link.mass_flow_kg_s)
        inlet_upstream_h = inlet.specific_enthalpy_J_kg if mass_in >= 0.0 else fluid.specific_enthalpy_J_kg
        outlet_upstream_h = fluid.specific_enthalpy_J_kg if mass_out >= 0.0 else outlet.specific_enthalpy_J_kg
        inlet_energy = mass_in * inlet_upstream_h
        outlet_energy = mass_out * outlet_upstream_h
        heat_from_wall = self.fluid_wall_UA * (state.wall_temperature_K - fluid.temperature_K)
        heat_from_ambient = self.ambient_UA * (self.ambient_temperature - state.wall_temperature_K)
        return DynamicHEMPipeDerivative(
            mass_kg_s=mass_in - mass_out,
            internal_energy_W=inlet_energy - outlet_energy + heat_from_wall,
            wall_temperature_K_s=(heat_from_ambient - heat_from_wall) / self.wall_heat_capacity,
            mass_flow_in_kg_s=mass_in,
            mass_flow_out_kg_s=mass_out,
            inlet_enthalpy_flow_W=inlet_energy,
            outlet_enthalpy_flow_W=outlet_energy,
            heat_from_wall_W=heat_from_wall,
            heat_from_ambient_W=heat_from_ambient,
        )

    def derivative_from_flows(
        self,
        state: DynamicHEMPipeState,
        flows: Iterable[MassEnergyFlow],
    ) -> DynamicHEMPipeDerivative:
        """Evaluate the same balance from signed network boundary streams.

        A dynamic pipe is represented in a network as a stateful boundary node
        with its two explicit hydraulic half-segments attached.  The network
        owns the pressure solve; this method owns the finite-volume inventory
        and wall balance.  Outflow enthalpy is always taken from the pipe's
        current thermodynamic state, preserving the control-volume upwind
        contract even if a caller supplies inconsistent metadata.
        """

        fluid = self.thermo(state)
        mass_rate = 0.0
        boundary_energy = 0.0
        incoming_mass = 0.0
        outgoing_mass = 0.0
        incoming_energy = 0.0
        outgoing_energy = 0.0
        for flow in flows:
            mass = float(flow.mass_flow_kg_s)
            if not math.isfinite(mass) or not math.isfinite(flow.specific_enthalpy_J_kg):
                raise ValueError("dynamic pipe boundary flows must be finite")
            mass_rate += mass
            if mass >= 0.0:
                incoming_mass += mass
                energy = mass * flow.specific_enthalpy_J_kg
                incoming_energy += energy
                boundary_energy += energy
            else:
                outgoing = -mass
                outgoing_mass += outgoing
                energy = outgoing * fluid.specific_enthalpy_J_kg
                outgoing_energy += energy
                boundary_energy -= energy
        heat_from_wall = self.fluid_wall_UA * (state.wall_temperature_K - fluid.temperature_K)
        heat_from_ambient = self.ambient_UA * (self.ambient_temperature - state.wall_temperature_K)
        return DynamicHEMPipeDerivative(
            mass_kg_s=mass_rate,
            internal_energy_W=boundary_energy + heat_from_wall,
            wall_temperature_K_s=(heat_from_ambient - heat_from_wall) / self.wall_heat_capacity,
            mass_flow_in_kg_s=incoming_mass,
            mass_flow_out_kg_s=outgoing_mass,
            inlet_enthalpy_flow_W=incoming_energy,
            outlet_enthalpy_flow_W=outgoing_energy,
            heat_from_wall_W=heat_from_wall,
            heat_from_ambient_W=heat_from_ambient,
        )

    def step(
        self,
        state: DynamicHEMPipeState,
        inlet: ThermoState,
        outlet: ThermoState,
        time_step_s: float,
    ) -> DynamicHEMPipeStepResult:
        """Advance one RK4 step and report the local conservation closures."""

        if not math.isfinite(time_step_s) or time_step_s <= 0.0:
            raise ValueError("time_step_s must be finite and positive")
        k1 = self.derivative(state, inlet, outlet)
        k2 = self.derivative(self._advance(state, k1, time_step_s / 2.0), inlet, outlet)
        k3 = self.derivative(self._advance(state, k2, time_step_s / 2.0), inlet, outlet)
        k4 = self.derivative(self._advance(state, k3, time_step_s), inlet, outlet)
        combined = DynamicHEMPipeDerivative(
            mass_kg_s=(k1.mass_kg_s + 2.0 * k2.mass_kg_s + 2.0 * k3.mass_kg_s + k4.mass_kg_s) / 6.0,
            internal_energy_W=(k1.internal_energy_W + 2.0 * k2.internal_energy_W + 2.0 * k3.internal_energy_W + k4.internal_energy_W) / 6.0,
            wall_temperature_K_s=(k1.wall_temperature_K_s + 2.0 * k2.wall_temperature_K_s + 2.0 * k3.wall_temperature_K_s + k4.wall_temperature_K_s) / 6.0,
            mass_flow_in_kg_s=(k1.mass_flow_in_kg_s + 2.0 * k2.mass_flow_in_kg_s + 2.0 * k3.mass_flow_in_kg_s + k4.mass_flow_in_kg_s) / 6.0,
            mass_flow_out_kg_s=(k1.mass_flow_out_kg_s + 2.0 * k2.mass_flow_out_kg_s + 2.0 * k3.mass_flow_out_kg_s + k4.mass_flow_out_kg_s) / 6.0,
            inlet_enthalpy_flow_W=(k1.inlet_enthalpy_flow_W + 2.0 * k2.inlet_enthalpy_flow_W + 2.0 * k3.inlet_enthalpy_flow_W + k4.inlet_enthalpy_flow_W) / 6.0,
            outlet_enthalpy_flow_W=(k1.outlet_enthalpy_flow_W + 2.0 * k2.outlet_enthalpy_flow_W + 2.0 * k3.outlet_enthalpy_flow_W + k4.outlet_enthalpy_flow_W) / 6.0,
            heat_from_wall_W=(k1.heat_from_wall_W + 2.0 * k2.heat_from_wall_W + 2.0 * k3.heat_from_wall_W + k4.heat_from_wall_W) / 6.0,
            heat_from_ambient_W=(k1.heat_from_ambient_W + 2.0 * k2.heat_from_ambient_W + 2.0 * k3.heat_from_ambient_W + k4.heat_from_ambient_W) / 6.0,
        )
        next_state = self._advance(state, combined, time_step_s)
        fluid = self.thermo(next_state)
        final_derivative = self.derivative(next_state, inlet, outlet)
        mass_residual = final_derivative.mass_kg_s - (
            final_derivative.mass_flow_in_kg_s - final_derivative.mass_flow_out_kg_s
        )
        energy_residual = final_derivative.internal_energy_W - (
            final_derivative.inlet_enthalpy_flow_W
            - final_derivative.outlet_enthalpy_flow_W
            + final_derivative.heat_from_wall_W
        )
        wall_residual = (
            self.wall_heat_capacity * final_derivative.wall_temperature_K_s
            - (final_derivative.heat_from_ambient_W - final_derivative.heat_from_wall_W)
        )
        return DynamicHEMPipeStepResult(
            state=next_state,
            fluid=fluid,
            derivative=final_derivative,
            mass_balance_residual_kg_s=mass_residual,
            energy_balance_residual_W=energy_residual,
            wall_energy_balance_residual_W=wall_residual,
        )


__all__ = [
    "DynamicHEMPipe",
    "DynamicHEMPipeDerivative",
    "DynamicHEMPipeState",
    "DynamicHEMPipeStepResult",
]
