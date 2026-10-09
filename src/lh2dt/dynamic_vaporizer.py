"""Finite-volume transient vaporizer control volume.

The steady :class:`~lh2dt.vaporizer.Vaporizer` is useful for a pressure
network, but a cooldown or warm-up procedure also needs a finite inventory
and a wall thermal state.  ``DynamicVaporizer`` keeps those states explicit:

* fluid mass and internal energy are conserved control-volume states;
* a lumped wall exchanges heat with the fluid and an ambient reservoir; and
* the heat-transfer cap is an explicit design input, never inferred from a
  plant time series.

The class is deliberately independent from the steady network solver.  A
caller may attach hydraulic half-segments through the same dynamic-node
adapter pattern used by ``DynamicHEMPipe`` without changing the balances in
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any, Iterable, Sequence

from .properties import HydrogenProperties, ThermoState
from .streams import MassEnergyFlow


@dataclass(frozen=True)
class DynamicVaporizerState:
    """Finite-volume fluid and wall state."""

    mass_kg: float
    internal_energy_J: float
    wall_temperature_K: float


@dataclass(frozen=True)
class DynamicVaporizerDerivative:
    """Conservative derivative and explicit heat/transport terms."""

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
class DynamicVaporizerStepResult:
    """One RK4 update and local conservation diagnostics."""

    state: DynamicVaporizerState
    fluid: ThermoState
    derivative: DynamicVaporizerDerivative
    mass_balance_residual_kg_s: float
    energy_balance_residual_W: float
    wall_energy_balance_residual_W: float


class DynamicVaporizer:
    """Transient heated control volume with explicit forward-flow contract.

    ``fluid_volume_m3`` and ``wall_heat_capacity_J_K`` are geometry/design
    inputs.  ``fluid_wall_UA_W_K`` and ``ambient_UA_W_K`` are replaceable
    closure parameters.  Heat from the wall is limited symmetrically by
    ``maximum_heat_W`` so a warm-up and a cooldown use the same conservation
    equation.  The hydraulic pressure drop remains a separate replaceable
    link, which keeps this class usable with different pipe and valve models.
    """

    def __init__(
        self,
        fluid_volume_m3: float,
        wall_heat_capacity_J_K: float,
        fluid_wall_UA_W_K: float,
        ambient_temperature_K: float,
        ambient_UA_W_K: float,
        maximum_heat_W: float,
        properties: HydrogenProperties | None = None,
    ) -> None:
        positive = {
            "fluid_volume_m3": fluid_volume_m3,
            "wall_heat_capacity_J_K": wall_heat_capacity_J_K,
            "maximum_heat_W": maximum_heat_W,
            "ambient_temperature_K": ambient_temperature_K,
        }
        for name, value in positive.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric, not bool")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in {
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

        self.fluid_volume = float(fluid_volume_m3)
        self.wall_heat_capacity = float(wall_heat_capacity_J_K)
        self.fluid_wall_UA = float(fluid_wall_UA_W_K)
        self.ambient_temperature = float(ambient_temperature_K)
        self.ambient_UA = float(ambient_UA_W_K)
        self.maximum_heat = float(maximum_heat_W)
        self.properties = properties or HydrogenProperties()

    def initialize(
        self,
        fluid: ThermoState,
        *,
        wall_temperature_K: float | None = None,
    ) -> DynamicVaporizerState:
        """Create a state from an explicit thermodynamic fluid state."""

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
        return DynamicVaporizerState(
            mass_kg=mass,
            internal_energy_J=mass * fluid.specific_internal_energy_J_kg,
            wall_temperature_K=wall_temperature,
        )

    def thermo(self, state: DynamicVaporizerState) -> ThermoState:
        """Recover the EOS state from conserved mass and internal energy."""

        if not math.isfinite(state.mass_kg) or state.mass_kg <= 0.0:
            raise ValueError("dynamic vaporizer mass must be finite and positive")
        if not math.isfinite(state.internal_energy_J):
            raise ValueError("dynamic vaporizer internal energy must be finite")
        if not math.isfinite(state.wall_temperature_K) or state.wall_temperature_K <= 0.0:
            raise ValueError("dynamic vaporizer wall temperature must be finite and positive")
        return self.properties.from_rho_u(
            state.mass_kg / self.fluid_volume,
            state.internal_energy_J / state.mass_kg,
        )

    def _heat_from_wall(self, fluid_temperature_K: float, wall_temperature_K: float) -> float:
        raw = self.fluid_wall_UA * (wall_temperature_K - fluid_temperature_K)
        return max(-self.maximum_heat, min(self.maximum_heat, raw))

    def derivative(
        self,
        state: DynamicVaporizerState,
        inlet: ThermoState,
        mass_flow_in_kg_s: float,
        mass_flow_out_kg_s: float,
        *,
        ambient_temperature_K: float | None = None,
    ) -> DynamicVaporizerDerivative:
        """Evaluate the forward-flow mass, energy, and wall balances."""

        values = {
            "mass_flow_in_kg_s": mass_flow_in_kg_s,
            "mass_flow_out_kg_s": mass_flow_out_kg_s,
            "ambient_temperature_K": self.ambient_temperature if ambient_temperature_K is None else ambient_temperature_K,
        }
        for name, value in values.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric, not bool")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        ambient_temperature = float(values["ambient_temperature_K"])
        if ambient_temperature <= 0.0:
            raise ValueError("ambient_temperature_K must be positive")

        fluid = self.thermo(state)
        mass_in = float(mass_flow_in_kg_s)
        mass_out = float(mass_flow_out_kg_s)
        heat_from_wall = self._heat_from_wall(fluid.temperature_K, state.wall_temperature_K)
        heat_from_ambient = self.ambient_UA * (ambient_temperature - state.wall_temperature_K)
        inlet_energy = mass_in * inlet.specific_enthalpy_J_kg
        outlet_energy = mass_out * fluid.specific_enthalpy_J_kg
        return DynamicVaporizerDerivative(
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
        state: DynamicVaporizerState,
        flows: Iterable[MassEnergyFlow],
        *,
        ambient_temperature_K: float | None = None,
        outlet_specific_enthalpy_J_kg: float | None = None,
    ) -> DynamicVaporizerDerivative:
        """Build the same balance from signed network boundary streams.

        Positive flow is into the vaporizer.  Incoming stream enthalpy is
        supplied by the stream; outgoing flow uses the current control-volume
        enthalpy, preserving an upwind transport contract.  A caller that
        represents several control volumes as one coarse mixed node may pass
        an explicit aggregate outlet enthalpy.  That option keeps the
        aggregate network energy ledger conservative; individual-port
        assemblies should leave it unset.
        """

        fluid = self.thermo(state)
        outlet_enthalpy = fluid.specific_enthalpy_J_kg
        if outlet_specific_enthalpy_J_kg is not None:
            if isinstance(outlet_specific_enthalpy_J_kg, bool):
                raise ValueError("outlet_specific_enthalpy_J_kg must be numeric, not bool")
            try:
                outlet_enthalpy = float(outlet_specific_enthalpy_J_kg)
            except (TypeError, ValueError) as exc:
                raise ValueError("outlet_specific_enthalpy_J_kg must be numeric") from exc
            if not math.isfinite(outlet_enthalpy):
                raise ValueError("outlet_specific_enthalpy_J_kg must be finite")
        mass_in = 0.0
        mass_out = 0.0
        inlet_energy = 0.0
        outlet_energy = 0.0
        for flow in flows:
            mass = float(flow.mass_flow_kg_s)
            enthalpy = float(flow.specific_enthalpy_J_kg)
            if not math.isfinite(mass) or not math.isfinite(enthalpy):
                raise ValueError("dynamic vaporizer boundary flows must be finite")
            if mass >= 0.0:
                mass_in += mass
                inlet_energy += mass * enthalpy
            else:
                outgoing = -mass
                mass_out += outgoing
                outlet_energy += outgoing * outlet_enthalpy
        base = self.derivative(
            state,
            fluid,
            mass_in,
            mass_out,
            ambient_temperature_K=ambient_temperature_K,
        )
        return DynamicVaporizerDerivative(
            mass_kg_s=base.mass_kg_s,
            internal_energy_W=inlet_energy - outlet_energy + base.heat_from_wall_W,
            wall_temperature_K_s=base.wall_temperature_K_s,
            mass_flow_in_kg_s=mass_in,
            mass_flow_out_kg_s=mass_out,
            inlet_enthalpy_flow_W=inlet_energy,
            outlet_enthalpy_flow_W=outlet_energy,
            heat_from_wall_W=base.heat_from_wall_W,
            heat_from_ambient_W=base.heat_from_ambient_W,
        )

    @staticmethod
    def _advance(
        state: DynamicVaporizerState,
        derivative: DynamicVaporizerDerivative,
        time_step_s: float,
    ) -> DynamicVaporizerState:
        result = DynamicVaporizerState(
            mass_kg=state.mass_kg + derivative.mass_kg_s * time_step_s,
            internal_energy_J=state.internal_energy_J + derivative.internal_energy_W * time_step_s,
            wall_temperature_K=state.wall_temperature_K + derivative.wall_temperature_K_s * time_step_s,
        )
        if result.mass_kg <= 0.0 or not math.isfinite(result.mass_kg):
            raise ValueError("dynamic vaporizer step would produce non-positive mass")
        if not math.isfinite(result.internal_energy_J) or result.wall_temperature_K <= 0.0:
            raise ValueError("dynamic vaporizer step produced an invalid state")
        return result

    def step(
        self,
        state: DynamicVaporizerState,
        inlet: ThermoState,
        mass_flow_in_kg_s: float,
        mass_flow_out_kg_s: float,
        time_step_s: float,
        *,
        ambient_temperature_K: float | None = None,
    ) -> DynamicVaporizerStepResult:
        """Advance one RK4 step and return conservation residuals."""

        if isinstance(time_step_s, bool):
            raise ValueError("time_step_s must be numeric")
        try:
            dt = float(time_step_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("time_step_s must be numeric") from exc
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("time_step_s must be finite and positive")

        def deriv(current: DynamicVaporizerState) -> DynamicVaporizerDerivative:
            return self.derivative(
                current,
                inlet,
                mass_flow_in_kg_s,
                mass_flow_out_kg_s,
                ambient_temperature_K=ambient_temperature_K,
            )

        k1 = deriv(state)
        k2 = deriv(self._advance(state, k1, dt / 2.0))
        k3 = deriv(self._advance(state, k2, dt / 2.0))
        k4 = deriv(self._advance(state, k3, dt))
        combined = DynamicVaporizerDerivative(
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
        next_state = self._advance(state, combined, dt)
        fluid = self.thermo(next_state)
        final_derivative = deriv(next_state)
        mass_residual = final_derivative.mass_kg_s - (
            final_derivative.mass_flow_in_kg_s - final_derivative.mass_flow_out_kg_s
        )
        energy_residual = final_derivative.internal_energy_W - (
            final_derivative.inlet_enthalpy_flow_W
            - final_derivative.outlet_enthalpy_flow_W
            + final_derivative.heat_from_wall_W
        )
        wall_residual = self.wall_heat_capacity * final_derivative.wall_temperature_K_s - (
            final_derivative.heat_from_ambient_W - final_derivative.heat_from_wall_W
        )
        return DynamicVaporizerStepResult(
            state=next_state,
            fluid=fluid,
            derivative=final_derivative,
            mass_balance_residual_kg_s=mass_residual,
            energy_balance_residual_W=energy_residual,
            wall_energy_balance_residual_W=wall_residual,
        )

    def export_accident_history(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Export one explicit atmospheric outlet history for PRISM.

        This is an accident boundary, not a normal ``evaluate``/``step``
        snapshot.  The caller must provide the failed opening, source state,
        ambient boundary and finite horizon.  A horizon that stops with
        residual inventory is labelled ``synthetic_diagnostic_horizon`` so a
        downstream QRA cannot mistake the bounded trace for a qualified
        inventory-depletion history.
        """
        if not isinstance(request, Mapping):
            raise ValueError("accident export request must be a mapping")

        def text(name: str) -> str:
            value = request.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            return value.strip()

        def number(name: str, default: float | None = None, *, positive: bool = True) -> float:
            value = request.get(name, default)
            if isinstance(value, bool) or value is None:
                raise ValueError(f"{name} must be numeric")
            try:
                result = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be numeric") from error
            if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
                comparator = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {comparator}")
            return result

        event_id = text("event_id")
        component_id = text("component_id")
        port_id = text("port_id")
        source_pressure = number("source_pressure_pa_abs", request.get("upstream_pressure_pa_abs"))
        source_temperature = number("source_temperature_k", request.get("upstream_temperature_k"))
        ambient_pressure = number("ambient_pressure_pa_abs", 101325.0)
        ambient_temperature = number("ambient_temperature_k", 288.15)
        opening_diameter = number("failure_opening_diameter_m")
        horizon = number("horizon_s")
        time_step = number("time_step_s", min(0.05, horizon))
        discharge_coefficient = number("discharge_coefficient", 0.8)
        stroke_time = number("stroke_time_s", max(time_step, 0.05))
        wall_temperature = number("wall_temperature_k", source_temperature)
        if horizon < time_step:
            time_step = horizon
        if source_pressure <= ambient_pressure:
            raise ValueError("source pressure must exceed ambient pressure for an outward accident")

        inlet = self.properties.from_pT(source_pressure, source_temperature)
        # CoolProp uses ``supercritical`` for states above both critical
        # coordinates, while its low-pressure gas branch is exposed as
        # ``supercritical_gas``.  Both are single-phase gas-like source
        # states for this atmospheric outlet contract; liquid-like states
        # remain rejected explicitly.
        if inlet.phase not in {"gas", "supercritical_gas", "supercritical"}:
            raise ValueError("DynamicVaporizer accident export requires a gas-like source state")
        state = self.initialize(inlet, wall_temperature_K=wall_temperature)
        initial_mass = float(state.mass_kg)
        outlet = self.properties.from_pT(ambient_pressure, ambient_temperature)
        from .valve import StrokeLimitedCommandedValve, Valve

        area = math.pi * opening_diameter**2 / 4.0
        valve = StrokeLimitedCommandedValve(
            Valve(area, discharge_coefficient, properties=self.properties, allow_reverse=False),
            stroke_time_s=stroke_time,
            initial_opening=0.0,
        )
        valve.set_target_opening(1.0)
        steps: list[dict[str, Any]] = []
        cumulative_mass = 0.0
        cumulative_enthalpy = 0.0
        elapsed = 0.0
        choked_count = 0
        depleted = False
        while elapsed < horizon - 1.0e-12:
            dt = min(time_step, horizon - elapsed)
            fluid = self.thermo(state)
            opening = valve.advance(dt)
            hydraulic = valve.evaluate(fluid, outlet)
            mass_flow = float(hydraulic.mass_flow_kg_s)
            if not math.isfinite(mass_flow) or mass_flow < 0.0:
                raise ValueError("accident outlet returned an invalid outward mass flow")
            exit_state = self.properties.from_ph(
                ambient_pressure, float(hydraulic.outlet_specific_enthalpy_J_kg)
            )
            if exit_state.phase not in {"gas", "supercritical_gas", "supercritical"}:
                raise ValueError("accident outlet is not gas-like at the declared ambient boundary")
            effective_area = area * opening
            enthalpy = float(hydraulic.outlet_specific_enthalpy_J_kg)
            density = float(exit_state.density_kg_m3)
            steps.append({
                "time_s": elapsed,
                "mass_flow_kg_s": mass_flow,
                "pressure_pa_abs": ambient_pressure,
                "specific_enthalpy_j_kg": enthalpy,
                "temperature_k": float(exit_state.temperature_K),
                "density_kg_m3": density,
                "effective_area_m2": effective_area,
                "velocity_m_s": (
                    mass_flow / (density * effective_area)
                    if mass_flow > 0.0 and effective_area > 0.0 else 0.0
                ),
                "area_provenance": "explicit atmospheric valve area scaled by stroke state",
                "velocity_origin": "mass_continuity_from_declared_area",
                "provider_source_state": {
                    "source_pressure_pa_abs": float(fluid.pressure_Pa),
                    "source_temperature_k": float(fluid.temperature_K),
                    "source_density_kg_m3": float(fluid.density_kg_m3),
                    "source_specific_enthalpy_j_kg": float(fluid.specific_enthalpy_J_kg),
                    "throat_pressure_pa_abs": float(hydraulic.throat_pressure_Pa),
                    "throat_mass_flux_kg_m2_s": float(hydraulic.mass_flux_kg_m2_s),
                    "choked": bool(hydraulic.choked),
                },
            })
            if hydraulic.choked:
                choked_count += 1
            result = self.step(state, inlet, 0.0, mass_flow, dt)
            cumulative_mass += mass_flow * dt
            cumulative_enthalpy += mass_flow * dt * enthalpy
            state = result.state
            elapsed += dt
            if state.mass_kg <= max(initial_mass * 1.0e-9, 1.0e-12):
                depleted = True
                break

        termination_basis = "inventory_depletion" if depleted else "synthetic_diagnostic_horizon"
        termination_provenance = (
            "DynamicVaporizer finite-volume mass state reached the explicit positive inventory threshold"
            if depleted else
            "DynamicVaporizer finite-volume state stopped at the caller-declared bounded horizon; residual inventory retained"
        )
        return {
            "provider_export_schema": "prism.external_accident_history.v1",
            "provider_model": f"{type(self).__module__}.{type(self).__qualname__}",
            "provider_source_digest": request.get("provider_source_digest"),
            "provider_state_snapshot_digest": request.get("state_snapshot_digest"),
            "data": {
                "event_kind": "accidental_leak",
                "event_id": event_id,
                "component_id": component_id,
                "port_id": port_id,
                "duration_s": elapsed,
                "available_mass_kg": initial_mass,
                "cumulative_mass_out_kg": cumulative_mass,
                "cumulative_static_enthalpy_out_j": cumulative_enthalpy,
                "termination_basis": termination_basis,
                "termination_provenance": termination_provenance,
                "fluid": "Hydrogen",
                "steps": steps,
                "provider_meta": {
                    "initial_mass_kg": initial_mass,
                    "final_mass_kg": float(state.mass_kg),
                    "residual_mass_kg": initial_mass - cumulative_mass - float(state.mass_kg),
                    "time_step_s": time_step,
                    "valve": {
                        "area_m2": area,
                        "discharge_coefficient": discharge_coefficient,
                        "stroke_time_s": stroke_time,
                        "final_opening": valve.opening,
                        "choked_intervals": choked_count,
                    },
                },
            },
        }


@dataclass(frozen=True)
class ParallelDynamicVaporizerState:
    """Explicit conserved state for every unit in a parallel vaporizer bank."""

    unit_states: tuple[DynamicVaporizerState, ...]


@dataclass(frozen=True)
class ParallelDynamicVaporizerDerivative:
    """Per-unit derivatives with aggregate conservative ledger properties."""

    unit_derivatives: tuple[DynamicVaporizerDerivative, ...]

    @property
    def mass_kg_s(self) -> float:
        return sum(item.mass_kg_s for item in self.unit_derivatives)

    @property
    def mass_flow_in_kg_s(self) -> float:
        """Aggregate positive boundary mass flow into the coarse node."""

        return sum(item.mass_flow_in_kg_s for item in self.unit_derivatives)

    @property
    def mass_flow_out_kg_s(self) -> float:
        """Aggregate positive boundary mass flow out of the coarse node."""

        return sum(item.mass_flow_out_kg_s for item in self.unit_derivatives)

    @property
    def internal_energy_W(self) -> float:
        return sum(item.internal_energy_W for item in self.unit_derivatives)

    @property
    def inlet_enthalpy_flow_W(self) -> float:
        """Aggregate incoming enthalpy transport rate."""

        return sum(item.inlet_enthalpy_flow_W for item in self.unit_derivatives)

    @property
    def outlet_enthalpy_flow_W(self) -> float:
        """Aggregate outgoing enthalpy transport rate."""

        return sum(item.outlet_enthalpy_flow_W for item in self.unit_derivatives)

    @property
    def heat_from_wall_W(self) -> float:
        """Aggregate wall-to-fluid heat rate."""

        return sum(item.heat_from_wall_W for item in self.unit_derivatives)

    @property
    def energy_balance_residual_W(self) -> float:
        """Residual of the aggregate fluid energy balance."""

        return self.internal_energy_W - (
            self.inlet_enthalpy_flow_W
            - self.outlet_enthalpy_flow_W
            + self.heat_from_wall_W
        )

    @property
    def ambient_heat_W(self) -> float:
        return sum(item.ambient_heat_W for item in self.unit_derivatives)


class ParallelDynamicVaporizer:
    """Aggregate transient vaporizer node with explicit replaceable units.

    A coarse network node may represent several vaporizers while each unit
    retains its own fluid inventory and wall temperature.  Hydraulic flow is
    split only by the explicit ``flow_weights`` and active mask; no operating
    split or capacity is inferred from observations.  A refined P&ID
    can expose the same units as individual :class:`DynamicVaporizer` nodes
    without changing their state or balance contracts.
    """

    def __init__(
        self,
        units: Sequence[DynamicVaporizer],
        *,
        unit_ids: Sequence[str] | None = None,
        active: Sequence[bool] | None = None,
        flow_weights: Sequence[float] | None = None,
    ) -> None:
        if not units:
            raise ValueError("at least one dynamic vaporizer unit is required")
        self.units = tuple(units)
        properties = self.units[0].properties
        if any(
            getattr(unit.properties, "fluid", None)
            != getattr(properties, "fluid", None)
            for unit in self.units[1:]
        ):
            raise ValueError("all dynamic vaporizer units must use the same fluid backend")
        ids = tuple(
            str(item).strip()
            for item in (unit_ids or (f"unit-{index + 1}" for index in range(len(self.units))))
        )
        if len(ids) != len(self.units) or any(not item for item in ids):
            raise ValueError("unit_ids must match units and be non-empty")
        if len(set(ids)) != len(ids):
            raise ValueError("unit_ids must be unique")
        if active is None:
            mask = tuple(True for _ in self.units)
        else:
            if isinstance(active, (str, bytes)) or len(active) != len(self.units):
                raise ValueError("active must match units")
            if any(not isinstance(item, bool) for item in active):
                raise ValueError("active entries must be bool")
            mask = tuple(active)
        if flow_weights is None:
            weights = tuple(unit.fluid_volume for unit in self.units)
        else:
            if isinstance(flow_weights, (str, bytes)) or len(flow_weights) != len(self.units):
                raise ValueError("flow_weights must match units")
            weights = tuple(float(item) for item in flow_weights)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("flow_weights must be finite and non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one flow weight must be positive")
        self.properties = properties
        self.unit_ids = ids
        self._active = list(mask)
        self.flow_weights = weights

    @property
    def active_unit_ids(self) -> tuple[str, ...]:
        return tuple(
            unit_id for unit_id, enabled in zip(self.unit_ids, self._active) if enabled
        )

    @property
    def active_mask(self) -> tuple[bool, ...]:
        """Return the explicit hydraulic participation mask in unit order."""

        return tuple(self._active)

    def set_active(self, unit_id: str, enabled: bool) -> None:
        """Explicitly enable or disable one unit's hydraulic participation."""

        if not isinstance(enabled, bool):
            raise ValueError("enabled must be bool")
        try:
            index = self.unit_ids.index(str(unit_id))
        except ValueError as exc:
            raise KeyError(f"unknown parallel dynamic vaporizer unit: {unit_id}") from exc
        self._active[index] = enabled

    def initialize(
        self,
        fluid: ThermoState,
        *,
        wall_temperature_K: float | None = None,
    ) -> ParallelDynamicVaporizerState:
        """Initialize every unit from one explicit thermodynamic state."""

        return ParallelDynamicVaporizerState(tuple(
            unit.initialize(fluid, wall_temperature_K=wall_temperature_K)
            for unit in self.units
        ))

    def thermo(self, state: ParallelDynamicVaporizerState) -> ThermoState:
        """Return the volume-averaged boundary state for the coarse bank node."""

        if len(state.unit_states) != len(self.units):
            raise ValueError("parallel dynamic vaporizer state does not match units")
        total_mass = sum(item.mass_kg for item in state.unit_states)
        total_energy = sum(item.internal_energy_J for item in state.unit_states)
        total_volume = sum(unit.fluid_volume for unit in self.units)
        if not math.isfinite(total_mass) or total_mass <= 0.0:
            raise ValueError("parallel dynamic vaporizer mass must be finite and positive")
        if not math.isfinite(total_energy):
            raise ValueError("parallel dynamic vaporizer energy must be finite")
        return self.properties.from_rho_u(
            total_mass / total_volume,
            total_energy / total_mass,
        )

    def _active_fractions(self) -> tuple[float, ...]:
        total = sum(
            weight for weight, enabled in zip(self.flow_weights, self._active) if enabled
        )
        if total <= 0.0:
            return tuple(0.0 for _ in self.units)
        return tuple(
            weight / total if enabled else 0.0
            for weight, enabled in zip(self.flow_weights, self._active)
        )

    def derivative_from_flows(
        self,
        state: ParallelDynamicVaporizerState,
        flows: Iterable[MassEnergyFlow],
        *,
        ambient_temperature_K: float | None = None,
    ) -> ParallelDynamicVaporizerDerivative:
        """Distribute signed boundary streams across active units."""

        if len(state.unit_states) != len(self.units):
            raise ValueError("parallel dynamic vaporizer state does not match units")
        stream_values = tuple(flows)
        fractions = self._active_fractions()
        if not any(fractions) and any(
            flow.mass_flow_kg_s != 0.0 for flow in stream_values
        ):
            raise ValueError(
                "parallel dynamic vaporizer cannot accept boundary flow without an active positive-weight unit"
            )
        aggregate_outlet_enthalpy = self.thermo(state).specific_enthalpy_J_kg
        derivatives = []
        for unit_state, unit, fraction in zip(state.unit_states, self.units, fractions):
            unit_flows = tuple(
                MassEnergyFlow(
                    flow.mass_flow_kg_s * fraction,
                    flow.specific_enthalpy_J_kg,
                    source=flow.source,
                )
                for flow in stream_values
            )
            derivatives.append(unit.derivative_from_flows(
                unit_state,
                unit_flows,
                ambient_temperature_K=ambient_temperature_K,
                outlet_specific_enthalpy_J_kg=aggregate_outlet_enthalpy,
            ))
        return ParallelDynamicVaporizerDerivative(tuple(derivatives))

    def advance(
        self,
        state: ParallelDynamicVaporizerState,
        derivative: ParallelDynamicVaporizerDerivative,
        time_step_s: float,
    ) -> ParallelDynamicVaporizerState:
        if len(state.unit_states) != len(self.units) or len(derivative.unit_derivatives) != len(self.units):
            raise ValueError("parallel dynamic vaporizer state and derivative do not match units")
        return ParallelDynamicVaporizerState(tuple(
            unit._advance(unit_state, unit_derivative, time_step_s)
            for unit, unit_state, unit_derivative in zip(
                self.units, state.unit_states, derivative.unit_derivatives
            )
        ))

    @staticmethod
    def combine_derivatives(
        k1: ParallelDynamicVaporizerDerivative,
        k2: ParallelDynamicVaporizerDerivative,
        k3: ParallelDynamicVaporizerDerivative,
        k4: ParallelDynamicVaporizerDerivative,
    ) -> ParallelDynamicVaporizerDerivative:
        if not (len(k1.unit_derivatives) == len(k2.unit_derivatives) == len(k3.unit_derivatives) == len(k4.unit_derivatives)):
            raise ValueError("parallel dynamic vaporizer derivatives do not match")
        combined = tuple(
            DynamicVaporizerDerivative(
                mass_kg_s=(a.mass_kg_s + 2.0 * b.mass_kg_s + 2.0 * c.mass_kg_s + d.mass_kg_s) / 6.0,
                internal_energy_W=(a.internal_energy_W + 2.0 * b.internal_energy_W + 2.0 * c.internal_energy_W + d.internal_energy_W) / 6.0,
                wall_temperature_K_s=(a.wall_temperature_K_s + 2.0 * b.wall_temperature_K_s + 2.0 * c.wall_temperature_K_s + d.wall_temperature_K_s) / 6.0,
                mass_flow_in_kg_s=(a.mass_flow_in_kg_s + 2.0 * b.mass_flow_in_kg_s + 2.0 * c.mass_flow_in_kg_s + d.mass_flow_in_kg_s) / 6.0,
                mass_flow_out_kg_s=(a.mass_flow_out_kg_s + 2.0 * b.mass_flow_out_kg_s + 2.0 * c.mass_flow_out_kg_s + d.mass_flow_out_kg_s) / 6.0,
                inlet_enthalpy_flow_W=(a.inlet_enthalpy_flow_W + 2.0 * b.inlet_enthalpy_flow_W + 2.0 * c.inlet_enthalpy_flow_W + d.inlet_enthalpy_flow_W) / 6.0,
                outlet_enthalpy_flow_W=(a.outlet_enthalpy_flow_W + 2.0 * b.outlet_enthalpy_flow_W + 2.0 * c.outlet_enthalpy_flow_W + d.outlet_enthalpy_flow_W) / 6.0,
                heat_from_wall_W=(a.heat_from_wall_W + 2.0 * b.heat_from_wall_W + 2.0 * c.heat_from_wall_W + d.heat_from_wall_W) / 6.0,
                heat_from_ambient_W=(a.heat_from_ambient_W + 2.0 * b.heat_from_ambient_W + 2.0 * c.heat_from_ambient_W + d.heat_from_ambient_W) / 6.0,
            )
            for a, b, c, d in zip(
                k1.unit_derivatives,
                k2.unit_derivatives,
                k3.unit_derivatives,
                k4.unit_derivatives,
            )
        )
        return ParallelDynamicVaporizerDerivative(combined)


__all__ = [
    "DynamicVaporizer",
    "DynamicVaporizerDerivative",
    "DynamicVaporizerState",
    "DynamicVaporizerStepResult",
    "ParallelDynamicVaporizer",
    "ParallelDynamicVaporizerDerivative",
    "ParallelDynamicVaporizerState",
]
