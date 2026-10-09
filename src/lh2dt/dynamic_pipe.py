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
from collections.abc import Mapping
from typing import Any, Iterable

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

    def export_accident_history(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Export a native finite-volume gas-pipe accidental outlet history.

        A failed opening is an explicit accident boundary.  The pipe owns its
        finite fluid and wall state while the caller owns the opening,
        atmospheric boundary and finite observation horizon.  The initial
        state may be a gas, saturated liquid, or explicit
        homogeneous two-phase state.  Liquid/two-phase outlets are flashed at
        the declared atmospheric boundary and exported with ground-liquid and
        gas mass branches.  Airborne liquid remains zero until an independent
        droplet adapter is supplied; the method never silently converts a
        liquid source to a gas-only packet.
        """
        if not isinstance(request, Mapping):
            raise ValueError("accident export request must be a mapping")

        def text(name: str) -> str:
            value = request.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            return value.strip()

        def number(
            name: str,
            default: float | None = None,
            *,
            positive: bool = True,
        ) -> float:
            value = request.get(name, default)
            if isinstance(value, bool) or value is None:
                raise ValueError(f"{name} must be numeric")
            try:
                result = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be numeric") from error
            valid = result > 0.0 if positive else result >= 0.0
            if not math.isfinite(result) or not valid:
                comparator = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {comparator}")
            return result

        def vector(name: str, default: tuple[float, float, float]) -> list[float]:
            value = request.get(name, default)
            if (
                not isinstance(value, (tuple, list))
                or len(value) != 3
            ):
                raise ValueError(f"{name} must contain three coordinates")
            result: list[float] = []
            for index, item in enumerate(value):
                if isinstance(item, bool):
                    raise ValueError(f"{name}[{index}] must be numeric")
                try:
                    coordinate = float(item)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{name}[{index}] must be numeric") from error
                if not math.isfinite(coordinate):
                    raise ValueError(f"{name}[{index}] must be finite")
                result.append(coordinate)
            if math.isclose(sum(item * item for item in result), 0.0, abs_tol=1.0e-20):
                raise ValueError(f"{name} must not be the zero vector")
            norm = math.sqrt(sum(item * item for item in result))
            return [item / norm for item in result]

        event_id = text("event_id")
        component_id = text("component_id")
        port_id = text("port_id")
        if request.get("failure_mode") == "two_sided_rupture":
            raise ValueError(
                "DynamicHEMPipe single-outlet export cannot represent a two-sided rupture; "
                "use a coupled two-exit release-set provider"
            )
        source_pressure = number(
            "initial_pressure_pa_abs", request.get("source_pressure_pa_abs")
        )
        source_temperature = number("initial_temperature_k", request.get("source_temperature_k"))
        ambient_pressure = number("ambient_pressure_pa_abs", 101325.0)
        ambient_temperature = number("ambient_temperature_k", 288.15)
        opening_diameter = number("failure_opening_diameter_m")
        horizon = number("horizon_s")
        time_step = number("time_step_s", min(0.05, horizon))
        discharge_coefficient = number("discharge_coefficient", 0.8)
        if discharge_coefficient > 1.0:
            raise ValueError("discharge_coefficient must not exceed one")
        isolation_value = request.get("isolation_time_s")
        isolation_time = None if isolation_value is None else number(
            "isolation_time_s", positive=False
        )
        if isolation_time is not None and isolation_time > horizon:
            raise ValueError("isolation_time_s must not exceed horizon_s")
        post_isolation_delay = number(
            "post_isolation_observation_s", 0.0, positive=False
        )
        effective_horizon = horizon if isolation_time is None else min(
            horizon, isolation_time + post_isolation_delay
        )
        if effective_horizon <= 0.0:
            raise ValueError("isolation horizon must leave a positive observation interval")
        wall_temperature = number("wall_temperature_k", source_temperature)
        if source_pressure <= ambient_pressure:
            raise ValueError("initial pressure must exceed ambient pressure for an outward accident")
        area = math.pi * opening_diameter**2 / 4.0
        if opening_diameter > self.diameter:
            raise ValueError("failure opening diameter must not exceed pipe diameter")

        initial_quality = request.get("initial_quality")
        if initial_quality is None:
            source = self.properties.from_pT(source_pressure, source_temperature)
        else:
            quality = number("initial_quality", positive=False)
            if not 0.0 <= quality <= 1.0:
                raise ValueError("initial_quality must lie between zero and one")
            liquid = self.properties.saturated_liquid(source_pressure)
            vapor = self.properties.saturated_vapor(source_pressure)
            source = self.properties.from_ph(
                source_pressure,
                liquid.specific_enthalpy_J_kg
                + quality * (vapor.specific_enthalpy_J_kg - liquid.specific_enthalpy_J_kg),
            )
        source_is_gas = source.phase in {"gas", "supercritical_gas", "supercritical"}
        if not source_is_gas and source.quality is None:
            raise ValueError("DynamicHEMPipe accident export source phase is unsupported")
        source_phase_basis = (
            "two_phase"
            if source.quality is not None and 0.0 < float(source.quality) < 1.0
            else "liquid"
            if source.quality is not None and float(source.quality) <= 1.0e-8
            else "gas"
        )
        ambient = self.properties.from_pT(ambient_pressure, ambient_temperature)
        from .valve import Valve

        valve = Valve(
            area,
            discharge_coefficient,
            properties=self.properties,
            allow_reverse=False,
        )
        state = self.initialize(source, wall_temperature_K=wall_temperature)
        initial_mass = float(state.mass_kg)
        steps: list[dict[str, Any]] = []
        cumulative_mass = 0.0
        cumulative_enthalpy = 0.0
        elapsed = 0.0
        choked_count = 0
        pressure_equalized = False

        def derivative_at(
            candidate: DynamicHEMPipeState, opening: float
        ) -> tuple[DynamicHEMPipeDerivative, Any, ThermoState]:
            fluid = self.thermo(candidate)
            hydraulic = valve.evaluate(fluid, ambient, opening=opening)
            if not math.isfinite(hydraulic.mass_flow_kg_s) or hydraulic.mass_flow_kg_s < 0.0:
                raise ValueError("native pipe outlet returned an invalid outward mass flow")
            derivative = self.derivative_from_flows(
                candidate,
                (MassEnergyFlow(-hydraulic.mass_flow_kg_s, hydraulic.outlet_specific_enthalpy_J_kg, source="DynamicHEMPipe native accidental outlet"),),
            )
            return derivative, hydraulic, fluid

        def combined(*derivatives: DynamicHEMPipeDerivative) -> DynamicHEMPipeDerivative:
            weights = (1.0, 2.0, 2.0, 1.0)
            total = sum(weights)
            names = (
                "mass_kg_s", "internal_energy_W", "wall_temperature_K_s",
                "mass_flow_in_kg_s", "mass_flow_out_kg_s",
                "inlet_enthalpy_flow_W", "outlet_enthalpy_flow_W",
                "heat_from_wall_W", "heat_from_ambient_W",
            )
            values = {
                name: sum(weight * float(getattr(item, name)) for weight, item in zip(weights, derivatives)) / total
                for name in names
            }
            return DynamicHEMPipeDerivative(**values)

        while elapsed < effective_horizon - 1.0e-12:
            dt = min(time_step, effective_horizon - elapsed)
            opening_at = lambda absolute_time: (
                0.0
                if isolation_time is not None and absolute_time >= isolation_time
                else 1.0
            )
            k1, hydraulic, fluid = derivative_at(state, opening_at(elapsed))
            flow_threshold = max(
                1.0e-12,
                initial_mass * 1.0e-10 / max(effective_horizon, 1.0e-12),
            )
            pressure_check_allowed = (
                isolation_time is None
                or elapsed >= effective_horizon - 1.0e-12
            )
            if pressure_check_allowed and (
                fluid.pressure_Pa <= ambient_pressure * (1.0 + 1.0e-8)
                or hydraulic.mass_flow_kg_s <= flow_threshold
            ):
                pressure_equalized = True
                break
            if k1.mass_kg_s < 0.0:
                safe_dt = 0.25 * state.mass_kg / (-k1.mass_kg_s)
                if safe_dt < dt:
                    dt = max(safe_dt, 1.0e-12)
            k2, _, _ = derivative_at(
                self._advance(state, k1, dt / 2.0), opening_at(elapsed + dt / 2.0)
            )
            k3, _, _ = derivative_at(
                self._advance(state, k2, dt / 2.0), opening_at(elapsed + dt / 2.0)
            )
            k4, _, _ = derivative_at(
                self._advance(state, k3, dt), opening_at(elapsed + dt)
            )
            averaged = combined(k1, k2, k3, k4)
            next_state = self._advance(state, averaged, dt)
            next_fluid = self.thermo(next_state)
            if next_fluid.pressure_Pa <= ambient_pressure:
                # The finite-volume step can straddle the atmospheric
                # boundary. Locate the pressure-equalisation point inside
                # this RK4 interval instead of retaining a below-ambient
                # terminal state caused by numerical overshoot.
                lo = 0.0
                hi = 1.0
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    trial = self._advance(state, averaged, dt * mid)
                    if self.thermo(trial).pressure_Pa > ambient_pressure:
                        lo = mid
                    else:
                        hi = mid
                event_fraction = 0.5 * (lo + hi)
                dt *= event_fraction
                next_state = self._advance(state, averaged, dt)
                pressure_equalized = True
            mass_out = max(0.0, -averaged.mass_kg_s * dt)
            enthalpy = float(hydraulic.outlet_specific_enthalpy_J_kg)
            exit_state = self.properties.from_ph(ambient_pressure, enthalpy)
            if source_is_gas:
                if exit_state.phase not in {"gas", "supercritical_gas", "supercritical"}:
                    raise ValueError("native pipe outlet is not gas-like at the atmospheric boundary")
                gas_flow = mass_out / dt
                ground_flow = 0.0
                gas_velocity = (
                    gas_flow / (exit_state.density_kg_m3 * area)
                    if gas_flow > 0.0 else 0.0
                )
                # Preserve the native compressible source state when the
                # pipe outlet is gas-like.  The valve solver already returns
                # the native throat pressure and mass flux; reconstructing
                # the throat with the provider's upstream entropy keeps the
                # optional downstream plane handoff source-bound without
                # making PRISM infer a near-field state.
                throat_state = self.properties.from_ps(
                    float(hydraulic.throat_pressure_Pa),
                    float(fluid.specific_entropy_J_kgK),
                )
                step = {
                    "time_s": elapsed,
                    "mass_flow_kg_s": gas_flow,
                    "pressure_pa_abs": ambient_pressure,
                    "specific_enthalpy_j_kg": enthalpy,
                    "temperature_k": float(exit_state.temperature_K),
                    "density_kg_m3": float(exit_state.density_kg_m3),
                    "effective_area_m2": area,
                    "velocity_m_s": gas_velocity,
                    "area_provenance": "explicit failed pipe opening area",
                    "velocity_origin": "mass_continuity_from_declared_area",
                    "provider_source_state": {
                        "source_pressure_pa_abs": float(fluid.pressure_Pa),
                        "source_temperature_k": float(fluid.temperature_K),
                        "source_density_kg_m3": float(fluid.density_kg_m3),
                        "source_specific_enthalpy_j_kg": float(fluid.specific_enthalpy_J_kg),
                        "source_specific_entropy_j_kgk": float(fluid.specific_entropy_J_kgK),
                        "throat_pressure_pa_abs": float(hydraulic.throat_pressure_Pa),
                        "throat_temperature_k": float(throat_state.temperature_K),
                        "throat_density_kg_m3": float(throat_state.density_kg_m3),
                        "throat_specific_enthalpy_j_kg": float(throat_state.specific_enthalpy_J_kg),
                        "throat_mass_flux_kg_m2_s": float(hydraulic.mass_flux_kg_m2_s),
                        "choked": bool(hydraulic.choked),
                        "provider_source_state_provenance": (
                            "native DynamicHEMPipe upstream entropy and hydraulic throat result"
                        ),
                    },
                }
            else:
                quality = exit_state.quality
                if quality is None:
                    if exit_state.phase in {"gas", "supercritical_gas", "supercritical"}:
                        quality = 1.0
                    elif exit_state.phase in {"liquid", "supercritical_liquid"}:
                        quality = 0.0
                    else:
                        raise ValueError("native pipe outlet has an unsupported atmospheric phase")
                quality = min(1.0, max(0.0, float(quality)))
                gas_flow = mass_out * quality / dt
                ground_flow = mass_out * (1.0 - quality) / dt
                step = {
                    "time_s": elapsed,
                    "total_mass_flow_kg_s": mass_out / dt,
                    "gas_mass_flow_kg_s": gas_flow,
                    "ground_liquid_mass_flow_kg_s": ground_flow,
                    "airborne_liquid_mass_flow_kg_s": 0.0,
                    "specific_enthalpy_j_kg": enthalpy,
                    "gas_pressure_pa_abs": ambient_pressure if gas_flow > 1.0e-12 else None,
                    "gas_temperature_k": float(exit_state.temperature_K) if gas_flow > 1.0e-12 else None,
                    "gas_density_kg_m3": float(exit_state.density_kg_m3) if gas_flow > 1.0e-12 else None,
                    "gas_effective_area_m2": area if gas_flow > 1.0e-12 else None,
                    "gas_velocity_m_s": (
                        gas_flow / (exit_state.density_kg_m3 * area)
                        if gas_flow > 1.0e-12 else None
                    ),
                    "gas_velocity_origin": "mass_continuity_from_declared_area" if gas_flow > 1.0e-12 else None,
                    "droplet_class_outcomes": [{
                        "ground_liquid_mass_flow_kg_s": ground_flow,
                        "airborne_liquid_mass_flow_kg_s": 0.0,
                        "flight_time_s": number("droplet_flight_time_s", 0.0, positive=False),
                        "impact_position_m": list(vector("impact_position_m", (0.0, 0.0, 0.001))),
                        "droplet_diameter_m": number("droplet_diameter_m", 1.0e-3),
                    }],
                }
            steps.append(step)
            if hydraulic.choked:
                choked_count += 1
            cumulative_mass += mass_out
            cumulative_enthalpy += mass_out * enthalpy
            state = next_state
            elapsed += dt
            if pressure_equalized:
                break
            if state.mass_kg <= max(initial_mass * 1.0e-9, 1.0e-12):
                break

        depleted = state.mass_kg <= max(initial_mass * 1.0e-9, 1.0e-12)
        isolation_success = (
            isolation_time is not None
            and elapsed >= effective_horizon - 1.0e-12
            and effective_horizon >= isolation_time
        )
        termination_basis = (
            "inventory_depletion"
            if depleted
            else "reviewed_isolation_success"
            if isolation_success
            else "model_pressure_equalization"
            if pressure_equalized
            else "synthetic_diagnostic_horizon"
        )
        termination_provenance = (
            "DynamicHEMPipe finite-volume mass state reached the explicit positive inventory threshold"
            if depleted else
            "DynamicHEMPipe native outlet closed at the caller-declared isolation time and remained closed through the observation delay"
            if isolation_success else
            "DynamicHEMPipe native pressure reached the atmospheric boundary and the outlet mass flow fell below the resolved termination threshold"
            if pressure_equalized else
            "DynamicHEMPipe finite-volume state stopped at the caller-declared bounded horizon; residual inventory retained"
        )
        final_fluid = self.thermo(state)
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
                "cumulative_specific_enthalpy_out_j": cumulative_enthalpy,
                "cumulative_static_enthalpy_out_j": cumulative_enthalpy,
                "termination_basis": termination_basis,
                "termination_provenance": termination_provenance,
                "upstream_service": "LH2",
                "fluid": "Hydrogen",
                "phase_basis": source_phase_basis,
                "reference_area_m2": area,
                "reference_area_provenance": "explicit failed pipe opening area",
                "droplet_partition_provenance": (
                    "DynamicHEMPipe HEM outlet flash at ambient pressure; ground liquid retained; airborne liquid set to zero"
                    if not source_is_gas else
                    "DynamicHEMPipe gas outlet at ambient pressure; no liquid partition"
                ),
                "steps": steps,
                "provider_meta": {
                    "native_inventory_owner": "DynamicHEMPipe",
                    "initial_mass_kg": initial_mass,
                    "final_mass_kg": float(state.mass_kg),
                    "residual_mass_kg": initial_mass - cumulative_mass - float(state.mass_kg),
                    "final_pressure_pa_abs": float(final_fluid.pressure_Pa),
                    "pressure_equalized": pressure_equalized,
                    "isolation_time_s": isolation_time,
                    "post_isolation_observation_s": post_isolation_delay,
                    "isolation_success": isolation_success,
                    "time_step_s": time_step,
                    "valve": {
                        "area_m2": area,
                        "discharge_coefficient": discharge_coefficient,
                        "choked_intervals": choked_count,
                    },
                },
            },
        }

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
