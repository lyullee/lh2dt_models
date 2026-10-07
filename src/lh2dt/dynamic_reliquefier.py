"""Finite-volume transient reliquefier control volume.

The steady :class:`~lh2dt.reliquefier.Reliquefier` is the default thermal
link for a pressure network.  This module provides its replaceable transient
counterpart when the cold-side and fluid inventory must be represented as a
dynamic node.  It reuses the same conserved fluid mass/internal-energy
equations as :class:`~lh2dt.dynamic_vaporizer.DynamicVaporizer`; the explicit
cold-side boundary removes heat through a declared UA and temperature.

No cooling capacity, UA, or cold-side temperature is inferred from operating
time series.  A manufacturer performance map can replace this boundary
closure later without changing the dynamic-node or two-port contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .dynamic_vaporizer import (
    DynamicVaporizer,
    DynamicVaporizerDerivative,
    DynamicVaporizerState,
    DynamicVaporizerStepResult,
)
from .properties import HydrogenProperties
from .streams import MassEnergyFlow


DynamicReliquefierState = DynamicVaporizerState
DynamicReliquefierDerivative = DynamicVaporizerDerivative
DynamicReliquefierStepResult = DynamicVaporizerStepResult


@dataclass(frozen=True)
class ParallelDynamicReliquefierState:
    """Explicit state of every unit in an aggregate reliquefier node."""

    unit_states: tuple[DynamicReliquefierState, ...]


@dataclass(frozen=True)
class ParallelDynamicReliquefierDerivative:
    """Per-unit derivatives plus aggregate ledger properties."""

    unit_derivatives: tuple[DynamicReliquefierDerivative, ...]

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


class DynamicReliquefier(DynamicVaporizer):
    """Transient condenser/reliquefier with an explicit cold-side boundary.

    The inherited state and derivative types are intentionally shared with
    ``DynamicVaporizer`` so a caller can replace a heated or cooled finite
    volume without changing network assembly.  ``cold_side_UA_W_K`` and
    ``maximum_cooling_W`` are design or manufacturer inputs, not fitted
    values.  Positive heat into the fluid is still represented by the common
    signed ``heat_from_wall_W`` term; normal reliquefaction operation makes it
    negative.
    """

    def __init__(
        self,
        fluid_volume_m3: float,
        wall_heat_capacity_J_K: float,
        fluid_wall_UA_W_K: float,
        cold_side_temperature_K: float,
        cold_side_UA_W_K: float,
        maximum_cooling_W: float,
        properties: HydrogenProperties | None = None,
    ) -> None:
        super().__init__(
            fluid_volume_m3=fluid_volume_m3,
            wall_heat_capacity_J_K=wall_heat_capacity_J_K,
            fluid_wall_UA_W_K=fluid_wall_UA_W_K,
            ambient_temperature_K=cold_side_temperature_K,
            ambient_UA_W_K=cold_side_UA_W_K,
            maximum_heat_W=maximum_cooling_W,
            properties=properties,
        )
        # Expose semantic names while retaining the common implementation
        # attributes used by the dynamic network adapter.
        self.cold_side_temperature = self.ambient_temperature
        self.cold_side_UA = self.ambient_UA
        self.maximum_cooling = self.maximum_heat


class ParallelDynamicReliquefier:
    """Aggregate dynamic node containing explicitly replaceable units.

    A multi-unit bank may be represented as one coarse
    process node.  This adapter preserves each unit's conserved mass,
    internal energy, wall state, cold-side boundary, and active flag while
    exposing one volume-weighted boundary state to that coarse node.  A
    future refined P&ID can map each unit to its own boundary without
    changing the individual ``DynamicReliquefier`` contracts.

    ``flow_weights`` are an explicit hydraulic split for the coarse node.  By
    default they are proportional to unit fluid volume, which keeps equal
    initial states equal under the common pressure boundary.  They are not
    fitted from a time series.
    """

    def __init__(
        self,
        units: Sequence[DynamicReliquefier],
        *,
        unit_ids: Sequence[str] | None = None,
        active: Sequence[bool] | None = None,
        flow_weights: Sequence[float] | None = None,
    ) -> None:
        if not units:
            raise ValueError("at least one dynamic reliquefier unit is required")
        self.units = tuple(units)
        properties = self.units[0].properties
        if any(
            getattr(unit.properties, "fluid", None)
            != getattr(properties, "fluid", None)
            for unit in self.units[1:]
        ):
            raise ValueError("all dynamic reliquefier units must use the same fluid backend")
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
            unit_id
            for unit_id, enabled in zip(self.unit_ids, self._active)
            if enabled
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
            raise KeyError(f"unknown parallel dynamic reliquefier unit: {unit_id}") from exc
        self._active[index] = enabled

    def initialize(
        self,
        fluid: object,
        *,
        wall_temperature_K: float | None = None,
    ) -> ParallelDynamicReliquefierState:
        """Initialize every unit from the same explicit thermodynamic state."""

        return ParallelDynamicReliquefierState(tuple(
            unit.initialize(fluid, wall_temperature_K=wall_temperature_K)
            for unit in self.units
        ))

    def thermo(self, state: ParallelDynamicReliquefierState):
        """Return the volume-averaged boundary state for the coarse bank node."""

        if len(state.unit_states) != len(self.units):
            raise ValueError("parallel dynamic reliquefier state does not match units")
        total_mass = sum(item.mass_kg for item in state.unit_states)
        total_energy = sum(item.internal_energy_J for item in state.unit_states)
        total_volume = sum(unit.fluid_volume for unit in self.units)
        if not math.isfinite(total_mass) or total_mass <= 0.0:
            raise ValueError("parallel dynamic reliquefier mass must be finite and positive")
        if not math.isfinite(total_energy):
            raise ValueError("parallel dynamic reliquefier energy must be finite")
        return self.properties.from_rho_u(
            total_mass / total_volume,
            total_energy / total_mass,
        )

    def _active_fractions(self) -> tuple[float, ...]:
        total = sum(
            weight
            for weight, enabled in zip(self.flow_weights, self._active)
            if enabled
        )
        if total <= 0.0:
            return tuple(0.0 for _ in self.units)
        return tuple(
            weight / total if enabled else 0.0
            for weight, enabled in zip(self.flow_weights, self._active)
        )

    def derivative_from_flows(
        self,
        state: ParallelDynamicReliquefierState,
        flows: Iterable[MassEnergyFlow],
        *,
        ambient_temperature_K: float | None = None,
    ) -> ParallelDynamicReliquefierDerivative:
        """Distribute signed boundary streams across the active units."""

        if len(state.unit_states) != len(self.units):
            raise ValueError("parallel dynamic reliquefier state does not match units")
        stream_values = tuple(flows)
        fractions = self._active_fractions()
        if not any(fractions) and any(
            flow.mass_flow_kg_s != 0.0 for flow in stream_values
        ):
            raise ValueError(
                "parallel dynamic reliquefier cannot accept boundary flow without an active positive-weight unit"
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
        return ParallelDynamicReliquefierDerivative(tuple(derivatives))

    def advance(
        self,
        state: ParallelDynamicReliquefierState,
        derivative: ParallelDynamicReliquefierDerivative,
        time_step_s: float,
    ) -> ParallelDynamicReliquefierState:
        if len(state.unit_states) != len(self.units) or len(derivative.unit_derivatives) != len(self.units):
            raise ValueError("parallel dynamic reliquefier state and derivative do not match units")
        return ParallelDynamicReliquefierState(tuple(
            unit._advance(unit_state, unit_derivative, time_step_s)
            for unit, unit_state, unit_derivative in zip(
                self.units, state.unit_states, derivative.unit_derivatives
            )
        ))

    @staticmethod
    def combine_derivatives(
        k1: ParallelDynamicReliquefierDerivative,
        k2: ParallelDynamicReliquefierDerivative,
        k3: ParallelDynamicReliquefierDerivative,
        k4: ParallelDynamicReliquefierDerivative,
    ) -> ParallelDynamicReliquefierDerivative:
        if not (len(k1.unit_derivatives) == len(k2.unit_derivatives) == len(k3.unit_derivatives) == len(k4.unit_derivatives)):
            raise ValueError("parallel dynamic reliquefier derivatives do not match")
        combined = tuple(
            DynamicReliquefierDerivative(
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
        return ParallelDynamicReliquefierDerivative(combined)


__all__ = [
    "DynamicReliquefier",
    "DynamicReliquefierDerivative",
    "DynamicReliquefierState",
    "DynamicReliquefierStepResult",
    "ParallelDynamicReliquefier",
    "ParallelDynamicReliquefierDerivative",
    "ParallelDynamicReliquefierState",
]
