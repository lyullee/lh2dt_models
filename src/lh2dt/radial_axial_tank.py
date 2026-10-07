"""Conservative two-column, two-phase cryogenic tank integration.

The tank resolves a wall layer and a core column in both liquid and vapor.
Each phase uses :class:`RadialAxialCirculationTransport` for buoyancy-driven
axial exchange.  Molecular axial/radial conduction, wall patches, an
equilibrium liquid-vapor interface, and external material ports are assembled
as conservative mass and energy source terms.  A single
:class:`CommonPressureRigidVolume` then enforces the rigid-volume DAE.

No facility data are fitted.  Conductances and geometric flow paths are
explicit replaceable inputs, so later CFD-derived or literature closures can
replace individual terms without changing the state or port contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import brentq

from .common_pressure import (
    CommonPressureCellDerivative,
    CommonPressureCellState,
    CommonPressureRigidVolume,
)
from .properties import HydrogenProperties, ThermoState
from .phase_change import kinetic_gas_film_heat_transfer_coefficient
from .radial_axial_transport import (
    RadialAxialBoundaryGeometry,
    RadialAxialCirculationTransport,
    RadialAxialPhaseState,
    RadialAxialTransportResult,
)
from .streams import MassEnergyFlow


@dataclass(frozen=True)
class RadialAxialTankParameters:
    """Geometry and closure inputs for a two-column tank.

    Conductances are totals for the named connection.  Axial fluid
    conductance is applied to each radial column at every boundary.  Wall
    totals are distributed by ``wall_patch_area_fractions``; an empty tuple
    means equal patches.
    """

    total_volume_m3: float
    liquid_level_count: int
    vapor_level_count: int
    liquid_wall_volume_fraction: float
    vapor_wall_volume_fraction: float
    liquid_boundaries: tuple[RadialAxialBoundaryGeometry, ...]
    vapor_boundaries: tuple[RadialAxialBoundaryGeometry, ...]
    wall_heat_capacity_J_K: float
    ambient_UA_W_K: float = 0.0
    wall_to_fluid_UA_W_K: float = 0.0
    wall_axial_conductance_W_K: float = 0.0
    liquid_radial_conductance_W_K: float = 0.0
    vapor_radial_conductance_W_K: float = 0.0
    liquid_axial_conductance_W_K: float = 0.0
    vapor_axial_conductance_W_K: float = 0.0
    liquid_interface_UA_W_K: tuple[float, float] = (0.0, 0.0)
    vapor_interface_UA_W_K: tuple[float, float] = (0.0, 0.0)
    vapor_interface_heat_transfer_model: str = "conductance"
    vapor_interface_area_m2: tuple[float, float] | None = None
    gas_interface_energy_accommodation_coefficient: float = 1.0
    distributed_boiling_model: str = "off"
    wall_patch_area_fractions: tuple[float, ...] = ()
    gravity_m_s2: float = 9.80665
    minimum_pressure_Pa: float = 10_000.0
    maximum_pressure_Pa: float = 1_250_000.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_volume_m3) or self.total_volume_m3 <= 0.0:
            raise ValueError("total_volume_m3 must be finite and positive")
        for name in ("liquid_level_count", "vapor_level_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        for name in ("liquid_wall_volume_fraction", "vapor_wall_volume_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if len(self.liquid_boundaries) != self.liquid_level_count - 1:
            raise ValueError("liquid boundary count must equal liquid levels minus one")
        if len(self.vapor_boundaries) != self.vapor_level_count - 1:
            raise ValueError("vapor boundary count must equal vapor levels minus one")
        positive = (
            "wall_heat_capacity_J_K",
            "gravity_m_s2",
            "minimum_pressure_Pa",
            "maximum_pressure_Pa",
        )
        nonnegative = (
            "ambient_UA_W_K",
            "wall_to_fluid_UA_W_K",
            "wall_axial_conductance_W_K",
            "liquid_radial_conductance_W_K",
            "vapor_radial_conductance_W_K",
            "liquid_axial_conductance_W_K",
            "vapor_axial_conductance_W_K",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("liquid_interface_UA_W_K", "vapor_interface_UA_W_K"):
            values = getattr(self, name)
            if len(values) != 2 or any(
                not math.isfinite(value) or value < 0.0 for value in values
            ):
                raise ValueError(f"{name} must contain two finite non-negative values")
        if self.vapor_interface_heat_transfer_model not in {
            "conductance", "kinetic_gas_film"
        }:
            raise ValueError(
                "vapor_interface_heat_transfer_model must be 'conductance' "
                "or 'kinetic_gas_film'"
            )
        if (
            not math.isfinite(self.gas_interface_energy_accommodation_coefficient)
            or not 0.0 < self.gas_interface_energy_accommodation_coefficient <= 1.0
        ):
            raise ValueError(
                "gas_interface_energy_accommodation_coefficient must be finite "
                "and in (0, 1]"
            )
        if self.vapor_interface_area_m2 is not None:
            if len(self.vapor_interface_area_m2) != 2 or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.vapor_interface_area_m2
            ):
                raise ValueError(
                    "vapor_interface_area_m2 must contain two finite positive values"
                )
        if (
            self.vapor_interface_heat_transfer_model == "kinetic_gas_film"
            and self.vapor_interface_area_m2 is None
        ):
            raise ValueError(
                "kinetic_gas_film requires explicit vapor_interface_area_m2"
            )
        if self.maximum_pressure_Pa <= self.minimum_pressure_Pa:
            raise ValueError("maximum_pressure_Pa must exceed minimum_pressure_Pa")
        if self.distributed_boiling_model not in {
            "off", "saturated_liquid_complementarity"
        }:
            raise ValueError(
                "distributed_boiling_model must be 'off' or "
                "'saturated_liquid_complementarity'"
            )
        patch_count = self.liquid_level_count + self.vapor_level_count
        if self.wall_patch_area_fractions:
            if len(self.wall_patch_area_fractions) != patch_count:
                raise ValueError("wall_patch_area_fractions length must match wall patches")
            if any(
                not math.isfinite(value) or value <= 0.0
                for value in self.wall_patch_area_fractions
            ):
                raise ValueError("wall patch fractions must be finite and positive")
            if not math.isclose(
                sum(self.wall_patch_area_fractions), 1.0,
                rel_tol=0.0, abs_tol=1.0e-12,
            ):
                raise ValueError("wall patch fractions must sum to one")


@dataclass(frozen=True)
class RadialAxialTankState:
    pressure_Pa: float
    liquid_wall_masses_kg: tuple[float, ...]
    liquid_wall_enthalpies_J_kg: tuple[float, ...]
    liquid_core_masses_kg: tuple[float, ...]
    liquid_core_enthalpies_J_kg: tuple[float, ...]
    vapor_wall_masses_kg: tuple[float, ...]
    vapor_wall_enthalpies_J_kg: tuple[float, ...]
    vapor_core_masses_kg: tuple[float, ...]
    vapor_core_enthalpies_J_kg: tuple[float, ...]
    wall_temperatures_K: tuple[float, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.pressure_Pa) or self.pressure_Pa <= 0.0:
            raise ValueError("pressure_Pa must be finite and positive")
        mass_groups = (
            self.liquid_wall_masses_kg,
            self.liquid_core_masses_kg,
            self.vapor_wall_masses_kg,
            self.vapor_core_masses_kg,
        )
        enthalpy_groups = (
            self.liquid_wall_enthalpies_J_kg,
            self.liquid_core_enthalpies_J_kg,
            self.vapor_wall_enthalpies_J_kg,
            self.vapor_core_enthalpies_J_kg,
        )
        if any(not group for group in mass_groups):
            raise ValueError("all four fluid blocks require at least one cell")
        if any(
            len(masses) != len(enthalpies)
            for masses, enthalpies in zip(mass_groups, enthalpy_groups)
        ):
            raise ValueError("each fluid block must have matching mass and enthalpy lengths")
        if any(
            not math.isfinite(value) or value <= 0.0
            for group in mass_groups for value in group
        ):
            raise ValueError("all cell masses must be finite and positive")
        if any(
            not math.isfinite(value)
            for group in enthalpy_groups for value in group
        ):
            raise ValueError("all cell enthalpies must be finite")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.wall_temperatures_K
        ):
            raise ValueError("all wall temperatures must be finite and positive")


@dataclass(frozen=True)
class RadialAxialTankThermo:
    cells: tuple[ThermoState, ...]
    cell_volumes_m3: tuple[float, ...]
    total_volume_m3: float
    volume_residual_m3: float


@dataclass(frozen=True)
class RadialAxialTankDerivative:
    pressure_rate_Pa_s: float
    mass_rates_kg_s: tuple[float, ...]
    enthalpy_rates_J_kg_s: tuple[float, ...]
    wall_temperature_rates_K_s: tuple[float, ...]
    enthalpy_inventory_sources_W: tuple[float, ...]
    interface_phase_change_rates_kg_s: tuple[float, float]
    interface_energy_residual_W: float
    distributed_boiling_rates_kg_s: tuple[float, ...]
    distributed_boiling_rate_kg_s: float
    distributed_boiling_latent_power_W: float
    liquid_transport: RadialAxialTransportResult
    vapor_transport: RadialAxialTransportResult
    common_pressure: CommonPressureCellDerivative
    boundary_mass_rate_kg_s: float
    boundary_energy_rate_W: float
    total_mass_rate_kg_s: float
    total_energy_rate_W: float
    mass_conservation_residual_kg_s: float
    energy_conservation_residual_W: float


@dataclass(frozen=True)
class RadialAxialTankStepResult:
    state: RadialAxialTankState
    thermo: RadialAxialTankThermo
    derivative: RadialAxialTankDerivative
    total_mass_residual_kg: float
    total_energy_residual_J: float


class RadialAxialTank:
    """Assemble the conservative 2 x N liquid and vapor tank equations."""

    def __init__(
        self,
        parameters: RadialAxialTankParameters,
        properties: HydrogenProperties | None = None,
    ) -> None:
        self.parameters = parameters
        self.properties = properties or HydrogenProperties()
        self._dae = CommonPressureRigidVolume(
            parameters.total_volume_m3, self.properties
        )
        self._liquid_transport = RadialAxialCirculationTransport(
            parameters.liquid_boundaries,
            gravity_m_s2=parameters.gravity_m_s2,
            properties=self.properties,
            closed_loop_turns=True,
        )
        self._vapor_transport = RadialAxialCirculationTransport(
            parameters.vapor_boundaries,
            gravity_m_s2=parameters.gravity_m_s2,
            properties=self.properties,
            closed_loop_turns=True,
        )
        patch_count = parameters.liquid_level_count + parameters.vapor_level_count
        fractions = parameters.wall_patch_area_fractions or (
            (1.0 / patch_count,) * patch_count
        )
        self._wall_fractions = np.asarray(fractions, dtype=float)
        self._wall_capacities = (
            parameters.wall_heat_capacity_J_K * self._wall_fractions
        )
        self._ambient_ua = parameters.ambient_UA_W_K * self._wall_fractions
        self._wall_fluid_ua = parameters.wall_to_fluid_UA_W_K * self._wall_fractions

    @property
    def cell_count(self) -> int:
        p = self.parameters
        return 2 * (p.liquid_level_count + p.vapor_level_count)

    @property
    def wall_patch_count(self) -> int:
        return self.parameters.liquid_level_count + self.parameters.vapor_level_count

    def cell_index(self, phase: str, column: str, level: int) -> int:
        """Return the stable flattened index used by ports and heat sources."""

        p = self.parameters
        if phase not in {"liquid", "vapor"}:
            raise ValueError("phase must be 'liquid' or 'vapor'")
        if column not in {"wall", "core"}:
            raise ValueError("column must be 'wall' or 'core'")
        count = p.liquid_level_count if phase == "liquid" else p.vapor_level_count
        if not isinstance(level, int) or not 0 <= level < count:
            raise ValueError("level lies outside the requested phase")
        if phase == "liquid":
            return level if column == "wall" else p.liquid_level_count + level
        start = 2 * p.liquid_level_count
        return start + level if column == "wall" else start + p.vapor_level_count + level

    def _validate_state_lengths(self, state: RadialAxialTankState) -> None:
        p = self.parameters
        liquid_groups = (
            state.liquid_wall_masses_kg,
            state.liquid_wall_enthalpies_J_kg,
            state.liquid_core_masses_kg,
            state.liquid_core_enthalpies_J_kg,
        )
        vapor_groups = (
            state.vapor_wall_masses_kg,
            state.vapor_wall_enthalpies_J_kg,
            state.vapor_core_masses_kg,
            state.vapor_core_enthalpies_J_kg,
        )
        if any(len(group) != p.liquid_level_count for group in liquid_groups):
            raise ValueError("liquid state lengths do not match parameters")
        if any(len(group) != p.vapor_level_count for group in vapor_groups):
            raise ValueError("vapor state lengths do not match parameters")
        if len(state.wall_temperatures_K) != self.wall_patch_count:
            raise ValueError("wall temperature count does not match wall patches")

    @staticmethod
    def _flatten_state(
        state: RadialAxialTankState,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        masses = (
            *state.liquid_wall_masses_kg,
            *state.liquid_core_masses_kg,
            *state.vapor_wall_masses_kg,
            *state.vapor_core_masses_kg,
        )
        enthalpies = (
            *state.liquid_wall_enthalpies_J_kg,
            *state.liquid_core_enthalpies_J_kg,
            *state.vapor_wall_enthalpies_J_kg,
            *state.vapor_core_enthalpies_J_kg,
        )
        return masses, enthalpies

    def _unflatten_state(
        self,
        pressure_Pa: float,
        masses: Sequence[float],
        enthalpies: Sequence[float],
        wall_temperatures_K: Sequence[float],
    ) -> RadialAxialTankState:
        nl = self.parameters.liquid_level_count
        nv = self.parameters.vapor_level_count
        masses = tuple(float(value) for value in masses)
        enthalpies = tuple(float(value) for value in enthalpies)
        return RadialAxialTankState(
            pressure_Pa=float(pressure_Pa),
            liquid_wall_masses_kg=masses[:nl],
            liquid_wall_enthalpies_J_kg=enthalpies[:nl],
            liquid_core_masses_kg=masses[nl:2 * nl],
            liquid_core_enthalpies_J_kg=enthalpies[nl:2 * nl],
            vapor_wall_masses_kg=masses[2 * nl:2 * nl + nv],
            vapor_wall_enthalpies_J_kg=enthalpies[2 * nl:2 * nl + nv],
            vapor_core_masses_kg=masses[2 * nl + nv:],
            vapor_core_enthalpies_J_kg=enthalpies[2 * nl + nv:],
            wall_temperatures_K=tuple(float(value) for value in wall_temperatures_K),
        )

    def initialize_uniform(
        self,
        pressure_Pa: float,
        liquid_volume_fraction: float,
        liquid_temperature_K: float,
        vapor_temperature_K: float,
        wall_temperature_K: float | None = None,
    ) -> RadialAxialTankState:
        """Create a volume-consistent uniform single-phase state."""

        if not 0.0 < liquid_volume_fraction < 1.0:
            raise ValueError("liquid_volume_fraction must lie between zero and one")
        liquid = self.properties.from_pT(pressure_Pa, liquid_temperature_K)
        vapor = self.properties.from_pT(pressure_Pa, vapor_temperature_K)
        if liquid.quality is not None or vapor.quality is not None:
            raise ValueError("initial bulk cells must be single-phase")
        if liquid.density_kg_m3 <= vapor.density_kg_m3:
            raise ValueError("liquid initial state must be denser than vapor")
        p = self.parameters
        liquid_volume = p.total_volume_m3 * liquid_volume_fraction
        vapor_volume = p.total_volume_m3 - liquid_volume
        lw_volume = liquid_volume * p.liquid_wall_volume_fraction / p.liquid_level_count
        lc_volume = liquid_volume * (1.0 - p.liquid_wall_volume_fraction) / p.liquid_level_count
        vw_volume = vapor_volume * p.vapor_wall_volume_fraction / p.vapor_level_count
        vc_volume = vapor_volume * (1.0 - p.vapor_wall_volume_fraction) / p.vapor_level_count
        wall_temperature = (
            0.5 * (liquid_temperature_K + vapor_temperature_K)
            if wall_temperature_K is None else wall_temperature_K
        )
        return RadialAxialTankState(
            pressure_Pa=float(pressure_Pa),
            liquid_wall_masses_kg=(liquid.density_kg_m3 * lw_volume,) * p.liquid_level_count,
            liquid_wall_enthalpies_J_kg=(liquid.specific_enthalpy_J_kg,) * p.liquid_level_count,
            liquid_core_masses_kg=(liquid.density_kg_m3 * lc_volume,) * p.liquid_level_count,
            liquid_core_enthalpies_J_kg=(liquid.specific_enthalpy_J_kg,) * p.liquid_level_count,
            vapor_wall_masses_kg=(vapor.density_kg_m3 * vw_volume,) * p.vapor_level_count,
            vapor_wall_enthalpies_J_kg=(vapor.specific_enthalpy_J_kg,) * p.vapor_level_count,
            vapor_core_masses_kg=(vapor.density_kg_m3 * vc_volume,) * p.vapor_level_count,
            vapor_core_enthalpies_J_kg=(vapor.specific_enthalpy_J_kg,) * p.vapor_level_count,
            wall_temperatures_K=(float(wall_temperature),) * self.wall_patch_count,
        )

    def initialize_saturated(
        self,
        pressure_Pa: float,
        liquid_volume_fraction: float,
        wall_temperature_K: float | None = None,
    ) -> RadialAxialTankState:
        """Create an exactly saturated, rigid-volume-consistent state."""

        if not 0.0 < liquid_volume_fraction < 1.0:
            raise ValueError("liquid_volume_fraction must lie between zero and one")
        liquid = self.properties.saturated_liquid(pressure_Pa)
        vapor = self.properties.saturated_vapor(pressure_Pa)
        p = self.parameters
        liquid_volume = p.total_volume_m3 * liquid_volume_fraction
        vapor_volume = p.total_volume_m3 - liquid_volume
        lw_volume = liquid_volume * p.liquid_wall_volume_fraction / p.liquid_level_count
        lc_volume = liquid_volume * (1.0 - p.liquid_wall_volume_fraction) / p.liquid_level_count
        vw_volume = vapor_volume * p.vapor_wall_volume_fraction / p.vapor_level_count
        vc_volume = vapor_volume * (1.0 - p.vapor_wall_volume_fraction) / p.vapor_level_count
        wall_temperature = (
            liquid.temperature_K if wall_temperature_K is None else wall_temperature_K
        )
        return RadialAxialTankState(
            pressure_Pa=float(pressure_Pa),
            liquid_wall_masses_kg=(liquid.density_kg_m3 * lw_volume,) * p.liquid_level_count,
            liquid_wall_enthalpies_J_kg=(liquid.specific_enthalpy_J_kg,) * p.liquid_level_count,
            liquid_core_masses_kg=(liquid.density_kg_m3 * lc_volume,) * p.liquid_level_count,
            liquid_core_enthalpies_J_kg=(liquid.specific_enthalpy_J_kg,) * p.liquid_level_count,
            vapor_wall_masses_kg=(vapor.density_kg_m3 * vw_volume,) * p.vapor_level_count,
            vapor_wall_enthalpies_J_kg=(vapor.specific_enthalpy_J_kg,) * p.vapor_level_count,
            vapor_core_masses_kg=(vapor.density_kg_m3 * vc_volume,) * p.vapor_level_count,
            vapor_core_enthalpies_J_kg=(vapor.specific_enthalpy_J_kg,) * p.vapor_level_count,
            wall_temperatures_K=(float(wall_temperature),) * self.wall_patch_count,
        )

    def thermo(self, state: RadialAxialTankState) -> RadialAxialTankThermo:
        self._validate_state_lengths(state)
        masses, enthalpies = self._flatten_state(state)
        cells, volumes = self._dae.thermo(CommonPressureCellState(
            state.pressure_Pa, masses, enthalpies
        ))
        nl = self.parameters.liquid_level_count
        liquid = cells[:2 * nl]
        vapor = cells[2 * nl:]
        if min(cell.density_kg_m3 for cell in liquid) <= max(
            cell.density_kg_m3 for cell in vapor
        ):
            raise ValueError("all labeled liquid cells must be denser than vapor cells")
        total_volume = float(np.sum(volumes))
        return RadialAxialTankThermo(
            cells=cells,
            cell_volumes_m3=tuple(float(value) for value in volumes),
            total_volume_m3=total_volume,
            volume_residual_m3=total_volume - self.parameters.total_volume_m3,
        )

    @staticmethod
    def _finite_vector(
        values: Sequence[float] | None, count: int, name: str
    ) -> np.ndarray:
        if values is None:
            return np.zeros(count)
        array = np.asarray(tuple(values), dtype=float)
        if array.shape != (count,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain {count} finite values")
        return array

    def derivative(
        self,
        state: RadialAxialTankState,
        ambient_temperature_K: float,
        cell_streams: Sequence[Iterable[MassEnergyFlow]] | None = None,
        direct_fluid_heat_W: Sequence[float] | None = None,
        direct_wall_heat_W: Sequence[float] | None = None,
        include_circulation_sources: bool = True,
    ) -> RadialAxialTankDerivative:
        """Evaluate all conservative sources and the common-pressure DAE."""

        if not math.isfinite(ambient_temperature_K) or ambient_temperature_K <= 0.0:
            raise ValueError("ambient_temperature_K must be finite and positive")
        thermo = self.thermo(state)
        p = self.parameters
        nl = p.liquid_level_count
        nv = p.vapor_level_count
        nc = self.cell_count
        masses, enthalpies = self._flatten_state(state)
        cells = thermo.cells
        mass_rates = np.zeros(nc)
        energy_sources = self._finite_vector(
            direct_fluid_heat_W, nc, "direct_fluid_heat_W"
        )
        direct_wall = self._finite_vector(
            direct_wall_heat_W, self.wall_patch_count, "direct_wall_heat_W"
        )

        liquid_transport = self._liquid_transport.evaluate(
            state.pressure_Pa,
            RadialAxialPhaseState(
                state.liquid_wall_masses_kg,
                state.liquid_wall_enthalpies_J_kg,
                state.liquid_core_masses_kg,
                state.liquid_core_enthalpies_J_kg,
            ),
        )
        vapor_transport = self._vapor_transport.evaluate(
            state.pressure_Pa,
            RadialAxialPhaseState(
                state.vapor_wall_masses_kg,
                state.vapor_wall_enthalpies_J_kg,
                state.vapor_core_masses_kg,
                state.vapor_core_enthalpies_J_kg,
            ),
        )
        if include_circulation_sources:
            mass_rates[:nl] += liquid_transport.wall_mass_rates_kg_s
            mass_rates[nl:2 * nl] += liquid_transport.core_mass_rates_kg_s
            mass_rates[2 * nl:2 * nl + nv] += vapor_transport.wall_mass_rates_kg_s
            mass_rates[2 * nl + nv:] += vapor_transport.core_mass_rates_kg_s
            energy_sources[:nl] += liquid_transport.wall_enthalpy_flow_W
            energy_sources[nl:2 * nl] += liquid_transport.core_enthalpy_flow_W
            energy_sources[2 * nl:2 * nl + nv] += vapor_transport.wall_enthalpy_flow_W
            energy_sources[2 * nl + nv:] += vapor_transport.core_enthalpy_flow_W

        def pair_conduction(left: int, right: int, conductance: float) -> None:
            heat = conductance * (
                cells[left].temperature_K - cells[right].temperature_K
            )
            energy_sources[left] -= heat
            energy_sources[right] += heat

        for level in range(nl):
            pair_conduction(level, nl + level, p.liquid_radial_conductance_W_K)
        for level in range(nv):
            pair_conduction(
                2 * nl + level,
                2 * nl + nv + level,
                p.vapor_radial_conductance_W_K,
            )
        for level in range(nl - 1):
            pair_conduction(level, level + 1, p.liquid_axial_conductance_W_K)
            pair_conduction(
                nl + level, nl + level + 1, p.liquid_axial_conductance_W_K
            )
        for level in range(nv - 1):
            pair_conduction(
                2 * nl + level,
                2 * nl + level + 1,
                p.vapor_axial_conductance_W_K,
            )
            pair_conduction(
                2 * nl + nv + level,
                2 * nl + nv + level + 1,
                p.vapor_axial_conductance_W_K,
            )

        saturated_liquid = self.properties.saturated_liquid(state.pressure_Pa)
        saturated_vapor = self.properties.saturated_vapor(state.pressure_Pa)
        latent_heat = (
            saturated_vapor.specific_enthalpy_J_kg
            - saturated_liquid.specific_enthalpy_J_kg
        )
        if not math.isfinite(latent_heat) or latent_heat <= 0.0:
            raise ValueError("interface phase change requires positive latent heat")
        interface_rates: list[float] = []
        interface_energy_residual = 0.0
        interface_pairs = (
            (nl - 1, 2 * nl, p.liquid_interface_UA_W_K[0], p.vapor_interface_UA_W_K[0]),
            (2 * nl - 1, 2 * nl + nv, p.liquid_interface_UA_W_K[1], p.vapor_interface_UA_W_K[1]),
        )
        for liquid_index, vapor_index, liquid_ua, vapor_ua in interface_pairs:
            q_liquid = liquid_ua * (
                cells[liquid_index].temperature_K - saturated_liquid.temperature_K
            )
            if p.vapor_interface_heat_transfer_model == "kinetic_gas_film":
                # The molecular-impingement coefficient is a gas-side closure;
                # the explicit interface area is kept separate from the
                # conductance fallback so either path can be replaced without
                # changing the phase-change or port contracts.
                areas = p.vapor_interface_area_m2
                if areas is None:  # guarded by parameter validation
                    raise RuntimeError(
                        "kinetic_gas_film requires vapor interface areas"
                    )
                area = areas[len(interface_rates)]
                kinetic = kinetic_gas_film_heat_transfer_coefficient(
                    pressure_Pa=cells[vapor_index].pressure_Pa,
                    temperature_K=cells[vapor_index].temperature_K,
                    isobaric_heat_capacity_J_kgK=(
                        self.properties.isobaric_heat_capacity_J_kgK(
                            cells[vapor_index]
                        )
                    ),
                    specific_gas_constant_J_kgK=(
                        self.properties.specific_gas_constant_J_kgK()
                    ),
                    energy_accommodation_coefficient=(
                        p.gas_interface_energy_accommodation_coefficient
                    ),
                )
                q_vapor = kinetic.heat_transfer_coefficient_W_m2K * area * (
                    cells[vapor_index].temperature_K - saturated_vapor.temperature_K
                )
            else:
                q_vapor = vapor_ua * (
                    cells[vapor_index].temperature_K - saturated_vapor.temperature_K
                )
            rate = (q_liquid + q_vapor) / latent_heat
            interface_rates.append(rate)
            mass_rates[liquid_index] -= rate
            mass_rates[vapor_index] += rate
            liquid_source = -q_liquid - rate * saturated_liquid.specific_enthalpy_J_kg
            vapor_source = -q_vapor + rate * saturated_vapor.specific_enthalpy_J_kg
            energy_sources[liquid_index] += liquid_source
            energy_sources[vapor_index] += vapor_source
            interface_energy_residual += liquid_source + vapor_source

        wall_to_fluid = np.zeros(self.wall_patch_count)
        wall_cell_indices = (*range(nl), *range(2 * nl, 2 * nl + nv))
        for patch, cell_index in enumerate(wall_cell_indices):
            heat = self._wall_fluid_ua[patch] * (
                state.wall_temperatures_K[patch] - cells[cell_index].temperature_K
            )
            wall_to_fluid[patch] = heat
            energy_sources[cell_index] += heat
        q_ambient = self._ambient_ua * (
            ambient_temperature_K - np.asarray(state.wall_temperatures_K)
        )
        q_wall_axial = np.zeros(self.wall_patch_count)
        for patch in range(self.wall_patch_count - 1):
            heat = p.wall_axial_conductance_W_K * (
                state.wall_temperatures_K[patch + 1]
                - state.wall_temperatures_K[patch]
            )
            q_wall_axial[patch] += heat
            q_wall_axial[patch + 1] -= heat
        q_wall_net = q_ambient + direct_wall - wall_to_fluid + q_wall_axial
        wall_temperature_rates = q_wall_net / self._wall_capacities

        boundary_mass = 0.0
        boundary_stream_energy = 0.0
        if cell_streams is None:
            stream_groups: tuple[tuple[MassEnergyFlow, ...], ...] = (
                ((),) * nc
            )
        else:
            if len(cell_streams) != nc:
                raise ValueError(f"cell_streams must contain {nc} cell groups")
            stream_groups = tuple(tuple(group) for group in cell_streams)
        for index, group in enumerate(stream_groups):
            for stream in group:
                if not math.isfinite(stream.mass_flow_kg_s) or not math.isfinite(
                    stream.specific_enthalpy_J_kg
                ):
                    raise ValueError("all material streams must be finite")
                mass_rates[index] += stream.mass_flow_kg_s
                energy_sources[index] += stream.enthalpy_flow_W
                boundary_mass += stream.mass_flow_kg_s
                boundary_stream_energy += stream.enthalpy_flow_W

        distributed_boiling = np.zeros(2 * nl)
        distributed_boiling_latent_power = 0.0
        if p.distributed_boiling_model == "saturated_liquid_complementarity":
            masses_array = np.asarray(masses, dtype=float)
            enthalpies_array = np.asarray(enthalpies, dtype=float)
            volumes = np.asarray(thermo.cell_volumes_m3, dtype=float)
            specific_volumes = volumes / masses_array
            vp = np.empty(nc)
            vh = np.empty(nc)
            for index, cell in enumerate(cells):
                vp[index], vh[index] = (
                    self.properties.specific_volume_derivatives_ph(cell)
                )
            reduced_sources = energy_sources - enthalpies_array * mass_rates
            denominator = float(np.sum(masses_array * vp + vh * volumes))
            if not math.isfinite(denominator) or abs(denominator) < 1.0e-20:
                raise ValueError("distributed-boiling pressure equation is singular")
            saturation_slope = (
                self.properties.saturation_enthalpy_pressure_derivative_J_kg_Pa(
                    state.pressure_Pa, 0.0
                )
            )
            saturation_tolerance = latent_heat * 1.0e-7
            eligible = {
                index
                for index in range(2 * nl)
                if enthalpies_array[index]
                >= saturated_liquid.specific_enthalpy_J_kg - saturation_tolerance
            }
            base_numerator = float(np.sum(
                specific_volumes * mass_rates + vh * reduced_sources
            ))

            def vapor_recipient(liquid_index: int) -> int:
                return 2 * nl if liquid_index < nl else 2 * nl + nv

            def solve_active(active: list[int]) -> tuple[float, np.ndarray]:
                matrix = np.zeros((len(active) + 1, len(active) + 1))
                rhs = np.zeros(len(active) + 1)
                matrix[0, 0] = denominator
                rhs[0] = -base_numerator
                for column, liquid_index in enumerate(active, start=1):
                    vapor_index = vapor_recipient(liquid_index)
                    vapor_source_enthalpy = (
                        saturated_vapor.specific_enthalpy_J_kg
                        - enthalpies_array[vapor_index]
                    )
                    matrix[0, column] = (
                        specific_volumes[vapor_index]
                        - specific_volumes[liquid_index]
                        - vh[liquid_index] * latent_heat
                        + vh[vapor_index] * vapor_source_enthalpy
                    )
                    matrix[column, 0] = (
                        masses_array[liquid_index] * saturation_slope
                        - volumes[liquid_index]
                    )
                    matrix[column, column] = latent_heat
                    rhs[column] = reduced_sources[liquid_index]
                solution = np.linalg.solve(matrix, rhs)
                return float(solution[0]), np.asarray(solution[1:])

            active: list[int] = []
            pressure_rate, active_rates = solve_active(active)
            rate_tolerance = 1.0e-12
            for _ in range(4 * nl + 2):
                negative = [
                    (active[index], rate)
                    for index, rate in enumerate(active_rates)
                    if rate < -rate_tolerance
                ]
                if negative:
                    active.remove(min(negative, key=lambda pair: pair[1])[0])
                    pressure_rate, active_rates = solve_active(active)
                    continue
                violations: list[tuple[int, float]] = []
                for liquid_index in sorted(eligible.difference(active)):
                    free_margin_rate = (
                        reduced_sources[liquid_index]
                        + (
                            volumes[liquid_index]
                            - masses_array[liquid_index] * saturation_slope
                        ) * pressure_rate
                    ) / masses_array[liquid_index]
                    if free_margin_rate > rate_tolerance:
                        violations.append((liquid_index, free_margin_rate))
                if not violations:
                    break
                active.append(max(violations, key=lambda pair: pair[1])[0])
                active.sort()
                pressure_rate, active_rates = solve_active(active)
            else:
                raise RuntimeError("distributed-boiling active set did not converge")

            for liquid_index, rate in zip(active, active_rates):
                rate = max(0.0, float(rate))
                vapor_index = vapor_recipient(liquid_index)
                distributed_boiling[liquid_index] = rate
                mass_rates[liquid_index] -= rate
                mass_rates[vapor_index] += rate
                reduced_sources[liquid_index] -= rate * latent_heat
                reduced_sources[vapor_index] += rate * (
                    saturated_vapor.specific_enthalpy_J_kg
                    - enthalpies_array[vapor_index]
                )
            distributed_boiling_latent_power = float(
                np.sum(distributed_boiling) * latent_heat
            )
            energy_sources = reduced_sources + enthalpies_array * mass_rates

        dae = self._dae.derivative(
            CommonPressureCellState(state.pressure_Pa, masses, enthalpies),
            mass_rates,
            energy_sources,
        )
        boundary_energy = float(
            boundary_stream_energy
            + np.sum(self._finite_vector(direct_fluid_heat_W, nc, "direct_fluid_heat_W"))
            + np.sum(q_ambient + direct_wall)
        )
        total_energy = float(
            dae.total_fluid_energy_rate_W
            + np.sum(self._wall_capacities * wall_temperature_rates)
        )
        total_mass = dae.total_mass_rate_kg_s
        return RadialAxialTankDerivative(
            pressure_rate_Pa_s=dae.pressure_rate_Pa_s,
            mass_rates_kg_s=dae.mass_rates_kg_s,
            enthalpy_rates_J_kg_s=dae.enthalpy_rates_J_kg_s,
            wall_temperature_rates_K_s=tuple(
                float(value) for value in wall_temperature_rates
            ),
            enthalpy_inventory_sources_W=tuple(
                float(value) for value in energy_sources
            ),
            interface_phase_change_rates_kg_s=(
                float(interface_rates[0]), float(interface_rates[1])
            ),
            interface_energy_residual_W=float(interface_energy_residual),
            distributed_boiling_rates_kg_s=tuple(
                float(value) for value in distributed_boiling
            ),
            distributed_boiling_rate_kg_s=float(np.sum(distributed_boiling)),
            distributed_boiling_latent_power_W=distributed_boiling_latent_power,
            liquid_transport=liquid_transport,
            vapor_transport=vapor_transport,
            common_pressure=dae,
            boundary_mass_rate_kg_s=float(boundary_mass),
            boundary_energy_rate_W=boundary_energy,
            total_mass_rate_kg_s=total_mass,
            total_energy_rate_W=total_energy,
            mass_conservation_residual_kg_s=total_mass - boundary_mass,
            energy_conservation_residual_W=total_energy - boundary_energy,
        )

    def total_mass_kg(self, state: RadialAxialTankState) -> float:
        return float(sum(self._flatten_state(state)[0]))

    def recommended_explicit_time_step_s(
        self,
        state: RadialAxialTankState,
        ambient_temperature_K: float,
        cell_streams: Sequence[Iterable[MassEnergyFlow]] | None = None,
        direct_fluid_heat_W: Sequence[float] | None = None,
        direct_wall_heat_W: Sequence[float] | None = None,
        *,
        safety_factor: float = 0.5,
        maximum_time_step_s: float | None = None,
    ) -> float:
        """Return a conservative explicit-step bound from current rates.

        The bound is derived from the current pressure, cell masses, and wall
        temperatures together with the first-principles derivative.  It is a
        caller-facing guard for explicit integration; it does not alter the
        equations, fit a coefficient, or guarantee that phase projection will
        succeed.  A caller should still handle a rejected step and reduce the
        interval when a nonlinear EOS or phase-admissibility check requires it.
        """

        factor = float(safety_factor)
        if not math.isfinite(factor) or not 0.0 < factor < 1.0:
            raise ValueError("safety_factor must be finite and lie in (0, 1)")
        if maximum_time_step_s is not None:
            maximum = float(maximum_time_step_s)
            if not math.isfinite(maximum) or maximum <= 0.0:
                raise ValueError("maximum_time_step_s must be finite and positive")
        else:
            maximum = math.inf

        derivative = self.derivative(
            state,
            ambient_temperature_K,
            cell_streams=cell_streams,
            direct_fluid_heat_W=direct_fluid_heat_W,
            direct_wall_heat_W=direct_wall_heat_W,
        )
        limits: list[float] = [maximum]

        pressure_rate = derivative.pressure_rate_Pa_s
        if pressure_rate > 0.0:
            limits.append((self.parameters.maximum_pressure_Pa - state.pressure_Pa) / pressure_rate)
        elif pressure_rate < 0.0:
            limits.append((state.pressure_Pa - self.parameters.minimum_pressure_Pa) / -pressure_rate)

        masses = np.asarray(self.cell_masses_kg(state), dtype=float)
        mass_rates = np.asarray(derivative.mass_rates_kg_s, dtype=float)
        for mass, rate in zip(masses, mass_rates):
            if rate < 0.0:
                limits.append(mass / -rate)

        wall_temperatures = np.asarray(state.wall_temperatures_K, dtype=float)
        wall_rates = np.asarray(derivative.wall_temperature_rates_K_s, dtype=float)
        for temperature, rate in zip(wall_temperatures, wall_rates):
            if rate < 0.0:
                limits.append(temperature / -rate)

        finite_limits = [value for value in limits if math.isfinite(value)]
        if any(value <= 0.0 for value in finite_limits):
            raise ValueError(
                "current derivative has no positive explicit step before a physical bound"
            )
        return float(factor * min(finite_limits, default=math.inf))

    def cell_masses_kg(self, state: RadialAxialTankState) -> tuple[float, ...]:
        """Return masses in the stable port-index order."""

        self._validate_state_lengths(state)
        return self._flatten_state(state)[0]

    def cell_enthalpies_J_kg(
        self, state: RadialAxialTankState
    ) -> tuple[float, ...]:
        """Return specific enthalpies in the stable port-index order."""

        self._validate_state_lengths(state)
        return self._flatten_state(state)[1]

    def total_energy_J(self, state: RadialAxialTankState) -> float:
        masses, enthalpies = self._flatten_state(state)
        fluid_internal = (
            float(np.dot(masses, enthalpies))
            - state.pressure_Pa * self.parameters.total_volume_m3
        )
        wall = float(np.dot(self._wall_capacities, state.wall_temperatures_K))
        return fluid_internal + wall

    def project_state(
        self,
        state: RadialAxialTankState,
        target_fluid_internal_energy_J: float | None = None,
    ) -> RadialAxialTankState:
        """Project to rigid volume while preserving mass and fluid energy.

        A common enthalpy correction is applied to unconstrained cells while
        pressure is solved from the EOS volume constraint.  When the
        saturated-liquid complementarity is enabled, labeled cells that cross
        into the two-phase interval are fixed to the appropriate saturation
        boundary.  The projection is numerical, contains no fitted parameter,
        and preserves the supplied total fluid internal energy exactly.
        """

        self._validate_state_lengths(state)
        masses, enthalpies_tuple = self._flatten_state(state)
        masses_array = np.asarray(masses, dtype=float)
        enthalpies = np.asarray(enthalpies_tuple, dtype=float)
        volume = self.parameters.total_volume_m3
        if target_fluid_internal_energy_J is None:
            target_fluid_internal_energy_J = (
                float(np.dot(masses_array, enthalpies)) - state.pressure_Pa * volume
            )
        if not math.isfinite(target_fluid_internal_energy_J):
            raise ValueError("target fluid internal energy must be finite")

        pmin = self.parameters.minimum_pressure_Pa
        pmax = self.parameters.maximum_pressure_Pa
        volume_tolerance = max(volume, 1.0) * 1.0e-12
        nl = self.parameters.liquid_level_count
        liquid_indices = set(range(2 * nl))
        vapor_indices = set(range(2 * nl, self.cell_count))
        active_liquid: set[int] = set()
        active_vapor: set[int] = set()
        saturation_at_hint_l = self.properties.saturated_liquid(state.pressure_Pa)
        saturation_at_hint_v = self.properties.saturated_vapor(state.pressure_Pa)
        latent_at_hint = (
            saturation_at_hint_v.specific_enthalpy_J_kg
            - saturation_at_hint_l.specific_enthalpy_J_kg
        )
        phase_tolerance = latent_at_hint * 1.0e-8
        for index in liquid_indices:
            if enthalpies[index] >= saturation_at_hint_v.specific_enthalpy_J_kg:
                raise ValueError("labeled liquid cell crossed the complete two-phase interval")
            if enthalpies[index] > saturation_at_hint_l.specific_enthalpy_J_kg + phase_tolerance:
                active_liquid.add(index)
        for index in vapor_indices:
            if enthalpies[index] <= saturation_at_hint_l.specific_enthalpy_J_kg:
                raise ValueError("labeled vapor cell crossed the complete two-phase interval")
            if enthalpies[index] < saturation_at_hint_v.specific_enthalpy_J_kg - phase_tolerance:
                active_vapor.add(index)
        if (
            (active_liquid or active_vapor)
            and self.parameters.distributed_boiling_model == "off"
        ):
            raise ValueError(
                "projection entered the two-phase region while distributed boiling is off"
            )

        def solve_projection() -> tuple[float, float, np.ndarray]:
            fixed = active_liquid | active_vapor
            free = [index for index in range(self.cell_count) if index not in fixed]
            free_mass = float(np.sum(masses_array[free]))
            if free_mass <= 0.0:
                raise ValueError("phase projection requires at least one unconstrained cell")

            def build(pressure: float) -> tuple[float, np.ndarray]:
                saturated_l = self.properties.saturated_liquid(pressure)
                saturated_v = self.properties.saturated_vapor(pressure)
                adjusted = enthalpies.copy()
                for index in active_liquid:
                    adjusted[index] = saturated_l.specific_enthalpy_J_kg
                for index in active_vapor:
                    adjusted[index] = saturated_v.specific_enthalpy_J_kg
                fixed_energy = float(np.dot(masses_array[list(fixed)], adjusted[list(fixed)])) if fixed else 0.0
                free_base_energy = float(np.dot(masses_array[free], enthalpies[free]))
                correction = (
                    target_fluid_internal_energy_J
                    + pressure * volume
                    - fixed_energy
                    - free_base_energy
                ) / free_mass
                adjusted[free] = enthalpies[free] + correction
                actual_volume = 0.0
                for mass, specific_enthalpy in zip(masses_array, adjusted):
                    cell = self.properties.from_ph(pressure, float(specific_enthalpy))
                    actual_volume += mass / cell.density_kg_m3
                return actual_volume - volume, adjusted

            pressure = min(max(state.pressure_Pa, pmin), pmax)
            converged = False
            try:
                for _ in range(12):
                    residual = build(pressure)[0]
                    if abs(residual) <= volume_tolerance:
                        converged = True
                        break
                    increment = max(pressure * 1.0e-5, 0.1)
                    lower = max(pmin, pressure - increment)
                    upper = min(pmax, pressure + increment)
                    slope = (build(upper)[0] - build(lower)[0]) / (upper - lower)
                    if not math.isfinite(slope) or abs(slope) < 1.0e-20:
                        break
                    candidate = pressure - residual / slope
                    pressure = min(
                        max(candidate, max(pmin, pressure * 0.5)),
                        min(pmax, pressure * 1.5),
                    )
            except (ValueError, RuntimeError, OverflowError):
                converged = False
            if not converged:
                samples: list[tuple[float, float]] = []
                candidates = sorted(set(
                    [float(value) for value in np.geomspace(pmin, pmax, 180)]
                    + [min(max(state.pressure_Pa, pmin), pmax)]
                ))
                for candidate in candidates:
                    try:
                        value = build(candidate)[0]
                    except (ValueError, RuntimeError, OverflowError):
                        continue
                    if math.isfinite(value):
                        samples.append((candidate, value))
                brackets = [
                    (left[0], right[0])
                    for left, right in zip(samples, samples[1:])
                    if left[1] * right[1] < 0.0
                ]
                if not brackets:
                    raise ValueError(
                        "no pressure/enthalpy phase projection satisfies volume"
                    )
                low, high = min(
                    brackets,
                    key=lambda pair: abs(
                        math.log(math.sqrt(pair[0] * pair[1]) / state.pressure_Pa)
                    ),
                )
                pressure = float(brentq(
                    lambda value: build(value)[0], low, high,
                    xtol=1.0e-7, rtol=1.0e-11,
                ))
            residual, adjusted = build(pressure)
            return pressure, residual, adjusted

        for _ in range(self.cell_count + 1):
            pressure, residual, adjusted = solve_projection()
            saturated_l = self.properties.saturated_liquid(pressure)
            saturated_v = self.properties.saturated_vapor(pressure)
            latent = saturated_v.specific_enthalpy_J_kg - saturated_l.specific_enthalpy_J_kg
            tolerance_h = latent * 1.0e-8
            if any(
                adjusted[index] >= saturated_v.specific_enthalpy_J_kg
                for index in liquid_indices.difference(active_liquid)
            ):
                raise ValueError(
                    "labeled liquid cell crossed the complete two-phase interval; "
                    "reduce the time step"
                )
            if any(
                adjusted[index] <= saturated_l.specific_enthalpy_J_kg
                for index in vapor_indices.difference(active_vapor)
            ):
                raise ValueError(
                    "labeled vapor cell crossed the complete two-phase interval; "
                    "reduce the time step"
                )
            new_liquid = {
                index for index in liquid_indices.difference(active_liquid)
                if adjusted[index] > saturated_l.specific_enthalpy_J_kg + tolerance_h
            }
            new_vapor = {
                index for index in vapor_indices.difference(active_vapor)
                if adjusted[index] < saturated_v.specific_enthalpy_J_kg - tolerance_h
            }
            if not new_liquid and not new_vapor:
                break
            if self.parameters.distributed_boiling_model == "off":
                raise ValueError(
                    "projection entered the two-phase region while distributed boiling is off"
                )
            active_liquid.update(new_liquid)
            active_vapor.update(new_vapor)
        else:
            raise RuntimeError("radial-axial phase projection active set did not converge")
        projected = self._unflatten_state(
            pressure, masses_array, adjusted, state.wall_temperatures_K
        )
        projected_thermo = self.thermo(projected)
        tolerance = max(volume, 1.0) * 1.0e-10
        if abs(residual) > tolerance or abs(projected_thermo.volume_residual_m3) > tolerance:
            raise RuntimeError("radial-axial tank projection did not close volume")
        projected_fluid_energy = (
            float(np.dot(masses_array, adjusted)) - pressure * volume
        )
        energy_tolerance = max(abs(target_fluid_internal_energy_J), 1.0) * 1.0e-10
        if not math.isclose(
            projected_fluid_energy,
            target_fluid_internal_energy_J,
            rel_tol=0.0,
            abs_tol=energy_tolerance,
        ):
            raise RuntimeError("radial-axial tank projection did not conserve energy")
        return projected

    @staticmethod
    def _implicit_circulation_block(
        masses_kg: np.ndarray,
        enthalpies_J_kg: np.ndarray,
        boundary_flows_kg_s: Sequence[float],
        time_step_s: float,
    ) -> np.ndarray:
        """Implicit donor-cell solve for local closed circulation loops."""

        level_count = len(masses_kg) // 2
        if masses_kg.shape != (2 * level_count,) or enthalpies_J_kg.shape != (
            2 * level_count,
        ):
            raise ValueError("circulation block requires matching wall/core arrays")
        if len(boundary_flows_kg_s) != level_count - 1:
            raise ValueError("circulation flow count must equal levels minus one")
        matrix = np.diag(masses_kg / time_step_s)
        rhs = masses_kg * enthalpies_J_kg / time_step_s
        for lower, signed_flow in enumerate(boundary_flows_kg_s):
            magnitude = abs(float(signed_flow))
            if magnitude == 0.0:
                continue
            upper = lower + 1
            if signed_flow >= 0.0:
                cycle = (
                    lower,
                    upper,
                    level_count + upper,
                    level_count + lower,
                )
            else:
                cycle = (
                    upper,
                    lower,
                    level_count + lower,
                    level_count + upper,
                )
            for donor, receiver in zip(cycle, (*cycle[1:], cycle[0])):
                matrix[receiver, receiver] += magnitude
                matrix[receiver, donor] -= magnitude
        result = np.linalg.solve(matrix, rhs)
        old_inventory = float(np.dot(masses_kg, enthalpies_J_kg))
        new_inventory = float(np.dot(masses_kg, result))
        tolerance = max(abs(old_inventory), 1.0) * 1.0e-11
        if not math.isclose(
            old_inventory, new_inventory, rel_tol=0.0, abs_tol=tolerance
        ):
            raise RuntimeError("implicit circulation did not conserve enthalpy")
        return result

    def step_euler(
        self,
        state: RadialAxialTankState,
        time_step_s: float,
        ambient_temperature_K: float,
        cell_streams: Sequence[Iterable[MassEnergyFlow]] | None = None,
        direct_fluid_heat_W: Sequence[float] | None = None,
        direct_wall_heat_W: Sequence[float] | None = None,
    ) -> RadialAxialTankStepResult:
        """Advance explicit sources, implicit circulation, and project DAE.

        The closed natural-circulation loops are stiff relative to external
        heating in small vapor inventories.  Their donor-cell transport is
        therefore solved by backward Euler with frozen momentum flow while
        all remaining sources use forward Euler.  Both substeps conserve the
        extensive inventories used by the final volume/energy projection.
        """

        if not math.isfinite(time_step_s) or time_step_s <= 0.0:
            raise ValueError("time_step_s must be finite and positive")
        initial_mass = self.total_mass_kg(state)
        initial_energy = self.total_energy_J(state)
        derivative = self.derivative(
            state,
            ambient_temperature_K,
            cell_streams,
            direct_fluid_heat_W,
            direct_wall_heat_W,
            include_circulation_sources=False,
        )
        masses, enthalpies = self._flatten_state(state)
        provisional_masses = np.asarray(masses) + time_step_s * np.asarray(
            derivative.mass_rates_kg_s
        )
        if np.any(provisional_masses <= 0.0):
            raise ValueError("time step would exhaust a fluid cell")
        provisional_enthalpies = np.asarray(enthalpies) + time_step_s * np.asarray(
            derivative.enthalpy_rates_J_kg_s
        )
        nl = self.parameters.liquid_level_count
        nv = self.parameters.vapor_level_count
        provisional_enthalpies[:2 * nl] = self._implicit_circulation_block(
            provisional_masses[:2 * nl],
            provisional_enthalpies[:2 * nl],
            derivative.liquid_transport.boundary_mass_flows_kg_s,
            time_step_s,
        )
        provisional_enthalpies[2 * nl:] = self._implicit_circulation_block(
            provisional_masses[2 * nl:],
            provisional_enthalpies[2 * nl:],
            derivative.vapor_transport.boundary_mass_flows_kg_s,
            time_step_s,
        )
        provisional_walls = np.asarray(state.wall_temperatures_K) + time_step_s * np.asarray(
            derivative.wall_temperature_rates_K_s
        )
        if np.any(provisional_walls <= 0.0):
            raise ValueError("time step would produce a non-positive wall temperature")
        provisional = self._unflatten_state(
            state.pressure_Pa + time_step_s * derivative.pressure_rate_Pa_s,
            provisional_masses,
            provisional_enthalpies,
            provisional_walls,
        )
        target_total_energy = (
            initial_energy + time_step_s * derivative.boundary_energy_rate_W
        )
        target_fluid_energy = target_total_energy - float(
            np.dot(self._wall_capacities, provisional_walls)
        )
        projected = self.project_state(provisional, target_fluid_energy)
        thermo = self.thermo(projected)
        expected_mass = initial_mass + time_step_s * derivative.boundary_mass_rate_kg_s
        expected_energy = target_total_energy
        return RadialAxialTankStepResult(
            state=projected,
            thermo=thermo,
            derivative=derivative,
            total_mass_residual_kg=self.total_mass_kg(projected) - expected_mass,
            total_energy_residual_J=self.total_energy_J(projected) - expected_energy,
        )
