"""Conservative vertical finite-volume tank with a common pressure.

Each Lagrangian fluid cell stores mass and specific enthalpy.  The common
pressure derivative follows from the rigid-volume constraint, while each cell
enthalpy equation includes ``V dp/dt``.  This makes internal pressure work,
axial conduction and phase change cancel exactly in the complete-tank energy
balance.  Closure multipliers are explicit and default to molecular
conduction; no facility-data calibration is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import brentq

from .geometry import HorizontalVesselGeometry
from .natural_convection import (
    churchill_chu_vertical_plate,
    daigle_horizontal_surface,
    daigle_vertical_wall_boundary_layer,
    daigle_vertical_wall_uniform_heat_flux,
    yang_west_2015_cryogenic_tank,
)
from .properties import HydrogenProperties, ThermoState
from .phase_change import (
    kinetic_gas_film_heat_transfer_coefficient,
    schrage_mass_flux,
)
from .streams import MassEnergyFlow


@dataclass(frozen=True)
class LayeredTankParameters:
    liquid_cell_count: int = 3
    vapor_cell_count: int = 3
    wall_node_count: int = 8
    wall_heat_capacity_J_K: float = 1.0
    ambient_UA_W_K: float = 0.0
    wall_liquid_U_W_m2K: float = 0.0
    wall_vapor_U_W_m2K: float = 0.0
    wall_liquid_heat_transfer_multiplier: float = 1.0
    wall_vapor_heat_transfer_multiplier: float = 1.0
    wall_heat_transfer_model: str = "constant"
    wall_axial_conductance_W_K: float = 0.0
    connected_vapor_volume_m3: float = 0.0
    connected_wall_heat_capacity_J_K: float = 0.0
    connected_ambient_UA_W_K: float = 0.0
    connected_wall_vapor_UA_W_K: float = 0.0
    liquid_axial_conduction_multiplier: float = 1.0
    vapor_axial_conduction_multiplier: float = 1.0
    liquid_interface_conduction_multiplier: float = 1.0
    vapor_interface_conduction_multiplier: float = 1.0
    interface_heat_transfer_model: str = "conduction"
    gas_interface_energy_accommodation_coefficient: float = 1.0
    interface_phase_change_model: str = "equilibrium_energy_jump"
    schrage_accommodation_coefficient: float = 0.001
    distributed_boiling_model: str = "off"
    vertical_wall_circulation_model: str = "off"
    gravity_m_s2: float = 9.80665
    minimum_pressure_Pa: float = 10_000.0
    maximum_pressure_Pa: float = 1_250_000.0
    # Optional per-wall-node ambient conductance.  When supplied, its sum
    # must equal ambient_UA_W_K.  This keeps upper/lower insulation zones and
    # explicit penetration allocations editable without hiding them in a
    # fitted scalar UA; the empty tuple preserves the historical uniform
    # distribution.
    ambient_UA_profile_W_K: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("liquid_cell_count", "vapor_cell_count"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.wall_node_count, int) or self.wall_node_count < 2:
            raise ValueError("wall_node_count must be an integer of at least two")
        positive = (
            "wall_heat_capacity_J_K", "liquid_axial_conduction_multiplier",
            "vapor_axial_conduction_multiplier", "liquid_interface_conduction_multiplier",
            "vapor_interface_conduction_multiplier", "minimum_pressure_Pa",
            "maximum_pressure_Pa", "gravity_m_s2",
            "gas_interface_energy_accommodation_coefficient",
            "wall_liquid_heat_transfer_multiplier",
            "wall_vapor_heat_transfer_multiplier",
        )
        nonnegative = (
            "ambient_UA_W_K", "wall_liquid_U_W_m2K",
            "wall_vapor_U_W_m2K", "wall_axial_conductance_W_K",
            "connected_vapor_volume_m3", "connected_wall_heat_capacity_J_K",
            "connected_ambient_UA_W_K", "connected_wall_vapor_UA_W_K",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        profile = tuple(self.ambient_UA_profile_W_K)
        if profile:
            if len(profile) != self.wall_node_count:
                raise ValueError(
                    "ambient_UA_profile_W_K length must match wall_node_count"
                )
            if any(
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in profile
            ):
                raise ValueError(
                    "ambient_UA_profile_W_K must contain finite non-negative values"
                )
            if not math.isclose(
                sum(float(value) for value in profile),
                self.ambient_UA_W_K,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "ambient_UA_profile_W_K must sum to ambient_UA_W_K"
                )
            object.__setattr__(self, "ambient_UA_profile_W_K", profile)
        if self.maximum_pressure_Pa <= self.minimum_pressure_Pa:
            raise ValueError("maximum_pressure_Pa must exceed minimum_pressure_Pa")
        if self.interface_heat_transfer_model not in {
            "conduction", "daigle_2013", "kinetic_gas_film"
        }:
            raise ValueError(
                "interface_heat_transfer_model must be 'conduction', "
                "'daigle_2013', or 'kinetic_gas_film'"
            )
        if self.interface_phase_change_model not in {
            "equilibrium_energy_jump", "schrage_kinetic_energy_balance"
        }:
            raise ValueError(
                "interface_phase_change_model must be "
                "'equilibrium_energy_jump' or "
                "'schrage_kinetic_energy_balance'"
            )
        if (
            not math.isfinite(self.schrage_accommodation_coefficient)
            or not 0.0 < self.schrage_accommodation_coefficient <= 1.0
        ):
            raise ValueError(
                "schrage_accommodation_coefficient must be finite and in (0, 1]"
            )
        if self.distributed_boiling_model not in {
            "off", "saturated_liquid_complementarity",
            "phase_change_complementarity",
        }:
            raise ValueError(
                "distributed_boiling_model must be 'off' or "
                "'saturated_liquid_complementarity' or "
                "'phase_change_complementarity'"
            )
        if self.wall_heat_transfer_model not in {
            "constant", "churchill_chu_vertical_plate",
            "yang_west_2015_cryogenic_tank",
        }:
            raise ValueError(
                "wall_heat_transfer_model must be 'constant' or "
                "'churchill_chu_vertical_plate' or "
                "'yang_west_2015_cryogenic_tank'"
            )
        if self.vertical_wall_circulation_model not in {
            "off",
            "daigle_2013_reduced",
            "daigle_2013_uniform_heat_flux_reduced",
        }:
            raise ValueError(
                "vertical_wall_circulation_model must be 'off', "
                "'daigle_2013_reduced', or "
                "'daigle_2013_uniform_heat_flux_reduced'"
            )


@dataclass(frozen=True)
class LayeredTankState:
    pressure_Pa: float
    liquid_masses_kg: tuple[float, ...]
    liquid_specific_enthalpies_J_kg: tuple[float, ...]
    vapor_masses_kg: tuple[float, ...]
    vapor_specific_enthalpies_J_kg: tuple[float, ...]
    wall_temperatures_K: tuple[float, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.pressure_Pa) or self.pressure_Pa <= 0.0:
            raise ValueError("pressure_Pa must be finite and positive")
        if len(self.liquid_masses_kg) != len(self.liquid_specific_enthalpies_J_kg):
            raise ValueError("Liquid mass and enthalpy arrays must have equal length")
        if len(self.vapor_masses_kg) != len(self.vapor_specific_enthalpies_J_kg):
            raise ValueError("Vapor mass and enthalpy arrays must have equal length")
        if not self.liquid_masses_kg or not self.vapor_masses_kg or not self.wall_temperatures_K:
            raise ValueError("Liquid, vapor and wall arrays cannot be empty")
        if any(not math.isfinite(value) or value <= 0.0 for value in (
            *self.liquid_masses_kg, *self.vapor_masses_kg,
        )):
            raise ValueError("All cell masses must be finite and positive")
        if any(not math.isfinite(value) for value in (
            *self.liquid_specific_enthalpies_J_kg,
            *self.vapor_specific_enthalpies_J_kg,
        )):
            raise ValueError("All cell enthalpies must be finite")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.wall_temperatures_K):
            raise ValueError("All wall temperatures must be finite and positive")


@dataclass(frozen=True)
class LayeredThermoState:
    pressure_Pa: float
    liquid: tuple[ThermoState, ...]
    vapor: tuple[ThermoState, ...]
    cell_volumes_m3: tuple[float, ...]
    cell_edge_heights_m: tuple[float, ...]
    cell_center_heights_m: tuple[float, ...]
    wall_overlap_areas_m2: tuple[tuple[float, ...], ...]
    liquid_height_m: float
    interface_area_m2: float
    volume_residual_m3: float
    tank_cell_volumes_m3: tuple[float, ...]
    # The wall state is exposed to boundary models so a decomposed vacuum
    # insulation law can be connected to the same wall control volumes.  It
    # remains an observation of the model state, not a fitted temperature.
    wall_temperatures_K: tuple[float, ...]


@dataclass(frozen=True)
class LayeredTankDerivative:
    pressure_Pa_s: float
    liquid_mass_rates_kg_s: tuple[float, ...]
    liquid_enthalpy_rates_J_kg_s: tuple[float, ...]
    vapor_mass_rates_kg_s: tuple[float, ...]
    vapor_enthalpy_rates_J_kg_s: tuple[float, ...]
    wall_temperature_rates_K_s: tuple[float, ...]
    evaporation_rate_kg_s: float
    liquid_interface_heat_W: float
    vapor_interface_heat_W: float
    latent_phase_change_power_W: float
    distributed_boiling_rates_kg_s: tuple[float, ...]
    distributed_boiling_rate_kg_s: float
    distributed_boiling_latent_power_W: float
    interface_temperature_K: float
    interface_pressure_Pa: float
    interface_mass_flux_kg_m2_s: float
    interface_kinetic_residual_kg_m2_s: float
    interface_energy_residual_W: float
    liquid_wall_circulation_rates_kg_s: tuple[float, ...]
    vapor_wall_circulation_rates_kg_s: tuple[float, ...]
    liquid_circulation_lower_cell_heat_W: tuple[float, ...]
    vapor_circulation_lower_cell_heat_W: tuple[float, ...]
    liquid_wall_heat_W: tuple[float, ...]
    vapor_wall_heat_W: tuple[float, ...]
    ambient_heat_W: float
    total_mass_rate_kg_s: float
    total_energy_rate_W: float
    boundary_mass_rate_kg_s: float
    boundary_energy_rate_W: float


@dataclass(frozen=True)
class LayeredTankStepResult:
    time_s: float
    state: LayeredTankState
    thermo: LayeredThermoState
    cumulative_boundary_mass_kg: float
    cumulative_boundary_energy_J: float
    cumulative_evaporation_mass_kg: float
    cumulative_liquid_interface_heat_J: float
    cumulative_vapor_interface_heat_J: float
    cumulative_latent_phase_change_energy_J: float
    cumulative_distributed_boiling_mass_kg: float
    cumulative_distributed_boiling_latent_energy_J: float
    total_mass_residual_kg: float
    total_energy_residual_J: float


CellFlows = tuple[Sequence[Iterable[MassEnergyFlow]], Sequence[Iterable[MassEnergyFlow]]]
FlowCallback = Callable[[float, LayeredThermoState], CellFlows]
FluidHeatCallback = Callable[[float, LayeredThermoState], tuple[Sequence[float], Sequence[float]]]
WallHeatCallback = Callable[[float, LayeredThermoState], Sequence[float]]


@dataclass(frozen=True)
class _InterfaceTransfer:
    liquid_state: ThermoState
    vapor_state: ThermoState
    liquid_heat_W: float
    vapor_heat_W: float
    evaporation_rate_kg_s: float
    mass_flux_kg_m2_s: float
    kinetic_residual_kg_m2_s: float
    energy_residual_W: float


class LayeredTank:
    """Vertically ordered Lagrangian liquid and vapor cells in a rigid vessel."""

    def __init__(
        self,
        shape: HorizontalVesselGeometry,
        parameters: LayeredTankParameters,
        properties: HydrogenProperties | None = None,
    ) -> None:
        self.shape = shape
        self.parameters = parameters
        self.properties = properties or HydrogenProperties()
        self._wall_edges_m = np.linspace(0.0, shape.diameter, parameters.wall_node_count + 1)
        wall_cumulative = np.asarray([
            shape.at_height(float(height)).wetted_inner_area_m2
            for height in self._wall_edges_m
        ])
        self._wall_area_edges_m2 = wall_cumulative
        self._wall_areas_m2 = np.diff(wall_cumulative)
        if np.any(self._wall_areas_m2 <= 0.0):
            raise ValueError("Wall nodes must have positive surface areas")
        self._wall_heat_capacities_J_K = (
            parameters.wall_heat_capacity_J_K
            * self._wall_areas_m2 / shape.inner_surface_area_m2
        )
        if parameters.ambient_UA_profile_W_K:
            self._wall_ambient_UA_W_K = np.asarray(
                parameters.ambient_UA_profile_W_K,
                dtype=float,
            )
        else:
            self._wall_ambient_UA_W_K = (
                parameters.ambient_UA_W_K
                * self._wall_areas_m2 / shape.inner_surface_area_m2
            )
        # A normally open instrument/nozzle manifold shares the tank ullage
        # pressure but contributes its own gas compliance and metal thermal
        # inertia.  The connected metal is represented by the uppermost wall
        # node, which is the attachment location for a top-vapor manifold.
        self._wall_heat_capacities_J_K[-1] += (
            parameters.connected_wall_heat_capacity_J_K
        )
        self._wall_ambient_UA_W_K[-1] += parameters.connected_ambient_UA_W_K

    @property
    def total_rigid_volume_m3(self) -> float:
        """Tank volume plus pressure-equal connected vapor dead volume."""

        return self.shape.volume_m3 + self.parameters.connected_vapor_volume_m3

    def _validate_lengths(self, state: LayeredTankState) -> None:
        p = self.parameters
        if len(state.liquid_masses_kg) != p.liquid_cell_count:
            raise ValueError("State liquid cell count does not match parameters")
        if len(state.vapor_masses_kg) != p.vapor_cell_count:
            raise ValueError("State vapor cell count does not match parameters")
        if len(state.wall_temperatures_K) != p.wall_node_count:
            raise ValueError("State wall node count does not match parameters")

    def initialize_saturated(
        self,
        pressure_Pa: float,
        liquid_volume_fraction: float,
        wall_temperature_K: float | None = None,
    ) -> LayeredTankState:
        if not 0.0 < liquid_volume_fraction < 1.0:
            raise ValueError("liquid_volume_fraction must lie between zero and one")
        liquid = self.properties.saturated_liquid(pressure_Pa)
        vapor = self.properties.saturated_vapor(pressure_Pa)
        p = self.parameters
        liquid_cell_volume = self.shape.volume_m3 * liquid_volume_fraction / p.liquid_cell_count
        vapor_cell_volume = self.shape.volume_m3 * (1.0 - liquid_volume_fraction) / p.vapor_cell_count
        temperature = liquid.temperature_K if wall_temperature_K is None else wall_temperature_K
        return LayeredTankState(
            pressure_Pa=float(pressure_Pa),
            liquid_masses_kg=tuple(
                liquid.density_kg_m3 * liquid_cell_volume for _ in range(p.liquid_cell_count)
            ),
            liquid_specific_enthalpies_J_kg=tuple(
                liquid.specific_enthalpy_J_kg for _ in range(p.liquid_cell_count)
            ),
            vapor_masses_kg=tuple(
                vapor.density_kg_m3 * (
                    vapor_cell_volume
                    + (p.connected_vapor_volume_m3 if index == p.vapor_cell_count - 1 else 0.0)
                )
                for index in range(p.vapor_cell_count)
            ),
            vapor_specific_enthalpies_J_kg=tuple(
                vapor.specific_enthalpy_J_kg for _ in range(p.vapor_cell_count)
            ),
            wall_temperatures_K=tuple(float(temperature) for _ in range(p.wall_node_count)),
        )

    def initialize_temperature_profiles(
        self,
        pressure_Pa: float,
        liquid_volume_fraction: float,
        liquid_temperatures_K: Sequence[float] | None = None,
        vapor_temperatures_K: Sequence[float] | None = None,
        wall_temperatures_K: Sequence[float] | float | None = None,
    ) -> LayeredTankState:
        """Create a volume-consistent state with measured or assumed profiles.

        Cells are ordered from bottom to top within each phase.  Each cell in
        a phase receives the same geometric volume, while its mass follows
        from the supplied local temperature and the common tank pressure.
        Omitting a phase profile initializes that phase at saturation.  A
        temperature equal to the saturation temperature is also evaluated
        with the corresponding saturated-liquid or saturated-vapor state so
        the phase remains unambiguous at the coexistence line.

        This initializer carries pre-lock-up thermal history into the dynamic
        model without fitting a heat-transfer coefficient to the subsequent
        pressure trace.
        """

        if not math.isfinite(pressure_Pa) or pressure_Pa <= 0.0:
            raise ValueError("pressure_Pa must be finite and positive")
        if not 0.0 < liquid_volume_fraction < 1.0:
            raise ValueError("liquid_volume_fraction must lie between zero and one")

        p = self.parameters
        saturated_liquid = self.properties.saturated_liquid(pressure_Pa)
        saturated_vapor = self.properties.saturated_vapor(pressure_Pa)
        saturation_temperature_K = saturated_liquid.temperature_K

        def temperatures(
            supplied: Sequence[float] | None,
            count: int,
            default_K: float,
            name: str,
        ) -> tuple[float, ...]:
            values = (default_K,) * count if supplied is None else tuple(float(x) for x in supplied)
            if len(values) != count:
                raise ValueError(f"{name} length must match its configured cell count")
            if any(not math.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError(f"{name} values must be finite and positive")
            return values

        liquid_temperatures = temperatures(
            liquid_temperatures_K,
            p.liquid_cell_count,
            saturation_temperature_K,
            "liquid_temperatures_K",
        )
        vapor_temperatures = temperatures(
            vapor_temperatures_K,
            p.vapor_cell_count,
            saturation_temperature_K,
            "vapor_temperatures_K",
        )

        def phase_state(temperature_K: float, phase: str) -> ThermoState:
            if math.isclose(
                temperature_K,
                saturation_temperature_K,
                rel_tol=0.0,
                abs_tol=1.0e-8,
            ):
                return saturated_liquid if phase == "liquid" else saturated_vapor
            state = self.properties.from_pT(pressure_Pa, temperature_K)
            if state.quality is not None:
                raise ValueError(
                    f"{phase} temperature profile must identify a single-phase state "
                    "or the saturation temperature"
                )
            return state

        liquid_states = tuple(phase_state(value, "liquid") for value in liquid_temperatures)
        vapor_states = tuple(phase_state(value, "vapor") for value in vapor_temperatures)
        if min(state.density_kg_m3 for state in liquid_states) <= max(
            state.density_kg_m3 for state in vapor_states
        ):
            raise ValueError("liquid profile must remain denser than the vapor profile")

        if wall_temperatures_K is None:
            wall_temperatures = (saturation_temperature_K,) * p.wall_node_count
        elif isinstance(wall_temperatures_K, (int, float)) and not isinstance(
            wall_temperatures_K, bool
        ):
            wall_temperatures = (float(wall_temperatures_K),) * p.wall_node_count
        else:
            wall_temperatures = tuple(float(value) for value in wall_temperatures_K)
        if len(wall_temperatures) != p.wall_node_count:
            raise ValueError("wall_temperatures_K length must match wall_node_count")
        if any(not math.isfinite(value) or value <= 0.0 for value in wall_temperatures):
            raise ValueError("wall_temperatures_K values must be finite and positive")

        liquid_cell_volume = (
            self.shape.volume_m3 * liquid_volume_fraction / p.liquid_cell_count
        )
        vapor_cell_volume = (
            self.shape.volume_m3 * (1.0 - liquid_volume_fraction) / p.vapor_cell_count
        )
        return LayeredTankState(
            pressure_Pa=float(pressure_Pa),
            liquid_masses_kg=tuple(
                state.density_kg_m3 * liquid_cell_volume for state in liquid_states
            ),
            liquid_specific_enthalpies_J_kg=tuple(
                state.specific_enthalpy_J_kg for state in liquid_states
            ),
            vapor_masses_kg=tuple(
                state.density_kg_m3 * (
                    vapor_cell_volume
                    + (p.connected_vapor_volume_m3 if index == p.vapor_cell_count - 1 else 0.0)
                )
                for index, state in enumerate(vapor_states)
            ),
            vapor_specific_enthalpies_J_kg=tuple(
                state.specific_enthalpy_J_kg for state in vapor_states
            ),
            wall_temperatures_K=wall_temperatures,
        )

    def _volume_at_pressure(self, pressure_Pa: float, state: LayeredTankState) -> float:
        volume = 0.0
        for mass, enthalpy in zip(state.liquid_masses_kg, state.liquid_specific_enthalpies_J_kg):
            volume += mass / self.properties.from_ph(pressure_Pa, enthalpy).density_kg_m3
        for mass, enthalpy in zip(state.vapor_masses_kg, state.vapor_specific_enthalpies_J_kg):
            volume += mass / self.properties.from_ph(pressure_Pa, enthalpy).density_kg_m3
        return volume

    def _volume_and_slope_at_pressure(
        self, pressure_Pa: float, state: LayeredTankState
    ) -> tuple[float, float]:
        """Return total volume and ``dV/dp`` at fixed cell mass and enthalpy."""

        volume = 0.0
        slope = 0.0
        masses = (*state.liquid_masses_kg, *state.vapor_masses_kg)
        enthalpies = (
            *state.liquid_specific_enthalpies_J_kg,
            *state.vapor_specific_enthalpies_J_kg,
        )
        for mass, enthalpy in zip(masses, enthalpies):
            fluid = self.properties.from_ph(pressure_Pa, enthalpy)
            volume += mass / fluid.density_kg_m3
            dv_dp_h, _ = self.properties.specific_volume_derivatives_ph(fluid)
            slope += mass * dv_dp_h
        return volume, slope

    def project_pressure(self, state: LayeredTankState) -> LayeredTankState:
        """Project integration drift back onto the rigid-volume constraint."""

        self._validate_lengths(state)
        pmin = self.parameters.minimum_pressure_Pa
        pmax = self.parameters.maximum_pressure_Pa
        residual = lambda pressure: (
            self._volume_at_pressure(pressure, state) - self.total_rigid_volume_m3
        )
        hint = min(max(state.pressure_Pa, pmin), pmax)
        try:
            tolerance = max(self.total_rigid_volume_m3, 1.0) * 1e-11
            pressure = hint
            for _ in range(12):
                volume, slope = self._volume_and_slope_at_pressure(pressure, state)
                error = volume - self.total_rigid_volume_m3
                if abs(error) <= tolerance:
                    break
                if not math.isfinite(slope) or abs(slope) < 1e-20:
                    raise ValueError
                candidate = pressure - error / slope
                pressure = min(max(candidate, max(pmin, pressure * 0.5)), min(pmax, pressure * 1.5))
            else:
                raise ValueError
        except (ValueError, RuntimeError, OverflowError):
            samples: list[tuple[float, float]] = []
            for candidate in np.geomspace(pmin, pmax, 100):
                try:
                    value = residual(float(candidate))
                except (ValueError, RuntimeError, OverflowError):
                    continue
                if math.isfinite(value):
                    samples.append((float(candidate), value))
            brackets = [
                (a[0], b[0]) for a, b in zip(samples, samples[1:]) if a[1] * b[1] < 0.0
            ]
            if not brackets:
                raise ValueError("No pressure satisfies the layered rigid-volume constraint")
            low, high = min(
                brackets,
                key=lambda pair: abs(math.log(math.sqrt(pair[0] * pair[1]) / hint)),
            )
            pressure = float(brentq(residual, low, high, xtol=1e-7, rtol=1e-11))
        return LayeredTankState(
            pressure, state.liquid_masses_kg, state.liquid_specific_enthalpies_J_kg,
            state.vapor_masses_kg, state.vapor_specific_enthalpies_J_kg,
            state.wall_temperatures_K,
        )

    def _project_phase_change_manifold(
        self, state: LayeredTankState
    ) -> LayeredTankState:
        """Project both liquid boiling and vapor condensation boundaries.

        The existing saturated-liquid projection is intentionally retained as
        a separate legacy closure.  This variant handles either side of the
        phase boundary for the optional two-sided projection model.  It changes
        only the cells that crossed a saturation branch, then solves one
        receiving-cell enthalpy together with pressure so rigid volume and
        total fluid internal energy remain unchanged.  The receiving-cell
        choice is deterministic and is a numerical manifold projection.  The
        validated differential rate closure remains the one-sided
        saturated-liquid boiling model; vapor-side condensation is kept as a
        projection gate until an independent rate closure is validated.
        """

        if self.parameters.distributed_boiling_model != "phase_change_complementarity":
            return state
        self._validate_lengths(state)
        nl = self.parameters.liquid_cell_count
        nv = self.parameters.vapor_cell_count
        saturated_liquid = self.properties.saturated_liquid(state.pressure_Pa)
        saturated_vapor = self.properties.saturated_vapor(state.pressure_Pa)
        liquid_tolerance = max(
            abs(saturated_vapor.specific_enthalpy_J_kg
                - saturated_liquid.specific_enthalpy_J_kg),
            1.0,
        ) * 1.0e-7
        liquid_active: list[int] = []
        vapor_active: list[int] = []
        for index, enthalpy in enumerate(state.liquid_specific_enthalpies_J_kg):
            if enthalpy <= saturated_liquid.specific_enthalpy_J_kg + liquid_tolerance:
                continue
            candidate = self.properties.from_ph(state.pressure_Pa, enthalpy)
            if candidate.quality is None or candidate.quality >= 1.0:
                raise ValueError(
                    "Phase-complementarity projection crossed the complete "
                    f"liquid-to-vapor interval (cell={index}, quality={candidate.quality}); "
                    "reduce the integration time step"
                )
            liquid_active.append(index)
        for local, enthalpy in enumerate(state.vapor_specific_enthalpies_J_kg):
            if enthalpy >= saturated_vapor.specific_enthalpy_J_kg - liquid_tolerance:
                continue
            candidate = self.properties.from_ph(state.pressure_Pa, enthalpy)
            if candidate.quality is None or candidate.quality <= 0.0:
                raise ValueError(
                    "Phase-complementarity projection crossed the complete "
                    f"vapor-to-liquid interval (cell={local}, quality={candidate.quality}); "
                    "reduce the integration time step"
                )
            vapor_active.append(local)
        if not liquid_active and not vapor_active:
            return state

        masses = np.asarray((*state.liquid_masses_kg, *state.vapor_masses_kg), dtype=float)
        enthalpies = np.asarray((
            *state.liquid_specific_enthalpies_J_kg,
            *state.vapor_specific_enthalpies_J_kg,
        ), dtype=float)
        target_internal_energy = float(
            np.dot(masses, enthalpies) - state.pressure_Pa * self.total_rigid_volume_m3
        )
        fixed_enthalpies = enthalpies.copy()

        # Cells that lie on the saturation boundary must follow the common
        # pressure while the projection is solved.  Leaving the other
        # saturated cells at their old enthalpy would make a pressure increase
        # turn them into an invalid labelled-vapor cell (or a pressure decrease
        # turn a labelled-liquid cell into the two-phase region).  Subcooled
        # liquid and superheated vapor cells retain their independent energy.
        liquid_follow = set(liquid_active)
        vapor_follow = set(vapor_active)
        for index, enthalpy in enumerate(state.liquid_specific_enthalpies_J_kg):
            if abs(enthalpy - saturated_liquid.specific_enthalpy_J_kg) <= liquid_tolerance:
                liquid_follow.add(index)
        for local, enthalpy in enumerate(state.vapor_specific_enthalpies_J_kg):
            if abs(enthalpy - saturated_vapor.specific_enthalpy_J_kg) <= liquid_tolerance:
                vapor_follow.add(local)

        # A pressure projection can make both labelled phases appear to cross
        # their old saturation values.  Choose the receiving phase from the
        # larger enthalpy excursion, then let every other near-saturated cell
        # follow the new pressure.  The selected receiver is removed from that
        # set and carries the one scalar energy correction.
        liquid_excess = sum(
            masses[index] * max(
                state.liquid_specific_enthalpies_J_kg[index]
                - saturated_liquid.specific_enthalpy_J_kg, 0.0
            )
            for index in liquid_active
        )
        vapor_deficit = sum(
            masses[nl + local] * max(
                saturated_vapor.specific_enthalpy_J_kg
                - state.vapor_specific_enthalpies_J_kg[local], 0.0
            )
            for local in vapor_active
        )
        if liquid_excess >= vapor_deficit:
            receiver = nl + (nv - 1)
            vapor_follow.discard(nv - 1)
        else:
            receiver = nl - 1
            liquid_follow.discard(nl - 1)
        follow_global = set(liquid_follow) | {nl + index for index in vapor_follow}
        follow_global.discard(receiver)

        fixed_mask = np.ones(nl + nv, dtype=bool)
        fixed_mask[list(follow_global)] = False
        fixed_mask[receiver] = False
        fixed_enthalpy_total = float(np.dot(masses[fixed_mask], fixed_enthalpies[fixed_mask]))

        def build(pressure: float) -> tuple[float, float]:
            liquid = self.properties.saturated_liquid(pressure)
            vapor = self.properties.saturated_vapor(pressure)
            values = fixed_enthalpies.copy()
            for index in liquid_follow:
                values[index] = liquid.specific_enthalpy_J_kg
            for local in vapor_follow:
                values[nl + local] = vapor.specific_enthalpy_J_kg
            receiver_enthalpy = (
                target_internal_energy
                + pressure * self.total_rigid_volume_m3
                - fixed_enthalpy_total
                - float(np.dot(masses[list(follow_global)], values[list(follow_global)]))
            ) / masses[receiver]
            values[receiver] = receiver_enthalpy
            volume = 0.0
            for mass, enthalpy in zip(masses, values):
                volume += mass / self.properties.from_ph(pressure, float(enthalpy)).density_kg_m3
            return volume - self.total_rigid_volume_m3, receiver_enthalpy

        pmin = self.parameters.minimum_pressure_Pa
        pmax = self.parameters.maximum_pressure_Pa
        hint = min(max(state.pressure_Pa, pmin), pmax)
        tolerance = max(self.total_rigid_volume_m3, 1.0) * 1.0e-11
        pressure = hint
        converged = False
        for _ in range(12):
            residual, _ = build(pressure)
            if abs(residual) <= tolerance:
                converged = True
                break
            increment = max(pressure * 1.0e-5, 0.1)
            lower = max(pmin, pressure - increment)
            upper = min(pmax, pressure + increment)
            slope = (build(upper)[0] - build(lower)[0]) / (upper - lower)
            if not math.isfinite(slope) or abs(slope) < 1.0e-20:
                break
            pressure = min(max(pressure - residual / slope, pmin), pmax)
        if not converged:
            samples: list[tuple[float, float]] = []
            for candidate in np.geomspace(pmin, pmax, 120):
                try:
                    residual = build(float(candidate))[0]
                except (ValueError, RuntimeError, OverflowError):
                    continue
                if math.isfinite(residual):
                    samples.append((float(candidate), residual))
            brackets = [
                (left[0], right[0])
                for left, right in zip(samples, samples[1:])
                if left[1] * right[1] < 0.0
            ]
            if not brackets:
                raise ValueError("No conservative phase-complementarity projection exists")
            low, high = min(
                brackets,
                key=lambda pair: abs(math.log(math.sqrt(pair[0] * pair[1]) / hint)),
            )
            pressure = float(brentq(
                lambda value: build(value)[0], low, high, xtol=1.0e-7, rtol=1.0e-11
            ))
        receiver_enthalpy = build(pressure)[1]
        liquid_enthalpies = list(state.liquid_specific_enthalpies_J_kg)
        for index in liquid_follow:
            liquid_enthalpies[index] = self.properties.saturated_liquid(pressure).specific_enthalpy_J_kg
        vapor_enthalpies = list(state.vapor_specific_enthalpies_J_kg)
        for local in vapor_follow:
            vapor_enthalpies[local] = self.properties.saturated_vapor(pressure).specific_enthalpy_J_kg
        if receiver < nl:
            liquid_enthalpies[receiver] = receiver_enthalpy
        else:
            vapor_enthalpies[receiver - nl] = receiver_enthalpy
        projected = LayeredTankState(
            pressure_Pa=pressure,
            liquid_masses_kg=state.liquid_masses_kg,
            liquid_specific_enthalpies_J_kg=tuple(liquid_enthalpies),
            vapor_masses_kg=state.vapor_masses_kg,
            vapor_specific_enthalpies_J_kg=tuple(vapor_enthalpies),
            wall_temperatures_K=state.wall_temperatures_K,
        )
        final_volume = self._volume_at_pressure(pressure, projected)
        final_internal_energy = self.total_energy_J(projected) - float(
            np.sum(self._wall_heat_capacities_J_K * np.asarray(projected.wall_temperatures_K))
        )
        if not math.isclose(
            final_volume, self.total_rigid_volume_m3,
            rel_tol=0.0, abs_tol=tolerance,
        ):
            raise RuntimeError("Phase-complementarity projection did not close volume")
        energy_tolerance = max(abs(target_internal_energy), 1.0) * 1.0e-11
        if not math.isclose(
            final_internal_energy, target_internal_energy,
            rel_tol=0.0, abs_tol=energy_tolerance,
        ):
            raise RuntimeError("Phase-complementarity projection did not conserve energy")
        return projected

    def _project_distributed_boiling_manifold(
        self, state: LayeredTankState
    ) -> LayeredTankState:
        """Project Runge--Kutta stages onto the saturated-liquid boundary.

        The one-sided differential boiling complementarity is tangent to the
        saturation manifold.  Explicit Runge--Kutta stages nevertheless leave a small
        normal error.  This projection sets only the offending
        liquid cells to saturated liquid and solves for common pressure and
        the bottom-vapor enthalpy so that rigid volume and total fluid internal
        energy are both preserved exactly.  A state that crosses the complete
        two-phase interval is rejected; accepted step size is assessed by
        ordinary time-step convergence rather than an empirical quality cap.
        """

        if self.parameters.distributed_boiling_model == "phase_change_complementarity":
            return self._project_phase_change_manifold(state)
        if self.parameters.distributed_boiling_model == "off":
            return state
        saturated = self.properties.saturated_liquid(state.pressure_Pa)
        active: list[int] = []
        for index, enthalpy in enumerate(state.liquid_specific_enthalpies_J_kg):
            if enthalpy <= saturated.specific_enthalpy_J_kg:
                continue
            candidate = self.properties.from_ph(state.pressure_Pa, enthalpy)
            if candidate.quality is None or candidate.quality >= 1.0:
                raise ValueError(
                    "Distributed-boiling phase projection crossed the complete "
                    f"two-phase interval (cell={index}, quality={candidate.quality}); "
                    "reduce the integration time step"
                )
            active.append(index)
        if not active:
            return state

        nl = self.parameters.liquid_cell_count
        vapor_recipient = 0
        masses = np.asarray((*state.liquid_masses_kg, *state.vapor_masses_kg))
        enthalpies = np.asarray((
            *state.liquid_specific_enthalpies_J_kg,
            *state.vapor_specific_enthalpies_J_kg,
        ))
        recipient_global = nl + vapor_recipient
        target_internal_energy = float(
            np.dot(masses, enthalpies) - state.pressure_Pa * self.total_rigid_volume_m3
        )
        fixed_mask = np.ones(len(masses), dtype=bool)
        fixed_mask[active] = False
        fixed_mask[recipient_global] = False
        fixed_enthalpy_total = float(np.dot(
            masses[fixed_mask], enthalpies[fixed_mask]
        ))

        def build(pressure: float) -> tuple[float, float, float]:
            liquid = self.properties.saturated_liquid(pressure)
            active_enthalpy_total = float(
                np.sum(masses[active]) * liquid.specific_enthalpy_J_kg
            )
            recipient_enthalpy = (
                target_internal_energy
                + pressure * self.total_rigid_volume_m3
                - fixed_enthalpy_total
                - active_enthalpy_total
            ) / masses[recipient_global]
            volume = 0.0
            for index, (mass, enthalpy) in enumerate(zip(masses, enthalpies)):
                if index in active:
                    density = liquid.density_kg_m3
                elif index == recipient_global:
                    density = self.properties.from_ph(
                        pressure, recipient_enthalpy
                    ).density_kg_m3
                else:
                    density = self.properties.from_ph(
                        pressure, enthalpy
                    ).density_kg_m3
                volume += mass / density
            return (
                volume - self.total_rigid_volume_m3,
                recipient_enthalpy,
                liquid.specific_enthalpy_J_kg,
            )

        pmin = self.parameters.minimum_pressure_Pa
        pmax = self.parameters.maximum_pressure_Pa
        pressure = min(max(state.pressure_Pa, pmin), pmax)
        tolerance = max(self.total_rigid_volume_m3, 1.0) * 1.0e-11
        converged = False
        for _ in range(12):
            residual, _, _ = build(pressure)
            if abs(residual) <= tolerance:
                converged = True
                break
            increment = max(pressure * 1.0e-5, 0.1)
            lower = max(pmin, pressure - increment)
            upper = min(pmax, pressure + increment)
            slope = (build(upper)[0] - build(lower)[0]) / (upper - lower)
            if not math.isfinite(slope) or abs(slope) < 1.0e-20:
                break
            pressure = min(max(pressure - residual / slope, pmin), pmax)
        if not converged:
            samples: list[tuple[float, float]] = []
            for candidate in np.geomspace(pmin, pmax, 120):
                try:
                    residual = build(float(candidate))[0]
                except (ValueError, RuntimeError, OverflowError):
                    continue
                if math.isfinite(residual):
                    samples.append((float(candidate), residual))
            brackets = [
                (left[0], right[0])
                for left, right in zip(samples, samples[1:])
                if left[1] * right[1] < 0.0
            ]
            if not brackets:
                raise ValueError(
                    "No conservative distributed-boiling phase projection exists"
                )
            low, high = min(
                brackets,
                key=lambda pair: abs(
                    math.log(math.sqrt(pair[0] * pair[1]) / state.pressure_Pa)
                ),
            )
            pressure = float(brentq(
                lambda value: build(value)[0], low, high, xtol=1e-7, rtol=1e-11
            ))
        _, recipient_enthalpy, saturated_liquid_enthalpy = build(pressure)
        liquid_enthalpies = list(state.liquid_specific_enthalpies_J_kg)
        for index in active:
            liquid_enthalpies[index] = saturated_liquid_enthalpy
        vapor_enthalpies = list(state.vapor_specific_enthalpies_J_kg)
        vapor_enthalpies[vapor_recipient] = recipient_enthalpy
        projected = LayeredTankState(
            pressure_Pa=pressure,
            liquid_masses_kg=state.liquid_masses_kg,
            liquid_specific_enthalpies_J_kg=tuple(liquid_enthalpies),
            vapor_masses_kg=state.vapor_masses_kg,
            vapor_specific_enthalpies_J_kg=tuple(vapor_enthalpies),
            wall_temperatures_K=state.wall_temperatures_K,
        )
        final_volume = self._volume_at_pressure(pressure, projected)
        final_internal_energy = float(
            np.dot(masses, np.asarray((
                *projected.liquid_specific_enthalpies_J_kg,
                *projected.vapor_specific_enthalpies_J_kg,
            ))) - pressure * self.total_rigid_volume_m3
        )
        if not math.isclose(
            final_volume, self.total_rigid_volume_m3,
            rel_tol=0.0, abs_tol=tolerance
        ):
            raise RuntimeError("Distributed-boiling projection did not close volume")
        energy_tolerance = max(abs(target_internal_energy), 1.0) * 1.0e-11
        if not math.isclose(
            final_internal_energy, target_internal_energy,
            rel_tol=0.0, abs_tol=energy_tolerance,
        ):
            raise RuntimeError("Distributed-boiling projection did not conserve energy")
        return projected

    def thermo(self, state: LayeredTankState) -> LayeredThermoState:
        self._validate_lengths(state)
        liquid = tuple(
            self.properties.from_ph(state.pressure_Pa, enthalpy)
            for enthalpy in state.liquid_specific_enthalpies_J_kg
        )
        vapor = tuple(
            self.properties.from_ph(state.pressure_Pa, enthalpy)
            for enthalpy in state.vapor_specific_enthalpies_J_kg
        )
        quality_tolerance = 1.0e-7
        if any(
            item.quality is not None and item.quality > quality_tolerance
            for item in liquid
        ):
            raise ValueError(
                "A labeled liquid cell entered the two-phase or vapor region; "
                "phase transfer must occur through the interface model"
            )
        if any(
            item.quality is not None and item.quality < 1.0 - quality_tolerance
            for item in vapor
        ):
            raise ValueError(
                "A labeled vapor cell entered the liquid or two-phase region; "
                "phase transfer must occur through the interface model"
            )
        if min(item.density_kg_m3 for item in liquid) <= max(item.density_kg_m3 for item in vapor):
            raise ValueError("Every liquid cell must be denser than every vapor cell")
        masses = (*state.liquid_masses_kg, *state.vapor_masses_kg)
        thermo = (*liquid, *vapor)
        volumes = np.asarray([
            mass / item.density_kg_m3 for mass, item in zip(masses, thermo)
        ])
        tank_volumes = volumes.copy()
        tank_volumes[-1] -= self.parameters.connected_vapor_volume_m3
        tolerance = max(self.total_rigid_volume_m3, 1.0) * 1e-8
        if tank_volumes[-1] <= 0.0:
            raise ValueError(
                "Top vapor inventory is too small for the connected vapor dead volume"
            )
        cumulative = np.r_[0.0, np.cumsum(tank_volumes)]
        if cumulative[-1] < -tolerance or cumulative[-1] > self.shape.volume_m3 + tolerance:
            raise ValueError("Layered tank cell volumes lie outside the vessel")
        geometry_volume = np.clip(cumulative, 0.0, self.shape.volume_m3)
        edges = np.asarray([
            self.shape.height_from_liquid_volume(float(value)) for value in geometry_volume
        ])
        centers = np.asarray([
            self.shape.height_from_liquid_volume(float(value))
            for value in (cumulative[:-1] + cumulative[1:]) / 2.0
        ])
        fluid_area_edges = np.asarray([
            self.shape.at_height(float(height)).wetted_inner_area_m2 for height in edges
        ])
        overlap = np.maximum(
            0.0,
            np.minimum(fluid_area_edges[1:, None], self._wall_area_edges_m2[None, 1:])
            - np.maximum(fluid_area_edges[:-1, None], self._wall_area_edges_m2[None, :-1]),
        )
        interface_index = self.parameters.liquid_cell_count
        interface_geometry = self.shape.at_height(float(edges[interface_index]))
        return LayeredThermoState(
            pressure_Pa=state.pressure_Pa,
            liquid=liquid,
            vapor=vapor,
            cell_volumes_m3=tuple(float(value) for value in volumes),
            cell_edge_heights_m=tuple(float(value) for value in edges),
            cell_center_heights_m=tuple(float(value) for value in centers),
            wall_overlap_areas_m2=tuple(tuple(float(value) for value in row) for row in overlap),
            liquid_height_m=float(edges[interface_index]),
            interface_area_m2=interface_geometry.interface_area_m2,
            volume_residual_m3=float(
                np.sum(volumes) - self.total_rigid_volume_m3
            ),
            tank_cell_volumes_m3=tuple(float(value) for value in tank_volumes),
            wall_temperatures_K=tuple(float(value) for value in state.wall_temperatures_K),
        )

    @property
    def wall_area_fractions(self) -> tuple[float, ...]:
        """Return the fixed geometric area fraction of each wall node."""

        total = float(np.sum(self._wall_areas_m2))
        if total <= 0.0:
            raise RuntimeError("Wall area must be positive")
        return tuple(float(value / total) for value in self._wall_areas_m2)

    @staticmethod
    def _validate_vector(values: Sequence[float], length: int, name: str) -> np.ndarray:
        result = np.asarray(tuple(values), dtype=float)
        if result.shape != (length,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain {length} finite values")
        return result

    @staticmethod
    def _flow_terms(
        flows: Sequence[Iterable[MassEnergyFlow]],
        cells: tuple[ThermoState, ...],
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        if len(flows) != len(cells):
            raise ValueError("Flow groups must match the number of cells")
        mass = np.zeros(len(cells))
        enthalpy_residual = np.zeros(len(cells))
        boundary_mass = 0.0
        boundary_energy = 0.0
        for index, (group, owner) in enumerate(zip(flows, cells)):
            for flow in group:
                if not math.isfinite(flow.mass_flow_kg_s) or not math.isfinite(flow.specific_enthalpy_J_kg):
                    raise ValueError("Flows must contain finite values")
                stream_h = (
                    flow.specific_enthalpy_J_kg
                    if flow.mass_flow_kg_s >= 0.0
                    else owner.specific_enthalpy_J_kg
                )
                mass[index] += flow.mass_flow_kg_s
                enthalpy_residual[index] += flow.mass_flow_kg_s * (
                    stream_h - owner.specific_enthalpy_J_kg
                )
                boundary_mass += flow.mass_flow_kg_s
                boundary_energy += flow.mass_flow_kg_s * stream_h
        return mass, enthalpy_residual, boundary_mass, boundary_energy

    @staticmethod
    def _harmonic_mean(a: float, b: float) -> float:
        return 2.0 * a * b / (a + b)

    def _interface_heat_W(
        self,
        bulk: ThermoState,
        saturated: ThermoState,
        area_m2: float,
        distance_m: float,
        conduction_multiplier: float,
        convection_is_unstable: bool,
        gas_side: bool = False,
    ) -> float:
        """Heat from a bulk cell to the saturated interface.

        The optional natural-convection closure is Daigle et al. (2013),
        Eqs. (20)--(22) and (33)--(34): use the larger of molecular
        conduction and horizontal-surface convection only for the unstable
        orientation.  The published correlation is not extrapolated above
        ``Ra=1e11``.
        """

        if (
            gas_side
            and self.parameters.interface_heat_transfer_model == "kinetic_gas_film"
        ):
            kinetic = kinetic_gas_film_heat_transfer_coefficient(
                pressure_Pa=bulk.pressure_Pa,
                temperature_K=bulk.temperature_K,
                isobaric_heat_capacity_J_kgK=(
                    self.properties.isobaric_heat_capacity_J_kgK(bulk)
                ),
                specific_gas_constant_J_kgK=(
                    self.properties.specific_gas_constant_J_kgK()
                ),
                energy_accommodation_coefficient=(
                    self.parameters.gas_interface_energy_accommodation_coefficient
                ),
            )
            return kinetic.heat_transfer_coefficient_W_m2K * area_m2 * (
                bulk.temperature_K - saturated.temperature_K
            )

        conductivity = self._harmonic_mean(
            self.properties.thermal_conductivity_W_mK(bulk),
            self.properties.thermal_conductivity_W_mK(saturated),
        )
        # Positive heat is from the bulk cell toward the saturated interface.
        # The correlation may act with either sign; the caller determines
        # whether the horizontal orientation is buoyantly unstable.
        delta_temperature = bulk.temperature_K - saturated.temperature_K
        conductance = conduction_multiplier * conductivity * area_m2 / distance_m
        conduction = conductance * delta_temperature
        if (
            self.parameters.interface_heat_transfer_model == "conduction"
            or not convection_is_unstable
            or delta_temperature == 0.0
            or area_m2 <= 0.0
        ):
            return conduction
        characteristic_length = math.sqrt(area_m2 / math.pi)
        correlation = daigle_horizontal_surface(
            delta_temperature_K=delta_temperature,
            characteristic_length_m=characteristic_length,
            density_kg_m3=bulk.density_kg_m3,
            viscosity_Pa_s=self.properties.viscosity_Pa_s(bulk),
            thermal_conductivity_W_mK=conductivity,
            isobaric_heat_capacity_J_kgK=(
                self.properties.isobaric_heat_capacity_J_kgK(bulk)
            ),
            expansion_coefficient_1_K=(
                self.properties.isobaric_expansion_coefficient_1_K(bulk)
            ),
            gravity_m_s2=self.parameters.gravity_m_s2,
        )
        if correlation.heat_transfer_coefficient_W_m2K is None:
            return conduction
        convection = (
            correlation.heat_transfer_coefficient_W_m2K
            * area_m2 * delta_temperature
        )
        return convection if abs(convection) > abs(conduction) else conduction

    def _interface_transfer(
        self,
        *,
        pressure_Pa: float,
        liquid_bulk: ThermoState,
        vapor_bulk: ThermoState,
        area_m2: float,
        liquid_distance_m: float,
        vapor_distance_m: float,
    ) -> _InterfaceTransfer:
        """Solve the selected massless-interface heat and phase-change closure."""

        p = self.parameters

        def at_temperature(interface_temperature_K: float) -> _InterfaceTransfer:
            interface_pressure = self.properties.saturation_pressure_Pa(
                interface_temperature_K
            )
            saturated_liquid = self.properties.saturated_liquid(interface_pressure)
            saturated_vapor = self.properties.saturated_vapor(interface_pressure)
            liquid_heat = self._interface_heat_W(
                liquid_bulk,
                saturated_liquid,
                area_m2,
                liquid_distance_m,
                p.liquid_interface_conduction_multiplier,
                convection_is_unstable=(
                    liquid_bulk.temperature_K > interface_temperature_K
                ),
            )
            vapor_heat = self._interface_heat_W(
                vapor_bulk,
                saturated_vapor,
                area_m2,
                vapor_distance_m,
                p.vapor_interface_conduction_multiplier,
                convection_is_unstable=(
                    interface_temperature_K > vapor_bulk.temperature_K
                ),
                gas_side=True,
            )
            latent_heat = (
                saturated_vapor.specific_enthalpy_J_kg
                - saturated_liquid.specific_enthalpy_J_kg
            )
            kinetic = schrage_mass_flux(
                interface_pressure_Pa=interface_pressure,
                interface_temperature_K=interface_temperature_K,
                vapor_pressure_Pa=pressure_Pa,
                vapor_temperature_K=vapor_bulk.temperature_K,
                specific_gas_constant_J_kgK=(
                    self.properties.specific_gas_constant_J_kgK()
                ),
                accommodation_coefficient=p.schrage_accommodation_coefficient,
            )
            evaporation = kinetic.mass_flux_kg_m2_s * area_m2
            residual = liquid_heat + vapor_heat - evaporation * latent_heat
            return _InterfaceTransfer(
                liquid_state=saturated_liquid,
                vapor_state=saturated_vapor,
                liquid_heat_W=liquid_heat,
                vapor_heat_W=vapor_heat,
                evaporation_rate_kg_s=evaporation,
                mass_flux_kg_m2_s=kinetic.mass_flux_kg_m2_s,
                kinetic_residual_kg_m2_s=0.0,
                energy_residual_W=residual,
            )

        equilibrium_liquid = self.properties.saturated_liquid(pressure_Pa)
        equilibrium_vapor = self.properties.saturated_vapor(pressure_Pa)
        equilibrium_temperature = equilibrium_liquid.temperature_K
        if p.interface_phase_change_model == "equilibrium_energy_jump":
            liquid_heat = self._interface_heat_W(
                liquid_bulk,
                equilibrium_liquid,
                area_m2,
                liquid_distance_m,
                p.liquid_interface_conduction_multiplier,
                convection_is_unstable=(
                    liquid_bulk.temperature_K > equilibrium_temperature
                ),
            )
            vapor_heat = self._interface_heat_W(
                vapor_bulk,
                equilibrium_vapor,
                area_m2,
                vapor_distance_m,
                p.vapor_interface_conduction_multiplier,
                convection_is_unstable=(
                    equilibrium_temperature > vapor_bulk.temperature_K
                ),
                gas_side=True,
            )
            latent_heat = (
                equilibrium_vapor.specific_enthalpy_J_kg
                - equilibrium_liquid.specific_enthalpy_J_kg
            )
            evaporation = (liquid_heat + vapor_heat) / latent_heat
            return _InterfaceTransfer(
                liquid_state=equilibrium_liquid,
                vapor_state=equilibrium_vapor,
                liquid_heat_W=liquid_heat,
                vapor_heat_W=vapor_heat,
                evaporation_rate_kg_s=evaporation,
                mass_flux_kg_m2_s=evaporation / area_m2,
                kinetic_residual_kg_m2_s=0.0,
                energy_residual_W=0.0,
            )

        if (
            abs(liquid_bulk.temperature_K - equilibrium_temperature) <= 1.0e-10
            and abs(vapor_bulk.temperature_K - equilibrium_temperature) <= 1.0e-10
        ):
            return _InterfaceTransfer(
                liquid_state=equilibrium_liquid,
                vapor_state=equilibrium_vapor,
                liquid_heat_W=0.0,
                vapor_heat_W=0.0,
                evaporation_rate_kg_s=0.0,
                mass_flux_kg_m2_s=0.0,
                kinetic_residual_kg_m2_s=0.0,
                energy_residual_W=0.0,
            )

        def enforce_energy_balance(result: _InterfaceTransfer) -> _InterfaceTransfer:
            latent_heat = (
                result.vapor_state.specific_enthalpy_J_kg
                - result.liquid_state.specific_enthalpy_J_kg
            )
            evaporation = (
                result.liquid_heat_W + result.vapor_heat_W
            ) / latent_heat
            return _InterfaceTransfer(
                liquid_state=result.liquid_state,
                vapor_state=result.vapor_state,
                liquid_heat_W=result.liquid_heat_W,
                vapor_heat_W=result.vapor_heat_W,
                evaporation_rate_kg_s=evaporation,
                mass_flux_kg_m2_s=evaporation / area_m2,
                kinetic_residual_kg_m2_s=(
                    evaporation / area_m2 - result.mass_flux_kg_m2_s
                ),
                energy_residual_W=0.0,
            )

        # The kinetic interface temperature is the root of the interface
        # energy balance.  Start close to the common-pressure saturation
        # temperature because the Schrage pressure term makes the root stiff,
        # then expand the bracket without leaving the two-phase envelope.
        temperature_span = (
            self.properties.critical_temperature_K()
            - self.properties.triple_temperature_K()
        )
        margin = max(1.0e-6, temperature_span * 1.0e-7)
        minimum_temperature = self.properties.triple_temperature_K() + margin
        maximum_temperature = self.properties.critical_temperature_K() - margin
        center = min(max(equilibrium_temperature, minimum_temperature), maximum_temperature)
        center_result = at_temperature(center)
        scale_W = max(
            1.0,
            abs(center_result.liquid_heat_W) + abs(center_result.vapor_heat_W),
        )
        if abs(center_result.energy_residual_W) <= 1.0e-10 * scale_W:
            return enforce_energy_balance(center_result)

        step = 0.01
        bracket: tuple[float, float] | None = None
        for _ in range(20):
            lower = max(minimum_temperature, center - step)
            upper = min(maximum_temperature, center + step)
            lower_residual = at_temperature(lower).energy_residual_W
            upper_residual = at_temperature(upper).energy_residual_W
            if lower_residual == 0.0:
                return enforce_energy_balance(at_temperature(lower))
            if upper_residual == 0.0:
                return enforce_energy_balance(at_temperature(upper))
            if lower_residual * upper_residual < 0.0:
                bracket = (lower, upper)
                break
            if lower == minimum_temperature and upper == maximum_temperature:
                break
            step *= 2.0
        if bracket is None:
            raise ValueError(
                "Schrage interface energy balance has no root inside the "
                "hydrogen saturation-temperature range"
            )
        interface_temperature = brentq(
            lambda temperature: at_temperature(temperature).energy_residual_W,
            *bracket,
            xtol=1.0e-10,
            rtol=1.0e-12,
        )
        return enforce_energy_balance(at_temperature(float(interface_temperature)))

    def _phase_wall_circulation(
        self,
        *,
        cells: tuple[ThermoState, ...],
        state_enthalpies_J_kg: Sequence[float],
        global_start: int,
        phase_bottom_m: float,
        phase_top_m: float,
        centers_m: np.ndarray,
        edges_m: np.ndarray,
        overlap_m2: np.ndarray,
        wall_temperatures_K: np.ndarray,
        phase_wall_heat_W: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return conservative bulk exchange driven by vertical wall layers.

        Daigle et al. (2013) retain separate wall-boundary and central-bulk
        control volumes.  This reduced form assumes negligible storage in the
        boundary layer.  At every internal horizontal boundary, the wall-leg
        flow is paired with an equal core return flow.  Cell masses therefore
        do not change, while ``m_dot * (h_upper - h_lower)`` is added to the
        lower cell and subtracted from the upper cell.

        A hot lower-cell wall layer contributes upward flow; a cold upper-cell
        wall layer contributes downward flow.  If both occur, their exchange
        magnitudes add.  The reported signed wall rate is positive upward.
        No correlation is extrapolated outside the range enforced by
        :func:`daigle_vertical_wall_boundary_layer`.
        """

        count = len(cells)
        heat = np.zeros(count)
        signed_rates = np.zeros(max(0, count - 1))
        lower_cell_heat = np.zeros(max(0, count - 1))
        if (
            self.parameters.vertical_wall_circulation_model == "off"
            or count < 2
        ):
            return heat, signed_rates, lower_cell_heat

        enthalpies = np.asarray(tuple(state_enthalpies_J_kg), dtype=float)
        for local in range(count - 1):
            lower_global = global_start + local
            upper_global = lower_global + 1
            upward = 0.0
            downward = 0.0

            for direction, donor_local, donor_global, origin_m in (
                (1.0, local, lower_global, phase_bottom_m),
                (-1.0, local + 1, upper_global, phase_top_m),
            ):
                wall_areas = overlap_m2[donor_global]
                wall_area = float(np.sum(wall_areas))
                cell_height = float(edges_m[donor_global + 1] - edges_m[donor_global])
                if wall_area <= 0.0 or cell_height <= 0.0:
                    continue
                characteristic_height = abs(float(centers_m[donor_global]) - origin_m)
                if characteristic_height <= 0.0:
                    continue
                effective_perimeter = wall_area / cell_height
                fluid = cells[donor_local]
                common = dict(
                    characteristic_height_m=characteristic_height,
                    exchange_area_m2=wall_area,
                    wall_perimeter_m=effective_perimeter,
                    density_kg_m3=fluid.density_kg_m3,
                    viscosity_Pa_s=self.properties.viscosity_Pa_s(fluid),
                    thermal_conductivity_W_mK=(
                        self.properties.thermal_conductivity_W_mK(fluid)
                    ),
                    isobaric_heat_capacity_J_kgK=(
                        self.properties.isobaric_heat_capacity_J_kgK(fluid)
                    ),
                    expansion_coefficient_1_K=(
                        self.properties.isobaric_expansion_coefficient_1_K(fluid)
                    ),
                    gravity_m_s2=self.parameters.gravity_m_s2,
                )
                if self.parameters.vertical_wall_circulation_model == (
                    "daigle_2013_uniform_heat_flux_reduced"
                ):
                    wall_heat_flux = phase_wall_heat_W[donor_local] / wall_area
                    if direction * wall_heat_flux <= 0.0:
                        continue
                    result = daigle_vertical_wall_uniform_heat_flux(
                        wall_heat_flux_W_m2=wall_heat_flux,
                        **common,
                    )
                else:
                    wall_temperature = float(
                        np.dot(wall_areas, wall_temperatures_K) / wall_area
                    )
                    delta_temperature = (
                        wall_temperature - cells[donor_local].temperature_K
                    )
                    if direction * delta_temperature <= 0.0:
                        continue
                    result = daigle_vertical_wall_boundary_layer(
                        wall_minus_bulk_temperature_K=delta_temperature,
                        **common,
                    )
                if result.mass_flow_kg_s is None:
                    continue
                magnitude = abs(result.mass_flow_kg_s)
                if direction > 0.0:
                    upward += magnitude
                else:
                    downward += magnitude

            exchange_rate = upward + downward
            signed_rates[local] = upward - downward
            lower_gain = exchange_rate * (
                enthalpies[local + 1] - enthalpies[local]
            )
            lower_cell_heat[local] = lower_gain
            heat[local] += lower_gain
            heat[local + 1] -= lower_gain
        return heat, signed_rates, lower_cell_heat

    def derivative(
        self,
        state: LayeredTankState,
        liquid_flows: Sequence[Iterable[MassEnergyFlow]],
        vapor_flows: Sequence[Iterable[MassEnergyFlow]],
        ambient_temperature_K: float,
        direct_liquid_heat_W: Sequence[float] | None = None,
        direct_vapor_heat_W: Sequence[float] | None = None,
        direct_wall_heat_W: Sequence[float] | None = None,
    ) -> LayeredTankDerivative:
        if not math.isfinite(ambient_temperature_K) or ambient_temperature_K <= 0.0:
            raise ValueError("ambient_temperature_K must be finite and positive")
        thermo = self.thermo(state)
        p = self.parameters
        nl = p.liquid_cell_count
        nv = p.vapor_cell_count
        nc = nl + nv
        q_fluid = np.zeros(nc)
        q_wall_to_fluid = np.zeros(p.wall_node_count)
        direct_liquid = self._validate_vector(
            direct_liquid_heat_W if direct_liquid_heat_W is not None else np.zeros(nl),
            nl, "direct_liquid_heat_W",
        )
        direct_vapor = self._validate_vector(
            direct_vapor_heat_W if direct_vapor_heat_W is not None else np.zeros(nv),
            nv, "direct_vapor_heat_W",
        )
        q_fluid[:nl] += direct_liquid
        q_fluid[nl:] += direct_vapor
        direct_wall = self._validate_vector(
            direct_wall_heat_W if direct_wall_heat_W is not None else np.zeros(p.wall_node_count),
            p.wall_node_count, "direct_wall_heat_W",
        )
        cells = (*thermo.liquid, *thermo.vapor)
        overlap = np.asarray(thermo.wall_overlap_areas_m2)
        q_cells_from_wall = np.zeros(nc)
        for cell_index, cell in enumerate(cells):
            coefficient = p.wall_liquid_U_W_m2K if cell_index < nl else p.wall_vapor_U_W_m2K
            phase_height = (
                thermo.liquid_height_m
                if cell_index < nl
                else self.shape.diameter - thermo.liquid_height_m
            )
            for wall_index, area in enumerate(overlap[cell_index]):
                if area <= 0.0:
                    continue
                delta_temperature = (
                    state.wall_temperatures_K[wall_index] - cell.temperature_K
                )
                if p.wall_heat_transfer_model == "constant":
                    heat = coefficient * area * delta_temperature
                elif p.wall_heat_transfer_model == "churchill_chu_vertical_plate":
                    heat = churchill_chu_vertical_plate(
                        wall_minus_bulk_temperature_K=delta_temperature,
                        characteristic_height_m=phase_height,
                        exchange_area_m2=float(area),
                        density_kg_m3=cell.density_kg_m3,
                        viscosity_Pa_s=self.properties.viscosity_Pa_s(cell),
                        thermal_conductivity_W_mK=(
                            self.properties.thermal_conductivity_W_mK(cell)
                        ),
                        isobaric_heat_capacity_J_kgK=(
                            self.properties.isobaric_heat_capacity_J_kgK(cell)
                        ),
                        expansion_coefficient_1_K=(
                            self.properties.isobaric_expansion_coefficient_1_K(cell)
                        ),
                        gravity_m_s2=p.gravity_m_s2,
                    ).heat_flow_W
                else:
                    heat = yang_west_2015_cryogenic_tank(
                        phase="liquid" if cell_index < nl else "vapor",
                        wall_minus_bulk_temperature_K=delta_temperature,
                        characteristic_height_m=phase_height,
                        exchange_area_m2=float(area),
                        density_kg_m3=cell.density_kg_m3,
                        viscosity_Pa_s=self.properties.viscosity_Pa_s(cell),
                        thermal_conductivity_W_mK=(
                            self.properties.thermal_conductivity_W_mK(cell)
                        ),
                        isobaric_heat_capacity_J_kgK=(
                            self.properties.isobaric_heat_capacity_J_kgK(cell)
                        ),
                        expansion_coefficient_1_K=(
                            self.properties.isobaric_expansion_coefficient_1_K(cell)
                        ),
                        gravity_m_s2=p.gravity_m_s2,
                    ).heat_flow_W
                heat *= (
                    p.wall_liquid_heat_transfer_multiplier
                    if cell_index < nl
                    else p.wall_vapor_heat_transfer_multiplier
                )
                q_fluid[cell_index] += heat
                q_cells_from_wall[cell_index] += heat
                q_wall_to_fluid[wall_index] += heat

        # Preserve the phase-specific wall heat before adding any separately
        # connected vapor boundary.  The uniform-heat-flux Daigle circulation
        # closure needs this actual wall-to-fluid heat, rather than only direct
        # external heat inputs, to remain active in a physical tank run.
        phase_wall_heat_W = q_cells_from_wall.copy()
        phase_wall_heat_W[:nl] += direct_liquid
        phase_wall_heat_W[nl:] += direct_vapor

        if p.connected_wall_vapor_UA_W_K > 0.0:
            connected_heat = p.connected_wall_vapor_UA_W_K * (
                state.wall_temperatures_K[-1] - thermo.vapor[-1].temperature_K
            )
            q_fluid[-1] += connected_heat
            q_cells_from_wall[-1] += connected_heat
            q_wall_to_fluid[-1] += connected_heat

        centers = np.asarray(thermo.cell_center_heights_m)
        edges = np.asarray(thermo.cell_edge_heights_m)

        def axial(start: int, count: int, multiplier: float) -> None:
            for local in range(count - 1):
                lower = start + local
                upper = lower + 1
                boundary = self.shape.at_height(float(edges[upper]))
                distance = centers[upper] - centers[lower]
                k_lower = self.properties.thermal_conductivity_W_mK(cells[lower])
                k_upper = self.properties.thermal_conductivity_W_mK(cells[upper])
                conductance = (
                    multiplier * self._harmonic_mean(k_lower, k_upper)
                    * boundary.interface_area_m2 / distance
                )
                heat = conductance * (cells[upper].temperature_K - cells[lower].temperature_K)
                q_fluid[lower] += heat
                q_fluid[upper] -= heat

        axial(0, nl, p.liquid_axial_conduction_multiplier)
        axial(nl, nv, p.vapor_axial_conduction_multiplier)

        wall_temperatures = np.asarray(state.wall_temperatures_K)
        (
            liquid_circulation_heat,
            liquid_circulation_rates,
            liquid_circulation_lower_heat,
        ) = (
            self._phase_wall_circulation(
                cells=thermo.liquid,
                state_enthalpies_J_kg=state.liquid_specific_enthalpies_J_kg,
                global_start=0,
                phase_bottom_m=0.0,
                phase_top_m=thermo.liquid_height_m,
                centers_m=centers,
                edges_m=edges,
                overlap_m2=overlap,
                wall_temperatures_K=wall_temperatures,
                phase_wall_heat_W=phase_wall_heat_W[:nl],
            )
        )
        (
            vapor_circulation_heat,
            vapor_circulation_rates,
            vapor_circulation_lower_heat,
        ) = (
            self._phase_wall_circulation(
                cells=thermo.vapor,
                state_enthalpies_J_kg=state.vapor_specific_enthalpies_J_kg,
                global_start=nl,
                phase_bottom_m=thermo.liquid_height_m,
                phase_top_m=self.shape.diameter,
                centers_m=centers,
                edges_m=edges,
                overlap_m2=overlap,
                wall_temperatures_K=wall_temperatures,
                phase_wall_heat_W=phase_wall_heat_W[nl:],
            )
        )
        q_fluid[:nl] += liquid_circulation_heat
        q_fluid[nl:] += vapor_circulation_heat

        top_liquid = thermo.liquid[-1]
        bottom_vapor = thermo.vapor[0]
        interface_height = thermo.liquid_height_m
        liquid_distance = max(interface_height - centers[nl - 1], 1e-9)
        vapor_distance = max(centers[nl] - interface_height, 1e-9)
        interface = self._interface_transfer(
            pressure_Pa=state.pressure_Pa,
            liquid_bulk=top_liquid,
            vapor_bulk=bottom_vapor,
            area_m2=thermo.interface_area_m2,
            liquid_distance_m=liquid_distance,
            vapor_distance_m=vapor_distance,
        )
        saturated_liquid = interface.liquid_state
        saturated_vapor = interface.vapor_state
        q_liquid_interface = interface.liquid_heat_W
        q_vapor_interface = interface.vapor_heat_W
        latent_heat = (
            saturated_vapor.specific_enthalpy_J_kg
            - saturated_liquid.specific_enthalpy_J_kg
        )
        evaporation = interface.evaporation_rate_kg_s
        q_fluid[nl - 1] -= q_liquid_interface
        q_fluid[nl] -= q_vapor_interface

        liquid_mass, liquid_r, boundary_lm, boundary_le = self._flow_terms(
            liquid_flows, thermo.liquid
        )
        vapor_mass, vapor_r, boundary_vm, boundary_ve = self._flow_terms(
            vapor_flows, thermo.vapor
        )
        liquid_mass[-1] -= evaporation
        vapor_mass[0] += evaporation
        liquid_r[-1] += -evaporation * (
            saturated_liquid.specific_enthalpy_J_kg - top_liquid.specific_enthalpy_J_kg
        )
        vapor_r[0] += evaporation * (
            saturated_vapor.specific_enthalpy_J_kg - bottom_vapor.specific_enthalpy_J_kg
        )

        masses = np.asarray((*state.liquid_masses_kg, *state.vapor_masses_kg))
        mass_rates = np.r_[liquid_mass, vapor_mass]
        residual_terms = np.r_[liquid_r, vapor_r]
        volumes = np.asarray(thermo.cell_volumes_m3)
        specific_volumes = volumes / masses
        vp = np.empty(nc)
        vh = np.empty(nc)
        for index, cell in enumerate(cells):
            vp[index], vh[index] = self.properties.specific_volume_derivatives_ph(cell)
        source = q_fluid + residual_terms
        denominator = float(np.sum(masses * vp + vh * volumes))
        if not math.isfinite(denominator) or abs(denominator) < 1e-20:
            raise ValueError("Layered pressure equation is singular")

        distributed_boiling = np.zeros(nl)
        distributed_boiling_latent_power = 0.0
        distributed_condensation_rate = 0.0
        if p.distributed_boiling_model == "saturated_liquid_complementarity":
            saturated_liquid_at_pressure = self.properties.saturated_liquid(
                state.pressure_Pa
            )
            saturated_vapor_at_pressure = self.properties.saturated_vapor(
                state.pressure_Pa
            )
            latent = (
                saturated_vapor_at_pressure.specific_enthalpy_J_kg
                - saturated_liquid_at_pressure.specific_enthalpy_J_kg
            )
            if not math.isfinite(latent) or latent <= 0.0:
                raise ValueError("Distributed boiling requires positive latent heat")
            saturation_slope = (
                self.properties.saturation_enthalpy_pressure_derivative_J_kg_Pa(
                    state.pressure_Pa, 0.0
                )
            )
            if not math.isfinite(saturation_slope):
                raise ValueError("Saturated-liquid enthalpy derivative must be finite")
            saturation_tolerance = latent * 1.0e-7
            eligible = {
                index for index, enthalpy in enumerate(
                    state.liquid_specific_enthalpies_J_kg
                )
                if enthalpy >= (
                    saturated_liquid_at_pressure.specific_enthalpy_J_kg
                    - saturation_tolerance
                )
            }
            base_numerator = float(
                np.sum(specific_volumes * mass_rates + vh * source)
            )
            vapor_index = nl

            def solve_active(active: list[int]) -> tuple[float, np.ndarray]:
                matrix = np.zeros((len(active) + 1, len(active) + 1))
                rhs = np.zeros(len(active) + 1)
                matrix[0, 0] = denominator
                rhs[0] = -base_numerator
                for column, liquid_index in enumerate(active, start=1):
                    vapor_source_enthalpy = (
                        saturated_vapor_at_pressure.specific_enthalpy_J_kg
                        - cells[vapor_index].specific_enthalpy_J_kg
                    )
                    volume_coefficient = (
                        specific_volumes[vapor_index]
                        - specific_volumes[liquid_index]
                        - vh[liquid_index] * latent
                        + vh[vapor_index] * vapor_source_enthalpy
                    )
                    matrix[0, column] = volume_coefficient
                    matrix[column, 0] = (
                        masses[liquid_index] * saturation_slope
                        - volumes[liquid_index]
                    )
                    matrix[column, column] = latent
                    rhs[column] = source[liquid_index]
                solution = np.linalg.solve(matrix, rhs)
                return float(solution[0]), np.asarray(solution[1:])

            active: list[int] = []
            pressure_rate, active_rates = solve_active(active)
            rate_tolerance = 1.0e-12
            for _ in range(2 * nl + 2):
                negative = [
                    (active[index], rate)
                    for index, rate in enumerate(active_rates)
                    if rate < -rate_tolerance
                ]
                if negative:
                    remove = min(negative, key=lambda pair: pair[1])[0]
                    active.remove(remove)
                    pressure_rate, active_rates = solve_active(active)
                    continue
                violations: list[tuple[int, float]] = []
                for liquid_index in sorted(eligible.difference(active)):
                    free_margin_rate = (
                        source[liquid_index]
                        + (
                            volumes[liquid_index]
                            - masses[liquid_index] * saturation_slope
                        ) * pressure_rate
                    ) / masses[liquid_index]
                    if free_margin_rate > rate_tolerance:
                        violations.append((liquid_index, free_margin_rate))
                if not violations:
                    break
                active.append(max(violations, key=lambda pair: pair[1])[0])
                active.sort()
                pressure_rate, active_rates = solve_active(active)
            else:
                raise RuntimeError("Distributed-boiling active set did not converge")

            for liquid_index, rate in zip(active, active_rates):
                rate = max(0.0, float(rate))
                distributed_boiling[liquid_index] = rate
                mass_rates[liquid_index] -= rate
                mass_rates[vapor_index] += rate
                source[liquid_index] -= rate * latent
                source[vapor_index] += rate * (
                    saturated_vapor_at_pressure.specific_enthalpy_J_kg
                    - cells[vapor_index].specific_enthalpy_J_kg
                )
            distributed_boiling_latent_power = float(
                np.sum(distributed_boiling) * latent
            )
            reconstructed_pressure_rate = -float(np.sum(
                specific_volumes * mass_rates + vh * source
            )) / denominator
            if not math.isclose(
                reconstructed_pressure_rate, pressure_rate, rel_tol=1e-10,
                abs_tol=1e-10,
            ):
                raise RuntimeError("Distributed-boiling pressure solve is inconsistent")
            pressure_rate = reconstructed_pressure_rate
        elif p.distributed_boiling_model == "phase_change_complementarity":
            """Apply the optional two-sided distributed phase-change closure.

            The liquid and vapor branches use the same complementarity idea:
            an active branch is constrained to its saturation enthalpy slope,
            while the transfer rate is nonnegative and consumes/releases the
            latent heat.  The rate solve is conservative in the rigid-volume
            pressure equation.  It is intentionally enabled only by the
            explicit ``phase_change_complementarity`` option; the historical
            default and the one-sided closure are unchanged.
            """
            saturated_liquid = self.properties.saturated_liquid(state.pressure_Pa)
            saturated_vapor = self.properties.saturated_vapor(state.pressure_Pa)
            latent = (
                saturated_vapor.specific_enthalpy_J_kg
                - saturated_liquid.specific_enthalpy_J_kg
            )
            if not math.isfinite(latent) or latent <= 0.0:
                raise ValueError("Two-sided phase change requires positive latent heat")
            liquid_slope = self.properties.saturation_enthalpy_pressure_derivative_J_kg_Pa(
                state.pressure_Pa, 0.0
            )
            vapor_slope = self.properties.saturation_enthalpy_pressure_derivative_J_kg_Pa(
                state.pressure_Pa, 1.0
            )
            if not math.isfinite(liquid_slope) or not math.isfinite(vapor_slope):
                raise ValueError("Saturation enthalpy derivatives must be finite")
            saturation_tolerance = max(abs(latent), 1.0) * 1.0e-7
            liquid_candidates = {
                index
                for index, enthalpy in enumerate(state.liquid_specific_enthalpies_J_kg)
                if enthalpy >= saturated_liquid.specific_enthalpy_J_kg - saturation_tolerance
            }
            vapor_candidates = {
                local
                for local, enthalpy in enumerate(state.vapor_specific_enthalpies_J_kg)
                if enthalpy <= saturated_vapor.specific_enthalpy_J_kg + saturation_tolerance
            }
            vapor_recipient = 0
            liquid_recipient = nl - 1
            base_numerator = float(np.sum(specific_volumes * mass_rates + vh * source))

            def transfer_delta(kind: str, index: int) -> tuple[np.ndarray, np.ndarray]:
                delta_mass = np.zeros(nc)
                delta_source = np.zeros(nc)
                if kind == "evaporation":
                    vapor_index = nl + vapor_recipient
                    delta_mass[index] -= 1.0
                    delta_mass[vapor_index] += 1.0
                    delta_source[index] -= latent
                    delta_source[vapor_index] += (
                        saturated_vapor.specific_enthalpy_J_kg
                        - cells[vapor_index].specific_enthalpy_J_kg
                    )
                else:
                    vapor_index = nl + index
                    delta_mass[vapor_index] -= 1.0
                    delta_mass[liquid_recipient] += 1.0
                    delta_source[vapor_index] += latent
                    delta_source[liquid_recipient] += (
                        saturated_liquid.specific_enthalpy_J_kg
                        - cells[liquid_recipient].specific_enthalpy_J_kg
                    )
                return delta_mass, delta_source

            all_transfers = [
                ("evaporation", index) for index in sorted(liquid_candidates)
            ] + [
                ("condensation", local) for local in sorted(vapor_candidates)
            ]
            # A common pressure/temperature boundary has one net phase
            # direction at a differential instant.  Locking the active set to
            # the stronger side prevents a mathematically singular
            # evaporation/condensation pair that would otherwise represent a
            # zero-net exchange between the same receiving cells.
            transfers = all_transfers

            def tangent_cell(transfer: tuple[str, int]) -> tuple[int, float]:
                kind, index = transfer
                if kind == "evaporation":
                    return index, liquid_slope
                return nl + index, vapor_slope

            def solve_two_sided(active: list[int]) -> tuple[float, np.ndarray]:
                size = len(active) + 1
                matrix = np.zeros((size, size))
                rhs = np.zeros(size)
                matrix[0, 0] = denominator
                rhs[0] = -base_numerator
                deltas = [transfer_delta(*transfers[item]) for item in active]
                for column, (delta_mass, delta_source) in enumerate(deltas, start=1):
                    matrix[0, column] = float(
                        np.sum(specific_volumes * delta_mass + vh * delta_source)
                    )
                for row, item in enumerate(active, start=1):
                    cell_index, slope = tangent_cell(transfers[item])
                    matrix[row, 0] = masses[cell_index] * slope - volumes[cell_index]
                    rhs[row] = source[cell_index]
                    for column, (delta_mass, delta_source) in enumerate(deltas, start=1):
                        matrix[row, column] = -delta_source[cell_index]
                solution = np.linalg.solve(matrix, rhs)
                return float(solution[0]), np.asarray(solution[1:])

            active: list[int] = []
            pressure_rate, active_rates = solve_two_sided(active)
            rate_tolerance = 1.0e-12
            liquid_margin = 0.0
            vapor_margin = 0.0
            for transfer in all_transfers:
                cell_index, slope = tangent_cell(transfer)
                margin = (
                    source[cell_index]
                    + (volumes[cell_index] - masses[cell_index] * slope) * pressure_rate
                ) / masses[cell_index]
                if transfer[0] == "evaporation":
                    liquid_margin = max(liquid_margin, margin)
                else:
                    vapor_margin = max(vapor_margin, -margin)
            if liquid_margin > rate_tolerance or vapor_margin > rate_tolerance:
                direction = (
                    "evaporation"
                    if liquid_margin >= vapor_margin
                    else "condensation"
                )
                transfers = [
                    transfer for transfer in all_transfers if transfer[0] == direction
                ]
                pressure_rate, active_rates = solve_two_sided(active)
            for _ in range(2 * len(transfers) + 2):
                negative = [
                    (active[index], rate)
                    for index, rate in enumerate(active_rates)
                    if rate < -rate_tolerance
                ]
                if negative:
                    remove = min(negative, key=lambda pair: pair[1])[0]
                    active.remove(remove)
                    pressure_rate, active_rates = solve_two_sided(active)
                    continue
                violations: list[tuple[int, float]] = []
                active_set = set(active)
                for item, transfer in enumerate(transfers):
                    if item in active_set:
                        continue
                    cell_index, slope = tangent_cell(transfer)
                    free_margin_rate = (
                        source[cell_index]
                        + (volumes[cell_index] - masses[cell_index] * slope) * pressure_rate
                    ) / masses[cell_index]
                    kind, _ = transfer
                    if (kind == "evaporation" and free_margin_rate > rate_tolerance) or (
                        kind == "condensation" and free_margin_rate < -rate_tolerance
                    ):
                        violations.append((item, abs(free_margin_rate)))
                if not violations:
                    break
                active.append(max(violations, key=lambda pair: pair[1])[0])
                active.sort()
                pressure_rate, active_rates = solve_two_sided(active)
            else:
                raise RuntimeError("Two-sided phase-change active set did not converge")

            evaporation_total = 0.0
            condensation_total = 0.0
            for item, rate in zip(active, active_rates):
                rate = max(0.0, float(rate))
                kind, index = transfers[item]
                delta_mass, delta_source = transfer_delta(kind, index)
                mass_rates += rate * delta_mass
                source += rate * delta_source
                if kind == "evaporation":
                    distributed_boiling[index] += rate
                    evaporation_total += rate
                else:
                    condensation_total += rate
            distributed_condensation_rate = condensation_total
            distributed_boiling_latent_power = (evaporation_total - condensation_total) * latent
            reconstructed_pressure_rate = -float(np.sum(
                specific_volumes * mass_rates + vh * source
            )) / denominator
            if not math.isclose(
                reconstructed_pressure_rate, pressure_rate, rel_tol=1e-10,
                abs_tol=1e-10,
            ):
                raise RuntimeError("Two-sided phase-change pressure solve is inconsistent")
            pressure_rate = reconstructed_pressure_rate
        else:
            pressure_rate = -float(
                np.sum(specific_volumes * mass_rates + vh * source)
            ) / denominator
        enthalpy_rates = (source + volumes * pressure_rate) / masses

        q_ambient = self._wall_ambient_UA_W_K * (
            ambient_temperature_K - np.asarray(state.wall_temperatures_K)
        )
        q_wall_axial = np.zeros(p.wall_node_count)
        for index in range(p.wall_node_count - 1):
            heat = p.wall_axial_conductance_W_K * (
                state.wall_temperatures_K[index + 1] - state.wall_temperatures_K[index]
            )
            q_wall_axial[index] += heat
            q_wall_axial[index + 1] -= heat
        q_wall_net = q_ambient + direct_wall - q_wall_to_fluid + q_wall_axial
        wall_temperature_rates = q_wall_net / self._wall_heat_capacities_J_K

        total_mass_rate = float(np.sum(mass_rates))
        total_fluid_energy_rate = float(
            np.sum(masses * enthalpy_rates + np.asarray([cell.specific_enthalpy_J_kg for cell in cells]) * mass_rates)
            - self.total_rigid_volume_m3 * pressure_rate
        )
        total_energy_rate = total_fluid_energy_rate + float(
            np.sum(self._wall_heat_capacities_J_K * wall_temperature_rates)
        )
        boundary_mass = boundary_lm + boundary_vm
        boundary_energy = (
            boundary_le + boundary_ve + float(np.sum(q_ambient + direct_wall))
        )
        # Direct fluid heat is already inside q_fluid; add only its explicit
        # boundary contribution here. Wall-fluid and all internal paths cancel.
        boundary_energy += float(np.sum(direct_liquid) + np.sum(direct_vapor))
        return LayeredTankDerivative(
            pressure_Pa_s=pressure_rate,
            liquid_mass_rates_kg_s=tuple(float(value) for value in mass_rates[:nl]),
            liquid_enthalpy_rates_J_kg_s=tuple(float(value) for value in enthalpy_rates[:nl]),
            vapor_mass_rates_kg_s=tuple(float(value) for value in mass_rates[nl:]),
            vapor_enthalpy_rates_J_kg_s=tuple(float(value) for value in enthalpy_rates[nl:]),
            wall_temperature_rates_K_s=tuple(float(value) for value in wall_temperature_rates),
            evaporation_rate_kg_s=float(
                evaporation + np.sum(distributed_boiling)
            ),
            liquid_interface_heat_W=float(q_liquid_interface),
            vapor_interface_heat_W=float(q_vapor_interface),
            latent_phase_change_power_W=float(
                evaporation * latent_heat + distributed_boiling_latent_power
            ),
            distributed_boiling_rates_kg_s=tuple(
                float(value) for value in distributed_boiling
            ),
            distributed_boiling_rate_kg_s=float(
                np.sum(distributed_boiling) - distributed_condensation_rate
            ),
            distributed_boiling_latent_power_W=distributed_boiling_latent_power,
            interface_temperature_K=float(saturated_liquid.temperature_K),
            interface_pressure_Pa=float(saturated_liquid.pressure_Pa),
            interface_mass_flux_kg_m2_s=float(interface.mass_flux_kg_m2_s),
            interface_kinetic_residual_kg_m2_s=float(
                interface.kinetic_residual_kg_m2_s
            ),
            interface_energy_residual_W=float(interface.energy_residual_W),
            liquid_wall_circulation_rates_kg_s=tuple(
                float(value) for value in liquid_circulation_rates
            ),
            vapor_wall_circulation_rates_kg_s=tuple(
                float(value) for value in vapor_circulation_rates
            ),
            liquid_circulation_lower_cell_heat_W=tuple(
                float(value) for value in liquid_circulation_lower_heat
            ),
            vapor_circulation_lower_cell_heat_W=tuple(
                float(value) for value in vapor_circulation_lower_heat
            ),
            liquid_wall_heat_W=tuple(
                float(value) for value in q_cells_from_wall[:nl]
            ),
            vapor_wall_heat_W=tuple(
                float(value) for value in q_cells_from_wall[nl:]
            ),
            ambient_heat_W=float(np.sum(q_ambient)),
            total_mass_rate_kg_s=total_mass_rate,
            total_energy_rate_W=total_energy_rate,
            boundary_mass_rate_kg_s=float(boundary_mass),
            boundary_energy_rate_W=float(boundary_energy),
        )

    @staticmethod
    def _advance(
        state: LayeredTankState, derivative: LayeredTankDerivative, dt_s: float
    ) -> LayeredTankState:
        return LayeredTankState(
            state.pressure_Pa + derivative.pressure_Pa_s * dt_s,
            tuple(m + dm * dt_s for m, dm in zip(
                state.liquid_masses_kg, derivative.liquid_mass_rates_kg_s
            )),
            tuple(h + dh * dt_s for h, dh in zip(
                state.liquid_specific_enthalpies_J_kg,
                derivative.liquid_enthalpy_rates_J_kg_s,
            )),
            tuple(m + dm * dt_s for m, dm in zip(
                state.vapor_masses_kg, derivative.vapor_mass_rates_kg_s
            )),
            tuple(h + dh * dt_s for h, dh in zip(
                state.vapor_specific_enthalpies_J_kg,
                derivative.vapor_enthalpy_rates_J_kg_s,
            )),
            tuple(t + dT * dt_s for t, dT in zip(
                state.wall_temperatures_K, derivative.wall_temperature_rates_K_s
            )),
        )

    def total_mass_kg(self, state: LayeredTankState) -> float:
        return float(sum(state.liquid_masses_kg) + sum(state.vapor_masses_kg))

    def total_energy_J(self, state: LayeredTankState) -> float:
        thermo = self.thermo(state)
        masses = np.asarray((*state.liquid_masses_kg, *state.vapor_masses_kg))
        enthalpies = np.asarray((
            *state.liquid_specific_enthalpies_J_kg,
            *state.vapor_specific_enthalpies_J_kg,
        ))
        fluid = float(
            np.sum(masses * enthalpies)
            - state.pressure_Pa * self.total_rigid_volume_m3
        )
        wall = float(np.sum(
            self._wall_heat_capacities_J_K * np.asarray(state.wall_temperatures_K)
        ))
        return fluid + wall

    def simulate(
        self,
        initial: LayeredTankState,
        duration_s: float,
        time_step_s: float,
        ambient_temperature_K: float | Callable[[float], float],
        flow_callback: FlowCallback | None = None,
        fluid_heat_callback: FluidHeatCallback | None = None,
        wall_heat_callback: WallHeatCallback | None = None,
        minimum_step_s: float | None = None,
    ) -> list[LayeredTankStepResult]:
        if duration_s <= 0.0 or time_step_s <= 0.0:
            raise ValueError("duration_s and time_step_s must be positive")
        if minimum_step_s is not None and (
            not math.isfinite(minimum_step_s)
            or minimum_step_s <= 0.0
            or minimum_step_s > time_step_s
        ):
            raise ValueError(
                "minimum_step_s must be finite, positive, and no larger than time_step_s"
            )
        # RK4 stages can briefly cross a phase manifold even when the final
        # requested step would be physically admissible. Callers that need an
        # event-safe diagnostic can opt into deterministic substepping by
        # supplying ``minimum_step_s``; the historical fixed-step behaviour
        # remains the production default until that policy is separately
        # benchmarked.
        minimum_step = (
            float(minimum_step_s)
            if minimum_step_s is not None
            else float(time_step_s)
        )
        p = self.parameters
        empty_flows: CellFlows = (
            tuple(() for _ in range(p.liquid_cell_count)),
            tuple(() for _ in range(p.vapor_cell_count)),
        )
        state = self.project_pressure(initial)
        initial_mass = self.total_mass_kg(state)
        initial_energy = self.total_energy_J(state)
        cumulative_mass = 0.0
        cumulative_energy = 0.0
        cumulative_evaporation = 0.0
        cumulative_liquid_interface_heat = 0.0
        cumulative_vapor_interface_heat = 0.0
        cumulative_latent_phase_change_energy = 0.0
        cumulative_distributed_boiling_mass = 0.0
        cumulative_distributed_boiling_latent_energy = 0.0
        results: list[LayeredTankStepResult] = []

        def evaluate(time: float, candidate: LayeredTankState) -> tuple[LayeredTankDerivative, LayeredTankState]:
            projected = self.project_pressure(candidate)
            projected = self._project_distributed_boiling_manifold(projected)
            thermo = self.thermo(projected)
            liquid_flows, vapor_flows = (
                flow_callback(time, thermo) if flow_callback is not None else empty_flows
            )
            liquid_heat, vapor_heat = (
                fluid_heat_callback(time, thermo)
                if fluid_heat_callback is not None
                else (np.zeros(p.liquid_cell_count), np.zeros(p.vapor_cell_count))
            )
            wall_heat = (
                wall_heat_callback(time, thermo)
                if wall_heat_callback is not None
                else np.zeros(p.wall_node_count)
            )
            ambient = ambient_temperature_K(time) if callable(ambient_temperature_K) else ambient_temperature_K
            return self.derivative(
                projected, liquid_flows, vapor_flows, ambient,
                liquid_heat, vapor_heat, wall_heat,
            ), projected

        def rk4_step(
            step_time: float,
            step_state: LayeredTankState,
            step_dt: float,
        ) -> tuple[LayeredTankState, LayeredTankDerivative]:
            """Advance one admissible RK4 step, preserving phase checks."""

            k1, state = evaluate(step_time, step_state)
            k2, _ = evaluate(step_time + step_dt / 2.0, self._advance(state, k1, step_dt / 2.0))
            k3, _ = evaluate(step_time + step_dt / 2.0, self._advance(state, k2, step_dt / 2.0))
            k4, _ = evaluate(step_time + step_dt, self._advance(state, k3, step_dt))

            def combine(name: str) -> tuple[float, ...]:
                a = np.asarray(getattr(k1, name)); b = np.asarray(getattr(k2, name))
                c = np.asarray(getattr(k3, name)); d = np.asarray(getattr(k4, name))
                return tuple(float(value) for value in (a + 2*b + 2*c + d) / 6.0)

            averaged = LayeredTankDerivative(
                pressure_Pa_s=(k1.pressure_Pa_s + 2*k2.pressure_Pa_s + 2*k3.pressure_Pa_s + k4.pressure_Pa_s) / 6.0,
                liquid_mass_rates_kg_s=combine("liquid_mass_rates_kg_s"),
                liquid_enthalpy_rates_J_kg_s=combine("liquid_enthalpy_rates_J_kg_s"),
                vapor_mass_rates_kg_s=combine("vapor_mass_rates_kg_s"),
                vapor_enthalpy_rates_J_kg_s=combine("vapor_enthalpy_rates_J_kg_s"),
                wall_temperature_rates_K_s=combine("wall_temperature_rates_K_s"),
                evaporation_rate_kg_s=(k1.evaporation_rate_kg_s + 2*k2.evaporation_rate_kg_s + 2*k3.evaporation_rate_kg_s + k4.evaporation_rate_kg_s) / 6.0,
                liquid_interface_heat_W=(k1.liquid_interface_heat_W + 2*k2.liquid_interface_heat_W + 2*k3.liquid_interface_heat_W + k4.liquid_interface_heat_W) / 6.0,
                vapor_interface_heat_W=(k1.vapor_interface_heat_W + 2*k2.vapor_interface_heat_W + 2*k3.vapor_interface_heat_W + k4.vapor_interface_heat_W) / 6.0,
                latent_phase_change_power_W=(k1.latent_phase_change_power_W + 2*k2.latent_phase_change_power_W + 2*k3.latent_phase_change_power_W + k4.latent_phase_change_power_W) / 6.0,
                distributed_boiling_rates_kg_s=combine(
                    "distributed_boiling_rates_kg_s"
                ),
                distributed_boiling_rate_kg_s=(k1.distributed_boiling_rate_kg_s + 2*k2.distributed_boiling_rate_kg_s + 2*k3.distributed_boiling_rate_kg_s + k4.distributed_boiling_rate_kg_s) / 6.0,
                distributed_boiling_latent_power_W=(k1.distributed_boiling_latent_power_W + 2*k2.distributed_boiling_latent_power_W + 2*k3.distributed_boiling_latent_power_W + k4.distributed_boiling_latent_power_W) / 6.0,
                interface_temperature_K=(k1.interface_temperature_K + 2*k2.interface_temperature_K + 2*k3.interface_temperature_K + k4.interface_temperature_K) / 6.0,
                interface_pressure_Pa=(k1.interface_pressure_Pa + 2*k2.interface_pressure_Pa + 2*k3.interface_pressure_Pa + k4.interface_pressure_Pa) / 6.0,
                interface_mass_flux_kg_m2_s=(k1.interface_mass_flux_kg_m2_s + 2*k2.interface_mass_flux_kg_m2_s + 2*k3.interface_mass_flux_kg_m2_s + k4.interface_mass_flux_kg_m2_s) / 6.0,
                interface_kinetic_residual_kg_m2_s=(k1.interface_kinetic_residual_kg_m2_s + 2*k2.interface_kinetic_residual_kg_m2_s + 2*k3.interface_kinetic_residual_kg_m2_s + k4.interface_kinetic_residual_kg_m2_s) / 6.0,
                interface_energy_residual_W=(k1.interface_energy_residual_W + 2*k2.interface_energy_residual_W + 2*k3.interface_energy_residual_W + k4.interface_energy_residual_W) / 6.0,
                liquid_wall_circulation_rates_kg_s=combine("liquid_wall_circulation_rates_kg_s"),
                vapor_wall_circulation_rates_kg_s=combine("vapor_wall_circulation_rates_kg_s"),
                liquid_circulation_lower_cell_heat_W=combine("liquid_circulation_lower_cell_heat_W"),
                vapor_circulation_lower_cell_heat_W=combine("vapor_circulation_lower_cell_heat_W"),
                liquid_wall_heat_W=combine("liquid_wall_heat_W"),
                vapor_wall_heat_W=combine("vapor_wall_heat_W"),
                ambient_heat_W=(k1.ambient_heat_W + 2*k2.ambient_heat_W + 2*k3.ambient_heat_W + k4.ambient_heat_W) / 6.0,
                total_mass_rate_kg_s=(k1.total_mass_rate_kg_s + 2*k2.total_mass_rate_kg_s + 2*k3.total_mass_rate_kg_s + k4.total_mass_rate_kg_s) / 6.0,
                total_energy_rate_W=(k1.total_energy_rate_W + 2*k2.total_energy_rate_W + 2*k3.total_energy_rate_W + k4.total_energy_rate_W) / 6.0,
                boundary_mass_rate_kg_s=(k1.boundary_mass_rate_kg_s + 2*k2.boundary_mass_rate_kg_s + 2*k3.boundary_mass_rate_kg_s + k4.boundary_mass_rate_kg_s) / 6.0,
                boundary_energy_rate_W=(k1.boundary_energy_rate_W + 2*k2.boundary_energy_rate_W + 2*k3.boundary_energy_rate_W + k4.boundary_energy_rate_W) / 6.0,
            )
            state = self.project_pressure(self._advance(state, averaged, step_dt))
            state = self._project_distributed_boiling_manifold(state)


            return state, averaged

        time = 0.0
        while time < duration_s - 1e-12:
            dt = min(time_step_s, duration_s - time)
            while True:
                try:
                    candidate_state, averaged = rk4_step(time, state, dt)
                    break
                except (ValueError, RuntimeError, OverflowError):
                    if dt <= minimum_step * (1.0 + 1.0e-12):
                        raise
                    dt = max(minimum_step, dt / 2.0)
            state = candidate_state
            cumulative_mass += averaged.boundary_mass_rate_kg_s * dt
            cumulative_energy += averaged.boundary_energy_rate_W * dt
            cumulative_evaporation += averaged.evaporation_rate_kg_s * dt
            cumulative_liquid_interface_heat += averaged.liquid_interface_heat_W * dt
            cumulative_vapor_interface_heat += averaged.vapor_interface_heat_W * dt
            cumulative_latent_phase_change_energy += averaged.latent_phase_change_power_W * dt
            cumulative_distributed_boiling_mass += (
                averaged.distributed_boiling_rate_kg_s * dt
            )
            cumulative_distributed_boiling_latent_energy += (
                averaged.distributed_boiling_latent_power_W * dt
            )
            time += dt
            results.append(LayeredTankStepResult(
                time_s=time,
                state=state,
                thermo=self.thermo(state),
                cumulative_boundary_mass_kg=cumulative_mass,
                cumulative_boundary_energy_J=cumulative_energy,
                cumulative_evaporation_mass_kg=cumulative_evaporation,
                cumulative_liquid_interface_heat_J=cumulative_liquid_interface_heat,
                cumulative_vapor_interface_heat_J=cumulative_vapor_interface_heat,
                cumulative_latent_phase_change_energy_J=cumulative_latent_phase_change_energy,
                cumulative_distributed_boiling_mass_kg=(
                    cumulative_distributed_boiling_mass
                ),
                cumulative_distributed_boiling_latent_energy_J=(
                    cumulative_distributed_boiling_latent_energy
                ),
                total_mass_residual_kg=self.total_mass_kg(state) - initial_mass - cumulative_mass,
                total_energy_residual_J=self.total_energy_J(state) - initial_energy - cumulative_energy,
            ))
        return results
