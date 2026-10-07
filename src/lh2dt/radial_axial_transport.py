"""Conservative wall-layer/core circulation on a radial-axial phase grid.

The operator in this module is the connection layer between the independent
buoyancy-loop momentum model and a future common-pressure two-phase tank.  One
single-phase region is represented by two radial columns (wall layer and core)
and an arbitrary number of axial levels.  At every horizontal boundary a
closed circulation loop moves equal mass upward in one column and downward in
the other.  Upwind enthalpy follows the actual flow direction.

The operator returns mass and enthalpy-flow source terms only.  It neither
changes pressure nor integrates state.  A tank DAE can therefore combine the
sources with wall heat, phase change, external ports, and its common-pressure
rigid-volume constraint without changing this transport contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .natural_circulation import (
    CirculationLegGeometry,
    NaturalCirculationLoop,
    NaturalCirculationResult,
)
from .properties import HydrogenProperties, ThermoState


@dataclass(frozen=True)
class RadialAxialBoundaryGeometry:
    """Hydraulic geometry for one boundary between adjacent axial levels."""

    vertical_separation_m: float
    wall_leg: CirculationLegGeometry
    core_leg: CirculationLegGeometry

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.vertical_separation_m)
            or self.vertical_separation_m <= 0.0
        ):
            raise ValueError(
                "vertical_separation_m must be finite and positive"
            )


@dataclass(frozen=True)
class RadialAxialPhaseState:
    """Mass and specific enthalpy for wall and core cells at each level."""

    wall_masses_kg: tuple[float, ...]
    wall_specific_enthalpies_J_kg: tuple[float, ...]
    core_masses_kg: tuple[float, ...]
    core_specific_enthalpies_J_kg: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.wall_masses_kg)
        if count < 2:
            raise ValueError("radial-axial phase grid requires at least two levels")
        if not all(
            len(values) == count
            for values in (
                self.wall_specific_enthalpies_J_kg,
                self.core_masses_kg,
                self.core_specific_enthalpies_J_kg,
            )
        ):
            raise ValueError("all wall/core state arrays must have equal length")
        for masses in (self.wall_masses_kg, self.core_masses_kg):
            if any(not math.isfinite(value) or value <= 0.0 for value in masses):
                raise ValueError("all cell masses must be finite and positive")
        for enthalpies in (
            self.wall_specific_enthalpies_J_kg,
            self.core_specific_enthalpies_J_kg,
        ):
            if any(not math.isfinite(value) for value in enthalpies):
                raise ValueError("all cell enthalpies must be finite")

    @property
    def level_count(self) -> int:
        return len(self.wall_masses_kg)


@dataclass(frozen=True)
class RadialAxialTransportResult:
    """Internal conservative source terms for one single-phase grid."""

    wall_mass_rates_kg_s: tuple[float, ...]
    core_mass_rates_kg_s: tuple[float, ...]
    wall_enthalpy_flow_W: tuple[float, ...]
    core_enthalpy_flow_W: tuple[float, ...]
    boundary_mass_flows_kg_s: tuple[float, ...]
    boundary_results: tuple[NaturalCirculationResult, ...]
    total_mass_residual_kg_s: float
    total_enthalpy_flow_residual_W: float
    level_mass_residuals_kg_s: tuple[float, ...]


class RadialAxialCirculationTransport:
    """Evaluate conservative circulation between wall and core columns.

    Positive boundary flow means upward transport in the wall column and an
    equal downward transport in the core column.  Negative flow reverses both
    directions.  Representative momentum states are obtained by mass-weighted
    averaging of the two cells adjacent to each boundary at the supplied
    common pressure.  Fluxes themselves use donor-cell (upwind) enthalpy.
    """

    def __init__(
        self,
        boundaries: Sequence[RadialAxialBoundaryGeometry],
        gravity_m_s2: float = 9.80665,
        properties: HydrogenProperties | None = None,
        closed_loop_turns: bool = False,
    ) -> None:
        self.boundaries = tuple(boundaries)
        if not self.boundaries:
            raise ValueError("at least one axial boundary is required")
        if not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
            raise ValueError("gravity_m_s2 must be finite and positive")
        if not isinstance(closed_loop_turns, bool):
            raise ValueError("closed_loop_turns must be bool")
        self.gravity = float(gravity_m_s2)
        self.properties = properties or HydrogenProperties()
        self.closed_loop_turns = closed_loop_turns

    def _states(
        self, pressure_Pa: float, state: RadialAxialPhaseState
    ) -> tuple[tuple[ThermoState, ...], tuple[ThermoState, ...]]:
        if not math.isfinite(pressure_Pa) or pressure_Pa <= 0.0:
            raise ValueError("pressure_Pa must be finite and positive")
        wall = tuple(
            self.properties.from_ph(pressure_Pa, enthalpy)
            for enthalpy in state.wall_specific_enthalpies_J_kg
        )
        core = tuple(
            self.properties.from_ph(pressure_Pa, enthalpy)
            for enthalpy in state.core_specific_enthalpies_J_kg
        )
        quality_tolerance = 1.0e-10
        for cell in (*wall, *core):
            if (
                cell.quality is not None
                and quality_tolerance < cell.quality < 1.0 - quality_tolerance
            ):
                raise ValueError(
                    "radial-axial circulation transport is single-phase; "
                    "two-phase cells require a replaceable momentum model"
                )
        def phase_label(cell: ThermoState) -> str:
            if cell.quality is not None:
                return "liquid" if cell.quality <= quality_tolerance else "vapor"
            return "vapor" if cell.phase in {"gas", "supercritical_gas"} else "liquid"

        phases = {phase_label(cell) for cell in (*wall, *core)}
        if len(phases) != 1:
            raise ValueError("all wall/core cells must belong to one phase")
        return wall, core

    def evaluate(
        self, pressure_Pa: float, state: RadialAxialPhaseState
    ) -> RadialAxialTransportResult:
        if len(self.boundaries) != state.level_count - 1:
            raise ValueError(
                "boundary count must equal state.level_count - 1"
            )
        wall_states, core_states = self._states(pressure_Pa, state)
        count = state.level_count
        wall_mass = np.zeros(count)
        core_mass = np.zeros(count)
        wall_energy = np.zeros(count)
        core_energy = np.zeros(count)
        flows: list[float] = []
        results: list[NaturalCirculationResult] = []

        wall_masses = np.asarray(state.wall_masses_kg, dtype=float)
        core_masses = np.asarray(state.core_masses_kg, dtype=float)
        wall_h = np.asarray(state.wall_specific_enthalpies_J_kg, dtype=float)
        core_h = np.asarray(state.core_specific_enthalpies_J_kg, dtype=float)

        for lower, geometry in enumerate(self.boundaries):
            upper = lower + 1
            wall_pair_mass = wall_masses[lower] + wall_masses[upper]
            core_pair_mass = core_masses[lower] + core_masses[upper]
            wall_average_h = float(
                (
                    wall_masses[lower] * wall_h[lower]
                    + wall_masses[upper] * wall_h[upper]
                )
                / wall_pair_mass
            )
            core_average_h = float(
                (
                    core_masses[lower] * core_h[lower]
                    + core_masses[upper] * core_h[upper]
                )
                / core_pair_mass
            )
            wall_average = self.properties.from_ph(pressure_Pa, wall_average_h)
            core_average = self.properties.from_ph(pressure_Pa, core_average_h)
            loop = NaturalCirculationLoop(
                vertical_separation_m=geometry.vertical_separation_m,
                rising_leg=geometry.wall_leg,
                descending_leg=geometry.core_leg,
                gravity_m_s2=self.gravity,
                properties=self.properties,
            )
            result = loop.evaluate(wall_average, core_average)
            mass_flow = result.mass_flow_kg_s
            flows.append(mass_flow)
            results.append(result)

            # Signed mass flow is positive upward in the wall leg and downward
            # in the core leg.  These formulas remain valid after reversal.
            wall_mass[lower] -= mass_flow
            wall_mass[upper] += mass_flow
            core_mass[lower] += mass_flow
            core_mass[upper] -= mass_flow

            if mass_flow >= 0.0:
                wall_donor_h = wall_h[lower]
                core_donor_h = core_h[upper]
            else:
                wall_donor_h = wall_h[upper]
                core_donor_h = core_h[lower]
            wall_power = mass_flow * wall_donor_h
            core_power = mass_flow * core_donor_h
            wall_energy[lower] -= wall_power
            wall_energy[upper] += wall_power
            core_energy[lower] += core_power
            core_energy[upper] -= core_power

            if self.closed_loop_turns:
                # Complete the local loop with radial turns at its upper and
                # lower ends. Fixed Eulerian cells then have zero net mass
                # accumulation from steady circulation while enthalpy moves.
                if mass_flow >= 0.0:
                    magnitude = mass_flow
                    wall_mass[upper] -= magnitude
                    core_mass[upper] += magnitude
                    turn_power = magnitude * wall_h[upper]
                    wall_energy[upper] -= turn_power
                    core_energy[upper] += turn_power
                    core_mass[lower] -= magnitude
                    wall_mass[lower] += magnitude
                    turn_power = magnitude * core_h[lower]
                    core_energy[lower] -= turn_power
                    wall_energy[lower] += turn_power
                else:
                    magnitude = -mass_flow
                    wall_mass[lower] -= magnitude
                    core_mass[lower] += magnitude
                    turn_power = magnitude * wall_h[lower]
                    wall_energy[lower] -= turn_power
                    core_energy[lower] += turn_power
                    core_mass[upper] -= magnitude
                    wall_mass[upper] += magnitude
                    turn_power = magnitude * core_h[upper]
                    core_energy[upper] -= turn_power
                    wall_energy[upper] += turn_power

        total_mass = float(np.sum(wall_mass) + np.sum(core_mass))
        total_energy = float(np.sum(wall_energy) + np.sum(core_energy))
        level_mass = wall_mass + core_mass
        return RadialAxialTransportResult(
            wall_mass_rates_kg_s=tuple(float(x) for x in wall_mass),
            core_mass_rates_kg_s=tuple(float(x) for x in core_mass),
            wall_enthalpy_flow_W=tuple(float(x) for x in wall_energy),
            core_enthalpy_flow_W=tuple(float(x) for x in core_energy),
            boundary_mass_flows_kg_s=tuple(flows),
            boundary_results=tuple(results),
            total_mass_residual_kg_s=total_mass,
            total_enthalpy_flow_residual_W=total_energy,
            level_mass_residuals_kg_s=tuple(float(x) for x in level_mass),
        )
