"""Reusable common-pressure rigid-volume DAE for arbitrary fluid cells.

Each cell stores mass and specific enthalpy while all cells share pressure.
Callers supply cell mass rates and complete enthalpy-inventory rates
``d(m h)/dt`` from ports, heat, phase change, and internal transport.  The
system differentiates the rigid total-volume constraint to obtain ``dp/dt``
and then returns cell ``dh/dt``.  It owns no tank geometry or closure model,
so axial, radial-axial, and future component discretizations can reuse the
same thermodynamic coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .properties import HydrogenProperties, ThermoState


@dataclass(frozen=True)
class CommonPressureCellState:
    pressure_Pa: float
    masses_kg: tuple[float, ...]
    specific_enthalpies_J_kg: tuple[float, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.pressure_Pa) or self.pressure_Pa <= 0.0:
            raise ValueError("pressure_Pa must be finite and positive")
        if not self.masses_kg:
            raise ValueError("at least one fluid cell is required")
        if len(self.masses_kg) != len(self.specific_enthalpies_J_kg):
            raise ValueError("mass and enthalpy arrays must have equal length")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.masses_kg
        ):
            raise ValueError("all cell masses must be finite and positive")
        if any(
            not math.isfinite(value)
            for value in self.specific_enthalpies_J_kg
        ):
            raise ValueError("all cell enthalpies must be finite")


@dataclass(frozen=True)
class CommonPressureCellDerivative:
    pressure_rate_Pa_s: float
    mass_rates_kg_s: tuple[float, ...]
    enthalpy_rates_J_kg_s: tuple[float, ...]
    cell_volumes_m3: tuple[float, ...]
    volume_constraint_residual_m3: float
    differentiated_volume_residual_m3_s: float
    total_mass_rate_kg_s: float
    total_fluid_energy_rate_W: float
    imposed_enthalpy_inventory_rate_W: float
    energy_residual_W: float


class CommonPressureRigidVolume:
    """Evaluate a common-pressure cell DAE at one state.

    ``enthalpy_inventory_rates_W`` is the full rate of each cell's extensive
    enthalpy ``m*h`` before pressure coupling.  For a material stream it is
    ``m_dot*h_stream``; for heat it is ``Q_dot``; conservative internal
    exchanges appear with equal and opposite signs.  The model performs no
    parameter fitting and uses only EOS derivatives.
    """

    def __init__(
        self,
        total_volume_m3: float,
        properties: HydrogenProperties | None = None,
    ) -> None:
        if not math.isfinite(total_volume_m3) or total_volume_m3 <= 0.0:
            raise ValueError("total_volume_m3 must be finite and positive")
        self.total_volume_m3 = float(total_volume_m3)
        self.properties = properties or HydrogenProperties()

    @staticmethod
    def _vector(values: Sequence[float], count: int, name: str) -> np.ndarray:
        array = np.asarray(tuple(values), dtype=float)
        if array.shape != (count,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain {count} finite values")
        return array

    def thermo(
        self, state: CommonPressureCellState
    ) -> tuple[tuple[ThermoState, ...], np.ndarray]:
        cells = tuple(
            self.properties.from_ph(state.pressure_Pa, enthalpy)
            for enthalpy in state.specific_enthalpies_J_kg
        )
        quality_tolerance = 1.0e-10
        if any(
            cell.quality is not None
            and quality_tolerance < cell.quality < 1.0 - quality_tolerance
            for cell in cells
        ):
            raise ValueError(
                "CommonPressureRigidVolume requires single-phase cells; "
                "phase-boundary handling belongs to the tank closure"
            )
        masses = np.asarray(state.masses_kg, dtype=float)
        volumes = masses / np.asarray(
            [cell.density_kg_m3 for cell in cells], dtype=float
        )
        return cells, volumes

    def derivative(
        self,
        state: CommonPressureCellState,
        mass_rates_kg_s: Sequence[float],
        enthalpy_inventory_rates_W: Sequence[float],
    ) -> CommonPressureCellDerivative:
        cells, volumes = self.thermo(state)
        count = len(cells)
        masses = np.asarray(state.masses_kg, dtype=float)
        enthalpies = np.asarray(state.specific_enthalpies_J_kg, dtype=float)
        mass_rates = self._vector(mass_rates_kg_s, count, "mass_rates_kg_s")
        inventory_rates = self._vector(
            enthalpy_inventory_rates_W,
            count,
            "enthalpy_inventory_rates_W",
        )
        # m*dh/dt excluding the common pressure term.
        reduced_sources = inventory_rates - enthalpies * mass_rates
        specific_volumes = volumes / masses
        vp = np.empty(count)
        vh = np.empty(count)
        for index, cell in enumerate(cells):
            vp[index], vh[index] = (
                self.properties.specific_volume_derivatives_ph(cell)
            )
        denominator = float(np.sum(masses * vp + vh * volumes))
        if not math.isfinite(denominator) or abs(denominator) < 1.0e-20:
            raise ValueError("common-pressure volume equation is singular")
        pressure_rate = -float(
            np.sum(specific_volumes * mass_rates + vh * reduced_sources)
        ) / denominator
        enthalpy_rates = (
            reduced_sources + volumes * pressure_rate
        ) / masses

        differentiated_volume = float(np.sum(
            specific_volumes * mass_rates
            + masses * vp * pressure_rate
            + masses * vh * enthalpy_rates
        ))
        total_volume = float(np.sum(volumes))
        fluid_energy_rate = float(
            np.sum(masses * enthalpy_rates + enthalpies * mass_rates)
            - total_volume * pressure_rate
        )
        imposed_rate = float(np.sum(inventory_rates))
        return CommonPressureCellDerivative(
            pressure_rate_Pa_s=pressure_rate,
            mass_rates_kg_s=tuple(float(value) for value in mass_rates),
            enthalpy_rates_J_kg_s=tuple(
                float(value) for value in enthalpy_rates
            ),
            cell_volumes_m3=tuple(float(value) for value in volumes),
            volume_constraint_residual_m3=(
                total_volume - self.total_volume_m3
            ),
            differentiated_volume_residual_m3_s=differentiated_volume,
            total_mass_rate_kg_s=float(np.sum(mass_rates)),
            total_fluid_energy_rate_W=fluid_energy_rate,
            imposed_enthalpy_inventory_rate_W=imposed_rate,
            energy_residual_W=fluid_energy_rate - imposed_rate,
        )
