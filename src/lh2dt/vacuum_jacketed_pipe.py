"""Geometry and thermal boundary for a vacuum-jacketed process pipe.

The dynamic pipe balance and the vacuum-insulation heat-leak balance are
deliberately separate models.  This module supplies the explicit geometry
that connects them without introducing a fitted overall ``UA``.  A caller
must still declare the construction conductances, vacuum pressure and wall
heat capacity; no time series is used to infer any of them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .vacuum_insulation import (
    LocalizedHeatPath,
    VacuumInsulation,
    VacuumInsulationParameters,
)


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


@dataclass(frozen=True)
class VacuumJacketedPipeEnvelope:
    """Explicit coaxial geometry for a vacuum-jacketed cryogenic pipe.

    ``process_inner_diameter_m`` is the fluid bore, ``process_outer_diameter_m``
    is the outside of the process tube, and ``jacket_inner_diameter_m`` is the
    inside of the vacuum jacket.  Keeping all three diameters explicit avoids
    inferring schedule thickness or vacuum gap from a nominal pipe size.
    """

    length_m: float
    process_inner_diameter_m: float
    process_outer_diameter_m: float
    jacket_inner_diameter_m: float

    def __post_init__(self) -> None:
        length = _positive(self.length_m, "length_m")
        inner = _positive(self.process_inner_diameter_m, "process_inner_diameter_m")
        process_outer = _positive(
            self.process_outer_diameter_m, "process_outer_diameter_m"
        )
        jacket_inner = _positive(
            self.jacket_inner_diameter_m, "jacket_inner_diameter_m"
        )
        if process_outer <= inner:
            raise ValueError("process_outer_diameter_m must exceed process_inner_diameter_m")
        if jacket_inner <= process_outer:
            raise ValueError("jacket_inner_diameter_m must exceed process_outer_diameter_m")
        object.__setattr__(self, "length_m", length)
        object.__setattr__(self, "process_inner_diameter_m", inner)
        object.__setattr__(self, "process_outer_diameter_m", process_outer)
        object.__setattr__(self, "jacket_inner_diameter_m", jacket_inner)

    @property
    def fluid_volume_m3(self) -> float:
        """Process-fluid geometric volume before any declared dead volume."""

        return math.pi * self.process_inner_diameter_m**2 * self.length_m / 4.0

    @property
    def process_wall_volume_m3(self) -> float:
        """Process-tube material volume from the declared diameters."""

        return (
            math.pi
            * (self.process_outer_diameter_m**2 - self.process_inner_diameter_m**2)
            * self.length_m
            / 4.0
        )

    @property
    def process_inner_surface_area_m2(self) -> float:
        return math.pi * self.process_inner_diameter_m * self.length_m

    @property
    def jacket_inner_surface_area_m2(self) -> float:
        return math.pi * self.jacket_inner_diameter_m * self.length_m

    @property
    def vacuum_annulus_area_m2(self) -> float:
        """Cross-sectional area available to the vacuum annulus."""

        return (
            math.pi
            * (self.jacket_inner_diameter_m**2 - self.process_outer_diameter_m**2)
            / 4.0
        )

    @property
    def radial_vacuum_gap_m(self) -> float:
        return (self.jacket_inner_diameter_m - self.process_outer_diameter_m) / 2.0

    def make_insulation(
        self,
        *,
        effective_emissivity: float,
        mli_solid_conductance_W_K: float,
        support_conductance_W_K: float,
        residual_gas_coefficient_W_m2K_Pa: float,
        maximum_valid_vacuum_pressure_Pa: float,
        localized_heat_paths: tuple[LocalizedHeatPath, ...] = (),
    ) -> VacuumInsulation:
        """Create a path-decomposed heat-leak boundary over the jacket area."""

        return VacuumInsulation(
            VacuumInsulationParameters(
                area_m2=self.jacket_inner_surface_area_m2,
                effective_emissivity=effective_emissivity,
                mli_solid_conductance_W_K=mli_solid_conductance_W_K,
                support_conductance_W_K=support_conductance_W_K,
                residual_gas_coefficient_W_m2K_Pa=residual_gas_coefficient_W_m2K_Pa,
                maximum_valid_vacuum_pressure_Pa=maximum_valid_vacuum_pressure_Pa,
                localized_heat_paths=localized_heat_paths,
            )
        )

    def dynamic_pipe_geometry(self) -> dict[str, float]:
        """Return only geometry needed to construct a ``DynamicHEMPipe``.

        Wall heat capacity and both thermal conductances intentionally remain
        absent: they are construction inputs and must be supplied explicitly.
        """

        return {
            "length_m": self.length_m,
            "inner_diameter_m": self.process_inner_diameter_m,
            "fluid_volume_m3": self.fluid_volume_m3,
        }


__all__ = ["VacuumJacketedPipeEnvelope"]
