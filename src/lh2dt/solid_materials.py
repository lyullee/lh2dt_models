"""Traceable solid-material closures for cryogenic vessel inventories.

The correlations in this module are forward property evaluations.  They do
not contain facility-data fitting or inverse estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .geometry import HorizontalVesselGeometry


NIST_316_SOURCE_URL = (
    "https://trc.nist.gov/cryogenics/materials/316Stainless/316Stainless_rev.htm"
)

# NIST cryogenic-material database: log10(y) polynomial in log10(T).
_NIST_316_SPECIFIC_HEAT_4_TO_50_K = (
    12.2486,
    -80.6422,
    218.743,
    -308.854,
    239.5296,
    -89.9982,
    3.15315,
    8.44996,
    -1.91368,
)
_NIST_316_SPECIFIC_HEAT_50_TO_300_K = (
    -1879.464,
    3643.198,
    76.70125,
    -6176.028,
    7437.6247,
    -4305.7217,
    1382.4627,
    -237.22704,
    17.05262,
)

_NIST_316_THERMAL_CONDUCTIVITY = (
    -1.4087,
    1.3982,
    0.2543,
    -0.6260,
    0.2334,
    0.4256,
    -0.4658,
    0.1650,
    -0.0199,
)


def nist_316_thermal_conductivity_W_mK(temperature_K: float) -> float:
    """Return NIST's 316 stainless thermal conductivity correlation.

    NIST publishes the same log-polynomial form used for the specific heat,
    with a 4--300 K data range and approximately two-percent curve-fit error.
    The result is used only for explicit material-conduction boundaries; it is
    never inferred from a facility time series.
    """

    temperature = float(temperature_K)
    if not math.isfinite(temperature) or not 4.0 <= temperature <= 300.0:
        raise ValueError("temperature_K must lie within the NIST 4--300 K range")
    log_temperature = math.log10(temperature)
    exponent = sum(
        coefficient * log_temperature**power
        for power, coefficient in enumerate(_NIST_316_THERMAL_CONDUCTIVITY)
    )
    return 10.0**exponent


def nist_316_specific_heat_J_kgK(temperature_K: float) -> float:
    """Return NIST's fitted specific heat for 316 stainless steel.

    NIST reports a 4--300 K equation range and about two-percent curve-fit
    error.  The 316 correlation is used as the public-property surrogate for
    an SA-240 Type 316L inner vessel; that grade substitution must remain
    visible in evidence records.
    """

    temperature = float(temperature_K)
    if not math.isfinite(temperature) or not 4.0 <= temperature <= 300.0:
        raise ValueError("temperature_K must lie within the NIST 4--300 K range")
    coefficients = (
        _NIST_316_SPECIFIC_HEAT_4_TO_50_K
        if temperature <= 50.0
        else _NIST_316_SPECIFIC_HEAT_50_TO_300_K
    )
    log_temperature = math.log10(temperature)
    exponent = sum(
        coefficient * log_temperature**power
        for power, coefficient in enumerate(coefficients)
    )
    return 10.0**exponent


@dataclass(frozen=True)
class ThinWallThermalInventory:
    cylinder_area_m2: float
    two_head_area_m2: float
    metal_volume_m3: float
    metal_mass_kg: float
    specific_heat_J_kgK: float
    heat_capacity_J_K: float
    evaluation_temperature_K: float
    density_kg_m3: float
    shell_thickness_m: float
    head_thickness_m: float


def horizontal_vessel_316_wall_inventory(
    geometry: HorizontalVesselGeometry,
    *,
    shell_thickness_m: float,
    head_thickness_m: float,
    evaluation_temperature_K: float,
    density_kg_m3: float = 7_950.0,
) -> ThinWallThermalInventory:
    """Build the inner-vessel metal mass and local heat capacity.

    The thin-wall approximation multiplies the cylindrical and two-head
    middle-surface areas by their documented thicknesses.  Attachments,
    nozzles and supports are excluded unless separately represented.
    """

    shell_thickness = float(shell_thickness_m)
    head_thickness = float(head_thickness_m)
    density = float(density_kg_m3)
    for name, value in (
        ("shell_thickness_m", shell_thickness),
        ("head_thickness_m", head_thickness),
        ("density_kg_m3", density),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    cylinder_area = math.pi * geometry.diameter * geometry.straight_length
    two_head_area = geometry.inner_surface_area_m2 - cylinder_area
    metal_volume = (
        cylinder_area * shell_thickness + two_head_area * head_thickness
    )
    metal_mass = density * metal_volume
    specific_heat = nist_316_specific_heat_J_kgK(evaluation_temperature_K)
    return ThinWallThermalInventory(
        cylinder_area_m2=cylinder_area,
        two_head_area_m2=two_head_area,
        metal_volume_m3=metal_volume,
        metal_mass_kg=metal_mass,
        specific_heat_J_kgK=specific_heat,
        heat_capacity_J_K=metal_mass * specific_heat,
        evaluation_temperature_K=float(evaluation_temperature_K),
        density_kg_m3=density,
        shell_thickness_m=shell_thickness,
        head_thickness_m=head_thickness,
    )


def horizontal_vessel_316_wall_axial_conductance_W_K(
    geometry: HorizontalVesselGeometry,
    *,
    shell_thickness_m: float,
    evaluation_temperature_K: float,
    wall_node_count: int,
) -> float:
    """Return adjacent-node axial conductance for the cylindrical inner shell.

    The shell cross-section for axial conduction is ``pi*D*t`` and the
    center-to-center distance is the straight length divided by
    ``wall_node_count - 1``.  Heads are excluded from this one-dimensional
    axial bridge; their heat capacity and fluid overlap remain represented by
    :func:`horizontal_vessel_316_wall_inventory`.
    """

    thickness = float(shell_thickness_m)
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("shell_thickness_m must be finite and positive")
    if not isinstance(wall_node_count, int) or wall_node_count < 2:
        raise ValueError("wall_node_count must be an integer of at least two")
    if geometry.straight_length <= 0.0:
        raise ValueError("geometry straight length must be positive")
    distance = geometry.straight_length / (wall_node_count - 1)
    area = math.pi * geometry.diameter * thickness
    conductance = (
        nist_316_thermal_conductivity_W_mK(evaluation_temperature_K)
        * area
        / distance
    )
    if not math.isfinite(conductance) or conductance <= 0.0:
        raise ValueError("computed wall axial conductance must be finite and positive")
    return conductance
