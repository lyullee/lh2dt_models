"""Published natural-convection closures used by layered cryogenic tanks.

The functions in this module are algebraic forward evaluations.  They expose
their validity range and contain no coefficient fitting or facility-data
calibration.  Signs describe the physical direction of heat or boundary-layer
flow; Rayleigh and Nusselt numbers are reported as positive magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HorizontalNaturalConvection:
    rayleigh_number: float
    prandtl_number: float
    psi: float
    nusselt_number: float | None
    heat_transfer_coefficient_W_m2K: float | None
    regime: str


@dataclass(frozen=True)
class VerticalBoundaryLayer:
    grashof_number: float
    rayleigh_number: float
    prandtl_number: float
    psi: float
    nusselt_number: float | None
    heat_transfer_coefficient_W_m2K: float | None
    heat_flow_W: float | None
    velocity_m_s: float | None
    hydrodynamic_thickness_m: float | None
    thermal_thickness_m: float | None
    mass_flow_kg_s: float | None
    regime: str


@dataclass(frozen=True)
class VerticalPlateHeatTransfer:
    """Mean vertical-plate natural-convection heat-transfer result."""

    grashof_number: float
    rayleigh_number: float
    prandtl_number: float
    nusselt_number: float
    heat_transfer_coefficient_W_m2K: float
    heat_flow_W: float


@dataclass(frozen=True)
class CryogenicTankHeatTransfer:
    """Tank-geometry natural-convection correlation result."""

    phase: str
    grashof_number: float
    rayleigh_number: float
    prandtl_number: float
    nusselt_number: float
    heat_transfer_coefficient_W_m2K: float
    heat_flow_W: float
    regime: str


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _dimensionless_groups(
    *,
    delta_temperature_K: float,
    characteristic_length_m: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
    thermal_conductivity_W_mK: float,
    isobaric_heat_capacity_J_kgK: float,
    expansion_coefficient_1_K: float,
    gravity_m_s2: float,
) -> tuple[float, float, float, float]:
    length = _positive(characteristic_length_m, "characteristic_length_m")
    density = _positive(density_kg_m3, "density_kg_m3")
    viscosity = _positive(viscosity_Pa_s, "viscosity_Pa_s")
    conductivity = _positive(
        thermal_conductivity_W_mK, "thermal_conductivity_W_mK"
    )
    heat_capacity = _positive(
        isobaric_heat_capacity_J_kgK, "isobaric_heat_capacity_J_kgK"
    )
    expansion = _positive(expansion_coefficient_1_K, "expansion_coefficient_1_K")
    gravity = _positive(gravity_m_s2, "gravity_m_s2")
    delta_temperature = abs(_finite(delta_temperature_K, "delta_temperature_K"))
    kinematic_viscosity = viscosity / density
    prandtl = viscosity * heat_capacity / conductivity
    grashof = (
        gravity * expansion * delta_temperature * length**3
        / kinematic_viscosity**2
    )
    rayleigh = grashof * prandtl
    psi = (1.0 + (0.492 / prandtl) ** (9.0 / 16.0)) ** (-16.0 / 9.0)
    return grashof, rayleigh, prandtl, psi


def churchill_chu_vertical_plate(
    *,
    wall_minus_bulk_temperature_K: float,
    characteristic_height_m: float,
    exchange_area_m2: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
    thermal_conductivity_W_mK: float,
    isobaric_heat_capacity_J_kgK: float,
    expansion_coefficient_1_K: float,
    gravity_m_s2: float = 9.80665,
) -> VerticalPlateHeatTransfer:
    """Evaluate the Churchill--Chu mean vertical-plate correlation.

    The correlation supplies a wall-to-bulk heat-transfer coefficient only;
    it does not infer a circulation mass flow.  Positive heat denotes transfer
    from a warmer wall to the fluid.  Fluid properties are supplied explicitly
    so callers can choose and record their property-evaluation convention.
    """

    signed_delta = _finite(
        wall_minus_bulk_temperature_K, "wall_minus_bulk_temperature_K"
    )
    height = _positive(characteristic_height_m, "characteristic_height_m")
    area = _positive(exchange_area_m2, "exchange_area_m2")
    conductivity = _positive(
        thermal_conductivity_W_mK, "thermal_conductivity_W_mK"
    )
    grashof, rayleigh, prandtl, _ = _dimensionless_groups(
        delta_temperature_K=signed_delta,
        characteristic_length_m=height,
        density_kg_m3=density_kg_m3,
        viscosity_Pa_s=viscosity_Pa_s,
        thermal_conductivity_W_mK=conductivity,
        isobaric_heat_capacity_J_kgK=isobaric_heat_capacity_J_kgK,
        expansion_coefficient_1_K=expansion_coefficient_1_K,
        gravity_m_s2=gravity_m_s2,
    )
    nusselt = (
        0.825
        + 0.387 * rayleigh ** (1.0 / 6.0)
        / (1.0 + (0.492 / prandtl) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    ) ** 2
    coefficient = nusselt * conductivity / height
    return VerticalPlateHeatTransfer(
        grashof_number=grashof,
        rayleigh_number=rayleigh,
        prandtl_number=prandtl,
        nusselt_number=nusselt,
        heat_transfer_coefficient_W_m2K=coefficient,
        heat_flow_W=coefficient * area * signed_delta,
    )


def yang_west_2015_cryogenic_tank(
    *,
    phase: str,
    wall_minus_bulk_temperature_K: float,
    characteristic_height_m: float,
    exchange_area_m2: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
    thermal_conductivity_W_mK: float,
    isobaric_heat_capacity_J_kgK: float,
    expansion_coefficient_1_K: float,
    gravity_m_s2: float = 9.80665,
) -> CryogenicTankHeatTransfer:
    """Evaluate Yang and West's 2015 cryogenic-tank wall correlation.

    NASA/AIAA 2015-3856 extracted separate liquid and ullage correlations
    from CFD of cylindrical tanks with domed ends.  The characteristic length
    is supplied explicitly because the paper defines it from the wetted phase
    height.  The result is a wall-to-bulk heat-transfer coefficient only and
    must not be interpreted as a circulation mass-flow model.

    The published ranges are ``Ra <= 5e13`` for liquid and ``Ra < 1e12`` for
    ullage gas.  Values outside those ranges are rejected instead of
    extrapolated.  The source does not unambiguously identify the simulated
    cryogen, so callers must retain that limitation in validation records.
    """

    if phase not in {"liquid", "vapor"}:
        raise ValueError("phase must be 'liquid' or 'vapor'")
    signed_delta = _finite(
        wall_minus_bulk_temperature_K, "wall_minus_bulk_temperature_K"
    )
    height = _positive(characteristic_height_m, "characteristic_height_m")
    area = _positive(exchange_area_m2, "exchange_area_m2")
    conductivity = _positive(
        thermal_conductivity_W_mK, "thermal_conductivity_W_mK"
    )
    grashof, rayleigh, prandtl, _ = _dimensionless_groups(
        delta_temperature_K=signed_delta,
        characteristic_length_m=height,
        density_kg_m3=density_kg_m3,
        viscosity_Pa_s=viscosity_Pa_s,
        thermal_conductivity_W_mK=conductivity,
        isobaric_heat_capacity_J_kgK=isobaric_heat_capacity_J_kgK,
        expansion_coefficient_1_K=expansion_coefficient_1_K,
        gravity_m_s2=gravity_m_s2,
    )
    if phase == "liquid":
        if rayleigh <= 1.0e7:
            nusselt = 0.642 * rayleigh ** (1.0 / 6.0)
            regime = "liquid-low"
        elif rayleigh <= 1.0e10:
            nusselt = 0.167 * rayleigh ** 0.25
            regime = "liquid-intermediate"
        elif rayleigh <= 5.0e13:
            nusselt = 0.00053 * math.sqrt(rayleigh)
            regime = "liquid-high"
        else:
            raise ValueError(
                "Yang-West liquid tank correlation requires Ra <= 5e13"
            )
    else:
        if rayleigh <= 1.0e7:
            nusselt = 4.5
            regime = "vapor-low"
        elif rayleigh < 1.0e12:
            nusselt = 0.08 * rayleigh ** 0.25
            regime = "vapor-high"
        else:
            raise ValueError(
                "Yang-West ullage tank correlation requires Ra < 1e12"
            )
    coefficient = nusselt * conductivity / height
    return CryogenicTankHeatTransfer(
        phase=phase,
        grashof_number=grashof,
        rayleigh_number=rayleigh,
        prandtl_number=prandtl,
        nusselt_number=nusselt,
        heat_transfer_coefficient_W_m2K=coefficient,
        heat_flow_W=coefficient * area * signed_delta,
        regime=regime,
    )


def daigle_horizontal_surface(
    *,
    delta_temperature_K: float,
    characteristic_length_m: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
    thermal_conductivity_W_mK: float,
    isobaric_heat_capacity_J_kgK: float,
    expansion_coefficient_1_K: float,
    gravity_m_s2: float = 9.80665,
) -> HorizontalNaturalConvection:
    """Evaluate Daigle et al. (2013) Eqs. 10 and 20--22.

    The caller must separately determine whether the surface orientation is
    buoyantly unstable.  Values below ``Ra=1e4`` return ``regime='conduction'``;
    values above ``Ra=1e11`` are rejected instead of extrapolated.
    """

    _, rayleigh, prandtl, psi = _dimensionless_groups(
        delta_temperature_K=delta_temperature_K,
        characteristic_length_m=characteristic_length_m,
        density_kg_m3=density_kg_m3,
        viscosity_Pa_s=viscosity_Pa_s,
        thermal_conductivity_W_mK=thermal_conductivity_W_mK,
        isobaric_heat_capacity_J_kgK=isobaric_heat_capacity_J_kgK,
        expansion_coefficient_1_K=expansion_coefficient_1_K,
        gravity_m_s2=gravity_m_s2,
    )
    if rayleigh < 1.0e4:
        return HorizontalNaturalConvection(
            rayleigh, prandtl, psi, None, None, "conduction"
        )
    if rayleigh < 1.0e7:
        nusselt = 0.54 * (rayleigh * psi) ** 0.25
        regime = "laminar"
    elif rayleigh <= 1.0e11:
        nusselt = 0.15 * (rayleigh * psi) ** (1.0 / 3.0)
        regime = "turbulent"
    else:
        raise ValueError("Daigle horizontal-surface correlation requires Ra <= 1e11")
    coefficient = (
        nusselt * _positive(
            thermal_conductivity_W_mK, "thermal_conductivity_W_mK"
        ) / _positive(characteristic_length_m, "characteristic_length_m")
    )
    return HorizontalNaturalConvection(
        rayleigh, prandtl, psi, nusselt, coefficient, regime
    )


def daigle_vertical_wall_boundary_layer(
    *,
    wall_minus_bulk_temperature_K: float,
    characteristic_height_m: float,
    exchange_area_m2: float,
    wall_perimeter_m: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
    thermal_conductivity_W_mK: float,
    isobaric_heat_capacity_J_kgK: float,
    expansion_coefficient_1_K: float,
    gravity_m_s2: float = 9.80665,
) -> VerticalBoundaryLayer:
    """Evaluate Daigle et al. (2013) Eqs. 9--16 for a vertical wall.

    Positive heat and mass flow denote a hot-wall boundary layer moving up;
    negative values denote a cold-wall layer moving down.  The mass-flow rate
    is the correlation estimate through a boundary layer of the supplied wall
    perimeter.  Below ``Ra=1e5`` the paper's correlation is not applied and
    optional outputs are ``None``.  ``Ra>=1e11`` is rejected.
    """

    signed_delta = _finite(
        wall_minus_bulk_temperature_K, "wall_minus_bulk_temperature_K"
    )
    height = _positive(characteristic_height_m, "characteristic_height_m")
    area = _positive(exchange_area_m2, "exchange_area_m2")
    perimeter = _positive(wall_perimeter_m, "wall_perimeter_m")
    density = _positive(density_kg_m3, "density_kg_m3")
    viscosity = _positive(viscosity_Pa_s, "viscosity_Pa_s")
    conductivity = _positive(
        thermal_conductivity_W_mK, "thermal_conductivity_W_mK"
    )
    grashof, rayleigh, prandtl, psi = _dimensionless_groups(
        delta_temperature_K=signed_delta,
        characteristic_length_m=height,
        density_kg_m3=density,
        viscosity_Pa_s=viscosity,
        thermal_conductivity_W_mK=conductivity,
        isobaric_heat_capacity_J_kgK=isobaric_heat_capacity_J_kgK,
        expansion_coefficient_1_K=expansion_coefficient_1_K,
        gravity_m_s2=gravity_m_s2,
    )
    if signed_delta == 0.0 or rayleigh < 1.0e5:
        return VerticalBoundaryLayer(
            grashof, rayleigh, prandtl, psi,
            None, None, None, None, None, None, None, "outside-low",
        )
    if rayleigh < 1.0e9:
        nusselt = 0.68 + 0.503 * (rayleigh * psi) ** 0.25
        regime = "laminar"
        mass_coefficient = 0.0833
        thickness = height * 3.93 * (
            (0.952 + prandtl) / (grashof * prandtl**2)
        ) ** 0.25
    elif rayleigh < 1.0e11:
        nusselt = 0.15 * (rayleigh * psi) ** (1.0 / 3.0)
        regime = "turbulent"
        mass_coefficient = 0.1436
        thickness = (
            height * 0.565
            * ((1.0 + 0.494 * prandtl ** (2.0 / 3.0)) / grashof) ** 0.1
            / prandtl ** (8.0 / 15.0)
        )
    else:
        raise ValueError("Daigle vertical-wall correlation requires Ra < 1e11")

    coefficient = nusselt * conductivity / height
    direction = math.copysign(1.0, signed_delta)
    kinematic_viscosity = viscosity / density
    velocity_magnitude = (
        1.185 * kinematic_viscosity / height
        * (grashof / (1.0 + 0.494 * prandtl ** (2.0 / 3.0))) ** 0.5
    )
    thermal_thickness = thickness / math.sqrt(prandtl)
    mass_flow_magnitude = (
        mass_coefficient * perimeter * density * velocity_magnitude * thickness
    )
    return VerticalBoundaryLayer(
        grashof_number=grashof,
        rayleigh_number=rayleigh,
        prandtl_number=prandtl,
        psi=psi,
        nusselt_number=nusselt,
        heat_transfer_coefficient_W_m2K=coefficient,
        heat_flow_W=coefficient * area * signed_delta,
        velocity_m_s=direction * velocity_magnitude,
        hydrodynamic_thickness_m=thickness,
        thermal_thickness_m=thermal_thickness,
        mass_flow_kg_s=direction * mass_flow_magnitude,
        regime=regime,
    )


def daigle_vertical_wall_uniform_heat_flux(
    *,
    wall_heat_flux_W_m2: float,
    characteristic_height_m: float,
    exchange_area_m2: float,
    wall_perimeter_m: float,
    density_kg_m3: float,
    viscosity_Pa_s: float,
    thermal_conductivity_W_mK: float,
    isobaric_heat_capacity_J_kgK: float,
    expansion_coefficient_1_K: float,
    gravity_m_s2: float = 9.80665,
) -> VerticalBoundaryLayer:
    """Evaluate Daigle et al. (2013) Appendix A for uniform wall flux.

    Appendix A replaces the temperature-difference Rayleigh number by
    ``Ra* = g rho^2 cp beta |q''| x^4 / (mu k^2)``.  Equations (A1),
    (A2), and (A4) recover the equivalent wall temperature difference;
    Eqs. (13)--(16) then provide the boundary-layer mass flow.  Positive
    heat flux denotes heating and upward wall flow.  Negative heat flux
    denotes cooling and downward wall flow.
    """

    signed_flux = _finite(wall_heat_flux_W_m2, "wall_heat_flux_W_m2")
    height = _positive(characteristic_height_m, "characteristic_height_m")
    area = _positive(exchange_area_m2, "exchange_area_m2")
    perimeter = _positive(wall_perimeter_m, "wall_perimeter_m")
    density = _positive(density_kg_m3, "density_kg_m3")
    viscosity = _positive(viscosity_Pa_s, "viscosity_Pa_s")
    conductivity = _positive(
        thermal_conductivity_W_mK, "thermal_conductivity_W_mK"
    )
    heat_capacity = _positive(
        isobaric_heat_capacity_J_kgK, "isobaric_heat_capacity_J_kgK"
    )
    expansion = _positive(expansion_coefficient_1_K, "expansion_coefficient_1_K")
    gravity = _positive(gravity_m_s2, "gravity_m_s2")
    prandtl = viscosity * heat_capacity / conductivity
    psi = (1.0 + (0.492 / prandtl) ** (9.0 / 16.0)) ** (-16.0 / 9.0)
    modified_rayleigh = (
        gravity * density**2 * heat_capacity * expansion
        * abs(signed_flux) * height**4 / (viscosity * conductivity**2)
    )
    if signed_flux == 0.0 or modified_rayleigh < 1.0e5:
        return VerticalBoundaryLayer(
            0.0, 0.0, prandtl, psi,
            None, None, None, None, None, None, None, "outside-low",
        )
    if modified_rayleigh < 1.0e9:
        nusselt = 0.631 * (modified_rayleigh * psi) ** 0.2
        regime = "laminar-uniform-flux"
        mass_coefficient = 0.0833
    elif modified_rayleigh < 1.0e11:
        nusselt = 0.241 * (modified_rayleigh * psi) ** 0.25
        regime = "turbulent-uniform-flux"
        mass_coefficient = 0.1436
    else:
        raise ValueError(
            "Daigle uniform-flux vertical-wall correlation requires Ra* < 1e11; "
            f"calculated Ra*={modified_rayleigh:.6g}"
        )

    coefficient = nusselt * conductivity / height
    equivalent_delta = signed_flux / coefficient
    rayleigh = modified_rayleigh / nusselt
    grashof = rayleigh / prandtl
    kinematic_viscosity = viscosity / density
    velocity_magnitude = (
        1.185 * kinematic_viscosity / height
        * (grashof / (1.0 + 0.494 * prandtl ** (2.0 / 3.0))) ** 0.5
    )
    if modified_rayleigh < 1.0e9:
        thickness = height * 3.93 * (
            (0.952 + prandtl) / (grashof * prandtl**2)
        ) ** 0.25
    else:
        thickness = (
            height * 0.565
            * ((1.0 + 0.494 * prandtl ** (2.0 / 3.0)) / grashof) ** 0.1
            / prandtl ** (8.0 / 15.0)
        )
    thermal_thickness = thickness / math.sqrt(prandtl)
    mass_flow_magnitude = (
        mass_coefficient * perimeter * density * velocity_magnitude * thickness
    )
    direction = math.copysign(1.0, equivalent_delta)
    return VerticalBoundaryLayer(
        grashof_number=grashof,
        rayleigh_number=rayleigh,
        prandtl_number=prandtl,
        psi=psi,
        nusselt_number=nusselt,
        heat_transfer_coefficient_W_m2K=coefficient,
        heat_flow_W=signed_flux * area,
        velocity_m_s=direction * velocity_magnitude,
        hydrodynamic_thickness_m=thickness,
        thermal_thickness_m=thermal_thickness,
        mass_flow_kg_s=direction * mass_flow_magnitude,
        regime=regime,
    )
