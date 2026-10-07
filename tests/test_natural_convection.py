import math

import pytest

from lh2dt import (
    churchill_chu_vertical_plate,
    daigle_horizontal_surface,
    daigle_vertical_wall_boundary_layer,
    daigle_vertical_wall_uniform_heat_flux,
    yang_west_2015_cryogenic_tank,
)


BASE = dict(
    density_kg_m3=2.0,
    viscosity_Pa_s=1.0e-5,
    thermal_conductivity_W_mK=0.1,
    isobaric_heat_capacity_J_kgK=1_000.0,
    expansion_coefficient_1_K=0.02,
    gravity_m_s2=9.80665,
)


def test_horizontal_correlation_matches_published_equations():
    result = daigle_horizontal_surface(
        delta_temperature_K=10.0,
        characteristic_length_m=0.5,
        **BASE,
    )
    viscosity = BASE["viscosity_Pa_s"] / BASE["density_kg_m3"]
    prandtl = (
        BASE["viscosity_Pa_s"] * BASE["isobaric_heat_capacity_J_kgK"]
        / BASE["thermal_conductivity_W_mK"]
    )
    grashof = (
        BASE["gravity_m_s2"] * BASE["expansion_coefficient_1_K"] * 10.0
        * 0.5**3 / viscosity**2
    )
    rayleigh = grashof * prandtl
    psi = (1.0 + (0.492 / prandtl) ** (9.0 / 16.0)) ** (-16.0 / 9.0)
    expected_nusselt = 0.15 * (rayleigh * psi) ** (1.0 / 3.0)

    assert result.regime == "turbulent"
    assert result.rayleigh_number == pytest.approx(rayleigh)
    assert result.prandtl_number == pytest.approx(prandtl)
    assert result.nusselt_number == pytest.approx(expected_nusselt)
    assert result.heat_transfer_coefficient_W_m2K == pytest.approx(
        expected_nusselt * BASE["thermal_conductivity_W_mK"] / 0.5
    )


def test_churchill_chu_vertical_plate_matches_published_equation():
    result = churchill_chu_vertical_plate(
        wall_minus_bulk_temperature_K=10.0,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        **BASE,
    )
    kinematic_viscosity = BASE["viscosity_Pa_s"] / BASE["density_kg_m3"]
    prandtl = (
        BASE["viscosity_Pa_s"] * BASE["isobaric_heat_capacity_J_kgK"]
        / BASE["thermal_conductivity_W_mK"]
    )
    grashof = (
        BASE["gravity_m_s2"] * BASE["expansion_coefficient_1_K"] * 10.0
        * 0.5**3 / kinematic_viscosity**2
    )
    rayleigh = grashof * prandtl
    nusselt = (
        0.825
        + 0.387 * rayleigh ** (1.0 / 6.0)
        / (1.0 + (0.492 / prandtl) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    ) ** 2

    assert result.grashof_number == pytest.approx(grashof)
    assert result.rayleigh_number == pytest.approx(rayleigh)
    assert result.prandtl_number == pytest.approx(prandtl)
    assert result.nusselt_number == pytest.approx(nusselt)
    assert result.heat_transfer_coefficient_W_m2K == pytest.approx(
        nusselt * BASE["thermal_conductivity_W_mK"] / 0.5
    )
    assert result.heat_flow_W == pytest.approx(
        result.heat_transfer_coefficient_W_m2K * 0.75 * 10.0
    )


def test_churchill_chu_heat_direction_reverses_and_zero_delta_has_zero_heat():
    hot = churchill_chu_vertical_plate(
        wall_minus_bulk_temperature_K=10.0,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        **BASE,
    )
    cold = churchill_chu_vertical_plate(
        wall_minus_bulk_temperature_K=-10.0,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        **BASE,
    )
    zero = churchill_chu_vertical_plate(
        wall_minus_bulk_temperature_K=0.0,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        **BASE,
    )

    assert cold.heat_flow_W == pytest.approx(-hot.heat_flow_W)
    assert cold.nusselt_number == pytest.approx(hot.nusselt_number)
    assert zero.rayleigh_number == 0.0
    assert zero.heat_flow_W == 0.0


def test_vertical_boundary_layer_matches_daigle_laminar_equations():
    result = daigle_vertical_wall_boundary_layer(
        wall_minus_bulk_temperature_K=10.0,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        wall_perimeter_m=1.5,
        **BASE,
    )
    nu = BASE["viscosity_Pa_s"] / BASE["density_kg_m3"]
    pr = BASE["viscosity_Pa_s"] * BASE["isobaric_heat_capacity_J_kgK"] / BASE[
        "thermal_conductivity_W_mK"
    ]
    gr = (
        BASE["gravity_m_s2"] * BASE["expansion_coefficient_1_K"] * 10.0
        * 0.5**3 / nu**2
    )
    ra = gr * pr
    psi = (1.0 + (0.492 / pr) ** (9.0 / 16.0)) ** (-16.0 / 9.0)
    nusselt = 0.68 + 0.503 * (ra * psi) ** 0.25
    velocity = 1.185 * nu / 0.5 * (
        gr / (1.0 + 0.494 * pr ** (2.0 / 3.0))
    ) ** 0.5
    thickness = 0.5 * 3.93 * ((0.952 + pr) / (gr * pr**2)) ** 0.25

    assert result.regime == "laminar"
    assert result.grashof_number == pytest.approx(gr)
    assert result.rayleigh_number == pytest.approx(ra)
    assert result.nusselt_number == pytest.approx(nusselt)
    assert result.velocity_m_s == pytest.approx(velocity)
    assert result.hydrodynamic_thickness_m == pytest.approx(thickness)
    assert result.thermal_thickness_m == pytest.approx(thickness / math.sqrt(pr))
    assert result.mass_flow_kg_s == pytest.approx(
        0.0833 * 1.5 * BASE["density_kg_m3"] * velocity * thickness
    )
    assert result.heat_flow_W == pytest.approx(
        nusselt * BASE["thermal_conductivity_W_mK"] / 0.5 * 0.75 * 10.0
    )


def test_vertical_boundary_layer_direction_reverses_without_changing_magnitude():
    hot = daigle_vertical_wall_boundary_layer(
        wall_minus_bulk_temperature_K=10.0,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        wall_perimeter_m=1.5,
        **BASE,
    )
    cold = daigle_vertical_wall_boundary_layer(
        wall_minus_bulk_temperature_K=-10.0,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        wall_perimeter_m=1.5,
        **BASE,
    )
    assert cold.heat_flow_W == pytest.approx(-hot.heat_flow_W)
    assert cold.velocity_m_s == pytest.approx(-hot.velocity_m_s)
    assert cold.mass_flow_kg_s == pytest.approx(-hot.mass_flow_kg_s)
    assert cold.rayleigh_number == pytest.approx(hot.rayleigh_number)


def test_uniform_heat_flux_boundary_layer_matches_appendix_a():
    result = daigle_vertical_wall_uniform_heat_flux(
        wall_heat_flux_W_m2=1.0,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        wall_perimeter_m=1.5,
        **BASE,
    )
    pr = BASE["viscosity_Pa_s"] * BASE["isobaric_heat_capacity_J_kgK"] / BASE[
        "thermal_conductivity_W_mK"
    ]
    psi = (1.0 + (0.492 / pr) ** (9.0 / 16.0)) ** (-16.0 / 9.0)
    ra_star = (
        BASE["gravity_m_s2"] * BASE["density_kg_m3"] ** 2
        * BASE["isobaric_heat_capacity_J_kgK"]
        * BASE["expansion_coefficient_1_K"] * 1.0 * 0.5**4
        / (
            BASE["viscosity_Pa_s"]
            * BASE["thermal_conductivity_W_mK"] ** 2
        )
    )
    nusselt = 0.631 * (ra_star * psi) ** 0.2
    rayleigh = ra_star / nusselt
    grashof = rayleigh / pr
    kinematic_viscosity = BASE["viscosity_Pa_s"] / BASE["density_kg_m3"]
    velocity = 1.185 * kinematic_viscosity / 0.5 * (
        grashof / (1.0 + 0.494 * pr ** (2.0 / 3.0))
    ) ** 0.5
    thickness = 0.5 * 3.93 * (
        (0.952 + pr) / (grashof * pr**2)
    ) ** 0.25

    assert result.regime == "laminar-uniform-flux"
    assert result.rayleigh_number == pytest.approx(rayleigh)
    assert result.grashof_number == pytest.approx(grashof)
    assert result.nusselt_number == pytest.approx(nusselt)
    assert result.heat_flow_W == pytest.approx(0.75)
    assert result.velocity_m_s == pytest.approx(velocity)
    assert result.hydrodynamic_thickness_m == pytest.approx(thickness)
    assert result.mass_flow_kg_s == pytest.approx(
        0.0833 * 1.5 * BASE["density_kg_m3"] * velocity * thickness
    )

    cold = daigle_vertical_wall_uniform_heat_flux(
        wall_heat_flux_W_m2=-1.0,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        wall_perimeter_m=1.5,
        **BASE,
    )
    assert cold.heat_flow_W == pytest.approx(-result.heat_flow_W)
    assert cold.mass_flow_kg_s == pytest.approx(-result.mass_flow_kg_s)


def test_correlations_report_low_range_and_reject_high_range():
    low = daigle_vertical_wall_boundary_layer(
        wall_minus_bulk_temperature_K=1.0e-6,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        wall_perimeter_m=1.5,
        **BASE,
    )
    assert low.regime == "outside-low"
    assert low.heat_flow_W is None
    assert low.mass_flow_kg_s is None

    with pytest.raises(ValueError, match="Ra < 1e11"):
        daigle_vertical_wall_boundary_layer(
            wall_minus_bulk_temperature_K=100.0,
            characteristic_height_m=2.0,
            exchange_area_m2=1.0,
            wall_perimeter_m=1.0,
            **BASE,
        )


def test_invalid_transport_inputs_are_rejected():
    with pytest.raises(ValueError, match="viscosity_Pa_s"):
        daigle_horizontal_surface(
            delta_temperature_K=5.0,
            characteristic_length_m=0.5,
            **{**BASE, "viscosity_Pa_s": 0.0},
        )


def test_yang_west_liquid_tank_piecewise_equations_and_direction():
    cases = (
        (1.0e-3, 0.5, 0.642, 1.0 / 6.0, "liquid-low"),
        (10.0, 0.5, 0.167, 0.25, "liquid-intermediate"),
        (100.0, 2.0, 0.00053, 0.5, "liquid-high"),
    )
    for delta, height, coefficient, exponent, regime in cases:
        result = yang_west_2015_cryogenic_tank(
            phase="liquid",
            wall_minus_bulk_temperature_K=delta,
            characteristic_height_m=height,
            exchange_area_m2=0.75,
            **BASE,
        )
        assert result.regime == regime
        assert result.nusselt_number == pytest.approx(
            coefficient * result.rayleigh_number**exponent
        )
        assert result.heat_flow_W == pytest.approx(
            result.heat_transfer_coefficient_W_m2K * 0.75 * delta
        )
    cold = yang_west_2015_cryogenic_tank(
        phase="liquid",
        wall_minus_bulk_temperature_K=-10.0,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        **BASE,
    )
    hot = yang_west_2015_cryogenic_tank(
        phase="liquid",
        wall_minus_bulk_temperature_K=10.0,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        **BASE,
    )
    assert cold.heat_flow_W == pytest.approx(-hot.heat_flow_W)


def test_yang_west_ullage_equations_and_ranges():
    low = yang_west_2015_cryogenic_tank(
        phase="vapor",
        wall_minus_bulk_temperature_K=1.0e-3,
        characteristic_height_m=0.5,
        exchange_area_m2=0.75,
        **BASE,
    )
    high = yang_west_2015_cryogenic_tank(
        phase="vapor",
        wall_minus_bulk_temperature_K=100.0,
        characteristic_height_m=2.0,
        exchange_area_m2=0.75,
        **BASE,
    )
    assert low.regime == "vapor-low"
    assert low.nusselt_number == 4.5
    assert high.regime == "vapor-high"
    assert high.nusselt_number == pytest.approx(
        0.08 * high.rayleigh_number**0.25
    )
    with pytest.raises(ValueError, match="phase"):
        yang_west_2015_cryogenic_tank(
            phase="two-phase",
            wall_minus_bulk_temperature_K=1.0,
            characteristic_height_m=0.5,
            exchange_area_m2=0.75,
            **BASE,
        )
    with pytest.raises(ValueError, match="Ra < 1e12"):
        yang_west_2015_cryogenic_tank(
            phase="vapor",
            wall_minus_bulk_temperature_K=100.0,
            characteristic_height_m=3.0,
            exchange_area_m2=0.75,
            **BASE,
        )
    with pytest.raises(ValueError, match="Ra <= 5e13"):
        yang_west_2015_cryogenic_tank(
            phase="liquid",
            wall_minus_bulk_temperature_K=100.0,
            characteristic_height_m=10.0,
            exchange_area_m2=0.75,
            **BASE,
        )
