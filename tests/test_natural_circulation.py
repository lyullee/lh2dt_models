import math

import pytest

from lh2dt import (
    CirculationLegGeometry,
    HydrogenProperties,
    NaturalCirculationLoop,
)


PROPS = HydrogenProperties()


def make_loop(diameter_m: float = 0.02) -> NaturalCirculationLoop:
    area = math.pi * diameter_m**2 / 4.0
    leg = CirculationLegGeometry(
        length_m=2.0,
        hydraulic_diameter_m=diameter_m,
        flow_area_m2=area,
        roughness_m=1.0e-6,
        local_loss_coefficient=1.0,
    )
    return NaturalCirculationLoop(1.0, leg, leg, properties=PROPS)


def test_density_head_closes_momentum_and_conserves_mass_and_energy():
    rising = PROPS.from_pT(500_000.0, 21.0)
    descending = PROPS.from_pT(500_000.0, 20.0)
    result = make_loop().evaluate(rising, descending)

    expected_head = 9.80665 * (
        descending.density_kg_m3 - rising.density_kg_m3
    )
    assert result.mass_flow_kg_s > 0.0
    assert result.buoyancy_pressure_Pa == pytest.approx(expected_head)
    assert result.hydraulic_residual_Pa == pytest.approx(0.0, abs=2.0e-9)
    assert result.mass_residual_kg_s == 0.0
    assert result.energy_residual_W == 0.0
    assert result.upper_control_volume_heat_W == pytest.approx(
        -result.lower_control_volume_heat_W
    )


def test_circulation_direction_reverses_and_is_zero_without_density_difference():
    warm = PROPS.from_pT(500_000.0, 21.0)
    cold = PROPS.from_pT(500_000.0, 20.0)
    loop = make_loop()

    forward = loop.evaluate(warm, cold)
    reverse = loop.evaluate(cold, warm)
    still = loop.evaluate(cold, cold)

    assert forward.mass_flow_kg_s > 0.0
    assert reverse.mass_flow_kg_s < 0.0
    assert reverse.hydraulic_residual_Pa == pytest.approx(0.0, abs=2.0e-9)
    assert still.mass_flow_kg_s == 0.0
    assert still.buoyancy_pressure_Pa == 0.0


def test_laminar_loop_recovers_two_leg_hagen_poiseuille_limit():
    diameter = 0.005
    area = math.pi * diameter**2 / 4.0
    leg = CirculationLegGeometry(2.0, diameter, area)
    loop = NaturalCirculationLoop(1.0, leg, leg, properties=PROPS)
    rising = PROPS.from_pT(500_000.0, 20.0001)
    descending = PROPS.from_pT(500_000.0, 20.0)

    result = loop.evaluate(rising, descending)
    mu_rising = PROPS.viscosity_Pa_s(rising)
    mu_descending = PROPS.viscosity_Pa_s(descending)
    linear_resistance = 32.0 * (
        mu_rising * leg.length_m
        / (rising.density_kg_m3 * area * diameter**2)
        + mu_descending * leg.length_m
        / (descending.density_kg_m3 * area * diameter**2)
    )
    expected = result.buoyancy_pressure_Pa / linear_resistance

    assert result.rising_leg_reynolds_number < 2_300.0
    assert result.descending_leg_reynolds_number < 2_300.0
    assert result.mass_flow_kg_s == pytest.approx(expected, rel=2.0e-10)


def test_loop_rejects_two_phase_pressure_mismatch_and_invalid_geometry():
    liquid = PROPS.from_pT(500_000.0, 20.0)
    other_pressure = PROPS.from_pT(490_000.0, 20.0)
    saturated_liquid = PROPS.saturated_liquid(500_000.0)
    saturated_vapor = PROPS.saturated_vapor(500_000.0)
    two_phase = PROPS.from_ph(
        500_000.0,
        0.5
        * (
            saturated_liquid.specific_enthalpy_J_kg
            + saturated_vapor.specific_enthalpy_J_kg
        ),
    )
    loop = make_loop()

    with pytest.raises(ValueError, match="single-phase"):
        loop.evaluate(two_phase, liquid)
    with pytest.raises(ValueError, match="same representative pressure"):
        loop.evaluate(liquid, other_pressure)
    with pytest.raises(ValueError, match="flow_area_m2"):
        CirculationLegGeometry(1.0, 0.01, 0.0)
