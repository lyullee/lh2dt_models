import math

import pytest

from lh2dt import (
    kinetic_gas_film_heat_transfer_coefficient,
    schrage_mass_flux,
)


def test_schrage_mass_flux_matches_published_equation():
    result = schrage_mass_flux(
        interface_pressure_Pa=102_000.0,
        interface_temperature_K=20.4,
        vapor_pressure_Pa=101_000.0,
        vapor_temperature_K=20.6,
        specific_gas_constant_J_kgK=4_124.0,
        accommodation_coefficient=0.001,
    )
    expected = (
        2.0 * 0.001 / (2.0 - 0.001)
        / math.sqrt(2.0 * math.pi * 4_124.0)
        * (102_000.0 / math.sqrt(20.4) - 101_000.0 / math.sqrt(20.6))
    )
    assert result.mass_flux_kg_m2_s == pytest.approx(expected)
    assert result.accommodation_coefficient == 0.001


def test_schrage_mass_flux_is_zero_at_equilibrium_and_changes_sign():
    equilibrium = schrage_mass_flux(
        interface_pressure_Pa=101_325.0,
        interface_temperature_K=20.3,
        vapor_pressure_Pa=101_325.0,
        vapor_temperature_K=20.3,
        specific_gas_constant_J_kgK=4_124.0,
    )
    evaporation = schrage_mass_flux(
        interface_pressure_Pa=102_000.0,
        interface_temperature_K=20.3,
        vapor_pressure_Pa=101_325.0,
        vapor_temperature_K=20.3,
        specific_gas_constant_J_kgK=4_124.0,
    )
    condensation = schrage_mass_flux(
        interface_pressure_Pa=101_325.0,
        interface_temperature_K=20.3,
        vapor_pressure_Pa=102_000.0,
        vapor_temperature_K=20.3,
        specific_gas_constant_J_kgK=4_124.0,
    )
    assert equilibrium.mass_flux_kg_m2_s == 0.0
    assert evaporation.mass_flux_kg_m2_s > 0.0
    assert condensation.mass_flux_kg_m2_s == pytest.approx(
        -evaporation.mass_flux_kg_m2_s
    )


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, float("nan")])
def test_schrage_rejects_invalid_accommodation_coefficient(value):
    with pytest.raises(ValueError, match="accommodation_coefficient"):
        schrage_mass_flux(
            interface_pressure_Pa=101_325.0,
            interface_temperature_K=20.3,
            vapor_pressure_Pa=101_325.0,
            vapor_temperature_K=20.3,
            specific_gas_constant_J_kgK=4_124.0,
            accommodation_coefficient=value,
        )


def test_kinetic_gas_film_coefficient_matches_molecular_impingement_formula():
    result = kinetic_gas_film_heat_transfer_coefficient(
        pressure_Pa=101_325.0,
        temperature_K=20.5,
        isobaric_heat_capacity_J_kgK=14_300.0,
        specific_gas_constant_J_kgK=4_124.0,
        energy_accommodation_coefficient=0.5,
    )
    impingement = 101_325.0 / math.sqrt(2.0 * math.pi * 4_124.0 * 20.5)
    assert result.impingement_mass_flux_kg_m2_s == pytest.approx(impingement)
    assert result.heat_transfer_coefficient_W_m2K == pytest.approx(
        0.5 * impingement * 14_300.0
    )


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, float("nan")])
def test_kinetic_gas_film_rejects_invalid_energy_accommodation(value):
    with pytest.raises(ValueError, match="energy_accommodation_coefficient"):
        kinetic_gas_film_heat_transfer_coefficient(
            pressure_Pa=101_325.0,
            temperature_K=20.5,
            isobaric_heat_capacity_J_kgK=14_300.0,
            specific_gas_constant_J_kgK=4_124.0,
            energy_accommodation_coefficient=value,
        )
