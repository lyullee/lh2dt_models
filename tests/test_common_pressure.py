import math

import pytest

from lh2dt import (
    CirculationLegGeometry,
    CommonPressureCellState,
    CommonPressureRigidVolume,
    HydrogenProperties,
    RadialAxialBoundaryGeometry,
    RadialAxialCirculationTransport,
    RadialAxialPhaseState,
)


PROPS = HydrogenProperties()
PRESSURE = 500_000.0


def h(temperature_K: float) -> float:
    return PROPS.from_pT(PRESSURE, temperature_K).specific_enthalpy_J_kg


def total_volume(state: CommonPressureCellState) -> float:
    return sum(
        mass / PROPS.from_ph(PRESSURE, enthalpy).density_kg_m3
        for mass, enthalpy in zip(
            state.masses_kg, state.specific_enthalpies_J_kg
        )
    )


def test_common_pressure_dae_closes_radial_axial_internal_transport():
    diameter = 0.02
    area = math.pi * diameter**2 / 4.0
    leg = CirculationLegGeometry(2.0, diameter, area, 1.0e-6, 1.0)
    phase = RadialAxialPhaseState(
        wall_masses_kg=(2.0, 2.5, 3.0),
        wall_specific_enthalpies_J_kg=(h(21.0), h(21.2), h(21.4)),
        core_masses_kg=(3.0, 3.5, 4.0),
        core_specific_enthalpies_J_kg=(h(20.0), h(20.1), h(20.2)),
    )
    transport = RadialAxialCirculationTransport(
        (
            RadialAxialBoundaryGeometry(0.8, leg, leg),
            RadialAxialBoundaryGeometry(1.1, leg, leg),
        ),
        properties=PROPS,
    ).evaluate(PRESSURE, phase)
    state = CommonPressureCellState(
        PRESSURE,
        (*phase.wall_masses_kg, *phase.core_masses_kg),
        (
            *phase.wall_specific_enthalpies_J_kg,
            *phase.core_specific_enthalpies_J_kg,
        ),
    )
    derivative = CommonPressureRigidVolume(
        total_volume(state), PROPS
    ).derivative(
        state,
        (*transport.wall_mass_rates_kg_s, *transport.core_mass_rates_kg_s),
        (*transport.wall_enthalpy_flow_W, *transport.core_enthalpy_flow_W),
    )

    assert derivative.volume_constraint_residual_m3 == pytest.approx(
        0.0, abs=1e-15
    )
    assert derivative.differentiated_volume_residual_m3_s == pytest.approx(
        0.0, abs=1e-15
    )
    assert derivative.total_mass_rate_kg_s == pytest.approx(0.0, abs=1e-15)
    assert derivative.imposed_enthalpy_inventory_rate_W == pytest.approx(
        0.0, abs=1e-10
    )
    assert derivative.total_fluid_energy_rate_W == pytest.approx(
        0.0, abs=1e-9
    )
    assert derivative.energy_residual_W == pytest.approx(0.0, abs=1e-9)


def test_external_stream_and_heat_match_open_system_energy_rate():
    enthalpies = (h(20.0), h(20.5))
    state = CommonPressureCellState(PRESSURE, (2.0, 3.0), enthalpies)
    system = CommonPressureRigidVolume(total_volume(state), PROPS)
    inlet_mass_rate = 0.01
    inlet_enthalpy = h(21.0)
    heat = 25.0
    derivative = system.derivative(
        state,
        (inlet_mass_rate, 0.0),
        (inlet_mass_rate * inlet_enthalpy + heat, 0.0),
    )
    expected_energy = inlet_mass_rate * inlet_enthalpy + heat

    assert derivative.total_mass_rate_kg_s == pytest.approx(inlet_mass_rate)
    assert derivative.total_fluid_energy_rate_W == pytest.approx(
        expected_energy, rel=1e-12
    )
    assert derivative.energy_residual_W == pytest.approx(0.0, abs=1e-9)
    assert derivative.differentiated_volume_residual_m3_s == pytest.approx(
        0.0, abs=1e-15
    )


def test_common_pressure_contract_rejects_two_phase_and_bad_source_shape():
    sat_l = PROPS.saturated_liquid(PRESSURE).specific_enthalpy_J_kg
    sat_v = PROPS.saturated_vapor(PRESSURE).specific_enthalpy_J_kg
    state = CommonPressureCellState(
        PRESSURE, (1.0, 1.0), (h(20.0), 0.5 * (sat_l + sat_v))
    )
    system = CommonPressureRigidVolume(1.0, PROPS)
    with pytest.raises(ValueError, match="single-phase"):
        system.derivative(state, (0.0, 0.0), (0.0, 0.0))

    valid = CommonPressureCellState(
        PRESSURE, (1.0, 1.0), (h(20.0), h(20.5))
    )
    with pytest.raises(ValueError, match="mass_rates_kg_s"):
        system.derivative(valid, (0.0,), (0.0, 0.0))
