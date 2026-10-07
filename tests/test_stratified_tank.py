import math

import pytest

from lh2dt import (
    HorizontalVesselGeometry,
    MassEnergyFlow,
    StratifiedTank,
    StratifiedTankParameters,
    StratifiedTankState,
)


def make_tank(**overrides):
    shape = HorizontalVesselGeometry(2.0, 4.0, 0.5, quadrature_order=48)
    values = dict(
        lower_wall_heat_capacity_J_K=2.0e6,
        upper_wall_heat_capacity_J_K=1.5e6,
        lower_ambient_UA_W_K=0.0,
        upper_ambient_UA_W_K=0.0,
        wall_liquid_U_W_m2K=5.0,
        wall_vapor_U_W_m2K=2.0,
        liquid_interface_UA_W_K=10.0,
        vapor_interface_UA_W_K=8.0,
        wall_axial_UA_W_K=20.0,
        wall_zone_split_height_m=1.0,
    )
    values.update(overrides)
    return StratifiedTank(shape, StratifiedTankParameters(**values))


def test_saturated_initialization_recovers_pressure_volume_and_wall_areas():
    tank = make_tank()
    state = tank.initialize_saturated(101_325.0, 0.37)
    thermo = tank.thermo(state, 101_325.0)

    assert thermo.pressure_Pa == pytest.approx(101_325.0, rel=1e-9)
    assert thermo.liquid_volume_m3 + thermo.vapor_volume_m3 == pytest.approx(
        tank.shape.volume_m3, rel=1e-11
    )
    assert thermo.liquid_volume_m3 / tank.shape.volume_m3 == pytest.approx(0.37, rel=1e-9)
    assert (
        thermo.lower_wet_area_m2 + thermo.lower_dry_area_m2
        + thermo.upper_wet_area_m2 + thermo.upper_dry_area_m2
    ) == pytest.approx(tank.shape.inner_surface_area_m2, rel=1e-11)


def test_equilibrium_closed_tank_has_zero_derivative():
    tank = make_tank()
    state = tank.initialize_saturated(101_325.0, 0.5)
    saturation_temperature = tank.thermo(state, 101_325.0).liquid.temperature_K
    derivative = tank.derivative(state, (), (), saturation_temperature, pressure_hint_Pa=101_325.0)

    assert derivative.liquid_mass_kg_s == pytest.approx(0.0, abs=1e-12)
    assert derivative.vapor_mass_kg_s == pytest.approx(0.0, abs=1e-12)
    assert derivative.liquid_internal_energy_W == pytest.approx(0.0, abs=1e-8)
    assert derivative.vapor_internal_energy_W == pytest.approx(0.0, abs=1e-8)
    assert derivative.lower_wall_temperature_K_s == pytest.approx(0.0, abs=1e-12)
    assert derivative.upper_wall_temperature_K_s == pytest.approx(0.0, abs=1e-12)


def test_phase_change_is_an_internal_mass_and_energy_transfer():
    tank = make_tank()
    initial = tank.initialize_saturated(101_325.0, 0.5)
    state = StratifiedTankState(
        initial.liquid_mass_kg,
        initial.liquid_internal_energy_J,
        initial.vapor_mass_kg,
        initial.vapor_internal_energy_J + 1_000.0,
        initial.lower_wall_temperature_K,
        initial.upper_wall_temperature_K,
    )
    thermo = tank.thermo(state, 101_325.0)
    derivative = tank.derivative(
        state, (), (), thermo.liquid.temperature_K, pressure_hint_Pa=thermo.pressure_Pa
    )
    parameters = tank.parameters

    total_mass_rate = derivative.liquid_mass_kg_s + derivative.vapor_mass_kg_s
    total_energy_rate = (
        derivative.liquid_internal_energy_W + derivative.vapor_internal_energy_W
        + parameters.lower_wall_heat_capacity_J_K * derivative.lower_wall_temperature_K_s
        + parameters.upper_wall_heat_capacity_J_K * derivative.upper_wall_temperature_K_s
    )
    assert abs(derivative.evaporation_rate_kg_s) > 0.0
    assert total_mass_rate == pytest.approx(0.0, abs=1e-14)
    assert total_energy_rate == pytest.approx(0.0, abs=1e-8)


def test_boundary_balance_uses_phase_owned_enthalpy_for_outflow():
    tank = make_tank(lower_ambient_UA_W_K=3.0, upper_ambient_UA_W_K=4.0)
    state = tank.initialize_saturated(101_325.0, 0.5)
    thermo = tank.thermo(state, 101_325.0)
    liquid_flows = (MassEnergyFlow(-0.01, 9.9e9, "liquid outlet"),)
    vapor_flows = (MassEnergyFlow(0.002, thermo.vapor.specific_enthalpy_J_kg + 500.0, "vapor inlet"),)
    lower_direct = 11.0
    upper_direct = 13.0
    derivative = tank.derivative(
        state, liquid_flows, vapor_flows, 295.0, lower_direct, upper_direct, thermo.pressure_Pa
    )
    p = tank.parameters
    total_energy_rate = (
        derivative.liquid_internal_energy_W + derivative.vapor_internal_energy_W
        + p.lower_wall_heat_capacity_J_K * derivative.lower_wall_temperature_K_s
        + p.upper_wall_heat_capacity_J_K * derivative.upper_wall_temperature_K_s
    )
    expected = (
        liquid_flows[0].mass_flow_kg_s * thermo.liquid.specific_enthalpy_J_kg
        + vapor_flows[0].enthalpy_flow_W
        + derivative.lower_ambient_heat_W + derivative.upper_ambient_heat_W
        + lower_direct + upper_direct
    )
    assert total_energy_rate == pytest.approx(expected, abs=1e-8)


def test_rk4_boundary_ledgers_close_for_external_wall_heat():
    tank = make_tank()
    initial = tank.initialize_saturated(101_325.0, 0.5)
    results = tank.simulate(
        initial,
        duration_s=2.0,
        time_step_s=1.0,
        ambient_temperature_K=initial.upper_wall_temperature_K,
        flow_callback=lambda _time, _thermo: ((), ()),
        wall_heat_callback=lambda _time, _thermo: (100.0, 300.0),
    )
    final = results[-1]

    assert final.cumulative_boundary_mass_kg == pytest.approx(0.0, abs=1e-14)
    assert final.cumulative_boundary_energy_J == pytest.approx(800.0, rel=1e-12)
    assert final.total_mass_residual_kg == pytest.approx(0.0, abs=1e-12)
    assert final.total_energy_residual_J == pytest.approx(0.0, abs=2e-6)
    assert math.isfinite(final.thermo.pressure_Pa)
