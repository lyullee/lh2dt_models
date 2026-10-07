import math

import pytest

from lh2dt import (
    CirculationLegGeometry,
    HydrogenProperties,
    MassEnergyFlow,
    RadialAxialBoundaryGeometry,
    RadialAxialTank,
    RadialAxialTankParameters,
    RadialAxialTankState,
)


PROPS = HydrogenProperties()
PRESSURE = 300_000.0


def boundary(height: float = 0.5) -> RadialAxialBoundaryGeometry:
    diameter = 0.025
    area = math.pi * diameter**2 / 4.0
    leg = CirculationLegGeometry(
        length_m=0.7,
        hydraulic_diameter_m=diameter,
        flow_area_m2=area,
        roughness_m=1.0e-6,
        local_loss_coefficient=1.2,
    )
    return RadialAxialBoundaryGeometry(height, leg, leg)


def make_tank(**changes) -> RadialAxialTank:
    values = dict(
        total_volume_m3=1.2,
        liquid_level_count=2,
        vapor_level_count=2,
        liquid_wall_volume_fraction=0.25,
        vapor_wall_volume_fraction=0.30,
        liquid_boundaries=(boundary(),),
        vapor_boundaries=(boundary(),),
        wall_heat_capacity_J_K=20_000.0,
        ambient_UA_W_K=0.0,
        wall_to_fluid_UA_W_K=12.0,
        wall_axial_conductance_W_K=0.8,
        liquid_radial_conductance_W_K=0.7,
        vapor_radial_conductance_W_K=0.2,
        liquid_axial_conductance_W_K=0.4,
        vapor_axial_conductance_W_K=0.1,
        liquid_interface_UA_W_K=(3.0, 2.0),
        vapor_interface_UA_W_K=(1.5, 1.0),
    )
    values.update(changes)
    return RadialAxialTank(RadialAxialTankParameters(**values), PROPS)


def nonuniform_state(tank: RadialAxialTank) -> RadialAxialTankState:
    saturation = PROPS.saturated_liquid(PRESSURE).temperature_K
    base = tank.initialize_uniform(
        PRESSURE,
        liquid_volume_fraction=0.55,
        liquid_temperature_K=saturation - 0.35,
        vapor_temperature_K=saturation + 0.45,
        wall_temperature_K=saturation + 0.15,
    )

    def hs(temperatures):
        return tuple(
            PROPS.from_pT(PRESSURE, temperature).specific_enthalpy_J_kg
            for temperature in temperatures
        )

    return tank.project_state(RadialAxialTankState(
        base.pressure_Pa,
        base.liquid_wall_masses_kg,
        hs((saturation - 0.30, saturation - 0.20)),
        base.liquid_core_masses_kg,
        hs((saturation - 0.65, saturation - 0.55)),
        base.vapor_wall_masses_kg,
        hs((saturation + 0.65, saturation + 0.85)),
        base.vapor_core_masses_kg,
        hs((saturation + 0.35, saturation + 0.45)),
        (
            saturation + 0.25,
            saturation + 0.30,
            saturation + 0.40,
            saturation + 0.50,
        ),
    ))


def test_closed_radial_axial_tank_derivative_is_conservative():
    tank = make_tank()
    state = nonuniform_state(tank)
    derivative = tank.derivative(
        state,
        ambient_temperature_K=state.wall_temperatures_K[0],
    )

    assert any(
        abs(value) > 0.0
        for value in derivative.liquid_transport.boundary_mass_flows_kg_s
    )
    assert any(
        abs(value) > 0.0
        for value in derivative.vapor_transport.boundary_mass_flows_kg_s
    )
    assert any(
        abs(value) > 0.0
        for value in derivative.interface_phase_change_rates_kg_s
    )
    assert derivative.interface_energy_residual_W == pytest.approx(0.0, abs=1e-10)
    assert derivative.boundary_mass_rate_kg_s == 0.0
    assert derivative.mass_conservation_residual_kg_s == pytest.approx(0.0, abs=1e-14)
    assert derivative.energy_conservation_residual_W == pytest.approx(0.0, abs=2e-9)
    assert derivative.common_pressure.differentiated_volume_residual_m3_s == pytest.approx(
        0.0, abs=2e-14
    )


def test_external_cell_port_and_heat_close_open_system_balances():
    tank = make_tank(ambient_UA_W_K=4.0)
    state = nonuniform_state(tank)
    inlet_index = tank.cell_index("liquid", "core", 0)
    outlet_index = tank.cell_index("vapor", "wall", 1)
    streams = [tuple() for _ in range(tank.cell_count)]
    inlet_h = PROPS.from_pT(state.pressure_Pa, 20.0).specific_enthalpy_J_kg
    outlet_h = tank.thermo(state).cells[outlet_index].specific_enthalpy_J_kg
    streams[inlet_index] = (MassEnergyFlow(0.015, inlet_h, "fill"),)
    streams[outlet_index] = (MassEnergyFlow(-0.004, outlet_h, "vent"),)
    fluid_heat = [0.0] * tank.cell_count
    fluid_heat[tank.cell_index("vapor", "core", 0)] = 8.0
    wall_heat = (1.0, 2.0, 3.0, 4.0)
    ambient = max(state.wall_temperatures_K) + 5.0
    derivative = tank.derivative(
        state,
        ambient_temperature_K=ambient,
        cell_streams=streams,
        direct_fluid_heat_W=fluid_heat,
        direct_wall_heat_W=wall_heat,
    )

    expected_mass = 0.015 - 0.004
    assert derivative.boundary_mass_rate_kg_s == pytest.approx(expected_mass)
    assert derivative.total_mass_rate_kg_s == pytest.approx(expected_mass)
    assert derivative.mass_conservation_residual_kg_s == pytest.approx(0.0, abs=1e-13)
    assert derivative.total_energy_rate_W == pytest.approx(
        derivative.boundary_energy_rate_W, abs=2e-9
    )
    assert derivative.energy_conservation_residual_W == pytest.approx(0.0, abs=2e-9)


def test_projected_step_preserves_closed_mass_energy_and_volume():
    tank = make_tank()
    state = nonuniform_state(tank)
    result = tank.step_euler(
        state,
        time_step_s=0.02,
        ambient_temperature_K=state.wall_temperatures_K[0],
    )

    assert result.total_mass_residual_kg == pytest.approx(0.0, abs=2e-13)
    assert result.total_energy_residual_J == pytest.approx(0.0, abs=2e-7)
    assert result.thermo.volume_residual_m3 == pytest.approx(0.0, abs=2e-10)
    assert result.state.pressure_Pa > 0.0
    assert all(mass > 0.0 for mass in (
        *result.state.liquid_wall_masses_kg,
        *result.state.liquid_core_masses_kg,
        *result.state.vapor_wall_masses_kg,
        *result.state.vapor_core_masses_kg,
    ))


def test_recommended_explicit_time_step_is_a_conservative_caller_guard():
    tank = make_tank()
    state = nonuniform_state(tank)
    heat = [0.0] * tank.cell_count
    heat[tank.cell_index("vapor", "core", 0)] = 100.0
    limit = tank.recommended_explicit_time_step_s(
        state,
        ambient_temperature_K=state.wall_temperatures_K[0],
        direct_fluid_heat_W=heat,
        maximum_time_step_s=2.0,
    )

    assert math.isfinite(limit)
    assert 0.0 < limit <= 1.0
    with pytest.raises(ValueError, match="safety_factor"):
        tank.recommended_explicit_time_step_s(
            state,
            ambient_temperature_K=state.wall_temperatures_K[0],
            safety_factor=1.0,
        )


def test_stable_cell_index_and_source_shapes_are_enforced():
    tank = make_tank()
    assert tank.cell_index("liquid", "wall", 0) == 0
    assert tank.cell_index("liquid", "core", 0) == 2
    assert tank.cell_index("vapor", "wall", 0) == 4
    assert tank.cell_index("vapor", "core", 0) == 6
    state = nonuniform_state(tank)
    with pytest.raises(ValueError, match="cell_streams"):
        tank.derivative(state, 25.0, cell_streams=(tuple(),))
    with pytest.raises(ValueError, match="direct_wall_heat_W"):
        tank.derivative(state, 25.0, direct_wall_heat_W=(0.0,))
    with pytest.raises(ValueError, match="phase"):
        tank.cell_index("two_phase", "wall", 0)


def test_kinetic_gas_film_is_a_replaceable_vapor_interface_closure():
    conduction = make_tank()
    kinetic = make_tank(
        vapor_interface_heat_transfer_model="kinetic_gas_film",
        vapor_interface_area_m2=(0.05, 0.07),
        gas_interface_energy_accommodation_coefficient=0.001,
    )
    state = nonuniform_state(kinetic)
    conduction_derivative = conduction.derivative(
        state, ambient_temperature_K=state.wall_temperatures_K[0]
    )
    kinetic_derivative = kinetic.derivative(
        state, ambient_temperature_K=state.wall_temperatures_K[0]
    )

    assert kinetic_derivative.interface_energy_residual_W == pytest.approx(0.0, abs=1e-10)
    assert kinetic_derivative.energy_conservation_residual_W == pytest.approx(0.0, abs=2e-9)
    assert kinetic_derivative.interface_phase_change_rates_kg_s != pytest.approx(
        conduction_derivative.interface_phase_change_rates_kg_s
    )


def test_kinetic_gas_film_requires_explicit_interface_area():
    with pytest.raises(ValueError, match="vapor_interface_area_m2"):
        make_tank(vapor_interface_heat_transfer_model="kinetic_gas_film")


def test_saturated_endpoint_state_is_accepted_as_single_phase_limit():
    tank = make_tank(distributed_boiling_model="saturated_liquid_complementarity")
    state = tank.initialize_saturated(PRESSURE, 0.55)
    thermo = tank.thermo(state)
    derivative = tank.derivative(
        state, ambient_temperature_K=state.wall_temperatures_K[0]
    )

    assert all(cell.quality == 0.0 for cell in thermo.cells[:4])
    assert all(cell.quality == 1.0 for cell in thermo.cells[4:])
    assert derivative.liquid_transport.boundary_mass_flows_kg_s == pytest.approx((0.0,))
    assert derivative.vapor_transport.boundary_mass_flows_kg_s == pytest.approx((0.0,))
    assert derivative.mass_conservation_residual_kg_s == pytest.approx(0.0, abs=1e-14)
    assert derivative.energy_conservation_residual_W == pytest.approx(0.0, abs=1e-9)


def test_saturated_liquid_boiling_complementarity_and_step_projection():
    tank = make_tank(distributed_boiling_model="saturated_liquid_complementarity")
    state = tank.initialize_saturated(PRESSURE, 0.55)
    heat = [0.0] * tank.cell_count
    heated = tank.cell_index("liquid", "wall", 0)
    heat[heated] = 100.0
    derivative = tank.derivative(
        state,
        ambient_temperature_K=state.wall_temperatures_K[0],
        direct_fluid_heat_W=heat,
    )
    slope = PROPS.saturation_enthalpy_pressure_derivative_J_kg_Pa(
        state.pressure_Pa, 0.0
    )

    assert derivative.distributed_boiling_rate_kg_s > 0.0
    assert derivative.distributed_boiling_rates_kg_s[heated] > 0.0
    assert derivative.enthalpy_rates_J_kg_s[heated] == pytest.approx(
        slope * derivative.pressure_rate_Pa_s, rel=2e-11
    )
    assert derivative.distributed_boiling_latent_power_W > 0.0
    assert derivative.mass_conservation_residual_kg_s == pytest.approx(0.0, abs=1e-13)
    assert derivative.energy_conservation_residual_W == pytest.approx(0.0, abs=2e-9)

    result = tank.step_euler(
        state,
        time_step_s=0.02,
        ambient_temperature_K=state.wall_temperatures_K[0],
        direct_fluid_heat_W=heat,
    )
    liquid_cells = result.thermo.cells[:4]
    assert all(
        cell.quality is None or cell.quality <= 1.0e-10
        for cell in liquid_cells
    )
    assert result.total_mass_residual_kg == pytest.approx(0.0, abs=2e-13)
    assert result.total_energy_residual_J == pytest.approx(0.0, abs=2e-7)
    assert result.thermo.volume_residual_m3 == pytest.approx(0.0, abs=2e-10)
