import pytest

from lh2dt import (
    HorizontalVesselGeometry,
    LayeredTank,
    LayeredTankParameters,
    LayeredTankState,
    MassEnergyFlow,
    reduce_layered_state_to_nasa_energy_partition,
    schrage_mass_flux,
)


def make_tank(liquid_cells=3, vapor_cells=3, **overrides):
    shape = HorizontalVesselGeometry(2.0, 4.0, 0.5, quadrature_order=48)
    values = dict(
        liquid_cell_count=liquid_cells,
        vapor_cell_count=vapor_cells,
        wall_node_count=6,
        wall_heat_capacity_J_K=3.0e6,
    )
    values.update(overrides)
    return LayeredTank(shape, LayeredTankParameters(**values))


def empty_flows(tank):
    p = tank.parameters
    return (
        tuple(() for _ in range(p.liquid_cell_count)),
        tuple(() for _ in range(p.vapor_cell_count)),
    )


def test_saturated_initialization_closes_volume_and_wall_overlap():
    tank = make_tank()
    state = tank.initialize_saturated(101_325.0, 0.43)
    thermo = tank.thermo(state)

    assert thermo.volume_residual_m3 == pytest.approx(0.0, abs=1e-12)
    assert sum(thermo.cell_volumes_m3[:3]) / tank.shape.volume_m3 == pytest.approx(0.43)
    assert sum(map(sum, thermo.wall_overlap_areas_m2)) == pytest.approx(
        tank.shape.inner_surface_area_m2, rel=1e-12
    )
    assert all(
        sum(row) > 0.0 for row in thermo.wall_overlap_areas_m2
    )


def test_profiled_initialization_preserves_temperature_gradient_and_volume():
    tank = make_tank(liquid_cells=2, vapor_cells=3)
    pressure = 101_325.0
    saturation_temperature = tank.properties.saturated_liquid(pressure).temperature_K
    liquid_temperatures = (saturation_temperature - 0.10, saturation_temperature)
    vapor_temperatures = (
        saturation_temperature,
        saturation_temperature + 0.75,
        saturation_temperature + 1.50,
    )
    wall_temperatures = tuple(
        saturation_temperature + 0.25 * index for index in range(6)
    )

    state = tank.initialize_temperature_profiles(
        pressure,
        0.43,
        liquid_temperatures,
        vapor_temperatures,
        wall_temperatures,
    )
    thermo = tank.thermo(state)

    assert thermo.volume_residual_m3 == pytest.approx(0.0, abs=2e-11)
    assert sum(thermo.cell_volumes_m3[:2]) / tank.shape.volume_m3 == pytest.approx(
        0.43, abs=2e-12
    )
    assert tuple(item.temperature_K for item in thermo.liquid) == pytest.approx(
        liquid_temperatures, abs=2e-8
    )
    assert tuple(item.temperature_K for item in thermo.vapor) == pytest.approx(
        vapor_temperatures, abs=2e-8
    )
    assert state.wall_temperatures_K == pytest.approx(wall_temperatures)


def test_profiled_initialization_rejects_wrong_profile_lengths():
    tank = make_tank(liquid_cells=2, vapor_cells=3)
    with pytest.raises(ValueError, match="liquid_temperatures_K length"):
        tank.initialize_temperature_profiles(
            101_325.0,
            0.5,
            liquid_temperatures_K=(20.0,),
        )


def test_connected_vapor_dead_volume_closes_common_pressure_volume():
    dead_volume = 0.35
    tank = make_tank(connected_vapor_volume_m3=dead_volume)
    state = tank.initialize_saturated(101_325.0, 0.43)
    thermo = tank.thermo(state)

    assert tank.total_rigid_volume_m3 == pytest.approx(
        tank.shape.volume_m3 + dead_volume
    )
    assert sum(thermo.cell_volumes_m3) == pytest.approx(
        tank.total_rigid_volume_m3, abs=2e-11
    )
    assert sum(thermo.tank_cell_volumes_m3) == pytest.approx(
        tank.shape.volume_m3, abs=2e-11
    )
    assert (
        thermo.cell_volumes_m3[-1] - thermo.tank_cell_volumes_m3[-1]
    ) == pytest.approx(dead_volume, abs=2e-11)
    assert thermo.volume_residual_m3 == pytest.approx(0.0, abs=2e-11)


def test_connected_nozzle_wall_heat_is_conservative_and_has_finite_inertia():
    tank = make_tank(
        connected_vapor_volume_m3=0.10,
        connected_wall_heat_capacity_J_K=2.0e5,
        connected_ambient_UA_W_K=4.0,
        connected_wall_vapor_UA_W_K=15.0,
    )
    initial = tank.initialize_saturated(101_325.0, 0.5)
    warm = LayeredTankState(
        initial.pressure_Pa,
        initial.liquid_masses_kg,
        initial.liquid_specific_enthalpies_J_kg,
        initial.vapor_masses_kg,
        initial.vapor_specific_enthalpies_J_kg,
        tuple((*initial.wall_temperatures_K[:-1], initial.wall_temperatures_K[-1] + 4.0)),
    )
    ambient = tank.thermo(initial).vapor[-1].temperature_K + 10.0
    derivative = tank.derivative(warm, *empty_flows(tank), ambient)

    assert derivative.vapor_wall_heat_W[-1] > 0.0
    assert derivative.total_mass_rate_kg_s == pytest.approx(0.0, abs=1e-14)
    assert derivative.total_energy_rate_W == pytest.approx(
        derivative.boundary_energy_rate_W, abs=2e-8
    )
    assert derivative.wall_temperature_rates_K_s[-1] != 0.0

    final = tank.simulate(warm, 1.0, 0.1, ambient)[-1]
    assert final.total_mass_residual_kg == pytest.approx(0.0, abs=3e-11)
    assert final.total_energy_residual_J == pytest.approx(0.0, abs=5e-4)


def test_connected_attachment_parameters_reject_negative_values():
    with pytest.raises(ValueError, match="connected_vapor_volume_m3"):
        make_tank(connected_vapor_volume_m3=-1.0)


def test_wall_ambient_ua_profile_preserves_explicit_total_and_changes_distribution():
    uniform = make_tank(ambient_UA_W_K=12.0)
    profiled = make_tank(
        ambient_UA_W_K=12.0,
        ambient_UA_profile_W_K=(0.5, 0.5, 1.0, 2.0, 3.0, 5.0),
    )
    assert sum(profiled._wall_ambient_UA_W_K) == pytest.approx(12.0)
    assert tuple(profiled._wall_ambient_UA_W_K) != pytest.approx(
        tuple(uniform._wall_ambient_UA_W_K)
    )


def test_wall_ambient_ua_profile_rejects_wrong_total_or_length():
    with pytest.raises(ValueError, match="length"):
        make_tank(
            ambient_UA_W_K=12.0,
            ambient_UA_profile_W_K=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="sum to ambient_UA_W_K"):
        make_tank(
            ambient_UA_W_K=12.0,
            ambient_UA_profile_W_K=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )


def test_closed_saturated_state_is_stationary():
    tank = make_tank()
    state = tank.initialize_saturated(101_325.0, 0.5)
    temperature = tank.thermo(state).liquid[0].temperature_K
    derivative = tank.derivative(state, *empty_flows(tank), temperature)

    assert derivative.pressure_Pa_s == pytest.approx(0.0, abs=1e-12)
    assert derivative.evaporation_rate_kg_s == pytest.approx(0.0, abs=1e-14)
    assert derivative.total_mass_rate_kg_s == pytest.approx(0.0, abs=1e-14)
    assert derivative.total_energy_rate_W == pytest.approx(0.0, abs=1e-10)
    assert derivative.boundary_energy_rate_W == pytest.approx(0.0, abs=1e-10)


def test_internal_stratification_and_phase_change_conserve_total_balances():
    tank = make_tank()
    initial = tank.initialize_saturated(101_325.0, 0.5)
    vapor_h = [
        value + 2_000.0 for value in initial.vapor_specific_enthalpies_J_kg
    ]
    vapor_h[0] += 1_000.0
    state = tank.project_pressure(LayeredTankState(
        initial.pressure_Pa,
        initial.liquid_masses_kg,
        initial.liquid_specific_enthalpies_J_kg,
        initial.vapor_masses_kg,
        tuple(vapor_h),
        initial.wall_temperatures_K,
    ))
    temperature = tank.thermo(state).liquid[0].temperature_K
    derivative = tank.derivative(state, *empty_flows(tank), temperature)

    assert abs(derivative.evaporation_rate_kg_s) > 0.0
    assert derivative.latent_phase_change_power_W == pytest.approx(
        derivative.liquid_interface_heat_W + derivative.vapor_interface_heat_W,
        abs=1e-10,
    )
    assert derivative.total_mass_rate_kg_s == pytest.approx(0.0, abs=1e-14)
    assert derivative.total_energy_rate_W == pytest.approx(0.0, abs=1e-9)
    assert derivative.boundary_energy_rate_W == pytest.approx(0.0, abs=1e-12)


def test_outflow_uses_owning_cell_enthalpy_and_boundary_ledger_closes():
    tank = make_tank()
    state = tank.initialize_saturated(101_325.0, 0.5)
    thermo = tank.thermo(state)
    liquid_flows, vapor_flows = empty_flows(tank)
    liquid_flows = list(liquid_flows)
    liquid_flows[0] = (MassEnergyFlow(-0.01, 9.9e9, "outlet"),)
    derivative = tank.derivative(
        state, tuple(liquid_flows), vapor_flows, thermo.liquid[0].temperature_K
    )
    expected = -0.01 * thermo.liquid[0].specific_enthalpy_J_kg

    assert derivative.boundary_mass_rate_kg_s == pytest.approx(-0.01)
    assert derivative.boundary_energy_rate_W == pytest.approx(expected, abs=1e-8)
    assert derivative.total_energy_rate_W == pytest.approx(expected, abs=1e-8)


def test_rk4_direct_heat_mass_energy_and_volume_ledgers_close():
    tank = make_tank()
    initial = tank.initialize_saturated(101_325.0, 0.5)
    temperature = tank.thermo(initial).liquid[0].temperature_K
    results = tank.simulate(
        initial,
        duration_s=2.0,
        time_step_s=0.2,
        ambient_temperature_K=temperature,
        fluid_heat_callback=lambda _time, _thermo: ((0.0, 0.0, 0.0), (0.0, 0.0, 100.0)),
    )
    final = results[-1]

    assert final.cumulative_boundary_energy_J == pytest.approx(200.0, abs=1e-10)
    assert final.total_mass_residual_kg == pytest.approx(0.0, abs=2e-12)
    assert final.total_energy_residual_J == pytest.approx(0.0, abs=2e-5)
    assert final.thermo.volume_residual_m3 == pytest.approx(0.0, abs=2e-9)
    assert final.thermo.vapor[-1].temperature_K > final.thermo.vapor[0].temperature_K
    assert final.cumulative_latent_phase_change_energy_J == pytest.approx(
        final.cumulative_liquid_interface_heat_J + final.cumulative_vapor_interface_heat_J,
        abs=1e-8,
    )

    partition = reduce_layered_state_to_nasa_energy_partition(
        tank=tank,
        initial_state=initial,
        final_result=final,
        test_number=0,
        heating_mode="unit-test",
    )
    assert partition.component_closure_error_percent_point == pytest.approx(
        0.0, abs=1e-12
    )
    assert (
        partition.liquid_energy_J
        + partition.vapor_energy_J
        + partition.evaporation_energy_J
    ) == pytest.approx(partition.total_heat_input_J, abs=1e-10)


def test_uniform_cells_recover_the_same_grouped_solution():
    coarse = make_tank(1, 1)
    fine = make_tank(3, 3)
    coarse_initial = coarse.initialize_saturated(101_325.0, 0.5)
    fine_initial = fine.initialize_saturated(101_325.0, 0.5)
    temperature = coarse.thermo(coarse_initial).liquid[0].temperature_K
    coarse_final = coarse.simulate(
        coarse_initial, 1.0, 0.1, temperature,
        fluid_heat_callback=lambda _time, _thermo: ((90.0,), (30.0,)),
    )[-1]
    fine_final = fine.simulate(
        fine_initial, 1.0, 0.1, temperature,
        fluid_heat_callback=lambda _time, _thermo: (
            (30.0, 30.0, 30.0), (10.0, 10.0, 10.0)
        ),
    )[-1]

    assert fine_final.state.pressure_Pa == pytest.approx(
        coarse_final.state.pressure_Pa, rel=2e-8
    )
    assert fine_final.total_energy_residual_J == pytest.approx(0.0, abs=2e-5)


def test_daigle_interface_convection_is_inactive_for_stably_superheated_vapor():
    conduction = make_tank(interface_heat_transfer_model="conduction")
    convection = make_tank(interface_heat_transfer_model="daigle_2013")
    initial = conduction.initialize_saturated(101_325.0, 0.5)
    vapor_h = [
        value + 2_000.0 for value in initial.vapor_specific_enthalpies_J_kg
    ]
    vapor_h[0] += 200.0
    perturbed = LayeredTankState(
        initial.pressure_Pa,
        initial.liquid_masses_kg,
        initial.liquid_specific_enthalpies_J_kg,
        initial.vapor_masses_kg,
        tuple(vapor_h),
        initial.wall_temperatures_K,
    )
    conduction_state = conduction.project_pressure(perturbed)
    convection_state = convection.project_pressure(perturbed)
    ambient = conduction.thermo(conduction_state).liquid[0].temperature_K
    conduction_rate = conduction.derivative(
        conduction_state, *empty_flows(conduction), ambient
    )
    convection_rate = convection.derivative(
        convection_state, *empty_flows(convection), ambient
    )

    assert convection_rate.vapor_interface_heat_W == pytest.approx(
        conduction_rate.vapor_interface_heat_W, rel=1e-12
    )
    assert convection_rate.total_mass_rate_kg_s == pytest.approx(0.0, abs=1e-13)
    assert convection_rate.total_energy_rate_W == pytest.approx(0.0, abs=1e-8)


def test_unknown_interface_heat_transfer_model_is_rejected():
    with pytest.raises(ValueError):
        make_tank(interface_heat_transfer_model="fitted")


def test_kinetic_gas_film_is_stationary_at_saturated_equilibrium():
    tank = make_tank(interface_heat_transfer_model="kinetic_gas_film")
    state = tank.initialize_saturated(101_325.0, 0.5)
    thermo = tank.thermo(state)
    derivative = tank.derivative(
        state, *empty_flows(tank), thermo.liquid[0].temperature_K
    )
    assert derivative.vapor_interface_heat_W == pytest.approx(0.0, abs=1e-12)
    assert derivative.evaporation_rate_kg_s == pytest.approx(0.0, abs=1e-14)
    assert derivative.total_energy_rate_W == pytest.approx(0.0, abs=1e-10)


def test_kinetic_gas_film_changes_only_gas_side_interface_heat():
    conduction = make_tank(interface_heat_transfer_model="conduction")
    kinetic = make_tank(interface_heat_transfer_model="kinetic_gas_film")
    initial = conduction.initialize_saturated(101_325.0, 0.5)
    vapor_h = tuple(value + 2_000.0 for value in initial.vapor_specific_enthalpies_J_kg)
    perturbed = LayeredTankState(
        initial.pressure_Pa,
        initial.liquid_masses_kg,
        initial.liquid_specific_enthalpies_J_kg,
        initial.vapor_masses_kg,
        vapor_h,
        initial.wall_temperatures_K,
    )
    conduction_state = conduction.project_pressure(perturbed)
    kinetic_state = kinetic.project_pressure(perturbed)
    ambient = conduction.thermo(conduction_state).liquid[0].temperature_K
    conduction_rate = conduction.derivative(
        conduction_state, *empty_flows(conduction), ambient
    )
    kinetic_rate = kinetic.derivative(
        kinetic_state, *empty_flows(kinetic), ambient
    )
    assert kinetic_rate.liquid_interface_heat_W == pytest.approx(
        conduction_rate.liquid_interface_heat_W, rel=1e-12
    )
    assert kinetic_rate.vapor_interface_heat_W != pytest.approx(
        conduction_rate.vapor_interface_heat_W, rel=1e-6
    )
    assert kinetic_rate.total_energy_rate_W == pytest.approx(0.0, abs=1e-8)


def test_schrage_coupled_interface_is_stationary_at_saturated_equilibrium():
    tank = make_tank(
        interface_phase_change_model="schrage_kinetic_energy_balance"
    )
    state = tank.initialize_saturated(101_325.0, 0.5)
    thermo = tank.thermo(state)
    derivative = tank.derivative(
        state, *empty_flows(tank), thermo.liquid[0].temperature_K
    )

    assert derivative.interface_temperature_K == pytest.approx(
        thermo.liquid[-1].temperature_K, abs=1e-12
    )
    assert derivative.interface_pressure_Pa == pytest.approx(
        state.pressure_Pa, rel=1e-12
    )
    assert derivative.interface_mass_flux_kg_m2_s == 0.0
    assert derivative.evaporation_rate_kg_s == 0.0
    assert derivative.interface_energy_residual_W == 0.0
    assert derivative.interface_kinetic_residual_kg_m2_s == 0.0
    assert derivative.total_mass_rate_kg_s == pytest.approx(0.0, abs=1e-14)
    assert derivative.total_energy_rate_W == pytest.approx(0.0, abs=1e-10)


def test_schrage_coupled_interface_satisfies_kinetics_and_energy_balance():
    tank = make_tank(
        interface_phase_change_model="schrage_kinetic_energy_balance"
    )
    initial = tank.initialize_saturated(101_325.0, 0.5)
    state = tank.project_pressure(LayeredTankState(
        initial.pressure_Pa,
        initial.liquid_masses_kg,
        initial.liquid_specific_enthalpies_J_kg,
        initial.vapor_masses_kg,
        tuple(value + 2_000.0 for value in initial.vapor_specific_enthalpies_J_kg),
        initial.wall_temperatures_K,
    ))
    thermo = tank.thermo(state)
    derivative = tank.derivative(
        state, *empty_flows(tank), thermo.liquid[0].temperature_K
    )
    kinetic = schrage_mass_flux(
        interface_pressure_Pa=derivative.interface_pressure_Pa,
        interface_temperature_K=derivative.interface_temperature_K,
        vapor_pressure_Pa=state.pressure_Pa,
        vapor_temperature_K=thermo.vapor[0].temperature_K,
        specific_gas_constant_J_kgK=tank.properties.specific_gas_constant_J_kgK(),
        accommodation_coefficient=(
            tank.parameters.schrage_accommodation_coefficient
        ),
    )

    assert derivative.interface_mass_flux_kg_m2_s == pytest.approx(
        kinetic.mass_flux_kg_m2_s, rel=1e-4, abs=1e-12
    )
    assert derivative.latent_phase_change_power_W == pytest.approx(
        derivative.liquid_interface_heat_W + derivative.vapor_interface_heat_W,
        abs=1e-12,
    )
    assert derivative.interface_energy_residual_W == 0.0
    assert derivative.interface_kinetic_residual_kg_m2_s == pytest.approx(
        0.0, abs=1e-12
    )
    assert derivative.total_mass_rate_kg_s == pytest.approx(0.0, abs=1e-13)
    assert derivative.total_energy_rate_W == pytest.approx(0.0, abs=1e-8)


def test_unknown_phase_change_model_and_invalid_schrage_coefficient_are_rejected():
    with pytest.raises(ValueError, match="interface_phase_change_model"):
        make_tank(interface_phase_change_model="fitted")
    with pytest.raises(ValueError, match="schrage_accommodation_coefficient"):
        make_tank(schrage_accommodation_coefficient=0.0)


def test_schrage_coupled_interface_time_step_converges_and_conserves():
    tank = make_tank(
        interface_phase_change_model="schrage_kinetic_energy_balance"
    )
    initial = tank.initialize_saturated(101_325.0, 0.5)
    state = tank.project_pressure(LayeredTankState(
        initial.pressure_Pa,
        initial.liquid_masses_kg,
        initial.liquid_specific_enthalpies_J_kg,
        initial.vapor_masses_kg,
        tuple(value + 2_000.0 for value in initial.vapor_specific_enthalpies_J_kg),
        initial.wall_temperatures_K,
    ))
    ambient = tank.thermo(state).liquid[0].temperature_K
    coarse = tank.simulate(state, 0.2, 0.1, ambient)[-1]
    fine = tank.simulate(state, 0.2, 0.05, ambient)[-1]

    assert coarse.state.pressure_Pa == pytest.approx(
        fine.state.pressure_Pa, rel=1e-12
    )
    assert coarse.cumulative_evaporation_mass_kg == pytest.approx(
        fine.cumulative_evaporation_mass_kg, rel=1e-9
    )
    for result in (coarse, fine):
        assert result.total_mass_residual_kg == pytest.approx(0.0, abs=2e-12)
        assert result.total_energy_residual_J == pytest.approx(0.0, abs=2e-6)


def test_churchill_chu_wall_heat_is_directional_and_conservative():
    tank = make_tank(wall_heat_transfer_model="churchill_chu_vertical_plate")
    initial = tank.initialize_saturated(101_325.0, 0.5)
    warm_wall = LayeredTankState(
        initial.pressure_Pa,
        initial.liquid_masses_kg,
        initial.liquid_specific_enthalpies_J_kg,
        initial.vapor_masses_kg,
        initial.vapor_specific_enthalpies_J_kg,
        tuple(value + 5.0 for value in initial.wall_temperatures_K),
    )
    derivative = tank.derivative(
        warm_wall, *empty_flows(tank), warm_wall.wall_temperatures_K[0]
    )

    assert all(value > 0.0 for value in derivative.liquid_wall_heat_W)
    assert all(value > 0.0 for value in derivative.vapor_wall_heat_W)
    assert derivative.total_mass_rate_kg_s == pytest.approx(0.0, abs=1e-14)
    assert derivative.total_energy_rate_W == pytest.approx(0.0, abs=1e-8)
    assert derivative.boundary_energy_rate_W == pytest.approx(0.0, abs=1e-12)


def test_churchill_chu_wall_heat_is_phase_mesh_independent_for_uniform_state():
    coarse = make_tank(
        1, 1, wall_heat_transfer_model="churchill_chu_vertical_plate"
    )
    fine = make_tank(
        3, 3, wall_heat_transfer_model="churchill_chu_vertical_plate"
    )

    def wall_heat(tank):
        initial = tank.initialize_saturated(101_325.0, 0.5)
        state = LayeredTankState(
            initial.pressure_Pa,
            initial.liquid_masses_kg,
            initial.liquid_specific_enthalpies_J_kg,
            initial.vapor_masses_kg,
            initial.vapor_specific_enthalpies_J_kg,
            tuple(value + 5.0 for value in initial.wall_temperatures_K),
        )
        derivative = tank.derivative(
            state, *empty_flows(tank), state.wall_temperatures_K[0]
        )
        return sum(derivative.liquid_wall_heat_W), sum(derivative.vapor_wall_heat_W)

    coarse_liquid, coarse_vapor = wall_heat(coarse)
    fine_liquid, fine_vapor = wall_heat(fine)
    assert fine_liquid == pytest.approx(coarse_liquid, rel=2e-12)
    assert fine_vapor == pytest.approx(coarse_vapor, rel=2e-12)


def test_unknown_wall_heat_transfer_model_is_rejected():
    with pytest.raises(ValueError, match="wall_heat_transfer_model"):
        make_tank(wall_heat_transfer_model="calibrated")


def test_phase_wall_heat_transfer_multipliers_scale_selected_phase_only():
    base = make_tank(
        1, 1, wall_heat_transfer_model="churchill_chu_vertical_plate"
    )
    scaled = make_tank(
        1,
        1,
        wall_heat_transfer_model="churchill_chu_vertical_plate",
        wall_liquid_heat_transfer_multiplier=0.5,
        wall_vapor_heat_transfer_multiplier=2.0,
    )

    def heat(tank):
        initial = tank.initialize_saturated(101_325.0, 0.5)
        warm = LayeredTankState(
            initial.pressure_Pa,
            initial.liquid_masses_kg,
            initial.liquid_specific_enthalpies_J_kg,
            initial.vapor_masses_kg,
            initial.vapor_specific_enthalpies_J_kg,
            tuple(value + 4.0 for value in initial.wall_temperatures_K),
        )
        derivative = tank.derivative(
            warm, *empty_flows(tank), initial.wall_temperatures_K[0]
        )
        return sum(derivative.liquid_wall_heat_W), sum(
            derivative.vapor_wall_heat_W
        )

    base_liquid, base_vapor = heat(base)
    scaled_liquid, scaled_vapor = heat(scaled)
    assert scaled_liquid == pytest.approx(0.5 * base_liquid, rel=1e-12)
    assert scaled_vapor == pytest.approx(2.0 * base_vapor, rel=1e-12)


def test_phase_wall_heat_transfer_multipliers_must_be_positive():
    with pytest.raises(ValueError, match="wall_vapor_heat_transfer_multiplier"):
        make_tank(wall_vapor_heat_transfer_multiplier=0.0)


def test_yang_west_tank_wall_heat_is_mesh_independent_and_conservative():
    coarse = make_tank(
        1, 1, wall_heat_transfer_model="yang_west_2015_cryogenic_tank"
    )
    fine = make_tank(
        3, 3, wall_heat_transfer_model="yang_west_2015_cryogenic_tank"
    )

    def wall_heat(tank):
        initial = tank.initialize_saturated(101_325.0, 0.5)
        state = LayeredTankState(
            initial.pressure_Pa,
            initial.liquid_masses_kg,
            initial.liquid_specific_enthalpies_J_kg,
            initial.vapor_masses_kg,
            initial.vapor_specific_enthalpies_J_kg,
            tuple(value + 1.0 for value in initial.wall_temperatures_K),
        )
        derivative = tank.derivative(
            state, *empty_flows(tank), state.wall_temperatures_K[0]
        )
        assert derivative.total_mass_rate_kg_s == pytest.approx(0.0, abs=1e-14)
        assert derivative.total_energy_rate_W == pytest.approx(0.0, abs=1e-8)
        return sum(derivative.liquid_wall_heat_W), sum(
            derivative.vapor_wall_heat_W
        )

    coarse_liquid, coarse_vapor = wall_heat(coarse)
    fine_liquid, fine_vapor = wall_heat(fine)
    assert fine_liquid == pytest.approx(coarse_liquid, rel=2e-12)
    assert fine_vapor == pytest.approx(coarse_vapor, rel=2e-12)


def test_reduced_vertical_wall_circulation_is_directional_and_conservative():
    shape = HorizontalVesselGeometry(0.2, 0.4, 0.05, quadrature_order=48)
    tank = LayeredTank(
        shape,
        LayeredTankParameters(
            liquid_cell_count=3,
            vapor_cell_count=3,
            wall_node_count=6,
            wall_heat_capacity_J_K=1.0e4,
            vertical_wall_circulation_model="daigle_2013_reduced",
        ),
    )
    initial = tank.initialize_saturated(101_325.0, 0.5)
    state = tank.project_pressure(LayeredTankState(
        initial.pressure_Pa,
        initial.liquid_masses_kg,
        tuple(
            value + offset for value, offset in zip(
                initial.liquid_specific_enthalpies_J_kg,
                (-1_000.0, -600.0, -200.0),
            )
        ),
        initial.vapor_masses_kg,
        tuple(
            value + offset for value, offset in zip(
                initial.vapor_specific_enthalpies_J_kg,
                (200.0, 600.0, 1_000.0),
            )
        ),
        initial.wall_temperatures_K,
    ))
    thermo = tank.thermo(state)
    derivative = tank.derivative(
        state, *empty_flows(tank), thermo.liquid[0].temperature_K
    )

    assert all(value > 0.0 for value in derivative.liquid_wall_circulation_rates_kg_s)
    assert all(value < 0.0 for value in derivative.vapor_wall_circulation_rates_kg_s)
    for rate, heat, lower_h, upper_h in zip(
        derivative.liquid_wall_circulation_rates_kg_s,
        derivative.liquid_circulation_lower_cell_heat_W,
        state.liquid_specific_enthalpies_J_kg[:-1],
        state.liquid_specific_enthalpies_J_kg[1:],
    ):
        assert heat == pytest.approx(abs(rate) * (upper_h - lower_h), rel=1e-12)
    for rate, heat, lower_h, upper_h in zip(
        derivative.vapor_wall_circulation_rates_kg_s,
        derivative.vapor_circulation_lower_cell_heat_W,
        state.vapor_specific_enthalpies_J_kg[:-1],
        state.vapor_specific_enthalpies_J_kg[1:],
    ):
        assert heat == pytest.approx(abs(rate) * (upper_h - lower_h), rel=1e-12)
    assert derivative.total_mass_rate_kg_s == pytest.approx(0.0, abs=1e-14)
    assert derivative.total_energy_rate_W == pytest.approx(0.0, abs=1e-10)
    assert derivative.boundary_energy_rate_W == pytest.approx(0.0, abs=1e-12)

    final = tank.simulate(
        state,
        duration_s=0.1,
        time_step_s=0.01,
        ambient_temperature_K=initial.wall_temperatures_K[0],
    )[-1]
    assert final.total_mass_residual_kg == pytest.approx(0.0, abs=1e-12)
    assert final.total_energy_residual_J == pytest.approx(0.0, abs=2e-6)


def test_unknown_vertical_wall_circulation_model_is_rejected():
    with pytest.raises(ValueError):
        make_tank(vertical_wall_circulation_model="calibrated")


def test_uniform_wall_heat_flux_drives_reduced_circulation_without_fitting():
    shape = HorizontalVesselGeometry(0.2, 0.4, 0.05, quadrature_order=48)
    tank = LayeredTank(
        shape,
        LayeredTankParameters(
            liquid_cell_count=3,
            vapor_cell_count=3,
            wall_node_count=6,
            wall_heat_capacity_J_K=1.0e4,
            vertical_wall_circulation_model=(
                "daigle_2013_uniform_heat_flux_reduced"
            ),
        ),
    )
    initial = tank.initialize_saturated(101_325.0, 0.5)
    state = tank.project_pressure(LayeredTankState(
        initial.pressure_Pa,
        initial.liquid_masses_kg,
        tuple(
            value + offset for value, offset in zip(
                initial.liquid_specific_enthalpies_J_kg,
                (-1_000.0, -600.0, -200.0),
            )
        ),
        initial.vapor_masses_kg,
        tuple(
            value + offset for value, offset in zip(
                initial.vapor_specific_enthalpies_J_kg,
                (200.0, 600.0, 1_000.0),
            )
        ),
        initial.wall_temperatures_K,
    ))
    thermo = tank.thermo(state)
    cell_wall_areas = tuple(sum(row) for row in thermo.wall_overlap_areas_m2)
    liquid_heat = tuple(0.01 * area for area in cell_wall_areas[:3])
    vapor_heat = tuple(0.01 * area for area in cell_wall_areas[3:])
    derivative = tank.derivative(
        state,
        *empty_flows(tank),
        thermo.liquid[0].temperature_K,
        liquid_heat,
        vapor_heat,
    )

    assert all(value >= 0.0 for value in derivative.liquid_wall_circulation_rates_kg_s)
    assert all(value >= 0.0 for value in derivative.vapor_wall_circulation_rates_kg_s)
    assert any(value > 0.0 for value in derivative.liquid_wall_circulation_rates_kg_s)
    assert any(value > 0.0 for value in derivative.vapor_wall_circulation_rates_kg_s)
    expected_heat = sum(liquid_heat) + sum(vapor_heat)
    assert derivative.boundary_energy_rate_W == pytest.approx(expected_heat, rel=1e-12)
    assert derivative.total_energy_rate_W == pytest.approx(expected_heat, abs=2e-12)


def test_labeled_cells_cannot_silently_cross_into_two_phase_region():
    tank = make_tank()
    initial = tank.initialize_saturated(101_325.0, 0.5)
    liquid_h = list(initial.liquid_specific_enthalpies_J_kg)
    saturated = tank.properties.saturated_vapor(initial.pressure_Pa)
    liquid_h[-1] = 0.5 * (
        initial.liquid_specific_enthalpies_J_kg[-1]
        + saturated.specific_enthalpy_J_kg
    )
    invalid = LayeredTankState(
        initial.pressure_Pa,
        initial.liquid_masses_kg,
        tuple(liquid_h),
        initial.vapor_masses_kg,
        initial.vapor_specific_enthalpies_J_kg,
        initial.wall_temperatures_K,
    )
    with pytest.raises(ValueError, match="labeled liquid cell"):
        tank.thermo(invalid)


def test_saturated_liquid_boiling_complementarity_is_tangent_and_conservative():
    tank = make_tank(
        1, 1,
        distributed_boiling_model="saturated_liquid_complementarity",
    )
    state = tank.initialize_saturated(101_325.0, 0.5)
    temperature = tank.thermo(state).liquid[0].temperature_K
    derivative = tank.derivative(
        state,
        *empty_flows(tank),
        temperature,
        direct_liquid_heat_W=(100.0,),
        direct_vapor_heat_W=(0.0,),
    )
    saturation_slope = (
        tank.properties.saturation_enthalpy_pressure_derivative_J_kg_Pa(
            state.pressure_Pa, 0.0
        )
    )
    latent = (
        tank.properties.saturated_vapor(state.pressure_Pa).specific_enthalpy_J_kg
        - tank.properties.saturated_liquid(state.pressure_Pa).specific_enthalpy_J_kg
    )

    assert derivative.distributed_boiling_rate_kg_s > 0.0
    assert derivative.distributed_boiling_rates_kg_s == pytest.approx(
        (derivative.distributed_boiling_rate_kg_s,)
    )
    assert derivative.liquid_enthalpy_rates_J_kg_s[0] == pytest.approx(
        saturation_slope * derivative.pressure_Pa_s, rel=1e-12
    )
    assert derivative.distributed_boiling_latent_power_W == pytest.approx(
        derivative.distributed_boiling_rate_kg_s * latent, rel=1e-12
    )
    assert derivative.total_mass_rate_kg_s == pytest.approx(0.0, abs=1e-14)
    assert derivative.total_energy_rate_W == pytest.approx(100.0, abs=1e-10)
    assert derivative.boundary_energy_rate_W == pytest.approx(100.0, abs=1e-12)


def test_distributed_boiling_projection_conserves_and_is_time_step_independent():
    tank = make_tank(
        1, 1,
        distributed_boiling_model="saturated_liquid_complementarity",
    )
    initial = tank.initialize_saturated(101_325.0, 0.5)
    temperature = tank.thermo(initial).liquid[0].temperature_K

    def run(step):
        return tank.simulate(
            initial,
            duration_s=2.0,
            time_step_s=step,
            ambient_temperature_K=temperature,
            fluid_heat_callback=lambda _time, _thermo: ((100.0,), (0.0,)),
        )[-1]

    coarse = run(0.5)
    fine = run(0.1)
    assert coarse.state.pressure_Pa == pytest.approx(
        fine.state.pressure_Pa, rel=1e-12
    )
    assert coarse.cumulative_distributed_boiling_mass_kg == pytest.approx(
        fine.cumulative_distributed_boiling_mass_kg, rel=1e-11
    )
    for result in (coarse, fine):
        assert result.cumulative_distributed_boiling_mass_kg > 0.0
        assert result.cumulative_latent_phase_change_energy_J == pytest.approx(
            result.cumulative_distributed_boiling_latent_energy_J
            + result.cumulative_liquid_interface_heat_J
            + result.cumulative_vapor_interface_heat_J,
            abs=1e-8,
        )
        assert result.total_mass_residual_kg == pytest.approx(0.0, abs=2e-12)
        assert result.total_energy_residual_J == pytest.approx(0.0, abs=2e-6)
        assert result.thermo.volume_residual_m3 == pytest.approx(0.0, abs=2e-9)
        assert result.thermo.liquid[0].quality in (None, 0.0)


def test_unknown_distributed_boiling_model_is_rejected():
    with pytest.raises(ValueError, match="distributed_boiling_model"):
        make_tank(distributed_boiling_model="fitted")


def test_distributed_boiling_active_set_handles_multiple_and_subcooled_cells():
    tank = make_tank(
        3, 1,
        distributed_boiling_model="saturated_liquid_complementarity",
    )
    saturated = tank.initialize_saturated(101_325.0, 0.5)
    temperature = tank.thermo(saturated).liquid[0].temperature_K
    active = tank.derivative(
        saturated,
        *empty_flows(tank),
        temperature,
        direct_liquid_heat_W=(100.0, 100.0, 100.0),
        direct_vapor_heat_W=(0.0,),
    )
    slope = tank.properties.saturation_enthalpy_pressure_derivative_J_kg_Pa(
        saturated.pressure_Pa, 0.0
    )
    assert all(rate > 0.0 for rate in active.distributed_boiling_rates_kg_s)
    assert active.liquid_enthalpy_rates_J_kg_s == pytest.approx(
        (slope * active.pressure_Pa_s,) * 3, rel=1e-12
    )

    subcooled = tank.project_pressure(LayeredTankState(
        saturated.pressure_Pa,
        saturated.liquid_masses_kg,
        tuple(value - 1_000.0 for value in saturated.liquid_specific_enthalpies_J_kg),
        saturated.vapor_masses_kg,
        saturated.vapor_specific_enthalpies_J_kg,
        saturated.wall_temperatures_K,
    ))
    inactive = tank.derivative(
        subcooled,
        *empty_flows(tank),
        tank.thermo(subcooled).liquid[0].temperature_K,
        direct_liquid_heat_W=(100.0, 100.0, 100.0),
        direct_vapor_heat_W=(0.0,),
    )
    assert inactive.distributed_boiling_rates_kg_s == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("crossing", ["liquid", "vapor"])
def test_phase_change_projection_closes_pressure_projected_crossing(crossing):
    """Both phase labels remain valid after a rigid-volume projection.

    A pressure solve can move every near-saturated cell a small distance across
    its old saturation enthalpy.  The two-sided projection must move those
    cells onto the new manifold and keep the projected enthalpy ledger fixed.
    """
    tank = make_tank(
        3,
        3,
        distributed_boiling_model="phase_change_complementarity",
    )
    initial = tank.initialize_saturated(101_325.0, 0.5)
    liquid_h = list(initial.liquid_specific_enthalpies_J_kg)
    vapor_h = list(initial.vapor_specific_enthalpies_J_kg)
    if crossing == "liquid":
        liquid_h[0] += 100.0
    else:
        vapor_h[0] -= 100.0
    candidate = tank.project_pressure(LayeredTankState(
        initial.pressure_Pa,
        initial.liquid_masses_kg,
        tuple(liquid_h),
        initial.vapor_masses_kg,
        tuple(vapor_h),
        initial.wall_temperatures_K,
    ))
    masses = (*candidate.liquid_masses_kg, *candidate.vapor_masses_kg)
    enthalpies = (
        *candidate.liquid_specific_enthalpies_J_kg,
        *candidate.vapor_specific_enthalpies_J_kg,
    )
    target_enthalpy_ledger_J = (
        sum(mass * enthalpy for mass, enthalpy in zip(masses, enthalpies))
        - candidate.pressure_Pa * tank.shape.volume_m3
    )

    projected = tank._project_distributed_boiling_manifold(candidate)
    thermo = tank.thermo(projected)
    final_masses = (*projected.liquid_masses_kg, *projected.vapor_masses_kg)
    final_enthalpies = (
        *projected.liquid_specific_enthalpies_J_kg,
        *projected.vapor_specific_enthalpies_J_kg,
    )
    final_enthalpy_ledger_J = (
        sum(mass * enthalpy for mass, enthalpy in zip(final_masses, final_enthalpies))
        - projected.pressure_Pa * tank.shape.volume_m3
    )

    assert thermo.volume_residual_m3 == pytest.approx(0.0, abs=2e-9)
    assert final_enthalpy_ledger_J == pytest.approx(target_enthalpy_ledger_J, abs=2e-5)
    assert all(cell.quality in (None, 0.0) for cell in thermo.liquid)
    assert all(cell.quality in (None, 1.0) for cell in thermo.vapor)


def test_phase_change_projection_rejects_a_complete_branch_crossing():
    tank = make_tank(
        1,
        1,
        distributed_boiling_model="phase_change_complementarity",
    )
    initial = tank.initialize_saturated(101_325.0, 0.5)
    latent = (
        tank.properties.saturated_vapor(initial.pressure_Pa).specific_enthalpy_J_kg
        - tank.properties.saturated_liquid(initial.pressure_Pa).specific_enthalpy_J_kg
    )
    invalid = LayeredTankState(
        initial.pressure_Pa,
        initial.liquid_masses_kg,
        (initial.liquid_specific_enthalpies_J_kg[0] + 1.1 * latent,),
        initial.vapor_masses_kg,
        initial.vapor_specific_enthalpies_J_kg,
        initial.wall_temperatures_K,
    )
    with pytest.raises(ValueError, match="reduce the integration time step"):
        tank._project_distributed_boiling_manifold(invalid)


@pytest.mark.parametrize(
    ("direct_liquid_heat_W", "direct_vapor_heat_W", "expected_sign"),
    [(100.0, 0.0, 1.0), (0.0, -100.0, -1.0)],
)
def test_two_sided_phase_change_rate_closure_handles_both_directions(
    direct_liquid_heat_W, direct_vapor_heat_W, expected_sign
):
    tank = make_tank(
        1,
        1,
        distributed_boiling_model="phase_change_complementarity",
    )
    state = tank.initialize_saturated(101_325.0, 0.5)
    temperature = tank.thermo(state).liquid[0].temperature_K
    derivative = tank.derivative(
        state,
        *empty_flows(tank),
        temperature,
        direct_liquid_heat_W=(direct_liquid_heat_W,),
        direct_vapor_heat_W=(direct_vapor_heat_W,),
    )
    latent = (
        tank.properties.saturated_vapor(state.pressure_Pa).specific_enthalpy_J_kg
        - tank.properties.saturated_liquid(state.pressure_Pa).specific_enthalpy_J_kg
    )
    assert derivative.distributed_boiling_rate_kg_s * expected_sign > 0.0
    assert derivative.distributed_boiling_latent_power_W == pytest.approx(
        derivative.distributed_boiling_rate_kg_s * latent, rel=1e-12
    )
    assert derivative.total_mass_rate_kg_s == pytest.approx(0.0, abs=1e-14)
    assert derivative.total_energy_rate_W == pytest.approx(
        direct_liquid_heat_W + direct_vapor_heat_W, abs=1e-10
    )
    assert derivative.boundary_energy_rate_W == pytest.approx(
        direct_liquid_heat_W + direct_vapor_heat_W, abs=1e-12
    )
    if expected_sign > 0.0:
        slope = tank.properties.saturation_enthalpy_pressure_derivative_J_kg_Pa(
            state.pressure_Pa, 0.0
        )
        assert derivative.liquid_enthalpy_rates_J_kg_s[0] == pytest.approx(
            slope * derivative.pressure_Pa_s, rel=1e-12
        )
    else:
        slope = tank.properties.saturation_enthalpy_pressure_derivative_J_kg_Pa(
            state.pressure_Pa, 1.0
        )
        assert derivative.vapor_enthalpy_rates_J_kg_s[0] == pytest.approx(
            slope * derivative.pressure_Pa_s, rel=1e-12
        )


def test_two_sided_phase_change_closes_vapor_condensation_during_outflow():
    """An explicit vapor outflow may cool the ullage through condensation.

    The optional two-sided closure must keep the labelled vapor cell on its
    valid branch while retaining the open-system mass and energy ledgers.  This
    is the physical boundary encountered by a closed-seat leakage diagnostic;
    the production default remains the historical one-sided closure.
    """
    tank = make_tank(
        1,
        1,
        distributed_boiling_model="phase_change_complementarity",
    )
    initial = tank.initialize_saturated(101_325.0, 0.5)
    outflow_kg_s = 1.0e-4

    def flow_callback(_time, thermo):
        vapor_flow = (MassEnergyFlow(
            -outflow_kg_s,
            thermo.vapor[-1].specific_enthalpy_J_kg,
            "vapor_outflow",
        ),)
        return (tuple(()),), (vapor_flow,)

    result = tank.simulate(
        initial,
        duration_s=2.0,
        time_step_s=0.2,
        ambient_temperature_K=initial.wall_temperatures_K[0],
        flow_callback=flow_callback,
        minimum_step_s=0.02,
    )[-1]

    assert result.thermo.vapor[0].quality in (None, 1.0)
    assert result.total_mass_residual_kg == pytest.approx(0.0, abs=1.0e-11)
    assert result.total_energy_residual_J == pytest.approx(0.0, abs=20.0)
    assert result.cumulative_boundary_mass_kg == pytest.approx(
        -outflow_kg_s * 2.0, rel=1.0e-10
    )
