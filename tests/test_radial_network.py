import math

import pytest

from lh2dt import (
    CirculationLegGeometry,
    HydrogenProperties,
    RadialAxialBoundaryGeometry,
    RadialAxialTank,
    RadialAxialTankNetworkSimulator,
    RadialAxialTankParameters,
    RadialAxialTankPort,
    SteadyNetwork,
    Valve,
)


def _tank() -> RadialAxialTank:
    diameter = 0.025
    area = math.pi * diameter**2 / 4.0
    leg = CirculationLegGeometry(
        length_m=0.7,
        hydraulic_diameter_m=diameter,
        flow_area_m2=area,
        roughness_m=1.0e-6,
        local_loss_coefficient=1.2,
    )
    boundary = RadialAxialBoundaryGeometry(0.5, leg, leg)
    properties = HydrogenProperties()
    return RadialAxialTank(
        RadialAxialTankParameters(
            total_volume_m3=1.2,
            liquid_level_count=2,
            vapor_level_count=2,
            liquid_wall_volume_fraction=0.25,
            vapor_wall_volume_fraction=0.30,
            liquid_boundaries=(boundary,),
            vapor_boundaries=(boundary,),
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
        ),
        properties,
    )


def test_radial_axial_tank_cell_ports_preserve_network_balance():
    properties = HydrogenProperties()
    tank = _tank()
    saturation_temperature = properties.saturated_liquid(300_000.0).temperature_K
    initial = tank.initialize_uniform(
        300_000.0,
        liquid_volume_fraction=0.55,
        liquid_temperature_K=saturation_temperature - 0.35,
        vapor_temperature_K=saturation_temperature + 0.45,
        wall_temperature_K=saturation_temperature,
    )
    liquid_index = tank.cell_index("liquid", "core", 0)
    vapor_index = tank.cell_index("vapor", "wall", 1)
    source = properties.from_pT(400_000.0, 20.0)
    sink = properties.from_pT(100_000.0, 80.0)
    network = SteadyNetwork(properties)
    thermo = tank.thermo(initial)
    network.add_boundary("source", source)
    network.add_boundary("liquid_port", thermo.cells[liquid_index])
    network.add_boundary("vapor_port", thermo.cells[vapor_index])
    network.add_boundary("sink", sink)
    network.add_link(
        "fill", "source", "liquid_port",
        Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False),
    )
    network.add_link(
        "vent", "vapor_port", "sink",
        Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False),
    )
    simulator = RadialAxialTankNetworkSimulator(
        tank,
        network,
        (RadialAxialTankPort("liquid_port", liquid_index), RadialAxialTankPort("vapor_port", vapor_index)),
        ambient_temperature_K=saturation_temperature,
    )
    results = simulator.simulate(initial, duration_s=0.04, time_step_s=0.02)
    assert len(results) == 2
    assert all(result.network_solution.pressure_solver_success for result in results)
    assert results[-1].tank.total_mass_residual_kg == pytest.approx(0.0, abs=2.0e-10)
    assert results[-1].tank.total_energy_residual_J == pytest.approx(0.0, abs=2.0e-5)
    assert results[-1].tank.thermo.volume_residual_m3 == pytest.approx(0.0, abs=2.0e-10)
    assert results[-1].tank.derivative.boundary_mass_rate_kg_s != 0.0


def test_radial_port_index_rejects_unknown_cell():
    tank = _tank()
    properties = HydrogenProperties()
    state = tank.initialize_uniform(300_000.0, 0.55, 19.0, 22.0, 20.0)
    network = SteadyNetwork(properties)
    network.add_boundary("port", tank.thermo(state).cells[0])
    with pytest.raises(ValueError):
        RadialAxialTankNetworkSimulator(
            tank, network, (RadialAxialTankPort("port", tank.cell_count),), 20.0
        )


def test_radial_command_callback_reads_current_cell_state_before_network_solve():
    tank = _tank()
    properties = HydrogenProperties()
    initial = tank.initialize_uniform(300_000.0, 0.55, 19.0, 22.0, 20.0)
    thermo = tank.thermo(initial)
    cell_index = tank.cell_index("liquid", "core", 0)
    stale = properties.from_pT(100_000.0, 80.0)
    network = SteadyNetwork(properties)
    network.add_boundary("liquid_port", stale)
    observed = []

    def command(_time_s, current_network):
        observed.append(current_network.boundaries["liquid_port"].state.pressure_Pa)

    simulator = RadialAxialTankNetworkSimulator(
        tank,
        network,
        (RadialAxialTankPort("liquid_port", cell_index),),
        ambient_temperature_K=20.0,
        command_callback=command,
    )

    simulator._solve(thermo, 0.0)

    assert observed == [pytest.approx(thermo.cells[cell_index].pressure_Pa)]
