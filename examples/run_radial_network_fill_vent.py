"""Synthetic radial/axial tank fill-to-vent network example."""

from __future__ import annotations

import json
import math

from lh2dt import (
    CirculationLegGeometry,
    CommandedValve,
    HydrogenProperties,
    RadialAxialBoundaryGeometry,
    RadialAxialTank,
    RadialAxialTankNetworkSimulator,
    RadialAxialTankParameters,
    RadialAxialTankPort,
    SteadyNetwork,
    Valve,
)


def main() -> None:
    properties = HydrogenProperties()
    diameter = 0.025
    area = math.pi * diameter**2 / 4.0
    leg = CirculationLegGeometry(0.7, diameter, area, 1.0e-6, 1.2)
    boundary = RadialAxialBoundaryGeometry(0.5, leg, leg)
    tank = RadialAxialTank(
        RadialAxialTankParameters(
            total_volume_m3=1.2,
            liquid_level_count=2,
            vapor_level_count=2,
            liquid_wall_volume_fraction=0.25,
            vapor_wall_volume_fraction=0.30,
            liquid_boundaries=(boundary,), vapor_boundaries=(boundary,),
            wall_heat_capacity_J_K=20_000.0,
            wall_to_fluid_UA_W_K=12.0,
            wall_axial_conductance_W_K=0.8,
            liquid_radial_conductance_W_K=0.7,
            vapor_radial_conductance_W_K=0.2,
            liquid_axial_conductance_W_K=0.4,
            vapor_axial_conductance_W_K=0.1,
            liquid_interface_UA_W_K=(3.0, 2.0),
            vapor_interface_UA_W_K=(1.5, 1.0),
        ), properties,
    )
    saturation_temperature = properties.saturated_liquid(300_000.0).temperature_K
    initial = tank.initialize_uniform(
        300_000.0, 0.55, saturation_temperature - 0.35,
        saturation_temperature + 0.45, saturation_temperature,
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
    fill = CommandedValve(Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False))
    vent = CommandedValve(Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False))
    network.add_link("fill", "source", "liquid_port", fill)
    network.add_link("vent", "vapor_port", "sink", vent)

    def command(time_s: float, _network: SteadyNetwork) -> None:
        fill.set_opening(1.0 if time_s < 0.04 else 0.0)
        vent.set_opening(0.0 if time_s < 0.04 else 1.0)

    simulator = RadialAxialTankNetworkSimulator(
        tank, network,
        (RadialAxialTankPort("liquid_port", liquid_index), RadialAxialTankPort("vapor_port", vapor_index)),
        saturation_temperature, command_callback=command,
    )
    result = simulator.simulate(initial, duration_s=0.08, time_step_s=0.02)[-1]
    print(json.dumps({
        "note": "synthetic design parameters; no center data fitting",
        "pressure_Pa": result.tank.thermo.cells[liquid_index].pressure_Pa,
        "total_mass_residual_kg": result.tank.total_mass_residual_kg,
        "total_energy_residual_J": result.tank.total_energy_residual_J,
        "volume_residual_m3": result.tank.thermo.volume_residual_m3,
        "link_mass_flow_kg_s": {
            name: flow.mass_flow_kg_s
            for name, flow in result.network_solution.link_flows.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
