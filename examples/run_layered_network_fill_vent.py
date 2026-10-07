"""Synthetic layered-tank fill/vent path using explicit network cell ports."""

from __future__ import annotations

import json

from lh2dt import (
    HydrogenProperties,
    HorizontalVesselGeometry,
    LayeredTank,
    LayeredTankNetworkSimulator,
    LayeredTankParameters,
    LayeredTankPort,
    LayeredTankState,
    CommandedValve,
    SteadyNetwork,
    Valve,
)


def main() -> None:
    properties = HydrogenProperties()
    tank = LayeredTank(
        HorizontalVesselGeometry(1.0, 2.0, 0.25, quadrature_order=40),
        LayeredTankParameters(
            liquid_cell_count=2,
            vapor_cell_count=2,
            wall_node_count=4,
            wall_heat_capacity_J_K=1.0e6,
        ),
        properties,
    )
    saturated = tank.initialize_saturated(200_000.0, 0.5)
    initial = tank.project_pressure(LayeredTankState(
        saturated.pressure_Pa,
        saturated.liquid_masses_kg,
        saturated.liquid_specific_enthalpies_J_kg,
        saturated.vapor_masses_kg,
        tuple(value + 5_000.0 for value in saturated.vapor_specific_enthalpies_J_kg),
        saturated.wall_temperatures_K,
    ))
    source = properties.from_pT(300_000.0, 20.0)
    sink = properties.from_pT(100_000.0, 80.0)
    network = SteadyNetwork(properties)
    thermo = tank.thermo(initial)
    network.add_boundary("source", source)
    network.add_boundary("liquid_port", thermo.liquid[0])
    network.add_boundary("vapor_port", thermo.vapor[1])
    network.add_boundary("vent_sink", sink)
    fill_valve = CommandedValve(
        Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False)
    )
    vent_valve = CommandedValve(
        Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False)
    )
    network.add_link(
        "fill", "source", "liquid_port",
        fill_valve,
        source_port_id="SOURCE:OUT", target_port_id="TANK:L0",
    )
    network.add_link(
        "vent", "vapor_port", "vent_sink",
        vent_valve,
        source_port_id="TANK:V1", target_port_id="VENT:IN",
    )
    def command(time_s: float, _network: SteadyNetwork) -> None:
        fill_valve.set_opening(1.0 if time_s < 0.25 else 0.0)
        vent_valve.set_opening(0.0 if time_s < 0.25 else 1.0)

    simulator = LayeredTankNetworkSimulator(
        tank,
        network,
        ports=(
            LayeredTankPort("liquid_port", "liquid", 0),
            LayeredTankPort("vapor_port", "vapor", 1),
        ),
        ambient_temperature_K=20.0,
        command_callback=command,
    )
    result = simulator.simulate(initial, duration_s=0.5, time_step_s=0.1)[-1]
    print(json.dumps({
        "note": "synthetic design parameters; no center data fitting",
        "time_s": result.tank.time_s,
        "pressure_Pa": result.tank.thermo.pressure_Pa,
        "cumulative_boundary_mass_kg": result.tank.cumulative_boundary_mass_kg,
        "total_mass_residual_kg": result.tank.total_mass_residual_kg,
        "total_energy_residual_J": result.tank.total_energy_residual_J,
        "link_mass_flow_kg_s": {
            name: flow.mass_flow_kg_s
            for name, flow in result.network_solution.link_flows.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
