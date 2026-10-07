"""Synthetic two-region tank fill/vent path with separate phase ports."""

from __future__ import annotations

import json

from lh2dt import (
    CommandedValve,
    HydrogenProperties,
    HorizontalVesselGeometry,
    StratifiedTank,
    StratifiedTankNetworkSimulator,
    StratifiedTankParameters,
    StratifiedTankPort,
    SteadyNetwork,
    Valve,
)


def main() -> None:
    properties = HydrogenProperties()
    tank = StratifiedTank(
        HorizontalVesselGeometry(2.0, 4.0, 0.5, quadrature_order=48),
        StratifiedTankParameters(
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
        ),
        properties,
    )
    initial = tank.initialize_saturated(200_000.0, 0.5)
    thermo = tank.thermo(initial)
    network = SteadyNetwork(properties)
    network.add_boundary("source", properties.from_pT(300_000.0, 20.0))
    network.add_boundary("liquid_port", thermo.liquid)
    network.add_boundary("vapor_port", thermo.vapor)
    network.add_boundary("vent_sink", properties.from_pT(100_000.0, 80.0))
    fill = CommandedValve(Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False))
    vent = CommandedValve(Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False))
    network.add_link("fill", "source", "liquid_port", fill)
    network.add_link("vent", "vapor_port", "vent_sink", vent)

    def command(time_s: float, _network: SteadyNetwork) -> None:
        fill.set_opening(1.0 if time_s < 0.25 else 0.0)
        vent.set_opening(0.0 if time_s < 0.25 else 1.0)

    simulator = StratifiedTankNetworkSimulator(
        tank,
        network,
        (StratifiedTankPort("liquid_port", "liquid"), StratifiedTankPort("vapor_port", "vapor")),
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

