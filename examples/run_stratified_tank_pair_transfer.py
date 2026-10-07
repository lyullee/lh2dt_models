"""Synthetic two-tank transfer path for the parallel-storage contract."""

from __future__ import annotations

import json

from lh2dt import (
    HydrogenProperties,
    HorizontalVesselGeometry,
    StratifiedTank,
    StratifiedTankEnsemblePort,
    StratifiedTankNetworkEnsembleSimulator,
    StratifiedTankParameters,
    SteadyNetwork,
    Valve,
)


def _make_tank(properties: HydrogenProperties) -> StratifiedTank:
    return StratifiedTank(
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


def main() -> None:
    properties = HydrogenProperties()
    tank_a = _make_tank(properties)
    tank_b = _make_tank(properties)
    state_a = tank_a.initialize_saturated(200_000.0, 0.5)
    state_b = tank_b.initialize_saturated(100_000.0, 0.5)
    network = SteadyNetwork(properties)
    network.add_boundary("tank_a_liquid", tank_a.thermo(state_a).liquid)
    network.add_boundary("tank_b_liquid", tank_b.thermo(state_b).liquid)
    network.add_link(
        "tank_transfer",
        "tank_a_liquid",
        "tank_b_liquid",
        Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False),
    )
    simulator = StratifiedTankNetworkEnsembleSimulator(
        {"tank_a": tank_a, "tank_b": tank_b},
        network,
        (
            StratifiedTankEnsemblePort("tank_a", "tank_a_liquid", "liquid"),
            StratifiedTankEnsemblePort("tank_b", "tank_b_liquid", "liquid"),
        ),
        ambient_temperature_K=20.0,
    )
    result = simulator.simulate(
        {"tank_a": state_a, "tank_b": state_b},
        duration_s=1.0,
        time_step_s=0.2,
    )[-1]
    total_mass_residual = sum(abs(value) for value in result.total_mass_residual_kg.values())
    total_energy_residual = sum(abs(value) for value in result.total_energy_residual_J.values())
    print(json.dumps({
        "note": "synthetic design parameters; no center data fitting",
        "time_s": result.time_s,
        "tank_pressure_Pa": {
            name: thermo.pressure_Pa for name, thermo in result.thermo_states.items()
        },
        "tank_boundary_mass_kg": result.cumulative_boundary_mass_kg,
        "total_mass_residual_kg": total_mass_residual,
        "total_energy_residual_J": total_energy_residual,
        "link_mass_flow_kg_s": {
            name: flow.mass_flow_kg_s
            for name, flow in result.network_solution.link_flows.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()

