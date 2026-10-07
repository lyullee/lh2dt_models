"""Run a synthetic two-phase HEM pipe into a dynamic equilibrium tank."""

from __future__ import annotations

import json

from lh2dt import (
    DynamicNetworkSimulator,
    HEMPipe,
    HydrogenProperties,
    HomogeneousTank,
    SteadyNetwork,
    TankGeometry,
    homogeneous_tank_node,
)


def main() -> None:
    properties = HydrogenProperties()
    geometry = TankGeometry(
        volume_m3=1.0,
        wall_heat_capacity_J_K=1.0e8,
        ambient_UA_W_K=1.0e-12,
        fluid_wall_UA_W_K=1.0e-12,
    )
    tank = HomogeneousTank(geometry, properties)
    initial = tank.initialize_saturated(200_000.0, 0.5, wall_temperature_K=20.0)
    liquid = properties.saturated_liquid(350_000.0)
    vapor = properties.saturated_vapor(350_000.0)
    source = properties.from_ph(
        350_000.0,
        liquid.specific_enthalpy_J_kg + 0.7 * (
            vapor.specific_enthalpy_J_kg - liquid.specific_enthalpy_J_kg
        ),
    )
    network = SteadyNetwork(properties)
    network.add_boundary("source", source)
    network.add_boundary("tank", tank.thermo(initial))
    network.add_link(
        "hem_fill",
        "source",
        "tank",
        HEMPipe(
            length_m=5.0,
            inner_diameter_m=0.02,
            roughness_m=1.0e-6,
            local_loss_coefficient=2.0,
            properties=properties,
        ),
    )
    simulator = DynamicNetworkSimulator(
        network,
        {"tank": homogeneous_tank_node("tank", tank, initial, 20.0)},
    )
    result = simulator.simulate(duration_s=0.2, time_step_s=0.05)[-1]
    final_state = result.states["tank"]
    initial_total_energy = (
        initial.hydrogen_internal_energy_J
        + geometry.wall_heat_capacity_J_K * initial.wall_temperature_K
    )
    final_total_energy = (
        final_state.hydrogen_internal_energy_J
        + geometry.wall_heat_capacity_J_K * final_state.wall_temperature_K
    )
    print(json.dumps({
        "note": "synthetic HEM two-phase linkage; no center-data fitting",
        "time_s": result.time_s,
        "initial_quality": tank.thermo(initial).quality,
        "final_quality": result.thermo_states["tank"].quality,
        "hem_pipe_mass_flow_kg_s": result.network_solution.link_flows["hem_fill"].mass_flow_kg_s,
        "cumulative_boundary_mass_kg": result.cumulative_boundary_mass_kg["tank"],
        "mass_residual_kg": final_state.hydrogen_mass_kg
        - initial.hydrogen_mass_kg
        - result.cumulative_boundary_mass_kg["tank"],
        "cumulative_boundary_energy_J": result.cumulative_boundary_energy_J["tank"],
        "energy_residual_J": final_total_energy
        - initial_total_energy
        - result.cumulative_boundary_energy_J["tank"],
        "pressure_solver_success": result.network_solution.pressure_solver_success,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
