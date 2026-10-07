"""Run synthetic first-principles chains for supply, vaporization, and reliquefaction.

The numbers are deliberately design placeholders.  The example demonstrates
that independently replaceable two-port components share mass and energy
through one network contract; it is not a center-parameter calibration.
"""

from __future__ import annotations

import json

from lh2dt import (
    HydrogenProperties,
    Pipe,
    Reliquefier,
    ReliquefierLink,
    SteadyNetwork,
    Valve,
    Vaporizer,
    VaporizerLink,
    VirtualPump,
)


def _network_summary(solution):
    return {
        "pressure_solver_success": solution.pressure_solver_success,
        "node_mass_residual_kg_s": solution.node_mass_residual_kg_s,
        "node_energy_residual_W": solution.node_energy_residual_W,
        "link_mass_flow_kg_s": {
            name: flow.mass_flow_kg_s for name, flow in solution.link_flows.items()
        },
    }


def main() -> None:
    properties = HydrogenProperties()

    liquid_source = properties.from_pT(600_000.0, 20.0)
    gas_sink = properties.from_pT(300_000.0, 293.15)
    vaporizer_network = SteadyNetwork(properties)
    vaporizer_network.add_boundary("liquid_source", liquid_source)
    vaporizer_network.add_boundary("gas_sink", gas_sink)
    vaporizer_network.add_junction("vaporizer_inlet", 500_000.0, liquid_source.specific_enthalpy_J_kg)
    vaporizer_network.add_link(
        "feed_valve",
        "liquid_source",
        "vaporizer_inlet",
        Valve(2.0e-7, 0.8, properties, pressure_samples=32),
    )
    vaporizer = VaporizerLink(
        Valve(2.0e-7, 0.8, properties, pressure_samples=32),
        Vaporizer(500.0, 100_000.0, properties, integration_segments=32),
        ambient_temperature_K=290.0,
        properties=properties,
    )
    vaporizer_network.add_link("vaporizer", "vaporizer_inlet", "gas_sink", vaporizer)
    vaporizer_solution = vaporizer_network.solve()
    vaporizer_result = vaporizer.evaluate(
        vaporizer_solution.node_states["vaporizer_inlet"], gas_sink
    )

    bog_source = properties.from_pT(600_000.0, 80.0)
    liquid_sink = properties.from_pT(200_000.0, 20.0)
    reliquefier_network = SteadyNetwork(properties)
    reliquefier_network.add_boundary("bog_source", bog_source)
    reliquefier_network.add_boundary("liquid_sink", liquid_sink)
    reliquefier = ReliquefierLink(
        Valve(2.0e-7, 0.8, properties, pressure_samples=32),
        Reliquefier(20_000.0, properties),
        properties=properties,
    )
    reliquefier_network.add_link("reliquefier", "bog_source", "liquid_sink", reliquefier)
    reliquefier_solution = reliquefier_network.solve()
    reliquefier_result = reliquefier.evaluate(bog_source, liquid_sink)

    pump_source = properties.from_pT(200_000.0, 20.0)
    pump_sink = properties.from_ph(250_000.0, pump_source.specific_enthalpy_J_kg)
    pump_network = SteadyNetwork(properties)
    pump_network.add_boundary("pump_source", pump_source)
    pump_network.add_boundary("pump_sink", pump_sink)
    pump_network.add_junction("pump_discharge", 350_000.0, pump_source.specific_enthalpy_J_kg)
    pump = VirtualPump(300_000.0, 0.2, 0.7, properties, npsh_required_m=5.0)
    pump_network.add_link("virtual_pump", "pump_source", "pump_discharge", pump)
    pump_network.add_link(
        "supply_pipe",
        "pump_discharge",
        "pump_sink",
        Pipe(20.0, 0.02, 1.0e-6, local_loss_coefficient=2.0, properties=properties),
    )
    pump_solution = pump_network.solve()
    pump_result = pump.evaluate(pump_source, pump_solution.node_states["pump_discharge"])

    print(json.dumps({
        "note": "synthetic design parameters only; no center data fitting",
        "vaporizer_chain": {
            **_network_summary(vaporizer_solution),
            "thermal_power_W": vaporizer_result.thermal_power_W,
        },
        "reliquefier_chain": {
            **_network_summary(reliquefier_solution),
            "thermal_power_W": reliquefier_result.thermal_power_W,
        },
        "pump_chain": {
            **_network_summary(pump_solution),
            "shaft_power_W": pump_result.shaft_power_W,
            "npsh_available_m": pump_result.npsh_available_m,
            "cavitation_margin_m": pump_result.cavitation_margin_m,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
