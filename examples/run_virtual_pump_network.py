"""Illustrative virtual-LH2-pump and pipe network with no fitted parameters."""

from __future__ import annotations

import json

from lh2dt import HydrogenProperties, Pipe, SteadyNetwork, VirtualPump


def main() -> None:
    properties = HydrogenProperties()
    source = properties.from_pT(200_000.0, 20.0)
    demand = properties.from_ph(250_000.0, source.specific_enthalpy_J_kg)
    pump = VirtualPump(
        shutoff_pressure_rise_Pa=300_000.0,
        runout_mass_flow_kg_s=0.2,
        isentropic_efficiency=0.7,
        properties=properties,
        npsh_required_m=5.0,
    )
    network = SteadyNetwork(properties)
    network.add_boundary("source", source)
    network.add_boundary("demand", demand)
    network.add_junction("pump_discharge", 350_000.0, source.specific_enthalpy_J_kg)
    network.add_link("virtual_pump", "source", "pump_discharge", pump)
    network.add_link(
        "supply_pipe",
        "pump_discharge",
        "demand",
        Pipe(
            length_m=20.0,
            inner_diameter_m=0.02,
            roughness_m=1.0e-6,
            local_loss_coefficient=2.0,
            properties=properties,
        ),
    )
    solution = network.solve()
    pump_result = pump.evaluate(source, solution.node_states["pump_discharge"])
    print(json.dumps({
        "note": "illustrative design parameters; not center equipment values",
        "pressure_solver_success": solution.pressure_solver_success,
        "discharge_pressure_Pa": solution.node_states["pump_discharge"].pressure_Pa,
        "mass_flow_kg_s": pump_result.mass_flow_kg_s,
        "shaft_power_W": pump_result.shaft_power_W,
        "hydraulic_power_W": pump_result.hydraulic_power_W,
        "npsh_available_m": pump_result.npsh_available_m,
        "npsh_required_m": pump_result.npsh_required_m,
        "cavitation_margin_m": pump_result.cavitation_margin_m,
        "node_mass_residual_kg_s": solution.node_mass_residual_kg_s,
        "node_energy_residual_W": solution.node_energy_residual_W,
    }, indent=2))


if __name__ == "__main__":
    main()
