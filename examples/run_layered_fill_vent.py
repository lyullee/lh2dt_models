"""Minimal explicit-port fill/vent scenario for the layered tank."""

from __future__ import annotations

import json

from lh2dt import (
    HorizontalVesselGeometry,
    HydrogenProperties,
    LayeredTank,
    LayeredTankParameters,
    LayeredTankScenario,
    LayeredTankState,
    LayeredValveBoundaryConnection,
    Valve,
)


def main() -> None:
    properties = HydrogenProperties()
    tank = LayeredTank(
        HorizontalVesselGeometry(1.0, 2.0, 0.25),
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
        tuple(
            value + 5_000.0
            for value in saturated.vapor_specific_enthalpies_J_kg
        ),
        saturated.wall_temperatures_K,
    ))
    fill_source = properties.from_pT(300_000.0, 20.0)
    vent_sink = properties.from_pT(100_000.0, 80.0)

    def small_valve() -> Valve:
        return Valve(
            1.0e-8, 0.8, properties,
            pressure_samples=16, allow_reverse=False,
        )

    scenario = LayeredTankScenario(tank, [
        LayeredValveBoundaryConnection(
            "bottom-fill", fill_source, small_valve(), "left", "liquid", 0,
            lambda time_s, _cell: 1.0 if time_s < 0.5 else 0.0,
        ),
        LayeredValveBoundaryConnection(
            "top-vent", vent_sink, small_valve(), "right", "vapor", 1,
            lambda time_s, _cell: 0.0 if time_s < 0.5 else 1.0,
        ),
    ])
    results = scenario.simulate(
        initial,
        duration_s=1.0,
        time_step_s=0.1,
        ambient_temperature_K=tank.thermo(initial).liquid[0].temperature_K,
    )
    final = results[-1]
    print(json.dumps({
        "note": "numerical connection example; not center boundary data",
        "final_pressure_Pa": final.state.pressure_Pa,
        "cumulative_boundary_mass_kg": final.cumulative_boundary_mass_kg,
        "cumulative_boundary_energy_J": final.cumulative_boundary_energy_J,
        "total_mass_residual_kg": final.total_mass_residual_kg,
        "total_energy_residual_J": final.total_energy_residual_J,
    }, indent=2))


if __name__ == "__main__":
    main()
