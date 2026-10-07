"""Run the conservative 2 x N liquid/vapor tank integration check."""

from __future__ import annotations

import json
import math
from pathlib import Path

from lh2dt import (
    CirculationLegGeometry,
    HydrogenProperties,
    RadialAxialBoundaryGeometry,
    RadialAxialTank,
    RadialAxialTankParameters,
    RadialAxialTankState,
)


def boundary(height_m: float) -> RadialAxialBoundaryGeometry:
    diameter = 0.025
    area = math.pi * diameter**2 / 4.0
    leg = CirculationLegGeometry(0.7, diameter, area, 1.0e-6, 1.2)
    return RadialAxialBoundaryGeometry(height_m, leg, leg)


def main() -> None:
    properties = HydrogenProperties()
    pressure = 300_000.0
    saturation_temperature = properties.saturated_liquid(pressure).temperature_K
    tank = RadialAxialTank(
        RadialAxialTankParameters(
            total_volume_m3=1.2,
            liquid_level_count=2,
            vapor_level_count=2,
            liquid_wall_volume_fraction=0.25,
            vapor_wall_volume_fraction=0.30,
            liquid_boundaries=(boundary(0.5),),
            vapor_boundaries=(boundary(0.5),),
            wall_heat_capacity_J_K=20_000.0,
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
    base = tank.initialize_uniform(
        pressure,
        liquid_volume_fraction=0.55,
        liquid_temperature_K=saturation_temperature - 0.35,
        vapor_temperature_K=saturation_temperature + 0.45,
        wall_temperature_K=saturation_temperature + 0.15,
    )

    def enthalpies(temperatures_K: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            properties.from_pT(pressure, temperature).specific_enthalpy_J_kg
            for temperature in temperatures_K
        )

    state = tank.project_state(RadialAxialTankState(
        base.pressure_Pa,
        base.liquid_wall_masses_kg,
        enthalpies((saturation_temperature - 0.30, saturation_temperature - 0.20)),
        base.liquid_core_masses_kg,
        enthalpies((saturation_temperature - 0.65, saturation_temperature - 0.55)),
        base.vapor_wall_masses_kg,
        enthalpies((saturation_temperature + 0.65, saturation_temperature + 0.85)),
        base.vapor_core_masses_kg,
        enthalpies((saturation_temperature + 0.35, saturation_temperature + 0.45)),
        (
            saturation_temperature + 0.25,
            saturation_temperature + 0.30,
            saturation_temperature + 0.40,
            saturation_temperature + 0.50,
        ),
    ))
    initial_mass = tank.total_mass_kg(state)
    initial_energy = tank.total_energy_J(state)
    initial_pressure = state.pressure_Pa
    maximum_volume_residual = 0.0
    maximum_step_mass_residual = 0.0
    maximum_step_energy_residual = 0.0
    cumulative_interface_transfer = [0.0, 0.0]
    first_derivative = None
    time_step = 0.01
    for _ in range(100):
        result = tank.step_euler(
            state,
            time_step_s=time_step,
            ambient_temperature_K=state.wall_temperatures_K[0],
        )
        if first_derivative is None:
            first_derivative = result.derivative
        for index, rate in enumerate(result.derivative.interface_phase_change_rates_kg_s):
            cumulative_interface_transfer[index] += rate * time_step
        maximum_volume_residual = max(
            maximum_volume_residual, abs(result.thermo.volume_residual_m3)
        )
        maximum_step_mass_residual = max(
            maximum_step_mass_residual, abs(result.total_mass_residual_kg)
        )
        maximum_step_energy_residual = max(
            maximum_step_energy_residual, abs(result.total_energy_residual_J)
        )
        state = result.state

    assert first_derivative is not None
    final_masses = tank.cell_masses_kg(state)
    report = {
        "model": "RadialAxialTank",
        "topology": {
            "liquid_grid": "2 radial columns x 2 axial levels",
            "vapor_grid": "2 radial columns x 2 axial levels",
            "wall_patches": 4,
            "common_pressure": True,
        },
        "integration": {
            "duration_s": 1.0,
            "time_step_s": time_step,
            "steps": 100,
        },
        "initial": {
            "pressure_Pa": initial_pressure,
            "total_mass_kg": initial_mass,
            "total_energy_J": initial_energy,
        },
        "first_derivative": {
            "liquid_boundary_mass_flows_kg_s": list(
                first_derivative.liquid_transport.boundary_mass_flows_kg_s
            ),
            "vapor_boundary_mass_flows_kg_s": list(
                first_derivative.vapor_transport.boundary_mass_flows_kg_s
            ),
            "interface_phase_change_rates_kg_s": list(
                first_derivative.interface_phase_change_rates_kg_s
            ),
            "differentiated_volume_residual_m3_s": (
                first_derivative.common_pressure.differentiated_volume_residual_m3_s
            ),
            "mass_conservation_residual_kg_s": (
                first_derivative.mass_conservation_residual_kg_s
            ),
            "energy_conservation_residual_W": (
                first_derivative.energy_conservation_residual_W
            ),
        },
        "final": {
            "pressure_Pa": state.pressure_Pa,
            "pressure_change_Pa": state.pressure_Pa - initial_pressure,
            "total_mass_change_kg": tank.total_mass_kg(state) - initial_mass,
            "total_energy_change_J": tank.total_energy_J(state) - initial_energy,
            "cumulative_interface_transfer_kg": cumulative_interface_transfer,
            "minimum_cell_mass_kg": min(final_masses),
            "maximum_volume_residual_m3": maximum_volume_residual,
            "maximum_step_mass_residual_kg": maximum_step_mass_residual,
            "maximum_step_energy_residual_J": maximum_step_energy_residual,
        },
    }
    output = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "benchmarks"
        / "radial_axial_tank_integration_validation.json"
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
