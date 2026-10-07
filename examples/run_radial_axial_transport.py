"""Validate conservative circulation on a three-level wall/core grid."""

from __future__ import annotations

import json
import math
from pathlib import Path

from lh2dt import (
    CirculationLegGeometry,
    HydrogenProperties,
    RadialAxialBoundaryGeometry,
    RadialAxialCirculationTransport,
    RadialAxialPhaseState,
)


def main() -> None:
    properties = HydrogenProperties()
    pressure = 500_000.0

    def h(temperature_K: float) -> float:
        return properties.from_pT(
            pressure, temperature_K
        ).specific_enthalpy_J_kg

    diameter = 0.02
    area = math.pi * diameter**2 / 4.0
    leg = CirculationLegGeometry(
        length_m=2.0,
        hydraulic_diameter_m=diameter,
        flow_area_m2=area,
        roughness_m=1.0e-6,
        local_loss_coefficient=1.0,
    )
    boundaries = (
        RadialAxialBoundaryGeometry(0.8, leg, leg),
        RadialAxialBoundaryGeometry(1.1, leg, leg),
    )
    state = RadialAxialPhaseState(
        wall_masses_kg=(2.0, 2.5, 3.0),
        wall_specific_enthalpies_J_kg=(h(21.0), h(21.2), h(21.4)),
        core_masses_kg=(3.0, 3.5, 4.0),
        core_specific_enthalpies_J_kg=(h(20.0), h(20.1), h(20.2)),
    )
    result = RadialAxialCirculationTransport(
        boundaries, properties=properties
    ).evaluate(pressure, state)

    dt = 0.5
    old_masses = (*state.wall_masses_kg, *state.core_masses_kg)
    old_h = (
        *state.wall_specific_enthalpies_J_kg,
        *state.core_specific_enthalpies_J_kg,
    )
    mass_rates = (
        *result.wall_mass_rates_kg_s,
        *result.core_mass_rates_kg_s,
    )
    powers = (
        *result.wall_enthalpy_flow_W,
        *result.core_enthalpy_flow_W,
    )
    new_masses = tuple(
        mass + dt * rate for mass, rate in zip(old_masses, mass_rates)
    )
    old_enthalpy_inventory = sum(
        mass * enthalpy for mass, enthalpy in zip(old_masses, old_h)
    )
    new_enthalpy_inventory = sum(
        mass * enthalpy + dt * power
        for mass, enthalpy, power in zip(old_masses, old_h, powers)
    )

    payload = {
        "source": {
            "title": "Multi-node Modeling of Cryogenic Tank Pressurization",
            "url": "https://ntrs.nasa.gov/citations/20200000048",
        },
        "model": (
            "three axial levels by two radial columns; common reference "
            "pressure; momentum-derived paired wall/core circulation; "
            "donor-cell enthalpy transport"
        ),
        "calibration": "none",
        "geometry_note": (
            "illustrative verification geometry; not center equipment data"
        ),
        "boundary_mass_flows_kg_s": result.boundary_mass_flows_kg_s,
        "wall_mass_rates_kg_s": result.wall_mass_rates_kg_s,
        "core_mass_rates_kg_s": result.core_mass_rates_kg_s,
        "wall_enthalpy_flow_W": result.wall_enthalpy_flow_W,
        "core_enthalpy_flow_W": result.core_enthalpy_flow_W,
        "level_mass_residuals_kg_s": result.level_mass_residuals_kg_s,
        "total_mass_residual_kg_s": result.total_mass_residual_kg_s,
        "total_enthalpy_flow_residual_W": (
            result.total_enthalpy_flow_residual_W
        ),
        "finite_advance_check": {
            "time_step_s": dt,
            "minimum_new_cell_mass_kg": min(new_masses),
            "mass_inventory_change_kg": sum(new_masses) - sum(old_masses),
            "enthalpy_inventory_change_J": (
                new_enthalpy_inventory - old_enthalpy_inventory
            ),
        },
        "integration_gate": (
            "The next tank layer must combine these internal sources with "
            "the common-pressure rigid-volume DAE, wall heat, interface phase "
            "change, and external ports before NASA D-4171 comparison."
        ),
    }
    output = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "benchmarks"
        / "radial_axial_transport_validation.json"
    )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
