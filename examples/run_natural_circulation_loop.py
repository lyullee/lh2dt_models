"""Validate the coefficient-free natural-circulation loop building block."""

from __future__ import annotations

import json
import math
from pathlib import Path

from lh2dt import (
    CirculationLegGeometry,
    HydrogenProperties,
    NaturalCirculationLoop,
)


def main() -> None:
    properties = HydrogenProperties()
    diameter = 0.02
    area = math.pi * diameter**2 / 4.0
    geometry = CirculationLegGeometry(
        length_m=2.0,
        hydraulic_diameter_m=diameter,
        flow_area_m2=area,
        roughness_m=1.0e-6,
        local_loss_coefficient=1.0,
    )
    loop = NaturalCirculationLoop(
        vertical_separation_m=1.0,
        rising_leg=geometry,
        descending_leg=geometry,
        properties=properties,
    )
    warm = properties.from_pT(500_000.0, 21.0)
    cold = properties.from_pT(500_000.0, 20.0)
    forward = loop.evaluate(warm, cold)
    reverse = loop.evaluate(cold, warm)
    still = loop.evaluate(cold, cold)

    laminar_diameter = 0.005
    laminar_area = math.pi * laminar_diameter**2 / 4.0
    laminar_leg = CirculationLegGeometry(
        length_m=2.0,
        hydraulic_diameter_m=laminar_diameter,
        flow_area_m2=laminar_area,
    )
    laminar_loop = NaturalCirculationLoop(
        1.0, laminar_leg, laminar_leg, properties=properties
    )
    slightly_warm = properties.from_pT(500_000.0, 20.0001)
    laminar = laminar_loop.evaluate(slightly_warm, cold)
    linear_resistance = 32.0 * (
        properties.viscosity_Pa_s(slightly_warm)
        * laminar_leg.length_m
        / (
            slightly_warm.density_kg_m3
            * laminar_area
            * laminar_diameter**2
        )
        + properties.viscosity_Pa_s(cold)
        * laminar_leg.length_m
        / (cold.density_kg_m3 * laminar_area * laminar_diameter**2)
    )
    analytic_laminar_flow = (
        laminar.buoyancy_pressure_Pa / linear_resistance
    )

    def compact(result) -> dict:
        return {
            "mass_flow_kg_s": result.mass_flow_kg_s,
            "buoyancy_pressure_Pa": result.buoyancy_pressure_Pa,
            "friction_pressure_Pa": result.friction_pressure_Pa,
            "hydraulic_residual_Pa": result.hydraulic_residual_Pa,
            "rising_leg_reynolds_number": (
                result.rising_leg_reynolds_number
            ),
            "descending_leg_reynolds_number": (
                result.descending_leg_reynolds_number
            ),
            "lower_control_volume_heat_W": (
                result.lower_control_volume_heat_W
            ),
            "upper_control_volume_heat_W": (
                result.upper_control_volume_heat_W
            ),
            "mass_residual_kg_s": result.mass_residual_kg_s,
            "energy_residual_W": result.energy_residual_W,
        }

    payload = {
        "source": {
            "title": "Multi-node Modeling of Cryogenic Tank Pressurization",
            "url": "https://ntrs.nasa.gov/citations/20200000048",
            "architectural_use": (
                "radial/axial nodes connected by mass, momentum, and energy "
                "balances to permit natural-convection recirculation"
            ),
        },
        "model": (
            "two-leg single-phase loop; exact density-difference buoyancy "
            "head; Darcy-Weisbach and local losses; conservative paired "
            "enthalpy transport"
        ),
        "calibration": "none",
        "geometry_note": (
            "illustrative verification geometry; not center equipment data"
        ),
        "cases": {
            "warm_rising_leg": compact(forward),
            "direction_reversal": compact(reverse),
            "equal_density_zero_flow": compact(still),
            "laminar_limit": {
                **compact(laminar),
                "analytic_hagen_poiseuille_mass_flow_kg_s": (
                    analytic_laminar_flow
                ),
                "relative_error": abs(
                    laminar.mass_flow_kg_s / analytic_laminar_flow - 1.0
                ),
            },
        },
        "integration_gate": (
            "Do not connect this loop to the current one-column layered tank "
            "until separate wall-layer and core fluid states plus hydraulic "
            "areas and loss coefficients are defined."
        ),
    }
    output = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "benchmarks"
        / "natural_circulation_loop_validation.json"
    )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["cases"], indent=2))


if __name__ == "__main__":
    main()
