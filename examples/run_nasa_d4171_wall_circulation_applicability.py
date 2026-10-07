"""Audit Daigle Appendix-A applicability for NASA TN D-4171.

The script evaluates the modified vertical-wall Rayleigh number without
running or fitting the tank model.  It uses the same Table-I heat partition
and initial geometry as the layered benchmark.
"""

from __future__ import annotations

import csv
import json
from importlib.resources import files

import numpy as np

from lh2dt import HorizontalVesselGeometry, HydrogenProperties, LayeredTank
from lh2dt import LayeredTankParameters


BTU_HR_FT2_TO_W_M2 = 3.154590745


def modified_rayleigh(fluid, props, heat_flux_W_m2, height_m):
    return (
        9.80665
        * fluid.density_kg_m3**2
        * props.isobaric_heat_capacity_J_kgK(fluid)
        * props.isobaric_expansion_coefficient_1_K(fluid)
        * abs(heat_flux_W_m2)
        * height_m**4
        / (
            props.viscosity_Pa_s(fluid)
            * props.thermal_conductivity_W_mK(fluid) ** 2
        )
    )


def main():
    props = HydrogenProperties()
    diameter_m = 9.0 * 0.0254
    shape = HorizontalVesselGeometry(
        diameter_m, 0.0, diameter_m / 2.0, quadrature_order=48
    )
    data_path = files("lh2dt").joinpath("data/nasa_tn_d4171_table1.csv")
    rows = []
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))

    for cell_count in (3, 6, 10):
        tank = LayeredTank(
            shape,
            LayeredTankParameters(
                liquid_cell_count=cell_count,
                vapor_cell_count=cell_count,
                wall_node_count=max(8, 2 * cell_count + 2),
                wall_heat_capacity_J_K=1.0,
            ),
            props,
        )
        for record in source:
            fill = float(record["initial_fill_percent"]) / 100.0
            state = tank.initialize_saturated(101_325.0, fill)
            thermo = tank.thermo(state)
            geometry = shape.at_height(thermo.liquid_height_m)
            wet_flux = (
                float(record["wetted_heat_flux_Btu_hr_ft2"])
                * BTU_HR_FT2_TO_W_M2
            )
            dry_flux = (
                float(record["dry_heat_flux_Btu_hr_ft2"])
                * BTU_HR_FT2_TO_W_M2
            )
            average_flux = (
                float(record["average_heat_flux_Btu_hr_ft2"])
                * BTU_HR_FT2_TO_W_M2
            )
            partition_heat = (
                wet_flux * geometry.wetted_inner_area_m2
                + dry_flux * geometry.dry_inner_area_m2
            )
            scale = average_flux * shape.inner_surface_area_m2 / partition_heat
            wet_flux *= scale
            dry_flux *= scale

            centers = np.asarray(thermo.cell_center_heights_m)
            liquid_ra = [
                modified_rayleigh(
                    thermo.liquid[index], props, wet_flux, centers[index]
                )
                for index in range(cell_count - 1)
            ]
            vapor_ra = [
                modified_rayleigh(
                    thermo.vapor[index],
                    props,
                    dry_flux,
                    centers[cell_count + index] - thermo.liquid_height_m,
                )
                for index in range(cell_count - 1)
            ]
            values = liquid_ra + vapor_ra
            rows.append({
                "test_number": int(record["test_number"]),
                "heating_mode": record["heating_mode"],
                "cells_per_phase": cell_count,
                "wetted_heat_flux_W_m2": wet_flux,
                "dry_heat_flux_W_m2": dry_flux,
                "minimum_modified_rayleigh": min(values),
                "maximum_modified_rayleigh": max(values),
                "interfaces_in_range_1e5_to_1e11": int(sum(
                    1.0e5 <= value < 1.0e11 for value in values
                )),
                "interfaces_below_1e5": int(sum(value < 1.0e5 for value in values)),
                "interfaces_at_or_above_1e11": int(sum(
                    value >= 1.0e11 for value in values
                )),
                "interface_count": len(values),
            })

    summary = []
    for cell_count in (3, 6, 10):
        selected = [row for row in rows if row["cells_per_phase"] == cell_count]
        summary.append({
            "cells_per_phase": cell_count,
            "tests_fully_in_range": sum(
                row["interfaces_at_or_above_1e11"] == 0
                and row["interfaces_below_1e5"] == 0
                for row in selected
            ),
            "tests_with_high_range_interfaces": sum(
                row["interfaces_at_or_above_1e11"] > 0 for row in selected
            ),
            "maximum_modified_rayleigh": max(
                row["maximum_modified_rayleigh"] for row in selected
            ),
        })
    print(json.dumps({
        "scope": (
            "initial-state preflight only; the runtime guard repeats this check "
            "as pressure, properties, geometry, and local wall area change"
        ),
        "published_range": "1e5 <= Ra* < 1e11",
        "summary": summary,
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
