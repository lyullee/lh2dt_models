"""Run the fixed-parameter cell-count study for NASA TN D-4171 test 15."""

from __future__ import annotations

import json

from lh2dt import run_nasa_d4171_layered_conduction_benchmark


def main() -> None:
    rows = []
    for cell_count in (3, 6, 10):
        report = run_nasa_d4171_layered_conduction_benchmark(
            time_step_s=10.0,
            test_numbers=(15,),
            liquid_cell_count=cell_count,
            vapor_cell_count=cell_count,
        )
        pressure = report.points[0]
        energy = report.energy_partitions[0]
        rows.append({
            "liquid_cell_count": cell_count,
            "vapor_cell_count": cell_count,
            "time_step_s": 10.0,
            "pressure_predicted_over_observed": pressure.predicted_over_observed,
            "liquid_energy_percent": energy.liquid_energy_percent,
            "vapor_energy_percent": energy.vapor_energy_percent,
            "evaporation_energy_percent": energy.evaporation_energy_percent,
            "liquid_bulk_energy_percent": energy.liquid_bulk_energy_percent,
            "liquid_layer_energy_percent": energy.liquid_layer_energy_percent,
            "component_closure_error_percent_point": (
                energy.component_closure_error_percent_point
            ),
        })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
