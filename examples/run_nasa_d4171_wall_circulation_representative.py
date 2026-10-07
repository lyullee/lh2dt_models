"""Run two applicability-gated Daigle wall-circulation cases."""

from __future__ import annotations

import csv
import json
from importlib.resources import files

from lh2dt import run_nasa_d4171_layered_wall_circulation_benchmark


def main():
    observations = {}
    path = files("lh2dt").joinpath("data/nasa_tn_d4171_table2.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            observations[int(row["test_number"])] = {
                "liquid_energy_percent": float(row["total_liquid_energy_percent"]),
                "vapor_energy_percent": float(row["vapor_energy_percent"]),
                "evaporation_energy_percent": float(row["evaporation_energy_percent"]),
                "liquid_layer_energy_percent": float(row["liquid_layer_energy_percent"]),
            }

    output = {
        "model": (
            "3+3 common-pressure cells, Daigle horizontal interface, "
            "Daigle Appendix-A reduced wall circulation"
        ),
        "time_step_s": 10.0,
        "calibration": "none",
        "cases": [],
    }
    for test_number in (1, 15):
        try:
            report = run_nasa_d4171_layered_wall_circulation_benchmark(
                time_step_s=10.0,
                test_numbers=(test_number,),
            )
            pressure = report.points[0]
            energy = report.energy_partitions[0]
            output["cases"].append({
                "test_number": test_number,
                "status": "completed",
                "pressure_predicted_over_observed": (
                    pressure.predicted_over_observed
                ),
                "predicted": {
                    "liquid_energy_percent": energy.liquid_energy_percent,
                    "vapor_energy_percent": energy.vapor_energy_percent,
                    "evaporation_energy_percent": (
                        energy.evaporation_energy_percent
                    ),
                    "liquid_layer_energy_percent": (
                        energy.liquid_layer_energy_percent
                    ),
                },
                "observed": observations.get(test_number),
            })
        except ValueError as error:
            output["cases"].append({
                "test_number": test_number,
                "status": "rejected-outside-correlation-range",
                "reason": str(error),
            })
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
