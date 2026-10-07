"""Run all NASA TN D-4171 cases with the 2 x N radial-axial tank."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lh2dt import (
    run_nasa_d4171_energy_partition_benchmark,
    run_nasa_d4171_radial_axial_benchmark,
)


def main() -> None:
    time_step = 20.0
    benchmark = run_nasa_d4171_radial_axial_benchmark(time_step_s=time_step)
    coarse = run_nasa_d4171_radial_axial_benchmark(
        time_step_s=60.0, test_numbers=(1, 10, 15)
    )
    observed = {
        point.test_number: point
        for point in run_nasa_d4171_energy_partition_benchmark().points
    }
    predicted_energy = {
        point.test_number: point for point in benchmark.energy_partitions
    }
    rows = []
    energy_errors = []
    for point in benchmark.points:
        energy = predicted_energy[point.test_number]
        row = {
            "test_number": point.test_number,
            "heating_mode": point.heating_mode,
            "initial_fill_fraction": point.initial_fill_fraction,
            "duration_s": point.predicted_duration_s,
            "observed_pressure_rise_Pa_s": point.observed_pressure_rise_Pa_s,
            "predicted_pressure_rise_Pa_s": point.predicted_pressure_rise_Pa_s,
            "predicted_over_observed": point.predicted_over_observed,
            "predicted_energy_percent": {
                "liquid": energy.liquid_energy_percent,
                "vapor": energy.vapor_energy_percent,
                "evaporation": energy.evaporation_energy_percent,
                "closure_error_percent_point": (
                    energy.component_closure_error_percent_point
                ),
            },
        }
        if point.test_number in observed:
            source = observed[point.test_number]
            errors = {
                "liquid": energy.liquid_energy_percent
                - source.observed_liquid_energy_percent,
                "vapor": energy.vapor_energy_percent
                - source.observed_vapor_energy_percent,
                "evaporation": energy.evaporation_energy_percent
                - source.observed_evaporation_energy_percent,
            }
            row["observed_energy_percent"] = {
                "liquid": source.observed_liquid_energy_percent,
                "vapor": source.observed_vapor_energy_percent,
                "evaporation": source.observed_evaporation_energy_percent,
            }
            row["energy_error_percent_point"] = errors
            energy_errors.append((point.heating_mode, errors))
        rows.append(row)

    def energy_summary(items) -> dict:
        values = list(items)
        return {
            "count": len(values),
            "liquid_mae_percent_point": float(np.mean([
                abs(item[1]["liquid"]) for item in values
            ])),
            "vapor_mae_percent_point": float(np.mean([
                abs(item[1]["vapor"]) for item in values
            ])),
            "evaporation_mae_percent_point": float(np.mean([
                abs(item[1]["evaporation"]) for item in values
            ])),
        }

    fine_by_test = {point.test_number: point for point in benchmark.points}
    time_step_check = {}
    for point in coarse.points:
        fine = fine_by_test[point.test_number]
        time_step_check[str(point.test_number)] = {
            "coarse_time_step_s": 60.0,
            "fine_time_step_s": time_step,
            "coarse_pressure_ratio": point.predicted_over_observed,
            "fine_pressure_ratio": fine.predicted_over_observed,
            "relative_change_percent": abs(
                fine.predicted_over_observed / point.predicted_over_observed - 1.0
            ) * 100.0,
        }

    payload = {
        "source": {
            "report": benchmark.source_report,
            "url": benchmark.source_url,
        },
        "model": benchmark.model,
        "calibration": "none",
        "assumptions": list(benchmark.assumptions),
        "time_step_s": time_step,
        "pressure_summary": {
            "overall": benchmark.overall.__dict__,
            "by_heating_mode": {
                mode: summary.__dict__
                for mode, summary in benchmark.by_heating_mode.items()
            },
        },
        "energy_summary": {
            "overall": energy_summary(energy_errors),
            "by_heating_mode": {
                mode: energy_summary(
                    item for item in energy_errors if item[0] == mode
                )
                for mode in sorted({item[0] for item in energy_errors})
            },
        },
        "time_step_check": time_step_check,
        "tests": rows,
        "interpretation": (
            "This is an unfitted forward validation of the first complete "
            "radial-axial tank assembly.  The equal radial-volume mesh, smooth "
            "flow paths, and initial-state molecular conductances remain "
            "explicit coarse-grid assumptions and are not facility-fitted."
        ),
    }
    output = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "benchmarks"
        / "nasa_tn_d4171_radial_axial_results.json"
    )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "pressure_summary": payload["pressure_summary"],
        "energy_summary": payload["energy_summary"],
        "time_step_check": payload["time_step_check"],
    }, indent=2))


if __name__ == "__main__":
    main()
