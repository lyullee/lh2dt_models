"""Run the six NASA TN D-4171 lower-heating cases with distributed boiling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lh2dt import (
    run_nasa_d4171_energy_partition_benchmark,
    run_nasa_d4171_layered_distributed_boiling_benchmark,
)


def row(point, energy, observed=None) -> dict:
    result = {
        "test_number": point.test_number,
        "heating_mode": point.heating_mode,
        "time_step_s": 20.0,
        "predicted_over_observed_pressure_rise": point.predicted_over_observed,
        "predicted_energy_percent": {
            "liquid": energy.liquid_energy_percent,
            "vapor": energy.vapor_energy_percent,
            "evaporation": energy.evaporation_energy_percent,
        },
    }
    if observed is not None:
        result["observed_energy_percent"] = {
            "liquid": observed.observed_liquid_energy_percent,
            "vapor": observed.observed_vapor_energy_percent,
            "evaporation": observed.observed_evaporation_energy_percent,
        }
    return result


def main() -> None:
    tests = tuple(range(9, 15))
    benchmark = run_nasa_d4171_layered_distributed_boiling_benchmark(
        time_step_s=20.0, test_numbers=tests
    )
    convergence = run_nasa_d4171_layered_distributed_boiling_benchmark(
        time_step_s=10.0, test_numbers=(10,)
    )
    observed = {
        point.test_number: point
        for point in run_nasa_d4171_energy_partition_benchmark().points
    }
    energy = {point.test_number: point for point in benchmark.energy_partitions}
    rows = [
        row(point, energy[point.test_number], observed.get(point.test_number))
        for point in benchmark.points
    ]
    pressure_ratios = np.asarray([
        item["predicted_over_observed_pressure_rise"] for item in rows
    ])
    convergence_point = convergence.points[0]
    convergence_energy = convergence.energy_partitions[0]
    coarse = next(item for item in rows if item["test_number"] == 10)
    payload = {
        "source": {
            "report": "NASA TN D-4171",
            "url": "https://ntrs.nasa.gov/citations/19670028965",
        },
        "model": (
            "3+3-cell common-pressure tank; molecular axial/interface "
            "conduction; equilibrium interface energy jump; saturated-liquid "
            "boiling complementarity and conservative DAE projection"
        ),
        "calibration": "none",
        "time_step_s": 20.0,
        "summary": {
            "count": len(rows),
            "mean_predicted_over_observed_pressure_rise": float(
                pressure_ratios.mean()
            ),
            "mean_absolute_percentage_error_percent": float(
                np.mean(np.abs(pressure_ratios - 1.0)) * 100.0
            ),
        },
        "time_step_check_test_10": {
            "coarse_time_step_s": 20.0,
            "fine_time_step_s": 10.0,
            "coarse_pressure_ratio": coarse[
                "predicted_over_observed_pressure_rise"
            ],
            "fine_pressure_ratio": convergence_point.predicted_over_observed,
            "relative_pressure_change_percent": abs(
                convergence_point.predicted_over_observed
                / coarse["predicted_over_observed_pressure_rise"]
                - 1.0
            ) * 100.0,
            "fine_predicted_energy_percent": {
                "liquid": convergence_energy.liquid_energy_percent,
                "vapor": convergence_energy.vapor_energy_percent,
                "evaporation": convergence_energy.evaporation_energy_percent,
            },
        },
        "tests": rows,
        "interpretation": (
            "The complementarity removes the phase-admissibility failure in "
            "all six lower-heating cases without fitted coefficients.  It is "
            "a numerical/thermodynamic gate result, not validation of the "
            "remaining high-Rayleigh wall-circulation closure."
        ),
    }
    output = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "benchmarks"
        / "nasa_tn_d4171_distributed_boiling_lower_results.json"
    )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
