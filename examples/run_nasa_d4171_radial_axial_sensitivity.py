"""Run an unfitted sensitivity sweep for the radial-axial NASA mapping.

The sweep spans explicit coarse-grid assumptions only.  It is intended to
bound applicability and expose numerical/model-form sensitivity; it does not
select a parameter from the observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lh2dt import (
    run_nasa_d4171_energy_partition_benchmark,
    run_nasa_d4171_radial_axial_benchmark,
)


def run(output: Path) -> dict:
    wall_fractions = (0.25, 0.50, 0.75)
    local_losses = (0.0, 1.0, 4.0)
    test_numbers = (1, 10, 15)  # uniform, lower, upper heating
    observed_energy = {
        point.test_number: point
        for point in run_nasa_d4171_energy_partition_benchmark().points
    }
    rows = []
    for wall_fraction in wall_fractions:
        for local_loss in local_losses:
            base = {
                "wall_volume_fraction": wall_fraction,
                "core_volume_fraction": 1.0 - wall_fraction,
                "local_loss_coefficient": local_loss,
            }
            try:
                report = run_nasa_d4171_radial_axial_benchmark(
                    time_step_s=60.0,
                    test_numbers=test_numbers,
                    wall_volume_fraction=wall_fraction,
                    local_loss_coefficient=local_loss,
                )
            except ValueError as error:
                rows.append({
                    **base,
                    "status": "rejected_by_physical_or_phase_guard",
                    "error": str(error),
                })
                continue
            rows.append({
                **base,
                "status": "completed",
                "pressure": report.overall.__dict__,
                "pressure_by_heating_mode": {
                    mode: summary.__dict__
                    for mode, summary in report.by_heating_mode.items()
                },
                "energy_partition": [
                    {
                        "test_number": point.test_number,
                        "heating_mode": point.heating_mode,
                        "liquid_energy_percent": point.liquid_energy_percent,
                        "vapor_energy_percent": point.vapor_energy_percent,
                        "evaporation_energy_percent": point.evaporation_energy_percent,
                        "closure_error_percent_point": point.component_closure_error_percent_point,
                    }
                    for point in report.energy_partitions
                ],
                "energy_mae_percent_point": {
                    "liquid": sum(
                        abs(
                            point.liquid_energy_percent
                            - observed_energy[point.test_number].observed_liquid_energy_percent
                        )
                        for point in report.energy_partitions
                    ) / len(report.energy_partitions),
                    "vapor": sum(
                        abs(
                            point.vapor_energy_percent
                            - observed_energy[point.test_number].observed_vapor_energy_percent
                        )
                        for point in report.energy_partitions
                    ) / len(report.energy_partitions),
                    "evaporation": sum(
                        abs(
                            point.evaporation_energy_percent
                            - observed_energy[point.test_number].observed_evaporation_energy_percent
                        )
                        for point in report.energy_partitions
                    ) / len(report.energy_partitions),
                },
                "assumptions": list(report.assumptions),
            })
    payload = {
        "source": "NASA TN D-4171",
        "model": "coefficient-free radial-axial common-pressure tank",
        "test_numbers": list(test_numbers),
        "time_step_s": 60.0,
        "sweep": rows,
        "method": (
            "Forward sensitivity only.  The wall/core split and local-loss "
            "values are explicit coarse-grid inputs; no observation is used "
            "to choose or optimize them."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/nasa_tn_d4171_radial_axial_sensitivity.json"),
    )
    args = parser.parse_args()
    payload = run(args.output)
    print(json.dumps({
        "output": str(args.output),
        "runs": len(payload["sweep"]),
        "tests_per_run": len(payload["test_numbers"]),
    }, indent=2))


if __name__ == "__main__":
    main()
