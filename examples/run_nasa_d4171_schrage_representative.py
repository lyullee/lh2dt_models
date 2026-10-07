"""Compare equilibrium and coupled-Schrage phase change without fitting."""

from __future__ import annotations

import json
from pathlib import Path

from lh2dt import (
    run_nasa_d4171_layered_conduction_benchmark,
    run_nasa_d4171_layered_schrage_benchmark,
)


def summarize(report) -> dict:
    point = report.points[0]
    energy = report.energy_partitions[0]
    return {
        "status": "completed",
        "pressure_predicted_over_observed": point.predicted_over_observed,
        "liquid_energy_percent": energy.liquid_energy_percent,
        "vapor_energy_percent": energy.vapor_energy_percent,
        "evaporation_energy_percent": energy.evaporation_energy_percent,
    }


def main() -> None:
    cases: dict[str, dict] = {}
    for test_number in (1, 15):
        equilibrium = run_nasa_d4171_layered_conduction_benchmark(
            time_step_s=10.0, test_numbers=(test_number,)
        )
        schrage = run_nasa_d4171_layered_schrage_benchmark(
            time_step_s=10.0, test_numbers=(test_number,)
        )
        cases[str(test_number)] = {
            "equilibrium_energy_jump": summarize(equilibrium),
            "schrage_kinetic_energy_balance": summarize(schrage),
        }

    lower_heating_steps: dict[str, dict] = {}
    for time_step_s in (10.0, 1.0, 0.1, 0.01):
        try:
            report = run_nasa_d4171_layered_schrage_benchmark(
                time_step_s=time_step_s, test_numbers=(10,)
            )
            lower_heating_steps[str(time_step_s)] = summarize(report)
        except ValueError as error:
            lower_heating_steps[str(time_step_s)] = {
                "status": "rejected",
                "reason": str(error),
            }

    payload = {
        "source": "NASA TN D-4171 Tables I and II",
        "parameter_policy": "no fitting; Schrage accommodation coefficient fixed at 0.001",
        "mesh": "3 liquid + 3 vapor cells",
        "representative_tests": cases,
        "lower_heating_test_10_time_step_screen": lower_heating_steps,
        "interpretation": [
            "The coupled Schrage closure increases predicted pressure and shifts energy from liquid storage toward vapor and evaporation in tests 1 and 15.",
            "It therefore does not repair the missing vapor-to-liquid energy path in the upper-heating test.",
            "Test 10 is rejected at every screened time step because a directly heated saturated lower liquid cell enters the two-phase region; the current model permits phase transfer only at the top liquid-vapor interface.",
            "A distributed wetted-wall boiling or conservative phase-projection model is required before the lower-heating Schrage benchmark is admissible.",
        ],
    }
    output = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "benchmarks"
        / "nasa_tn_d4171_schrage_representative.json"
    )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
