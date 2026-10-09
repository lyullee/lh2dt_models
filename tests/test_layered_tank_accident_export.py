import math

import pytest

from lh2dt import HorizontalVesselGeometry, LayeredTank, LayeredTankParameters


def make_tank() -> LayeredTank:
    shape = HorizontalVesselGeometry(2.0, 4.0, 0.5, quadrature_order=48)
    return LayeredTank(
        shape,
        LayeredTankParameters(
            liquid_cell_count=3,
            vapor_cell_count=3,
            wall_node_count=6,
            wall_heat_capacity_J_K=3.0e6,
        ),
    )


def request() -> dict[str, object]:
    return {
        "event_id": "tank-A/leak-001",
        "component_id": "tank-A",
        "port_id": "bottom-liquid-outlet",
        "initial_pressure_pa_abs": 500_000.0,
        "liquid_volume_fraction": 0.43,
        "ambient_pressure_pa_abs": 101_325.0,
        "ambient_temperature_k": 288.15,
        "failure_opening_diameter_m": 1.0e-3,
        "discharge_coefficient": 0.7,
        "horizon_s": 0.05,
        "time_step_s": 0.05,
        "impact_position_m": [0.0, 0.0, 0.5],
        "direction_enu": [0.0, 0.0, -1.0],
        "provider_source_digest": "a" * 64,
        "state_snapshot_digest": "b" * 64,
    }


def test_native_export_closes_inventory_and_liquid_flash_partition():
    result = make_tank().export_accident_history(request())
    data = result["data"]
    steps = data["steps"]

    assert result["provider_export_schema"] == "prism.external_accident_history.v1"
    assert data["event_kind"] == "accidental_leak"
    assert data["termination_basis"] == "synthetic_diagnostic_horizon"
    assert data["phase_basis"] == "two_phase"
    assert data["duration_s"] == pytest.approx(0.05)
    assert len(steps) == 1
    step = steps[0]
    assert step["total_mass_flow_kg_s"] == pytest.approx(
        step["gas_mass_flow_kg_s"]
        + step["ground_liquid_mass_flow_kg_s"]
        + step["airborne_liquid_mass_flow_kg_s"]
    )
    assert step["airborne_liquid_mass_flow_kg_s"] == 0.0
    assert step["droplet_class_outcomes"][0]["ground_liquid_mass_flow_kg_s"] == pytest.approx(
        step["ground_liquid_mass_flow_kg_s"]
    )
    assert data["cumulative_mass_out_kg"] > 0.0
    assert data["provider_meta"]["residual_mass_kg"] == pytest.approx(0.0, abs=1.0e-10)
    assert data["provider_meta"]["maximum_mass_balance_residual_kg"] == pytest.approx(
        0.0, abs=1.0e-10
    )


def test_native_export_rejects_non_outward_initial_pressure():
    payload = request()
    payload["initial_pressure_pa_abs"] = payload["ambient_pressure_pa_abs"]
    with pytest.raises(ValueError, match="exceed ambient pressure"):
        make_tank().export_accident_history(payload)


def test_native_export_rejects_zero_direction():
    payload = request()
    payload["direction_enu"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="zero vector"):
        make_tank().export_accident_history(payload)


def test_native_export_records_provider_owned_isolation_success():
    payload = request()
    payload.update({
        "horizon_s": 0.1,
        "time_step_s": 0.05,
        "minimum_step_s": 0.0005,
        "isolation_time_s": 0.05,
        "post_isolation_observation_s": 0.01,
    })
    result = make_tank().export_accident_history(payload)
    data = result["data"]

    assert data["termination_basis"] == "reviewed_isolation_success"
    assert data["provider_meta"]["isolation_success"] is True
    assert data["steps"][0]["total_mass_flow_kg_s"] > 0.0
    assert data["steps"][1]["total_mass_flow_kg_s"] == pytest.approx(0.0)
    assert data["cumulative_mass_out_kg"] < data["available_mass_kg"]
