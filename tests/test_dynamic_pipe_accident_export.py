import pytest

from lh2dt.dynamic_pipe import DynamicHEMPipe


def make_pipe() -> DynamicHEMPipe:
    return DynamicHEMPipe(
        length_m=10.0,
        inner_diameter_m=0.1,
        roughness_m=1.0e-6,
        wall_heat_capacity_J_K=1.0e5,
        ambient_temperature_K=288.15,
    )


def request() -> dict[str, object]:
    return {
        "event_id": "pipe-A/leak-001",
        "component_id": "pipe-A",
        "port_id": "outlet",
        "initial_pressure_pa_abs": 1.0e6,
        "initial_temperature_k": 300.0,
        "ambient_pressure_pa_abs": 101_325.0,
        "ambient_temperature_k": 288.15,
        "failure_opening_diameter_m": 1.0e-3,
        "discharge_coefficient": 0.8,
        "horizon_s": 0.1,
        "time_step_s": 0.05,
        "position_enu_m": [0.0, 0.0, 1.0],
        "direction_enu": [0.0, 0.0, -1.0],
        "provider_source_digest": "a" * 64,
        "state_snapshot_digest": "b" * 64,
    }


def test_native_pipe_export_closes_gas_inventory():
    result = make_pipe().export_accident_history(request())
    data = result["data"]
    assert result["provider_model"] == "lh2dt.dynamic_pipe.DynamicHEMPipe"
    assert data["termination_basis"] == "synthetic_diagnostic_horizon"
    assert len(data["steps"]) == 2
    assert data["cumulative_mass_out_kg"] > 0.0
    assert data["provider_meta"]["residual_mass_kg"] == pytest.approx(0.0, abs=1.0e-12)
    assert all(step["mass_flow_kg_s"] >= 0.0 for step in data["steps"])
    assert all(step["provider_source_state"]["choked"] for step in data["steps"])


def test_native_pipe_export_resolves_explicit_two_phase_source():
    payload = request()
    payload.update({
        "initial_pressure_pa_abs": 500_000.0,
        "initial_temperature_k": 27.0,
        "initial_quality": 0.1,
        "horizon_s": 0.01,
        "time_step_s": 0.005,
    })
    result = make_pipe().export_accident_history(payload)
    data = result["data"]
    assert data["phase_basis"] == "two_phase"
    assert data["cumulative_specific_enthalpy_out_j"] > 0.0
    assert data["cumulative_mass_out_kg"] > 0.0
    assert data["steps"][0]["gas_mass_flow_kg_s"] > 0.0
    assert data["steps"][0]["ground_liquid_mass_flow_kg_s"] > 0.0
    assert data["steps"][0]["airborne_liquid_mass_flow_kg_s"] == 0.0
    assert data["provider_meta"]["residual_mass_kg"] == pytest.approx(0.0, abs=1.0e-12)


def test_native_pipe_export_rejects_opening_larger_than_pipe():
    payload = request()
    payload["failure_opening_diameter_m"] = 0.2
    with pytest.raises(ValueError, match="must not exceed pipe diameter"):
        make_pipe().export_accident_history(payload)


def test_native_pipe_export_records_provider_owned_isolation_success():
    payload = request()
    payload.update({
        "horizon_s": 0.1,
        "time_step_s": 0.05,
        "isolation_time_s": 0.05,
        "post_isolation_observation_s": 0.01,
    })
    result = make_pipe().export_accident_history(payload)
    data = result["data"]

    assert data["termination_basis"] == "reviewed_isolation_success"
    assert data["provider_meta"]["isolation_success"] is True
    assert data["steps"][0]["mass_flow_kg_s"] > 0.0
    assert data["steps"][1]["mass_flow_kg_s"] == pytest.approx(0.0)
    assert data["cumulative_mass_out_kg"] < data["available_mass_kg"]
