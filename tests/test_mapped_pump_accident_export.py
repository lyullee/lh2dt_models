import pytest

from lh2dt.pump import MappedPump


def make_pump() -> MappedPump:
    return MappedPump(
        [(0.0, 300_000.0), (0.01, 150_000.0), (0.02, 0.0)],
        reference_speed_rpm=3000.0,
        speed_rpm=3000.0,
        isentropic_efficiency=0.7,
    )


def request() -> dict[str, object]:
    return {
        "event_id": "pump-A/leak-001",
        "component_id": "pump-A",
        "port_id": "discharge",
        "source_pressure_pa_abs": 500_000.0,
        "source_temperature_k": 27.0,
        "ambient_pressure_pa_abs": 101_325.0,
        "ambient_temperature_k": 288.15,
        "failure_opening_diameter_m": 1.0e-3,
        "discharge_coefficient": 0.8,
        "horizon_s": 0.1,
        "time_step_s": 0.05,
        "available_mass_kg": 1.0,
        "termination_basis": "synthetic_diagnostic_horizon",
        "termination_provenance": "caller-declared bounded upstream inventory horizon",
        "pump_pressure_rise_pa": 100_000.0,
        "provider_source_digest": "a" * 64,
        "state_snapshot_digest": "b" * 64,
    }


def test_native_pump_export_preserves_map_and_upstream_inventory_ledger():
    result = make_pump().export_accident_history(request())
    data = result["data"]
    assert result["provider_model"] == "lh2dt.pump.MappedPump"
    assert data["phase_basis"] == "two_phase"
    assert data["termination_basis"] == "synthetic_diagnostic_horizon"
    assert data["provider_meta"]["native_inventory_owner"] == "upstream_provider_declared"
    assert data["provider_meta"]["pump_operating_point"]["map_limited"] is False
    assert data["cumulative_mass_out_kg"] > 0.0
    assert data["cumulative_mass_out_kg"] < data["available_mass_kg"]
    for step in data["steps"]:
        assert step["total_mass_flow_kg_s"] == pytest.approx(
            step["gas_mass_flow_kg_s"]
            + step["ground_liquid_mass_flow_kg_s"]
            + step["airborne_liquid_mass_flow_kg_s"]
        )
        assert step["droplet_class_outcomes"][0]["ground_liquid_mass_flow_kg_s"] == pytest.approx(
            step["ground_liquid_mass_flow_kg_s"]
        )


def test_native_pump_export_requires_explicit_upstream_termination_ledger():
    payload = request()
    del payload["termination_basis"]
    with pytest.raises(ValueError, match="termination_basis"):
        make_pump().export_accident_history(payload)


def test_native_pump_export_rejects_gas_source():
    payload = request()
    payload.update({"source_pressure_pa_abs": 1.0e6, "source_temperature_k": 300.0})
    with pytest.raises(ValueError, match="requires a liquid source state"):
        make_pump().export_accident_history(payload)

