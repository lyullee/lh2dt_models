import pytest

from lh2dt.dynamic_reliquefier import DynamicReliquefier
from lh2dt.dynamic_vaporizer import DynamicVaporizer
from lh2dt.properties import HydrogenProperties
from lh2dt.valve import StrokeLimitedCommandedValve, Valve
from lh2dt.vent_stack import VentStack


def _request() -> dict[str, object]:
    return {
        "event_id": "unit-accident-001",
        "component_id": "vaporizer-A",
        "port_id": "outlet",
        "source_pressure_pa_abs": 500_000.0,
        "source_temperature_k": 300.0,
        "ambient_pressure_pa_abs": 101_325.0,
        "ambient_temperature_k": 288.15,
        "failure_opening_diameter_m": 1.0e-3,
        "horizon_s": 0.5,
        "time_step_s": 0.05,
        "provider_source_digest": "sha256:source",
        "state_snapshot_digest": "sha256:state",
    }


@pytest.fixture(params=["vaporizer", "reliquefier"])
def model(request):
    if request.param == "vaporizer":
        return DynamicVaporizer(
            fluid_volume_m3=0.01,
            wall_heat_capacity_J_K=10_000.0,
            fluid_wall_UA_W_K=10.0,
            ambient_temperature_K=300.0,
            ambient_UA_W_K=5.0,
            maximum_heat_W=1_000.0,
        )
    return DynamicReliquefier(
        fluid_volume_m3=0.01,
        wall_heat_capacity_J_K=10_000.0,
        fluid_wall_UA_W_K=10.0,
        cold_side_temperature_K=25.0,
        cold_side_UA_W_K=5.0,
        maximum_cooling_W=1_000.0,
    )


def test_accident_export_has_explicit_history_and_mass_closure(model):
    result = model.export_accident_history(_request())
    assert result["provider_export_schema"] == "prism.external_accident_history.v1"
    assert result["provider_source_digest"] == "sha256:source"
    assert result["provider_state_snapshot_digest"] == "sha256:state"

    data = result["data"]
    assert data["event_kind"] == "accidental_leak"
    assert data["termination_basis"] == "synthetic_diagnostic_horizon"
    assert data["duration_s"] == pytest.approx(0.5)
    assert len(data["steps"]) == 10
    assert any(step["mass_flow_kg_s"] > 0.0 for step in data["steps"])
    assert all("provider_source_state" in step for step in data["steps"])
    assert data["provider_meta"]["residual_mass_kg"] == pytest.approx(0.0, abs=1.0e-12)
    assert data["provider_meta"]["valve"]["choked_intervals"] > 0


def test_accident_export_rejects_liquid_source(model):
    request = _request()
    request.update({"source_pressure_pa_abs": 2_000_000.0, "source_temperature_k": 30.0})
    with pytest.raises(ValueError, match="gas-like source state"):
        model.export_accident_history(request)


def test_accident_export_requires_outward_pressure(model):
    request = _request()
    request["ambient_pressure_pa_abs"] = request["source_pressure_pa_abs"]
    with pytest.raises(ValueError, match="exceed ambient pressure"):
        model.export_accident_history(request)


def test_stroke_limited_valve_native_export_requires_upstream_ledger():
    properties = HydrogenProperties()
    valve = StrokeLimitedCommandedValve(
        Valve(7.853981633974483e-7, 0.8, properties=properties, allow_reverse=False),
        stroke_time_s=0.05,
    )
    valve.set_target_opening(1.0)
    request = _request()
    request.update({
        "available_mass_kg": 1.0,
        "termination_basis": "synthetic_diagnostic_horizon",
        "termination_provenance": "caller-declared bounded valve diagnostic horizon",
    })
    result = valve.export_accident_history(request)
    assert result["provider_export_schema"] == "prism.external_accident_history.v1"
    assert result["data"]["provider_meta"]["native_inventory_owner"] == "upstream_provider_declared"
    assert result["data"]["cumulative_mass_out_kg"] > 0.0
    assert result["data"]["cumulative_mass_out_kg"] < result["data"]["available_mass_kg"]
    assert len(result["data"]["steps"]) == 10


def test_stroke_limited_valve_rejects_missing_termination_ledger():
    properties = HydrogenProperties()
    valve = StrokeLimitedCommandedValve(
        Valve(7.853981633974483e-7, 0.8, properties=properties, allow_reverse=False),
        stroke_time_s=0.05,
    )
    request = _request()
    request["available_mass_kg"] = 1.0
    with pytest.raises(ValueError, match="termination_basis"):
        valve.export_accident_history(request)


def test_vent_stack_native_export_uses_fixed_native_area():
    stack = VentStack(
        length_m=0.1,
        inner_diameter_m=1.0e-3,
        roughness_m=1.0e-6,
        ambient_temperature_K=288.15,
        properties=HydrogenProperties(),
    )
    request = _request()
    request.update({
        "available_mass_kg": 1.0,
        "termination_basis": "synthetic_diagnostic_horizon",
        "termination_provenance": "caller-declared bounded vent diagnostic horizon",
    })
    result = stack.export_accident_history(request)
    assert result["provider_model"] == "lh2dt.vent_stack.VentStack"
    assert result["data"]["provider_meta"]["native_inventory_owner"] == "upstream_provider_declared"
    assert len(result["data"]["steps"]) == 10
    assert result["data"]["cumulative_mass_out_kg"] > 0.0
