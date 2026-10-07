import pytest

from lh2dt import (
    AmbientAirBoundary,
    AmbientAirVaporizer,
    FlowUAAdjustment,
    HydrogenProperties,
    PressureDropRelation,
    Vaporizer,
)
from lh2dt.component_factory import build_asset_model


def test_explicit_relation_is_monotone_and_bounds_outlet_pressure():
    relation = PressureDropRelation(constant_Pa=100.0, quadratic_Pa_s2_per_kg2=2.0e5)
    assert relation.drop_Pa(0.0) == pytest.approx(100.0)
    assert relation.drop_Pa(0.02) > relation.drop_Pa(0.01)
    assert relation.outlet_pressure_Pa(300_000.0, 0.01) < 300_000.0


def test_relation_rejects_negative_coefficients_and_excessive_loss():
    with pytest.raises(ValueError):
        PressureDropRelation(linear_Pa_s_per_kg=-1.0)
    relation = PressureDropRelation(constant_Pa=200_000.0)
    with pytest.raises(ValueError):
        relation.outlet_pressure_Pa(100_000.0, 0.01)


def test_vaporizer_inlet_pressure_helper_uses_explicit_relation_only():
    properties = HydrogenProperties()
    inlet = properties.saturated_liquid(350_000.0)
    relation = PressureDropRelation(quadratic_Pa_s2_per_kg2=2.0e6)
    vaporizer = Vaporizer(
        1_000.0,
        100_000.0,
        properties,
        integration_segments=32,
        pressure_drop_relation=relation,
    )
    result = vaporizer.evaluate_from_inlet_pressure(inlet, 350_000.0, 0.01, 293.15)
    expected_pressure = relation.outlet_pressure_Pa(350_000.0, 0.01)
    assert result.outlet.pressure_Pa == pytest.approx(expected_pressure)


def test_default_relation_preserves_existing_zero_drop_behavior():
    vaporizer = Vaporizer(1_000.0, 100_000.0)
    assert vaporizer.outlet_pressure_from_inlet(350_000.0, 0.01) == pytest.approx(350_000.0)


def test_factory_accepts_explicit_relation_without_fitting():
    record = {
        "asset_id": "VP-TEST",
        "component_model": "ambient_air_vaporizer",
        "source": "explicit geometry fixture",
        "model_parameters": {
            "external_area_m2": 30.0,
            "characteristic_length_m": 4.0,
            "hydraulic_diameter_m": 0.2,
            "internal_UA_W_K": 2_000.0,
            "wall_resistance_K_W": 1.0e-4,
            "emissivity": 0.85,
            "maximum_heat_W": 60_000.0,
            "frost_conductivity_W_mK": 0.2,
            "pressure_drop_relation": {
                "quadratic_Pa_s2_per_kg2": 1.0e6,
            },
        },
    }
    built = build_asset_model(record)
    assert built.ready
    assert isinstance(built.component, AmbientAirVaporizer)
    assert built.component.outlet_pressure_from_inlet(300_000.0, 0.01) == pytest.approx(299_900.0)


def test_flow_ua_adjustment_is_explicit_monotone_and_default_is_unchanged():
    relation = FlowUAAdjustment(
        reference_mass_flow_kg_s=0.01,
        exponent=0.8,
        minimum_scale=0.5,
        maximum_scale=3.0,
    )
    assert relation.scale(0.0) == pytest.approx(0.5)
    assert relation.scale(0.02) > relation.scale(0.01)
    base = Vaporizer(1_000.0, 100_000.0)
    adjusted = Vaporizer(1_000.0, 100_000.0, flow_ua_adjustment=relation)
    assert base.effective_UA_W_K(0.01) == pytest.approx(1_000.0)
    assert adjusted.effective_UA_W_K(0.02) > adjusted.effective_UA_W_K(0.01)


def test_flow_ua_factory_relation_is_optional_and_no_fit():
    record = {
        "asset_id": "VP-FLOW-UA-TEST",
        "component_model": "vaporizer",
        "source": "explicit geometry fixture",
        "model_parameters": {
            "UA_W_K": 1_000.0,
            "maximum_heat_W": 100_000.0,
            "integration_segments": 32,
            "flow_ua_adjustment": {
                "reference_mass_flow_kg_s": 0.01,
                "exponent": 0.8,
                "minimum_scale": 0.5,
                "maximum_scale": 3.0,
            },
        },
    }
    built = build_asset_model(record)
    assert built.ready, built.issues
    assert built.component.flow_ua_adjustment == FlowUAAdjustment(
        0.01, 0.8, 0.5, 3.0
    )
