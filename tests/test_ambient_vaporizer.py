import pytest

from lh2dt import (
    AmbientAirBoundary,
    AmbientAirVaporizer,
    FrostLayerState,
    HydrogenProperties,
)
from lh2dt.component_factory import build_asset_model


def model(**overrides):
    parameters = dict(
        external_area_m2=30.0,
        characteristic_length_m=4.0,
        hydraulic_diameter_m=0.20,
        internal_UA_W_K=2_000.0,
        wall_resistance_K_W=1.0e-4,
        emissivity=0.85,
        maximum_heat_W=60_000.0,
        frost_conductivity_W_mK=0.20,
        frost_deposition_efficiency=0.50,
        integration_segments=32,
    )
    parameters.update(overrides)
    return AmbientAirVaporizer(**parameters)


def operating_point():
    properties = HydrogenProperties()
    inlet = properties.saturated_liquid(350_000.0)
    return inlet, 300_000.0, 0.010, AmbientAirBoundary(293.15, 0.70, 1.0)


def test_wind_increases_conductance_and_heat_delivery():
    vaporizer = model()
    inlet, pressure, flow, still = operating_point()
    windy = AmbientAirBoundary(still.temperature_K, still.relative_humidity, 5.0)
    still_result = vaporizer.evaluate_boundary(inlet, pressure, flow, still)
    windy_result = vaporizer.evaluate_boundary(inlet, pressure, flow, windy)
    assert windy_result.heat_transfer.effective_UA_W_K > still_result.heat_transfer.effective_UA_W_K
    assert windy_result.heat_into_hydrogen_W >= still_result.heat_into_hydrogen_W


def test_frost_adds_resistance_and_reduces_outlet_temperature():
    vaporizer = model()
    inlet, pressure, flow, boundary = operating_point()
    clean = vaporizer.evaluate_boundary(inlet, pressure, flow, boundary, FrostLayerState(0.0))
    frosted = vaporizer.evaluate_boundary(inlet, pressure, flow, boundary, FrostLayerState(0.010))
    assert frosted.heat_transfer.frost_resistance_K_W > 0.0
    assert frosted.heat_transfer.effective_UA_W_K < clean.heat_transfer.effective_UA_W_K
    assert frosted.outlet.temperature_K < clean.outlet.temperature_K


def test_frost_growth_requires_a_cold_surface_and_available_water_vapor():
    vaporizer = model()
    initial = FrostLayerState()
    humid = AmbientAirBoundary(293.15, 0.80, 2.0)
    grown = vaporizer.advance_frost(initial, humid, 250.0, 3600.0)
    warm = vaporizer.advance_frost(initial, humid, 275.0, 3600.0)
    dry = vaporizer.advance_frost(initial, AmbientAirBoundary(293.15, 0.0, 2.0), 250.0, 3600.0)
    assert grown.thickness_m > 0.0
    assert warm == initial
    assert dry == initial


def test_frost_thaws_when_surface_warms_without_becoming_negative():
    vaporizer = model()
    boundary = AmbientAirBoundary(293.15, 0.80, 2.0)
    initial = FrostLayerState(0.010)
    thawed = vaporizer.advance_frost(initial, boundary, 280.0, 3600.0)
    exhausted = vaporizer.advance_frost(initial, boundary, 280.0, 10_000_000.0)
    assert 0.0 <= thawed.thickness_m < initial.thickness_m
    assert exhausted.thickness_m == 0.0


def test_declared_minimum_approach_is_a_physical_outlet_boundary():
    vaporizer = model(minimum_approach_K=8.0)
    inlet, pressure, flow, boundary = operating_point()
    result = vaporizer.evaluate_boundary(inlet, pressure, flow, boundary)
    assert result.outlet.temperature_K <= boundary.temperature_K - 8.0 + 1.0e-6
    assert vaporizer.minimum_approach_K == pytest.approx(8.0)


def test_factory_builds_weather_and_frost_aware_vaporizer():
    record = {
        "asset_id": "VP-TEST",
        "component_model": "ambient_air_vaporizer",
        "source": "explicit test fixture",
        "model_parameters": {
            "external_area_m2": 30.0,
            "characteristic_length_m": 4.0,
            "hydraulic_diameter_m": 0.2,
            "internal_UA_W_K": 2000.0,
            "wall_resistance_K_W": 0.0001,
            "emissivity": 0.85,
            "maximum_heat_W": 60000.0,
            "frost_conductivity_W_mK": 0.2,
            "minimum_approach_K": 6.0,
        },
    }
    built = build_asset_model(record)
    assert built.ready
    assert isinstance(built.component, AmbientAirVaporizer)
    assert built.component.minimum_approach_K == pytest.approx(6.0)


@pytest.mark.parametrize(
    "field,value",
    [("emissivity", 1.1), ("wall_resistance_K_W", -1.0), ("external_area_m2", 0.0)],
)
def test_factory_rejects_invalid_physical_parameters(field, value):
    parameters = {
        "external_area_m2": 30.0,
        "characteristic_length_m": 4.0,
        "hydraulic_diameter_m": 0.2,
        "internal_UA_W_K": 2000.0,
        "wall_resistance_K_W": 0.0001,
        "emissivity": 0.85,
        "maximum_heat_W": 60000.0,
        "frost_conductivity_W_mK": 0.2,
    }
    parameters[field] = value
    built = build_asset_model({
        "asset_id": "VP-BAD",
        "component_model": "ambient_air_vaporizer",
        "source": "explicit test fixture",
        "model_parameters": parameters,
    })
    assert not built.ready
    assert built.issues
