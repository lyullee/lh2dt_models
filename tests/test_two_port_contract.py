from types import SimpleNamespace

import pytest

from lh2dt.contracts import validate_thermal_link_result, validate_two_port_result


def _result(mass_flow: float, upstream_side):
    return SimpleNamespace(
        mass_flow_kg_s=mass_flow,
        outlet_specific_enthalpy_J_kg=1.0e5,
        upstream_side=upstream_side,
    )


def test_two_port_contract_requires_direction_consistent_with_flow_sign():
    validate_two_port_result(_result(1.0, "left"))
    validate_two_port_result(_result(-1.0, "right"))
    validate_two_port_result(_result(0.0, None))

    with pytest.raises(ValueError, match="positive flow"):
        validate_two_port_result(_result(1.0, "right"))
    with pytest.raises(ValueError, match="negative flow"):
        validate_two_port_result(_result(-1.0, "left"))
    with pytest.raises(ValueError, match="zero-flow"):
        validate_two_port_result(_result(0.0, "left"))


def test_two_port_contract_keeps_upstream_side_optional_for_generic_components():
    validate_two_port_result(
        SimpleNamespace(
            mass_flow_kg_s=1.0,
            outlet_specific_enthalpy_J_kg=1.0e5,
        )
    )


def test_thermal_link_contract_closes_reported_heat_with_enthalpy_change():
    valid = SimpleNamespace(
        mass_flow_kg_s=-2.0,
        outlet_specific_enthalpy_J_kg=90.0,
        hydraulic_outlet_specific_enthalpy_J_kg=100.0,
        thermal_power_W=-20.0,
        upstream_side="right",
    )
    validate_thermal_link_result(valid)

    invalid = SimpleNamespace(
        mass_flow_kg_s=2.0,
        outlet_specific_enthalpy_J_kg=110.0,
        hydraulic_outlet_specific_enthalpy_J_kg=100.0,
        thermal_power_W=30.0,
        upstream_side="left",
    )
    with pytest.raises(ValueError, match="thermal energy closure mismatch"):
        validate_thermal_link_result(invalid)
