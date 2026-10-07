import pytest

from lh2dt import HydrogenProperties, Vaporizer, VaporizerOperatingBoundary


def test_operating_boundary_requires_explicit_closed_or_flowing_state():
    with pytest.raises(ValueError):
        VaporizerOperatingBoundary("closed", 0.1)
    with pytest.raises(ValueError):
        VaporizerOperatingBoundary("flowing", 0.0)
    with pytest.raises(ValueError):
        VaporizerOperatingBoundary("flowing", 600.0, maximum_normal_flow_Nm3_h=500.0)


def test_closed_boundary_has_zero_transport_and_flowing_boundary_uses_public_reference_state():
    properties = HydrogenProperties()
    closed = VaporizerOperatingBoundary("closed", 0.0, maximum_normal_flow_Nm3_h=500.0)
    flowing = VaporizerOperatingBoundary("flowing", 100.0, maximum_normal_flow_Nm3_h=500.0)
    assert closed.is_closed is True
    assert closed.mass_flow_kg_s(properties) == pytest.approx(0.0)
    assert flowing.is_closed is False
    assert flowing.mass_flow_kg_s(properties) > 0.0


def test_vaporizer_evaluates_explicit_closed_boundary_without_hidden_capacity_default():
    properties = HydrogenProperties()
    inlet = properties.saturated_liquid(300_000.0)
    model = Vaporizer(100.0, 10_000.0, properties)
    boundary = VaporizerOperatingBoundary("closed", 0.0)
    result = model.evaluate_operating_boundary(inlet, 300_000.0, boundary, 300.0)
    assert result.heat_into_hydrogen_W == pytest.approx(0.0)
    assert result.requested_heat_W == pytest.approx(0.0)
    assert result.outlet.temperature_K == pytest.approx(inlet.temperature_K)
