import pytest

from lh2dt import Compressor, HydrogenProperties


def test_compressor_closes_isentropic_efficiency_energy_balance():
    props = HydrogenProperties()
    inlet = props.from_pT(200_000.0, 40.0)
    compressor = Compressor(
        props, pressure_ratio=2.5, isentropic_efficiency=0.72
    )
    result = compressor.evaluate(inlet, 0.08)
    assert result.outlet.pressure_Pa == pytest.approx(500_000.0)
    assert result.outlet.specific_enthalpy_J_kg > inlet.specific_enthalpy_J_kg
    assert result.shaft_power_W == pytest.approx(
        result.isentropic_power_W / 0.72
    )
    assert result.pressure_rise_Pa == pytest.approx(300_000.0)


def test_compressor_rejects_liquid_and_invalid_operating_values():
    props = HydrogenProperties()
    liquid = props.saturated_liquid(200_000.0)
    compressor = Compressor(
        props, pressure_ratio=2.0, isentropic_efficiency=0.75
    )
    with pytest.raises(ValueError, match="vapor inlet"):
        compressor.evaluate(liquid, 0.1)
    with pytest.raises(ValueError, match="greater than 1"):
        Compressor(props, pressure_ratio=1.0, isentropic_efficiency=0.75)
