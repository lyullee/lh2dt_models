import pytest

from lh2dt import (
    FirstOrderTransmitter,
    HartDynamicVariables,
    PMD75BHartObservation,
    TransmitterParameters,
    pmd75b_hart_variable_meanings,
)


def test_pmd75b_default_transform_keeps_pv_and_qv_on_one_measurement_cell():
    observation = PMD75BHartObservation()
    state = observation.initialize(120.0)
    state, variables = observation.advance(
        state,
        raw_sensor_pressure=80.0,
        position_adjusted_pressure=120.0,
        sensor_temperature=15.0,
        electronics_temperature=28.0,
        time_step_s=1.0,
    )

    assert variables.as_dict() == {"PV": 120.0, "SV": 15.0, "TV": 28.0, "QV": 80.0}
    assert state.pv_transmitter_state is None


def test_pv_damping_does_not_delay_qv_or_reassign_temperatures():
    transmitter = FirstOrderTransmitter(
        TransmitterParameters(0.0, 200.0, time_constant_s=10.0)
    )
    observation = PMD75BHartObservation(transmitter)
    state = observation.initialize(0.0)
    state, variables = observation.advance(
        state,
        raw_sensor_pressure=100.0,
        position_adjusted_pressure=100.0,
        sensor_temperature=20.0,
        electronics_temperature=30.0,
        time_step_s=1.0,
    )

    assert variables.qv == pytest.approx(100.0)
    assert variables.pv < 20.0
    assert variables.sv == pytest.approx(20.0)
    assert variables.tv == pytest.approx(30.0)


def test_hart_contract_does_not_infer_range_or_units():
    observation = PMD75BHartObservation()
    assert observation.pv_transmitter is None
    assert set(pmd75b_hart_variable_meanings()) == {"PV", "SV", "TV", "QV"}
    assert pmd75b_hart_variable_meanings()["QV"].startswith("sensor pressure")


def test_hart_variables_reject_nonfinite_values():
    with pytest.raises(ValueError):
        HartDynamicVariables(pv=float("nan"), sv=1.0, tv=2.0, qv=3.0)

