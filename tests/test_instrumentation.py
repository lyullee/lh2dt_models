import pytest

from lh2dt import (
    FirstOrderTransmitter,
    TransmitterParameters,
    absolute_to_gauge_pressure_Pa,
    gauge_to_absolute_pressure_Pa,
)


def test_exact_first_order_update_is_partition_invariant_for_constant_input():
    transmitter = FirstOrderTransmitter(
        TransmitterParameters(0.0, 200.0, time_constant_s=10.0)
    )
    state_one = transmitter.initialize(0.0)
    state_one, _ = transmitter.advance(state_one, 100.0, 10.0)

    state_ten = transmitter.initialize(0.0)
    for _ in range(10):
        state_ten, _ = transmitter.advance(state_ten, 100.0, 1.0)

    assert state_ten.filtered_process_value == pytest.approx(
        state_one.filtered_process_value, rel=1e-14
    )


def test_transmitter_applies_scale_bias_range_resolution_and_uncertainty():
    transmitter = FirstOrderTransmitter(
        TransmitterParameters(
            lower_range=0.0,
            upper_range=100.0,
            scale_factor=1.01,
            bias=0.2,
            resolution=0.5,
            accuracy_fraction_of_span=0.001,
        )
    )
    state = transmitter.initialize(10.0)
    reading = transmitter.read(state)
    assert reading.unsaturated_indicated_value == pytest.approx(10.3)
    assert reading.indicated_value == pytest.approx(10.5)
    assert reading.quantization_error_bound == pytest.approx(0.25)
    assert reading.stated_accuracy_bound == pytest.approx(0.1)
    assert reading.saturated is False

    saturated = transmitter.read(transmitter.initialize(200.0))
    assert saturated.indicated_value == pytest.approx(100.0)
    assert saturated.saturated is True


def test_pressure_basis_conversion_round_trip_requires_explicit_atmosphere():
    atmosphere = 99_800.0
    absolute = 450_000.0
    gauge = absolute_to_gauge_pressure_Pa(absolute, atmosphere)
    assert gauge == pytest.approx(350_200.0)
    assert gauge_to_absolute_pressure_Pa(gauge, atmosphere) == pytest.approx(absolute)


def test_invalid_transmitter_contract_is_rejected():
    with pytest.raises(ValueError):
        TransmitterParameters(1.0, 1.0)
    with pytest.raises(ValueError):
        TransmitterParameters(0.0, 1.0, time_constant_s=-1.0)
    with pytest.raises(ValueError):
        TransmitterParameters(0.0, 1.0, resolution=-1.0)
