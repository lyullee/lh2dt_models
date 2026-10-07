import pytest

from lh2dt import PIDController, PIDParameters, PIDState


def test_output_limit_prevents_integral_windup():
    controller = PIDController(
        PIDParameters(1.0, 1.0, 0.0, output_minimum=0.0, output_maximum=1.0)
    )
    state = PIDState()
    for _ in range(100):
        result = controller.step(state, setpoint=10.0, measurement=0.0, time_step_s=1.0)
        state = result.state
    assert result.output == pytest.approx(1.0)
    assert result.saturated is True
    assert state.integral_error_s == pytest.approx(0.0)

    released = controller.step(state, setpoint=0.0, measurement=0.5, time_step_s=1.0)
    assert released.output == pytest.approx(0.0)


def test_derivative_on_measurement_avoids_setpoint_kick():
    controller = PIDController(PIDParameters(0.0, 0.0, 2.0, output_minimum=-100.0, output_maximum=100.0))
    state = PIDState(previous_measurement=5.0)
    result = controller.step(state, setpoint=100.0, measurement=5.0, time_step_s=1.0)
    assert result.derivative_term == pytest.approx(0.0)


def test_pi_controller_closes_a_first_order_loop():
    controller = PIDController(
        PIDParameters(2.0, 0.5, 0.0, output_minimum=0.0, output_maximum=2.0)
    )
    state = PIDState()
    process_value = 0.0
    dt = 0.01
    for _ in range(2_000):
        result = controller.step(state, 1.0, process_value, dt)
        state = result.state
        process_value += dt * (result.output - process_value) / 2.0
    assert process_value == pytest.approx(1.0, abs=0.01)
    assert result.output == pytest.approx(1.0, abs=0.01)


def test_reverse_action_changes_error_direction():
    direct = PIDController(PIDParameters(1.0, 0.0, 0.0, output_minimum=-10.0, output_maximum=10.0))
    reverse = PIDController(
        PIDParameters(1.0, 0.0, 0.0, output_minimum=-10.0, output_maximum=10.0, reverse_acting=True)
    )
    assert direct.step(PIDState(), 2.0, 1.0, 1.0).output == pytest.approx(1.0)
    assert reverse.step(PIDState(), 2.0, 1.0, 1.0).output == pytest.approx(-1.0)
