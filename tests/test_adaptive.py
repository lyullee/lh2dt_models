import pytest

from lh2dt import AdaptiveIntegrationError, AdaptiveStepPolicy, HomogeneousTank, TankGeometry


def test_policy_hits_event_without_crossing_it():
    policy = AdaptiveStepPolicy(event_times_s=(50.0,), minimum_step_s=1.0, maximum_step_s=60.0)
    assert policy.limit_to_event(30.0, 60.0, 100.0) == pytest.approx(20.0)
    assert policy.limit_to_event(50.0, 60.0, 100.0) == pytest.approx(50.0)


def test_homogeneous_tank_adaptive_run_keeps_event_timestamp():
    tank = HomogeneousTank(TankGeometry(2.0, 2.0e6, 5.0, 100.0))
    initial = tank.initialize_saturated(300_000.0, 0.5)
    results = tank.simulate(
        initial, duration_s=100.0, time_step_s=30.0,
        ambient_temperature_K=tank.thermo(initial).temperature_K,
        flow_callback=lambda _time, _thermo: [],
        adaptive_policy=AdaptiveStepPolicy(
            minimum_step_s=1.0, maximum_step_s=60.0, event_times_s=(50.0,),
        ),
    )
    times = [result.time_s for result in results]
    assert any(abs(value - 50.0) < 1.0e-12 for value in times)
    assert times[-1] == pytest.approx(100.0)
