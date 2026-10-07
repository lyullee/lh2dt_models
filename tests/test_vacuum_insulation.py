import pytest

from lh2dt import (
    HomogeneousTank,
    LocalizedHeatPath,
    TankGeometry,
    VacuumInsulation,
    VacuumInsulationParameters,
    make_layered_wall_heat_callback,
)


def make_boundary(**overrides):
    values = dict(
        area_m2=10.0,
        effective_emissivity=0.01,
        mli_solid_conductance_W_K=0.02,
        support_conductance_W_K=0.03,
        residual_gas_coefficient_W_m2K_Pa=0.004,
        maximum_valid_vacuum_pressure_Pa=10.0,
    )
    values.update(overrides)
    return VacuumInsulation(VacuumInsulationParameters(**values))


def test_parallel_heat_paths_sum_and_vacuum_degradation_is_explicit():
    boundary = make_boundary()
    high_vacuum = boundary.evaluate(295.0, 21.0, 0.01)
    degraded = boundary.evaluate(295.0, 21.0, 0.10)

    assert high_vacuum.total_heat_into_cold_W == pytest.approx(
        high_vacuum.radiation_heat_W
        + high_vacuum.mli_solid_heat_W
        + high_vacuum.support_heat_W
        + high_vacuum.residual_gas_heat_W
    )
    assert degraded.radiation_heat_W == pytest.approx(high_vacuum.radiation_heat_W)
    assert degraded.mli_solid_heat_W == pytest.approx(high_vacuum.mli_solid_heat_W)
    assert degraded.support_heat_W == pytest.approx(high_vacuum.support_heat_W)
    assert degraded.residual_gas_heat_W == pytest.approx(10.0 * high_vacuum.residual_gas_heat_W)
    assert degraded.total_heat_into_cold_W > high_vacuum.total_heat_into_cold_W


def test_localized_penetration_paths_are_explicit_and_additive():
    boundary = make_boundary(
        localized_heat_paths=(
            LocalizedHeatPath("nozzle_N1", 0.010),
            LocalizedHeatPath("instrument_I1", 0.005),
        )
    )
    heat = boundary.evaluate(295.0, 21.0, 0.01)

    assert heat.localized_heat_W == pytest.approx(274.0 * 0.015)
    assert heat.total_heat_into_cold_W == pytest.approx(
        heat.radiation_heat_W
        + heat.mli_solid_heat_W
        + heat.support_heat_W
        + heat.residual_gas_heat_W
        + heat.localized_heat_W
    )
    assert boundary.parameters.localized_heat_paths[0].path_id == "nozzle_N1"


@pytest.mark.parametrize(
    "paths",
    [
        (LocalizedHeatPath("same", 0.01), LocalizedHeatPath("same", 0.02)),
        (object(),),
    ],
)
def test_localized_penetration_paths_require_typed_unique_inputs(paths):
    values = dict(
        area_m2=1.0,
        effective_emissivity=0.01,
        mli_solid_conductance_W_K=0.0,
        support_conductance_W_K=0.0,
        residual_gas_coefficient_W_m2K_Pa=0.0,
        maximum_valid_vacuum_pressure_Pa=10.0,
        localized_heat_paths=paths,
    )
    with pytest.raises(ValueError):
        VacuumInsulationParameters(**values)


def test_equal_temperature_and_reversed_temperature_limits():
    boundary = make_boundary()
    equal = boundary.evaluate(40.0, 40.0, 1.0)
    reverse = boundary.evaluate(20.0, 40.0, 0.1)

    assert equal.total_heat_into_cold_W == pytest.approx(0.0)
    assert equal.effective_UA_W_K == pytest.approx(0.0)
    assert reverse.total_heat_into_cold_W < 0.0
    assert reverse.effective_UA_W_K > 0.0
    limited = make_boundary(maximum_valid_vacuum_pressure_Pa=0.5)
    with pytest.raises(ValueError, match="validity limit"):
        limited.evaluate(295.0, 21.0, 0.6)


def test_homogeneous_tank_accepts_decomposed_heat_as_boundary_power():
    boundary = make_boundary(
        area_m2=2.0,
        effective_emissivity=0.001,
        mli_solid_conductance_W_K=0.001,
        support_conductance_W_K=0.001,
        residual_gas_coefficient_W_m2K_Pa=0.0001,
    )
    tank = HomogeneousTank(TankGeometry(2.0, 2.0e6, 1.0e-12, 100.0))
    initial = tank.initialize_saturated(300_000.0, 0.5)

    def heat(_time, _thermo, wall_temperature_K):
        return boundary.evaluate(295.0, wall_temperature_K, 0.01).total_heat_into_cold_W

    results = tank.simulate(
        initial,
        duration_s=4.0,
        time_step_s=1.0,
        ambient_temperature_K=initial.wall_temperature_K,
        flow_callback=lambda _time, _thermo: (),
        wall_heat_callback=heat,
    )
    final = results[-1]
    assert final.cumulative_boundary_energy_J > 0.0
    assert final.total_energy_residual_J == pytest.approx(0.0, abs=2.0e-5)


def test_layered_wall_distribution_preserves_decomposed_heat_for_equal_wall_temperature():
    boundary = make_boundary(
        area_m2=6.0,
        localized_heat_paths=(LocalizedHeatPath("nozzle", 0.01),),
    )
    temperatures = (21.0, 21.0, 21.0)
    fractions = (0.2, 0.3, 0.5)
    distributed = boundary.layered_wall_heat_distribution(
        295.0,
        temperatures,
        0.01,
        area_fractions=fractions,
    )
    expected = boundary.evaluate(295.0, 21.0, 0.01).total_heat_into_cold_W
    assert sum(distributed) == pytest.approx(expected)
    assert distributed[0] / sum(distributed) == pytest.approx(fractions[0])
    assert distributed[2] / sum(distributed) == pytest.approx(fractions[2])


def test_layered_wall_distribution_accepts_explicit_penetration_allocation():
    boundary = make_boundary(
        area_m2=4.0,
        localized_heat_paths=(LocalizedHeatPath("instrument", 0.02),),
    )
    distributed = boundary.layered_wall_heat_distribution(
        295.0,
        (21.0, 21.0),
        0.01,
        area_fractions=(0.5, 0.5),
        localized_path_fractions={"instrument": (0.0, 1.0)},
    )
    base = make_boundary(
        area_m2=4.0,
        localized_heat_paths=(),
    ).evaluate(295.0, 21.0, 0.01).total_heat_into_cold_W
    path_heat = 0.02 * (295.0 - 21.0)
    assert sum(distributed) == pytest.approx(base + path_heat)
    assert distributed[1] > distributed[0]


def test_layered_wall_callback_reads_wall_state_and_supports_time_varying_warm_boundary():
    boundary = make_boundary(area_m2=2.0)
    callback = make_layered_wall_heat_callback(
        boundary,
        vacuum_pressure_Pa=0.01,
        warm_temperature_K=lambda time_s: 295.0 + time_s,
        area_fractions=(0.25, 0.75),
    )
    class Thermo:
        wall_temperatures_K = (21.0, 23.0)
    first = callback(0.0, Thermo())
    second = callback(10.0, Thermo())
    assert len(first) == 2
    assert sum(second) > sum(first)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("area_m2", 0.0),
        ("effective_emissivity", 1.1),
        ("mli_solid_conductance_W_K", -1.0),
        ("support_conductance_W_K", -1.0),
        ("residual_gas_coefficient_W_m2K_Pa", -1.0),
    ],
)
def test_invalid_vacuum_boundary_parameters_are_rejected(parameter, value):
    values = dict(
        area_m2=1.0,
        effective_emissivity=0.01,
        mli_solid_conductance_W_K=0.0,
        support_conductance_W_K=0.0,
        residual_gas_coefficient_W_m2K_Pa=0.0,
        maximum_valid_vacuum_pressure_Pa=10.0,
    )
    values[parameter] = value
    with pytest.raises(ValueError):
        VacuumInsulationParameters(**values)
