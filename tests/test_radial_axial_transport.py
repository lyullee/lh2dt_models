import math

import pytest

from lh2dt import (
    CirculationLegGeometry,
    HydrogenProperties,
    RadialAxialBoundaryGeometry,
    RadialAxialCirculationTransport,
    RadialAxialPhaseState,
)


PROPS = HydrogenProperties()
PRESSURE = 500_000.0


def enthalpy(temperature_K: float) -> float:
    return PROPS.from_pT(PRESSURE, temperature_K).specific_enthalpy_J_kg


def boundary(height: float = 1.0) -> RadialAxialBoundaryGeometry:
    diameter = 0.02
    area = math.pi * diameter**2 / 4.0
    leg = CirculationLegGeometry(
        length_m=2.0,
        hydraulic_diameter_m=diameter,
        flow_area_m2=area,
        roughness_m=1.0e-6,
        local_loss_coefficient=1.0,
    )
    return RadialAxialBoundaryGeometry(height, leg, leg)


def test_positive_wall_upflow_uses_upwind_enthalpy_and_closes_balances():
    wall_h = (enthalpy(21.0), enthalpy(21.2))
    core_h = (enthalpy(20.0), enthalpy(20.2))
    state = RadialAxialPhaseState(
        wall_masses_kg=(2.0, 3.0),
        wall_specific_enthalpies_J_kg=wall_h,
        core_masses_kg=(4.0, 5.0),
        core_specific_enthalpies_J_kg=core_h,
    )
    result = RadialAxialCirculationTransport(
        (boundary(),), properties=PROPS
    ).evaluate(PRESSURE, state)
    flow = result.boundary_mass_flows_kg_s[0]

    assert flow > 0.0
    assert result.wall_mass_rates_kg_s == pytest.approx((-flow, flow))
    assert result.core_mass_rates_kg_s == pytest.approx((flow, -flow))
    assert result.wall_enthalpy_flow_W == pytest.approx(
        (-flow * wall_h[0], flow * wall_h[0])
    )
    assert result.core_enthalpy_flow_W == pytest.approx(
        (flow * core_h[1], -flow * core_h[1])
    )
    assert result.level_mass_residuals_kg_s == pytest.approx((0.0, 0.0))
    assert result.total_mass_residual_kg_s == pytest.approx(0.0, abs=1e-15)
    assert result.total_enthalpy_flow_residual_W == pytest.approx(
        0.0, abs=1e-10
    )


def test_reversed_loop_switches_donor_cells_without_losing_conservation():
    wall_h = (enthalpy(20.0), enthalpy(20.2))
    core_h = (enthalpy(21.0), enthalpy(21.2))
    state = RadialAxialPhaseState(
        wall_masses_kg=(2.0, 3.0),
        wall_specific_enthalpies_J_kg=wall_h,
        core_masses_kg=(4.0, 5.0),
        core_specific_enthalpies_J_kg=core_h,
    )
    result = RadialAxialCirculationTransport(
        (boundary(),), properties=PROPS
    ).evaluate(PRESSURE, state)
    flow = result.boundary_mass_flows_kg_s[0]

    assert flow < 0.0
    assert result.wall_enthalpy_flow_W == pytest.approx(
        (-flow * wall_h[1], flow * wall_h[1])
    )
    assert result.core_enthalpy_flow_W == pytest.approx(
        (flow * core_h[0], -flow * core_h[0])
    )
    assert result.total_mass_residual_kg_s == pytest.approx(0.0, abs=1e-15)
    assert result.total_enthalpy_flow_residual_W == pytest.approx(
        0.0, abs=1e-10
    )


def test_three_level_grid_conserves_each_level_and_whole_phase():
    state = RadialAxialPhaseState(
        wall_masses_kg=(2.0, 2.5, 3.0),
        wall_specific_enthalpies_J_kg=(
            enthalpy(21.0), enthalpy(21.2), enthalpy(21.4)
        ),
        core_masses_kg=(3.0, 3.5, 4.0),
        core_specific_enthalpies_J_kg=(
            enthalpy(20.0), enthalpy(20.1), enthalpy(20.2)
        ),
    )
    result = RadialAxialCirculationTransport(
        (boundary(0.8), boundary(1.1)), properties=PROPS
    ).evaluate(PRESSURE, state)

    assert len(result.boundary_results) == 2
    assert all(flow > 0.0 for flow in result.boundary_mass_flows_kg_s)
    assert result.level_mass_residuals_kg_s == pytest.approx(
        (0.0, 0.0, 0.0), abs=1e-15
    )
    assert result.total_mass_residual_kg_s == pytest.approx(0.0, abs=1e-15)
    assert result.total_enthalpy_flow_residual_W == pytest.approx(
        0.0, abs=1e-9
    )


def test_transport_sources_preserve_inventory_in_a_finite_state_advance():
    state = RadialAxialPhaseState(
        wall_masses_kg=(2.0, 2.5, 3.0),
        wall_specific_enthalpies_J_kg=(
            enthalpy(21.0), enthalpy(21.2), enthalpy(21.4)
        ),
        core_masses_kg=(3.0, 3.5, 4.0),
        core_specific_enthalpies_J_kg=(
            enthalpy(20.0), enthalpy(20.1), enthalpy(20.2)
        ),
    )
    result = RadialAxialCirculationTransport(
        (boundary(0.8), boundary(1.1)), properties=PROPS
    ).evaluate(PRESSURE, state)
    dt = 0.5
    old_masses = (*state.wall_masses_kg, *state.core_masses_kg)
    rates = (*result.wall_mass_rates_kg_s, *result.core_mass_rates_kg_s)
    old_h = (
        *state.wall_specific_enthalpies_J_kg,
        *state.core_specific_enthalpies_J_kg,
    )
    powers = (*result.wall_enthalpy_flow_W, *result.core_enthalpy_flow_W)
    new_masses = tuple(mass + dt * rate for mass, rate in zip(old_masses, rates))
    new_enthalpy_inventories = tuple(
        mass * h + dt * power
        for mass, h, power in zip(old_masses, old_h, powers)
    )

    assert min(new_masses) > 0.0
    assert sum(new_masses) == pytest.approx(sum(old_masses), abs=1e-14)
    assert sum(new_enthalpy_inventories) == pytest.approx(
        sum(mass * h for mass, h in zip(old_masses, old_h)), abs=1e-8
    )


def test_equal_wall_core_states_give_zero_transport():
    h = (enthalpy(20.0), enthalpy(20.5))
    state = RadialAxialPhaseState((1.0, 1.0), h, (1.0, 1.0), h)
    result = RadialAxialCirculationTransport(
        (boundary(),), properties=PROPS
    ).evaluate(PRESSURE, state)

    assert result.boundary_mass_flows_kg_s == (0.0,)
    assert result.wall_mass_rates_kg_s == (0.0, 0.0)
    assert result.core_enthalpy_flow_W == (0.0, 0.0)


def test_closed_loop_turns_fix_cell_masses_and_redistribute_enthalpy():
    state = RadialAxialPhaseState(
        wall_masses_kg=(2.0, 2.0),
        wall_specific_enthalpies_J_kg=(enthalpy(21.0), enthalpy(21.4)),
        core_masses_kg=(2.0, 2.0),
        core_specific_enthalpies_J_kg=(enthalpy(20.0), enthalpy(20.2)),
    )
    result = RadialAxialCirculationTransport(
        (boundary(0.8),), properties=PROPS, closed_loop_turns=True
    ).evaluate(PRESSURE, state)

    assert result.boundary_mass_flows_kg_s[0] > 0.0
    assert result.wall_mass_rates_kg_s == pytest.approx((0.0, 0.0), abs=1e-15)
    assert result.core_mass_rates_kg_s == pytest.approx((0.0, 0.0), abs=1e-15)
    assert any(abs(value) > 0.0 for value in result.wall_enthalpy_flow_W)
    assert result.total_enthalpy_flow_residual_W == pytest.approx(0.0, abs=1e-10)


def test_closed_loop_turns_requires_boolean_flag():
    with pytest.raises(ValueError, match="closed_loop_turns must be bool"):
        RadialAxialCirculationTransport((boundary(),), closed_loop_turns="false")


def test_grid_rejects_topology_mismatch_and_two_phase_cells():
    liquid_h = enthalpy(20.0)
    valid = RadialAxialPhaseState(
        (1.0, 1.0, 1.0),
        (liquid_h, liquid_h, liquid_h),
        (1.0, 1.0, 1.0),
        (liquid_h, liquid_h, liquid_h),
    )
    with pytest.raises(ValueError, match="boundary count"):
        RadialAxialCirculationTransport(
            (boundary(),), properties=PROPS
        ).evaluate(PRESSURE, valid)

    sat_l = PROPS.saturated_liquid(PRESSURE).specific_enthalpy_J_kg
    sat_v = PROPS.saturated_vapor(PRESSURE).specific_enthalpy_J_kg
    two_phase_h = 0.5 * (sat_l + sat_v)
    invalid = RadialAxialPhaseState(
        (1.0, 1.0),
        (liquid_h, two_phase_h),
        (1.0, 1.0),
        (liquid_h, liquid_h),
    )
    with pytest.raises(ValueError, match="single-phase"):
        RadialAxialCirculationTransport(
            (boundary(),), properties=PROPS
        ).evaluate(PRESSURE, invalid)
