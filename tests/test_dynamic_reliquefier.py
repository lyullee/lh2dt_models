import pytest
from dataclasses import dataclass

from lh2dt import (
    DynamicReliquefier,
    DynamicReliquefierState,
    HydrogenProperties,
    DynamicNetworkSimulator,
    SteadyNetwork,
    dynamic_reliquefier_node,
    ParallelDynamicReliquefier,
    ParallelDynamicReliquefierState,
    parallel_dynamic_reliquefier_node,
    MassEnergyFlow,
)


@dataclass(frozen=True)
class _LinearResult:
    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float


class _LinearResistance:
    def __init__(self, resistance_Pa_s_kg: float):
        self.resistance = resistance_Pa_s_kg

    def evaluate(self, left, right):
        flow = (left.pressure_Pa - right.pressure_Pa) / self.resistance
        return _LinearResult(
            flow,
            left.specific_enthalpy_J_kg if flow >= 0.0 else right.specific_enthalpy_J_kg,
        )


def test_dynamic_reliquefier_cools_inventory_and_closes_local_ledgers():
    properties = HydrogenProperties()
    reliquefier = DynamicReliquefier(
        fluid_volume_m3=0.01,
        wall_heat_capacity_J_K=1.0e5,
        fluid_wall_UA_W_K=2.0e3,
        cold_side_temperature_K=15.0,
        cold_side_UA_W_K=4.0e3,
        maximum_cooling_W=2.0e4,
        properties=properties,
    )
    initial_fluid = properties.from_pT(300_000.0, 25.0)
    state = reliquefier.initialize(initial_fluid, wall_temperature_K=20.0)
    result = reliquefier.step(
        state,
        initial_fluid,
        mass_flow_in_kg_s=0.0,
        mass_flow_out_kg_s=0.0,
        time_step_s=1.0,
    )

    assert isinstance(result.state, DynamicReliquefierState)
    assert result.derivative.heat_from_wall_W < 0.0
    assert result.fluid.temperature_K < initial_fluid.temperature_K
    assert result.mass_balance_residual_kg_s == pytest.approx(0.0, abs=1.0e-14)
    assert result.energy_balance_residual_W == pytest.approx(0.0, abs=1.0e-9)
    assert result.wall_energy_balance_residual_W == pytest.approx(0.0, abs=1.0e-9)
    assert abs(result.derivative.heat_from_wall_W) <= reliquefier.maximum_cooling


def test_dynamic_reliquefier_node_keeps_cold_side_boundary_explicit():
    properties = HydrogenProperties()
    reliquefier = DynamicReliquefier(
        fluid_volume_m3=0.01,
        wall_heat_capacity_J_K=1.0e5,
        fluid_wall_UA_W_K=1.0e3,
        cold_side_temperature_K=14.0,
        cold_side_UA_W_K=2.0e3,
        maximum_cooling_W=1.0e4,
        properties=properties,
    )
    fluid = properties.from_pT(300_000.0, 25.0)
    state = reliquefier.initialize(fluid, wall_temperature_K=18.0)
    node = dynamic_reliquefier_node("reliq", reliquefier, state)
    assert node.name == "reliq"
    assert node.thermo(state).pressure_Pa == pytest.approx(fluid.pressure_Pa, rel=1.0e-9)
    with pytest.raises(TypeError):
        dynamic_reliquefier_node("bad", object(), state)


def test_dynamic_reliquefier_node_closes_mass_and_energy_through_two_links():
    properties = HydrogenProperties()
    reliquefier = DynamicReliquefier(
        fluid_volume_m3=0.01,
        wall_heat_capacity_J_K=1.0e5,
        fluid_wall_UA_W_K=1.0e3,
        cold_side_temperature_K=14.0,
        cold_side_UA_W_K=2.0e3,
        maximum_cooling_W=1.0e4,
        properties=properties,
    )
    source = properties.from_pT(350_000.0, 25.0)
    sink = properties.from_pT(250_000.0, 20.0)
    initial_fluid = properties.from_pT(300_000.0, 25.0)
    initial_state = reliquefier.initialize(initial_fluid, wall_temperature_K=18.0)
    network = SteadyNetwork(properties)
    network.add_boundary("source", source)
    network.add_boundary("reliq", reliquefier.thermo(initial_state))
    network.add_boundary("sink", sink)
    network.add_link("inlet", "source", "reliq", _LinearResistance(1.0e8))
    network.add_link("outlet", "reliq", "sink", _LinearResistance(1.0e8))
    simulator = DynamicNetworkSimulator(
        network,
        {"reliq": dynamic_reliquefier_node("reliq", reliquefier, initial_state)},
    )

    result = simulator.simulate(duration_s=0.1, time_step_s=0.1)[-1]
    final_state = result.states["reliq"]
    assert final_state.mass_kg > 0.0
    assert final_state.internal_energy_J < initial_state.internal_energy_J
    assert result.network_solution.pressure_solver_success
    assert final_state.mass_kg - initial_state.mass_kg == pytest.approx(
        result.cumulative_boundary_mass_kg["reliq"], abs=1.0e-10
    )


def test_parallel_dynamic_reliquefier_keeps_unit_states_and_explicit_flow_split():
    properties = HydrogenProperties()
    units = tuple(
        DynamicReliquefier(
            fluid_volume_m3=volume,
            wall_heat_capacity_J_K=1.0e5,
            fluid_wall_UA_W_K=1.0e3,
            cold_side_temperature_K=14.0,
            cold_side_UA_W_K=2.0e3,
            maximum_cooling_W=1.0e4,
            properties=properties,
        )
        for volume in (0.01, 0.02)
    )
    bank = ParallelDynamicReliquefier(
        units,
        unit_ids=("L-A", "L-B"),
        flow_weights=(1.0, 3.0),
    )
    fluid = properties.from_pT(300_000.0, 25.0)
    state = bank.initialize(fluid, wall_temperature_K=18.0)
    assert isinstance(state, ParallelDynamicReliquefierState)
    assert bank.active_unit_ids == ("L-A", "L-B")
    assert bank.active_mask == (True, True)
    derivative = bank.derivative_from_flows(
        state,
        (
            MassEnergyFlow(0.4, fluid.specific_enthalpy_J_kg, source="inlet"),
            MassEnergyFlow(-0.1, fluid.specific_enthalpy_J_kg, source="outlet"),
        ),
    )
    assert len(derivative.unit_derivatives) == 2
    assert derivative.mass_kg_s == pytest.approx(0.3)
    assert derivative.mass_flow_in_kg_s == pytest.approx(0.4)
    assert derivative.mass_flow_out_kg_s == pytest.approx(0.1)
    assert derivative.energy_balance_residual_W == pytest.approx(0.0, abs=1.0e-8)
    assert derivative.unit_derivatives[0].mass_flow_in_kg_s == pytest.approx(0.1)
    assert derivative.unit_derivatives[1].mass_flow_in_kg_s == pytest.approx(0.3)
    assert derivative.ambient_heat_W < 0.0
    assert bank.thermo(state).pressure_Pa == pytest.approx(fluid.pressure_Pa, rel=1.0e-9)
    bank.set_active("L-A", False)
    assert bank.active_unit_ids == ("L-B",)
    assert bank.active_mask == (False, True)
    only_b = bank.derivative_from_flows(
        state,
        (MassEnergyFlow(0.4, fluid.specific_enthalpy_J_kg, source="inlet"),),
    )
    assert only_b.unit_derivatives[0].mass_flow_in_kg_s == pytest.approx(0.0)
    assert only_b.unit_derivatives[1].mass_flow_in_kg_s == pytest.approx(0.4)


def test_parallel_dynamic_reliquefier_requires_explicit_boolean_activation():
    properties = HydrogenProperties()
    unit = DynamicReliquefier(
        fluid_volume_m3=0.01,
        wall_heat_capacity_J_K=1.0e5,
        fluid_wall_UA_W_K=1.0e3,
        cold_side_temperature_K=14.0,
        cold_side_UA_W_K=2.0e3,
        maximum_cooling_W=1.0e4,
        properties=properties,
    )
    with pytest.raises(ValueError, match="active entries must be bool"):
        ParallelDynamicReliquefier((unit,), active=("false",))
    bank = ParallelDynamicReliquefier((unit,))
    with pytest.raises(ValueError, match="enabled must be bool"):
        bank.set_active("unit-1", "false")


def test_parallel_dynamic_reliquefier_rejects_boundary_flow_without_active_weight():
    properties = HydrogenProperties()
    fluid = properties.from_pT(300_000.0, 25.0)
    unit = DynamicReliquefier(
        fluid_volume_m3=0.01,
        wall_heat_capacity_J_K=1.0e5,
        fluid_wall_UA_W_K=1.0e3,
        cold_side_temperature_K=14.0,
        cold_side_UA_W_K=2.0e3,
        maximum_cooling_W=1.0e4,
        properties=properties,
    )
    bank = ParallelDynamicReliquefier((unit,), active=(False,))
    state = bank.initialize(fluid)
    with pytest.raises(ValueError, match="cannot accept boundary flow"):
        bank.derivative_from_flows(state, (
            MassEnergyFlow(0.04, fluid.specific_enthalpy_J_kg),
            MassEnergyFlow(-0.04, fluid.specific_enthalpy_J_kg),
        ))
    closed = bank.derivative_from_flows(state, ())
    assert closed.mass_kg_s == 0.0
    assert closed.mass_flow_in_kg_s == 0.0
    assert closed.mass_flow_out_kg_s == 0.0


def test_parallel_dynamic_reliquefier_node_uses_reusable_dynamic_network_contract():
    properties = HydrogenProperties()
    units = tuple(
        DynamicReliquefier(
            fluid_volume_m3=0.01,
            wall_heat_capacity_J_K=1.0e5,
            fluid_wall_UA_W_K=1.0e3,
            cold_side_temperature_K=14.0,
            cold_side_UA_W_K=2.0e3,
            maximum_cooling_W=1.0e4,
            properties=properties,
        )
        for _ in range(2)
    )
    bank = ParallelDynamicReliquefier(units, unit_ids=("L-A", "L-B"))
    fluid = properties.from_pT(300_000.0, 25.0)
    state = bank.initialize(fluid, wall_temperature_K=18.0)
    node = parallel_dynamic_reliquefier_node("bank", bank, state)
    assert node.name == "bank"
    assert node.thermo(state).pressure_Pa == pytest.approx(fluid.pressure_Pa, rel=1.0e-9)
    with pytest.raises(TypeError):
        parallel_dynamic_reliquefier_node("bad", object(), state)

    network = SteadyNetwork(properties)
    network.add_boundary("source", fluid)
    network.add_boundary("bank", bank.thermo(state))
    network.add_boundary("sink", properties.from_pT(250_000.0, 20.0))
    network.add_link("inlet", "source", "bank", _LinearResistance(1.0e8))
    network.add_link("outlet", "bank", "sink", _LinearResistance(1.0e8))
    simulator = DynamicNetworkSimulator(
        network,
        {"bank": node},
    )
    result = simulator.simulate(duration_s=0.1, time_step_s=0.1)[-1]
    final_state = result.states["bank"]
    assert len(final_state.unit_states) == 2
    assert result.network_solution.pressure_solver_success
    assert result.cumulative_boundary_mass_kg["bank"] == pytest.approx(
        final_state.unit_states[0].mass_kg + final_state.unit_states[1].mass_kg
        - sum(item.mass_kg for item in state.unit_states),
        abs=1.0e-10,
    )
