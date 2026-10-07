import math

import pytest

from lh2dt import (
    DynamicNetworkSimulator,
    DynamicVaporizer,
    ParallelDynamicVaporizer,
    HydrogenProperties,
    MassEnergyFlow,
    SteadyNetwork,
    Valve,
    build_asset_model,
    build_dynamic_vaporizer_bank,
    dynamic_vaporizer_node,
    parallel_dynamic_vaporizer_node,
)


def test_dynamic_vaporizer_preserves_mass_energy_and_wall_balances():
    properties = HydrogenProperties()
    inlet = properties.from_pT(300_000.0, 20.0)
    model = DynamicVaporizer(
        fluid_volume_m3=0.01,
        wall_heat_capacity_J_K=10_000.0,
        fluid_wall_UA_W_K=100.0,
        ambient_temperature_K=300.0,
        ambient_UA_W_K=25.0,
        maximum_heat_W=2_000.0,
        properties=properties,
    )
    state = model.initialize(inlet, wall_temperature_K=300.0)
    result = model.step(state, inlet, 0.01, 0.01, 0.1)

    assert result.state.mass_kg > 0.0
    assert result.state.wall_temperature_K < 300.0
    assert abs(result.mass_balance_residual_kg_s) <= 1.0e-12
    assert abs(result.energy_balance_residual_W) <= 1.0e-9
    assert abs(result.wall_energy_balance_residual_W) <= 1.0e-9


def test_dynamic_vaporizer_signed_stream_adapter_uses_upwind_enthalpy():
    properties = HydrogenProperties()
    inlet = properties.from_pT(300_000.0, 20.0)
    model = DynamicVaporizer(
        fluid_volume_m3=0.01,
        wall_heat_capacity_J_K=10_000.0,
        fluid_wall_UA_W_K=0.0,
        ambient_temperature_K=300.0,
        ambient_UA_W_K=0.0,
        maximum_heat_W=100.0,
        properties=properties,
    )
    state = model.initialize(inlet)
    derivative = model.derivative_from_flows(
        state,
        (
            MassEnergyFlow(0.02, inlet.specific_enthalpy_J_kg, source="in"),
            MassEnergyFlow(-0.01, inlet.specific_enthalpy_J_kg, source="out"),
        ),
    )

    assert derivative.mass_flow_in_kg_s == pytest.approx(0.02)
    assert derivative.mass_flow_out_kg_s == pytest.approx(0.01)
    assert derivative.mass_kg_s == pytest.approx(0.01)
    assert derivative.outlet_enthalpy_flow_W == pytest.approx(
        0.01 * inlet.specific_enthalpy_J_kg
    )


def test_dynamic_vaporizer_rejects_reverse_flow_and_bad_parameters():
    properties = HydrogenProperties()
    inlet = properties.from_pT(300_000.0, 20.0)
    with pytest.raises(ValueError):
        DynamicVaporizer(0.0, 1.0, 1.0, 300.0, 1.0, 1.0, properties)
    model = DynamicVaporizer(0.01, 10_000.0, 100.0, 300.0, 25.0, 2_000.0, properties)
    state = model.initialize(inlet)
    with pytest.raises(ValueError):
        model.derivative(state, inlet, -0.01, 0.01)
    with pytest.raises(ValueError):
        model.step(state, inlet, 0.01, 0.01, math.nan)


def test_component_factory_constructs_dynamic_vaporizer_from_explicit_record():
    record = {
        "asset_id": "DYNAMIC-VAP-TEST",
        "component_model": "dynamic_vaporizer",
        "model_parameters": {
            "fluid_volume_m3": 0.01,
            "wall_heat_capacity_J_K": 10_000.0,
            "fluid_wall_UA_W_K": 100.0,
            "ambient_temperature_K": 300.0,
            "ambient_UA_W_K": 25.0,
            "maximum_heat_W": 2_000.0,
        },
        "source": "explicit independent test closure",
    }
    build = build_asset_model(record)
    assert build.ready is True
    assert isinstance(build.component, DynamicVaporizer)


def test_dynamic_vaporizer_node_connects_explicit_hydraulic_links_and_closes_ledger():
    properties = HydrogenProperties()
    inlet_fluid = properties.saturated_liquid(300_000.0)
    vaporizer = DynamicVaporizer(
        fluid_volume_m3=0.01,
        wall_heat_capacity_J_K=100_000.0,
        fluid_wall_UA_W_K=1_000.0,
        ambient_temperature_K=300.0,
        ambient_UA_W_K=10.0,
        maximum_heat_W=100_000.0,
        properties=properties,
    )
    initial_state = vaporizer.initialize(inlet_fluid, wall_temperature_K=280.0)
    source = properties.from_ph(400_000.0, inlet_fluid.specific_enthalpy_J_kg)
    sink = properties.from_ph(200_000.0, inlet_fluid.specific_enthalpy_J_kg)

    network = SteadyNetwork(properties)
    network.add_boundary("source", source)
    network.add_boundary("vaporizer", vaporizer.thermo(initial_state))
    network.add_boundary("sink", sink)
    network.add_link(
        "vaporizer_inlet",
        "source",
        "vaporizer",
        Valve(1.0e-5, 0.8, properties),
        source_port_id="SOURCE:OUT",
        target_port_id="VAP:IN",
    )
    network.add_link(
        "vaporizer_outlet",
        "vaporizer",
        "sink",
        Valve(1.0e-5, 0.8, properties),
        source_port_id="VAP:OUT",
        target_port_id="SINK:IN",
    )

    result = DynamicNetworkSimulator(
        network,
        {"vaporizer": dynamic_vaporizer_node("vaporizer", vaporizer, initial_state)},
    ).simulate(duration_s=1.0e-3, time_step_s=1.0e-3)[-1]

    inlet_flow = result.network_solution.link_flows["vaporizer_inlet"]
    outlet_flow = result.network_solution.link_flows["vaporizer_outlet"]
    assert inlet_flow.mass_flow_kg_s > 0.0
    assert outlet_flow.mass_flow_kg_s > 0.0
    assert inlet_flow.source_port_id == "SOURCE:OUT"
    assert outlet_flow.target_port_id == "SINK:IN"
    final_state = result.states["vaporizer"]
    assert final_state.mass_kg - initial_state.mass_kg == pytest.approx(
        result.cumulative_boundary_mass_kg["vaporizer"], abs=1.0e-12
    )
    initial_total_energy = initial_state.internal_energy_J + vaporizer.wall_heat_capacity * initial_state.wall_temperature_K
    final_total_energy = final_state.internal_energy_J + vaporizer.wall_heat_capacity * final_state.wall_temperature_K
    assert final_total_energy - initial_total_energy == pytest.approx(
        result.cumulative_boundary_energy_J["vaporizer"], abs=1.0e-8
    )
    assert result.node_ambient_heat_W["vaporizer"] > 0.0


def test_parallel_dynamic_vaporizer_preserves_explicit_unit_state_and_flow_split():
    properties = HydrogenProperties()
    inlet = properties.from_pT(300_000.0, 20.0)
    units = tuple(
        DynamicVaporizer(
            fluid_volume_m3=0.01,
            wall_heat_capacity_J_K=10_000.0,
            fluid_wall_UA_W_K=100.0,
            ambient_temperature_K=300.0,
            ambient_UA_W_K=25.0,
            maximum_heat_W=2_000.0,
            properties=properties,
        )
        for _ in range(2)
    )
    bank = ParallelDynamicVaporizer(
        units,
        unit_ids=("VP-A", "VP-B"),
        flow_weights=(1.0, 3.0),
    )
    state = bank.initialize(inlet, wall_temperature_K=300.0)
    derivative = bank.derivative_from_flows(
        state,
        (
            MassEnergyFlow(0.04, inlet.specific_enthalpy_J_kg, source="in"),
            MassEnergyFlow(-0.02, inlet.specific_enthalpy_J_kg, source="out"),
        ),
    )

    assert bank.active_unit_ids == ("VP-A", "VP-B")
    assert bank.active_mask == (True, True)
    assert derivative.mass_kg_s == pytest.approx(0.02)
    assert derivative.mass_flow_in_kg_s == pytest.approx(0.04)
    assert derivative.mass_flow_out_kg_s == pytest.approx(0.02)
    assert derivative.energy_balance_residual_W == pytest.approx(0.0, abs=1.0e-8)
    assert derivative.unit_derivatives[0].mass_flow_in_kg_s == pytest.approx(0.01)
    assert derivative.unit_derivatives[1].mass_flow_in_kg_s == pytest.approx(0.03)
    node = parallel_dynamic_vaporizer_node("bank", bank, state)
    assert node.name == "bank"

    bank.set_active("VP-A", False)
    disabled = bank.derivative_from_flows(
        state,
        (MassEnergyFlow(0.04, inlet.specific_enthalpy_J_kg, source="in"),),
    )
    assert bank.active_unit_ids == ("VP-B",)
    assert bank.active_mask == (False, True)
    assert disabled.unit_derivatives[0].mass_flow_in_kg_s == pytest.approx(0.0)
    assert disabled.unit_derivatives[1].mass_flow_in_kg_s == pytest.approx(0.04)


def test_parallel_dynamic_vaporizer_requires_explicit_boolean_activation():
    properties = HydrogenProperties()
    unit = DynamicVaporizer(
        fluid_volume_m3=0.01,
        wall_heat_capacity_J_K=10_000.0,
        fluid_wall_UA_W_K=100.0,
        ambient_temperature_K=300.0,
        ambient_UA_W_K=25.0,
        maximum_heat_W=2_000.0,
        properties=properties,
    )
    with pytest.raises(ValueError, match="active entries must be bool"):
        ParallelDynamicVaporizer((unit,), active=("false",))
    bank = ParallelDynamicVaporizer((unit,))
    with pytest.raises(ValueError, match="enabled must be bool"):
        bank.set_active("unit-1", "false")


def test_parallel_dynamic_vaporizer_rejects_boundary_flow_without_active_weight():
    properties = HydrogenProperties()
    fluid = properties.from_pT(300_000.0, 20.0)
    unit = DynamicVaporizer(
        fluid_volume_m3=0.01,
        wall_heat_capacity_J_K=10_000.0,
        fluid_wall_UA_W_K=100.0,
        ambient_temperature_K=300.0,
        ambient_UA_W_K=25.0,
        maximum_heat_W=2_000.0,
        properties=properties,
    )
    bank = ParallelDynamicVaporizer((unit,), active=(False,))
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


def test_dynamic_vaporizer_bank_factory_requires_complete_explicit_units():
    def record(asset_id: str, model: str = "dynamic_vaporizer") -> dict:
        return {
            "asset_id": asset_id,
            "component_model": model,
            "model_parameters": {
                "fluid_volume_m3": 0.01,
                "wall_heat_capacity_J_K": 10_000.0,
                "fluid_wall_UA_W_K": 100.0,
                "ambient_temperature_K": 300.0,
                "ambient_UA_W_K": 25.0,
                "maximum_heat_W": 2_000.0,
            },
            "source": "explicit independent bank test closure",
        }

    build = build_dynamic_vaporizer_bank(
        [record("VP-A"), record("VP-B")],
        unit_ids=("VP-A", "VP-B"),
        flow_weights=(1.0, 1.0),
    )
    assert build.ready is True
    assert isinstance(build.component, ParallelDynamicVaporizer)

    incomplete_record = record("VP-B")
    incomplete_record["model_parameters"]["ambient_UA_W_K"] = None
    incomplete = build_dynamic_vaporizer_bank([record("VP-A"), incomplete_record])
    assert incomplete.ready is False
    assert incomplete.component is None
    assert any("ambient_UA_W_K" in issue for issue in incomplete.issues)

    wrong_model = build_dynamic_vaporizer_bank([record("VP-A", "vaporizer")])
    assert wrong_model.ready is False
    assert wrong_model.component is None
