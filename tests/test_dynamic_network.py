import unittest
from dataclasses import dataclass

from lh2dt import (
    DynamicNetworkSimulator,
    DynamicNodeModel,
    HEMPipe,
    HorizontalVesselGeometry,
    HydrogenProperties,
    HomogeneousTank,
    StratifiedTank,
    StratifiedTankParameters,
    SteadyNetwork,
    TankGeometry,
    homogeneous_tank_node,
    stratified_tank_node,
)


@dataclass(frozen=True)
class LinearResult:
    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float


class LinearResistance:
    def __init__(self, resistance_Pa_s_kg: float):
        self.resistance = resistance_Pa_s_kg

    def evaluate(self, left, right):
        flow = (left.pressure_Pa - right.pressure_Pa) / self.resistance
        return LinearResult(
            flow,
            left.specific_enthalpy_J_kg if flow >= 0.0 else right.specific_enthalpy_J_kg,
        )


@dataclass(frozen=True)
class InvalidThermalResult:
    """Malformed thermal link output used to exercise the dynamic network gate."""

    mass_flow_kg_s: float = 1.0
    outlet_specific_enthalpy_J_kg: float = 110.0
    hydraulic_outlet_specific_enthalpy_J_kg: float = 100.0
    thermal_power_W: float = 30.0
    upstream_side: str = "left"

    def evaluate(self, left, right):
        return self


class DynamicNetworkTests(unittest.TestCase):
    def test_dynamic_network_validates_thermal_link_energy_closure(self):
        properties = HydrogenProperties()
        state = properties.from_pT(500_000.0, 80.0)
        network = SteadyNetwork(properties)
        network.add_boundary("source", state)
        network.add_boundary("dynamic_sink", state)
        network.add_link("invalid_thermal", "source", "dynamic_sink", InvalidThermalResult())

        model = DynamicNodeModel(
            name="dynamic_sink",
            initial_state=state,
            thermo=lambda current: current,
            derivative=lambda _state, _flows, _time: None,
            advance=lambda current, _derivative, _dt: current,
            combine_derivatives=lambda *_derivatives: None,
        )
        simulator = DynamicNetworkSimulator(network, {"dynamic_sink": model})
        with self.assertRaisesRegex(ValueError, "thermal energy closure mismatch"):
            simulator.simulate(duration_s=0.1, time_step_s=0.1)

    def test_command_callback_reads_current_rk4_stage_boundary_state(self):
        properties = HydrogenProperties()
        network = SteadyNetwork(properties)
        # Deliberately seed the network with a stale boundary state.  The
        # callback must see the dynamic node's current stage state instead.
        network.add_boundary("node", properties.from_pT(50_000.0, 20.0))
        observed_pressures = []
        model = DynamicNodeModel(
            name="node",
            initial_state=100_000.0,
            thermo=lambda pressure: properties.from_pT(float(pressure), 20.0),
            derivative=lambda _state, _flows, _time: None,
            advance=lambda state, _derivative, _dt: state,
            combine_derivatives=lambda *_derivatives: None,
        )

        def command_callback(_time_s, current_network):
            observed_pressures.append(
                current_network.boundaries["node"].state.pressure_Pa
            )

        simulator = DynamicNetworkSimulator(
            network,
            {"node": model},
            command_callback=command_callback,
        )
        simulator.simulate(duration_s=0.1, time_step_s=0.1)

        self.assertGreaterEqual(len(observed_pressures), 1)
        self.assertAlmostEqual(observed_pressures[0], 100_000.0)

    def test_two_dynamic_tanks_share_mass_through_reusable_network(self):
        properties = HydrogenProperties()
        geometry = TankGeometry(
            volume_m3=1.0,
            wall_heat_capacity_J_K=1.0e8,
            ambient_UA_W_K=1.0e-9,
            fluid_wall_UA_W_K=1.0e-9,
        )
        tank_a = HomogeneousTank(geometry, properties)
        tank_b = HomogeneousTank(geometry, properties)
        state_a = tank_a.initialize_saturated(200_000.0, 0.5, wall_temperature_K=20.0)
        state_b = tank_b.initialize_saturated(100_000.0, 0.5, wall_temperature_K=20.0)
        network = SteadyNetwork(properties)
        network.add_boundary("tank_a", tank_a.thermo(state_a))
        network.add_boundary("tank_b", tank_b.thermo(state_b))
        network.add_link(
            "transfer",
            "tank_a",
            "tank_b",
            LinearResistance(1.0e8),
            source_port_id="TANK_A:P",
            target_port_id="TANK_B:A",
        )
        model_a = homogeneous_tank_node("tank_a", tank_a, state_a, 20.0)
        model_b = homogeneous_tank_node("tank_b", tank_b, state_b, 20.0)
        simulator = DynamicNetworkSimulator(network, {"tank_a": model_a, "tank_b": model_b})
        initial_mass = state_a.hydrogen_mass_kg + state_b.hydrogen_mass_kg
        results = simulator.simulate(duration_s=100.0, time_step_s=10.0)
        self.assertEqual(len(results), 10)
        self.assertLess(results[-1].node_mass_flow_kg_s["tank_a"], 0.0)
        self.assertGreater(results[-1].node_mass_flow_kg_s["tank_b"], 0.0)
        self.assertEqual(results[-1].node_flows["tank_a"][0].source, "TANK_A:P")
        self.assertEqual(results[-1].node_flows["tank_b"][0].source, "TANK_A:P")
        final_mass = sum(
            results[-1].states[name].hydrogen_mass_kg
            for name in ("tank_a", "tank_b")
        )
        self.assertAlmostEqual(final_mass, initial_mass, delta=1.0e-7)
        self.assertAlmostEqual(results[-1].network_boundary_mass_flow_kg_s, 0.0, delta=1.0e-12)
        self.assertAlmostEqual(results[-1].cumulative_network_boundary_mass_kg, 0.0, delta=1.0e-10)
        self.assertAlmostEqual(results[-1].network_boundary_transport_energy_W, 0.0, delta=1.0e-5)
        self.assertAlmostEqual(results[-1].cumulative_network_boundary_energy_J, 0.0, delta=1.0e-3)
        self.assertAlmostEqual(results[-1].cumulative_network_boundary_ambient_heat_J, 0.0, delta=1.0e-3)
        self.assertEqual(network.links[0].source_port_id, "TANK_A:P")
        self.assertEqual(network.links[0].target_port_id, "TANK_B:A")

    def test_dynamic_node_must_be_explicit_boundary(self):
        properties = HydrogenProperties()
        geometry = TankGeometry(1.0, 1.0e6, 1.0, 1.0)
        tank = HomogeneousTank(geometry, properties)
        state = tank.initialize_saturated(100_000.0, 0.5, wall_temperature_K=20.0)
        network = SteadyNetwork(properties)
        with self.assertRaises(KeyError):
            DynamicNetworkSimulator(
                network,
                {"tank": homogeneous_tank_node("tank", tank, state, 20.0)},
            )

    def test_stratified_tank_node_requires_explicit_phase_and_couples_liquid_port(self):
        properties = HydrogenProperties()
        tank = StratifiedTank(
            shape=HorizontalVesselGeometry(2.0, 4.0, 0.5),
            parameters=StratifiedTankParameters(
                lower_wall_heat_capacity_J_K=2.0e6,
                upper_wall_heat_capacity_J_K=1.5e6,
                lower_ambient_UA_W_K=0.0,
                upper_ambient_UA_W_K=0.0,
                wall_liquid_U_W_m2K=5.0,
                wall_vapor_U_W_m2K=2.0,
                liquid_interface_UA_W_K=10.0,
                vapor_interface_UA_W_K=8.0,
                wall_axial_UA_W_K=20.0,
                wall_zone_split_height_m=1.0,
            ),
            properties=properties,
        )
        initial = tank.initialize_saturated(100_000.0, 0.5)
        with self.assertRaises(ValueError):
            stratified_tank_node("tank", tank, initial, "unknown", 300.0)

        source = properties.saturated_liquid(200_000.0)
        network = SteadyNetwork(properties)
        network.add_boundary("source", source)
        network.add_boundary("tank", tank.thermo(initial).liquid)
        network.add_link(
            "liquid_fill",
            "source",
            "tank",
            LinearResistance(1.0e8),
            source_port_id="SOURCE:OUT",
            target_port_id="TANK:LIQUID",
        )
        model = stratified_tank_node("tank", tank, initial, "liquid", 300.0)
        simulator = DynamicNetworkSimulator(network, {"tank": model})
        results = simulator.simulate(duration_s=0.2, time_step_s=0.05)

        final = results[-1]
        final_state = final.states["tank"]
        initial_mass = initial.liquid_mass_kg + initial.vapor_mass_kg
        final_mass = final_state.liquid_mass_kg + final_state.vapor_mass_kg
        self.assertGreater(final.cumulative_boundary_mass_kg["tank"], 0.0)
        self.assertAlmostEqual(
            final_mass - initial_mass,
            final.cumulative_boundary_mass_kg["tank"],
            delta=1.0e-9,
        )
        self.assertGreater(final.network_boundary_mass_flow_kg_s, 0.0)

    def test_hem_pipe_connects_two_phase_tank_and_tracks_conservative_ledger(self):
        properties = HydrogenProperties()
        geometry = TankGeometry(
            volume_m3=1.0,
            wall_heat_capacity_J_K=1.0e8,
            ambient_UA_W_K=1.0e-12,
            fluid_wall_UA_W_K=1.0e-12,
        )
        tank = HomogeneousTank(geometry, properties)
        initial = tank.initialize_saturated(200_000.0, 0.5, wall_temperature_K=20.0)
        liquid = properties.saturated_liquid(350_000.0)
        vapor = properties.saturated_vapor(350_000.0)
        source = properties.from_ph(
            350_000.0,
            liquid.specific_enthalpy_J_kg + 0.7 * (
                vapor.specific_enthalpy_J_kg - liquid.specific_enthalpy_J_kg
            ),
        )
        network = SteadyNetwork(properties)
        network.add_boundary("source", source)
        network.add_boundary("tank", tank.thermo(initial))
        network.add_link(
            "hem_fill",
            "source",
            "tank",
            HEMPipe(
                length_m=5.0,
                inner_diameter_m=0.02,
                roughness_m=1.0e-6,
                local_loss_coefficient=2.0,
                properties=properties,
            ),
        )
        simulator = DynamicNetworkSimulator(
            network,
            {"tank": homogeneous_tank_node("tank", tank, initial, 20.0)},
        )

        results = simulator.simulate(duration_s=0.2, time_step_s=0.05)
        final = results[-1]
        final_state = final.states["tank"]
        initial_total_energy = initial.hydrogen_internal_energy_J + geometry.wall_heat_capacity_J_K * initial.wall_temperature_K
        final_total_energy = final_state.hydrogen_internal_energy_J + geometry.wall_heat_capacity_J_K * final_state.wall_temperature_K

        self.assertTrue(final.thermo_states["tank"].quality is not None)
        self.assertGreater(final.cumulative_boundary_mass_kg["tank"], 0.0)
        self.assertAlmostEqual(
            final_state.hydrogen_mass_kg - initial.hydrogen_mass_kg,
            final.cumulative_boundary_mass_kg["tank"],
            delta=1.0e-10,
        )
        self.assertAlmostEqual(
            final_total_energy - initial_total_energy,
            final.cumulative_boundary_energy_J["tank"],
            delta=1.0e-3,
        )
        self.assertAlmostEqual(
            final.cumulative_boundary_ambient_heat_J["tank"],
            0.0,
            delta=1.0e-3,
        )


if __name__ == "__main__":
    unittest.main()
