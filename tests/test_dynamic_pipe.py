import unittest

from lh2dt import (
    DynamicHEMPipe,
    DynamicHEMPipeState,
    DynamicNetworkSimulator,
    HydrogenProperties,
    SteadyNetwork,
    build_asset_model,
    dynamic_hem_pipe_node,
)


class DynamicHEMPipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.properties = HydrogenProperties()
        liquid = cls.properties.saturated_liquid(300_000.0)
        vapor = cls.properties.saturated_vapor(300_000.0)
        cls.two_phase = cls.properties.from_ph(
            300_000.0,
            liquid.specific_enthalpy_J_kg
            + 0.2 * (vapor.specific_enthalpy_J_kg - liquid.specific_enthalpy_J_kg),
        )

    def test_equal_endpoints_are_stationary_without_heat(self):
        pipe = DynamicHEMPipe(
            10.0,
            0.02,
            1.0e-6,
            1.0e5,
            properties=self.properties,
        )
        state = pipe.initialize(self.two_phase, wall_temperature_K=self.two_phase.temperature_K)
        result = pipe.step(state, self.two_phase, self.two_phase, 1.0)
        self.assertAlmostEqual(result.state.mass_kg, state.mass_kg, delta=1.0e-14)
        self.assertAlmostEqual(result.state.internal_energy_J, state.internal_energy_J, delta=1.0e-9)
        self.assertAlmostEqual(result.fluid.pressure_Pa, self.two_phase.pressure_Pa, delta=1.0e-7)
        self.assertEqual(result.derivative.mass_flow_in_kg_s, 0.0)
        self.assertEqual(result.derivative.mass_flow_out_kg_s, 0.0)
        self.assertAlmostEqual(result.mass_balance_residual_kg_s, 0.0, delta=1.0e-15)
        self.assertAlmostEqual(result.energy_balance_residual_W, 0.0, delta=1.0e-12)
        self.assertAlmostEqual(result.wall_energy_balance_residual_W, 0.0, delta=1.0e-12)

    def test_two_phase_throughflow_has_finite_conservative_fluxes(self):
        pipe = DynamicHEMPipe(
            2.0,
            0.02,
            1.0e-6,
            1.0e5,
            properties=self.properties,
        )
        state = pipe.initialize(self.two_phase, wall_temperature_K=self.two_phase.temperature_K)
        inlet = self.properties.from_ph(400_000.0, self.two_phase.specific_enthalpy_J_kg)
        outlet = self.properties.from_ph(200_000.0, self.two_phase.specific_enthalpy_J_kg)
        derivative = pipe.derivative(state, inlet, outlet)
        self.assertGreater(derivative.mass_flow_in_kg_s, 0.0)
        self.assertGreater(derivative.mass_flow_out_kg_s, 0.0)
        self.assertAlmostEqual(
            derivative.mass_kg_s,
            derivative.mass_flow_in_kg_s - derivative.mass_flow_out_kg_s,
            delta=1.0e-12,
        )
        self.assertAlmostEqual(
            derivative.internal_energy_W,
            derivative.inlet_enthalpy_flow_W
            - derivative.outlet_enthalpy_flow_W
            + derivative.heat_from_wall_W,
            delta=1.0e-8,
        )
        result = pipe.step(state, inlet, outlet, 1.0e-3)
        self.assertTrue(result.fluid.quality is not None)
        self.assertTrue(result.fluid.pressure_Pa > 0.0)
        self.assertAlmostEqual(result.mass_balance_residual_kg_s, 0.0, delta=1.0e-12)
        self.assertAlmostEqual(result.energy_balance_residual_W, 0.0, delta=1.0e-8)

    def test_wall_and_ambient_heat_terms_close_first_law(self):
        pipe = DynamicHEMPipe(
            10.0,
            0.05,
            1.0e-6,
            1.0e6,
            fluid_wall_UA_W_K=1.0,
            ambient_temperature_K=300.0,
            ambient_UA_W_K=5.0,
            properties=self.properties,
        )
        state = pipe.initialize(self.two_phase, wall_temperature_K=20.0)
        derivative = pipe.derivative(state, self.two_phase, self.two_phase)
        self.assertLess(derivative.heat_from_wall_W, 0.0)
        self.assertGreater(derivative.heat_from_ambient_W, 0.0)
        self.assertAlmostEqual(
            pipe.wall_heat_capacity * derivative.wall_temperature_K_s,
            derivative.heat_from_ambient_W - derivative.heat_from_wall_W,
            delta=1.0e-12,
        )
        result = pipe.step(state, self.two_phase, self.two_phase, 1.0e-3)
        self.assertGreater(result.state.wall_temperature_K, state.wall_temperature_K)
        self.assertAlmostEqual(result.wall_energy_balance_residual_W, 0.0, delta=1.0e-10)
        self.assertAlmostEqual(result.energy_balance_residual_W, 0.0, delta=1.0e-8)

    def test_state_and_step_validation(self):
        pipe = DynamicHEMPipe(1.0, 0.02, 1.0e-6, 1.0e5, properties=self.properties)
        with self.assertRaises(ValueError):
            pipe.thermo(DynamicHEMPipeState(0.0, 1.0, 20.0))
        state = pipe.initialize(self.two_phase, wall_temperature_K=self.two_phase.temperature_K)
        with self.assertRaises(ValueError):
            pipe.step(state, self.two_phase, self.two_phase, 0.0)

    def test_constructor_and_wall_temperature_reject_boolean_numeric_inputs(self):
        for kwargs in (
            {"length_m": True},
            {"inner_diameter_m": True},
            {"roughness_m": True},
            {"wall_heat_capacity_J_K": True},
            {"local_loss_coefficient": True},
            {"fluid_wall_UA_W_K": True},
            {"ambient_temperature_K": True},
            {"ambient_UA_W_K": True},
            {"fluid_volume_m3": True},
        ):
            base = {
                "length_m": 1.0,
                "inner_diameter_m": 0.02,
                "roughness_m": 1.0e-6,
                "wall_heat_capacity_J_K": 1.0e5,
            }
            base.update(kwargs)
            with self.assertRaises(ValueError):
                DynamicHEMPipe(properties=self.properties, **base)
        pipe = DynamicHEMPipe(1.0, 0.02, 1.0e-6, 1.0e5, properties=self.properties)
        with self.assertRaises(ValueError):
            pipe.initialize(self.two_phase, wall_temperature_K=True)

    def test_dynamic_pipe_node_connects_explicit_half_segments_and_closes_ledgers(self):
        pipe = DynamicHEMPipe(
            2.0,
            0.02,
            1.0e-6,
            1.0e5,
            properties=self.properties,
        )
        initial_state = pipe.initialize(
            self.two_phase,
            wall_temperature_K=self.two_phase.temperature_K,
        )
        source = self.properties.from_ph(400_000.0, self.two_phase.specific_enthalpy_J_kg)
        sink = self.properties.from_ph(200_000.0, self.two_phase.specific_enthalpy_J_kg)
        network = SteadyNetwork(self.properties)
        network.add_boundary("source", source)
        network.add_boundary("pipe", pipe.thermo(initial_state))
        network.add_boundary("sink", sink)
        network.add_link(
            "pipe_inlet",
            "source",
            "pipe",
            pipe.inlet_hydraulic_component,
            source_port_id="SOURCE:OUT",
            target_port_id="PIPE:IN",
        )
        network.add_link(
            "pipe_outlet",
            "pipe",
            "sink",
            pipe.outlet_hydraulic_component,
            source_port_id="PIPE:OUT",
            target_port_id="SINK:IN",
        )
        simulator = DynamicNetworkSimulator(
            network,
            {"pipe": dynamic_hem_pipe_node("pipe", pipe, initial_state)},
        )
        result = simulator.simulate(duration_s=1.0e-3, time_step_s=1.0e-3)[-1]
        inlet_flow = result.network_solution.link_flows["pipe_inlet"]
        outlet_flow = result.network_solution.link_flows["pipe_outlet"]
        self.assertGreater(inlet_flow.mass_flow_kg_s, 0.0)
        self.assertGreater(outlet_flow.mass_flow_kg_s, 0.0)
        self.assertEqual(inlet_flow.source_port_id, "SOURCE:OUT")
        self.assertEqual(outlet_flow.target_port_id, "SINK:IN")
        final_state = result.states["pipe"]
        self.assertAlmostEqual(
            final_state.mass_kg - initial_state.mass_kg,
            result.cumulative_boundary_mass_kg["pipe"],
            delta=1.0e-12,
        )
        initial_total_energy = initial_state.internal_energy_J + pipe.wall_heat_capacity * initial_state.wall_temperature_K
        final_total_energy = final_state.internal_energy_J + pipe.wall_heat_capacity * final_state.wall_temperature_K
        self.assertAlmostEqual(
            final_total_energy - initial_total_energy,
            result.cumulative_boundary_energy_J["pipe"],
            delta=1.0e-8,
        )
        self.assertAlmostEqual(
            result.network_boundary_mass_flow_kg_s,
            result.node_mass_flow_kg_s["pipe"],
            delta=1.0e-12,
        )
        self.assertAlmostEqual(
            result.network_boundary_transport_energy_W,
            result.node_transport_energy_W["pipe"],
            delta=1.0e-5,
        )

    def test_factory_built_dynamic_pipe_connects_to_network_adapter(self):
        build = build_asset_model({
            "asset_id": "SYNTH-DYNAMIC-PIPE",
            "component_model": "dynamic_hem_pipe",
            "model_parameters": {
                "length_m": 2.0,
                "inner_diameter_m": 0.02,
                "roughness_m": 1.0e-6,
                "wall_heat_capacity_J_K": 1.0e5,
                "local_loss_coefficient": 0.0,
                "fluid_wall_UA_W_K": 0.0,
                "ambient_temperature_K": 300.0,
                "ambient_UA_W_K": 0.0,
                "fluid_volume_m3": 5.0e-4,
            },
            "source": "independent factory-to-network test",
        })
        self.assertTrue(build.ready, build.issues)
        pipe = build.component
        initial_state = pipe.initialize(self.two_phase)
        source = self.properties.from_ph(400_000.0, self.two_phase.specific_enthalpy_J_kg)
        sink = self.properties.from_ph(200_000.0, self.two_phase.specific_enthalpy_J_kg)
        network = SteadyNetwork(self.properties)
        network.add_boundary("source", source)
        network.add_boundary("pipe", pipe.thermo(initial_state))
        network.add_boundary("sink", sink)
        network.add_link("pipe_inlet", "source", "pipe", pipe.inlet_hydraulic_component)
        network.add_link("pipe_outlet", "pipe", "sink", pipe.outlet_hydraulic_component)
        result = DynamicNetworkSimulator(
            network,
            {"pipe": dynamic_hem_pipe_node("pipe", pipe, initial_state)},
        ).simulate(duration_s=1.0e-3, time_step_s=1.0e-3)[-1]
        self.assertGreater(result.network_solution.link_flows["pipe_inlet"].mass_flow_kg_s, 0.0)
        self.assertAlmostEqual(
            result.states["pipe"].mass_kg - initial_state.mass_kg,
            result.cumulative_boundary_mass_kg["pipe"],
            delta=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
