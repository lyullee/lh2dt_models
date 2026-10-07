import unittest
from dataclasses import dataclass

from lh2dt import (
    HydrogenProperties,
    MappedPump,
    Pipe,
    Reliquefier,
    ReliquefierLink,
    SteadyNetwork,
    Vaporizer,
    VaporizerLink,
    VirtualPump,
    build_asset_model,
)


@dataclass(frozen=True)
class LinearResult:
    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float


class LinearResistance:
    def __init__(self, resistance_Pa_s_kg):
        self.resistance = resistance_Pa_s_kg

    def evaluate(self, left, right):
        flow = (left.pressure_Pa - right.pressure_Pa) / self.resistance
        return LinearResult(
            flow,
            left.specific_enthalpy_J_kg if flow >= 0.0 else right.specific_enthalpy_J_kg,
        )


class NonFiniteResult:
    mass_flow_kg_s = float("nan")
    outlet_specific_enthalpy_J_kg = 0.0

    def evaluate(self, left, right):
        return self


class InvalidThermoResult:
    mass_flow_kg_s = 0.1
    outlet_specific_enthalpy_J_kg = -1.0e12

    def evaluate(self, left, right):
        return self


@dataclass(frozen=True)
class InvalidThermalResult:
    mass_flow_kg_s: float = 1.0
    outlet_specific_enthalpy_J_kg: float = 110.0
    hydraulic_outlet_specific_enthalpy_J_kg: float = 100.0
    thermal_power_W: float = 30.0
    upstream_side: str = "left"

    def evaluate(self, left, right):
        return self


class NetworkTests(unittest.TestCase):
    def test_check_valve_preserves_forward_network_direction_and_blocks_reverse(self):
        properties = HydrogenProperties()
        source = properties.from_pT(600_000.0, 20.0)
        sink = properties.from_ph(500_000.0, source.specific_enthalpy_J_kg)
        build = build_asset_model(
            {
                "asset_id": "SYNTH-CHECK-VALVE",
                "component_model": "check_valve",
                "model_parameters": {
                    "full_open_area_m2": 2.0e-6,
                    "discharge_coefficient": 0.8,
                    "cracking_pressure_Pa": 1_000.0,
                },
                "source": "independent factory-to-network test",
            },
            properties,
        )
        self.assertTrue(build.ready, build.issues)
        check = build.component
        network = SteadyNetwork(properties)
        network.add_boundary("source", source)
        network.add_boundary("sink", sink)
        network.add_link(
            "check",
            "source",
            "sink",
            check,
            source_port_id="SOURCE:OUT",
            target_port_id="SINK:IN",
        )
        solution = network.solve()
        self.assertTrue(solution.pressure_solver_success)
        flow = solution.link_flows["check"]
        self.assertGreater(flow.mass_flow_kg_s, 0.0)
        self.assertEqual(flow.upstream_port_id, "SOURCE:OUT")
        self.assertAlmostEqual(
            flow.transmitted_specific_enthalpy_J_kg,
            source.specific_enthalpy_J_kg,
            delta=1.0e-9,
        )

        reverse = check.evaluate(sink, source)
        self.assertEqual(reverse.mass_flow_kg_s, 0.0)
        self.assertIsNone(reverse.upstream_side)

    def test_branch_junction_closes_mass_and_energy(self):
        properties = HydrogenProperties()
        source = properties.from_pT(800_000.0, 80.0)
        sink_a = properties.from_pT(200_000.0, 80.0)
        sink_b = properties.from_pT(300_000.0, 80.0)
        network = SteadyNetwork(properties)
        network.add_boundary("source", source)
        network.add_boundary("sink_a", sink_a)
        network.add_boundary("sink_b", sink_b)
        network.add_junction("header", 500_000.0, source.specific_enthalpy_J_kg)
        network.add_link("inlet", "source", "header", LinearResistance(1.0e7))
        network.add_link("outlet_a", "header", "sink_a", LinearResistance(2.0e7))
        network.add_link("outlet_b", "header", "sink_b", LinearResistance(3.0e7))
        solution = network.solve()
        self.assertTrue(solution.pressure_solver_success)
        self.assertLess(abs(solution.node_mass_residual_kg_s["header"]), 1e-9)
        self.assertLess(abs(solution.node_energy_residual_W["header"]), 1e-4)
        self.assertAlmostEqual(
            solution.node_states["header"].specific_enthalpy_J_kg,
            source.specific_enthalpy_J_kg,
            delta=1e-6,
        )

    def test_invalid_connection_rejected(self):
        properties = HydrogenProperties()
        network = SteadyNetwork(properties)
        network.add_boundary("source", properties.from_pT(500_000.0, 80.0))
        with self.assertRaises(KeyError):
            network.add_link("bad", "source", "missing", LinearResistance(1.0))

    def test_two_port_result_contract_rejects_nonfinite_flow(self):
        properties = HydrogenProperties()
        state = properties.from_pT(500_000.0, 80.0)
        network = SteadyNetwork(properties)
        network.add_boundary("source", state)
        network.add_boundary("sink", state)
        network.add_link("invalid", "source", "sink", NonFiniteResult())
        with self.assertRaisesRegex(ValueError, "link invalid mass_flow_kg_s must be finite"):
            network.solve()

    def test_network_rejects_transmitted_enthalpy_without_downstream_eos_state(self):
        properties = HydrogenProperties()
        source = properties.from_pT(500_000.0, 80.0)
        sink = properties.from_pT(400_000.0, 80.0)
        network = SteadyNetwork(properties)
        network.add_boundary("source", source)
        network.add_boundary("sink", sink)
        network.add_link("invalid_state", "source", "sink", InvalidThermoResult())
        with self.assertRaisesRegex(ValueError, "transmitted enthalpy does not define a valid downstream EOS state"):
            network.solve()

    def test_network_validates_thermal_link_energy_closure(self):
        properties = HydrogenProperties()
        state = properties.from_pT(500_000.0, 80.0)
        network = SteadyNetwork(properties)
        network.add_boundary("source", state)
        network.add_boundary("sink", state)
        network.add_link("invalid_thermal", "source", "sink", InvalidThermalResult())
        with self.assertRaisesRegex(ValueError, "thermal energy closure mismatch"):
            network.solve()

    def test_port_ids_are_recorded_and_require_a_pair(self):
        properties = HydrogenProperties()
        network = SteadyNetwork(properties)
        state = properties.from_pT(500_000.0, 80.0)
        network.add_boundary("source", state)
        network.add_boundary("sink", state)
        network.add_link(
            "pipe",
            "source",
            "sink",
            LinearResistance(1.0e7),
            source_port_id="SOURCE:OUT",
            target_port_id="SINK:IN",
        )
        assert network.links[0].source_port_id == "SOURCE:OUT"
        assert network.links[0].target_port_id == "SINK:IN"
        solution = network.solve()
        assert solution.link_flows["pipe"].source_port_id == "SOURCE:OUT"
        assert solution.link_flows["pipe"].target_port_id == "SINK:IN"
        with self.assertRaises(ValueError):
            network.add_link(
                "incomplete",
                "source",
                "sink",
                LinearResistance(1.0e7),
                source_port_id="SOURCE:OUT",
            )
        with self.assertRaises(ValueError):
            network.add_link(
                "duplicate-port",
                "source",
                "sink",
                LinearResistance(1.0e7),
                source_port_id="SOURCE:OUT",
                target_port_id="SINK:IN-2",
            )

    def test_virtual_lh2_pump_connects_to_pressure_network(self):
        properties = HydrogenProperties()
        source = properties.from_pT(200_000.0, 20.0)
        sink = properties.from_ph(100_000.0, source.specific_enthalpy_J_kg)
        network = SteadyNetwork(properties)
        network.add_boundary("source", source)
        network.add_boundary("sink", sink)
        network.add_junction("discharge", 350_000.0, source.specific_enthalpy_J_kg)
        network.add_link(
            "pump",
            "source",
            "discharge",
            VirtualPump(300_000.0, 0.2, 0.7, properties),
        )
        network.add_link(
            "downstream",
            "discharge",
            "sink",
            LinearResistance(2.0e6),
        )
        solution = network.solve()
        self.assertTrue(solution.pressure_solver_success)
        self.assertLess(abs(solution.node_mass_residual_kg_s["discharge"]), 1.0e-9)
        self.assertLess(abs(solution.node_energy_residual_W["discharge"]), 1.0e-4)
        self.assertGreater(
            solution.node_states["discharge"].pressure_Pa,
            source.pressure_Pa,
        )
        self.assertGreater(
            solution.node_states["discharge"].specific_enthalpy_J_kg,
            source.specific_enthalpy_J_kg,
        )
        self.assertIsNotNone(solution.link_flows["pump"].momentum_residual_Pa)
        self.assertAlmostEqual(
            solution.link_flows["pump"].momentum_residual_Pa,
            0.0,
            delta=1.0e-8,
        )

    def test_factory_virtual_pump_connects_to_pressure_network(self):
        properties = HydrogenProperties()
        build = build_asset_model({
            "asset_id": "SYNTH-VIRTUAL-PUMP",
            "component_model": "virtual_pump",
            "model_parameters": {
                "shutoff_pressure_rise_Pa": 300_000.0,
                "runout_mass_flow_kg_s": 0.2,
                "isentropic_efficiency": 0.7,
                "NPSHr_m": 0.5,
            },
            "source": "independent factory-to-network test",
        })
        self.assertTrue(build.ready, build.issues)
        source = properties.from_pT(200_000.0, 20.0)
        sink = properties.from_ph(100_000.0, source.specific_enthalpy_J_kg)
        network = SteadyNetwork(properties)
        network.add_boundary("source", source)
        network.add_boundary("sink", sink)
        network.add_junction("discharge", 350_000.0, source.specific_enthalpy_J_kg)
        network.add_link("pump", "source", "discharge", build.component)
        network.add_link("downstream", "discharge", "sink", LinearResistance(2.0e6))
        solution = network.solve()
        self.assertTrue(solution.pressure_solver_success)
        self.assertAlmostEqual(
            solution.link_flows["pump"].momentum_residual_Pa,
            0.0,
            delta=1.0e-8,
        )

    def test_mapped_pump_replaces_virtual_pump_in_pressure_network(self):
        properties = HydrogenProperties()
        source = properties.from_pT(200_000.0, 20.0)
        sink = properties.from_ph(100_000.0, source.specific_enthalpy_J_kg)
        network = SteadyNetwork(properties)
        network.add_boundary("source", source)
        network.add_boundary("sink", sink)
        network.add_junction("discharge", 350_000.0, source.specific_enthalpy_J_kg)
        network.add_link(
            "pump",
            "source",
            "discharge",
            MappedPump(
                [(0.0, 300_000.0), (0.1, 225_000.0), (0.2, 0.0)],
                reference_speed_rpm=10_000.0,
                speed_rpm=10_000.0,
                isentropic_efficiency=0.7,
                npsh_required_m=0.5,
                properties=properties,
            ),
        )
        network.add_link("downstream", "discharge", "sink", LinearResistance(2.0e6))
        solution = network.solve()
        self.assertTrue(solution.pressure_solver_success)
        self.assertLess(abs(solution.node_mass_residual_kg_s["discharge"]), 1.0e-9)
        self.assertLess(abs(solution.node_energy_residual_W["discharge"]), 1.0e-4)
        self.assertGreater(solution.link_flows["pump"].mass_flow_kg_s, 0.0)
        self.assertAlmostEqual(
            solution.link_flows["pump"].momentum_residual_Pa,
            0.0,
            delta=1.0e-8,
        )

    def test_factory_vent_stack_connects_to_back_pressure_network(self):
        properties = HydrogenProperties()
        build = build_asset_model({
            "asset_id": "SYNTH-VENT-STACK",
            "component_model": "vent_stack",
            "model_parameters": {
                "length_m": 10.0,
                "inner_diameter_m": 0.05,
                "roughness_m": 1.0e-6,
                "local_loss_coefficient": 2.0,
                "elevation_change_m": 0.0,
                "ambient_temperature_K": 300.0,
                "ambient_UA_W_K": 0.0,
            },
            "source": "independent factory-to-network test",
        }, properties)
        self.assertTrue(build.ready, build.issues)
        source = properties.from_pT(600_000.0, 300.0)
        sink = properties.from_ph(500_000.0, source.specific_enthalpy_J_kg)
        network = SteadyNetwork(properties)
        network.add_boundary("source", source)
        network.add_boundary("sink", sink)
        network.add_link("vent", "source", "sink", build.component)
        solution = network.solve()
        self.assertTrue(solution.pressure_solver_success)
        flow = solution.link_flows["vent"]
        self.assertGreater(flow.mass_flow_kg_s, 0.0)
        self.assertAlmostEqual(flow.momentum_residual_Pa, 0.0, delta=1.0e-8)
        self.assertEqual(set(solution.link_momentum_residuals_Pa), {"vent"})

        # The same factory-built object retains the explicit one-way boundary
        # when the connected back pressure is higher than the source.
        reverse = build.component.evaluate(sink, source)
        self.assertEqual(reverse.mass_flow_kg_s, 0.0)
        self.assertIsNone(reverse.momentum_residual_Pa)

    def test_network_preserves_optional_pipe_momentum_residual_on_link(self):
        properties = HydrogenProperties()
        source = properties.from_pT(600_000.0, 20.0)
        sink = properties.from_ph(500_000.0, source.specific_enthalpy_J_kg)
        network = SteadyNetwork(properties)
        network.add_boundary("source", source)
        network.add_boundary("sink", sink)
        network.add_link(
            "pipe",
            "source",
            "sink",
            Pipe(
                10.0,
                0.02,
                1.0e-6,
                local_loss_coefficient=2.0,
                properties=properties,
            ),
        )
        solution = network.solve()
        link = solution.link_flows["pipe"]
        self.assertIsNotNone(link.momentum_residual_Pa)
        self.assertAlmostEqual(link.momentum_residual_Pa, 0.0, delta=1.0e-8)
        self.assertEqual(set(solution.link_momentum_residuals_Pa), {"pipe"})
        self.assertAlmostEqual(solution.max_abs_momentum_residual_Pa, 0.0, delta=1.0e-8)

        wrapped_network = SteadyNetwork(properties)
        wrapped_network.add_boundary("source", source)
        wrapped_network.add_boundary("sink", sink)
        wrapped_network.add_link(
            "vaporizer_link",
            "source",
            "sink",
            VaporizerLink(
                Pipe(
                    10.0,
                    0.02,
                    1.0e-6,
                    local_loss_coefficient=2.0,
                    properties=properties,
                ),
                Vaporizer(100.0, 10_000.0, properties),
                ambient_temperature_K=300.0,
                properties=properties,
            ),
        )
        wrapped_solution = wrapped_network.solve()
        wrapped_link = wrapped_solution.link_flows["vaporizer_link"]
        self.assertIsNotNone(wrapped_link.momentum_residual_Pa)
        self.assertAlmostEqual(wrapped_link.momentum_residual_Pa, 0.0, delta=1.0e-8)
        self.assertAlmostEqual(
            wrapped_solution.max_abs_momentum_residual_Pa,
            0.0,
            delta=1.0e-8,
        )

        reliquefier_network = SteadyNetwork(properties)
        reliquefier_network.add_boundary("source", source)
        reliquefier_network.add_boundary("sink", sink)
        reliquefier_network.add_link(
            "reliquefier_link",
            "source",
            "sink",
            ReliquefierLink(
                Pipe(
                    10.0,
                    0.02,
                    1.0e-6,
                    local_loss_coefficient=2.0,
                    properties=properties,
                ),
                Reliquefier(10_000.0, properties),
                properties=properties,
            ),
        )
        reliquefier_solution = reliquefier_network.solve()
        reliquefier_link = reliquefier_solution.link_flows["reliquefier_link"]
        self.assertIsNotNone(reliquefier_link.momentum_residual_Pa)
        self.assertAlmostEqual(reliquefier_link.momentum_residual_Pa, 0.0, delta=1.0e-8)
        self.assertAlmostEqual(
            reliquefier_solution.max_abs_momentum_residual_Pa,
            0.0,
            delta=1.0e-8,
        )

    def test_network_leaves_momentum_residual_unset_for_generic_components(self):
        properties = HydrogenProperties()
        state = properties.from_pT(500_000.0, 80.0)
        network = SteadyNetwork(properties)
        network.add_boundary("source", state)
        network.add_boundary("sink", properties.from_ph(400_000.0, state.specific_enthalpy_J_kg))
        network.add_link("generic", "source", "sink", LinearResistance(1.0e7))
        solution = network.solve()
        self.assertIsNone(solution.link_flows["generic"].momentum_residual_Pa)
        self.assertEqual(solution.link_momentum_residuals_Pa, {})
        self.assertIsNone(solution.max_abs_momentum_residual_Pa)


if __name__ == "__main__":
    unittest.main()
