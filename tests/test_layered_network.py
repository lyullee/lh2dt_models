import unittest

from lh2dt import (
    HydrogenProperties,
    HorizontalVesselGeometry,
    LayeredTank,
    LayeredTankNetworkSimulator,
    LayeredTankParameters,
    LayeredTankPort,
    LayeredTankState,
    CommandedValve,
    SteadyNetwork,
    Valve,
)


class LayeredNetworkTests(unittest.TestCase):
    def test_liquid_and_vapor_ports_drive_layered_tank_from_network(self):
        properties = HydrogenProperties()
        tank = LayeredTank(
            HorizontalVesselGeometry(1.0, 2.0, 0.25, quadrature_order=40),
            LayeredTankParameters(
                liquid_cell_count=2,
                vapor_cell_count=2,
                wall_node_count=4,
                wall_heat_capacity_J_K=1.0e6,
            ),
            properties,
        )
        saturated = tank.initialize_saturated(200_000.0, 0.5)
        initial = tank.project_pressure(LayeredTankState(
            saturated.pressure_Pa,
            saturated.liquid_masses_kg,
            saturated.liquid_specific_enthalpies_J_kg,
            saturated.vapor_masses_kg,
            tuple(value + 5_000.0 for value in saturated.vapor_specific_enthalpies_J_kg),
            saturated.wall_temperatures_K,
        ))
        source = properties.from_pT(300_000.0, 20.0)
        sink = properties.from_pT(100_000.0, 80.0)
        network = SteadyNetwork(properties)
        network.add_boundary("source", source)
        network.add_boundary("liquid_port", tank.thermo(initial).liquid[0])
        network.add_boundary("vapor_port", tank.thermo(initial).vapor[1])
        network.add_boundary("vent_sink", sink)
        network.add_link(
            "fill",
            "source",
            "liquid_port",
            Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False),
            source_port_id="SOURCE:OUT",
            target_port_id="TANK:L0",
        )
        network.add_link(
            "vent",
            "vapor_port",
            "vent_sink",
            Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False),
            source_port_id="TANK:V1",
            target_port_id="VENT:IN",
        )
        simulator = LayeredTankNetworkSimulator(
            tank,
            network,
            ports=(
                LayeredTankPort("liquid_port", "liquid", 0),
                LayeredTankPort("vapor_port", "vapor", 1),
            ),
            ambient_temperature_K=20.0,
        )
        results = simulator.simulate(initial, duration_s=0.5, time_step_s=0.1)
        self.assertEqual(len(results), 5)
        self.assertGreater(results[0].tank.cumulative_boundary_mass_kg, 0.0)
        self.assertTrue(all(result.network_solution.pressure_solver_success for result in results))
        self.assertLess(
            abs(results[-1].tank.total_mass_residual_kg),
            1.0e-8,
        )
        self.assertEqual(network.links[0].source_port_id, "SOURCE:OUT")
        self.assertEqual(network.links[1].target_port_id, "VENT:IN")

    def test_port_cell_index_is_checked(self):
        properties = HydrogenProperties()
        tank = LayeredTank(
            HorizontalVesselGeometry(1.0, 2.0, 0.25),
            LayeredTankParameters(liquid_cell_count=1, vapor_cell_count=1, wall_node_count=2),
            properties,
        )
        state = tank.initialize_saturated(200_000.0, 0.5)
        network = SteadyNetwork(properties)
        network.add_boundary("port", tank.thermo(state).liquid[0])
        with self.assertRaises(ValueError):
            LayeredTankNetworkSimulator(
                tank,
                network,
                (LayeredTankPort("port", "liquid", 2),),
                ambient_temperature_K=20.0,
            )

    def test_command_callback_reads_current_cell_state_before_network_solve(self):
        properties = HydrogenProperties()
        tank = LayeredTank(
            HorizontalVesselGeometry(1.0, 2.0, 0.25, quadrature_order=40),
            LayeredTankParameters(
                liquid_cell_count=1, vapor_cell_count=1, wall_node_count=2,
                wall_heat_capacity_J_K=1.0e6,
            ), properties,
        )
        initial = tank.initialize_saturated(200_000.0, 0.5)
        thermo = tank.thermo(initial)
        stale = properties.from_pT(100_000.0, 80.0)
        network = SteadyNetwork(properties)
        network.add_boundary("liquid_port", stale)
        network.add_boundary("vapor_port", stale)
        observed = []

        def command(_time_s, current_network):
            observed.append(current_network.boundaries["liquid_port"].state.pressure_Pa)

        simulator = LayeredTankNetworkSimulator(
            tank,
            network,
            (
                LayeredTankPort("liquid_port", "liquid", 0),
                LayeredTankPort("vapor_port", "vapor", 0),
            ),
            ambient_temperature_K=20.0,
            command_callback=command,
        )

        simulator._solve(thermo, 0.0)

        self.assertEqual(len(observed), 1)
        self.assertAlmostEqual(observed[0], thermo.liquid[0].pressure_Pa)

    def test_command_callback_switches_from_fill_to_vent(self):
        properties = HydrogenProperties()
        tank = LayeredTank(
            HorizontalVesselGeometry(1.0, 2.0, 0.25, quadrature_order=40),
            LayeredTankParameters(
                liquid_cell_count=2, vapor_cell_count=2, wall_node_count=4,
                wall_heat_capacity_J_K=1.0e6,
            ), properties,
        )
        saturated = tank.initialize_saturated(200_000.0, 0.5)
        initial = tank.project_pressure(LayeredTankState(
            saturated.pressure_Pa,
            saturated.liquid_masses_kg,
            saturated.liquid_specific_enthalpies_J_kg,
            saturated.vapor_masses_kg,
            tuple(value + 5_000.0 for value in saturated.vapor_specific_enthalpies_J_kg),
            saturated.wall_temperatures_K,
        ))
        source = properties.from_pT(300_000.0, 20.0)
        sink = properties.from_pT(100_000.0, 80.0)
        network = SteadyNetwork(properties)
        thermo = tank.thermo(initial)
        network.add_boundary("source", source)
        network.add_boundary("liquid_port", thermo.liquid[0])
        network.add_boundary("vapor_port", thermo.vapor[1])
        network.add_boundary("vent_sink", sink)
        fill = CommandedValve(Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False))
        vent = CommandedValve(Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False))
        network.add_link("fill", "source", "liquid_port", fill)
        network.add_link("vent", "vapor_port", "vent_sink", vent)

        def command(time_s, _network):
            fill.set_opening(1.0 if time_s < 0.25 else 0.0)
            vent.set_opening(0.0 if time_s < 0.25 else 1.0)

        simulator = LayeredTankNetworkSimulator(
            tank, network,
            (LayeredTankPort("liquid_port", "liquid", 0), LayeredTankPort("vapor_port", "vapor", 1)),
            ambient_temperature_K=20.0,
            command_callback=command,
        )
        results = simulator.simulate(initial, duration_s=0.5, time_step_s=0.1)
        self.assertGreater(results[1].tank.cumulative_boundary_mass_kg, 0.0)
        self.assertLess(
            results[-1].tank.cumulative_boundary_mass_kg,
            results[1].tank.cumulative_boundary_mass_kg,
        )


if __name__ == "__main__":
    unittest.main()
