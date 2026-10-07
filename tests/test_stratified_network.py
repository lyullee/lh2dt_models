import unittest

import pytest

from lh2dt import (
    HydrogenProperties,
    HorizontalVesselGeometry,
    StratifiedTank,
    StratifiedTankNetworkSimulator,
    StratifiedTankNetworkEnsembleSimulator,
    StratifiedTankEnsemblePort,
    StratifiedTankParameters,
    StratifiedTankPort,
    SteadyNetwork,
    Valve,
)


class StratifiedNetworkTests(unittest.TestCase):
    def _tank(self):
        properties = HydrogenProperties()
        tank = StratifiedTank(
            HorizontalVesselGeometry(2.0, 4.0, 0.5, quadrature_order=48),
            StratifiedTankParameters(
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
            properties,
        )
        return properties, tank, tank.initialize_saturated(200_000.0, 0.5)

    def test_closed_tank_preserves_mass_and_energy_through_phase_ports(self):
        properties, tank, initial = self._tank()
        network = SteadyNetwork(properties)
        network.add_boundary("liquid_port", tank.thermo(initial).liquid)
        network.add_boundary("vapor_port", tank.thermo(initial).vapor)
        simulator = StratifiedTankNetworkSimulator(
            tank,
            network,
            (
                StratifiedTankPort("liquid_port", "liquid"),
                StratifiedTankPort("vapor_port", "vapor"),
            ),
            ambient_temperature_K=20.0,
        )

        results = simulator.simulate(initial, duration_s=1.0, time_step_s=0.2)

        self.assertEqual(len(results), 5)
        self.assertTrue(all(result.network_solution.pressure_solver_success for result in results))
        self.assertLess(abs(results[-1].tank.total_mass_residual_kg), 1.0e-10)
        self.assertLess(abs(results[-1].tank.total_energy_residual_J), 1.0e-5)

    def test_liquid_fill_and_vapor_vent_are_aggregated_by_phase(self):
        properties, tank, initial = self._tank()
        thermo = tank.thermo(initial)
        network = SteadyNetwork(properties)
        network.add_boundary("source", properties.from_pT(300_000.0, 20.0))
        network.add_boundary("liquid_port", thermo.liquid)
        network.add_boundary("vapor_port", thermo.vapor)
        network.add_boundary("vent_sink", properties.from_pT(100_000.0, 80.0))
        network.add_link(
            "fill",
            "source",
            "liquid_port",
            Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False),
        )
        network.add_link(
            "vent",
            "vapor_port",
            "vent_sink",
            Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False),
        )
        simulator = StratifiedTankNetworkSimulator(
            tank,
            network,
            (
                StratifiedTankPort("liquid_port", "liquid"),
                StratifiedTankPort("vapor_port", "vapor"),
            ),
            ambient_temperature_K=20.0,
        )

        results = simulator.simulate(initial, duration_s=0.5, time_step_s=0.1)

        self.assertEqual(len(results), 5)
        self.assertTrue(all(result.network_solution.pressure_solver_success for result in results))
        self.assertTrue(any(result.tank.cumulative_boundary_mass_kg != 0.0 for result in results))
        self.assertLess(abs(results[-1].tank.total_mass_residual_kg), 1.0e-8)
        self.assertLess(abs(results[-1].tank.total_energy_residual_J), 1.0e-2)

    def test_port_mapping_requires_boundary_and_unique_nodes(self):
        properties, tank, initial = self._tank()
        network = SteadyNetwork(properties)
        network.add_boundary("port", tank.thermo(initial).liquid)
        with self.assertRaises(ValueError):
            StratifiedTankNetworkSimulator(
                tank,
                network,
                (StratifiedTankPort("port", "liquid"), StratifiedTankPort("port", "vapor")),
                ambient_temperature_K=20.0,
            )
        with self.assertRaises(KeyError):
            StratifiedTankNetworkSimulator(
                tank,
                network,
                (StratifiedTankPort("missing", "liquid"),),
                ambient_temperature_K=20.0,
            )

    def test_command_callback_reads_current_phase_state_before_network_solve(self):
        properties, tank, initial = self._tank()
        thermo = tank.thermo(initial)
        stale = properties.from_pT(100_000.0, 80.0)
        network = SteadyNetwork(properties)
        network.add_boundary("liquid_port", stale)
        network.add_boundary("vapor_port", stale)
        observed = []

        def command(_time_s, current_network):
            observed.append(current_network.boundaries["liquid_port"].state.pressure_Pa)

        simulator = StratifiedTankNetworkSimulator(
            tank,
            network,
            (
                StratifiedTankPort("liquid_port", "liquid"),
                StratifiedTankPort("vapor_port", "vapor"),
            ),
            ambient_temperature_K=20.0,
            command_callback=command,
        )

        simulator._solve(thermo, 0.0)

        assert observed == [pytest.approx(thermo.liquid.pressure_Pa)]

    def test_two_stratified_tanks_share_one_network_and_conserve_total_inventory(self):
        properties, tank_a, state_a = self._tank()
        _, tank_b, state_b = self._tank()
        state_b = tank_b.initialize_saturated(100_000.0, 0.5)
        network = SteadyNetwork(properties)
        network.add_boundary("tank_a_liquid", tank_a.thermo(state_a).liquid)
        network.add_boundary("tank_b_liquid", tank_b.thermo(state_b).liquid)
        network.add_link(
            "tank_transfer",
            "tank_a_liquid",
            "tank_b_liquid",
            Valve(1.0e-10, 0.8, properties, pressure_samples=16, allow_reverse=False),
        )
        simulator = StratifiedTankNetworkEnsembleSimulator(
            {"tank_a": tank_a, "tank_b": tank_b},
            network,
            (
                StratifiedTankEnsemblePort("tank_a", "tank_a_liquid", "liquid"),
                StratifiedTankEnsemblePort("tank_b", "tank_b_liquid", "liquid"),
            ),
            ambient_temperature_K=20.0,
        )

        initial_total_mass = (
            state_a.liquid_mass_kg + state_a.vapor_mass_kg
            + state_b.liquid_mass_kg + state_b.vapor_mass_kg
        )
        initial_total_energy = tank_a._total_energy_J(state_a) + tank_b._total_energy_J(state_b)
        results = simulator.simulate(
            {"tank_a": state_a, "tank_b": state_b},
            duration_s=1.0,
            time_step_s=0.2,
        )

        final = results[-1]
        final_total_mass = sum(
            state.liquid_mass_kg + state.vapor_mass_kg
            for state in final.states.values()
        )
        final_total_energy = sum(
            simulator.tanks[name]._total_energy_J(state)
            for name, state in final.states.items()
        )
        self.assertEqual(len(results), 5)
        self.assertTrue(all(result.network_solution.pressure_solver_success for result in results))
        self.assertLess(final.cumulative_boundary_mass_kg["tank_a"], 0.0)
        self.assertGreater(final.cumulative_boundary_mass_kg["tank_b"], 0.0)
        self.assertAlmostEqual(final_total_mass, initial_total_mass, delta=1.0e-8)
        self.assertAlmostEqual(final_total_energy, initial_total_energy, delta=1.0e-2)
        self.assertLess(abs(final.total_mass_residual_kg["tank_a"]), 1.0e-8)
        self.assertLess(abs(final.total_mass_residual_kg["tank_b"]), 1.0e-8)

    def test_ensemble_requires_states_for_every_tank(self):
        properties, tank, state = self._tank()
        network = SteadyNetwork(properties)
        network.add_boundary("port", tank.thermo(state).liquid)
        simulator = StratifiedTankNetworkEnsembleSimulator(
            {"tank": tank},
            network,
            (StratifiedTankEnsemblePort("tank", "port", "liquid"),),
            ambient_temperature_K=20.0,
        )
        with self.assertRaises(ValueError):
            simulator.simulate({}, duration_s=0.2, time_step_s=0.1)


if __name__ == "__main__":
    unittest.main()
