import unittest

from lh2dt import HomogeneousTank, MassEnergyFlow, TankGeometry


class TankTests(unittest.TestCase):
    def setUp(self):
        self.geometry = TankGeometry(
            volume_m3=2.0,
            wall_heat_capacity_J_K=2.0e6,
            ambient_UA_W_K=5.0,
            fluid_wall_UA_W_K=100.0,
        )
        self.tank = HomogeneousTank(self.geometry)

    def test_saturated_initialization_roundtrip(self):
        state = self.tank.initialize_saturated(300_000.0, 0.5)
        thermo = self.tank.thermo(state)
        self.assertAlmostEqual(thermo.pressure_Pa, 300_000.0, delta=3.0)
        self.assertIsNotNone(thermo.quality)

    def test_closed_equilibrium_has_zero_derivative(self):
        state = self.tank.initialize_saturated(300_000.0, 0.5)
        thermo = self.tank.thermo(state)
        derivative = self.tank.derivative(state, [], thermo.temperature_K)
        self.assertAlmostEqual(derivative.mass_kg_s, 0.0)
        self.assertAlmostEqual(derivative.internal_energy_W, 0.0, delta=1e-8)
        self.assertAlmostEqual(derivative.wall_temperature_K_s, 0.0, delta=1e-12)

    def test_open_system_mass_and_total_energy_ledgers_close(self):
        initial = self.tank.initialize_saturated(300_000.0, 0.5)
        inlet = self.tank.properties.saturated_liquid(500_000.0)

        def flows(_time, _tank_state):
            return [MassEnergyFlow(0.01, inlet.specific_enthalpy_J_kg, "fill")]

        results = self.tank.simulate(
            initial,
            duration_s=20.0,
            time_step_s=1.0,
            ambient_temperature_K=self.tank.thermo(initial).temperature_K,
            flow_callback=flows,
        )
        final = results[-1]
        self.assertAlmostEqual(final.state.hydrogen_mass_kg - initial.hydrogen_mass_kg, 0.2, delta=1e-9)
        self.assertAlmostEqual(final.cumulative_boundary_mass_kg, 0.2, delta=1e-9)
        self.assertAlmostEqual(final.total_energy_residual_J, 0.0, delta=0.05)

    def test_outflow_uses_tank_enthalpy(self):
        state = self.tank.initialize_saturated(300_000.0, 0.5)
        thermo = self.tank.thermo(state)
        derivative = self.tank.derivative(
            state,
            [MassEnergyFlow(-0.1, 1.0e9, "outlet")],
            thermo.temperature_K,
        )
        self.assertAlmostEqual(derivative.internal_energy_W, -0.1 * thermo.specific_enthalpy_J_kg)


if __name__ == "__main__":
    unittest.main()

