import unittest

from lh2dt import (
    CommandedValve,
    HydrogenProperties,
    PressureReliefValve,
    StrokeLimitedCommandedValve,
    TemperatureProtectionValve,
    Valve,
)


class CommandedValveTests(unittest.TestCase):
    def test_opening_command_is_explicit_and_scales_flow(self):
        properties = HydrogenProperties()
        left = properties.from_pT(300_000.0, 20.0)
        right = properties.from_pT(100_000.0, 20.0)
        commanded = CommandedValve(
            Valve(1.0e-8, 0.8, properties, pressure_samples=16, allow_reverse=False)
        )
        self.assertEqual(commanded.evaluate(left, right).mass_flow_kg_s, 0.0)
        commanded.set_opening(0.5)
        half = commanded.evaluate(left, right).mass_flow_kg_s
        commanded.set_opening(1.0)
        full = commanded.evaluate(left, right).mass_flow_kg_s
        self.assertGreater(half, 0.0)
        self.assertAlmostEqual(half / full, 0.5, delta=1.0e-12)
        with self.assertRaises(ValueError):
            commanded.set_opening(1.1)
        with self.assertRaises(ValueError):
            commanded.set_opening(True)

    def test_stroke_limited_command_moves_at_declared_rate(self):
        properties = HydrogenProperties()
        left = properties.from_pT(300_000.0, 20.0)
        right = properties.from_pT(100_000.0, 20.0)
        actuator = StrokeLimitedCommandedValve(
            Valve(1.0e-8, 0.8, properties, pressure_samples=16, allow_reverse=False),
            stroke_time_s=2.0,
        )
        actuator.set_target_opening(1.0)
        self.assertEqual(actuator.opening, 0.0)
        self.assertEqual(actuator.target_opening, 1.0)
        self.assertEqual(actuator.evaluate(left, right).mass_flow_kg_s, 0.0)
        self.assertAlmostEqual(actuator.advance(0.5), 0.25)
        quarter = actuator.evaluate(left, right).mass_flow_kg_s
        self.assertGreater(quarter, 0.0)
        self.assertAlmostEqual(actuator.advance(1.5), 1.0)
        full = actuator.evaluate(left, right).mass_flow_kg_s
        self.assertAlmostEqual(quarter / full, 0.25, delta=1.0e-12)
        self.assertAlmostEqual(actuator.advance(0.0), 1.0)

    def test_stroke_limited_command_validates_time_and_stroke(self):
        properties = HydrogenProperties()
        base = Valve(1.0e-8, 0.8, properties, pressure_samples=16)
        with self.assertRaises(ValueError):
            StrokeLimitedCommandedValve(base, stroke_time_s=0.0)
        actuator = StrokeLimitedCommandedValve(base, stroke_time_s=1.0)
        with self.assertRaises(ValueError):
            actuator.advance(-0.1)
        with self.assertRaises(ValueError):
            actuator.set_target_opening(1.1)

    def test_pressure_relief_valve_opens_at_set_pressure_and_reseats(self):
        properties = HydrogenProperties()
        left = properties.from_pT(300_000.0, 20.0)
        right = properties.from_pT(100_000.0, 20.0)
        relief = PressureReliefValve(
            Valve(1.0e-8, 0.8, properties, pressure_samples=16, allow_reverse=False),
            set_pressure_Pa=250_000.0,
            reseat_pressure_Pa=200_000.0,
        )
        opened = relief.evaluate(left, right)
        self.assertTrue(relief.is_open)
        self.assertGreater(opened.mass_flow_kg_s, 0.0)
        between = properties.from_pT(225_000.0, 20.0)
        held = relief.evaluate(between, right)
        self.assertTrue(relief.is_open)
        self.assertGreater(held.mass_flow_kg_s, 0.0)
        below = properties.from_pT(180_000.0, 20.0)
        closed = relief.evaluate(below, right)
        self.assertFalse(relief.is_open)
        self.assertEqual(closed.mass_flow_kg_s, 0.0)

    def test_pressure_relief_valve_blocks_reverse_flow(self):
        properties = HydrogenProperties()
        left = properties.from_pT(100_000.0, 20.0)
        right = properties.from_pT(300_000.0, 20.0)
        relief = PressureReliefValve(
            Valve(1.0e-8, 0.8, properties, pressure_samples=16, allow_reverse=True),
            set_pressure_Pa=250_000.0,
            reseat_pressure_Pa=200_000.0,
        )
        result = relief.evaluate(left, right)
        self.assertEqual(result.mass_flow_kg_s, 0.0)
        self.assertFalse(relief.is_open)

    def test_pressure_relief_valve_requires_ordered_pressures(self):
        properties = HydrogenProperties()
        with self.assertRaises(ValueError):
            PressureReliefValve(
                Valve(1.0e-8, 0.8, properties, pressure_samples=16),
                set_pressure_Pa=200_000.0,
                reseat_pressure_Pa=250_000.0,
            )

    def test_temperature_protection_valve_opens_and_reseats(self):
        properties = HydrogenProperties()
        hot = properties.from_pT(300_000.0, 40.0)
        warm = properties.from_pT(300_000.0, 35.0)
        cold = properties.from_pT(300_000.0, 30.0)
        right = properties.from_pT(100_000.0, 20.0)
        trip = TemperatureProtectionValve(
            Valve(1.0e-8, 0.8, properties, pressure_samples=16, allow_reverse=False),
            trip_temperature_K=38.0,
            reset_temperature_K=32.0,
        )
        self.assertGreater(trip.evaluate(hot, right).mass_flow_kg_s, 0.0)
        self.assertTrue(trip.is_open)
        self.assertGreater(trip.evaluate(warm, right).mass_flow_kg_s, 0.0)
        self.assertEqual(trip.evaluate(cold, right).mass_flow_kg_s, 0.0)
        self.assertFalse(trip.is_open)

    def test_temperature_protection_valve_requires_ordered_temperatures(self):
        properties = HydrogenProperties()
        with self.assertRaises(ValueError):
            TemperatureProtectionValve(
                Valve(1.0e-8, 0.8, properties, pressure_samples=16),
                trip_temperature_K=30.0,
                reset_temperature_K=35.0,
            )


if __name__ == "__main__":
    unittest.main()
