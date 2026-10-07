import math
import unittest

from lh2dt import (
    HomogeneousTank,
    HorizontalVesselGeometry,
    HydrostaticDifferentialPressure,
    TankGeometry,
)


class HorizontalGeometryTests(unittest.TestCase):
    def test_ellipsoidal_head_volume_matches_analytic_formula(self):
        geometry = HorizontalVesselGeometry(2.0, 5.0, 0.5)
        expected = math.pi * 1.0**2 * 5.0 + 4.0 / 3.0 * math.pi * 0.5 * 1.0**2
        self.assertAlmostEqual(geometry.volume_m3, expected, places=12)
        self.assertAlmostEqual(geometry.at_height(2.0).liquid_volume_m3, expected, delta=2e-10)

    def test_half_fill_symmetry_and_interface_area(self):
        radius = 1.0
        straight = 5.0
        head_depth = 0.5
        geometry = HorizontalVesselGeometry(2.0 * radius, straight, head_depth)
        half = geometry.at_height(radius)
        self.assertAlmostEqual(half.fill_volume_fraction, 0.5, delta=2e-12)
        expected_interface = 2.0 * radius * straight + math.pi * head_depth * radius
        self.assertAlmostEqual(half.interface_area_m2, expected_interface, delta=2e-10)
        self.assertAlmostEqual(half.wetted_inner_area_m2, half.dry_inner_area_m2, delta=2e-10)

    def test_volume_height_roundtrip_and_complement(self):
        geometry = HorizontalVesselGeometry(2.4, 8.0, 0.6)
        for fraction in (0.01, 0.1, 0.35, 0.5, 0.8, 0.99):
            volume = geometry.volume_m3 * fraction
            height = geometry.height_from_liquid_volume(volume)
            state = geometry.at_height(height)
            self.assertAlmostEqual(state.liquid_volume_m3, volume, delta=1e-10 * geometry.volume_m3)
            complement = geometry.at_height(geometry.diameter - height)
            self.assertAlmostEqual(
                state.liquid_volume_m3 + complement.liquid_volume_m3,
                geometry.volume_m3,
                delta=2e-10 * geometry.volume_m3,
            )

    def test_flat_head_limit_includes_end_plates(self):
        geometry = HorizontalVesselGeometry(2.0, 5.0, 0.0)
        self.assertAlmostEqual(geometry.volume_m3, 5.0 * math.pi, places=12)
        self.assertAlmostEqual(
            geometry.inner_surface_area_m2,
            2.0 * math.pi * 1.0 * 5.0 + 2.0 * math.pi,
            places=12,
        )

    def test_geometry_rejects_boolean_and_nonfinite_numeric_inputs(self):
        base = {
            "inner_diameter_m": 2.0,
            "straight_length_m": 5.0,
            "head_depth_m": 0.5,
        }
        for name in ("inner_diameter_m", "straight_length_m", "head_depth_m"):
            values = dict(base)
            values[name] = True
            with self.assertRaises(ValueError):
                HorizontalVesselGeometry(**values)
        with self.assertRaises(ValueError):
            HorizontalVesselGeometry(**base, quadrature_order=True)
        geometry = HorizontalVesselGeometry(**base)
        with self.assertRaises(ValueError):
            geometry.at_height(True)
        with self.assertRaises(ValueError):
            geometry.at_height(float("nan"))
        with self.assertRaises(ValueError):
            geometry.height_from_liquid_volume(True)
        with self.assertRaises(ValueError):
            geometry.height_from_liquid_volume(float("inf"))

    def test_homogeneous_tank_level_uses_phase_inventory(self):
        shape = HorizontalVesselGeometry(2.0, 5.0, 0.5)
        tank = HomogeneousTank(TankGeometry(shape.volume_m3, 1e6, 1.0, 20.0))
        state = tank.initialize_saturated(300_000.0, 0.5)
        inventory = tank.phase_inventory(state)
        self.assertAlmostEqual(
            inventory.liquid_volume_m3 + inventory.vapor_volume_m3,
            shape.volume_m3,
            delta=1e-8,
        )
        self.assertAlmostEqual(tank.liquid_level_m(state, shape), shape.radius, delta=1e-8)


class DifferentialPressureTests(unittest.TestCase):
    def test_dp_and_height_inverse(self):
        sensor = HydrostaticDifferentialPressure(0.0, 2.0)
        height = 1.2
        dp = sensor.evaluate(height, 70.0, 2.0)
        reconstructed = sensor.height_from_differential_pressure(dp, 70.0, 2.0)
        self.assertAlmostEqual(reconstructed, height, places=12)

    def test_vapor_head_is_retained(self):
        sensor = HydrostaticDifferentialPressure(0.0, 2.0)
        dp = sensor.evaluate(1.0, 70.0, 2.0)
        expected = 9.80665 * (70.0 * 1.0 + 2.0 * 1.0)
        self.assertAlmostEqual(dp, expected, places=12)

    def test_out_of_range_dpt_returns_bounded_tap_span_instead_of_exact_level(self):
        sensor = HydrostaticDifferentialPressure(0.0, 2.0)
        envelope = sensor.height_envelope_from_differential_pressure(
            1.0,
            70.0,
            2.0,
            valid_pressure_range_Pa=(10.0, 100.0),
        )
        self.assertEqual(envelope.status, "below_valid_range")
        self.assertFalse(envelope.exact)
        self.assertEqual(envelope.lower_height_m, 0.0)
        self.assertEqual(envelope.upper_height_m, 2.0)

    def test_dpt_resolution_is_reported_as_an_interval(self):
        sensor = HydrostaticDifferentialPressure(0.0, 2.0)
        reported = sensor.evaluate(1.2, 70.0, 2.0)
        envelope = sensor.height_envelope_from_differential_pressure(
            reported,
            70.0,
            2.0,
            valid_pressure_range_Pa=(0.0, 2000.0),
            uncertainty_Pa=5.0,
        )
        self.assertEqual(envelope.status, "resolved_interval")
        self.assertFalse(envelope.exact)
        self.assertLess(envelope.lower_height_m, 1.2)
        self.assertGreater(envelope.upper_height_m, 1.2)


if __name__ == "__main__":
    unittest.main()
