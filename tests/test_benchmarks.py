import unittest

from lh2dt import (
    run_nasa_d4171_benchmark,
    run_nasa_d4171_energy_partition_benchmark,
    run_nasa_d4171_layered_conduction_benchmark,
    run_nasa_d4171_layered_schrage_benchmark,
    run_nasa_d4171_radial_axial_benchmark,
    run_nasa_d4171_two_region_limit_benchmark,
)


class NasaD4171BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_nasa_d4171_benchmark()

    def test_dataset_identity_and_cardinality(self):
        self.assertEqual(self.report.source_report, "NASA TN D-4171")
        self.assertEqual(len(self.report.points), 21)
        self.assertEqual({point.test_number for point in self.report.points}, set(range(1, 22)))
        self.assertIn("no parameter is fitted", self.report.assumptions[-1])

    def test_homogeneous_model_is_structurally_rejected(self):
        self.assertTrue(all(point.predicted_over_observed < 1.0 for point in self.report.points))
        ratios = {
            mode: summary.mean_predicted_over_observed
            for mode, summary in self.report.by_heating_mode.items()
        }
        self.assertGreater(ratios["lower"], ratios["uniform"])
        self.assertGreater(ratios["uniform"], ratios["upper"])
        self.assertAlmostEqual(ratios["lower"], 0.8117, delta=0.002)
        self.assertAlmostEqual(ratios["upper"], 0.2319, delta=0.002)

    def test_reported_error_is_not_a_fit_score(self):
        self.assertEqual(self.report.overall.count, 21)
        self.assertGreater(self.report.overall.mean_absolute_percentage_error_percent, 50.0)
        self.assertGreater(self.report.overall.normalized_rmse, 0.35)


class NasaD4171TwoRegionLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_nasa_d4171_two_region_limit_benchmark()

    def test_unfitted_two_region_limit_has_all_tests_and_explicit_scope(self):
        self.assertEqual(len(self.report.points), 21)
        self.assertIn("bounding limit", self.report.model)
        self.assertIn("no parameter is fitted", self.report.assumptions[-1])

    def test_heat_location_response_is_resolved(self):
        ratios = {
            mode: summary.mean_predicted_over_observed
            for mode, summary in self.report.by_heating_mode.items()
        }
        self.assertLess(ratios["lower"], ratios["uniform"])
        self.assertLess(ratios["uniform"], ratios["upper"])
        self.assertAlmostEqual(ratios["lower"], 1.034, delta=0.01)
        self.assertGreater(self.report.overall.mean_predicted_over_observed, 1.3)


class NasaD4171EnergyPartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_nasa_d4171_energy_partition_benchmark()

    def test_corrected_table_two_identity_and_rounding_closure(self):
        self.assertEqual(self.report.source_report, "NASA TN D-4171 Table II")
        self.assertEqual(len(self.report.points), 17)
        self.assertLessEqual(max(
            abs(point.component_closure_error_percent_point)
            for point in self.report.points
        ), 0.11)
        self.assertLessEqual(max(
            abs(point.liquid_closure_error_percent_point)
            for point in self.report.points
        ), 0.11)

    def test_top_heating_requires_large_cross_phase_energy_transfer(self):
        top = self.report.by_heating_mode["upper"]
        bottom = self.report.by_heating_mode["lower"]
        self.assertGreater(top.mean_inferred_cross_phase_transfer_percent, 75.0)
        self.assertLess(abs(bottom.mean_inferred_cross_phase_transfer_percent), 2.0)

    def test_zero_transfer_limit_is_rejected_by_energy_partition(self):
        summary = self.report.overall
        self.assertGreater(summary.zero_transfer_liquid_mae_percent_point, 20.0)
        self.assertGreater(summary.zero_transfer_vapor_mae_percent_point, 30.0)
        self.assertGreater(summary.zero_transfer_evaporation_mae_percent_point, 10.0)


class NasaD4171LayeredConductionTests(unittest.TestCase):
    def test_invalid_time_step_is_rejected_before_simulation(self):
        with self.assertRaises(ValueError):
            run_nasa_d4171_layered_conduction_benchmark(time_step_s=0.0)

    def test_invalid_cell_count_is_rejected_before_simulation(self):
        with self.assertRaises(ValueError):
            run_nasa_d4171_layered_conduction_benchmark(liquid_cell_count=0)

    def test_invalid_schrage_accommodation_is_rejected_before_simulation(self):
        with self.assertRaisesRegex(ValueError, "schrage_accommodation_coefficient"):
            run_nasa_d4171_layered_schrage_benchmark(
                test_numbers=(1,), accommodation_coefficient=0.0
            )

    def test_lower_heating_is_rejected_without_distributed_boiling(self):
        with self.assertRaisesRegex(ValueError, "labeled liquid cell"):
            run_nasa_d4171_layered_conduction_benchmark(
                time_step_s=0.01, test_numbers=(10,)
            )


class NasaD4171RadialAxialTests(unittest.TestCase):
    def test_single_case_mapping_is_unfitted_conservative_and_complete(self):
        report = run_nasa_d4171_radial_axial_benchmark(
            time_step_s=60.0, test_numbers=(1,)
        )
        self.assertEqual(report.source_report, "NASA TN D-4171")
        self.assertEqual(len(report.points), 1)
        self.assertEqual(report.points[0].test_number, 1)
        self.assertIn("radial-axial", report.model)
        self.assertIn("no parameter is fitted", report.assumptions[-1])
        self.assertAlmostEqual(
            report.energy_partitions[0].component_closure_error_percent_point,
            0.0,
            delta=1.0e-10,
        )

    def test_invalid_radial_grid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            run_nasa_d4171_radial_axial_benchmark(
                test_numbers=(1,), liquid_level_count=1
            )
        with self.assertRaises(ValueError):
            run_nasa_d4171_radial_axial_benchmark(
                test_numbers=(1,), wall_volume_fraction=1.0
            )
        with self.assertRaises(ValueError):
            run_nasa_d4171_radial_axial_benchmark(
                test_numbers=(1,), local_loss_coefficient=-1.0
            )

    def test_local_loss_is_exposed_as_an_unfitted_hydraulic_input(self):
        report = run_nasa_d4171_radial_axial_benchmark(
            time_step_s=60.0, test_numbers=(1,), local_loss_coefficient=2.0
        )
        self.assertTrue(any("local-loss coefficient 2" in item for item in report.assumptions))

    def test_adaptive_step_control_is_explicitly_recorded(self):
        report = run_nasa_d4171_radial_axial_benchmark(
            time_step_s=60.0, test_numbers=(1,), adaptive_step_control=True
        )
        self.assertTrue(any("adaptive explicit step control is enabled" in item for item in report.assumptions))
        with self.assertRaises(ValueError):
            run_nasa_d4171_radial_axial_benchmark(
                time_step_s=60.0, test_numbers=(1,), adaptive_step_control=1
            )


if __name__ == "__main__":
    unittest.main()
