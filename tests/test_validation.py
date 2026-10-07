import unittest

import numpy as np

from lh2dt import compare_series, evaluate_series_metrics


class ValidationTests(unittest.TestCase):
    def test_metrics_do_not_fit_or_shift_data(self):
        metrics = compare_series(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.5, 1.5]),
            np.array([0.4, 1.7]),
            observed_uncertainty=0.15,
        )
        self.assertEqual(metrics.count, 2)
        self.assertAlmostEqual(metrics.mean_error, -0.05)
        self.assertAlmostEqual(metrics.mean_absolute_error, 0.15)
        self.assertAlmostEqual(metrics.within_uncertainty_fraction, 0.5)

    def test_nonoverlap_rejected(self):
        with self.assertRaises(ValueError):
            compare_series(
                np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                np.array([2.0]), np.array([2.0]),
            )

    def test_partial_observation_coverage_is_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "outside the model time range"):
            compare_series(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 1.0, 2.0]),
                np.array([0.5, 2.5]),
                np.array([0.5, 2.5]),
            )

    def test_partial_observation_coverage_requires_explicit_opt_in(self):
        metrics = compare_series(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.5, 2.5]),
            np.array([0.5, 2.5]),
            require_full_observation_coverage=False,
        )
        self.assertEqual(metrics.count, 1)

    def test_observation_time_reversal_is_rejected_without_repair(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            compare_series(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 1.0, 2.0]),
                np.array([1.5, 0.5]),
                np.array([1.5, 0.5]),
            )

    def test_nonfinite_model_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            compare_series(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, np.nan, 2.0]),
                np.array([0.5]),
                np.array([0.5]),
            )

    def test_uncertainty_must_be_positive_and_finite(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            compare_series(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 1.0, 2.0]),
                np.array([0.5]),
                np.array([0.5]),
                observed_uncertainty=0.0,
            )

    def test_capacity_normalized_mae_is_optional_and_explicit(self):
        metrics = compare_series(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 2.0, 4.0]),
            np.array([0.5, 1.5]),
            np.array([0.0, 2.0]),
            capacity_reference=10.0,
        )
        self.assertAlmostEqual(metrics.mean_absolute_error, 1.0)
        self.assertAlmostEqual(metrics.normalized_mae_by_capacity_percent, 10.0)

    def test_capacity_normalized_mae_rejects_nonpositive_reference(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            compare_series(
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([0.5]),
                np.array([0.5]),
                capacity_reference=0.0,
            )

    def test_explicit_acceptance_criteria_can_accept_reject_or_block(self):
        accepted_metrics = compare_series(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.5, 1.5]),
            np.array([0.5, 1.5]),
            observed_uncertainty=0.1,
            capacity_reference=10.0,
        )
        accepted = evaluate_series_metrics(
            accepted_metrics,
            max_nmape_percent=5.0,
            max_rmse=0.1,
            min_within_uncertainty_fraction=1.0,
        )
        self.assertEqual(accepted.status, "accepted")

        rejected_metrics = compare_series(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 2.0, 4.0]),
            np.array([0.5, 1.5]),
            np.array([0.0, 2.0]),
            capacity_reference=10.0,
        )
        rejected = evaluate_series_metrics(rejected_metrics, max_nmape_percent=5.0)
        self.assertEqual(rejected.status, "rejected")
        self.assertIn("nMAPE", rejected.reasons[0])

        blocked = evaluate_series_metrics(
            rejected_metrics,
            max_nmape_percent=5.0,
            min_within_uncertainty_fraction=0.5,
        )
        self.assertEqual(blocked.status, "blocked")
        self.assertIn("observed_uncertainty", blocked.reasons[0])

    def test_acceptance_criteria_are_explicit_and_validated(self):
        metrics = compare_series(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.5]),
            np.array([0.5]),
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_series_metrics(metrics)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            evaluate_series_metrics(metrics, min_within_uncertainty_fraction=1.1)


if __name__ == "__main__":
    unittest.main()
