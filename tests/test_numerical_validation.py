import numpy as np
import pytest

from lh2dt import compare_step_refinement


def test_step_refinement_exposes_unresolved_terminal_pressure():
    # The same initial pressure and 60 s endpoint are compared on two grids.
    # A good conservation residual cannot make the 90 kPa coarse endpoint
    # agree with the 200 kPa refined endpoint.
    comparison = compare_step_refinement(
        np.array([0.0, 60.0]),
        np.array([400_000.0, 90_000.0]),
        np.array([0.0, 30.0, 60.0]),
        np.array([400_000.0, 230_000.0, 200_000.0]),
        reference_scale=400_000.0,
    )
    assert comparison.endpoint_absolute_difference == 110_000.0
    assert comparison.endpoint_difference_fraction_of_scale == pytest.approx(0.275)
    assert comparison.maximum_absolute_difference == 110_000.0
    assert comparison.coarse_max_step_s == 60.0
    assert comparison.refined_max_step_s == 30.0


def test_step_refinement_rejects_nonmatching_time_horizons():
    with pytest.raises(ValueError, match="same initial and final"):
        compare_step_refinement(
            np.array([0.0, 60.0]), np.array([1.0, 2.0]),
            np.array([0.0, 20.0, 40.0]), np.array([1.0, 1.5, 2.0]),
            reference_scale=1.0,
        )


def test_step_refinement_rejects_invalid_grid_and_scale():
    with pytest.raises(ValueError, match="strictly increasing"):
        compare_step_refinement(
            np.array([0.0, 60.0]), np.array([1.0, 2.0]),
            np.array([0.0, 30.0, 30.0, 60.0]), np.array([1.0, 1.5, 1.5, 2.0]),
            reference_scale=1.0,
        )
    with pytest.raises(ValueError, match="reference_scale"):
        compare_step_refinement(
            np.array([0.0, 60.0]), np.array([1.0, 2.0]),
            np.array([0.0, 30.0, 60.0]), np.array([1.0, 1.5, 2.0]),
            reference_scale=True,
        )
