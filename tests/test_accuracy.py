import numpy as np

from lh2dt import AccuracyTargetDefinition, assess_accuracy_target, compare_series


def _target(status: str = "confirmed", accuracy: float = 94.0, error: float | None = 6.0):
    return AccuracyTargetDefinition(
        target_accuracy_percent=accuracy,
        target_error_percent=error,
        metric_name="nMAPE",
        scope="steady-state simulation versus field data",
        denominator="explicit equipment capacity",
        status=status,
        source_file="evidence.pdf",
        source_locator="p.1",
        reviewer="reviewer",
        reviewed_at="2026-01-01T00:00:00Z",
    )


def _metrics(error: float, capacity: float = 10.0):
    return compare_series(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 1.5]),
        np.array([error, error]),
        capacity_reference=capacity,
    )


def test_unconfirmed_94_percent_target_is_blocked():
    result = assess_accuracy_target(_metrics(0.6), _target(status="unconfirmed"))
    assert result.status == "blocked"
    assert result.accuracy_percent is None
    assert any("not confirmed" in reason for reason in result.blockers)


def test_confirmed_target_converts_nmape_to_accuracy_without_fitting():
    result = assess_accuracy_target(_metrics(0.6), _target())
    assert result.status == "accepted"
    assert result.error_percent == 6.0
    assert result.accuracy_percent == 94.0
    assert result.blockers == ()


def test_confirmed_target_rejects_below_target():
    result = assess_accuracy_target(_metrics(0.7), _target())
    assert result.status == "rejected"
    assert result.accuracy_percent == 93.0


def test_target_definition_requires_consistent_accuracy_and_error():
    target = _target(error=5.0)
    result = assess_accuracy_target(_metrics(0.6), target)
    assert result.status == "blocked"
    assert any("inconsistent" in reason for reason in result.blockers)


def test_target_requires_explicit_capacity_metric():
    metrics = compare_series(
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([0.5]),
        np.array([0.5]),
    )
    result = assess_accuracy_target(metrics, _target())
    assert result.status == "blocked"
    assert result.blockers == ("nMAPE requires an explicit capacity_reference",)
