import csv

import numpy as np
import pytest

from lh2dt import (
    ChannelSpec,
    compare_forward_window,
    convert_engineering_value,
    load_observation_csv,
    select_quiet_windows,
    split_validation_window,
)


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "PT", "PT_Q", "EVENT"])
        writer.writeheader()
        writer.writerows(rows)


def test_pressure_conversion_requires_explicit_basis():
    assert convert_engineering_value(0.383, unit="MPa", quantity="pressure", pressure_basis="gauge") == pytest.approx(484325.0)
    assert convert_engineering_value(25.0, unit="mmH2O", quantity="differential_pressure", pressure_basis="differential") == pytest.approx(245.16625)
    with pytest.raises(ValueError):
        convert_engineering_value(1.0, unit="MPa", quantity="pressure", pressure_basis="unknown")


def test_csv_loader_normalises_units_and_preserves_quality(tmp_path):
    path = tmp_path / "observations.csv"
    _write_csv(path, [
        {"timestamp": "2026-01-01T00:00:00+09:00", "PT": "0.383", "PT_Q": "good", "EVENT": "0"},
        {"timestamp": "2026-01-01T00:01:00+09:00", "PT": "", "PT_Q": "bad", "EVENT": "0"},
        {"timestamp": "2026-01-01T00:02:00+09:00", "PT": "0.393", "PT_Q": "good", "EVENT": "0"},
    ])
    table = load_observation_csv(path, [
        ChannelSpec("PT", "tank_pressure", "pressure", "MPa", "gauge", "PT_Q"),
        ChannelSpec("EVENT", "event", "value", ""),
    ])
    assert table.time_s.tolist() == [0.0, 60.0, 120.0]
    assert table.values["tank_pressure"][0] == pytest.approx(484325.0)
    assert not table.valid_masks["tank_pressure"][1]
    assert table.quality_flags["tank_pressure"] == ("good", "bad", "good")


def test_csv_loader_can_explicitly_select_one_asset_row_subset(tmp_path):
    path = tmp_path / "multi_asset.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset", "timestamp", "PT"])
        writer.writeheader()
        writer.writerows([
            {"asset": "A", "timestamp": "2026-01-01T00:00:00+00:00", "PT": "1"},
            {"asset": "B", "timestamp": "2026-01-01T00:00:00+00:00", "PT": "2"},
        ])
    table = load_observation_csv(
        path, [ChannelSpec("PT", "pressure", "pressure", "bar", "absolute")],
        row_filter=lambda row: row["asset"] == "B",
    )
    assert table.row_count == 1
    assert table.values["pressure"][0] == pytest.approx(200_000.0)


def test_loader_requires_explicit_timestamp_repair(tmp_path):
    path = tmp_path / "unordered.csv"
    _write_csv(path, [
        {"timestamp": "2026-01-01T00:01:00+00:00", "PT": "1", "PT_Q": "ok", "EVENT": "0"},
        {"timestamp": "2026-01-01T00:00:00+00:00", "PT": "1", "PT_Q": "ok", "EVENT": "0"},
    ])
    spec = [ChannelSpec("PT", "pressure", "pressure", "bar", "absolute")]
    with pytest.raises(ValueError, match="sort_timestamps"):
        load_observation_csv(path, spec)
    table = load_observation_csv(path, spec, sort_timestamps=True)
    assert table.repairs == ("sorted timestamps by explicit policy",)


def test_quiet_window_split_and_blind_comparison(tmp_path):
    path = tmp_path / "long.csv"
    rows = []
    for minute in range(0, 7 * 60 + 1):
        # Seven-hour pressure excursion with a constant command/event state.
        rows.append({
            "timestamp": f"2026-01-01T{minute // 60:02d}:{minute % 60:02d}:00+00:00",
            "PT": str(100_000.0 + 10_000.0 * minute / (7 * 60)),
            "PT_Q": "ok",
            "EVENT": "0",
        })
    _write_csv(path, rows)
    table = load_observation_csv(path, [
        ChannelSpec("PT", "pressure", "pressure", "Pa", "absolute"),
        ChannelSpec("EVENT", "event", "value", ""),
    ])
    windows = select_quiet_windows(
        table, "pressure", min_duration_s=6 * 3600, max_duration_s=7 * 3600,
        min_pressure_span_Pa=10_000.0, event_columns=("event",),
    )
    assert windows
    split = split_validation_window(table, windows[0], initialization_s=3600.0, blind_duration_s=4 * 3600)
    assert split.initialization_end_index < split.blind_start_index
    assert split.blind_duration_s >= 4 * 3600 - 60
    metrics = compare_forward_window(
        table.time_s, table.values["pressure"], table, split, "pressure",
        capacity_reference=1_000_000.0,
    )
    assert metrics.count > 200
    assert metrics.normalized_mae_by_capacity_percent == pytest.approx(0.0)
