"""Auditable observation loading and fixed-parameter validation windows.

This module is deliberately an adapter layer.  It accepts a facility-owned
CSV export, normalises timestamps and engineering units, and returns arrays
that can be compared with a forward model.  It never writes observations into
the model state and contains no optimiser or parameter-fitting path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .validation import SeriesMetrics, compare_series


_PRESSURE_UNITS = {
    "pa": 1.0,
    "kpa": 1.0e3,
    "mpa": 1.0e6,
    "bar": 1.0e5,
    "psi": 6894.757293168,
    "mmh2o": 9.80665,
    "mmh2o@4c": 9.80665,
}
_VALUE_UNITS = {
    "": 1.0,
    "1": 1.0,
    "kg": 1.0,
    "g": 1.0e-3,
    "m3": 1.0,
    "l": 1.0e-3,
    "w": 1.0,
    "kw": 1.0e3,
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
}


def _finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalise_unit(unit: str) -> str:
    return str(unit).strip().lower().replace("²", "2").replace(" ", "")


def convert_engineering_value(
    value: float,
    *,
    unit: str,
    quantity: str = "value",
    pressure_basis: str = "absolute",
    atmospheric_pressure_Pa: float = 101_325.0,
) -> float:
    """Convert one value to the model boundary convention.

    Pressure is returned as absolute Pa, temperature as K, and other values
    use the declared SI-compatible multiplier.  Gauge pressure must be
    declared explicitly; an unlabeled pressure is rejected rather than
    guessed from its magnitude.
    """

    number = _finite(value, "value")
    key = _normalise_unit(unit)
    kind = str(quantity).strip().lower()
    if kind in {"pressure", "differential_pressure"}:
        if key not in _PRESSURE_UNITS:
            raise ValueError(f"unsupported pressure unit: {unit}")
        basis = str(pressure_basis).strip().lower()
        if kind == "differential_pressure":
            if basis != "differential":
                raise ValueError("differential pressure_basis must be 'differential'")
            pressure = number * _PRESSURE_UNITS[key]
            if pressure < 0.0:
                raise ValueError("differential pressure must be non-negative")
            return pressure
        if basis not in {"absolute", "gauge"}:
            raise ValueError("pressure_basis must be 'absolute' or 'gauge'")
        pressure = number * _PRESSURE_UNITS[key]
        if basis == "gauge":
            atmosphere = _finite(atmospheric_pressure_Pa, "atmospheric_pressure_Pa")
            if atmosphere <= 0.0:
                raise ValueError("atmospheric_pressure_Pa must be positive")
            pressure += atmosphere
        if pressure <= 0.0:
            raise ValueError("converted absolute pressure must be positive")
        return pressure
    if kind in {"temperature", "temp"}:
        if key in {"k", "kelvin"}:
            result = number
        elif key in {"c", "degc", "°c", "celsius"}:
            result = number + 273.15
        else:
            raise ValueError(f"unsupported temperature unit: {unit}")
        if result <= 0.0:
            raise ValueError("converted thermodynamic temperature must be positive")
        return result
    if key not in _VALUE_UNITS:
        raise ValueError(f"unsupported unit for {quantity}: {unit}")
    return number * _VALUE_UNITS[key]


@dataclass(frozen=True)
class ChannelSpec:
    """Mapping from one exported column to a canonical model quantity."""

    source_column: str
    canonical_name: str
    quantity: str = "value"
    unit: str = ""
    pressure_basis: str = "absolute"
    quality_column: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not str(self.source_column).strip() or not str(self.canonical_name).strip():
            raise ValueError("channel column names must be non-empty")
        if self.quantity.lower() not in {"pressure", "differential_pressure", "temperature", "temp", "value"}:
            raise ValueError("unsupported channel quantity")
        if self.quantity.lower() == "differential_pressure" and self.pressure_basis.lower() != "differential":
            raise ValueError("differential-pressure channels require pressure_basis='differential'")


@dataclass(frozen=True)
class ObservationTable:
    """Normalised, immutable observation table.

    ``time_s`` is seconds from the first accepted timestamp.  ``values`` are
    canonical SI values; missing numeric fields remain NaN and are visible in
    ``valid_masks``.  The original timestamp strings are retained for audit.
    """

    time_s: np.ndarray
    timestamps: tuple[datetime, ...]
    original_timestamp_strings: tuple[str, ...]
    values: Mapping[str, np.ndarray]
    valid_masks: Mapping[str, np.ndarray]
    quality_flags: Mapping[str, tuple[str, ...]]
    source_path: str
    repairs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        t = np.asarray(self.time_s, dtype=float)
        if t.ndim != 1 or len(t) == 0 or np.any(~np.isfinite(t)):
            raise ValueError("time_s must be a non-empty finite vector")
        if len(t) > 1 and np.any(np.diff(t) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        if len(self.timestamps) != len(t):
            raise ValueError("timestamps and time_s length mismatch")
        for name, values in self.values.items():
            array = np.asarray(values, dtype=float)
            if array.shape != t.shape:
                raise ValueError(f"values[{name}] length mismatch")
            mask = np.asarray(self.valid_masks[name], dtype=bool)
            if mask.shape != t.shape:
                raise ValueError(f"valid_masks[{name}] length mismatch")

    @property
    def row_count(self) -> int:
        return len(self.time_s)

    def column(self, canonical_name: str, *, valid_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if canonical_name not in self.values:
            raise KeyError(f"unknown observation column: {canonical_name}")
        values = np.asarray(self.values[canonical_name], dtype=float)
        mask = np.asarray(self.valid_masks[canonical_name], dtype=bool)
        if valid_only:
            return self.time_s[mask], values[mask]
        return self.time_s.copy(), values.copy()


def _parse_timestamp(raw: str) -> datetime:
    text = str(raw).strip()
    if not text:
        raise ValueError("timestamp cannot be empty")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"timestamp is not ISO-8601: {raw!r}") from exc


def load_observation_csv(
    path: str | Path,
    channels: Sequence[ChannelSpec],
    *,
    timestamp_column: str = "timestamp",
    sort_timestamps: bool = False,
    duplicate_policy: str = "error",
    quality_ok_values: Iterable[str] = ("", "0", "1", "ok", "good", "valid", "true"),
    atmospheric_pressure_Pa: float = 101_325.0,
    row_filter: Callable[[Mapping[str, str]], bool] | None = None,
) -> ObservationTable:
    """Load and normalise a CSV export with explicit repair policy.

    Sorting or duplicate handling is never implicit.  Set ``sort_timestamps``
    or ``duplicate_policy='keep_first'|'keep_last'`` only when that decision is
    part of the validation record; the selected operation is written to
    ``ObservationTable.repairs``.
    """

    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or timestamp_column not in reader.fieldnames:
            raise ValueError(f"CSV must contain timestamp column {timestamp_column!r}")
        fieldnames = set(reader.fieldnames)
        for channel in channels:
            if channel.required and channel.source_column not in fieldnames:
                raise ValueError(f"CSV is missing required channel {channel.source_column!r}")
            if channel.quality_column and channel.quality_column not in fieldnames:
                raise ValueError(f"CSV is missing quality column {channel.quality_column!r}")
        rows = [row for row in reader if row_filter is None or row_filter(row)]
    if not rows:
        raise ValueError("CSV contains no data rows")
    parsed: list[tuple[datetime, str, dict[str, float], dict[str, str]]] = []
    for row_index, row in enumerate(rows, start=2):
        try:
            timestamp_raw = str(row.get(timestamp_column, ""))
            timestamp = _parse_timestamp(timestamp_raw)
            converted: dict[str, float] = {}
            flags: dict[str, str] = {}
            for channel in channels:
                raw = row.get(channel.source_column, "")
                flag = "" if channel.quality_column is None else str(row.get(channel.quality_column, "")).strip()
                flags[channel.canonical_name] = flag
                if raw is None or not str(raw).strip():
                    converted[channel.canonical_name] = float("nan")
                else:
                    converted[channel.canonical_name] = convert_engineering_value(
                        float(raw), unit=channel.unit, quantity=channel.quantity,
                        pressure_basis=channel.pressure_basis,
                        atmospheric_pressure_Pa=atmospheric_pressure_Pa,
                    )
            parsed.append((timestamp, timestamp_raw, converted, flags))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid CSV row {row_index}: {exc}") from exc

    repairs: list[str] = []
    if any(parsed[index][0] > parsed[index + 1][0] for index in range(len(parsed) - 1)):
        if not sort_timestamps:
            raise ValueError("timestamps are not increasing; set sort_timestamps=True explicitly")
        parsed.sort(key=lambda item: item[0])
        repairs.append("sorted timestamps by explicit policy")
    duplicate_policy = str(duplicate_policy).strip().lower()
    duplicate_indices = [index for index in range(len(parsed) - 1) if parsed[index][0] == parsed[index + 1][0]]
    if duplicate_indices:
        if duplicate_policy == "error":
            raise ValueError("duplicate timestamps found; choose an explicit duplicate_policy")
        if duplicate_policy not in {"keep_first", "keep_last"}:
            raise ValueError("duplicate_policy must be error, keep_first, or keep_last")
        selected: list[tuple[datetime, str, dict[str, float], dict[str, str]]] = []
        for item in parsed:
            if selected and item[0] == selected[-1][0]:
                if duplicate_policy == "keep_last":
                    selected[-1] = item
            else:
                selected.append(item)
        parsed = selected
        repairs.append(f"duplicate timestamps resolved with {duplicate_policy}")

    first = parsed[0][0]
    time_s = np.array([(item[0] - first).total_seconds() for item in parsed], dtype=float)
    if len(time_s) > 1 and np.any(np.diff(time_s) <= 0.0):
        raise ValueError("timestamps must be strictly increasing after repairs")
    values = {
        channel.canonical_name: np.array([item[2][channel.canonical_name] for item in parsed], dtype=float)
        for channel in channels
    }
    accepted = {str(item).strip().lower() for item in quality_ok_values}
    masks: dict[str, np.ndarray] = {}
    flags_by_column: dict[str, tuple[str, ...]] = {}
    for channel in channels:
        flags = tuple(item[3][channel.canonical_name] for item in parsed)
        flags_by_column[channel.canonical_name] = flags
        quality_mask = np.array([flag.strip().lower() in accepted for flag in flags], dtype=bool)
        masks[channel.canonical_name] = np.isfinite(values[channel.canonical_name]) & quality_mask
    return ObservationTable(
        time_s=time_s,
        timestamps=tuple(item[0] for item in parsed),
        original_timestamp_strings=tuple(item[1] for item in parsed),
        values=values,
        valid_masks=masks,
        quality_flags=flags_by_column,
        source_path=str(path_obj),
        repairs=tuple(repairs),
    )


@dataclass(frozen=True)
class QuietWindow:
    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    pressure_span_Pa: float

    @property
    def row_count(self) -> int:
        return self.end_index - self.start_index + 1


def select_quiet_windows(
    table: ObservationTable,
    pressure_column: str,
    *,
    min_duration_s: float = 6.0 * 3600.0,
    max_duration_s: float = 12.0 * 3600.0,
    min_pressure_span_Pa: float = 10_000.0,
    event_columns: Sequence[str] = (),
) -> tuple[QuietWindow, ...]:
    """Return stable-boundary windows suitable for blind forward validation.

    A window is contiguous finite pressure data with no change in the listed
    discrete event columns.  The pressure span must meet the explicit lower
    bound, so a nearly constant trace cannot appear accurate only because its
    denominator is tiny.  The returned window can be split into an
    initialization portion and a blind prediction portion with
    :func:`split_validation_window`.
    """

    if min_duration_s <= 0.0 or max_duration_s < min_duration_s:
        raise ValueError("duration bounds are invalid")
    if min_pressure_span_Pa < 0.0:
        raise ValueError("min_pressure_span_Pa must be non-negative")
    time, pressure = table.column(pressure_column)
    valid = table.valid_masks[pressure_column] & np.isfinite(pressure)
    for column in event_columns:
        if column not in table.values:
            raise KeyError(f"unknown event column: {column}")
        valid &= table.valid_masks[column]
    starts: list[int] = []
    index = 0
    while index < len(time):
        if not valid[index]:
            index += 1
            continue
        start = index
        index += 1
        while index < len(time) and valid[index]:
            if event_columns and any(
                table.values[column][index] != table.values[column][index - 1]
                for column in event_columns
            ):
                break
            index += 1
        end = index - 1
        if end <= start:
            continue
        # Use the longest allowed duration ending at each contiguous segment.
        cursor = start
        while cursor < end:
            limit = cursor
            while limit < end and time[limit + 1] - time[cursor] <= max_duration_s:
                limit += 1
            while limit > cursor and time[limit] - time[cursor] >= min_duration_s:
                span = float(np.ptp(pressure[cursor:limit + 1]))
                if span >= min_pressure_span_Pa:
                    starts.append(cursor)
                    break
                limit -= 1
            cursor = max(cursor + 1, limit + 1)
    windows: list[QuietWindow] = []
    for start in starts:
        end = start
        while end + 1 < len(time) and time[end + 1] - time[start] <= max_duration_s and valid[end + 1]:
            end += 1
        duration = float(time[end] - time[start])
        if duration < min_duration_s:
            continue
        windows.append(QuietWindow(start, end, float(time[start]), float(time[end]), duration, float(np.ptp(pressure[start:end + 1]))))
    # Deduplicate overlapping candidates while retaining the first occurrence.
    unique: list[QuietWindow] = []
    seen: set[tuple[int, int]] = set()
    for window in windows:
        key = (window.start_index, window.end_index)
        if key not in seen:
            unique.append(window)
            seen.add(key)
    return tuple(unique)


@dataclass(frozen=True)
class ValidationSplit:
    window: QuietWindow
    initialization_start_index: int
    initialization_end_index: int
    blind_start_index: int
    blind_end_index: int
    blind_start_time_s: float
    blind_end_time_s: float

    @property
    def blind_duration_s(self) -> float:
        return self.blind_end_time_s - self.blind_start_time_s


def split_validation_window(
    table: ObservationTable,
    window: QuietWindow,
    *,
    initialization_s: float = 3600.0,
    blind_duration_s: float | None = None,
) -> ValidationSplit:
    """Split a selected window without using blind observations for setup."""

    if initialization_s <= 0.0:
        raise ValueError("initialization_s must be positive")
    init_end_time = window.start_time_s + initialization_s
    init_end = int(np.searchsorted(table.time_s, init_end_time, side="right")) - 1
    blind_start = init_end + 1
    blind_end = window.end_index
    if blind_duration_s is not None:
        if blind_duration_s <= 0.0:
            raise ValueError("blind_duration_s must be positive")
        blind_end = int(np.searchsorted(table.time_s, table.time_s[blind_start] + blind_duration_s, side="right")) - 1
        blind_end = min(blind_end, window.end_index)
    if init_end < window.start_index or blind_start > blind_end:
        raise ValueError("window is too short for the requested initialization and blind periods")
    return ValidationSplit(
        window, window.start_index, init_end, blind_start, blind_end,
        float(table.time_s[blind_start]), float(table.time_s[blind_end]),
    )


def compare_forward_window(
    model_time_s: Sequence[float],
    model_values: Sequence[float],
    table: ObservationTable,
    split: ValidationSplit,
    observed_column: str,
    *,
    capacity_reference: float | None = None,
    uncertainty_column: str | None = None,
) -> SeriesMetrics:
    """Compare a model trajectory only against the split's blind interval."""

    model_t = np.asarray(model_time_s, dtype=float)
    model_y = np.asarray(model_values, dtype=float)
    values = table.values[observed_column]
    valid = table.valid_masks[observed_column].copy()
    valid[:split.blind_start_index] = False
    valid[split.blind_end_index + 1:] = False
    if uncertainty_column is None:
        uncertainty = None
    else:
        uncertainty = table.values[uncertainty_column]
        valid &= table.valid_masks[uncertainty_column]
    if not np.any(valid):
        raise ValueError("blind interval contains no valid observations")
    return compare_series(
        model_t,
        model_y,
        table.time_s[valid],
        values[valid],
        None if uncertainty is None else uncertainty[valid],
        capacity_reference=capacity_reference,
        require_full_observation_coverage=True,
    )


__all__ = [
    "ChannelSpec",
    "ObservationTable",
    "QuietWindow",
    "ValidationSplit",
    "compare_forward_window",
    "convert_engineering_value",
    "load_observation_csv",
    "select_quiet_windows",
    "split_validation_window",
]
