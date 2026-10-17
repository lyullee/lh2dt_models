"""Provider-owned evidence for finite-volume accident state feedback.

The LH2 equipment models already solve their pressure and inventory states
inside each transient step.  This module gives those models one small,
deterministic export helper so a downstream risk evaluator can distinguish a
dynamic source from a constant-state diagnostic packet.  The helper does not
infer closure from a mass total: the caller must supply the authoritative
state history and independently calculated residuals.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def build_dynamic_state_closure(
    state_history: Sequence[Mapping[str, Any]],
    *,
    max_abs_inventory_residual_kg: float,
    inventory_tolerance_kg: float,
    max_abs_pressure_residual_pa: float,
    pressure_tolerance_pa: float,
    provenance: str,
) -> dict[str, Any]:
    """Return an explicit ``prism.provider_dynamic_state_closure.v1`` record.

    ``state_history`` must be the provider's own finite-volume states after
    each resolved step (including the initial state).  The digest is over a
    canonical JSON representation and therefore binds the declaration to the
    exact state rows used by the provider.  Residuals are supplied by the
    caller; this function validates their units and bounds but never derives
    them from the release packet.
    """

    if not isinstance(state_history, Sequence) or isinstance(
        state_history, (str, bytes, bytearray)
    ) or not state_history:
        raise ValueError("state_history must contain at least one provider state")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(state_history):
        if not isinstance(row, Mapping):
            raise ValueError(f"state_history[{index}] must be a mapping")
        # Reject NaN/Infinity before hashing: the digest must be stable across
        # Python JSON implementations and safe to re-verify downstream.
        try:
            encoded = json.dumps(
                dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"state_history[{index}] is not JSON-canonical") from error
        rows.append(json.loads(encoded))

    def finite_nonnegative(name: str, value: float, *, positive: bool = False) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric")
        result = float(value)
        if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
            comparator = "positive" if positive else "non-negative"
            raise ValueError(f"{name} must be finite and {comparator}")
        return result

    max_inventory = finite_nonnegative(
        "max_abs_inventory_residual_kg", max_abs_inventory_residual_kg
    )
    inventory_tolerance = finite_nonnegative(
        "inventory_tolerance_kg", inventory_tolerance_kg, positive=True
    )
    max_pressure = finite_nonnegative(
        "max_abs_pressure_residual_pa", max_abs_pressure_residual_pa
    )
    pressure_tolerance = finite_nonnegative(
        "pressure_tolerance_pa", pressure_tolerance_pa, positive=True
    )
    if max_inventory > inventory_tolerance:
        raise ValueError("inventory residual exceeds declared tolerance")
    if max_pressure > pressure_tolerance:
        raise ValueError("pressure residual exceeds declared tolerance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("provenance must be a non-empty string")

    digest_payload = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256(digest_payload).hexdigest()
    return {
        "schema_version": "prism.provider_dynamic_state_closure.v1",
        "inventory_feedback_closed": True,
        "pressure_feedback_closed": True,
        "state_history_digest": digest,
        "max_abs_inventory_residual_kg": max_inventory,
        "inventory_tolerance_kg": inventory_tolerance,
        "max_abs_pressure_residual_pa": max_pressure,
        "pressure_tolerance_pa": pressure_tolerance,
        "provenance": provenance.strip(),
    }


__all__ = ["build_dynamic_state_closure"]
