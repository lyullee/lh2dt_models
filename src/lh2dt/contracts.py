"""Runtime contracts for replaceable two-port component results.

The network solver deliberately accepts any component implementing
``evaluate(left, right)``.  This module validates the small result contract at
that boundary so a replaceable model cannot silently inject a non-finite flow,
enthalpy, or momentum residual into a conservation solve.
"""

from __future__ import annotations

import math
from typing import Any


_OPTIONAL_NUMERIC_FIELDS = (
    "momentum_residual_Pa",
    "pressure_drop_Pa",
    "heat_into_fluid_W",
    "thermal_power_W",
    "hydraulic_outlet_specific_enthalpy_J_kg",
    "throat_pressure_Pa",
    "mass_flux_kg_m2_s",
)

# Keep this aligned with the zero-flow branch used by the link wrappers.  A
# component may expose a tiny numerical flow while converging to a closed
# state; treating that value as zero avoids rejecting an otherwise valid
# replaceable component because of round-off noise.
_ZERO_FLOW_TOLERANCE_KG_S = 1.0e-14


def _finite_number(value: Any, field: str, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} {field} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} {field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} {field} must be finite")
    return number


def validate_two_port_result(result: object, *, context: str = "two-port result") -> None:
    """Validate the common result fields returned by a two-port component.

    Required fields are the signed mass flow and downstream specific
    enthalpy.  Optional fields are checked only when a component exposes them,
    preserving the intentionally extensible protocol.  The flow sign remains
    the network convention: positive is left-to-right and negative is
    right-to-left.
    """

    for field in ("mass_flow_kg_s", "outlet_specific_enthalpy_J_kg"):
        if not hasattr(result, field):
            raise ValueError(f"{context} is missing required field {field}")
        _finite_number(getattr(result, field), field, context)

    for field in _OPTIONAL_NUMERIC_FIELDS:
        if hasattr(result, field):
            value = getattr(result, field)
            if value is not None:
                _finite_number(value, field, context)

    if hasattr(result, "upstream_side"):
        upstream_side = getattr(result, "upstream_side")
        if upstream_side not in {None, "left", "right"}:
            raise ValueError(
                f"{context} upstream_side must be None, 'left', or 'right'"
            )
        mass_flow = float(getattr(result, "mass_flow_kg_s"))
        if abs(mass_flow) <= _ZERO_FLOW_TOLERANCE_KG_S:
            if upstream_side is not None:
                raise ValueError(
                    f"{context} zero-flow result must have upstream_side=None"
                )
        elif mass_flow > 0.0 and upstream_side != "left":
            raise ValueError(
                f"{context} positive flow must have upstream_side='left'"
            )
        elif mass_flow < 0.0 and upstream_side != "right":
            raise ValueError(
                f"{context} negative flow must have upstream_side='right'"
            )


def validate_transmitted_thermo_state(
    *,
    mass_flow_kg_s: float,
    outlet_specific_enthalpy_J_kg: float,
    downstream_pressure_Pa: float,
    properties: Any,
    context: str = "two-port result",
    zero_flow_tolerance_kg_s: float = 1.0e-14,
) -> None:
    """Check that a transported enthalpy has a valid EOS state downstream.

    The generic two-port protocol deliberately does not require a particular
    property package.  A network that *does* have its shared EOS can perform
    this second boundary check after a link has returned a non-zero stream.
    Zero-flow results carry no transported material, so their placeholder
    enthalpy is intentionally not interpreted as a thermodynamic state.
    """

    mass = _finite_number(mass_flow_kg_s, "mass_flow_kg_s", context)
    enthalpy = _finite_number(
        outlet_specific_enthalpy_J_kg,
        "outlet_specific_enthalpy_J_kg",
        context,
    )
    pressure = _finite_number(downstream_pressure_Pa, "downstream_pressure_Pa", context)
    if abs(mass) <= zero_flow_tolerance_kg_s:
        return
    try:
        properties.from_ph(pressure, enthalpy)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            f"{context} transmitted enthalpy does not define a valid downstream EOS state"
        ) from exc


def validate_thermal_link_result(
    result: object,
    *,
    context: str = "thermal link result",
    zero_flow_tolerance_kg_s: float = _ZERO_FLOW_TOLERANCE_KG_S,
) -> None:
    """Validate the energy closure reported by a hydraulic/thermal link.

    A thermal link exposes the hydraulic outlet enthalpy before the thermal
    unit and the transported outlet enthalpy after it. For non-zero flow, the
    reported heat rate must equal ``abs(mdot) * (h_out - h_hydraulic)``.
    """

    validate_two_port_result(result, context=context)
    for field in ("thermal_power_W", "hydraulic_outlet_specific_enthalpy_J_kg"):
        if not hasattr(result, field):
            raise ValueError(f"{context} is missing required field {field}")
        _finite_number(getattr(result, field), field, context)

    mass_flow = float(getattr(result, "mass_flow_kg_s"))
    if abs(mass_flow) <= zero_flow_tolerance_kg_s:
        return
    outlet_h = float(getattr(result, "outlet_specific_enthalpy_J_kg"))
    hydraulic_h = float(getattr(result, "hydraulic_outlet_specific_enthalpy_J_kg"))
    thermal_power = float(getattr(result, "thermal_power_W"))
    expected_power = abs(mass_flow) * (outlet_h - hydraulic_h)
    scale = max(abs(expected_power), abs(thermal_power), 1.0)
    if abs(thermal_power - expected_power) > max(1.0e-6, 1.0e-8 * scale):
        raise ValueError(
            f"{context} thermal energy closure mismatch: "
            f"reported={thermal_power:.12g} W, expected={expected_power:.12g} W"
        )


__all__ = [
    "validate_thermal_link_result",
    "validate_transmitted_thermo_state",
    "validate_two_port_result",
]
