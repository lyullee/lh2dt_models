"""Optional ortho/para hydrogen composition physics.

The default short-duration calculations use a para-hydrogen property package
and do not need an independent composition state. This module provides the
state and energy transform required when a long-duration, normal-hydrogen, or
catalytic-conversion boundary is explicitly declared.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


# Rotational temperature of H2, K.  This is the rigid-rotor constant used by
# the equilibrium partition-function closure; it is independent of facility
# data and is intentionally exposed in the module for traceability.
ROTATIONAL_TEMPERATURE_K = 85.4


@dataclass(frozen=True)
class OrthoParaKineticsParameters:
    """Explicit composition-boundary parameters.

    ``conversion_rate_constant_s`` is zero for a frozen composition and must
    be supplied when a catalyst or a validated long-duration conversion
    boundary is present.  ``conversion_enthalpy_J_kg`` is the positive heat
    released per kilogram of para-fraction increase; callers may set it to
    zero when only composition, not the energy coupling, is under study.
    """

    conversion_rate_constant_s: float = 0.0
    conversion_enthalpy_J_kg: float = 0.0
    rotational_temperature_K: float = ROTATIONAL_TEMPERATURE_K
    maximum_rotational_level: int = 24

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.conversion_rate_constant_s)
            or self.conversion_rate_constant_s < 0.0
        ):
            raise ValueError("conversion_rate_constant_s must be finite and non-negative")
        if (
            not math.isfinite(self.conversion_enthalpy_J_kg)
            or self.conversion_enthalpy_J_kg < 0.0
        ):
            raise ValueError("conversion_enthalpy_J_kg must be finite and non-negative")
        if (
            not math.isfinite(self.rotational_temperature_K)
            or self.rotational_temperature_K <= 0.0
        ):
            raise ValueError("rotational_temperature_K must be finite and positive")
        if not isinstance(self.maximum_rotational_level, int) or self.maximum_rotational_level < 1:
            raise ValueError("maximum_rotational_level must be a positive integer")


@dataclass(frozen=True)
class OrthoParaState:
    """Mass-independent para fraction state in the interval ``[0, 1]``."""

    para_fraction: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.para_fraction) or not 0.0 <= self.para_fraction <= 1.0:
            raise ValueError("para_fraction must be finite and lie in [0, 1]")

    @property
    def ortho_fraction(self) -> float:
        return 1.0 - self.para_fraction


def equilibrium_para_fraction(
    temperature_K: float,
    *,
    rotational_temperature_K: float = ROTATIONAL_TEMPERATURE_K,
    maximum_rotational_level: int = 24,
) -> float:
    """Return the rigid-rotor equilibrium para fraction.

    Even rotational levels carry para nuclear-spin weight one and odd levels
    carry ortho weight three.  The finite sum is deterministic, bounded and
    suitable for a reduced-order tank state; increasing the level cutoff is a
    caller-controlled accuracy choice rather than a fitted parameter.
    """

    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    if not math.isfinite(rotational_temperature_K) or rotational_temperature_K <= 0.0:
        raise ValueError("rotational_temperature_K must be finite and positive")
    if not isinstance(maximum_rotational_level, int) or maximum_rotational_level < 1:
        raise ValueError("maximum_rotational_level must be a positive integer")
    para_partition = 0.0
    ortho_partition = 0.0
    for level in range(maximum_rotational_level + 1):
        term = (2.0 * level + 1.0) * math.exp(
            -rotational_temperature_K * level * (level + 1) / temperature_K
        )
        if level % 2 == 0:
            para_partition += term
        else:
            ortho_partition += 3.0 * term
    total = para_partition + ortho_partition
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("ortho/para partition function is not finite")
    return para_partition / total


def para_fraction_rate_s(
    state: OrthoParaState,
    temperature_K: float,
    parameters: OrthoParaKineticsParameters,
) -> float:
    """Evaluate the first-order relaxation rate toward equilibrium."""

    target = equilibrium_para_fraction(
        temperature_K,
        rotational_temperature_K=parameters.rotational_temperature_K,
        maximum_rotational_level=parameters.maximum_rotational_level,
    )
    return parameters.conversion_rate_constant_s * (target - state.para_fraction)


def advance_ortho_para_state(
    state: OrthoParaState,
    temperature_K: float,
    time_step_s: float,
    parameters: OrthoParaKineticsParameters,
) -> OrthoParaState:
    """Advance composition with the exact constant-temperature relaxation step."""

    if not math.isfinite(time_step_s) or time_step_s < 0.0:
        raise ValueError("time_step_s must be finite and non-negative")
    target = equilibrium_para_fraction(
        temperature_K,
        rotational_temperature_K=parameters.rotational_temperature_K,
        maximum_rotational_level=parameters.maximum_rotational_level,
    )
    decay = math.exp(-parameters.conversion_rate_constant_s * time_step_s)
    para = target + (state.para_fraction - target) * decay
    # Roundoff at the limits is clipped only to the mathematical state domain;
    # No observation is involved in this projection.
    return OrthoParaState(min(1.0, max(0.0, para)))


def conversion_heat_rate_W(
    state: OrthoParaState,
    temperature_K: float,
    total_hydrogen_mass_kg: float,
    parameters: OrthoParaKineticsParameters,
) -> float:
    """Return heat released to the tank by the current composition change."""

    if not math.isfinite(total_hydrogen_mass_kg) or total_hydrogen_mass_kg < 0.0:
        raise ValueError("total_hydrogen_mass_kg must be finite and non-negative")
    return (
        total_hydrogen_mass_kg
        * parameters.conversion_enthalpy_J_kg
        * para_fraction_rate_s(state, temperature_K, parameters)
    )


__all__ = [
    "ROTATIONAL_TEMPERATURE_K",
    "OrthoParaKineticsParameters",
    "OrthoParaState",
    "equilibrium_para_fraction",
    "para_fraction_rate_s",
    "advance_ortho_para_state",
    "conversion_heat_rate_W",
]
