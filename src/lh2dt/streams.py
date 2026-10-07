"""Conservative stream contracts used by every component."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class MassEnergyFlow:
    """A signed flow into a control volume.

    ``mass_flow_kg_s`` is positive into the receiving control volume.  The
    enthalpy belongs to the material crossing the boundary, so energy transport
    is exactly ``mass_flow * specific_enthalpy``.
    """

    mass_flow_kg_s: float
    specific_enthalpy_J_kg: float
    source: str = ""

    @property
    def enthalpy_flow_W(self) -> float:
        return self.mass_flow_kg_s * self.specific_enthalpy_J_kg


@dataclass(frozen=True)
class EnergyTransfer:
    heat_W: float = 0.0
    shaft_work_into_fluid_W: float = 0.0

    @property
    def total_into_fluid_W(self) -> float:
        return self.heat_W + self.shaft_work_into_fluid_W


def mix_streams(streams: Iterable[MassEnergyFlow]) -> MassEnergyFlow:
    """Mix positive incoming streams by exact mass and enthalpy balance."""

    items = list(streams)
    if not items:
        raise ValueError("At least one incoming stream is required")
    if any(not math.isfinite(s.mass_flow_kg_s) or s.mass_flow_kg_s <= 0.0 for s in items):
        raise ValueError("Mixing accepts positive finite incoming mass flows")
    mass = sum(s.mass_flow_kg_s for s in items)
    energy = sum(s.enthalpy_flow_W for s in items)
    return MassEnergyFlow(mass, energy / mass, source="mixed")


def split_stream(stream: MassEnergyFlow, fractions: Iterable[float]) -> tuple[MassEnergyFlow, ...]:
    """Split a stream without changing its specific enthalpy."""

    values = tuple(float(x) for x in fractions)
    if not values or any(x < 0.0 or not math.isfinite(x) for x in values):
        raise ValueError("Split fractions must be finite and non-negative")
    if abs(sum(values) - 1.0) > 1e-12:
        raise ValueError("Split fractions must sum to one")
    return tuple(
        MassEnergyFlow(stream.mass_flow_kg_s * x, stream.specific_enthalpy_J_kg, stream.source)
        for x in values
    )

