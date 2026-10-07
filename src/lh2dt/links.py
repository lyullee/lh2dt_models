"""Composable steady links that couple hydraulics and thermal equipment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

from .properties import HydrogenProperties, ThermoState
from .contracts import (
    validate_thermal_link_result,
    validate_transmitted_thermo_state,
    validate_two_port_result,
)
from .reliquefier import Reliquefier
from .vaporizer import Vaporizer
from .ambient_vaporizer import AmbientAirVaporizer


class HydraulicResult(Protocol):
    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float
    # Optional at runtime; wrappers read it with getattr so generic hydraulic
    # components remain valid without inventing a residual.


class HydraulicComponent(Protocol):
    def evaluate(self, left: ThermoState, right: ThermoState) -> HydraulicResult: ...


@dataclass(frozen=True)
class ThermalLinkResult:
    """Hydraulic plus thermal result with optional hydraulic closure evidence."""
    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float
    thermal_power_W: float
    hydraulic_outlet_specific_enthalpy_J_kg: float
    upstream_side: str | None
    momentum_residual_Pa: float | None = None


@dataclass(frozen=True)
class ParallelThermalLinkResult:
    """Aggregate result with unit-level and common hydraulic closure evidence.

    ``momentum_residual_Pa`` is populated only when every active unit exposes
    the same residual within numerical tolerance.  The full per-unit values
    remain available in ``unit_momentum_residuals_Pa``.
    """

    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float
    thermal_power_W: float
    hydraulic_outlet_specific_enthalpy_J_kg: float
    upstream_side: str | None
    active_unit_ids: tuple[str, ...]
    unit_results: tuple[ThermalLinkResult, ...]
    momentum_residual_Pa: float | None = None
    unit_momentum_residuals_Pa: tuple[float | None, ...] = ()


class ParallelThermalLink:
    """Connect independently modeled two-port thermal units in parallel.

    Each unit sees the same left/right pressure boundary.  Unit flows and
    thermal powers are summed, while outlet enthalpy is mixed by the absolute
    unit mass flows.  Opposite non-zero flow directions are rejected instead
    of silently creating an unmodeled recirculation path.

    ``unit_ids`` and ``active`` are explicit inputs.  The class does not
    infer the number of operating units from a capacity label or a measured
    time history, which keeps an aggregate P&ID node replaceable by a bank of
    separately parameterized components.
    """

    def __init__(
        self,
        links: Sequence[object],
        *,
        unit_ids: Sequence[str] | None = None,
        active: Sequence[bool] | None = None,
    ) -> None:
        if not links:
            raise ValueError("at least one parallel thermal link is required")
        self.links = tuple(links)
        ids = tuple(str(item).strip() for item in (unit_ids or (f"unit-{i + 1}" for i in range(len(self.links)))))
        if len(ids) != len(self.links) or any(not item for item in ids):
            raise ValueError("unit_ids must match links and be non-empty")
        if len(set(ids)) != len(ids):
            raise ValueError("unit_ids must be unique")
        if active is None:
            mask = tuple(True for _ in self.links)
        else:
            if isinstance(active, (str, bytes)) or len(active) != len(self.links):
                raise ValueError("active must match links")
            if any(not isinstance(item, bool) for item in active):
                raise ValueError("active entries must be bool")
            mask = tuple(active)
        self.unit_ids = ids
        self._active = list(mask)

    @property
    def active_unit_ids(self) -> tuple[str, ...]:
        return tuple(unit_id for unit_id, enabled in zip(self.unit_ids, self._active) if enabled)

    @property
    def active_mask(self) -> tuple[bool, ...]:
        """Return the explicit hydraulic participation mask in unit order."""

        return tuple(self._active)

    def set_active(self, unit_id: str, enabled: bool) -> None:
        """Explicitly change one unit's operating state."""

        if not isinstance(enabled, bool):
            raise ValueError("enabled must be bool")
        try:
            index = self.unit_ids.index(str(unit_id))
        except ValueError as exc:
            raise KeyError(f"unknown parallel unit: {unit_id}") from exc
        self._active[index] = enabled

    def evaluate_mass_flow(self, left: ThermoState, right: ThermoState) -> float:
        """Return the active bank's hydraulic flow for a pressure residual.

        Heat exchange changes outlet enthalpy after each unit's hydraulic
        flow is determined.  The complete `evaluate` method still handles
        mixing and energy conservation at the converged network state.
        """

        flows: list[float] = []
        for unit_id, link, enabled in zip(self.unit_ids, self.links, self._active):
            if not enabled:
                continue
            quick = getattr(link, "evaluate_mass_flow", None)
            if callable(quick):
                flow = float(quick(left, right))
            else:
                result = link.evaluate(left, right)
                validate_two_port_result(result, context=f"parallel unit {unit_id}")
                flow = float(result.mass_flow_kg_s)
            if not math.isfinite(flow):
                raise ValueError(f"parallel unit {unit_id} returned non-finite mass flow")
            flows.append(flow)
        signs = {1 if flow > 1.0e-14 else -1 for flow in flows if abs(flow) > 1.0e-14}
        if len(signs) > 1:
            raise ValueError("parallel units returned opposing non-zero flow directions")
        return sum(flows)

    def evaluate(self, left: ThermoState, right: ThermoState) -> ParallelThermalLinkResult:
        selected = tuple(
            (unit_id, link)
            for unit_id, link, enabled in zip(self.unit_ids, self.links, self._active)
            if enabled
        )
        if not selected:
            return ParallelThermalLinkResult(
                0.0,
                left.specific_enthalpy_J_kg,
                0.0,
                left.specific_enthalpy_J_kg,
                None,
                (),
                (),
            )
        results = tuple(link.evaluate(left, right) for _, link in selected)
        for (unit_id, _), result in zip(selected, results):
            validate_thermal_link_result(result, context=f"parallel unit {unit_id}")
        signs = {1 if result.mass_flow_kg_s > 1.0e-14 else -1 for result in results if abs(result.mass_flow_kg_s) > 1.0e-14}
        if len(signs) > 1:
            raise ValueError("parallel units returned opposing non-zero flow directions")
        total_flow = sum(float(result.mass_flow_kg_s) for result in results)
        weights = [abs(float(result.mass_flow_kg_s)) for result in results]
        weight_sum = sum(weights)
        if weight_sum <= 1.0e-14:
            hydraulic_h = left.specific_enthalpy_J_kg
            outlet_h = hydraulic_h
            upstream_side = None
        else:
            hydraulic_h = sum(weight * result.hydraulic_outlet_specific_enthalpy_J_kg for weight, result in zip(weights, results)) / weight_sum
            outlet_h = sum(weight * result.outlet_specific_enthalpy_J_kg for weight, result in zip(weights, results)) / weight_sum
            upstream_side = next((result.upstream_side for result in results if result.upstream_side is not None), None)
        unit_momentum_residuals = tuple(
            None
            if getattr(result, "momentum_residual_Pa", None) is None
            else float(getattr(result, "momentum_residual_Pa"))
            for result in results
        )
        common_momentum_residual = None
        if unit_momentum_residuals and all(value is not None for value in unit_momentum_residuals):
            first = float(unit_momentum_residuals[0])
            if all(abs(float(value) - first) <= 1.0e-8 for value in unit_momentum_residuals[1:]):
                common_momentum_residual = first
        aggregate = ParallelThermalLinkResult(
            mass_flow_kg_s=total_flow,
            outlet_specific_enthalpy_J_kg=outlet_h,
            thermal_power_W=sum(float(result.thermal_power_W) for result in results),
            hydraulic_outlet_specific_enthalpy_J_kg=hydraulic_h,
            upstream_side=upstream_side,
            active_unit_ids=tuple(unit_id for unit_id, _ in selected),
            unit_results=results,
            momentum_residual_Pa=common_momentum_residual,
            unit_momentum_residuals_Pa=unit_momentum_residuals,
        )
        validate_thermal_link_result(aggregate, context="parallel thermal link")
        return aggregate


def _flow_orientation(
    mass_flow_kg_s: float,
    left: ThermoState,
    right: ThermoState,
) -> tuple[ThermoState, float, str | None]:
    if mass_flow_kg_s > 0.0:
        return left, right.pressure_Pa, "left"
    if mass_flow_kg_s < 0.0:
        return right, left.pressure_Pa, "right"
    return left, right.pressure_Pa, None


def _optional_momentum_residual(result: object) -> float | None:
    value = getattr(result, "momentum_residual_Pa", None)
    return None if value is None else float(value)


class VaporizerLink:
    """Hydraulic element followed by a steady vaporizer thermal model."""

    def __init__(
        self,
        hydraulic: HydraulicComponent,
        vaporizer: Vaporizer | AmbientAirVaporizer,
        ambient_temperature_K: float,
        properties: HydrogenProperties | None = None,
    ) -> None:
        if ambient_temperature_K <= 0.0:
            raise ValueError("ambient_temperature_K must be positive")
        self.hydraulic = hydraulic
        self.vaporizer = vaporizer
        self.ambient_temperature_K = float(ambient_temperature_K)
        self.properties = properties or vaporizer.properties

    def evaluate_mass_flow(self, left: ThermoState, right: ThermoState) -> float:
        """Use the unchanged hydraulic law without repeating heat integration."""

        hydraulic = self.hydraulic.evaluate(left, right)
        validate_two_port_result(hydraulic, context="vaporizer hydraulic link")
        return float(hydraulic.mass_flow_kg_s)

    def evaluate(self, left: ThermoState, right: ThermoState) -> ThermalLinkResult:
        hydraulic = self.hydraulic.evaluate(left, right)
        validate_two_port_result(hydraulic, context="vaporizer hydraulic link")
        momentum_residual = _optional_momentum_residual(hydraulic)
        mass_flow = float(hydraulic.mass_flow_kg_s)
        upstream, outlet_pressure, upstream_side = _flow_orientation(
            mass_flow, left, right
        )
        hydraulic_h = float(hydraulic.outlet_specific_enthalpy_J_kg)
        if abs(mass_flow) <= 1.0e-14:
            result = ThermalLinkResult(0.0, hydraulic_h, 0.0, hydraulic_h, None, momentum_residual)
            validate_two_port_result(result, context="vaporizer link")
            return result
        thermal_inlet = self.properties.from_ph(outlet_pressure, hydraulic_h)
        result = self.vaporizer.evaluate(
            thermal_inlet,
            outlet_pressure,
            abs(mass_flow),
            self.ambient_temperature_K,
        )
        result = ThermalLinkResult(
            mass_flow_kg_s=mass_flow,
            outlet_specific_enthalpy_J_kg=result.outlet.specific_enthalpy_J_kg,
            thermal_power_W=result.heat_into_hydrogen_W,
            hydraulic_outlet_specific_enthalpy_J_kg=hydraulic_h,
            upstream_side=upstream_side,
            momentum_residual_Pa=momentum_residual,
        )
        validate_thermal_link_result(result, context="vaporizer link")
        validate_transmitted_thermo_state(
            mass_flow_kg_s=result.mass_flow_kg_s,
            outlet_specific_enthalpy_J_kg=result.outlet_specific_enthalpy_J_kg,
            downstream_pressure_Pa=outlet_pressure,
            properties=self.properties,
            context="vaporizer link",
        )
        return result


class ReliquefierLink:
    """Hydraulic element followed by a fixed-capacity reliquefier."""

    def __init__(
        self,
        hydraulic: HydraulicComponent,
        reliquefier: Reliquefier,
        target_quality: float = 0.0,
        properties: HydrogenProperties | None = None,
    ) -> None:
        if not 0.0 <= target_quality <= 1.0:
            raise ValueError("target_quality must be in [0, 1]")
        self.hydraulic = hydraulic
        self.reliquefier = reliquefier
        self.target_quality = float(target_quality)
        self.properties = properties or reliquefier.properties

    def evaluate_mass_flow(self, left: ThermoState, right: ThermoState) -> float:
        """Use the unchanged hydraulic law during junction pressure solves."""

        hydraulic = self.hydraulic.evaluate(left, right)
        validate_two_port_result(hydraulic, context="reliquefier hydraulic link")
        return float(hydraulic.mass_flow_kg_s)

    def evaluate(self, left: ThermoState, right: ThermoState) -> ThermalLinkResult:
        hydraulic = self.hydraulic.evaluate(left, right)
        validate_two_port_result(hydraulic, context="reliquefier hydraulic link")
        momentum_residual = _optional_momentum_residual(hydraulic)
        mass_flow = float(hydraulic.mass_flow_kg_s)
        upstream, outlet_pressure, upstream_side = _flow_orientation(
            mass_flow, left, right
        )
        hydraulic_h = float(hydraulic.outlet_specific_enthalpy_J_kg)
        if abs(mass_flow) <= 1.0e-14:
            result = ThermalLinkResult(0.0, hydraulic_h, 0.0, hydraulic_h, None, momentum_residual)
            validate_two_port_result(result, context="reliquefier link")
            return result
        thermal_inlet = self.properties.from_ph(outlet_pressure, hydraulic_h)
        result = self.reliquefier.evaluate(
            thermal_inlet,
            outlet_pressure,
            abs(mass_flow),
            self.target_quality,
        )
        result = ThermalLinkResult(
            mass_flow_kg_s=mass_flow,
            outlet_specific_enthalpy_J_kg=result.outlet.specific_enthalpy_J_kg,
            thermal_power_W=-result.heat_removed_W,
            hydraulic_outlet_specific_enthalpy_J_kg=hydraulic_h,
            upstream_side=upstream_side,
            momentum_residual_Pa=momentum_residual,
        )
        validate_thermal_link_result(result, context="reliquefier link")
        validate_transmitted_thermo_state(
            mass_flow_kg_s=result.mass_flow_kg_s,
            outlet_specific_enthalpy_J_kg=result.outlet_specific_enthalpy_J_kg,
            downstream_pressure_Pa=outlet_pressure,
            properties=self.properties,
            context="reliquefier link",
        )
        return result
