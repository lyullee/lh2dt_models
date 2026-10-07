"""Steady control-volume vaporizer energy balance."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .properties import HydrogenProperties, ThermoState


@dataclass(frozen=True)
class VaporizerResult:
    outlet: ThermoState
    heat_into_hydrogen_W: float
    requested_heat_W: float
    capacity_limited: bool


@dataclass(frozen=True)
class PressureDropRelation:
    """Explicit pressure-loss relation for a vaporizer flow boundary.

    The relation is supplied from equipment geometry or a manufacturer
    performance curve.  It is deliberately independent of operating time
    histories and therefore cannot act as a fitting or inverse-estimation
    parameter.  The non-negative polynomial is evaluated as

    ``delta_p = constant + linear*m_dot + quadratic*m_dot**2``.
    """

    constant_Pa: float = 0.0
    linear_Pa_s_per_kg: float = 0.0
    quadratic_Pa_s2_per_kg2: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.constant_Pa,
            self.linear_Pa_s_per_kg,
            self.quadratic_Pa_s2_per_kg2,
        )
        if any(isinstance(value, bool) for value in values):
            raise ValueError("pressure-drop coefficients must be numeric")
        try:
            numeric = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError("pressure-drop coefficients must be numeric") from exc
        if any(not math.isfinite(value) or value < 0.0 for value in numeric):
            raise ValueError("pressure-drop coefficients must be finite and non-negative")
        object.__setattr__(self, "constant_Pa", numeric[0])
        object.__setattr__(self, "linear_Pa_s_per_kg", numeric[1])
        object.__setattr__(self, "quadratic_Pa_s2_per_kg2", numeric[2])

    def drop_Pa(self, mass_flow_kg_s: float) -> float:
        """Return the pressure loss for a positive mass-flow boundary."""

        if isinstance(mass_flow_kg_s, bool):
            raise ValueError("mass_flow_kg_s must be numeric")
        try:
            mass_flow = float(mass_flow_kg_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("mass_flow_kg_s must be numeric") from exc
        if not math.isfinite(mass_flow) or mass_flow < 0.0:
            raise ValueError("mass_flow_kg_s must be finite and non-negative")
        return (
            self.constant_Pa
            + self.linear_Pa_s_per_kg * mass_flow
            + self.quadratic_Pa_s2_per_kg2 * mass_flow * mass_flow
        )

    def outlet_pressure_Pa(
        self,
        inlet_pressure_Pa: float,
        mass_flow_kg_s: float,
    ) -> float:
        """Return inlet pressure minus the explicit flow-dependent loss."""

        if isinstance(inlet_pressure_Pa, bool):
            raise ValueError("inlet_pressure_Pa must be numeric")
        try:
            inlet_pressure = float(inlet_pressure_Pa)
        except (TypeError, ValueError) as exc:
            raise ValueError("inlet_pressure_Pa must be numeric") from exc
        if not math.isfinite(inlet_pressure) or inlet_pressure <= 0.0:
            raise ValueError("inlet_pressure_Pa must be finite and positive")
        outlet_pressure = inlet_pressure - self.drop_Pa(mass_flow_kg_s)
        if outlet_pressure <= 0.0:
            raise ValueError(
                "pressure-drop relation exceeds the supplied inlet pressure"
            )
        return outlet_pressure


@dataclass(frozen=True)
class FlowUAAdjustment:
    """Optional explicit mass-flow scaling for vaporizer conductance.

    Internal convection and flow distribution can make an equipment UA vary
    with mass flow.  This relation exposes that effect as a declared design
    law rather than inferring it from a time series:

    ``UA(flow) = UA_reference * clip(flow/reference_flow, 0, inf)**exponent``.

    The scale bounds are explicit to keep the relation finite over scenario
    sweeps.  The default production path leaves this relation unset, so the
    historical constant-UA behaviour is preserved.
    """

    reference_mass_flow_kg_s: float
    exponent: float = 0.0
    minimum_scale: float = 0.0
    maximum_scale: float = 10.0

    def __post_init__(self) -> None:
        values = (
            self.reference_mass_flow_kg_s,
            self.exponent,
            self.minimum_scale,
            self.maximum_scale,
        )
        if any(isinstance(value, bool) for value in values):
            raise ValueError("flow-UA relation values must be numeric")
        try:
            numeric = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError("flow-UA relation values must be numeric") from exc
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("flow-UA relation values must be finite")
        reference, exponent, minimum, maximum = numeric
        if reference <= 0.0:
            raise ValueError("reference_mass_flow_kg_s must be positive")
        if exponent < 0.0:
            raise ValueError("exponent must be non-negative")
        if minimum < 0.0 or maximum <= 0.0 or maximum < minimum:
            raise ValueError("flow-UA scale bounds must be non-negative and ordered")
        object.__setattr__(self, "reference_mass_flow_kg_s", reference)
        object.__setattr__(self, "exponent", exponent)
        object.__setattr__(self, "minimum_scale", minimum)
        object.__setattr__(self, "maximum_scale", maximum)

    def scale(self, mass_flow_kg_s: float) -> float:
        """Return the bounded conductance scale at a non-negative flow."""

        if isinstance(mass_flow_kg_s, bool):
            raise ValueError("mass_flow_kg_s must be numeric")
        try:
            mass_flow = float(mass_flow_kg_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("mass_flow_kg_s must be numeric") from exc
        if not math.isfinite(mass_flow) or mass_flow < 0.0:
            raise ValueError("mass_flow_kg_s must be finite and non-negative")
        raw = (mass_flow / self.reference_mass_flow_kg_s) ** self.exponent
        return min(self.maximum_scale, max(self.minimum_scale, raw))


@dataclass(frozen=True)
class VaporizerOperatingBoundary:
    """Explicit operating state for a vaporizer flow boundary.

    A drawing capacity is an equipment envelope, not an operating flow.  The
    caller must therefore declare either ``closed`` with zero flow or
    ``flowing`` with a positive normal volumetric flow.  The conversion uses
    the public normal hydrogen reference state and does not infer a flow from
    a plant observation or a model residual.
    """

    mode: str
    normal_flow_Nm3_h: float
    maximum_normal_flow_Nm3_h: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"closed", "flowing"}:
            raise ValueError("mode must be 'closed' or 'flowing'")
        if isinstance(self.normal_flow_Nm3_h, bool):
            raise ValueError("normal_flow_Nm3_h must be numeric")
        try:
            flow = float(self.normal_flow_Nm3_h)
        except (TypeError, ValueError) as exc:
            raise ValueError("normal_flow_Nm3_h must be numeric") from exc
        if not math.isfinite(flow) or flow < 0.0:
            raise ValueError("normal_flow_Nm3_h must be finite and non-negative")
        maximum = self.maximum_normal_flow_Nm3_h
        if maximum is not None:
            if isinstance(maximum, bool):
                raise ValueError("maximum_normal_flow_Nm3_h must be numeric")
            try:
                maximum = float(maximum)
            except (TypeError, ValueError) as exc:
                raise ValueError("maximum_normal_flow_Nm3_h must be numeric") from exc
            if not math.isfinite(maximum) or maximum <= 0.0:
                raise ValueError("maximum_normal_flow_Nm3_h must be finite and positive")
            if flow > maximum:
                raise ValueError("normal_flow_Nm3_h exceeds the declared boundary")
        if self.mode == "closed" and flow != 0.0:
            raise ValueError("closed operating boundary requires zero flow")
        if self.mode == "flowing" and flow <= 0.0:
            raise ValueError("flowing operating boundary requires positive flow")
        object.__setattr__(self, "normal_flow_Nm3_h", flow)
        object.__setattr__(self, "maximum_normal_flow_Nm3_h", maximum)

    @property
    def is_closed(self) -> bool:
        return self.mode == "closed"

    def mass_flow_kg_s(self, properties: HydrogenProperties) -> float:
        """Convert the declared normal flow using the public reference state."""

        if not isinstance(properties, HydrogenProperties):
            raise TypeError("properties must be HydrogenProperties")
        normal_state = properties.from_pT(101_325.0, 273.15)
        return self.normal_flow_Nm3_h / 3600.0 * normal_state.density_kg_m3


class Vaporizer:
    def __init__(
        self,
        UA_W_K: float,
        maximum_heat_W: float,
        properties: HydrogenProperties | None = None,
        integration_segments: int = 192,
        minimum_approach_K: float = 0.0,
        pressure_drop_relation: PressureDropRelation | None = None,
        flow_ua_adjustment: FlowUAAdjustment | None = None,
    ) -> None:
        if isinstance(UA_W_K, bool) or isinstance(maximum_heat_W, bool):
            raise ValueError("UA_W_K and maximum_heat_W must be numeric")
        try:
            ua = float(UA_W_K)
            maximum_heat = float(maximum_heat_W)
        except (TypeError, ValueError) as exc:
            raise ValueError("UA_W_K and maximum_heat_W must be numeric") from exc
        if not math.isfinite(ua) or not math.isfinite(maximum_heat):
            raise ValueError("UA_W_K and maximum_heat_W must be finite")
        if ua <= 0.0 or maximum_heat <= 0.0:
            raise ValueError("UA_W_K and maximum_heat_W must be positive")
        if not isinstance(integration_segments, int) or integration_segments < 32:
            raise ValueError("integration_segments must be an integer of at least 32")
        if isinstance(minimum_approach_K, bool):
            raise ValueError("minimum_approach_K must be numeric")
        try:
            minimum_approach = float(minimum_approach_K)
        except (TypeError, ValueError) as exc:
            raise ValueError("minimum_approach_K must be numeric") from exc
        if not math.isfinite(minimum_approach) or minimum_approach < 0.0:
            raise ValueError("minimum_approach_K must be finite and non-negative")
        self.UA = ua
        self.maximum_heat = maximum_heat
        self.properties = properties or HydrogenProperties()
        self.integration_segments = integration_segments
        # A positive approach is a physical outlet-temperature boundary for
        # ambient-heated equipment: the hydrogen cannot reach ambient exactly.
        # It is a design/envelope input, never an inferred fit parameter.
        self.minimum_approach_K = minimum_approach
        if pressure_drop_relation is not None and not isinstance(
            pressure_drop_relation, PressureDropRelation
        ):
            raise TypeError("pressure_drop_relation must be PressureDropRelation or None")
        self.pressure_drop_relation = pressure_drop_relation or PressureDropRelation()
        if flow_ua_adjustment is not None and not isinstance(
            flow_ua_adjustment, FlowUAAdjustment
        ):
            raise TypeError("flow_ua_adjustment must be FlowUAAdjustment or None")
        self.flow_ua_adjustment = flow_ua_adjustment
        self.parameterization = "UA"
        self.heat_transfer_area_m2: float | None = None
        self.resistance_terms_m2K_W: tuple[float, ...] = ()

    @classmethod
    def from_resistance_network(
        cls,
        *,
        area_m2: float,
        resistance_terms_m2K_W: Sequence[float],
        maximum_heat_W: float,
        properties: HydrogenProperties | None = None,
        integration_segments: int = 192,
        minimum_approach_K: float = 0.0,
        pressure_drop_relation: PressureDropRelation | None = None,
        flow_ua_adjustment: FlowUAAdjustment | None = None,
    ) -> "Vaporizer":
        """Construct a vaporizer from explicit area-normalized resistances.

        ``resistance_terms_m2K_W`` contains positive terms such as hot-side
        convection, wall conduction, fouling, and cold-side convection.  The
        effective conductance is calculated as
        ``UA = area_m2 / sum(resistance_terms_m2K_W)``.  No term is inferred
        from a time history; callers must provide the complete
        resistance basis explicitly.
        """

        if isinstance(area_m2, bool):
            raise ValueError("area_m2 must be numeric")
        try:
            area = float(area_m2)
        except (TypeError, ValueError) as exc:
            raise ValueError("area_m2 must be numeric") from exc
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("area_m2 must be finite and positive")
        if isinstance(resistance_terms_m2K_W, (str, bytes)):
            raise ValueError("resistance_terms_m2K_W must be a non-empty sequence")
        try:
            raw_terms = tuple(resistance_terms_m2K_W)
        except TypeError as exc:
            raise ValueError("resistance_terms_m2K_W must contain numeric values") from exc
        terms_list: list[float] = []
        for index, term in enumerate(raw_terms):
            if isinstance(term, bool):
                raise ValueError(f"resistance_terms_m2K_W[{index}] must be numeric")
            try:
                terms_list.append(float(term))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "resistance_terms_m2K_W must contain numeric values"
                ) from exc
        terms = tuple(terms_list)
        if not terms:
            raise ValueError("resistance_terms_m2K_W must be a non-empty sequence")
        if any(not math.isfinite(term) or term <= 0.0 for term in terms):
            raise ValueError("resistance terms must be finite and positive")
        total_resistance = sum(terms)
        if not math.isfinite(total_resistance) or total_resistance <= 0.0:
            raise ValueError("total thermal resistance must be finite and positive")
        vaporizer = cls(
            area / total_resistance,
            maximum_heat_W,
            properties,
            integration_segments,
            minimum_approach_K,
            pressure_drop_relation,
            flow_ua_adjustment,
        )
        vaporizer.parameterization = "resistance_network"
        vaporizer.heat_transfer_area_m2 = area
        vaporizer.resistance_terms_m2K_W = terms
        return vaporizer

    def outlet_pressure_from_inlet(
        self,
        inlet_pressure_Pa: float,
        mass_flow_kg_s: float,
    ) -> float:
        """Apply the explicit equipment pressure-loss relation."""

        return self.pressure_drop_relation.outlet_pressure_Pa(
            inlet_pressure_Pa, mass_flow_kg_s
        )

    def effective_UA_W_K(self, mass_flow_kg_s: float) -> float:
        """Return the declared flow-dependent conductance, if enabled."""

        scale = 1.0 if self.flow_ua_adjustment is None else self.flow_ua_adjustment.scale(mass_flow_kg_s)
        effective = self.UA * scale
        if not math.isfinite(effective) or effective <= 0.0:
            raise ValueError("flow-UA relation must produce a positive finite conductance")
        return effective

    def evaluate_from_inlet_pressure(
        self,
        inlet: ThermoState,
        inlet_pressure_Pa: float,
        mass_flow_kg_s: float,
        ambient_temperature_K: float,
    ) -> VaporizerResult:
        """Evaluate using inlet pressure and an explicit loss relation.

        The existing :meth:`evaluate` API remains available for networks that
        already solve the outlet pressure.  This helper closes the common
        standalone-equipment case without inferring a pressure loss from
        observations.
        """

        outlet_pressure = self.outlet_pressure_from_inlet(
            inlet_pressure_Pa, mass_flow_kg_s
        )
        return self.evaluate(
            inlet, outlet_pressure, mass_flow_kg_s, ambient_temperature_K
        )

    def evaluate_operating_boundary(
        self,
        inlet: ThermoState,
        outlet_pressure_Pa: float,
        operating_boundary: VaporizerOperatingBoundary,
        ambient_temperature_K: float,
    ) -> VaporizerResult:
        """Evaluate using an explicit closed or flowing operating state."""

        if not isinstance(operating_boundary, VaporizerOperatingBoundary):
            raise TypeError("operating_boundary must be VaporizerOperatingBoundary")
        return self.evaluate(
            inlet,
            outlet_pressure_Pa,
            operating_boundary.mass_flow_kg_s(self.properties),
            ambient_temperature_K,
        )

    def evaluate(
        self,
        inlet: ThermoState,
        outlet_pressure_Pa: float,
        mass_flow_kg_s: float,
        ambient_temperature_K: float,
    ) -> VaporizerResult:
        if any(
            isinstance(value, bool)
            for value in (outlet_pressure_Pa, mass_flow_kg_s, ambient_temperature_K)
        ):
            raise ValueError("vaporizer operating values must be numeric")
        try:
            mass_flow = float(mass_flow_kg_s)
            outlet_pressure = float(outlet_pressure_Pa)
            ambient_temperature = float(ambient_temperature_K)
        except (TypeError, ValueError) as exc:
            raise ValueError("vaporizer operating values must be numeric") from exc
        if not math.isfinite(mass_flow) or mass_flow < 0.0:
            raise ValueError("mass_flow_kg_s must be finite and non-negative")
        if not math.isfinite(outlet_pressure) or outlet_pressure <= 0.0:
            raise ValueError("outlet_pressure_Pa must be finite and positive")
        if not math.isfinite(ambient_temperature) or ambient_temperature <= 0.0:
            raise ValueError("ambient_temperature_K must be finite and positive")
        pressure_adjusted_inlet = self.properties.from_ph(
            outlet_pressure, inlet.specific_enthalpy_J_kg
        )
        if mass_flow == 0.0:
            # A closed upstream/downstream valve is a valid operating state.
            # There is no transported enthalpy and therefore no steady
            # vaporizer duty.  Returning the pressure-adjusted stagnant state
            # lets a network carry the state through an idle interval instead
            # of rejecting the interval or inventing a residual flow.
            return VaporizerResult(
                pressure_adjusted_inlet,
                0.0,
                0.0,
                False,
            )
        effective_ambient_temperature = ambient_temperature - self.minimum_approach_K
        if effective_ambient_temperature <= 0.0:
            raise ValueError(
                "ambient_temperature_K must exceed minimum_approach_K"
            )
        if effective_ambient_temperature <= pressure_adjusted_inlet.temperature_K:
            return VaporizerResult(
                pressure_adjusted_inlet,
                0.0, 0.0, False,
            )
        ambient_limit = self.properties.from_pT(
            outlet_pressure, effective_ambient_temperature
        )
        requested = max(0.0, mass_flow * (
            ambient_limit.specific_enthalpy_J_kg - inlet.specific_enthalpy_J_kg
        ))
        if requested == 0.0:
            return VaporizerResult(pressure_adjusted_inlet, 0.0, 0.0, False)

        inlet_h = inlet.specific_enthalpy_J_kg
        capacity_h = inlet_h + self.maximum_heat / mass_flow
        upper_h = min(capacity_h, ambient_limit.specific_enthalpy_J_kg)
        if upper_h >= ambient_limit.specific_enthalpy_J_kg:
            # A finite conductance approaches, but never exactly reaches, the
            # infinite-reservoir ambient temperature.
            upper_h = ambient_limit.specific_enthalpy_J_kg - max(
                1.0e-8 * abs(ambient_limit.specific_enthalpy_J_kg - inlet_h),
                1.0e-6,
            )

        saturation_points: list[float] = []
        try:
            saturated_liquid = self.properties.saturated_liquid(outlet_pressure)
            saturated_vapor = self.properties.saturated_vapor(outlet_pressure)
            saturation_points = [
                value for value in (
                    saturated_liquid.specific_enthalpy_J_kg,
                    saturated_vapor.specific_enthalpy_J_kg,
                ) if inlet_h < value < upper_h
            ]
        except (ValueError, RuntimeError):
            saturation_points = []

        enthalpy_grid = np.unique(np.r_[
            np.linspace(inlet_h, upper_h, self.integration_segments + 1),
            saturation_points,
        ])
        temperature_grid = np.asarray([
            self.properties.from_ph(outlet_pressure, float(enthalpy)).temperature_K
            for enthalpy in enthalpy_grid
        ])
        integrand = 1.0 / np.maximum(
            effective_ambient_temperature - temperature_grid, 1.0e-12
        )
        increments = (
            0.5 * (integrand[1:] + integrand[:-1]) * np.diff(enthalpy_grid)
            * mass_flow
        )
        cumulative_UA = np.r_[0.0, np.cumsum(increments)]
        effective_UA = self.effective_UA_W_K(mass_flow)
        if cumulative_UA[-1] <= effective_UA:
            outlet_h = upper_h
        else:
            upper_index = int(np.searchsorted(cumulative_UA, effective_UA, side="right"))
            lower_index = upper_index - 1
            fraction = (
                (effective_UA - cumulative_UA[lower_index])
                / (cumulative_UA[upper_index] - cumulative_UA[lower_index])
            )
            outlet_h = float(
                enthalpy_grid[lower_index]
                + fraction * (enthalpy_grid[upper_index] - enthalpy_grid[lower_index])
            )
        heat = mass_flow * (outlet_h - inlet_h)
        return VaporizerResult(
            self.properties.from_ph(outlet_pressure, outlet_h),
            heat,
            requested,
            heat < requested * (1.0 - 1e-12),
        )
