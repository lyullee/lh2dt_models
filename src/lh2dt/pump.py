"""Adiabatic liquid-hydrogen pump energy balance."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Any, Sequence

from .properties import HydrogenProperties, ThermoState


_LIQUID_QUALITY_EPSILON = 1.0e-7


@dataclass(frozen=True)
class PumpResult:
    outlet: ThermoState
    pressure_rise_Pa: float
    shaft_power_W: float
    hydraulic_power_W: float
    efficiency: float


@dataclass(frozen=True)
class VirtualPumpResult:
    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float
    pressure_rise_Pa: float
    shaft_power_W: float
    hydraulic_power_W: float
    npsh_available_m: float
    npsh_required_m: float | None
    cavitation_margin_m: float | None
    upstream_side: str | None
    momentum_residual_Pa: float | None = None

    @property
    def operating_point_physical_valid(self) -> bool:
        """Whether the reported point satisfies the explicit NPSH contract.

        A virtual pump can still return a mathematically closed point when the
        available NPSH is below the declared requirement.  Keep that point
        visible for screening, but expose one reusable physical-operability
        predicate so callers do not accidentally treat it as an admissible
        pump operating point.
        """

        return self.cavitation_margin_m is None or self.cavitation_margin_m >= 0.0


class Pump:
    """Prescribed pressure-rise pump.

    A manufacturer H-Q-N map can replace the optional configured operating
    point later.  The conservation equation stays unchanged.  When no
    operating point is configured, callers must continue to provide the
    pressure rise and efficiency explicitly to :meth:`evaluate`.
    """

    def __init__(
        self,
        properties: HydrogenProperties | None = None,
        *,
        pressure_rise_Pa: float | None = None,
        isentropic_efficiency: float | None = None,
    ) -> None:
        if (pressure_rise_Pa is None) != (isentropic_efficiency is None):
            raise ValueError(
                "pressure_rise_Pa and isentropic_efficiency must be supplied together"
            )
        if pressure_rise_Pa is not None:
            if isinstance(pressure_rise_Pa, bool) or isinstance(isentropic_efficiency, bool):
                raise ValueError("pump operating point values must be numeric")
            try:
                pressure_rise_value = float(pressure_rise_Pa)
                efficiency_value = float(isentropic_efficiency)
            except (TypeError, ValueError) as exc:
                raise ValueError("pump operating point values must be numeric") from exc
            if not math.isfinite(pressure_rise_value) or pressure_rise_value <= 0.0:
                raise ValueError("pressure_rise_Pa must be finite and positive")
            if not math.isfinite(efficiency_value) or not 0.0 < efficiency_value <= 1.0:
                raise ValueError("isentropic_efficiency must be finite and in (0, 1]")
        self.properties = properties or HydrogenProperties()
        self.default_pressure_rise_Pa = (
            None if pressure_rise_Pa is None else float(pressure_rise_Pa)
        )
        self.default_isentropic_efficiency = (
            None if isentropic_efficiency is None else float(isentropic_efficiency)
        )

    def evaluate(
        self,
        inlet: ThermoState,
        mass_flow_kg_s: float,
        pressure_rise_Pa: float | None = None,
        isentropic_efficiency: float | None = None,
    ) -> PumpResult:
        if pressure_rise_Pa is None:
            pressure_rise_Pa = self.default_pressure_rise_Pa
        if isentropic_efficiency is None:
            isentropic_efficiency = self.default_isentropic_efficiency
        if pressure_rise_Pa is None or isentropic_efficiency is None:
            raise ValueError(
                "pressure_rise_Pa and isentropic_efficiency must be supplied "
                "or configured on Pump"
            )
        if any(
            isinstance(value, bool)
            for value in (mass_flow_kg_s, pressure_rise_Pa, isentropic_efficiency)
        ):
            raise ValueError("pump operating values must be numeric")
        try:
            mass_flow_value = float(mass_flow_kg_s)
            pressure_rise_value = float(pressure_rise_Pa)
            efficiency_value = float(isentropic_efficiency)
        except (TypeError, ValueError) as exc:
            raise ValueError("pump operating values must be numeric") from exc
        if not math.isfinite(mass_flow_value) or not math.isfinite(pressure_rise_value):
            raise ValueError("mass flow and pressure rise must be finite")
        if mass_flow_value <= 0.0 or pressure_rise_value <= 0.0:
            raise ValueError("mass flow and pressure rise must be positive")
        if not math.isfinite(efficiency_value) or not 0.0 < efficiency_value <= 1.0:
            raise ValueError("isentropic_efficiency must be in (0, 1]")
        outlet_pressure = inlet.pressure_Pa + pressure_rise_value
        ideal_outlet = self.properties.from_ps(outlet_pressure, inlet.specific_entropy_J_kgK)
        ideal_dh = ideal_outlet.specific_enthalpy_J_kg - inlet.specific_enthalpy_J_kg
        actual_dh = ideal_dh / efficiency_value
        outlet = self.properties.from_ph(
            outlet_pressure,
            inlet.specific_enthalpy_J_kg + actual_dh,
        )
        hydraulic_power = mass_flow_value * pressure_rise_value / inlet.density_kg_m3
        return PumpResult(
            outlet=outlet,
            pressure_rise_Pa=pressure_rise_value,
            shaft_power_W=mass_flow_value * actual_dh,
            hydraulic_power_W=hydraulic_power,
            efficiency=efficiency_value,
        )


class VirtualPump:
    """Network-compatible LH2 pump with an explicit quadratic H-Q curve.

    ``shutoff_pressure_rise_Pa`` and ``runout_mass_flow_kg_s`` define
    ``Δp = Δp_shutoff * (1 - (mdot/mdot_runout)^2)``.  They are design or
    manufacturer parameters and are never inferred from facility time series.
    If explicit ``reference_speed_rpm`` and ``speed_rpm`` values are supplied,
    affinity laws scale pressure rise by ``N²`` and runout flow by ``N``;
    ``set_speed_rpm`` can then be used by a scenario command schedule. Without
    those optional speed inputs the original fixed curve remains active. The
    initial implementation is forward-only and accepts liquid inlet states.
    """

    def __init__(
        self,
        shutoff_pressure_rise_Pa: float,
        runout_mass_flow_kg_s: float,
        isentropic_efficiency: float,
        properties: HydrogenProperties | None = None,
        npsh_required_m: float | None = None,
        reference_speed_rpm: float | None = None,
        speed_rpm: float | None = None,
    ) -> None:
        def finite_number(name: str, value: object, *, positive: bool = False) -> float:
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            if positive and number <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            return number

        shutoff_pressure_rise = finite_number(
            "shutoff_pressure_rise_Pa", shutoff_pressure_rise_Pa, positive=True
        )
        runout_mass_flow = finite_number(
            "runout_mass_flow_kg_s", runout_mass_flow_kg_s, positive=True
        )
        efficiency = finite_number("isentropic_efficiency", isentropic_efficiency)
        if not 0.0 < efficiency <= 1.0:
            raise ValueError("isentropic_efficiency must be finite and in (0, 1]")
        npsh_required = None
        if npsh_required_m is not None:
            npsh_required = finite_number("npsh_required_m", npsh_required_m)
            if npsh_required < 0.0:
                raise ValueError("npsh_required_m must be finite and non-negative")
        if (reference_speed_rpm is None) != (speed_rpm is None):
            raise ValueError(
                "reference_speed_rpm and speed_rpm must be supplied together"
            )
        reference_speed = None
        operating_speed = None
        if reference_speed_rpm is not None:
            reference_speed = finite_number(
                "reference_speed_rpm", reference_speed_rpm, positive=True
            )
            operating_speed = finite_number("speed_rpm", speed_rpm, positive=True)
        self.shutoff_pressure_rise_Pa = shutoff_pressure_rise
        self.runout_mass_flow_kg_s = runout_mass_flow
        self.efficiency = efficiency
        self.npsh_required_m = npsh_required
        self.reference_speed_rpm = reference_speed
        self._speed_rpm = operating_speed
        self.speed_ratio = (
            1.0 if reference_speed is None else operating_speed / reference_speed
        )
        self.properties = properties or HydrogenProperties()
        self._pump = Pump(self.properties)

    @property
    def speed_rpm(self) -> float | None:
        """Current commanded speed when an explicit speed basis is configured."""

        return self._speed_rpm

    @speed_rpm.setter
    def speed_rpm(self, value: float) -> None:
        if self.reference_speed_rpm is None:
            raise ValueError(
                "reference_speed_rpm is required before commanding speed_rpm"
            )
        if isinstance(value, bool):
            raise ValueError("speed_rpm must be numeric")
        try:
            speed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("speed_rpm must be numeric") from exc
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed_rpm must be finite and positive")
        self._speed_rpm = speed
        self.speed_ratio = speed / self.reference_speed_rpm

    def set_speed_rpm(self, speed_rpm: float) -> None:
        """Set the explicit operating speed for a command schedule."""

        self.speed_rpm = speed_rpm

    def evaluate(self, left: ThermoState, right: ThermoState) -> VirtualPumpResult:
        # ``quality=None`` is used by CoolProp for both single-phase liquid
        # and single-phase gas states.  Checking quality alone therefore
        # allows a superheated/supercritical gas into a liquid pump whenever
        # it is not on the saturation dome.  Keep saturated liquid
        # (quality=0) and single-phase liquid states admissible, but reject
        # every non-liquid phase explicitly.  The phase label comes from the
        # shared EOS facade and is not inferred from plant data.
        phase = str(left.phase).strip().lower()
        liquid_phase = "liquid" in phase and "gas" not in phase
        if (
            (left.quality is not None and left.quality > _LIQUID_QUALITY_EPSILON)
            or (left.quality is None and not liquid_phase)
        ):
            raise ValueError("VirtualPump requires a liquid inlet state")
        saturation_pressure = self.properties.saturation_pressure_Pa(
            left.temperature_K
        )
        npsh_available = (
            left.pressure_Pa - saturation_pressure
        ) / (left.density_kg_m3 * 9.80665)
        cavitation_margin = (
            None if self.npsh_required_m is None
            else npsh_available - self.npsh_required_m
        )
        required_rise = max(0.0, right.pressure_Pa - left.pressure_Pa)
        speed_ratio = self.speed_ratio
        shutoff_pressure_rise = self.shutoff_pressure_rise_Pa * speed_ratio**2
        runout_mass_flow = self.runout_mass_flow_kg_s * speed_ratio

        def momentum_residual(mass_flow_kg_s: float) -> float:
            """Return the signed H-Q pressure closure in Pa."""

            curve_rise = shutoff_pressure_rise * (
                1.0 - (float(mass_flow_kg_s) / runout_mass_flow) ** 2
            )
            return curve_rise - (right.pressure_Pa - left.pressure_Pa)

        if required_rise >= shutoff_pressure_rise:
            return VirtualPumpResult(
                0.0, left.specific_enthalpy_J_kg, required_rise,
                0.0, 0.0, npsh_available, self.npsh_required_m,
                cavitation_margin, None, momentum_residual(0.0),
            )
        mass_flow = runout_mass_flow * math.sqrt(
            1.0 - required_rise / shutoff_pressure_rise
        )
        if required_rise <= 0.0:
            return VirtualPumpResult(
                mass_flow, left.specific_enthalpy_J_kg, 0.0,
                0.0, 0.0, npsh_available, self.npsh_required_m,
                cavitation_margin, "left", momentum_residual(mass_flow),
            )
        result = self._pump.evaluate(
            left, mass_flow, required_rise, self.efficiency
        )
        return VirtualPumpResult(
            mass_flow_kg_s=mass_flow,
            outlet_specific_enthalpy_J_kg=result.outlet.specific_enthalpy_J_kg,
            pressure_rise_Pa=required_rise,
            shaft_power_W=result.shaft_power_W,
            hydraulic_power_W=result.hydraulic_power_W,
            npsh_available_m=npsh_available,
            npsh_required_m=self.npsh_required_m,
            cavitation_margin_m=cavitation_margin,
            upstream_side="left",
            momentum_residual_Pa=momentum_residual(mass_flow),
        )


@dataclass(frozen=True)
class MappedPumpResult:
    """Result of an explicit H-Q-N performance-map evaluation.

    The map itself is an explicit design/manufacturer input.  It is never
    estimated from facility time series.  ``map_limited`` indicates that the
    requested pressure difference was outside the supplied map envelope and
    the nearest tabulated flow point was used; the hydraulic residual remains
    visible so callers cannot mistake a clipped map for a closed operating
    point.
    """

    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float
    pressure_rise_Pa: float
    shaft_power_W: float
    hydraulic_power_W: float
    npsh_available_m: float
    npsh_required_m: float | None
    cavitation_margin_m: float | None
    upstream_side: str | None
    map_head_pressure_rise_Pa: float
    map_limited: bool
    momentum_residual_Pa: float
    inlet_quality: float | None = None

    @property
    def operating_point_physical_valid(self) -> bool:
        """Whether the explicit map envelope and NPSH requirement are met."""

        return (
            not self.map_limited
            and (
                self.inlet_quality is None
                or self.inlet_quality <= _LIQUID_QUALITY_EPSILON
            )
            and (
                self.cavitation_margin_m is None
                or self.cavitation_margin_m >= 0.0
            )
        )


def _normalise_curve(
    name: str,
    points: Sequence[Sequence[float]],
    *,
    y_min: float | None = None,
    y_max: float | None = None,
    non_increasing_y: bool = False,
) -> tuple[tuple[float, float], ...]:
    if isinstance(points, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of (x, y) points")
    try:
        raw_points = tuple(points)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of (x, y) points") from exc
    if len(raw_points) < 2:
        raise ValueError(f"{name} requires at least two points")
    result: list[tuple[float, float]] = []
    previous_x: float | None = None
    for index, raw in enumerate(raw_points):
        if isinstance(raw, (str, bytes)):
            raise ValueError(f"{name}[{index}] must contain two numeric values")
        try:
            pair = tuple(raw)
        except TypeError as exc:
            raise ValueError(f"{name}[{index}] must contain two numeric values") from exc
        if len(pair) != 2:
            raise ValueError(f"{name}[{index}] must contain two numeric values")
        if any(isinstance(value, bool) for value in pair):
            raise ValueError(f"{name}[{index}] values must be numeric")
        try:
            x, y = (float(pair[0]), float(pair[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] values must be numeric") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"{name}[{index}] values must be finite")
        if x < 0.0:
            raise ValueError(f"{name}[{index}] x must be non-negative")
        if previous_x is not None and x <= previous_x:
            raise ValueError(f"{name} x values must be strictly increasing")
        if y_min is not None and y < y_min:
            raise ValueError(f"{name}[{index}] y is below its allowed range")
        if y_max is not None and y > y_max:
            raise ValueError(f"{name}[{index}] y is above its allowed range")
        if non_increasing_y and result and y > result[-1][1] + 1.0e-12:
            raise ValueError(f"{name} y values must be non-increasing")
        result.append((x, y))
        previous_x = x
    return tuple(result)


def _linear_curve_value(points: tuple[tuple[float, float], ...], x: float) -> float:
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            fraction = (x - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    return points[-1][1]


class MappedPump:
    """Network-compatible pump driven by an explicit H-Q-N map.

    ``head_curve_points`` are ``(mass_flow_kg_s, pressure_rise_Pa)`` points at
    ``reference_speed_rpm``.  The affinity laws scale flow by ``N/N_ref`` and
    pressure rise by ``(N/N_ref)^2``.  Efficiency and NPSHr may be supplied as
    explicit maps or as constant design values.  Map endpoints are clamped;
    no extrapolation is performed, and a clipped evaluation reports
    ``map_limited=True`` and a non-zero momentum residual when applicable.
    ``inlet_quality_tolerance`` is an explicit near-saturated-liquid numerical
    closure; it is not a performance fit and the resulting quality and NPSH
    margin remain exposed in :class:`MappedPumpResult`.
    """

    def __init__(
        self,
        head_curve_points: Sequence[Sequence[float]],
        *,
        reference_speed_rpm: float,
        speed_rpm: float,
        isentropic_efficiency: float | None = None,
        efficiency_curve_points: Sequence[Sequence[float]] | None = None,
        npsh_required_m: float | None = None,
        npsh_curve_points: Sequence[Sequence[float]] | None = None,
        inlet_quality_tolerance: float = 1.0e-7,
        properties: HydrogenProperties | None = None,
    ) -> None:
        def positive(name: str, value: object) -> float:
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            return number

        self.reference_speed_rpm = positive("reference_speed_rpm", reference_speed_rpm)
        self._speed_rpm = positive("speed_rpm", speed_rpm)
        self.speed_ratio = self._speed_rpm / self.reference_speed_rpm
        self._head_curve = _normalise_curve(
            "head_curve_points", head_curve_points,
            y_min=0.0, non_increasing_y=True,
        )
        if self._head_curve[0][0] != 0.0:
            raise ValueError("head_curve_points must start at zero mass flow")
        if self._head_curve[0][1] <= 0.0:
            raise ValueError("zero-flow head must be positive")
        self._efficiency_curve = None
        if efficiency_curve_points is not None:
            self._efficiency_curve = _normalise_curve(
                "efficiency_curve_points", efficiency_curve_points,
                y_min=1.0e-12, y_max=1.0,
            )
        if isentropic_efficiency is None and self._efficiency_curve is None:
            raise ValueError("provide isentropic_efficiency or efficiency_curve_points")
        if isentropic_efficiency is not None:
            if isinstance(isentropic_efficiency, bool):
                raise ValueError("isentropic_efficiency must be numeric")
            try:
                efficiency = float(isentropic_efficiency)
            except (TypeError, ValueError) as exc:
                raise ValueError("isentropic_efficiency must be numeric") from exc
            if not math.isfinite(efficiency) or not 0.0 < efficiency <= 1.0:
                raise ValueError("isentropic_efficiency must be finite and in (0, 1]")
            self.isentropic_efficiency = efficiency
        else:
            self.isentropic_efficiency = None
        self._npsh_curve = None
        if npsh_curve_points is not None:
            self._npsh_curve = _normalise_curve(
                "npsh_curve_points", npsh_curve_points, y_min=0.0,
            )
        if npsh_required_m is not None:
            if isinstance(npsh_required_m, bool):
                raise ValueError("npsh_required_m must be numeric")
            try:
                npsh = float(npsh_required_m)
            except (TypeError, ValueError) as exc:
                raise ValueError("npsh_required_m must be numeric") from exc
            if not math.isfinite(npsh) or npsh < 0.0:
                raise ValueError("npsh_required_m must be finite and non-negative")
            self.npsh_required_m = npsh
        else:
            self.npsh_required_m = None
        if isinstance(inlet_quality_tolerance, bool):
            raise ValueError("inlet_quality_tolerance must be numeric")
        try:
            quality_tolerance = float(inlet_quality_tolerance)
        except (TypeError, ValueError) as exc:
            raise ValueError("inlet_quality_tolerance must be numeric") from exc
        if not math.isfinite(quality_tolerance) or quality_tolerance < 0.0:
            raise ValueError("inlet_quality_tolerance must be finite and non-negative")
        self.inlet_quality_tolerance = quality_tolerance
        self.properties = properties or HydrogenProperties()
        self._pump = Pump(self.properties)

    @property
    def speed_rpm(self) -> float:
        """Current commanded speed used by the affinity-law H-Q map."""

        return self._speed_rpm

    @speed_rpm.setter
    def speed_rpm(self, value: float) -> None:
        if isinstance(value, bool):
            raise ValueError("speed_rpm must be numeric")
        try:
            speed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("speed_rpm must be numeric") from exc
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("speed_rpm must be finite and positive")
        self._speed_rpm = speed
        if hasattr(self, "reference_speed_rpm"):
            self.speed_ratio = speed / self.reference_speed_rpm

    def set_speed_rpm(self, speed_rpm: float) -> None:
        """Set the explicit operating speed for a command schedule."""

        self.speed_rpm = speed_rpm

    @property
    def head_curve_points(self) -> tuple[tuple[float, float], ...]:
        ratio = self.speed_ratio
        return tuple((flow * ratio, head * ratio * ratio) for flow, head in self._head_curve)

    def _operating_point(self, pressure_difference_Pa: float) -> tuple[float, float, bool]:
        curve = self.head_curve_points
        requested = max(0.0, float(pressure_difference_Pa))
        shutoff = curve[0][1]
        minimum = curve[-1][1]
        if requested >= shutoff:
            return 0.0, shutoff, requested > shutoff
        if requested <= minimum:
            return curve[-1][0], minimum, requested < minimum
        for (flow0, head0), (flow1, head1) in zip(curve, curve[1:]):
            if head0 >= requested >= head1:
                fraction = (head0 - requested) / (head0 - head1)
                flow = flow0 + fraction * (flow1 - flow0)
                return flow, requested, False
        raise RuntimeError("H-Q map inversion failed")

    def evaluate(self, left: ThermoState, right: ThermoState) -> MappedPumpResult:
        phase = str(left.phase).strip().lower()
        liquid_phase = "liquid" in phase and "gas" not in phase
        if (left.quality is not None and left.quality > self.inlet_quality_tolerance) or (
            left.quality is None and not liquid_phase
        ):
            raise ValueError("MappedPump requires a liquid inlet state")
        saturation_pressure = self.properties.saturation_pressure_Pa(left.temperature_K)
        npsh_available = (
            left.pressure_Pa - saturation_pressure
        ) / (left.density_kg_m3 * 9.80665)
        required_rise = max(0.0, right.pressure_Pa - left.pressure_Pa)
        mass_flow, map_head, limited = self._operating_point(required_rise)
        if self._efficiency_curve is None:
            efficiency = float(self.isentropic_efficiency)
        else:
            efficiency = _linear_curve_value(
                self._efficiency_curve, mass_flow / self.speed_ratio
            )
        if self._npsh_curve is None:
            npsh_required = self.npsh_required_m
        else:
            npsh_required = _linear_curve_value(
                self._npsh_curve, mass_flow / self.speed_ratio
            ) * self.speed_ratio * self.speed_ratio
        cavitation_margin = (
            None if npsh_required is None else npsh_available - npsh_required
        )
        if mass_flow <= 1.0e-14 or required_rise <= 0.0:
            shaft = hydraulic = 0.0
            outlet_h = left.specific_enthalpy_J_kg
            upstream_side = None if mass_flow <= 1.0e-14 else "left"
        else:
            result = self._pump.evaluate(
                left, mass_flow, required_rise, efficiency
            )
            shaft = result.shaft_power_W
            hydraulic = result.hydraulic_power_W
            outlet_h = result.outlet.specific_enthalpy_J_kg
            upstream_side = "left"
        return MappedPumpResult(
            mass_flow_kg_s=mass_flow,
            outlet_specific_enthalpy_J_kg=outlet_h,
            pressure_rise_Pa=required_rise,
            shaft_power_W=shaft,
            hydraulic_power_W=hydraulic,
            npsh_available_m=npsh_available,
            npsh_required_m=npsh_required,
            cavitation_margin_m=cavitation_margin,
            upstream_side=upstream_side,
            map_head_pressure_rise_Pa=map_head,
            map_limited=limited,
            momentum_residual_Pa=map_head - required_rise,
            inlet_quality=left.quality,
        )

    def export_accident_history(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Export a fixed-source liquid/two-phase pump outlet accident.

        ``MappedPump`` is a performance map, not a finite-volume inventory
        owner.  The request therefore must carry an explicit upstream mass
        ledger and termination basis.  The H-Q-N operating point is evaluated
        and echoed as provenance; the failed atmospheric opening is evaluated
        by a separate native valve boundary so normal pump flow is never
        silently promoted to an accident release.
        """
        if not isinstance(request, Mapping):
            raise ValueError("accident export request must be a mapping")

        def text(name: str) -> str:
            value = request.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            return value.strip()

        def number(name: str, default: float | None = None, *, positive: bool = True) -> float:
            value = request.get(name, default)
            if isinstance(value, bool) or value is None:
                raise ValueError(f"{name} must be numeric")
            try:
                result = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be numeric") from error
            valid = result > 0.0 if positive else result >= 0.0
            if not math.isfinite(result) or not valid:
                comparator = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {comparator}")
            return result

        def vector(name: str, default: tuple[float, float, float]) -> list[float]:
            value = request.get(name, default)
            if not isinstance(value, (tuple, list)) or len(value) != 3:
                raise ValueError(f"{name} must contain three coordinates")
            result: list[float] = []
            for index, item in enumerate(value):
                if isinstance(item, bool):
                    raise ValueError(f"{name}[{index}] must be numeric")
                try:
                    coordinate = float(item)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{name}[{index}] must be numeric") from error
                if not math.isfinite(coordinate):
                    raise ValueError(f"{name}[{index}] must be finite")
                result.append(coordinate)
            if math.isclose(sum(item * item for item in result), 0.0, abs_tol=1.0e-20):
                raise ValueError(f"{name} must not be the zero vector")
            return result

        event_id = text("event_id")
        component_id = text("component_id")
        port_id = text("port_id")
        source_pressure = number(
            "source_pressure_pa_abs", request.get("initial_pressure_pa_abs")
        )
        source_temperature = number(
            "source_temperature_k", request.get("initial_temperature_k")
        )
        ambient_pressure = number("ambient_pressure_pa_abs", 101325.0)
        ambient_temperature = number("ambient_temperature_k", 288.15)
        opening_diameter = number("failure_opening_diameter_m")
        horizon = number("horizon_s")
        time_step = number("time_step_s", min(0.05, horizon))
        available_mass = number("available_mass_kg")
        termination_basis = text("termination_basis")
        termination_provenance = text("termination_provenance")
        droplet_diameter = number("droplet_diameter_m", 1.0e-3)
        flight_time = number("droplet_flight_time_s", 0.0, positive=False)
        impact_position = vector("impact_position_m", (0.0, 0.0, 0.001))
        if source_pressure <= ambient_pressure:
            raise ValueError("source pressure must exceed ambient pressure for an outward accident")
        source = self.properties.from_pT(source_pressure, source_temperature)
        phase = str(source.phase).strip().lower()
        if not ("liquid" in phase and "gas" not in phase) or (
            source.quality is not None and source.quality > self.inlet_quality_tolerance
        ):
            raise ValueError("MappedPump accident export requires a liquid source state")
        ambient = self.properties.from_pT(ambient_pressure, ambient_temperature)
        pump_pressure_rise = number("pump_pressure_rise_pa", 0.0, positive=False)
        pump_discharge = self.properties.from_pT(
            source_pressure + pump_pressure_rise,
            source_temperature,
        )
        pump_point = self.evaluate(source, pump_discharge)
        from .valve import Valve

        area = math.pi * opening_diameter**2 / 4.0
        discharge_coefficient = number("discharge_coefficient", 0.8)
        if discharge_coefficient > 1.0:
            raise ValueError("discharge_coefficient must not exceed one")
        valve = Valve(
            area,
            discharge_coefficient,
            properties=self.properties,
            allow_reverse=False,
        )
        steps: list[dict[str, Any]] = []
        cumulative_mass = 0.0
        cumulative_enthalpy = 0.0
        elapsed = 0.0
        choked_count = 0
        while elapsed < horizon - 1.0e-12:
            dt = min(time_step, horizon - elapsed)
            hydraulic = valve.evaluate(source, ambient)
            mass_flow = float(hydraulic.mass_flow_kg_s)
            if not math.isfinite(mass_flow) or mass_flow < 0.0:
                raise ValueError("native pump accident outlet returned an invalid mass flow")
            interval_mass = mass_flow * dt
            if cumulative_mass + interval_mass > available_mass + 1.0e-12:
                raise ValueError("pump accident history exceeds the upstream available mass")
            enthalpy = float(hydraulic.outlet_specific_enthalpy_J_kg)
            exit_state = self.properties.from_ph(ambient_pressure, enthalpy)
            quality = exit_state.quality
            if quality is None:
                if exit_state.phase in {"gas", "supercritical_gas", "supercritical"}:
                    quality = 1.0
                elif "liquid" in exit_state.phase:
                    quality = 0.0
                else:
                    raise ValueError("native pump accident outlet has an unsupported phase")
            quality = min(1.0, max(0.0, float(quality)))
            gas_flow = mass_flow * quality
            ground_flow = mass_flow - gas_flow
            gas_fields: dict[str, Any] = {}
            if gas_flow > 1.0e-12:
                gas_fields = {
                    "gas_pressure_pa_abs": ambient_pressure,
                    "gas_temperature_k": float(exit_state.temperature_K),
                    "gas_density_kg_m3": float(exit_state.density_kg_m3),
                    "gas_effective_area_m2": area,
                    "gas_velocity_m_s": gas_flow / (exit_state.density_kg_m3 * area),
                    "gas_velocity_origin": "mass_continuity_from_declared_area",
                }
            steps.append({
                "time_s": elapsed,
                "total_mass_flow_kg_s": mass_flow,
                "gas_mass_flow_kg_s": gas_flow,
                "ground_liquid_mass_flow_kg_s": ground_flow,
                "airborne_liquid_mass_flow_kg_s": 0.0,
                "specific_enthalpy_j_kg": enthalpy,
                "droplet_class_outcomes": [{
                    "ground_liquid_mass_flow_kg_s": ground_flow,
                    "airborne_liquid_mass_flow_kg_s": 0.0,
                    "flight_time_s": flight_time,
                    "impact_position_m": impact_position,
                    "droplet_diameter_m": droplet_diameter,
                }],
                **gas_fields,
                "provider_source_state": {
                    "source_pressure_pa_abs": source_pressure,
                    "source_temperature_k": source_temperature,
                    "source_specific_enthalpy_j_kg": float(source.specific_enthalpy_J_kg),
                    "throat_pressure_pa_abs": float(hydraulic.throat_pressure_Pa),
                    "throat_mass_flux_kg_m2_s": float(hydraulic.mass_flux_kg_m2_s),
                    "choked": bool(hydraulic.choked),
                },
            })
            if hydraulic.choked:
                choked_count += 1
            cumulative_mass += interval_mass
            cumulative_enthalpy += interval_mass * enthalpy
            elapsed += dt
        phase_basis = "two_phase" if any(
            step["gas_mass_flow_kg_s"] > 1.0e-12
            and step["ground_liquid_mass_flow_kg_s"] > 1.0e-12
            for step in steps
        ) else "liquid"
        return {
            "provider_export_schema": "prism.external_accident_history.v1",
            "provider_model": f"{type(self).__module__}.{type(self).__qualname__}",
            "provider_source_digest": request.get("provider_source_digest"),
            "provider_state_snapshot_digest": request.get("state_snapshot_digest"),
            "data": {
                "event_kind": "accidental_leak",
                "event_id": event_id,
                "component_id": component_id,
                "port_id": port_id,
                "duration_s": elapsed,
                "available_mass_kg": available_mass,
                "cumulative_mass_out_kg": cumulative_mass,
                "cumulative_specific_enthalpy_out_j": cumulative_enthalpy,
                "termination_basis": termination_basis,
                "termination_provenance": termination_provenance,
                "upstream_service": "LH2",
                "fluid": "Hydrogen",
                "phase_basis": phase_basis,
                "reference_area_m2": area,
                "reference_area_provenance": "explicit failed atmospheric pump opening area",
                "droplet_partition_provenance": "native pump outlet flash at ambient pressure; ground liquid retained; airborne liquid set to zero",
                "steps": steps,
                "provider_meta": {
                    "native_inventory_owner": "upstream_provider_declared",
                    "pump_operating_point": {
                        "mass_flow_kg_s": pump_point.mass_flow_kg_s,
                        "pressure_rise_pa": pump_point.pressure_rise_Pa,
                        "map_head_pressure_rise_pa": pump_point.map_head_pressure_rise_Pa,
                        "map_limited": pump_point.map_limited,
                        "momentum_residual_pa": pump_point.momentum_residual_Pa,
                        "cavitation_margin_m": pump_point.cavitation_margin_m,
                    },
                    "time_step_s": time_step,
                    "valve": {
                        "area_m2": area,
                        "discharge_coefficient": discharge_coefficient,
                        "choked_intervals": choked_count,
                    },
                },
            },
        }
