"""Horizontal vessel geometry independent of the thermodynamic tank model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class HorizontalGeometryState:
    liquid_height_m: float
    liquid_volume_m3: float
    vapor_volume_m3: float
    fill_volume_fraction: float
    interface_area_m2: float
    wetted_inner_area_m2: float
    dry_inner_area_m2: float


class HorizontalVesselGeometry:
    """Horizontal cylinder with two half-ellipsoidal or flat heads.

    ``head_depth_m`` is the axial depth of one half-ellipsoidal head.  A zero
    depth represents flat end plates.  The geometry is an explicit input: a
    drawing's overall length must not be treated as straight-shell length until
    its dimension convention and head type are confirmed.
    """

    def __init__(
        self,
        inner_diameter_m: float,
        straight_length_m: float,
        head_depth_m: float,
        quadrature_order: int = 160,
    ) -> None:
        values: dict[str, float] = {}
        for name, value in {
            "inner_diameter_m": inner_diameter_m,
            "straight_length_m": straight_length_m,
            "head_depth_m": head_depth_m,
        }.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric, not bool")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            values[name] = number
        if values["inner_diameter_m"] <= 0.0 or values["straight_length_m"] < 0.0 or values["head_depth_m"] < 0.0:
            raise ValueError("diameter must be positive; straight length and head depth non-negative")
        if values["straight_length_m"] == 0.0 and values["head_depth_m"] == 0.0:
            raise ValueError("straight length and head depth cannot both be zero")
        if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, (int, np.integer)):
            raise ValueError("quadrature_order must be an integer")
        if quadrature_order < 24:
            raise ValueError("quadrature_order must be at least 24")
        self.diameter = values["inner_diameter_m"]
        self.radius = self.diameter / 2.0
        self.straight_length = values["straight_length_m"]
        self.head_depth = values["head_depth_m"]
        nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
        self._theta = (nodes + 1.0) * math.pi / 4.0
        self._theta_weight = weights * math.pi / 4.0

    @property
    def overall_length_m(self) -> float:
        return self.straight_length + 2.0 * self.head_depth

    @property
    def volume_m3(self) -> float:
        cylinder = math.pi * self.radius**2 * self.straight_length
        heads = 4.0 / 3.0 * math.pi * self.head_depth * self.radius**2
        return cylinder + heads

    @staticmethod
    def _segment_area(radius: np.ndarray | float, level_from_center: float) -> np.ndarray:
        r = np.asarray(radius, dtype=float)
        result = np.zeros_like(r)
        full = level_from_center >= r
        empty = level_from_center <= -r
        partial = ~(full | empty)
        result[full] = math.pi * r[full] ** 2
        if np.any(partial):
            rp = r[partial]
            z = float(level_from_center)
            # Area below a horizontal chord y=z.
            result[partial] = (
                rp**2 * np.arccos(-z / rp)
                + z * np.sqrt(np.maximum(0.0, rp**2 - z**2))
            )
        return result

    @staticmethod
    def _wet_angle(radius: np.ndarray | float, level_from_center: float) -> np.ndarray:
        r = np.asarray(radius, dtype=float)
        result = np.zeros_like(r)
        full = level_from_center >= r
        empty = level_from_center <= -r
        partial = ~(full | empty)
        result[full] = 2.0 * math.pi
        if np.any(partial):
            result[partial] = 2.0 * np.arccos(-level_from_center / r[partial])
        return result

    def _head_integrals(self, z: float) -> tuple[float, float, float, float]:
        """Return both-head volume, interface, wet area, and total area."""

        if self.head_depth == 0.0:
            segment = float(self._segment_area(np.array([self.radius]), z)[0])
            return 0.0, 0.0, 2.0 * segment, 2.0 * math.pi * self.radius**2
        theta = self._theta
        weight = self._theta_weight
        radius = self.radius * np.sin(theta)
        dx_dtheta = self.head_depth * np.sin(theta)
        ds_dtheta = np.sqrt(
            (self.head_depth * np.sin(theta))**2
            + (self.radius * np.cos(theta))**2
        )
        segment = self._segment_area(radius, z)
        chord = np.where(np.abs(z) < radius, 2.0 * np.sqrt(np.maximum(0.0, radius**2 - z**2)), 0.0)
        angle = self._wet_angle(radius, z)
        # Factor two accounts for the two identical heads.
        volume = 2.0 * float(np.sum(weight * segment * dx_dtheta))
        interface = 2.0 * float(np.sum(weight * chord * dx_dtheta))
        wet_area = 2.0 * float(np.sum(weight * radius * angle * ds_dtheta))
        total_area = 2.0 * float(np.sum(weight * radius * (2.0 * math.pi) * ds_dtheta))
        return volume, interface, wet_area, total_area

    @property
    def inner_surface_area_m2(self) -> float:
        cylinder = 2.0 * math.pi * self.radius * self.straight_length
        return cylinder + self._head_integrals(self.radius)[3]

    def at_height(self, liquid_height_m: float) -> HorizontalGeometryState:
        if isinstance(liquid_height_m, bool):
            raise ValueError("liquid_height_m must be numeric, not bool")
        try:
            height = float(liquid_height_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("liquid_height_m must be numeric") from exc
        if not math.isfinite(height):
            raise ValueError("liquid_height_m must be finite")
        if not 0.0 <= height <= self.diameter:
            raise ValueError("liquid_height_m must lie within the inner diameter")
        z = height - self.radius
        cylinder_segment = float(self._segment_area(np.array([self.radius]), z)[0])
        cylinder_volume = cylinder_segment * self.straight_length
        cylinder_chord = 0.0
        if abs(z) < self.radius:
            cylinder_chord = 2.0 * math.sqrt(self.radius**2 - z**2) * self.straight_length
        cylinder_wet = float(self._wet_angle(np.array([self.radius]), z)[0]) * self.radius * self.straight_length
        head_volume, head_interface, head_wet, head_total = self._head_integrals(z)
        liquid_volume = cylinder_volume + head_volume
        total_area = 2.0 * math.pi * self.radius * self.straight_length + head_total
        wet_area = cylinder_wet + head_wet
        # Clamp quadrature roundoff at empty/full limits.
        liquid_volume = min(max(liquid_volume, 0.0), self.volume_m3)
        wet_area = min(max(wet_area, 0.0), total_area)
        return HorizontalGeometryState(
            liquid_height_m=height,
            liquid_volume_m3=liquid_volume,
            vapor_volume_m3=self.volume_m3 - liquid_volume,
            fill_volume_fraction=liquid_volume / self.volume_m3,
            interface_area_m2=cylinder_chord + head_interface,
            wetted_inner_area_m2=wet_area,
            dry_inner_area_m2=total_area - wet_area,
        )

    def height_from_liquid_volume(self, liquid_volume_m3: float) -> float:
        if isinstance(liquid_volume_m3, bool):
            raise ValueError("liquid_volume_m3 must be numeric, not bool")
        try:
            volume = float(liquid_volume_m3)
        except (TypeError, ValueError) as exc:
            raise ValueError("liquid_volume_m3 must be numeric") from exc
        if not math.isfinite(volume):
            raise ValueError("liquid_volume_m3 must be finite")
        tolerance = max(self.volume_m3, 1.0) * 1e-12
        if volume < -tolerance or volume > self.volume_m3 + tolerance:
            raise ValueError("liquid_volume_m3 lies outside the vessel")
        if volume <= tolerance:
            return 0.0
        if volume >= self.volume_m3 - tolerance:
            return self.diameter
        return float(brentq(
            lambda height: self.at_height(height).liquid_volume_m3 - volume,
            0.0,
            self.diameter,
            xtol=1e-12,
            rtol=1e-12,
        ))


@dataclass(frozen=True)
class DptEngineeringLevelScale:
    """Convert a declared DPT engineering span into a bounded liquid inventory.

    The scale is an observation-layer contract, not a fitted tank parameter.
    A field level table supplies the end points and a declared liquid density
    supplies the hydrostatic height interpretation.  The vessel geometry then
    converts that height to volume and mass, retaining the nonlinear horizontal
    vessel cross-section instead of assuming mass is linear in DPT.

    ``clamp`` is explicit because a transmitter value outside the engineering
    table must not silently create an impossible liquid height.  Keeping the
    clamped result visible lets an audit report distinguish a bounded estimate
    from an in-range conversion.
    """

    dpt_low_mmH2O: float
    dpt_high_mmH2O: float
    liquid_height_low_m: float
    liquid_height_high_m: float
    liquid_density_kg_m3: float
    clamp: bool = True

    def __post_init__(self) -> None:
        numeric = {
            "dpt_low_mmH2O": self.dpt_low_mmH2O,
            "dpt_high_mmH2O": self.dpt_high_mmH2O,
            "liquid_height_low_m": self.liquid_height_low_m,
            "liquid_height_high_m": self.liquid_height_high_m,
            "liquid_density_kg_m3": self.liquid_density_kg_m3,
        }
        for name, value in numeric.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric, not bool")
            try:
                converted = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(converted):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, converted)
        if self.dpt_high_mmH2O <= self.dpt_low_mmH2O:
            raise ValueError("DPT span must be increasing")
        if self.liquid_height_high_m <= self.liquid_height_low_m:
            raise ValueError("liquid-height span must be increasing")
        if self.liquid_density_kg_m3 <= 0.0:
            raise ValueError("liquid_density_kg_m3 must be positive")
        if not isinstance(self.clamp, bool):
            raise ValueError("clamp must be boolean")

    @property
    def dpt_span_mmH2O(self) -> float:
        return self.dpt_high_mmH2O - self.dpt_low_mmH2O

    @property
    def liquid_height_span_m(self) -> float:
        return self.liquid_height_high_m - self.liquid_height_low_m

    def _bounded_fraction(self, dpt_mmH2O: float) -> tuple[float, str]:
        try:
            value = float(dpt_mmH2O)
        except (TypeError, ValueError) as exc:
            raise ValueError("dpt_mmH2O must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError("dpt_mmH2O must be finite")
        fraction = (value - self.dpt_low_mmH2O) / self.dpt_span_mmH2O
        if 0.0 <= fraction <= 1.0:
            return fraction, "in_range"
        if not self.clamp:
            raise ValueError("DPT value lies outside the declared engineering span")
        if fraction < 0.0:
            return 0.0, "below_range_clamped"
        return 1.0, "above_range_clamped"

    def liquid_height_from_dpt(self, dpt_mmH2O: float) -> tuple[float, str]:
        """Return liquid height and whether the value was clamped."""

        fraction, status = self._bounded_fraction(dpt_mmH2O)
        height = self.liquid_height_low_m + fraction * self.liquid_height_span_m
        return height, status

    def dpt_from_liquid_height(self, liquid_height_m: float) -> float:
        try:
            height = float(liquid_height_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("liquid_height_m must be numeric") from exc
        if not math.isfinite(height):
            raise ValueError("liquid_height_m must be finite")
        fraction = (height - self.liquid_height_low_m) / self.liquid_height_span_m
        if not 0.0 <= fraction <= 1.0:
            if not self.clamp:
                raise ValueError("liquid height lies outside the declared engineering span")
            fraction = min(max(fraction, 0.0), 1.0)
        return self.dpt_low_mmH2O + fraction * self.dpt_span_mmH2O

    def liquid_inventory_from_dpt(
        self,
        dpt_mmH2O: float,
        geometry: "HorizontalVesselGeometry",
    ) -> dict[str, float | str]:
        """Return bounded height, liquid volume and mass from the DPT scale."""

        if not isinstance(geometry, HorizontalVesselGeometry):
            raise TypeError("geometry must be HorizontalVesselGeometry")
        height, status = self.liquid_height_from_dpt(dpt_mmH2O)
        if height > geometry.diameter:
            if not self.clamp:
                raise ValueError("engineering level exceeds vessel diameter")
            height = geometry.diameter
            status = "above_geometry_clamped"
        volume = geometry.at_height(height).liquid_volume_m3
        mass = volume * self.liquid_density_kg_m3
        return {
            "dpt_mmH2O": float(dpt_mmH2O),
            "liquid_height_m": height,
            "liquid_volume_m3": volume,
            "liquid_mass_kg": mass,
            "status": status,
        }


@dataclass(frozen=True)
class HydrostaticLevelEnvelope:
    """An interval-valued level result from a differential-pressure signal.

    A DPT value outside its declared valid range is not converted into a
    fictitious exact level.  The safe physical result is the full span
    between the two impulse-line taps until the transmitter range or signal
    state is resolved.
    """

    lower_height_m: float
    upper_height_m: float
    reported_pressure_Pa: float
    pressure_interval_Pa: tuple[float, float]
    status: str
    exact: bool

    def __post_init__(self) -> None:
        if self.lower_height_m > self.upper_height_m:
            raise ValueError("lower_height_m must not exceed upper_height_m")
        if self.status not in {
            "exact", "resolved_interval", "below_valid_range",
            "above_valid_range", "out_of_physical_tap_span",
        }:
            raise ValueError("unsupported hydrostatic envelope status")
        if self.exact and self.status != "exact":
            raise ValueError("exact envelope must use status='exact'")


@dataclass(frozen=True)
class HydrostaticDifferentialPressure:
    lower_tap_height_m: float
    upper_tap_height_m: float
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        if self.lower_tap_height_m >= self.upper_tap_height_m:
            raise ValueError("lower tap must be below upper tap")
        if self.gravity_m_s2 <= 0.0:
            raise ValueError("gravity must be positive")

    @staticmethod
    def _pressure_relative_to_interface(
        height_m: float,
        liquid_height_m: float,
        liquid_density_kg_m3: float,
        vapor_density_kg_m3: float,
        gravity_m_s2: float,
    ) -> float:
        if height_m <= liquid_height_m:
            return liquid_density_kg_m3 * gravity_m_s2 * (liquid_height_m - height_m)
        return -vapor_density_kg_m3 * gravity_m_s2 * (height_m - liquid_height_m)

    def evaluate(
        self,
        liquid_height_m: float,
        liquid_density_kg_m3: float,
        vapor_density_kg_m3: float,
    ) -> float:
        if liquid_density_kg_m3 <= 0.0 or vapor_density_kg_m3 <= 0.0:
            raise ValueError("phase densities must be positive")
        lower = self._pressure_relative_to_interface(
            self.lower_tap_height_m, liquid_height_m,
            liquid_density_kg_m3, vapor_density_kg_m3, self.gravity_m_s2,
        )
        upper = self._pressure_relative_to_interface(
            self.upper_tap_height_m, liquid_height_m,
            liquid_density_kg_m3, vapor_density_kg_m3, self.gravity_m_s2,
        )
        return lower - upper

    def height_from_differential_pressure(
        self,
        differential_pressure_Pa: float,
        liquid_density_kg_m3: float,
        vapor_density_kg_m3: float,
    ) -> float:
        """Invert the common case where the interface is between both taps."""

        density_difference = liquid_density_kg_m3 - vapor_density_kg_m3
        if density_difference <= 0.0:
            raise ValueError("liquid density must exceed vapor density")
        numerator = (
            differential_pressure_Pa / self.gravity_m_s2
            + liquid_density_kg_m3 * self.lower_tap_height_m
            - vapor_density_kg_m3 * self.upper_tap_height_m
        )
        height = numerator / density_difference
        if not self.lower_tap_height_m <= height <= self.upper_tap_height_m:
            raise ValueError("inverted interface lies outside the tap span")
        return height

    def height_envelope_from_differential_pressure(
        self,
        differential_pressure_Pa: float,
        liquid_density_kg_m3: float,
        vapor_density_kg_m3: float,
        *,
        valid_pressure_range_Pa: tuple[float, float] | None = None,
        uncertainty_Pa: float = 0.0,
    ) -> HydrostaticLevelEnvelope:
        """Return a bounded level interval without inventing a DPT scale.

        ``valid_pressure_range_Pa`` is the transmitter's declared operating
        range.  If it is absent, only the geometric tap-span limits are
        applied.  A value outside the declared range is classified and mapped
        to the full tap span; it is never treated as an exact inventory.  A
        finite ``uncertainty_Pa`` expands an in-range result into an interval.
        """

        if isinstance(differential_pressure_Pa, bool):
            raise ValueError("differential_pressure_Pa must be numeric")
        try:
            reported = float(differential_pressure_Pa)
            uncertainty = float(uncertainty_Pa)
        except (TypeError, ValueError) as exc:
            raise ValueError("differential-pressure inputs must be numeric") from exc
        if not math.isfinite(reported):
            raise ValueError("differential_pressure_Pa must be finite")
        if not math.isfinite(uncertainty) or uncertainty < 0.0:
            raise ValueError("uncertainty_Pa must be finite and non-negative")
        if valid_pressure_range_Pa is not None:
            if len(valid_pressure_range_Pa) != 2:
                raise ValueError("valid_pressure_range_Pa must contain two values")
            valid_low, valid_high = map(float, valid_pressure_range_Pa)
            if (
                not math.isfinite(valid_low)
                or not math.isfinite(valid_high)
                or valid_low < 0.0
                or valid_high <= valid_low
            ):
                raise ValueError("valid_pressure_range_Pa must be an increasing non-negative range")
        else:
            valid_low = -math.inf
            valid_high = math.inf

        if reported < valid_low:
            return HydrostaticLevelEnvelope(
                self.lower_tap_height_m,
                self.upper_tap_height_m,
                reported,
                (valid_low, valid_low),
                "below_valid_range",
                False,
            )
        if reported > valid_high:
            return HydrostaticLevelEnvelope(
                self.lower_tap_height_m,
                self.upper_tap_height_m,
                reported,
                (valid_high, valid_high),
                "above_valid_range",
                False,
            )

        physical_low = self.evaluate(
            self.lower_tap_height_m, liquid_density_kg_m3, vapor_density_kg_m3
        )
        physical_high = self.evaluate(
            self.upper_tap_height_m, liquid_density_kg_m3, vapor_density_kg_m3
        )
        physical_low, physical_high = sorted((physical_low, physical_high))
        if reported < physical_low or reported > physical_high:
            return HydrostaticLevelEnvelope(
                self.lower_tap_height_m,
                self.upper_tap_height_m,
                reported,
                (reported, reported),
                "out_of_physical_tap_span",
                False,
            )

        low_pressure = max(physical_low, reported - uncertainty)
        high_pressure = min(physical_high, reported + uncertainty)
        low_height = self.height_from_differential_pressure(
            low_pressure, liquid_density_kg_m3, vapor_density_kg_m3
        )
        high_height = self.height_from_differential_pressure(
            high_pressure, liquid_density_kg_m3, vapor_density_kg_m3
        )
        lower_height, upper_height = sorted((low_height, high_height))
        exact = uncertainty == 0.0
        return HydrostaticLevelEnvelope(
            lower_height,
            upper_height,
            reported,
            (low_pressure, high_pressure),
            "exact" if exact else "resolved_interval",
            exact,
        )
