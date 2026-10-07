"""Momentum-balance building block for buoyancy-driven tank circulation.

NASA multi-node cryogenic-tank models resolve recirculation by connecting
radial and axial fluid nodes with branches that satisfy mass, momentum, and
energy conservation. This module provides the corresponding reduced closed-
loop primitive. It does not infer flow from a heat-transfer coefficient and
contains no facility-data calibration.

The two representative leg states must be single phase and evaluated at the
same reference pressure. The signed circulation rate follows from the exact
density-difference buoyancy head and Darcy-Weisbach plus local losses in both
legs. Positive flow travels upward through ``rising_leg`` and downward through
``descending_leg``. The paired paths give zero net mass transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.optimize import brentq

from .properties import HydrogenProperties, ThermoState


@dataclass(frozen=True)
class CirculationLegGeometry:
    """Hydraulic geometry and fixed losses for one loop leg."""

    length_m: float
    hydraulic_diameter_m: float
    flow_area_m2: float
    roughness_m: float = 0.0
    local_loss_coefficient: float = 0.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.length_m, "length_m"),
            (self.hydraulic_diameter_m, "hydraulic_diameter_m"),
            (self.flow_area_m2, "flow_area_m2"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.roughness_m) or self.roughness_m < 0.0:
            raise ValueError("roughness_m must be finite and non-negative")
        if (
            not math.isfinite(self.local_loss_coefficient)
            or self.local_loss_coefficient < 0.0
        ):
            raise ValueError(
                "local_loss_coefficient must be finite and non-negative"
            )


@dataclass(frozen=True)
class NaturalCirculationResult:
    """Steady result of the closed-loop momentum balance."""

    mass_flow_kg_s: float
    buoyancy_pressure_Pa: float
    friction_pressure_Pa: float
    hydraulic_residual_Pa: float
    rising_leg_reynolds_number: float
    descending_leg_reynolds_number: float
    rising_leg_darcy_friction_factor: float
    descending_leg_darcy_friction_factor: float
    lower_control_volume_heat_W: float
    upper_control_volume_heat_W: float
    mass_residual_kg_s: float
    energy_residual_W: float


class NaturalCirculationLoop:
    """Solve a two-leg, single-phase natural-circulation loop.

    The buoyancy driving pressure is
    ``dp_b = g H (rho_descending - rho_rising)``. Each leg contributes
    Darcy-Weisbach and local loss pressure using its own density, viscosity,
    area, diameter, length, and roughness. The only empirical inputs are the
    explicitly exposed wall roughness and local-loss coefficients; they are
    geometry/design parameters and are never inferred from facility time
    series.

    The returned lower/upper heat rates are the conservative enthalpy transport
    of the closed loop. They are equal and opposite and can be applied to two
    stacked control volumes without changing either control-volume mass.
    """

    def __init__(
        self,
        vertical_separation_m: float,
        rising_leg: CirculationLegGeometry,
        descending_leg: CirculationLegGeometry,
        gravity_m_s2: float = 9.80665,
        properties: HydrogenProperties | None = None,
    ) -> None:
        if (
            not math.isfinite(vertical_separation_m)
            or vertical_separation_m <= 0.0
        ):
            raise ValueError(
                "vertical_separation_m must be finite and positive"
            )
        if not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
            raise ValueError("gravity_m_s2 must be finite and positive")
        self.height = float(vertical_separation_m)
        self.rising_leg = rising_leg
        self.descending_leg = descending_leg
        self.gravity = float(gravity_m_s2)
        self.properties = properties or HydrogenProperties()

    @staticmethod
    def _friction_factor(reynolds: float, relative_roughness: float) -> float:
        if reynolds <= 0.0:
            return 0.0
        if reynolds < 2_300.0:
            return 64.0 / reynolds
        # Haaland explicit approximation to the Colebrook-White relation.
        return 1.0 / (
            -1.8
            * math.log10(
                (relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds
            )
        ) ** 2

    @staticmethod
    def _validate_states(rising: ThermoState, descending: ThermoState) -> None:
        tolerance = 1.0e-10
        if any(
            state.quality is not None
            and tolerance < state.quality < 1.0 - tolerance
            for state in (rising, descending)
        ):
            raise ValueError(
                "NaturalCirculationLoop is single-phase; two-phase leg states "
                "require a replaceable two-phase momentum model"
            )
        pressure_scale = max(
            abs(rising.pressure_Pa), abs(descending.pressure_Pa), 1.0
        )
        if abs(rising.pressure_Pa - descending.pressure_Pa) > 1.0e-6 * pressure_scale:
            raise ValueError(
                "leg states must use the same representative pressure"
            )
        def phase_label(state: ThermoState) -> str:
            if state.quality is not None:
                return "liquid" if state.quality <= tolerance else "vapor"
            return "vapor" if state.phase in {"gas", "supercritical_gas"} else "liquid"

        if phase_label(rising) != phase_label(descending):
            raise ValueError("leg states must belong to the same single phase")

    def _leg_loss(
        self,
        mass_flow_magnitude_kg_s: float,
        state: ThermoState,
        geometry: CirculationLegGeometry,
    ) -> tuple[float, float, float]:
        if mass_flow_magnitude_kg_s <= 0.0:
            return 0.0, 0.0, 0.0
        viscosity = self.properties.viscosity_Pa_s(state)
        reynolds = (
            mass_flow_magnitude_kg_s
            * geometry.hydraulic_diameter_m
            / (geometry.flow_area_m2 * viscosity)
        )
        friction = self._friction_factor(
            reynolds, geometry.roughness_m / geometry.hydraulic_diameter_m
        )
        loss_coefficient = (
            friction * geometry.length_m / geometry.hydraulic_diameter_m
            + geometry.local_loss_coefficient
        )
        pressure_loss = (
            0.5
            * loss_coefficient
            * mass_flow_magnitude_kg_s**2
            / (state.density_kg_m3 * geometry.flow_area_m2**2)
        )
        return pressure_loss, reynolds, friction

    def evaluate(
        self,
        rising_leg_state: ThermoState,
        descending_leg_state: ThermoState,
    ) -> NaturalCirculationResult:
        self._validate_states(rising_leg_state, descending_leg_state)
        buoyancy = (
            self.gravity
            * self.height
            * (
                descending_leg_state.density_kg_m3
                - rising_leg_state.density_kg_m3
            )
        )
        if abs(buoyancy) <= 1.0e-14:
            return NaturalCirculationResult(
                0.0, buoyancy, 0.0, buoyancy,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            )

        target = abs(buoyancy)

        def pressure_loss(magnitude: float) -> float:
            rising_loss, _, _ = self._leg_loss(
                magnitude, rising_leg_state, self.rising_leg
            )
            descending_loss, _, _ = self._leg_loss(
                magnitude, descending_leg_state, self.descending_leg
            )
            return rising_loss + descending_loss

        upper = 1.0
        for _ in range(80):
            if pressure_loss(upper) >= target:
                break
            upper *= 2.0
        else:
            raise RuntimeError("failed to bracket natural-circulation flow")

        magnitude = float(
            brentq(
                lambda value: pressure_loss(value) - target,
                0.0,
                upper,
                xtol=1.0e-14,
                rtol=1.0e-12,
            )
        )
        direction = math.copysign(1.0, buoyancy)
        mass_flow = direction * magnitude
        rising_loss, rising_re, rising_f = self._leg_loss(
            magnitude, rising_leg_state, self.rising_leg
        )
        descending_loss, descending_re, descending_f = self._leg_loss(
            magnitude, descending_leg_state, self.descending_leg
        )
        signed_loss = direction * (rising_loss + descending_loss)
        lower_heat = mass_flow * (
            descending_leg_state.specific_enthalpy_J_kg
            - rising_leg_state.specific_enthalpy_J_kg
        )
        upper_heat = -lower_heat
        return NaturalCirculationResult(
            mass_flow_kg_s=mass_flow,
            buoyancy_pressure_Pa=buoyancy,
            friction_pressure_Pa=signed_loss,
            hydraulic_residual_Pa=buoyancy - signed_loss,
            rising_leg_reynolds_number=rising_re,
            descending_leg_reynolds_number=descending_re,
            rising_leg_darcy_friction_factor=rising_f,
            descending_leg_darcy_friction_factor=descending_f,
            lower_control_volume_heat_W=lower_heat,
            upper_control_volume_heat_W=upper_heat,
            mass_residual_kg_s=0.0,
            energy_residual_W=lower_heat + upper_heat,
        )
