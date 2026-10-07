"""Quasi-steady single-phase pipe using Darcy-Weisbach momentum balance."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .properties import HydrogenProperties, ThermoState


@dataclass(frozen=True)
class PipeResult:
    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float
    reynolds_number: float
    darcy_friction_factor: float
    heat_into_fluid_W: float
    upstream_side: str | None
    pressure_drop_Pa: float = 0.0
    momentum_residual_Pa: float = 0.0


class Pipe:
    """Adiabatic or prescribed-heat-leak pipe.

    The hydraulic calculation applies only to a single-phase upstream state.
    Two-phase transport requires a replaceable correlation/model and is rejected
    rather than silently extrapolated.
    """

    def __init__(
        self,
        length_m: float,
        inner_diameter_m: float,
        roughness_m: float,
        local_loss_coefficient: float = 0.0,
        heat_into_fluid_W: float = 0.0,
        elevation_change_m: float = 0.0,
        properties: HydrogenProperties | None = None,
    ) -> None:
        raw_values = {
            "length_m": length_m,
            "inner_diameter_m": inner_diameter_m,
            "roughness_m": roughness_m,
            "local_loss_coefficient": local_loss_coefficient,
            "heat_into_fluid_W": heat_into_fluid_W,
            "elevation_change_m": elevation_change_m,
        }
        values: dict[str, float] = {}
        for name, value in raw_values.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            values[name] = number
        if values["length_m"] <= 0.0 or values["inner_diameter_m"] <= 0.0:
            raise ValueError("length_m and inner_diameter_m must be positive")
        if values["roughness_m"] < 0.0 or values["local_loss_coefficient"] < 0.0:
            raise ValueError("roughness and local loss must be non-negative")
        self.length = values["length_m"]
        self.diameter = values["inner_diameter_m"]
        self.roughness = values["roughness_m"]
        self.local_k = values["local_loss_coefficient"]
        self.heat = values["heat_into_fluid_W"]
        self.dz = values["elevation_change_m"]
        self.properties = properties or HydrogenProperties()

    @staticmethod
    def _friction_factor(reynolds: float, relative_roughness: float) -> float:
        if reynolds <= 0.0:
            return 0.0
        if reynolds < 2300.0:
            return 64.0 / reynolds
        # Haaland explicit approximation to Colebrook-White.
        return 1.0 / (-1.8 * math.log10((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds)) ** 2

    def _forward(self, upstream: ThermoState, downstream_pressure_Pa: float, elevation_sign: float) -> PipeResult:
        if upstream.quality is not None:
            raise ValueError("Pipe model is single-phase; two-phase upstream state requires another model")
        rho = upstream.density_kg_m3
        viscosity = self.properties.viscosity_Pa_s(upstream)
        driving_pressure = upstream.pressure_Pa - downstream_pressure_Pa - rho * 9.80665 * self.dz * elevation_sign
        if driving_pressure <= 0.0:
            return PipeResult(
                0.0,
                upstream.specific_enthalpy_J_kg,
                0.0,
                0.0,
                0.0,
                None,
                pressure_drop_Pa=upstream.pressure_Pa - downstream_pressure_Pa,
                # A zero-flow result is only a closed momentum state when
                # the pressure difference equals the hydrostatic head.  The
                # residual therefore keeps the signed total-head deficit
                # visible instead of fabricating a zero closure.
                momentum_residual_Pa=driving_pressure,
            )
        area = math.pi * self.diameter**2 / 4.0
        friction = 0.02
        velocity = 0.0
        reynolds = 0.0
        for _ in range(30):
            loss = friction * self.length / self.diameter + self.local_k
            if loss <= 0.0:
                raise ValueError("Pipe requires friction or a positive local loss")
            velocity = math.sqrt(2.0 * driving_pressure / (rho * loss))
            reynolds = rho * velocity * self.diameter / viscosity
            updated = self._friction_factor(reynolds, self.roughness / self.diameter)
            if abs(updated - friction) < 1e-10:
                friction = updated
                break
            friction = updated
        mass_flow = rho * area * velocity
        outlet_h = upstream.specific_enthalpy_J_kg
        heat = self.heat
        if mass_flow > 1e-12:
            outlet_h += heat / mass_flow
        dynamic_pressure_loss = 0.5 * rho * velocity**2 * loss
        modeled_pressure_drop = rho * 9.80665 * self.dz * elevation_sign + dynamic_pressure_loss
        momentum_residual = (
            upstream.pressure_Pa - downstream_pressure_Pa - modeled_pressure_drop
        )
        return PipeResult(
            mass_flow,
            outlet_h,
            reynolds,
            friction,
            heat,
            "left",
            pressure_drop_Pa=modeled_pressure_drop,
            momentum_residual_Pa=momentum_residual,
        )

    def evaluate(self, left: ThermoState, right: ThermoState) -> PipeResult:
        # Select the direction from the available total-pressure driving
        # force, not pressure alone.  This matters for cryogenic lines with
        # a substantial elevation change: a higher-pressure lower endpoint
        # can still be below the static head required to rise, in which case
        # the physically admissible direction is from the higher endpoint
        # downwards.
        gravity = 9.80665
        left_drive = (
            left.pressure_Pa
            - right.pressure_Pa
            - left.density_kg_m3 * gravity * self.dz
        )
        right_drive = (
            right.pressure_Pa
            - left.pressure_Pa
            + right.density_kg_m3 * gravity * self.dz
        )
        if left_drive >= right_drive:
            return self._forward(left, right.pressure_Pa, 1.0)
        reverse = self._forward(right, left.pressure_Pa, -1.0)
        return PipeResult(
            -reverse.mass_flow_kg_s,
            reverse.outlet_specific_enthalpy_J_kg,
            reverse.reynolds_number,
            reverse.darcy_friction_factor,
            reverse.heat_into_fluid_W,
            "right" if reverse.mass_flow_kg_s else None,
            pressure_drop_Pa=reverse.pressure_drop_Pa,
            momentum_residual_Pa=reverse.momentum_residual_Pa,
        )


class HEMPipe(Pipe):
    """Homogeneous-equilibrium two-phase extension of :class:`Pipe`.

    The phases share one velocity and the EOS mixture density is used in the
    Darcy--Weisbach momentum balance.  Dynamic viscosity is the explicit
    quality-weighted saturated-phase closure
    ``mu = (1-x) mu_l,sat + x mu_v,sat``.  This is a replaceable HEM closure;
    it is not inferred from plant time series and it does not claim to resolve
    slip, phase segregation, or transient flashing fronts.
    """

    def _effective_viscosity(self, state: ThermoState) -> float:
        if state.quality is None:
            return self.properties.viscosity_Pa_s(state)
        quality = min(max(float(state.quality), 0.0), 1.0)
        liquid = self.properties.saturated_liquid(state.pressure_Pa)
        vapor = self.properties.saturated_vapor(state.pressure_Pa)
        liquid_mu = self.properties.viscosity_Pa_s(liquid)
        vapor_mu = self.properties.viscosity_Pa_s(vapor)
        return (1.0 - quality) * liquid_mu + quality * vapor_mu

    def _forward(self, upstream: ThermoState, downstream_pressure_Pa: float, elevation_sign: float) -> PipeResult:
        rho = upstream.density_kg_m3
        viscosity = self._effective_viscosity(upstream)
        driving_pressure = upstream.pressure_Pa - downstream_pressure_Pa - rho * 9.80665 * self.dz * elevation_sign
        if driving_pressure <= 0.0:
            return PipeResult(
                0.0,
                upstream.specific_enthalpy_J_kg,
                0.0,
                0.0,
                0.0,
                None,
                pressure_drop_Pa=upstream.pressure_Pa - downstream_pressure_Pa,
                momentum_residual_Pa=driving_pressure,
            )
        area = math.pi * self.diameter**2 / 4.0
        friction = 0.02
        velocity = 0.0
        reynolds = 0.0
        for _ in range(30):
            loss = friction * self.length / self.diameter + self.local_k
            if loss <= 0.0:
                raise ValueError("Pipe requires friction or a positive local loss")
            velocity = math.sqrt(2.0 * driving_pressure / (rho * loss))
            reynolds = rho * velocity * self.diameter / viscosity
            updated = self._friction_factor(reynolds, self.roughness / self.diameter)
            if abs(updated - friction) < 1e-10:
                friction = updated
                break
            friction = updated
        mass_flow = rho * area * velocity
        outlet_h = upstream.specific_enthalpy_J_kg
        heat = self.heat
        if mass_flow > 1e-12:
            outlet_h += heat / mass_flow
        dynamic_pressure_loss = 0.5 * rho * velocity**2 * loss
        modeled_pressure_drop = rho * 9.80665 * self.dz * elevation_sign + dynamic_pressure_loss
        momentum_residual = (
            upstream.pressure_Pa - downstream_pressure_Pa - modeled_pressure_drop
        )
        return PipeResult(
            mass_flow,
            outlet_h,
            reynolds,
            friction,
            heat,
            "left",
            pressure_drop_Pa=modeled_pressure_drop,
            momentum_residual_Pa=momentum_residual,
        )
