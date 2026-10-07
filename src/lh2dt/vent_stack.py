"""First-principles terminal vent-stack component.

The stack is a one-way two-port component.  Its hydraulic part is an
explicit homogeneous-equilibrium pipe, while the exposed wall exchanges heat
with a prescribed ambient reservoir.  The downstream node still supplies the
actual back pressure, so a stack can be connected to an atmosphere, header, or
another explicitly modeled boundary without hiding that choice in the
component.

The model deliberately does not infer a discharge coefficient, stack loss, or
ambient temperature from operating time histories.  A manufacturer correlation
or a more detailed stack model can replace this component without changing
the two-port contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .pipe import HEMPipe
from .properties import HydrogenProperties, ThermoState


@dataclass(frozen=True)
class VentStackResult:
    """Hydraulic, thermal, and momentum result for one stack evaluation."""

    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float
    reynolds_number: float
    darcy_friction_factor: float
    heat_into_fluid_W: float
    upstream_side: str | None
    pressure_drop_Pa: float
    # ``None`` means the one-way boundary blocked the requested direction;
    # no hydraulic momentum equation was evaluated for that state.
    momentum_residual_Pa: float | None


class VentStack:
    """One-way HEM stack with an explicit ambient heat-transfer boundary.

    ``ambient_UA_W_K`` is the total wall-to-ambient conductance for the stack
    section.  The fluid pressure is obtained from the connected downstream
    state; ``ambient_temperature_K`` is an explicit environmental input.  The
    enthalpy ODE

    ``d h / d(UA) = (T_ambient - T(h, p)) / m_dot``

    is integrated with RK4 at the downstream pressure.  This is a finite-NTU
    control-volume closure and does not impose the ambient temperature at the
    outlet.  Reverse flow is disabled by default because a discharge stack is
    a terminal path; setting ``allow_reverse=True`` is an explicit modeling
    choice for a stack that is known to be bidirectional.
    """

    def __init__(
        self,
        length_m: float,
        inner_diameter_m: float,
        roughness_m: float,
        local_loss_coefficient: float = 0.0,
        elevation_change_m: float = 0.0,
        ambient_temperature_K: float = 300.0,
        ambient_UA_W_K: float = 0.0,
        properties: HydrogenProperties | None = None,
        integration_segments: int = 64,
        allow_reverse: bool = False,
    ) -> None:
        if isinstance(ambient_temperature_K, bool) or isinstance(ambient_UA_W_K, bool):
            raise ValueError("ambient temperature and UA must be numeric")
        if not math.isfinite(ambient_temperature_K) or ambient_temperature_K <= 0.0:
            raise ValueError("ambient_temperature_K must be finite and positive")
        if not math.isfinite(ambient_UA_W_K) or ambient_UA_W_K < 0.0:
            raise ValueError("ambient_UA_W_K must be finite and non-negative")
        if isinstance(integration_segments, bool) or not isinstance(integration_segments, int) or integration_segments < 8:
            raise ValueError("integration_segments must be an integer of at least 8")
        if not isinstance(allow_reverse, bool):
            raise ValueError("allow_reverse must be bool")
        self.properties = properties or HydrogenProperties()
        self.ambient_temperature_K = float(ambient_temperature_K)
        self.ambient_UA_W_K = float(ambient_UA_W_K)
        self.integration_segments = int(integration_segments)
        self.allow_reverse = allow_reverse
        # No fixed heat term is hidden in the hydraulic part.  All thermal
        # transfer is reported from the explicit ambient-UA closure below.
        self.pipe = HEMPipe(
            length_m=length_m,
            inner_diameter_m=inner_diameter_m,
            roughness_m=roughness_m,
            local_loss_coefficient=local_loss_coefficient,
            heat_into_fluid_W=0.0,
            elevation_change_m=elevation_change_m,
            properties=self.properties,
        )

    @property
    def length(self) -> float:
        return self.pipe.length

    @property
    def diameter(self) -> float:
        return self.pipe.diameter

    @property
    def roughness(self) -> float:
        return self.pipe.roughness

    @property
    def local_k(self) -> float:
        return self.pipe.local_k

    @property
    def dz(self) -> float:
        return self.pipe.dz

    def _ambient_outlet_enthalpy(
        self,
        pressure_Pa: float,
        inlet_specific_enthalpy_J_kg: float,
        mass_flow_kg_s: float,
    ) -> tuple[float, float]:
        """Integrate the finite-UA ambient exchange and return ``(h, Q)``."""

        if mass_flow_kg_s <= 0.0 or self.ambient_UA_W_K == 0.0:
            return inlet_specific_enthalpy_J_kg, 0.0
        target = self.properties.from_pT(
            pressure_Pa, self.ambient_temperature_K
        ).specific_enthalpy_J_kg
        inlet = float(inlet_specific_enthalpy_J_kg)
        if abs(target - inlet) <= 1.0e-12:
            return inlet, 0.0

        lower = min(inlet, target)
        upper = max(inlet, target)
        delta_ua = self.ambient_UA_W_K / self.integration_segments

        def rhs(enthalpy: float) -> float:
            state = self.properties.from_ph(pressure_Pa, enthalpy)
            return (self.ambient_temperature_K - state.temperature_K) / mass_flow_kg_s

        enthalpy = inlet
        for _ in range(self.integration_segments):
            k1 = rhs(enthalpy)
            k2 = rhs(min(upper, max(lower, enthalpy + 0.5 * delta_ua * k1)))
            k3 = rhs(min(upper, max(lower, enthalpy + 0.5 * delta_ua * k2)))
            k4 = rhs(min(upper, max(lower, enthalpy + delta_ua * k3)))
            candidate = enthalpy + delta_ua * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            enthalpy = min(upper, max(lower, candidate))
        return enthalpy, mass_flow_kg_s * (enthalpy - inlet)

    @staticmethod
    def _zero_result(upstream: ThermoState, pressure_difference_Pa: float) -> VentStackResult:
        return VentStackResult(
            mass_flow_kg_s=0.0,
            outlet_specific_enthalpy_J_kg=upstream.specific_enthalpy_J_kg,
            reynolds_number=0.0,
            darcy_friction_factor=0.0,
            heat_into_fluid_W=0.0,
            upstream_side=None,
            pressure_drop_Pa=pressure_difference_Pa,
            # A directional block is a discrete boundary condition, not a
            # solved zero-flow momentum state.  Keep the residual unavailable
            # rather than fabricating a closed hydraulic equation.
            momentum_residual_Pa=None,
        )

    def evaluate(self, left: ThermoState, right: ThermoState) -> VentStackResult:
        """Evaluate flow and heat transfer using the connected back pressure."""

        hydraulic = self.pipe.evaluate(left, right)
        mass_flow = float(hydraulic.mass_flow_kg_s)
        # Elevation can overcome (or reverse) the static-pressure ordering.
        # Enforce the one-way boundary on the hydraulic flow direction, not
        # on pressure alone, so gravity-driven forward flow remains possible.
        if not self.allow_reverse and mass_flow < 0.0:
            return self._zero_result(left, left.pressure_Pa - right.pressure_Pa)
        if abs(mass_flow) <= 1.0e-14:
            return VentStackResult(
                mass_flow_kg_s=mass_flow,
                outlet_specific_enthalpy_J_kg=hydraulic.outlet_specific_enthalpy_J_kg,
                reynolds_number=hydraulic.reynolds_number,
                darcy_friction_factor=hydraulic.darcy_friction_factor,
                heat_into_fluid_W=0.0,
                upstream_side=hydraulic.upstream_side,
                pressure_drop_Pa=hydraulic.pressure_drop_Pa,
                momentum_residual_Pa=hydraulic.momentum_residual_Pa,
            )

        if mass_flow > 0.0:
            upstream = left
            downstream_pressure = right.pressure_Pa
            upstream_side = "left"
        else:
            upstream = right
            downstream_pressure = left.pressure_Pa
            upstream_side = "right"
        outlet_h, heat = self._ambient_outlet_enthalpy(
            downstream_pressure,
            upstream.specific_enthalpy_J_kg,
            abs(mass_flow),
        )
        return VentStackResult(
            mass_flow_kg_s=mass_flow,
            outlet_specific_enthalpy_J_kg=outlet_h,
            reynolds_number=hydraulic.reynolds_number,
            darcy_friction_factor=hydraulic.darcy_friction_factor,
            heat_into_fluid_W=heat if mass_flow > 0.0 else -heat,
            upstream_side=upstream_side,
            pressure_drop_Pa=hydraulic.pressure_drop_Pa,
            momentum_residual_Pa=hydraulic.momentum_residual_Pa,
        )


__all__ = ["VentStack", "VentStackResult"]
