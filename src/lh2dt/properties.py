"""Thermodynamic state conversion with one explicit energy reference.

All pressures are absolute Pa, temperatures K, and mass-specific energies J/kg.
CoolProp's ParaHydrogen backend supplies the equation of state.  Components do
not carry their own property correlations, which prevents energy-reference
mismatches when they are connected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import CoolProp.CoolProp as CP


@dataclass(frozen=True)
class ThermoState:
    pressure_Pa: float
    temperature_K: float
    density_kg_m3: float
    specific_enthalpy_J_kg: float
    specific_internal_energy_J_kg: float
    specific_entropy_J_kgK: float
    quality: float | None
    phase: str


class HydrogenProperties:
    """Small validated facade around the CoolProp ParaHydrogen EOS."""

    def __init__(self, fluid: str = "ParaHydrogen") -> None:
        self.fluid = fluid

    @staticmethod
    def _finite(value: float, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric, not bool")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def _state(self, key1: str, value1: float, key2: str, value2: float) -> ThermoState:
        value1 = self._finite(value1, key1)
        value2 = self._finite(value2, key2)
        pressure = float(CP.PropsSI("P", key1, value1, key2, value2, self.fluid))
        temperature = float(CP.PropsSI("T", key1, value1, key2, value2, self.fluid))
        quality_raw = float(CP.PropsSI("Q", key1, value1, key2, value2, self.fluid))
        # CoolProp can return an endpoint quality a few ulps outside [0, 1]
        # after a pressure/enthalpy round trip.  Preserve the explicit
        # saturated endpoint rather than misclassifying the same branch as
        # single-phase liquid or vapor in a circulation closure.
        quality_tolerance = 1.0e-8
        if -quality_tolerance <= quality_raw <= 1.0 + quality_tolerance:
            quality = min(max(quality_raw, 0.0), 1.0)
        else:
            quality = None
        return ThermoState(
            pressure_Pa=pressure,
            temperature_K=temperature,
            density_kg_m3=float(CP.PropsSI("Dmass", key1, value1, key2, value2, self.fluid)),
            specific_enthalpy_J_kg=float(CP.PropsSI("Hmass", key1, value1, key2, value2, self.fluid)),
            specific_internal_energy_J_kg=float(CP.PropsSI("Umass", key1, value1, key2, value2, self.fluid)),
            specific_entropy_J_kgK=float(CP.PropsSI("Smass", key1, value1, key2, value2, self.fluid)),
            quality=quality,
            phase=str(CP.PhaseSI(key1, value1, key2, value2, self.fluid)),
        )

    def from_pT(self, pressure_Pa: float, temperature_K: float) -> ThermoState:
        return self._state(
            "P", self._positive(pressure_Pa, "pressure_Pa"),
            "T", self._positive(temperature_K, "temperature_K"),
        )

    def from_ph(self, pressure_Pa: float, specific_enthalpy_J_kg: float) -> ThermoState:
        return self._state(
            "P", self._positive(pressure_Pa, "pressure_Pa"),
            "Hmass", self._finite(specific_enthalpy_J_kg, "specific_enthalpy_J_kg"),
        )

    def from_ps(self, pressure_Pa: float, specific_entropy_J_kgK: float) -> ThermoState:
        return self._state(
            "P", self._positive(pressure_Pa, "pressure_Pa"),
            "Smass", self._finite(specific_entropy_J_kgK, "specific_entropy_J_kgK"),
        )

    def from_rho_u(self, density_kg_m3: float, specific_internal_energy_J_kg: float) -> ThermoState:
        return self._state(
            "Dmass", self._positive(density_kg_m3, "density_kg_m3"),
            "Umass", self._finite(specific_internal_energy_J_kg, "specific_internal_energy_J_kg"),
        )

    def from_prho(self, pressure_Pa: float, density_kg_m3: float) -> ThermoState:
        return self._state(
            "P", self._positive(pressure_Pa, "pressure_Pa"),
            "Dmass", self._positive(density_kg_m3, "density_kg_m3"),
        )

    def from_pu(self, pressure_Pa: float, specific_internal_energy_J_kg: float) -> ThermoState:
        return self._state(
            "P", self._positive(pressure_Pa, "pressure_Pa"),
            "Umass", self._finite(specific_internal_energy_J_kg, "specific_internal_energy_J_kg"),
        )

    def saturated_liquid(self, pressure_Pa: float) -> ThermoState:
        return self._state("P", self._positive(pressure_Pa, "pressure_Pa"), "Q", 0.0)

    def saturated_vapor(self, pressure_Pa: float) -> ThermoState:
        return self._state("P", self._positive(pressure_Pa, "pressure_Pa"), "Q", 1.0)

    def saturation_pressure_Pa(self, temperature_K: float) -> float:
        return float(CP.PropsSI(
            "P", "T", self._positive(temperature_K, "temperature_K"),
            "Q", 0.0, self.fluid,
        ))

    def viscosity_Pa_s(self, state: ThermoState) -> float:
        return float(CP.PropsSI(
            "VISCOSITY", "P", state.pressure_Pa, "Hmass", state.specific_enthalpy_J_kg, self.fluid
        ))

    def thermal_conductivity_W_mK(self, state: ThermoState) -> float:
        return float(CP.PropsSI(
            "CONDUCTIVITY", "P", state.pressure_Pa,
            "Hmass", state.specific_enthalpy_J_kg, self.fluid,
        ))

    def isobaric_heat_capacity_J_kgK(self, state: ThermoState) -> float:
        return float(CP.PropsSI(
            "Cpmass", "P", state.pressure_Pa,
            "Hmass", state.specific_enthalpy_J_kg, self.fluid,
        ))

    def isochoric_heat_capacity_J_kgK(self, state: ThermoState) -> float:
        return float(CP.PropsSI(
            "Cvmass", "P", state.pressure_Pa,
            "Hmass", state.specific_enthalpy_J_kg, self.fluid,
        ))

    def specific_gas_constant_J_kgK(self) -> float:
        return float(
            CP.PropsSI("gas_constant", self.fluid)
            / CP.PropsSI("molar_mass", self.fluid)
        )

    def molar_mass_kg_mol(self) -> float:
        """Return the backend molar mass as an explicit property contract."""

        return float(CP.PropsSI("molar_mass", self.fluid))

    def triple_temperature_K(self) -> float:
        return float(CP.PropsSI("Ttriple", self.fluid))

    def triple_pressure_Pa(self) -> float:
        """Return the backend triple-point pressure in Pa absolute."""

        return float(CP.PropsSI("ptriple", self.fluid))

    def critical_temperature_K(self) -> float:
        return float(CP.PropsSI("Tcrit", self.fluid))

    def critical_pressure_Pa(self) -> float:
        """Return the backend critical pressure in Pa absolute."""

        return float(CP.PropsSI("pcrit", self.fluid))

    def saturation_enthalpy_pressure_derivative_J_kg_Pa(
        self, pressure_Pa: float, quality: float
    ) -> float:
        """Return ``dh_sat/dp`` along the coexistence curve.

        CoolProp evaluates this as an equation-of-state derivative along the
        saturation line.  Only the saturated-liquid and saturated-vapor
        branches are exposed because intermediate quality is not a material
        branch for the component models.
        """

        pressure = self._positive(pressure_Pa, "pressure_Pa")
        quality = self._finite(quality, "quality")
        if quality not in (0.0, 1.0):
            raise ValueError("quality must be exactly 0.0 or 1.0")
        return float(CP.PropsSI(
            "d(Hmass)/d(P)|sigma", "P", pressure, "Q", quality, self.fluid
        ))

    def isobaric_expansion_coefficient_1_K(self, state: ThermoState) -> float:
        return float(CP.PropsSI(
            "ISOBARIC_EXPANSION_COEFFICIENT", "P", state.pressure_Pa,
            "Hmass", state.specific_enthalpy_J_kg, self.fluid,
        ))

    def specific_volume_derivatives_ph(self, state: ThermoState) -> tuple[float, float]:
        """Return ``dv/dp|h`` and ``dv/dh|p`` in SI units."""

        density = state.density_kg_m3
        drho_dp_h = float(CP.PropsSI(
            "d(Dmass)/d(P)|Hmass", "P", state.pressure_Pa,
            "Hmass", state.specific_enthalpy_J_kg, self.fluid,
        ))
        drho_dh_p = float(CP.PropsSI(
            "d(Dmass)/d(Hmass)|P", "P", state.pressure_Pa,
            "Hmass", state.specific_enthalpy_J_kg, self.fluid,
        ))
        factor = -1.0 / density**2
        return factor * drho_dp_h, factor * drho_dh_p
