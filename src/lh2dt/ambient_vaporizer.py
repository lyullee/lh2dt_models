"""Frost-aware ambient-air vaporizer physics.

The model keeps the hydrogen-side enthalpy integration implemented by
``Vaporizer`` and calculates its effective conductance from explicit air-side
natural/forced convection, radiation, wall/internal resistance, and frost.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .properties import HydrogenProperties, ThermoState
from .vaporizer import FlowUAAdjustment, PressureDropRelation, Vaporizer, VaporizerResult


STEFAN_BOLTZMANN = 5.670374419e-8
WATER_VAPOR_GAS_CONSTANT = 461.52
ICE_FROST_LATENT_HEAT_J_KG = 333_500.0


@dataclass(frozen=True)
class AmbientAirBoundary:
    temperature_K: float
    relative_humidity: float = 0.60
    wind_speed_m_s: float = 0.0
    pressure_Pa: float = 101_325.0

    def __post_init__(self) -> None:
        values = (
            self.temperature_K,
            self.relative_humidity,
            self.wind_speed_m_s,
            self.pressure_Pa,
        )
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
            raise ValueError("ambient-air boundary values must be finite numbers")
        if self.temperature_K <= 0.0 or self.pressure_Pa <= 0.0:
            raise ValueError("temperature_K and pressure_Pa must be positive")
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError("relative_humidity must lie in [0, 1]")
        if self.wind_speed_m_s < 0.0:
            raise ValueError("wind_speed_m_s must be non-negative")


@dataclass(frozen=True)
class FrostLayerState:
    thickness_m: float = 0.0
    density_kg_m3: float = 300.0

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not math.isfinite(float(value))
            for value in (self.thickness_m, self.density_kg_m3)
        ):
            raise ValueError("frost state values must be finite numbers")
        if self.thickness_m < 0.0 or self.density_kg_m3 <= 0.0:
            raise ValueError("frost thickness must be non-negative and density positive")


@dataclass(frozen=True)
class AmbientVaporizerHeatTransfer:
    effective_UA_W_K: float
    natural_convection_W_m2K: float
    forced_convection_W_m2K: float
    mixed_convection_W_m2K: float
    radiation_W_m2K: float
    external_resistance_K_W: float
    frost_resistance_K_W: float
    total_resistance_K_W: float


@dataclass(frozen=True)
class AmbientAirVaporizerResult:
    outlet: ThermoState
    heat_into_hydrogen_W: float
    requested_heat_W: float
    capacity_limited: bool
    surface_temperature_K: float
    heat_transfer: AmbientVaporizerHeatTransfer
    frost_state: FrostLayerState


def _air_properties(temperature_K: float, pressure_Pa: float) -> tuple[float, float, float, float, float]:
    """Return density, dynamic viscosity, conductivity, cp and Pr for air."""

    temperature = min(max(float(temperature_K), 180.0), 400.0)
    density = pressure_Pa / (287.058 * temperature)
    viscosity = (
        1.716e-5 * (temperature / 273.15) ** 1.5
        * (273.15 + 111.0) / (temperature + 111.0)
    )
    conductivity = 0.0241 * (temperature / 273.15) ** 0.90
    cp = 1006.0
    prandtl = cp * viscosity / conductivity
    return density, viscosity, conductivity, cp, prandtl


def _saturation_vapor_pressure_Pa(temperature_K: float, *, over_ice: bool) -> float:
    temperature_C = min(max(temperature_K - 273.15, -100.0), 80.0)
    if over_ice:
        return 611.15 * math.exp(22.452 * temperature_C / (temperature_C + 272.55))
    return 610.94 * math.exp(17.625 * temperature_C / (temperature_C + 243.04))


class AmbientAirVaporizer:
    """Steady LH2 vaporizer with weather and frost dependent conductance."""

    def __init__(
        self,
        *,
        external_area_m2: float,
        characteristic_length_m: float,
        hydraulic_diameter_m: float,
        internal_UA_W_K: float,
        wall_resistance_K_W: float,
        emissivity: float,
        maximum_heat_W: float,
        frost_conductivity_W_mK: float = 0.20,
        frost_deposition_efficiency: float = 1.0,
        default_relative_humidity: float = 0.60,
        default_wind_speed_m_s: float = 0.0,
        default_frost_thickness_m: float = 0.0,
        minimum_approach_K: float = 0.0,
        pressure_drop_relation: PressureDropRelation | None = None,
        flow_ua_adjustment: FlowUAAdjustment | None = None,
        properties: HydrogenProperties | None = None,
        integration_segments: int = 96,
    ) -> None:
        numeric = {
            "external_area_m2": external_area_m2,
            "characteristic_length_m": characteristic_length_m,
            "hydraulic_diameter_m": hydraulic_diameter_m,
            "internal_UA_W_K": internal_UA_W_K,
            "wall_resistance_K_W": wall_resistance_K_W,
            "emissivity": emissivity,
            "maximum_heat_W": maximum_heat_W,
            "frost_conductivity_W_mK": frost_conductivity_W_mK,
            "frost_deposition_efficiency": frost_deposition_efficiency,
            "default_relative_humidity": default_relative_humidity,
            "default_wind_speed_m_s": default_wind_speed_m_s,
            "default_frost_thickness_m": default_frost_thickness_m,
            "minimum_approach_K": minimum_approach_K,
        }
        if any(isinstance(value, bool) for value in numeric.values()):
            raise ValueError("ambient-air vaporizer parameters must be numeric")
        try:
            numeric = {key: float(value) for key, value in numeric.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("ambient-air vaporizer parameters must be numeric") from exc
        if any(not math.isfinite(value) for value in numeric.values()):
            raise ValueError("ambient-air vaporizer parameters must be finite")
        for name in (
            "external_area_m2", "characteristic_length_m", "hydraulic_diameter_m",
            "internal_UA_W_K", "maximum_heat_W", "frost_conductivity_W_mK",
        ):
            if numeric[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
        if numeric["wall_resistance_K_W"] < 0.0:
            raise ValueError("wall_resistance_K_W must be non-negative")
        if not 0.0 <= numeric["emissivity"] <= 1.0:
            raise ValueError("emissivity must lie in [0, 1]")
        for name in ("frost_deposition_efficiency", "default_relative_humidity"):
            if not 0.0 <= numeric[name] <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if numeric["default_wind_speed_m_s"] < 0.0 or numeric["default_frost_thickness_m"] < 0.0:
            raise ValueError("default wind speed and frost thickness must be non-negative")
        if numeric["minimum_approach_K"] < 0.0:
            raise ValueError("minimum_approach_K must be non-negative")
        if not isinstance(integration_segments, int) or integration_segments < 32:
            raise ValueError("integration_segments must be an integer of at least 32")

        self.external_area_m2 = numeric["external_area_m2"]
        self.characteristic_length_m = numeric["characteristic_length_m"]
        self.hydraulic_diameter_m = numeric["hydraulic_diameter_m"]
        self.internal_UA_W_K = numeric["internal_UA_W_K"]
        self.wall_resistance_K_W = numeric["wall_resistance_K_W"]
        self.emissivity = numeric["emissivity"]
        self.maximum_heat = numeric["maximum_heat_W"]
        self.frost_conductivity_W_mK = numeric["frost_conductivity_W_mK"]
        self.frost_deposition_efficiency = numeric["frost_deposition_efficiency"]
        self.default_relative_humidity = numeric["default_relative_humidity"]
        self.default_wind_speed_m_s = numeric["default_wind_speed_m_s"]
        self.default_frost_thickness_m = numeric["default_frost_thickness_m"]
        self.minimum_approach_K = numeric["minimum_approach_K"]
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
        self.properties = properties or HydrogenProperties()
        self.integration_segments = integration_segments
        self.parameterization = "ambient_air_frost_resistance_network"

    def heat_transfer(
        self,
        boundary: AmbientAirBoundary,
        surface_temperature_K: float,
        frost_state: FrostLayerState | None = None,
    ) -> AmbientVaporizerHeatTransfer:
        frost = frost_state or FrostLayerState(self.default_frost_thickness_m)
        surface_temperature = float(surface_temperature_K)
        if not math.isfinite(surface_temperature) or surface_temperature <= 0.0:
            raise ValueError("surface_temperature_K must be finite and positive")
        film_temperature = 0.5 * (boundary.temperature_K + surface_temperature)
        density, viscosity, conductivity, cp, prandtl = _air_properties(
            film_temperature, boundary.pressure_Pa
        )
        kinematic_viscosity = viscosity / density
        alpha = conductivity / (density * cp)
        beta = 1.0 / film_temperature
        delta_temperature = abs(boundary.temperature_K - surface_temperature)
        rayleigh = (
            9.80665 * beta * delta_temperature * self.characteristic_length_m ** 3
            / max(kinematic_viscosity * alpha, 1.0e-30)
        )
        natural_nusselt = (
            0.825
            + 0.387 * rayleigh ** (1.0 / 6.0)
            / (1.0 + (0.492 / prandtl) ** (9.0 / 16.0)) ** (8.0 / 27.0)
        ) ** 2
        h_natural = natural_nusselt * conductivity / self.characteristic_length_m

        reynolds = (
            density * boundary.wind_speed_m_s * self.hydraulic_diameter_m
            / max(viscosity, 1.0e-30)
        )
        forced_nusselt = 0.3
        if reynolds > 0.0:
            forced_nusselt += (
                0.62 * reynolds ** 0.5 * prandtl ** (1.0 / 3.0)
                / (1.0 + (0.4 / prandtl) ** (2.0 / 3.0)) ** 0.25
                * (1.0 + (reynolds / 282_000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0)
            )
        h_forced = forced_nusselt * conductivity / self.hydraulic_diameter_m
        h_mixed = (h_natural ** 3 + h_forced ** 3) ** (1.0 / 3.0)
        h_radiation = (
            self.emissivity * STEFAN_BOLTZMANN
            * (boundary.temperature_K + surface_temperature)
            * (boundary.temperature_K ** 2 + surface_temperature ** 2)
        )
        external_conductance = self.external_area_m2 * (h_mixed + h_radiation)
        external_resistance = 1.0 / max(external_conductance, 1.0e-30)
        frost_resistance = (
            frost.thickness_m
            / (self.frost_conductivity_W_mK * self.external_area_m2)
        )
        total_resistance = (
            external_resistance + frost_resistance + self.wall_resistance_K_W
            + 1.0 / self.internal_UA_W_K
        )
        return AmbientVaporizerHeatTransfer(
            1.0 / total_resistance,
            h_natural,
            h_forced,
            h_mixed,
            h_radiation,
            external_resistance,
            frost_resistance,
            total_resistance,
        )

    def advance_frost(
        self,
        frost_state: FrostLayerState,
        boundary: AmbientAirBoundary,
        surface_temperature_K: float,
        duration_s: float,
    ) -> FrostLayerState:
        duration = float(duration_s)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("duration_s must be finite and non-negative")
        surface_temperature = float(surface_temperature_K)
        if not math.isfinite(surface_temperature) or surface_temperature <= 0.0:
            raise ValueError("surface_temperature_K must be finite and positive")
        if duration == 0.0:
            return frost_state
        if surface_temperature >= 273.15:
            # When the surface warms above the ice point, the same explicit
            # thermal resistance network supplies the melt power.  This is a
            # bounded state update: frost cannot become negative, and no
            # plant time series is used to infer a thaw rate.
            if frost_state.thickness_m == 0.0:
                return frost_state
            transfer = self.heat_transfer(boundary, surface_temperature, frost_state)
            melt_power = transfer.effective_UA_W_K * (surface_temperature - 273.15)
            melted_mass = max(melt_power, 0.0) * duration / ICE_FROST_LATENT_HEAT_J_KG
            removed_thickness = melted_mass / (
                frost_state.density_kg_m3 * self.external_area_m2
            )
            return FrostLayerState(
                max(0.0, frost_state.thickness_m - removed_thickness),
                frost_state.density_kg_m3,
            )
        transfer = self.heat_transfer(boundary, surface_temperature_K, frost_state)
        film_temperature = 0.5 * (boundary.temperature_K + surface_temperature_K)
        density, _, _, cp, _ = _air_properties(film_temperature, boundary.pressure_Pa)
        ambient_pressure = _saturation_vapor_pressure_Pa(
            boundary.temperature_K, over_ice=boundary.temperature_K < 273.15
        )
        surface_pressure = _saturation_vapor_pressure_Pa(surface_temperature_K, over_ice=True)
        ambient_vapor_density = (
            boundary.relative_humidity * ambient_pressure
            / (WATER_VAPOR_GAS_CONSTANT * boundary.temperature_K)
        )
        surface_vapor_density = (
            surface_pressure / (WATER_VAPOR_GAS_CONSTANT * surface_temperature_K)
        )
        # Lewis analogy with Le≈1: h_m = h/(rho*cp).
        mass_transfer_velocity = transfer.mixed_convection_W_m2K / (density * cp)
        deposition_flux = (
            self.frost_deposition_efficiency * mass_transfer_velocity
            * max(ambient_vapor_density - surface_vapor_density, 0.0)
        )
        added_thickness = deposition_flux * duration / frost_state.density_kg_m3
        return FrostLayerState(
            frost_state.thickness_m + added_thickness,
            frost_state.density_kg_m3,
        )

    def evaluate_boundary(
        self,
        inlet: ThermoState,
        outlet_pressure_Pa: float,
        mass_flow_kg_s: float,
        boundary: AmbientAirBoundary,
        frost_state: FrostLayerState | None = None,
    ) -> AmbientAirVaporizerResult:
        frost = frost_state or FrostLayerState(self.default_frost_thickness_m)
        surface_temperature = min(
            boundary.temperature_K - 1.0e-3,
            max(inlet.temperature_K, 180.0),
        )
        result: VaporizerResult | None = None
        transfer: AmbientVaporizerHeatTransfer | None = None
        for _ in range(8):
            transfer = self.heat_transfer(boundary, surface_temperature, frost)
            core = Vaporizer(
                transfer.effective_UA_W_K,
                self.maximum_heat,
                self.properties,
                self.integration_segments,
                self.minimum_approach_K,
                self.pressure_drop_relation,
                self.flow_ua_adjustment,
            )
            result = core.evaluate(
                inlet, outlet_pressure_Pa, mass_flow_kg_s, boundary.temperature_K
            )
            external_conductance = 1.0 / transfer.external_resistance_K_W
            estimated_surface = boundary.temperature_K - (
                result.heat_into_hydrogen_W / max(external_conductance, 1.0e-30)
            )
            estimated_surface = min(boundary.temperature_K, max(80.0, estimated_surface))
            if abs(estimated_surface - surface_temperature) < 0.02:
                surface_temperature = estimated_surface
                break
            surface_temperature = 0.55 * surface_temperature + 0.45 * estimated_surface
        assert result is not None and transfer is not None
        transfer = self.heat_transfer(boundary, surface_temperature, frost)
        result = Vaporizer(
            transfer.effective_UA_W_K,
            self.maximum_heat,
            self.properties,
            self.integration_segments,
            self.minimum_approach_K,
            self.pressure_drop_relation,
            self.flow_ua_adjustment,
        ).evaluate(inlet, outlet_pressure_Pa, mass_flow_kg_s, boundary.temperature_K)
        return AmbientAirVaporizerResult(
            result.outlet,
            result.heat_into_hydrogen_W,
            result.requested_heat_W,
            result.capacity_limited,
            surface_temperature,
            transfer,
            frost,
        )

    def evaluate(
        self,
        inlet: ThermoState,
        outlet_pressure_Pa: float,
        mass_flow_kg_s: float,
        ambient_temperature_K: float,
    ) -> AmbientAirVaporizerResult:
        return self.evaluate_boundary(
            inlet,
            outlet_pressure_Pa,
            mass_flow_kg_s,
            AmbientAirBoundary(
                ambient_temperature_K,
                self.default_relative_humidity,
                self.default_wind_speed_m_s,
            ),
            FrostLayerState(self.default_frost_thickness_m),
        )

    def outlet_pressure_from_inlet(
        self,
        inlet_pressure_Pa: float,
        mass_flow_kg_s: float,
    ) -> float:
        """Apply the explicit equipment pressure-loss relation."""

        return self.pressure_drop_relation.outlet_pressure_Pa(
            inlet_pressure_Pa, mass_flow_kg_s
        )

    def evaluate_from_inlet_pressure(
        self,
        inlet: ThermoState,
        inlet_pressure_Pa: float,
        mass_flow_kg_s: float,
        boundary: AmbientAirBoundary,
        frost_state: FrostLayerState | None = None,
    ) -> AmbientAirVaporizerResult:
        """Evaluate from inlet pressure using explicit flow pressure loss."""

        return self.evaluate_boundary(
            inlet,
            self.outlet_pressure_from_inlet(inlet_pressure_Pa, mass_flow_kg_s),
            mass_flow_kg_s,
            boundary,
            frost_state,
        )
