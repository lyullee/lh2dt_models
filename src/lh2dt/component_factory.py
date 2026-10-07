"""Explicit construction of reusable component models from asset records.

An asset with a ``null`` required parameter never reaches a constructor.  The
factory therefore makes missing engineering inputs visible instead of silently
inventing a value from an operating trace. Complete records can be used for
synthetic or literature-based assemblies through the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .catalog import REQUIRED_PARAMETERS, load_asset_catalog
from .ambient_vaporizer import AmbientAirVaporizer
from .compressor import Compressor
from .dynamic_pipe import DynamicHEMPipe
from .dynamic_reliquefier import DynamicReliquefier, ParallelDynamicReliquefier
from .dynamic_vaporizer import DynamicVaporizer, ParallelDynamicVaporizer
from .geometry import HorizontalVesselGeometry
from .natural_circulation import CirculationLegGeometry
from .pipe import HEMPipe, Pipe
from .properties import HydrogenProperties
from .pump import MappedPump, Pump, VirtualPump
from .radial_axial_transport import RadialAxialBoundaryGeometry
from .radial_axial_tank import RadialAxialTank, RadialAxialTankParameters
from .reliquefier import Reliquefier
from .stratified_tank import StratifiedTank, StratifiedTankParameters
from .tank import HomogeneousTank, TankGeometry
from .valve import (
    CheckValve,
    PressureReliefValve,
    StrokeLimitedCommandedValve,
    TemperatureProtectionValve,
    Valve,
)
from .vaporizer import FlowUAAdjustment, PressureDropRelation, Vaporizer
from .vent_stack import VentStack


@dataclass(frozen=True)
class AssetModelBuild:
    """One deterministic build attempt and its audit information."""

    asset_id: str
    component_model: str
    component: Any | None
    parameter_source: str
    missing_parameters: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def parameters_complete(self) -> bool:
        return not self.missing_parameters

    @property
    def ready(self) -> bool:
        return self.component is not None and not self.issues


@dataclass(frozen=True)
class DynamicReliquefierBankBuild:
    """Deterministic composition result for an explicit dynamic bank."""

    unit_ids: tuple[str, ...]
    unit_builds: tuple[AssetModelBuild, ...]
    component: ParallelDynamicReliquefier | None
    issues: tuple[str, ...]

    @property
    def parameters_complete(self) -> bool:
        return bool(self.unit_builds) and all(
            build.parameters_complete for build in self.unit_builds
        )

    @property
    def ready(self) -> bool:
        return self.component is not None and not self.issues


@dataclass(frozen=True)
class DynamicVaporizerBankBuild:
    """Deterministic composition result for an explicit dynamic bank."""

    unit_ids: tuple[str, ...]
    unit_builds: tuple[AssetModelBuild, ...]
    component: ParallelDynamicVaporizer | None
    issues: tuple[str, ...]

    @property
    def parameters_complete(self) -> bool:
        return bool(self.unit_builds) and all(
            build.parameters_complete for build in self.unit_builds
        )

    @property
    def ready(self) -> bool:
        return self.component is not None and not self.issues


def _missing_parameters(model: str, parameters: Mapping[str, Any]) -> tuple[str, ...]:
    required = REQUIRED_PARAMETERS.get(model)
    if required is None:
        return ()
    return tuple(name for name in required if parameters.get(name) is None)


def _parameter_value_issues(model: str, parameters: Mapping[str, Any]) -> tuple[str, ...]:
    """Reject non-finite or out-of-domain values before constructors run."""

    required = REQUIRED_PARAMETERS.get(model, ())
    values: dict[str, float] = {}
    issues: list[str] = []
    structured = {
        "liquid_boundaries", "vapor_boundaries",
        "liquid_interface_UA_W_K", "vapor_interface_UA_W_K",
        "vapor_interface_area_m2", "wall_patch_area_fractions",
        "resistance_terms_m2K_W",
        "head_curve_points", "efficiency_curve_points", "npsh_curve_points",
    }
    for name in required:
        if name in structured:
            continue
        value = parameters.get(name)
        if isinstance(value, bool):
            issues.append(f"parameter {name} must be numeric")
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            issues.append(f"parameter {name} must be numeric")
            continue
        if not math.isfinite(number):
            issues.append(f"parameter {name} must be finite")
            continue
        values[name] = number
    positive = {
        "homogeneous_tank": (
            "volume_m3", "wall_heat_capacity_J_K", "ambient_UA_W_K",
            "fluid_wall_UA_W_K", "inner_diameter_m",
        ),
        "stratified_tank": (
            "lower_wall_heat_capacity_J_K", "upper_wall_heat_capacity_J_K",
            "minimum_pressure_Pa", "maximum_pressure_Pa",
        ),
        "radial_axial_tank": (
            "total_volume_m3", "wall_heat_capacity_J_K",
            "minimum_pressure_Pa", "maximum_pressure_Pa",
        ),
        "vaporizer": ("UA_W_K", "maximum_heat_W"),
        "vaporizer_resistance_network": ("heat_transfer_area_m2", "maximum_heat_W"),
        "ambient_air_vaporizer": (
            "external_area_m2", "characteristic_length_m", "hydraulic_diameter_m",
            "internal_UA_W_K", "maximum_heat_W", "frost_conductivity_W_mK",
        ),
        "dynamic_vaporizer": (
            "fluid_volume_m3", "wall_heat_capacity_J_K",
            "ambient_temperature_K", "maximum_heat_W",
        ),
        "dynamic_reliquefier": (
            "fluid_volume_m3", "wall_heat_capacity_J_K",
            "cold_side_temperature_K", "maximum_cooling_W",
        ),
        "reliquefier": ("cooling_capacity_W",),
        "pump": ("pressure_rise_Pa",),
        "compressor": ("pressure_ratio",),
        "virtual_pump": ("shutoff_pressure_rise_Pa", "runout_mass_flow_kg_s"),
        "mapped_pump": ("reference_speed_rpm", "speed_rpm"),
        "valve": ("full_open_area_m2",),
        "stroke_limited_commanded_valve": (
            "full_open_area_m2", "stroke_time_s",
        ),
        "pressure_relief_valve": (
            "full_open_area_m2", "set_pressure_Pa", "reseat_pressure_Pa",
        ),
        "temperature_protection_valve": (
            "full_open_area_m2", "trip_temperature_K", "reset_temperature_K",
        ),
        "pipe": ("length_m", "inner_diameter_m"),
        "hem_pipe": ("length_m", "inner_diameter_m"),
        "dynamic_hem_pipe": (
            "length_m", "inner_diameter_m", "wall_heat_capacity_J_K",
            "ambient_temperature_K",
        ),
    }.get(model, ())
    for name in positive:
        if name in values and values[name] <= 0.0:
            issues.append(f"parameter {name} must be positive")
    if model == "vaporizer_resistance_network":
        terms = parameters.get("resistance_terms_m2K_W")
        if isinstance(terms, (str, bytes)) or not isinstance(terms, (list, tuple)) or not terms:
            issues.append("parameter resistance_terms_m2K_W must be a non-empty sequence")
        else:
            for index, term in enumerate(terms):
                if isinstance(term, bool):
                    issues.append(f"parameter resistance_terms_m2K_W[{index}] must be numeric")
                    continue
                try:
                    number = float(term)
                except (TypeError, ValueError):
                    issues.append(f"parameter resistance_terms_m2K_W[{index}] must be numeric")
                    continue
                if not math.isfinite(number) or number <= 0.0:
                    issues.append(
                        f"parameter resistance_terms_m2K_W[{index}] must be finite and positive"
                    )
    if model == "ambient_air_vaporizer":
        for name in ("wall_resistance_K_W", "default_wind_speed_m_s", "default_frost_thickness_m"):
            if name in values and values[name] < 0.0:
                issues.append(f"parameter {name} must be non-negative")
        for name in ("emissivity", "frost_deposition_efficiency", "default_relative_humidity"):
            if name in values and not 0.0 <= values[name] <= 1.0:
                issues.append(f"parameter {name} must lie in [0, 1]")
    if model == "homogeneous_tank":
        for name in ("straight_length_m", "head_depth_m"):
            if name in values and values[name] < 0.0:
                issues.append(f"parameter {name} must be non-negative")
        if values.get("straight_length_m", 0.0) == 0.0 and values.get("head_depth_m", 0.0) == 0.0:
            issues.append("parameters straight_length_m and head_depth_m cannot both be zero")
    if model == "radial_axial_tank":
        for name in ("liquid_level_count", "vapor_level_count"):
            value = parameters.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                issues.append(f"parameter {name} must be an integer of at least two")
        for name in ("liquid_wall_volume_fraction", "vapor_wall_volume_fraction"):
            if name in values and not 0.0 < values[name] < 1.0:
                issues.append(f"parameter {name} must lie strictly between zero and one")
        for name in (
            "wall_axial_conductance_W_K", "liquid_radial_conductance_W_K",
            "vapor_radial_conductance_W_K", "liquid_axial_conductance_W_K",
            "vapor_axial_conductance_W_K", "ambient_UA_W_K",
        ):
            if name in values and values[name] < 0.0:
                issues.append(f"parameter {name} must be non-negative")
    if model in {"valve", "stroke_limited_commanded_valve"} and "discharge_coefficient" in values and not 0.0 < values["discharge_coefficient"] <= 1.0:
        issues.append("parameter discharge_coefficient must lie in (0, 1]")
    if model in {
        "valve", "stroke_limited_commanded_valve",
        "pressure_relief_valve", "temperature_protection_valve",
    } and "closed_leakage_CdA_m2" in values:
        leakage = values["closed_leakage_CdA_m2"]
        if leakage < 0.0:
            issues.append("parameter closed_leakage_CdA_m2 must be non-negative")
        elif (
            "full_open_area_m2" in values
            and "discharge_coefficient" in values
            and leakage > values["full_open_area_m2"] * values["discharge_coefficient"]
        ):
            issues.append(
                "parameter closed_leakage_CdA_m2 must not exceed full-open CdA"
            )
    if model == "stroke_limited_commanded_valve":
        if "stroke_time_s" in values and values["stroke_time_s"] <= 0.0:
            issues.append("parameter stroke_time_s must be positive")
        if "initial_opening" in parameters:
            try:
                initial_opening = float(parameters["initial_opening"])
            except (TypeError, ValueError):
                issues.append("parameter initial_opening must be numeric")
            else:
                if not math.isfinite(initial_opening) or not 0.0 <= initial_opening <= 1.0:
                    issues.append("parameter initial_opening must be finite and in [0, 1]")
    if model == "check_valve":
        if "discharge_coefficient" in values and not 0.0 < values["discharge_coefficient"] <= 1.0:
            issues.append("parameter discharge_coefficient must lie in (0, 1]")
        if "cracking_pressure_Pa" in values and values["cracking_pressure_Pa"] < 0.0:
            issues.append("parameter cracking_pressure_Pa must be non-negative")
    if model == "pressure_relief_valve":
        if "reseat_pressure_Pa" in values and "set_pressure_Pa" in values:
            if values["reseat_pressure_Pa"] > values["set_pressure_Pa"]:
                issues.append("parameter reseat_pressure_Pa must not exceed set_pressure_Pa")
        if "discharge_coefficient" in values and not 0.0 < values["discharge_coefficient"] <= 1.0:
            issues.append("parameter discharge_coefficient must lie in (0, 1]")
    if model == "temperature_protection_valve":
        if "reset_temperature_K" in values and "trip_temperature_K" in values:
            if values["reset_temperature_K"] > values["trip_temperature_K"]:
                issues.append("parameter reset_temperature_K must not exceed trip_temperature_K")
        if "discharge_coefficient" in values and not 0.0 < values["discharge_coefficient"] <= 1.0:
            issues.append("parameter discharge_coefficient must lie in (0, 1]")
    if model in {"pump", "virtual_pump"} and "isentropic_efficiency" in values and not 0.0 < values["isentropic_efficiency"] <= 1.0:
        issues.append("parameter isentropic_efficiency must lie in (0, 1]")
    for name in ("roughness_m", "local_loss_coefficient", "NPSHr_m"):
        if name in values and values[name] < 0.0:
            issues.append(f"parameter {name} must be non-negative")
    if model == "dynamic_hem_pipe":
        for name in ("fluid_wall_UA_W_K", "ambient_UA_W_K"):
            if name in values and values[name] < 0.0:
                issues.append(f"parameter {name} must be non-negative")
        if "fluid_volume_m3" in parameters and parameters["fluid_volume_m3"] is not None:
            value = parameters["fluid_volume_m3"]
            if isinstance(value, bool):
                issues.append("parameter fluid_volume_m3 must be numeric")
            else:
                try:
                    volume = float(value)
                except (TypeError, ValueError):
                    issues.append("parameter fluid_volume_m3 must be numeric")
                else:
                    if not math.isfinite(volume) or volume <= 0.0:
                        issues.append("parameter fluid_volume_m3 must be finite and positive")
    if model == "dynamic_vaporizer":
        for name in ("fluid_wall_UA_W_K", "ambient_UA_W_K"):
            if name in values and values[name] < 0.0:
                issues.append(f"parameter {name} must be non-negative")
    if model == "dynamic_reliquefier":
        for name in ("fluid_wall_UA_W_K", "cold_side_UA_W_K"):
            if name in values and values[name] < 0.0:
                issues.append(f"parameter {name} must be non-negative")
    if model == "vent_stack":
        if "ambient_temperature_K" in values and values["ambient_temperature_K"] <= 0.0:
            issues.append("parameter ambient_temperature_K must be positive")
        if "ambient_UA_W_K" in values and values["ambient_UA_W_K"] < 0.0:
            issues.append("parameter ambient_UA_W_K must be non-negative")
    return tuple(issues)


def build_asset_model(
    asset: Mapping[str, Any],
    properties: HydrogenProperties | None = None,
    *,
    model_override: str | None = None,
) -> AssetModelBuild:
    """Build one model only when the asset record is complete.

    ``asset`` uses the fields listed in ``catalog.REQUIRED_PARAMETERS``.  The returned
    object can be a dynamic tank, a hydraulic two-port component, or a thermal
    control-volume component; topology binding remains the responsibility of
    ``assembly.py``.

    ``model_override`` is an explicit model substitution for a complete input
    record. It is never inferred from ``target_component_model`` or another
    catalog field. This keeps baseline and promoted models interchangeable
    while preventing silent construction from incomplete asset records.
    """

    asset_id = str(asset.get("asset_id", ""))
    model = str(model_override).strip() if model_override is not None else str(asset.get("component_model", ""))
    parameters = asset.get("model_parameters", {})
    if not isinstance(parameters, Mapping):
        parameters = {}
    source = str(asset.get("source", "")).strip()
    issues: list[str] = []
    if not asset_id:
        issues.append("asset_id is required")
    if model_override is not None and not str(model_override).strip():
        issues.append("model_override must be non-empty when provided")
    if model not in REQUIRED_PARAMETERS:
        issues.append(f"unknown component model: {model or '<empty>'}")
    if not source:
        issues.append("parameter source is required")
    missing = _missing_parameters(model, parameters)
    if missing:
        return AssetModelBuild(
            asset_id=asset_id,
            component_model=model,
            component=None,
            parameter_source=source,
            missing_parameters=missing,
            issues=tuple(issues),
        )
    if issues:
        return AssetModelBuild(
            asset_id=asset_id,
            component_model=model,
            component=None,
            parameter_source=source,
            missing_parameters=missing,
            issues=tuple(issues),
        )
    value_issues = _parameter_value_issues(model, parameters)
    if value_issues:
        return AssetModelBuild(
            asset_id=asset_id,
            component_model=model,
            component=None,
            parameter_source=source,
            missing_parameters=missing,
            issues=value_issues,
        )

    props = properties or HydrogenProperties()
    try:
        def radial_leg(raw: Any, context: str) -> CirculationLegGeometry:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{context} must be an object")
            return CirculationLegGeometry(
                length_m=float(raw["length_m"]),
                hydraulic_diameter_m=float(raw["hydraulic_diameter_m"]),
                flow_area_m2=float(raw["flow_area_m2"]),
                roughness_m=float(raw.get("roughness_m", 0.0)),
                local_loss_coefficient=float(raw.get("local_loss_coefficient", 0.0)),
            )

        def radial_boundary(raw: Any, index: int, phase: str) -> RadialAxialBoundaryGeometry:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{phase}_boundaries[{index}] must be an object")
            return RadialAxialBoundaryGeometry(
                vertical_separation_m=float(raw["vertical_separation_m"]),
                wall_leg=radial_leg(raw["wall_leg"], f"{phase}_boundaries[{index}].wall_leg"),
                core_leg=radial_leg(raw["core_leg"], f"{phase}_boundaries[{index}].core_leg"),
            )

        if model == "homogeneous_tank":
            component = HomogeneousTank(
                TankGeometry(
                    volume_m3=float(parameters["volume_m3"]),
                    wall_heat_capacity_J_K=float(parameters["wall_heat_capacity_J_K"]),
                    ambient_UA_W_K=float(parameters["ambient_UA_W_K"]),
                    fluid_wall_UA_W_K=float(parameters["fluid_wall_UA_W_K"]),
                ),
                props,
            )
        elif model == "stratified_tank":
            component = StratifiedTank(
                HorizontalVesselGeometry(
                    inner_diameter_m=float(parameters["inner_diameter_m"]),
                    straight_length_m=float(parameters["straight_length_m"]),
                    head_depth_m=float(parameters["head_depth_m"]),
                ),
                StratifiedTankParameters(
                    lower_wall_heat_capacity_J_K=float(parameters["lower_wall_heat_capacity_J_K"]),
                    upper_wall_heat_capacity_J_K=float(parameters["upper_wall_heat_capacity_J_K"]),
                    lower_ambient_UA_W_K=float(parameters["lower_ambient_UA_W_K"]),
                    upper_ambient_UA_W_K=float(parameters["upper_ambient_UA_W_K"]),
                    wall_liquid_U_W_m2K=float(parameters["wall_liquid_U_W_m2K"]),
                    wall_vapor_U_W_m2K=float(parameters["wall_vapor_U_W_m2K"]),
                    liquid_interface_UA_W_K=float(parameters["liquid_interface_UA_W_K"]),
                    vapor_interface_UA_W_K=float(parameters["vapor_interface_UA_W_K"]),
                    wall_axial_UA_W_K=float(parameters["wall_axial_UA_W_K"]),
                    wall_zone_split_height_m=float(parameters["wall_zone_split_height_m"]),
                    minimum_pressure_Pa=float(parameters["minimum_pressure_Pa"]),
                    maximum_pressure_Pa=float(parameters["maximum_pressure_Pa"]),
                ),
                props,
            )
        elif model == "radial_axial_tank":
            liquid_boundaries = tuple(
                radial_boundary(raw, index, "liquid")
                for index, raw in enumerate(parameters["liquid_boundaries"])
            )
            vapor_boundaries = tuple(
                radial_boundary(raw, index, "vapor")
                for index, raw in enumerate(parameters["vapor_boundaries"])
            )
            optional_tuple = lambda name: (
                None
                if parameters.get(name) is None
                else tuple(float(value) for value in parameters[name])
            )
            component = RadialAxialTank(
                RadialAxialTankParameters(
                    total_volume_m3=float(parameters["total_volume_m3"]),
                    liquid_level_count=int(parameters["liquid_level_count"]),
                    vapor_level_count=int(parameters["vapor_level_count"]),
                    liquid_wall_volume_fraction=float(parameters["liquid_wall_volume_fraction"]),
                    vapor_wall_volume_fraction=float(parameters["vapor_wall_volume_fraction"]),
                    liquid_boundaries=liquid_boundaries,
                    vapor_boundaries=vapor_boundaries,
                    wall_heat_capacity_J_K=float(parameters["wall_heat_capacity_J_K"]),
                    ambient_UA_W_K=float(parameters["ambient_UA_W_K"]),
                    wall_to_fluid_UA_W_K=float(parameters["wall_to_fluid_UA_W_K"]),
                    wall_axial_conductance_W_K=float(parameters["wall_axial_conductance_W_K"]),
                    liquid_radial_conductance_W_K=float(parameters["liquid_radial_conductance_W_K"]),
                    vapor_radial_conductance_W_K=float(parameters["vapor_radial_conductance_W_K"]),
                    liquid_axial_conductance_W_K=float(parameters["liquid_axial_conductance_W_K"]),
                    vapor_axial_conductance_W_K=float(parameters["vapor_axial_conductance_W_K"]),
                    liquid_interface_UA_W_K=tuple(float(value) for value in parameters["liquid_interface_UA_W_K"]),
                    vapor_interface_UA_W_K=tuple(float(value) for value in parameters["vapor_interface_UA_W_K"]),
                    vapor_interface_heat_transfer_model=str(parameters.get("vapor_interface_heat_transfer_model", "conductance")),
                    vapor_interface_area_m2=optional_tuple("vapor_interface_area_m2"),
                    gas_interface_energy_accommodation_coefficient=float(parameters.get("gas_interface_energy_accommodation_coefficient", 1.0)),
                    distributed_boiling_model=str(parameters.get("distributed_boiling_model", "off")),
                    wall_patch_area_fractions=tuple(float(value) for value in parameters.get("wall_patch_area_fractions", ())),
                    gravity_m_s2=float(parameters.get("gravity_m_s2", 9.80665)),
                    minimum_pressure_Pa=float(parameters["minimum_pressure_Pa"]),
                    maximum_pressure_Pa=float(parameters["maximum_pressure_Pa"]),
                ),
                props,
            )
        elif model == "vaporizer":
            pressure_drop = parameters.get("pressure_drop_relation")
            pressure_drop_relation = (
                None
                if pressure_drop is None
                else PressureDropRelation(**pressure_drop)
            )
            flow_ua = parameters.get("flow_ua_adjustment")
            flow_ua_adjustment = (
                None if flow_ua is None else FlowUAAdjustment(**flow_ua)
            )
            component = Vaporizer(
                float(parameters["UA_W_K"]),
                float(parameters["maximum_heat_W"]),
                props,
                integration_segments=int(parameters.get("integration_segments", 192)),
                minimum_approach_K=float(parameters.get("minimum_approach_K", 0.0)),
                pressure_drop_relation=pressure_drop_relation,
                flow_ua_adjustment=flow_ua_adjustment,
            )
        elif model == "vaporizer_resistance_network":
            pressure_drop = parameters.get("pressure_drop_relation")
            pressure_drop_relation = (
                None
                if pressure_drop is None
                else PressureDropRelation(**pressure_drop)
            )
            flow_ua = parameters.get("flow_ua_adjustment")
            flow_ua_adjustment = (
                None if flow_ua is None else FlowUAAdjustment(**flow_ua)
            )
            component = Vaporizer.from_resistance_network(
                area_m2=float(parameters["heat_transfer_area_m2"]),
                resistance_terms_m2K_W=parameters["resistance_terms_m2K_W"],
                maximum_heat_W=float(parameters["maximum_heat_W"]),
                properties=props,
                integration_segments=int(parameters.get("integration_segments", 192)),
                minimum_approach_K=float(parameters.get("minimum_approach_K", 0.0)),
                pressure_drop_relation=pressure_drop_relation,
                flow_ua_adjustment=flow_ua_adjustment,
            )
        elif model == "ambient_air_vaporizer":
            pressure_drop = parameters.get("pressure_drop_relation")
            pressure_drop_relation = (
                None
                if pressure_drop is None
                else PressureDropRelation(**pressure_drop)
            )
            flow_ua = parameters.get("flow_ua_adjustment")
            flow_ua_adjustment = (
                None if flow_ua is None else FlowUAAdjustment(**flow_ua)
            )
            component = AmbientAirVaporizer(
                external_area_m2=float(parameters["external_area_m2"]),
                characteristic_length_m=float(parameters["characteristic_length_m"]),
                hydraulic_diameter_m=float(parameters["hydraulic_diameter_m"]),
                internal_UA_W_K=float(parameters["internal_UA_W_K"]),
                wall_resistance_K_W=float(parameters["wall_resistance_K_W"]),
                emissivity=float(parameters["emissivity"]),
                maximum_heat_W=float(parameters["maximum_heat_W"]),
                frost_conductivity_W_mK=float(parameters["frost_conductivity_W_mK"]),
                frost_deposition_efficiency=float(parameters.get("frost_deposition_efficiency", 1.0)),
                default_relative_humidity=float(parameters.get("default_relative_humidity", 0.60)),
                default_wind_speed_m_s=float(parameters.get("default_wind_speed_m_s", 0.0)),
                default_frost_thickness_m=float(parameters.get("default_frost_thickness_m", 0.0)),
                minimum_approach_K=float(parameters.get("minimum_approach_K", 0.0)),
                pressure_drop_relation=pressure_drop_relation,
                flow_ua_adjustment=flow_ua_adjustment,
                properties=props,
                integration_segments=int(parameters.get("integration_segments", 96)),
            )
        elif model == "dynamic_vaporizer":
            component = DynamicVaporizer(
                fluid_volume_m3=float(parameters["fluid_volume_m3"]),
                wall_heat_capacity_J_K=float(parameters["wall_heat_capacity_J_K"]),
                fluid_wall_UA_W_K=float(parameters["fluid_wall_UA_W_K"]),
                ambient_temperature_K=float(parameters["ambient_temperature_K"]),
                ambient_UA_W_K=float(parameters["ambient_UA_W_K"]),
                maximum_heat_W=float(parameters["maximum_heat_W"]),
                properties=props,
            )
        elif model == "dynamic_reliquefier":
            component = DynamicReliquefier(
                fluid_volume_m3=float(parameters["fluid_volume_m3"]),
                wall_heat_capacity_J_K=float(parameters["wall_heat_capacity_J_K"]),
                fluid_wall_UA_W_K=float(parameters["fluid_wall_UA_W_K"]),
                cold_side_temperature_K=float(parameters["cold_side_temperature_K"]),
                cold_side_UA_W_K=float(parameters["cold_side_UA_W_K"]),
                maximum_cooling_W=float(parameters["maximum_cooling_W"]),
                properties=props,
            )
        elif model == "reliquefier":
            component = Reliquefier(float(parameters["cooling_capacity_W"]), props)
        elif model == "valve":
            component = Valve(
                float(parameters["full_open_area_m2"]),
                float(parameters["discharge_coefficient"]),
                props,
                closed_leakage_CdA_m2=float(
                    parameters.get("closed_leakage_CdA_m2", 0.0)
                ),
            )
        elif model == "stroke_limited_commanded_valve":
            component = StrokeLimitedCommandedValve(
                Valve(
                    float(parameters["full_open_area_m2"]),
                    float(parameters["discharge_coefficient"]),
                    props,
                    closed_leakage_CdA_m2=float(
                        parameters.get("closed_leakage_CdA_m2", 0.0)
                    ),
                ),
                stroke_time_s=float(parameters["stroke_time_s"]),
                initial_opening=float(parameters.get("initial_opening", 0.0)),
            )
        elif model == "check_valve":
            component = CheckValve(
                float(parameters["full_open_area_m2"]),
                float(parameters["discharge_coefficient"]),
                cracking_pressure_Pa=float(parameters["cracking_pressure_Pa"]),
                properties=props,
            )
        elif model == "pressure_relief_valve":
            component = PressureReliefValve(
                Valve(
                    float(parameters["full_open_area_m2"]),
                    float(parameters["discharge_coefficient"]),
                    props,
                    allow_reverse=False,
                    closed_leakage_CdA_m2=float(
                        parameters.get("closed_leakage_CdA_m2", 0.0)
                    ),
                ),
                set_pressure_Pa=float(parameters["set_pressure_Pa"]),
                reseat_pressure_Pa=float(parameters["reseat_pressure_Pa"]),
            )
        elif model == "temperature_protection_valve":
            component = TemperatureProtectionValve(
                Valve(
                    float(parameters["full_open_area_m2"]),
                    float(parameters["discharge_coefficient"]),
                    props,
                    allow_reverse=False,
                    closed_leakage_CdA_m2=float(
                        parameters.get("closed_leakage_CdA_m2", 0.0)
                    ),
                ),
                trip_temperature_K=float(parameters["trip_temperature_K"]),
                reset_temperature_K=float(parameters["reset_temperature_K"]),
            )
        elif model in {"pipe", "hem_pipe"}:
            pipe_type = HEMPipe if model == "hem_pipe" else Pipe
            component = pipe_type(
                length_m=float(parameters["length_m"]),
                inner_diameter_m=float(parameters["inner_diameter_m"]),
                roughness_m=float(parameters["roughness_m"]),
                local_loss_coefficient=float(parameters["local_loss_coefficient"]),
                heat_into_fluid_W=float(parameters["heat_into_fluid_W"]),
                elevation_change_m=float(parameters["elevation_change_m"]),
                properties=props,
            )
        elif model == "dynamic_hem_pipe":
            dynamic_kwargs: dict[str, float] = {}
            if parameters.get("fluid_volume_m3") is not None:
                dynamic_kwargs["fluid_volume_m3"] = float(parameters["fluid_volume_m3"])
            component = DynamicHEMPipe(
                length_m=float(parameters["length_m"]),
                inner_diameter_m=float(parameters["inner_diameter_m"]),
                roughness_m=float(parameters["roughness_m"]),
                wall_heat_capacity_J_K=float(parameters["wall_heat_capacity_J_K"]),
                local_loss_coefficient=float(parameters["local_loss_coefficient"]),
                fluid_wall_UA_W_K=float(parameters["fluid_wall_UA_W_K"]),
                ambient_temperature_K=float(parameters["ambient_temperature_K"]),
                ambient_UA_W_K=float(parameters["ambient_UA_W_K"]),
                properties=props,
                **dynamic_kwargs,
            )
        elif model == "vent_stack":
            component = VentStack(
                length_m=float(parameters["length_m"]),
                inner_diameter_m=float(parameters["inner_diameter_m"]),
                roughness_m=float(parameters["roughness_m"]),
                local_loss_coefficient=float(parameters["local_loss_coefficient"]),
                elevation_change_m=float(parameters["elevation_change_m"]),
                ambient_temperature_K=float(parameters["ambient_temperature_K"]),
                ambient_UA_W_K=float(parameters["ambient_UA_W_K"]),
                properties=props,
            )
        elif model == "virtual_pump":
            speed_kwargs: dict[str, float] = {}
            speed_keys = {"reference_speed_rpm", "speed_rpm"}
            supplied_speed_keys = speed_keys & set(parameters)
            if supplied_speed_keys and supplied_speed_keys != speed_keys:
                raise ValueError(
                    "parameters reference_speed_rpm and speed_rpm must be supplied together"
                )
            elif supplied_speed_keys == speed_keys:
                speed_kwargs = {
                    "reference_speed_rpm": float(parameters["reference_speed_rpm"]),
                    "speed_rpm": float(parameters["speed_rpm"]),
                }
            component = VirtualPump(
                float(parameters["shutoff_pressure_rise_Pa"]),
                float(parameters["runout_mass_flow_kg_s"]),
                float(parameters["isentropic_efficiency"]),
                props,
                npsh_required_m=float(parameters["NPSHr_m"]),
                **speed_kwargs,
            )
        elif model == "pump":
            # Pump is an explicit energy-balance object rather than a network
            # two-port.  Keep a complete asset operating point on the object;
            # callers may still override it explicitly at evaluation time.
            component = Pump(
                props,
                pressure_rise_Pa=float(parameters["pressure_rise_Pa"]),
                isentropic_efficiency=float(parameters["isentropic_efficiency"]),
            )
        elif model == "compressor":
            component = Compressor(
                props,
                pressure_ratio=float(parameters["pressure_ratio"]),
                isentropic_efficiency=float(parameters["isentropic_efficiency"]),
            )
        elif model == "mapped_pump":
            efficiency_points = parameters.get("efficiency_curve_points")
            npsh_points = parameters.get("npsh_curve_points")
            component = MappedPump(
                parameters["head_curve_points"],
                reference_speed_rpm=float(parameters["reference_speed_rpm"]),
                speed_rpm=float(parameters["speed_rpm"]),
                isentropic_efficiency=(
                    None if parameters.get("isentropic_efficiency") is None
                    else float(parameters["isentropic_efficiency"])
                ),
                efficiency_curve_points=efficiency_points,
                npsh_required_m=(
                    None if parameters.get("NPSHr_m") is None
                    else float(parameters["NPSHr_m"])
                ),
                npsh_curve_points=npsh_points,
                properties=props,
            )
        else:  # pragma: no cover - guarded by REQUIRED_PARAMETERS above
            raise ValueError(f"unsupported component model: {model}")
    except (TypeError, ValueError, KeyError) as exc:
        issues.append(f"constructor rejected parameters: {exc}")
        component = None
    return AssetModelBuild(
        asset_id=asset_id,
        component_model=model,
        component=component,
        parameter_source=source,
        missing_parameters=missing,
        issues=tuple(issues),
    )


def build_dynamic_reliquefier_bank(
    assets: Sequence[Mapping[str, Any]],
    properties: HydrogenProperties | None = None,
    *,
    unit_ids: Sequence[str] | None = None,
    active: Sequence[bool] | None = None,
    flow_weights: Sequence[float] | None = None,
) -> DynamicReliquefierBankBuild:
    """Build a multi-unit dynamic bank from complete explicit asset records.

    This is a composition helper rather than a new physical model.  Each
    record must independently pass :func:`build_asset_model` as a
    ``dynamic_reliquefier``.  No capacity label, unit count, or operating trace
    history is used to create or activate a unit.
    """

    if isinstance(assets, (str, bytes)):
        raise TypeError("assets must be a sequence of asset records")
    records = tuple(assets)
    issues: list[str] = []
    if not records:
        issues.append("at least one dynamic reliquefier asset is required")
    if unit_ids is None:
        ids = tuple(str(record.get("asset_id", "")).strip() for record in records)
    else:
        ids = tuple(str(item).strip() for item in unit_ids)
        if len(ids) != len(records):
            issues.append("unit_ids must match assets")
    if any(not item for item in ids):
        issues.append("dynamic reliquefier unit IDs must be non-empty")
    if len(set(ids)) != len(ids):
        issues.append("dynamic reliquefier unit IDs must be unique")

    builds: list[AssetModelBuild] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            issues.append(f"dynamic reliquefier asset {index} must be an object")
            continue
        if str(record.get("component_model", "")) != "dynamic_reliquefier":
            issues.append(
                f"dynamic reliquefier asset {record.get('asset_id', index)} "
                "must declare component_model=dynamic_reliquefier"
            )
        build = build_asset_model(record, properties)
        builds.append(build)
        if not build.ready:
            issues.extend(
                f"{build.asset_id or index}: {issue}"
                for issue in (*build.missing_parameters, *build.issues)
            )

    component: ParallelDynamicReliquefier | None = None
    if not issues and len(builds) == len(records):
        try:
            component = ParallelDynamicReliquefier(
                tuple(build.component for build in builds),  # type: ignore[arg-type]
                unit_ids=ids,
                active=active,
                flow_weights=flow_weights,
            )
        except (TypeError, ValueError) as exc:
            issues.append(f"dynamic reliquefier bank rejected composition: {exc}")
    return DynamicReliquefierBankBuild(
        unit_ids=ids,
        unit_builds=tuple(builds),
        component=component,
        issues=tuple(issues),
    )


def build_dynamic_vaporizer_bank(
    assets: Sequence[Mapping[str, Any]],
    properties: HydrogenProperties | None = None,
    *,
    unit_ids: Sequence[str] | None = None,
    active: Sequence[bool] | None = None,
    flow_weights: Sequence[float] | None = None,
) -> DynamicVaporizerBankBuild:
    """Build a multi-unit dynamic vaporizer bank from explicit asset records.

    Every record must independently pass :func:`build_asset_model` as a
    ``dynamic_vaporizer``.  The helper never infers unit count, capacity split,
    or activation from observations.
    """

    if isinstance(assets, (str, bytes)):
        raise TypeError("assets must be a sequence of asset records")
    records = tuple(assets)
    issues: list[str] = []
    if not records:
        issues.append("at least one dynamic vaporizer asset is required")
    if unit_ids is None:
        ids = tuple(str(record.get("asset_id", "")).strip() for record in records)
    else:
        ids = tuple(str(item).strip() for item in unit_ids)
        if len(ids) != len(records):
            issues.append("unit_ids must match assets")
    if any(not item for item in ids):
        issues.append("dynamic vaporizer unit IDs must be non-empty")
    if len(set(ids)) != len(ids):
        issues.append("dynamic vaporizer unit IDs must be unique")

    builds: list[AssetModelBuild] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            issues.append(f"dynamic vaporizer asset {index} must be an object")
            continue
        if str(record.get("component_model", "")) != "dynamic_vaporizer":
            issues.append(
                f"dynamic vaporizer asset {record.get('asset_id', index)} "
                "must declare component_model=dynamic_vaporizer"
            )
        build = build_asset_model(record, properties)
        builds.append(build)
        if not build.ready:
            issues.extend(
                f"{build.asset_id or index}: {issue}"
                for issue in (*build.missing_parameters, *build.issues)
            )

    component: ParallelDynamicVaporizer | None = None
    if not issues and len(builds) == len(records):
        try:
            component = ParallelDynamicVaporizer(
                tuple(build.component for build in builds),  # type: ignore[arg-type]
                unit_ids=ids,
                active=active,
                flow_weights=flow_weights,
            )
        except (TypeError, ValueError) as exc:
            issues.append(f"dynamic vaporizer bank rejected composition: {exc}")
    return DynamicVaporizerBankBuild(
        unit_ids=ids,
        unit_builds=tuple(builds),
        component=component,
        issues=tuple(issues),
    )


def build_catalog_models(
    catalog: Mapping[str, Any],
    properties: HydrogenProperties | None = None,
    *,
    model_overrides: Mapping[str, str] | None = None,
) -> tuple[AssetModelBuild, ...]:
    """Build every catalog asset without mutating the input catalog.

    ``model_overrides`` is keyed by asset ID and is an explicit, caller-owned
    substitution map. Unknown asset IDs are rejected so a misspelled
    promotion request cannot silently do nothing.
    """

    assets = catalog.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("Asset catalog assets must be a list")
    overrides = dict(model_overrides or {})
    asset_ids = {
        str(asset.get("asset_id", ""))
        for asset in assets
        if isinstance(asset, Mapping)
    }
    unknown_overrides = sorted(set(overrides) - asset_ids)
    if unknown_overrides:
        raise KeyError(
            "model_overrides contain unknown asset IDs: "
            + ", ".join(unknown_overrides)
        )
    return tuple(
        build_asset_model(
            asset,
            properties,
            model_override=overrides.get(str(asset.get("asset_id", ""))),
        )
        for asset in assets
    )


def load_and_build_catalog(
    path: str,
    properties: HydrogenProperties | None = None,
    *,
    model_overrides: Mapping[str, str] | None = None,
) -> tuple[AssetModelBuild, ...]:
    """Load a policy-checked catalog and return explicit build attempts."""

    return build_catalog_models(
        load_asset_catalog(path), properties, model_overrides=model_overrides
    )
