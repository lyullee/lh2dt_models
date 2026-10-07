"""Reusable component parameter registry and readiness helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


REQUIRED_PARAMETERS = {
    "homogeneous_tank": (
        "volume_m3", "wall_heat_capacity_J_K", "ambient_UA_W_K", "fluid_wall_UA_W_K",
        "inner_diameter_m", "straight_length_m", "head_depth_m",
    ),
    "stratified_tank": (
        "inner_diameter_m", "straight_length_m", "head_depth_m",
        "lower_wall_heat_capacity_J_K", "upper_wall_heat_capacity_J_K",
        "lower_ambient_UA_W_K", "upper_ambient_UA_W_K",
        "wall_liquid_U_W_m2K", "wall_vapor_U_W_m2K",
        "liquid_interface_UA_W_K", "vapor_interface_UA_W_K",
        "wall_axial_UA_W_K", "wall_zone_split_height_m",
        "minimum_pressure_Pa", "maximum_pressure_Pa",
    ),
    "radial_axial_tank": (
        "total_volume_m3", "liquid_level_count", "vapor_level_count",
        "liquid_wall_volume_fraction", "vapor_wall_volume_fraction",
        "liquid_boundaries", "vapor_boundaries", "wall_heat_capacity_J_K",
        "ambient_UA_W_K", "wall_to_fluid_UA_W_K", "wall_axial_conductance_W_K",
        "liquid_radial_conductance_W_K", "vapor_radial_conductance_W_K",
        "liquid_axial_conductance_W_K", "vapor_axial_conductance_W_K",
        "liquid_interface_UA_W_K", "vapor_interface_UA_W_K",
        "minimum_pressure_Pa", "maximum_pressure_Pa",
    ),
    "vaporizer": ("UA_W_K", "maximum_heat_W"),
    "vaporizer_resistance_network": (
        "heat_transfer_area_m2", "resistance_terms_m2K_W", "maximum_heat_W",
    ),
    "ambient_air_vaporizer": (
        "external_area_m2", "characteristic_length_m", "hydraulic_diameter_m",
        "internal_UA_W_K", "wall_resistance_K_W", "emissivity",
        "maximum_heat_W", "frost_conductivity_W_mK",
    ),
    "dynamic_vaporizer": (
        "fluid_volume_m3", "wall_heat_capacity_J_K", "fluid_wall_UA_W_K",
        "ambient_temperature_K", "ambient_UA_W_K", "maximum_heat_W",
    ),
    "dynamic_reliquefier": (
        "fluid_volume_m3", "wall_heat_capacity_J_K", "fluid_wall_UA_W_K",
        "cold_side_temperature_K", "cold_side_UA_W_K", "maximum_cooling_W",
    ),
    "reliquefier": ("cooling_capacity_W",),
    "pump": ("pressure_rise_Pa", "isentropic_efficiency"),
    "compressor": ("pressure_ratio", "isentropic_efficiency"),
    "virtual_pump": (
        "shutoff_pressure_rise_Pa", "runout_mass_flow_kg_s",
        "isentropic_efficiency", "NPSHr_m",
    ),
    "mapped_pump": (
        "head_curve_points", "reference_speed_rpm", "speed_rpm",
    ),
    "valve": ("full_open_area_m2", "discharge_coefficient"),
    "stroke_limited_commanded_valve": (
        "full_open_area_m2", "discharge_coefficient", "stroke_time_s",
    ),
    "check_valve": (
        "full_open_area_m2", "discharge_coefficient", "cracking_pressure_Pa",
    ),
    "pressure_relief_valve": (
        "full_open_area_m2", "discharge_coefficient",
        "set_pressure_Pa", "reseat_pressure_Pa",
    ),
    "temperature_protection_valve": (
        "full_open_area_m2", "discharge_coefficient",
        "trip_temperature_K", "reset_temperature_K",
    ),
    "pipe": ("length_m", "inner_diameter_m", "roughness_m", "local_loss_coefficient", "heat_into_fluid_W", "elevation_change_m"),
    "hem_pipe": ("length_m", "inner_diameter_m", "roughness_m", "local_loss_coefficient", "heat_into_fluid_W", "elevation_change_m"),
    "dynamic_hem_pipe": (
        "length_m", "inner_diameter_m", "roughness_m", "wall_heat_capacity_J_K",
        "local_loss_coefficient", "fluid_wall_UA_W_K", "ambient_temperature_K",
        "ambient_UA_W_K",
    ),
    "vent_stack": (
        "length_m", "inner_diameter_m", "roughness_m", "local_loss_coefficient",
        "elevation_change_m", "ambient_temperature_K", "ambient_UA_W_K",
    ),
}


@dataclass(frozen=True)
class AssetReadiness:
    asset_id: str
    component_model: str
    parameters_complete: bool
    validation_status: str
    ready: bool
    missing_parameters: tuple[str, ...]
    status: str


def load_asset_catalog(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "0.1":
        raise ValueError("Unsupported asset catalog schema")
    if data.get("parameter_policy", {}).get("plant_data_optimization") is not False:
        raise ValueError("Catalog must explicitly disable plant-data parameter optimization")
    return data


def assess_readiness(catalog: dict) -> tuple[AssetReadiness, ...]:
    results = []
    for asset in catalog.get("assets", []):
        model = asset["component_model"]
        required = REQUIRED_PARAMETERS.get(model)
        if required is None:
            raise ValueError(f"Unknown component model: {model}")
        parameters = asset.get("model_parameters", {})
        missing = tuple(name for name in required if parameters.get(name) is None)
        validation_status = asset.get("validation_status", "not_started")
        if validation_status not in {"not_started", "baseline_only", "failed", "passed"}:
            raise ValueError(f"Unknown validation status: {validation_status}")
        parameters_complete = not missing
        results.append(AssetReadiness(
            asset_id=asset["asset_id"],
            component_model=model,
            parameters_complete=parameters_complete,
            validation_status=validation_status,
            ready=parameters_complete and validation_status == "passed",
            missing_parameters=missing,
            status=asset.get("status", ""),
        ))
    return tuple(results)
