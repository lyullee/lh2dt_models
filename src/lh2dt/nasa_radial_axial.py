"""NASA TN D-4171 mapping for the conservative radial-axial tank."""

from __future__ import annotations

import csv
from importlib.resources import files
import math
from typing import Iterable

import numpy as np

from .benchmarks import (
    BTU_HR_FT2_TO_W_M2,
    PSI_TO_PA,
    LayeredEnergyPartitionPoint,
    NasaD4171Benchmark,
    PressureRisePoint,
    _summary,
)
from .geometry import HorizontalVesselGeometry
from .natural_circulation import CirculationLegGeometry
from .properties import HydrogenProperties
from .radial_axial_transport import RadialAxialBoundaryGeometry
from .radial_axial_tank import (
    RadialAxialTank,
    RadialAxialTankParameters,
    RadialAxialTankState,
)


def _phase_geometry(
    *,
    shape: HorizontalVesselGeometry,
    start_volume_m3: float,
    end_volume_m3: float,
    level_count: int,
    wall_volume_fraction: float,
    thermal_conductivity_W_mK: float,
    local_loss_coefficient: float,
) -> tuple[
    tuple[RadialAxialBoundaryGeometry, ...],
    tuple[float, ...],
    tuple[float, ...],
    float,
    float,
]:
    """Reduce an equal-volume spherical phase mesh to 2 x N geometry."""

    phase_volume = end_volume_m3 - start_volume_m3
    edge_volumes = np.linspace(start_volume_m3, end_volume_m3, level_count + 1)
    edge_heights = tuple(
        shape.height_from_liquid_volume(float(value)) for value in edge_volumes
    )
    center_heights = tuple(
        shape.height_from_liquid_volume(
            start_volume_m3 + (index + 0.5) * phase_volume / level_count
        )
        for index in range(level_count)
    )
    cumulative_wall_area = tuple(
        shape.at_height(height).wetted_inner_area_m2 for height in edge_heights
    )
    wall_patch_areas = tuple(
        cumulative_wall_area[index + 1] - cumulative_wall_area[index]
        for index in range(level_count)
    )
    core_fraction = 1.0 - wall_volume_fraction
    boundaries: list[RadialAxialBoundaryGeometry] = []
    axial_conductances: list[float] = []
    for index in range(level_count - 1):
        boundary_height = edge_heights[index + 1]
        cross_section = shape.at_height(boundary_height).interface_area_m2
        outer_radius = math.sqrt(cross_section / math.pi)
        core_radius = outer_radius * math.sqrt(core_fraction)
        wall_area = cross_section * wall_volume_fraction
        core_area = cross_section * core_fraction
        separation = center_heights[index + 1] - center_heights[index]
        boundaries.append(RadialAxialBoundaryGeometry(
            vertical_separation_m=separation,
            wall_leg=CirculationLegGeometry(
                separation,
                max(2.0 * (outer_radius - core_radius), 1.0e-9),
                wall_area,
                0.0,
                local_loss_coefficient,
            ),
            core_leg=CirculationLegGeometry(
                separation,
                max(2.0 * core_radius, 1.0e-9),
                core_area,
                0.0,
                local_loss_coefficient,
            ),
        ))
        axial_conductances.append(
            thermal_conductivity_W_mK
            * 0.5 * (wall_area + core_area)
            / separation
        )

    phase_mid_volume = 0.5 * (start_volume_m3 + end_volume_m3)
    representative_height = shape.height_from_liquid_volume(phase_mid_volume)
    representative_area = shape.at_height(representative_height).interface_area_m2
    outer_radius = math.sqrt(representative_area / math.pi)
    core_radius = outer_radius * math.sqrt(core_fraction)
    core_centroid = 2.0 * core_radius / 3.0
    annulus_centroid = (
        2.0
        / 3.0
        * (outer_radius**3 - core_radius**3)
        / (outer_radius**2 - core_radius**2)
    )
    radial_distance = max(annulus_centroid - core_centroid, 1.0e-9)
    phase_height = edge_heights[-1] - edge_heights[0]
    radial_interface_area = 2.0 * math.pi * core_radius * phase_height
    radial_conductance_per_level = (
        thermal_conductivity_W_mK
        * radial_interface_area
        / radial_distance
        / level_count
    )
    return (
        tuple(boundaries),
        center_heights,
        wall_patch_areas,
        radial_conductance_per_level,
        float(np.mean(axial_conductances)),
    )


def _energy_partition(
    *,
    tank: RadialAxialTank,
    initial_state: RadialAxialTankState,
    final_state: RadialAxialTankState,
    total_heat_input_J: float,
    evaporation_energy_J: float,
    test_number: int,
    heating_mode: str,
) -> LayeredEnergyPartitionPoint:
    if total_heat_input_J <= 0.0:
        raise ValueError("NASA radial-axial energy partition requires positive heat")
    nl = tank.parameters.liquid_level_count
    initial_thermo = tank.thermo(initial_state)
    final_thermo = tank.thermo(final_state)
    initial_masses = tank.cell_masses_kg(initial_state)
    final_masses = tank.cell_masses_kg(final_state)
    vapor_start = 2 * nl
    initial_vapor_energy = sum(
        mass * cell.specific_internal_energy_J_kg
        for mass, cell in zip(
            initial_masses[vapor_start:], initial_thermo.cells[vapor_start:]
        )
    )
    final_vapor_energy = sum(
        mass * cell.specific_internal_energy_J_kg
        for mass, cell in zip(
            final_masses[vapor_start:], final_thermo.cells[vapor_start:]
        )
    )
    vapor_energy = final_vapor_energy - initial_vapor_energy
    liquid_energy = total_heat_input_J - vapor_energy - evaporation_energy_J
    scale = 100.0 / total_heat_input_J
    liquid_percent = liquid_energy * scale
    vapor_percent = vapor_energy * scale
    evaporation_percent = evaporation_energy_J * scale
    return LayeredEnergyPartitionPoint(
        test_number=int(test_number),
        heating_mode=heating_mode,
        total_heat_input_J=float(total_heat_input_J),
        vapor_energy_J=float(vapor_energy),
        evaporation_energy_J=float(evaporation_energy_J),
        liquid_energy_J=float(liquid_energy),
        liquid_bulk_energy_J=None,
        liquid_layer_energy_J=None,
        vapor_energy_percent=float(vapor_percent),
        evaporation_energy_percent=float(evaporation_percent),
        liquid_energy_percent=float(liquid_percent),
        liquid_bulk_energy_percent=None,
        liquid_layer_energy_percent=None,
        component_closure_error_percent_point=float(
            liquid_percent + vapor_percent + evaporation_percent - 100.0
        ),
        liquid_closure_error_percent_point=None,
    )


def run_nasa_d4171_radial_axial_benchmark(
    properties: HydrogenProperties | None = None,
    time_step_s: float = 20.0,
    test_numbers: Iterable[int] | None = None,
    liquid_level_count: int = 2,
    vapor_level_count: int = 2,
    wall_volume_fraction: float = 0.5,
    local_loss_coefficient: float = 0.0,
    vapor_interface_heat_transfer_model: str = "conductance",
    gas_interface_energy_accommodation_coefficient: float = 1.0,
    adaptive_step_control: bool = False,
) -> NasaD4171Benchmark:
    """Run the coefficient-free 2 x N tank against D-4171 Table I.

    Hydraulic dimensions, cell positions, molecular conductances, interface
    areas, and wall-heat weights are derived from the published 9-inch sphere
    and each test's initial fill.  The equal radial-volume split is a fixed
    coarse-grid choice.  No observation is used to alter a model input.
    """

    if not math.isfinite(time_step_s) or time_step_s <= 0.0:
        raise ValueError("time_step_s must be finite and positive")
    for value, name in (
        (liquid_level_count, "liquid_level_count"),
        (vapor_level_count, "vapor_level_count"),
    ):
        if not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer of at least two")
    if not math.isfinite(wall_volume_fraction) or not 0.0 < wall_volume_fraction < 1.0:
        raise ValueError("wall_volume_fraction must lie strictly between zero and one")
    if not math.isfinite(local_loss_coefficient) or local_loss_coefficient < 0.0:
        raise ValueError("local_loss_coefficient must be finite and non-negative")
    if vapor_interface_heat_transfer_model not in {"conductance", "kinetic_gas_film"}:
        raise ValueError(
            "vapor_interface_heat_transfer_model must be 'conductance' or "
            "'kinetic_gas_film'"
        )
    if (
        not math.isfinite(gas_interface_energy_accommodation_coefficient)
        or not 0.0 < gas_interface_energy_accommodation_coefficient <= 1.0
    ):
        raise ValueError(
            "gas_interface_energy_accommodation_coefficient must be finite "
            "and in (0, 1]"
        )
    if not isinstance(adaptive_step_control, bool):
        raise ValueError("adaptive_step_control must be boolean")
    selected_tests = None if test_numbers is None else {int(value) for value in test_numbers}
    if selected_tests is not None and (
        not selected_tests or not selected_tests.issubset(set(range(1, 22)))
    ):
        raise ValueError("test_numbers must be a non-empty selection from 1 through 21")

    props = properties or HydrogenProperties()
    diameter_m = 9.0 * 0.0254
    shape = HorizontalVesselGeometry(
        inner_diameter_m=diameter_m,
        straight_length_m=0.0,
        head_depth_m=diameter_m / 2.0,
        quadrature_order=48,
    )
    initial_pressure_Pa = 101_325.0
    final_pressure_Pa = 100.0 * PSI_TO_PA
    saturated_liquid = props.saturated_liquid(initial_pressure_Pa)
    saturated_vapor = props.saturated_vapor(initial_pressure_Pa)
    liquid_k = props.thermal_conductivity_W_mK(saturated_liquid)
    vapor_k = props.thermal_conductivity_W_mK(saturated_vapor)
    data_path = files("lh2dt").joinpath("data/nasa_tn_d4171_table1.csv")
    points: list[PressureRisePoint] = []
    energy_partitions: list[LayeredEnergyPartitionPoint] = []

    with data_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            test_number = int(row["test_number"])
            if selected_tests is not None and test_number not in selected_tests:
                continue
            fill = float(row["initial_fill_percent"]) / 100.0
            liquid_volume = shape.volume_m3 * fill
            liquid_geometry = _phase_geometry(
                shape=shape,
                start_volume_m3=0.0,
                end_volume_m3=liquid_volume,
                level_count=liquid_level_count,
                wall_volume_fraction=wall_volume_fraction,
                thermal_conductivity_W_mK=liquid_k,
                local_loss_coefficient=local_loss_coefficient,
            )
            vapor_geometry = _phase_geometry(
                shape=shape,
                start_volume_m3=liquid_volume,
                end_volume_m3=shape.volume_m3,
                level_count=vapor_level_count,
                wall_volume_fraction=wall_volume_fraction,
                thermal_conductivity_W_mK=vapor_k,
                local_loss_coefficient=local_loss_coefficient,
            )
            interface_height = shape.height_from_liquid_volume(liquid_volume)
            interface_area = shape.at_height(interface_height).interface_area_m2
            liquid_distance = max(
                interface_height - liquid_geometry[1][-1], 1.0e-9
            )
            vapor_distance = max(
                vapor_geometry[1][0] - interface_height, 1.0e-9
            )
            radial_fractions = (
                wall_volume_fraction, 1.0 - wall_volume_fraction
            )
            liquid_interface_ua = tuple(
                liquid_k * interface_area * fraction / liquid_distance
                for fraction in radial_fractions
            )
            vapor_interface_ua = tuple(
                vapor_k * interface_area * fraction / vapor_distance
                for fraction in radial_fractions
            )
            patch_areas = (*liquid_geometry[2], *vapor_geometry[2])
            patch_area_total = sum(patch_areas)
            tank = RadialAxialTank(RadialAxialTankParameters(
                total_volume_m3=shape.volume_m3,
                liquid_level_count=liquid_level_count,
                vapor_level_count=vapor_level_count,
                liquid_wall_volume_fraction=wall_volume_fraction,
                vapor_wall_volume_fraction=wall_volume_fraction,
                liquid_boundaries=liquid_geometry[0],
                vapor_boundaries=vapor_geometry[0],
                wall_heat_capacity_J_K=1.0,
                liquid_radial_conductance_W_K=liquid_geometry[3],
                vapor_radial_conductance_W_K=vapor_geometry[3],
                liquid_axial_conductance_W_K=liquid_geometry[4],
                vapor_axial_conductance_W_K=vapor_geometry[4],
                liquid_interface_UA_W_K=liquid_interface_ua,
                vapor_interface_UA_W_K=vapor_interface_ua,
                vapor_interface_heat_transfer_model=vapor_interface_heat_transfer_model,
                vapor_interface_area_m2=tuple(
                    interface_area * fraction for fraction in radial_fractions
                ),
                gas_interface_energy_accommodation_coefficient=(
                    gas_interface_energy_accommodation_coefficient
                ),
                distributed_boiling_model="saturated_liquid_complementarity",
                wall_patch_area_fractions=tuple(
                    area / patch_area_total for area in patch_areas
                ),
            ), props)
            initial = tank.initialize_saturated(initial_pressure_Pa, fill)
            geometry = shape.at_height(interface_height)
            liquid_heat_W = (
                float(row["wetted_heat_flux_Btu_hr_ft2"])
                * BTU_HR_FT2_TO_W_M2
                * geometry.wetted_inner_area_m2
            )
            vapor_heat_W = (
                float(row["dry_heat_flux_Btu_hr_ft2"])
                * BTU_HR_FT2_TO_W_M2
                * geometry.dry_inner_area_m2
            )
            total_heat_W = (
                float(row["average_heat_flux_Btu_hr_ft2"])
                * BTU_HR_FT2_TO_W_M2
                * shape.inner_surface_area_m2
            )
            partition_scale = total_heat_W / (liquid_heat_W + vapor_heat_W)
            liquid_weights = np.asarray(liquid_geometry[2]) / sum(liquid_geometry[2])
            vapor_weights = np.asarray(vapor_geometry[2]) / sum(vapor_geometry[2])
            direct_heat = np.zeros(tank.cell_count)
            direct_heat[:liquid_level_count] = (
                liquid_weights * liquid_heat_W * partition_scale
            )
            vapor_wall_start = 2 * liquid_level_count
            direct_heat[
                vapor_wall_start:vapor_wall_start + vapor_level_count
            ] = vapor_weights * vapor_heat_W * partition_scale
            observed = float(row["pressure_rise_psi_min"]) * PSI_TO_PA / 60.0
            duration = (final_pressure_Pa - initial_pressure_Pa) / observed
            state = initial
            elapsed = 0.0
            total_heat_input = 0.0
            evaporation_energy = 0.0
            while elapsed < duration:
                requested_step = min(time_step_s, duration - elapsed)
                step = requested_step
                if adaptive_step_control:
                    step = min(
                        requested_step,
                        tank.recommended_explicit_time_step_s(
                            state,
                            ambient_temperature_K=saturated_liquid.temperature_K,
                            direct_fluid_heat_W=direct_heat,
                            maximum_time_step_s=requested_step,
                        ),
                    )
                    if not math.isfinite(step) or step <= 0.0:
                        raise ValueError(
                            f"NASA TN D-4171 radial-axial test {test_number} "
                            "has no positive adaptive explicit time step"
                        )
                attempt_step = step
                result = None
                last_error: ValueError | None = None
                for _ in range(24 if adaptive_step_control else 1):
                    try:
                        result = tank.step_euler(
                            state,
                            time_step_s=attempt_step,
                            ambient_temperature_K=saturated_liquid.temperature_K,
                            direct_fluid_heat_W=direct_heat,
                        )
                        break
                    except ValueError as error:
                        last_error = error
                        if not adaptive_step_control:
                            break
                        attempt_step *= 0.5
                        if attempt_step <= max(duration, 1.0) * 1.0e-12:
                            break
                if result is None:
                    assert last_error is not None
                    raise ValueError(
                        f"NASA TN D-4171 radial-axial test {test_number} "
                        f"failed at t={elapsed:g} s, dt={attempt_step:g} s: {last_error}"
                    ) from last_error
                step = attempt_step
                latent = (
                    props.saturated_vapor(state.pressure_Pa).specific_enthalpy_J_kg
                    - props.saturated_liquid(state.pressure_Pa).specific_enthalpy_J_kg
                )
                evaporation_energy += step * (
                    sum(result.derivative.interface_phase_change_rates_kg_s) * latent
                    + result.derivative.distributed_boiling_latent_power_W
                )
                total_heat_input += step * result.derivative.boundary_energy_rate_W
                state = result.state
                elapsed += step
            predicted = (state.pressure_Pa - initial_pressure_Pa) / duration
            points.append(PressureRisePoint(
                test_number=test_number,
                heating_mode=row["heating_mode"],
                initial_fill_fraction=fill,
                net_average_heat_flux_W_m2=total_heat_W / shape.inner_surface_area_m2,
                observed_pressure_rise_Pa_s=observed,
                predicted_pressure_rise_Pa_s=predicted,
                predicted_over_observed=predicted / observed,
                predicted_duration_s=duration,
            ))
            energy_partitions.append(_energy_partition(
                tank=tank,
                initial_state=initial,
                final_state=state,
                total_heat_input_J=total_heat_input,
                evaporation_energy_J=evaporation_energy,
                test_number=test_number,
                heating_mode=row["heating_mode"],
            ))

    expected_tests = set(range(1, 22)) if selected_tests is None else selected_tests
    if {point.test_number for point in points} != expected_tests:
        raise RuntimeError("NASA radial-axial benchmark test selection is incomplete")
    modes = sorted({point.heating_mode for point in points})
    return NasaD4171Benchmark(
        source_title=(
            "Normal Gravity Self-Pressurization of 9-Inch Diameter "
            "Spherical Liquid Hydrogen Tankage"
        ),
        source_report="NASA TN D-4171",
        source_url="https://ntrs.nasa.gov/citations/19670028965",
        model=(
            f"liquid 2x{liquid_level_count} and vapor 2x{vapor_level_count} "
            "radial-axial common-pressure tank"
        ),
        assumptions=(
            "9 inch internal spherical geometry with equal-volume axial cells",
            f"wall/core radial volume fractions are {wall_volume_fraction:g}/"
            f"{1.0 - wall_volume_fraction:g} as a fixed coarse-grid choice",
            f"smooth hydraulic paths with specified local-loss coefficient {local_loss_coefficient:g}",
            "molecular conductances use CoolProp properties at the initial saturated state",
            f"vapor interface heat-transfer model is {vapor_interface_heat_transfer_model}",
            f"gas-side energy accommodation coefficient is {gas_interface_energy_accommodation_coefficient:g}",
            f"adaptive explicit step control is {'enabled' if adaptive_step_control else 'disabled'}",
            "Table I net wetted/dry heat enters only the corresponding wall-layer cells",
            "the heat partition is rescaled to preserve reported total average heat input",
            "saturated liquid uses coefficient-free pressure-coupled boiling complementarity",
            "the comparison horizon is the reported time to nominal 100 psia",
            "no parameter is fitted to pressure-rise or energy-partition observations",
        ),
        points=tuple(points),
        overall=_summary(points),
        by_heating_mode={
            mode: _summary(point for point in points if point.heating_mode == mode)
            for mode in modes
        },
        energy_partitions=tuple(energy_partitions),
    )
