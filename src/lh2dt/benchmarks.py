"""Independent public-data benchmarks for model-scope decisions.

These routines execute fixed-parameter forward calculations. They contain no
parameter search, calibration, or optimization path.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from importlib.resources import files
import math
from typing import Iterable

import numpy as np

from .properties import HydrogenProperties
from .geometry import HorizontalVesselGeometry
from .layered_tank import (
    LayeredTank,
    LayeredTankParameters,
    LayeredTankState,
    LayeredTankStepResult,
)
from .stratified_tank import StratifiedTank, StratifiedTankParameters


BTU_HR_FT2_TO_W_M2 = 3.154590745
PSI_TO_PA = 6_894.757293168


@dataclass(frozen=True)
class PressureRisePoint:
    test_number: int
    heating_mode: str
    initial_fill_fraction: float
    net_average_heat_flux_W_m2: float
    observed_pressure_rise_Pa_s: float
    predicted_pressure_rise_Pa_s: float
    predicted_over_observed: float
    predicted_duration_s: float


@dataclass(frozen=True)
class PressureRiseSummary:
    count: int
    mean_predicted_over_observed: float
    mean_absolute_percentage_error_percent: float
    root_mean_squared_error_Pa_s: float
    normalized_rmse: float | None


@dataclass(frozen=True)
class LayeredEnergyPartitionPoint:
    test_number: int
    heating_mode: str
    total_heat_input_J: float
    vapor_energy_J: float
    evaporation_energy_J: float
    liquid_energy_J: float
    liquid_bulk_energy_J: float | None
    liquid_layer_energy_J: float | None
    vapor_energy_percent: float
    evaporation_energy_percent: float
    liquid_energy_percent: float
    liquid_bulk_energy_percent: float | None
    liquid_layer_energy_percent: float | None
    component_closure_error_percent_point: float
    liquid_closure_error_percent_point: float | None


@dataclass(frozen=True)
class NasaD4171Benchmark:
    source_title: str
    source_report: str
    source_url: str
    model: str
    assumptions: tuple[str, ...]
    points: tuple[PressureRisePoint, ...]
    overall: PressureRiseSummary
    by_heating_mode: dict[str, PressureRiseSummary]
    energy_partitions: tuple[LayeredEnergyPartitionPoint, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def reduce_layered_state_to_nasa_energy_partition(
    *,
    tank: LayeredTank,
    initial_state: LayeredTankState,
    final_result: LayeredTankStepResult,
    test_number: int,
    heating_mode: str,
) -> LayeredEnergyPartitionPoint:
    """Reduce a layered result using NASA TN D-4171 Appendix D definitions.

    Vapor energy follows Eq. D4, evaporation energy is the integrated
    instantaneous latent power corresponding to Eq. D6, and liquid energy is
    the residual in Eq. D7.  The optional bulk/layer subdivision follows Eqs.
    D10--D14, using the lower liquid cells as the modeled bulk temperature.
    """

    initial_thermo = tank.thermo(initial_state)
    final_state = final_result.state
    final_thermo = final_result.thermo
    total_heat = final_result.cumulative_boundary_energy_J
    if not math.isfinite(total_heat) or total_heat <= 0.0:
        raise ValueError("NASA energy partition requires positive total boundary heat")
    initial_vapor_energy = sum(
        mass * fluid.specific_internal_energy_J_kg
        for mass, fluid in zip(initial_state.vapor_masses_kg, initial_thermo.vapor)
    )
    final_vapor_energy = sum(
        mass * fluid.specific_internal_energy_J_kg
        for mass, fluid in zip(final_state.vapor_masses_kg, final_thermo.vapor)
    )
    vapor_energy = final_vapor_energy - initial_vapor_energy
    evaporation_energy = final_result.cumulative_latent_phase_change_energy_J
    liquid_energy = total_heat - vapor_energy - evaporation_energy

    initial_liquid_energy = sum(
        mass * fluid.specific_internal_energy_J_kg
        for mass, fluid in zip(initial_state.liquid_masses_kg, initial_thermo.liquid)
    )
    final_liquid_energy = sum(
        mass * fluid.specific_internal_energy_J_kg
        for mass, fluid in zip(final_state.liquid_masses_kg, final_thermo.liquid)
    )
    liquid_mass = sum(final_state.liquid_masses_kg)
    # NASA Eq. D9 defines the average liquid internal energy from the
    # residual liquid heat.  This also avoids tying the reduction to a
    # particular internal-energy bookkeeping convention for transferred mass.
    initial_liquid_average_u = initial_liquid_energy / sum(initial_state.liquid_masses_kg)
    liquid_average_u = initial_liquid_average_u + liquid_energy / liquid_mass
    bulk_count = max(1, len(final_thermo.liquid) - 1)
    bulk_mass = sum(final_state.liquid_masses_kg[:bulk_count])
    bulk_u = sum(
        mass * fluid.specific_internal_energy_J_kg
        for mass, fluid in zip(
            final_state.liquid_masses_kg[:bulk_count], final_thermo.liquid[:bulk_count]
        )
    ) / bulk_mass
    saturated_u = tank.properties.saturated_liquid(
        final_state.pressure_Pa
    ).specific_internal_energy_J_kg
    denominator = saturated_u - bulk_u
    reduced_bulk_energy: float | None = None
    reduced_layer_energy: float | None = None
    if abs(denominator) > 1e-9:
        inferred_bulk_mass = liquid_mass * (
            saturated_u + bulk_u - 2.0 * liquid_average_u
        ) / denominator
        tolerance = liquid_mass * 1e-8
        if -tolerance <= inferred_bulk_mass <= liquid_mass + tolerance:
            inferred_bulk_mass = min(max(inferred_bulk_mass, 0.0), liquid_mass)
            reduced_bulk_energy = inferred_bulk_mass * bulk_u - initial_liquid_energy
            reduced_layer_energy = liquid_energy - reduced_bulk_energy

    scale = 100.0 / total_heat
    liquid_percent = liquid_energy * scale
    vapor_percent = vapor_energy * scale
    evaporation_percent = evaporation_energy * scale
    bulk_percent = None if reduced_bulk_energy is None else reduced_bulk_energy * scale
    layer_percent = None if reduced_layer_energy is None else reduced_layer_energy * scale
    return LayeredEnergyPartitionPoint(
        test_number=int(test_number),
        heating_mode=heating_mode,
        total_heat_input_J=float(total_heat),
        vapor_energy_J=float(vapor_energy),
        evaporation_energy_J=float(evaporation_energy),
        liquid_energy_J=float(liquid_energy),
        liquid_bulk_energy_J=None if reduced_bulk_energy is None else float(reduced_bulk_energy),
        liquid_layer_energy_J=None if reduced_layer_energy is None else float(reduced_layer_energy),
        vapor_energy_percent=float(vapor_percent),
        evaporation_energy_percent=float(evaporation_percent),
        liquid_energy_percent=float(liquid_percent),
        liquid_bulk_energy_percent=None if bulk_percent is None else float(bulk_percent),
        liquid_layer_energy_percent=None if layer_percent is None else float(layer_percent),
        component_closure_error_percent_point=float(
            liquid_percent + vapor_percent + evaporation_percent - 100.0
        ),
        liquid_closure_error_percent_point=(
            None if bulk_percent is None or layer_percent is None
            else float(bulk_percent + layer_percent - liquid_percent)
        ),
    )


@dataclass(frozen=True)
class EnergyPartitionPoint:
    test_number: int
    heating_mode: str
    average_fill_fraction: float
    direct_wetted_wall_energy_percent: float
    observed_liquid_energy_percent: float
    observed_vapor_energy_percent: float
    observed_evaporation_energy_percent: float
    observed_liquid_bulk_energy_percent: float
    observed_liquid_layer_energy_percent: float
    inferred_cross_phase_transfer_percent: float
    zero_transfer_liquid_error_percent_point: float
    zero_transfer_vapor_error_percent_point: float
    zero_transfer_evaporation_error_percent_point: float
    component_closure_error_percent_point: float
    liquid_closure_error_percent_point: float


@dataclass(frozen=True)
class EnergyPartitionSummary:
    count: int
    mean_observed_liquid_energy_percent: float
    mean_observed_vapor_energy_percent: float
    mean_observed_evaporation_energy_percent: float
    mean_observed_liquid_layer_energy_percent: float
    mean_inferred_cross_phase_transfer_percent: float
    zero_transfer_liquid_mae_percent_point: float
    zero_transfer_vapor_mae_percent_point: float
    zero_transfer_evaporation_mae_percent_point: float


@dataclass(frozen=True)
class NasaD4171EnergyPartitionBenchmark:
    source_title: str
    source_report: str
    source_url: str
    comparison_model: str
    assumptions: tuple[str, ...]
    points: tuple[EnergyPartitionPoint, ...]
    overall: EnergyPartitionSummary
    by_heating_mode: dict[str, EnergyPartitionSummary]

    def to_dict(self) -> dict:
        return asdict(self)


def homogeneous_closed_tank_pressure_rise(
    *,
    tank_volume_m3: float,
    tank_inner_area_m2: float,
    initial_liquid_volume_fraction: float,
    net_average_heat_flux_W_m2: float,
    initial_pressure_Pa: float,
    final_pressure_Pa: float,
    properties: HydrogenProperties | None = None,
) -> tuple[float, float]:
    """Return average HEM pressure-rise rate and duration at fixed inventory."""

    values = (
        tank_volume_m3,
        tank_inner_area_m2,
        net_average_heat_flux_W_m2,
        initial_pressure_Pa,
        final_pressure_Pa,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("geometry, heat flux, and pressures must be finite and positive")
    if not 0.0 < initial_liquid_volume_fraction < 1.0:
        raise ValueError("initial_liquid_volume_fraction must lie between zero and one")
    if final_pressure_Pa <= initial_pressure_Pa:
        raise ValueError("final pressure must exceed initial pressure")

    props = properties or HydrogenProperties()
    liquid = props.saturated_liquid(initial_pressure_Pa)
    vapor = props.saturated_vapor(initial_pressure_Pa)
    liquid_volume = tank_volume_m3 * initial_liquid_volume_fraction
    vapor_volume = tank_volume_m3 - liquid_volume
    liquid_mass = liquid.density_kg_m3 * liquid_volume
    vapor_mass = vapor.density_kg_m3 * vapor_volume
    mass = liquid_mass + vapor_mass
    initial_energy = (
        liquid_mass * liquid.specific_internal_energy_J_kg
        + vapor_mass * vapor.specific_internal_energy_J_kg
    )
    density = mass / tank_volume_m3
    final = props.from_prho(final_pressure_Pa, density)
    required_energy = mass * final.specific_internal_energy_J_kg - initial_energy
    heat_rate = net_average_heat_flux_W_m2 * tank_inner_area_m2
    if required_energy <= 0.0:
        raise ValueError("requested final state does not require positive heat input")
    duration = required_energy / heat_rate
    return (final_pressure_Pa - initial_pressure_Pa) / duration, duration


def _summary(points: Iterable[PressureRisePoint]) -> PressureRiseSummary:
    items = tuple(points)
    observed = np.asarray([point.observed_pressure_rise_Pa_s for point in items])
    predicted = np.asarray([point.predicted_pressure_rise_Pa_s for point in items])
    ratios = predicted / observed
    rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    span = float(np.ptp(observed))
    return PressureRiseSummary(
        count=len(items),
        mean_predicted_over_observed=float(np.mean(ratios)),
        mean_absolute_percentage_error_percent=float(np.mean(np.abs(ratios - 1.0)) * 100.0),
        root_mean_squared_error_Pa_s=rmse,
        normalized_rmse=rmse / span if span > 0.0 else None,
    )


def _energy_partition_summary(
    points: Iterable[EnergyPartitionPoint],
) -> EnergyPartitionSummary:
    items = tuple(points)
    if not items:
        raise ValueError("At least one energy-partition point is required")
    return EnergyPartitionSummary(
        count=len(items),
        mean_observed_liquid_energy_percent=float(np.mean([
            point.observed_liquid_energy_percent for point in items
        ])),
        mean_observed_vapor_energy_percent=float(np.mean([
            point.observed_vapor_energy_percent for point in items
        ])),
        mean_observed_evaporation_energy_percent=float(np.mean([
            point.observed_evaporation_energy_percent for point in items
        ])),
        mean_observed_liquid_layer_energy_percent=float(np.mean([
            point.observed_liquid_layer_energy_percent for point in items
        ])),
        mean_inferred_cross_phase_transfer_percent=float(np.mean([
            point.inferred_cross_phase_transfer_percent for point in items
        ])),
        zero_transfer_liquid_mae_percent_point=float(np.mean(np.abs([
            point.zero_transfer_liquid_error_percent_point for point in items
        ]))),
        zero_transfer_vapor_mae_percent_point=float(np.mean(np.abs([
            point.zero_transfer_vapor_error_percent_point for point in items
        ]))),
        zero_transfer_evaporation_mae_percent_point=float(np.mean(np.abs([
            point.zero_transfer_evaporation_error_percent_point for point in items
        ]))),
    )


def run_nasa_d4171_energy_partition_benchmark() -> NasaD4171EnergyPartitionBenchmark:
    """Audit Table II against the zero-interphase-transfer limiting model.

    This comparison identifies energy paths that the limiting model cannot
    represent.  It performs no coefficient search and does not use the table to
    alter a model parameter.
    """

    data_path = files("lh2dt").joinpath("data/nasa_tn_d4171_table2.csv")
    points: list[EnergyPartitionPoint] = []
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            direct_wet = float(row["direct_wetted_wall_energy_percent"])
            liquid = float(row["total_liquid_energy_percent"])
            vapor = float(row["vapor_energy_percent"])
            evaporation = float(row["evaporation_energy_percent"])
            bulk = float(row["liquid_bulk_energy_percent"])
            layer = float(row["liquid_layer_energy_percent"])
            # With no interphase transfer, wetted-wall heat remains in liquid,
            # dry-wall heat remains in vapor, and evaporation receives zero.
            predicted_liquid = direct_wet
            predicted_vapor = 100.0 - direct_wet
            predicted_evaporation = 0.0
            points.append(EnergyPartitionPoint(
                test_number=int(row["test_number"]),
                heating_mode=row["heating_mode"],
                average_fill_fraction=float(row["average_fill_percent"]) / 100.0,
                direct_wetted_wall_energy_percent=direct_wet,
                observed_liquid_energy_percent=liquid,
                observed_vapor_energy_percent=vapor,
                observed_evaporation_energy_percent=evaporation,
                observed_liquid_bulk_energy_percent=bulk,
                observed_liquid_layer_energy_percent=layer,
                inferred_cross_phase_transfer_percent=(liquid + evaporation - direct_wet),
                zero_transfer_liquid_error_percent_point=predicted_liquid - liquid,
                zero_transfer_vapor_error_percent_point=predicted_vapor - vapor,
                zero_transfer_evaporation_error_percent_point=predicted_evaporation - evaporation,
                component_closure_error_percent_point=liquid + vapor + evaporation - 100.0,
                liquid_closure_error_percent_point=bulk + layer - liquid,
            ))
    expected_tests = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19, 20}
    if len(points) != 17 or {point.test_number for point in points} != expected_tests:
        raise RuntimeError("NASA TN D-4171 Table II must contain the 17 published tests")
    if max(abs(point.component_closure_error_percent_point) for point in points) > 0.11:
        raise RuntimeError("NASA TN D-4171 Table II component energy balance does not close")
    if max(abs(point.liquid_closure_error_percent_point) for point in points) > 0.11:
        raise RuntimeError("NASA TN D-4171 Table II liquid energy subdivision does not close")
    modes = sorted({point.heating_mode for point in points})
    return NasaD4171EnergyPartitionBenchmark(
        source_title="Normal Gravity Self-Pressurization of 9-Inch Diameter Spherical Liquid Hydrogen Tankage",
        source_report="NASA TN D-4171 Table II",
        source_url="https://ntrs.nasa.gov/citations/19670028965",
        comparison_model="two-region zero-interphase-transfer energy partition",
        assumptions=(
            "published percentages are treated as rounded observations",
            "wetted-wall energy remains in the liquid in the zero-transfer limit",
            "dry-wall energy remains in the vapor in the zero-transfer limit",
            "evaporation energy is zero in the zero-transfer limit",
            "inferred cross-phase transfer is liquid plus evaporation minus wetted-wall input",
            "no parameter is fitted to Table II",
        ),
        points=tuple(points),
        overall=_energy_partition_summary(points),
        by_heating_mode={
            mode: _energy_partition_summary(
                point for point in points if point.heating_mode == mode
            )
            for mode in modes
        },
    )


def run_nasa_d4171_benchmark(
    properties: HydrogenProperties | None = None,
) -> NasaD4171Benchmark:
    """Run the HEM model against NASA TN D-4171 Table I, without fitting."""

    diameter_m = 9.0 * 0.0254
    volume_m3 = math.pi * diameter_m**3 / 6.0
    inner_area_m2 = math.pi * diameter_m**2
    initial_pressure_Pa = 101_325.0
    final_pressure_Pa = 100.0 * PSI_TO_PA
    data_path = files("lh2dt").joinpath("data/nasa_tn_d4171_table1.csv")
    points: list[PressureRisePoint] = []
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            observed = float(row["pressure_rise_psi_min"]) * PSI_TO_PA / 60.0
            heat_flux = float(row["average_heat_flux_Btu_hr_ft2"]) * BTU_HR_FT2_TO_W_M2
            predicted, duration = homogeneous_closed_tank_pressure_rise(
                tank_volume_m3=volume_m3,
                tank_inner_area_m2=inner_area_m2,
                initial_liquid_volume_fraction=float(row["initial_fill_percent"]) / 100.0,
                net_average_heat_flux_W_m2=heat_flux,
                initial_pressure_Pa=initial_pressure_Pa,
                final_pressure_Pa=final_pressure_Pa,
                properties=properties,
            )
            points.append(PressureRisePoint(
                test_number=int(row["test_number"]),
                heating_mode=row["heating_mode"],
                initial_fill_fraction=float(row["initial_fill_percent"]) / 100.0,
                net_average_heat_flux_W_m2=heat_flux,
                observed_pressure_rise_Pa_s=observed,
                predicted_pressure_rise_Pa_s=predicted,
                predicted_over_observed=predicted / observed,
                predicted_duration_s=duration,
            ))
    if len(points) != 21 or len({point.test_number for point in points}) != 21:
        raise RuntimeError("NASA TN D-4171 benchmark must contain 21 unique tests")
    modes = sorted({point.heating_mode for point in points})
    return NasaD4171Benchmark(
        source_title="Normal Gravity Self-Pressurization of 9-Inch Diameter Spherical Liquid Hydrogen Tankage",
        source_report="NASA TN D-4171",
        source_url="https://ntrs.nasa.gov/citations/19670028965",
        model="closed rigid homogeneous-equilibrium hydrogen inventory",
        assumptions=(
            "9 inch internal spherical diameter",
            "tests begin at 1 atmosphere absolute pressure",
            "nominal endpoint is 100 psia absolute",
            "Table I average heat flux multiplied by full inner spherical area",
            "all net heat enters a spatially homogeneous equilibrium hydrogen inventory",
            "no parameter is fitted to the 21 pressure-rise observations",
        ),
        points=tuple(points),
        overall=_summary(points),
        by_heating_mode={
            mode: _summary(point for point in points if point.heating_mode == mode)
            for mode in modes
        },
    )


def run_nasa_d4171_two_region_limit_benchmark(
    properties: HydrogenProperties | None = None,
    time_step_s: float = 60.0,
) -> NasaD4171Benchmark:
    """Run the zero-interphase-transfer limit of the two-region tank model.

    The reported mean heat flux to hydrogen is partitioned using Table I's
    wetted- and dry-wall fluxes and the initial spherical areas.  The two
    partitioned rates are rescaled together so their sum exactly equals the
    reported total rate.  No heat-transfer coefficient is inferred from the
    pressure observations.  This is a bounding calculation, not a completed
    stratified-tank validation.
    """

    if not math.isfinite(time_step_s) or time_step_s <= 0.0:
        raise ValueError("time_step_s must be finite and positive")
    props = properties or HydrogenProperties()
    diameter_m = 9.0 * 0.0254
    shape = HorizontalVesselGeometry(
        inner_diameter_m=diameter_m,
        straight_length_m=0.0,
        head_depth_m=diameter_m / 2.0,
        quadrature_order=48,
    )
    parameters = StratifiedTankParameters(
        lower_wall_heat_capacity_J_K=1.0,
        upper_wall_heat_capacity_J_K=1.0,
        lower_ambient_UA_W_K=0.0,
        upper_ambient_UA_W_K=0.0,
        wall_liquid_U_W_m2K=0.0,
        wall_vapor_U_W_m2K=0.0,
        liquid_interface_UA_W_K=0.0,
        vapor_interface_UA_W_K=0.0,
        wall_axial_UA_W_K=0.0,
        wall_zone_split_height_m=shape.radius,
        minimum_pressure_Pa=10_000.0,
        maximum_pressure_Pa=1_250_000.0,
    )
    tank = StratifiedTank(shape, parameters, props)
    initial_pressure_Pa = 101_325.0
    final_pressure_Pa = 100.0 * PSI_TO_PA
    initial_temperature_K = props.saturated_liquid(initial_pressure_Pa).temperature_K
    data_path = files("lh2dt").joinpath("data/nasa_tn_d4171_table1.csv")
    points: list[PressureRisePoint] = []
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            fill = float(row["initial_fill_percent"]) / 100.0
            initial = tank.initialize_saturated(initial_pressure_Pa, fill)
            initial_geometry = shape.at_height(
                shape.height_from_liquid_volume(shape.volume_m3 * fill)
            )
            liquid_heat_W = (
                float(row["wetted_heat_flux_Btu_hr_ft2"])
                * BTU_HR_FT2_TO_W_M2
                * initial_geometry.wetted_inner_area_m2
            )
            vapor_heat_W = (
                float(row["dry_heat_flux_Btu_hr_ft2"])
                * BTU_HR_FT2_TO_W_M2
                * initial_geometry.dry_inner_area_m2
            )
            total_heat_W = (
                float(row["average_heat_flux_Btu_hr_ft2"])
                * BTU_HR_FT2_TO_W_M2
                * shape.inner_surface_area_m2
            )
            partition_total = liquid_heat_W + vapor_heat_W
            if partition_total <= 0.0:
                raise ValueError("NASA heat partition must be positive")
            partition_scale = total_heat_W / partition_total
            liquid_heat_W *= partition_scale
            vapor_heat_W *= partition_scale
            observed = float(row["pressure_rise_psi_min"]) * PSI_TO_PA / 60.0
            evaluation_duration = (final_pressure_Pa - initial_pressure_Pa) / observed
            result = tank.simulate(
                initial,
                duration_s=evaluation_duration,
                time_step_s=time_step_s,
                ambient_temperature_K=initial_temperature_K,
                flow_callback=lambda _time, _thermo: ((), ()),
                fluid_heat_callback=(
                    lambda _time, _thermo, ql=liquid_heat_W, qv=vapor_heat_W: (ql, qv)
                ),
            )[-1]
            predicted = (result.thermo.pressure_Pa - initial_pressure_Pa) / evaluation_duration
            points.append(PressureRisePoint(
                test_number=int(row["test_number"]),
                heating_mode=row["heating_mode"],
                initial_fill_fraction=fill,
                net_average_heat_flux_W_m2=total_heat_W / shape.inner_surface_area_m2,
                observed_pressure_rise_Pa_s=observed,
                predicted_pressure_rise_Pa_s=predicted,
                predicted_over_observed=predicted / observed,
                predicted_duration_s=evaluation_duration,
            ))
    modes = sorted({point.heating_mode for point in points})
    return NasaD4171Benchmark(
        source_title="Normal Gravity Self-Pressurization of 9-Inch Diameter Spherical Liquid Hydrogen Tankage",
        source_report="NASA TN D-4171",
        source_url="https://ntrs.nasa.gov/citations/19670028965",
        model="two-region liquid/vapor zero-interphase-transfer bounding limit",
        assumptions=(
            "9 inch internal spherical diameter represented by two hemispherical heads",
            "tests begin at 1 atmosphere absolute pressure",
            "the comparison horizon is the reported time to nominal 100 psia",
            "Table I net wetted-wall heat enters the liquid region",
            "Table I net dry-wall heat enters the vapor region",
            "the partition is rescaled to preserve the reported total average heat input",
            "interphase heat and mass transfer are zero to define an upper-response limit",
            "no parameter is fitted to the 21 pressure-rise observations",
        ),
        points=tuple(points),
        overall=_summary(points),
        by_heating_mode={
            mode: _summary(point for point in points if point.heating_mode == mode)
            for mode in modes
        },
    )


def _run_nasa_d4171_layered_benchmark(
    interface_heat_transfer_model: str,
    vertical_wall_circulation_model: str = "off",
    interface_phase_change_model: str = "equilibrium_energy_jump",
    schrage_accommodation_coefficient: float = 0.001,
    distributed_boiling_model: str = "off",
    properties: HydrogenProperties | None = None,
    time_step_s: float = 60.0,
    test_numbers: Iterable[int] | None = None,
    liquid_cell_count: int = 3,
    vapor_cell_count: int = 3,
    wall_node_count: int | None = None,
) -> NasaD4171Benchmark:
    """Run one fixed interface-closure variant of the layered tank.

    Table I's measured net wetted- and dry-wall heat rates are imposed directly
    on the requested liquid and vapor cell counts.  Within each phase, heat is
    distributed by the initial cell wall-contact area.  The model then resolves
    axial molecular conduction and interface phase change without a searched or
    calibrated multiplier.  This is a fixed-parameter baseline for deciding
    which closure mechanisms must be added next.
    """

    if not math.isfinite(time_step_s) or time_step_s <= 0.0:
        raise ValueError("time_step_s must be finite and positive")
    if not isinstance(liquid_cell_count, int) or liquid_cell_count < 1:
        raise ValueError("liquid_cell_count must be a positive integer")
    if not isinstance(vapor_cell_count, int) or vapor_cell_count < 1:
        raise ValueError("vapor_cell_count must be a positive integer")
    if wall_node_count is None:
        wall_node_count = max(8, liquid_cell_count + vapor_cell_count + 2)
    if not isinstance(wall_node_count, int) or wall_node_count < 2:
        raise ValueError("wall_node_count must be an integer of at least two")
    props = properties or HydrogenProperties()
    selected_tests = None if test_numbers is None else {int(value) for value in test_numbers}
    if selected_tests is not None and not selected_tests:
        raise ValueError("test_numbers cannot be empty")
    if selected_tests is not None and not selected_tests.issubset(set(range(1, 22))):
        raise ValueError("test_numbers must be selected from 1 through 21")
    diameter_m = 9.0 * 0.0254
    shape = HorizontalVesselGeometry(
        inner_diameter_m=diameter_m,
        straight_length_m=0.0,
        head_depth_m=diameter_m / 2.0,
        quadrature_order=48,
    )
    tank = LayeredTank(
        shape,
        LayeredTankParameters(
            liquid_cell_count=liquid_cell_count,
            vapor_cell_count=vapor_cell_count,
            wall_node_count=wall_node_count,
            wall_heat_capacity_J_K=1.0,
            ambient_UA_W_K=0.0,
            wall_liquid_U_W_m2K=0.0,
            wall_vapor_U_W_m2K=0.0,
            wall_axial_conductance_W_K=0.0,
            liquid_axial_conduction_multiplier=1.0,
            vapor_axial_conduction_multiplier=1.0,
            liquid_interface_conduction_multiplier=1.0,
            vapor_interface_conduction_multiplier=1.0,
            interface_heat_transfer_model=interface_heat_transfer_model,
            interface_phase_change_model=interface_phase_change_model,
            schrage_accommodation_coefficient=schrage_accommodation_coefficient,
            distributed_boiling_model=distributed_boiling_model,
            vertical_wall_circulation_model=vertical_wall_circulation_model,
            minimum_pressure_Pa=10_000.0,
            maximum_pressure_Pa=1_250_000.0,
        ),
        props,
    )
    initial_pressure_Pa = 101_325.0
    final_pressure_Pa = 100.0 * PSI_TO_PA
    initial_temperature_K = props.saturated_liquid(initial_pressure_Pa).temperature_K
    data_path = files("lh2dt").joinpath("data/nasa_tn_d4171_table1.csv")
    points: list[PressureRisePoint] = []
    energy_partitions: list[LayeredEnergyPartitionPoint] = []
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if selected_tests is not None and int(row["test_number"]) not in selected_tests:
                continue
            fill = float(row["initial_fill_percent"]) / 100.0
            initial = tank.initialize_saturated(initial_pressure_Pa, fill)
            initial_thermo = tank.thermo(initial)
            overlap = np.asarray(initial_thermo.wall_overlap_areas_m2)
            cell_wall_areas = np.sum(overlap, axis=1)
            liquid_weights = (
                cell_wall_areas[:liquid_cell_count]
                / np.sum(cell_wall_areas[:liquid_cell_count])
            )
            vapor_weights = (
                cell_wall_areas[liquid_cell_count:]
                / np.sum(cell_wall_areas[liquid_cell_count:])
            )
            geometry = shape.at_height(initial_thermo.liquid_height_m)
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
            partition_total = liquid_heat_W + vapor_heat_W
            if partition_total <= 0.0:
                raise ValueError("NASA heat partition must be positive")
            partition_scale = total_heat_W / partition_total
            liquid_cell_heat = tuple(
                float(value) for value in liquid_weights * liquid_heat_W * partition_scale
            )
            vapor_cell_heat = tuple(
                float(value) for value in vapor_weights * vapor_heat_W * partition_scale
            )
            observed = float(row["pressure_rise_psi_min"]) * PSI_TO_PA / 60.0
            evaluation_duration = (final_pressure_Pa - initial_pressure_Pa) / observed
            try:
                result = tank.simulate(
                    initial,
                    duration_s=evaluation_duration,
                    time_step_s=time_step_s,
                    ambient_temperature_K=initial_temperature_K,
                    fluid_heat_callback=(
                        lambda _time, _thermo, ql=liquid_cell_heat, qv=vapor_cell_heat: (ql, qv)
                    ),
                )[-1]
            except ValueError as error:
                raise ValueError(
                    f"NASA TN D-4171 test {row['test_number']} failed at "
                    f"time_step_s={time_step_s}: {error}"
                ) from error
            predicted = (result.thermo.pressure_Pa - initial_pressure_Pa) / evaluation_duration
            points.append(PressureRisePoint(
                test_number=int(row["test_number"]),
                heating_mode=row["heating_mode"],
                initial_fill_fraction=fill,
                net_average_heat_flux_W_m2=total_heat_W / shape.inner_surface_area_m2,
                observed_pressure_rise_Pa_s=observed,
                predicted_pressure_rise_Pa_s=predicted,
                predicted_over_observed=predicted / observed,
                predicted_duration_s=evaluation_duration,
            ))
            energy_partitions.append(reduce_layered_state_to_nasa_energy_partition(
                tank=tank,
                initial_state=initial,
                final_result=result,
                test_number=int(row["test_number"]),
                heating_mode=row["heating_mode"],
            ))
    expected_tests = set(range(1, 22)) if selected_tests is None else selected_tests
    if len(points) != len(expected_tests) or {point.test_number for point in points} != expected_tests:
        raise RuntimeError("NASA TN D-4171 benchmark test selection is incomplete")
    modes = sorted({point.heating_mode for point in points})
    return NasaD4171Benchmark(
        source_title="Normal Gravity Self-Pressurization of 9-Inch Diameter Spherical Liquid Hydrogen Tankage",
        source_report="NASA TN D-4171",
        source_url="https://ntrs.nasa.gov/citations/19670028965",
        model=(
            f"{liquid_cell_count}+{vapor_cell_count}-cell common-pressure tank "
            + (
                "with molecular interface conduction"
                if interface_heat_transfer_model == "conduction"
                else "with Daigle horizontal-interface natural convection"
            )
            + (
                " and equilibrium energy-jump phase change"
                if interface_phase_change_model == "equilibrium_energy_jump"
                else " and Schrage kinetic-energy-balance phase change"
            )
            + (
                " and saturated-liquid complementarity boiling"
                if distributed_boiling_model
                == "saturated_liquid_complementarity"
                else ""
            )
        ),
        assumptions=(
            "9 inch internal spherical diameter represented by two hemispherical heads",
            "tests begin at 1 atmosphere absolute pressure",
            "the comparison horizon is the reported time to nominal 100 psia",
            "Table I net wetted-wall heat enters the liquid cells by initial wall-contact area",
            "Table I net dry-wall heat enters the vapor cells by initial wall-contact area",
            "the partition is rescaled to preserve the reported total average heat input",
            (
                "axial and interface transport use CoolProp molecular conductivity without enhancement"
                if interface_heat_transfer_model == "conduction"
                else "interface heat uses Daigle et al. 2013 Eqs. 20-22 and 33-34 without fitted multipliers"
            ),
            (
                "phase change uses a saturated equilibrium energy jump"
                if interface_phase_change_model == "equilibrium_energy_jump"
                else (
                    "phase change solves Schrage kinetics and the interface "
                    f"energy balance with fixed accommodation coefficient {schrage_accommodation_coefficient:g}"
                )
            ),
            (
                "saturated liquid cells use a pressure-coupled boiling complementarity without fitted coefficients"
                if distributed_boiling_model
                == "saturated_liquid_complementarity"
                else "distributed wetted-wall boiling is disabled"
            ),
            "no parameter is fitted to the 21 pressure-rise observations",
        ),
        points=tuple(points),
        overall=_summary(points),
        by_heating_mode={
            mode: _summary(point for point in points if point.heating_mode == mode)
            for mode in modes
        },
        energy_partitions=tuple(energy_partitions),
    )


def run_nasa_d4171_layered_conduction_benchmark(
    properties: HydrogenProperties | None = None,
    time_step_s: float = 60.0,
    test_numbers: Iterable[int] | None = None,
    liquid_cell_count: int = 3,
    vapor_cell_count: int = 3,
    wall_node_count: int | None = None,
) -> NasaD4171Benchmark:
    """Run the common-pressure six-cell tank with molecular conduction only."""

    return _run_nasa_d4171_layered_benchmark(
        "conduction", properties=properties, time_step_s=time_step_s,
        test_numbers=test_numbers, liquid_cell_count=liquid_cell_count,
        vapor_cell_count=vapor_cell_count, wall_node_count=wall_node_count,
    )


def run_nasa_d4171_layered_natural_convection_benchmark(
    properties: HydrogenProperties | None = None,
    time_step_s: float = 60.0,
    test_numbers: Iterable[int] | None = None,
    liquid_cell_count: int = 3,
    vapor_cell_count: int = 3,
    wall_node_count: int | None = None,
) -> NasaD4171Benchmark:
    """Run the six-cell tank with the published Daigle interface closure."""

    return _run_nasa_d4171_layered_benchmark(
        "daigle_2013", properties=properties, time_step_s=time_step_s,
        test_numbers=test_numbers, liquid_cell_count=liquid_cell_count,
        vapor_cell_count=vapor_cell_count, wall_node_count=wall_node_count,
    )


def run_nasa_d4171_layered_schrage_benchmark(
    properties: HydrogenProperties | None = None,
    time_step_s: float = 10.0,
    test_numbers: Iterable[int] | None = None,
    liquid_cell_count: int = 3,
    vapor_cell_count: int = 3,
    wall_node_count: int | None = None,
    accommodation_coefficient: float = 0.001,
) -> NasaD4171Benchmark:
    """Run molecular interface heat transfer with coupled Schrage kinetics."""

    return _run_nasa_d4171_layered_benchmark(
        "conduction",
        interface_phase_change_model="schrage_kinetic_energy_balance",
        schrage_accommodation_coefficient=accommodation_coefficient,
        properties=properties,
        time_step_s=time_step_s,
        test_numbers=test_numbers,
        liquid_cell_count=liquid_cell_count,
        vapor_cell_count=vapor_cell_count,
        wall_node_count=wall_node_count,
    )


def run_nasa_d4171_layered_distributed_boiling_benchmark(
    properties: HydrogenProperties | None = None,
    time_step_s: float = 10.0,
    test_numbers: Iterable[int] | None = None,
    liquid_cell_count: int = 3,
    vapor_cell_count: int = 3,
    wall_node_count: int | None = None,
) -> NasaD4171Benchmark:
    """Run molecular transport with coefficient-free saturated-cell boiling."""

    return _run_nasa_d4171_layered_benchmark(
        "conduction",
        distributed_boiling_model="saturated_liquid_complementarity",
        properties=properties,
        time_step_s=time_step_s,
        test_numbers=test_numbers,
        liquid_cell_count=liquid_cell_count,
        vapor_cell_count=vapor_cell_count,
        wall_node_count=wall_node_count,
    )


def run_nasa_d4171_layered_wall_circulation_benchmark(
    properties: HydrogenProperties | None = None,
    time_step_s: float = 60.0,
    test_numbers: Iterable[int] | None = None,
    liquid_cell_count: int = 3,
    vapor_cell_count: int = 3,
    wall_node_count: int | None = None,
) -> NasaD4171Benchmark:
    """Run the layered tank with Daigle interface and wall circulation.

    NASA Table I reports net wetted- and dry-wall heat fluxes rather than wall
    temperatures.  The benchmark therefore uses Daigle Appendix A's uniform-
    heat-flux form for the vertical boundary-layer circulation.  Coefficients
    are the published values and are not calibrated against the 21 tests.
    """

    return _run_nasa_d4171_layered_benchmark(
        "daigle_2013",
        vertical_wall_circulation_model=(
            "daigle_2013_uniform_heat_flux_reduced"
        ),
        properties=properties,
        time_step_s=time_step_s,
        test_numbers=test_numbers,
        liquid_cell_count=liquid_cell_count,
        vapor_cell_count=vapor_cell_count,
        wall_node_count=wall_node_count,
    )
