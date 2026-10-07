"""Decomposed heat leak through a vacuum-insulated cryogenic boundary.

The model keeps the distributed paths and discrete penetrations explicit:
radiation, MLI/solid conduction, distributed support conduction,
residual-gas conduction, and separately declared local heat bridges.  It is
a boundary model rather than a fitted overall-UA correlation.  The gas
coefficient and construction conductances therefore remain declared inputs
until a vacuum/MLI performance test, geometry/material calculation, or
manufacturer heat-leak value exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from dataclasses import replace
from collections.abc import Callable, Mapping, Sequence


STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class LocalizedHeatPath:
    """One explicitly declared warm-to-cold thermal penetration.

    This represents a nozzle, support, instrument stem, or other discrete
    penetration through the vacuum jacket.  ``conductance_W_K`` is a
    construction input (geometry/material or a manufacturer value), not an
    inverse-fit coefficient.  Several paths are kept separate so an authoring
    tool can add, remove, or resize a physical penetration without changing a
    lumped tank ``UA``.
    """

    path_id: str
    conductance_W_K: float

    def __post_init__(self) -> None:
        if not isinstance(self.path_id, str) or not self.path_id.strip():
            raise ValueError("path_id must be a non-empty string")
        if _finite(self.conductance_W_K, "conductance_W_K") < 0.0:
            raise ValueError("conductance_W_K must be non-negative")

    def heat_flow(self, warm_temperature_K: float, cold_temperature_K: float) -> float:
        warm = _finite(warm_temperature_K, "warm_temperature_K")
        cold = _finite(cold_temperature_K, "cold_temperature_K")
        if warm <= 0.0 or cold <= 0.0:
            raise ValueError("boundary temperatures must be positive")
        return self.conductance_W_K * (warm - cold)


@dataclass(frozen=True)
class VacuumInsulationParameters:
    """Geometry and construction closures for one insulation zone.

    ``effective_emissivity`` represents the complete shield stack as viewed
    between the warm and cold boundaries.  ``residual_gas_coefficient`` has
    units W/(m2 K Pa) and is intended for the molecular-flow range where gas
    conduction is approximately proportional to pressure.  The two solid
    conductances separate distributed MLI/spacer contact from the distributed
    support path.  Discrete necks, nozzles, and instrument penetrations are
    supplied through ``localized_heat_paths``.
    """

    area_m2: float
    effective_emissivity: float
    mli_solid_conductance_W_K: float
    support_conductance_W_K: float
    residual_gas_coefficient_W_m2K_Pa: float
    maximum_valid_vacuum_pressure_Pa: float
    localized_heat_paths: tuple[LocalizedHeatPath, ...] = ()

    def __post_init__(self) -> None:
        area = _finite(self.area_m2, "area_m2")
        emissivity = _finite(self.effective_emissivity, "effective_emissivity")
        if area <= 0.0:
            raise ValueError("area_m2 must be positive")
        if not 0.0 <= emissivity <= 1.0:
            raise ValueError("effective_emissivity must be in [0, 1]")
        for name in (
            "mli_solid_conductance_W_K",
            "support_conductance_W_K",
            "residual_gas_coefficient_W_m2K_Pa",
        ):
            if _finite(getattr(self, name), name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if _finite(
            self.maximum_valid_vacuum_pressure_Pa,
            "maximum_valid_vacuum_pressure_Pa",
        ) <= 0.0:
            raise ValueError("maximum_valid_vacuum_pressure_Pa must be positive")
        paths = tuple(self.localized_heat_paths)
        if any(not isinstance(path, LocalizedHeatPath) for path in paths):
            raise ValueError("localized_heat_paths must contain LocalizedHeatPath values")
        path_ids = [path.path_id for path in paths]
        if len(path_ids) != len(set(path_ids)):
            raise ValueError("localized_heat_paths path_id values must be unique")
        object.__setattr__(self, "localized_heat_paths", paths)


@dataclass(frozen=True)
class VacuumInsulationHeatFlow:
    """Signed heat flow into the cold boundary and its parallel paths."""

    total_heat_into_cold_W: float
    radiation_heat_W: float
    mli_solid_heat_W: float
    support_heat_W: float
    residual_gas_heat_W: float
    effective_UA_W_K: float
    vacuum_pressure_Pa: float
    localized_heat_W: float = 0.0


class VacuumInsulation:
    """Evaluate a vacuum-jacket heat leak without collapsing its causes."""

    def __init__(self, parameters: VacuumInsulationParameters) -> None:
        self.parameters = parameters

    def evaluate(
        self,
        warm_temperature_K: float,
        cold_temperature_K: float,
        vacuum_pressure_Pa: float,
    ) -> VacuumInsulationHeatFlow:
        warm = _finite(warm_temperature_K, "warm_temperature_K")
        cold = _finite(cold_temperature_K, "cold_temperature_K")
        pressure = _finite(vacuum_pressure_Pa, "vacuum_pressure_Pa")
        if warm <= 0.0 or cold <= 0.0:
            raise ValueError("boundary temperatures must be positive")
        if pressure < 0.0:
            raise ValueError("vacuum_pressure_Pa must be non-negative")

        p = self.parameters
        if pressure > p.maximum_valid_vacuum_pressure_Pa:
            raise ValueError(
                "vacuum_pressure_Pa exceeds the declared molecular-flow validity limit"
            )
        delta_temperature = warm - cold
        radiation = (
            STEFAN_BOLTZMANN_W_M2_K4
            * p.area_m2
            * p.effective_emissivity
            * (warm**4 - cold**4)
        )
        mli_solid = p.mli_solid_conductance_W_K * delta_temperature
        support = p.support_conductance_W_K * delta_temperature
        residual_gas = (
            p.residual_gas_coefficient_W_m2K_Pa
            * p.area_m2
            * pressure
            * delta_temperature
        )
        localized_heat = sum(
            path.heat_flow(warm, cold) for path in p.localized_heat_paths
        )
        total = radiation + mli_solid + support + residual_gas + localized_heat
        effective_ua = 0.0 if delta_temperature == 0.0 else total / delta_temperature
        return VacuumInsulationHeatFlow(
            total_heat_into_cold_W=total,
            radiation_heat_W=radiation,
            mli_solid_heat_W=mli_solid,
            support_heat_W=support,
            residual_gas_heat_W=residual_gas,
            effective_UA_W_K=effective_ua,
            vacuum_pressure_Pa=pressure,
            localized_heat_W=localized_heat,
        )

    def heat_callback(self, vacuum_pressure_Pa: float):
        """Return a stratified-tank wall callback for this insulation zone.

        The callback returns a single total heat flow.  A caller with separate
        lower and upper wall zones should instantiate one boundary per zone
        and return both values from its tank callback.
        """

        pressure = _finite(vacuum_pressure_Pa, "vacuum_pressure_Pa")
        if pressure < 0.0:
            raise ValueError("vacuum_pressure_Pa must be non-negative")

        def callback(warm_temperature_K: float, cold_temperature_K: float) -> float:
            return self.evaluate(
                warm_temperature_K,
                cold_temperature_K,
                pressure,
            ).total_heat_into_cold_W

        return callback

    def layered_wall_heat_distribution(
        self,
        warm_temperature_K: float,
        cold_temperatures_K: Sequence[float],
        vacuum_pressure_Pa: float,
        area_fractions: Sequence[float] | None = None,
        localized_path_fractions: Mapping[str, Sequence[float]] | None = None,
    ) -> tuple[float, ...]:
        """Distribute the decomposed boundary heat into tank wall nodes.

        The total insulation area is partitioned by explicit geometric area
        fractions.  Distributed radiation, MLI/solid, support, and residual
        gas paths are evaluated on each fraction using that node's wall
        temperature.  Discrete penetrations are distributed by the supplied
        ``localized_path_fractions``; when omitted they follow the same area
        fractions.  No overall-UA or time-series fit is introduced.
        """

        cold = tuple(float(value) for value in cold_temperatures_K)
        if not cold:
            raise ValueError("cold_temperatures_K cannot be empty")
        if any(not math.isfinite(value) or value <= 0.0 for value in cold):
            raise ValueError("cold_temperatures_K must contain finite positive values")
        if area_fractions is None:
            fractions = tuple(1.0 / len(cold) for _ in cold)
        else:
            fractions = tuple(float(value) for value in area_fractions)
            if len(fractions) != len(cold):
                raise ValueError("area_fractions must match cold_temperatures_K")
            if any(not math.isfinite(value) or value < 0.0 for value in fractions):
                raise ValueError("area_fractions must be finite and non-negative")
            if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("area_fractions must sum to one")

        paths = self.parameters.localized_heat_paths
        path_fractions: dict[str, tuple[float, ...]] = {}
        for path in paths:
            supplied = (
                localized_path_fractions.get(path.path_id)
                if localized_path_fractions is not None
                else None
            )
            values = fractions if supplied is None else tuple(float(value) for value in supplied)
            if len(values) != len(cold):
                raise ValueError(
                    f"localized path fractions for {path.path_id} must match wall nodes"
                )
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(
                    f"localized path fractions for {path.path_id} must be finite and non-negative"
                )
            if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(
                    f"localized path fractions for {path.path_id} must sum to one"
                )
            path_fractions[path.path_id] = values
        if localized_path_fractions is not None:
            unknown = set(localized_path_fractions) - set(path_fractions)
            if unknown:
                raise ValueError(
                    "localized_path_fractions contains unknown paths: "
                    + ", ".join(sorted(unknown))
                )

        warm = _finite(warm_temperature_K, "warm_temperature_K")
        if warm <= 0.0:
            raise ValueError("warm_temperature_K must be positive")
        result: list[float] = []
        for index, (fraction, cold_temperature) in enumerate(zip(fractions, cold)):
            local_paths = tuple(
                LocalizedHeatPath(
                    path.path_id,
                    path.conductance_W_K * path_fractions[path.path_id][index],
                )
                for path in paths
            )
            local_parameters = replace(
                self.parameters,
                area_m2=self.parameters.area_m2 * fraction,
                mli_solid_conductance_W_K=(
                    self.parameters.mli_solid_conductance_W_K * fraction
                ),
                support_conductance_W_K=(
                    self.parameters.support_conductance_W_K * fraction
                ),
                localized_heat_paths=local_paths,
            )
            local = VacuumInsulation(local_parameters).evaluate(
                warm,
                cold_temperature,
                vacuum_pressure_Pa,
            )
            result.append(float(local.total_heat_into_cold_W))
        return tuple(result)


def make_layered_wall_heat_callback(
    insulation: VacuumInsulation,
    vacuum_pressure_Pa: float,
    warm_temperature_K: float | Callable[[float], float],
    area_fractions: Sequence[float] | None = None,
    localized_path_fractions: Mapping[str, Sequence[float]] | None = None,
):
    """Create a ``LayeredTank`` wall callback from a vacuum boundary.

    ``warm_temperature_K`` may be constant or a function of simulation time.
    The callback consumes ``LayeredThermoState.wall_temperatures_K`` and
    returns one heat input per wall node.  Explicit area and penetration
    allocations keep the boundary editable and traceable.
    """

    pressure = _finite(vacuum_pressure_Pa, "vacuum_pressure_Pa")
    if pressure < 0.0:
        raise ValueError("vacuum_pressure_Pa must be non-negative")

    def callback(time_s: float, thermo) -> tuple[float, ...]:
        warm = warm_temperature_K(time_s) if callable(warm_temperature_K) else warm_temperature_K
        return insulation.layered_wall_heat_distribution(
            warm,
            thermo.wall_temperatures_K,
            pressure,
            area_fractions=area_fractions,
            localized_path_fractions=localized_path_fractions,
        )

    return callback
