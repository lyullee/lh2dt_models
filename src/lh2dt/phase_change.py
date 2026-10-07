"""Interfacial phase-change closures with explicit physical parameters."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SchrageMassFlux:
    """Signed Schrage interfacial mass flux and its two pressure terms."""

    mass_flux_kg_m2_s: float
    accommodation_coefficient: float
    interface_pressure_term_Pa_per_sqrt_K: float
    vapor_pressure_term_Pa_per_sqrt_K: float


@dataclass(frozen=True)
class KineticGasFilmHeatTransfer:
    """Kinetic-theory gas-side heat-transfer coefficient.

    The coefficient treats gas molecules striking the interface as a
    Maxwellian impingement flux.  The energy accommodation coefficient is
    explicit so this closure can be replaced by a measured or literature
    value without changing the tank equations.  It is independent of the
    Schrage mass accommodation coefficient.
    """

    heat_transfer_coefficient_W_m2K: float
    impingement_mass_flux_kg_m2_s: float
    energy_accommodation_coefficient: float


def schrage_mass_flux(
    *,
    interface_pressure_Pa: float,
    interface_temperature_K: float,
    vapor_pressure_Pa: float,
    vapor_temperature_K: float,
    specific_gas_constant_J_kgK: float,
    accommodation_coefficient: float = 0.001,
) -> SchrageMassFlux:
    """Evaluate the near-equilibrium Schrage mass-flux equation.

    Positive flux denotes evaporation from the interface into the vapor.  The
    default accommodation coefficient is the fixed value used in most cases
    of Kartuzova and Kassemi's NASA K-Site CFD study; it is a replaceable
    literature default, not a calibrated tank parameter.  This kinetic closure
    can be used independently or coupled to a separately solved interface
    state and energy balance.
    """

    positive = {
        "interface_pressure_Pa": interface_pressure_Pa,
        "interface_temperature_K": interface_temperature_K,
        "vapor_pressure_Pa": vapor_pressure_Pa,
        "vapor_temperature_K": vapor_temperature_K,
        "specific_gas_constant_J_kgK": specific_gas_constant_J_kgK,
    }
    for name, raw in positive.items():
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    accommodation = float(accommodation_coefficient)
    if not math.isfinite(accommodation) or not 0.0 < accommodation <= 1.0:
        raise ValueError(
            "accommodation_coefficient must be finite and in the interval (0, 1]"
        )

    interface_term = float(interface_pressure_Pa) / math.sqrt(
        float(interface_temperature_K)
    )
    vapor_term = float(vapor_pressure_Pa) / math.sqrt(float(vapor_temperature_K))
    kinetic_factor = (
        2.0 * accommodation / (2.0 - accommodation)
        / math.sqrt(2.0 * math.pi * float(specific_gas_constant_J_kgK))
    )
    return SchrageMassFlux(
        mass_flux_kg_m2_s=kinetic_factor * (interface_term - vapor_term),
        accommodation_coefficient=accommodation,
        interface_pressure_term_Pa_per_sqrt_K=interface_term,
        vapor_pressure_term_Pa_per_sqrt_K=vapor_term,
    )


def kinetic_gas_film_heat_transfer_coefficient(
    *,
    pressure_Pa: float,
    temperature_K: float,
    isobaric_heat_capacity_J_kgK: float,
    specific_gas_constant_J_kgK: float,
    energy_accommodation_coefficient: float = 1.0,
) -> KineticGasFilmHeatTransfer:
    """Return a gas-side interfacial heat-transfer coefficient.

    A molecular impingement flux ``p/sqrt(2*pi*R*T)`` carries the sensible
    energy ``cp*(T_bulk-T_interface)`` per unit mass when the collision has
    energy accommodation.  Therefore ``h = alpha * j_m * cp``.  This is a
    kinetic-theory closure, not a fit to facility histories; it is intended
    for the gas-side film only and does not replace the Schrage mass flux.
    """

    positive = {
        "pressure_Pa": pressure_Pa,
        "temperature_K": temperature_K,
        "isobaric_heat_capacity_J_kgK": isobaric_heat_capacity_J_kgK,
        "specific_gas_constant_J_kgK": specific_gas_constant_J_kgK,
    }
    for name, raw in positive.items():
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    accommodation = float(energy_accommodation_coefficient)
    if not math.isfinite(accommodation) or not 0.0 < accommodation <= 1.0:
        raise ValueError(
            "energy_accommodation_coefficient must be finite and in the interval (0, 1]"
        )
    impingement = float(pressure_Pa) / math.sqrt(
        2.0 * math.pi * float(specific_gas_constant_J_kgK) * float(temperature_K)
    )
    coefficient = (
        accommodation
        * impingement
        * float(isobaric_heat_capacity_J_kgK)
    )
    return KineticGasFilmHeatTransfer(
        heat_transfer_coefficient_W_m2K=coefficient,
        impingement_mass_flux_kg_m2_s=impingement,
        energy_accommodation_coefficient=accommodation,
    )
