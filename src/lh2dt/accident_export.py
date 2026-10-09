"""Small native accident-history helpers for fixed-source outlet models.

These helpers are intentionally bounded.  A valve or vent stack does not own
the upstream inventory, so the caller must provide an explicit available-mass
ledger and termination basis.  The helper never turns a normal-state
``evaluate`` result into an inventory-depletion history.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from .properties import HydrogenProperties


def export_fixed_source_accident_history(
    request: Mapping[str, Any],
    *,
    properties: HydrogenProperties,
    provider_model: str,
    evaluate: Callable[[Any, Any, float], Any],
    advance_opening: Callable[[float], float],
    opening_area_m2: float,
    area_provenance: str,
) -> Mapping[str, Any]:
    """Evaluate a declared fixed-source atmospheric outlet history."""
    if not isinstance(request, Mapping):
        raise ValueError("accident export request must be a mapping")

    def text(name: str, *, required: bool = True, default: str | None = None) -> str:
        value = request.get(name, default)
        if not isinstance(value, str) or (required and not value.strip()):
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    def number(name: str, default: float | None = None, *, positive: bool = True) -> float:
        value = request.get(name, default)
        if isinstance(value, bool) or value is None:
            raise ValueError(f"{name} must be numeric")
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
        if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
            comparator = "positive" if positive else "non-negative"
            raise ValueError(f"{name} must be finite and {comparator}")
        return result

    event_id = text("event_id")
    component_id = text("component_id")
    port_id = text("port_id")
    source_pressure = number("source_pressure_pa_abs", request.get("upstream_pressure_pa_abs"))
    source_temperature = number("source_temperature_k", request.get("upstream_temperature_k"))
    ambient_pressure = number("ambient_pressure_pa_abs", 101325.0)
    ambient_temperature = number("ambient_temperature_k", 288.15)
    opening_diameter = number("failure_opening_diameter_m")
    horizon = number("horizon_s")
    time_step = number("time_step_s", min(0.05, horizon))
    available_mass = number("available_mass_kg")
    termination_basis = text("termination_basis")
    termination_provenance = text("termination_provenance")
    release_end = request.get("release_end_s")
    if release_end is not None:
        release_end_s = number("release_end_s")
        if release_end_s > horizon:
            raise ValueError("release_end_s must not exceed horizon_s")
    else:
        release_end_s = horizon
    if source_pressure <= ambient_pressure:
        raise ValueError("source pressure must exceed ambient pressure for an outward accident")
    declared_area = math.pi * opening_diameter**2 / 4.0
    if not math.isclose(declared_area, opening_area_m2, rel_tol=1.0e-6, abs_tol=1.0e-12):
        raise ValueError("failure opening diameter disagrees with the native outlet area")

    inlet = properties.from_pT(source_pressure, source_temperature)
    if inlet.phase not in {"gas", "supercritical_gas", "supercritical"}:
        raise ValueError("fixed-source accident export requires a gas-like source state")
    outlet_boundary = properties.from_pT(ambient_pressure, ambient_temperature)
    steps: list[dict[str, Any]] = []
    cumulative_mass = 0.0
    cumulative_enthalpy = 0.0
    elapsed = 0.0
    choked_count = 0
    while elapsed < release_end_s - 1.0e-12:
        dt = min(time_step, release_end_s - elapsed)
        opening = float(advance_opening(dt))
        if not math.isfinite(opening) or not 0.0 <= opening <= 1.0:
            raise ValueError("native outlet returned an invalid opening")
        hydraulic = evaluate(inlet, outlet_boundary, opening)
        mass_flow = float(getattr(hydraulic, "mass_flow_kg_s"))
        if not math.isfinite(mass_flow) or mass_flow < 0.0:
            raise ValueError("native outlet returned an invalid outward mass flow")
        exit_state = properties.from_ph(
            ambient_pressure, float(getattr(hydraulic, "outlet_specific_enthalpy_J_kg"))
        )
        if exit_state.phase not in {"gas", "supercritical_gas", "supercritical"}:
            raise ValueError("accident outlet is not gas-like at the declared ambient boundary")
        enthalpy = float(getattr(hydraulic, "outlet_specific_enthalpy_J_kg"))
        effective_area = declared_area * opening
        density = float(exit_state.density_kg_m3)
        steps.append({
            "time_s": elapsed,
            "mass_flow_kg_s": mass_flow,
            "pressure_pa_abs": ambient_pressure,
            "specific_enthalpy_j_kg": enthalpy,
            "temperature_k": float(exit_state.temperature_K),
            "density_kg_m3": density,
            "effective_area_m2": effective_area,
            "velocity_m_s": (
                mass_flow / (density * effective_area)
                if mass_flow > 0.0 and effective_area > 0.0 else 0.0
            ),
            "area_provenance": area_provenance,
            "velocity_origin": "mass_continuity_from_declared_area",
            "provider_source_state": {
                "source_pressure_pa_abs": float(inlet.pressure_Pa),
                "source_temperature_k": float(inlet.temperature_K),
                "source_density_kg_m3": float(inlet.density_kg_m3),
                "source_specific_enthalpy_j_kg": float(inlet.specific_enthalpy_J_kg),
                "opening": opening,
                "choked": bool(getattr(hydraulic, "choked", False)),
            },
        })
        if bool(getattr(hydraulic, "choked", False)):
            choked_count += 1
        cumulative_mass += mass_flow * dt
        cumulative_enthalpy += mass_flow * dt * enthalpy
        if cumulative_mass > available_mass + 1.0e-12:
            raise ValueError("fixed-source accident history exceeds declared available mass")
        elapsed += dt

    return {
        "provider_export_schema": "prism.external_accident_history.v1",
        "provider_model": provider_model,
        "provider_source_digest": request.get("provider_source_digest"),
        "provider_state_snapshot_digest": request.get("state_snapshot_digest"),
        "data": {
            "event_kind": "accidental_leak",
            "event_id": event_id,
            "component_id": component_id,
            "port_id": port_id,
            "duration_s": elapsed,
            "available_mass_kg": available_mass,
            "cumulative_mass_out_kg": cumulative_mass,
            "cumulative_static_enthalpy_out_j": cumulative_enthalpy,
            "termination_basis": termination_basis,
            "termination_provenance": termination_provenance,
            "fluid": "Hydrogen",
            "steps": steps,
            "provider_meta": {
                "native_inventory_owner": "upstream_provider_declared",
                "initial_source_phase": inlet.phase,
                "time_step_s": time_step,
                "valve": {
                    "area_m2": declared_area,
                    "final_opening": float(steps[-1]["provider_source_state"].get("opening", 0.0)) if steps else 0.0,
                    "choked_intervals": choked_count,
                },
            },
        },
    }


__all__ = ["export_fixed_source_accident_history"]
