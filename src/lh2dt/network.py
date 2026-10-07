"""Pressure/enthalpy junction solver for replaceable two-port components."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol

import numpy as np
from scipy.optimize import least_squares

from .properties import HydrogenProperties, ThermoState
from .contracts import (
    validate_thermal_link_result,
    validate_transmitted_thermo_state,
    validate_two_port_result,
)


class TwoPortResult(Protocol):
    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float


class TwoPortComponent(Protocol):
    def evaluate(self, left: ThermoState, right: ThermoState) -> TwoPortResult: ...


@dataclass(frozen=True)
class BoundaryNode:
    name: str
    state: ThermoState


@dataclass(frozen=True)
class JunctionNode:
    name: str
    initial_pressure_Pa: float
    initial_specific_enthalpy_J_kg: float


@dataclass(frozen=True)
class NetworkLink:
    name: str
    left: str
    right: str
    component: TwoPortComponent
    source_port_id: str | None = None
    target_port_id: str | None = None


@dataclass(frozen=True)
class LinkSolution:
    name: str
    left: str
    right: str
    mass_flow_kg_s: float
    transmitted_specific_enthalpy_J_kg: float
    momentum_residual_Pa: float | None = None
    source_port_id: str | None = None
    target_port_id: str | None = None

    @property
    def upstream_port_id(self) -> str | None:
        """Return the port from which the signed material flow originates."""

        return self.source_port_id if self.mass_flow_kg_s >= 0.0 else self.target_port_id


@dataclass(frozen=True)
class NetworkSolution:
    """Solved link states and an explicit conservation-aware success verdict.

    `pressure_solver_success` requires optimizer termination, enthalpy
    iteration convergence, and junction mass/energy closure.  The separate
    flags distinguish a numerical optimizer stop from a physical solution.
    """

    node_states: dict[str, ThermoState]
    link_flows: dict[str, LinkSolution]
    node_mass_residual_kg_s: dict[str, float]
    node_energy_residual_W: dict[str, float]
    pressure_solver_success: bool
    enthalpy_iterations: int
    link_momentum_residuals_Pa: dict[str, float] = field(default_factory=dict)
    pressure_optimizer_success: bool = True
    enthalpy_solver_success: bool = True
    junction_conservation_success: bool = True
    node_mass_closure_tolerance_kg_s: dict[str, float] = field(default_factory=dict)
    node_energy_closure_tolerance_W: dict[str, float] = field(default_factory=dict)

    @property
    def max_abs_momentum_residual_Pa(self) -> float | None:
        """Return the largest reported link momentum residual.

        Generic components may not expose a momentum closure equation.  Such
        links are intentionally absent from ``link_momentum_residuals_Pa``
        instead of being assigned a fabricated zero, so callers can
        distinguish a proven closure from an unavailable one.
        """

        if not self.link_momentum_residuals_Pa:
            return None
        return max(abs(value) for value in self.link_momentum_residuals_Pa.values())


class SteadyNetwork:
    """Solve zero-storage junctions by mass and enthalpy conservation.

    Dynamic vessels remain boundary nodes during one network solve; after the
    flows are known, their own conservation equations advance their states.
    This separation allows a tank, pipe, or valve model to be replaced without
    changing the connection algorithm.
    """

    def __init__(self, properties: HydrogenProperties | None = None) -> None:
        self.properties = properties or HydrogenProperties()
        self.boundaries: dict[str, BoundaryNode] = {}
        self.junctions: dict[str, JunctionNode] = {}
        self.links: list[NetworkLink] = []
        self._port_links: dict[str, str] = {}
        # Dynamic simulations solve the same topology repeatedly while the
        # boundary states change continuously.  Retain the last converged
        # junction state as a numerical initial guess; this does not alter the
        # governing equations or invent a physical state when no prior solve
        # exists.
        self._last_junction_pressures: dict[str, float] | None = None
        self._last_junction_enthalpies: dict[str, float] | None = None

    def add_boundary(self, name: str, state: ThermoState) -> None:
        self._require_new_name(name)
        self.boundaries[name] = BoundaryNode(name, state)

    def set_boundary_state(self, name: str, state: ThermoState) -> None:
        """Replace one boundary state while preserving topology and port IDs.

        Dynamic storage models use the same network topology at every time
        step.  Updating a boundary state explicitly keeps that coupling
        visible and avoids rebuilding components or inferring a state from a
        measurement.
        """
        if name not in self.boundaries:
            raise KeyError(f"Unknown boundary node: {name}")
        self.boundaries[name] = BoundaryNode(name, state)

    def set_boundary_states(self, states: dict[str, ThermoState]) -> None:
        for name, state in states.items():
            self.set_boundary_state(name, state)

    def add_junction(self, name: str, initial_pressure_Pa: float, initial_specific_enthalpy_J_kg: float) -> None:
        self._require_new_name(name)
        if initial_pressure_Pa <= 0.0:
            raise ValueError("initial_pressure_Pa must be positive")
        self.junctions[name] = JunctionNode(name, initial_pressure_Pa, initial_specific_enthalpy_J_kg)

    def add_link(
        self,
        name: str,
        left: str,
        right: str,
        component: TwoPortComponent,
        *,
        source_port_id: str | None = None,
        target_port_id: str | None = None,
    ) -> None:
        nodes = set(self.boundaries) | set(self.junctions)
        if left not in nodes or right not in nodes:
            raise KeyError("Both link endpoints must already exist")
        if left == right:
            raise ValueError("A link cannot connect a node to itself")
        if (source_port_id is None) != (target_port_id is None):
            raise ValueError("source_port_id and target_port_id must be supplied together")
        if source_port_id is not None and (
            not str(source_port_id).strip() or not str(target_port_id).strip()
        ):
            raise ValueError("port IDs must be non-empty when supplied")
        if source_port_id is not None and source_port_id == target_port_id:
            raise ValueError("A link cannot use the same source and target port ID")
        if source_port_id is not None:
            for port_id in (source_port_id, target_port_id):
                if port_id in self._port_links:
                    raise ValueError(
                        f"Port ID {port_id} is already assigned to link {self._port_links[port_id]}"
                    )
        if any(link.name == name for link in self.links):
            raise ValueError(f"Duplicate link name: {name}")
        self.links.append(NetworkLink(
            name,
            left,
            right,
            component,
            None if source_port_id is None else str(source_port_id),
            None if target_port_id is None else str(target_port_id),
        ))
        if source_port_id is not None:
            self._port_links[str(source_port_id)] = name
            self._port_links[str(target_port_id)] = name

    def _require_new_name(self, name: str) -> None:
        if not name or name in self.boundaries or name in self.junctions:
            raise ValueError(f"Invalid or duplicate node name: {name}")

    def _states(self, pressures: dict[str, float], enthalpies: dict[str, float]) -> dict[str, ThermoState]:
        states = {name: node.state for name, node in self.boundaries.items()}
        for name in self.junctions:
            states[name] = self.properties.from_ph(pressures[name], enthalpies[name])
        return states

    def _flows(self, states: dict[str, ThermoState]) -> dict[str, LinkSolution]:
        flows: dict[str, LinkSolution] = {}
        for link in self.links:
            result = link.component.evaluate(states[link.left], states[link.right])
            if all(
                hasattr(result, field)
                for field in ("thermal_power_W", "hydraulic_outlet_specific_enthalpy_J_kg")
            ):
                validate_thermal_link_result(result, context=f"link {link.name}")
            else:
                validate_two_port_result(result, context=f"link {link.name}")
            flows[link.name] = LinkSolution(
                name=link.name,
                left=link.left,
                right=link.right,
                mass_flow_kg_s=float(result.mass_flow_kg_s),
                transmitted_specific_enthalpy_J_kg=float(result.outlet_specific_enthalpy_J_kg),
                momentum_residual_Pa=(
                    None
                    if getattr(result, "momentum_residual_Pa", None) is None
                    else float(getattr(result, "momentum_residual_Pa"))
                ),
                source_port_id=link.source_port_id,
                target_port_id=link.target_port_id,
            )
        return flows

    def _validate_transmitted_states(
        self,
        states: dict[str, ThermoState],
        flows: dict[str, LinkSolution],
    ) -> None:
        """Validate non-zero link enthalpies against the shared EOS.

        This is kept outside the nonlinear residual callback so solver
        iterations do not add a second CoolProp inversion for every trial
        point.  The final link solution still cannot expose an enthalpy that
        is disconnected from the pressure state at its downstream endpoint.
        """

        for flow in flows.values():
            downstream = (
                states[flow.right]
                if flow.mass_flow_kg_s >= 0.0
                else states[flow.left]
            )
            validate_transmitted_thermo_state(
                mass_flow_kg_s=flow.mass_flow_kg_s,
                outlet_specific_enthalpy_J_kg=flow.transmitted_specific_enthalpy_J_kg,
                downstream_pressure_Pa=downstream.pressure_Pa,
                properties=self.properties,
                context=f"link {flow.name}",
            )

    def _mass_residual(self, flows: dict[str, LinkSolution], node: str) -> float:
        residual = 0.0
        for flow in flows.values():
            if flow.right == node:
                residual += flow.mass_flow_kg_s
            if flow.left == node:
                residual -= flow.mass_flow_kg_s
        return residual

    def _pressure_mass_residuals(
        self,
        states: dict[str, ThermoState],
        junction_names: tuple[str, ...],
    ) -> np.ndarray:
        """Evaluate only hydraulic flow during the pressure root solve.

        A thermal link's downstream heat calculation cannot affect its
        hydraulic flow for the supplied endpoint states.  Components without
        an explicit mass-only method retain the complete evaluation path.
        """

        residuals = {name: 0.0 for name in junction_names}
        for link in self.links:
            left = states[link.left]
            right = states[link.right]
            quick = getattr(link.component, "evaluate_mass_flow", None)
            if callable(quick):
                mass_flow = float(quick(left, right))
            else:
                result = link.component.evaluate(left, right)
                validate_two_port_result(result, context=f"link {link.name}")
                mass_flow = float(result.mass_flow_kg_s)
            if not math.isfinite(mass_flow):
                raise ValueError(f"link {link.name} returned non-finite mass flow")
            if link.left in residuals:
                residuals[link.left] -= mass_flow
            if link.right in residuals:
                residuals[link.right] += mass_flow
        return np.asarray([residuals[name] for name in junction_names], dtype=float)

    def _check_mass_only_contract(
        self,
        states: dict[str, ThermoState],
        flows: dict[str, LinkSolution],
    ) -> None:
        """Reject a fast hydraulic path that disagrees with full evaluation."""

        for link in self.links:
            quick = getattr(link.component, "evaluate_mass_flow", None)
            if not callable(quick):
                continue
            mass_flow = float(quick(states[link.left], states[link.right]))
            full_flow = flows[link.name].mass_flow_kg_s
            if not math.isfinite(mass_flow) or not math.isclose(
                mass_flow, full_flow, rel_tol=1.0e-10, abs_tol=1.0e-12
            ):
                raise ValueError(f"link {link.name} mass-only flow disagrees with full evaluation")

    def solve(
        self,
        relative_tolerance: float = 1e-8,
        max_enthalpy_iterations: int = 50,
        max_pressure_nfev: int = 300,
        *,
        mass_closure_absolute_tolerance_kg_s: float = 1.0e-7,
        energy_closure_absolute_tolerance_W: float = 1.0e-2,
        closure_relative_tolerance: float = 1.0e-6,
    ) -> NetworkSolution:
        """Solve junction states and verify signed mass/energy conservation.

        Closure tolerances are numerical acceptance limits, independent of
        equipment ratings or measured-data accuracy targets.  The energy
        tolerance includes the energy carried by the permitted mass residual,
        so the same tiny transport error is not counted twice.  A least-squares
        stop with material irreducible residual is reported as unsuccessful.
        """
        if (
            not isinstance(max_pressure_nfev, int)
            or isinstance(max_pressure_nfev, bool)
            or max_pressure_nfev < 1
        ):
            raise ValueError("max_pressure_nfev must be a positive integer")
        for name, value in (
            ("mass_closure_absolute_tolerance_kg_s", mass_closure_absolute_tolerance_kg_s),
            ("energy_closure_absolute_tolerance_W", energy_closure_absolute_tolerance_W),
            ("closure_relative_tolerance", closure_relative_tolerance),
        ):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite and nonnegative")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite and nonnegative") from exc
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not self.boundaries:
            raise ValueError("At least one boundary node is required")
        if not self.junctions:
            states = {name: node.state for name, node in self.boundaries.items()}
            flows = self._flows(states)
            self._validate_transmitted_states(states, flows)
            return NetworkSolution(
                states,
                flows,
                {},
                {},
                True,
                0,
                self._momentum_residuals(flows),
            )
        junction_names = tuple(self.junctions)
        pressure_min = max(1.0e3, min(node.state.pressure_Pa for node in self.boundaries.values()) * 0.2)
        pressure_max = max(node.state.pressure_Pa for node in self.boundaries.values()) * 5.0
        if (
            self._last_junction_pressures is not None
            and self._last_junction_enthalpies is not None
            and set(self._last_junction_pressures) == set(junction_names)
            and set(self._last_junction_enthalpies) == set(junction_names)
        ):
            pressures = {
                name: float(np.clip(self._last_junction_pressures[name], pressure_min, pressure_max))
                for name in junction_names
            }
            enthalpies = {
                name: float(self._last_junction_enthalpies[name])
                for name in junction_names
            }
        else:
            pressures = {name: self.junctions[name].initial_pressure_Pa for name in junction_names}
            enthalpies = {name: self.junctions[name].initial_specific_enthalpy_J_kg for name in junction_names}
        solver_success = False
        enthalpy_converged = False
        iteration = 0
        for iteration in range(1, max_enthalpy_iterations + 1):
            x0 = np.array([
                np.clip(pressures[name], pressure_min, pressure_max)
                for name in junction_names
            ])

            def residual(x: np.ndarray) -> np.ndarray:
                local_p = dict(zip(junction_names, x))
                states = self._states(local_p, enthalpies)
                return self._pressure_mass_residuals(states, junction_names)

            result = least_squares(
                residual,
                x0,
                bounds=(np.full_like(x0, pressure_min), np.full_like(x0, pressure_max)),
                xtol=relative_tolerance,
                ftol=relative_tolerance,
                # The kg/s/Pa gradient can be very small for a high-
                # resistance path even before mass conservation is reached.
                # Tighten the optimizer's dimensional gradient test; the
                # explicit closure check below remains the success gate.
                gtol=min(relative_tolerance, 1.0e-15),
                max_nfev=max_pressure_nfev,
            )
            solver_success = bool(result.success)
            pressures = dict(zip(junction_names, result.x))
            states = self._states(pressures, enthalpies)
            flows = self._flows(states)
            updated = dict(enthalpies)
            for name in junction_names:
                incoming_mass = 0.0
                incoming_energy = 0.0
                for flow in flows.values():
                    arrives = (flow.right == name and flow.mass_flow_kg_s > 0.0) or (
                        flow.left == name and flow.mass_flow_kg_s < 0.0
                    )
                    if arrives:
                        incoming_mass += abs(flow.mass_flow_kg_s)
                        incoming_energy += abs(flow.mass_flow_kg_s) * flow.transmitted_specific_enthalpy_J_kg
                if incoming_mass > 1e-14:
                    updated[name] = incoming_energy / incoming_mass
            relative_change = max(
                abs(updated[name] - enthalpies[name]) / max(abs(enthalpies[name]), 1.0)
                for name in junction_names
            )
            enthalpies = updated
            if relative_change < relative_tolerance:
                enthalpy_converged = True
                break
        states = self._states(pressures, enthalpies)
        flows = self._flows(states)
        self._check_mass_only_contract(states, flows)
        self._validate_transmitted_states(states, flows)
        mass_residual = {name: self._mass_residual(flows, name) for name in junction_names}
        energy_residual: dict[str, float] = {}
        for name in junction_names:
            residual_W = 0.0
            for flow in flows.values():
                if flow.right == name:
                    residual_W += flow.mass_flow_kg_s * (
                        flow.transmitted_specific_enthalpy_J_kg if flow.mass_flow_kg_s > 0.0
                        else states[name].specific_enthalpy_J_kg
                    )
                if flow.left == name:
                    residual_W -= flow.mass_flow_kg_s * (
                        states[name].specific_enthalpy_J_kg if flow.mass_flow_kg_s > 0.0
                        else flow.transmitted_specific_enthalpy_J_kg
                    )
            energy_residual[name] = residual_W
        junction_conservation_success = True
        mass_tolerances: dict[str, float] = {}
        energy_tolerances: dict[str, float] = {}
        for name in junction_names:
            incident = [
                flow for flow in flows.values()
                if name in (flow.left, flow.right)
            ]
            mass_scale = sum(abs(flow.mass_flow_kg_s) for flow in incident)
            energy_scale = sum(
                abs(flow.mass_flow_kg_s) * max(
                    abs(flow.transmitted_specific_enthalpy_J_kg),
                    abs(states[name].specific_enthalpy_J_kg),
                )
                for flow in incident
            )
            mass_tolerance = max(
                mass_closure_absolute_tolerance_kg_s,
                closure_relative_tolerance * mass_scale,
            )
            maximum_specific_enthalpy = max(
                (
                    max(
                        abs(flow.transmitted_specific_enthalpy_J_kg),
                        abs(states[name].specific_enthalpy_J_kg),
                    )
                    for flow in incident
                ),
                default=0.0,
            )
            energy_tolerance = max(
                energy_closure_absolute_tolerance_W,
                closure_relative_tolerance * energy_scale,
                mass_tolerance * maximum_specific_enthalpy,
            )
            mass_tolerances[name] = mass_tolerance
            energy_tolerances[name] = energy_tolerance
            if (
                abs(mass_residual[name]) > mass_tolerance
                or abs(energy_residual[name]) > energy_tolerance
            ):
                junction_conservation_success = False
        network_success = solver_success and enthalpy_converged and junction_conservation_success
        if network_success:
            self._last_junction_pressures = dict(pressures)
            self._last_junction_enthalpies = dict(enthalpies)
        return NetworkSolution(
            states,
            flows,
            mass_residual,
            energy_residual,
            network_success,
            iteration,
            self._momentum_residuals(flows),
            pressure_optimizer_success=solver_success,
            enthalpy_solver_success=enthalpy_converged,
            junction_conservation_success=junction_conservation_success,
            node_mass_closure_tolerance_kg_s=mass_tolerances,
            node_energy_closure_tolerance_W=energy_tolerances,
        )

    @staticmethod
    def _momentum_residuals(flows: dict[str, LinkSolution]) -> dict[str, float]:
        """Collect only link-level momentum closures supplied by components."""

        return {
            name: float(flow.momentum_residual_Pa)
            for name, flow in flows.items()
            if flow.momentum_residual_Pa is not None
        }

