"""Port-level coupling between ``StratifiedTank`` and a steady network.

``StratifiedTank`` has one liquid and one vapor control volume, each with its
own thermodynamic boundary state.  This adapter keeps those states as separate
network boundaries and aggregates every connected link into the corresponding
phase balance.  It deliberately does not collapse the two phases into one
``DynamicNodeModel`` state: doing so would lose the rigid-volume pressure
closure and the liquid/vapor energy ledgers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Literal, Mapping, Sequence

from .network import LinkSolution, NetworkSolution, SteadyNetwork
from .stratified_tank import (
    StratifiedTank,
    StratifiedTankDerivative,
    StratifiedTankState,
    StratifiedTankStepResult,
    StratifiedThermoState,
)
from .streams import MassEnergyFlow


StratifiedPhase = Literal["liquid", "vapor"]


@dataclass(frozen=True)
class StratifiedTankPort:
    """Map one network boundary node to a tank phase control volume."""

    node_name: str
    phase: StratifiedPhase


@dataclass(frozen=True)
class StratifiedTankNetworkStepResult:
    """One tank integration result and the corresponding network solution."""

    tank: StratifiedTankStepResult
    network_solution: NetworkSolution


@dataclass(frozen=True)
class StratifiedTankEnsemblePort:
    """Map one network boundary node to one phase of one tank in an ensemble."""

    tank_name: str
    node_name: str
    phase: StratifiedPhase


@dataclass(frozen=True)
class StratifiedTankNetworkEnsembleStepResult:
    """One shared-network integration result for multiple two-region tanks."""

    time_s: float
    states: dict[str, StratifiedTankState]
    thermo_states: dict[str, StratifiedThermoState]
    network_solution: NetworkSolution
    cumulative_boundary_mass_kg: dict[str, float]
    cumulative_boundary_energy_J: dict[str, float]
    total_mass_residual_kg: dict[str, float]
    total_energy_residual_J: dict[str, float]


class StratifiedTankNetworkSimulator:
    """Advance a two-region tank through reusable two-port components.

    Each boundary node listed in ``ports`` receives the liquid or vapor
    thermodynamic state.  Link flows are converted to signed
    :class:`MassEnergyFlow` values at the tank boundary, then passed to the
    tank's first-principles conservation equations.  A phase may expose more
    than one node, which is useful for separate fill, withdrawal, and vent
    lines while retaining one common phase inventory.
    """

    def __init__(
        self,
        tank: StratifiedTank,
        network: SteadyNetwork,
        ports: Sequence[StratifiedTankPort],
        ambient_temperature_K: float | Callable[[float], float],
        command_callback: Callable[[float, SteadyNetwork], None] | None = None,
    ) -> None:
        if not ports:
            raise ValueError("At least one stratified-tank port is required")
        self.tank = tank
        self.network = network
        self.ports = tuple(ports)
        self.ambient_temperature_K = ambient_temperature_K
        self.command_callback = command_callback
        names = [port.node_name for port in self.ports]
        if len(names) != len(set(names)):
            raise ValueError("Each network node may map to only one tank port")
        missing = [name for name in names if name not in network.boundaries]
        if missing:
            raise KeyError("Stratified tank ports must be boundary nodes: " + ", ".join(missing))
        if any(port.node_name in network.junctions for port in self.ports):
            raise ValueError("Stratified tank ports cannot be junction nodes")
        for port in self.ports:
            if port.phase not in {"liquid", "vapor"}:
                raise ValueError(f"Unsupported stratified tank port phase: {port.phase}")

    @staticmethod
    def _node_flow(node: str, flow: LinkSolution, states: dict[str, object]) -> MassEnergyFlow | None:
        """Express one network link as a signed stream at ``node``."""

        if node not in (flow.left, flow.right):
            return None
        mass = flow.mass_flow_kg_s
        owner = states[node]
        if not hasattr(owner, "specific_enthalpy_J_kg"):
            raise TypeError("Network node state must expose specific_enthalpy_J_kg")
        owner_h = float(owner.specific_enthalpy_J_kg)
        if node == flow.right:
            if mass >= 0.0:
                return MassEnergyFlow(mass, flow.transmitted_specific_enthalpy_J_kg, source=flow.upstream_port_id or flow.name)
            return MassEnergyFlow(mass, owner_h, source=flow.upstream_port_id or flow.name)
        if mass >= 0.0:
            return MassEnergyFlow(-mass, owner_h, source=flow.upstream_port_id or flow.name)
        return MassEnergyFlow(-mass, flow.transmitted_specific_enthalpy_J_kg, source=flow.upstream_port_id or flow.name)

    def _boundary_states(self, thermo: StratifiedThermoState) -> dict[str, object]:
        return {
            port.node_name: thermo.liquid if port.phase == "liquid" else thermo.vapor
            for port in self.ports
        }

    def _solve(
        self,
        thermo: StratifiedThermoState,
        time_s: float,
    ) -> tuple[NetworkSolution, tuple[tuple[MassEnergyFlow, ...], tuple[MassEnergyFlow, ...]]]:
        port_states = self._boundary_states(thermo)
        # Publish the current stage before evaluating commands.  A pressure-
        # triggered valve must see the same phase state that the solver will
        # use, rather than the previous stage's boundary state.
        self.network.set_boundary_states(port_states)
        if self.command_callback is not None:
            self.command_callback(time_s, self.network)
        solution = self.network.solve()
        liquid: list[MassEnergyFlow] = []
        vapor: list[MassEnergyFlow] = []
        for port in self.ports:
            stream = [
                candidate
                for link in solution.link_flows.values()
                for candidate in (self._node_flow(port.node_name, link, solution.node_states),)
                if candidate is not None
            ]
            (liquid if port.phase == "liquid" else vapor).extend(stream)
        return solution, (tuple(liquid), tuple(vapor))

    def simulate(
        self,
        initial: StratifiedTankState,
        duration_s: float,
        time_step_s: float,
    ) -> list[StratifiedTankNetworkStepResult]:
        """Integrate the tank and network and return one result per step."""

        def flow_callback(time_s: float, thermo: StratifiedThermoState):
            _solution, flows = self._solve(thermo, time_s)
            return flows

        tank_results = self.tank.simulate(
            initial=initial,
            duration_s=duration_s,
            time_step_s=time_step_s,
            ambient_temperature_K=self.ambient_temperature_K,
            flow_callback=flow_callback,
        )
        results: list[StratifiedTankNetworkStepResult] = []
        for tank_result in tank_results:
            solution, _flows = self._solve(tank_result.thermo, tank_result.time_s)
            results.append(StratifiedTankNetworkStepResult(tank_result, solution))
        return results


class StratifiedTankNetworkEnsembleSimulator:
    """Advance multiple ``StratifiedTank`` objects through one network.

    The ensemble adapter is the phase-aware counterpart of
    ``DynamicNetworkSimulator``.  Every tank retains separate liquid and vapor
    states while all tank ports participate in the same pressure/enthalpy
    network solve.  This is the required structure for parallel storage tanks,
    selectable return tanks, and tank-to-tank transfer paths.
    """

    def __init__(
        self,
        tanks: Mapping[str, StratifiedTank],
        network: SteadyNetwork,
        ports: Sequence[StratifiedTankEnsemblePort],
        ambient_temperature_K: float | Callable[[float], float],
        command_callback: Callable[[float, SteadyNetwork], None] | None = None,
    ) -> None:
        if not tanks:
            raise ValueError("At least one stratified tank is required")
        if not ports:
            raise ValueError("At least one stratified-tank ensemble port is required")
        self.tanks = dict(tanks)
        if any(not name for name in self.tanks):
            raise ValueError("Stratified tank names must be non-empty")
        self.network = network
        self.ports = tuple(ports)
        self.ambient_temperature_K = ambient_temperature_K
        self.command_callback = command_callback
        node_names = [port.node_name for port in self.ports]
        if len(node_names) != len(set(node_names)):
            raise ValueError("Each network node may map to only one stratified tank port")
        missing_tanks = sorted({port.tank_name for port in self.ports} - set(self.tanks))
        if missing_tanks:
            raise KeyError("Unknown stratified tanks in ports: " + ", ".join(missing_tanks))
        missing_nodes = [name for name in node_names if name not in network.boundaries]
        if missing_nodes:
            raise KeyError("Stratified tank ensemble ports must be boundary nodes: " + ", ".join(missing_nodes))
        if any(name in network.junctions for name in node_names):
            raise ValueError("Stratified tank ensemble ports cannot be junction nodes")
        for port in self.ports:
            if port.phase not in {"liquid", "vapor"}:
                raise ValueError(f"Unsupported stratified tank port phase: {port.phase}")
        self._ports_by_tank: dict[str, tuple[StratifiedTankEnsemblePort, ...]] = {
            name: tuple(port for port in self.ports if port.tank_name == name)
            for name in self.tanks
        }

    @staticmethod
    def _node_flow(node: str, flow: LinkSolution, states: Mapping[str, object]) -> MassEnergyFlow | None:
        if node not in (flow.left, flow.right):
            return None
        mass = flow.mass_flow_kg_s
        owner = states[node]
        if not hasattr(owner, "specific_enthalpy_J_kg"):
            raise TypeError("Network node state must expose specific_enthalpy_J_kg")
        owner_h = float(owner.specific_enthalpy_J_kg)
        if node == flow.right:
            if mass >= 0.0:
                return MassEnergyFlow(mass, flow.transmitted_specific_enthalpy_J_kg, source=flow.name)
            return MassEnergyFlow(mass, owner_h, source=flow.name)
        if mass >= 0.0:
            return MassEnergyFlow(-mass, owner_h, source=flow.name)
        return MassEnergyFlow(-mass, flow.transmitted_specific_enthalpy_J_kg, source=flow.name)

    def _solve(
        self,
        states: Mapping[str, StratifiedTankState],
        pressure_hints: Mapping[str, float],
        time_s: float,
    ) -> tuple[
        NetworkSolution,
        dict[str, StratifiedThermoState],
        dict[str, tuple[tuple[MassEnergyFlow, ...], tuple[MassEnergyFlow, ...]]],
    ]:
        thermos = {
            name: self.tanks[name].thermo(state, pressure_hints.get(name))
            for name, state in states.items()
        }
        boundary_states: dict[str, object] = {}
        for tank_name, tank_ports in self._ports_by_tank.items():
            thermo = thermos[tank_name]
            for port in tank_ports:
                boundary_states[port.node_name] = (
                    thermo.liquid if port.phase == "liquid" else thermo.vapor
                )
        # Keep command evaluation stage-consistent with DynamicNetworkSimulator:
        # controls read the newly published phase boundary states.
        self.network.set_boundary_states(boundary_states)
        if self.command_callback is not None:
            self.command_callback(time_s, self.network)
        solution = self.network.solve()
        phase_flows: dict[str, tuple[list[MassEnergyFlow], list[MassEnergyFlow]]] = {
            name: ([], []) for name in self.tanks
        }
        for port in self.ports:
            target = phase_flows[port.tank_name][0 if port.phase == "liquid" else 1]
            target.extend(
                stream
                for link in solution.link_flows.values()
                for stream in (self._node_flow(port.node_name, link, solution.node_states),)
                if stream is not None
            )
        return solution, thermos, {
            name: (tuple(values[0]), tuple(values[1]))
            for name, values in phase_flows.items()
        }

    @staticmethod
    def _combine_derivatives(
        first: StratifiedTankDerivative,
        second: StratifiedTankDerivative,
        third: StratifiedTankDerivative,
        fourth: StratifiedTankDerivative,
    ) -> StratifiedTankDerivative:
        values = {
            field.name: (
                getattr(first, field.name)
                + 2.0 * getattr(second, field.name)
                + 2.0 * getattr(third, field.name)
                + getattr(fourth, field.name)
            ) / 6.0
            for field in fields(StratifiedTankDerivative)
        }
        return StratifiedTankDerivative(**values)

    def _evaluate_stage(
        self,
        states: Mapping[str, StratifiedTankState],
        pressure_hints: Mapping[str, float],
        time_s: float,
    ) -> tuple[
        NetworkSolution,
        dict[str, StratifiedThermoState],
        dict[str, StratifiedTankDerivative],
        dict[str, float],
        dict[str, float],
    ]:
        solution, thermos, phase_flows = self._solve(states, pressure_hints, time_s)
        derivatives: dict[str, StratifiedTankDerivative] = {}
        boundary_mass: dict[str, float] = {}
        boundary_energy: dict[str, float] = {}
        for name, tank in self.tanks.items():
            liquid_flows, vapor_flows = phase_flows[name]
            thermo = thermos[name]
            ambient = (
                self.ambient_temperature_K(time_s)
                if callable(self.ambient_temperature_K)
                else self.ambient_temperature_K
            )
            derivative = tank._derivative_from_thermo(
                states[name], thermo, liquid_flows, vapor_flows, float(ambient),
                0.0, 0.0, 0.0, 0.0,
            )
            derivatives[name] = derivative
            liquid_mass, liquid_energy = tank._flow_balance(liquid_flows, thermo.liquid)
            vapor_mass, vapor_energy = tank._flow_balance(vapor_flows, thermo.vapor)
            boundary_mass[name] = liquid_mass + vapor_mass
            boundary_energy[name] = (
                liquid_energy + vapor_energy
                + derivative.lower_ambient_heat_W + derivative.upper_ambient_heat_W
            )
        return solution, thermos, derivatives, boundary_mass, boundary_energy

    def simulate(
        self,
        initial_states: Mapping[str, StratifiedTankState],
        duration_s: float,
        time_step_s: float,
    ) -> list[StratifiedTankNetworkEnsembleStepResult]:
        """Integrate all tanks with shared RK4 network stages."""

        if duration_s <= 0.0 or time_step_s <= 0.0:
            raise ValueError("duration_s and time_step_s must be positive")
        if set(initial_states) != set(self.tanks):
            missing = sorted(set(self.tanks) - set(initial_states))
            extra = sorted(set(initial_states) - set(self.tanks))
            raise ValueError(f"Initial tank states mismatch; missing={missing}, extra={extra}")
        states = dict(initial_states)
        initial_mass = {
            name: state.liquid_mass_kg + state.vapor_mass_kg
            for name, state in states.items()
        }
        initial_energy = {
            name: self.tanks[name]._total_energy_J(state)
            for name, state in states.items()
        }
        cumulative_mass = {name: 0.0 for name in self.tanks}
        cumulative_energy = {name: 0.0 for name in self.tanks}
        pressure_hints = {
            name: self.tanks[name].thermo(state).pressure_Pa
            for name, state in states.items()
        }
        results: list[StratifiedTankNetworkEnsembleStepResult] = []
        time_s = 0.0
        while time_s < duration_s - 1.0e-12:
            dt = min(time_step_s, duration_s - time_s)
            s1, t1, k1, m1, e1 = self._evaluate_stage(states, pressure_hints, time_s)
            half_1 = {
                name: self.tanks[name]._advance(states[name], k1[name], dt / 2.0)
                for name in states
            }
            s2, t2, k2, m2, e2 = self._evaluate_stage(half_1, {name: t1[name].pressure_Pa for name in states}, time_s + dt / 2.0)
            half_2 = {
                name: self.tanks[name]._advance(states[name], k2[name], dt / 2.0)
                for name in states
            }
            s3, t3, k3, m3, e3 = self._evaluate_stage(half_2, {name: t2[name].pressure_Pa for name in states}, time_s + dt / 2.0)
            full_3 = {
                name: self.tanks[name]._advance(states[name], k3[name], dt)
                for name in states
            }
            s4, t4, k4, m4, e4 = self._evaluate_stage(full_3, {name: t3[name].pressure_Pa for name in states}, time_s + dt)
            states = {
                name: self.tanks[name]._advance(
                    states[name], self._combine_derivatives(k1[name], k2[name], k3[name], k4[name]), dt
                )
                for name in states
            }
            for name in states:
                cumulative_mass[name] += dt * (m1[name] + 2.0 * m2[name] + 2.0 * m3[name] + m4[name]) / 6.0
                cumulative_energy[name] += dt * (e1[name] + 2.0 * e2[name] + 2.0 * e3[name] + e4[name]) / 6.0
            time_s += dt
            solution, thermos, _flows = self._solve(
                states, {name: t4[name].pressure_Pa for name in states}, time_s
            )
            pressure_hints = {name: thermo.pressure_Pa for name, thermo in thermos.items()}
            results.append(StratifiedTankNetworkEnsembleStepResult(
                time_s=time_s,
                states=dict(states),
                thermo_states=dict(thermos),
                network_solution=solution,
                cumulative_boundary_mass_kg=dict(cumulative_mass),
                cumulative_boundary_energy_J=dict(cumulative_energy),
                total_mass_residual_kg={
                    name: state.liquid_mass_kg + state.vapor_mass_kg - initial_mass[name] - cumulative_mass[name]
                    for name, state in states.items()
                },
                total_energy_residual_J={
                    name: self.tanks[name]._total_energy_J(state) - initial_energy[name] - cumulative_energy[name]
                    for name, state in states.items()
                },
            ))
        return results
