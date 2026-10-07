"""Port-level coupling for the conservative radial/axial tank model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .network import LinkSolution, NetworkSolution, SteadyNetwork
from .radial_axial_tank import RadialAxialTank, RadialAxialTankState, RadialAxialTankStepResult, RadialAxialTankThermo
from .streams import MassEnergyFlow


@dataclass(frozen=True)
class RadialAxialTankPort:
    """Map a network boundary node to a stable radial/axial cell index."""

    node_name: str
    cell_index: int


@dataclass(frozen=True)
class RadialAxialTankNetworkStepResult:
    tank: RadialAxialTankStepResult
    network_solution: NetworkSolution


class RadialAxialTankNetworkSimulator:
    """Advance a radial/axial tank through the reusable steady network.

    The radial/axial tank currently exposes a conservative semi-implicit
    ``step_euler`` integrator because its circulation transport is stiff. The
    network and cell-port contract is independent of that time integrator.
    """

    def __init__(
        self,
        tank: RadialAxialTank,
        network: SteadyNetwork,
        ports: Sequence[RadialAxialTankPort],
        ambient_temperature_K: float | Callable[[float], float],
        command_callback: Callable[[float, SteadyNetwork], None] | None = None,
    ) -> None:
        if not ports:
            raise ValueError("At least one radial/axial tank port is required")
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
            raise KeyError("Radial/axial tank ports must be boundary nodes: " + ", ".join(missing))
        for port in self.ports:
            if not isinstance(port.cell_index, int) or not 0 <= port.cell_index < tank.cell_count:
                raise ValueError(f"Invalid radial/axial cell index {port.cell_index}")

    @staticmethod
    def _node_flow(node: str, flow: LinkSolution, states: dict[str, object]) -> MassEnergyFlow | None:
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

    def _solve(
        self,
        thermo: RadialAxialTankThermo,
        time_s: float,
    ) -> tuple[NetworkSolution, tuple[tuple[MassEnergyFlow, ...], ...]]:
        port_states = {
            port.node_name: thermo.cells[port.cell_index]
            for port in self.ports
        }
        # Publish the current cell state before callbacks so a pressure-based
        # command cannot read a stale boundary from the prior step.
        self.network.set_boundary_states(port_states)
        if self.command_callback is not None:
            self.command_callback(time_s, self.network)
        solution = self.network.solve()
        cell_streams: list[list[MassEnergyFlow]] = [
            [] for _ in range(self.tank.cell_count)
        ]
        for port in self.ports:
            streams = [
                stream
                for link in solution.link_flows.values()
                for stream in (self._node_flow(port.node_name, link, solution.node_states),)
                if stream is not None
            ]
            cell_streams[port.cell_index].extend(streams)
        return solution, tuple(tuple(group) for group in cell_streams)

    def simulate(
        self,
        initial: RadialAxialTankState,
        duration_s: float,
        time_step_s: float,
    ) -> list[RadialAxialTankNetworkStepResult]:
        if duration_s <= 0.0 or time_step_s <= 0.0:
            raise ValueError("duration_s and time_step_s must be positive")
        state = self.tank.project_state(initial)
        time_s = 0.0
        results: list[RadialAxialTankNetworkStepResult] = []
        while time_s < duration_s - 1.0e-12:
            dt = min(time_step_s, duration_s - time_s)
            thermo = self.tank.thermo(state)
            _solution, streams = self._solve(thermo, time_s)
            ambient = (
                self.ambient_temperature_K(time_s)
                if callable(self.ambient_temperature_K)
                else self.ambient_temperature_K
            )
            tank_result = self.tank.step_euler(
                state,
                time_step_s=dt,
                ambient_temperature_K=float(ambient),
                cell_streams=streams,
            )
            state = tank_result.state
            time_s += dt
            solution, _streams = self._solve(tank_result.thermo, time_s)
            results.append(RadialAxialTankNetworkStepResult(tank_result, solution))
        return results
