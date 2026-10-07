"""Port-level coupling between ``LayeredTank`` and a steady component network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from .layered_tank import CellFlows, LayeredTank, LayeredTankState, LayeredTankStepResult, LayeredThermoState
from .network import LinkSolution, NetworkSolution, SteadyNetwork
from .streams import MassEnergyFlow


LayerPhase = Literal["liquid", "vapor"]


@dataclass(frozen=True)
class LayeredTankPort:
    """Map one network boundary node to one explicit tank cell."""

    node_name: str
    phase: LayerPhase
    cell_index: int


@dataclass(frozen=True)
class LayeredTankNetworkStepResult:
    tank: LayeredTankStepResult
    network_solution: NetworkSolution


class LayeredTankNetworkSimulator:
    """Run a layered tank through the same reusable 2-port network solver.

    A tank may expose any number of liquid and vapor cell ports.  Each port is
    a normal ``SteadyNetwork`` boundary, so adding a pipe, valve, vaporizer or
    reliquefier does not require changing the tank equations.
    """

    def __init__(
        self,
        tank: LayeredTank,
        network: SteadyNetwork,
        ports: Sequence[LayeredTankPort],
        ambient_temperature_K: float | Callable[[float], float],
        command_callback: Callable[[float, SteadyNetwork], None] | None = None,
    ) -> None:
        if not ports:
            raise ValueError("At least one layered-tank port is required")
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
            raise KeyError("Layered tank ports must be boundary nodes: " + ", ".join(missing))
        if any(port.node_name in network.junctions for port in self.ports):
            raise ValueError("Layered tank ports cannot be junction nodes")
        for port in self.ports:
            if port.phase not in {"liquid", "vapor"}:
                raise ValueError(f"Unsupported layered tank port phase: {port.phase}")
            count = (
                tank.parameters.liquid_cell_count
                if port.phase == "liquid"
                else tank.parameters.vapor_cell_count
            )
            if not isinstance(port.cell_index, int) or not 0 <= port.cell_index < count:
                raise ValueError(
                    f"Invalid {port.phase} cell index {port.cell_index} for port {port.node_name}"
                )

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

    def _boundary_states(self, thermo: LayeredThermoState) -> dict[str, object]:
        states: dict[str, object] = {}
        for port in self.ports:
            cells = thermo.liquid if port.phase == "liquid" else thermo.vapor
            states[port.node_name] = cells[port.cell_index]
        return states

    def _solve(
        self,
        thermo: LayeredThermoState,
        time_s: float,
    ) -> tuple[NetworkSolution, CellFlows]:
        port_states = self._boundary_states(thermo)
        # Publish the current cell states before applying pressure/temperature
        # commands so callbacks observe the same RK stage as the solver.
        self.network.set_boundary_states(port_states)
        if self.command_callback is not None:
            self.command_callback(time_s, self.network)
        solution = self.network.solve()
        liquid: list[list[MassEnergyFlow]] = [
            [] for _ in range(self.tank.parameters.liquid_cell_count)
        ]
        vapor: list[list[MassEnergyFlow]] = [
            [] for _ in range(self.tank.parameters.vapor_cell_count)
        ]
        for port in self.ports:
            streams = [
                stream
                for link in solution.link_flows.values()
                for stream in (self._node_flow(port.node_name, link, solution.node_states),)
                if stream is not None
            ]
            target = liquid if port.phase == "liquid" else vapor
            target[port.cell_index].extend(streams)
        return solution, (tuple(tuple(group) for group in liquid), tuple(tuple(group) for group in vapor))

    def simulate(
        self,
        initial: LayeredTankState,
        duration_s: float,
        time_step_s: float,
    ) -> list[LayeredTankNetworkStepResult]:
        """Integrate the tank with its network and return final network states."""

        def flow_callback(time_s: float, thermo: LayeredThermoState):
            _solution, flows = self._solve(thermo, time_s)
            return flows

        tank_results = self.tank.simulate(
            initial=initial,
            duration_s=duration_s,
            time_step_s=time_step_s,
            ambient_temperature_K=self.ambient_temperature_K,
            flow_callback=flow_callback,
        )
        results: list[LayeredTankNetworkStepResult] = []
        for tank_result in tank_results:
            solution, _flows = self._solve(tank_result.thermo, tank_result.time_s)
            results.append(LayeredTankNetworkStepResult(tank_result, solution))
        return results
