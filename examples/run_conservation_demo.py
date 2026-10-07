"""Run non-plant reference checks for a filling tank and a branch network."""

from dataclasses import dataclass

from lh2dt import (
    HomogeneousTank,
    HydrogenProperties,
    MassEnergyFlow,
    SteadyNetwork,
    TankGeometry,
)


@dataclass(frozen=True)
class LinearResult:
    mass_flow_kg_s: float
    outlet_specific_enthalpy_J_kg: float


class LinearResistance:
    def __init__(self, resistance_Pa_s_kg: float) -> None:
        self.resistance = resistance_Pa_s_kg

    def evaluate(self, left, right):
        flow = (left.pressure_Pa - right.pressure_Pa) / self.resistance
        upstream_h = left.specific_enthalpy_J_kg if flow >= 0.0 else right.specific_enthalpy_J_kg
        return LinearResult(flow, upstream_h)


def main() -> None:
    properties = HydrogenProperties()
    tank = HomogeneousTank(TankGeometry(2.0, 2.0e6, 5.0, 100.0), properties)
    initial = tank.initialize_saturated(300_000.0, 0.5)
    source = properties.saturated_liquid(500_000.0)
    results = tank.simulate(
        initial,
        duration_s=20.0,
        time_step_s=1.0,
        ambient_temperature_K=tank.thermo(initial).temperature_K,
        flow_callback=lambda _time, _state: [MassEnergyFlow(0.01, source.specific_enthalpy_J_kg, "fill")],
    )
    final = results[-1]
    print("Tank reference case")
    print(f"  added mass: {final.state.hydrogen_mass_kg - initial.hydrogen_mass_kg:.9f} kg")
    print(f"  total-energy residual: {final.total_energy_residual_J:.6e} J")
    print(f"  final pressure: {final.thermo.pressure_Pa:.3f} Pa")

    network = SteadyNetwork(properties)
    network.add_boundary("source", properties.from_pT(800_000.0, 80.0))
    network.add_boundary("sink_a", properties.from_pT(200_000.0, 80.0))
    network.add_boundary("sink_b", properties.from_pT(300_000.0, 80.0))
    network.add_junction("header", 500_000.0, properties.from_pT(800_000.0, 80.0).specific_enthalpy_J_kg)
    network.add_link("inlet", "source", "header", LinearResistance(1.0e7))
    network.add_link("outlet_a", "header", "sink_a", LinearResistance(2.0e7))
    network.add_link("outlet_b", "header", "sink_b", LinearResistance(3.0e7))
    solution = network.solve()
    print("Branch-network reference case")
    print(f"  header pressure: {solution.node_states['header'].pressure_Pa:.3f} Pa")
    print(f"  mass residual: {solution.node_mass_residual_kg_s['header']:.6e} kg/s")
    print(f"  energy residual: {solution.node_energy_residual_W['header']:.6e} W")


if __name__ == "__main__":
    main()

