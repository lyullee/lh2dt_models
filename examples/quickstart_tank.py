"""Run a small, no-fitting homogeneous LH2 tank calculation."""

from lh2dt import HydrogenProperties, HomogeneousTank, MassEnergyFlow, TankGeometry


def main() -> None:
    properties = HydrogenProperties()
    tank = HomogeneousTank(
        TankGeometry(
            volume_m3=10.0,
            wall_heat_capacity_J_K=2.0e6,
            ambient_UA_W_K=8.0,
            fluid_wall_UA_W_K=15.0,
        ),
        properties,
    )
    state = tank.initialize_saturated(130_000.0, liquid_volume_fraction=0.70)

    def closed_boundary(_time_s: float, thermo):
        return [MassEnergyFlow(0.0, thermo.specific_enthalpy_J_kg, "closed")]

    trace = tank.simulate(
        state,
        duration_s=600.0,
        time_step_s=10.0,
        ambient_temperature_K=293.15,
        flow_callback=closed_boundary,
    )
    last = trace[-1]
    print(f"samples={len(trace)}")
    print(f"pressure_Pa={last.thermo.pressure_Pa:.3f}")
    print(f"temperature_K={last.thermo.temperature_K:.3f}")
    print(f"energy_residual_J={last.total_energy_residual_J:.6e}")


if __name__ == "__main__":
    main()

