# LH2DT Interface Reference

## 1. Installation and import

The distribution name is `lh2dt_models`. The implementation namespace remains
`lh2dt` for compatibility with existing examples.

```bash
python -m pip install lh2dt_models
```

```python
from lh2dt import HydrogenProperties, LayeredTank, MassEnergyFlow
```

The installed distribution also exposes `lh2dt_models`, but public examples and
submodules use `lh2dt` as the canonical import path.

## 2. Common interface contract

Use SI units internally: Pa absolute, K, kg, kg/s, J/kg, W, and m³. Positive
mass flow enters a control volume and negative mass flow leaves it. `ThermoState`
should be passed between components rather than separate pressure and temperature
numbers.

Every dynamic component should expose:

1. an immutable parameter object;
2. an explicit initial state;
3. a `thermo` or equivalent state projection;
4. a `step`/`simulate` method;
5. cumulative mass and energy ledgers;
6. conservation residuals.

## 3. Tank construction and runtime inputs

`HomogeneousTank` requires volume, wall heat capacity, ambient UA, and fluid-wall
UA. Its runtime inputs are initial pressure, liquid volume fraction, wall and
ambient temperature, signed flows, direct heat, and integration time.

`StratifiedTank` requires horizontal geometry plus lower/upper wall heat capacity,
ambient UA, wall-liquid and wall-vapor transfer, interface UA, axial wall
conduction, wall split height, and pressure bounds.

`LayeredTank` requires cell counts, wall thermal parameters, optional connected
dead-volume and nozzle-wall capacity, interface/boiling/circulation closures,
ambient or vacuum heat boundaries, and pressure bounds. Its state is a common
pressure plus liquid-cell, vapor-cell, and wall-node arrays.

`RadialAxialTank` requires radial/axial level counts, wall volume fractions,
radial and axial conductances, interface UA, wall capacity, ambient UA, and
pressure limits.

Runtime tank inputs are signed per-cell flows, direct heat callbacks, ambient
temperature, optional relief/leakage boundary, and the time step. Outputs are
pressure, level, cell thermodynamic states, wall temperatures, phase-change rates,
heat rates, cumulative ledgers, and mass/energy residuals.

## 4. Component ports

`Valve`, `Pipe`, `HEMPipe`, vaporizers, reliquefiers, pumps, and compressors accept
one or more `ThermoState` ports plus component parameters. They return an outlet
state, signed mass flow, enthalpy/heat/power information, and momentum or energy
diagnostics where applicable.

Use `NetworkLink` when a component is placed in a `SteadyNetwork` or
`DynamicNetworkSimulator`. Do not manually duplicate an internal flow in the
external mass ledger; network aggregation cancels internal transfers.

## 5. Valve and pipe interfaces

```python
result = valve.evaluate(
    upstream=upstream_state,
    downstream=downstream_state,
    opening=0.5,
)
```

The valve result contains mass flow, outlet state, throat pressure, choked status,
flow direction, and residual information. `CommandedValve` and
`StrokeLimitedCommandedValve` add a time-varying command and actuator dynamics.
`CheckValve`, `PressureReliefValve`, and `TemperatureProtectionValve` add direction,
pressure-hysteresis, and temperature-trip logic.

`Pipe` uses length, inner diameter, roughness, local-loss coefficient, elevation,
and direct heat. `DynamicHEMPipe` additionally accepts fluid volume, wall capacity,
fluid-wall UA, ambient temperature, ambient UA, and a time step; it returns fluid
and wall state plus three residual ledgers.

## 6. Vaporizer and reliquefier interfaces

Steady vaporizer inputs are inlet state, outlet pressure, mass flow, heat-source
temperature, UA, approach-temperature limit, and maximum heat duty. Dynamic
vaporizers add fluid/wall state, wall capacity, fluid-wall UA, ambient UA, and a
time step. Parallel banks return per-unit results and aggregate flow/heat ledgers.

Reliquefiers use inlet state, outlet pressure, mass flow, cooling capacity, and
target quality. Dynamic units add fluid/wall thermal storage, cold-side temperature,
cold-side UA, and a time step.

## 7. Network interface

`SteadyNetwork` receives junction nodes, network links, and boundary conditions.
It returns node states, link solutions, mass/energy residuals, solver status, and
momentum residuals.

`DynamicNetworkSimulator` receives dynamic node models and a command callback. A
dynamic adapter must provide current state, thermodynamic projection, derivative,
advance operation, external mass/energy rate, and residuals.

## 8. Measurement and HART interface

`FirstOrderTransmitter` accepts range, time constant, scale, bias, resolution, and
accuracy parameters. It returns filtered process value, indicated value,
saturation, quantization bound, and stated accuracy bound.

`PMD75BHartObservation.advance` accepts raw sensor pressure, position-adjusted
pressure, sensor temperature, electronics temperature, and time step. It returns
PV, SV, TV, and QV. The HART variables are measurement-layer values and must not be
double-injected as independent plant states.

## 9. Data binding and validation

An external dataset should provide `timestamp`, channel names, units, pressure
reference, timezone, sampling interval, missing-data flags, and provenance. The
recommended binding sequence is:

1. sort and validate timestamps;
2. validate units and absolute/gauge pressure;
3. initialize from the first valid row;
4. bind only physical operating boundaries as model inputs;
5. run a forward simulation;
6. compare later observations without feedback;
7. report mass, energy, and momentum residuals.

## 10. Output schema

For a portable time series, export `timestamp`, `pressure_Pa_abs`, `temperature_K`,
`liquid_mass_kg`, `vapor_mass_kg`, `liquid_level_m`, `evaporation_rate_kg_s`,
`heat_leak_W`, `mass_residual_kg`, and `energy_residual_J`. Instrument readings
should be explicitly suffixed, for example `PT1105A_pv`, `DPT1101_qv`, and
`TT1116_sv`.

## 11. Model selection and multi-unit use

Model families are fidelity alternatives. Instantiate one selected model per
physical asset. Four physical tanks therefore use four independent tank instances,
each with its own state and boundary conditions. Connect them through a network
only when the process topology includes an actual connection.

## 12. Fitting policy

The production baseline is physics-only. Geometry, properties, public closures,
manufacturer ratings, and explicit boundaries are configuration inputs. Any
parameter estimation for a quantitative report must be isolated in a separate
experiment and must not silently modify the production model defaults.
