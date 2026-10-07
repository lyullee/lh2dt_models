# LH2DT Model Reference

This document describes the states, conservation laws, closures, and interfaces
implemented by `lh2dt_models`. The package contains generic first-principles
components; site tags, historian files, P&IDs, photographs, and private asset
catalogs are intentionally excluded.

## 1. Architecture

The library has four layers:

1. **Property layer**: Para-Hydrogen thermodynamic states and saturation states.
2. **Equipment layer**: tanks, insulation, pipes, valves, vaporizers,
   reliquefiers, pumps, and compressors.
3. **Network layer**: steady and dynamic port connections with mass, energy, and
   momentum closure diagnostics.
4. **Measurement/control layer**: transmitter dynamics, HART PV/SV/TV/QV
   observations, and PID control.

Internal units are Pa absolute, K, kg, kg/s, J/kg, W, and m³. Gauge pressure is
converted at the boundary and is never mixed with absolute pressure internally.

## 2. Common thermodynamic contract

`ThermoState` carries pressure, temperature, density, specific enthalpy, specific
internal energy, entropy, quality, and phase. `HydrogenProperties` supplies
`from_pT`, `from_ph`, `from_rho_u`, `saturated_liquid`, and `saturated_vapor`.

`MassEnergyFlow` carries signed mass flow and specific enthalpy:

```python
MassEnergyFlow(
    mass_flow_kg_s=0.01,
    specific_enthalpy_J_kg=1.2e5,
    source="upstream-valve",
)
```

The enthalpy flow is \(\dot H=\dot m h\). Positive flow enters the control
volume and negative flow leaves it.

## 3. Tank models

### HomogeneousTank

`HomogeneousTank` represents the vessel as one average fluid control volume and a
single wall thermal mass. Construction inputs are vessel volume, wall heat
capacity, ambient-wall UA, and fluid-wall UA. Runtime inputs are initial absolute
pressure, liquid volume fraction, wall temperature, ambient temperature, signed
mass/energy flows, direct wall heat, and the integration step.

The state is total hydrogen mass, total hydrogen internal energy, and wall
temperature. The output includes the thermodynamic state, liquid/vapor phase
inventory, cumulative boundary mass/energy, and an energy residual.

### StratifiedTank

`StratifiedTank` has separate liquid and vapor control volumes and independent
lower and upper wall temperatures. Its parameters include lower/upper wall heat
capacity and ambient UA, wall-liquid and wall-vapor heat transfer, interface UA,
axial wall conduction, a wall-zone split height, and pressure bounds.

Outputs include liquid and vapor states, liquid height and interface area,
evaporation rate, lower/upper wall heat rates, cumulative boundary ledgers, and
mass/energy residuals.

### LayeredTank

`LayeredTank` is the reference high-fidelity tank model used in the current
TK-1101/TK-1102 forward diagnostics. It stores arrays of liquid cells, vapor
cells, and wall nodes under a rigid-vessel common-pressure constraint.

Configuration inputs include cell counts, wall heat capacity and UA, axial
conduction, connected vapor dead volume, connected wall heat capacity, interface
heat-transfer and phase-change closures, distributed boiling, wall circulation,
ambient/vacuum heat boundaries, and pressure limits.

Initial inputs include pressure, liquid volume fraction or explicit cell mass and
enthalpy profiles, vapor profiles, wall-temperature profiles, and ambient state.
Runtime inputs include per-cell flows, direct heat callbacks, external heat
boundaries, optional leakage/relief boundaries, and the time step.

The step result contains:

- cell-level liquid and vapor masses and enthalpies;
- cell-level wall temperatures;
- pressure, liquid height, interface area, and cell geometry;
- interface temperature, pressure, mass flux, evaporation rate, and latent power;
- distributed boiling rates and wall circulation rates;
- cumulative boundary mass/energy and cumulative evaporation;
- total mass and energy residuals.

### RadialAxialTank

`RadialAxialTank` resolves liquid and vapor in axial levels and radial core/wall
regions. It accepts total volume, level counts, wall volume fractions, axial and
radial conductances, interface UA, wall heat capacity, ambient UA, and pressure
bounds. It returns cell temperatures, masses, enthalpies, circulation, local heat
rates, pressure, level, and conservation residuals.

## 4. Heat boundaries and insulation

`VacuumInsulation` separates radiation, MLI/solid conduction, support conduction,
residual-gas conduction, and local heat paths. The total heat leak is

\[
Q_{in}=Q_{rad}+Q_{MLI}+Q_{support}+Q_{gas}+Q_{local}.
\]

`VacuumJacketedPipeEnvelope` derives process and jacket geometry. Natural
convection and natural-circulation models supply heat-transfer coefficients,
heat rates, circulation flows, and residuals to the tank or pipe equations.

## 5. Flow-path models

`Valve` uses effective open area, discharge coefficient, opening, and upstream/
downstream states. It reports mass flow, outlet state, throat pressure, choked
status, flow direction, and momentum residual. Check, relief, and temperature
protection valves add one-way, pressure-hysteresis, and temperature-trip logic.

`Pipe` uses Darcy-Weisbach friction, local losses, elevation, and a specified
direct heat input. `HEMPipe` adds homogeneous-equilibrium two-phase expansion.
`DynamicHEMPipe` integrates fluid and wall states. `VentStack` adds vent-stack
heat exchange and reverse-flow policy.

## 6. Vaporization and reliquefaction

`Vaporizer` uses UA, maximum heat duty, approach-temperature and pressure-boundary
inputs to calculate outlet state, required heat, delivered heat, and capacity
limiting. `AmbientAirVaporizer` adds ambient convection, radiation, frost growth/
thaw, humidity, and wind. `DynamicVaporizer` integrates fluid and wall state;
`ParallelDynamicVaporizer` returns unit-level and bank-level ledgers.

`Reliquefier` uses cooling capacity and target quality. `DynamicReliquefier` adds
fluid/wall thermal storage and cold-side UA; the parallel bank aggregates multiple
active units while retaining each unit's state and residuals.

## 7. Pumps and compressor

`Pump` uses pressure rise and isentropic efficiency. `VirtualPump` interpolates a
shutoff-pressure/runout-flow H-Q relation. `MappedPump` uses explicit H-Q,
efficiency, and NPSH curves. Outputs include outlet state, flow, head, power,
efficiency, NPSH margin, and momentum residual where applicable.

`Compressor` uses pressure ratio and isentropic efficiency to calculate outlet
state, pressure rise, isentropic power, and shaft power.

## 8. Network and closure

`SteadyNetwork` solves junction pressure/enthalpy and link flows. The dynamic
network integrates storage nodes, dynamic pipes, vaporizers, reliquefiers, and
external boundaries. Internal transfers cancel in the aggregate ledger; external
mass and energy boundaries remain explicit. Link momentum residuals and node mass/
energy residuals are returned for every solve.

## 9. Measurement and control

`FirstOrderTransmitter` represents time constant, range, scale/bias, resolution,
saturation, and stated accuracy. `PMD75BHartObservation` maps explicit sensor
inputs to:

- PV: damped and position-adjusted pressure;
- SV: sensor temperature;
- TV: electronics temperature;
- QV: raw sensor pressure before damping and position adjustment.

These are observation-layer values and do not create additional plant states.
`PIDController` returns output, error, P/I/D terms, saturation, and next state.

## 10. Verification policy

Use the following order: validate property states, check timestep convergence,
check mass/energy/momentum residuals, freeze initial and boundary conditions,
run a forward calculation without observed-trajectory feedback, and compare the
result against an independent time series. Field-data fitting is kept outside the
production physics baseline.
