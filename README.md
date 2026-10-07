# lh2dt_models

First-principles, composable liquid-hydrogen equipment and process-network models for Python.

lh2dt_models is the distribution name on PyPI. The implementation package is imported as
lh2dt for compatibility with the original model examples, and lh2dt_models is also provided
as a compatibility namespace. Both imports are supported:

~~~python
import lh2dt
import lh2dt_models
~~~

The library is intended for engineering studies, digital-twin prototyping, model-in-the-loop
testing, and reproducible research. It provides explicit thermodynamic states, mass and energy
flows, component contracts, dynamic storage-tank models, cryogenic heat-transfer paths, valves,
pipes, vaporizers, reliquefiers, pumps, compressors, vent stacks, instrumentation, and connected
steady or dynamic process networks.

The public repository contains model code, public-reference benchmark data, examples, tests, and
English/Korean technical documentation. It deliberately does **not** contain facility tags,
historian exports, photographs, P&IDs, site-specific configuration files, or other private
operational records. The same model contracts can therefore be connected to a different facility
without publishing the facility identity or raw telemetry.

## Contents

- [What the library does](#what-the-library-does)
- [Design principles](#design-principles)
- [Installation](#installation)
- [A first dynamic tank simulation](#a-first-dynamic-tank-simulation)
- [Model architecture](#model-architecture)
- [Available model families](#available-model-families)
- [Connecting components](#connecting-components)
- [Inputs, outputs, and unit conventions](#inputs-outputs-and-unit-conventions)
- [Dynamic simulation workflow](#dynamic-simulation-workflow)
- [Instrumentation and HART observations](#instrumentation-and-hart-observations)
- [Observation loading and blind validation](#observation-loading-and-blind-validation)
- [Private asset input provenance](#private-asset-input-provenance)
- [Adaptive integration](#adaptive-integration)
- [Visual network authoring contract](#visual-network-authoring-contract)
- [Validation and reproducibility](#validation-and-reproducibility)
- [Choosing a model level](#choosing-a-model-level)
- [Project layout](#project-layout)
- [Documentation](#documentation)
- [Development](#development)
- [Release, DOI, and citation](#release-doi-and-citation)
- [Scope and responsible use](#scope-and-responsible-use)
- [License](#license)

## What the library does

The library represents an LH2 process as a set of replaceable physical models:

~~~text
property package
      |
thermodynamic states <--> two-port components <--> links and networks
      |                           |
      +--> tank balances          +--> valve, pipe, vaporizer, pump, compressor...
      |
      +--> sensor and HART observation layer
~~~

Each model exposes physical parameters and returns quantities with explicit units. A tank can be
used on its own, connected to a pipe and a valve, or inserted into a larger network. A simple
homogeneous tank and a spatially resolved layered or radial-axial tank share the same property and
flow conventions, so a study can increase spatial resolution without changing the surrounding
interface.

The implementation focuses on physics that are useful when a model must remain portable:

- para-hydrogen thermodynamic properties from CoolProp;
- mass, energy, and momentum accounting;
- vacuum-jacket and insulation heat paths;
- liquid, vapor, wall, and connected-line thermal capacitances;
- equilibrium and non-equilibrium phase-change options;
- compressible cryogenic flow through valves and pipes;
- transient component and network states;
- measurement dynamics and a PMD75B-style HART observation contract.

The code is parameter driven. It does not silently fit field time series, infer unknown ranges, or
replace a missing unit conversion. Site data may be used to specify initial conditions, boundary
conditions, or an independent validation case, while the physical model parameters remain visible
and reviewable.

## Observation loading and blind validation

The `lh2dt.data_pipeline` adapter accepts a facility-owned CSV or historian export through explicit
`ChannelSpec` declarations. It converts gauge pressure to absolute Pa, differential pressure to a
separate Pa quantity, and Celsius to K. Missing values remain visible in a validity mask; no
interpolation or sensor feedback is performed.

```python
from lh2dt import ChannelSpec, load_observation_csv, select_quiet_windows

channels = [
    ChannelSpec("PT", "tank_pressure", "pressure", "MPa", "gauge"),
    ChannelSpec("DPT", "level_dpt", "differential_pressure", "mmH2O", "differential"),
    ChannelSpec("TT", "tank_temperature", "temperature", "degC"),
]
table = load_observation_csv("private/export.csv", channels)
windows = select_quiet_windows(
    table, "tank_pressure", min_duration_s=6*3600,
    max_duration_s=12*3600, min_pressure_span_Pa=10_000,
)
```

`split_validation_window` reserves the first 30–60 minutes for a declared initial
wall/vapor state and keeps the following 4–8 hours blind. `compare_forward_window`
then applies the existing `compare_series` metric only to that blind interval. Sorting,
duplicate resolution, quality flags, event boundaries, and the capacity denominator are
all explicit in the input and result objects. See
[`docs/실측_비교_파이프라인.md`](docs/실측_비교_파이프라인.md).

## Private asset input provenance

Facility-specific tank dimensions, insulation paths, connected dead volumes, valve boundaries,
and instrument mappings belong in a private JSON/YAML file. The `asset_inputs` helpers require
`privacy: "private"`, disable plant-data optimisation, and audit every supplied parameter for a
unit, source, uncertainty, and asset applicability. The public package contains no TK-1101/TK-1102
telemetry or site configuration. See [`docs/비공개_설비_입력_스키마.md`](docs/비공개_설비_입력_스키마.md).

## Adaptive integration

For long transients or sharp valve/phase-change events, pass an
`AdaptiveStepPolicy` to `HomogeneousTank.simulate`. The policy can hit declared event times,
retry a trial state at a smaller step when the EOS domain or pressure/temperature change limit
is violated, and raise `AdaptiveIntegrationError` with the time, attempted step, retry count,
and reason when the minimum step is exhausted. It changes numerical resolution only; it does
not fit or alter physical parameters.

```python
from lh2dt import AdaptiveStepPolicy

policy = AdaptiveStepPolicy(
    minimum_step_s=0.5,
    maximum_step_s=120.0,
    event_times_s=(3600.0, 7200.0),
)
trace = tank.simulate(
    initial, duration_s=12*3600, time_step_s=60,
    ambient_temperature_K=293.15, flow_callback=boundary,
    adaptive_policy=policy,
)
```

## Visual network authoring contract

`NetworkDraft` is the GUI-neutral representation for an Aspen/HYSYS-like editor. It stores
component icons as nodes, typed ports, canvas positions, metadata, and pipe/valve links in JSON.
`load_network_draft` and `save_network_draft` validate node and port references before a desktop
or web editor hands the draft to a physical network builder. The draft is separate from the
numerical solver, so a user can move, replace, duplicate, or resize a component without changing
the model equations.

## Design principles

### Explicit physical contracts

Thermodynamic states, streams, component results, and network links are ordinary Python data
objects. A caller can inspect every pressure, temperature, enthalpy, mass flow, heat rate, and
residual rather than receiving an opaque solver result.

### Replaceable model levels

The same process can be represented at several levels:

1. A homogeneous equilibrium tank for quick system studies.
2. A layered tank for axial stratification, wall nodes, connected vapor volume, and phase-interface
   heat transfer.
3. A radial-axial tank for distributed liquid/vapor and wall cells with radial and axial
   conduction or circulation terms.

The surrounding links and data contracts remain stable while the internal tank model is replaced.

### Conservation before curve fitting

The primary checks are conservation and validity checks: finite values, valid EOS states, mass
balance, energy balance, pressure bounds, flow direction, and time-step convergence. Optional
time-series metrics such as nMAPE are applied after the physical simulation and are not used to
hide a missing boundary condition or an unknown sensor unit.

### SI units at the model boundary

All public model calls use SI units. Gauge pressure, engineering units, raw transmitter counts,
and display scaling belong in an adapter at the boundary. The internal state is always absolute
pressure and thermodynamic temperature.

## Installation

### Install a released package

Once the PyPI release is available, install it with:

~~~bash
python -m pip install lh2dt_models
~~~

The package requires Python 3.10 or newer and installs:

- CoolProp >= 8.0
- numpy >= 2.0
- scipy >= 1.14

### Install from GitHub for development

~~~bash
git clone https://github.com/lyullee/lh2dt_models.git
cd lh2dt_models
python -m venv .venv

# Windows PowerShell
.\\.venv\\Scripts\\Activate.ps1

# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
~~~

The editable install makes changes under src immediately available to examples and tests.

## A first dynamic tank simulation

The following example creates a ten-cubic-metre homogeneous tank, initializes it at saturation,
and advances it for ten minutes. A positive mass flow enters the tank; a negative mass flow leaves
the tank.

~~~python
from lh2dt import HydrogenProperties, HomogeneousTank, MassEnergyFlow, TankGeometry

properties = HydrogenProperties(fluid="ParaHydrogen")

tank = HomogeneousTank(
    TankGeometry(
        volume_m3=10.0,
        wall_heat_capacity_J_K=2.0e6,
        ambient_UA_W_K=8.0,
        fluid_wall_UA_W_K=15.0,
    ),
    properties,
)

initial = tank.initialize_saturated(
    pressure_Pa=130_000.0,
    liquid_volume_fraction=0.70,
    wall_temperature_K=293.15,
)

def boundary(_time_s, thermo):
    return [
        MassEnergyFlow(
            mass_flow_kg_s=0.0,
            specific_enthalpy_J_kg=thermo.specific_enthalpy_J_kg,
            source="closed boundary",
        )
    ]

trace = tank.simulate(
    initial,
    duration_s=600.0,
    time_step_s=10.0,
    ambient_temperature_K=293.15,
    flow_callback=boundary,
)

last = trace[-1]
print(f"pressure = {last.thermo.pressure_Pa:.1f} Pa")
print(f"temperature = {last.thermo.temperature_K:.3f} K")
print(f"energy residual = {last.total_energy_residual_J:.3e} J")
~~~

The complete runnable version is [examples/quickstart_tank.py](examples/quickstart_tank.py). The result trace contains the
thermodynamic state, phase inventory, wall state, and conservation diagnostics for every step.

## Model architecture

### 1. Property package

HydrogenProperties is the common equation-of-state and property service. It provides conversions
among pressure-temperature, pressure-enthalpy, pressure-internal-energy, saturation, density,
specific enthalpy, specific internal energy, entropy, quality, and phase labels. All components
receive the same property object so a connected network uses one consistent thermodynamic basis.

Typical calls are:

~~~python
from lh2dt import HydrogenProperties

props = HydrogenProperties(fluid="ParaHydrogen")
state_pt = props.from_pT(130_000.0, 20.5)
state_ph = props.from_ph(130_000.0, state_pt.specific_enthalpy_J_kg)
sat_liquid = props.saturated_liquid(130_000.0)
sat_vapor = props.saturated_vapor(130_000.0)
~~~

The exact convenience methods are discoverable from the class docstring and type signatures. The
important interface rule is that the returned ThermoState carries pressure in Pa, temperature in K,
and specific quantities in SI units.

### 2. Storage-tank balances

The tank family integrates fluid and wall inventories. Depending on model level, the state may
include:

- total mass and total internal energy;
- liquid and vapor inventories;
- liquid and vapor cell enthalpies;
- wall-node temperatures and heat capacities;
- connected vapor volume and connected-wall capacitance;
- interface position or phase-volume fractions;
- radial and axial cell inventories;
- optional circulation and distributed-boiling source terms.

For a closed control volume, the governing bookkeeping is expressed as:

~~~text
dM/dt = sum(mass inflow) - sum(mass outflow)
dU/dt = sum(mass flow * h) + Q_wall + Q_direct
~~~

The pressure is obtained from the selected property model and the current inventory. Layered and
radial-axial models additionally enforce a common rigid-vessel pressure through a pressure
projection step. The projection keeps cell volumes and the phase state physically consistent after
a time step.

### 3. Heat-transfer and insulation paths

Heat transfer is represented by explicit terms rather than a hidden correction factor. Depending
on the selected model, a tank may include:

- ambient-to-wall conductance (ambient_UA_W_K);
- wall-to-liquid and wall-to-vapor conductance;
- axial wall conduction;
- liquid and vapor axial conduction;
- interface conduction or a gas-interface accommodation coefficient;
- vacuum-jacket radiation and residual-gas conduction;
- MLI and support conduction through the vacuum-insulation utilities;
- connected pipe/nozzle wall heat capacity and ambient heat leak;
- natural-convection and natural-circulation source terms.

The layered model accepts separate liquid-side and vapor-side wall heat-transfer parameters. The
radial-axial model accepts separate radial and axial conductances and optional wall-patch area
fractions. This makes it possible to represent a warm upper wall and a colder lower wall without
changing the external network interface.

### 4. Phase change

The phase-change module supports an equilibrium energy-jump interface and a Schrage mass-flux
formulation with an accommodation coefficient. The layered and radial-axial tanks also expose an
optional distributed-boiling formulation. These options are controlled by named parameters, so a
study can record which mechanism was used and reproduce it later.

### 5. Flow components

Two-port components evaluate a left and right ThermoState and return a result containing at least
the signed mass flow and downstream specific enthalpy. Optional result fields report pressure drop,
momentum residual, heat rate, throat pressure, and mass flux.

The supplied component families include:

- Valve, CheckValve, CommandedValve, and StrokeLimitedCommandedValve;
- PressureReliefValve and TemperatureProtectionValve;
- Pipe and HEMPipe;
- dynamic HEM pipe with fluid and wall states;
- Vaporizer and AmbientVaporizer;
- Reliquefier and DynamicReliquefier;
- Pump and virtual H-Q pump;
- Compressor;
- VentStack.

The valve and pipe models account for compressible pressure-driven flow. HEM components use a
homogeneous-equilibrium-mixture treatment where appropriate. The vaporizer combines pressure drop,
ambient heat transfer, maximum heat duty, approach temperature, and optional flow-dependent UA.

### 6. Networks and links

SteadyNetwork solves a connected pressure network using component evaluations and node states.
DynamicNetworkSimulator advances selected dynamic nodes while resolving the hydraulic network at
each time step. Layered, stratified, and radial network wrappers preserve the same idea for spatial
tank states.

Network links carry MassEnergyFlow objects. A link can also report thermal power and hydraulic
outlet enthalpy; the validation layer checks that the reported thermal power closes with
abs(m_dot) * (h_out - h_hydraulic_out).

## Available model families

| Family | Main entry points | Physical content | Typical use |
|---|---|---|---|
| Properties | HydrogenProperties | Para-hydrogen EOS and derived properties | Every model and adapter |
| Homogeneous tank | TankGeometry, HomogeneousTank | One equilibrium fluid state plus wall capacitance | Fast system studies and initialization |
| Layered tank | LayeredTankParameters, LayeredTank | Liquid/vapor cells, wall nodes, interface heat transfer, connected vapor volume | Stratification and boil-off studies |
| Radial-axial tank | RadialAxialTankParameters, RadialAxialTank | Distributed liquid/vapor/wall cells, radial/axial conduction, circulation | Spatial sensitivity and detailed transient studies |
| Stratified networks | StratifiedNetwork | Multiple stratified tanks and links | Multi-tank fill, transfer, and vent studies |
| Radial networks | RadialNetwork | Radial-axial tank nodes and hydraulic links | Spatial network studies |
| Valves | Valve, CheckValve, commanded and protection valves | Pressure-driven opening, check behavior, relief and temperature trips | Isolation, fill, vent, and protection paths |
| Pipes | Pipe, HEMPipe, DynamicPipe | Darcy-Weisbach friction, local losses, elevation, heat transfer, dynamic line volume | Transfer and vent piping |
| Vaporizers | Vaporizer, AmbientVaporizer, DynamicVaporizer | Heat duty, approach temperature, pressure drop, transient wall/flow states | LH2-to-gas delivery studies |
| Reliquefiers | Reliquefier, DynamicReliquefier | Rejection/removal of vapor enthalpy and transient capacity | Optional closed-loop studies |
| Rotating equipment | Pump, Compressor | Head/pressure-ratio and efficiency contracts | Supply and compression paths |
| Instrumentation | FirstOrderTransmitter, PMD75BHartObservation | Sensor lag, range conversion, HART dynamic variables | Measurement-layer simulation |
| Validation | validation, numerical_validation, accuracy, benchmarks | Residuals, convergence, nMAPE, public benchmark fixtures | Verification and reproducible comparison |

## Connecting components

The smallest integration unit is a two-port component. A caller supplies a left and right state;
the component returns a signed flow. Positive flow is left-to-right, negative flow is right-to-left,
and zero flow is a valid closed result.

~~~python
from lh2dt import HEMPipe, HydrogenProperties, Valve

props = HydrogenProperties()
upstream = props.from_pT(180_000.0, 21.0)
downstream = props.from_pT(130_000.0, 20.4)

valve = Valve(
    full_open_area_m2=2.0e-4,
    discharge_coefficient=0.72,
    properties=props,
)
pipe = HEMPipe(
    length_m=8.0,
    inner_diameter_m=0.012,
    roughness_m=1.5e-6,
    local_loss_coefficient=2.0,
    properties=props,
)

valve_result = valve.evaluate(upstream, downstream, opening=0.50)
pipe_result = pipe.evaluate(upstream, downstream)
print(valve_result.mass_flow_kg_s)
print(pipe_result.pressure_drop_Pa)
~~~

For a reusable process network, define named nodes and links and let the network solver evaluate
the pressure-dependent components. The dynamic simulator can update commands at each time step:

~~~python
commands = {"vent_opening": 0.0}

def command_callback(time_s, network):
    # An application adapter applies this command to the CommandedValve
    # used by the corresponding network link.
    commands["vent_opening"] = 0.20 if time_s > 300.0 else 0.0

simulator = DynamicNetworkSimulator(
    network=steady_network,
    dynamic_nodes=dynamic_nodes,
    command_callback=command_callback,
)
trace = simulator.simulate(duration_s=900.0, time_step_s=5.0)
~~~

The exact network construction is application-specific. Start with
[examples/run_component_linkage_demo.py](examples/run_component_linkage_demo.py) and
[examples/run_hem_pipe_dynamic_network.py](examples/run_hem_pipe_dynamic_network.py).

## Inputs, outputs, and unit conventions

### Required unit convention

| Quantity | Unit | Sign or reference convention |
|---|---|---|
| Pressure | Pa | Absolute pressure inside the model |
| Gauge pressure | Pa(g) at an external boundary | Convert before calling the model |
| Temperature | K | Thermodynamic temperature |
| Mass | kg | Positive inventory |
| Mass flow | kg/s | Positive left-to-right or into a control volume |
| Specific enthalpy | J/kg | Transported with the mass flow |
| Specific internal energy | J/kg | Fluid inventory energy |
| Heat rate | W | Positive into the selected control volume |
| Heat capacity | J/K | Lumped wall or line capacitance |
| Conductance / UA | W/K | Positive heat-transfer conductance |
| Volume | m³ | Rigid geometric or connected volume |
| Area | m² | Flow or heat-transfer area |
| Length / elevation | m | Elevation increase follows the pipe convention |
| Time | s | Integration and command time |

Convert gauge pressure at the adapter boundary:

~~~python
from lh2dt import gauge_to_absolute_pressure_Pa

absolute_pressure = gauge_to_absolute_pressure_Pa(
    gauge_pressure_Pa=0.383e6,
    ambient_pressure_Pa=101_325.0,
)
~~~

Do not put a display value such as 0.383 MPa(g) directly into pressure_Pa. The model cannot infer
whether a number is gauge or absolute pressure, and an incorrect reference pressure changes both
saturation properties and mass inventory.

### Common tank inputs

For a homogeneous tank:

- rigid volume;
- wall heat capacity;
- ambient-to-wall UA;
- fluid-to-wall UA;
- initial pressure and liquid volume fraction;
- wall temperature, ambient temperature, and boundary streams.

For a layered tank, add liquid/vapor cell counts, wall-node count, separate wall-side
conductances, connected vapor volume, connected wall capacitance, wall and fluid axial conductance,
interface model, phase-change model, and optional circulation/distributed-boiling settings.

For a radial-axial tank, add liquid/vapor level counts, radial boundary geometry, fluid/wall volume
fractions, radial and axial conductances, interface UA, wall patch area fractions, and optional
circulation sources.

### Common outputs

Every tank step returns a state and diagnostics that can be recorded in a table or adapted to a
historian interface:

- time;
- pressure, temperature, density, enthalpy, internal energy, entropy, phase, and quality;
- total mass and energy;
- liquid/vapor inventory and level-related quantities where available;
- wall temperatures and heat rates;
- boundary mass and energy ledgers;
- mass, energy, and momentum residuals;
- pressure-projection and numerical diagnostics.

Network steps additionally expose node states, link results, commanded openings, pressure-solver
status, and boundary ledgers. Instrumentation adapters can then convert physical outputs to the
same engineering units and dynamic response used by the external control system.

## Dynamic simulation workflow

The recommended workflow is explicit:

1. **Define the property package.** Create one HydrogenProperties instance for the connected study.
2. **Build geometry and equipment.** Use documented geometry, material, insulation, valve, pipe,
   and heat-transfer parameters.
3. **Choose the minimum model level that resolves the question.** Start homogeneous, then replace
   the tank with layered or radial-axial resolution if stratification or wall gradients matter.
4. **Initialize a valid thermodynamic state.** Use a public initialize method rather than manually
   constructing inconsistent mass, enthalpy, and pressure values.
5. **Define boundaries and commands.** Supply ambient temperature, inlet/outlet streams, valve
   commands, equipment duties, and known line volumes explicitly.
6. **Advance the state.** Use simulate, step, or DynamicNetworkSimulator with a documented time step.
7. **Inspect conservation.** Reject a run with non-finite values, invalid EOS states, large mass or
   energy residuals, or a failed pressure solve.
8. **Export a reproducible trace.** Store parameter values, initial state, boundary functions,
   solver settings, package version, and output trace together.
9. **Compare to independent observations.** Align timestamps and units at the adapter boundary;
   calculate nMAPE or another metric only after the physical run is complete.

The first 30–60 minutes of a real transient may be used to define a physically justified initial
wall or vapor state. If that practice is used, record the initialization window separately from the
blind prediction window so the validation result remains auditable.

## Instrumentation and HART observations

The instrumentation layer is separate from the process model. This prevents a transmitter range,
first-order lag, display scaling, or HART variable assignment from being confused with physical
pressure or temperature.

### First-order transmitter

TransmitterParameters defines engineering range, time constant, bias, and optional noise or
quantization settings. FirstOrderTransmitter.advance takes a process value and returns a new
transmitter state plus a reading. Use this layer when a comparison must reproduce a real sensor
response rather than an ideal process measurement.

### PMD75B-style HART observation

PMD75BHartObservation models an observation contract for a 4–20 mA HART differential-pressure
transmitter. It keeps raw sensor pressure, position-adjusted pressure, sensor temperature, and
electronics temperature as separate inputs and exposes dynamic variables as an explicit mapping.
The HART layer does not invent a physical pressure range or level scaling; those values must be
provided by the instrument configuration adapter.

~~~python
from lh2dt import PMD75BHartObservation

observation = PMD75BHartObservation()
hart_state = observation.initialize(position_adjusted_pressure=130_000.0)
hart_state, variables = observation.advance(
    hart_state,
    raw_sensor_pressure=130_000.0,
    position_adjusted_pressure=130_000.0,
    sensor_temperature=20.5,
    electronics_temperature=293.15,
    time_step_s=1.0,
)
print(variables.as_dict())
~~~

This separation allows a facility adapter to map a real PT/DPT/TT tag, unit, timestamp, HART
address, and PV/SV/TV/QV assignment without changing the tank or network equations.

## Validation and reproducibility

The repository includes a public-reference validation suite and model-level tests. The current
release test suite contains 340 passing tests covering properties, tank balances, stratification,
radial-axial transport, phase change, insulation, natural circulation, valves, pipes, vaporizers,
reliquefiers, networks, instrumentation, and numerical validation.

Run the checks locally:

~~~bash
pytest -q
python -m compileall -q src examples
python -m build
python -m twine check dist/*
~~~

The benchmark examples are based on public NASA TN D-4171 reference cases included under
src/lh2dt/data/. They are regression fixtures for the implementation; they are not a claim that
one facility has the same geometry or operating boundary.

The accuracy helper defines the normalized mean absolute percentage error used for an aligned
comparison:

~~~text
nMAPE = (100 / N) * sum(abs(y_i - y_hat_i) / y_c)
~~~

The denominator y_c is an explicitly supplied comparison scale. A validation report should state
the scale, timestamp alignment, excluded or unavailable intervals, model parameter set, and whether
parameters were fixed from design data or estimated for a declared calibration case. The library
itself does not claim a universal facility-level accuracy percentage.

## Choosing a model level

| Question | Recommended starting point | Upgrade when |
|---|---|---|
| How does total tank pressure respond to a heat leak? | HomogeneousTank | A single fluid state cannot represent the observed transient |
| Does a warm upper wall or vapor space change the response? | LayeredTank | Axial cells are still too coarse |
| Are radial wall paths, local hot spots, or circulation important? | RadialAxialTank | A network-level question requires connected equipment |
| Does a transfer or vent path control the pressure? | Tank + Valve + HEMPipe | Line inventory and wall thermal lag matter |
| Does a gas-delivery path need heat duty and pressure drop? | Vaporizer or DynamicVaporizer | The equipment wall and flow history must be resolved |
| Does the sensor response affect the comparison? | FirstOrderTransmitter / HART layer | A real instrument adapter is available |

Increasing spatial resolution adds state variables and solver work. It should be done because the
physical question requires it, not because a larger model automatically improves a metric.

## Project layout

~~~text
lh2dt_models/
├── src/lh2dt/                 # canonical implementation package
│   ├── properties.py          # hydrogen property and EOS service
│   ├── tank.py                # homogeneous tank
│   ├── layered_tank.py        # layered liquid/vapor/wall tank
│   ├── radial_axial_tank.py   # distributed radial-axial tank
│   ├── pipe.py                # steady and HEM pipe models
│   ├── dynamic_pipe.py        # dynamic line inventory and wall state
│   ├── valve.py               # flow, command, check, relief, protection valves
│   ├── vaporizer.py           # steady vaporizer and pressure-drop relations
│   ├── dynamic_vaporizer.py   # transient vaporizer model
│   ├── reliquefier.py         # steady and transient reliquefier models
│   ├── pump.py, compressor.py # rotating-equipment contracts
│   ├── network.py             # steady network
│   ├── dynamic_network.py     # dynamic network simulator
│   ├── instrumentation.py     # transmitter dynamics
│   ├── hart.py                # HART observation contract
│   ├── validation.py          # conservation and state checks
│   └── data/                  # public benchmark data only
├── src/lh2dt_models/           # compatibility import namespace
├── examples/                   # runnable model and benchmark examples
├── tests/                      # unit, conservation, and numerical tests
├── docs/                       # detailed English and Korean references
├── CITATION.cff                # citation metadata
├── .zenodo.json                # Zenodo metadata template
├── LICENSE                     # MIT license
└── pyproject.toml              # package, dependency, and build metadata
~~~

The package does not load site-specific files implicitly. A facility-specific application should
keep its adapters, tag maps, calibration records, and operational data outside this public package
and pass validated SI values through the documented interfaces.

## Documentation

### English

- docs/model_reference.md — model hierarchy, state variables, physics, equations, and implementation
  notes.
- docs/interface_reference.md — component, stream, network, instrumentation, and adapter
  interfaces.
- docs/release_and_citation.md — release, DOI, and citation instructions.

### Korean

- docs/모델_상세_및_물리방정식.md — 상세 모델과 물리방정식.
- docs/인터페이스_상세_및_연결가이드.md — 인터페이스와 연결 가이드.
- docs/모델_사용_및_인터페이스.md — 사용 절차와 예제.
- docs/물리모델_구조와_방정식.md — 구조와 방정식 개요.
- docs/유닛별_입출력_사양.md — 유닛별 입출력 사양.
- docs/검증과_재현성.md — 검증과 재현성 절차.
- docs/배포_및_인용_가이드.md — 배포·DOI·인용 가이드.

## Development

Install the development extras and run the complete suite:

~~~bash
python -m pip install -e ".[dev]"
pytest -q
~~~

Before submitting a change:

1. Add or update a focused test for the physical behavior or interface contract.
2. Run the complete test suite.
3. Run examples touched by the change.
4. Check that all new public parameters have SI units and clear names.
5. Record any change in model assumptions or benchmark behavior in the English and Korean
   documentation when applicable.

The GitHub Actions workflow runs tests and package builds on supported Python versions. The release
workflow is configured for a trusted PyPI publisher so a GitHub release can publish the exact source
distribution and wheel without storing a long-lived upload token in the repository.

## Release, DOI, and citation

The project is released from Git tags. A release contains:

- the source distribution (.tar.gz);
- the wheel (.whl);
- the exact commit used to build both artifacts;
- CITATION.cff and .zenodo.json metadata.

Zenodo can archive each GitHub release and assign a persistent DOI. The DOI for a release is created
by Zenodo after the repository is enabled and the release is processed; do not invent or hard-code a
DOI before it is visible in the Zenodo record. After a DOI is assigned, cite the specific release
record rather than an unversioned development checkout.

For package users, cite the release tag and repository URL. For academic work, include the model
version, property backend, model level, physical parameter source, time step, boundary conditions,
and validation data selection in addition to the citation.

## Scope and responsible use

This is an engineering and research model library. It is not a safety-rated controller, a pressure
vessel design-code implementation, or a substitute for regulatory certification. Users must verify
the applicable pressure range, fluid property range, insulation construction, valve coefficient,
pipe roughness, phase-change correlation, heat-transfer boundary, and numerical time step for their
application.

The library intentionally leaves safety decisions visible to the application: protection devices,
trip logic, operating limits, and alarm policies should be represented as explicit components or
controllers and reviewed by a qualified engineer. A converged numerical result is not, by itself,
proof that a physical installation is safe.

## License

This project is distributed under the MIT License. See LICENSE.
