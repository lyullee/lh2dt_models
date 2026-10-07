import math
import unittest

from lh2dt import (
    CheckValve,
    HydrogenProperties,
    HEMPipe,
    Pipe,
    Pump,
    MappedPump,
    Reliquefier,
    Valve,
    Vaporizer,
    VaporizerLink,
    VirtualPump,
    ReliquefierLink,
    VentStack,
)


class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.properties = HydrogenProperties()

    def test_valve_is_directional_and_isenthalpic(self):
        upstream = self.properties.from_pT(600_000.0, 35.0)
        downstream = self.properties.from_pT(200_000.0, 35.0)
        valve = Valve(2.0e-6, 0.8, pressure_samples=24)
        result = valve.evaluate(upstream, downstream, opening=0.5)
        self.assertGreater(result.mass_flow_kg_s, 0.0)
        self.assertEqual(result.upstream_side, "left")
        self.assertAlmostEqual(result.outlet_specific_enthalpy_J_kg, upstream.specific_enthalpy_J_kg)
        reverse = valve.evaluate(downstream, upstream, opening=0.5)
        self.assertLess(reverse.mass_flow_kg_s, 0.0)

    def test_closed_valve_has_zero_flow(self):
        left = self.properties.from_pT(600_000.0, 35.0)
        right = self.properties.from_pT(200_000.0, 35.0)
        result = Valve(2.0e-6, 0.8, pressure_samples=24).evaluate(left, right, opening=0.0)
        self.assertEqual(result.mass_flow_kg_s, 0.0)

    def test_closed_valve_can_represent_finite_seat_leakage(self):
        left = self.properties.from_pT(600_000.0, 35.0)
        right = self.properties.from_pT(200_000.0, 35.0)
        leakage_CdA = 2.0e-9
        valve = Valve(
            2.0e-6,
            0.8,
            pressure_samples=24,
            closed_leakage_CdA_m2=leakage_CdA,
        )
        closed = valve.evaluate(left, right, opening=0.0)
        opened = valve.evaluate(left, right, opening=1.0)
        self.assertGreater(closed.mass_flow_kg_s, 0.0)
        self.assertAlmostEqual(
            closed.mass_flow_kg_s,
            leakage_CdA * closed.mass_flux_kg_m2_s,
        )
        self.assertAlmostEqual(
            opened.mass_flow_kg_s,
            0.8 * 2.0e-6 * opened.mass_flux_kg_m2_s,
        )

    def test_closed_valve_rejects_leakage_above_full_open_CdA(self):
        with self.assertRaises(ValueError):
            Valve(2.0e-6, 0.8, closed_leakage_CdA_m2=2.0e-6)

    def test_valve_near_equal_pressure_stays_within_expansion_interval(self):
        upstream = self.properties.from_pT(300_000.0, 300.0)
        downstream = self.properties.from_pT(299_999.9999, 300.0)
        area = 1.0e-4
        discharge_coefficient = 0.8
        result = Valve(area, discharge_coefficient, self.properties).evaluate(
            upstream, downstream
        )
        expected_incompressible_limit = (
            discharge_coefficient * area
            * math.sqrt(2.0 * upstream.density_kg_m3
                        * (upstream.pressure_Pa - downstream.pressure_Pa))
        )
        self.assertGreaterEqual(result.throat_pressure_Pa, downstream.pressure_Pa)
        self.assertLessEqual(result.throat_pressure_Pa, upstream.pressure_Pa)
        self.assertAlmostEqual(
            result.mass_flow_kg_s / expected_incompressible_limit, 1.0, delta=0.01
        )
        self.assertFalse(result.choked)

    def test_check_valve_requires_cracking_drop_and_blocks_reverse_flow(self):
        upstream = self.properties.from_pT(600_000.0, 35.0)
        just_below = self.properties.from_pT(501_000.0, 35.0)
        downstream = self.properties.from_pT(500_000.0, 35.0)
        check = CheckValve(
            2.0e-6,
            0.8,
            cracking_pressure_Pa=2_000.0,
            pressure_samples=24,
        )

        closed = check.evaluate(just_below, downstream)
        forward = check.evaluate(upstream, downstream)
        reverse = check.evaluate(downstream, upstream)

        self.assertEqual(closed.mass_flow_kg_s, 0.0)
        self.assertGreater(forward.mass_flow_kg_s, 0.0)
        self.assertEqual(reverse.mass_flow_kg_s, 0.0)
        self.assertEqual(forward.outlet_specific_enthalpy_J_kg, upstream.specific_enthalpy_J_kg)

    def test_check_valve_rejects_negative_cracking_pressure(self):
        with self.assertRaises(ValueError):
            CheckValve(2.0e-6, 0.8, cracking_pressure_Pa=-1.0)

    def test_check_valve_near_cracking_pressure_uses_remaining_drop_only(self):
        upstream = self.properties.from_pT(300_000.0, 300.0)
        downstream = self.properties.from_pT(297_999.9999, 300.0)
        cracking = 2_000.0
        result = CheckValve(
            1.0e-4, 0.8, cracking_pressure_Pa=cracking,
            properties=self.properties,
        ).evaluate(upstream, downstream)
        self.assertGreater(result.mass_flow_kg_s, 0.0)
        self.assertGreaterEqual(
            result.throat_pressure_Pa, downstream.pressure_Pa + cracking
        )
        self.assertFalse(result.choked)

    def test_valve_family_rejects_boolean_inputs(self):
        with self.assertRaises(ValueError):
            Valve(True, 0.8)
        with self.assertRaises(ValueError):
            Valve(2.0e-6, True)
        with self.assertRaises(ValueError):
            Valve(2.0e-6, 0.8, pressure_samples=True)
        with self.assertRaises(ValueError):
            CheckValve(2.0e-6, 0.8, cracking_pressure_Pa=True)
        with self.assertRaises(ValueError):
            Valve(2.0e-6, 0.8, allow_reverse="false")

    def test_vaporizer_and_vent_stack_reject_boolean_thermal_inputs(self):
        with self.assertRaises(ValueError):
            Vaporizer.from_resistance_network(
                area_m2=10.0,
                resistance_terms_m2K_W=[0.1, True],
                maximum_heat_W=1.0e5,
            )
        with self.assertRaises(ValueError):
            VentStack(10.0, 0.05, 1.0e-6, ambient_temperature_K=True)
        with self.assertRaises(ValueError):
            VentStack(10.0, 0.05, 1.0e-6, ambient_UA_W_K=True)

    def test_valve_recovers_ideal_gas_choked_flow_limit(self):
        upstream = self.properties.from_pT(100_000.0, 300.0)
        downstream = self.properties.from_pT(10_000.0, 300.0)
        result = Valve(
            1.0, 1.0, self.properties, pressure_samples=240
        ).evaluate(upstream, downstream)
        cp = self.properties.isobaric_heat_capacity_J_kgK(upstream)
        cv = self.properties.isochoric_heat_capacity_J_kgK(upstream)
        gamma = cp / cv
        gas_constant = self.properties.specific_gas_constant_J_kgK()
        ideal_flux = upstream.pressure_Pa * math.sqrt(
            gamma / (gas_constant * upstream.temperature_K)
        ) * (2.0 / (gamma + 1.0)) ** (
            (gamma + 1.0) / (2.0 * (gamma - 1.0))
        )
        self.assertTrue(result.choked)
        self.assertAlmostEqual(result.mass_flux_kg_m2_s / ideal_flux, 1.0, delta=0.005)

    def test_pipe_momentum_and_heat_balance(self):
        left = self.properties.from_pT(600_000.0, 20.0)
        right = self.properties.from_ph(500_000.0, left.specific_enthalpy_J_kg)
        pipe = Pipe(10.0, 0.02, 1.0e-6, local_loss_coefficient=2.0, heat_into_fluid_W=50.0)
        result = pipe.evaluate(left, right)
        self.assertGreater(result.mass_flow_kg_s, 0.0)
        self.assertGreater(result.reynolds_number, 0.0)
        self.assertAlmostEqual(
            result.mass_flow_kg_s * (result.outlet_specific_enthalpy_J_kg - left.specific_enthalpy_J_kg),
            50.0,
            delta=1e-7,
        )
        self.assertAlmostEqual(result.momentum_residual_Pa, 0.0, delta=1.0e-8)
        self.assertAlmostEqual(
            result.pressure_drop_Pa,
            left.pressure_Pa - right.pressure_Pa,
            delta=1.0e-8,
        )

    def test_pipe_chooses_reverse_flow_when_elevation_overcomes_pressure_difference(self):
        # The right endpoint is higher.  Its pressure is only slightly lower,
        # so the pressure-only direction heuristic would incorrectly return
        # zero flow even though the available total head drives the fluid
        # downwards from right to left.
        left = self.properties.from_pT(300_000.0, 20.0)
        right = self.properties.from_ph(299_500.0, left.specific_enthalpy_J_kg)
        pipe = Pipe(
            10.0,
            0.02,
            1.0e-6,
            local_loss_coefficient=2.0,
            elevation_change_m=20.0,
            properties=self.properties,
        )
        result = pipe.evaluate(left, right)
        self.assertLess(result.mass_flow_kg_s, 0.0)
        self.assertEqual(result.upstream_side, "right")
        self.assertAlmostEqual(result.momentum_residual_Pa, 0.0, delta=1.0e-8)

    def test_pipe_zero_flow_at_equal_pressure_and_elevation_has_closed_momentum(self):
        state = self.properties.from_pT(300_000.0, 20.0)
        pipe = Pipe(
            10.0,
            0.02,
            1.0e-6,
            local_loss_coefficient=2.0,
            properties=self.properties,
        )
        result = pipe.evaluate(state, state)
        self.assertEqual(result.mass_flow_kg_s, 0.0)
        self.assertAlmostEqual(result.momentum_residual_Pa, 0.0, delta=1.0e-12)

    def test_pipe_rejects_two_phase_input(self):
        left = self.properties.from_ph(
            300_000.0,
            0.5 * (
                self.properties.saturated_liquid(300_000.0).specific_enthalpy_J_kg
                + self.properties.saturated_vapor(300_000.0).specific_enthalpy_J_kg
            ),
        )
        right = self.properties.from_pT(200_000.0, 30.0)
        with self.assertRaises(ValueError):
            Pipe(10.0, 0.02, 1e-6, local_loss_coefficient=1.0).evaluate(left, right)

    def test_pipe_rejects_nonfinite_constructor_inputs(self):
        with self.assertRaises(ValueError):
            Pipe(float("nan"), 0.02, 1.0e-6)
        with self.assertRaises(ValueError):
            HEMPipe(1.0, 0.02, float("inf"))
        with self.assertRaises(ValueError):
            Pipe(1.0, 0.02, 1.0e-6, local_loss_coefficient=float("nan"))

    def test_hem_pipe_accepts_two_phase_input_and_closes_heat_balance(self):
        pressure = 300_000.0
        liquid = self.properties.saturated_liquid(pressure)
        vapor = self.properties.saturated_vapor(pressure)
        inlet = self.properties.from_ph(
            pressure,
            liquid.specific_enthalpy_J_kg + 0.35 * (
                vapor.specific_enthalpy_J_kg - liquid.specific_enthalpy_J_kg
            ),
        )
        outlet = self.properties.from_ph(250_000.0, inlet.specific_enthalpy_J_kg)
        pipe = HEMPipe(
            10.0,
            0.02,
            1.0e-6,
            local_loss_coefficient=2.0,
            heat_into_fluid_W=25.0,
        )
        result = pipe.evaluate(inlet, outlet)
        self.assertIsNotNone(inlet.quality)
        self.assertGreater(result.mass_flow_kg_s, 0.0)
        self.assertGreater(result.reynolds_number, 0.0)
        self.assertAlmostEqual(
            result.mass_flow_kg_s * (
                result.outlet_specific_enthalpy_J_kg - inlet.specific_enthalpy_J_kg
            ),
            25.0,
            delta=1e-7,
        )
        self.assertAlmostEqual(result.momentum_residual_Pa, 0.0, delta=1.0e-8)

    def test_hem_pipe_reduces_to_single_phase_pipe_contract(self):
        left = self.properties.from_pT(600_000.0, 20.0)
        right = self.properties.from_ph(500_000.0, left.specific_enthalpy_J_kg)
        ordinary = Pipe(10.0, 0.02, 1.0e-6, local_loss_coefficient=2.0).evaluate(left, right)
        hem = HEMPipe(10.0, 0.02, 1.0e-6, local_loss_coefficient=2.0).evaluate(left, right)
        self.assertAlmostEqual(hem.mass_flow_kg_s, ordinary.mass_flow_kg_s, delta=1e-12)
        self.assertAlmostEqual(hem.outlet_specific_enthalpy_J_kg, ordinary.outlet_specific_enthalpy_J_kg, delta=1e-12)

    def test_vent_stack_is_one_way_and_closes_momentum(self):
        upstream = self.properties.from_pT(600_000.0, 300.0)
        downstream = self.properties.from_ph(500_000.0, upstream.specific_enthalpy_J_kg)
        stack = VentStack(
            10.0,
            0.05,
            1.0e-6,
            local_loss_coefficient=2.0,
            ambient_temperature_K=300.0,
            ambient_UA_W_K=0.0,
            properties=self.properties,
        )
        result = stack.evaluate(upstream, downstream)
        self.assertGreater(result.mass_flow_kg_s, 0.0)
        self.assertAlmostEqual(result.heat_into_fluid_W, 0.0, delta=1.0e-12)
        self.assertAlmostEqual(result.momentum_residual_Pa, 0.0, delta=1.0e-8)
        reverse = stack.evaluate(downstream, upstream)
        self.assertEqual(reverse.mass_flow_kg_s, 0.0)
        self.assertIsNone(reverse.upstream_side)
        self.assertIsNone(reverse.momentum_residual_Pa)

    def test_vent_stack_blocks_gravity_driven_reverse_flow_despite_higher_left_pressure(self):
        left = self.properties.from_pT(300_000.0, 20.0)
        right = self.properties.from_pT(295_000.0, 20.0)
        stack = VentStack(20.0, 0.05, 1.0e-6, elevation_change_m=20.0,
                          properties=self.properties)
        self.assertLess(stack.pipe.evaluate(left, right).mass_flow_kg_s, 0.0)
        result = stack.evaluate(left, right)
        self.assertEqual(result.mass_flow_kg_s, 0.0)
        self.assertIsNone(result.upstream_side)
        self.assertIsNone(result.momentum_residual_Pa)

    def test_vent_stack_allows_gravity_driven_forward_flow_despite_lower_left_pressure(self):
        left = self.properties.from_pT(295_000.0, 20.0)
        right = self.properties.from_pT(300_000.0, 20.0)
        stack = VentStack(20.0, 0.05, 1.0e-6, elevation_change_m=-20.0,
                          properties=self.properties)
        result = stack.evaluate(left, right)
        self.assertGreater(result.mass_flow_kg_s, 0.0)
        self.assertEqual(result.upstream_side, "left")
        self.assertAlmostEqual(result.momentum_residual_Pa, 0.0, delta=1.0e-8)

    def test_vent_stack_ambient_exchange_closes_first_law_without_overshoot(self):
        upstream = self.properties.from_pT(600_000.0, 100.0)
        downstream = self.properties.from_ph(500_000.0, upstream.specific_enthalpy_J_kg)
        stack = VentStack(
            10.0,
            0.05,
            1.0e-6,
            local_loss_coefficient=2.0,
            ambient_temperature_K=300.0,
            ambient_UA_W_K=100.0,
            properties=self.properties,
        )
        result = stack.evaluate(upstream, downstream)
        self.assertGreater(result.mass_flow_kg_s, 0.0)
        self.assertGreater(result.heat_into_fluid_W, 0.0)
        self.assertAlmostEqual(
            result.mass_flow_kg_s
            * (result.outlet_specific_enthalpy_J_kg - upstream.specific_enthalpy_J_kg),
            result.heat_into_fluid_W,
            delta=1.0e-6,
        )
        outlet = self.properties.from_ph(
            downstream.pressure_Pa, result.outlet_specific_enthalpy_J_kg
        )
        self.assertLess(outlet.temperature_K, 300.0)
        self.assertGreater(outlet.temperature_K, upstream.temperature_K)

    def test_vent_stack_can_be_explicitly_bidirectional(self):
        upstream = self.properties.from_pT(500_000.0, 300.0)
        downstream = self.properties.from_ph(600_000.0, upstream.specific_enthalpy_J_kg)
        stack = VentStack(
            10.0,
            0.05,
            1.0e-6,
            local_loss_coefficient=2.0,
            ambient_temperature_K=300.0,
            ambient_UA_W_K=0.0,
            properties=self.properties,
            allow_reverse=True,
        )
        result = stack.evaluate(upstream, downstream)
        self.assertLess(result.mass_flow_kg_s, 0.0)
        self.assertEqual(result.upstream_side, "right")

    def test_vent_stack_requires_boolean_direction_flag(self):
        with self.assertRaises(ValueError):
            VentStack(10.0, 0.05, 1.0e-6, allow_reverse="false")

    def test_vent_stack_rejects_boolean_integration_segments(self):
        with self.assertRaises(ValueError):
            VentStack(10.0, 0.05, 1.0e-6, integration_segments=True)

    def test_laminar_pipe_recovers_hagen_poiseuille_limit(self):
        left = self.properties.from_pT(300_000.0, 20.0)
        right = self.properties.from_ph(
            299_990.0, left.specific_enthalpy_J_kg
        )
        length = 10.0
        diameter = 0.005
        result = Pipe(length, diameter, 0.0).evaluate(left, right)
        viscosity = self.properties.viscosity_Pa_s(left)
        expected_mass_flow = (
            left.density_kg_m3 * math.pi * diameter**4 * 10.0
            / (128.0 * viscosity * length)
        )
        self.assertLess(result.reynolds_number, 2_300.0)
        self.assertAlmostEqual(
            result.mass_flow_kg_s / expected_mass_flow, 1.0, delta=1.0e-6
        )

    def test_vaporizer_first_law(self):
        inlet = self.properties.saturated_liquid(300_000.0)
        result = Vaporizer(500.0, 200_000.0).evaluate(inlet, 280_000.0, 0.05, 290.0)
        self.assertAlmostEqual(
            0.05 * (result.outlet.specific_enthalpy_J_kg - inlet.specific_enthalpy_J_kg),
            result.heat_into_hydrogen_W,
            delta=1e-7,
        )

    def test_vaporizer_closed_flow_returns_stagnant_pressure_adjusted_state(self):
        inlet = self.properties.saturated_liquid(300_000.0)
        result = Vaporizer(500.0, 200_000.0).evaluate(
            inlet, 280_000.0, 0.0, 290.0
        )
        assert result.heat_into_hydrogen_W == 0.0
        assert result.requested_heat_W == 0.0
        assert result.capacity_limited is False
        assert result.outlet.pressure_Pa == 280_000.0
        assert result.outlet.specific_enthalpy_J_kg == inlet.specific_enthalpy_J_kg

    def test_vaporizer_recovers_constant_cp_infinite_reservoir_limit(self):
        inlet = self.properties.from_pT(100_000.0, 300.0)
        mass_flow = 0.1
        conductance = 1_000.0
        ambient = 350.0
        result = Vaporizer(
            conductance, 1.0e9, self.properties
        ).evaluate(inlet, inlet.pressure_Pa, mass_flow, ambient)
        cp = self.properties.isobaric_heat_capacity_J_kgK(inlet)
        expected_temperature = ambient - (ambient - inlet.temperature_K) * math.exp(
            -conductance / (mass_flow * cp)
        )
        self.assertAlmostEqual(
            result.outlet.temperature_K, expected_temperature, delta=0.2
        )

    def test_vaporizer_applies_explicit_ambient_approach_boundary(self):
        inlet = self.properties.from_pT(100_000.0, 100.0)
        mass_flow = 0.05
        no_approach = Vaporizer(
            1_000.0, 1.0e9, self.properties
        ).evaluate(inlet, inlet.pressure_Pa, mass_flow, 320.0)
        approach_limited = Vaporizer(
            1_000.0, 1.0e9, self.properties, minimum_approach_K=20.0
        ).evaluate(inlet, inlet.pressure_Pa, mass_flow, 340.0)
        self.assertAlmostEqual(
            approach_limited.outlet.temperature_K,
            no_approach.outlet.temperature_K,
            delta=1.0e-8,
        )

    def test_vaporizer_rejects_invalid_approach_boundary(self):
        with self.assertRaises(ValueError):
            Vaporizer(1_000.0, 1.0e6, self.properties, minimum_approach_K=-1.0)
        with self.assertRaises(ValueError):
            Vaporizer(1_000.0, 1.0e6, self.properties, minimum_approach_K=True)
        inlet = self.properties.from_pT(100_000.0, 100.0)
        vaporizer = Vaporizer(
            1_000.0, 1.0e6, self.properties, minimum_approach_K=300.0
        )
        with self.assertRaises(ValueError):
            vaporizer.evaluate(inlet, inlet.pressure_Pa, 0.1, 290.0)

    def test_vaporizer_resistance_network_derives_explicit_conductance(self):
        vaporizer = Vaporizer.from_resistance_network(
            area_m2=10.0,
            resistance_terms_m2K_W=(0.1, 0.2, 0.05),
            maximum_heat_W=2_000.0,
            properties=self.properties,
        )

        self.assertAlmostEqual(vaporizer.UA, 10.0 / 0.35)
        self.assertEqual(vaporizer.parameterization, "resistance_network")
        self.assertEqual(vaporizer.heat_transfer_area_m2, 10.0)
        self.assertEqual(vaporizer.resistance_terms_m2K_W, (0.1, 0.2, 0.05))

    def test_vaporizer_resistance_network_rejects_incomplete_terms(self):
        with self.assertRaises(ValueError):
            Vaporizer.from_resistance_network(
                area_m2=10.0,
                resistance_terms_m2K_W=(),
                maximum_heat_W=2_000.0,
                properties=self.properties,
            )
        with self.assertRaises(ValueError):
            Vaporizer.from_resistance_network(
                area_m2=10.0,
                resistance_terms_m2K_W=(0.1, 0.0),
                maximum_heat_W=2_000.0,
                properties=self.properties,
            )

    def test_vaporizer_rejects_nonfinite_mass_flow(self):
        inlet = self.properties.from_pT(100_000.0, 300.0)
        with self.assertRaises(ValueError):
            Vaporizer(1_000.0, 1.0e6, self.properties).evaluate(
                inlet, inlet.pressure_Pa, float("nan"), 350.0
            )

    def test_vaporizer_rejects_boolean_physical_inputs(self):
        with self.assertRaises(ValueError):
            Vaporizer(True, 1.0e6, self.properties)
        with self.assertRaises(ValueError):
            Vaporizer(1_000.0, False, self.properties)
        inlet = self.properties.from_pT(100_000.0, 300.0)
        vaporizer = Vaporizer(1_000.0, 1.0e6, self.properties)
        for operating_values in (
            (True, 0.1, 350.0),
            (inlet.pressure_Pa, True, 350.0),
            (inlet.pressure_Pa, 0.1, False),
        ):
            with self.assertRaises(ValueError):
                vaporizer.evaluate(inlet, *operating_values)

    def test_pump_first_law_and_entropy(self):
        inlet = self.properties.from_pT(300_000.0, 20.0)
        result = Pump().evaluate(inlet, 0.1, 200_000.0, 0.7)
        self.assertAlmostEqual(
            result.shaft_power_W,
            0.1 * (result.outlet.specific_enthalpy_J_kg - inlet.specific_enthalpy_J_kg),
            delta=1e-7,
        )
        self.assertGreaterEqual(result.outlet.specific_entropy_J_kgK, inlet.specific_entropy_J_kgK - 1e-6)

    def test_pump_recovers_incompressible_hydraulic_power_limit(self):
        inlet = self.properties.from_pT(300_000.0, 20.0)
        result = Pump(self.properties).evaluate(inlet, 0.1, 10_000.0, 1.0)
        self.assertAlmostEqual(
            result.shaft_power_W / result.hydraulic_power_W, 1.0, delta=1.0e-4
        )

    def test_configured_pump_uses_explicit_default_operating_point(self):
        inlet = self.properties.from_pT(300_000.0, 20.0)
        pump = Pump(
            self.properties,
            pressure_rise_Pa=25_000.0,
            isentropic_efficiency=0.8,
        )
        result = pump.evaluate(inlet, 0.1)
        self.assertAlmostEqual(result.pressure_rise_Pa, 25_000.0)
        self.assertAlmostEqual(result.efficiency, 0.8)

    def test_pump_rejects_incomplete_configured_operating_point(self):
        with self.assertRaisesRegex(ValueError, "supplied together"):
            Pump(self.properties, pressure_rise_Pa=25_000.0)

    def test_pump_rejects_nonfinite_evaluation_inputs(self):
        inlet = self.properties.from_pT(300_000.0, 20.0)
        for kwargs in (
            {"mass_flow_kg_s": math.nan},
            {"mass_flow_kg_s": math.inf},
            {"pressure_rise_Pa": math.nan},
            {"isentropic_efficiency": math.nan},
        ):
            with self.assertRaises(ValueError):
                Pump().evaluate(
                    inlet,
                    kwargs.pop("mass_flow_kg_s", 0.1),
                    kwargs.pop("pressure_rise_Pa", 200_000.0),
                    kwargs.pop("isentropic_efficiency", 0.7),
                )

    def test_virtual_pump_quadratic_curve_and_first_law(self):
        inlet = self.properties.from_pT(300_000.0, 20.0)
        outlet_node = self.properties.from_ph(
            525_000.0, inlet.specific_enthalpy_J_kg
        )
        pump = VirtualPump(300_000.0, 0.2, 0.7, self.properties)
        result = pump.evaluate(inlet, outlet_node)
        self.assertAlmostEqual(result.mass_flow_kg_s, 0.1, delta=1.0e-12)
        self.assertAlmostEqual(result.pressure_rise_Pa, 225_000.0)
        self.assertAlmostEqual(
            result.shaft_power_W,
            result.mass_flow_kg_s * (
                result.outlet_specific_enthalpy_J_kg
                - inlet.specific_enthalpy_J_kg
            ),
            delta=1.0e-8,
        )
        self.assertAlmostEqual(result.momentum_residual_Pa, 0.0, delta=1.0e-8)
        self.assertTrue(result.operating_point_physical_valid)
        speed_pump = VirtualPump(
            300_000.0,
            0.2,
            0.7,
            self.properties,
            reference_speed_rpm=10_000.0,
            speed_rpm=10_000.0,
        )
        low_pressure_outlet = self.properties.from_ph(
            inlet.pressure_Pa + 50_000.0,
            inlet.specific_enthalpy_J_kg,
        )
        nominal = speed_pump.evaluate(inlet, low_pressure_outlet)
        speed_pump.set_speed_rpm(5_000.0)
        reduced = speed_pump.evaluate(inlet, low_pressure_outlet)
        self.assertAlmostEqual(speed_pump.speed_ratio, 0.5)
        self.assertLess(reduced.mass_flow_kg_s, nominal.mass_flow_kg_s)
        self.assertAlmostEqual(reduced.momentum_residual_Pa, 0.0, delta=1.0e-8)

    def test_mapped_pump_inverts_explicit_hq_map_and_scales_speed(self):
        inlet = self.properties.from_pT(300_000.0, 20.0)
        outlet_node = self.properties.from_ph(380_000.0, inlet.specific_enthalpy_J_kg)
        pump = MappedPump(
            [(0.0, 200_000.0), (0.01, 150_000.0), (0.02, 0.0)],
            reference_speed_rpm=10_000.0,
            speed_rpm=10_000.0,
            isentropic_efficiency=0.7,
            properties=self.properties,
        )
        result = pump.evaluate(inlet, outlet_node)
        self.assertAlmostEqual(result.mass_flow_kg_s, 0.01 + (150_000.0 - 80_000.0) / 150_000.0 * 0.01, delta=1.0e-12)
        self.assertAlmostEqual(result.map_head_pressure_rise_Pa, 80_000.0, delta=1.0e-8)
        self.assertAlmostEqual(result.momentum_residual_Pa, 0.0, delta=1.0e-8)
        self.assertFalse(result.map_limited)
        self.assertTrue(result.operating_point_physical_valid)
        faster = MappedPump(
            [(0.0, 200_000.0), (0.01, 150_000.0), (0.02, 0.0)],
            reference_speed_rpm=10_000.0,
            speed_rpm=20_000.0,
            isentropic_efficiency=0.7,
            properties=self.properties,
        )
        self.assertAlmostEqual(faster.head_curve_points[1][0], 0.02)
        self.assertAlmostEqual(faster.head_curve_points[1][1], 600_000.0)

    def test_mapped_pump_speed_command_updates_affinity_law_ratio(self):
        pump = MappedPump(
            [(0.0, 320_000.0), (0.01, 160_000.0), (0.02, 0.0)],
            reference_speed_rpm=10_000.0,
            speed_rpm=10_000.0,
            isentropic_efficiency=0.7,
            npsh_required_m=0.5,
            properties=self.properties,
        )

        self.assertAlmostEqual(pump.head_curve_points[0][1], 320_000.0)
        pump.set_speed_rpm(5_000.0)
        self.assertAlmostEqual(pump.speed_ratio, 0.5)
        self.assertAlmostEqual(pump.head_curve_points[0][1], 80_000.0)

        with self.assertRaises(ValueError):
            pump.set_speed_rpm(0.0)

    def test_mapped_pump_clips_outside_map_without_extrapolation(self):
        inlet = self.properties.from_pT(300_000.0, 20.0)
        high_pressure = self.properties.from_ph(600_000.0, inlet.specific_enthalpy_J_kg)
        pump = MappedPump(
            [(0.0, 200_000.0), (0.01, 100_000.0)],
            reference_speed_rpm=10_000.0,
            speed_rpm=10_000.0,
            isentropic_efficiency=0.7,
            properties=self.properties,
        )
        result = pump.evaluate(inlet, high_pressure)
        self.assertEqual(result.mass_flow_kg_s, 0.0)
        self.assertTrue(result.map_limited)
        self.assertFalse(result.operating_point_physical_valid)
        self.assertLess(result.momentum_residual_Pa, 0.0)
        with self.assertRaises(ValueError):
            MappedPump(
                [(0.0, 100.0), (0.01, 120.0)],
                reference_speed_rpm=10_000.0,
                speed_rpm=10_000.0,
                isentropic_efficiency=0.7,
            )

    def test_mapped_pump_near_saturated_quality_tolerance_is_explicit(self):
        liquid = self.properties.saturated_liquid(300_000.0)
        vapor = self.properties.saturated_vapor(300_000.0)
        mixed = self.properties.from_ph(
            300_000.0,
            liquid.specific_enthalpy_J_kg
            + 0.01 * (vapor.specific_enthalpy_J_kg - liquid.specific_enthalpy_J_kg),
        )
        outlet = self.properties.from_ph(380_000.0, mixed.specific_enthalpy_J_kg)
        default_pump = MappedPump(
            [(0.0, 200_000.0), (0.01, 150_000.0), (0.02, 0.0)],
            reference_speed_rpm=10_000.0,
            speed_rpm=10_000.0,
            isentropic_efficiency=0.7,
            properties=self.properties,
        )
        with self.assertRaises(ValueError):
            default_pump.evaluate(mixed, outlet)
        tolerant_pump = MappedPump(
            [(0.0, 200_000.0), (0.01, 150_000.0), (0.02, 0.0)],
            reference_speed_rpm=10_000.0,
            speed_rpm=10_000.0,
            isentropic_efficiency=0.7,
            inlet_quality_tolerance=0.02,
            properties=self.properties,
        )
        result = tolerant_pump.evaluate(mixed, outlet)
        self.assertAlmostEqual(mixed.quality, 0.01, delta=1.0e-10)
        self.assertAlmostEqual(result.inlet_quality, 0.01, delta=1.0e-10)
        self.assertGreater(result.mass_flow_kg_s, 0.0)
        self.assertFalse(result.operating_point_physical_valid)

    def test_virtual_pump_reports_npsh_margin_without_hiding_cavitation_risk(self):
        inlet = self.properties.from_pT(300_000.0, 20.0)
        outlet_node = self.properties.from_ph(
            400_000.0, inlet.specific_enthalpy_J_kg
        )
        unconstrained = VirtualPump(
            300_000.0, 0.2, 0.7, self.properties
        ).evaluate(inlet, outlet_node)
        required = unconstrained.npsh_available_m + 1.0
        checked = VirtualPump(
            300_000.0, 0.2, 0.7, self.properties,
            npsh_required_m=required,
        ).evaluate(inlet, outlet_node)
        self.assertGreater(checked.npsh_available_m, 0.0)
        self.assertAlmostEqual(checked.cavitation_margin_m, -1.0, delta=1.0e-12)
        self.assertFalse(checked.operating_point_physical_valid)

    def test_virtual_pump_shutoff_reports_curve_pressure_closure(self):
        inlet = self.properties.from_pT(300_000.0, 20.0)
        outlet = self.properties.from_ph(
            inlet.pressure_Pa + 300_000.0,
            inlet.specific_enthalpy_J_kg,
        )
        result = VirtualPump(300_000.0, 0.2, 0.7, self.properties).evaluate(inlet, outlet)
        self.assertEqual(result.mass_flow_kg_s, 0.0)
        self.assertAlmostEqual(result.momentum_residual_Pa, 0.0, delta=1.0e-8)

    def test_virtual_pump_rejects_single_phase_gas_with_no_quality_value(self):
        # CoolProp reports quality=None outside the saturation dome, so the
        # liquid-only contract must inspect the explicit phase as well.
        gas = self.properties.from_pT(300_000.0, 100.0)
        outlet = self.properties.from_pT(500_000.0, 100.0)
        pump = VirtualPump(300_000.0, 0.2, 0.7, self.properties)
        with self.assertRaisesRegex(ValueError, "liquid inlet"):
            pump.evaluate(gas, outlet)

    def test_virtual_pump_rejects_nonfinite_constructor_inputs(self):
        with self.assertRaises(ValueError):
            VirtualPump(float("nan"), 0.2, 0.7, self.properties)
        with self.assertRaises(ValueError):
            VirtualPump(300_000.0, float("inf"), 0.7, self.properties)
        with self.assertRaises(ValueError):
            VirtualPump(300_000.0, 0.2, float("nan"), self.properties)
        with self.assertRaises(ValueError):
            VirtualPump(300_000.0, 0.2, 0.7, self.properties, npsh_required_m="bad")
        with self.assertRaises(ValueError):
            VirtualPump(300_000.0, 0.2, 0.7, self.properties, speed_rpm=10_000.0)
        with self.assertRaises(ValueError):
            VirtualPump(
                300_000.0,
                0.2,
                0.7,
                self.properties,
                reference_speed_rpm=10_000.0,
            )

    def test_reliquefier_first_law(self):
        inlet = self.properties.from_pT(300_000.0, 80.0)
        result = Reliquefier(20_000.0).evaluate(inlet, 300_000.0, 0.05)
        self.assertAlmostEqual(
            0.05 * (inlet.specific_enthalpy_J_kg - result.outlet.specific_enthalpy_J_kg),
            result.heat_removed_W,
            delta=1e-7,
        )
        self.assertGreaterEqual(result.liquefied_mass_fraction, 0.0)

    def test_reliquefier_recovers_saturated_lever_rule(self):
        pressure = 300_000.0
        mass_flow = 0.05
        liquid = self.properties.saturated_liquid(pressure)
        vapor = self.properties.saturated_vapor(pressure)
        half_latent_power = 0.5 * mass_flow * (
            vapor.specific_enthalpy_J_kg - liquid.specific_enthalpy_J_kg
        )
        result = Reliquefier(
            half_latent_power, self.properties
        ).evaluate(vapor, pressure, mass_flow, target_quality=0.0)
        self.assertAlmostEqual(result.outlet.quality, 0.5, delta=1.0e-10)
        self.assertAlmostEqual(result.liquefied_mass_fraction, 0.5, delta=1.0e-10)

    def test_reliquefier_rejects_nonfinite_operating_values(self):
        inlet = self.properties.from_pT(300_000.0, 80.0)
        reliquefier = Reliquefier(20_000.0, self.properties)
        with self.assertRaises(ValueError):
            reliquefier.evaluate(inlet, 300_000.0, float("inf"))
        with self.assertRaises(ValueError):
            reliquefier.evaluate(inlet, float("nan"), 0.05)

    def test_vaporizer_link_composes_hydraulic_and_thermal_balances(self):
        left = self.properties.from_pT(600_000.0, 20.0)
        right = self.properties.from_pT(300_000.0, 40.0)
        link = VaporizerLink(
            Valve(2.0e-7, 0.8, self.properties, pressure_samples=32),
            Vaporizer(500.0, 100_000.0, self.properties, integration_segments=64),
            290.0,
        )
        result = link.evaluate(left, right)
        self.assertGreater(result.mass_flow_kg_s, 0.0)
        self.assertGreater(result.thermal_power_W, 0.0)
        self.assertAlmostEqual(
            result.mass_flow_kg_s * (
                result.outlet_specific_enthalpy_J_kg
                - result.hydraulic_outlet_specific_enthalpy_J_kg
            ),
            result.thermal_power_W,
            delta=1.0e-7,
        )

    def test_reliquefier_link_composes_hydraulic_and_thermal_balances(self):
        left = self.properties.from_pT(600_000.0, 80.0)
        right = self.properties.from_pT(300_000.0, 30.0)
        link = ReliquefierLink(
            Valve(2.0e-7, 0.8, self.properties, pressure_samples=32),
            Reliquefier(20_000.0, self.properties),
        )
        result = link.evaluate(left, right)
        self.assertGreater(result.mass_flow_kg_s, 0.0)
        self.assertLess(result.thermal_power_W, 0.0)
        self.assertAlmostEqual(
            result.mass_flow_kg_s * (
                result.outlet_specific_enthalpy_J_kg
                - result.hydraulic_outlet_specific_enthalpy_J_kg
            ),
            result.thermal_power_W,
            delta=1.0e-7,
        )


if __name__ == "__main__":
    unittest.main()
