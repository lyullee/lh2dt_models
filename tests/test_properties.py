import unittest

from lh2dt import HydrogenProperties


class PropertyTests(unittest.TestCase):
    def test_state_energy_identity(self):
        properties = HydrogenProperties()
        state = properties.from_pT(500_000.0, 40.0)
        self.assertAlmostEqual(
            state.specific_enthalpy_J_kg,
            state.specific_internal_energy_J_kg + state.pressure_Pa / state.density_kg_m3,
            delta=1e-5 * abs(state.specific_enthalpy_J_kg),
        )

    def test_saturated_states_are_ordered(self):
        properties = HydrogenProperties()
        liquid = properties.saturated_liquid(300_000.0)
        vapor = properties.saturated_vapor(300_000.0)
        self.assertGreater(liquid.density_kg_m3, vapor.density_kg_m3)
        self.assertLess(liquid.specific_enthalpy_J_kg, vapor.specific_enthalpy_J_kg)
        self.assertEqual(liquid.quality, 0.0)
        self.assertEqual(vapor.quality, 1.0)
        self.assertAlmostEqual(
            properties.saturation_pressure_Pa(liquid.temperature_K),
            300_000.0,
            delta=1.0e-4,
        )

    def test_saturation_endpoint_roundtrip_is_clamped_within_property_tolerance(self):
        properties = HydrogenProperties()
        vapor = properties.saturated_vapor(101_325.0)
        reconstructed = properties.from_ph(
            vapor.pressure_Pa, vapor.specific_enthalpy_J_kg + 1.0e-8
        )
        self.assertEqual(reconstructed.quality, 1.0)

    def test_pressure_density_roundtrip(self):
        properties = HydrogenProperties()
        source = properties.from_pT(500_000.0, 30.0)
        reconstructed = properties.from_prho(source.pressure_Pa, source.density_kg_m3)
        self.assertAlmostEqual(reconstructed.pressure_Pa, source.pressure_Pa, delta=1e-5)
        self.assertAlmostEqual(reconstructed.density_kg_m3, source.density_kg_m3, delta=1e-10)

    def test_specific_volume_derivatives_match_finite_difference(self):
        properties = HydrogenProperties()
        state = properties.from_pT(500_000.0, 40.0)
        vp, vh = properties.specific_volume_derivatives_ph(state)
        dp = 10.0
        dh = 1.0
        volume = lambda p, h: 1.0 / properties.from_ph(p, h).density_kg_m3
        vp_fd = (
            volume(state.pressure_Pa + dp, state.specific_enthalpy_J_kg)
            - volume(state.pressure_Pa - dp, state.specific_enthalpy_J_kg)
        ) / (2.0 * dp)
        vh_fd = (
            volume(state.pressure_Pa, state.specific_enthalpy_J_kg + dh)
            - volume(state.pressure_Pa, state.specific_enthalpy_J_kg - dh)
        ) / (2.0 * dh)
        self.assertAlmostEqual(vp, vp_fd, delta=abs(vp_fd) * 1e-5)
        self.assertAlmostEqual(vh, vh_fd, delta=abs(vh_fd) * 1e-5)

    def test_transport_properties_needed_by_natural_convection_are_positive(self):
        properties = HydrogenProperties()
        state = properties.from_pT(300_000.0, 30.0)
        self.assertGreater(properties.viscosity_Pa_s(state), 0.0)
        self.assertGreater(properties.thermal_conductivity_W_mK(state), 0.0)
        self.assertGreater(properties.isobaric_heat_capacity_J_kgK(state), 0.0)
        self.assertGreater(properties.isochoric_heat_capacity_J_kgK(state), 0.0)
        self.assertGreater(properties.specific_gas_constant_J_kgK(), 0.0)
        self.assertGreater(properties.isobaric_expansion_coefficient_1_K(state), 0.0)

    def test_backend_reference_constants_are_exposed(self):
        properties = HydrogenProperties()
        self.assertAlmostEqual(properties.molar_mass_kg_mol(), 0.00201588, delta=1.0e-10)
        self.assertAlmostEqual(properties.triple_temperature_K(), 13.8033, delta=1.0e-4)
        self.assertAlmostEqual(properties.triple_pressure_Pa(), 7041.0867515, delta=1.0e-3)
        self.assertAlmostEqual(properties.critical_temperature_K(), 32.93785507, delta=1.0e-7)
        self.assertAlmostEqual(properties.critical_pressure_Pa(), 1285776.1785, delta=1.0e-1)

    def test_saturation_enthalpy_pressure_derivative_matches_finite_difference(self):
        properties = HydrogenProperties()
        pressure = 101_325.0
        step = 10.0
        for quality, state_function in (
            (0.0, properties.saturated_liquid),
            (1.0, properties.saturated_vapor),
        ):
            analytic = (
                properties.saturation_enthalpy_pressure_derivative_J_kg_Pa(
                    pressure, quality
                )
            )
            finite_difference = (
                state_function(pressure + step).specific_enthalpy_J_kg
                - state_function(pressure - step).specific_enthalpy_J_kg
            ) / (2.0 * step)
            self.assertAlmostEqual(
                analytic, finite_difference, delta=abs(finite_difference) * 1e-7
            )
        with self.assertRaises(ValueError):
            properties.saturation_enthalpy_pressure_derivative_J_kg_Pa(
                pressure, 0.5
            )

    def test_state_inputs_reject_boolean_and_nonfinite_values(self):
        properties = HydrogenProperties()
        state = properties.from_pT(300_000.0, 20.0)
        for builder, kind in (
            (properties.from_pT, "pT"),
            (properties.from_ph, "ph"),
            (properties.from_ps, "ps"),
            (properties.from_rho_u, "rho_u"),
            (properties.from_pu, "pu"),
        ):
            with self.assertRaises(ValueError):
                if kind == "rho_u":
                    builder(state.density_kg_m3, True)
                else:
                    builder(300_000.0, True)
        with self.assertRaises(ValueError):
            properties.from_ph(300_000.0, float("nan"))
        with self.assertRaises(ValueError):
            properties.from_ps(300_000.0, float("inf"))
        with self.assertRaises(ValueError):
            properties.saturation_enthalpy_pressure_derivative_J_kg_Pa(
                300_000.0, True
            )


if __name__ == "__main__":
    unittest.main()
