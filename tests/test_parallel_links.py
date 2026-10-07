import unittest

from lh2dt import (
    HydrogenProperties,
    ParallelThermalLink,
    Pipe,
    Reliquefier,
    ReliquefierLink,
    ThermalLinkResult,
    Valve,
    Vaporizer,
    VaporizerLink,
)


class _StubLink:
    def __init__(self, flow: float, hydraulic_h: float, outlet_h: float, thermal_power: float, side: str):
        self.result = ThermalLinkResult(flow, outlet_h, thermal_power, hydraulic_h, side)

    def evaluate(self, _left, _right):
        return self.result


class _NonFiniteLink:
    def evaluate(self, _left, _right):
        return ThermalLinkResult(float("nan"), 0.0, 0.0, 0.0, None)


class ParallelThermalLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.properties = HydrogenProperties()

    def test_four_independent_reliquefiers_sum_flow_and_heat(self):
        left = self.properties.from_pT(600_000.0, 80.0)
        right = self.properties.from_pT(300_000.0, 30.0)
        units = tuple(
            ReliquefierLink(
                Valve(2.0e-7, 0.8, self.properties, pressure_samples=32),
                Reliquefier(5_000.0, self.properties),
            )
            for _ in range(4)
        )
        bank = ParallelThermalLink(units, unit_ids=("L-1101A", "L-1101B", "L-1101C", "L-1101D"))
        result = bank.evaluate(left, right)
        self.assertEqual(result.active_unit_ids, ("L-1101A", "L-1101B", "L-1101C", "L-1101D"))
        self.assertEqual(len(result.unit_results), 4)
        self.assertAlmostEqual(result.mass_flow_kg_s, sum(item.mass_flow_kg_s for item in result.unit_results))
        self.assertAlmostEqual(result.thermal_power_W, sum(item.thermal_power_W for item in result.unit_results))
        self.assertAlmostEqual(
            result.mass_flow_kg_s * (
                result.outlet_specific_enthalpy_J_kg - result.hydraulic_outlet_specific_enthalpy_J_kg
            ),
            result.thermal_power_W,
            delta=1.0e-7,
        )

        pipe_source = self.properties.from_pT(600_000.0, 20.0)
        pipe_sink = self.properties.from_ph(500_000.0, pipe_source.specific_enthalpy_J_kg)
        pipe_link = VaporizerLink(
            Pipe(10.0, 0.02, 1.0e-6, local_loss_coefficient=2.0, properties=self.properties),
            Vaporizer(100.0, 10_000.0, self.properties),
            ambient_temperature_K=300.0,
            properties=self.properties,
        )
        pipe_bank = ParallelThermalLink((pipe_link, pipe_link), unit_ids=("P-A", "P-B"))
        pipe_bank_result = pipe_bank.evaluate(pipe_source, pipe_sink)
        self.assertAlmostEqual(pipe_bank_result.momentum_residual_Pa, 0.0, delta=1.0e-8)
        self.assertEqual(len(pipe_bank_result.unit_momentum_residuals_Pa), 2)
        for residual in pipe_bank_result.unit_momentum_residuals_Pa:
            self.assertAlmostEqual(residual, 0.0, delta=1.0e-8)

    def test_explicit_unit_deactivation_changes_only_bank_sum(self):
        left = self.properties.from_pT(600_000.0, 80.0)
        right = self.properties.from_pT(300_000.0, 30.0)
        unit = ReliquefierLink(
            Valve(2.0e-7, 0.8, self.properties, pressure_samples=32),
            Reliquefier(5_000.0, self.properties),
        )
        bank = ParallelThermalLink((unit, unit, unit, unit), unit_ids=("A", "B", "C", "D"))
        all_units = bank.evaluate(left, right)
        bank.set_active("D", False)
        three_units = bank.evaluate(left, right)
        self.assertEqual(three_units.active_unit_ids, ("A", "B", "C"))
        self.assertAlmostEqual(three_units.mass_flow_kg_s, all_units.mass_flow_kg_s * 0.75, delta=1.0e-12)
        self.assertAlmostEqual(three_units.thermal_power_W, all_units.thermal_power_W * 0.75, delta=1.0e-7)

    def test_parallel_link_requires_explicit_boolean_activation(self):
        link = _StubLink(1.0, 10.0, 12.0, 2.0, "left")
        with self.assertRaisesRegex(ValueError, "active entries must be bool"):
            ParallelThermalLink((link,), active=("false",))
        bank = ParallelThermalLink((link,))
        self.assertEqual(bank.active_mask, (True,))
        with self.assertRaisesRegex(ValueError, "enabled must be bool"):
            bank.set_active("unit-1", "false")

    def test_opposing_parallel_flows_are_rejected(self):
        bank = ParallelThermalLink(
            (
                _StubLink(1.0, 10.0, 12.0, 2.0, "left"),
                _StubLink(-1.0, 11.0, 13.0, 3.0, "right"),
            ),
            unit_ids=("forward", "reverse"),
        )
        state = self.properties.from_pT(300_000.0, 20.0)
        with self.assertRaises(ValueError):
            bank.evaluate(state, state)

    def test_parallel_link_validates_each_replaceable_unit_result(self):
        bank = ParallelThermalLink((_NonFiniteLink(),), unit_ids=("invalid",))
        state = self.properties.from_pT(300_000.0, 20.0)
        with self.assertRaisesRegex(
            ValueError, "parallel unit invalid mass_flow_kg_s must be finite"
        ):
            bank.evaluate(state, state)


if __name__ == "__main__":
    unittest.main()
