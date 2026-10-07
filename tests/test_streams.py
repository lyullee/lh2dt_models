import unittest

from lh2dt import MassEnergyFlow, mix_streams, split_stream


class StreamTests(unittest.TestCase):
    def test_mix_conserves_mass_and_enthalpy_flow(self):
        streams = [MassEnergyFlow(2.0, 100.0), MassEnergyFlow(1.0, 400.0)]
        mixed = mix_streams(streams)
        self.assertEqual(mixed.mass_flow_kg_s, 3.0)
        self.assertEqual(mixed.specific_enthalpy_J_kg, 200.0)
        self.assertEqual(mixed.enthalpy_flow_W, sum(x.enthalpy_flow_W for x in streams))

    def test_split_conserves_mass_and_energy(self):
        inlet = MassEnergyFlow(4.0, 123.0)
        outlets = split_stream(inlet, [0.25, 0.75])
        self.assertAlmostEqual(sum(x.mass_flow_kg_s for x in outlets), inlet.mass_flow_kg_s)
        self.assertAlmostEqual(sum(x.enthalpy_flow_W for x in outlets), inlet.enthalpy_flow_W)

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):
            split_stream(MassEnergyFlow(1.0, 1.0), [0.2, 0.2])


if __name__ == "__main__":
    unittest.main()

