import pytest

from lh2dt import LocalizedHeatPath, VacuumJacketedPipeEnvelope


def test_envelope_exposes_explicit_fluid_wall_and_vacuum_geometry():
    envelope = VacuumJacketedPipeEnvelope(
        length_m=10.0,
        process_inner_diameter_m=0.0525,
        process_outer_diameter_m=0.0603,
        jacket_inner_diameter_m=0.1082,
    )

    assert envelope.fluid_volume_m3 == pytest.approx(
        3.141592653589793 * 0.0525**2 * 10.0 / 4.0
    )
    assert envelope.process_wall_volume_m3 > 0.0
    assert envelope.jacket_inner_surface_area_m2 > envelope.process_inner_surface_area_m2
    assert envelope.vacuum_annulus_area_m2 > 0.0
    assert envelope.radial_vacuum_gap_m == pytest.approx((0.1082 - 0.0603) / 2.0)


def test_envelope_builds_path_decomposed_insulation_without_fitting():
    envelope = VacuumJacketedPipeEnvelope(10.0, 0.0525, 0.0603, 0.1082)
    insulation = envelope.make_insulation(
        effective_emissivity=0.02,
        mli_solid_conductance_W_K=0.01,
        support_conductance_W_K=0.02,
        residual_gas_coefficient_W_m2K_Pa=0.001,
        maximum_valid_vacuum_pressure_Pa=5.0,
        localized_heat_paths=(LocalizedHeatPath("support-1", 0.003),),
    )
    heat = insulation.evaluate(300.0, 25.0, 0.1)

    assert insulation.parameters.area_m2 == pytest.approx(
        envelope.jacket_inner_surface_area_m2
    )
    assert heat.total_heat_into_cold_W > heat.radiation_heat_W
    assert heat.localized_heat_W == pytest.approx(0.003 * 275.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length_m": 0.0, "process_inner_diameter_m": 0.05, "process_outer_diameter_m": 0.06, "jacket_inner_diameter_m": 0.1},
        {"length_m": 1.0, "process_inner_diameter_m": 0.06, "process_outer_diameter_m": 0.05, "jacket_inner_diameter_m": 0.1},
        {"length_m": 1.0, "process_inner_diameter_m": 0.05, "process_outer_diameter_m": 0.06, "jacket_inner_diameter_m": 0.06},
    ],
)
def test_envelope_rejects_ambiguous_or_nonphysical_diameters(kwargs):
    with pytest.raises(ValueError):
        VacuumJacketedPipeEnvelope(**kwargs)


def test_dynamic_pipe_geometry_does_not_hide_thermal_parameters():
    envelope = VacuumJacketedPipeEnvelope(2.0, 0.02, 0.025, 0.05)
    assert envelope.dynamic_pipe_geometry() == {
        "length_m": 2.0,
        "inner_diameter_m": 0.02,
        "fluid_volume_m3": pytest.approx(3.141592653589793 * 0.02**2 * 2.0 / 4.0),
    }
