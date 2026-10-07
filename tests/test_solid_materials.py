import math

import pytest

from lh2dt import (
    HorizontalVesselGeometry,
    horizontal_vessel_316_wall_axial_conductance_W_K,
    horizontal_vessel_316_wall_inventory,
    nist_316_specific_heat_J_kgK,
    nist_316_thermal_conductivity_W_mK,
)


def test_nist_316_specific_heat_matches_published_polynomial_values():
    assert nist_316_specific_heat_J_kgK(20.0) == pytest.approx(13.6072593, rel=1e-7)
    assert nist_316_specific_heat_J_kgK(30.0) == pytest.approx(29.2192132, rel=1e-7)
    assert nist_316_specific_heat_J_kgK(100.0) == pytest.approx(273.0234813, rel=1e-7)


def test_nist_316_specific_heat_rejects_extrapolation():
    with pytest.raises(ValueError, match="4--300 K"):
        nist_316_specific_heat_J_kgK(3.9)
    with pytest.raises(ValueError, match="4--300 K"):
        nist_316_specific_heat_J_kgK(301.0)


def test_nist_316_thermal_conductivity_matches_published_polynomial_values():
    assert nist_316_thermal_conductivity_W_mK(20.0) == pytest.approx(2.1686224, rel=1e-7)
    assert nist_316_thermal_conductivity_W_mK(100.0) == pytest.approx(9.2235902, rel=1e-7)
    assert nist_316_thermal_conductivity_W_mK(293.0) == pytest.approx(15.1233439, rel=1e-7)


def test_nist_316_thermal_conductivity_rejects_extrapolation():
    with pytest.raises(ValueError, match="4--300 K"):
        nist_316_thermal_conductivity_W_mK(3.9)
    with pytest.raises(ValueError, match="4--300 K"):
        nist_316_thermal_conductivity_W_mK(301.0)


def test_horizontal_vessel_wall_inventory_uses_separate_shell_and_head_thicknesses():
    geometry = HorizontalVesselGeometry(2.0, 4.0, 0.5, quadrature_order=48)
    inventory = horizontal_vessel_316_wall_inventory(
        geometry,
        shell_thickness_m=0.013,
        head_thickness_m=0.0124,
        evaluation_temperature_K=25.0,
    )
    expected_cylinder_area = math.pi * 2.0 * 4.0
    expected_metal_volume = (
        expected_cylinder_area * 0.013
        + (geometry.inner_surface_area_m2 - expected_cylinder_area) * 0.0124
    )

    assert inventory.cylinder_area_m2 == pytest.approx(expected_cylinder_area)
    assert inventory.metal_volume_m3 == pytest.approx(expected_metal_volume)
    assert inventory.metal_mass_kg == pytest.approx(7_950.0 * expected_metal_volume)
    assert inventory.heat_capacity_J_K == pytest.approx(
        inventory.metal_mass_kg * inventory.specific_heat_J_kgK
    )


def test_wall_axial_conductance_is_geometry_and_material_based():
    geometry = HorizontalVesselGeometry(2.0, 4.0, 0.5, quadrature_order=48)
    value = horizontal_vessel_316_wall_axial_conductance_W_K(
        geometry,
        shell_thickness_m=0.013,
        evaluation_temperature_K=25.0,
        wall_node_count=6,
    )
    expected = (
        nist_316_thermal_conductivity_W_mK(25.0)
        * math.pi * 2.0 * 0.013
        / (4.0 / 5.0)
    )
    assert value == pytest.approx(expected)
