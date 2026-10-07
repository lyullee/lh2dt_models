import math

import pytest

from lh2dt import (
    OrthoParaKineticsParameters,
    OrthoParaState,
    advance_ortho_para_state,
    conversion_heat_rate_W,
    equilibrium_para_fraction,
    para_fraction_rate_s,
)


def test_equilibrium_fraction_is_physically_bounded_and_cold_rich_in_para():
    cold = equilibrium_para_fraction(20.0)
    warm = equilibrium_para_fraction(300.0)
    assert 0.0 < warm < 1.0
    assert cold > warm


def test_zero_rate_keeps_composition_and_energy_boundary_zero():
    parameters = OrthoParaKineticsParameters()
    state = OrthoParaState(0.25)
    advanced = advance_ortho_para_state(state, 25.0, 86_400.0, parameters)
    assert advanced == state
    assert conversion_heat_rate_W(state, 25.0, 100.0, parameters) == 0.0


def test_exact_relaxation_moves_toward_equilibrium_without_overshoot():
    parameters = OrthoParaKineticsParameters(
        conversion_rate_constant_s=2.0e-4,
        conversion_enthalpy_J_kg=700_000.0,
    )
    state = OrthoParaState(0.25)
    target = equilibrium_para_fraction(20.0)
    advanced = advance_ortho_para_state(state, 20.0, 1_000.0, parameters)
    assert state.para_fraction < advanced.para_fraction < target
    assert para_fraction_rate_s(state, 20.0, parameters) > 0.0
    assert conversion_heat_rate_W(state, 20.0, 4.0, parameters) > 0.0


def test_invalid_composition_and_mass_are_rejected():
    with pytest.raises(ValueError):
        OrthoParaState(1.1)
    with pytest.raises(ValueError):
        conversion_heat_rate_W(OrthoParaState(0.5), 20.0, -1.0, OrthoParaKineticsParameters())
