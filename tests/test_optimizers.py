"""Tests for yamatsumi.optimizers."""
from __future__ import annotations

import numpy as np
import pytest

from yamatsumi.optimizers import (
    OptimizerState,
    step_fire,
    step_quickmin,
    step_sd,
)
from yamatsumi.poscar import Poscar


def _poscar(positions, selective_dynamics=None):
    return Poscar(
        comment="t", scale=1.0, lattice=np.eye(3), species=["H"], counts=[len(positions)],
        coord_type="Direct", positions=np.array(positions, dtype=float),
        selective_dynamics=selective_dynamics,
    )


def test_step_sd_direction_and_magnitude():
    p = _poscar([[0.0, 0.0, 0.0]])
    forces = np.array([[1.0, 0.0, 0.0]])
    updated = step_sd(p, forces, step_size=0.1)
    assert np.allclose(updated.positions, [[0.1, 0.0, 0.0]], atol=1e-12)


def test_step_sd_no_op_below_force_threshold():
    p = _poscar([[0.25, 0.25, 0.25]])
    forces = np.array([[1e-12, 0.0, 0.0]])
    updated = step_sd(p, forces, step_size=0.1)
    assert updated is p


def test_step_quickmin_aligned_velocity_accelerates_then_caps():
    p = _poscar([[0.0, 0.0, 0.0]])
    state = OptimizerState(n_atoms=1)
    state.velocity = np.array([[1.0, 0.0, 0.0]])
    forces = np.array([[1.0, 0.0, 0.0]])
    updated = step_quickmin(p, forces, state, step_size=0.1)
    # P=1>0, F_norm2=1 -> v=(1,0,0); dt=0.1; v=(1.1,0,0); v_max=1.1>0.1(step_size)
    # -> capped so |v|==step_size==0.1 -> v=(0.1,0,0); new_cart = 0 + v (v is
    # already the capped per-step displacement -- see the bug-fix comment in
    # _quickmin_kernel; it must NOT be scaled by dt a second time).
    expected_x = 0.1
    assert np.allclose(updated.positions, [[expected_x, 0.0, 0.0]], atol=1e-9)
    assert np.allclose(state.velocity, [[0.1, 0.0, 0.0]], atol=1e-9)


def test_step_quickmin_opposing_velocity_resets_to_zero_before_euler():
    p = _poscar([[0.0, 0.0, 0.0]])
    state = OptimizerState(n_atoms=1)
    state.velocity = np.array([[-1.0, 0.0, 0.0]])
    forces = np.array([[1.0, 0.0, 0.0]])
    updated = step_quickmin(p, forces, state, step_size=0.1)
    # P=-1<=0 -> v=0; dt=0.1; v=(0.1,0,0); v_max=0.1, not > step_size -> no cap
    # -> new_cart = 0 + v = (0.1,0,0).
    assert np.allclose(updated.positions, [[0.1, 0.0, 0.0]], atol=1e-12)
    assert np.allclose(state.velocity, [[0.1, 0.0, 0.0]], atol=1e-12)


def test_step_quickmin_displacement_never_exceeds_step_size_for_small_force():
    """
    Regression for the dt-double-application bug: previously, new_cart_flat
    was computed as cart_flat + v*dt where v was already the step_size-capped
    displacement, applying the (unbounded, as max_f -> 0) step_size/max_f
    factor twice. A small residual force (well above the 1e-10 no-op guard,
    but far below a typical force_tol) must still produce a displacement no
    larger than step_size -- concretely, force=1e-4 eV/Ang with step_size=0.05
    Ang previously produced a 25 Ang displacement (500x the cap).
    """
    p = _poscar([[0.5, 0.5, 0.5]])
    state = OptimizerState(n_atoms=1)
    forces = np.array([[1e-4, 0.0, 0.0]])
    step_size = 0.05
    updated = step_quickmin(p, forces, state, step_size=step_size)
    disp = np.linalg.norm(updated.positions - p.positions)  # lattice=eye(3) here
    assert disp <= step_size + 1e-9
    assert disp == pytest.approx(step_size, abs=1e-9)


def test_step_fire_d4_caps_velocity_before_euler_step():
    """
    D-4 regression: the velocity cap must be applied BEFORE the Euler
    displacement, so the capped (not the raw) velocity determines the
    position update. With dt starting at 0.1, zero initial velocity, and a
    large force, an uncapped-order bug would move the atom 10x further
    than the correct capped-before behaviour.
    """
    p = _poscar([[0.0, 0.0, 0.0]])
    state = OptimizerState(n_atoms=1, dt0=0.1, alpha0=0.1, f_dec=0.5)
    forces = np.array([[10.0, 0.0, 0.0]])
    updated = step_fire(p, forces, state, step_size=0.05, is_ci=False, tangent=None)

    # P = F.v = 0 (v starts at 0) -> not > 0 -> v=0, dt *= f_dec -> 0.05
    # v = 0 + F*dt = (10,0,0)*0.05 = (0.5,0,0); v_max=0.5 > step_size(0.05)
    # -> capped v = (0.05,0,0); new_cart = 0 + v*dt = 0.05*0.05 = 0.0025
    assert updated.positions[0, 0] == pytest.approx(0.0025, abs=1e-12)
    assert state.dt == pytest.approx(0.05, abs=1e-12)
    assert np.allclose(state.velocity, [[0.05, 0.0, 0.0]], atol=1e-12)


def test_step_fire_ci_uses_tangent_split():
    p = _poscar([[0.0, 0.0, 0.0]])
    state = OptimizerState(n_atoms=1, dt0=0.1)
    forces = np.array([[1.0, 1.0, 0.0]])
    tangent = np.array([1.0, 0.0, 0.0])
    updated = step_fire(p, forces, state, step_size=0.5, is_ci=True, tangent=tangent)
    # Should not raise, and should produce a finite, distinct-from-input position.
    assert updated.positions.shape == (1, 3)
    assert not np.allclose(updated.positions, p.positions)


@pytest.mark.parametrize("stepper_name", ["sd", "quickmin", "fire"])
def test_selective_dynamics_freezes_fixed_atom(stepper_name):
    """Bug C: a fixed atom's position must be bitwise unchanged after any stepper."""
    sd_mask = np.array([[True, True, True], [False, False, False]])
    p = _poscar([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], selective_dynamics=sd_mask)
    forces = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    if stepper_name == "sd":
        updated = step_sd(p, forces, step_size=0.1)
    elif stepper_name == "quickmin":
        state = OptimizerState(n_atoms=2)
        updated = step_quickmin(p, forces, state, step_size=0.1)
    else:
        state = OptimizerState(n_atoms=2, dt0=0.1)
        updated = step_fire(p, forces, state, step_size=0.1, is_ci=False, tangent=None)

    assert np.array_equal(updated.positions[1], p.positions[1]), (
        f"{stepper_name}: fixed atom moved from {p.positions[1]} to {updated.positions[1]}"
    )
    assert not np.array_equal(updated.positions[0], p.positions[0]), (
        f"{stepper_name}: free atom did not move at all"
    )


def test_steppers_share_lattice_and_selective_dynamics_by_reference():
    """2a: steppers must return via with_positions(), not a full deep copy --
    lattice/selective_dynamics on the output should be the same object."""
    sd_mask = np.array([[True, True, True]])
    p = _poscar([[0.0, 0.0, 0.0]], selective_dynamics=sd_mask)
    forces = np.array([[1.0, 0.0, 0.0]])

    updated_sd = step_sd(p, forces, step_size=0.1)
    assert updated_sd.lattice is p.lattice
    assert updated_sd.selective_dynamics is p.selective_dynamics

    state_qm = OptimizerState(n_atoms=1)
    updated_qm = step_quickmin(p, forces, state_qm, step_size=0.1)
    assert updated_qm.lattice is p.lattice
    assert updated_qm.selective_dynamics is p.selective_dynamics

    state_fire = OptimizerState(n_atoms=1, dt0=0.1)
    updated_fire = step_fire(p, forces, state_fire, step_size=0.1, is_ci=False, tangent=None)
    assert updated_fire.lattice is p.lattice
    assert updated_fire.selective_dynamics is p.selective_dynamics


def test_numba_consistency_if_available():
    from yamatsumi.numba_shims import NUMBA_AVAILABLE
    from yamatsumi.optimizers import _sd_kernel, _quickmin_kernel, _max_atom_norm

    if not NUMBA_AVAILABLE:
        pytest.skip("numba not installed")

    cart = np.array([0.0, 0.0, 0.0])
    forces = np.array([1.0, 2.0, 3.0])
    max_f = _max_atom_norm(forces, 1)
    jit_out   = _sd_kernel(cart, forces, 0.1, max_f)
    plain_out = _sd_kernel.py_func(cart, forces, 0.1, max_f)
    assert np.allclose(jit_out, plain_out, atol=1e-12)

    velocity = np.array([0.1, 0.1, 0.1])
    jit_qm   = _quickmin_kernel(cart, forces, velocity, 0.1, max_f, 1)
    plain_qm = _quickmin_kernel.py_func(cart, forces, velocity, 0.1, max_f, 1)
    assert np.allclose(jit_qm[0], plain_qm[0], atol=1e-12)
    assert np.allclose(jit_qm[1], plain_qm[1], atol=1e-12)
