"""
Tests for yamatsumi.neb_forces.

The hand-checked toy example below was computed independently (by hand,
not by running the module under test) so the expected numbers are a real
oracle, not a tautology:

  3 images, 1 atom, Cartesian (0,0,0) / (1,0,0) / (1,2,0), energies
  (-1.0, 0.0, -0.5) -- a genuine local energy maximum at the middle image,
  forcing the weighted (non-monotonic) tangent branch -- F_true=(-0.2,0.3,0)
  at the middle image, spring_const=2.0.
"""
from __future__ import annotations

import numpy as np
import pytest

from yamatsumi.errors import NEBError
from yamatsumi.neb_forces import compute_neb_forces
from yamatsumi.poscar import Poscar


def _poscar(position, lattice=None):
    lattice = np.eye(3) if lattice is None else lattice
    return Poscar(
        comment="t", scale=1.0, lattice=lattice, species=["H"], counts=[1],
        coord_type="Direct", positions=np.array([position], dtype=float),
    )


def _three_image_case():
    images = [_poscar([0.0, 0.0, 0.0]), _poscar([1.0, 0.0, 0.0]), _poscar([1.0, 2.0, 0.0])]
    true_forces = [
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[-0.2, 0.3, 0.0]]),
        np.array([[0.0, 0.0, 0.0]]),
    ]
    energies = [-1.0, 0.0, -0.5]
    return images, true_forces, energies


def test_weighted_tangent_non_ci():
    images, true_forces, energies = _three_image_case()
    neb_forces, tangents = compute_neb_forces(images, true_forces, energies, spring_const=2.0, ci_index=-1)

    tangent = tangents[0]
    assert np.allclose(tangent, [0.242535625036333, 0.970142500145332, 0.0], atol=1e-9)

    f_eff = neb_forces[0].ravel()
    assert np.allclose(f_eff, [0.226247720660901, 2.004990882643605, 0.0], atol=1e-9)


def test_weighted_tangent_ci():
    images, true_forces, energies = _three_image_case()
    neb_forces, tangents = compute_neb_forces(images, true_forces, energies, spring_const=2.0, ci_index=1)

    f_eff = neb_forces[0].ravel()
    assert np.allclose(f_eff, [-0.317647058823529, -0.170588235294118, 0.0], atol=1e-9)


def test_monotonic_increasing_tangent_is_forward_difference():
    images = [_poscar([0.0, 0.0, 0.0]), _poscar([1.0, 0.0, 0.0]), _poscar([2.0, 0.0, 0.0])]
    true_forces = [np.zeros((1, 3)) for _ in range(3)]
    energies = [-1.0, 0.0, 1.0]  # strictly increasing
    _, tangents = compute_neb_forces(images, true_forces, energies, spring_const=1.0, ci_index=-1)
    assert np.allclose(tangents[0], [1.0, 0.0, 0.0], atol=1e-12)


def test_monotonic_decreasing_tangent_is_backward_difference():
    images = [_poscar([0.0, 0.0, 0.0]), _poscar([1.0, 0.0, 0.0]), _poscar([2.0, 0.0, 0.0])]
    true_forces = [np.zeros((1, 3)) for _ in range(3)]
    energies = [1.0, 0.0, -1.0]  # strictly decreasing
    _, tangents = compute_neb_forces(images, true_forces, energies, spring_const=1.0, ci_index=-1)
    assert np.allclose(tangents[0], [1.0, 0.0, 0.0], atol=1e-12)


def test_length_mismatch_raises():
    images, true_forces, energies = _three_image_case()
    with pytest.raises(NEBError):
        compute_neb_forces(images, true_forces[:-1], energies, spring_const=2.0, ci_index=-1)
    with pytest.raises(NEBError):
        compute_neb_forces(images, true_forces, energies[:-1], spring_const=2.0, ci_index=-1)


def test_too_few_images_raises():
    images = [_poscar([0.0, 0.0, 0.0]), _poscar([1.0, 0.0, 0.0])]
    true_forces = [np.zeros((1, 3)), np.zeros((1, 3))]
    with pytest.raises(NEBError):
        compute_neb_forces(images, true_forces, [0.0, 0.0], spring_const=1.0, ci_index=-1)


def test_selective_dynamics_masks_force_upstream():
    """Bug C: a force on a fixed atom must never reach the NEB force output."""
    images, true_forces, energies = _three_image_case()
    images[1] = Poscar(
        comment="t", scale=1.0, lattice=np.eye(3), species=["H"], counts=[1],
        coord_type="Direct", positions=images[1].positions,
        selective_dynamics=np.array([[False, False, False]]),
    )
    neb_forces, _ = compute_neb_forces(images, true_forces, energies, spring_const=2.0, ci_index=-1)
    assert np.allclose(neb_forces[0], 0.0)


def test_numba_consistency_if_available():
    from yamatsumi.numba_shims import NUMBA_AVAILABLE
    from yamatsumi.neb_forces import _tangent_kernel, _neb_force_kernel

    if not NUMBA_AVAILABLE:
        pytest.skip("numba not installed")

    dR_plus  = np.array([0.0, 2.0, 0.0])
    dR_minus = np.array([1.0, 0.0, 0.0])
    jit_t   = _tangent_kernel(dR_plus, dR_minus, -1.0, 0.0, -0.5)
    plain_t = _tangent_kernel.py_func(dR_plus, dR_minus, -1.0, 0.0, -0.5)
    assert np.allclose(jit_t, plain_t, atol=1e-12)

    F_true = np.array([-0.2, 0.3, 0.0])
    jit_f   = _neb_force_kernel(F_true, jit_t, dR_plus, dR_minus, 2.0, False)
    plain_f = _neb_force_kernel.py_func(F_true, plain_t, dR_plus, dR_minus, 2.0, False)
    assert np.allclose(jit_f, plain_f, atol=1e-12)


# ---------------------------------------------------------------------------
# Degenerate energies: the weighted tangent must not collapse to zero
# ---------------------------------------------------------------------------

_DRP = np.array([0.30, 0.02, 0.0])   # R_{i+1} - R_i
_DRM = np.array([0.08, -0.01, 0.0])  # R_i - R_{i-1}


def test_tangent_falls_back_to_bisection_when_three_energies_are_equal():
    """
    With e_prev = e_i = e_next both energy weights are zero, and the
    Henkelman-Jonsson weighted tangent was the zero vector. The bisection
    tangent of the same paper (JCP 113, 9901 (2000)) is used instead.
    """
    from yamatsumi.neb_forces import _tangent_kernel
    t = _tangent_kernel(_DRP, _DRM, -10.0, -10.0, -10.0)
    expected = _DRM / np.linalg.norm(_DRM) + _DRP / np.linalg.norm(_DRP)
    assert np.allclose(t, expected / np.linalg.norm(expected), atol=1e-14)


def test_spring_force_survives_equal_energies():
    """|dR+| - |dR-| = 0.22 Ang must still produce a spring force."""
    from yamatsumi.neb_forces import _neb_force_kernel, _tangent_kernel
    f_true = np.array([0.4, -0.2, 0.0])
    t = _tangent_kernel(_DRP, _DRM, -10.0, -10.0, -10.0)
    f_neb = _neb_force_kernel(f_true, t, _DRP, _DRM, 5.0, False)
    spring = 5.0 * (np.linalg.norm(_DRP) - np.linalg.norm(_DRM))
    assert np.dot(f_neb, t) == pytest.approx(spring, rel=1e-12)
