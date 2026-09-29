"""Tests for yamatsumi.interpolate."""
from __future__ import annotations

import numpy as np
import pytest

from yamatsumi.errors import NEBError
from yamatsumi.geometry import cartesian_to_direct, direct_to_cartesian
from yamatsumi.interpolate import (
    _idpp_objective_and_gradient,
    _pairwise_cartesian,
    check_endpoint_consistency,
    idpp_interpolate_images,
    interpolate_images,
)
from yamatsumi.poscar import Poscar


def _poscar(positions, lattice=None, species=("H",), counts=(1,), selective_dynamics=None):
    lattice = np.eye(3) * 10.0 if lattice is None else lattice
    return Poscar(
        comment="t", scale=1.0, lattice=lattice, species=list(species), counts=list(counts),
        coord_type="Direct", positions=np.array(positions, dtype=float),
        selective_dynamics=selective_dynamics,
    )


def test_cartesian_coord_type_raises():
    """3e: a Poscar left in Cartesian coordinates must be rejected up front."""
    init  = _poscar([[0.0, 0.0, 0.0]])
    final = _poscar([[0.1, 0.1, 0.1]])
    init.coord_type = "Cartesian"
    with pytest.raises(NEBError):
        check_endpoint_consistency(init, final)


def test_atom_count_mismatch_raises():
    init = _poscar([[0.0, 0.0, 0.0]])
    final = _poscar([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]], counts=(2,))
    with pytest.raises(NEBError):
        check_endpoint_consistency(init, final)


def test_species_mismatch_raises():
    init = _poscar([[0.0, 0.0, 0.0]], species=("H",))
    final = _poscar([[0.0, 0.0, 0.0]], species=("O",))
    with pytest.raises(NEBError):
        check_endpoint_consistency(init, final)


def test_counts_mismatch_raises():
    init = _poscar([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]], species=("H", "O"), counts=(1, 1))
    final = _poscar([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]], species=("H", "O"), counts=(2, 0))
    with pytest.raises(NEBError):
        check_endpoint_consistency(init, final)


def test_lattice_mismatch_raises():
    init = _poscar([[0.0, 0.0, 0.0]], lattice=np.eye(3) * 10.0)
    final = _poscar([[0.0, 0.0, 0.0]], lattice=np.eye(3) * 11.0)
    with pytest.raises(NEBError):
        check_endpoint_consistency(init, final)


def test_lattice_mismatch_from_different_scale_bake_in():
    """
    Interaction with the Bug-B fix: two endpoints that look identical before
    scale is baked in (same raw lattice text) but used a different POSCAR
    scale factor now resolve to genuinely different physical lattices, and
    must be caught here rather than silently passing.
    """
    raw = np.eye(3)
    init  = _poscar([[0.0, 0.0, 0.0]], lattice=raw * 1.0)
    final = _poscar([[0.0, 0.0, 0.0]], lattice=raw * 2.0)  # as if scale=2.0 was baked in
    with pytest.raises(NEBError):
        check_endpoint_consistency(init, final)


def test_selective_dynamics_presence_mismatch_raises():
    init  = _poscar([[0.0, 0.0, 0.0]], selective_dynamics=np.array([[True, True, True]]))
    final = _poscar([[0.0, 0.0, 0.0]], selective_dynamics=None)
    with pytest.raises(NEBError):
        check_endpoint_consistency(init, final)


def test_selective_dynamics_value_mismatch_raises():
    init  = _poscar([[0.0, 0.0, 0.0]], selective_dynamics=np.array([[True, True, True]]))
    final = _poscar([[0.0, 0.0, 0.0]], selective_dynamics=np.array([[False, True, True]]))
    with pytest.raises(NEBError):
        check_endpoint_consistency(init, final)


def test_selective_dynamics_matching_masks_pass():
    mask = np.array([[True, True, False]])
    init  = _poscar([[0.0, 0.0, 0.0]], selective_dynamics=mask)
    final = _poscar([[0.1, 0.1, 0.1]], selective_dynamics=mask.copy())
    check_endpoint_consistency(init, final)  # must not raise


def test_interpolate_basic_count_and_endpoints():
    init  = _poscar([[0.1, 0.1, 0.1]])
    final = _poscar([[0.3, 0.3, 0.3]])
    images = interpolate_images(init, final, num_images=3)
    assert len(images) == 5
    assert np.allclose(images[0].positions, init.positions)
    assert np.allclose(images[-1].positions, final.positions)
    # linear, no MIC wraparound needed: frac = 1/4, 2/4, 3/4 of the way
    assert np.allclose(images[1].positions, [[0.15, 0.15, 0.15]])
    assert np.allclose(images[2].positions, [[0.2, 0.2, 0.2]])
    assert np.allclose(images[3].positions, [[0.25, 0.25, 0.25]])


def test_interpolate_mic_crosses_boundary_the_short_way():
    """0.05 -> 0.95 must interpolate through 0 (short way), not through 0.5."""
    init  = _poscar([[0.05, 0.0, 0.0]])
    final = _poscar([[0.95, 0.0, 0.0]])
    images = interpolate_images(init, final, num_images=1)
    mid_x = images[1].positions[0, 0]
    # short path: 0.05 -> 0.0 -> -0.05 (i.e. 0.95 mod 1); midpoint should be
    # near 0.0 (mod 1), not near 0.5.
    assert (mid_x % 1.0) < 0.1 or (mid_x % 1.0) > 0.9


def test_interpolate_propagates_selective_dynamics_to_intermediates():
    mask = np.array([[False, False, False]])
    init  = _poscar([[0.1, 0.1, 0.1]], selective_dynamics=mask)
    final = _poscar([[0.3, 0.3, 0.3]], selective_dynamics=mask.copy())
    images = interpolate_images(init, final, num_images=2)
    for img in images:
        assert img.selective_dynamics is not None
        assert np.array_equal(img.selective_dynamics, mask)


# ---------------------------------------------------------------------------
# 1c: IDPP interpolation
# ---------------------------------------------------------------------------

def test_idpp_gradient_matches_finite_difference():
    """
    The analytic IDPP gradient must match an independently computed
    finite-difference gradient of the same objective -- this validates the
    hand-derived gradient formula itself, not just self-consistency.
    """
    # Positions chosen so every pairwise fractional separation stays well
    # clear of +-0.5 in each component: mic_displacement's "d - round(d)"
    # wrap has a genuine slope discontinuity exactly at a 0.5 separation, so
    # a pair sitting on that boundary would make finite differences spurious
    # regardless of whether the analytic gradient itself is correct.
    lattice = np.eye(3) * 10.0
    positions = np.array([
        [0.20, 0.30, 0.50],
        [0.45, 0.50, 0.50],
        [0.60, 0.35, 0.50],
    ])
    _, d_current = _pairwise_cartesian(positions, lattice)
    d_target = d_current * 1.2  # arbitrary, distinct from current -> nonzero delta
    n = 3
    off_diag = ~np.eye(n, dtype=bool)
    weights = np.zeros((n, n))
    weights[off_diag] = 1.0 / (d_target[off_diag] ** 4)

    _, grad = _idpp_objective_and_gradient(positions, lattice, d_target, weights)

    numerical_grad = np.zeros_like(grad)
    h = 1e-6
    cart0 = direct_to_cartesian(positions, lattice)
    for i in range(n):
        for c in range(3):
            cart_plus = cart0.copy(); cart_plus[i, c] += h
            obj_plus, _ = _idpp_objective_and_gradient(
                cartesian_to_direct(cart_plus, lattice), lattice, d_target, weights
            )
            cart_minus = cart0.copy(); cart_minus[i, c] -= h
            obj_minus, _ = _idpp_objective_and_gradient(
                cartesian_to_direct(cart_minus, lattice), lattice, d_target, weights
            )
            numerical_grad[i, c] = (obj_plus - obj_minus) / (2 * h)

    assert np.allclose(grad, numerical_grad, atol=1e-4)


def test_idpp_endpoints_match_input_exactly():
    init  = _poscar([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]], species=("H", "H"), counts=(1, 1))
    final = _poscar([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]], species=("H", "H"), counts=(1, 1))
    images = idpp_interpolate_images(init, final, num_images=3)
    assert len(images) == 5
    assert np.allclose(images[0].positions, init.positions)
    assert np.allclose(images[-1].positions, final.positions)


def test_idpp_increases_close_approach_distance_relative_to_linear():
    """
    Two atoms that pass close to each other under plain linear interpolation
    (the pathological case IDPP exists to fix) must end up farther apart at
    the midpoint image after IDPP relaxation.

    Atom A moves 0.3 -> 0.6 and atom B moves 0.7 -> 0.4 (each a 0.3
    fractional displacement, safely under the +-0.5 MIC-wrap boundary, so
    interpolate_images's per-atom MIC shortest path is the direct one).
    Their independent linear paths cross near the middle: at the midpoint
    image A is at 0.45 and B at 0.55, only 0.1 fractional (1.0 Ang) apart,
    even though they start and end 4.0/2.0 Ang apart (MIC-correct pair
    distance) -- exactly the kind of close approach IDPP's pairwise-distance
    target (linearly interpolated between 4.0 and 2.0 Ang) should relax away
    from.
    """
    init  = _poscar([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]], species=("H", "H"), counts=(1, 1))
    final = _poscar([[0.6, 0.5, 0.5], [0.4, 0.5, 0.5]], species=("H", "H"), counts=(1, 1))
    lattice = init.lattice

    linear_images = interpolate_images(init, final, num_images=1)
    idpp_images = idpp_interpolate_images(init, final, num_images=1)

    def midpoint_distance(img):
        return float(_pairwise_cartesian(img.positions, lattice)[1][0, 1])

    linear_dist = midpoint_distance(linear_images[1])
    idpp_dist   = midpoint_distance(idpp_images[1])
    assert linear_dist == pytest.approx(1.0, abs=1e-9)
    assert idpp_dist > linear_dist + 1.0


def test_idpp_rejects_coincident_atoms_instead_of_producing_nan():
    """
    Two atoms at the same position make IDPP's 1/d^4 pair weight infinite,
    the objective's gradient NaN, and every relaxed image's coordinates NaN
    -- which would then be written out to POSCAR files silently, since NaN
    propagates through the steepest-descent loop without raising. A
    duplicated atom in an endpoint must be reported explicitly instead.
    """
    duplicated = [[0.3, 0.5, 0.5], [0.3, 0.5, 0.5]]
    init  = _poscar(duplicated, species=("H", "H"), counts=(1, 1))
    final = _poscar([[0.6, 0.5, 0.5], [0.4, 0.5, 0.5]], species=("H", "H"), counts=(1, 1))

    with pytest.raises(NEBError, match="duplicated atom"):
        idpp_interpolate_images(init, final, num_images=1)


def test_idpp_rejects_coincident_atoms_in_final_endpoint_too():
    init  = _poscar([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]], species=("H", "H"), counts=(1, 1))
    final = _poscar([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], species=("H", "H"), counts=(1, 1))

    with pytest.raises(NEBError, match="duplicated atom"):
        idpp_interpolate_images(init, final, num_images=1)


def test_idpp_single_atom_has_no_pairs_and_still_works():
    """A 1-atom cell has no interatomic pairs at all; the coincident-atom
    check must not misfire on the (empty) off-diagonal."""
    init  = _poscar([[0.3, 0.5, 0.5]])
    final = _poscar([[0.6, 0.5, 0.5]])
    images = idpp_interpolate_images(init, final, num_images=2)
    assert len(images) == 4
    assert np.all(np.isfinite(images[1].positions))


def test_idpp_freezes_fixed_atoms_at_linear_position():
    mask  = np.array([[False, False, False], [True, True, True]])
    init  = _poscar([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]], species=("H", "H"), counts=(1, 1),
                     selective_dynamics=mask)
    final = _poscar([[0.2, 0.5, 0.5], [0.3, 0.5, 0.5]], species=("H", "H"), counts=(1, 1),
                     selective_dynamics=mask.copy())
    linear_images = interpolate_images(init, final, num_images=2)
    idpp_images   = idpp_interpolate_images(init, final, num_images=2)
    for lin_img, idpp_img in zip(linear_images, idpp_images):
        assert np.array_equal(idpp_img.positions[0], lin_img.positions[0])


# ---------------------------------------------------------------------------
# ADR 0001: a VASP-wrapped final endpoint must be unwrapped against init
# ---------------------------------------------------------------------------

def _two_atom(z_second):
    return Poscar(
        comment="e", scale=1.0, lattice=np.eye(3) * 10.0, species=["H"], counts=[2],
        coord_type="Direct", positions=np.array([[0.5, 0.5, 0.4], [0.5, 0.5, z_second]]),
    )


def test_wrapped_final_endpoint_is_unwrapped_against_init():
    """
    Atom 2 moves from z = 0.02 to z = -0.02 across the cell boundary. VASP
    writes the relaxed final endpoint's CONTCAR wrapped into [0, 1), as
    z = 0.98. The intermediate images already used the MIC displacement,
    but the final image was used verbatim, so the last segment contained a
    whole lattice vector (measured max|F_NEB| = 49 eV/Ang on that image).
    """
    images = interpolate_images(_two_atom(0.02), _two_atom(0.98), 3)
    assert images[-1].positions[1, 2] == pytest.approx(-0.02)
    steps = [images[k + 1].positions - images[k].positions for k in range(len(images) - 1)]
    for s in steps:
        assert np.allclose(s, steps[0]), "every segment must be the same MIC step"


def test_unwrapped_final_is_bit_identical_when_nothing_is_wrapped():
    final = _two_atom(0.03)
    images = interpolate_images(_two_atom(0.02), final, 3)
    assert np.array_equal(images[-1].positions, final.positions)


def test_idpp_also_returns_the_unwrapped_final_endpoint():
    images = idpp_interpolate_images(_two_atom(0.02), _two_atom(0.98), 3)
    assert images[-1].positions[1, 2] == pytest.approx(-0.02)
