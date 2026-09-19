"""
Tests for yamatsumi.geometry.

The sheared-lattice round-trip test below is the canonical regression test
for "Bug A": an earlier implementation of cartesian_to_direct applied the
inverse lattice matrix's rows instead of its columns to a Cartesian vector,
which is invisible for symmetric (axis-aligned orthogonal) lattices but
wrong for any sheared/non-orthogonal cell. Orthogonal-lattice tests alone
would NOT catch a regression here -- see also test_poscar.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from yamatsumi.errors import NEBError
from yamatsumi.geometry import cartesian_to_direct, direct_to_cartesian, mic_displacement


def test_round_trip_orthogonal_lattice():
    lattice = np.diag([2.0, 3.0, 4.0])
    frac = np.array([[0.25, 0.5, 0.75], [0.1, 0.9, 0.3]])
    cart = direct_to_cartesian(frac, lattice)
    back = cartesian_to_direct(cart, lattice)
    assert np.allclose(back, frac, atol=1e-12)


def test_round_trip_sheared_lattice_bug_a_regression():
    """Canonical Bug-A regression: a sheared (non-symmetric) lattice matrix."""
    lattice = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    frac = np.array([[0.5, 0.5, 0.0]])

    cart = direct_to_cartesian(frac, lattice)
    assert np.allclose(cart, [[1.0, 0.5, 0.0]], atol=1e-12)

    back = cartesian_to_direct(cart, lattice)
    assert np.allclose(back, frac, atol=1e-12), (
        f"cartesian_to_direct did not invert direct_to_cartesian for a "
        f"sheared lattice: expected {frac.tolist()}, got {back.tolist()}"
    )


def test_round_trip_hexagonal_like_lattice():
    """A second, more physically realistic non-orthogonal (hexagonal-like) cell."""
    a = 3.0
    c = 5.0
    lattice = np.array([
        [a, 0.0, 0.0],
        [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, c],
    ])
    frac = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.5], [0.0, 0.0, 0.0], [0.2, 0.8, 0.9]])
    cart = direct_to_cartesian(frac, lattice)
    back = cartesian_to_direct(cart, lattice)
    assert np.allclose(back, frac, atol=1e-12)


def test_singular_lattice_raises():
    singular = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(NEBError):
        cartesian_to_direct(np.array([[0.1, 0.2, 0.3]]), singular)


@pytest.mark.parametrize(
    "p0,p1,expected",
    [
        ([0.0], [0.6], [-0.4]),
        ([0.0], [-0.6], [0.4]),
        ([0.9], [0.1], [0.2]),
        ([0.1], [0.9], [-0.2]),
    ],
)
def test_mic_displacement_wraps_shortest_path(p0, p1, expected):
    d = mic_displacement(np.array(p0 + [0.0, 0.0]), np.array(p1 + [0.0, 0.0]))
    assert np.allclose(d[0], expected[0], atol=1e-12)


def test_mic_displacement_batch_shape():
    p0 = np.zeros((4, 3))
    p1 = np.full((4, 3), 0.6)
    d = mic_displacement(p0, p1)
    assert d.shape == (4, 3)
    assert np.allclose(d, -0.4, atol=1e-12)
