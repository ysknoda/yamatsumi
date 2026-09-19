"""
Coordinate transforms and minimum-image-convention (MIC) helpers.

Lattice convention: lattice rows are the lattice vectors, and fractional
(Direct) coordinates combine with them as a row vector:

    cart = frac @ lattice          (cart_k = sum_j frac_j * lattice[j, k])

so the inverse direction is:

    frac = cart @ inv(lattice)     (frac_k = sum_j cart_j * inv(lattice)[j, k])

Bug-fix note: an earlier pure-Python implementation of the inverse transform
applied the inverse matrix's *rows* to the Cartesian vector instead of its
*columns* -- equivalent to using ``inv(lattice)`` where ``inv(lattice).T`` was
required. That was invisible for symmetric lattice matrices (axis-aligned
cubic/tetragonal/orthorhombic cells) but silently corrupted every hexagonal,
monoclinic, triclinic, or otherwise sheared cell. Expressing the transform as
a plain ``positions @ inv(lattice)`` matrix product (as done here) is
unambiguous and correct for any lattice shape.
"""
from __future__ import annotations

import numpy as np

from .errors import NEBError


def direct_to_cartesian(positions: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert fractional (Direct) coordinates to Cartesian (Ang)."""
    return np.asarray(positions, dtype=float) @ np.asarray(lattice, dtype=float)


def cartesian_to_direct(positions: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert Cartesian (Ang) coordinates to fractional (Direct)."""
    lattice = np.asarray(lattice, dtype=float)
    try:
        inv = np.linalg.inv(lattice)
    except np.linalg.LinAlgError as exc:
        raise NEBError("Singular lattice matrix - cannot invert.") from exc
    return np.asarray(positions, dtype=float) @ inv


def mic_displacement(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """
    Minimum-image-convention (MIC) displacement p1 - p0 in fractional coords.
    Each component is wrapped into [-0.5, 0.5) so the shortest periodic path
    is always chosen. Works element-wise on single 3-vectors or (N, 3) arrays.
    """
    d = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    return d - np.round(d)
