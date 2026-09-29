"""
Improved-tangent CI-NEB force engine.

Reference: G. Henkelman, B. P. Uberuaga, H. Jonsson, "A climbing image
nudged elastic band method for finding saddle points and minimum energy
paths", J. Chem. Phys. 113, 9901 (2000).

Only pure array/scalar kernels (`_tangent_kernel`, `_neb_force_kernel`) are
`@njit`-decorated; the outer functions own all `Poscar`/logging/IO.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from .constraints import build_mask
from .errors import NEBError
from .geometry import direct_to_cartesian
from .numba_shims import njit
from .poscar import Poscar

logger = logging.getLogger(__name__)


@njit(cache=True)
def _tangent_kernel(
    dR_plus:  np.ndarray,
    dR_minus: np.ndarray,
    e_prev:   float,
    e_i:      float,
    e_next:   float,
) -> np.ndarray:
    """Pure-array Henkelman improved-tangent kernel for a single image."""
    if e_next > e_i and e_i > e_prev:
        t = dR_plus
    elif e_next < e_i and e_i < e_prev:
        t = dR_minus
    else:
        dE_plus  = abs(e_next - e_i)
        dE_minus = abs(e_i - e_prev)
        w_max = max(dE_plus, dE_minus)
        w_min = min(dE_plus, dE_minus)
        if e_next > e_prev:
            t = dR_plus * w_max + dR_minus * w_min
        else:
            t = dR_plus * w_min + dR_minus * w_max
    norm = np.sqrt(np.sum(t * t))
    if norm <= 1e-15:
        # e_prev == e_i == e_next makes both energy weights zero, so the
        # weighted tangent vanishes and with it the spring force and the CI
        # inversion. Fall back to the bisection tangent of the same paper,
        # tau = dR-/|dR-| + dR+/|dR+| (Henkelman & Jonsson, JCP 113, 9901
        # (2000)), which is nonzero whenever the neighbours are distinct.
        n_plus  = np.sqrt(np.sum(dR_plus * dR_plus))
        n_minus = np.sqrt(np.sum(dR_minus * dR_minus))
        t = np.zeros_like(dR_plus)
        if n_plus > 1e-15:
            t = t + dR_plus / n_plus
        if n_minus > 1e-15:
            t = t + dR_minus / n_minus
        norm = np.sqrt(np.sum(t * t))
    if norm > 1e-15:
        return t / norm
    return np.zeros_like(t)


@njit(cache=True)
def _neb_force_kernel(
    F_true:       np.ndarray,
    tangent:      np.ndarray,
    dR_plus:      np.ndarray,
    dR_minus:     np.ndarray,
    spring_const: float,
    is_ci:        bool,
) -> np.ndarray:
    """Pure-array CI-NEB effective-force kernel for a single image."""
    proj = np.sum(F_true * tangent)
    if is_ci:
        return F_true - 2.0 * proj * tangent
    norm_plus  = np.sqrt(np.sum(dR_plus * dR_plus))
    norm_minus = np.sqrt(np.sum(dR_minus * dR_minus))
    f_spring = tangent * (spring_const * (norm_plus - norm_minus))
    return (F_true - proj * tangent) + f_spring


def compute_neb_forces(
    images       : List[Poscar],
    true_forces  : List[np.ndarray],
    energies     : List[float],
    spring_const : float,
    ci_index     : int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute the NEB effective force for each intermediate image.

    Returns (neb_forces, tangents):
      neb_forces : list of (N, 3) force arrays (intermediates only)
      tangents   : list of flat 3N unit tangent vectors (intermediates only)

    Selective Dynamics: forces on constrained (fixed) degrees of freedom are
    zeroed immediately after assembling `true_forces`, upstream of the
    tangent/spring/CI math -- a fixed atom therefore both (a) never receives
    a net NEB displacement and (b) never contributes a spuriously nonzero
    residual to the max|F_NEB| convergence check.

    Periodicity invariant (important, and easy to break from elsewhere):
    `dR_plus`/`dR_minus` below are plain Cartesian differences between
    adjacent images, deliberately *without* a minimum-image wrap. That is
    correct only because nothing in this package ever wraps fractional
    coordinates back into [0, 1): `interpolate_images` applies MIC once when
    building the initial path and thereafter every image carries whatever
    unwrapped coordinates that produced (values below 0 or above 1 included),
    `write_poscar`/`read_poscar` round-trip them verbatim, and VASP's own
    re-wrapped CONTCAR is deleted rather than read. The chain therefore stays
    mutually MIC-consistent, so a plain difference already *is* the minimum-
    image displacement. Applying MIC per image pair here instead would be
    wrong for any displacement legitimately exceeding half a cell, and
    conversely, wrapping coordinates anywhere upstream (e.g. "tidying up"
    POSCAR output into [0, 1)) would silently corrupt every tangent in a
    system where an atom crosses a cell boundary.
    """
    n_img = len(images)
    if len(true_forces) != n_img or len(energies) != n_img:
        raise NEBError(
            f"compute_neb_forces: length mismatch - "
            f"images={n_img}, true_forces={len(true_forces)}, "
            f"energies={len(energies)}."
        )
    if n_img < 3:
        raise NEBError(
            "compute_neb_forces: need at least 3 images "
            "(2 endpoints + 1 intermediate)."
        )

    n_atoms  = images[0].n_atoms
    lat      = images[0].lattice
    cart_all = [direct_to_cartesian(img.positions, lat) for img in images]
    masked_forces = [
        np.asarray(f, dtype=float) * build_mask(img.selective_dynamics, (n_atoms, 3))
        for f, img in zip(true_forces, images)
    ]

    neb_forces_out: List[np.ndarray] = []
    tangents_out:   List[np.ndarray] = []

    for i in range(1, n_img - 1):
        dR_plus  = (cart_all[i + 1] - cart_all[i]).ravel()
        dR_minus = (cart_all[i] - cart_all[i - 1]).ravel()
        tangent  = _tangent_kernel(
            dR_plus, dR_minus, energies[i - 1], energies[i], energies[i + 1],
        )
        if energies[i - 1] == energies[i] == energies[i + 1]:
            logger.warning(
                f"  Image [{i}] and both neighbours have identical energies "
                f"({energies[i]:.8f} eV): using the bisection tangent."
            )
        if not np.any(tangent):
            logger.warning(
                f"  Image [{i}] coincides with a neighbour: its tangent is zero, "
                "so it gets no spring force this step."
            )
        tangents_out.append(tangent)

        F_true = masked_forces[i].ravel()
        is_ci  = (i == ci_index)
        F_eff  = _neb_force_kernel(
            F_true, tangent, dR_plus, dR_minus, spring_const, is_ci,
        )

        if is_ci:
            proj = float(np.dot(F_true, tangent))
            logger.debug(f"  Image [{i}] CLIMBING  |F.tau| = {abs(proj):.4f} eV/Ang")

        # Re-apply the mask to the *effective* force: the spring-force term
        # is purely geometric (derived from inter-image distances, not from
        # true_forces) so masking true_forces upstream alone does not zero
        # it for a fixed atom. Masking here too ensures a fixed atom both
        # never receives a net displacement AND never contributes a
        # spuriously nonzero residual to the max|F_NEB| convergence check.
        mask_i = build_mask(images[i].selective_dynamics, (n_atoms, 3)).ravel()
        F_eff  = F_eff * mask_i

        neb_forces_out.append(F_eff.reshape(n_atoms, 3))

    return neb_forces_out, tangents_out
