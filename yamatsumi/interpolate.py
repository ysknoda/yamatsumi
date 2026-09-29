"""NEB image interpolation via the minimum-image convention (MIC)."""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from .constraints import apply_mask, build_mask
from .errors import NEBError
from .geometry import cartesian_to_direct, direct_to_cartesian, mic_displacement
from .poscar import Poscar

logger = logging.getLogger(__name__)


def check_endpoint_consistency(init: Poscar, final: Poscar) -> None:
    """
    Raise NEBError if init and final POSCARs are incompatible for NEB
    interpolation: different atom count, species, counts, lattice, or
    Selective Dynamics mask.

    Also requires both endpoints' `.positions` to already be fractional
    (coord_type == "Direct"), since the MIC displacement this module computes
    is only meaningful in fractional coordinates. `read_poscar` always
    normalises to Direct, so this is unreachable via the CLI; it guards
    against a hand-built or future-caller-supplied Poscar in Cartesian
    coordinates silently corrupting the interpolation.
    """
    for label, poscar in (("init", init), ("final", final)):
        if poscar.coord_type != "Direct":
            raise NEBError(
                f"{label} Poscar must have coord_type == 'Direct' for NEB "
                f"interpolation (got {poscar.coord_type!r})."
            )
    if init.n_atoms != final.n_atoms:
        raise NEBError(
            f"Atom count mismatch: init={init.n_atoms}, final={final.n_atoms}."
        )
    if init.species != final.species:
        raise NEBError(
            f"Species mismatch: init={init.species}, final={final.species}."
        )
    if init.counts != final.counts:
        raise NEBError(
            f"Counts mismatch per species: init={init.counts}, "
            f"final={final.counts}."
        )

    tol = 1e-4
    diff = np.abs(np.asarray(init.lattice) - np.asarray(final.lattice))
    if np.any(diff > tol):
        r, c = (int(x) for x in np.argwhere(diff > tol)[0])
        raise NEBError(
            f"Lattice mismatch at row {r} col {c}: "
            f"init={init.lattice[r][c]:.6f}, "
            f"final={final.lattice[r][c]:.6f} "
            f"(diff={diff[r][c]:.2e} Ang, threshold={tol} Ang)."
        )

    init_sd, final_sd = init.selective_dynamics, final.selective_dynamics
    if (init_sd is None) != (final_sd is None):
        raise NEBError(
            "Selective Dynamics mismatch: present on only one endpoint "
            f"(init={'yes' if init_sd is not None else 'no'}, "
            f"final={'yes' if final_sd is not None else 'no'})."
        )
    if init_sd is not None and not np.array_equal(init_sd, final_sd):
        atom, comp = (int(x) for x in np.argwhere(init_sd != final_sd)[0])
        raise NEBError(
            f"Selective Dynamics mismatch at atom {atom + 1}, component "
            f"{'xyz'[comp]}: init={bool(init_sd[atom, comp])}, "
            f"final={bool(final_sd[atom, comp])}. Endpoints must agree on "
            "which degrees of freedom are fixed."
        )


def interpolate_images(
    init:       Poscar,
    final:      Poscar,
    num_images: int,
) -> List[Poscar]:
    """
    Create *num_images* linearly interpolated intermediate Poscar objects.
    MIC is applied per atom to ensure the shortest periodic path.
    Returns [ init, img_1, ..., img_N, final ].

    The returned final endpoint is *final* unwrapped against *init* (ADR
    0001): any atom whose fractional coordinate differs from init by more
    than half a cell is shifted by whole lattice vectors onto the same MIC
    path the intermediate images follow. VASP writes CONTCAR wrapped into
    [0, 1), so a relaxed final endpoint can put an atom that moved from
    z = 0.02 to z = -0.02 at z = 0.98, and the un-MIC'd tangent of the last
    segment would then contain a whole lattice vector. Atoms that need no
    shift keep their coordinates bit for bit.
    """
    check_endpoint_consistency(init, final)

    d = mic_displacement(init.positions, final.positions)  # (N, 3), MIC-wrapped

    shift = np.round(final.positions - init.positions)       # whole cells, per component
    if np.any(shift != 0.0):
        moved = sorted({int(a) for a in np.nonzero(np.any(shift != 0.0, axis=1))[0]})
        logger.info(
            "Unwrapped the final endpoint against the initial one by whole "
            f"lattice vectors for {len(moved)} atom(s): "
            + ", ".join(str(a + 1) for a in moved[:20])
            + (" ..." if len(moved) > 20 else "")
            + " (VASP writes CONTCAR wrapped into [0, 1); see docs/adr/0001)."
        )
        final = final.with_positions(final.positions - shift)

    images = [init.copy()]
    for i in range(1, num_images + 1):
        frac = i / (num_images + 1)
        img = init.copy()
        img.comment   = f"NEB image {i}/{num_images}  frac={frac:.4f}  (MIC)"
        img.positions = init.positions + frac * d
        images.append(img)

    images.append(final.copy())
    logger.info(
        f"Generated {num_images} intermediate images with MIC interpolation "
        f"({len(images)} total including endpoints)."
    )
    return images


# ---------------------------------------------------------------------------
# IDPP (Image Dependent Pair Potential) interpolation
#
# Smidstrup, Pedersen, Stokbro, Jonsson, "Improved initial guess for minimum
# energy path calculations", J. Chem. Phys. 140, 214106 (2014).
#
# Plain linear interpolation can route atoms through each other when
# multiple atoms move past one another along the path. IDPP instead relaxes
# each intermediate image's Cartesian positions toward a target pairwise
# interatomic-distance matrix that itself linearly interpolates between the
# two endpoints' distance matrices, weighted 1/d^4 so short (bonded-range)
# distances dominate the fit. This is a purely geometric, VASP-free
# refinement of the initial guess -- it does not touch energies or forces.
# ---------------------------------------------------------------------------

# Any two distinct atoms closer than this (Ang) are treated as a duplicated
# -atom input error rather than as physics: no real interatomic distance is
# anywhere near this small (the shortest known bond, H2, is ~0.74 Ang), and
# IDPP's 1/d^4 weighting divides by exactly this quantity.
_MIN_PAIR_DISTANCE_ANG = 1e-8


def _reject_coincident_atoms(label: str, dist: np.ndarray, off_diag: np.ndarray) -> None:
    """
    Raise NEBError if any two distinct atoms in the *label* endpoint sit at
    (essentially) the same position.

    IDPP weights each atom pair by 1/d_target^4, so a coincident pair makes
    the weight infinite, the objective's gradient NaN, and every relaxed
    image's coordinates NaN -- which would otherwise be written out to POSCAR
    files silently, since NaN propagates through the steepest-descent loop
    without raising anything.
    """
    too_close = (dist < _MIN_PAIR_DISTANCE_ANG) & off_diag
    if not np.any(too_close):
        return
    atom_i, atom_j = (int(x) for x in np.argwhere(too_close)[0])
    raise NEBError(
        f"IDPP interpolation: atoms {atom_i + 1} and {atom_j + 1} of the "
        f"{label} endpoint are {dist[atom_i, atom_j]:.3e} Ang apart, which is "
        "indistinguishable from the same position. Check that endpoint for a "
        "duplicated atom (IDPP weights atom pairs by 1/d^4 and cannot use a "
        "zero separation). Plain --path-interpolation linear does not use "
        "interatomic distances and is unaffected."
    )


def _pairwise_cartesian(positions: np.ndarray, lattice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    (N,N,3) MIC Cartesian displacement tensor (cart_diff[i,j] = R_j - R_i)
    and (N,N) MIC Cartesian distance matrix, for fractional `positions`.
    """
    frac_diff = mic_displacement(positions[:, None, :], positions[None, :, :])
    cart_diff = direct_to_cartesian(frac_diff, lattice)
    dist = np.linalg.norm(cart_diff, axis=2)
    return cart_diff, dist


def _idpp_objective_and_gradient(
    positions : np.ndarray,
    lattice   : np.ndarray,
    d_target  : np.ndarray,
    weights   : np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    S(R) = sum_{i<j} w_ij (d_ij(R) - D_ij)^2  and its Cartesian gradient.

    `weights`/`d_target` are full symmetric (N,N) matrices with a zero
    diagonal, so summing over the full matrix double-counts each i<j pair;
    halving the raw sum recovers the i<j-only objective.
    """
    n = positions.shape[0]
    cart_diff, dist = _pairwise_cartesian(positions, lattice)
    dist_safe = dist + np.eye(n)                    # avoid 0/0 on the diagonal
    unit  = cart_diff / dist_safe[:, :, None]         # unit[i,j] = (R_j-R_i)/d_ij, 0 on diagonal
    delta = dist - d_target                            # (N,N), diagonal irrelevant (weights=0 there)
    coef  = weights * delta                              # (N,N)
    grad  = -2.0 * np.einsum("ij,ijc->ic", coef, unit)     # (N,3); d/dR_i of the (i,j) and (j,i) terms
    objective = float(np.sum(weights * delta ** 2)) / 2.0
    return objective, grad


def _idpp_relax_image(
    positions : np.ndarray,
    lattice   : np.ndarray,
    d_target  : np.ndarray,
    weights   : np.ndarray,
    mask      : np.ndarray,
    max_steps : int,
    force_tol : float,
    step_size : float,
) -> np.ndarray:
    """
    Steepest-descent relaxation of one image's fractional positions against
    the IDPP pairwise-distance objective (mirrors optimizers.step_sd's
    max-displacement-cap idiom). Fixed degrees of freedom (per `mask`) never
    move. Returns the relaxed fractional positions.
    """
    pos = positions.copy()
    for _ in range(max_steps):
        _, grad = _idpp_objective_and_gradient(pos, lattice, d_target, weights)
        grad = apply_mask(grad, mask)
        max_g = float(np.max(np.linalg.norm(grad, axis=1)))
        if max_g < force_tol:
            break
        cart = direct_to_cartesian(pos, lattice)
        cart = cart - (step_size / max_g) * grad
        pos = cartesian_to_direct(cart, lattice)
    return pos


def idpp_interpolate_images(
    init:       Poscar,
    final:      Poscar,
    num_images: int,
    max_steps:  int = 300,
    force_tol:  float = 1e-3,
    step_size:  float = 0.01,
) -> List[Poscar]:
    """
    Create *num_images* intermediate Poscar objects via IDPP interpolation
    (see module notes above) instead of plain linear interpolation.

    Starts from interpolate_images()'s linear+MIC guess for every image,
    then relaxes each intermediate image independently toward its own
    linearly-interpolated target pairwise-distance matrix. Endpoints are
    never touched. Returns [ init, img_1, ..., img_N, final ].
    """
    linear_images = interpolate_images(init, final, num_images)

    lattice  = init.lattice
    n_atoms  = init.n_atoms
    mask     = build_mask(init.selective_dynamics, init.positions.shape)
    off_diag = ~np.eye(n_atoms, dtype=bool)

    _, d0 = _pairwise_cartesian(init.positions, lattice)
    _, d1 = _pairwise_cartesian(final.positions, lattice)
    _reject_coincident_atoms("init", d0, off_diag)
    _reject_coincident_atoms("final", d1, off_diag)

    images = [linear_images[0]]
    for k in range(1, num_images + 1):
        s = k / (num_images + 1)
        d_target = d0 + s * (d1 - d0)
        weights = np.zeros_like(d_target)
        weights[off_diag] = 1.0 / (d_target[off_diag] ** 4)

        img = linear_images[k]
        relaxed_pos = _idpp_relax_image(
            img.positions, lattice, d_target, weights, mask,
            max_steps, force_tol, step_size,
        )
        relaxed_img = img.with_positions(relaxed_pos)
        relaxed_img.comment = img.comment + "  (IDPP)"
        images.append(relaxed_img)

    images.append(linear_images[-1])
    logger.info(
        f"Generated {num_images} intermediate images with IDPP interpolation "
        f"({len(images)} total including endpoints)."
    )
    return images
