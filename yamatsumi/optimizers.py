"""
Geometry-optimiser suite: steepest-descent, Quick-Min, and FIRE.

FIRE hyper-parameters live in `OptimizerState` (not in CLI args), so
`step_fire` needs no reference to the argument namespace.

Only pure array/scalar kernels are `@njit`-decorated; `OptimizerState`
mutation and `Poscar` marshalling stay in the plain-Python wrapper
functions (`OptimizerState` is an ordinary Python class and cannot itself
cross the JIT boundary).

Selective Dynamics: every stepper masks both the incoming force and the
carried velocity against `poscar.selective_dynamics` (force masking already
happens once upstream in `neb_forces.compute_neb_forces`; re-applying it
here to velocity is deliberate defense in depth against momentum drifting a
fixed atom on a later step).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .constraints import apply_mask, build_mask
from .errors import NEBError
from .geometry import cartesian_to_direct, direct_to_cartesian
from .numba_shims import njit
from .poscar import Poscar


class OptimizerState:
    """
    Per-image mutable state for QuickMin and FIRE optimisers.

    Attributes
    ----------
    velocity : (N, 3) Cartesian velocity [Ang/step]
    dt       : current time step
    alpha    : FIRE mixing parameter
    n_pos    : consecutive steps with P >= 0
    """

    def __init__(
        self,
        n_atoms  : int,
        dt0      : float = 0.1,
        dt_max   : float = 1.0,
        n_min    : int   = 5,
        f_inc    : float = 1.1,
        f_dec    : float = 0.5,
        alpha0   : float = 0.1,
        f_alpha  : float = 0.99,
    ):
        self.n_atoms = n_atoms
        self._dt0    = dt0
        self.dt_max  = dt_max
        self.n_min   = n_min
        self.f_inc   = f_inc
        self.f_dec   = f_dec
        self._alpha0 = alpha0
        self.f_alpha = f_alpha
        self.velocity = np.zeros((n_atoms, 3), dtype=float)
        self.dt       = dt0
        self.alpha    = alpha0
        self.n_pos    = 0

    def reset_velocity(self, dt_factor: float = 0.5) -> None:
        """Zero velocity and reduce dt (called when P < 0 in FIRE)."""
        self.velocity = np.zeros((self.n_atoms, 3), dtype=float)
        self.dt      *= dt_factor
        self.alpha    = self._alpha0
        self.n_pos    = 0

    def full_reset(self) -> None:
        """Restore all scalars to initial values (called on CI change)."""
        self.velocity = np.zeros((self.n_atoms, 3), dtype=float)
        self.dt       = self._dt0
        self.alpha    = self._alpha0
        self.n_pos    = 0


def make_optimizer_states(n_images: int, n_atoms: int, args) -> List[OptimizerState]:
    """Create one OptimizerState per intermediate image."""
    return [
        OptimizerState(
            n_atoms = n_atoms,
            dt0     = args.fire_dt0,
            dt_max  = args.fire_dt_max,
            n_min   = args.fire_n_min,
            f_inc   = args.fire_f_inc,
            f_dec   = args.fire_f_dec,
            alpha0  = args.fire_alpha0,
            f_alpha = args.fire_f_alpha,
        )
        for _ in range(n_images)
    ]


# ---------------------------------------------------------------------------
# Shared kernel helper
# ---------------------------------------------------------------------------

@njit(cache=True)
def _max_atom_norm(v_flat: np.ndarray, n_atoms: int) -> float:
    """Largest per-atom Euclidean norm in a flat 3N vector."""
    max_v = 0.0
    for i in range(n_atoms):
        vx = v_flat[3 * i]
        vy = v_flat[3 * i + 1]
        vz = v_flat[3 * i + 2]
        norm = np.sqrt(vx * vx + vy * vy + vz * vz)
        if norm > max_v:
            max_v = norm
    return max_v


# ---------------------------------------------------------------------------
# Steepest-descent
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sd_kernel(cart_flat: np.ndarray, forces_flat: np.ndarray, step_size: float, max_f: float) -> np.ndarray:
    return cart_flat + (step_size / max_f) * forces_flat


def step_sd(poscar: Poscar, forces: np.ndarray, step_size: float) -> Poscar:
    """Steepest-descent step with maximum-displacement cap."""
    n = poscar.n_atoms
    mask = build_mask(poscar.selective_dynamics, (n, 3))
    forces = apply_mask(np.asarray(forces, dtype=float), mask)
    max_f = _max_atom_norm(forces.ravel(), n)
    if max_f < 1e-10:
        return poscar

    lat  = poscar.lattice
    cart = direct_to_cartesian(poscar.positions, lat).ravel()
    new_cart_flat = _sd_kernel(cart, forces.ravel(), step_size, max_f)

    return poscar.with_positions(cartesian_to_direct(new_cart_flat.reshape(n, 3), lat))


# ---------------------------------------------------------------------------
# Quick-Min
# ---------------------------------------------------------------------------

@njit(cache=True)
def _quickmin_kernel(
    cart_flat:     np.ndarray,
    forces_flat:   np.ndarray,
    velocity_flat: np.ndarray,
    step_size:     float,
    max_f:         float,
    n_atoms:       int,
):
    """Returns (new_cart_flat, new_velocity_flat)."""
    P = np.sum(velocity_flat * forces_flat)
    F_norm2 = np.sum(forces_flat * forces_flat)
    if P > 0.0 and F_norm2 > 1e-20:
        v = forces_flat * (P / F_norm2)
    else:
        v = np.zeros_like(forces_flat)

    # dt is not a physical timestep here: it is the same max-force-normalising
    # scale factor _sd_kernel uses (step_size / max_f), chosen so that
    # forces_flat * dt has max per-atom norm exactly step_size. Bug fix: v
    # (built from this dt-scaled force plus the velocity projection, then
    # explicitly capped at step_size below) IS therefore already the intended
    # per-step Cartesian displacement -- it must be added to cart_flat
    # directly. Multiplying it by dt a second time (the previous behaviour)
    # applied the unbounded step_size/max_f factor twice: harmless when
    # max_f happens to be O(step_size), but as max_f -> 0 (the near-converged
    # regime, exactly where the smallest, most careful steps are needed) dt
    # diverges and the position update exploded to a displacement far beyond
    # the step_size cap (e.g. max_f=1e-4 eV/Ang, step_size=0.05 Ang gave a 25
    # Ang single-step displacement -- 500x the intended cap). FIRE's velocity
    # (step_fire, below) legitimately multiplies by dt twice because FIRE's
    # own dt is a separately bounded (dt_max) adaptive timestep, not this
    # unbounded max_f-normalising factor -- that pattern is correct and is
    # not touched here.
    dt = step_size / max_f
    v = v + forces_flat * dt

    v_max = _max_atom_norm(v, n_atoms)
    if v_max > step_size:
        v = v * (step_size / v_max)

    new_cart_flat = cart_flat + v
    return new_cart_flat, v


def step_quickmin(
    poscar    : Poscar,
    forces    : np.ndarray,
    state     : OptimizerState,
    step_size : float,
) -> Poscar:
    """Velocity-projection (Quick-Min) step."""
    n = poscar.n_atoms
    mask = build_mask(poscar.selective_dynamics, (n, 3))
    forces = apply_mask(np.asarray(forces, dtype=float), mask)
    max_f = _max_atom_norm(forces.ravel(), n)
    if max_f < 1e-10:
        return poscar

    lat = poscar.lattice
    cart = direct_to_cartesian(poscar.positions, lat).ravel()
    velocity = apply_mask(np.asarray(state.velocity, dtype=float), mask).ravel()

    new_cart_flat, v_flat = _quickmin_kernel(cart, forces.ravel(), velocity, step_size, max_f, n)
    v_flat = apply_mask(v_flat.reshape(n, 3), mask).ravel()

    state.velocity = v_flat.reshape(n, 3)
    return poscar.with_positions(cartesian_to_direct(new_cart_flat.reshape(n, 3), lat))


# ---------------------------------------------------------------------------
# Image-individual FIRE
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fire_kernel_ci(
    F_flat, v_flat, tangent,
    dt, alpha, n_pos,
    n_min, f_inc, f_dec, alpha0, f_alpha, dt_max,
    step_size, n_atoms,
):
    """FIRE update for the climbing image (perpendicular/tangential split)."""
    F_t    = np.sum(F_flat * tangent)
    F_tang = tangent * F_t
    F_perp = F_flat - F_tang
    v_t    = np.sum(v_flat * tangent)
    v_tang = tangent * v_t
    v_perp = v_flat - v_tang

    P_perp   = np.sum(F_perp * v_perp)
    F_p_norm = np.sqrt(np.sum(F_perp * F_perp))

    if P_perp > 0.0 and F_p_norm > 1e-20:
        v_perp_norm = np.sqrt(np.sum(v_perp * v_perp))
        v_perp = v_perp * (1.0 - alpha) + F_perp * (alpha * v_perp_norm / F_p_norm)
        if n_pos > n_min:
            dt    = min(dt * f_inc, dt_max)
            alpha = alpha * f_alpha
        n_pos = n_pos + 1
    else:
        v_perp = np.zeros_like(v_perp)
        dt     = dt * f_dec
        alpha  = alpha0
        n_pos  = 0

    v_perp = v_perp + F_perp * dt
    v_tang = v_tang + F_tang * dt
    v_flat = v_perp + v_tang

    v_max = _max_atom_norm(v_flat, n_atoms)
    if v_max > step_size:
        v_flat = v_flat * (step_size / v_max)

    return v_flat, dt, alpha, n_pos


@njit(cache=True)
def _fire_kernel_plain(
    F_flat, v_flat,
    dt, alpha, n_pos,
    n_min, f_inc, f_dec, alpha0, f_alpha, dt_max,
    step_size, n_atoms,
):
    """FIRE update for a non-climbing image."""
    F_norm = np.sqrt(np.sum(F_flat * F_flat))
    P      = np.sum(F_flat * v_flat)
    if P > 0.0 and F_norm > 1e-20:
        v_norm = np.sqrt(np.sum(v_flat * v_flat))
        v_flat = v_flat * (1.0 - alpha) + F_flat * (alpha * v_norm / F_norm)
        if n_pos > n_min:
            dt    = min(dt * f_inc, dt_max)
            alpha = alpha * f_alpha
        n_pos = n_pos + 1
    else:
        v_flat = np.zeros_like(v_flat)
        dt     = dt * f_dec
        alpha  = alpha0
        n_pos  = 0

    v_flat = v_flat + F_flat * dt

    v_max = _max_atom_norm(v_flat, n_atoms)
    if v_max > step_size:
        v_flat = v_flat * (step_size / v_max)

    return v_flat, dt, alpha, n_pos


def step_fire(
    poscar    : Poscar,
    forces    : np.ndarray,
    state     : OptimizerState,
    step_size : float,
    is_ci     : bool,
    tangent   : Optional[np.ndarray],
) -> Poscar:
    """
    One FIRE step with CI split-component treatment.

    For the climbing image the power test P = F.v is applied only to the
    perpendicular component of the force/velocity so that the uphill
    tangential component does not trigger false resets. Velocity is capped
    BEFORE the Euler displacement so the capped velocity is used for both
    the position update and the stored state.
    """
    n = poscar.n_atoms
    lat = poscar.lattice
    mask = build_mask(poscar.selective_dynamics, (n, 3))

    forces_flat   = apply_mask(np.asarray(forces, dtype=float), mask).ravel()
    velocity_flat = apply_mask(np.asarray(state.velocity, dtype=float), mask).ravel()
    cart_flat     = direct_to_cartesian(poscar.positions, lat).ravel()

    if is_ci and tangent is not None:
        v_flat, new_dt, new_alpha, new_n_pos = _fire_kernel_ci(
            forces_flat, velocity_flat, np.asarray(tangent, dtype=float),
            state.dt, state.alpha, state.n_pos,
            state.n_min, state.f_inc, state.f_dec, state._alpha0, state.f_alpha, state.dt_max,
            step_size, n,
        )
    else:
        v_flat, new_dt, new_alpha, new_n_pos = _fire_kernel_plain(
            forces_flat, velocity_flat,
            state.dt, state.alpha, state.n_pos,
            state.n_min, state.f_inc, state.f_dec, state._alpha0, state.f_alpha, state.dt_max,
            step_size, n,
        )

    new_cart_flat = cart_flat + v_flat * new_dt
    v_flat = apply_mask(v_flat.reshape(n, 3), mask).ravel()

    state.dt       = new_dt
    state.alpha    = new_alpha
    state.n_pos    = new_n_pos
    state.velocity = v_flat.reshape(n, 3)

    return poscar.with_positions(cartesian_to_direct(new_cart_flat.reshape(n, 3), lat))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def update_all_images(
    images     : List[Poscar],
    neb_forces : List[np.ndarray],
    tangents   : List[np.ndarray],
    ci_index   : int,
    opt_states : List[OptimizerState],
    optimizer  : str,
    step_size  : float,
) -> List[Poscar]:
    """Apply the selected optimiser to all intermediate images."""
    opt     = optimizer.lower()
    updated = []
    for j, (fmat, tau) in enumerate(zip(neb_forces, tangents)):
        img_idx = j + 1
        state   = opt_states[j]
        poscar  = images[img_idx]
        is_ci   = (img_idx == ci_index)

        if opt == "sd":
            new_p = step_sd(poscar, fmat, step_size)
        elif opt == "quickmin":
            new_p = step_quickmin(poscar, fmat, state, step_size)
        elif opt == "fire":
            new_p = step_fire(poscar, fmat, state, step_size, is_ci, tau)
        else:
            raise NEBError(
                f"Unknown optimizer '{optimizer}'. "
                "Choose 'sd', 'quickmin', or 'fire'."
            )
        updated.append(new_p)
    return updated
