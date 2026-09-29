"""
Geometry-optimiser suite: steepest-descent, Quick-Min, and FIRE.

Semantics follow the VTST Tools optimizers (ADR 0002), with unit mass:
steepest descent moves ``sd_alpha * F`` (VTST SDALPHA), Quick-Min integrates
``v += F*dt; dx = v*dt`` with a fixed ``quickmin_dt`` (VTST TIMESTEP), and
``step_size`` caps every optimiser's per-atom displacement (VTST MAXMOVE).

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

from .constants import DEFAULT_QUICKMIN_DT, DEFAULT_SD_ALPHA
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
    velocity : (N, 3) Cartesian velocity (unit mass)
    dt       : current FIRE time step
    sd_alpha    : steepest-descent step per unit force [Ang^2/eV]
    quickmin_dt : fixed Quick-Min time step
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
        sd_alpha    : float = DEFAULT_SD_ALPHA,
        quickmin_dt : float = DEFAULT_QUICKMIN_DT,
    ):
        self.n_atoms = n_atoms
        self._dt0    = dt0
        self.dt_max  = dt_max
        self.n_min   = n_min
        self.f_inc   = f_inc
        self.f_dec   = f_dec
        self._alpha0 = alpha0
        self.f_alpha = f_alpha
        self.sd_alpha    = sd_alpha
        self.quickmin_dt = quickmin_dt
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
            sd_alpha    = args.sd_alpha,
            quickmin_dt = args.quickmin_dt,
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


@njit(cache=True)
def _cap_displacement(dx_flat: np.ndarray, step_size: float, n_atoms: int):
    """Scale *dx_flat* so no atom moves more than *step_size*. Returns the
    (possibly scaled) displacement and the factor applied (1.0 if none)."""
    d_max = _max_atom_norm(dx_flat, n_atoms)
    if d_max > step_size:
        factor = step_size / d_max
        return dx_flat * factor, factor
    return dx_flat, 1.0


# ---------------------------------------------------------------------------
# Steepest-descent
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sd_kernel(cart_flat: np.ndarray, forces_flat: np.ndarray, alpha: float,
               step_size: float, n_atoms: int) -> np.ndarray:
    dx, _ = _cap_displacement(forces_flat * alpha, step_size, n_atoms)
    return cart_flat + dx


def step_sd(poscar: Poscar, forces: np.ndarray, step_size: float,
            alpha: float = DEFAULT_SD_ALPHA) -> Poscar:
    """
    Steepest-descent step: move ``alpha * F`` [alpha in Ang^2/eV], with no
    atom moving more than *step_size* [Ang]. The step shrinks with the force,
    so the path can settle instead of limit-cycling at a fixed step length.
    """
    n = poscar.n_atoms
    mask = build_mask(poscar.selective_dynamics, (n, 3))
    forces = apply_mask(np.asarray(forces, dtype=float), mask)
    if _max_atom_norm(forces.ravel(), n) < 1e-10:
        return poscar

    lat  = poscar.lattice
    cart = direct_to_cartesian(poscar.positions, lat).ravel()
    new_cart_flat = _sd_kernel(cart, forces.ravel(), alpha, step_size, n)

    return poscar.with_positions(cartesian_to_direct(new_cart_flat.reshape(n, 3), lat))


# ---------------------------------------------------------------------------
# Quick-Min
# ---------------------------------------------------------------------------

@njit(cache=True)
def _quickmin_kernel(
    cart_flat:     np.ndarray,
    forces_flat:   np.ndarray,
    velocity_flat: np.ndarray,
    dt:            float,
    step_size:     float,
    n_atoms:       int,
):
    """
    One Quick-Min step (unit mass). Returns (new_cart_flat, new_velocity_flat).

    The velocity keeps only its component along the current force, and only
    while it points downhill; then ``v += F*dt`` and the atoms move ``v*dt``.
    Under a steady force the step therefore grows each iteration -- the whole
    point of Quick-Min over steepest descent. If the step exceeds
    *step_size*, the step and the stored velocity are scaled together so the
    velocity stays consistent with the move actually made.
    """
    P = np.sum(velocity_flat * forces_flat)
    F_norm2 = np.sum(forces_flat * forces_flat)
    if P > 0.0 and F_norm2 > 1e-20:
        v = forces_flat * (P / F_norm2)
    else:
        v = np.zeros_like(forces_flat)

    v = v + forces_flat * dt
    dx, factor = _cap_displacement(v * dt, step_size, n_atoms)
    return cart_flat + dx, v * factor


def step_quickmin(
    poscar    : Poscar,
    forces    : np.ndarray,
    state     : OptimizerState,
    step_size : float,
) -> Poscar:
    """Velocity-projection (Quick-Min) step with time step ``state.quickmin_dt``."""
    n = poscar.n_atoms
    mask = build_mask(poscar.selective_dynamics, (n, 3))
    forces = apply_mask(np.asarray(forces, dtype=float), mask)
    if _max_atom_norm(forces.ravel(), n) < 1e-10:
        return poscar

    lat = poscar.lattice
    cart = direct_to_cartesian(poscar.positions, lat).ravel()
    velocity = apply_mask(np.asarray(state.velocity, dtype=float), mask).ravel()

    new_cart_flat, v_flat = _quickmin_kernel(
        cart, forces.ravel(), velocity, state.quickmin_dt, step_size, n,
    )
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

    # Mask BEFORE moving: the climbing-image kernel works with the unmasked
    # NEB tangent, so its velocity can be nonzero on frozen components.
    v_flat = apply_mask(v_flat.reshape(n, 3), mask).ravel()
    # The kernels cap the velocity at step_size, which bounds the move only
    # while dt <= 1; cap the move itself as well (ADR 0002).
    dx_flat, _ = _cap_displacement(v_flat * new_dt, step_size, n)
    new_cart_flat = cart_flat + dx_flat

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
            new_p = step_sd(poscar, fmat, step_size, alpha=state.sd_alpha)
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
