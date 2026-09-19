"""
Reaction-kinetics quantities derived from a converged CI-NEB energy profile:
activation barriers and an approximate Arrhenius rate-constant estimate.

Kept separate from plotting.py (matplotlib-only) and neb_forces.py
(per-iteration NEB mechanics) since these are post-hoc, once-per-run scalar
quantities computed from the final energy profile alone.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from .constants import KB_EV_PER_K
from .errors import NEBError


def compute_barriers(energies: List[float], ci_index: int) -> Tuple[float, float]:
    """
    Forward and backward activation barriers (eV), relative to the initial
    and final endpoints respectively, given the full per-image energy list
    and the climbing-image (transition-state) index into it.
    """
    return energies[ci_index] - energies[0], energies[ci_index] - energies[-1]


def arrhenius_rate(barrier_ev: float, temperature_k: float, attempt_freq_hz: float) -> float:
    """
    Simple Arrhenius-law rate estimate: k = nu * exp(-Ea / (kB*T)).

    A crude order-of-magnitude estimate from the electronic activation
    barrier and a user-supplied attempt frequency alone -- not a rigorous
    transition-state-theory rate (which would also require the vibrational
    spectra of both the reactant and the transition state). Useful as a
    quick sanity check of whether a barrier is kinetically accessible at a
    given temperature, not as a quantitative rate prediction.

    Raises NEBError for a non-positive temperature or attempt frequency --
    both are unphysical, and temperature_k == 0.0 in particular would
    otherwise divide by zero here (ZeroDivisionError) with no context about
    which CLI input caused it.
    """
    if temperature_k <= 0.0:
        raise NEBError(
            f"arrhenius_rate: temperature_k must be positive (got {temperature_k})."
        )
    if attempt_freq_hz <= 0.0:
        raise NEBError(
            f"arrhenius_rate: attempt_freq_hz must be positive (got {attempt_freq_hz})."
        )
    return attempt_freq_hz * math.exp(-barrier_ev / (KB_EV_PER_K * temperature_k))
