"""Selective-Dynamics (per-atom constraint) mask helpers."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def build_mask(selective_dynamics: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """
    Return a boolean (N, 3) mask, True = free (unconstrained). When
    *selective_dynamics* is None (no Selective Dynamics block in the
    POSCAR), every degree of freedom is free.
    """
    if selective_dynamics is None:
        return np.ones(shape, dtype=bool)
    return np.asarray(selective_dynamics, dtype=bool)


def apply_mask(vec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out components where mask is False (constrained/fixed)."""
    return vec * mask
