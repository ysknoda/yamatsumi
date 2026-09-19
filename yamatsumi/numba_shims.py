"""
Optional numba JIT with a transparent no-op fallback.

Design rule enforced by every caller of `njit` in this package: only pure
array-and-scalar kernels are ever decorated -- `Poscar` / `OptimizerState` /
`Path` / logging / file IO must never cross the JIT boundary. That keeps the
fallback path (numba absent) behaviourally identical to the JIT path, and
keeps numba's ~100ms-1s first-call compilation cost confined to hot loops
that are actually called every NEB iteration (never to one-shot setup code).
"""
from __future__ import annotations

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """No-op stand-in for numba.njit when numba is not installed."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]            # used as a bare @njit
        return lambda fn: fn          # used as @njit(cache=True, ...)

__all__ = ["njit", "NUMBA_AVAILABLE"]
