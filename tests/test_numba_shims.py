"""
Tests for yamatsumi.numba_shims.

The fallback decorator's logic is tested directly against a standalone copy
of its implementation rather than by forcing yamatsumi.numba_shims to reload
with numba's import patched to fail: reload-based simulation mutates a
process-global module object, and a previous version of this test left it
mutated for the rest of the test session (a real bug caught by the very
tests it broke -- see git history / PR discussion). Testing the fallback's
logic directly avoids that whole class of cross-test state leakage.
"""
from __future__ import annotations

import pytest


def _fallback_njit(*args, **kwargs):
    """Exact copy of yamatsumi.numba_shims's fallback decorator logic."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda fn: fn


def test_fallback_bare_decorator_is_identity():
    @_fallback_njit
    def bare(x):
        return x + 1

    assert bare(1) == 2


def test_fallback_decorator_with_kwargs_is_identity():
    @_fallback_njit(cache=True)
    def with_kwargs(x):
        return x * 2

    assert with_kwargs(3) == 6


def test_numba_shims_exports_njit_and_availability_flag():
    from yamatsumi import numba_shims

    assert callable(numba_shims.njit)
    assert isinstance(numba_shims.NUMBA_AVAILABLE, bool)


def test_numba_kernels_match_python_reference_when_available():
    from yamatsumi.numba_shims import NUMBA_AVAILABLE

    if not NUMBA_AVAILABLE:
        pytest.skip("numba not installed")

    import numpy as np

    from yamatsumi.optimizers import _max_atom_norm

    v = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 5.0])
    jit_result   = _max_atom_norm(v, 2)
    plain_result = _max_atom_norm.py_func(v, 2)
    assert jit_result == pytest.approx(plain_result)
    assert jit_result == pytest.approx(5.0)
