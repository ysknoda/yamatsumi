"""Tests for yamatsumi.kinetics."""
from __future__ import annotations

import pytest

from yamatsumi.errors import NEBError
from yamatsumi.kinetics import arrhenius_rate, compute_barriers


def test_compute_barriers_basic():
    energies = [-10.0, -9.0, -8.5, -9.5, -10.2]
    ci_index = 2
    barrier_fwd, barrier_rev = compute_barriers(energies, ci_index)
    assert barrier_fwd == pytest.approx(-8.5 - (-10.0))   # 1.5
    assert barrier_rev == pytest.approx(-8.5 - (-10.2))   # 1.7


def test_compute_barriers_zero_when_ci_is_an_endpoint():
    energies = [-5.0, -4.0, -6.0]
    assert compute_barriers(energies, 0) == (0.0, 1.0)
    assert compute_barriers(energies, 2) == (-1.0, 0.0)


def test_arrhenius_rate_hand_computed():
    """
    k = nu * exp(-Ea / (kB*T)); Ea=0.5 eV, T=300 K, nu=1e13 Hz.
    kB*T = 8.617333262e-5 * 300 = 0.025851999786 eV, so Ea/(kB*T) =
    19.34086353624264 and k = 1e13 * exp(-19.34086...) = 39844.62015337332
    (independently computed, not merely re-deriving the implementation's
    own expression).
    """
    barrier_ev = 0.5
    temperature_k = 300.0
    attempt_freq_hz = 1.0e13
    assert arrhenius_rate(barrier_ev, temperature_k, attempt_freq_hz) == pytest.approx(
        39844.62015337332, rel=1e-9
    )


def test_arrhenius_rate_zero_barrier_equals_attempt_frequency():
    assert arrhenius_rate(0.0, 300.0, 1.0e13) == pytest.approx(1.0e13)


def test_arrhenius_rate_decreases_with_larger_barrier():
    low  = arrhenius_rate(0.2, 300.0, 1.0e13)
    high = arrhenius_rate(0.8, 300.0, 1.0e13)
    assert high < low


def test_arrhenius_rate_rejects_nonpositive_temperature():
    """
    Regression: temperature_k=0.0 previously raised an uninformative
    ZeroDivisionError (division by KB_EV_PER_K * 0.0) instead of a clear,
    package-consistent error.
    """
    with pytest.raises(NEBError):
        arrhenius_rate(0.5, 0.0, 1.0e13)
    with pytest.raises(NEBError):
        arrhenius_rate(0.5, -10.0, 1.0e13)


def test_arrhenius_rate_rejects_nonpositive_attempt_freq():
    with pytest.raises(NEBError):
        arrhenius_rate(0.5, 300.0, 0.0)
    with pytest.raises(NEBError):
        arrhenius_rate(0.5, 300.0, -1.0e13)
