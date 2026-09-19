"""Tests for yamatsumi.plotting."""
from __future__ import annotations

import builtins
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from yamatsumi.plotting import (
    _spline_fallback,
    plot_energy_profile,
    report_energy_profile,
    spline_curve,
)


def test_spline_curve_preserves_endpoints():
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    ys = [0.0, 0.3, -0.1, 0.4, 0.0]
    x_fine, y_fine = spline_curve(xs, ys, n_points=50)
    assert len(x_fine) == len(y_fine) == 50
    assert x_fine[0] == pytest.approx(xs[0])
    assert x_fine[-1] == pytest.approx(xs[-1])
    assert y_fine[0] == pytest.approx(ys[0], abs=1e-9)
    assert y_fine[-1] == pytest.approx(ys[-1], abs=1e-9)


def test_scipy_and_fallback_splines_agree():
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    ys = [0.0, 0.3, -0.1, 0.4, 0.0]
    x_scipy, y_scipy = spline_curve(xs, ys, n_points=40)
    x_fallback, y_fallback = _spline_fallback(xs, ys, n_points=40)
    assert np.allclose(x_scipy, x_fallback, atol=1e-9)
    assert np.allclose(y_scipy, y_fallback, atol=1e-6)


def test_spline_falls_back_without_scipy():
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy.interpolate" or name.startswith("scipy"):
            raise ImportError("scipy not installed (simulated)")
        return real_import(name, *args, **kwargs)

    xs = [0.0, 0.5, 1.0]
    ys = [0.0, 1.0, 0.0]
    with patch("builtins.__import__", side_effect=fake_import):
        x_fine, y_fine = spline_curve(xs, ys, n_points=10)
    assert len(x_fine) == 10


def test_report_energy_profile_logs_table(caplog):
    image_dirs = [Path("00"), Path("01"), Path("02")]
    energies = [-10.0, -9.5, -10.2]
    with caplog.at_level(logging.INFO):
        report_energy_profile(image_dirs, energies)
    text = caplog.text
    assert "00" in text and "01" in text and "02" in text
    assert "initial endpoint" in text
    assert "final endpoint" in text


def test_plot_logs_debug_when_interactive_show_fails(tmp_path, caplog):
    """4b: a failed interactive plt.show() must be logged, not silently swallowed."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    with patch.object(plt, "show", side_effect=RuntimeError("no display available")), \
         caplog.at_level(logging.DEBUG):
        plot_energy_profile(
            image_dirs=[Path("00"), Path("01")],
            energies=[0.0, 1.0],
            ci_index=1,
            phonon={"frequencies": [], "n_imaginary": 0, "is_transition_state": False},
            barrier_fwd=1.0,
            barrier_rev=1.0,
            output_path=tmp_path / "out.png",
            fmt="png",
            no_show=False,
        )
    assert "no display available" in caplog.text


def test_plot_skips_gracefully_without_matplotlib(tmp_path):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("matplotlib not installed (simulated)")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        plot_energy_profile(
            image_dirs=[Path("00"), Path("01")],
            energies=[0.0, 1.0],
            ci_index=1,
            phonon={"frequencies": [], "n_imaginary": 0, "is_transition_state": False},
            barrier_fwd=1.0,
            barrier_rev=1.0,
            output_path=tmp_path / "out.png",
            fmt="png",
            no_show=True,
        )
    assert not (tmp_path / "out.png").exists()
