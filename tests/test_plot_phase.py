"""Tests for yamatsumi.phases.plot_phase."""
from __future__ import annotations

import argparse
import logging
from unittest.mock import patch

from yamatsumi.phases.plot_phase import phase_plot


def _make_args(tmp_path, **overrides):
    ns = argparse.Namespace(
        work_dir=str(tmp_path),
        num_images=1,
        plot_out="energy_profile.png",
        plot_fmt="png",
        no_show=True,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _write_outcar(path, energy):
    path.write_text(f"  free  energy   TOTEN  =   {energy:.6f} eV\n")


def _stage_images(tmp_path):
    for name, e in [("0", -10.0), ("1", -9.0), ("2", -10.5)]:
        d = tmp_path / name
        d.mkdir()
        _write_outcar(d / "OUTCAR", e)


def test_phase_plot_parses_phonon_calculation_failed_true(tmp_path):
    """4a: the re-parsed phonon dict must reflect a failed prior phonon run."""
    _stage_images(tmp_path)
    (tmp_path / "yamatsumi_result.txt").write_text(
        "Phonon calculation failed   : True\n"
        "Imaginary frequencies found : 0\n"
        "TS criterion satisfied      : False\n"
    )
    captured = {}

    def fake_plot(**kwargs):
        captured.update(kwargs)

    with patch("yamatsumi.phases.plot_phase.plot_energy_profile", side_effect=fake_plot):
        phase_plot(_make_args(tmp_path))

    assert captured["phonon"]["phonon_calculation_failed"] is True


def test_phase_plot_defaults_to_not_failed_without_result_file(tmp_path):
    _stage_images(tmp_path)
    captured = {}

    def fake_plot(**kwargs):
        captured.update(kwargs)

    with patch("yamatsumi.phases.plot_phase.plot_energy_profile", side_effect=fake_plot):
        phase_plot(_make_args(tmp_path))

    assert captured["phonon"]["phonon_calculation_failed"] is False


def test_phase_plot_malformed_imaginary_count_logs_warning_not_silent(tmp_path, caplog):
    """4b: a malformed result-file line must be logged, not silently ignored."""
    _stage_images(tmp_path)
    (tmp_path / "yamatsumi_result.txt").write_text(
        "Imaginary frequencies found : not-a-number\n"
    )
    with patch("yamatsumi.phases.plot_phase.plot_energy_profile"), \
         caplog.at_level(logging.WARNING):
        phase_plot(_make_args(tmp_path))
    assert "Unparseable" in caplog.text
