"""Tests for yamatsumi.phonon (TS criterion logic; VASP itself is not invoked here)."""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from yamatsumi.phonon import verify_transition_state
from yamatsumi.poscar import Poscar


def _poscar():
    return Poscar(
        comment="t", scale=1.0, lattice=np.eye(3) * 10.0, species=["H"], counts=[1],
        coord_type="Direct", positions=np.array([[0.5, 0.5, 0.5]]),
    )


def _run_with_fake_frequencies(tmp_path, frequencies, vasp_ok=True, eigenvector_modes=None, tangent=None):
    copy_dir = tmp_path / "CopyDir"
    copy_dir.mkdir()
    (copy_dir / "INCAR").write_text("ENCUT = 400\n")
    image_dir = tmp_path / "00"
    image_dir.mkdir()

    with patch("yamatsumi.phonon.run_vasp", return_value=vasp_ok), \
         patch("yamatsumi.phonon.parse_frequencies_from_outcar", return_value=frequencies), \
         patch("yamatsumi.phonon.parse_eigenvectors_from_outcar", return_value=eigenvector_modes or []):
        return verify_transition_state(_poscar(), image_dir, "fake_vasp", copy_dir, tangent=tangent)


def test_exactly_one_imaginary_is_transition_state(tmp_path):
    result = _run_with_fake_frequencies(tmp_path, [-5.0, 10.0, 20.0])
    assert result["n_imaginary"] == 1
    assert result["is_transition_state"] is True


def test_zero_imaginary_is_not_transition_state(tmp_path):
    result = _run_with_fake_frequencies(tmp_path, [1.0, 2.0, 3.0])
    assert result["n_imaginary"] == 0
    assert result["is_transition_state"] is False


def test_two_or_more_imaginary_is_not_transition_state(tmp_path):
    result = _run_with_fake_frequencies(tmp_path, [-5.0, -3.0, 10.0])
    assert result["n_imaginary"] == 2
    assert result["is_transition_state"] is False


def test_vasp_failure_reports_unverified(tmp_path):
    result = _run_with_fake_frequencies(tmp_path, [], vasp_ok=False)
    assert result == dict(
        frequencies=[], n_imaginary=0, n_zero_modes=0, is_transition_state=False,
        phonon_calculation_failed=True, eigenvector_tangent_alignment=None,
    )


def test_successful_run_reports_not_failed(tmp_path):
    """4a: a clean run (regardless of TS outcome) must report failed=False."""
    result = _run_with_fake_frequencies(tmp_path, [1.0, 2.0, 3.0])
    assert result["phonon_calculation_failed"] is False


# ---------------------------------------------------------------------------
# 1d: eigenvector-tangent alignment diagnostic
# ---------------------------------------------------------------------------

def test_alignment_is_one_for_parallel_tangent(tmp_path):
    modes = [(-5.0, np.array([[1.0, 0.0, 0.0]])), (10.0, np.array([[0.0, 1.0, 0.0]]))]
    result = _run_with_fake_frequencies(
        tmp_path, [-5.0, 10.0], eigenvector_modes=modes, tangent=np.array([1.0, 0.0, 0.0]),
    )
    assert result["eigenvector_tangent_alignment"] == pytest.approx(1.0)


def test_alignment_antiparallel_tangent_still_reports_one(tmp_path):
    """Sign is arbitrary for a normal mode -- anti-parallel counts as aligned."""
    modes = [(-5.0, np.array([[1.0, 0.0, 0.0]])), (10.0, np.array([[0.0, 1.0, 0.0]]))]
    result = _run_with_fake_frequencies(
        tmp_path, [-5.0, 10.0], eigenvector_modes=modes, tangent=np.array([-1.0, 0.0, 0.0]),
    )
    assert result["eigenvector_tangent_alignment"] == pytest.approx(1.0)


def test_alignment_is_zero_for_orthogonal_tangent(tmp_path):
    modes = [(-5.0, np.array([[1.0, 0.0, 0.0]])), (10.0, np.array([[0.0, 1.0, 0.0]]))]
    result = _run_with_fake_frequencies(
        tmp_path, [-5.0, 10.0], eigenvector_modes=modes, tangent=np.array([0.0, 1.0, 0.0]),
    )
    assert result["eigenvector_tangent_alignment"] == pytest.approx(0.0, abs=1e-12)


def test_alignment_none_without_tangent(tmp_path):
    modes = [(-5.0, np.array([[1.0, 0.0, 0.0]])), (10.0, np.array([[0.0, 1.0, 0.0]]))]
    result = _run_with_fake_frequencies(tmp_path, [-5.0, 10.0], eigenvector_modes=modes)
    assert result["eigenvector_tangent_alignment"] is None


def test_alignment_none_when_not_a_transition_state(tmp_path):
    """Two imaginary modes -> not a TS -> alignment is not computed even with a tangent."""
    modes = [(-5.0, np.array([[1.0, 0.0, 0.0]])), (-3.0, np.array([[0.0, 1.0, 0.0]]))]
    result = _run_with_fake_frequencies(
        tmp_path, [-5.0, -3.0], eigenvector_modes=modes, tangent=np.array([1.0, 0.0, 0.0]),
    )
    assert result["eigenvector_tangent_alignment"] is None


def test_alignment_none_when_eigenvector_not_matched(tmp_path):
    """Frequency/eigenvector parse mismatch must degrade gracefully, not raise."""
    result = _run_with_fake_frequencies(
        tmp_path, [-5.0, 10.0], eigenvector_modes=[], tangent=np.array([1.0, 0.0, 0.0]),
    )
    assert result["eigenvector_tangent_alignment"] is None


def test_incar_relaxation_tags_are_stripped_and_phonon_tags_appended(tmp_path):
    copy_dir = tmp_path / "CopyDir"
    copy_dir.mkdir()
    (copy_dir / "INCAR").write_text("ENCUT = 400\nIBRION = 2\nNSW = 100\nPOTIM = 0.5\n")
    image_dir = tmp_path / "00"
    image_dir.mkdir()

    with patch("yamatsumi.phonon.run_vasp", return_value=True), \
         patch("yamatsumi.phonon.parse_frequencies_from_outcar", return_value=[1.0]):
        verify_transition_state(_poscar(), image_dir, "fake_vasp", copy_dir)

    incar_text = (image_dir / "phonon" / "INCAR").read_text()
    assert "IBRION = 2" not in incar_text
    assert "NSW = 100" not in incar_text
    assert "ENCUT = 400" in incar_text
    assert "IBRION = 5" in incar_text
    assert "NFREE  = 2" in incar_text


# ---------------------------------------------------------------------------
# ADR 0006: translational zero modes must not be counted as imaginary
# ---------------------------------------------------------------------------

_REAL_BULK_SI_PHONON = (
    Path(__file__).parent / "fixtures" / "real_vasp_excerpts"
    / "si8_bulk_ibrion6_vasp643_phonon.OUTCAR"
)


def _run_on_real_outcar(tmp_path, outcar_source, **kwargs):
    """Run verify_transition_state with VASP replaced by copying a real OUTCAR in."""
    copy_dir = tmp_path / "CopyDir"
    copy_dir.mkdir()
    (copy_dir / "INCAR").write_text("ENCUT = 400\n")
    image_dir = tmp_path / "00"
    image_dir.mkdir()

    def fake_run_vasp(phonon_dir, *args, **kw):
        shutil.copy(outcar_source, Path(phonon_dir) / "OUTCAR")
        return True

    with patch("yamatsumi.phonon.run_vasp", side_effect=fake_run_vasp):
        return verify_transition_state(_poscar(), image_dir, "fake_vasp", copy_dir, **kwargs)


def test_real_vasp_bulk_si_minimum_has_no_imaginary_modes(tmp_path):
    """
    Regression test on real VASP 6.4.3 output. Perfect diamond Si at its
    minimum: VASP labels the three translational modes 'f/i' at ~3e-6 meV.
    They were previously counted as three imaginary modes.
    """
    result = _run_on_real_outcar(tmp_path, _REAL_BULK_SI_PHONON)
    assert len(result["frequencies"]) == 24
    assert result["n_imaginary"] == 0
    assert result["n_zero_modes"] == 3
    assert result["is_transition_state"] is False


def test_zero_modes_do_not_hide_a_genuine_imaginary_mode(tmp_path):
    result = _run_with_fake_frequencies(tmp_path, [-3e-6, -3e-6, -3e-6, -5.0, 10.0, 20.0])
    assert result["n_imaginary"] == 1
    assert result["n_zero_modes"] == 3
    assert result["is_transition_state"] is True


def test_alignment_uses_the_imaginary_mode_not_a_translational_one(tmp_path):
    """
    The translational modes come first here, with the same negative sign as
    the genuine imaginary mode. The diagnostic must use the genuine one.
    """
    zero = np.array([[0.0, 1.0, 0.0]])
    modes = [(-3e-6, zero), (-3e-6, zero), (-3e-6, zero),
             (-5.0, np.array([[1.0, 0.0, 0.0]])), (10.0, np.array([[0.0, 0.0, 1.0]]))]
    result = _run_with_fake_frequencies(
        tmp_path, [f for f, _ in modes], eigenvector_modes=modes,
        tangent=np.array([1.0, 0.0, 0.0]),
    )
    assert result["is_transition_state"] is True
    assert result["eigenvector_tangent_alignment"] == pytest.approx(1.0)


def test_zero_mode_tolerance_is_configurable(tmp_path):
    """With the tolerance set to 0 the old sign-only rule is recovered."""
    result = _run_on_real_outcar(tmp_path, _REAL_BULK_SI_PHONON, zero_mode_tol_mev=0.0)
    assert result["n_imaginary"] == 3
    assert result["n_zero_modes"] == 0


def test_warns_when_full_hessian_zero_mode_count_is_not_three(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _run_with_fake_frequencies(tmp_path, [-5.0, 10.0, 20.0])
    assert "zero mode" in caplog.text.lower()
