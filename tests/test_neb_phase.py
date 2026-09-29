"""Tests for internal helpers in yamatsumi.phases.neb_phase."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from yamatsumi.cli import build_parser
from yamatsumi.directories import get_image_dirs
from yamatsumi.errors import NEBError
from yamatsumi.phases.neb_phase import (
    _compute_convergence,
    _last_completed_iteration,
    finalize,
    phase_neb,
)
from yamatsumi.poscar import Poscar, write_poscar


def test_compute_convergence_matches_hand_computed_values():
    """
    2b regression: vectorised max/RMS must match the same hand-computed
    values as the original per-atom Python-loop implementation.

    Image 1 forces [[3,4,0],[0,0,0]] -> per-atom norms [5, 0].
    Image 2 forces [[0,0,1]]         -> per-atom norm  [1].
    Flat norms = [5, 0, 1]; max = 5; RMS = sqrt((25+0+1)/3) = sqrt(26/3).
    """
    neb_forces = [
        np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
    ]
    image_dirs = [Path("00"), Path("01"), Path("02")]
    max_force, rms_force = _compute_convergence(
        neb_forces=neb_forces,
        image_dirs=image_dirs,
        ci_index=1,
        ci_active=False,
        opt_states=[None, None],
        optimizer="sd",
        force_tol=0.05,
        verbose=False,
        root=False,
    )
    assert max_force == pytest.approx(5.0)
    assert rms_force == pytest.approx((26.0 / 3.0) ** 0.5)


def _make_finalize_args(tmp_path, **overrides):
    """
    Args for finalize(), built from build_parser() so that every attribute
    finalize reads exists with its real default -- a hand-written Namespace
    silently falls behind whenever a CLI option is added.
    """
    ns = build_parser().parse_args([
        "--work-dir", str(tmp_path), "--dry-run", "--vasp-cmd", "unused", "--no-show",
    ])
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _dummy_poscar():
    return Poscar(
        comment="t", scale=1.0, lattice=np.eye(3) * 10.0, species=["H"], counts=[1],
        coord_type="Direct", positions=np.array([[0.5, 0.5, 0.5]]),
    )


def test_finalize_writes_arrhenius_rate_when_requested(tmp_path):
    """1b: --rate-estimate arrhenius must add a rate line to the result file."""
    images     = [_dummy_poscar(), _dummy_poscar(), _dummy_poscar()]
    energies   = [-10.0, -9.5, -10.2]
    image_dirs = [tmp_path / n for n in ("00", "01", "02")]
    args = _make_finalize_args(tmp_path, rate_estimate="arrhenius")

    with patch("yamatsumi.phases.neb_phase.plot_energy_profile"):
        finalize(images, energies, 1, image_dirs, tmp_path / "CopyDir", args)

    text = (tmp_path / "yamatsumi_result.txt").read_text()
    assert "Arrhenius rate estimate" in text

    barrier_fwd = energies[1] - energies[0]
    expected_rate = 1.0e13 * math.exp(-barrier_fwd / (8.617333262e-5 * 300.0))
    assert f"{expected_rate:.6e}" in text


def test_finalize_omits_rate_line_by_default(tmp_path):
    images     = [_dummy_poscar(), _dummy_poscar(), _dummy_poscar()]
    energies   = [-10.0, -9.5, -10.2]
    image_dirs = [tmp_path / n for n in ("00", "01", "02")]
    args = _make_finalize_args(tmp_path)  # rate_estimate="none" by default

    with patch("yamatsumi.phases.neb_phase.plot_energy_profile"):
        finalize(images, energies, 1, image_dirs, tmp_path / "CopyDir", args)

    text = (tmp_path / "yamatsumi_result.txt").read_text()
    assert "Arrhenius rate estimate" not in text


def test_finalize_passes_ci_tangent_and_writes_alignment(tmp_path):
    """1d: finalize() must pass ci_tangent through to verify_transition_state
    and surface the resulting alignment diagnostic in the result file."""
    images     = [_dummy_poscar(), _dummy_poscar(), _dummy_poscar()]
    energies   = [-10.0, -9.5, -10.2]
    image_dirs = [tmp_path / n for n in ("00", "01", "02")]
    args = _make_finalize_args(tmp_path, dry_run=False)

    fake_phonon = dict(
        frequencies=[-5.0], n_imaginary=1, is_transition_state=True,
        phonon_calculation_failed=False, eigenvector_tangent_alignment=0.987,
    )
    ci_tangent = np.array([1.0, 0.0, 0.0])
    with patch("yamatsumi.phases.neb_phase.plot_energy_profile"), \
         patch("yamatsumi.phases.neb_phase.verify_transition_state",
               return_value=fake_phonon) as mock_verify:
        finalize(
            images, energies, 1, image_dirs, tmp_path / "CopyDir", args,
            ci_tangent=ci_tangent,
        )

    assert mock_verify.call_args.kwargs["tangent"] is ci_tangent

    text = (tmp_path / "yamatsumi_result.txt").read_text()
    assert "Eigenvector-tangent alignment: 0.9870" in text


def test_finalize_dry_run_never_computes_alignment(tmp_path):
    images     = [_dummy_poscar(), _dummy_poscar(), _dummy_poscar()]
    energies   = [-10.0, -9.5, -10.2]
    image_dirs = [tmp_path / n for n in ("00", "01", "02")]
    args = _make_finalize_args(tmp_path)  # dry_run=True by default

    with patch("yamatsumi.phases.neb_phase.plot_energy_profile"):
        finalize(
            images, energies, 1, image_dirs, tmp_path / "CopyDir", args,
            ci_tangent=np.array([1.0, 0.0, 0.0]),
        )

    text = (tmp_path / "yamatsumi_result.txt").read_text()
    assert "Eigenvector-tangent alignment" not in text


def test_last_completed_iteration_missing_file_returns_zero(tmp_path):
    assert _last_completed_iteration(tmp_path / "nope.csv") == 0


def test_last_completed_iteration_reads_last_row(tmp_path):
    """4c: --resume must continue from the last row actually written, not restart at 1."""
    path = tmp_path / "convergence_history.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "iteration", "max_force_eV_per_Ang", "rms_force_eV_per_Ang",
            "ci_image", "ci_energy_eV", "ci_active",
        ])
        writer.writerow([1, "0.500000", "0.300000", "01", "-10.000000", True])
        writer.writerow([2, "0.400000", "0.250000", "01", "-10.100000", True])
        writer.writerow([3, "0.300000", "0.200000", "02", "-10.200000", True])
    assert _last_completed_iteration(path) == 3


def test_last_completed_iteration_header_only_returns_zero(tmp_path):
    path = tmp_path / "convergence_history.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "iteration", "max_force_eV_per_Ang", "rms_force_eV_per_Ang",
            "ci_image", "ci_energy_eV", "ci_active",
        ])
    assert _last_completed_iteration(path) == 0


def test_last_completed_iteration_ignores_unparseable_trailing_row(tmp_path):
    """A malformed trailing row must not crash resume; the last good row wins."""
    path = tmp_path / "convergence_history.csv"
    with open(path, "w", newline="") as fh:
        fh.write(
            "iteration,max_force_eV_per_Ang,rms_force_eV_per_Ang,"
            "ci_image,ci_energy_eV,ci_active\n"
            "1,0.5,0.3,01,-10.0,True\n"
            "2,0.4,0.25,01,-10.1,True\n"
            "not-a-number,truncated\n"
        )
    assert _last_completed_iteration(path) == 2


# ---------------------------------------------------------------------------
# Regression: --rate-estimate arrhenius must validate temperature/attempt-freq
# eagerly, before the (expensive) NEB loop, not only deep inside finalize()
# after convergence.
# ---------------------------------------------------------------------------

def _minimal_phase_neb_args(tmp_path, **overrides):
    """
    A real parsed argument namespace pointed at tmp_path, with `overrides`
    applied on top. Built from build_parser() rather than a hand-written
    Namespace so that every attribute phase_neb (or anything it calls, e.g.
    make_optimizer_states' FIRE hyper-parameters) reads is present by
    construction, instead of being discovered one AttributeError at a time.
    """
    copy_dir = tmp_path / "CopyDir"
    copy_dir.mkdir(exist_ok=True)
    args = build_parser().parse_args([
        "--copy-dir", str(copy_dir),
        "--work-dir", str(tmp_path / "NEB_run"),
        "--rate-estimate", "arrhenius",
        "--num-images", "3",
        "--optimizer", "fire",
    ])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_phase_neb_rejects_nonpositive_temperature_before_loop(tmp_path):
    """
    A bad --temperature must fail immediately (before even checking that the
    image directories from Phase 1 exist), not only after a full,
    potentially hours-long NEB run has already converged and reached
    finalize()'s own (also-guarded) arrhenius_rate() call.
    """
    args = _minimal_phase_neb_args(tmp_path, temperature=0.0)
    with pytest.raises(NEBError, match="temperature"):
        phase_neb(args)


def test_phase_neb_rejects_negative_attempt_freq_before_loop(tmp_path):
    args = _minimal_phase_neb_args(tmp_path, attempt_freq=-1.0)
    with pytest.raises(NEBError, match="attempt-freq"):
        phase_neb(args)


def test_phase_neb_allows_valid_rate_params_past_the_check(tmp_path):
    """
    Sanity check that the new validation does not false-positive on the
    default, valid temperature/attempt-freq: it must fail later, for the
    unrelated (expected, in this minimal test setup) reason that Phase 1's
    image directories do not exist yet -- not for the rate-estimate check.
    """
    args = _minimal_phase_neb_args(tmp_path)
    with pytest.raises(NEBError, match="Image directory not found"):
        phase_neb(args)


# ---------------------------------------------------------------------------
# Regression: an empty iteration range must fail with a clear message rather
# than skipping the loop body and crashing on an empty `energies` list.
# ---------------------------------------------------------------------------

def _stage_image_dirs(work_dir, num_images):
    """Create the Phase-1 image directories + POSCARs phase_neb expects."""
    for idir in get_image_dirs(work_dir, num_images):
        idir.mkdir(parents=True, exist_ok=True)
        write_poscar(str(idir / "POSCAR"), _dummy_poscar())


def _write_history(csv_path, last_iteration):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "iteration", "max_force_eV_per_Ang", "rms_force_eV_per_Ang",
            "ci_image", "ci_energy_eV", "ci_active",
        ])
        for i in range(1, last_iteration + 1):
            writer.writerow([i, "0.500000", "0.300000", "01", "-10.000000", True])


def test_phase_neb_resume_past_max_iter_reports_clearly(tmp_path):
    """
    The realistic --resume failure: a previous run used up every --max-iter
    iteration without converging, and is resumed without raising --max-iter.
    start_iteration then exceeds max_iter, the loop body never runs, and
    `energies` stays empty -- which used to reach report_energy_profile() and
    raise a bare IndexError. It must instead say what happened and what to do.
    """
    work_dir = tmp_path / "NEB_run"
    args = _minimal_phase_neb_args(
        tmp_path, work_dir=str(work_dir), rate_estimate="none",
        resume=True, max_iter=3, num_images=3,
    )
    _stage_image_dirs(work_dir, args.num_images)
    _write_history(work_dir / "convergence_history.csv", last_iteration=3)

    with pytest.raises(NEBError) as exc_info:
        phase_neb(args)
    message = str(exc_info.value)
    assert "--max-iter" in message
    assert "without converging" in message


def test_phase_neb_zero_max_iter_reports_clearly(tmp_path):
    """The same empty-range guard, reached without --resume."""
    work_dir = tmp_path / "NEB_run"
    args = _minimal_phase_neb_args(
        tmp_path, work_dir=str(work_dir), rate_estimate="none",
        resume=False, max_iter=0, num_images=3,
    )
    _stage_image_dirs(work_dir, args.num_images)

    with pytest.raises(NEBError, match="at least 1"):
        phase_neb(args)


def test_phase_neb_resume_within_max_iter_proceeds(tmp_path):
    """
    The guard must not false-positive on a resume that still has iterations
    left: with 3 completed and --max-iter 10 the run continues, failing only
    later for the unrelated (expected here) missing-OUTCAR reason.
    """
    work_dir = tmp_path / "NEB_run"
    args = _minimal_phase_neb_args(
        tmp_path, work_dir=str(work_dir), rate_estimate="none",
        resume=True, max_iter=10, num_images=3,
        log_interval=1, spring_const=5.0, force_tol=0.05, verbose=False,
    )
    _stage_image_dirs(work_dir, args.num_images)
    _write_history(work_dir / "convergence_history.csv", last_iteration=3)

    with pytest.raises(NEBError, match="Cannot read energy/forces"):
        phase_neb(args)


# ---------------------------------------------------------------------------
# MPI: no rank may rewrite its image files while another rank is still
# reading this iteration's OUTCARs/POSCARs
# ---------------------------------------------------------------------------

class _RecordingComm:
    """Fake 2-rank communicator that records collective calls into `events`."""

    def __init__(self, events, rank=0, size=2):
        self._events, self._rank, self._size = events, rank, size

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def Barrier(self):
        self._events.append("barrier")

    def allreduce(self, value, op=None):
        return value


def test_update_and_run_waits_for_every_rank_before_touching_files(tmp_path, monkeypatch):
    """
    Regression test for an MPI race. Every rank reads *all* image OUTCARs and
    POSCARs at the start of an iteration, then rewrites the files of its own
    images (prepare_image_directory deletes the old OUTCAR first). Without a
    barrier in between, a fast rank deleted an OUTCAR a slower rank was still
    reading: NEBError "Cannot read energy/forces", or, if the file was caught
    half-written, a silently wrong energy that made MPI and serial runs
    diverge.
    """
    from yamatsumi.phases import neb_phase

    events: list[str] = []
    n_inter = 3
    poscars = [_dummy_poscar() for _ in range(n_inter)]
    monkeypatch.setattr(neb_phase, "update_all_images", lambda **kw: [p.copy() for p in poscars])
    monkeypatch.setattr(neb_phase, "prepare_image_directory", lambda *a, **k: events.append("prepare"))
    monkeypatch.setattr(neb_phase, "write_poscar", lambda *a, **k: events.append("write"))
    monkeypatch.setattr(neb_phase, "run_vasp_for_images", lambda *a, **k: events.append("vasp"))

    image_dirs = [tmp_path / f"{i:02d}" for i in range(n_inter + 2)]
    neb_phase._update_and_run(
        images=[_dummy_poscar()] * (n_inter + 2), neb_forces=[None] * n_inter,
        tangents=[None] * n_inter, ci_index=2, opt_states=[None] * n_inter,
        image_dirs=image_dirs, copy_dir=tmp_path, iteration=1, n_inter=n_inter,
        optimizer="fire", step_size=0.05, vasp_cmd="unused",
        comm=_RecordingComm(events), local_workers=1, sanitize_mpi_env=False,
        vasp_timeout=None, root=True,
    )

    assert "prepare" in events, "rank 0 of 2 owns at least one image"
    first_touch = min(events.index(e) for e in ("prepare", "write") if e in events)
    assert "barrier" in events[:first_touch], (
        f"image files were touched before all ranks synchronised: {events}"
    )


def test_phase_neb_refuses_a_single_digit_work_dir_from_1_0_0a1(tmp_path):
    """
    ADR 0005: 1.0.0a1 named image directories "0".."4" for 3 images. Resuming
    such a run must stop with an explanation, not silently create "00".."04"
    and start again from scratch while the user thinks it was continued.
    """
    from yamatsumi.phases.neb_phase import phase_neb

    args = _minimal_phase_neb_args(tmp_path, resume=True)
    for i in range(5):
        (tmp_path / "NEB_run" / str(i)).mkdir(parents=True)
    with pytest.raises(NEBError, match="single-digit"):
        phase_neb(args)


def test_finalize_skip_ts_verification_runs_no_phonons_and_says_so(tmp_path):
    """
    --skip-ts-verification must not launch the phonon run, and the result
    must say the TS was not checked -- not "0 imaginary modes, not a TS".
    """
    images     = [_dummy_poscar(), _dummy_poscar(), _dummy_poscar()]
    energies   = [-10.0, -9.5, -10.2]
    image_dirs = [tmp_path / n for n in ("00", "01", "02")]
    args = _make_finalize_args(tmp_path, dry_run=False, skip_ts_verification=True)

    with patch("yamatsumi.phases.neb_phase.plot_energy_profile") as plot, \
         patch("yamatsumi.phases.neb_phase.verify_transition_state") as verify:
        finalize(images, energies, 1, image_dirs, tmp_path / "CopyDir", args)

    verify.assert_not_called()
    text = (tmp_path / "yamatsumi_result.txt").read_text()
    assert "SKIPPED" in text
    assert "TS criterion satisfied" not in text
    assert plot.call_args.kwargs["phonon"]["verification_skipped"] is True


def test_finalize_dry_run_is_reported_as_unverified(tmp_path):
    images     = [_dummy_poscar(), _dummy_poscar(), _dummy_poscar()]
    image_dirs = [tmp_path / n for n in ("00", "01", "02")]
    with patch("yamatsumi.phases.neb_phase.plot_energy_profile"):
        finalize(images, [-10.0, -9.5, -10.2], 1, image_dirs, tmp_path / "CopyDir",
                 _make_finalize_args(tmp_path))
    assert "SKIPPED" in (tmp_path / "yamatsumi_result.txt").read_text()
