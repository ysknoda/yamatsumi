"""
Phase 2: CI-NEB optimisation loop.

MPI notes: force/tangent/convergence computation is replicated on every
rank (cheap, and every rank already needs every image's energy/forces to
evaluate the improved tangent, so there is no second synchronisation point
to design around). Only two things are rank-specific: (a) which images a
rank writes a POSCAR for and runs VASP on -- `vasp_runner.run_vasp_for_images`
distributes that deterministically via `parallel.distribute_indices`, using
the same (n_inter, rank, size) on both the write and run steps so a rank
always writes the POSCAR for exactly the images it will also run VASP for;
and (b) all logging / CSV / TS_POSCAR / result-summary / plot output, which
is gated to rank 0 (`root`) only, so an MPI run does not print/write
everything N times.
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .. import parallel
from ..directories import (
    check_no_legacy_image_dirs,
    check_vasp_relaxation_disabled,
    get_image_dirs,
    prepare_image_directory,
)
from ..errors import NEBError
from ..kinetics import arrhenius_rate, compute_barriers
from ..neb_forces import compute_neb_forces
from ..optimizers import make_optimizer_states, update_all_images
from ..outcar import parse_energy_from_outcar, parse_forces_from_outcar
from ..phonon import verify_transition_state
from ..plotting import plot_energy_profile, report_energy_profile
from ..poscar import Poscar, read_poscar, write_poscar
from ..vasp_runner import run_vasp_for_images

logger = logging.getLogger(__name__)


def _rlog(root: bool, msg: str, level: int = logging.INFO) -> None:
    if root:
        logger.log(level, msg)


def _parse_iteration_data(
    image_dirs : List[Path],
    n_atoms    : int,
) -> Tuple[List[float], List[np.ndarray]]:
    """Read energies and forces from all OUTCAR files."""
    energies: List[float] = []
    true_forces: List[np.ndarray] = []
    for idir in image_dirs:
        e = parse_energy_from_outcar(idir / "OUTCAR")
        f = parse_forces_from_outcar(idir / "OUTCAR", n_atoms)
        if e is None or f is None:
            raise NEBError(f"Cannot read energy/forces from {idir / 'OUTCAR'}.")
        energies.append(e)
        true_forces.append(f)
    return energies, true_forces


def _log_energy_profile(
    image_dirs : List[Path],
    energies   : List[float],
    n_total    : int,
    root       : bool,
) -> None:
    """Log relative energy for every image."""
    e0 = energies[0]
    _rlog(root, "  Energy profile (relative to initial endpoint) [eV]:")
    for i, e in enumerate(energies):
        tag = " <-- endpoint" if i in (0, n_total - 1) else ""
        _rlog(root, f"    [{image_dirs[i].name}]  {e - e0:+.6f}{tag}")


def _compute_convergence(
    neb_forces : List[np.ndarray],
    image_dirs : List[Path],
    ci_index   : int,
    ci_active  : bool,
    opt_states : list,
    optimizer  : str,
    force_tol  : float,
    verbose    : bool,
    root       : bool,
) -> Tuple[float, float]:
    """Compute max|F_NEB| and RMS|F_NEB|; *** CI *** label only when active."""
    all_norms: List[np.ndarray] = []
    for j, fmat in enumerate(neb_forces):
        img_idx   = j + 1
        norms     = np.linalg.norm(np.asarray(fmat, dtype=float), axis=1)
        f_max_img = float(norms.max())
        all_norms.append(norms)

        ci_tag = " *** CI ***" if (ci_active and img_idx == ci_index) else ""
        fire_info = ""
        if optimizer == "fire" and verbose:
            st = opt_states[j]
            fire_info = f"  [FIRE: dt={st.dt:.4f} alpha={st.alpha:.4f} n_pos={st.n_pos}]"
        _rlog(
            root,
            f"  [{image_dirs[img_idx].name}]  "
            f"max|F_NEB| = {f_max_img:.5f} eV/Ang{ci_tag}{fire_info}",
        )

    norms_flat = np.concatenate(all_norms)
    max_force  = float(norms_flat.max())
    rms_force  = float(np.sqrt(np.mean(norms_flat ** 2)))
    _rlog(root, f"  Global  max|F_NEB| = {max_force:.5f} eV/Ang  (threshold = {force_tol:.5f} eV/Ang)")
    _rlog(root, f"  Global  RMS|F_NEB| = {rms_force:.5f} eV/Ang")
    return max_force, rms_force


def _append_convergence_history(
    csv_path   : Path,
    iteration  : int,
    max_force  : float,
    rms_force  : float,
    ci_index   : int,
    image_dirs : List[Path],
    energies   : List[float],
    ci_active  : bool,
    root       : bool,
) -> None:
    """Append one row to convergence_history.csv (root only)."""
    if not root:
        return
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow([
                "iteration", "max_force_eV_per_Ang", "rms_force_eV_per_Ang",
                "ci_image", "ci_energy_eV", "ci_active",
            ])
        writer.writerow([
            iteration,
            f"{max_force:.6f}",
            f"{rms_force:.6f}",
            image_dirs[ci_index].name,
            f"{energies[ci_index]:.6f}",
            ci_active,
        ])


def _archive_csv(csv_path: Path, root: bool) -> None:
    """If csv_path exists and --resume was NOT given, archive it (root only)."""
    if not root:
        return
    if csv_path.exists():
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = csv_path.with_name(csv_path.stem + f"_{ts}" + csv_path.suffix)
        csv_path.rename(archive)
        logger.info(f"  Existing convergence history archived: {archive.name}")


def _last_completed_iteration(csv_path: Path) -> int:
    """
    Return the iteration number of the last row in convergence_history.csv,
    or 0 if the file is absent/empty. Used by --resume so the iteration
    counter continues from where a previous, interrupted run left off
    instead of restarting at 1 and producing duplicate iteration numbers in
    the (not archived, since --resume keeps it) retained CSV.
    """
    if not csv_path.exists():
        return 0
    last = 0
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            try:
                last = int(row[0])
            except ValueError:
                continue
    return last


def _update_and_run(
    images           : List[Poscar],
    neb_forces       : List[np.ndarray],
    tangents         : List[np.ndarray],
    ci_index         : int,
    opt_states       : list,
    image_dirs       : List[Path],
    copy_dir         : Path,
    iteration        : int,
    n_inter          : int,
    optimizer        : str,
    step_size        : float,
    vasp_cmd         : str,
    comm,
    local_workers    : int,
    sanitize_mpi_env : bool,
    vasp_timeout     : Optional[float],
    root             : bool,
) -> List[Poscar]:
    """
    Update POSCARs with the selected optimiser and re-run VASP.

    Returns the updated intermediate-image Poscars (length n_inter, endpoints
    excluded). Force/tangent/optimiser-state computation is replicated on
    every rank (see module docstring), so this return value is already
    complete and identical on every rank -- serial callers can cache it
    directly instead of re-reading POSCAR from disk next iteration.
    """
    _rlog(root, f"  Updating positions [{optimizer.upper()}] ...")
    updated_poscars = update_all_images(
        images     = images,
        neb_forces = neb_forces,
        tangents   = tangents,
        ci_index   = ci_index,
        opt_states = opt_states,
        optimizer  = optimizer,
        step_size  = step_size,
    )
    for j, new_poscar in enumerate(updated_poscars):
        img_idx = j + 1
        new_poscar.comment = (
            f"CI-NEB iter {iteration}  image {img_idx}  "
            f"opt={optimizer}  "
            f"CI={'yes' if img_idx == ci_index else 'no'}"
        )

    # Every rank has just read *all* images' OUTCARs (and, under MPI, all
    # POSCARs) for this iteration. The writes below start with
    # prepare_image_directory(), which deletes the old OUTCAR, so without
    # this barrier a fast rank deletes or half-rewrites a file a slower rank
    # is still reading: either NEBError "Cannot read energy/forces" or,
    # worse, a silently wrong energy that makes ranks -- and MPI vs serial
    # runs -- diverge.
    parallel.barrier(comm)

    # Each rank writes POSCARs only for the images it will also run VASP
    # for below (same distribute_indices call in both places), avoiding a
    # concurrent-write race on the same file from multiple ranks.
    rank = parallel.get_rank(comm)
    size = parallel.get_size(comm)
    my_indices = set(parallel.distribute_indices(n_inter, rank, size))
    intermediate_dirs = image_dirs[1:1 + n_inter]

    for j in range(n_inter):
        if j in my_indices:
            idir = intermediate_dirs[j]
            prepare_image_directory(idir, copy_dir)
            write_poscar(str(idir / "POSCAR"), updated_poscars[j])

    _rlog(root, "  Launching VASP for intermediate images ...")
    run_vasp_for_images(
        intermediate_dirs, vasp_cmd, comm=comm,
        local_workers=local_workers, sanitize_mpi_env=sanitize_mpi_env,
        vasp_timeout=vasp_timeout,
    )
    return updated_poscars


def finalize(
    images     : List[Poscar],
    energies   : List[float],
    ci_index   : int,
    image_dirs : List[Path],
    copy_dir   : Path,
    args,
    ci_tangent : Optional[np.ndarray] = None,
) -> None:
    """
    Save TS_POSCAR, run phonon verification, write result summary, plot.
    Only ever called on rank 0 (see phase_neb), so no internal root gating.

    *ci_tangent* is the flat 3N CI-NEB tangent at the climbing image from
    the final (converged) iteration, if available -- passed through to
    verify_transition_state() for the eigenvector-tangent alignment
    diagnostic. None (the default) simply skips that diagnostic.
    """
    work_dir    = Path(args.work_dir).resolve()
    ts_poscar   = images[ci_index]
    ts_dir      = image_dirs[ci_index]
    e0          = energies[0]
    barrier_fwd, barrier_rev = compute_barriers(energies, ci_index)

    ts_path = work_dir / "TS_POSCAR"
    write_poscar(str(ts_path), ts_poscar)
    logger.info(f"\n  Transition-state POSCAR written: {ts_path}")
    logger.info(f"  Forward  activation barrier: {barrier_fwd:.4f} eV")
    logger.info(f"  Backward activation barrier: {barrier_rev:.4f} eV")

    rate = None
    if args.rate_estimate == "arrhenius":
        rate = arrhenius_rate(barrier_fwd, args.temperature, args.attempt_freq)
        logger.info(
            f"  Arrhenius rate estimate    : {rate:.4e} 1/s  "
            f"(T={args.temperature:.1f} K, nu={args.attempt_freq:.2e} Hz)"
        )

    skip_reason = (
        "dry run" if args.dry_run
        else "--skip-ts-verification" if args.skip_ts_verification
        else None
    )
    if skip_reason is not None:
        _rlog(True, f"  Phonon TS verification skipped ({skip_reason}).")
        phonon = dict(
            frequencies=[], n_imaginary=0, n_zero_modes=0, is_transition_state=False,
            phonon_calculation_failed=False, eigenvector_tangent_alignment=None,
            verification_skipped=True,
        )
    else:
        phonon = verify_transition_state(
            ts_poscar, ts_dir, args.vasp_cmd, copy_dir,
            vasp_timeout=args.vasp_timeout, tangent=ci_tangent,
            zero_mode_tol_mev=args.zero_mode_tol,
        )

    result_path = work_dir / "yamatsumi_result.txt"
    with open(result_path, "w") as fh:
        fh.write("CI-NEB Calculation Result\n" + "=" * 50 + "\n\n")
        fh.write(f"Transition-state image index : {ci_index}\n")
        fh.write(f"Transition-state directory   : {ts_dir}\n\n")
        fh.write(f"Forward  activation barrier  : {barrier_fwd:.6f} eV\n")
        fh.write(f"Backward activation barrier  : {barrier_rev:.6f} eV\n")
        if rate is not None:
            fh.write(
                f"Arrhenius rate estimate      : {rate:.6e} 1/s  "
                f"(T={args.temperature:.1f} K, nu={args.attempt_freq:.3e} Hz, "
                "order-of-magnitude only, not rigorous TST)\n"
            )
        fh.write("\nEnergy profile along reaction path:\n")
        for idir, e in zip(image_dirs, energies):
            fh.write(f"  {idir.name}  E = {e:.6f} eV  dE = {e - e0:+.6f} eV\n")
        if phonon.get("verification_skipped", False):
            fh.write(f"\nTransition-state phonon verification: SKIPPED ({skip_reason})\n")
        else:
            fh.write("\nTransition-state phonon verification (IBRION=5):\n")
            fh.write(f"  Phonon calculation failed   : {phonon['phonon_calculation_failed']}\n")
            fh.write(f"  Imaginary frequencies found : {phonon['n_imaginary']}\n")
            fh.write(
                f"  Zero modes (|nu| <= {args.zero_mode_tol:g} meV): "
                f"{phonon.get('n_zero_modes', 0)}\n"
            )
            fh.write(f"  TS criterion satisfied      : {phonon['is_transition_state']}\n")
            alignment = phonon.get("eigenvector_tangent_alignment")
            if alignment is not None:
                fh.write(
                    f"  Eigenvector-tangent alignment: {alignment:.4f}  "
                    "(1.0 = imaginary mode exactly along the NEB reaction coordinate)\n"
                )
            if phonon["frequencies"]:
                fh.write("  All frequencies (meV):\n")
                for k, freq in enumerate(phonon["frequencies"]):
                    tag = (" [imaginary]" if freq < -args.zero_mode_tol
                           else " [zero mode]" if abs(freq) <= args.zero_mode_tol else "")
                    fh.write(f"    {k + 1:4d}  {freq:10.4f} meV{tag}\n")

    logger.info(f"  Result summary written: {result_path}")

    plot_path = work_dir / args.plot_out
    plot_energy_profile(
        image_dirs  = image_dirs,
        energies    = energies,
        ci_index    = ci_index,
        phonon      = phonon,
        barrier_fwd = barrier_fwd,
        barrier_rev = barrier_rev,
        output_path = plot_path,
        fmt         = args.plot_fmt,
        no_show     = args.no_show,
    )


def phase_neb(args) -> None:
    """CI-NEB iterative optimisation loop."""
    copy_dir = Path(args.copy_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    csv_path = work_dir / "convergence_history.csv"
    check_vasp_relaxation_disabled(copy_dir, allow_override=args.allow_vasp_relaxation)
    if args.rate_estimate == "arrhenius":
        # Fail fast, before the (potentially hours-long, VASP-subprocess-
        # dominated) NEB loop below runs at all: arrhenius_rate() itself
        # would reject these same values, but only inside finalize() after
        # convergence -- far too late to be a useful check.
        if args.temperature <= 0.0:
            raise NEBError(
                f"--temperature must be positive for --rate-estimate arrhenius "
                f"(got {args.temperature})."
            )
        if args.attempt_freq <= 0.0:
            raise NEBError(
                f"--attempt-freq must be positive for --rate-estimate arrhenius "
                f"(got {args.attempt_freq})."
            )

    comm = parallel.get_comm()
    root = parallel.is_root(comm)

    _rlog(root, "=" * 60)
    _rlog(root, "PHASE 2 - CI-NEB Optimisation Loop")
    _rlog(root, f"  Optimiser    : {args.optimizer.upper()}")
    _rlog(root, f"  CI activates : iteration {args.ci_start_iter}")
    _rlog(root, f"  Resume mode  : {args.resume}")
    if parallel.is_parallel(comm):
        _rlog(root, f"  MPI ranks    : {parallel.get_size(comm)}")
    _rlog(root, "=" * 60)

    check_no_legacy_image_dirs(work_dir, args.num_images)
    image_dirs = get_image_dirs(work_dir, args.num_images)
    for idir in image_dirs:
        if not idir.exists():
            raise NEBError(
                f"Image directory not found: {idir}\n"
                "Run Phase 1 first (--phase init or --phase all)."
            )

    n_atoms = read_poscar(str(image_dirs[0] / "POSCAR")).n_atoms
    n_inter = args.num_images
    n_total = len(image_dirs)

    if args.resume:
        start_iteration = _last_completed_iteration(csv_path) + 1
        _rlog(root, f"  Resuming from iteration {start_iteration} "
                    f"(last completed: {start_iteration - 1}).")
    else:
        _archive_csv(csv_path, root)
        start_iteration = 1

    # An empty iteration range would skip the loop body entirely, leaving
    # `energies` empty and crashing the trailing report_energy_profile() with
    # an IndexError. The realistic route here is exactly the case --resume
    # exists for: a previous run that used up all --max-iter iterations
    # without converging, resumed without also raising --max-iter. Say so
    # explicitly instead of failing obscurely (or, worse, silently doing
    # nothing at all).
    if start_iteration > args.max_iter:
        raise NEBError(
            f"Nothing to do: the run would start at iteration "
            f"{start_iteration} but --max-iter is {args.max_iter}."
            + (
                f"\nThe previous run already completed all {args.max_iter} "
                "iterations without converging. To continue it, re-run with a "
                "larger --max-iter (the CI-NEB path picks up from the POSCARs "
                "already on disk)."
                if args.resume else
                "\n--max-iter must be at least 1."
            )
        )

    opt_states = make_optimizer_states(n_inter, n_atoms, args)
    max_force  = float("inf")
    rms_force  = float("inf")
    prev_ci    = None
    energies: List[float] = []
    images_cache: Optional[List[Poscar]] = None

    for iteration in range(start_iteration, args.max_iter + 1):
        _rlog(root, f"\n{'-' * 60}")
        _rlog(root, f"Iteration {iteration} / {args.max_iter}  [{args.optimizer.upper()}]")
        _rlog(root, f"{'-' * 60}")

        # (a) Parse OUTCAR data
        energies, true_forces = _parse_iteration_data(image_dirs, n_atoms)

        # (b) Energy profile
        if iteration % args.log_interval == 0:
            _log_energy_profile(image_dirs, energies, n_total, root)

        # (c) Identify climbing image
        inter_e  = energies[1:-1]
        ci_local = inter_e.index(max(inter_e))
        ci_index = ci_local + 1

        ci_active    = (iteration >= args.ci_start_iter)
        effective_ci = ci_index if ci_active else -1

        if ci_active:
            _rlog(
                root,
                f"  Climbing image: [{image_dirs[ci_index].name}]  "
                f"E = {energies[ci_index] - energies[0]:+.6f} eV",
            )
        else:
            _rlog(root, f"  Warm-up (CI activates at iter {args.ci_start_iter}): plain NEB")

        if ci_active and prev_ci is not None and ci_index != prev_ci:
            for old in [prev_ci, ci_index]:
                opt_states[old - 1].full_reset()
            _rlog(
                root,
                f"  CI changed [{image_dirs[prev_ci].name}] -> "
                f"[{image_dirs[ci_index].name}]: optimiser states reset.",
            )
        if ci_active:
            prev_ci = ci_index

        # (d) Compute NEB forces. Under MPI, each rank only writes POSCARs
        # for the images it ran VASP for, so a fresh disk read is required
        # to see the complete, globally up-to-date set; in serial mode the
        # previous iteration's _update_and_run() already computed the full,
        # correct set in memory (force/tangent/position math is replicated
        # on every rank, so there is nothing MPI-specific in the values
        # themselves -- only the disk sync need differs), so re-parsing the
        # same POSCARs back from disk would be pure overhead.
        if images_cache is not None and not parallel.is_parallel(comm):
            images = images_cache
        else:
            images = [read_poscar(str(idir / "POSCAR")) for idir in image_dirs]
        neb_forces, tangents = compute_neb_forces(
            images, true_forces, energies, args.spring_const, effective_ci
        )

        # (e) Convergence check
        max_force, rms_force = _compute_convergence(
            neb_forces = neb_forces,
            image_dirs = image_dirs,
            ci_index   = ci_index,
            ci_active  = ci_active,
            opt_states = opt_states,
            optimizer  = args.optimizer,
            force_tol  = args.force_tol,
            verbose    = args.verbose,
            root       = root,
        )

        _append_convergence_history(
            csv_path, iteration, max_force, rms_force,
            ci_index, image_dirs, energies, ci_active, root,
        )

        if max_force < args.force_tol:
            _rlog(
                root,
                f"\n{'=' * 60}\n"
                f"  CONVERGENCE REACHED at iteration {iteration}\n"
                f"  max|F_NEB| = {max_force:.5f} eV/Ang\n"
                f"  RMS|F_NEB| = {rms_force:.5f} eV/Ang\n"
                f"{'=' * 60}",
            )
            if root:
                report_energy_profile(image_dirs, energies)
                finalize(
                    images, energies, ci_index, image_dirs, copy_dir, args,
                    ci_tangent=tangents[ci_index - 1],
                )
            return

        if args.dry_run:
            _rlog(root, "Dry-run mode: stopping after first convergence check.")
            return

        # (f) Update positions and re-run VASP
        updated_intermediates = _update_and_run(
            images           = images,
            neb_forces       = neb_forces,
            tangents         = tangents,
            ci_index         = effective_ci,
            opt_states       = opt_states,
            image_dirs       = image_dirs,
            copy_dir         = copy_dir,
            iteration        = iteration,
            n_inter          = n_inter,
            optimizer        = args.optimizer,
            step_size        = args.step_size,
            vasp_cmd         = args.vasp_cmd,
            comm             = comm,
            local_workers    = args.local_workers,
            sanitize_mpi_env = args.sanitize_mpi_env,
            vasp_timeout     = args.vasp_timeout,
            root             = root,
        )
        images_cache = [images[0], *updated_intermediates, images[-1]]

    _rlog(
        root,
        f"\n{'!' * 60}\n"
        f"  CI-NEB did NOT converge within {args.max_iter} iterations.\n"
        f"  Last  max|F_NEB| = {max_force:.5f} eV/Ang\n"
        f"  Last  RMS|F_NEB| = {rms_force:.5f} eV/Ang\n"
        f"{'!' * 60}",
        level=logging.WARNING,
    )
    if root:
        report_energy_profile(image_dirs, energies)
