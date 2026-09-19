"""Command-line interface: argument parser and program entry point."""
from __future__ import annotations

import argparse
import logging
import sys

from . import parallel
from .constants import (
    DEFAULT_ATTEMPT_FREQ_HZ,
    DEFAULT_CI_START_ITER,
    DEFAULT_COPY_DIR,
    DEFAULT_FINAL_POSCAR,
    DEFAULT_FIRE_ALPHA0,
    DEFAULT_FIRE_DT0,
    DEFAULT_FIRE_DT_MAX,
    DEFAULT_FIRE_F_ALPHA,
    DEFAULT_FIRE_F_DEC,
    DEFAULT_FIRE_F_INC,
    DEFAULT_FIRE_N_MIN,
    DEFAULT_FORCE_TOL,
    DEFAULT_INIT_POSCAR,
    DEFAULT_LOCAL_WORKERS,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_MAX_ITER,
    DEFAULT_NUM_IMAGES,
    DEFAULT_OPTIMIZER,
    DEFAULT_PATH_INTERPOLATION,
    DEFAULT_PLOT_FMT,
    DEFAULT_PLOT_OUT,
    DEFAULT_RATE_ESTIMATE,
    DEFAULT_SPRING_CONST,
    DEFAULT_STEP_SIZE,
    DEFAULT_TEMPERATURE_K,
    DEFAULT_VASP_CMD,
    DEFAULT_WORK_DIR,
)
from .errors import NEBError
from .phases.init_phase import phase_init
from .phases.neb_phase import phase_neb
from .phases.plot_phase import phase_plot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    _EPILOG = """
Examples
--------
  # Full pipeline, 7 images, 4 MPI ranks (per-image VASP), Quick-Min optimiser:
  python -m yamatsumi --phase all --num-images 7 \\
                   --vasp-cmd "mpirun -np 4 vasp_std" --optimizer quickmin

  # Full pipeline with FIRE and a 3-iteration NEB warm-up:
  python -m yamatsumi --phase all --optimizer fire --ci-start-iter 4

  # Resume an interrupted NEB run (keeps existing convergence_history.csv):
  python -m yamatsumi --phase neb --resume

  # Log full energy profile every 5 iterations to reduce output:
  python -m yamatsumi --phase neb --log-interval 5

  # Re-plot the energy profile in SVG format:
  python -m yamatsumi --phase plot --plot-fmt svg --plot-out mep.svg

  # Non-default structure filenames:
  python -m yamatsumi --init IS/POSCAR --final FS/POSCAR --work-dir run01

  # Parallelise NEB images across a local thread pool (single workstation,
  # no MPI needed):
  python -m yamatsumi --phase all --local-workers 4

  # Parallelise NEB images across MPI ranks (one rank per image):
  mpirun -n 7 python -m yamatsumi --phase all --num-images 5

  # Better initial-path guess for multi-atom systems (IDPP) plus an
  # order-of-magnitude Arrhenius rate estimate at 500 K:
  python -m yamatsumi --phase all --path-interpolation idpp \\
                   --rate-estimate arrhenius --temperature 500

Notes
-----
  * CopyDir must contain INCAR, KPOINTS, and POTCAR (any POSCAR is ignored).
  * Endpoint images (00 and the last) are fixed after Phase 1.
  * Phonon verification (IBRION=5) runs automatically after convergence.
  * convergence_history.csv is written each iteration; archived on new runs.
  * Energy-profile plotting requires matplotlib  (pip install matplotlib).
  * For HPC clusters: --vasp-cmd "mpirun -np 16 vasp_std"
  * Launching this program itself under mpirun (e.g. "mpirun -n 7 python -m
    yamatsumi ...") distributes NEB images across ranks; --local-workers gives
    the same kind of speedup on a single machine without mpirun.
"""
    p = argparse.ArgumentParser(
        prog="yamatsumi",
        description=(
            "CI-NEB (Climbing Image Nudged Elastic Band) transition-state "
            "search using VASP.\n\n"
            "Phases:\n"
            "  init  -- interpolate images and run initial SP calculations\n"
            "  neb   -- CI-NEB geometry optimisation loop\n"
            "  all   -- init then neb  [default]\n"
            "  plot  -- re-plot the energy profile from an existing run\n\n"
            "Run with -h for the full option reference."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--phase", choices=["init", "neb", "all", "plot"], default="all",
        metavar="PHASE",
        help="Workflow phase: init | neb | all (default) | plot",
    )

    struct = p.add_argument_group("Input structures")
    struct.add_argument(
        "--init", default=DEFAULT_INIT_POSCAR, metavar="FILE",
        help=f"Initial-state POSCAR  (default: {DEFAULT_INIT_POSCAR})",
    )
    struct.add_argument(
        "--final", default=DEFAULT_FINAL_POSCAR, metavar="FILE",
        help=f"Final-state POSCAR    (default: {DEFAULT_FINAL_POSCAR})",
    )

    neb = p.add_argument_group("CI-NEB algorithm parameters")
    neb.add_argument(
        "--num-images", type=int, default=DEFAULT_NUM_IMAGES, metavar="N",
        help=f"Intermediate images excluding endpoints  (default: {DEFAULT_NUM_IMAGES})",
    )
    neb.add_argument(
        "--path-interpolation", default=DEFAULT_PATH_INTERPOLATION, metavar="MODE",
        choices=["linear", "idpp"],
        help=(
            "Initial-path generation (Phase 1): linear (plain MIC "
            "interpolation) | idpp (Image Dependent Pair Potential; tends "
            f"to avoid atom pass-through on multi-atom paths)  "
            f"(default: {DEFAULT_PATH_INTERPOLATION})"
        ),
    )
    neb.add_argument(
        "--spring-const", type=float, default=DEFAULT_SPRING_CONST, metavar="K",
        help=f"NEB spring constant k [eV/Ang]  (default: {DEFAULT_SPRING_CONST})",
    )
    neb.add_argument(
        "--force-tol", type=float, default=DEFAULT_FORCE_TOL, metavar="TOL",
        help=f"Convergence threshold: max|F_NEB| [eV/Ang]  (default: {DEFAULT_FORCE_TOL})",
    )
    neb.add_argument(
        "--step-size", type=float, default=DEFAULT_STEP_SIZE, metavar="DX",
        help=f"Max Cartesian displacement per step [Ang]  (default: {DEFAULT_STEP_SIZE})",
    )
    neb.add_argument(
        "--max-iter", type=int, default=DEFAULT_MAX_ITER, metavar="N",
        help=f"Maximum CI-NEB iterations  (default: {DEFAULT_MAX_ITER})",
    )
    neb.add_argument(
        "--ci-start-iter", type=int, default=DEFAULT_CI_START_ITER, metavar="N",
        help=(
            f"Iteration at which CI is activated; earlier iters run as plain "
            f"NEB (warm-up)  (default: {DEFAULT_CI_START_ITER})"
        ),
    )
    neb.add_argument(
        "--optimizer", default=DEFAULT_OPTIMIZER, metavar="ALG",
        choices=["sd", "quickmin", "fire"],
        help=(
            "Optimiser: sd (robust/slow) | quickmin (~2-4x faster) | "
            f"fire (~3-6x faster)  (default: {DEFAULT_OPTIMIZER})"
        ),
    )

    fire = p.add_argument_group(
        "FIRE hyper-parameters",
        description="Stored in OptimizerState. Defaults suit most runs.",
    )
    fire.add_argument("--fire-dt0",     type=float, default=DEFAULT_FIRE_DT0,
                       metavar="DT", help=f"Initial dt  (default: {DEFAULT_FIRE_DT0})")
    fire.add_argument("--fire-dt-max",  type=float, default=DEFAULT_FIRE_DT_MAX,
                       metavar="DT", help=f"Maximum dt  (default: {DEFAULT_FIRE_DT_MAX})")
    fire.add_argument("--fire-n-min",   type=int,   default=DEFAULT_FIRE_N_MIN,
                       metavar="N",  help=f"Min positive-power steps before dt increase  (default: {DEFAULT_FIRE_N_MIN})")
    fire.add_argument("--fire-f-inc",   type=float, default=DEFAULT_FIRE_F_INC,
                       metavar="F",  help=f"dt increase factor  (default: {DEFAULT_FIRE_F_INC})")
    fire.add_argument("--fire-f-dec",   type=float, default=DEFAULT_FIRE_F_DEC,
                       metavar="F",  help=f"dt decrease factor  (default: {DEFAULT_FIRE_F_DEC})")
    fire.add_argument("--fire-alpha0",  type=float, default=DEFAULT_FIRE_ALPHA0,
                       metavar="A",  help=f"Initial alpha  (default: {DEFAULT_FIRE_ALPHA0})")
    fire.add_argument("--fire-f-alpha", type=float, default=DEFAULT_FIRE_F_ALPHA,
                       metavar="F",  help=f"Alpha damping factor  (default: {DEFAULT_FIRE_F_ALPHA})")

    paths = p.add_argument_group("File and directory paths")
    paths.add_argument(
        "--copy-dir", default=DEFAULT_COPY_DIR, metavar="DIR",
        help=f"Template VASP inputs (POSCAR ignored)  (default: {DEFAULT_COPY_DIR})",
    )
    paths.add_argument(
        "--work-dir", default=DEFAULT_WORK_DIR, metavar="DIR",
        help=f"Top-level working directory  (default: {DEFAULT_WORK_DIR})",
    )
    paths.add_argument(
        "--vasp-cmd", default=DEFAULT_VASP_CMD, metavar="CMD",
        help=f"VASP executable / MPI command  (default: {DEFAULT_VASP_CMD})",
    )
    paths.add_argument(
        "--vasp-timeout", type=float, default=None, metavar="SEC",
        help=(
            "Kill and treat as failed any single VASP invocation running "
            "longer than SEC seconds  (default: no timeout)"
        ),
    )

    parallel_g = p.add_argument_group(
        "Parallelism",
        description=(
            "Priority: MPI (when this program itself is launched via mpirun "
            "with more than one rank) > --local-workers thread pool > serial."
        ),
    )
    parallel_g.add_argument(
        "--local-workers", type=int, default=DEFAULT_LOCAL_WORKERS, metavar="N",
        help=(
            "Run up to N image VASP calculations concurrently in a local "
            f"thread pool (single workstation, no MPI needed)  (default: {DEFAULT_LOCAL_WORKERS})"
        ),
    )
    parallel_g.add_argument(
        "--sanitize-mpi-env", action="store_true",
        help=(
            "Strip OMPI_*/PMI_*/PMIX_*/HYDRA_* environment variables before "
            "launching --vasp-cmd. Only useful if --vasp-cmd itself invokes "
            "mpirun/mpiexec from inside an already MPI-parallel yamatsumi run "
            "(nested MPI); leave off otherwise."
        ),
    )

    rate_g = p.add_argument_group(
        "Reaction-rate estimate (optional)",
        description=(
            "Arrhenius is a crude order-of-magnitude estimate from the "
            "electronic barrier and an assumed attempt frequency alone -- "
            "not a rigorous transition-state-theory rate."
        ),
    )
    rate_g.add_argument(
        "--rate-estimate", default=DEFAULT_RATE_ESTIMATE, metavar="MODE",
        choices=["none", "arrhenius"],
        help=f"Rate-constant estimate to report: none | arrhenius  (default: {DEFAULT_RATE_ESTIMATE})",
    )
    rate_g.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE_K, metavar="K",
        help=f"Temperature for --rate-estimate  (default: {DEFAULT_TEMPERATURE_K})",
    )
    rate_g.add_argument(
        "--attempt-freq", type=float, default=DEFAULT_ATTEMPT_FREQ_HZ, metavar="HZ",
        help=f"Attempt frequency (nu) for --rate-estimate arrhenius  (default: {DEFAULT_ATTEMPT_FREQ_HZ:.1e})",
    )

    plot_g = p.add_argument_group("Energy-profile plot (requires matplotlib)")
    plot_g.add_argument(
        "--plot-out", default=DEFAULT_PLOT_OUT, metavar="FILE",
        help=f"Output figure filename  (default: {DEFAULT_PLOT_OUT})",
    )
    plot_g.add_argument(
        "--plot-fmt", default=DEFAULT_PLOT_FMT, metavar="FMT",
        choices=["png", "svg", "pdf", "eps"],
        help="Figure format: png | svg | pdf | eps  (default: png)",
    )
    plot_g.add_argument(
        "--no-show", action="store_true",
        help="Save plot but do NOT open an interactive window (HPC).",
    )

    misc = p.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume an interrupted --phase neb run.  "
            "Skips archiving convergence_history.csv and reuses existing "
            "OUTCAR data from the last completed iteration."
        ),
    )
    misc.add_argument(
        "--log-interval", type=int, default=DEFAULT_LOG_INTERVAL, metavar="N",
        help=(
            f"Print the full energy profile every N iterations  "
            f"(default: {DEFAULT_LOG_INTERVAL}; 1 = every iteration)"
        ),
    )
    misc.add_argument(
        "--dry-run", action="store_true",
        help="Prepare files but do NOT launch VASP.",
    )
    misc.add_argument(
        "--allow-vasp-relaxation", action="store_true",
        help=(
            "Skip the startup check that CopyDir/INCAR keeps VASP as a "
            "single-point evaluator (NSW=0, IBRION=-1). Only pass this if "
            "you intentionally want VASP to relax ions internally; "
            "yamatsumi otherwise still pairs its result with the "
            "pre-relaxation POSCAR positions it wrote, silently."
        ),
    )
    misc.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return p


def validate_args(args) -> None:
    """
    Reject numeric CLI values that argparse's type= cannot catch but that
    either crash outright or silently produce meaningless physics.

    argparse only enforces that these parse as int/float, so every check
    below corresponds to a value that is syntactically fine yet has no valid
    interpretation:

    * ``--log-interval 0`` reaches ``iteration % 0`` -> ZeroDivisionError.
    * ``--vasp-timeout 0`` makes every VASP invocation time out instantly.
    * ``--step-size`` of 0 never moves an image (so the run can never
      converge); a negative value steps *along* the force's uphill direction
      and diverges.
    * ``--spring-const`` below 0 is an anti-restoring spring: images repel
      along the path instead of staying evenly spaced. (Exactly 0 is a
      legitimate "no spring" NEB variant and is allowed.)
    * ``--force-tol`` of 0 or less can never be satisfied by a norm.
    * ``--num-images`` below 1 leaves no intermediate image to optimise.

    ``--temperature`` / ``--attempt-freq`` are deliberately not checked here:
    they are consumed only by phase_neb, which validates them itself before
    starting its (expensive) optimisation loop.
    """
    positive_int   = [("--num-images", args.num_images),
                      ("--max-iter", args.max_iter),
                      ("--ci-start-iter", args.ci_start_iter),
                      ("--log-interval", args.log_interval),
                      ("--local-workers", args.local_workers)]
    positive_float = [("--step-size", args.step_size),
                      ("--force-tol", args.force_tol)]

    for flag, value in positive_int:
        if value < 1:
            raise NEBError(f"{flag} must be at least 1 (got {value}).")
    for flag, value in positive_float:
        if value <= 0.0:
            raise NEBError(f"{flag} must be positive (got {value}).")
    if args.spring_const < 0.0:
        raise NEBError(
            f"--spring-const must not be negative (got {args.spring_const}); "
            "a negative spring constant pushes images apart along the path "
            "instead of keeping them evenly spaced."
        )
    if args.vasp_timeout is not None and args.vasp_timeout <= 0.0:
        raise NEBError(
            f"--vasp-timeout must be positive (got {args.vasp_timeout}); "
            "omit it entirely for the default unbounded wait."
        )


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    comm = parallel.get_comm()
    try:
        validate_args(args)
        if args.phase in ("init", "all"):
            phase_init(args)
        if args.phase in ("neb", "all"):
            phase_neb(args)
        if args.phase == "plot":
            phase_plot(args)
    except NEBError as exc:
        logger.error(f"NEBError: {exc}")
        if parallel.is_parallel(comm):
            comm.Abort(1)
        sys.exit(1)
    except Exception:
        # Last-resort safety net: under MPI, an unanticipated exception on
        # one rank must not leave the others blocked forever at the next
        # collective call, so tear down the whole MPI job instead of just
        # this process.
        logger.exception("Unhandled exception in yamatsumi")
        if parallel.is_parallel(comm):
            comm.Abort(1)
        raise


if __name__ == "__main__":
    main()
