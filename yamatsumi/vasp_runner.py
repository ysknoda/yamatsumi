"""
VASP subprocess launcher, with optional MPI- or thread-pool-based
parallelism across NEB image directories.

The real wall-clock bottleneck of this workflow is the external VASP
subprocess itself (minutes-hours per image), not the in-memory NEB
numerics. Each image's VASP run is independent given that iteration's
positions, so `run_vasp_for_images` distributes them across MPI ranks
(when launched under mpirun with more than one rank) or a local thread
pool (single-workstation case; `subprocess.run` releases the GIL while
VASP runs, so threads deliver real wall-clock speedup with no new
dependency), falling back to fully serial execution otherwise.
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from . import parallel
from .errors import NEBError

logger = logging.getLogger(__name__)

# Environment-variable prefixes that can confuse a nested `mpirun` launched
# from inside an already-MPI-parallel driver process. SLURM_* is
# deliberately NOT included: legitimate srun-based --vasp-cmd setups need it.
_MPI_ENV_PREFIXES = ("OMPI_", "PMI_", "PMIX_", "HYDRA_")

# A deliberately short list of well-known VASP fatal-error banners that can
# appear in stdout/stderr while VASP itself still exits with code 0 (e.g. a
# fatal numerical error inside a loop VASP does not treat as a hard abort).
# Exit-code-only success detection would silently accept such a run.
_VASP_FATAL_ERROR_MARKERS = (
    "ZBRENT: fatal error",
    "VERY BAD NEWS! internal error",
    "Error EDDDAV: Call to ZHEGV failed",
)


def run_vasp(
    image_dir        : Path,
    vasp_cmd         : str,
    sanitize_mpi_env : bool = False,
    timeout          : Optional[float] = None,
) -> bool:
    """
    Run VASP inside *image_dir*; stdout/stderr redirected to vasp_run.log.
    Returns True on success, False on failure (does NOT raise).

    Success requires both a zero exit code AND the absence of any known
    fatal-error banner in the captured log (see _VASP_FATAL_ERROR_MARKERS).
    *timeout* (seconds) bounds the subprocess; a hung VASP process is killed
    and treated as a failure instead of blocking forever. `timeout=None`
    (the default) preserves the previous unbounded-wait behaviour.
    """
    log_file = image_dir / "vasp_run.log"
    logger.info(f"  Launching VASP: {image_dir.name}")

    env = None
    if sanitize_mpi_env:
        env = {k: v for k, v in os.environ.items() if not k.startswith(_MPI_ENV_PREFIXES)}

    try:
        argv = shlex.split(vasp_cmd)
        with open(log_file, "w") as lf:
            result = subprocess.run(
                argv,
                cwd=str(image_dir),
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=True,
                env=env,
                timeout=timeout,
            )
    except subprocess.CalledProcessError as exc:
        logger.error(f"  VASP error (rc={exc.returncode}) in {image_dir}. See {log_file}")
        return False
    except FileNotFoundError:
        logger.error(f"  VASP executable not found: '{shlex.split(vasp_cmd)[0]}'")
        return False
    except subprocess.TimeoutExpired:
        logger.error(
            f"  VASP timed out after {timeout}s in {image_dir} and was killed. "
            f"See {log_file}"
        )
        return False

    log_text = log_file.read_text(errors="replace")
    for marker in _VASP_FATAL_ERROR_MARKERS:
        if marker in log_text:
            logger.error(
                f"  VASP reported a fatal error despite exit code 0 in "
                f"{image_dir}: {marker!r}. See {log_file}"
            )
            return False

    logger.info(f"  Done (rc={result.returncode}): {image_dir.name}")
    return True


def run_vasp_for_images(
    image_dirs       : List[Path],
    vasp_cmd         : str,
    comm             = None,
    local_workers    : int = 1,
    sanitize_mpi_env : bool = False,
    vasp_timeout     : Optional[float] = None,
) -> None:
    """
    Run VASP for every directory in *image_dirs*.

    Dispatch priority: MPI (when `comm` represents more than one rank) >
    local thread pool (when `local_workers > 1`) > fully serial.

    Raises RuntimeError if any assigned run fails. Under MPI, failures are
    combined with `allreduce` (logical AND) before raising, so every rank
    learns of and raises on a failure together -- a rank that fails locally
    never leaves the others blocked waiting at the next collective call.
    """
    n_total = len(image_dirs)

    if parallel.is_parallel(comm):
        rank = parallel.get_rank(comm)
        size = parallel.get_size(comm)
        my_indices = parallel.distribute_indices(n_total, rank, size)

        local_ok = True
        for i in my_indices:
            local_ok = run_vasp(
                image_dirs[i], vasp_cmd, sanitize_mpi_env, timeout=vasp_timeout
            ) and local_ok

        all_ok = parallel.allreduce_all_ok(comm, local_ok)
        if not all_ok:
            raise NEBError(
                "One or more VASP runs failed under MPI parallel dispatch "
                "(see per-rank vasp_run.log files)."
            )
        return

    if local_workers > 1 and n_total > 1:
        with ThreadPoolExecutor(max_workers=local_workers) as pool:
            futures = {
                pool.submit(
                    run_vasp, image_dirs[i], vasp_cmd, sanitize_mpi_env, vasp_timeout
                ): i
                for i in range(n_total)
            }
            results = {futures[f]: f.result() for f in futures}
        if not all(results.values()):
            failed = [image_dirs[i].name for i, ok in results.items() if not ok]
            raise NEBError(f"VASP failed for image(s): {', '.join(failed)}")
        return

    for idir in image_dirs:
        if not run_vasp(idir, vasp_cmd, sanitize_mpi_env, timeout=vasp_timeout):
            raise NEBError(f"VASP failed for image: {idir.name}")
