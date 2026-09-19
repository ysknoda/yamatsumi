"""Phase 1: interpolate images, stage directories, run initial single-point VASP calcs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .. import parallel
from ..directories import (
    check_vasp_relaxation_disabled,
    get_image_dirs,
    prepare_image_directory,
)
from ..interpolate import (
    check_endpoint_consistency,
    idpp_interpolate_images,
    interpolate_images,
)
from ..poscar import Poscar, read_poscar, write_poscar
from ..vasp_runner import run_vasp_for_images

logger = logging.getLogger(__name__)


def phase_init(args) -> List[Poscar]:
    """
    Read endpoints, interpolate images (MIC), create directories, run
    initial single-point VASP calculations.
    """
    copy_dir = Path(args.copy_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    check_vasp_relaxation_disabled(copy_dir, allow_override=args.allow_vasp_relaxation)

    comm = parallel.get_comm()
    root = parallel.is_root(comm)

    if root:
        logger.info("=" * 60)
        logger.info("PHASE 1 - Initialisation")
        logger.info("=" * 60)
        logger.info(f"Reading initial structure : {args.init}")
        logger.info(f"Reading final   structure : {args.final}")

    init_poscar = read_poscar(args.init)
    final_poscar = read_poscar(args.final)
    if root:
        logger.info(
            f"  {init_poscar.n_atoms} atoms | "
            f"species {init_poscar.species} | counts {init_poscar.counts}"
        )

    check_endpoint_consistency(init_poscar, final_poscar)
    if root:
        logger.info("  Endpoint structures are consistent (species, counts, lattice).")

    if args.path_interpolation == "idpp":
        images = idpp_interpolate_images(init_poscar, final_poscar, args.num_images)
    else:
        images = interpolate_images(init_poscar, final_poscar, args.num_images)
    image_dirs = get_image_dirs(work_dir, args.num_images)
    n_total = len(image_dirs)

    # Stage only the directories this rank will also run VASP for below.
    # run_vasp_for_images() partitions this same `image_dirs` list with the
    # same distribute_indices() call, so the two partitions agree exactly --
    # mirroring what neb_phase._update_and_run() already does per iteration.
    #
    # Without this gating every rank writes every POSCAR and calls
    # prepare_image_directory() -- which deletes stale OUTCAR/CONTCAR -- on
    # every directory, with no barrier before VASP is launched. A rank that
    # falls even tens of milliseconds behind then truncates a POSCAR another
    # rank's VASP is already reading, or deletes an OUTCAR another rank's VASP
    # has just finished writing.
    #
    # In serial the partition degenerates to "every index" (size == 1), so
    # non-MPI behaviour is unchanged.
    rank = parallel.get_rank(comm)
    size = parallel.get_size(comm)
    if args.dry_run:
        # No VASP runs, so there is no work to partition. The root rank writes
        # the complete set -- a single writer, hence no race -- so that a dry
        # run still leaves a fully staged work directory to inspect.
        my_indices = set(range(n_total)) if root else set()
    else:
        my_indices = set(parallel.distribute_indices(n_total, rank, size))

    for idx, (img, idir) in enumerate(zip(images, image_dirs)):
        if idx in my_indices:
            prepare_image_directory(idir, copy_dir)
            write_poscar(str(idir / "POSCAR"), img)
        if root:
            label = (
                "initial endpoint"
                if idx == 0
                else "final endpoint"
                if idx == n_total - 1
                else f"intermediate image {idx}"
            )
            logger.info(f"  [{idir.name}] {label}")

    if args.dry_run:
        if root:
            logger.info("Dry-run mode: VASP not launched.")
        return images

    if root:
        logger.info("Running initial single-point VASP calculations ...")
    run_vasp_for_images(
        image_dirs,
        args.vasp_cmd,
        comm=comm,
        local_workers=args.local_workers,
        sanitize_mpi_env=args.sanitize_mpi_env,
        vasp_timeout=args.vasp_timeout,
    )

    if root:
        logger.info("Phase 1 complete.")
    return images
