"""Stand-alone phase: re-generate the energy-profile plot from an existing NEB_run."""
from __future__ import annotations

import logging
from pathlib import Path

from .. import parallel
from ..directories import check_no_legacy_image_dirs, get_image_dirs
from ..errors import NEBError
from ..kinetics import compute_barriers
from ..outcar import parse_energy_from_outcar
from ..plotting import plot_energy_profile, report_energy_profile

logger = logging.getLogger(__name__)


def phase_plot(args) -> None:
    """Re-generate the energy-profile plot from an existing NEB_run."""
    comm = parallel.get_comm()
    root = parallel.is_root(comm)

    work_dir   = Path(args.work_dir).resolve()
    check_no_legacy_image_dirs(work_dir, args.num_images)
    image_dirs = get_image_dirs(work_dir, args.num_images)
    for idir in image_dirs:
        if not (idir / "OUTCAR").exists():
            raise NEBError(
                f"OUTCAR not found in {idir}.\n"
                "Run at least Phase 1 before using --phase plot."
            )
    energies = []
    for idir in image_dirs:
        e = parse_energy_from_outcar(idir / "OUTCAR")
        if e is None:
            raise NEBError(f"Cannot parse energy from {idir / 'OUTCAR'}.")
        energies.append(e)
    inter_e     = energies[1:-1]
    ci_index    = inter_e.index(max(inter_e)) + 1
    barrier_fwd, barrier_rev = compute_barriers(energies, ci_index)

    phonon = dict(
        frequencies=[], n_imaginary=0, is_transition_state=False,
        phonon_calculation_failed=False, verification_skipped=False,
    )
    result_file = work_dir / "yamatsumi_result.txt"
    if result_file.exists():
        with open(result_file) as fh:
            for line in fh:
                if "phonon verification: SKIPPED" in line:
                    phonon["verification_skipped"] = True
                if "Phonon calculation failed" in line:
                    phonon["phonon_calculation_failed"] = (
                        line.split(":")[-1].strip().lower() == "true"
                    )
                if "Imaginary frequencies found" in line:
                    try:
                        phonon["n_imaginary"] = int(line.split(":")[-1].strip())
                    except ValueError:
                        logger.warning(
                            f"Unparseable 'Imaginary frequencies found' line in "
                            f"{result_file}: {line.strip()!r}"
                        )
                if "TS criterion satisfied" in line:
                    phonon["is_transition_state"] = (
                        line.split(":")[-1].strip().lower() == "true"
                    )

    if not root:
        return
    report_energy_profile(image_dirs, energies)
    plot_energy_profile(
        image_dirs  = image_dirs,
        energies    = energies,
        ci_index    = ci_index,
        phonon      = phonon,
        barrier_fwd = barrier_fwd,
        barrier_rev = barrier_rev,
        output_path = work_dir / args.plot_out,
        fmt         = args.plot_fmt,
        no_show     = args.no_show,
    )
