"""Transition-state verification via finite-difference phonon calculation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .constraints import apply_mask, build_mask
from .directories import prepare_image_directory
from .outcar import parse_eigenvectors_from_outcar, parse_frequencies_from_outcar
from .poscar import Poscar, write_poscar
from .vasp_runner import run_vasp

logger = logging.getLogger(__name__)


def verify_transition_state(
    poscar       : Poscar,
    image_dir    : Path,
    vasp_cmd     : str,
    copy_dir     : Path,
    vasp_timeout : Optional[float] = None,
    tangent      : Optional[np.ndarray] = None,
) -> dict:
    """
    Verify the TS criterion by running VASP finite-difference phonons (IBRION=5).
    Returns dict with keys: frequencies, n_imaginary, is_transition_state,
    phonon_calculation_failed, eigenvector_tangent_alignment.

    `phonon_calculation_failed` distinguishes "the phonon VASP subprocess
    itself failed" from "it ran fine and genuinely found the wrong number of
    imaginary modes" -- without it, both cases return the same
    n_imaginary=0/is_transition_state=False shape and a caller cannot tell a
    crashed, unverified calculation from a clean, verified non-TS result.

    When exactly one imaginary mode is found and *tangent* (the flat 3N
    Cartesian CI-NEB tangent at this image, as produced by
    neb_forces.compute_neb_forces) is supplied, `eigenvector_tangent_alignment`
    is set to |unit(eigenvector) . unit(tangent)| (both restricted to free
    degrees of freedom) -- a value near 1.0 means the saddle's unstable
    direction actually coincides with the intended NEB reaction coordinate;
    a low value despite a single imaginary mode flags a saddle whose
    instability points somewhere else. Left None when not computable
    (no/mismatched tangent, or the eigenvector could not be matched to the
    imaginary mode's frequency).
    """
    phonon_dir = image_dir / "phonon"
    prepare_image_directory(phonon_dir, copy_dir)
    write_poscar(str(phonon_dir / "POSCAR"), poscar)

    incar_path  = phonon_dir / "INCAR"
    remove_tags = {"IBRION", "NSW", "POTIM", "NFREE", "EDIFF", "ICHAIN", "IMAGES"}
    if incar_path.exists():
        with open(incar_path) as fh:
            incar_lines = fh.readlines()
    else:
        incar_lines = []
    filtered = [
        ln for ln in incar_lines
        if not any(ln.strip().upper().startswith(t) for t in remove_tags)
    ]
    filtered += [
        "\n# --- Finite-difference phonon (appended by yamatsumi) ---\n",
        "IBRION = 5\n", "NSW    = 1\n", "NFREE  = 2\n",
        "POTIM  = 0.015\n", "EDIFF  = 1E-8\n",
    ]
    with open(incar_path, "w") as fh:
        fh.writelines(filtered)

    logger.info(f"  Running phonon calculation for TS verification: {phonon_dir}")
    if not run_vasp(phonon_dir, vasp_cmd, timeout=vasp_timeout):
        logger.warning("  Phonon run failed - TS criterion cannot be verified.")
        return dict(
            frequencies=[], n_imaginary=0, is_transition_state=False,
            phonon_calculation_failed=True, eigenvector_tangent_alignment=None,
        )

    frequencies = parse_frequencies_from_outcar(phonon_dir / "OUTCAR")
    n_imaginary = sum(1 for f in frequencies if f < 0)
    is_ts       = (n_imaginary == 1)

    logger.info(
        "  Phonon frequencies (meV): "
        + "  ".join(f"{f:+.3f}" for f in frequencies)
    )
    logger.info(f"  Imaginary frequencies: {n_imaginary}")
    if is_ts:
        logger.info("  TS criterion SATISFIED: exactly one imaginary frequency.")
    else:
        logger.warning(
            f"  TS criterion NOT satisfied "
            f"(expected 1 imaginary, found {n_imaginary})."
        )

    alignment = None
    if is_ts and tangent is not None:
        imaginary_freq = next(f for f in frequencies if f < 0)
        modes = parse_eigenvectors_from_outcar(phonon_dir / "OUTCAR", poscar.n_atoms)
        matching = [evec for freq, evec in modes if freq == imaginary_freq]
        if matching:
            eigvec = matching[0]
            tan    = np.asarray(tangent, dtype=float).reshape(eigvec.shape)
            mask   = build_mask(poscar.selective_dynamics, eigvec.shape)
            eigvec = apply_mask(eigvec, mask).ravel()
            tan    = apply_mask(tan, mask).ravel()
            eig_norm = np.linalg.norm(eigvec)
            tan_norm = np.linalg.norm(tan)
            if eig_norm > 1e-12 and tan_norm > 1e-12:
                alignment = float(abs(np.dot(eigvec, tan)) / (eig_norm * tan_norm))
                logger.info(f"  Eigenvector-tangent alignment: {alignment:.4f}  (1.0 = perfectly aligned)")
        else:
            logger.warning(
                "  Could not locate the imaginary mode's eigenvector for the "
                "tangent-alignment diagnostic (frequency/eigenvector parse mismatch)."
            )

    return dict(
        frequencies                    = frequencies,
        n_imaginary                    = n_imaginary,
        is_transition_state            = is_ts,
        phonon_calculation_failed      = False,
        eigenvector_tangent_alignment  = alignment,
    )
