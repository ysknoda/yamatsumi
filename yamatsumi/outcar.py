"""VASP OUTCAR parsers (energy, ionic forces, vibrational frequencies)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def parse_energy_from_outcar(outcar_path: Path) -> Optional[float]:
    """
    Return the last TOTEN value (eV) from an OUTCAR, or None if not found.
    Relevant line:  '  free  energy   TOTEN  =   -123.456789 eV'

    A line whose value field cannot be parsed (e.g. VASP's "********"
    field-overflow marker) is skipped rather than raising, leaving whatever
    the last successfully-parsed occurrence was.
    """
    energy = None
    try:
        with open(outcar_path) as fh:
            for line in fh:
                if "TOTEN" in line and "eV" in line:
                    parts = line.split("=")
                    if len(parts) >= 2:
                        try:
                            energy = float(parts[-1].split()[0])
                        except (ValueError, IndexError):
                            logger.warning(
                                f"Unparseable TOTEN line in {outcar_path}: {line.strip()!r}"
                            )
    except OSError:
        logger.warning(f"Cannot open: {outcar_path}")
    return energy


def parse_forces_from_outcar(
    outcar_path: Path,
    n_atoms:     int,
) -> Optional[np.ndarray]:
    """
    Return the last set of ionic forces (N x 3, eV/Ang) from an OUTCAR.
    Forces appear after 'TOTAL-FORCE (eV/Angst)', then a dashed line,
    then N lines of: '  x y z fx fy fz'
    Returns None if the block cannot be found.

    A block containing an unparseable field (e.g. VASP's "********"
    field-overflow marker) is treated the same as a truncated block: it is
    discarded rather than raising, so a still-valid earlier occurrence (or
    None, if there is none) is returned instead.
    """
    forces = None
    try:
        with open(outcar_path) as fh:
            lines = fh.readlines()
    except OSError:
        logger.warning(f"Cannot open: {outcar_path}")
        return None

    for i, line in enumerate(lines):
        if "TOTAL-FORCE (eV/Angst)" in line:
            start = i + 2
            block = []
            for j in range(start, start + n_atoms):
                if j >= len(lines):
                    break  # truncated file: fewer lines than n_atoms requires
                parts = lines[j].split()
                if len(parts) < 6:
                    break
                try:
                    block.append([float(parts[3]), float(parts[4]), float(parts[5])])
                except ValueError:
                    logger.warning(
                        f"Unparseable force field in {outcar_path} line {j + 1}: "
                        f"{lines[j].strip()!r}"
                    )
                    break
            if len(block) == n_atoms:
                forces = block
    return np.array(forces, dtype=float) if forces is not None else None


_EIGENVECTOR_HEADER = "Eigenvectors and eigenvalues of the dynamical matrix"


def parse_frequencies_from_outcar(outcar_path: Path) -> List[float]:
    """
    Parse vibrational frequencies from OUTCAR (IBRION=5/6).
    Returns list of floats in meV; imaginary ones stored as negative values.

    VASP encodes the imaginary/real distinction with a token
    ('f/i=' vs 'f  =') rather than a literal minus sign in the meV column.

    The meV value is the last whitespace-separated token on the line when
    it has no trailing unit label; some VASP versions instead print a
    trailing "meV" label after the number, in which case the value is the
    second-to-last token. Both are tried (last token first) so this parser
    is not silently defeated by that formatting difference.

    Only lines after the last "Eigenvectors and eigenvalues of the dynamical
    matrix" header are scanned, and only those whose first token is a bare
    integer (the mode index) are treated as mode-header lines. Without this
    anchor, any earlier POSITION/TOTAL-FORCE block row happening to start
    with a non-negative coordinate (e.g. "1.234000 0.300000 ...") also has
    >=3 whitespace-separated tokens and a leading digit character, and would
    otherwise be misparsed as a spurious frequency.
    """
    frequencies: List[float] = []
    try:
        with open(outcar_path) as fh:
            lines = fh.readlines()
    except OSError:
        logger.warning(f"Cannot open OUTCAR for frequency parsing: {outcar_path}")
        return frequencies

    anchor = None
    for i, line in enumerate(lines):
        if _EIGENVECTOR_HEADER in line:
            anchor = i  # last occurrence wins, consistent with the energy/force parsers
    if anchor is None:
        return frequencies

    for line in lines[anchor + 1:]:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        is_imag = "f/i" in parts[1]
        freq_mev = None
        for token in (parts[-1], parts[-2] if len(parts) >= 2 else None):
            if token is None:
                continue
            try:
                freq_mev = float(token)
                break
            except ValueError:
                continue
        if freq_mev is None:
            continue
        frequencies.append(-freq_mev if is_imag else freq_mev)
    return frequencies


def parse_eigenvectors_from_outcar(
    outcar_path: Path,
    n_atoms:     int,
) -> List[Tuple[float, np.ndarray]]:
    """
    Parse vibrational mode frequencies (meV, imaginary as negative -- same
    convention as parse_frequencies_from_outcar) together with their
    per-atom displacement eigenvectors from OUTCAR (IBRION=5/6).

    Each mode is printed as a header line (the same "N f = ..." / "N f/i=
    ..." format parse_frequencies_from_outcar reads), then a column-label
    line ("X Y Z dx dy dz"), then one line per atom giving its Cartesian
    position followed by its (dx, dy, dz) displacement for that mode.
    Returns a list of (frequency_mev, displacement) pairs, displacement
    shaped (n_atoms, 3). A mode whose data block is truncated or malformed
    is skipped rather than raising or aborting the scan for later modes.

    Kept independent of parse_frequencies_from_outcar (the mode-header
    detection logic is intentionally duplicated) since that function's
    signature is stable and used elsewhere, and this one additionally needs
    n_atoms and returns a different shape.
    """
    modes: List[Tuple[float, np.ndarray]] = []
    try:
        with open(outcar_path) as fh:
            lines = fh.readlines()
    except OSError:
        logger.warning(f"Cannot open OUTCAR for eigenvector parsing: {outcar_path}")
        return modes

    anchor = None
    for i, line in enumerate(lines):
        if _EIGENVECTOR_HEADER in line:
            anchor = i
    if anchor is None:
        return modes

    i = anchor + 1
    while i < len(lines):
        parts = lines[i].strip().split()
        if len(parts) < 3 or not parts[0].isdigit():
            i += 1
            continue

        is_imag = "f/i" in parts[1]
        freq_mev = None
        for token in (parts[-1], parts[-2] if len(parts) >= 2 else None):
            if token is None:
                continue
            try:
                freq_mev = float(token)
                break
            except ValueError:
                continue
        if freq_mev is None:
            i += 1
            continue

        data_start = i + 2  # skip this header line and the "X Y Z dx dy dz" label line
        rows: List[List[float]] = []
        for j in range(data_start, data_start + n_atoms):
            if j >= len(lines):
                break
            row_parts = lines[j].split()
            if len(row_parts) < 6:
                break
            try:
                rows.append([float(row_parts[3]), float(row_parts[4]), float(row_parts[5])])
            except ValueError:
                break

        if len(rows) == n_atoms:
            modes.append((-freq_mev if is_imag else freq_mev, np.array(rows, dtype=float)))
            i = data_start + n_atoms
        else:
            i += 1

    return modes
