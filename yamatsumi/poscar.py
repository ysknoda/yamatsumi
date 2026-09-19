"""
VASP POSCAR / CONTCAR I/O.

Bug fixes applied here relative to the original single-file script:

* Scale-factor normalisation. The POSCAR "universal scaling factor" was
  previously left baked into neither `lattice` nor consistently applied to
  Cartesian positions, so every downstream Cartesian-space computation
  implicitly (and silently) assumed scale == 1.0. `read_poscar` now resolves
  the true linear scale factor immediately (handling VASP's negative-scale
  "target volume" convention too), bakes it into `lattice`, and always
  returns `scale == 1.0` -- mirroring how `coord_type` is already always
  normalised to "Direct".
* Selective Dynamics (per-atom T/F constraint flags) are now parsed into
  `Poscar.selective_dynamics` and faithfully re-emitted by `write_poscar`,
  instead of being silently discarded.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from .errors import NEBError
from .geometry import cartesian_to_direct

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Poscar:
    """
    Typed representation of a VASP POSCAR / CONTCAR file.

    After read_poscar() all positions are guaranteed to be in fractional
    (Direct) coordinates, coord_type == "Direct", and scale == 1.0 (the
    scale factor is baked directly into `lattice` at parse time -- see
    module docstring).

    Attributes
    ----------
    comment            : str                    - header comment line
    scale              : float                   - always 1.0 after read_poscar()
    lattice            : np.ndarray (3, 3)        - lattice vectors (rows), true Ang
    species            : List[str]                - element symbols (empty for VASP4)
    counts             : List[int]                 - atoms per species
    coord_type         : str                        - always "Direct" after read_poscar()
    positions          : np.ndarray (N, 3)           - fractional coordinates
    selective_dynamics : Optional[np.ndarray (N, 3)]  - bool, True = free (unconstrained).
                          None when the file has no "Selective dynamics" block.
    """
    comment            : str
    scale              : float
    lattice            : np.ndarray
    species            : List[str]
    counts             : List[int]
    coord_type         : str
    positions          : np.ndarray
    selective_dynamics : Optional[np.ndarray] = None

    @property
    def n_atoms(self) -> int:
        """Total number of atoms."""
        return sum(self.counts)

    def copy(self) -> "Poscar":
        """Return a deep copy."""
        return copy.deepcopy(self)

    def with_positions(self, positions: np.ndarray) -> "Poscar":
        """
        Return a new Poscar with `positions` replaced; all other fields
        (lattice, species, counts, selective_dynamics, comment) are shared
        by reference rather than deep-copied.

        Safe because nothing in this codebase mutates those fields in place
        after construction -- only `.positions`/`.comment` are ever
        reassigned wholesale, never mutated element-wise. Used in the
        optimiser hot loop (optimizers.py) where `copy()`'s full
        `copy.deepcopy` would otherwise needlessly duplicate the
        run-invariant lattice/species/counts/selective_dynamics arrays on
        every single step.
        """
        return dataclasses.replace(self, positions=positions)

    @property
    def lattice_volume(self) -> float:
        """Unit-cell volume in Ang^3."""
        return abs(np.linalg.det(self.lattice)) * self.scale ** 3

    def __eq__(self, other: object) -> bool:
        """Structural equality check (comment is excluded)."""
        if not isinstance(other, Poscar):
            return NotImplemented
        tol = 1e-9
        if self.selective_dynamics is None or other.selective_dynamics is None:
            sd_equal = self.selective_dynamics is other.selective_dynamics
        else:
            sd_equal = bool(np.array_equal(self.selective_dynamics, other.selective_dynamics))
        return (
            abs(self.scale - other.scale) < tol
            and np.allclose(self.lattice, other.lattice, atol=tol)
            and self.species    == other.species
            and self.counts     == other.counts
            and self.coord_type == other.coord_type
            and np.allclose(self.positions, other.positions, atol=tol)
            and sd_equal
        )


def read_poscar(filepath: str) -> Poscar:
    """
    Parse a VASP POSCAR / CONTCAR file and return a Poscar dataclass.

    Cartesian-coordinate files are converted to fractional (Direct) before
    returning; coord_type is always "Direct" and scale is always 1.0 in the
    returned object (see module docstring for the scale-normalisation fix).
    Input validation raises NEBError with a descriptive message on malformed
    input.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"POSCAR not found: {filepath}")

    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = [ln.rstrip("\n") for ln in fh]
    except OSError as exc:
        raise NEBError(f"Cannot read {filepath}: {exc}") from exc

    if len(lines) < 8:
        raise NEBError(
            f"{filepath}: file too short ({len(lines)} lines); "
            "a valid POSCAR needs at least 8 lines."
        )

    idx = 0
    comment = lines[idx].strip(); idx += 1

    # --- Scale factor ---
    try:
        scale = float(lines[idx].split()[0]); idx += 1
    except (ValueError, IndexError):
        raise NEBError(f"{filepath} line {idx}: cannot parse scale factor.")
    if scale == 0.0:
        raise NEBError(f"{filepath}: scale factor is zero.")

    # --- Lattice vectors (raw, as literally printed -- not yet scaled) ---
    raw_lattice_rows: List[List[float]] = []
    for row in range(3):
        try:
            vec = [float(x) for x in lines[idx].split()[:3]]
            if len(vec) < 3:
                raise ValueError
            raw_lattice_rows.append(vec)
            idx += 1
        except (ValueError, IndexError):
            raise NEBError(
                f"{filepath} line {idx}: cannot parse lattice vector {row + 1}."
            )

    raw_lattice = np.array(raw_lattice_rows, dtype=float)
    raw_det = np.linalg.det(raw_lattice)
    if abs(raw_det) < 1e-10:
        raise NEBError(f"{filepath}: lattice matrix is singular (det ~ 0).")

    # Bug-B fix: resolve the true linear scale factor -- VASP's negative-scale
    # convention means |scale| is a target cell volume (Ang^3), not a direct
    # linear factor -- then bake it into the lattice and normalise scale to 1.0.
    if scale > 0.0:
        linear_scale = scale
    else:
        target_volume = abs(scale)
        linear_scale = (target_volume / abs(raw_det)) ** (1.0 / 3.0)
    physical_lattice = raw_lattice * linear_scale

    # --- Species / counts ---
    tok = lines[idx].split()
    if not tok:
        raise NEBError(f"{filepath} line {idx + 1}: empty species/counts line.")

    if tok[0].lstrip("-").replace(".", "", 1).isdigit():
        # VASP4: no species symbols
        species: List[str] = []
        try:
            counts: List[int] = [int(x) for x in tok]
        except ValueError:
            raise NEBError(f"{filepath} line {idx + 1}: cannot parse atom counts.")
        idx += 1
    else:
        species = tok; idx += 1
        try:
            counts = [int(x) for x in lines[idx].split()]
        except (ValueError, IndexError):
            raise NEBError(f"{filepath} line {idx + 1}: cannot parse atom counts.")
        if len(species) != len(counts):
            raise NEBError(
                f"{filepath}: {len(species)} species symbols but {len(counts)} "
                f"atom counts (species={species}, counts={counts})."
            )
        idx += 1

    if not counts or any(c < 0 for c in counts):
        raise NEBError(f"{filepath}: invalid atom counts: {counts}.")
    n_atoms = sum(counts)
    if n_atoms == 0:
        raise NEBError(f"{filepath}: total atom count is zero.")

    # --- Optional Selective Dynamics ---
    has_selective_dynamics = (
        idx < len(lines) and lines[idx].strip() != ""
        and lines[idx].strip()[0].lower() == "s"
    )
    if has_selective_dynamics:
        idx += 1

    # --- Coordinate type ---
    if idx >= len(lines):
        raise NEBError(f"{filepath}: missing coordinate-type line.")
    coord_type = lines[idx].strip(); idx += 1
    if not coord_type:
        raise NEBError(f"{filepath}: empty coordinate-type line.")

    # --- Positions (+ optional per-atom Selective Dynamics T/F flags) ---
    if idx + n_atoms > len(lines):
        raise NEBError(
            f"{filepath}: expected {n_atoms} position lines but only "
            f"{len(lines) - idx} remain."
        )
    position_rows: List[List[float]] = []
    sd_rows: List[List[bool]] = []
    for i in range(n_atoms):
        try:
            parts = lines[idx + i].split()
            pos = [float(parts[0]), float(parts[1]), float(parts[2])]
            position_rows.append(pos)
        except (ValueError, IndexError):
            raise NEBError(
                f"{filepath} line {idx + i + 1}: cannot parse position for atom {i + 1}."
            )
        if has_selective_dynamics:
            if len(parts) < 6:
                raise NEBError(
                    f"{filepath} line {idx + i + 1}: missing Selective Dynamics "
                    f"flags for atom {i + 1}."
                )
            sd_rows.append([tok.upper() == "T" for tok in parts[3:6]])

    # --- Convert Cartesian -> Direct if needed (scale already resolved above) ---
    ct_lower = coord_type.strip().lower()
    if ct_lower.startswith("c") or ct_lower.startswith("k"):
        raw_cart = np.array(position_rows, dtype=float)
        physical_cart = raw_cart * linear_scale
        positions = cartesian_to_direct(physical_cart, physical_lattice)
        logger.debug(f"{filepath}: Cartesian coordinates converted to Direct.")
    elif ct_lower.startswith("d"):
        positions = np.array(position_rows, dtype=float)
    else:
        raise NEBError(
            f"{filepath}: unrecognised coordinate-type line {coord_type!r} "
            "(expected 'Direct', 'Cartesian', or 'Karthesian')."
        )

    return Poscar(
        comment            = comment,
        scale              = 1.0,
        lattice            = physical_lattice,
        species            = species,
        counts             = counts,
        coord_type         = "Direct",
        positions          = positions,
        selective_dynamics = (np.array(sd_rows, dtype=bool) if has_selective_dynamics else None),
    )


def write_poscar(filepath: str, poscar: Poscar) -> None:
    """
    Write a Poscar to disk in VASP5 / Direct format.

    Always writes "Direct" as the coordinate type and scale = 1.0 (both are
    canonical after read_poscar()). Re-emits "Selective dynamics" and
    per-atom T/F flags when `poscar.selective_dynamics` is set. Parent
    directories are created automatically.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        poscar.comment,
        f"  {poscar.scale:.6f}",
    ]
    for vec in poscar.lattice:
        lines.append("  {:>20.16f}  {:>20.16f}  {:>20.16f}".format(*vec))
    if poscar.species:
        lines.append("  " + "  ".join(poscar.species))
    lines.append("  " + "  ".join(str(c) for c in poscar.counts))

    has_sd = poscar.selective_dynamics is not None
    if has_sd:
        lines.append("Selective dynamics")
    lines.append("Direct")

    for i, pos in enumerate(poscar.positions):
        line = "  {:>20.16f}  {:>20.16f}  {:>20.16f}".format(*pos)
        if has_sd:
            flags = poscar.selective_dynamics[i]
            line += "  " + "  ".join("T" if f else "F" for f in flags)
        lines.append(line)

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.debug(f"POSCAR written: {path}")
