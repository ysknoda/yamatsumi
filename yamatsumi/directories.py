"""Image-directory layout and VASP-input template staging."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from .errors import NEBError

logger = logging.getLogger(__name__)


def prepare_image_directory(image_dir: Path, copy_dir: Path) -> None:
    """
    Create *image_dir* and copy template VASP inputs from *copy_dir*.

    POSCAR files found in copy_dir are intentionally skipped so that a
    pre-existing or freshly written image POSCAR is never accidentally
    overwritten by the template. Any pre-existing OUTCAR/CONTCAR already in
    *image_dir* is removed here, before the new VASP inputs are staged, so a
    VASP run that crashes or is interrupted can never leave a stale result
    behind that a later step would silently mistake for a fresh one.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("OUTCAR", "CONTCAR"):
        stale_path = image_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
            logger.debug(f"  Removed stale {stale_name}: {stale_path}")
    if not copy_dir.exists():
        raise NEBError(
            f"Template directory not found: {copy_dir}\n"
            "Populate it with INCAR, KPOINTS, POTCAR, etc."
        )
    if not (copy_dir / "INCAR").is_file():
        # VASP itself refuses to start without one ("No INCAR or vaspin.h5
        # found", exit code 1, checked on 6.6.1), but only after every image
        # has been staged and only as a generic per-image failure.
        raise NEBError(
            f"{copy_dir} has no INCAR. Every image copies its VASP inputs from "
            "this directory; add the single-point INCAR (NSW = 0, IBRION = -1)."
        )
    for src in copy_dir.iterdir():
        if src.is_file():
            if src.name.upper() == "POSCAR":
                logger.debug(f"  Skipped POSCAR in CopyDir: {src}")
                continue
            shutil.copy2(src, image_dir / src.name)
            logger.debug(f"  Copied {src.name} -> {image_dir.name}/")


def _parse_incar_tag(lines: List[str], tag: str) -> Optional[str]:
    """Return the value string of the last assignment to *tag* (case
    insensitive, "!"/"#" inline comments stripped), or None if absent.

    VASP accepts several ';'-separated assignments on one line
    ("IBRION = 2; NSW = 100"), so each line is split on ';' first; reading
    only the first '=' of a line used to hide the later tags entirely."""
    value = None
    for raw in lines:
        stripped = raw.split("!")[0].split("#")[0]
        for fragment in stripped.split(";"):
            fragment = fragment.strip()
            if not fragment or "=" not in fragment:
                continue
            key, _, val = fragment.partition("=")
            if key.strip().upper() == tag:
                value = val.strip()
    return value


def _parse_int_tag(incar_path: Path, lines: List[str], tag: str, default: int) -> int:
    """Integer value of *tag*, *default* if absent. An unreadable value raises
    NEBError: this is a safety check, so it must fail closed rather than fall
    back to the harmless-looking default."""
    raw = _parse_incar_tag(lines, tag)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise NEBError(
            f"{incar_path}: cannot read {tag} = {raw!r} as an integer, so "
            "yamatsumi cannot confirm VASP is a single-point evaluator. Fix the "
            "value (NSW = 0, IBRION = -1), or pass --allow-vasp-relaxation if "
            "the setup is intentional."
        ) from None


def check_vasp_relaxation_disabled(copy_dir: Path, allow_override: bool = False) -> None:
    """
    Raise NEBError unless CopyDir/INCAR keeps VASP as a single-point
    evaluator: NSW=0 and IBRION=-1 (VASP's own defaults when a tag is
    absent).

    yamatsumi drives atom positions entirely from the Python side -- each
    VASP invocation is meant to evaluate energy/forces at the positions
    yamatsumi just wrote, not to relax them further. If CopyDir's INCAR
    instead allows VASP to relax ions internally (NSW>0 or IBRION>=0),
    OUTCAR's last ionic step reflects VASP's own relaxed geometry while
    yamatsumi still pairs that energy/forces with the pre-relaxation POSCAR
    positions it wrote -- a silent, hard-to-detect violation of this
    method's core assumption. Pass allow_override=True (wired to the CLI's
    --allow-vasp-relaxation) to skip this check for an intentionally
    non-standard setup.
    """
    if allow_override:
        return
    incar_path = copy_dir / "INCAR"
    if not incar_path.exists():
        return  # a missing CopyDir/INCAR is reported by prepare_image_directory
    with open(incar_path) as fh:
        lines = fh.readlines()

    nsw    = _parse_int_tag(incar_path, lines, "NSW", default=0)
    ibrion = _parse_int_tag(incar_path, lines, "IBRION", default=-1)

    if nsw != 0 or ibrion != -1:
        raise NEBError(
            f"{incar_path}: NSW={nsw}, IBRION={ibrion} allow VASP to relax "
            "ions internally, but yamatsumi drives atom positions "
            "externally via its own NEB optimiser. Running as-is would "
            "silently pair VASP's internally-relaxed energy/forces with the "
            "pre-relaxation POSCAR positions yamatsumi wrote. Set NSW=0 and "
            "IBRION=-1 in CopyDir/INCAR (single-point evaluation), or pass "
            "--allow-vasp-relaxation if this is intentional."
        )


def get_image_dirs(work_dir: Path, num_images: int) -> List[Path]:
    """
    Return all image Path objects:
      [ work_dir/00, work_dir/01, ..., work_dir/<num_images+1> ]
    """
    total = num_images + 2
    width = len(str(total - 1))
    return [work_dir / str(i).zfill(width) for i in range(total)]
