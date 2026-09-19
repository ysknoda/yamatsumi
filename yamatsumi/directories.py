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
    for src in copy_dir.iterdir():
        if src.is_file():
            if src.name.upper() == "POSCAR":
                logger.debug(f"  Skipped POSCAR in CopyDir: {src}")
                continue
            shutil.copy2(src, image_dir / src.name)
            logger.debug(f"  Copied {src.name} -> {image_dir.name}/")


def _parse_incar_tag(lines: List[str], tag: str) -> Optional[str]:
    """Return the value string of the last INCAR line assigning *tag* (case
    insensitive, "!"/"#" inline comments stripped), or None if absent."""
    value = None
    for raw in lines:
        stripped = raw.split("!")[0].split("#")[0].strip()
        if not stripped or "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        if key.strip().upper() == tag:
            value = val.strip()
    return value


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
        return  # missing CopyDir/INCAR is reported by prepare_image_directory instead
    with open(incar_path) as fh:
        lines = fh.readlines()

    nsw_str    = _parse_incar_tag(lines, "NSW")
    ibrion_str = _parse_incar_tag(lines, "IBRION")
    try:
        nsw = int(float(nsw_str)) if nsw_str is not None else 0
    except ValueError:
        nsw = 0
    try:
        ibrion = int(float(ibrion_str)) if ibrion_str is not None else -1
    except ValueError:
        ibrion = -1

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
