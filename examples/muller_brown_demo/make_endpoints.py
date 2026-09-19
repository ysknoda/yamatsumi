#!/usr/bin/env python3
"""
Generate POSCAR_init / POSCAR_final / CopyDir for the Muller-Brown CI-NEB
demo (see README.md in this directory).

The two endpoints are minima #2 and #3 of the standard Muller-Bown surface
(a single clean saddle between them, rather than the full 3-minimum
chain), encoded as a single "atom" whose (x, y) Cartesian coordinate is the
Muller-Brown coordinate. See tests/fixtures/mock_vasp.py for the surface
definition and tests/test_mock_vasp_integration.py for where these
coordinates were derived.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running this script directly (``python examples/.../make_endpoints.py``)
# without first installing yamatsumi, by adding the repository root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yamatsumi.poscar import Poscar, write_poscar  # noqa: E402

LATTICE_SIZE = 20.0  # Ang; large relative to the surface's O(1) coordinate range

MIN2 = (0.623499, 0.028038)
MIN3 = (-0.050011, 0.466694)


def _poscar(x: float, y: float, comment: str) -> Poscar:
    lattice = np.eye(3) * LATTICE_SIZE
    positions = np.array([[x / LATTICE_SIZE, y / LATTICE_SIZE, 0.5]])
    return Poscar(
        comment=comment, scale=1.0, lattice=lattice, species=["H"], counts=[1],
        coord_type="Direct", positions=positions,
    )


def main() -> None:
    here = Path(__file__).parent
    write_poscar(str(here / "POSCAR_init"), _poscar(*MIN3, "Muller-Brown min3"))
    write_poscar(str(here / "POSCAR_final"), _poscar(*MIN2, "Muller-Brown min2"))

    copy_dir = here / "CopyDir"
    copy_dir.mkdir(exist_ok=True)
    (copy_dir / "INCAR").write_text("# placeholder -- ignored by the mock VASP executable\n")

    print(f"Wrote {here / 'POSCAR_init'}")
    print(f"Wrote {here / 'POSCAR_final'}")
    print(f"Prepared {copy_dir}/")


if __name__ == "__main__":
    main()
