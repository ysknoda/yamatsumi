"""
Tests for examples/si7_vacancy_vasp/run_validation.py that need no VASP.

Criterion A2 must judge only the OUTCARs of the run being evaluated: archived
earlier attempts or diagnostic runs kept under the same work directory must
not decide it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).parent.parent / "examples" / "si7_vacancy_vasp" / "run_validation.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("si7_run_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a2_ignores_outcars_outside_the_current_run(tmp_path):
    script = _load_script()
    for rel in (
        "relax_init",
        "neb/00",
        "neb/03/phonon",
        "attempt2_metastable_endpoints/neb/03",
        "diag/c3v",
    ):
        d = tmp_path / rel
        d.mkdir(parents=True)
        (d / "OUTCAR").write_text("")
    found = sorted(
        str(p.relative_to(tmp_path)) for p in script.current_run_outcars(tmp_path)
    )
    assert found == ["neb/00/OUTCAR", "neb/03/phonon/OUTCAR", "relax_init/OUTCAR"]


def _staged_images(tmp_path, script):
    """A straight on-axis path like the one Phase 1 interpolates."""
    import numpy as np

    from yamatsumi.poscar import Poscar, write_poscar

    lattice = np.eye(3) * 5.431
    for i in range(script.NUM_IMAGES + 2):
        pos = np.array([[0.25, 0.25, 0.25], [0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        pos[0] += 0.1 * i
        d = tmp_path / f"{i:02d}"
        d.mkdir()
        write_poscar(str(d / "POSCAR"),
                     Poscar("t", 1.0, lattice, ["Si"], [3], "Direct", pos))
    return lattice


def test_perturbation_breaks_the_axis_and_stays_within_bounds(tmp_path):
    import numpy as np

    from yamatsumi.geometry import direct_to_cartesian, mic_displacement
    from yamatsumi.poscar import read_poscar

    script = _load_script()
    lattice = _staged_images(tmp_path, script)
    before = [read_poscar(str(tmp_path / f"{i:02d}" / "POSCAR")).positions
              for i in range(script.NUM_IMAGES + 2)]
    script.perturb_images(tmp_path, max_ang=0.05, seed=1)
    after = [read_poscar(str(tmp_path / f"{i:02d}" / "POSCAR")).positions
             for i in range(script.NUM_IMAGES + 2)]

    assert np.array_equal(after[0], before[0])
    assert np.array_equal(after[-1], before[-1])
    axis = np.ones(3) / np.sqrt(3.0)
    for b, a in zip(before[1:-1], after[1:-1]):
        d = direct_to_cartesian(mic_displacement(b, a), lattice)
        assert np.linalg.norm(d, axis=1).max() <= 0.05 + 1e-9
        assert np.allclose(d.mean(axis=0), 0.0, atol=1e-9)
        off_axis = d[0] - (d[0] @ axis) * axis
        assert np.linalg.norm(off_axis) > 1e-3


def test_perturbation_is_reproducible(tmp_path):
    from yamatsumi.poscar import read_poscar

    script = _load_script()
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        _staged_images(tmp_path / sub, script)
        script.perturb_images(tmp_path / sub, max_ang=0.05, seed=7)
    for i in range(1, script.NUM_IMAGES + 1):
        pa = read_poscar(str(tmp_path / "a" / f"{i:02d}" / "POSCAR")).positions
        pb = read_poscar(str(tmp_path / "b" / f"{i:02d}" / "POSCAR")).positions
        assert (pa == pb).all()
