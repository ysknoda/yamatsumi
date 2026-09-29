"""
Tests for examples/si_vacancy_neb/make_endpoints.py, run as a script the way
a user runs it. The script must refuse inputs that would write a physically
wrong endpoint pair, and must never report a failed self-check as OK.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from yamatsumi.geometry import direct_to_cartesian, mic_displacement
from yamatsumi.poscar import read_poscar

SCRIPT = Path(__file__).parent.parent / "examples" / "si_vacancy_neb" / "make_endpoints.py"
A_SI = 5.431


def _run(tmp_path, *argv):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(SCRIPT), *argv, "--output-dir", str(out)],
                       capture_output=True, text=True)
    return r, out


def test_default_run_writes_a_nearest_neighbour_hop(tmp_path):
    r, out = _run(tmp_path)
    assert r.returncode == 0, r.stderr
    init, final = read_poscar(str(out / "POSCAR_init")), read_poscar(str(out / "POSCAR_final"))
    disp = np.linalg.norm(direct_to_cartesian(
        mic_displacement(init.positions, final.positions), init.lattice), axis=1)
    assert np.count_nonzero(disp > 1e-9) == 1
    assert disp.max() == pytest.approx(A_SI * np.sqrt(3) / 4, abs=1e-9)


@pytest.mark.parametrize("argv", [
    # Regression: -1 was used as a Python index by np.delete but as an
    # ordinal by the index bookkeeping, writing a 9.41 Ang "hop" through
    # occupied sites while printing "expected 2.351692 Ang ... OK".
    ["--vacancy-site", "-1"],
    ["--vacancy-site", "64"],            # out of range for 2x2x2 (raw IndexError)
    ["--supercell-size", "0"],           # raw ValueError from np.concatenate
    ["--supercell-size", "-2"],
    # Regression: wrote Ge at silicon's lattice constant (2.352 Ang bonds).
    ["--element", "Ge"],
])
def test_bad_input_writes_nothing_and_explains(tmp_path, argv):
    r, out = _run(tmp_path, *argv)
    assert r.returncode != 0
    assert not (out / "POSCAR_init").exists()
    assert "Traceback" not in r.stderr, r.stderr


def test_other_element_with_its_own_lattice_constant_is_accepted(tmp_path):
    r, out = _run(tmp_path, "--element", "Ge", "--lattice-constant", "5.658")
    assert r.returncode == 0, r.stderr
    assert read_poscar(str(out / "POSCAR_init")).species == ["Ge"]
