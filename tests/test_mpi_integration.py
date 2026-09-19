"""
MPI-parallel vs. serial regression test: running yamatsumi's driver process
itself under `mpirun` (distributing NEB images across ranks) must produce
results identical to plain serial execution.

This cannot run in a sandbox without a working mpirun/mpiexec + mpi4py MPI
runtime (confirmed absent in the environment this suite was developed in:
mpi4py's Python package imports, but `from mpi4py import MPI` raises
RuntimeError with no native libmpi linked, and neither mpirun nor mpiexec
exist on PATH) -- it self-skips via conftest.requires_real_mpi and is
meant to run in CI (with e.g. `apt-get install libopenmpi-dev openmpi-bin`)
or manually on your own HPC/workstation MPI setup. See also
docs/real_vasp_validation.md for the equivalent manual procedure with real
VASP.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from yamatsumi.directories import get_image_dirs
from yamatsumi.outcar import parse_energy_from_outcar
from yamatsumi.poscar import Poscar, write_poscar

from conftest import requires_real_mpi

MOCK_VASP = str(Path(__file__).parent / "fixtures" / "mock_vasp.py")
LATTICE_SIZE = 20.0


def _make_poscar(x, y, comment):
    import numpy as np
    lattice = np.eye(3) * LATTICE_SIZE
    positions = np.array([[x / LATTICE_SIZE, y / LATTICE_SIZE, 0.5]])
    return Poscar(
        comment=comment, scale=1.0, lattice=lattice, species=["H"], counts=[1],
        coord_type="Direct", positions=positions,
    )


def _run(tmp_path, work_dir_name, extra_argv):
    copy_dir = tmp_path / "CopyDir"
    copy_dir.mkdir(exist_ok=True)
    (copy_dir / "INCAR").write_text("# placeholder\n")
    init_path  = tmp_path / "POSCAR_init"
    final_path = tmp_path / "POSCAR_final"
    write_poscar(str(init_path), _make_poscar(-1.0, 0.3, "initial"))
    write_poscar(str(final_path), _make_poscar(1.0, 0.3, "final"))

    work_dir = tmp_path / work_dir_name
    vasp_cmd = f"{sys.executable} {MOCK_VASP} --pes doublewell"
    base_argv = [
        "-m", "yamatsumi", "--phase", "all",
        "--init", str(init_path), "--final", str(final_path),
        "--copy-dir", str(copy_dir), "--work-dir", str(work_dir),
        "--vasp-cmd", vasp_cmd, "--no-show",
        "--num-images", "3", "--optimizer", "fire",
        "--force-tol", "0.05", "--max-iter", "100",
    ]
    subprocess.run(
        [*extra_argv, sys.executable, *base_argv],
        check=True, cwd=str(Path(__file__).parent.parent),
    )
    image_dirs = get_image_dirs(work_dir, 3)
    return [parse_energy_from_outcar(idir / "OUTCAR") for idir in image_dirs]


@requires_real_mpi
def test_mpi_parallel_matches_serial(tmp_path):
    serial_energies = _run(tmp_path, "NEB_run_serial", extra_argv=[])

    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    mpi_energies = _run(tmp_path, "NEB_run_mpi", extra_argv=[mpirun, "-n", "3"])

    assert len(serial_energies) == len(mpi_energies)
    for e_serial, e_mpi in zip(serial_energies, mpi_energies):
        assert e_mpi == pytest.approx(e_serial, abs=1e-8)
