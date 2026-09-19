"""
Phase 1 (initialisation) tests.

The central invariant here is the MPI staging partition: under `mpirun`, each
image directory must be staged and written by exactly ONE rank. Every rank
writing every directory is a concurrent multi-writer race -- and because
`prepare_image_directory()` deletes stale OUTCAR/CONTCAR files, a rank lagging
even slightly behind can delete an OUTCAR another rank's VASP has just
written, or truncate a POSCAR another rank's VASP is reading.
"""

from __future__ import annotations

import numpy as np
import pytest

from yamatsumi import parallel
from yamatsumi.cli import build_parser
from yamatsumi.phases import init_phase
from yamatsumi.poscar import Poscar, write_poscar

LATTICE_SIZE = 20.0
NUM_IMAGES = 3
N_TOTAL = NUM_IMAGES + 2


class _FakeComm:
    """Minimal stand-in for an mpi4py communicator, single-process semantics."""

    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def allreduce(self, value, op=None):
        return value


def _make_poscar(x, comment):
    return Poscar(
        comment=comment,
        scale=1.0,
        lattice=np.eye(3) * LATTICE_SIZE,
        species=["H"],
        counts=[1],
        coord_type="Direct",
        positions=np.array([[x / LATTICE_SIZE, 0.3 / LATTICE_SIZE, 0.5]]),
    )


def _make_args(tmp_path, work_dir_name="NEB_run", extra=()):
    copy_dir = tmp_path / "CopyDir"
    copy_dir.mkdir(exist_ok=True)
    (copy_dir / "INCAR").write_text("# placeholder\n")
    init_path = tmp_path / "POSCAR_init"
    final_path = tmp_path / "POSCAR_final"
    write_poscar(str(init_path), _make_poscar(-1.0, "initial"))
    write_poscar(str(final_path), _make_poscar(1.0, "final"))
    return build_parser().parse_args(
        [
            "--phase",
            "init",
            "--init",
            str(init_path),
            "--final",
            str(final_path),
            "--copy-dir",
            str(copy_dir),
            "--work-dir",
            str(tmp_path / work_dir_name),
            "--num-images",
            str(NUM_IMAGES),
            *extra,
        ]
    )


# Captured once, at import time, so that repeated _run_rank() calls within a
# single test never wrap an already-traced write_poscar: monkeypatch unwinds at
# the end of a test, not between calls inside one.
_PRISTINE_WRITE_POSCAR = init_phase.write_poscar


def _run_rank(tmp_path, monkeypatch, rank, size, extra=()):
    """
    Run phase_init as if it were MPI rank `rank` of `size`, with VASP stubbed
    out, and return the list of image directory names this rank staged.
    """
    staged: list[str] = []

    monkeypatch.setattr(parallel, "get_comm", lambda: _FakeComm(rank, size))
    monkeypatch.setattr(init_phase, "run_vasp_for_images", lambda *a, **k: None)

    def traced_write(path, poscar):
        from pathlib import Path

        staged.append(Path(path).parent.name)
        return _PRISTINE_WRITE_POSCAR(path, poscar)

    monkeypatch.setattr(init_phase, "write_poscar", traced_write)
    init_phase.phase_init(_make_args(tmp_path, extra=extra))
    return staged


@pytest.mark.parametrize("size", [2, 3, 5, 7])
def test_each_image_dir_is_staged_by_exactly_one_rank(tmp_path, monkeypatch, size):
    """
    Regression test for the init_phase concurrent-write race.

    Before the fix every rank staged every directory, so with `size` ranks each
    of the N_TOTAL directories was written `size` times instead of once.
    """
    per_rank = [_run_rank(tmp_path, monkeypatch, rank, size) for rank in range(size)]

    all_staged = [d for staged in per_rank for d in staged]
    assert len(all_staged) == len(set(all_staged)), (
        f"with {size} ranks, these image directories were staged by more than "
        f"one rank (concurrent-write race): "
        f"{sorted({d for d in all_staged if all_staged.count(d) > 1})}"
    )
    assert len(all_staged) == N_TOTAL, (
        f"with {size} ranks, {len(all_staged)} directories were staged but "
        f"{N_TOTAL} exist -- every image directory must be staged by exactly "
        f"one rank, none left unstaged."
    )


def test_serial_run_still_stages_every_image_dir(tmp_path, monkeypatch):
    """The partition must degenerate to 'everything' on a single rank."""
    staged = _run_rank(tmp_path, monkeypatch, rank=0, size=1)
    assert len(staged) == N_TOTAL
    assert len(set(staged)) == N_TOTAL


def test_dry_run_root_stages_every_image_dir(tmp_path, monkeypatch):
    """
    A dry run launches no VASP, so there is no work to partition: the root rank
    writes the complete set (single writer, no race) and non-root ranks write
    nothing, leaving a fully staged directory to inspect either way.
    """
    root_staged = _run_rank(tmp_path, monkeypatch, 0, 3, extra=("--dry-run",))
    assert sorted(root_staged) == sorted(set(root_staged))
    assert len(root_staged) == N_TOTAL

    for rank in (1, 2):
        assert _run_rank(tmp_path, monkeypatch, rank, 3, extra=("--dry-run",)) == []
