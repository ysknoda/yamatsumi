"""Shared pytest fixtures for the yamatsumi test suite."""
from __future__ import annotations

import shutil
import sys

import pytest


@pytest.fixture()
def tmp_work_dir(tmp_path):
    """A fresh temporary working directory for a single test."""
    return tmp_path


def mpi_runtime_available() -> bool:
    """True only if mpirun/mpiexec exist AND `from mpi4py import MPI` succeeds."""
    if shutil.which("mpirun") is None and shutil.which("mpiexec") is None:
        return False
    try:
        from mpi4py import MPI  # noqa: F401
    except (ImportError, RuntimeError):
        return False
    return True


requires_real_mpi = pytest.mark.skipif(
    not mpi_runtime_available(),
    reason="No working mpirun/mpiexec + mpi4py MPI runtime available in this environment.",
)
