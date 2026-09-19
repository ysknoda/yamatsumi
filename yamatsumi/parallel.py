"""
Optional mpi4py integration with graceful degradation.

The availability check must catch more than ImportError: mpi4py's Python
package can import successfully while `from mpi4py import MPI` itself
raises RuntimeError when no native MPI library is linked to the process
(a real, easily-hit misconfiguration on workstations without an MPI runtime
installed -- not a hypothetical edge case).
"""
from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except (ImportError, RuntimeError):
    MPI = None
    _MPI_AVAILABLE = False


def get_comm():
    """Return the world communicator, or None if MPI is unavailable."""
    if not _MPI_AVAILABLE:
        return None
    return MPI.COMM_WORLD


def get_rank(comm) -> int:
    return comm.Get_rank() if comm is not None else 0


def get_size(comm) -> int:
    return comm.Get_size() if comm is not None else 1


def is_root(comm) -> bool:
    return get_rank(comm) == 0


def is_parallel(comm) -> bool:
    """True only when actually launched with more than one MPI rank."""
    return comm is not None and get_size(comm) > 1


def distribute_indices(n_items: int, rank: int, size: int) -> List[int]:
    """
    Split range(n_items) into `size` contiguous, non-overlapping blocks and
    return the block assigned to `rank`. Every item is covered by exactly
    one rank; ranks at or beyond n_items receive an empty list.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    base, extra = divmod(n_items, size)
    start = rank * base + min(rank, extra)
    count = base + (1 if rank < extra else 0)
    return list(range(start, start + count))


def allreduce_all_ok(comm, local_ok: bool) -> bool:
    """
    Reduce a per-rank success flag with logical AND across all ranks, so
    that a failure on any single rank is visible to (and raised by) every
    rank at the same point -- never only the rank where it happened. With
    no MPI (comm is None) this simply returns the local flag unchanged.
    """
    if comm is None:
        return local_ok
    # MPI may be None here even with a non-None `comm` when a test passes a
    # hand-written fake communicator without a real mpi4py MPI module behind
    # it; falling back to op=None keeps that usable without affecting real
    # MPI runs (where MPI.LAND is always used, exactly as before).
    op = MPI.LAND if MPI is not None else None
    return bool(comm.allreduce(local_ok, op=op))


def gather_by_index(comm, local_results: Dict[int, object], n_total: int) -> list:
    """
    Gather {index: value} dicts from every rank and reassemble a full,
    index-ordered list of length n_total. Reassembly keys off the index,
    not gather/arrival order, so the result is identical regardless of
    which rank computed which item -- the invariant that makes serial and
    MPI-parallel execution produce byte-identical results.
    """
    if comm is None:
        full = dict(local_results)
    else:
        gathered = comm.allgather(local_results)
        full = {}
        for d in gathered:
            full.update(d)
    missing = [i for i in range(n_total) if i not in full]
    if missing:
        raise RuntimeError(f"gather_by_index: missing results for indices {missing}")
    return [full[i] for i in range(n_total)]
