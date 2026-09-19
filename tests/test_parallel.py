"""
Tests for yamatsumi.parallel.

Real multi-process MPI cannot run in this environment (see
tests/test_mpi_integration.py), so `allreduce_all_ok` / `gather_by_index`
are exercised here with a tiny hand-written fake communicator rather than
mpi4py itself.
"""
from __future__ import annotations

import pytest

from yamatsumi import parallel


class _FakeComm:
    """Minimal stand-in for an mpi4py communicator, single-process semantics."""

    def __init__(self, rank, size, allgather_result=None, allreduce_result=None):
        self._rank = rank
        self._size = size
        self._allgather_result = allgather_result
        self._allreduce_result = allreduce_result

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def allgather(self, obj):
        return self._allgather_result if self._allgather_result is not None else [obj]

    def allreduce(self, value, op=None):
        return self._allreduce_result if self._allreduce_result is not None else value


def test_get_rank_size_root_serial_when_comm_none():
    assert parallel.get_rank(None) == 0
    assert parallel.get_size(None) == 1
    assert parallel.is_root(None) is True
    assert parallel.is_parallel(None) is False


def test_is_parallel_true_only_when_size_gt_1():
    assert parallel.is_parallel(_FakeComm(0, 1)) is False
    assert parallel.is_parallel(_FakeComm(0, 4)) is True
    assert parallel.is_root(_FakeComm(0, 4)) is True
    assert parallel.is_root(_FakeComm(1, 4)) is False


@pytest.mark.parametrize("n_items,size", [(7, 3), (5, 5), (2, 5), (0, 3), (10, 1)])
def test_distribute_indices_full_coverage_no_overlap(n_items, size):
    all_indices = []
    for rank in range(size):
        chunk = parallel.distribute_indices(n_items, rank, size)
        all_indices.extend(chunk)
        assert len(chunk) <= (n_items // size) + 1
    assert sorted(all_indices) == list(range(n_items))


def test_distribute_indices_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        parallel.distribute_indices(5, 0, 0)


def test_allreduce_all_ok_no_comm_passthrough():
    assert parallel.allreduce_all_ok(None, True) is True
    assert parallel.allreduce_all_ok(None, False) is False


def test_allreduce_all_ok_with_fake_comm():
    comm = _FakeComm(0, 3, allreduce_result=False)
    assert parallel.allreduce_all_ok(comm, True) is False


def test_gather_by_index_no_comm():
    result = parallel.gather_by_index(None, {0: "a", 1: "b"}, n_total=2)
    assert result == ["a", "b"]


def test_gather_by_index_merges_by_index_not_arrival_order():
    """The reassembly must key off the index, not gather/arrival order."""
    comm = _FakeComm(0, 3, allgather_result=[{2: "c"}, {0: "a"}, {1: "b"}])
    result = parallel.gather_by_index(comm, {}, n_total=3)
    assert result == ["a", "b", "c"]


def test_gather_by_index_missing_index_raises():
    comm = _FakeComm(0, 2, allgather_result=[{0: "a"}])
    with pytest.raises(RuntimeError):
        parallel.gather_by_index(comm, {}, n_total=2)
