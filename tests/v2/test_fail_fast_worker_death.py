"""Fail-fast (no-hang) regression tests for the v2 parallel paths.

Issue #48/#49 class: when a worker is SIGKILLed / OOM-killed, the v2
ProcessPoolExecutor/Pool paths must surface a ``BrokenProcessPool`` (or a
clear wrapped error) *promptly* rather than blocking forever on a result
that will never arrive.

The v2 parallel sites covered here:

* ``writer._write_v2_inmem``       -> ``_encode_one_shard_from_shm`` (writer.py:1331 — the
  parallel in-memory arm this file's write test actually submits; the family also has
  ``_encode_one_shard`` and the ``_gather_local`` / ``_gather_shm`` / ``_gather_backed``
  variants, all carrying the same hook)
* ``reader_cpu.read_v2_to_host``   -> ``_read_one_shard_into_shm``
* the dense read path             -> ``_scatter_one_shard_dense``

(``converter.convert_v1_to_v2`` -> ``_convert_one_shard`` was a fourth site until #154
part A deleted the converter.)

Each per-shard worker function carries the canonical test-only env hook
``workers._maybe_test_kill_self`` (``CELLSTREAM_TEST_KILL_SHARD_IDX`` ->
SIGKILL) — a single shared definition, now used only by the v2 workers (its v1
consumers went in #154 parts A and B). Because the executors use the spawn start
method, setting the env var in the parent before the executor is created propagates it to
workers via process-start env inheritance.

#154 part B deleted ``tests/test_write_fail_fast_worker_death.py`` with the v1 writer. That
file asserted two things beyond the exception class — no partial archive, and the write lock
released — for both v1 write paths. They are re-pinned in **two** places, because the two
surviving write paths hold different locks:

* the DIRECTORY writer, in ``test_writer_worker_death_fails_fast`` below (oracle:
  ``manifest.json`` absent, since ``header.h5ad`` / ``obs.parquet`` survive);
* the PACKED writer, in ``tests/v2/test_writer_shm_handoff.py::test_no_shm_leak_on_worker_death``
  (oracle: the ``.shad`` file absent, and its own ``<archive>.lock`` re-acquirable —
  ``write_packed`` takes that outer lock and calls the directory writer with
  ``unsafe_no_lock=True``).

So this file is not the only write-side fail-fast home; that shm-leak test kills a write worker
and asserts ``BrokenProcessPool`` too.

Anti-hang guard: every call is run on a ``threading.Thread`` with a
bounded ``join(timeout=...)``. If the thread is still alive after the
timeout the worker-death detection has regressed into a hang and the
test fails loudly (instead of the whole suite stalling for the timeout).
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from tests._archive_helpers import write_directory

# How long we let a worker-death call run before declaring a hang. Spawn
# interpreter startup + imports is ~0.5-1 s/worker on slow CI, so this is
# generous; a regressed (hanging) path would never finish.
_HANG_TIMEOUT_S = 60.0


def _make_uint16_adata(seed=17, n_obs=400, n_vars=200, density=0.05):
    rng = np.random.default_rng(seed)
    nnz = int(n_obs * n_vars * density)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, 50, size=nnz, dtype=np.uint16)
    csr_raw = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    csr = sp.csr_matrix(csr_raw.shape, dtype=np.float32)
    csr.data = csr_raw.data.astype(np.float32)
    csr.indices = csr_raw.indices.astype(np.int32)
    csr.indptr = csr_raw.indptr.astype(np.int64)
    return ad.AnnData(X=csr)


def _run_with_hang_guard(fn, timeout=_HANG_TIMEOUT_S):
    """Run *fn* on a thread; return the exception it raised.

    Asserts the thread finished within *timeout* (a hang would mean the
    fail-fast detection regressed) and that *fn* did raise.
    """
    box: dict = {}

    def _target():
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 — capture for assertions
            box["exc"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), (
        f"call did not return within {timeout}s after a worker died — "
        "the fail-fast path regressed into a hang (issue #48/#49)"
    )
    assert "exc" in box, "expected the call to raise after a worker died, but it returned cleanly"
    return box["exc"]


def test_writer_worker_death_fails_fast(tmp_path, monkeypatch):
    """_write_v2_inmem surfaces a dead encode worker promptly, no hang -- and leaves no
    readable archive behind, with the write lock released.

    The last two assertions came from ``tests/test_write_fail_fast_worker_death.py``, which
    #154 part B deleted with the v1 writer. That file pinned them for both v1 write paths;
    this test pinned only the exception class, so they would otherwise have gone unpinned for
    the v2 writer even though the properties still hold.

    ``manifest.json`` is the right oracle, NOT ``out.exists()``: ``header.h5ad`` and
    ``obs.parquet`` ARE left behind (verified empirically, not assumed). The manifest is
    published last precisely so a half-written archive is not readable, so its absence is
    what "no partial archive" means here.
    """
    from cellstream import lock

    adata = _make_uint16_adata(seed=1)
    out = tmp_path / "v2_arch"

    # Worker for shard 1 SIGKILLs itself before encoding -> BrokenProcessPool.
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    monkeypatch.setenv("CELLSTREAM_TEST_KILL_SHARD_IDX", "1")
    exc = _run_with_hang_guard(lambda: write_directory(adata, out, n_shards=4, n_workers=2))
    assert isinstance(exc, BrokenProcessPool), f"expected BrokenProcessPool, got {exc!r}"

    # No partial archive: the manifest is published last, so a failed write must leave none.
    assert not (out / "manifest.json").exists(), (
        f"failed write left a manifest.json (partial archive); dir holds "
        f"{sorted(p.name for p in out.iterdir())}"
    )
    # And the write lock must have been released, not left held by the dead pool.
    with lock.acquire_write_lock(out / ".cellstream.lock", op="write"):
        pass  # re-acquired cleanly -> it was released on failure


def test_reader_worker_death_fails_fast_and_unlinks_shm(tmp_path, monkeypatch):
    """read_v2_to_host surfaces a dead read worker promptly AND leaks no shm.

    Mirrors tests/v2/test_omp_env_propagation.py's shm-leak check: wrap
    allocate_shm_buffers to capture the segment names, then after the
    failure assert none survive in /dev/shm (the BaseException handler
    must close_and_unlink the bundle even when the executor breaks).
    """
    if sys.platform != "linux":
        pytest.skip("SharedMemory leak detection requires Linux /dev/shm")

    from cellstream import shm as _shm
    from cellstream.v2 import reader_cpu

    adata = _make_uint16_adata(seed=3, n_obs=200, n_vars=100)
    out = tmp_path / "v2_arch"
    write_directory(adata, out, n_shards=4, n_workers=2)

    allocated_names: list[str] = []
    original_alloc = _shm.allocate_shm_buffers

    def _capturing_alloc(**kwargs):
        bundle = original_alloc(**kwargs)
        allocated_names.extend([bundle.data.name, bundle.indices.name, bundle.indptr.name])
        return bundle

    # This test validates the multiprocess Python reader's fail-fast + shm-unlink
    # path; force the Python reader since the Rust decode path is single-process
    # (no spawn workers, no shm) and would bypass the machinery under test.
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    monkeypatch.setenv("CELLSTREAM_TEST_KILL_SHARD_IDX", "1")
    with mock.patch.object(_shm, "allocate_shm_buffers", side_effect=_capturing_alloc):
        exc = _run_with_hang_guard(
            lambda: reader_cpu.read_v2_to_host(out, cast_back=False, n_workers=2, omp_nthreads=4)
        )
    assert isinstance(exc, BrokenProcessPool), f"expected BrokenProcessPool, got {exc!r}"

    assert allocated_names, "no shm allocations were captured — test setup broken"
    leaked = [n for n in allocated_names if (Path("/dev/shm") / n).exists()]
    assert not leaked, f"shm segments leaked after worker death: {leaked}"


def test_reader_dense_worker_death_fails_fast_and_unlinks_shm(tmp_path, monkeypatch):
    """L15: the DENSE read path (container='dense', Python fallback) must also
    unlink its /dev/shm segment when a worker dies — mirrors the CSR leak test but
    patches allocate_dense_shm_buffer."""
    if sys.platform != "linux":
        pytest.skip("SharedMemory leak detection requires Linux /dev/shm")

    from cellstream import shm as _shm
    from cellstream.manifest import load_manifest
    from cellstream.v2 import reader_cpu
    from cellstream.v2.materialize import resolve_plan

    adata = _make_uint16_adata(seed=4, n_obs=200, n_vars=100)
    out = tmp_path / "v2_arch"
    write_directory(adata, out, n_shards=4, n_workers=2)

    allocated_names: list[str] = []
    original_alloc = _shm.allocate_dense_shm_buffer

    def _capturing_alloc(**kwargs):
        bundle = original_alloc(**kwargs)
        allocated_names.append(bundle.array.name)
        return bundle

    # Force the multiprocess Python dense reader (Rust dense scatter is single-process).
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    monkeypatch.setenv("CELLSTREAM_TEST_KILL_SHARD_IDX", "1")
    plan = resolve_plan(
        load_manifest(out / "manifest.json"),
        container="dense",
        data_dtype="float32",
        index_dtype=None,
        allow_lossy=False,
        cast_back=False,
        target="cpu",
    )
    with mock.patch.object(_shm, "allocate_dense_shm_buffer", side_effect=_capturing_alloc):
        exc = _run_with_hang_guard(
            lambda: reader_cpu.read_v2_to_host(out, plan=plan, n_workers=2, omp_nthreads=4)
        )
    assert isinstance(exc, BrokenProcessPool), f"expected BrokenProcessPool, got {exc!r}"
    assert allocated_names, "no dense shm allocation captured — test setup broken"
    leaked = [n for n in allocated_names if (Path("/dev/shm") / n).exists()]
    assert not leaked, f"dense shm segment leaked after worker death: {leaked}"
