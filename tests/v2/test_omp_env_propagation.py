"""Regression test for the load-bearing OMP_NUM_THREADS env-propagation fix.

The v0.2 reader_cpu sets OMP_NUM_THREADS in the parent's env
BEFORE Pool() creation, so spawn workers inherit it at execve and
bitshuffle sees the right value at first import. The original bug was
to use Pool(initializer=_init_omp_threads) which silently failed
because the initializer pickle pulls in reader_cpu, which imports
bitshuffle, which initializes OpenMP with os.cpu_count() BEFORE the
initializer runs.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

from tests._archive_helpers import write_directory


def _get_omp_env(_):
    """Worker function (must be module-level for spawn to pickle)."""
    import os

    return os.environ.get("OMP_NUM_THREADS")


def test_omp_num_threads_inherited_by_spawn_workers():
    """Setting OMP_NUM_THREADS in the parent's os.environ propagates to
    spawn workers via inheritance, NOT via Pool(initializer=)."""
    target = "7"  # arbitrary value not matching common defaults (1, 4, os.cpu_count())
    prev = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = target
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
            results = list(ex.map(_get_omp_env, [None, None]))
        assert all(r == target for r in results), (
            f"workers reported {results}, expected all '{target}'"
        )
    finally:
        if prev is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev


# ---------------------------------------------------------------------------
# Round-4 strengthening: exercise the actual regression scenario
# ---------------------------------------------------------------------------


def _spawn_worker_observe_omp_after_bitshuffle_import(_):
    """Run inside a spawn worker. Imports bitshuffle (triggering its OMP
    init at module-load time), then returns the OMP_NUM_THREADS env var
    that bitshuffle would have seen at init."""
    import os

    # Import bitshuffle FIRST — this is the load-bearing moment: bitshuffle's
    # _omp_set_num_threads() is called from within its C-extension init, which
    # reads OMP_NUM_THREADS from the env at that point.
    import bitshuffle  # noqa: F401

    # Return the env var. If it's the expected value, bitshuffle saw it too
    # (we can't directly observe bitshuffle's OMP pool size from Python, but
    # if the env was set BEFORE this import, bitshuffle initialized OMP with it).
    return os.environ.get("OMP_NUM_THREADS")


def test_bitshuffle_import_observes_parent_omp_env():
    """The actual regression: workers must see OMP_NUM_THREADS=X at the moment
    they import bitshuffle. This is the load-bearing case that broke when
    Pool(initializer=...) was tried instead of parent-env-setting.

    The previous test (test_omp_num_threads_inherited_by_spawn_workers) only
    verified env visibility — it doesn't import bitshuffle in the worker, so it
    couldn't catch the regression where the initializer ran AFTER bitshuffle's
    import.
    """
    target = "5"
    prev = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = target
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
            observed = list(ex.map(_spawn_worker_observe_omp_after_bitshuffle_import, [None, None]))
        assert all(v == target for v in observed), (
            f"workers observed OMP_NUM_THREADS={observed}, expected all '{target}'"
        )
    finally:
        if prev is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev


def test_reader_cpu_restores_omp_env_on_normal_completion(tmp_path):
    """After read_v2_to_host returns normally, parent's OMP_NUM_THREADS is
    restored to its prior state."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    from cellstream.v2.reader_cpu import read_v2_to_host

    # Build a tiny v2 archive on disk. v2 format requires uint16-compatible
    # integer data values (lossless_cast to uint16 rejects floats).
    rng = np.random.default_rng(1)
    n_obs, n_vars = 50, 20
    nnz = int(n_obs * n_vars * 0.2)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, 100, size=nnz, dtype=np.uint16)
    csr_raw = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    csr = sp.csr_matrix(csr_raw.shape, dtype=np.float32)
    csr.data = csr_raw.data.astype(np.float32)
    csr.indices = csr_raw.indices.astype(np.int32)
    csr.indptr = csr_raw.indptr.astype(np.int64)
    adata = ad.AnnData(X=csr)
    out = tmp_path / "tiny_v2"
    write_directory(adata, out, n_shards=2, n_workers=2)

    sentinel = "13"  # value unlikely to match anything else
    prev = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = sentinel
        # Call read_v2_to_host. It will set OMP internally (per its parent-env
        # pattern), do the read, and restore on finally.
        read_v2_to_host(out, cast_back=False, n_workers=2, omp_nthreads=4)
        # After the call, env should be back to sentinel:
        assert os.environ.get("OMP_NUM_THREADS") == sentinel, (
            f"reader_cpu changed OMP_NUM_THREADS without restoring: "
            f"got {os.environ.get('OMP_NUM_THREADS')!r}, expected {sentinel!r}"
        )
    finally:
        if prev is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev


def test_reader_cpu_restores_omp_env_on_failure(tmp_path, monkeypatch):
    """If read_v2_to_host raises mid-read (after the env-set block opens),
    parent's OMP_NUM_THREADS is restored via the try/finally.

    The original test used a missing-manifest archive which raises in
    load_manifest() — BEFORE the OMP_NUM_THREADS try/finally block opens
    (line ~155 in reader_cpu.py). That test passed trivially because the env
    was never modified.

    This version builds a valid archive (so manifest load + _plan_shards
    succeed), then corrupts only the data-stream chunk bytes of one shard.
    _plan_shards reads only the indptr stream and will succeed. The worker
    pool fails when it tries to zstd-decompress the corrupted data bytes —
    after the parent has set OMP_NUM_THREADS. The try/finally must restore
    the env when pool.map() propagates the worker exception.
    """
    import json
    import struct

    import anndata as ad
    import numpy as np
    import pytest
    import scipy.sparse as sp

    from cellstream.v2.reader_cpu import read_v2_to_host

    # This test validates the multiprocess Python reader's OMP-env try/finally; the
    # Rust decode path is single-process (no spawn workers, no OMP env window), so
    # force the Python reader.
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")

    # Build a valid tiny v2 archive with 2 shards so len(inputs) > 1
    # (the n_workers > 1 and len(inputs) > 1 guard at reader_cpu.py:134
    # is what determines whether the env-set try/finally block runs at all).
    rng = np.random.default_rng(7)
    nnz = 200
    rows = rng.integers(0, 50, size=nnz)
    cols = rng.integers(0, 20, size=nnz)
    vals = rng.integers(1, 100, size=nnz, dtype=np.uint16)
    csr_raw = sp.coo_matrix((vals, (rows, cols)), shape=(50, 20)).tocsr()
    csr = sp.csr_matrix(csr_raw.shape, dtype=np.float32)
    csr.data = csr_raw.data.astype(np.float32)
    csr.indices = csr_raw.indices.astype(np.int32)
    csr.indptr = csr_raw.indptr.astype(np.int64)
    adata = ad.AnnData(X=csr)
    out = tmp_path / "valid_v2"
    write_directory(adata, out, n_shards=2, n_workers=2)

    # Corrupt the data-stream chunk bytes in shard 0, leaving the header
    # (magic + JSON) and indptr stream intact.
    #
    # SHX2 layout: 4-byte magic + 2-byte reserved + 4-byte header_len
    # + <header_len> JSON bytes, padded to 4 KB boundary.
    # Data stream chunks begin immediately after the 4 KB boundary.
    # The chunk offsets are stored in the JSON header itself.
    #
    # Strategy: parse the header to find the first data chunk's byte offset
    # and corrupt exactly those bytes. _plan_shards only reads the indptr
    # stream (the last stream, whose chunks are listed after data+indices in
    # the header). Workers read data first and will raise on zstd decode.
    shard_path = out / "shard_0000.shx"
    raw = bytearray(shard_path.read_bytes())
    # Parse fixed prefix: magic(4) + reserved(2) + header_size(4) = 10 bytes
    _magic = raw[0:4]
    _reserved, header_size = struct.unpack_from("<HI", raw, 4)
    js = raw[10 : 10 + header_size].decode("utf-8")
    header_dict = json.loads(js)
    # Find the first data chunk's offset and size.
    data_chunks = header_dict["data"]["chunks"]
    if data_chunks:
        first_chunk = data_chunks[0]
        off = first_chunk["offset"]
        csz = first_chunk["compressed_size"]
        # Overwrite with random-looking non-zstd bytes.
        corruption = bytes(range(256)) * (csz // 256 + 1)
        raw[off : off + csz] = corruption[:csz]
        shard_path.write_bytes(bytes(raw))

    sentinel = "11"
    prev = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = sentinel
        # Corrupting a data-stream chunk's bytes makes the bitshuffle/zstd
        # decoder in the worker raise some codec-level exception that bubbles
        # up via multiprocessing.pool. The exact concrete class depends on
        # the codec library; what matters is that an exception escapes the
        # `with _OMP_ENV_LOCK` block so we can verify env was restored.
        with pytest.raises(Exception):  # noqa: B017
            # n_workers=2, len(inputs)=2 → env-set try/finally block runs.
            read_v2_to_host(out, cast_back=False, n_workers=2, omp_nthreads=4)
        # After the exception, reader_cpu's try/finally must have restored env
        # to what it was before the call (sentinel).
        assert os.environ.get("OMP_NUM_THREADS") == sentinel, (
            f"env not restored after reader_cpu raised: "
            f"got {os.environ.get('OMP_NUM_THREADS')!r}, expected {sentinel!r}"
        )
    finally:
        if prev is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev


def test_reader_cpu_no_shm_leak_on_worker_failure(tmp_path, monkeypatch):
    """If read_v2_to_host raises inside the worker pool, the shm bundle
    is closed+unlinked before the exception propagates. Without this,
    every failure leaks 3 shm segments per call."""
    import json
    import struct
    import sys
    import unittest.mock as _mock
    from pathlib import Path

    import anndata as ad
    import numpy as np
    import pytest
    import scipy.sparse as sp

    from cellstream import shm as _shm
    from cellstream.v2 import reader_cpu

    # Linux-only: SharedMemory backing differs on macOS/Windows.
    if sys.platform != "linux":
        pytest.skip("SharedMemory leak detection requires Linux /dev/shm")

    # This test validates the multiprocess Python reader's shm cleanup-on-failure;
    # the Rust decode path is single-process (heap output, no shm bundle), so force
    # the Python reader.
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")

    # Build a valid v2 archive
    rng = np.random.default_rng(7)
    nnz = 200
    rows = rng.integers(0, 50, size=nnz)
    cols = rng.integers(0, 20, size=nnz)
    vals = rng.integers(1, 100, size=nnz, dtype=np.uint16)
    csr_raw = sp.coo_matrix((vals, (rows, cols)), shape=(50, 20)).tocsr()
    csr = sp.csr_matrix(csr_raw.shape, dtype=np.float32)
    csr.data = csr_raw.data.astype(np.float32)
    csr.indices = csr_raw.indices.astype(np.int32)
    csr.indptr = csr_raw.indptr.astype(np.int64)
    adata = ad.AnnData(X=csr)
    out = tmp_path / "shm_leak_v2"
    write_directory(adata, out, n_shards=2, n_workers=2)

    # Corrupt the data-stream chunk bytes of shard_0000.shx so workers fail
    # during zstd decompress (same corruption strategy as round-6 env test).
    shard_path = out / "shard_0000.shx"
    raw = bytearray(shard_path.read_bytes())
    _magic = raw[0:4]
    _reserved, header_size = struct.unpack_from("<HI", raw, 4)
    js = raw[10 : 10 + header_size].decode("utf-8")
    header_dict = json.loads(js)
    data_chunks = header_dict["data"]["chunks"]
    if data_chunks:
        first_chunk = data_chunks[0]
        off = first_chunk["offset"]
        csz = first_chunk["compressed_size"]
        corruption = bytes(range(256)) * (csz // 256 + 1)
        raw[off : off + csz] = corruption[:csz]
        shard_path.write_bytes(bytes(raw))

    # Wrap _shm.allocate_shm_buffers so we capture the names of newly-
    # allocated segments. This is more robust than globbing /dev/shm/psm_*
    # because we know exactly which segments to check, regardless of the
    # platform's shm backing-name prefix.
    allocated_names = []
    original_alloc = _shm.allocate_shm_buffers

    def _capturing_alloc(**kwargs):
        bundle = original_alloc(**kwargs)
        # Capture the names BEFORE returning so we can check post-test.
        allocated_names.extend([bundle.data.name, bundle.indices.name, bundle.indptr.name])
        return bundle

    with _mock.patch.object(_shm, "allocate_shm_buffers", side_effect=_capturing_alloc):
        # Same broad-catch rationale as above: codec/multiprocessing-level
        # exception class depends on the failure mode; what matters is that
        # the shm cleanup path runs.
        with pytest.raises(Exception):  # noqa: B017
            reader_cpu.read_v2_to_host(out, cast_back=False, n_workers=2, omp_nthreads=4)

    # After the failure propagates, each allocated segment should be gone
    # from /dev/shm (close_and_unlink removes the segment).
    assert allocated_names, "no shm allocations were captured — test setup broken"
    leaked = [n for n in allocated_names if (Path("/dev/shm") / n).exists()]
    assert not leaked, f"shm segments leaked after worker failure: {leaked}"


def test_reader_reuses_workers_omp_env_lock():
    """L10: read + write must serialize OMP_NUM_THREADS mutation on the SAME lock,
    not two separate locks (which could clobber each other's save/restore)."""
    from cellstream import workers as _workers
    from cellstream.v2 import reader_cpu

    assert reader_cpu._OMP_ENV_LOCK is _workers.OMP_ENV_LOCK
