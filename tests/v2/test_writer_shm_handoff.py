# tests/v2/test_writer_shm_handoff.py
"""Tests for the v2 writer shared-memory source handoff (#76)."""

from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellstream import ShardedArchive, write_sharded
from tests._archive_helpers import archive_manifest, write_archive, write_directory


def _make_adata(seed=1, n_obs=600, n_vars=400, density=0.06):
    """Float32 CSR AnnData with integer-valued data (lossless to uint16)."""
    rng = np.random.default_rng(seed)
    raw = sp.random(n_obs, n_vars, density=density, format="csr", random_state=rng)
    csr = sp.csr_matrix(raw.shape, dtype=np.float32)
    csr.data = np.ceil(raw.data * 100).astype(np.float32)  # integer-valued, <= ~100
    csr.indices = raw.indices.astype(np.int32)
    csr.indptr = raw.indptr.astype(np.int64)
    return ad.AnnData(X=csr)


def test_encode_shard_streams_roundtrips(tmp_path):
    """_encode_shard_streams writes a valid .shx that reads back to the input rows."""
    from cellstream.v2.writer import _encode_shard_streams

    a = _make_adata(n_obs=50, n_vars=20, density=0.2)
    X = a.X
    n_rows, n_vars = X.shape
    data_u16 = X.data.astype(np.uint16)
    indices_u16 = X.indices.astype(np.uint16)
    indptr_i64 = X.indptr.astype(np.int64)

    res = _encode_shard_streams(
        tmp_path, 0, data_u16, indices_u16, indptr_i64, n_rows, n_vars, "bitshuffle"
    )
    assert res == {"shard_idx": 0, "nnz": int(X.nnz)}
    assert (tmp_path / "shard_0000.shx").exists()


def test_encode_one_shard_from_shm_matches_serial(tmp_path):
    """A shard encoded by the shm worker is byte-identical to the same shard
    range encoded by the serial path."""
    if sys.platform != "linux":
        pytest.skip("SharedMemory handoff path is Linux-only")
    from cellstream import shm as _shm
    from cellstream.v2.writer import _encode_one_shard, _encode_one_shard_from_shm

    a = _make_adata(n_obs=120, n_vars=40, density=0.2)
    X = a.X
    n_vars = X.shape[1]
    r0, r1 = 30, 90  # a middle shard range

    # --- Serial reference: slice in-parent, encode to dir_serial ---
    dir_serial = tmp_path / "serial"
    dir_serial.mkdir()
    sub = X[r0:r1]
    _encode_one_shard(
        (
            dir_serial,
            0,
            sub.data,
            sub.indices,
            sub.indptr,
            r1 - r0,
            n_vars,
            "bitshuffle",
            "uint16",
            "uint16",
        )
    )

    # --- shm path: fill the FULL source into shm, encode the same range ---
    dir_shm = tmp_path / "shm"
    dir_shm.mkdir()
    total_nnz = int(X.data.size)
    n_indptr = X.shape[0] + 1
    bundle = _shm.allocate_shm_buffers(
        data_nbytes=max(1, total_nnz * 2),
        indices_nbytes=max(1, total_nnz * 2),
        indptr_nbytes=max(1, n_indptr * 8),
    )
    try:
        np.ndarray((total_nnz,), dtype=np.uint16, buffer=bundle.data.buf)[:] = X.data
        np.ndarray((total_nnz,), dtype=np.uint16, buffer=bundle.indices.buf)[:] = X.indices
        np.ndarray((n_indptr,), dtype=np.int64, buffer=bundle.indptr.buf)[:] = X.indptr
        _encode_one_shard_from_shm(
            (
                dir_shm,
                0,
                bundle.data.name,
                bundle.indices.name,
                bundle.indptr.name,
                total_nnz,
                n_indptr,
                r0,
                r1,
                n_vars,
                "bitshuffle",
                "uint16",
                "uint16",
            )
        )
    finally:
        bundle.close_and_unlink()

    assert (dir_shm / "shard_0000.shx").read_bytes() == (dir_serial / "shard_0000.shx").read_bytes()


def test_can_use_shm_true_when_room(monkeypatch):
    import platform as _plat

    from cellstream import shm as _shm
    from cellstream.v2 import writer as _writer

    monkeypatch.setattr(_plat, "system", lambda: "Linux")
    monkeypatch.setattr(_shm, "_free_bytes", lambda _p: 10 * 2**30)  # 10 GiB free
    assert (
        _writer._can_use_shm(
            total_nnz=1_000_000, n_indptr=100_001, data_dtype=np.uint16, indices_dtype=np.uint16
        )
        is True
    )


def test_can_use_shm_false_when_insufficient(monkeypatch):
    import platform as _plat

    from cellstream import shm as _shm
    from cellstream.v2 import writer as _writer

    monkeypatch.setattr(_plat, "system", lambda: "Linux")
    monkeypatch.setattr(_shm, "_free_bytes", lambda _p: 1024)  # 1 KiB free
    assert (
        _writer._can_use_shm(
            total_nnz=1_000_000, n_indptr=100_001, data_dtype=np.uint16, indices_dtype=np.uint16
        )
        is False
    )


def test_can_use_shm_false_when_not_linux(monkeypatch):
    import platform as _plat

    from cellstream import shm as _shm
    from cellstream.v2 import writer as _writer

    monkeypatch.setattr(_plat, "system", lambda: "Darwin")
    monkeypatch.setattr(_shm, "_free_bytes", lambda _p: 10 * 2**30)
    assert (
        _writer._can_use_shm(
            total_nnz=1_000_000, n_indptr=100_001, data_dtype=np.uint16, indices_dtype=np.uint16
        )
        is False
    )


def _shard_bytes(d):
    return {p.name: p.read_bytes() for p in sorted(d.glob("shard_*.shx"))}


@pytest.mark.skipif(sys.platform != "linux", reason="shm handoff is Linux-only")
def test_parallel_write_uses_shm_allocation(tmp_path, monkeypatch):
    """A parallel v2 write (n_workers>1, n_shards>1) allocates a shm source
    bundle; a serial write does not."""
    import unittest.mock as _mock

    from cellstream import shm as _shm

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    monkeypatch.setattr(_shm, "_free_bytes", lambda _p: 1 << 40)  # force shm path (1 TiB)
    a = _make_adata(n_obs=800, n_vars=300, density=0.05)

    calls = {"n": 0}
    orig = _shm.allocate_shm_buffers

    def _spy(**kw):
        calls["n"] += 1
        return orig(**kw)

    # PUBLIC writer (#154 part B): which branch runs is decided by n_workers, and
    # n_workers is one of the two defaults that differ between the public and internal
    # writers -- so on write_directory this stays green even if write_sharded stops
    # forwarding it. Neither call reads a loose member, only the allocation spy.
    with _mock.patch.object(_shm, "allocate_shm_buffers", side_effect=_spy):
        write_archive(a, tmp_path / "par", n_shards=4, n_workers=4)
    assert calls["n"] == 1, "parallel write must allocate exactly one shm bundle"

    calls["n"] = 0
    with _mock.patch.object(_shm, "allocate_shm_buffers", side_effect=_spy):
        write_archive(a, tmp_path / "ser", n_shards=4, n_workers=1)
    assert calls["n"] == 0, "serial write must not allocate shm"


@pytest.mark.skipif(sys.platform != "linux", reason="shm handoff is Linux-only")
@pytest.mark.parametrize("codec_name", ["zstd-1-u16u16-bitshuffle", "v2-byte-filter"])
def test_parallel_shards_byte_identical_to_serial(tmp_path, codec_name):
    """Shard files from a parallel (shm) write are byte-identical to a serial write."""
    a = _make_adata(n_obs=800, n_vars=300, density=0.05)
    write_directory(
        a,
        tmp_path / "ser",
        n_shards=4,
        n_workers=1,
        codec=codec_name,
    )
    write_directory(
        a,
        tmp_path / "par",
        n_shards=4,
        n_workers=4,
        codec=codec_name,
    )
    assert _shard_bytes(tmp_path / "par") == _shard_bytes(tmp_path / "ser")


@pytest.mark.skipif(sys.platform != "linux", reason="shm handoff is Linux-only")
@pytest.mark.parametrize("codec_name", ["zstd-1-u16u16-bitshuffle", "v2-byte-filter"])
def test_parallel_roundtrip(tmp_path, codec_name):
    """A parallel-written archive reads back equal to the source.

    PACKED since #154 part B3a (route A): write_packed stages through write_v2_archive
    (packed/writer.py:192) with the same n_workers, so the shm handoff under test is
    identical -- only the container differs, and _pack_directory adds a pure byte concat
    with no shm of its own. Its sibling test_parallel_shards_byte_identical_to_serial
    stays on write_directory: that one COMPARES loose shard bytes (route B).
    """
    a = _make_adata(n_obs=800, n_vars=300, density=0.05)
    out = tmp_path / "par"
    write_archive(a, out, n_shards=4, n_workers=4, codec=codec_name)
    back = ShardedArchive(out).to_anndata(cast_back=True, n_workers=2)
    got = back.X.tocsr()
    want = a.X.tocsr()
    assert got.shape == want.shape
    assert np.array_equal(got.toarray(), want.toarray().astype(got.dtype))


@pytest.mark.skipif(sys.platform != "linux", reason="shm handoff is Linux-only")
def test_parallel_group_by_roundtrip(tmp_path):
    """Group-aware parallel write reads back correctly. Inverts the group
    permutation via the writer's `_orig_row` obs column and asserts the FULL
    matrix matches — a mis-sliced indptr/data range would corrupt values while
    preserving nnz, so an nnz-only check is too weak for the shm slicing."""
    a = _make_adata(n_obs=1000, n_vars=300, density=0.05)
    a.obs["grp"] = (np.arange(a.n_obs) % 5).astype(str)
    out = tmp_path / "grp"
    # Small target so groups land in multiple shards (n_shards > 1) and the
    # parallel shm path is actually exercised.
    write_archive(
        a,
        out,
        group_by="grp",
        n_workers=4,
        target_shard_bytes=8 * 1024,
    )
    assert archive_manifest(out)["n_shards"] > 1, (
        "group_by test must produce >1 shard to exercise the parallel path; "
        "reduce target_shard_bytes if this fails"
    )
    back = ShardedArchive(out).to_anndata(cast_back=True, n_workers=2)
    assert back.n_obs == a.n_obs
    # _orig_row[pos] = source row stored at position pos. inv = argsort(_orig_row)
    # reorders back into source order: (back[inv])[k] is source row k.
    orig = back.obs["_orig_row"].to_numpy().astype(np.int64)
    inv = np.argsort(orig)
    got = back.X.tocsr()[inv]
    want = a.X.tocsr()
    assert np.array_equal(got.toarray(), want.toarray().astype(got.dtype))


@pytest.mark.skipif(sys.platform != "linux", reason="shm leak detection requires /dev/shm")
def test_no_shm_leak_on_worker_death(tmp_path, monkeypatch):
    """A killed worker raises BrokenProcessPool, leaks no /dev/shm segment, publishes no
    archive, and releases the write lock.

    This is the **packed** (public ``write_sharded``) half of the fail-fast contract; the
    directory half is tests/v2/test_fail_fast_worker_death.py. The no-archive and lock-release
    assertions were added in #154 part B, which deleted
    tests/test_write_fail_fast_worker_death.py -- the only place that pinned them, and it pinned
    them for the two v1 write paths.

    Asserting them here as well as on the directory writer is the point, because the two paths
    use different locks: ``packed.writer.write_packed`` takes its own outer lock on
    ``<archive>.lock`` (packed/writer.py:202) and then calls the directory writer with
    ``unsafe_no_lock=True``, so re-acquiring the directory's ``.cellstream.lock`` says nothing
    about the lock that survives #154 part B3. Verified empirically before being asserted:
    after a killed worker the parent dir holds only ``killed.shad.lock`` -- no archive -- and
    that lock re-acquires cleanly.
    """
    import unittest.mock as _mock
    from concurrent.futures.process import BrokenProcessPool
    from pathlib import Path

    from cellstream import lock as _lock
    from cellstream import shm as _shm

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    monkeypatch.setattr(_shm, "_free_bytes", lambda _p: 1 << 40)  # force shm path (1 TiB)
    a = _make_adata(n_obs=800, n_vars=300, density=0.05)

    allocated = []
    orig = _shm.allocate_shm_buffers

    def _capturing(**kw):
        b = orig(**kw)
        allocated.extend([b.data.name, b.indices.name, b.indptr.name])
        return b

    out = tmp_path / "killed.shad"
    monkeypatch.setenv("CELLSTREAM_TEST_KILL_SHARD_IDX", "1")  # kill the 2nd shard's worker
    with _mock.patch.object(_shm, "allocate_shm_buffers", side_effect=_capturing):
        with pytest.raises(BrokenProcessPool):
            write_sharded(a, out, format="v2", n_shards=4, n_workers=4)

    assert allocated, "no shm allocation captured — test setup broken"
    leaked = [n for n in allocated if (Path("/dev/shm") / n).exists()]
    assert not leaked, f"shm segments leaked after worker death: {leaked}"

    # No partial archive. The packed file is ATOMICALLY PUBLISHED last: _pack_directory writes
    # a uuid-named temp beside the destination and os.replace()s it into place
    # (packed/writer.py:184), so a failure before that leaves no destination at all and its
    # ABSENCE is the whole assertion -- unlike the directory writer, where header.h5ad and
    # obs.parquet survive and manifest.json is the oracle.
    assert not out.exists(), (
        f"failed write published an archive; parent dir holds "
        f"{sorted(p.name for p in tmp_path.iterdir())}"
    )
    # And the PACKED write lock must have been released, not left held by the dead pool.
    with _lock.acquire_write_lock(out.with_suffix(out.suffix + ".lock"), op="write"):
        pass  # re-acquired cleanly -> released on failure


@pytest.mark.skipif(sys.platform != "linux", reason="shm handoff is Linux-only")
def test_omp_env_restored_after_parallel_write(tmp_path):
    """A parallel v2 write restores the parent's OMP_NUM_THREADS afterward."""
    import os

    a = _make_adata(n_obs=400, n_vars=200, density=0.05)
    sentinel = "13"
    prev = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = sentinel
        write_directory(a, tmp_path / "omp", n_shards=4, n_workers=4)
        assert os.environ.get("OMP_NUM_THREADS") == sentinel
    finally:
        if prev is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev


def test_falls_back_to_serial_when_shm_insufficient(tmp_path, monkeypatch):
    """When /dev/shm can't hold the source, the parallel write warns and falls
    back to serial (no shm allocation), still producing a correct archive."""
    import unittest.mock as _mock

    from cellstream import shm as _shm

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    a = _make_adata(n_obs=400, n_vars=200, density=0.05)
    monkeypatch.setattr(_shm, "_free_bytes", lambda _p: 1024)  # 1 KiB free

    calls = {"n": 0}
    orig = _shm.allocate_shm_buffers

    def _spy(**kw):
        calls["n"] += 1
        return orig(**kw)

    with _mock.patch.object(_shm, "allocate_shm_buffers", side_effect=_spy):
        with pytest.warns(UserWarning, match="falling back to serial"):
            write_archive(
                a,
                tmp_path / "fallback",
                n_shards=4,
                n_workers=4,
            )
    assert calls["n"] == 0, "fallback must not allocate shm"
    back = ShardedArchive(tmp_path / "fallback").to_anndata(cast_back=True, n_workers=1)
    assert back.n_obs == a.n_obs
    assert int(back.X.tocsr().nnz) == int(a.X.tocsr().nnz)


@pytest.mark.skipif(sys.platform != "linux", reason="shm handoff is Linux-only")
def test_nongrouped_oserror_falls_back_to_serial(tmp_path, monkeypatch):
    """An OSError from allocate_shm_buffers (e.g. /dev/shm filled AFTER the
    preflight — a TOCTOU race) must fall back to the serial path, not crash the
    write (ultrareview L11)."""
    import unittest.mock as _mock

    from cellstream import shm as _shm

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")  # shm path is the pure-Python fallback
    monkeypatch.setattr(_shm, "_free_bytes", lambda _p: 1 << 40)  # pass the preflight
    a = _make_adata(n_obs=400, n_vars=200, density=0.05)
    calls = {"n": 0}

    def _boom(**kw):
        calls["n"] += 1
        raise OSError("simulated /dev/shm exhaustion after the preflight")

    with _mock.patch.object(_shm, "allocate_shm_buffers", side_effect=_boom):
        with pytest.warns(UserWarning, match="falling back to serial"):
            write_archive(a, tmp_path / "out", n_shards=4, n_workers=4)
    assert calls["n"] >= 1, "parallel path must have attempted the shm allocation"
    back = ShardedArchive(tmp_path / "out").to_anndata(cast_back=True, n_workers=1)
    assert back.n_obs == a.n_obs
    assert int(back.X.tocsr().nnz) == int(a.X.tocsr().nnz)


@pytest.mark.skipif(sys.platform != "linux", reason="shm handoff is Linux-only")
def test_grouped_inmem_oserror_falls_back_to_serial(tmp_path, monkeypatch):
    """Same TOCTOU fallback for the grouped in-mem parallel gather path (L11)."""
    import unittest.mock as _mock

    from cellstream import shm as _shm

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")  # shm path is the pure-Python fallback
    monkeypatch.setattr(_shm, "_free_bytes", lambda _p: 1 << 40)
    a = _make_adata(n_obs=300, n_vars=120, density=0.08)
    a.obs["target_gene"] = [f"g{i % 6}" for i in range(a.n_obs)]
    calls = {"n": 0}

    def _boom(**kw):
        calls["n"] += 1
        raise OSError("simulated /dev/shm exhaustion after the preflight")

    with _mock.patch.object(_shm, "allocate_shm_buffers", side_effect=_boom):
        with pytest.warns(UserWarning, match="falling back to serial"):
            write_archive(
                a,
                tmp_path / "gout",
                group_by="target_gene",
                n_workers=4,
                target_shard_bytes=4096,
            )
    assert calls["n"] >= 1, "grouped parallel path must have attempted the shm allocation"
    back = ShardedArchive(tmp_path / "gout").to_anndata(cast_back=True, n_workers=1)
    assert back.n_obs == a.n_obs
    assert int(back.X.tocsr().nnz) == int(a.X.tocsr().nnz)
