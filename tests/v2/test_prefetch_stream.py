"""Prefetch (decode-ahead) streaming reader tests (#159)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import cellstream


def _toy_adata(labels, n_vars=50, seed=42):
    n = len(labels)
    rng = np.random.default_rng(seed)
    nnz = max(1, int(n * n_vars * 0.2))
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, 50, size=nnz, dtype=np.uint16)
    csr = sp.coo_matrix((vals, (rows, cols)), shape=(n, n_vars)).tocsr().astype("float32")
    csr.indices = csr.indices.astype(np.int32)
    csr.indptr = csr.indptr.astype(np.int64)
    return ad.AnnData(X=csr, obs={"gene": labels})


def _write_grouped(tmp_path, labels, *, reference=None, target_shard_bytes=4_000):
    """Grouped PACKED archive.

    The former directory-container variants were retired in #154. Prefetch-vs-serial
    byte identity and view.x()-vs-to_anndata parity are properties of the supported
    packed reader, so this helper builds that public form directly.
    """
    from cellstream.write import write_sharded

    a = _toy_adata(labels)
    out = tmp_path / "arc"
    write_sharded(
        a,
        out,
        format="v2",
        group_by="gene",
        reference=reference,
        target_shard_bytes=target_shard_bytes,
    )
    return out, a


# ---------------------------------------------------------------------------
# Task 1: iter_group_shards prefetch params + prefetch=0 guard
# ---------------------------------------------------------------------------


def test_prefetch_zero_rejects_decode_knobs(tmp_path):
    out, _ = _write_grouped(tmp_path, [f"g{i}" for i in range(40)])
    arc = cellstream.ShardedArchive(out)
    with pytest.raises(ValueError, match="prefetch"):
        list(arc.iter_group_shards(prefetch=0, cast_back=False))
    with pytest.raises(ValueError, match="prefetch"):
        list(arc.iter_group_shards(n_workers=4))  # prefetch defaults to 0
    with pytest.raises(ValueError, match="prefetch must be >= 0"):
        list(arc.iter_group_shards(prefetch=-1))


# ---------------------------------------------------------------------------
# Task 2: GroupShard pre-decoded to_anndata() contract
# ---------------------------------------------------------------------------


def test_groupshard_predecoded_returns_attached_and_rejects_kwargs():
    from cellstream.prefetch import _DecodedShard
    from cellstream.read import GroupShard

    csr = sp.csr_matrix(np.ones((3, 4), dtype="float32"))
    batch = ad.AnnData(X=csr, obs={"gene": ["a", "b", "c"]})
    ds = _DecodedShard(csr, batch, 0, 3)
    gs = GroupShard(
        archive=None,
        shard_index=0,
        global_start=0,
        global_stop=3,
        groups={"g": slice(0, 3)},
        _decoded=ds,
    )
    assert gs.x() is csr  # raw matrix, no AnnData built
    a1 = gs.to_anndata()
    assert a1 is gs.to_anndata()  # AnnData built lazily and cached
    np.testing.assert_array_equal(a1.X.toarray(), csr.toarray())
    with pytest.raises(ValueError, match="pre-decoded"):
        gs.to_anndata(cast_back=False)  # conflicting knob rejected
    with pytest.raises(ValueError, match="pre-decoded"):
        gs.x(cast_back=False)


# ---------------------------------------------------------------------------
# Task 3: _form_batches (consecutive-run batching)
# ---------------------------------------------------------------------------


def test_form_batches_consecutive_runs_and_caps():
    from cellstream.prefetch import _form_batches, _ShardPlan

    def P(i):
        return _ShardPlan(i, i * 10, i * 10 + 10, {})

    # consecutive 0..3, cap 2 -> [[0,1],[2,3]]
    b = _form_batches([P(0), P(1), P(2), P(3)], n_workers=2)
    assert [[p.shard_index for p in batch] for batch in b] == [[0, 1], [2, 3]]

    # gap at 2 (interior reference shard) splits the run; cap 16
    b = _form_batches([P(0), P(1), P(3), P(4)], n_workers=16)
    assert [[p.shard_index for p in batch] for batch in b] == [[0, 1], [3, 4]]

    # single shard
    assert [[p.shard_index for p in x] for x in _form_batches([P(5)], 16)] == [[5]]


# ---------------------------------------------------------------------------
# Task 4: _split_batch (per-shard views)
# ---------------------------------------------------------------------------


def test_split_batch_slices_per_shard():
    from cellstream.prefetch import _ShardPlan, _split_batch

    X = sp.csr_matrix(np.arange(6 * 4, dtype="float32").reshape(6, 4))
    batch = ad.AnnData(X=X)  # rows 0..5 == global rows 100..105
    plans = [_ShardPlan(0, 100, 103, {}), _ShardPlan(1, 103, 106, {})]
    parts = _split_batch(batch, plans, batch_global_start=100)
    assert [p.shard_index for p, _ in parts] == [0, 1]
    np.testing.assert_array_equal(parts[0][1].x.toarray(), X.toarray()[0:3])
    np.testing.assert_array_equal(parts[1][1].x.toarray(), X.toarray()[3:6])
    # carrier builds the AnnData lazily and caches it
    a0 = parts[0][1].to_anndata()
    assert a0 is parts[0][1].to_anndata()
    np.testing.assert_array_equal(a0.X.toarray(), X.toarray()[0:3])


# ---------------------------------------------------------------------------
# Task 5: iter_prefetched (producer thread + bounded queue)
# ---------------------------------------------------------------------------


class _FakeArchive:
    """Minimal archive: archive[start:stop].to_anndata(...) returns a view of a
    backing AnnData, recording the n_workers it was called with."""

    def __init__(self, backing):
        self._backing = backing
        self.calls = []

    def __getitem__(self, sl):
        outer = self
        start, stop = sl.start, sl.stop

        class _V:
            def to_anndata(self, **kw):
                outer.calls.append(kw.get("n_workers"))
                return outer._backing[start:stop]

        return _V()


def _backing(n_rows=8, n_vars=4):
    X = sp.csr_matrix(np.arange(n_rows * n_vars, dtype="float32").reshape(n_rows, n_vars))
    return ad.AnnData(X=X)


def test_iter_prefetched_orders_and_decodes_parallel():
    from cellstream.prefetch import _ShardPlan, iter_prefetched

    back = _backing(8)
    arc = _FakeArchive(back)
    plans = [_ShardPlan(i, i * 2, i * 2 + 2, {"g": slice(0, 2)}) for i in range(4)]
    got = list(
        iter_prefetched(arc, plans, prefetch=4, n_workers=2, decode_kwargs={"cast_back": False})
    )
    assert [p.shard_index for p, _ in got] == [0, 1, 2, 3]  # order preserved
    np.testing.assert_array_equal(got[0][1].x.toarray(), back.X.toarray()[0:2])
    np.testing.assert_array_equal(got[3][1].x.toarray(), back.X.toarray()[6:8])
    assert all(nw == 2 for nw in arc.calls)  # n_workers threaded through


def test_iter_prefetched_propagates_producer_error():
    from cellstream.prefetch import _ShardPlan, iter_prefetched

    class _Boom:
        def __getitem__(self, sl):
            class _V:
                def to_anndata(self, **kw):
                    raise RuntimeError("decode boom")

            return _V()

    plans = [_ShardPlan(0, 0, 2, {})]
    with pytest.raises(RuntimeError, match="decode boom"):
        list(iter_prefetched(_Boom(), plans, prefetch=2, n_workers=1, decode_kwargs={}))


def test_iter_prefetched_slow_consumer_no_hang():
    # Regression (Gemini #176 HIGH): a consumer slower than the producer with a
    # full queue must still receive the end-of-stream sentinel and terminate.
    import threading
    import time

    from cellstream.prefetch import _ShardPlan, iter_prefetched

    arc = _FakeArchive(_backing(16))
    plans = [_ShardPlan(i, i, i + 1, {}) for i in range(16)]
    result = []

    def run():
        for _, view in iter_prefetched(arc, plans, prefetch=1, n_workers=1, decode_kwargs={}):
            time.sleep(0.02)  # consumer slower than producer
            result.append(view)

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "iter_prefetched hung with a slow consumer"
    assert len(result) == 16


def test_iter_prefetched_early_break_joins_producer():
    import threading
    import time

    from cellstream.prefetch import _ShardPlan, iter_prefetched

    arc = _FakeArchive(_backing(20))
    plans = [_ShardPlan(i, i, i + 1, {}) for i in range(20)]
    gen = iter_prefetched(arc, plans, prefetch=2, n_workers=1, decode_kwargs={})
    next(gen)
    gen.close()  # consumer aborts early
    for _ in range(40):
        if not any(t.name == "cellstream-prefetch" for t in threading.enumerate()):
            break
        time.sleep(0.05)
    assert not any(t.name == "cellstream-prefetch" for t in threading.enumerate())


# ---------------------------------------------------------------------------
# Task 6: iter_group_shards(prefetch>=1) end-to-end vs serial
# ---------------------------------------------------------------------------


def _assert_csr_identical(a, b):
    a = a.tocsr()
    b = b.tocsr()
    assert a.dtype == b.dtype and a.indices.dtype == b.indices.dtype
    np.testing.assert_array_equal(a.indptr, b.indptr)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.data, b.data)


@pytest.mark.parametrize("cast_back", [True, False])
def test_prefetch_byte_identical_to_serial(tmp_path, cast_back):
    labels = [f"g{i // 3}" for i in range(120)]  # ~40 groups, several shards
    out, _ = _write_grouped(tmp_path, labels)

    serial = []
    for gs in cellstream.ShardedArchive(out).iter_group_shards():
        serial.append((gs.shard_index, dict(gs.groups), gs.to_anndata(cast_back=cast_back)))

    pref = []
    for gs in cellstream.ShardedArchive(out).iter_group_shards(
        prefetch=4, n_workers=2, cast_back=cast_back
    ):
        pref.append((gs.shard_index, dict(gs.groups), gs.to_anndata()))

    assert [s[0] for s in serial] == [s[0] for s in pref]  # same shard order
    assert len(serial) >= 2  # exercised >1 batch
    for (si_s, g_s, a_s), (si_p, g_p, a_p) in zip(serial, pref, strict=True):
        assert si_s == si_p and g_s == g_p
        _assert_csr_identical(a_s.X, a_p.X)


def test_prefetch_reference_excluded(tmp_path):
    labels = [f"g{i // 4}" for i in range(80)]
    out, _ = _write_grouped(tmp_path, labels, reference="g0")  # g0 becomes reference
    seen = set()
    for gs in cellstream.ShardedArchive(out).iter_group_shards(prefetch=3, n_workers=2):
        seen.update(gs.groups.keys())
        gs.to_anndata()  # decodes fine
    assert "g0" not in seen  # reference label excluded


def test_prefetch_x_outlives_anndata_wrapper(tmp_path):
    # Regression (Gemini #176): extracting X and discarding the AnnData wrapper +
    # GroupShard must keep the shm-backed buffers alive (batch pinned to the csr).
    import gc

    out, _ = _write_grouped(tmp_path, [f"g{i // 3}" for i in range(60)])
    it = cellstream.ShardedArchive(out).iter_group_shards(prefetch=2, n_workers=2, cast_back=False)
    gs = next(it)
    X = gs.to_anndata().X  # keep ONLY X; drop the AnnData + GroupShard + iterator
    expected = X.toarray().copy()
    del gs, it
    gc.collect()
    np.testing.assert_array_equal(X.toarray(), expected)  # buffers still valid
    assert X.indices.dtype == np.uint16  # narrow index dtype preserved


def test_prefetch_single_shard_and_prefetch_gt_nshards(tmp_path):
    out, _ = _write_grouped(
        tmp_path, [f"g{i}" for i in range(6)], target_shard_bytes=10_000_000
    )  # one shard
    got = list(cellstream.ShardedArchive(out).iter_group_shards(prefetch=99, n_workers=8))
    assert len(got) == 1
    assert got[0].to_anndata().n_obs == 6


# ---------------------------------------------------------------------------
# shm matrix-level bundle pin (attach_buffer_cleanup_matrix)
# ---------------------------------------------------------------------------


class _FakeBundle:
    def __init__(self):
        self.unlinked = False
        self.closed = False

    def unlink_segments(self):
        self.unlinked = True

    def close_and_unlink(self):
        self.closed = True


def test_attach_buffer_cleanup_matrix_csr():
    import gc

    from cellstream.shm import attach_buffer_cleanup_matrix

    b = _FakeBundle()
    X = sp.csr_matrix(np.ones((2, 3), dtype="float32"))
    attach_buffer_cleanup_matrix(X, b)
    assert X._cellstream_shm_bundle is b
    X.close_shared_memory()
    assert b.closed is True
    del X
    gc.collect()
    assert b.unlinked is True  # weakref.finalize fired unlink_segments on GC


def test_attach_buffer_cleanup_matrix_dense():
    import gc

    from cellstream.shm import _ShmDenseArray, attach_buffer_cleanup_matrix

    b = _FakeBundle()
    X = np.ones((2, 3), dtype="float32").view(_ShmDenseArray)
    attach_buffer_cleanup_matrix(X, b)
    assert X._cellstream_shm_bundle is b
    del X
    gc.collect()
    assert b.unlinked is True


# ---------------------------------------------------------------------------
# ShardedArchiveView.x() — lazy raw-matrix read (no AnnData)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cast_back", [True, False])
def test_view_x_matches_to_anndata_X(tmp_path, cast_back):
    out, _ = _write_grouped(tmp_path, [f"g{i // 3}" for i in range(60)])
    arc = cellstream.ShardedArchive(out)
    n = arc.n_obs
    x = arc[0:n].x(cast_back=cast_back)
    a = arc[0:n].to_anndata(cast_back=cast_back)
    _assert_csr_identical(x, a.X)


def test_view_x_dense_and_shm_outlives_wrapper(tmp_path):
    import gc

    out, _ = _write_grouped(tmp_path, [f"g{i // 3}" for i in range(60)])
    arc = cellstream.ShardedArchive(out)
    n = arc.n_obs
    # dense container returns an ndarray equal to to_anndata(dense).X
    xd = arc[0:n].x(container="dense")
    ad_dense = arc[0:n].to_anndata(container="dense")
    np.testing.assert_array_equal(np.asarray(xd), np.asarray(ad_dense.X))
    # shm-backed CSR survives dropping the view (bundle pinned to the matrix)
    view = arc[0:n]
    X = view.x(cast_back=False, output_backing="shm")
    expected = X.toarray().copy()
    del view, arc
    gc.collect()
    np.testing.assert_array_equal(X.toarray(), expected)
    assert X.indices.dtype == np.uint16


# ---------------------------------------------------------------------------
# GroupShard.x() — per-shard raw matrix accessor (prefetch + lazy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cast_back", [True, False])
@pytest.mark.parametrize("prefetch", [0, 4])
def test_groupshard_x_matches_to_anndata_X(tmp_path, cast_back, prefetch):
    out, _ = _write_grouped(tmp_path, [f"g{i // 3}" for i in range(120)])

    # reference: serial per-shard to_anndata().X
    ref = {}
    for gs in cellstream.ShardedArchive(out).iter_group_shards():
        ref[gs.shard_index] = gs.to_anndata(cast_back=cast_back).X

    # prefetch=0 (lazy) rejects non-default iterator knobs; pass n_workers/cast_back
    # only in prefetch mode and supply cast_back per-shard to gs.x() when lazy.
    extra = {} if prefetch == 0 else {"n_workers": 2, "cast_back": cast_back}
    it = cellstream.ShardedArchive(out).iter_group_shards(prefetch=prefetch, **extra)
    seen = 0
    for gs in it:
        X = gs.x(cast_back=cast_back) if prefetch == 0 else gs.x()
        _assert_csr_identical(X, ref[gs.shard_index])
        # gs.groups slices apply to the raw matrix exactly as to to_anndata().X
        a = gs.to_anndata() if prefetch else gs.to_anndata(cast_back=cast_back)
        for _label, sl in gs.groups.items():
            np.testing.assert_array_equal(X[sl].toarray(), a.X[sl].toarray())
        seen += 1
    assert seen >= 2


def test_groupshard_x_dense_container(tmp_path):
    out, _ = _write_grouped(tmp_path, [f"g{i // 3}" for i in range(60)])
    # prefetch dense: x() returns the dense ndarray. Bind the AnnData (a, not a
    # temporary) so its shm backing outlives the comparison — a temporary dense
    # AnnData's SharedMemory is munmapped on GC (pre-existing dense-AnnData
    # lifetime bug, orthogonal to this accessor; reproduces on the untouched
    # ShardedArchive.to_anndata(container="dense")).
    for gs in cellstream.ShardedArchive(out).iter_group_shards(
        prefetch=2, n_workers=2, container="dense"
    ):
        X = gs.x()
        a = gs.to_anndata()
        assert isinstance(X, np.ndarray)
        np.testing.assert_array_equal(np.asarray(X), np.asarray(a.X))
    # lazy dense
    for gs in cellstream.ShardedArchive(out).iter_group_shards():
        X = gs.x(container="dense")
        a = gs.to_anndata(container="dense")
        assert isinstance(X, np.ndarray)
        np.testing.assert_array_equal(np.asarray(X), np.asarray(a.X))


def test_groupshard_x_prefetch_dense_outlives_groupshard(tmp_path):
    # prefetch dense: extracting X and dropping the GroupShard keeps the dense
    # buffer mapped (the dense view is pinned to the batch via _cellstream_batch_ref).
    import gc

    out, _ = _write_grouped(tmp_path, [f"g{i // 3}" for i in range(60)])
    gs = next(
        cellstream.ShardedArchive(out).iter_group_shards(prefetch=2, n_workers=2, container="dense")
    )
    X = gs.x()
    expected = np.asarray(X).copy()
    del gs
    gc.collect()
    np.testing.assert_array_equal(np.asarray(X), expected)


def test_groupshard_x_prefetch_rejects_kwargs(tmp_path):
    out, _ = _write_grouped(tmp_path, [f"g{i // 3}" for i in range(60)])
    gs = next(cellstream.ShardedArchive(out).iter_group_shards(prefetch=2, n_workers=2))
    with pytest.raises(ValueError, match="pre-decoded"):
        gs.x(cast_back=False)


def test_groupshard_x_lazy_outlives_groupshard(tmp_path):
    import gc

    out, _ = _write_grouped(tmp_path, [f"g{i // 3}" for i in range(60)])
    gs = next(cellstream.ShardedArchive(out).iter_group_shards())  # lazy
    X = gs.x(cast_back=False, output_backing="shm")
    expected = X.toarray().copy()
    del gs
    gc.collect()
    np.testing.assert_array_equal(X.toarray(), expected)
    assert X.indices.dtype == np.uint16  # narrow dtype preserved
