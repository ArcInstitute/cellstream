"""PR4 (#219): the CellBlockSource adapters in isolation."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.cell.reader import open_cell
from cellstream.cell.source import (
    BackedH5adSource,
    CellArchiveSource,
    DenseH5adSource,
    InMemoryAnnDataSource,
    open_cell_source,
)
from cellstream.cell.writer import write_cell_archive

from .conftest import make_cell_adata


def _assert_blocks_reconstruct(src, X_expected):
    """row_nnz + every read_block must reconstruct the source matrix exactly."""
    np.testing.assert_array_equal(src.row_nnz(), np.diff(X_expected.indptr))
    for lo, hi in [(0, src.n_obs), (0, 1), (1, 3), (src.n_obs - 1, src.n_obs)]:
        np.testing.assert_allclose(src.read_block(lo, hi).toarray(), X_expected[lo:hi].toarray())


class _SliceSpy:
    """Wraps an h5py Dataset, recording the row-height of every hyperslab read. Delegates dtype /
    shape / __getitem__ so DenseH5adSource treats it exactly like the real dataset."""

    def __init__(self, ds):
        self._ds = ds
        self.dtype = ds.dtype
        self.shape = ds.shape
        self.heights = []

    def __getitem__(self, key):
        if isinstance(key, slice):
            lo = key.start or 0
            hi = key.stop if key.stop is not None else self.shape[0]
            self.heights.append(hi - lo)
        return self._ds[key]


def test_in_memory_source(cell_adata):
    src = open_cell_source(cell_adata)
    assert isinstance(src, InMemoryAnnDataSource)
    assert src.is_in_memory is True
    assert (src.n_obs, src.n_vars) == cell_adata.shape
    _assert_blocks_reconstruct(src, cell_adata.X.tocsr())
    src.close()


def test_backed_h5ad_source(tmp_path):
    a = make_cell_adata()
    p = tmp_path / "src.h5ad"
    a.write_h5ad(p)
    src = open_cell_source(p)
    assert isinstance(src, BackedH5adSource)
    assert src.is_in_memory is False
    _assert_blocks_reconstruct(src, a.X.tocsr())
    src.close()


def test_cell_archive_source(tmp_path, cell_adata):
    arc = tmp_path / "a.shad"
    write_cell_archive(cell_adata, arc)
    src = open_cell_source(arc)
    assert isinstance(src, CellArchiveSource)
    assert src.is_in_memory is False
    # read_block now gathers at the archive's OWN on-disk value width (#277), so data_dtype()
    # reports that -- still exactly what the BLOCKS contain, which is what
    # use_rust_for_cell_encode and Rust's match arm see. Reporting the reader's default float32
    # instead rounded integer counts above 2**24 and re-encoded them as truth.
    from cellstream.cell.reader import CellStore

    _s = CellStore(arc)
    try:
        _on_disk = np.dtype(_s.manifest["value_dtype_on_disk"])
    finally:
        _s.close()  # else a failure here leaks the mmap export into later tests
    assert src.data_dtype() == _on_disk
    _assert_blocks_reconstruct(src, cell_adata.X.tocsr())
    src.close()


@pytest.mark.parametrize("kind", ["memory", "h5ad", "dense_h5ad", "archive"])
def test_declared_dtypes_match_what_read_block_yields(tmp_path, cell_adata, kind):
    """THE SOURCE CONTRACT, and it is load-bearing: use_rust_for_cell_encode and
    _plan_cell_codec both decide from source.data_dtype() / .index_dtype(), while Rust's match
    arm sees the dtype of the array actually handed to it -- i.e. read_block's. If the two ever
    drift, the gate decides on a dtype Rust never sees. That IS the #212 r2 bug class (Gemini),
    which Codex re-found in the gate on #220. Pin it."""
    if kind == "memory":
        src = open_cell_source(cell_adata)
    elif kind == "h5ad":
        p = tmp_path / "s.h5ad"
        cell_adata.write_h5ad(p)
        src = open_cell_source(p)
    elif kind == "dense_h5ad":
        p = tmp_path / "d.h5ad"
        dense = cell_adata.copy()
        dense.X = cell_adata.X.toarray()  # dense X -> DenseH5adSource
        dense.write_h5ad(p)
        src = open_cell_source(p)
    else:
        p = tmp_path / "s.shad"
        write_cell_archive(cell_adata, p)
        src = open_cell_source(p)
    try:
        blk = src.read_block(0, min(4, src.n_obs))
        assert blk.data.dtype == src.data_dtype(), "data_dtype() lies about read_block's output"
        assert blk.indices.dtype == src.index_dtype(), (
            "index_dtype() lies about read_block's output"
        )
    finally:
        src.close()


@pytest.mark.parametrize("kind", ["memory", "h5ad", "dense_h5ad", "archive"])
def test_obs_var_uns_are_properties_not_methods(tmp_path, cell_adata, kind):
    """`obs`/`var`/`uns` are PROPERTIES, not methods (module docstring): cellstream.grouping
    duck-types on `.obs` (Task 3, #219 site 1 -- _resolve_storage_order now passes a source
    straight into grouping.resolve_reference). A `<bound method>` where a DataFrame belongs
    would break `grouping` in a confusing way (e.g. AttributeError on `.columns`), and no
    existing test read these attributes directly. Pin the contract for all three adapters."""
    if kind == "memory":
        src = open_cell_source(cell_adata)
    elif kind == "h5ad":
        p = tmp_path / "s.h5ad"
        cell_adata.write_h5ad(p)
        src = open_cell_source(p)
    elif kind == "dense_h5ad":
        p = tmp_path / "d.h5ad"
        dense = cell_adata.copy()
        dense.X = cell_adata.X.toarray()  # dense X -> DenseH5adSource
        dense.write_h5ad(p)
        src = open_cell_source(p)
    else:
        p = tmp_path / "s.shad"
        write_cell_archive(cell_adata, p)
        src = open_cell_source(p)
    try:
        assert isinstance(src.obs, pd.DataFrame)
        assert isinstance(src.var, pd.DataFrame)
        assert isinstance(src.uns, dict)
    finally:
        src.close()


def test_backed_h5ad_reports_dropped_fields(tmp_path):
    """A streaming source must still WARN when layout='cell' drops .layers/.raw. Returning
    a hard-coded empty list (or raw=None) would make a backed h5ad drop them SILENTLY while the
    in-memory path warns -- a regression no in-memory test could see."""
    a = make_cell_adata()
    a.layers["counts"] = a.X.copy()
    a.obsm["X_pca"] = np.zeros((a.n_obs, 3), dtype=np.float32)
    p = tmp_path / "rich.h5ad"
    a.write_h5ad(p)
    src = open_cell_source(p)
    assert set(src.dropped_fields()) == {"layers"}
    src.close()


def test_dense_h5ad_streams_via_dense_source(tmp_path):
    """A dense-X h5ad (encoding-type == "array") now streams via DenseH5adSource (#225), NOT the
    in-memory fallback -- the O(block) path covers dense inputs."""
    a = ad.AnnData(X=np.arange(20, dtype=np.float32).reshape(4, 5))
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p)
    src = open_cell_source(p)
    assert isinstance(src, DenseH5adSource)
    assert src.is_in_memory is False
    src.close()


def test_csc_h5ad_falls_back_to_in_memory(tmp_path):
    """A CSC-X h5ad has f["X"] as an HDF5 GROUP (encoding-type "csc_matrix"), not a dense Dataset,
    so it must NOT route to DenseH5adSource -- it keeps the in-memory fallback (no regression)."""
    a = ad.AnnData(X=sp.csc_matrix(np.arange(20, dtype=np.float32).reshape(4, 5)))
    p = tmp_path / "c.h5ad"
    a.write_h5ad(p)
    src = open_cell_source(p)
    assert isinstance(src, InMemoryAnnDataSource)
    assert src.is_in_memory is True
    src.close()


def test_cell_archive_source_from_path_owns_and_closes_its_store(tmp_path, cell_adata):
    """A path input: open_cell_source opened the CellStore itself -> it owns it -> close()
    must close it (the un-buggy direction of the #219 final-review Important finding)."""
    arc = tmp_path / "a.shad"
    write_cell_archive(cell_adata, arc)
    src = open_cell_source(arc)
    assert isinstance(src, CellArchiveSource)
    src.close()
    with pytest.raises(ValueError):
        src._s.gather_rows([0])  # the underlying CellStore's mmap is gone


def test_write_cell_archive_does_not_close_caller_supplied_store(tmp_path, cell_adata):
    """I2 (PR4 final review, Important): write_cell_archive(CellStore, ...) must NOT close the
    caller's store. open_cell_source(CellStore) -> CellArchiveSource does not own a store handed
    to it directly, so write_cell_archive's `finally: source.close()` must be a no-op for it."""
    src_path = tmp_path / "a.shad"
    write_cell_archive(cell_adata, src_path)
    store = open_cell(src_path)

    out_path = tmp_path / "b.shad"
    write_cell_archive(store, out_path)

    # the caller's store must still be usable after the write completed
    got = store.gather_rows([0, 1, 2])
    assert got.shape == (3, cell_adata.n_vars)
    store.close()


def test_shard_layout_shad_raises(tmp_path):
    import cellstream

    a = make_cell_adata()
    out = tmp_path / "shard.shad"
    cellstream.write_sharded(a, out, n_shards=2)
    with pytest.raises(ValueError, match="shard-layout"):
        open_cell_source(out)


def test_dense_h5ad_source_reconstructs(tmp_path):
    """DenseH5adSource (direct construction): row_nnz + read_block reconstruct the dense matrix;
    is_in_memory is False; declared dtypes are float32 data + int32 indices (the source contract:
    sp.csr_matrix(dense) yields int32, NOT choose_index_dtype)."""
    a = make_cell_adata()
    a.X = a.X.toarray().astype(np.float32)  # dense-X h5ad -> encoding-type "array"
    p = tmp_path / "dense.h5ad"
    a.write_h5ad(p)

    src = DenseH5adSource(p)
    try:
        assert src.is_in_memory is False
        assert (src.n_obs, src.n_vars) == a.shape
        assert src.data_dtype() == np.dtype(np.float32)
        assert src.index_dtype() == np.dtype(np.int32)
        _assert_blocks_reconstruct(src, sp.csr_matrix(a.X))
    finally:
        src.close()


def test_dense_h5ad_reports_dropped_fields(tmp_path):
    """A dense streaming source must still WARN when layout='cell' drops .layers -- drop-
    warning parity with BackedH5adSource and the in-memory path."""
    a = make_cell_adata()
    a.X = a.X.toarray().astype(np.float32)
    a.layers["counts"] = a.X.copy()
    a.obsm["X_pca"] = np.zeros((a.n_obs, 3), dtype=np.float32)
    p = tmp_path / "rich_dense.h5ad"
    a.write_h5ad(p)
    src = open_cell_source(p)
    assert isinstance(src, DenseH5adSource)
    assert set(src.dropped_fields()) == {"layers"}
    src.close()


def test_dense_source_reads_are_O_block(tmp_path, monkeypatch):
    """Structural O(block) proof (AC#2), matching the repo's AC#5 style: DenseH5adSource NEVER
    reads the full matrix in one shot. row_nnz() scans in bounded blocks (<= the byte budget,
    < n_obs); read_block(lo, hi) reads exactly one (hi-lo) hyperslab."""
    from cellstream.cell import format as fmt
    from cellstream.cell.source import DenseH5adSource

    a = make_cell_adata(n_obs=240, n_vars=60)
    a.X = a.X.toarray().astype(np.float32)  # 60 vars * 4 B = 240 B/row
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p)

    # Force the row_nnz scan into ~10-row blocks: budget 2400 B / 240 B-per-row = 10 rows.
    monkeypatch.setattr(fmt, "_BLOCK_BYTES", 240 * 10)

    src = DenseH5adSource(p)
    spy = _SliceSpy(src._X)
    src._X = spy
    try:
        rn = src.row_nnz()
        assert rn.shape == (a.n_obs,)
        np.testing.assert_array_equal(rn, np.diff(sp.csr_matrix(a.X).indptr))
        assert max(spy.heights) <= 10, "row_nnz scan block exceeded the byte budget"
        assert max(spy.heights) < a.n_obs, "row_nnz read the full matrix in one shot"

        spy.heights.clear()
        blk = src.read_block(3, 9)
        assert blk.shape == (6, a.n_vars)
        assert spy.heights == [6], "read_block did not read exactly one (hi-lo) hyperslab"
    finally:
        src.close()


def test_dense_h5ad_without_encoding_type_still_streams(tmp_path):
    """Robustness (Gemini #231 review): a dense X stored as an h5py Dataset routes to
    DenseH5adSource even when it LACKS encoding-type=="array" (older / third-party files) -- a
    sparse X is ALWAYS an HDF5 Group, so a Dataset X is provably dense. Prevents a memory-heavy
    in-memory fallback for such files."""
    import h5py

    a = ad.AnnData(X=np.arange(20, dtype=np.float32).reshape(4, 5))
    a.obs["g"] = ["a", "b", "a", "b"]
    p = tmp_path / "old_dense.h5ad"
    a.write_h5ad(p)
    with h5py.File(p, "r+") as f:  # strip encoding metadata -> an old dense file with none
        for attr in ("encoding-type", "encoding-version"):
            if attr in f["X"].attrs:
                del f["X"].attrs[attr]
    src = open_cell_source(p)
    try:
        assert isinstance(src, DenseH5adSource)
        assert src.is_in_memory is False
        _assert_blocks_reconstruct(src, sp.csr_matrix(a.X))
    finally:
        src.close()


def test_dense_read_block_chunks_tall_sparse_blocks(tmp_path, monkeypatch):
    """Regression (Gemini + Copilot #231, HIGH): the writer's block planner sizes blocks by CSR
    nnz cost, so a SPARSE-but-dense matrix can put very many rows in one block. read_block must NOT
    materialise that whole span as a single dense hyperslab (multi-TB OOM on real data) -- it
    chunks the dense read to the same budget and vstacks, byte-identically."""
    from cellstream.cell import format as fmt
    from cellstream.cell.source import DenseH5adSource

    rng = np.random.default_rng(3)
    X = np.zeros((200, 40), dtype=np.float32)  # sparse-but-dense: 1 nnz/row
    X[np.arange(200), rng.integers(0, 40, size=200)] = rng.integers(1, 9, size=200).astype(
        np.float32
    )
    p = tmp_path / "sparse_dense.h5ad"
    ad.AnnData(X=X).write_h5ad(p)

    monkeypatch.setattr(fmt, "_BLOCK_BYTES", 40 * 4 * 10)  # dense chunk budget -> chunk_rows = 10

    src = DenseH5adSource(p)
    spy = _SliceSpy(src._X)
    src._X = spy
    try:
        blk = src.read_block(0, 200)  # a block spanning ALL rows, as the sparse planner produces
        np.testing.assert_allclose(blk.toarray(), X)  # byte/value-correct through the vstack
        assert blk.indices.dtype == np.int32, "vstack must preserve the int32 index contract"
        assert blk.indptr.dtype == np.int32, "vstack must preserve int32 indptr too"
        assert max(spy.heights) <= 10, "read_block materialised more than chunk_rows dense rows"
        assert max(spy.heights) < 200, "read_block read the whole tall block as one dense hyperslab"
    finally:
        src.close()


def test_csr_h5ad_with_bytes_encoding_type_still_streams(tmp_path):
    """Robustness (Copilot #231): h5py can return X's encoding-type attr as bytes/np.bytes_ (a
    fixed-length string), which would make `enc == "csr_matrix"` fail and misroute a CSR file to
    the memory-heavy in-memory fallback. open_cell_source must decode bytes attrs before comparing."""
    import h5py

    a = ad.AnnData(X=sp.csr_matrix(np.arange(20, dtype=np.float32).reshape(4, 5)))
    p = tmp_path / "csr_bytes.h5ad"
    a.write_h5ad(p)
    with h5py.File(p, "r+") as f:  # rewrite encoding-type as a BYTES attr
        del f["X"].attrs["encoding-type"]
        f["X"].attrs.create("encoding-type", np.bytes_("csr_matrix"))
    src = open_cell_source(p)
    try:
        assert isinstance(src, BackedH5adSource)  # NOT InMemoryAnnDataSource
        assert src.is_in_memory is False
    finally:
        src.close()


def test_dense_source_init_failure_closes_backed_anndata(monkeypatch):
    """Gemini #231: if h5py.File (or a later init step) raises AFTER the backed AnnData opened,
    __init__ must close it rather than leak the handle."""
    import h5py

    import cellstream.cell.source as srcmod

    closed = {"a": False}

    class _FakeFile:
        def close(self):
            closed["a"] = True

    class _FakeAnnData:
        shape = (4, 5)
        file = _FakeFile()

    monkeypatch.setattr(srcmod.ad, "read_h5ad", lambda *a, **k: _FakeAnnData())

    def _boom(*a, **k):
        raise OSError("cannot open")

    monkeypatch.setattr(h5py, "File", _boom)  # fail after the backed AnnData is open

    with pytest.raises(OSError, match="cannot open"):
        srcmod.DenseH5adSource("whatever.h5ad")
    assert closed["a"], "backed AnnData handle leaked on init failure"
