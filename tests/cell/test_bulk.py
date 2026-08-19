"""Bulk materialize (read_h5ad) of a cell archive vs a scipy reference."""

from __future__ import annotations

import numpy as np
import pytest

import cellstream
from cellstream.cell.writer import write_cell_archive
from tests.cell.conftest import make_cell_adata


@pytest.fixture
def written(tmp_path):
    a = make_cell_adata()
    out = tmp_path / "u.shad"
    write_cell_archive(a, out)
    ref = a.X.tocsr()
    ref.sort_indices()
    return out, ref, a


@pytest.mark.parametrize("n_workers", [1, 4])
def test_bulk_csr_matches_reference(written, n_workers):
    from cellstream.cell.reader import read_cell_bulk

    out, ref, a = written
    got = read_cell_bulk(out, container="csr", data_dtype="uint16", n_workers=n_workers)
    G = got.X.tocsr()
    G.sort_indices()
    np.testing.assert_array_equal(G.indptr, ref.indptr)
    np.testing.assert_array_equal(G.indices.astype(np.int64), ref.indices.astype(np.int64))
    np.testing.assert_array_equal(G.data.astype(np.int64), ref.data.astype(np.int64))
    assert list(got.obs.columns) == list(a.obs.columns)
    assert list(got.var_names) == list(a.var_names)


def test_bulk_cast_back_float32(written):
    from cellstream.cell.reader import read_cell_bulk

    out, ref, a = written
    got = read_cell_bulk(out, cast_back=True, n_workers=2)
    assert got.X.dtype == np.float32
    G = got.X.tocsr()
    G.sort_indices()
    np.testing.assert_array_equal(G.data, ref.data.astype(np.float32))


def test_read_h5ad_routes_cell(written):
    out, ref, a = written
    got = cellstream.read_h5ad(out, container="csr", data_dtype="uint16", n_workers=2)
    G = got.X.tocsr()
    G.sort_indices()
    np.testing.assert_array_equal(G.indptr, ref.indptr)
    np.testing.assert_array_equal(G.data.astype(np.int64), ref.data.astype(np.int64))


def test_bulk_rust_matches_python(tmp_path, cell_adata):
    import numpy as np

    import cellstream
    from cellstream.cell._rust_dispatch import rust_available

    if not rust_available():
        import pytest

        pytest.skip("Rust core not built/enabled")
    p = tmp_path / "cell.shad"
    cellstream.write_sharded(cell_adata, p, layout="cell")

    a_rust = cellstream.read_h5ad(p)
    import os

    os.environ["CELLSTREAM_DISABLE_RUST"] = "1"
    try:
        a_py = cellstream.read_h5ad(p)
    finally:
        del os.environ["CELLSTREAM_DISABLE_RUST"]
    np.testing.assert_array_equal(a_rust.X.toarray(), a_py.X.toarray())
    np.testing.assert_array_equal(a_rust.X.indptr, a_py.X.indptr)
    assert list(a_rust.obs.columns) == list(a_py.obs.columns)


def test_bulk_heap_default_not_shm_backed(written):
    import cellstream

    out, ref, a = written
    got = cellstream.read_h5ad(out, container="csr", data_dtype="uint16", n_workers=2)
    # default (heap) must NOT leave the AnnData pinned to a /dev/shm bundle
    assert not hasattr(got, "close_shared_memory")
    assert getattr(got, "_cellstream_shm_bundle", None) is None
    G = got.X.tocsr()
    G.sort_indices()
    np.testing.assert_array_equal(G.indptr, ref.indptr)
    np.testing.assert_array_equal(G.data.astype(np.int64), ref.data.astype(np.int64))


def test_bulk_shm_backing_pins_bundle(written, monkeypatch):
    from cellstream.cell.reader import read_cell_bulk

    out, ref, a = written
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    got = read_cell_bulk(
        out, container="csr", data_dtype="uint16", n_workers=2, output_backing="shm"
    )
    assert hasattr(got, "close_shared_memory")
    G = got.X.tocsr()
    G.sort_indices()
    np.testing.assert_array_equal(G.data.astype(np.int64), ref.data.astype(np.int64))


def test_csr_direct_assign_valid_matches_naive_ctor():
    import scipy.sparse as sp

    from cellstream.cell.reader import _csr_direct_assign

    data = np.array([10, 20, 30, 40], dtype=np.uint16)
    indices = np.array([0, 2, 1, 2], dtype=np.int64)
    indptr = np.array([0, 2, 2, 4], dtype=np.int64)  # rows: [2 nnz, 0 nnz, 2 nnz]
    n_obs, n_vars = 3, 3

    got = _csr_direct_assign(data, indices, indptr, n_obs=n_obs, n_vars=n_vars)
    ref = sp.csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))
    assert got.shape == (n_obs, n_vars)
    assert got.data.dtype == data.dtype
    np.testing.assert_array_equal(got.toarray(), ref.toarray())
    # arrays assigned by reference (no scipy copy / recast)
    assert got.data is data
    assert got.indices is indices
    assert got.indptr is indptr


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ip: ip[:-1],  # wrong length (n_obs entries, not n_obs+1)
        lambda ip: np.array([1, 2, 2, 4], dtype=np.int64),  # indptr[0] != 0
        lambda ip: np.array([0, 2, 2, 5], dtype=np.int64),  # indptr[-1] != len(data)
        lambda ip: np.array([0, 3, 2, 4], dtype=np.int64),  # non-decreasing violated
    ],
)
def test_csr_direct_assign_rejects_corrupt_indptr(mutate):
    from cellstream.cell.reader import _csr_direct_assign

    data = np.array([10, 20, 30, 40], dtype=np.uint16)
    indices = np.array([0, 2, 1, 2], dtype=np.int64)
    indptr = np.array([0, 2, 2, 4], dtype=np.int64)
    with pytest.raises(ValueError):
        _csr_direct_assign(data, indices, mutate(indptr), n_obs=3, n_vars=3)


@pytest.mark.parametrize("idx_dtype", [np.int64, np.uint32])
def test_csr_direct_assign_check_indices_rejects_out_of_bounds(idx_dtype):
    # check_indices=True (pure-Python fallback path) must catch a column index
    # >= n_vars that the Rust decoder would have rejected in-decode.
    from cellstream.cell.reader import _csr_direct_assign

    data = np.array([1, 2], dtype=np.uint16)
    indices = np.array([0, 5], dtype=idx_dtype)  # 5 >= n_vars=3
    indptr = np.array([0, 2, 2, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="out of range"):
        _csr_direct_assign(data, indices, indptr, n_obs=3, n_vars=3, check_indices=True)


def test_csr_direct_assign_check_indices_rejects_negative_signed():
    # signed index dtype: the < 0 branch fires (unsigned dtypes skip it)
    from cellstream.cell.reader import _csr_direct_assign

    data = np.array([1, 2], dtype=np.uint16)
    indices = np.array([-1, 2], dtype=np.int64)
    indptr = np.array([0, 2, 2, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="out of range"):
        _csr_direct_assign(data, indices, indptr, n_obs=3, n_vars=3, check_indices=True)


def test_csr_direct_assign_default_skips_index_scan():
    # default check_indices=False (Rust bulk path — bounds already validated
    # in-decoder): an out-of-bounds index is NOT re-scanned (no O(nnz) cost).
    from cellstream.cell.reader import _csr_direct_assign

    data = np.array([1, 2], dtype=np.uint16)
    indices = np.array([0, 5], dtype=np.int64)  # 5 >= n_vars=3, but not checked
    indptr = np.array([0, 2, 2, 2], dtype=np.int64)
    X = _csr_direct_assign(data, indices, indptr, n_obs=3, n_vars=3)
    assert X.shape == (3, 3)
    assert X.indices is indices  # passed through untouched


def test_csr_direct_assign_sets_flags():
    # sortedness is intrinsic to pfordelta -> always cached; canonical is opt-in via the arg.
    from cellstream.cell.reader import _csr_direct_assign

    data = np.array([1, 2, 3], dtype=np.uint16)
    indices = np.array([0, 2, 1], dtype=np.int64)
    indptr = np.array([0, 2, 3], dtype=np.int64)
    x_default = _csr_direct_assign(data, indices, indptr, n_obs=2, n_vars=3)
    # Robust across scipy versions: older scipy initializes _has_canonical_format=False in the
    # ctor, scipy>=1.17 leaves it absent (lazy). getattr(...,False) reads the private cache
    # WITHOUT triggering the property's O(nnz) scan and treats "absent" as "not cached".
    assert getattr(x_default, "_has_sorted_indices", False) is True
    assert (
        getattr(x_default, "_has_canonical_format", False) is False
    )  # not forced (canonical=False)
    x_canon = _csr_direct_assign(data, indices, indptr, n_obs=2, n_vars=3, canonical=True)
    assert getattr(x_canon, "_has_canonical_format", False) is True  # cached -> scipy skips O(nnz)


def test_bulk_read_caches_canonical_flags(tmp_path):
    # default require_unique_indices=True -> canonical archive -> flags cached end-to-end.
    out = tmp_path / "d.shad"
    write_cell_archive(make_cell_adata(), out)
    X = cellstream.read_h5ad(out, container="csr", data_dtype="uint16", n_workers=2).X
    assert getattr(X, "_has_sorted_indices", False) is True
    assert getattr(X, "_has_canonical_format", False) is True


def test_bulk_read_no_canonical_when_optout(tmp_path):
    # require_unique_indices=False -> archive NOT marked canonical -> reader must not force the
    # flag. make_cell_adata() data IS canonical, so the public getter would scan+return True;
    # assert on the private cache to prove the reader did not force it.
    out = tmp_path / "n.shad"
    write_cell_archive(make_cell_adata(), out, require_unique_indices=False)
    X = cellstream.read_h5ad(out, container="csr", data_dtype="uint16", n_workers=2).X
    assert getattr(X, "_has_sorted_indices", False) is True  # sortedness still cached (intrinsic)
    assert getattr(X, "_has_canonical_format", False) is False  # canonical NOT force-cached
