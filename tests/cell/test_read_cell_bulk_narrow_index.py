"""#218: read_cell_bulk narrow/uint index dtype on both codecs + both engines."""

from __future__ import annotations

import contextlib
import os

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellstream.cell._rust_dispatch import rust_available, use_rust_for_cell_decode
from cellstream.cell.reader import read_cell_bulk
from cellstream.cell.writer import write_cell_archive


@contextlib.contextmanager
def no_rust():
    prev = os.environ.get("CELLSTREAM_DISABLE_RUST")
    os.environ["CELLSTREAM_DISABLE_RUST"] = "1"
    try:
        yield
    finally:
        if prev is None:
            del os.environ["CELLSTREAM_DISABLE_RUST"]
        else:
            os.environ["CELLSTREAM_DISABLE_RUST"] = prev


def _adata(csr):
    csr = csr.tocsr()
    csr.sort_indices()
    csr.sum_duplicates()
    x = sp.csr_matrix(csr.shape, dtype=csr.dtype)
    x.data = csr.data
    x.indices = csr.indices.astype(np.int32)
    x.indptr = csr.indptr.astype(np.int64)
    # var_names left as AnnData's default index — the tests never assert on them, and building a
    # 70k-element list for the wide-index cases is avoidable overhead. (Copilot #226.)
    return ad.AnnData(X=x, obs={"g": ["a"] * csr.shape[0]})


def _narrow(seed=3, n_obs=50, n_vars=40):
    rng = np.random.default_rng(seed)
    nnz = 300
    rows = rng.integers(0, n_obs, nnz)
    cols = rng.integers(0, n_vars, nnz)
    vals = rng.integers(1, 50, nnz, dtype=np.uint16)
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars))


@pytest.mark.parametrize("codec", ["pfordelta", "zstd"])
def test_cast_back_false_reads_uint16(tmp_path, codec):
    coo = _narrow()
    ref = coo.tocsr()
    ref.sort_indices()
    ref.sum_duplicates()
    out = tmp_path / "u.shad"
    write_cell_archive(_adata(coo), out, codec=codec, overwrite=True)

    got = read_cell_bulk(out, cast_back=False, n_workers=2)
    G = got.X.tocsr()
    G.sort_indices()
    assert got.X.indices.dtype == np.uint16
    np.testing.assert_array_equal(G.indices.astype(np.int64), ref.indices.astype(np.int64))
    np.testing.assert_array_equal(G.data.astype(np.int64), ref.data.astype(np.int64))


@pytest.mark.parametrize("codec", ["pfordelta", "zstd"])
@pytest.mark.parametrize("idt", ["uint16", "uint32"])
def test_explicit_uint_index_narrow(tmp_path, codec, idt):
    coo = _narrow()
    ref = coo.tocsr()
    ref.sort_indices()
    ref.sum_duplicates()
    out = tmp_path / "u.shad"
    write_cell_archive(_adata(coo), out, codec=codec, overwrite=True)

    got = read_cell_bulk(out, index_dtype=idt, n_workers=2)
    G = got.X.tocsr()
    G.sort_indices()
    assert got.X.indices.dtype == np.dtype(idt)
    np.testing.assert_array_equal(G.indices.astype(np.int64), ref.indices.astype(np.int64))


def test_gate_routes_uint_index_to_rust():
    # The gate routes uint16/uint32 output indices to Rust: _RUST_OUT_INDEX includes them, in
    # lockstep with cell.rs's decode_cells dispatch arms (#218).
    if not rust_available():
        pytest.skip("Rust core not built/enabled")
    assert use_rust_for_cell_decode("pfordelta", "uint16", "uint16", "uint16", "uint16") is True
    assert use_rust_for_cell_decode("zstd", "uint32", "uint32", "float32", "uint32") is True


def _wide(all_fit: bool, n_obs=8, n_vars=70000):
    # one deliberately high column when not all_fit (69999 >= 65536 -> overflows uint16).
    hi = 5000 if all_fit else 69999
    rows = np.array([0, 1, 2, 3], dtype=np.int64)
    cols = np.array([10, 200, hi, 42], dtype=np.int64)
    vals = np.array([1, 2, 3, 4], dtype=np.uint16)
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars))


@pytest.mark.parametrize("codec", ["pfordelta", "zstd"])
def test_narrow_uint_rust_matches_python(tmp_path, codec):
    if not rust_available():
        pytest.skip("Rust core not built/enabled")
    coo = _narrow()
    out = tmp_path / "u.shad"
    write_cell_archive(_adata(coo), out, codec=codec, overwrite=True)
    rust = read_cell_bulk(out, cast_back=False, n_workers=2).X.tocsr()
    with no_rust():
        py = read_cell_bulk(out, cast_back=False, n_workers=2).X.tocsr()
    assert rust.indices.dtype == py.indices.dtype == np.uint16
    np.testing.assert_array_equal(rust.indices, py.indices)
    np.testing.assert_array_equal(rust.data, py.data)
    np.testing.assert_array_equal(rust.indptr, py.indptr)


@pytest.mark.parametrize("codec", ["pfordelta", "zstd"])
def test_wide_uint16_overflow_raises(tmp_path, codec):
    from cellstream.errors import LossyCastError

    coo = _wide(all_fit=False)
    out = tmp_path / "w.shad"
    write_cell_archive(_adata(coo), out, codec=codec, overwrite=True)
    # both engines fail loud (Rust wide_fits / Python lossless_cast).
    with pytest.raises(LossyCastError):
        read_cell_bulk(out, index_dtype="uint16", n_workers=2)
    with no_rust(), pytest.raises(LossyCastError):
        read_cell_bulk(out, index_dtype="uint16", n_workers=2)


@pytest.mark.parametrize("codec", ["pfordelta", "zstd"])
def test_wide_uint16_allow_lossy_truncates_identically(tmp_path, codec):
    coo = _wide(all_fit=False)
    out = tmp_path / "w.shad"
    write_cell_archive(_adata(coo), out, codec=codec, overwrite=True)
    rust = read_cell_bulk(out, index_dtype="uint16", allow_lossy=True, n_workers=2).X.tocsr()
    with no_rust():
        py = read_cell_bulk(out, index_dtype="uint16", allow_lossy=True, n_workers=2).X.tocsr()
    # 69999 & 0xFFFF == 4463 on both engines (numpy astype low-bits == Rust `as u16`).
    np.testing.assert_array_equal(rust.indices, py.indices)
    assert 4463 in set(rust.indices.tolist())


@pytest.mark.parametrize("codec", ["pfordelta", "zstd"])
def test_wide_uint16_per_element_fits_succeeds(tmp_path, codec):
    # wide archive (n_vars>65536) whose indices all fit uint16 -> per-element check passes.
    coo = _wide(all_fit=True)
    out = tmp_path / "w.shad"
    write_cell_archive(_adata(coo), out, codec=codec, overwrite=True)
    got = read_cell_bulk(out, index_dtype="uint16", n_workers=2).X.tocsr()
    with no_rust():
        py = read_cell_bulk(out, index_dtype="uint16", n_workers=2).X.tocsr()
    np.testing.assert_array_equal(got.indices, py.indices)


def test_read_prologue_failure_closes_store(tmp_path, monkeypatch):
    # #218 / Copilot #226 r4: resolve_plan rejects a bad data_dtype AFTER CellStore is opened;
    # read_cell_bulk must close the store on that error path (not leak the mmap/file handle).
    from cellstream.cell.reader import CellStore

    out = tmp_path / "u.shad"
    write_cell_archive(_adata(_narrow()), out, overwrite=True)

    closed = []
    real_close = CellStore.close

    def spy_close(self):
        closed.append(True)
        return real_close(self)

    monkeypatch.setattr(CellStore, "close", spy_close)
    with pytest.raises((TypeError, ValueError)):
        read_cell_bulk(out, data_dtype="not_a_real_dtype", n_workers=2)
    assert closed, "read_cell_bulk leaked the CellStore when the read prologue raised"
