"""Rust zstd cell decode (PR2).

The gate: the Rust decoder must produce results IDENTICAL to PR1's pure-Python decoder
on the same archive. PR1's committed golden zstd frames (tests/cell/test_zstd_golden.py)
independently pin the on-disk bytes, so together they close the loop.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest
import scipy.sparse as sp

from cellstream.cell._rust_dispatch import rust_available, use_rust_for_cell_decode
from cellstream.cell.reader import open_cell, read_cell_bulk
from cellstream.cell.writer import write_cell_archive

pytestmark = pytest.mark.skipif(
    not rust_available(), reason="Rust extension unavailable / CELLSTREAM_DISABLE_RUST=1"
)


@contextlib.contextmanager
def _no_rust():
    """Force the pure-Python decode path.

    Works because rust_available() re-reads os.environ on EVERY call (it is not cached at
    import), so flipping the var mid-process reroutes the very next decode.
    """
    prev = os.environ.get("CELLSTREAM_DISABLE_RUST")
    os.environ["CELLSTREAM_DISABLE_RUST"] = "1"
    try:
        assert not rust_available()
        yield
    finally:
        if prev is None:
            os.environ.pop("CELLSTREAM_DISABLE_RUST", None)
        else:
            os.environ["CELLSTREAM_DISABLE_RUST"] = prev


def _counts(n_obs=64, n_var=500, density=0.08, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.random(n_obs, n_var, density=density, format="csr", random_state=rng)
    X.data = rng.integers(1, 300, size=X.nnz).astype(np.uint16)
    X = X.tocsr()
    X.sort_indices()
    return X


def _adata(X):
    import anndata as ad

    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(X.shape[1])]
    return a


def _log1p_f64(X):
    Xf = X.astype(np.float64)
    Xf.data = np.log1p(Xf.data)
    return Xf.tocsr()


# --------------------------------------------------------------------------- int


def test_gate_opens_for_zstd_int():
    assert use_rust_for_cell_decode("zstd", "uint32", "uint32", "float32", "int32") is True


def test_rust_zstd_int_matches_python_bulk(tmp_path):
    X = _counts()
    p = tmp_path / "z.shad"
    write_cell_archive(_adata(X), p, codec="zstd", overwrite=True)

    rust = read_cell_bulk(p).X.tocsr()
    with _no_rust():
        py = read_cell_bulk(p).X.tocsr()

    assert (rust != py).nnz == 0
    assert np.array_equal(rust.indices, py.indices)
    assert np.array_equal(rust.indptr, py.indptr)
    assert np.array_equal(rust.toarray(), X.toarray())


def test_rust_zstd_int_matches_python_gather(tmp_path):
    X = _counts()
    p = tmp_path / "z.shad"
    write_cell_archive(_adata(X), p, codec="zstd", overwrite=True)
    rows = np.array([5, 0, 63, 17, 17, 2], dtype=np.int64)  # unsorted + duplicate

    s = open_cell(p)
    rust = s.gather_rows(rows)
    s.close()
    with _no_rust():
        s = open_cell(p)
        py = s.gather_rows(rows)
        s.close()

    assert (rust != py).nnz == 0
    assert np.array_equal(rust.toarray(), X.toarray()[rows])


# ------------------------------------------------------------------------- float


def test_rust_gather_preserves_float64(tmp_path):
    """gather_rows' Rust branch must NOT truncate an f64 archive to float32.

    Regression guard: reader.py's Rust branch hard-coded dtype=np.float32, which was safe
    only while the Rust gate was pfordelta+integer-only. Opening the gate to zstd/float64
    would otherwise SILENTLY lose precision -- allow_lossy=True means Rust never raises.
    """
    Xf = _log1p_f64(_counts())
    p = tmp_path / "f64.shad"
    write_cell_archive(_adata(Xf), p, overwrite=True)  # f64 in -> f64 on disk (PR1 default)

    s = open_cell(p)
    assert s.manifest["value_dtype_on_disk"] == "float64"
    assert use_rust_for_cell_decode("zstd", "uint32", "float64", "float64", "int32") is True
    rows = np.array([3, 11, 0, 40], dtype=np.int64)
    got = s.gather_rows(rows)
    s.close()

    assert got.dtype == np.float64, "gather_rows truncated float64 to float32"
    assert np.array_equal(got.toarray(), Xf.toarray()[rows])  # bit-exact, not approx


@pytest.mark.parametrize(
    "dtype,kwargs",
    [
        ("float64", {}),
        ("float32", {"data_dtype": "float32", "allow_lossy": True}),
    ],
)
def test_rust_float_matches_python(tmp_path, dtype, kwargs):
    Xf = _log1p_f64(_counts())
    p = tmp_path / f"{dtype}.shad"
    write_cell_archive(_adata(Xf), p, codec="zstd", overwrite=True, **kwargs)
    with contextlib.closing(open_cell(p)) as _s:
        assert _s.manifest["value_dtype_on_disk"] == dtype

    rust = read_cell_bulk(p).X.tocsr()
    with _no_rust():
        py = read_cell_bulk(p).X.tocsr()

    assert rust.dtype == py.dtype == np.dtype(dtype)
    assert np.array_equal(rust.toarray(), py.toarray())


def test_rust_float_nan_inf_survive(tmp_path):
    """NaN/Inf must survive the Rust raw-zstd float path BIT-exactly."""
    X = _counts(n_obs=8, n_var=32, density=0.5).astype(np.float64)
    X.data[0], X.data[1], X.data[2] = np.nan, np.inf, -np.inf
    p = tmp_path / "nan.shad"
    write_cell_archive(_adata(X.tocsr()), p, overwrite=True)

    rust = read_cell_bulk(p).X.tocsr()
    with _no_rust():
        py = read_cell_bulk(p).X.tocsr()

    # NaN != NaN, so compare raw bits rather than values.
    assert rust.data.view(np.uint64).tolist() == py.data.view(np.uint64).tolist()
    assert np.isnan(rust.data[0]) and np.isinf(rust.data[1])


# --------------------------------------------------------------------- edge cases


def test_rust_zstd_edge_shapes(tmp_path):
    """Empty cells (incl. trailing), a single-nnz cell, and a cell spanning the full axis."""
    n_var = 40
    rows = [1] + [2] * n_var
    cols = [7] + list(range(n_var))
    vals = [3] + list(range(1, n_var + 1))
    X = sp.csr_matrix(
        (np.array(vals, dtype=np.uint16), (np.array(rows), np.array(cols))),
        shape=(4, n_var),  # row 0 and row 3 stay empty
    )
    X.sort_indices()
    p = tmp_path / "edge.shad"
    write_cell_archive(_adata(X), p, codec="zstd", overwrite=True)

    rust = read_cell_bulk(p).X.tocsr()
    with _no_rust():
        py = read_cell_bulk(p).X.tocsr()
    assert (rust != py).nnz == 0
    assert np.array_equal(rust.toarray(), X.toarray())

    s = open_cell(p)
    assert s.gather_rows(np.array([0], dtype=np.int64)).nnz == 0  # empty cell
    assert s.gather_rows(np.array([1], dtype=np.int64)).nnz == 1  # single nnz
    assert s.gather_rows(np.array([2], dtype=np.int64)).nnz == n_var  # full row
    assert s.gather_rows(np.array([3], dtype=np.int64)).nnz == 0  # trailing empty
    assert s.gather_rows(np.array([], dtype=np.int64)).shape == (0, n_var)
    s.close()


def test_rust_zstd_lossy_policy(tmp_path):
    """f64 archive read down to float32: raises without allow_lossy, casts with it."""
    from cellstream.errors import LossyCastError

    X = _log1p_f64(_counts(n_obs=8, n_var=32, density=0.5))  # fractional -> zstd float path
    X.data[0] = 1e300  # representable in float64, NOT in float32
    p = tmp_path / "lossy.shad"
    write_cell_archive(_adata(X), p, overwrite=True)
    with contextlib.closing(open_cell(p)) as _s:
        assert _s.manifest["value_dtype_on_disk"] == "float64"

    with pytest.raises(LossyCastError):
        read_cell_bulk(p, data_dtype="float32", allow_lossy=False)

    forced = read_cell_bulk(p, data_dtype="float32", allow_lossy=True).X.tocsr()
    assert forced.dtype == np.float32
    assert np.isinf(forced.data[0])  # 1e300 overflows to inf under a forced f32 cast


def test_rust_zstd_rejects_out_of_range_column(tmp_path):
    """A column index >= n_vars must raise cleanly, not crash or return garbage.

    Mirrors the pfordelta guard in test_pfor_rust_decode.py: decode with a too-small
    n_vars so a legitimate index trips the [0, n_vars) bound.
    """
    from cellstream.cell._rust_dispatch import decode_cells

    X = _counts(n_obs=10, n_var=300, density=0.2)
    p = tmp_path / "z.shad"
    write_cell_archive(_adata(X), p, codec="zstd", overwrite=True)

    s = open_cell(p)
    row_ids = np.arange(s.n_obs, dtype=np.int64)
    total = int(s.indptr[-1])
    od = np.empty(total, np.float32)
    oi = np.empty(total, np.int32)
    op = np.empty(s.n_obs + 1, np.int64)
    payload = np.frombuffer(s._mm, dtype=np.uint8)
    try:
        with pytest.raises(ValueError, match="out of range"):
            decode_cells(
                payload,
                s.offsets,
                s.indptr,
                row_ids,
                od,
                oi,
                op,
                codec="zstd",
                value_dtype_on_disk=s.manifest["value_dtype_on_disk"],
                n_vars=5,  # deliberately too small
                allow_lossy=True,
                n_threads=1,
            )
    finally:
        # np.frombuffer exports a pointer into the mmap and PINS it; close() would raise
        # BufferError("cannot close exported pointers exist") while the view is alive.
        del payload
        s.close()


def test_pfordelta_lossy_read_raises_lossy_not_buffererror(tmp_path):
    """Regression: a lossy read via read_cell_bulk's Rust path must raise LossyCastError.

    Pre-existing bug (NOT zstd-specific -- this uses pfordelta): the raised exception's
    traceback pinned _rust_dispatch.decode_cells' frame, whose payload_u8 IS the caller's
    mmap-backed view, so read_cell_bulk's `finally: store.close()` died with
    BufferError("cannot close exported pointers exist") -- MASKING the real error and
    silently breaking the allow_lossy=False contract.
    """
    from cellstream.errors import LossyCastError

    X = sp.random(8, 32, density=0.5, format="csr", random_state=np.random.default_rng(0))
    X.data = np.full(X.nnz, 100_000, dtype=np.uint32)  # > uint16 max -> lossy as uint16
    X = X.tocsr()
    X.sort_indices()
    p = tmp_path / "p.shad"
    write_cell_archive(_adata(X), p, codec="pfordelta", overwrite=True)
    with contextlib.closing(open_cell(p)) as _s:
        assert _s.manifest["codec"] == "pfordelta"

    with pytest.raises(LossyCastError):  # NOT BufferError
        read_cell_bulk(p, data_dtype="uint16", allow_lossy=False)


def test_rust_float32_nan_payloads_bit_exact(tmp_path):
    """f32 exotic NaN/Inf bit patterns must survive the Rust path bit-for-bit.

    Codex (PR #217) flagged that the f32 value path reads through WideVal::Float(f64) and
    casts back to f32, whereas the pure-Python decoder returns a raw np.frombuffer view --
    so arbitrary NaN payload/signaling bits might not be preserved. Empirically they ARE
    (x86 cvtss2sd/cvtsd2ss preserve the payload, and the crate is x86_64-gated), so this
    pins the behavior rather than papering over a non-bug: if a future refactor or a new
    target ever breaks it, this fails loudly.
    """
    n_var = 16
    X = sp.random(4, n_var, density=1.0, format="csr", random_state=np.random.default_rng(2))
    X = X.tocsr().astype(np.float32)
    X.data = np.linspace(0.1, 0.9, X.nnz).astype(np.float32)  # fractional -> zstd float path
    exotic = np.array(
        [0x7FC01234, 0x7F800001, 0xFFC00000, 0x7F800000, 0xFF800000], dtype=np.uint32
    ).view(np.float32)  # quiet-NaN w/ payload, signaling NaN, -NaN, +inf, -inf
    X.data[: exotic.size] = exotic

    p = tmp_path / "f32nan.shad"
    write_cell_archive(
        _adata(X), p, codec="zstd", data_dtype="float32", allow_lossy=True, overwrite=True
    )
    with contextlib.closing(open_cell(p)) as _s:
        assert _s.manifest["value_dtype_on_disk"] == "float32"

    rust = read_cell_bulk(p).X.tocsr()
    with _no_rust():
        py = read_cell_bulk(p).X.tocsr()

    assert rust.data.dtype == np.float32
    # Compare RAW BITS: NaN != NaN, and the whole point is payload preservation.
    assert rust.data.view(np.uint32).tolist() == py.data.view(np.uint32).tolist()
    assert rust.data.view(np.uint32)[: exotic.size].tolist() == exotic.view(np.uint32).tolist()
