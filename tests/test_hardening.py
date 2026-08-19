"""Hardening tests for items 1-7.

Each test is marked *FAILING* before the fix is applied.  After applying the
fix it must pass.  Run with:

    .venv/bin/python -m pytest tests/test_hardening.py -v
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_small_adata(n_obs=100, n_vars=200, density=0.1, seed=7, max_val=50):
    """Generic small CSR AnnData with values in [1, max_val]."""
    rng = np.random.default_rng(seed)
    nnz = max(1, int(n_obs * n_vars * density))
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, max_val + 1, size=nnz).astype(np.float32)
    coo = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars))
    csr = coo.tocsr()
    out = sp.csr_matrix(csr.shape, dtype=np.float32)
    out.data = csr.data.astype(np.float32)
    out.indices = csr.indices.astype(np.int32)
    out.indptr = csr.indptr.astype(np.int64)
    return ad.AnnData(X=out)


def _write_v2(tmp_path, adata, n_shards=2):
    from cellstream import write_sharded

    out = tmp_path / "v2"
    write_sharded(adata, out, format="v2", n_shards=n_shards, n_workers=1)
    return out


# ===========================================================================
# Item 1 — X.indices uint16 overflow guard in v2 writer
# ===========================================================================


def test_v2_writer_indices_overflow_uses_uint32(tmp_path):
    """A column index > 65535 is stored as uint32 on disk and round-trips.

    Behavior change: the v2 encoder now narrows indices to uint16 only when
    every column index fits (max <= 65535); otherwise it stores uint32 indices
    (n_var > 65536 is supported). Previously this raised LossyCastError.
    """
    from cellstream import ShardedArchive, write_sharded

    # Build a 10×70000 matrix with a nonzero in column 65536 (> uint16 max).
    n_obs, n_vars = 10, 70_000
    rows = np.array([0], dtype=np.int32)
    cols = np.array([65_536], dtype=np.int32)  # high column index → needs uint32
    vals = np.array([1.0], dtype=np.float32)
    csr = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    adata = ad.AnnData(X=csr)

    out = tmp_path / "v2_u32_indices"
    write_sharded(adata, out, format="v2", n_shards=1, n_workers=1)  # must NOT raise
    arc = ShardedArchive(out)  # container-agnostic (default packed); reads manifest
    assert arc.manifest["x_indices_dtype_on_disk"] == "uint32"
    back = arc.to_anndata(cast_back=True, n_workers=1)
    np.testing.assert_array_equal(back.X.toarray(), csr.toarray())


def test_v2_writer_indices_no_false_positive(tmp_path):
    """v2 writer must succeed when all column indices fit uint16 (<= 65535)."""
    adata = _make_small_adata(n_obs=50, n_vars=200)
    _write_v2(tmp_path, adata, n_shards=2)  # must not raise


def test_v2_writer_indices_max_uint16_succeeds(tmp_path):
    """A column index of exactly 65535 (uint16 max) is valid and round-trips.

    Guards against an off-by-one that would reject the boundary value 65535
    (only 65536+ overflows uint16).
    """
    from cellstream import ShardedArchive, write_sharded

    n_obs, n_vars = 10, 65_536  # valid column indices are 0..65535
    rows = np.array([0, 1, 2], dtype=np.int32)
    cols = np.array([0, 65_535, 100], dtype=np.int32)  # 65535 == uint16 max (valid)
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    csr = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    adata = ad.AnnData(X=csr)

    out = tmp_path / "v2_max_u16"
    write_sharded(adata, out, format="v2", n_shards=1, n_workers=1)  # must NOT raise
    back = ShardedArchive(out).to_anndata(cast_back=True, n_workers=1)
    np.testing.assert_array_equal(back.X.toarray(), csr.toarray())


def test_v2_writer_negative_index_rejected(tmp_path):
    """On the scan-based n_vars > 65536 branch, a negative index is rejected."""
    from cellstream import write_sharded
    from cellstream.errors import LossyCastError

    adata = _make_small_adata(n_obs=5, n_vars=100_000, density=0.00002)
    adata.X.indices[0] = -1  # corrupt: violate the CSR non-negative-index invariant

    out = tmp_path / "v2_neg"
    with pytest.raises(LossyCastError, match="X.indices"):
        write_sharded(adata, out, format="v2", n_shards=1, n_workers=1)
    # The archive is packed -- a single file -- so its ABSENCE is the assertion.
    # `(out / "manifest.json").exists()` is False for a packed path no matter what the
    # writer did, so the old check passed whether or not cleanup ran (#154 part B).
    assert not out.exists()


# ===========================================================================
# Item 2 — non-group v2 writes coerce CSR
# ===========================================================================


def test_v2_writer_csc_roundtrips(tmp_path):
    """write_sharded(..., format='v2') with a CSC matrix must write + round-trip correctly."""
    from cellstream import ShardedArchive, write_sharded

    adata_csr = _make_small_adata(n_obs=100, n_vars=50)
    expected = adata_csr.X.toarray()

    # Convert to CSC and wrap in AnnData
    adata_csc = ad.AnnData(X=sp.csc_matrix(adata_csr.X))

    out = tmp_path / "v2_csc"
    write_sharded(adata_csc, out, format="v2", n_shards=2, n_workers=1)

    back = ShardedArchive(out).to_anndata(cast_back=True, n_workers=1)
    np.testing.assert_array_equal(back.X.toarray(), expected)


def test_v2_writer_dense_roundtrips(tmp_path):
    """write_sharded(..., format='v2') with a dense X must write + round-trip correctly."""
    from cellstream import ShardedArchive, write_sharded

    adata_csr = _make_small_adata(n_obs=80, n_vars=40)
    expected = adata_csr.X.toarray()

    adata_dense = ad.AnnData(X=expected.copy())  # X is a numpy ndarray

    out = tmp_path / "v2_dense"
    write_sharded(adata_dense, out, format="v2", n_shards=2, n_workers=1)

    back = ShardedArchive(out).to_anndata(cast_back=True, n_workers=1)
    np.testing.assert_array_equal(back.X.toarray(), expected)


# ===========================================================================
# Item 3 — converter source-indices validation
# ===========================================================================


def test_v2_selector_out_of_range_int_raises_index_error(tmp_path):
    """archive[int_array] with index >= n_obs must raise IndexError."""
    adata = _make_small_adata(n_obs=100, n_vars=50)
    v2 = _write_v2(tmp_path, adata)

    from cellstream import ShardedArchive

    sh = ShardedArchive(v2)
    with pytest.raises(IndexError):
        sh[np.array([50, 150])].to_anndata(n_workers=1)  # 150 >= n_obs=100


def test_v2_selector_negative_int_raises_index_error(tmp_path):
    """archive[int_array] with negative index must raise IndexError."""
    adata = _make_small_adata(n_obs=100, n_vars=50)
    v2 = _write_v2(tmp_path, adata)

    from cellstream import ShardedArchive

    sh = ShardedArchive(v2)
    with pytest.raises(IndexError):
        sh[np.array([-1, 0, 5])].to_anndata(n_workers=1)


def test_v2_selector_wrong_length_mask_raises(tmp_path):
    """archive[bool_mask] with wrong length must raise ValueError or IndexError."""
    adata = _make_small_adata(n_obs=100, n_vars=50)
    v2 = _write_v2(tmp_path, adata)

    from cellstream import ShardedArchive

    sh = ShardedArchive(v2)
    bad_mask = np.ones(50, dtype=bool)  # length 50, not 100
    with pytest.raises((ValueError, IndexError)):
        sh[bad_mask].to_anndata(n_workers=1)


def test_v2_selector_valid_int_array_works(tmp_path):
    """archive[valid_int_array] must work and return the right rows."""
    adata = _make_small_adata(n_obs=100, n_vars=50)
    v2 = _write_v2(tmp_path, adata)

    from cellstream import ShardedArchive

    sh = ShardedArchive(v2)
    idx = np.array([0, 5, 10, 20])
    result = sh[idx].to_anndata(n_workers=1)
    assert result.n_obs == 4


def test_v2_selector_valid_bool_mask_works(tmp_path):
    """archive[valid_bool_mask] must work."""
    adata = _make_small_adata(n_obs=100, n_vars=50)
    v2 = _write_v2(tmp_path, adata)

    from cellstream import ShardedArchive

    sh = ShardedArchive(v2)
    mask = np.zeros(100, dtype=bool)
    mask[:10] = True
    result = sh[mask].to_anndata(n_workers=1)
    assert result.n_obs == 10


def test_v2_selector_scalar_int_works(tmp_path):
    """archive[int] selects a single row (v1 parity: scalar int -> 1 row)."""
    adata = _make_small_adata(n_obs=100, n_vars=50)
    v2 = _write_v2(tmp_path, adata)

    from cellstream import ShardedArchive

    sh = ShardedArchive(v2)
    result = sh[5].to_anndata(n_workers=1)
    assert result.n_obs == 1


def test_v2_selector_float_array_raises(tmp_path):
    """archive[float_array] raises TypeError (no silent truncation of 1.9 -> 1)."""
    adata = _make_small_adata(n_obs=100, n_vars=50)
    v2 = _write_v2(tmp_path, adata)

    from cellstream import ShardedArchive

    sh = ShardedArchive(v2)
    with pytest.raises(TypeError):
        sh[np.array([1.9, 2.1])].to_anndata(n_workers=1)


def test_v2_selector_scalar_bool_raises(tmp_path):
    """archive[True] must NOT be read as integer row 1 (bool is an int subclass)."""
    adata = _make_small_adata(n_obs=100, n_vars=50)
    v2 = _write_v2(tmp_path, adata)

    from cellstream import ShardedArchive

    sh = ShardedArchive(v2)
    with pytest.raises(ValueError):
        sh[True].to_anndata(n_workers=1)


# ===========================================================================
# Item 6 — reject (never silently drop) layers / obsp / varp at write time
# ===========================================================================


def test_v2_nonempty_layers_stored_not_dropped(tmp_path):
    """Non-empty layers are STORED (never silently dropped) at write time.

    Behavior change (multi-matrix release): a v2 write with non-empty layers
    used to raise ValueError (the old contract this test originally asserted).
    It now routes to the v4 multi-matrix packed writer instead, so the layer
    is preserved on disk rather than lost. The "never silently drop" intent
    from the original test is now enforced by the routing pre-flight in
    write_sharded. The two rejection cases this used to point at are both gone:
    format='v1' with the v1 writer (#154 part B) and container!='packed' with the
    directory container itself (part B3b) -- packed is now the only destination,
    so there is nothing left to reject, only the storing this test asserts.
    """
    from cellstream import write_sharded
    from cellstream.packed.reader import is_packed_file
    from cellstream.read import ShardedArchive

    adata = _make_small_adata(n_obs=60, n_vars=30)
    adata.layers["counts"] = adata.X.copy()

    out = tmp_path / "v2_layers.shad"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=1)  # must NOT raise
    assert is_packed_file(out)
    assert ShardedArchive(out).manifest["schema_version"] == 4


def test_v2_nonempty_obsp_roundtrips(tmp_path):
    """Non-empty obsp is stored and round-trips (Bucket 3), no longer rejected."""
    from cellstream import write_sharded
    from cellstream.read import ShardedArchive

    adata = _make_small_adata(n_obs=20, n_vars=10)
    adata.obsp["conn"] = sp.eye(20, format="csr")

    out = tmp_path / "v2_obsp"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=1)  # must NOT raise
    back = ShardedArchive(out).to_anndata()
    assert "conn" in back.obsp
    assert (back.obsp["conn"] != adata.obsp["conn"]).nnz == 0


def test_v2_nonempty_varp_roundtrips(tmp_path):
    """Non-empty varp is stored and round-trips (Bucket 3), no longer rejected."""
    from cellstream import write_sharded
    from cellstream.read import ShardedArchive

    adata = _make_small_adata(n_obs=20, n_vars=10)
    adata.varp["net"] = sp.eye(10, format="csr")

    out = tmp_path / "v2_varp"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=1)  # must NOT raise
    back = ShardedArchive(out).to_anndata()
    assert "net" in back.varp
    assert (back.varp["net"] != adata.varp["net"]).nnz == 0


# The two multi-matrix pre-flight tests that used to sit here
# (test_v1_nonempty_layers_rejected_before_any_shard and its PATH-source twin) were
# v1-only: their subject was that the layers/.raw rejection fires BEFORE the v1 fork pool
# writes any shard_*.h5ad, rather than inside write_header after all of them. They went
# with the v1 writer in #154. The non-packed arm that briefly carried their
# no-partial-artifact assertions went with the directory container in part B3b -- packed is
# now the only destination for a multi-matrix source, so there is no rejection left to
# fire before anything is written. What survives is the STORING, asserted above.


# ===========================================================================
# Item 7 — index-dtype off-by-one at total_nnz == 2**31
#
# NOT v1 code despite the test names: _target_indices_dtype lives in cellstream/h5ad_io.py
# and is reached from the LIVE public read_narrow_h5ad (h5ad_io.py, cast_back arm). The
# names date from when that module was _v1.py. Do not delete these with a v1 sweep.
# ===========================================================================


def test_v1_target_indices_dtype_boundary():
    """total_nnz == 2**31 must select int64, not int32 (CSR indptr overflow)."""
    from cellstream.h5ad_io import _target_indices_dtype

    boundary = 2**31
    dtype_at = _target_indices_dtype(boundary)
    assert dtype_at == np.dtype("int64"), (
        f"total_nnz == 2**31 returned {dtype_at!r}; expected int64 "
        "(indptr[-1] == 2**31 cannot be represented as int32)"
    )


def test_v1_target_indices_dtype_below_boundary():
    """total_nnz == 2**31 - 1 must still select int32."""
    from cellstream.h5ad_io import _target_indices_dtype

    just_below = 2**31 - 1
    dtype_below = _target_indices_dtype(just_below)
    assert dtype_below == np.dtype("int32"), (
        f"total_nnz == 2**31-1 returned {dtype_below!r}; expected int32"
    )


def test_v1_target_indices_dtype_above_boundary():
    """total_nnz > 2**31 must select int64."""
    from cellstream.h5ad_io import _target_indices_dtype

    just_above = 2**31 + 1
    dtype_above = _target_indices_dtype(just_above)
    assert dtype_above == np.dtype("int64"), (
        f"total_nnz == 2**31+1 returned {dtype_above!r}; expected int64"
    )
