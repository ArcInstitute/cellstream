from __future__ import annotations

import numpy as np
import pytest

from cellstream import shm as _shm
from cellstream.errors import InsufficientShmError
from tests._archive_helpers import archive_manifest, read_v2_host, write_archive


def test_dense_shm_bundle_roundtrip():
    b = _shm.allocate_dense_shm_buffer(nbytes=4 * 10)
    try:
        view = np.ndarray((10,), dtype=np.float32, buffer=b.array.buf)
        # demand-zero: fresh segment reads as 0 before any write.
        assert np.array_equal(view, np.zeros(10, dtype=np.float32))
        view[3] = 1.5
        assert view[3] == np.float32(1.5)
    finally:
        b.close_and_unlink()


def test_dense_preflight_raises_when_too_big(monkeypatch):
    monkeypatch.setattr(_shm, "_free_bytes", lambda path: 1024)
    with pytest.raises(InsufficientShmError, match="dense"):
        _shm.allocate_dense_shm_buffer(nbytes=10 * 1024 * 1024)


def _write_small_uint16_archive(tmp_path):
    """A small float32 AnnData with integral, in-range values -> narrows to uint16
    on disk. Returns the PACKED archive path (#154 part B3a).

    The subject of every test below is materialization -- plan resolution, dtype casts,
    dense-vs-CSR parity -- never the container. The tests that read INTERNALLY go through
    read_v2_host, which hands read_v2_to_host the same (manifest, shard_locator) pair
    ShardedArchive._reader_kwargs does for a packed archive (read.py:302).
    """
    import anndata as ad
    import scipy.sparse as sp

    X = sp.random(20, 50, density=0.3, format="csr", random_state=1).astype(np.float32)
    # integral float32 values in [1, 100] -> writer narrows X.data to uint16.
    X.data = np.floor(X.data * 100).astype(np.float32) + 1.0
    adata = ad.AnnData(X=X)
    out = tmp_path / "u16"
    write_archive(adata, out, n_shards=3)
    return out


def test_read_explicit_data_dtype_csr(tmp_path):
    from cellstream.v2.materialize import resolve_plan

    out = _write_small_uint16_archive(tmp_path)
    plan = resolve_plan(
        archive_manifest(out),
        container="csr",
        data_dtype="float64",
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    data, indices, indptr = read_v2_host(out, plan=plan)
    assert data.dtype == np.dtype(np.float64)


def test_read_default_plan_is_float32(tmp_path):
    out = _write_small_uint16_archive(tmp_path)
    data, indices, indptr = read_v2_host(out)  # no plan -> cast_back default
    assert data.dtype == np.dtype(np.float32)


def test_read_dense_equals_csr_toarray(tmp_path):
    import scipy.sparse as sp

    from cellstream import ShardedArchive
    from cellstream.v2.materialize import resolve_plan

    out = _write_small_uint16_archive(tmp_path)
    ar = ShardedArchive(out)
    d, i, p = read_v2_host(out)  # default CSR reference
    ref = sp.csr_matrix((d, i, p), shape=(ar.n_obs, ar.n_vars)).toarray()
    plan = resolve_plan(
        ar.manifest,
        container="dense",
        data_dtype="float32",
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    dense = read_v2_host(out, plan=plan)
    assert dense.shape == (ar.n_obs, ar.n_vars)
    assert dense.dtype == np.dtype(np.float32)
    assert np.array_equal(dense, ref)  # values match
    assert np.array_equal(dense == 0, ref == 0)  # zero-gaps invariant


def test_read_dense_float16(tmp_path):
    from cellstream.v2.materialize import resolve_plan

    out = _write_small_uint16_archive(tmp_path)
    plan = resolve_plan(
        archive_manifest(out),
        container="dense",
        data_dtype="float16",
        index_dtype=None,
        allow_lossy=True,
        cast_back=True,
        target="cpu",
    )
    dense = read_v2_host(out, plan=plan)
    assert dense.dtype == np.dtype(np.float16)


def _write_float64_nonintegral_archive(tmp_path):
    """A float64 archive with non-integral values -> stored as float64 on disk."""
    import anndata as ad
    import scipy.sparse as sp

    X = sp.random(15, 30, density=0.3, format="csr", random_state=3)  # float64, non-integral
    X.data = X.data + 0.5
    out = tmp_path / "f64"
    write_archive(ad.AnnData(X=X), out, n_shards=2)
    return out


def test_to_anndata_default_unchanged(tmp_path):
    import scipy.sparse as sp

    from cellstream import ShardedArchive

    out = _write_small_uint16_archive(tmp_path)
    a = ShardedArchive(out).to_anndata()
    assert sp.issparse(a.X) and a.X.dtype == np.dtype(np.float32)


def test_to_anndata_dense_and_dtype(tmp_path):
    from cellstream import ShardedArchive

    out = _write_small_uint16_archive(tmp_path)
    ar = ShardedArchive(out)
    ref = ar.to_anndata()  # default float32 csr
    a = ar.to_anndata(container="dense", data_dtype="float64")
    assert isinstance(a.X, np.ndarray)
    assert a.X.dtype == np.dtype(np.float64)
    assert np.array_equal(a.X, ref.X.toarray().astype(np.float64))
    assert list(a.obs.index) == list(ref.obs.index)  # obs stays aligned


def test_subset_dense_parity(tmp_path):
    from cellstream import ShardedArchive

    out = _write_small_uint16_archive(tmp_path)
    ar = ShardedArchive(out)
    ref = ar[2:9].to_anndata().X.toarray()
    a = ar[2:9].to_anndata(container="dense")
    assert a.X.shape == (7, ar.n_vars)
    assert np.array_equal(a.X, ref)


def test_lossy_data_dtype_raises_then_allowed(tmp_path):
    from cellstream import ShardedArchive
    from cellstream.errors import LossyCastError

    out = _write_float64_nonintegral_archive(tmp_path)
    ar = ShardedArchive(out)
    with pytest.raises(LossyCastError):
        ar.to_anndata(data_dtype="int32", n_workers=1)
    a = ar.to_anndata(data_dtype="int32", allow_lossy=True, n_workers=1)
    assert a.X.dtype == np.dtype(np.int32)


def test_read_h5ad_forwards_materialization(tmp_path):
    from cellstream import read_h5ad

    out = _write_small_uint16_archive(tmp_path)
    a = read_h5ad(out, container="dense", data_dtype="float64")
    assert isinstance(a.X, np.ndarray)
    assert a.X.dtype == np.dtype(np.float64)


def test_read_group_dense_parity(tmp_path):
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    from cellstream import ShardedArchive

    X = sp.random(40, 25, density=0.3, format="csr", random_state=7).astype(np.float32)
    X.data = np.floor(X.data * 50).astype(np.float32) + 1.0
    obs = pd.DataFrame({"target": (["NTC"] * 10 + ["g1"] * 15 + ["g2"] * 15)})
    adata = ad.AnnData(X=X, obs=obs)
    out = tmp_path / "grouped"
    write_archive(adata, out, group_by="target", reference="NTC")
    ar = ShardedArchive(out)
    ref = ar.read_group("g1").X.toarray()
    dense = ar.read_group("g1", container="dense").X
    assert isinstance(dense, np.ndarray)
    assert np.array_equal(dense, ref)


# --- GPU/torch knob guards that raise BEFORE any cupy/torch call (run without a GPU) ---


def test_to_torch_uint16_data_dtype_rejected(tmp_path):
    from cellstream import ShardedArchive

    out = _write_small_uint16_archive(tmp_path)
    ar = ShardedArchive(out)
    with pytest.raises(ValueError, match="uint16"):
        ar.to_torch(data_dtype="uint16")  # torch lacks uint16; raises before cupy/torch
