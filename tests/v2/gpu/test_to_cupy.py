"""ShardedArchive.to_cupy_anndata: hybrid CPU↔GPU dispatch."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

# GPU tests are gated by tests/v2/gpu/conftest.py (skipped unless cupy + a visible
# CUDA device are present). Individual GPU-decode tests also importorskip nvcomp.


def _make_test_adata(n_obs=1000, n_vars=400, density=0.05, seed=42):
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


def test_to_cupy_anndata_small_archive_uses_cpu_upload(tmp_path):
    """A small archive (under threshold) goes via CPU read + cupy.asarray upload."""
    adata = _make_test_adata()
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    sh = ShardedArchive(out)
    gpu_adata = sh.to_cupy_anndata(cast_back=True, n_streams=2)
    import cupyx.scipy.sparse as cusp

    assert isinstance(gpu_adata.X, cusp.csr_matrix)
    np.testing.assert_array_equal(gpu_adata.X.toarray().get(), adata.X.toarray())


def test_to_cupy_anndata_subset_works(tmp_path):
    """sh[a:b].to_cupy_anndata() returns the right slice on device."""
    adata = _make_test_adata()
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    sh = ShardedArchive(out)
    gpu_sub = sh[100:300].to_cupy_anndata(cast_back=True, n_streams=2)
    expected = adata.X[100:300].toarray()
    np.testing.assert_array_equal(gpu_sub.X.toarray().get(), expected)


# --- materialization knobs on the working CPU-decode -> device-upload path ---
# (contract tests; the module-level skip keeps them off until a GPU node runs them)


def test_to_cupy_dense_container_small_archive(tmp_path):
    """container='dense' uploads a dense cupy ndarray (single H2D memcpy)."""
    adata = _make_test_adata()
    import cupy as cp

    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    sh = ShardedArchive(out)
    gpu_adata = sh.to_cupy_anndata(container="dense", data_dtype="float32", n_streams=2)
    assert isinstance(gpu_adata.X, cp.ndarray)
    np.testing.assert_array_equal(gpu_adata.X.get(), adata.X.toarray())


def test_to_cupy_data_dtype_float16_dense(tmp_path):
    """data_dtype='float16' is honored for a DENSE device array.

    (A CSR cannot hold float16 — neither scipy.sparse nor cupyx support it — so
    float16 is container='dense' only; see test_gpu_csr_requires_float.)
    """
    import cupy as cp

    adata = _make_test_adata()
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    sh = ShardedArchive(out)
    gpu_adata = sh.to_cupy_anndata(
        container="dense", data_dtype="float16", allow_lossy=True, n_streams=2
    )
    assert isinstance(gpu_adata.X, cp.ndarray)
    assert gpu_adata.X.dtype == np.dtype(np.float16)


# --- forced on-GPU decode (_gpu_threshold_bytes=0) byte-exactness + x_cupy (#40) ---


def _write(adata, tmp_path, *, n_shards=2, **kw):
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=n_shards, n_workers=2, **kw)
    return ShardedArchive(out)


def test_gpu_decode_byte_exact_int(tmp_path):
    """Forced GPU decode of an integer (byteshuffle-bytedelta) archive == CPU."""
    pytest.importorskip("cupy")
    pytest.importorskip("nvidia.nvcomp")
    sh = _write(_make_test_adata(), tmp_path)
    gpu = sh.to_cupy_anndata(cast_back=True, _gpu_threshold_bytes=0)
    cpu = sh.to_anndata(cast_back=True)
    np.testing.assert_array_equal(gpu.X.toarray().get(), cpu.X.toarray())


def test_gpu_decode_byte_exact_float(tmp_path):
    """Forced GPU decode of a FLOAT (byteshuffle-bytedelta-rawdata) archive == CPU."""
    pytest.importorskip("cupy")
    pytest.importorskip("nvidia.nvcomp")
    rng = np.random.default_rng(7)
    n_obs, n_vars = 800, 300
    dense = np.zeros((n_obs, n_vars), np.float32)
    m = rng.random((n_obs, n_vars)) < 0.05
    dense[m] = (rng.random(int(m.sum())) * 9 + 0.5).astype(
        np.float32
    )  # fractional -> float on disk
    sh = _write(ad.AnnData(X=sp.csr_matrix(dense)), tmp_path)
    assert sh.manifest["x_data_dtype_on_disk"].startswith("float")
    gpu = sh.to_cupy_anndata(cast_back=True, _gpu_threshold_bytes=0)
    np.testing.assert_array_equal(gpu.X.toarray().get(), dense)


@pytest.mark.parametrize("kw", [{"cast_back": False}, {"data_dtype": "float16"}])
def test_gpu_csr_requires_float(tmp_path, kw):
    """A device CSR needs float32/float64 data — narrow uint16 / float16 raise."""
    pytest.importorskip("cupy")
    pytest.importorskip("nvidia.nvcomp")
    sh = _write(_make_test_adata(), tmp_path)
    with pytest.raises(NotImplementedError, match="float32/float64"):
        sh.to_cupy_anndata(_gpu_threshold_bytes=0, allow_lossy=True, **kw)


def test_gpu_narrow_dense_small(tmp_path):
    """Narrow (cast_back=False) IS available via container='dense' (small path)."""
    pytest.importorskip("cupy")
    pytest.importorskip("nvidia.nvcomp")
    import cupy as cp

    sh = _write(_make_test_adata(), tmp_path)
    gpu = sh.to_cupy_anndata(cast_back=False, container="dense")
    cpu = sh.to_anndata(cast_back=False, container="dense")
    assert isinstance(gpu.X, cp.ndarray)
    # A narrow INTEGER dtype (uint16 or, for always-uint32 archives, uint32) — not
    # float32; matches the CPU narrow read (don't hardcode the width).
    assert np.issubdtype(gpu.X.dtype, np.integer)
    assert gpu.X.dtype == cpu.X.dtype
    np.testing.assert_array_equal(gpu.X.get().astype(np.int64), np.asarray(cpu.X).astype(np.int64))


def test_x_cupy_matches_host_x(tmp_path):
    """sh[shard-aligned].x_cupy() == host .x() for the same rows."""
    pytest.importorskip("cupy")
    pytest.importorskip("nvidia.nvcomp")
    import cupyx.scipy.sparse as cusp

    sh = _write(_make_test_adata(n_obs=1000), tmp_path)
    off = sh.row_offsets
    xd = sh[off[0] : off[1]].x_cupy()
    assert isinstance(xd, cusp.csr_matrix)
    np.testing.assert_array_equal(xd.toarray().get(), sh[off[0] : off[1]].x().toarray())


def test_group_shard_x_cupy(tmp_path):
    """GroupShard.x_cupy() == GroupShard.x() on a grouped archive."""
    pytest.importorskip("cupy")
    pytest.importorskip("nvidia.nvcomp")
    import pandas as pd

    from cellstream import ShardedArchive, write_sharded

    rng = np.random.default_rng(1)
    n_obs, n_vars = 1200, 300
    X = sp.random(n_obs, n_vars, density=0.05, format="csr", dtype=np.float32, random_state=rng)
    X.data = np.round(X.data * 40 + 1).astype(np.float32)
    obs = pd.DataFrame({"g": rng.integers(0, 4, n_obs).astype(str)})
    out = tmp_path / "g"
    # group_by governs shard count (n_shards can't be combined with group_by).
    write_sharded(
        ad.AnnData(X=X, obs=obs),
        out,
        format="v2",
        n_workers=2,
        group_by="g",
        reference="0",
    )
    sh = ShardedArchive(out)
    seen = 0
    for gs in sh.iter_group_shards():
        np.testing.assert_array_equal(gs.x_cupy().toarray().get(), gs.x().toarray())
        seen += 1
    assert seen > 0


def test_gpu_dense_container_large_rejected(tmp_path):
    """container='dense' on the forced GPU-decode path raises (CSR-only)."""
    pytest.importorskip("cupy")
    pytest.importorskip("nvidia.nvcomp")
    sh = _write(_make_test_adata(), tmp_path)
    with pytest.raises(NotImplementedError, match="container='csr'"):
        sh.to_cupy_anndata(container="dense", data_dtype="float32", _gpu_threshold_bytes=0)
