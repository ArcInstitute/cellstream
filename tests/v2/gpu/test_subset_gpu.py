"""Subset reads on the GPU path: contiguous + non-contiguous.

All three tests skip on CPU-only hosts via tests/v2/gpu/conftest.py
(which checks cupy + getDeviceCount at collection time).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp

# GPU-gated by conftest.py. Small subsets take the CPU-decode+upload path (#40);
# arbitrary sub-shard on-GPU decode is #150.


def _make_test_adata(n_obs=2000, n_vars=400, density=0.05, seed=21):
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


def test_contiguous_subset_gpu(tmp_path):
    adata = _make_test_adata()
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=4, n_workers=2)
    sh = ShardedArchive(out)
    gpu = sh[200:700].to_cupy_anndata(cast_back=True, n_streams=2)
    np.testing.assert_array_equal(gpu.X.toarray().get(), adata.X[200:700].toarray())


def test_bool_mask_subset_gpu(tmp_path):
    adata = _make_test_adata()
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=4, n_workers=2)
    sh = ShardedArchive(out)
    rng = np.random.default_rng(7)
    mask = rng.random(adata.shape[0]) > 0.6
    gpu = sh[mask].to_cupy_anndata(cast_back=True, n_streams=2)
    np.testing.assert_array_equal(gpu.X.toarray().get(), adata.X[mask].toarray())


def test_int_array_subset_gpu(tmp_path):
    adata = _make_test_adata()
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=4, n_workers=2)
    sh = ShardedArchive(out)
    rng = np.random.default_rng(8)
    idx = rng.choice(adata.shape[0], size=200, replace=False)
    gpu = sh[idx].to_cupy_anndata(cast_back=True, n_streams=2)
    np.testing.assert_array_equal(gpu.X.toarray().get(), adata.X[np.sort(idx)].toarray())
