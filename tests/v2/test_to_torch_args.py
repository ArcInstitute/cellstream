"""CPU-side argument validation tests for to_torch().

These run without a GPU: cast_back=False raises ValueError before any GPU
code is executed, so they belong in the CPU-side test directory.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp


def _make_test_adata(n_obs=500, n_vars=200, density=0.05, seed=33):
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


def test_to_torch_cast_back_false_raises(tmp_path):
    adata = _make_test_adata()
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    sh = ShardedArchive(out)
    with pytest.raises(ValueError, match="uint16"):
        sh.to_torch(cast_back=False)


def test_to_torch_view_cast_back_false_raises(tmp_path):
    adata = _make_test_adata()
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    sh = ShardedArchive(out)
    with pytest.raises(ValueError, match="cast_back=True"):
        sh[100:300].to_torch(cast_back=False)
