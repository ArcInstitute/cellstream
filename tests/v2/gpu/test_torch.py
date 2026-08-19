"""ShardedArchive.to_torch — DLPack hand-off to torch.sparse_csr_tensor."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

# GPU-gated by tests/v2/gpu/conftest.py (cupy + a visible CUDA device).


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


def test_to_torch_returns_torch_sparse_csr(tmp_path):
    pytest.importorskip("torch")
    import torch

    adata = _make_test_adata()
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    sh = ShardedArchive(out)
    t = sh.to_torch(cast_back=True, n_streams=2)
    assert t.is_sparse_csr
    assert t.dtype == torch.float32
    assert t.crow_indices().dtype == t.col_indices().dtype  # matching-dtype invariant
    assert t.is_cuda


def test_to_torch_subset(tmp_path):
    pytest.importorskip("torch")
    adata = _make_test_adata()
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    sh = ShardedArchive(out)
    t = sh[100:300].to_torch(cast_back=True, n_streams=2)
    assert t.is_sparse_csr
    assert t.shape == (200, adata.shape[1])


def test_to_torch_dense_container(tmp_path):
    """container='dense' -> a dense torch tensor via zero-copy DLPack."""
    pytest.importorskip("torch")
    adata = _make_test_adata()
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    sh = ShardedArchive(out)
    t = sh.to_torch(container="dense", data_dtype="float32", n_streams=2)
    assert not t.is_sparse and not t.is_sparse_csr
    assert tuple(t.shape) == adata.shape
    np.testing.assert_array_equal(t.cpu().numpy(), adata.X.toarray())
