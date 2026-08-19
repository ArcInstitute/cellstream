"""Fixtures for packed-archive tests."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from tests._archive_helpers import write_directory


def _make_adata(seed=7, n_obs=240, n_vars=60):
    rng = np.random.default_rng(seed)
    nnz = 4000
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, 50, size=nnz, dtype=np.uint16)
    csr = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    x = sp.csr_matrix(csr.shape, dtype=np.uint16)
    x.data = csr.data.astype(np.uint16)
    x.indices = csr.indices.astype(np.uint16)
    x.indptr = csr.indptr.astype(np.int64)
    genes = [f"g{i:02d}" for i in range(8)]
    labels = rng.choice(genes, size=n_obs)
    obs = {"target_gene": labels, "n_umi": rng.integers(1, 1000, size=n_obs)}
    return ad.AnnData(X=x, obs=obs)


@pytest.fixture
def grouped_dir(tmp_path):
    """A small grouped v2 directory archive. Returns its Path."""
    adata = _make_adata()
    out = tmp_path / "dir_archive"
    write_directory(
        adata,
        out,
        group_by="target_gene",
        reference=["g00"],
        target_shard_bytes=8192,
    )
    return out


@pytest.fixture
def plain_dir(tmp_path):
    """A small non-grouped v2 directory archive. Returns its Path."""
    adata = _make_adata(seed=11)
    out = tmp_path / "plain_archive"
    write_directory(adata, out, n_shards=3)
    return out


@pytest.fixture
def sample_adata():
    """A fresh small AnnData (use this instead of importing _make_adata)."""
    return _make_adata()
