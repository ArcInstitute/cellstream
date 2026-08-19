"""Tests for the v2 row-gather primitives (#76 lean grouped write)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellstream.v2.row_gather import gather_csr_rows


def _csr(seed=0, n_obs=50, n_vars=12, density=0.3):
    X = sp.random(n_obs, n_vars, density=density, format="csr", random_state=seed)
    X.data = np.ceil(X.data * 10).astype(np.float32)
    X.indices = X.indices.astype(np.int32)
    X.indptr = X.indptr.astype(np.int64)
    return X


def test_gather_matches_fancy_index_in_order():
    X = _csr()
    oi = np.array([7, 3, 3, 20, 0, 49], dtype=np.int64)  # unsorted, duplicate
    d, i, p = gather_csr_rows(X.data, X.indices, X.indptr, oi)
    out = sp.csr_matrix((d, i, p), shape=(len(oi), X.shape[1]))
    assert (out.toarray() == X.toarray()[oi]).all()


def test_gather_empty_index_list():
    X = _csr()
    d, i, p = gather_csr_rows(X.data, X.indices, X.indptr, np.array([], dtype=np.int64))
    assert d.size == 0 and i.size == 0
    assert p.tolist() == [0]


def test_gather_preserves_dtypes():
    X = _csr()
    d, i, p = gather_csr_rows(X.data, X.indices, X.indptr, np.array([1, 2], dtype=np.int64))
    assert d.dtype == X.data.dtype and i.dtype == X.indices.dtype and p.dtype == np.int64


def test_read_backed_matches_inmem_gather(tmp_path):
    from cellstream.v2.row_gather import read_backed_csr_rows

    X = _csr(seed=2, n_obs=80, n_vars=16, density=0.25)
    p = tmp_path / "src.h5ad"
    ad.AnnData(X=X).write_h5ad(p)

    oi = np.array([60, 1, 1, 79, 0, 33, 12], dtype=np.int64)  # unsorted, dup
    d, i, ptr = read_backed_csr_rows(str(p), oi)
    out = sp.csr_matrix((d, i, ptr), shape=(len(oi), X.shape[1]))
    assert (out.toarray() == X.toarray()[oi]).all()


def test_read_backed_empty(tmp_path):
    from cellstream.v2.row_gather import read_backed_csr_rows

    X = _csr(seed=3, n_obs=20, n_vars=8)
    p = tmp_path / "src.h5ad"
    ad.AnnData(X=X).write_h5ad(p)
    d, i, ptr = read_backed_csr_rows(str(p), np.array([], dtype=np.int64))
    assert d.size == 0 and ptr.tolist() == [0]
