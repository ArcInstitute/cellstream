"""Shared fixtures for v2 tests."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def tiny_csr() -> sp.csr_matrix:
    """A 5x4 uint16 CSR matrix with nnz=8. Small enough to inspect by hand."""
    out = sp.csr_matrix((5, 4), dtype=np.uint16)
    out.data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
    out.indices = np.array([0, 2, 1, 3, 0, 2, 1, 3], dtype=np.uint16)
    out.indptr = np.array([0, 2, 4, 4, 6, 8], dtype=np.int64)
    return out


@pytest.fixture
def mid_csr() -> sp.csr_matrix:
    """A 1000x500 uint16 CSR matrix with ~20k nnz; UMI-shaped."""
    rng = np.random.default_rng(42)
    nnz = 20_000
    rows = rng.integers(0, 1000, size=nnz)
    cols = rng.integers(0, 500, size=nnz)
    vals = rng.integers(1, 50, size=nnz, dtype=np.uint16)
    out_csr = sp.coo_matrix((vals, (rows, cols)), shape=(1000, 500)).tocsr()
    casted = sp.csr_matrix(out_csr.shape, dtype=np.uint16)
    casted.data = out_csr.data.astype(np.uint16)
    casted.indices = out_csr.indices.astype(np.uint16)
    casted.indptr = out_csr.indptr.astype(np.int64)
    return casted
