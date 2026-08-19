"""v2.reader_gpu — full read on GPU."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

# GPU-gated by tests/v2/gpu/conftest.py (cupy + a visible CUDA device).


def _make_test_adata(seed=11, n_obs=2000, n_vars=300, density=0.05):
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


def test_gpu_full_read_matches_cpu(tmp_path):
    """to_device output matches CPU read of the same archive."""
    adata = _make_test_adata()
    from cellstream import write_sharded
    from cellstream.v2.reader_gpu import read_v2_to_device

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=4, n_workers=2)
    csr_dev = read_v2_to_device(out, cast_back=True, n_streams=2)
    # Pull back to host for comparison.
    host_data = csr_dev.data.get()
    host_indices = csr_dev.indices.get()
    host_indptr = csr_dev.indptr.get()
    host = sp.csr_matrix(
        (host_data, host_indices, host_indptr), shape=adata.shape, dtype=host_data.dtype
    )
    np.testing.assert_array_equal(host.toarray(), adata.X.toarray())


def test_gpu_oom_preflight_raises_before_allocation(tmp_path):
    """An archive that won't fit raises GpuOutOfMemoryError BEFORE allocation."""
    adata = _make_test_adata(n_obs=200, n_vars=80)
    from cellstream import write_sharded
    from cellstream.errors import GpuOutOfMemoryError
    from cellstream.v2.reader_gpu import read_v2_to_device

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    # Force the projected size to exceed available memory.
    with pytest.raises(GpuOutOfMemoryError):
        read_v2_to_device(out, cast_back=True, n_streams=2, _override_projected_bytes=10**15)


def test_gpu_cast_back_false_csr_raises(tmp_path):
    """cast_back=False (uint16) is unsupported for a device CSR.

    cupyx.scipy.sparse CSR data must be float32/float64, so a narrow CSR is
    impossible; read_v2_to_device raises a clear NotImplementedError (narrow is
    container='dense' only). See test_to_cupy.test_gpu_narrow_dense_small."""
    adata = _make_test_adata(n_obs=100, n_vars=50)
    from cellstream import write_sharded
    from cellstream.v2.reader_gpu import read_v2_to_device

    out = tmp_path / "arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    with pytest.raises(NotImplementedError, match="float32/float64"):
        read_v2_to_device(out, cast_back=False, n_streams=2)
