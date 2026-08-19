"""shard_io: write/read primitive for a single .h5ad with narrow dtypes."""

import h5py
import numpy as np
import pytest
import scipy.sparse as sp


def _make_small_adata(seed=0, n_obs=200, n_var=50):
    import anndata as ad

    rng = np.random.default_rng(seed)
    nnz = int(n_obs * n_var * 0.1)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_var, size=nnz)
    data = rng.integers(0, 200, size=nnz).astype(np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_var))
    a = ad.AnnData(X=X)
    a.obs["sample"] = (["s_a"] * (n_obs // 2)) + (["s_b"] * (n_obs - n_obs // 2))
    a.var["gene_id"] = [f"g{i}" for i in range(n_var)]
    return a


def test_write_one_h5ad_produces_uint16_on_disk(tmp_path):
    from cellstream.shard_io import write_one_h5ad

    a = _make_small_adata()
    out = tmp_path / "one.h5ad"
    write_one_h5ad(a, out, codec="blosc-zstd-1-u16u16")
    with h5py.File(out, "r") as f:
        assert f["X/data"].dtype == np.uint16
        assert f["X/indices"].dtype == np.uint16
        # Blosc filter present (third-party filters show as "unknown", not compression_opts)
        assert f["X/data"].compression is not None


def test_write_one_h5ad_refuses_lossy_data(tmp_path):
    from cellstream.errors import LossyCastError
    from cellstream.shard_io import write_one_h5ad

    a = _make_small_adata()
    a.X.data = a.X.data * 1000.0  # overflows uint16
    with pytest.raises(LossyCastError):
        write_one_h5ad(a, tmp_path / "lossy.h5ad", codec="blosc-zstd-1-u16u16")


def test_write_one_h5ad_round_trips_data(tmp_path):
    """Read back via stock anndata; data should round-trip after cast back."""
    import anndata as ad

    from cellstream.shard_io import write_one_h5ad

    a = _make_small_adata()
    out = tmp_path / "one.h5ad"
    write_one_h5ad(a, out, codec="blosc-zstd-1-u16u16")
    a2 = ad.read_h5ad(out)
    # On-disk dtypes are narrow; cast back manually for comparison
    a2.X.data = a2.X.data.astype(np.float32)
    a2.X.indices = a2.X.indices.astype(a.X.indices.dtype)
    assert (a2.X != a.X).nnz == 0
    assert list(a2.obs["sample"]) == list(a.obs["sample"])
    assert list(a2.var["gene_id"]) == list(a.var["gene_id"])
