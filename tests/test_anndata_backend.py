"""anndata backend registration is best-effort. Skip cleanly if the registry shape differs."""

import pytest

from tests._archive_helpers import write_archive


def test_registration_attempt_does_not_break_import():
    import cellstream

    assert hasattr(cellstream, "anndata_backend_registered")
    # The flag should be bool. We don't require True (older anndata or
    # registry-shape changes are acceptable).
    assert isinstance(cellstream.anndata_backend_registered, bool)


def test_anndata_read_h5ad_packed_if_registered(tmp_path):
    """The registered anndata.read_h5ad backend must also intercept a single-file
    packed archive (#106 Gap A) — packed is the v0.4.0 default container, so a
    default write read back via anndata.read_h5ad must route to cellstream rather
    than falling through to anndata's reader (which raises 'file signature not
    found')."""
    import cellstream

    if not cellstream.anndata_backend_registered:
        pytest.skip("anndata IO registry not available or shape mismatch")
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    rng = np.random.default_rng(14)
    n_obs, n_var = 120, 24
    nnz = int(n_obs * n_var * 0.1)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_var, size=nnz)
    data = rng.integers(0, 100, size=nnz).astype(np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_var))
    a = ad.AnnData(X=X)
    a.obs["s"] = ["a"] * n_obs
    out = tmp_path / "arch.shardpack"
    cellstream.write_sharded(a, out, n_shards=2, n_workers=2)
    from cellstream.packed.reader import is_packed_file

    assert is_packed_file(out)  # a single packed file, not a directory
    a2 = ad.read_h5ad(out)  # must route to cellstream's reader, not anndata's
    assert (a2.X != a.X).nnz == 0


def _write_small_dir(tmp_path):
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    rng = np.random.default_rng(21)
    n_obs, n_var = 80, 16
    nnz = int(n_obs * n_var * 0.1)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_var, size=nnz)
    data = rng.integers(1, 50, size=nnz).astype(np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_var))
    out = tmp_path / "arch"
    # Packed (#154 part B3a, route A): the "container" this test is about is the
    # MATERIALIZATION container (dense vs csr) forwarded through the backend, not the archive
    # container -- so the archive container is incidental and packed is the supported one.
    write_archive(ad.AnnData(X=X), out, n_shards=2, n_workers=2)
    return out


def test_anndata_read_h5ad_forwards_container_dense(tmp_path):
    """L7: the intercepted anndata.read_h5ad must forward materialization knobs
    (container/data_dtype/...) to the cellstream reader, not silently drop them."""
    import cellstream

    if not cellstream.anndata_backend_registered:
        pytest.skip("anndata IO registry not available or shape mismatch")
    import anndata as ad
    import scipy.sparse as sp

    out = _write_small_dir(tmp_path)
    a_dense = ad.read_h5ad(out, container="dense")
    assert not sp.issparse(a_dense.X), "container='dense' was dropped -> got a sparse X"


def test_anndata_read_h5ad_rejects_positional_args(tmp_path):
    """L7: a positional arg (e.g. anndata's `backed=`) is unsupported for cellstream
    archives and must fail loud rather than being silently dropped."""
    import cellstream

    if not cellstream.anndata_backend_registered:
        pytest.skip("anndata IO registry not available or shape mismatch")
    import anndata as ad

    out = _write_small_dir(tmp_path)
    with pytest.raises(TypeError):
        ad.read_h5ad(out, "r")  # positional `backed` -> rejected for archives
