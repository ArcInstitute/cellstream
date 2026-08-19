"""Packed v2 archives dispatch through the public ``ShardedArchive`` reader."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def test_packed_archive_constructs_with_schema_and_shape(tmp_path, mid_csr):
    """Packed construction exposes the hand-built archive's schema and shape."""
    from tests.v2.test_reader_cpu import _hand_build_v2_archive

    arch = tmp_path / "v2_arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=2)
    # v2 archives don't have a header.h5ad from the hand-built helper —
    # add a minimal one.
    _write_header_h5ad(arch, mid_csr)
    from cellstream import ShardedArchive
    from tests._archive_helpers import pack_directory

    sh = ShardedArchive(pack_directory(arch, tmp_path / "v2.shad"))
    assert sh.schema_version == 2
    assert sh.shape == mid_csr.shape


def test_v2_to_anndata_round_trips(tmp_path, mid_csr):
    """A packed public read returns an AnnData whose X equals the source CSR."""
    from tests.v2.test_reader_cpu import _hand_build_v2_archive

    arch = tmp_path / "v2_arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=2)
    _write_header_h5ad(arch, mid_csr)
    from cellstream import ShardedArchive
    from tests._archive_helpers import pack_directory

    sh = ShardedArchive(pack_directory(arch, tmp_path / "v2.shad"))
    adata = sh.to_anndata(cast_back=False, n_workers=2)
    np.testing.assert_array_equal(adata.X.data, mid_csr.data)
    np.testing.assert_array_equal(adata.X.indices, mid_csr.indices)
    np.testing.assert_array_equal(adata.X.indptr, mid_csr.indptr)


def test_subset_dispatches_correctly(tmp_path, mid_csr):
    """A packed slice preserves X internals and obs alignment."""
    from tests.v2.test_reader_cpu import _hand_build_v2_archive

    arch = tmp_path / "v2_arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=4)
    _write_header_h5ad(arch, mid_csr)
    from cellstream import ShardedArchive
    from tests._archive_helpers import pack_directory

    sh = ShardedArchive(pack_directory(arch, tmp_path / "v2.shad"))
    sub = sh[100:350]
    adata = sub.to_anndata(cast_back=False, n_workers=2)
    expected = mid_csr[100:350]
    np.testing.assert_array_equal(adata.X.data, expected.data)
    np.testing.assert_array_equal(adata.X.indices, expected.indices)
    np.testing.assert_array_equal(adata.X.indptr, expected.indptr)
    # obs labels: rows 100..349 of the original
    assert adata.obs.index.tolist() == [f"cell_{i}" for i in range(100, 350)]


def test_subset_with_obsm_does_not_crash(tmp_path, mid_csr):
    """A packed subset keeps obs and obsm aligned with X."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    from tests.v2.test_reader_cpu import _hand_build_v2_archive

    arch = tmp_path / "v2_arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=4)
    # Write a header with an obsm — this is the production-realistic case.
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(mid_csr.shape[0])])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(mid_csr.shape[1])])
    obsm = {"X_pca": np.arange(mid_csr.shape[0] * 4, dtype=np.float32).reshape(-1, 4)}
    empty_X = sp.csr_matrix(mid_csr.shape, dtype="float32")
    h = ad.AnnData(X=empty_X, obs=obs, var=var, obsm=obsm)
    h.write_h5ad(arch / "header.h5ad")
    from cellstream import ShardedArchive
    from tests._archive_helpers import pack_directory

    sh = ShardedArchive(pack_directory(arch, tmp_path / "v2.shad"))
    sub = sh[100:350].to_anndata(cast_back=False, n_workers=2)
    # obsm preserved with matching shape:
    assert sub.obsm["X_pca"].shape == (250, 4)
    # The right rows of obsm were taken (rows 100..349):
    expected_pca = obsm["X_pca"][100:350]
    np.testing.assert_array_equal(sub.obsm["X_pca"], expected_pca)


def test_subset_bool_mask_on_v2(tmp_path, mid_csr):
    """A boolean mask on a packed v2 archive returns matching rows."""
    import numpy as np

    from tests.v2.test_reader_cpu import _hand_build_v2_archive

    arch = tmp_path / "v2_arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=2)
    _write_header_h5ad(arch, mid_csr)
    rng = np.random.default_rng(7)
    mask = rng.random(mid_csr.shape[0]) > 0.7
    from cellstream import ShardedArchive
    from tests._archive_helpers import pack_directory

    sh = ShardedArchive(pack_directory(arch, tmp_path / "v2.shad"))
    adata = sh[mask].to_anndata(cast_back=False, n_workers=2)
    expected = mid_csr[mask]
    np.testing.assert_array_equal(adata.X.data, expected.data)
    assert adata.shape[0] == int(mask.sum())


def test_subset_int_array_on_v2(tmp_path, mid_csr):
    """An integer selector on packed v2 returns matching rows in sorted order."""
    import numpy as np

    from tests.v2.test_reader_cpu import _hand_build_v2_archive

    arch = tmp_path / "v2_arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=2)
    _write_header_h5ad(arch, mid_csr)
    rng = np.random.default_rng(8)
    idx = rng.choice(mid_csr.shape[0], size=100, replace=False)
    from cellstream import ShardedArchive
    from tests._archive_helpers import pack_directory

    sh = ShardedArchive(pack_directory(arch, tmp_path / "v2.shad"))
    adata = sh[idx].to_anndata(cast_back=False, n_workers=2)
    expected = mid_csr[np.sort(idx)]
    np.testing.assert_array_equal(adata.X.data, expected.data)


def _write_header_h5ad(arch_dir, csr):
    """Helper: write a minimal header.h5ad so ShardedArchive's obs/var
    accessors work. The v2 hand-build helper doesn't produce one."""
    import anndata as ad
    import pandas as pd

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(csr.shape[0])])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(csr.shape[1])])
    empty_X = sp.csr_matrix(csr.shape, dtype="float32")
    h = ad.AnnData(X=empty_X, obs=obs, var=var)
    h.write_h5ad(arch_dir / "header.h5ad")
