"""Opt-in byte-filter codec via the public write_sharded(format='v2', codec=...) API."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from tests._archive_helpers import archive_manifest


def _make_adata(seed=1, n_obs=600, n_vars=400, density=0.06):
    rng = np.random.default_rng(seed)
    nnz = int(n_obs * n_vars * density)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, 50, size=nnz, dtype=np.uint16)
    raw = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    csr = sp.csr_matrix(raw.shape, dtype=np.float32)
    csr.data = raw.data.astype(np.float32)
    csr.indices = raw.indices.astype(np.int32)
    csr.indptr = raw.indptr.astype(np.int64)
    return ad.AnnData(X=csr)


def test_write_sharded_v2_byte_filter_round_trips(tmp_path):
    from cellstream import ShardedArchive, write_sharded

    adata = _make_adata(seed=8)
    out = tmp_path / "pub_bf"
    write_sharded(
        adata,
        out,
        format="v2",
        n_shards=4,
        n_workers=2,
        codec="v2-byte-filter",
    )
    back = ShardedArchive(out).to_anndata(cast_back=True, n_workers=2)
    np.testing.assert_array_equal(back.X.toarray(), adata.X.toarray())
    m = archive_manifest(out)
    # Manifest `codec` is informational and now stores the public preset name
    # (the reader dispatches on per-shard ShxHeader.shuffle, not this field).
    assert m["codec"] == "v2-byte-filter"


def test_write_sharded_v2_unknown_codec_raises(tmp_path):
    from cellstream import write_sharded

    adata = _make_adata(seed=9)
    with pytest.raises(ValueError, match="Unknown v2 codec"):
        write_sharded(adata, tmp_path / "x", format="v2", n_shards=2, n_workers=1, codec="bogus")


def test_write_sharded_v2_default_codec_is_byte_filter(tmp_path):
    """As of v0.3.0, write_sharded(format='v2') with no codec= defaults to the
    byte-filter codec (the manifest `codec` field reads the public preset name).
    """
    from cellstream import write_sharded

    adata = _make_adata(seed=10)
    out = tmp_path / "pub_default"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    m = archive_manifest(out)
    assert m["codec"] == "v2-byte-filter"


def test_write_sharded_v2_byte_filter_subset_read(tmp_path):
    """A row subset of a byte-filter archive reads byte-exact (decode dispatch
    works on the streaming/subset path too — exercises _plan_shards too)."""
    from cellstream import ShardedArchive, write_sharded

    adata = _make_adata(seed=12, n_obs=500, n_vars=300)
    out = tmp_path / "sub_bf"
    write_sharded(adata, out, format="v2", n_shards=4, n_workers=2, codec="v2-byte-filter")
    rows = np.array([3, 17, 200, 401, 499])
    sub = ShardedArchive(out)[rows].to_anndata(cast_back=True, n_workers=2)
    np.testing.assert_array_equal(sub.X.toarray(), adata.X.toarray()[rows])
