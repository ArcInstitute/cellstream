# tests/test_multimatrix_integration.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellstream.read import ShardedArchive
from cellstream.write import write_sharded


def _full(n_obs=80, n_vars=9):
    rng = np.random.default_rng(2)
    X = sp.random(n_obs, n_vars, density=0.35, format="csr", random_state=2)
    X = sp.csr_matrix(
        (
            (X.data * 30 + 1).astype(np.uint16),
            X.indices.astype(np.uint16),
            X.indptr.astype(np.int64),
        ),
        shape=(n_obs, n_vars),
    )
    counts = X.copy()
    counts.data = (counts.data * 2).astype(np.uint16)
    lognorm = X.copy()
    lognorm.data = np.log1p(X.data.astype(np.float32))
    raw = sp.random(n_obs, n_vars + 3, density=0.35, format="csr", random_state=12)
    raw = sp.csr_matrix(
        (
            (raw.data * 18 + 1).astype(np.uint16),
            raw.indices.astype(np.uint16),
            raw.indptr.astype(np.int64),
        ),
        shape=(n_obs, n_vars + 3),
    )
    obs = pd.DataFrame(
        {"target_gene": rng.choice(list("abcd"), size=n_obs)}, index=[f"c{i}" for i in range(n_obs)]
    )
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]))
    a.layers["counts"] = counts
    a.layers["lognorm"] = lognorm
    a.raw = ad.AnnData(
        X=raw, obs=obs.copy(), var=pd.DataFrame(index=[f"r{i}" for i in range(n_vars + 3)])
    )
    return a


def _assert_round_trip(a, out):
    np.testing.assert_array_equal(out.X.tocsr().toarray(), a.X.toarray())
    for name in a.layers:
        L = out.layers[name]
        L = L.tocsr().toarray() if sp.issparse(L) else np.asarray(L)
        np.testing.assert_allclose(L, a.layers[name].toarray(), rtol=1e-6)
    np.testing.assert_array_equal(out.raw.X.tocsr().toarray(), a.raw.X.toarray())
    assert list(out.var.index) == list(a.var.index)
    assert list(out.raw.var.index) == list(a.raw.var.index)


def test_full_round_trip_flat_csr(tmp_path):
    a = _full()
    f = tmp_path / "flat.shad"
    write_sharded(a, f, format="v2", n_shards=4)
    _assert_round_trip(a, ShardedArchive(f).to_anndata())


def test_full_round_trip_grouped(tmp_path):
    a = _full()
    f = tmp_path / "grp.shad"
    write_sharded(a, f, format="v2", group_by="target_gene", target_shard_bytes=4096)
    arch = ShardedArchive(f)
    out = arch.to_anndata()
    # grouped reads return permuted order; restore via _orig_row before comparing.
    order = out.obs["_orig_row"].to_numpy().argsort()
    reordered = out[order]
    np.testing.assert_array_equal(reordered.X.tocsr().toarray(), a.X.toarray())
    np.testing.assert_array_equal(reordered.raw.X.tocsr().toarray(), a.raw.X.toarray())


def test_dense_container_round_trip(tmp_path):
    a = _full()
    f = tmp_path / "flat.shad"
    write_sharded(a, f, format="v2", n_shards=3)
    out = ShardedArchive(f).to_anndata()  # default rebuild
    np.testing.assert_array_equal(out.X.tocsr().toarray(), a.X.toarray())


def test_iter_group_shards_zips_families(tmp_path):
    a = _full()
    f = tmp_path / "grp.shad"
    write_sharded(a, f, format="v2", group_by="target_gene", target_shard_bytes=4096)
    arch = ShardedArchive(f)
    xs = list(arch.iter_group_shards(matrix="x"))
    cs = list(arch.iter_group_shards(matrix="layer:counts"))
    assert [g.shard_index for g in xs] == [g.shard_index for g in cs]
    for gx, gc in zip(xs, cs, strict=True):
        np.testing.assert_array_equal(gc.x().tocsr().toarray(), gx.x().tocsr().toarray() * 2)


def test_full_round_trip_from_h5ad_path(tmp_path):
    # The flagship multi-matrix use case: read an existing .h5ad file (with
    # layers/raw) from disk and write it as a cellstream archive. All other
    # integration/routing tests above feed write_sharded an in-memory
    # AnnData; this exercises the path-source branch specifically.
    a = _full()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    f = tmp_path / "out.shad"
    write_sharded(str(src), f, format="v2", n_shards=4)
    _assert_round_trip(a, ShardedArchive(f).to_anndata())
