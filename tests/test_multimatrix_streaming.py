from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellstream.read import ShardedArchive
from cellstream.write import write_sharded


def _grouped_multi(tmp_path, n_obs=60, n_vars=6):
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=8)
    X = sp.csr_matrix(
        (
            (X.data * 20 + 1).astype(np.uint16),
            X.indices.astype(np.uint16),
            X.indptr.astype(np.int64),
        ),
        shape=(n_obs, n_vars),
    )
    counts = X.copy()
    counts.data = (counts.data * 4).astype(np.uint16)
    obs = pd.DataFrame(
        {"target_gene": (["a"] * 30) + (["b"] * 30)}, index=[f"c{i}" for i in range(n_obs)]
    )
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]))
    a.layers["counts"] = counts
    f = tmp_path / "gm.shad"
    write_sharded(a, f, format="v2", group_by="target_gene", target_shard_bytes=4096)
    return ShardedArchive(f), a


def test_view_x_selects_family(tmp_path):
    arch, a = _grouped_multi(tmp_path)
    x_all = arch[:].x()
    counts_all = arch[:].x(matrix="layer:counts")
    # x_all is in grouped (permuted) order; _orig_row == perm, so a.X[perm] matches.
    perm = arch.obs["_orig_row"].to_numpy()
    np.testing.assert_array_equal(x_all.tocsr().toarray(), a.X.toarray()[perm])
    # counts family == 4x the x family, element-wise (same partition/order)
    np.testing.assert_array_equal(counts_all.tocsr().toarray(), (x_all.tocsr().toarray() * 4))


def test_iter_group_shards_matrix_selects_family(tmp_path):
    arch, a = _grouped_multi(tmp_path)
    x_shards = {
        gs.shard_index: gs.x().tocsr().toarray() for gs in arch.iter_group_shards(matrix="x")
    }
    c_shards = {
        gs.shard_index: gs.x().tocsr().toarray()
        for gs in arch.iter_group_shards(matrix="layer:counts")
    }
    assert set(x_shards) == set(c_shards)
    for i in x_shards:
        np.testing.assert_array_equal(c_shards[i], x_shards[i] * 4)


def test_view_layer_kwarg_sugar(tmp_path):
    arch, a = _grouped_multi(tmp_path)
    by_matrix = arch[:].x(matrix="layer:counts").tocsr().toarray()
    by_layer = arch[:].x(layer="counts").tocsr().toarray()
    np.testing.assert_array_equal(by_matrix, by_layer)


def test_layer_helper_returns_scoped_view(tmp_path):
    import pytest

    arch, a = _grouped_multi(tmp_path)
    perm = arch.obs["_orig_row"].to_numpy()
    got = arch.layer("counts").x().tocsr().toarray()
    np.testing.assert_array_equal(got, a.layers["counts"].toarray()[perm])
    with pytest.raises(KeyError):
        arch.layer("nope")


def test_view_to_anndata_selects_family(tmp_path):
    arch, a = _grouped_multi(tmp_path)
    x_all = arch[:].x()
    out = arch[:].to_anndata(matrix="layer:counts")
    # X is the counts family == 4x the x family, element-wise (same partition/order).
    np.testing.assert_array_equal(out.X.tocsr().toarray(), x_all.tocsr().toarray() * 4)
    # var for a layer:* family is shared with x's var (owner resolution).
    assert list(out.var.index) == list(arch.var.index)
    assert list(out.var.index) == list(a.var.index)

    # layer= sugar gives the same X as matrix="layer:counts".
    out_sugar = arch[:].to_anndata(layer="counts")
    np.testing.assert_array_equal(out_sugar.X.tocsr().toarray(), out.X.tocsr().toarray())

    # The scoped view returned by arch.layer(...) should also to_anndata() to
    # the same X (its own default family is already "layer:counts").
    out_scoped = arch.layer("counts").to_anndata()
    np.testing.assert_array_equal(out_scoped.X.tocsr().toarray(), out.X.tocsr().toarray())


def test_iter_group_shards_matrix_with_prefetch(tmp_path):
    arch, a = _grouped_multi(tmp_path)
    x_shards = {
        gs.shard_index: gs.x().tocsr().toarray()
        for gs in arch.iter_group_shards(matrix="x", prefetch=0)
    }
    c_shards = {
        gs.shard_index: gs.x().tocsr().toarray()
        for gs in arch.iter_group_shards(matrix="layer:counts", prefetch=2, n_workers=2)
    }
    assert set(x_shards) == set(c_shards)
    for i in x_shards:
        np.testing.assert_array_equal(c_shards[i], x_shards[i] * 4)
