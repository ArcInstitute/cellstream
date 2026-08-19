from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellstream.packed.reader import is_packed_file
from cellstream.read import ShardedArchive
from cellstream.write import write_sharded


def _adata(with_layer=True, with_raw=True, n_obs=40, n_vars=6):
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=3)
    X = sp.csr_matrix(
        (
            (X.data * 30 + 1).astype(np.uint16),
            X.indices.astype(np.uint16),
            X.indptr.astype(np.int64),
        ),
        shape=(n_obs, n_vars),
    )
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    if with_layer:
        a.layers["counts"] = X.copy()
    if with_raw:
        a.raw = ad.AnnData(
            X=X.copy(), obs=a.obs.copy(), var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
        )
    return a


def test_layer_present_emits_v4_packed(tmp_path):
    f = tmp_path / "multi.shad"
    write_sharded(_adata(with_layer=True, with_raw=False), f, format="v2")
    assert is_packed_file(f)
    assert ShardedArchive(f).manifest["schema_version"] == 4


def test_raw_only_emits_v4(tmp_path):
    f = tmp_path / "rawonly.shad"
    write_sharded(_adata(with_layer=False, with_raw=True), f, format="v2")
    assert ShardedArchive(f).manifest["schema_version"] == 4


def test_single_matrix_still_v2(tmp_path):
    f = tmp_path / "single.shad"
    write_sharded(_adata(with_layer=False, with_raw=False), f, format="v2", n_shards=2)
    assert ShardedArchive(f).manifest["schema_version"] == 2


def test_single_matrix_grouped_still_v3(tmp_path):
    a = _adata(with_layer=False, with_raw=False)
    a.obs["target_gene"] = (["a"] * 20) + (["b"] * 20)
    f = tmp_path / "grp.shad"
    write_sharded(a, f, format="v2", group_by="target_gene", target_shard_bytes=4096)
    assert ShardedArchive(f).manifest["schema_version"] == 3
