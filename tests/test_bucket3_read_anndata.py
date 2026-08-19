import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellstream import write_sharded
from cellstream.read import ShardedArchive


def _proc_adata(n=10, v=5):
    a = ad.AnnData(X=sp.random(n, v, density=0.5, format="csr", dtype="float32", random_state=7))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{j}" for j in range(v)]
    a.obsp["connectivities"] = sp.random(
        n, n, density=0.3, format="csr", dtype="float32", random_state=8
    )
    a.varp["vp"] = sp.random(v, v, density=0.4, format="csr", dtype="float32", random_state=9)
    return a


def test_single_matrix_v2_roundtrips_graphs(tmp_path):
    a = _proc_adata()
    dst = tmp_path / "arch.shad"
    write_sharded(a, dst, format="v2")
    back = ShardedArchive(dst).to_anndata()
    assert "connectivities" in back.obsp and "vp" in back.varp
    assert (back.obsp["connectivities"] != a.obsp["connectivities"]).nnz == 0


def test_single_matrix_v2_grouped_roundtrips_graphs(tmp_path):
    a = _proc_adata()
    a.obs["ct"] = (["A"] * 5) + (["B"] * 5)
    dst = tmp_path / "arch_g.shad"
    write_sharded(a, dst, format="v2", group_by="ct")
    back = ShardedArchive(dst).to_anndata()
    # returned in partition order; re-index by _orig_row to compare to source
    orig = np.asarray(back.obs["_orig_row"])
    exp = a.obsp["connectivities"][orig][:, orig]
    assert (back.obsp["connectivities"] != exp).nnz == 0
