import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellstream.v2 import writer as w


def _meta_source(n=8, v=4):
    a = ad.AnnData(X=sp.random(n, v, density=0.6, format="csr", dtype="float32", random_state=3))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{j}" for j in range(v)]
    a.obs["ct"] = (["A"] * (n // 2)) + (["B"] * (n - n // 2))
    a.obsp["conn"] = sp.random(n, n, density=0.3, format="csr", dtype="float32", random_state=4)
    a.varp["vp"] = sp.random(v, v, density=0.5, format="csr", dtype="float32", random_state=5)
    return a


def test_grouped_meta_carries_graphs_in_memory():
    a = _meta_source()
    gm = w._resolve_grouped_metadata(a)
    assert "conn" in gm.obsp and "vp" in gm.varp


def test_write_permuted_metadata_permutes_obsp_both_axes(tmp_path):
    a = _meta_source()
    gm = w._resolve_grouped_metadata(a)
    perm = np.array([7, 0, 6, 1, 5, 2, 4, 3])
    w._write_permuted_metadata(tmp_path, gm, perm)
    back = ad.read_h5ad(tmp_path / "header.h5ad")
    expected = a.obsp["conn"][perm][:, perm]
    assert (back.obsp["conn"] != expected).nnz == 0
    assert (back.varp["vp"] != a.varp["vp"]).nnz == 0


def test_grouped_meta_backed_path_reads_graphs(tmp_path):
    a = _meta_source()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    gm = w._resolve_grouped_metadata(str(src))
    assert "conn" in gm.obsp and "vp" in gm.varp
