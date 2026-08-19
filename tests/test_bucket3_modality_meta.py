import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellstream import header


def _mod(n=6, v=4):
    a = ad.AnnData(X=sp.random(n, v, density=0.5, format="csr", dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{j}" for j in range(v)]
    a.obs["qc"] = np.arange(n, dtype="float64")
    a.obsp["conn"] = sp.random(n, n, density=0.3, format="csr", dtype="float32")
    a.varp["vp"] = sp.random(v, v, density=0.4, format="csr", dtype="float32")
    return a


def test_modality_meta_stores_full_obs_and_graphs_no_perm(tmp_path):
    a = _mod()
    p = tmp_path / "meta.h5ad"
    header.write_modality_meta(a, p, perm=None)
    back = ad.read_h5ad(p)
    assert list(back.obs.columns) == ["qc"]
    np.testing.assert_array_equal(back.obs["qc"].to_numpy(), a.obs["qc"].to_numpy())
    assert (back.obsp["conn"] != a.obsp["conn"]).nnz == 0
    assert (back.varp["vp"] != a.varp["vp"]).nnz == 0


def test_modality_meta_permutes_obs_and_obsp_both_axes(tmp_path):
    a = _mod()
    perm = np.array([5, 0, 3, 1, 4, 2])
    p = tmp_path / "meta.h5ad"
    header.write_modality_meta(a, p, perm=perm)
    back = ad.read_h5ad(p)
    # obs row-permuted
    np.testing.assert_array_equal(back.obs["qc"].to_numpy(), a.obs["qc"].to_numpy()[perm])
    np.testing.assert_array_equal(np.asarray(back.obs_names), np.asarray(a.obs_names)[perm])
    # obsp both-axes permuted; varp untouched
    expected = a.obsp["conn"][perm][:, perm]
    assert (back.obsp["conn"] != expected).nnz == 0
    assert (back.varp["vp"] != a.varp["vp"]).nnz == 0
