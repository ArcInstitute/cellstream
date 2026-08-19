import anndata as ad
import pytest
import scipy.sparse as sp

from cellstream import header


def _adata_with_graphs(n=6, v=4):
    X = sp.random(n, v, density=0.5, format="csr", dtype="float32")
    a = ad.AnnData(X=X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{j}" for j in range(v)]
    a.obsp["connectivities"] = sp.random(n, n, density=0.3, format="csr", dtype="float32")
    a.varp["corr"] = sp.random(v, v, density=0.4, format="csr", dtype="float32")
    return a


def test_write_header_roundtrips_obsp_varp(tmp_path):
    a = _adata_with_graphs()
    p = tmp_path / "header.h5ad"
    header.write_header(a, p)
    back = ad.read_h5ad(p)
    assert set(back.obsp.keys()) == {"connectivities"}
    assert set(back.varp.keys()) == {"corr"}
    assert (back.obsp["connectivities"] != a.obsp["connectivities"]).nnz == 0
    assert (back.varp["corr"] != a.varp["corr"]).nnz == 0


def test_reject_modality_fields_only_raw(tmp_path):
    # obsp/varp on a modality no longer raise; .raw still does.
    a = _adata_with_graphs()
    header._reject_unsupported_modality_fields(a)  # must NOT raise (has obsp/varp)
    a2 = _adata_with_graphs()
    a2.raw = a2.copy()
    with pytest.raises(ValueError, match=r"\.raw is not supported"):
        header._reject_unsupported_modality_fields(a2)


def test_reject_unsupported_fields_deleted():
    assert not hasattr(header, "_reject_unsupported_fields")
