import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

mudata = pytest.importorskip("mudata")
from cellstream.mudata_adapter import mudata_to_native  # noqa: E402


def _mod(n, v, seed):
    a = ad.AnnData(X=sp.random(n, v, density=0.5, format="csr", dtype="float32", random_state=seed))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"m{seed}_{j}" for j in range(v)]
    return a


def test_mudata_to_native_routes_global_and_primary_graphs():
    n = 8
    rna, atac = _mod(n, 5, 1), _mod(n, 7, 2)
    rna.obs["rna_qc"] = np.arange(n, dtype="float64")  # primary-modality-only obs col
    rna.obsp["rna_conn"] = sp.random(n, n, density=0.3, format="csr", dtype="float32")
    md = mudata.MuData({"rna": rna, "atac": atac})
    md.obs["global_ct"] = ["A"] * n  # global-only obs col
    md.obsp["wnn"] = sp.random(n, n, density=0.3, format="csr", dtype="float32")

    primary_adata, modalities, primary_meta, primary_name = mudata_to_native(
        md, primary_modality="rna"
    )
    assert primary_name == "rna"
    # global WNN graph rides the header via primary_adata
    assert "wnn" in primary_adata.obsp
    # primary modality's own obs + own graph ride meta/x
    assert "rna_qc" in primary_meta.obs.columns
    assert "rna_conn" in primary_meta.obsp
    # atac returned as-is
    assert "atac" in modalities
