import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellstream import matrices as mx


def test_reassemble_anndata_attaches_graphs():
    n, v = 5, 3
    X = sp.random(n, v, density=0.5, format="csr", dtype="float32")
    spec = mx.MatrixSpec(key="x", role="x", owner=None, n_vars=v, X=X, var=None)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(v)])
    obsp = {"conn": sp.random(n, n, density=0.3, format="csr", dtype="float32")}
    varp = {"vp": sp.random(v, v, density=0.4, format="csr", dtype="float32")}
    out = mx.reassemble_anndata(
        [spec], obs=obs, var_by_key={"x": var}, uns={}, obsm={}, varm={}, obsp=obsp, varp=varp
    )
    assert "conn" in out.obsp and "vp" in out.varp


def test_reassemble_modality_anndata_attaches_graphs():
    n, v = 5, 3
    X = sp.random(n, v, density=0.5, format="csr", dtype="float32")
    obs = pd.DataFrame({"qc": np.arange(n)}, index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(v)])
    obsp = {"conn": sp.random(n, n, density=0.3, format="csr", dtype="float32")}
    out = mx.reassemble_modality_anndata(
        anchor_X=X, var=var, obs=obs, layer_items=[], obsm={}, varm={}, uns={}, obsp=obsp, varp={}
    )
    assert "qc" in out.obs.columns and "conn" in out.obsp
