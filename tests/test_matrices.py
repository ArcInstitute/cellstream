# tests/test_matrices.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream import matrices as mx


def _adata_with_layers_and_raw():
    X = sp.csr_matrix(np.array([[0, 1, 2], [3, 0, 0]], dtype=np.uint16))
    counts = sp.csr_matrix(np.array([[0, 10, 20], [30, 0, 0]], dtype=np.uint16))
    var = pd.DataFrame(index=["g0", "g1", "g2"])
    obs = pd.DataFrame({"ct": ["a", "b"]}, index=["c0", "c1"])
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.layers["counts"] = counts
    raw_X = sp.csr_matrix(np.array([[0, 1, 2, 0], [3, 0, 0, 9]], dtype=np.uint16))
    raw_var = pd.DataFrame(index=["g0", "g1", "g2", "g3"])
    a.raw = ad.AnnData(X=raw_X, obs=obs, var=raw_var)
    return a


def test_matrix_dir_sanitises_keys():
    assert mx.matrix_dir("x") == "x"
    assert mx.matrix_dir("raw") == "raw"
    assert mx.matrix_dir("layer:counts") == "layer_counts"
    assert mx.matrix_dir("layer:log 1p/norm") == "layer_log_1p_norm"


def test_var_member_name():
    assert mx.var_member_name("x") == "var/x.parquet"
    assert mx.var_member_name("raw") == "var/raw.parquet"


def test_decompose_order_and_roles():
    a = _adata_with_layers_and_raw()
    specs = mx.decompose_anndata(a)
    assert [s.key for s in specs] == ["x", "layer:counts", "raw"]
    x, layer, raw = specs
    assert x.role == "x" and x.owner is None and x.n_vars == 3 and x.var is not None
    assert layer.role == "layer:counts" and layer.owner == "x" and layer.var is None
    assert raw.role == "raw" and raw.owner is None and raw.n_vars == 4 and raw.var is not None
    np.testing.assert_array_equal(x.X.toarray(), a.X.toarray())
    np.testing.assert_array_equal(layer.X.toarray(), a.layers["counts"].toarray())
    np.testing.assert_array_equal(raw.X.toarray(), a.raw.X.toarray())


def test_decompose_single_matrix_is_just_x():
    a = ad.AnnData(X=sp.csr_matrix(np.eye(3, dtype=np.uint16)))
    specs = mx.decompose_anndata(a)
    assert [s.key for s in specs] == ["x"]


def test_reassemble_round_trip():
    a = _adata_with_layers_and_raw()
    specs = mx.decompose_anndata(a)
    var_by_key = {"x": a.var, "raw": a.raw.var}
    rebuilt = mx.reassemble_anndata(
        specs, obs=a.obs, var_by_key=var_by_key, uns={}, obsm={}, varm={}
    )
    np.testing.assert_array_equal(rebuilt.X.toarray(), a.X.toarray())
    np.testing.assert_array_equal(rebuilt.layers["counts"].toarray(), a.layers["counts"].toarray())
    assert list(rebuilt.var.index) == list(a.var.index)
    np.testing.assert_array_equal(rebuilt.raw.X.toarray(), a.raw.X.toarray())
    assert list(rebuilt.raw.var.index) == list(a.raw.var.index)


def test_reassemble_rejects_missing_primary():
    layer = mx.MatrixSpec(key="layer:c", role="layer:c", owner="x", n_vars=2, X=None, var=None)
    with pytest.raises(ValueError, match="primary 'x'"):
        mx.reassemble_anndata(
            [layer], obs=pd.DataFrame(index=["c0"]), var_by_key={}, uns={}, obsm={}, varm={}
        )


def test_decompose_rejects_sanitize_collision():
    a = ad.AnnData(X=sp.csr_matrix(np.eye(2, dtype=np.uint16)))
    a.layers["a/b"] = sp.csr_matrix(np.eye(2, dtype=np.uint16))
    a.layers["a_b"] = sp.csr_matrix(np.eye(2, dtype=np.uint16))
    with pytest.raises(ValueError, match="on-disk directory"):
        mx.decompose_anndata(a)
