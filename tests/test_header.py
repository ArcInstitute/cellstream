"""header.py: header.h5ad + obs.parquet write/read."""

import numpy as np
import pandas as pd
import pytest


def _make_adata_for_header():
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(1)
    n_obs, n_var = 60, 20
    X = sp.csr_matrix((n_obs, n_var), dtype=np.float32)
    a = ad.AnnData(X=X)
    a.obs["sample"] = ["a"] * 30 + ["b"] * 30
    a.obs["score"] = rng.standard_normal(n_obs)
    a.obs["sample"] = a.obs["sample"].astype("category")
    a.var["gene_id"] = [f"g{i}" for i in range(n_var)]
    a.var["highly_variable"] = rng.random(n_var) > 0.5
    a.uns["pipeline_version"] = "1.0"
    a.obsm["X_pca"] = rng.standard_normal((n_obs, 5))
    a.varm["coef"] = rng.standard_normal((n_var, 3))
    return a


def test_write_read_header_round_trip(tmp_path):
    from cellstream.header import read_header_obs, read_header_var, write_header

    a = _make_adata_for_header()
    path = tmp_path / "header.h5ad"
    write_header(a, path)
    obs = read_header_obs(path)
    var = read_header_var(path)
    assert list(obs["sample"]) == list(a.obs["sample"])
    pd.testing.assert_series_equal(obs["score"], a.obs["score"], check_names=False)
    pd.testing.assert_frame_equal(var, a.var)


def test_write_read_obs_parquet_round_trip(tmp_path):
    from cellstream.header import read_obs_parquet, write_obs_parquet

    a = _make_adata_for_header()
    path = tmp_path / "obs.parquet"
    write_obs_parquet(a.obs, path)
    obs2 = read_obs_parquet(path)
    pd.testing.assert_series_equal(obs2["score"], a.obs["score"], check_names=False)


def test_header_contains_no_x(tmp_path):
    """header.h5ad must not carry X (or X must be empty/zero)."""
    import h5py

    from cellstream.header import write_header

    a = _make_adata_for_header()
    path = tmp_path / "header.h5ad"
    write_header(a, path)
    with h5py.File(path, "r") as f:
        # Empty CSR: X/data exists with shape[0]==0, OR X is absent entirely.
        assert ("X/data" not in f) or (f["X/data"].shape[0] == 0)


def test_permute_cell_axis_fields_rows_and_both_axes():
    import numpy as np
    import scipy.sparse as sp

    from cellstream.header import permute_cell_axis_fields

    perm = np.array([2, 0, 1])
    dense = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=float)
    obsm_out, obsp_out = permute_cell_axis_fields(
        {"e": np.array([[10.0], [20.0], [30.0]])},
        {"g": sp.csr_matrix(dense)},
        perm,
    )
    np.testing.assert_array_equal(obsm_out["e"], np.array([[30.0], [10.0], [20.0]]))
    np.testing.assert_array_equal(obsp_out["g"].toarray(), dense[perm][:, perm])


def test_permute_cell_axis_fields_none_is_passthrough():
    import numpy as np

    from cellstream.header import permute_cell_axis_fields

    obsm = {"e": np.arange(6).reshape(3, 2)}
    obsm_out, obsp_out = permute_cell_axis_fields(obsm, {}, None)
    np.testing.assert_array_equal(obsm_out["e"], obsm["e"])
    assert obsp_out == {}


def test_permute_cell_axis_fields_dataframe_via_iloc():
    import numpy as np
    import pandas as pd

    from cellstream.header import permute_cell_axis_fields

    obsm_out, _ = permute_cell_axis_fields(
        {"df": pd.DataFrame({"x": [10, 20]})}, {}, np.array([1, 0])
    )
    assert list(obsm_out["df"]["x"]) == [20, 10]


# --- #237: concat_cell_axis_fields ------------------------------------------------------------


def _csr(n, m, seed):
    import scipy.sparse as sp

    return sp.random(n, m, density=0.4, random_state=seed, format="csr", dtype=np.float32)


def test_concat_cell_axis_fields_obsm_containers():
    """obsm row-concatenates per container kind: ndarray, pandas, sparse."""
    import scipy.sparse as sp

    from cellstream.header import concat_cell_axis_fields

    d0 = {
        "arr": np.arange(6.0).reshape(3, 2),
        "df": pd.DataFrame({"s": [1.0, 2.0, 3.0]}, index=["a", "b", "c"]),
        "sp": _csr(3, 4, 1),
    }
    d1 = {
        "arr": np.arange(6.0, 10.0).reshape(2, 2),
        "df": pd.DataFrame({"s": [4.0, 5.0]}, index=["d", "e"]),
        "sp": _csr(2, 4, 2),
    }
    obsm, obsp = concat_cell_axis_fields([d0, d1], [{}, {}])
    assert obsp == {}
    np.testing.assert_array_equal(obsm["arr"], np.arange(10.0).reshape(5, 2))
    pd.testing.assert_frame_equal(
        obsm["df"],
        pd.DataFrame({"s": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=["a", "b", "c", "d", "e"]),
    )
    assert sp.issparse(obsm["sp"]) and obsm["sp"].shape == (5, 4)
    np.testing.assert_array_equal(
        obsm["sp"].toarray(), np.vstack([d0["sp"].toarray(), d1["sp"].toarray()])
    )


def test_concat_cell_axis_fields_obsp_block_diagonal():
    """obsp assembles block-diagonal (anndata.concat(pairwise=True) semantics): each part's graph
    lands on its own diagonal block and cross-part entries are zero. block_diag rebases the
    row/column indices itself."""
    import scipy.sparse as sp

    from cellstream.header import concat_cell_axis_fields

    g0, g1 = _csr(3, 3, 5), _csr(2, 2, 6)
    _, obsp = concat_cell_axis_fields([{}, {}], [{"g": g0}, {"g": g1}])
    got = obsp["g"]
    assert sp.issparse(got) and got.shape == (5, 5)
    dense = got.toarray()
    np.testing.assert_array_equal(dense[:3, :3], g0.toarray())
    np.testing.assert_array_equal(dense[3:, 3:], g1.toarray())
    assert not dense[:3, 3:].any() and not dense[3:, :3].any()


def test_concat_cell_axis_fields_dense_obsp_stays_dense():
    """A key stored dense in EVERY part comes back dense -- concat(parts) must equal a single write
    of the same data, and write_cell_archive round-trips a dense obsp."""
    from cellstream.header import concat_cell_axis_fields

    d0 = np.arange(9.0).reshape(3, 3)
    d1 = np.arange(4.0).reshape(2, 2)
    _, obsp = concat_cell_axis_fields([{}, {}], [{"g": d0}, {"g": d1}])
    assert isinstance(obsp["g"], np.ndarray)
    np.testing.assert_array_equal(obsp["g"][:3, :3], d0)
    np.testing.assert_array_equal(obsp["g"][3:, 3:], d1)
    assert not obsp["g"][:3, 3:].any()


def test_concat_cell_axis_fields_promotes_dtype():
    """Value dtypes MAY differ and take numpy's natural promotion, mirroring _widest for X."""
    from cellstream.header import concat_cell_axis_fields

    obsm, _ = concat_cell_axis_fields(
        [{"a": np.zeros((2, 3), np.float32)}, {"a": np.zeros((3, 3), np.float64)}], [{}, {}]
    )
    assert obsm["a"].dtype == np.float64 and obsm["a"].shape == (5, 3)


def test_concat_cell_axis_fields_zero_row_part():
    """A part with no cells contributes an empty block rather than raising."""
    import scipy.sparse as sp

    from cellstream.header import concat_cell_axis_fields

    obsm, obsp = concat_cell_axis_fields(
        [{"a": np.ones((3, 2))}, {"a": np.ones((0, 2))}],
        [{"g": _csr(3, 3, 7)}, {"g": sp.csr_matrix((0, 0), dtype=np.float32)}],
    )
    assert obsm["a"].shape == (3, 2) and obsp["g"].shape == (3, 3)


def test_concat_cell_axis_fields_single_part_is_identity():
    """One part must round-trip unchanged -- concat_cell_archives accepts a single part."""
    from cellstream.header import concat_cell_axis_fields

    obsm, obsp = concat_cell_axis_fields(
        [{"a": np.arange(6.0).reshape(3, 2)}], [{"g": _csr(3, 3, 8)}]
    )
    np.testing.assert_array_equal(obsm["a"], np.arange(6.0).reshape(3, 2))
    assert obsp["g"].shape == (3, 3)


def test_concat_cell_axis_fields_empty_parts():
    """Parts carrying no annotations produce empty dicts, not errors."""
    from cellstream.header import concat_cell_axis_fields

    assert concat_cell_axis_fields([{}, {}, {}], [{}, {}, {}]) == ({}, {})


@pytest.mark.parametrize(
    "obsm_dicts, obsp_dicts, match",
    [
        ([{"a": np.zeros((2, 2))}, {}], [{}, {}], "obsm keys differ"),
        ([{}, {}], [{"g": np.zeros((2, 2))}, {}], "obsp keys differ"),
        (
            [{"a": np.zeros((2, 3))}, {"a": np.zeros((2, 4))}],
            [{}, {}],
            r"obsm\['a'\].*trailing shape",
        ),
        (
            [{"a": np.zeros((2, 2))}, {"a": pd.DataFrame(np.zeros((2, 2)))}],
            [{}, {}],
            "mixed container kinds",
        ),
        ([{}, {}], [{"g": np.zeros((2, 2))}, {"g": "SPARSE_2x2"}], "mixed container kinds"),
        (
            [{"a": pd.DataFrame({"x": [1.0, 2.0]})}, {"a": pd.DataFrame({"y": [3.0, 4.0]})}],
            [{}, {}],
            "different columns",
        ),
    ],
    ids=["obsm_keys", "obsp_keys", "obsm_width", "obsm_container", "obsp_dense_sparse", "df_cols"],
)
def test_concat_cell_axis_fields_rejects_mismatch(obsm_dicts, obsp_dicts, match):
    """Every cross-part disagreement fails loud, naming field, key and part -- never a silent drop
    or NaN-fill (the failure mode #232's guard exists to prevent)."""
    import scipy.sparse as sp

    from cellstream.header import concat_cell_axis_fields

    obsp_dicts = [
        {
            k: (sp.csr_matrix((2, 2), dtype=np.float32) if isinstance(v, str) else v)
            for k, v in d.items()
        }
        for d in obsp_dicts
    ]
    with pytest.raises(ValueError, match=match):
        concat_cell_axis_fields(obsm_dicts, obsp_dicts)


def test_concat_cell_axis_fields_rejects_pandas_obsp():
    """obsp is a cell x cell matrix; a pandas object there is a caller error, not a container to
    block-diagonalize."""
    from cellstream.header import concat_cell_axis_fields

    with pytest.raises(ValueError, match="pandas"):
        concat_cell_axis_fields([{}, {}], [{"g": pd.DataFrame(np.zeros((2, 2)))}] * 2)
