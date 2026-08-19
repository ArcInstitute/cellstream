from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream import matrices as mx


def _gex_atac():
    obs = pd.DataFrame({"ct": ["a", "b", "a"]}, index=["c0", "c1", "c2"])
    gex = ad.AnnData(
        X=sp.csr_matrix(np.arange(9, dtype=np.uint16).reshape(3, 3)),
        obs=obs.copy(),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    gex.layers["norm"] = sp.csr_matrix(np.ones((3, 3), dtype=np.uint16))
    atac = ad.AnnData(
        X=sp.csr_matrix(np.arange(12, dtype=np.uint16).reshape(3, 4)),
        obs=obs.copy(),
        var=pd.DataFrame(index=["p0", "p1", "p2", "p3"]),
    )
    atac.layers["tfidf"] = sp.csr_matrix(np.ones((3, 4), dtype=np.uint16))
    return gex, atac


def test_matrix_dir_modality_and_modality_layer():
    assert mx.matrix_dir("modality:atac") == "modality_atac"
    assert mx.matrix_dir("modality:atac:layer:tfidf") == "modality_atac__layer_tfidf"
    assert mx.matrix_dir("modality:AT AC/1") == "modality_AT_AC_1"


def test_meta_member_name():
    assert mx.meta_member_name("x") == "meta/x.h5ad"
    assert mx.meta_member_name("modality:atac") == "meta/modality_atac.h5ad"


def test_decompose_with_modality_roles_and_owners():
    gex, atac = _gex_atac()
    specs = mx.decompose(gex, modalities={"atac": atac})
    got = [(s.key, s.role, s.owner, s.n_vars) for s in specs]
    assert got == [
        ("x", "x", None, 3),
        ("layer:norm", "layer:norm", "x", 3),
        ("modality:atac", "modality:atac", None, 4),
        ("modality:atac:layer:tfidf", "layer:tfidf", "modality:atac", 4),
    ]
    # modality anchor carries its own var; modality layer carries none.
    by = {s.key: s for s in specs}
    assert list(by["modality:atac"].var.index) == ["p0", "p1", "p2", "p3"]
    assert by["modality:atac:layer:tfidf"].var is None
    np.testing.assert_array_equal(by["modality:atac"].X.toarray(), atac.X.toarray())


def test_decompose_no_modalities_matches_decompose_anndata():
    gex, _ = _gex_atac()
    assert mx.decompose(gex) == mx.decompose_anndata(gex)


def test_decompose_rejects_unaligned_n_obs():
    gex, atac = _gex_atac()
    atac2 = atac[:2].copy()
    with pytest.raises(ValueError, match="aligned|n_obs|obs_names"):
        mx.decompose(gex, modalities={"atac": atac2})


def test_decompose_rejects_reordered_obs_names():
    gex, atac = _gex_atac()
    atac2 = atac[[2, 1, 0]].copy()
    with pytest.raises(ValueError, match="aligned|obs_names|order"):
        mx.decompose(gex, modalities={"atac": atac2})


def test_decompose_accepts_modality_only_obs_column():
    # Bucket 3: per-modality obs columns are stored, not rejected.
    gex, atac = _gex_atac()
    atac.obs["atac_only"] = [1, 2, 3]
    specs = mx.decompose(gex, modalities={"atac": atac})  # must NOT raise
    assert any(s.role == "modality:atac" for s in specs)


def test_is_primary_family():
    assert mx.is_primary_family("x", None)
    assert mx.is_primary_family("raw", None)
    assert mx.is_primary_family("layer:norm", "x")
    assert not mx.is_primary_family("modality:atac", None)
    assert not mx.is_primary_family("layer:tfidf", "modality:atac")


def test_reassemble_modality_anndata():
    _, atac = _gex_atac()
    obs = atac.obs
    out = mx.reassemble_modality_anndata(
        anchor_X=atac.X,
        var=atac.var,
        obs=obs,
        layer_items=[("tfidf", atac.layers["tfidf"])],
        obsm={"X_lsi": np.zeros((3, 2))},
        varm={},
        uns={"k": 1},
    )
    np.testing.assert_array_equal(out.X.toarray(), atac.X.toarray())
    np.testing.assert_array_equal(out.layers["tfidf"].toarray(), atac.layers["tfidf"].toarray())
    assert list(out.var.index) == ["p0", "p1", "p2", "p3"]
    assert "X_lsi" in out.obsm and out.uns["k"] == 1


def test_decompose_rejects_modality_dir_collision():
    obs = pd.DataFrame(index=["c0", "c1", "c2"])
    gex = ad.AnnData(
        X=sp.csr_matrix(np.eye(3, dtype=np.uint16)),
        obs=obs.copy(),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    m1 = ad.AnnData(
        X=sp.csr_matrix(np.eye(3, dtype=np.uint16)),
        obs=obs.copy(),
        var=pd.DataFrame(index=["p0", "p1", "p2"]),
    )
    m2 = m1.copy()
    # modality names "a b" and "a_b" both sanitize to on-disk dir "modality_a_b"
    with pytest.raises(ValueError, match="on-disk directory"):
        mx.decompose(gex, modalities={"a b": m1, "a_b": m2})


def test_decompose_multiple_modalities_order_and_zero_layers():
    obs = pd.DataFrame(index=["c0", "c1", "c2"])
    gex = ad.AnnData(
        X=sp.csr_matrix(np.eye(3, dtype=np.uint16)),
        obs=obs.copy(),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    atac = ad.AnnData(
        X=sp.csr_matrix(np.eye(3, dtype=np.uint16)),
        obs=obs.copy(),
        var=pd.DataFrame(index=["p0", "p1", "p2"]),
    )
    adt = ad.AnnData(
        X=sp.csr_matrix(np.eye(3, dtype=np.uint16)),
        obs=obs.copy(),
        var=pd.DataFrame(index=["a0", "a1", "a2"]),
    )
    specs = mx.decompose(gex, modalities={"atac": atac, "adt": adt})
    # both modalities are layer-free -> just their anchors, in insertion order
    assert [s.key for s in specs] == ["x", "modality:atac", "modality:adt"]
    assert all(s.owner is None for s in specs)  # x + two own-var modality anchors


def test_decompose_rejects_modality_name_with_layer_marker():
    # A modality name containing the reserved ':layer:' marker would be silently
    # mis-parsed by matrix_dir()'s rsplit into a fake modality-layer directory
    # (T1). Reject it fail-loud at decompose time.
    gex, atac = _gex_atac()
    with pytest.raises(ValueError, match="reserved marker"):
        mx.decompose(gex, modalities={"a:layer:b": atac})
