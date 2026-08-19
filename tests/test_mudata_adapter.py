# tests/test_mudata_adapter.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.read import ShardedArchive
from cellstream.write import write_sharded

mudata = pytest.importorskip("mudata")


def _mudata(n_obs=30):
    def _csr(nv, s):
        m = sp.random(n_obs, nv, density=0.3, format="csr", random_state=s)
        return sp.csr_matrix(
            (
                (m.data * 20 + 1).astype(np.uint16),
                m.indices.astype(np.uint16),
                m.indptr.astype(np.int64),
            ),
            shape=(n_obs, nv),
        )

    # Bare-index obs (no per-modality columns): mudata's default MuData.update()
    # pulls each modality's OWN obs columns into the global obs with a
    # per-modality PREFIX (e.g. a "ct" column on both rna/atac would land as
    # "rna:ct"/"atac:ct" globally, never bare "ct" — confirmed against the
    # installed mudata 0.3.9). The multi-matrix format does not store
    # per-modality obs (only obsm/varm/uns get a sidecar; see
    # matrices._check_modality_aligned / write_modality_meta), so a per-modality
    # obs column is unrepresentable here regardless — keeping obs index-only
    # avoids that (documented, pre-existing, out-of-scope-for-Task-8) rejection
    # entirely; it is not what this test suite is exercising.
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    rna = ad.AnnData(
        X=_csr(5, 1), obs=obs.copy(), var=pd.DataFrame(index=[f"g{i}" for i in range(5)])
    )
    rna.obsm["X_umap"] = np.arange(n_obs * 2, dtype="float32").reshape(n_obs, 2)
    atac = ad.AnnData(
        X=_csr(7, 2), obs=obs.copy(), var=pd.DataFrame(index=[f"p{i}" for i in range(7)])
    )
    atac.layers["tfidf"] = _csr(7, 3)
    atac.obsm["X_lsi"] = np.arange(n_obs * 3, dtype="float32").reshape(n_obs, 3)
    md = mudata.MuData({"rna": rna, "atac": atac})
    md.obs["joint"] = list(range(n_obs))  # a global obs column
    md.obsm["X_joint"] = np.ones((n_obs, 2), dtype="float32")  # a global embedding
    md.uns["run"] = "demo"
    return md


def test_mudata_round_trip(tmp_path):
    md = _mudata()
    f = tmp_path / "mm.h5shad"
    write_sharded(md, f)
    back = ShardedArchive(f).to_mudata()
    assert set(back.mod.keys()) == {"rna", "atac"}
    np.testing.assert_array_equal(back["rna"].X.toarray(), md["rna"].X.toarray())
    np.testing.assert_array_equal(back["atac"].X.toarray(), md["atac"].X.toarray())
    np.testing.assert_array_equal(
        back["atac"].layers["tfidf"].toarray(), md["atac"].layers["tfidf"].toarray()
    )
    # per-modality metadata restored
    np.testing.assert_array_equal(back["rna"].obsm["X_umap"], md["rna"].obsm["X_umap"])
    np.testing.assert_array_equal(back["atac"].obsm["X_lsi"], md["atac"].obsm["X_lsi"])
    # global level restored
    np.testing.assert_array_equal(back.obsm["X_joint"], md.obsm["X_joint"])
    assert back.uns["run"] == "demo"
    assert list(back["rna"].var.index) == [f"g{i}" for i in range(5)]
    assert list(back["atac"].var.index) == [f"p{i}" for i in range(7)]


def test_to_mudata_without_modalities_raises(tmp_path):
    a = ad.AnnData(X=sp.csr_matrix(np.eye(4, dtype=np.uint16)))
    a.layers["norm"] = sp.csr_matrix(np.eye(4, dtype=np.uint16))
    f = tmp_path / "plain.shad"
    write_sharded(a, f)
    with pytest.raises(ValueError, match="modalit"):
        ShardedArchive(f).to_mudata()


def test_mudata_to_native_rejects_unaligned_primary():
    from cellstream.mudata_adapter import mudata_to_native

    md = _mudata()
    # Desync the primary modality's cell ORDER from the global obs (same cells).
    # Do NOT call md.update() (it would re-derive the global obs from the mods).
    rev = list(range(md.n_obs))[::-1]
    md.mod["rna"] = md.mod["rna"][rev].copy()
    with pytest.raises(ValueError, match="aligned"):
        mudata_to_native(md)


def test_mudata_round_trip_primary_with_raw(tmp_path):
    md = _mudata()
    md.mod["rna"].raw = md.mod["rna"].copy()  # primary modality has .raw (legit -> raw family)
    f = tmp_path / "mm_raw.shad"
    write_sharded(md, f)  # must NOT raise (primary_meta is a clean carrier)
    back = ShardedArchive(f).to_mudata()
    assert back["rna"].raw is not None  # primary .raw restored via to_anndata()
    np.testing.assert_array_equal(back["rna"].X.toarray(), md["rna"].X.toarray())


def test_mudata_primary_obsp_write_accepted(tmp_path):
    # Bucket 3: a primary modality's own obsp is stored (in meta/x.h5ad), no
    # longer rejected. Full round-trip is covered in test_bucket3_to_mudata.
    md = _mudata()
    n = md.mod["rna"].n_obs
    md.mod["rna"].obsp["conn"] = sp.random(n, n, density=0.1, format="csr", random_state=0)
    f = tmp_path / "x.shad"
    write_sharded(md, f)  # must NOT raise
    assert f.exists()


def test_mudata_rejects_single_modality(tmp_path):
    rna = ad.AnnData(
        X=sp.csr_matrix(np.eye(4, dtype=np.uint16)),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )
    md1 = mudata.MuData({"rna": rna})
    with pytest.raises(ValueError, match="single modality|>=2|modalit"):
        write_sharded(md1, tmp_path / "y.shad")


def test_mudata_rejects_genuine_global_varm(tmp_path):
    md = _mudata()
    md.varm["real_global"] = np.zeros((md.n_vars, 2))  # not a modality name -> genuine global
    with pytest.raises(ValueError, match="varm"):
        write_sharded(md, tmp_path / "z.shad")


def test_primary_modality_layer_roundtrips_via_mudata(tmp_path):
    # T8a: a PRIMARY modality (rna) carrying a .layers entry must round-trip
    # through the MuData write -> to_mudata read path (previously logic-verified
    # only; non-primary layers were covered, primary-via-MuData was not).
    n = 10
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    rna = ad.AnnData(
        X=sp.csr_matrix((np.arange(n * 4) % 7).astype(np.uint16).reshape(n, 4)),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )
    rna.layers["norm"] = sp.csr_matrix((np.arange(n * 4) % 5).astype(np.uint16).reshape(n, 4))
    atac = ad.AnnData(
        X=sp.csr_matrix((np.arange(n * 5) % 3).astype(np.uint16).reshape(n, 5)),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"p{i}" for i in range(5)]),
    )
    mdata = mudata.MuData({"rna": rna, "atac": atac})
    f = tmp_path / "mm.shad"
    write_sharded(mdata, f)
    out = ShardedArchive(f).to_mudata()
    assert "norm" in out.mod["rna"].layers
    np.testing.assert_array_equal(
        out.mod["rna"].layers["norm"].toarray(),
        rna.layers["norm"].toarray(),
    )


def test_global_varm_modality_named_masks_accepted():
    # T8b complement: mudata auto-creates modality-named boolean varm masks on
    # update; these must NOT trip the by-key-name global-varm guard (the guard is
    # deliberately key-name based; see mudata_to_native's LIMITATION note).
    from cellstream.mudata_adapter import mudata_to_native

    md = _mudata()
    primary, modalities, meta, name = mudata_to_native(md)
    assert name == "rna" and set(modalities) == {"atac"}
