from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellstream.header import read_header_obs
from cellstream.manifest import load_manifest
from cellstream.v2.materialize import resolve_plan
from cellstream.v2.reader_cpu import read_v2_to_host
from cellstream.v2.writer import write_v2_multi


def _gex_atac(n_obs=40):
    rng = np.random.default_rng(0)

    def _csr(nv, seed):
        m = sp.random(n_obs, nv, density=0.3, format="csr", random_state=seed)
        return sp.csr_matrix(
            (
                (m.data * 30 + 1).astype(np.uint16),
                m.indices.astype(np.uint16),
                m.indptr.astype(np.int64),
            ),
            shape=(n_obs, nv),
        )

    obs = pd.DataFrame({"g": rng.integers(0, 3, n_obs)}, index=[f"c{i}" for i in range(n_obs)])
    gex = ad.AnnData(
        X=_csr(6, 1), obs=obs.copy(), var=pd.DataFrame(index=[f"g{i}" for i in range(6)])
    )
    atac = ad.AnnData(
        X=_csr(9, 2), obs=obs.copy(), var=pd.DataFrame(index=[f"p{i}" for i in range(9)])
    )
    atac.layers["tfidf"] = _csr(9, 3)
    atac.obsm["X_lsi"] = rng.standard_normal((n_obs, 2)).astype("float32")
    return gex, atac


def _read_family(d, m, key):
    md = m["matrices"][key]
    sub = dict(m)
    sub.update(md)
    sub["matrices"] = None
    plan = resolve_plan(
        sub,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    data, indices, indptr = read_v2_to_host(
        d,
        plan=plan,
        n_workers=1,
        manifest=sub,
        shard_locator=lambda idx: (str(d / md["shard_filename_template"].format(idx)), 0, 0),
    )
    return sp.csr_matrix((data, indices, indptr), shape=(sub["n_obs_total"], md["n_vars"]))


def test_multi_writes_modality_family_and_meta(tmp_path):
    gex, atac = _gex_atac()
    d = tmp_path / "mm"
    write_v2_multi(
        gex,
        d,
        modalities={"atac": atac},
        primary_meta=None,
        primary_modality="rna",
        n_shards=3,
        n_workers=1,
    )
    m = load_manifest(d / "manifest.json")
    assert set(m["matrices"]) == {"x", "modality:atac", "modality:atac:layer:tfidf"}
    assert m["matrices"]["x"]["modality_name"] == "rna"
    assert m["matrices"]["modality:atac"]["modality_name"] == "atac"
    assert (d / "modality_atac" / "shard_0000.shx").exists()
    assert (d / "var" / "modality_atac.parquet").exists()
    assert (d / "meta" / "modality_atac.h5ad").exists()
    assert not (d / "meta" / "x.h5ad").exists()  # native primary -> no x sidecar
    X = _read_family(d, m, "modality:atac")
    np.testing.assert_array_equal(X.toarray(), atac.X.toarray())
    L = _read_family(d, m, "modality:atac:layer:tfidf")
    np.testing.assert_array_equal(L.toarray(), atac.layers["tfidf"].toarray())
    # A modality LAYER shares the modality anchor's var/meta — it must not get
    # its own (redundant) var parquet or meta sidecar.
    layer_md = m["matrices"]["modality:atac:layer:tfidf"]
    assert "var_filename" not in layer_md
    assert "meta_filename" not in layer_md
    assert not (d / "var" / "modality_atac__layer_tfidf.parquet").exists()
    assert not (d / "meta" / "modality_atac__layer_tfidf.h5ad").exists()


def test_multi_modality_meta_obsm_follows_group_by_permutation(tmp_path):
    """Under group_by (non-identity perm), a modality's obsm sidecar must be
    permuted to the SAME cell order as its on-disk X shards.

    Task 5 review bug: write_v2_multi writes every family's X shards in the
    shared permuted order (oi_lists, derived from perm), but used to pass the
    modality's obsm to write_modality_meta unpermuted — so sidecar row i and
    X shard row i silently referred to different cells whenever group_by
    reordered the cells.
    """
    n_obs = 24

    def _csr(nv, seed):
        mat = sp.random(n_obs, nv, density=0.3, format="csr", random_state=seed)
        return sp.csr_matrix(
            (
                (mat.data * 30 + 1).astype(np.uint16),
                mat.indices.astype(np.uint16),
                mat.indptr.astype(np.int64),
            ),
            shape=(n_obs, nv),
        )

    # Interleaved labels (0,1,2,3,0,1,2,3,...): grouping by "g" MUST move rows
    # (group 0's rows are scattered every 4th position), so the resulting perm
    # is guaranteed non-identity — no reliance on random-sort luck.
    obs = pd.DataFrame({"g": np.arange(n_obs) % 4}, index=[f"c{i}" for i in range(n_obs)])
    gex = ad.AnnData(
        X=_csr(6, 1), obs=obs.copy(), var=pd.DataFrame(index=[f"g{i}" for i in range(6)])
    )
    atac = ad.AnnData(
        X=_csr(9, 2), obs=obs.copy(), var=pd.DataFrame(index=[f"p{i}" for i in range(9)])
    )
    # Row i encodes cell identity distinctively, so any misalignment shows up
    # as a value mismatch rather than a coincidental pass.
    atac.obsm["X_lsi"] = np.tile(np.arange(n_obs, dtype="float32")[:, None], (1, 2))

    d = tmp_path / "mm_group"
    write_v2_multi(
        gex,
        d,
        modalities={"atac": atac},
        group_by="g",
        target_shard_bytes=8192,
        n_workers=1,
    )
    m = load_manifest(d / "manifest.json")

    # header.h5ad's obs["_orig_row"] IS the shared permutation (perm) applied
    # to every family's on-disk shard order.
    perm = read_header_obs(d / "header.h5ad")["_orig_row"].to_numpy()
    assert not np.array_equal(perm, np.arange(n_obs))  # group_by really reordered

    meta = ad.read_h5ad(d / "meta" / "modality_atac.h5ad")
    np.testing.assert_array_equal(meta.obsm["X_lsi"], atac.obsm["X_lsi"][perm])

    # X on disk must be permuted the SAME way, so X and obsm agree row-for-row.
    X = _read_family(d, m, "modality:atac")
    np.testing.assert_array_equal(X.toarray(), atac.X[perm].toarray())


def test_multi_primary_meta_writes_x_sidecar(tmp_path):
    gex, atac = _gex_atac()
    primary_meta = ad.AnnData(  # stands in for the primary modality's own metadata
        X=sp.csr_matrix((gex.n_obs, gex.n_vars), dtype="float32"),
        obs=pd.DataFrame(index=gex.obs_names),
        var=pd.DataFrame(index=gex.var_names),
    )
    primary_meta.obsm["X_umap"] = np.zeros((gex.n_obs, 2), dtype="float32")
    d = tmp_path / "mm2"
    write_v2_multi(
        gex,
        d,
        modalities={"atac": atac},
        primary_meta=primary_meta,
        primary_modality="rna",
        n_shards=2,
        n_workers=1,
    )
    m = load_manifest(d / "manifest.json")
    assert (d / "meta" / "x.h5ad").exists()
    assert m["matrices"]["x"]["meta_filename"] == "meta/x.h5ad"


def test_multi_no_modalities_omits_new_fields(tmp_path):
    # native AnnData(X + layers + raw) -> byte-identity guard at the manifest level
    gex, _ = _gex_atac()
    gex.layers["norm"] = gex.X.copy()
    gex.raw = ad.AnnData(X=gex.X.copy(), var=gex.var.copy())
    d = tmp_path / "plain"
    write_v2_multi(gex, d, n_shards=2, n_workers=1)
    m = load_manifest(d / "manifest.json")
    for md in m["matrices"].values():
        assert "modality_name" not in md
        assert "meta_filename" not in md
    assert not (d / "meta").exists()
