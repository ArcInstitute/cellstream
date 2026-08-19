from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellstream.manifest import load_manifest
from cellstream.v2.materialize import resolve_plan
from cellstream.v2.reader_cpu import read_v2_to_host
from cellstream.v2.writer import write_v2_archive, write_v2_multi


def _adata(seed=1, n_obs=60, n_vars=8):
    rng = np.random.default_rng(seed)
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=seed)
    X.data = (X.data * 40 + 1).astype(np.uint16)
    X = sp.csr_matrix(
        (X.data.astype(np.uint16), X.indices.astype(np.uint16), X.indptr.astype(np.int64)),
        shape=(n_obs, n_vars),
    )
    counts = X.copy()
    counts.data = (counts.data * 2).astype(np.uint16)
    raw = sp.random(n_obs, n_vars + 2, density=0.3, format="csr", random_state=seed + 9)
    raw = sp.csr_matrix(
        (
            (raw.data * 20 + 1).astype(np.uint16),
            raw.indices.astype(np.uint16),
            raw.indptr.astype(np.int64),
        ),
        shape=(n_obs, n_vars + 2),
    )
    genes = np.array([f"g{i:02d}" for i in range(n_vars)])
    obs = pd.DataFrame(
        {"target_gene": rng.choice(genes[:4], size=n_obs)}, index=[f"c{i}" for i in range(n_obs)]
    )
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=list(genes)))
    a.layers["counts"] = counts
    a.raw = ad.AnnData(X=raw, obs=obs, var=pd.DataFrame(index=[f"r{i}" for i in range(n_vars + 2)]))
    return a


def test_multi_writes_v4_structure_flat(tmp_path):
    a = _adata()
    d = tmp_path / "multi"
    write_v2_multi(a, d, n_shards=None, n_workers=1, target_shard_bytes=8192)
    m = load_manifest(d / "manifest.json")
    assert m["schema_version"] == 4
    assert set(m["matrices"]) == {"x", "layer:counts", "raw"}
    assert (d / "x" / "shard_0000.shx").exists()
    assert (d / "layer_counts" / "shard_0000.shx").exists()
    assert (d / "raw" / "shard_0000.shx").exists()
    assert (d / "var" / "x.parquet").exists()
    assert (d / "var" / "raw.parquet").exists()
    assert not (d / "var" / "layer_counts.parquet").exists()  # layer shares x's var


def test_multi_x_family_byte_identical_to_grouped_single(tmp_path):
    a = _adata()
    single = tmp_path / "single"
    x_only = ad.AnnData(X=a.X.copy(), obs=a.obs.copy(), var=a.var.copy())
    write_v2_archive(x_only, single, group_by="target_gene", target_shard_bytes=8192, n_workers=1)
    multi = tmp_path / "multi"
    write_v2_multi(a, multi, group_by="target_gene", target_shard_bytes=8192, n_workers=1)
    ms = load_manifest(single / "manifest.json")
    for i in range(ms["n_shards"]):
        assert (multi / "x" / f"shard_{i:04d}.shx").read_bytes() == (
            single / f"shard_{i:04d}.shx"
        ).read_bytes()


def test_multi_x_family_byte_identical_under_byte_filter_codec(tmp_path):
    """Byte-identity of the x-family must also hold under the PRODUCTION
    codec (``v2-byte-filter``, Rust engaged), where single-matrix write forwards
    ``_rust_dispatch.decide_dtypes`` while write_v2_multi sizes via
    ``narrow.decide_on_disk_dtypes`` — the two must agree (#130 unified the
    Py/Rust narrow scan). Guards the byte-identity binding constraint on the path
    write_sharded actually uses."""
    a = _adata()
    single = tmp_path / "single"
    x_only = ad.AnnData(X=a.X.copy(), obs=a.obs.copy(), var=a.var.copy())
    write_v2_archive(
        x_only,
        single,
        group_by="target_gene",
        target_shard_bytes=8192,
        n_workers=1,
        codec="v2-byte-filter",
    )
    multi = tmp_path / "multi"
    write_v2_multi(
        a,
        multi,
        group_by="target_gene",
        target_shard_bytes=8192,
        n_workers=1,
        codec="v2-byte-filter",
    )
    ms = load_manifest(single / "manifest.json")
    assert ms["n_shards"] == load_manifest(multi / "manifest.json")["n_shards"]
    for i in range(ms["n_shards"]):
        assert (multi / "x" / f"shard_{i:04d}.shx").read_bytes() == (
            single / f"shard_{i:04d}.shx"
        ).read_bytes()


def test_multi_overwrite_removes_dropped_family_dir(tmp_path):
    d = tmp_path / "multi"
    a_two_layers = _adata()
    a_two_layers.layers["extra"] = a_two_layers.layers["counts"].copy()
    write_v2_multi(a_two_layers, d, n_shards=None, n_workers=1, target_shard_bytes=8192)
    m1 = load_manifest(d / "manifest.json")
    assert set(m1["matrices"]) == {"x", "layer:counts", "layer:extra", "raw"}
    assert (d / "layer_extra" / "shard_0000.shx").exists()

    a_one_layer = _adata()
    write_v2_multi(
        a_one_layer, d, n_shards=None, n_workers=1, target_shard_bytes=8192, overwrite=True
    )

    assert not (d / "layer_extra").exists()
    assert (d / "x" / "shard_0000.shx").exists()
    assert (d / "layer_counts" / "shard_0000.shx").exists()
    m2 = load_manifest(d / "manifest.json")
    assert set(m2["matrices"]) == {"x", "layer:counts", "raw"}
    assert "layer:extra" not in m2["matrices"]


def test_multi_overwrite_cleans_unknown_prior_family_dir(tmp_path):
    """Overwrite cleanup must remove a PRIOR family dir whose key is not one
    matrix_dir() recognises (e.g. a future Plan-2 'modality:*'), using the old
    manifest's stored template — and must NOT crash on the unknown key."""
    import json

    d = tmp_path / "multi"
    write_v2_multi(_adata(), d, n_shards=None, n_workers=1, target_shard_bytes=8192)
    # Simulate an archive written by a newer writer: inject a family whose key
    # matrix_dir() would reject, with a real on-disk family dir + shard.
    man = json.loads((d / "manifest.json").read_text())
    man["matrices"]["modality:atac"] = {
        "role": "modality:atac",
        "owner": None,
        "n_vars": 5,
        "nnz_per_shard": man["matrices"]["x"]["nnz_per_shard"],
        "total_nnz": man["matrices"]["x"]["total_nnz"],
        "x_data_dtype_on_disk": "uint32",
        "x_indices_dtype_on_disk": "uint32",
        "x_data_dtype_min": "uint16",
        "x_indices_dtype_min": "uint16",
        "codec": man["matrices"]["x"]["codec"],
        "chunk_size_elements": man["matrices"]["x"]["chunk_size_elements"],
        "shard_filename_template": "modality_atac/shard_{:04d}.shx",
    }
    (d / "manifest.json").write_text(json.dumps(man))
    (d / "modality_atac").mkdir()
    (d / "modality_atac" / "shard_0000.shx").write_bytes(b"SHX2stale")

    # A plain Plan-1 overwrite must succeed (no ValueError on the unknown key)
    # AND remove the orphaned modality family dir.
    write_v2_multi(_adata(), d, n_shards=None, n_workers=1, target_shard_bytes=8192, overwrite=True)
    assert not (d / "modality_atac").exists()
    assert (d / "x" / "shard_0000.shx").exists()
    assert "modality:atac" not in load_manifest(d / "manifest.json")["matrices"]


def test_multi_layer_family_decodes_correctly_flat(tmp_path):
    a = _adata()
    d = tmp_path / "multi"
    write_v2_multi(a, d, n_shards=3, n_workers=1)
    m = load_manifest(d / "manifest.json")
    md = m["matrices"]["layer:counts"]
    sub = dict(m)
    sub.update(md)  # per-matrix fields override shared block
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
    X = sp.csr_matrix((data, indices, indptr), shape=(a.n_obs, md["n_vars"]))
    np.testing.assert_array_equal(X.toarray(), a.layers["counts"].toarray())
