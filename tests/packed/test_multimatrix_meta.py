# tests/packed/test_multimatrix_meta.py
from __future__ import annotations

import io

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellstream import header as _h
from cellstream.manifest import build_v4_manifest, dump_manifest
from cellstream.packed.reader import PackedArchive
from cellstream.packed.writer import _pack_directory as pack


def _v4_dir_with_meta(tmp_path):
    d = tmp_path / "v4dir"
    (d / "x").mkdir(parents=True)
    (d / "modality_atac").mkdir()
    (d / "var").mkdir()
    (d / "meta").mkdir()
    for i in range(2):
        (d / "x" / f"shard_{i:04d}.shx").write_bytes(b"SHX2x" + bytes([i]))
        (d / "modality_atac" / f"shard_{i:04d}.shx").write_bytes(b"SHX2a" + bytes([i]))
    pd.DataFrame(index=["g0", "g1", "g2"]).to_parquet(d / "var" / "x.parquet")
    pd.DataFrame(index=["p0", "p1", "p2", "p3"]).to_parquet(d / "var" / "modality_atac.parquet")
    atac = ad.AnnData(
        X=sp.csr_matrix((4, 4), dtype="float32"),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["p0", "p1", "p2", "p3"]),
    )
    atac.obsm["X_lsi"] = np.arange(8, dtype="float32").reshape(4, 2)
    _h.write_modality_meta(atac, d / "meta" / "modality_atac.h5ad")
    obs = pd.DataFrame(index=[f"c{i}" for i in range(4)])
    hh = ad.AnnData(
        X=sp.csr_matrix((4, 3), dtype="float32"),
        obs=obs,
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    _h.write_header(hh, d / "header.h5ad")
    _h.write_obs_parquet(obs, d / "obs.parquet")

    def _mx(n_vars, dir_, role, owner=None, has_var=True, mod=None, meta=False):
        e = {
            "role": role,
            "owner": owner,
            "n_vars": n_vars,
            "nnz_per_shard": [1, 1],
            "total_nnz": 2,
            "x_data_dtype_on_disk": "uint32",
            "x_indices_dtype_on_disk": "uint32",
            "x_data_dtype_min": "uint16",
            "x_indices_dtype_min": "uint16",
            "codec": "v2-byte-filter",
            "chunk_size_elements": 1048576,
            "shard_filename_template": f"{dir_}/shard_{{:04d}}.shx",
        }
        if has_var:
            e["var_filename"] = f"var/{dir_}.parquet"
        if mod is not None:
            e["modality_name"] = mod
        if meta:
            e["meta_filename"] = f"meta/{dir_}.h5ad"
        return e

    m = build_v4_manifest(
        n_obs_total=4,
        row_offsets=[0, 2, 4],
        matrices={
            "x": _mx(3, "x", "x", mod="rna"),
            "modality:atac": _mx(4, "modality_atac", "modality:atac", mod="atac", meta=True),
        },
        writer_fingerprint={"cellstream_version": "test"},
        created_at="2026-07-04T00:00:00Z",
    )
    dump_manifest(d / "manifest.json", m)
    return d


def test_packed_v4_meta_bytes(tmp_path):
    d = _v4_dir_with_meta(tmp_path)
    f = tmp_path / "a.shad"
    pack(d, f)
    pa = PackedArchive(f)
    back = ad.read_h5ad(io.BytesIO(pa.meta_bytes("modality:atac")))
    np.testing.assert_array_equal(back.obsm["X_lsi"], np.arange(8, dtype="float32").reshape(4, 2))
    # modality shard still locatable (Plan-1 generic path)
    fpath, base, length = pa.shard_location(1, matrix="modality:atac")
    assert base % 4096 == 0 and length > 0
