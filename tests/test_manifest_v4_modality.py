from __future__ import annotations

import pytest

from cellstream.errors import IncompatibleSchemaError
from cellstream.manifest import build_v4_manifest, validate_manifest


def _mx(n_vars, dir_, role, owner=None, has_var=True, modality_name=None, meta=False):
    d = {
        "role": role,
        "owner": owner,
        "n_vars": n_vars,
        "nnz_per_shard": [2, 3],
        "total_nnz": 5,
        "x_data_dtype_on_disk": "uint32",
        "x_indices_dtype_on_disk": "uint32",
        "x_data_dtype_min": "uint16",
        "x_indices_dtype_min": "uint16",
        "codec": "v2-byte-filter",
        "chunk_size_elements": 1048576,
        "shard_filename_template": f"{dir_}/shard_{{:04d}}.shx",
    }
    if has_var:
        d["var_filename"] = f"var/{dir_}.parquet"
    if modality_name is not None:
        d["modality_name"] = modality_name
    if meta:
        d["meta_filename"] = f"meta/{dir_}.h5ad"
    return d


def _good():
    return build_v4_manifest(
        n_obs_total=5,
        row_offsets=[0, 2, 5],
        matrices={
            "x": _mx(3, "x", "x", modality_name="rna", meta=True),
            "modality:atac": _mx(
                4, "modality_atac", "modality:atac", modality_name="atac", meta=True
            ),
            "modality:atac:layer:tfidf": _mx(
                4, "modality_atac__layer_tfidf", "layer:tfidf", owner="modality:atac", has_var=False
            ),
        },
        writer_fingerprint={"cellstream_version": "test"},
        created_at="2026-07-04T00:00:00Z",
    )


def test_good_modality_manifest_validates():
    validate_manifest(_good())  # must not raise


def test_layer_owner_must_be_own_var_family():
    m = _good()
    # point the modality layer at raw-like/layer owner -> reject
    m["matrices"]["bad:layer:z"] = _mx(
        3, "bad__layer_z", "layer:z", owner="modality:atac:layer:tfidf", has_var=False
    )
    with pytest.raises(IncompatibleSchemaError, match="owner|own-var"):
        validate_manifest(m)


def test_modality_requires_own_var():
    m = _good()
    del m["matrices"]["modality:atac"]["var_filename"]
    with pytest.raises(IncompatibleSchemaError, match="modality.*var"):
        validate_manifest(m)


def test_modality_present_requires_x_modality_name():
    m = _good()
    del m["matrices"]["x"]["modality_name"]
    with pytest.raises(IncompatibleSchemaError, match="modality_name"):
        validate_manifest(m)


def test_modality_owner_must_be_none():
    m = _good()
    m["matrices"]["modality:atac"]["owner"] = "x"
    with pytest.raises(IncompatibleSchemaError, match="owner"):
        validate_manifest(m)
