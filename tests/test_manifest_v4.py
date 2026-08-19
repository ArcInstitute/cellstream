# tests/test_manifest_v4.py
from __future__ import annotations

import pytest

from cellstream.errors import IncompatibleSchemaError
from cellstream.manifest import (
    SUPPORTED_SCHEMA_VERSIONS,
    build_v4_manifest,
    dump_manifest,
    load_manifest,
    validate_manifest,
)


def _matrix(n_vars, nnz_per_shard, dir_, role="x", owner=None, has_var=True):
    d = {
        "role": role,
        "owner": owner,
        "n_vars": n_vars,
        "nnz_per_shard": nnz_per_shard,
        "total_nnz": sum(nnz_per_shard),
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
    return d


def _good_v4():
    return build_v4_manifest(
        n_obs_total=10,
        row_offsets=[0, 5, 10],
        matrices={
            "x": _matrix(3, [4, 6], "x", role="x"),
            "layer:counts": _matrix(
                3, [4, 6], "layer_counts", role="layer:counts", owner="x", has_var=False
            ),
            "raw": _matrix(4, [5, 7], "raw", role="raw"),
        },
        writer_fingerprint={"cellstream_version": "test"},
        created_at="2026-07-04T00:00:00Z",
    )


def test_v4_in_supported_versions():
    assert SUPPORTED_SCHEMA_VERSIONS == (1, 2, 3, 4, 5, 6)


def test_build_v4_sets_shared_block():
    m = _good_v4()
    assert m["schema_version"] == 4
    assert m["n_shards"] == 2
    assert m["n_obs_total"] == 10
    assert m["n_vars"] == 3  # mirrors primary x
    assert m["shard_filename_template"] == "x/shard_{:04d}.shx"
    assert set(m["matrices"]) == {"x", "layer:counts", "raw"}


def test_v4_round_trip(tmp_path):
    m = _good_v4()
    p = tmp_path / "manifest.json"
    dump_manifest(p, m)
    assert load_manifest(p) == m


def test_v4_validate_rejects_bad_nnz_length():
    m = _good_v4()
    m["matrices"]["x"]["nnz_per_shard"] = [4, 6, 1]  # len 3 != n_shards 2
    with pytest.raises(IncompatibleSchemaError, match="nnz_per_shard"):
        validate_manifest(m)


def test_v4_validate_requires_one_primary():
    m = _good_v4()
    m["matrices"]["x"]["role"] = "raw"
    with pytest.raises(IncompatibleSchemaError, match="exactly one .*role.*x"):
        validate_manifest(m)


def test_v4_validate_rejects_dangling_owner():
    m = _good_v4()
    m["matrices"]["layer:counts"]["owner"] = "nope"
    with pytest.raises(IncompatibleSchemaError, match="owner"):
        validate_manifest(m)


def test_v4_validate_rejects_negative_n_vars():
    m = _good_v4()
    m["matrices"]["x"]["n_vars"] = -1
    with pytest.raises(IncompatibleSchemaError, match="n_vars"):
        validate_manifest(m)


def test_v4_validate_rejects_bool_n_vars():
    m = _good_v4()
    m["matrices"]["x"]["n_vars"] = True
    with pytest.raises(IncompatibleSchemaError, match="n_vars"):
        validate_manifest(m)
