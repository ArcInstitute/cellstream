"""Manifest validation for layout=cell (schema 5)."""

from __future__ import annotations

import pytest

from cellstream.errors import IncompatibleSchemaError
from cellstream.manifest import SUPPORTED_SCHEMA_VERSIONS, validate_manifest


def _good():
    return {
        "schema_version": 5,
        "layout": "cell",
        "codec": "pfordelta",
        "n_obs_total": 240,
        "n_vars": 60,
        "total_nnz": 4000,
        "value_dtype_on_disk": "uint16",
        "index_dtype_on_disk": "uint16",
        "x_data_dtype_min": "uint16",
        "x_indices_dtype_min": "uint16",
        "pyfastpfor_codec": "simdfastpfor256",
        "group_by": None,
        "reference": None,
        "sort_within_group": None,
        "sort_order": None,
        "payload_sha256": "0" * 64,
        "writer_fingerprint": {"cellstream_version": "x", "writer_kind": "cell"},
        "created_at": "2026-07-08T00:00:00Z",
    }


def test_schema_5_supported():
    assert 5 in SUPPORTED_SCHEMA_VERSIONS


def test_valid_cell_manifest_passes():
    validate_manifest(_good())  # no raise


def test_cell_manifest_skips_shard_checks():
    # A cell manifest has no n_shards / row_offsets; validation must not require them.
    m = _good()
    assert "n_shards" not in m and "row_offsets" not in m
    validate_manifest(m)


@pytest.mark.parametrize("field", ["n_obs_total", "n_vars", "total_nnz", "codec"])
def test_missing_required_field_raises(field):
    m = _good()
    del m[field]
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_bad_value_dtype_raises():
    m = _good()
    m["value_dtype_on_disk"] = "float32"
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_negative_n_obs_raises():
    m = _good()
    m["n_obs_total"] = -1
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_bool_n_obs_rejected():
    m = _good()
    m["n_obs_total"] = True
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_null_payload_sha256_ok():
    m = _good()
    m["payload_sha256"] = None
    m["indices_canonical"] = True
    validate_manifest(m)  # no raise: opt-in checksum off + canonical flag present


def test_indices_canonical_absent_ok():
    m = _good()
    m.pop("indices_canonical", None)
    validate_manifest(m)  # older archives lack the key -> tolerated


def test_bad_payload_sha256_raises():
    m = _good()
    m["payload_sha256"] = "deadbeef"  # not null and not 64 hex chars
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_nonhex_payload_sha256_raises():
    m = _good()
    m["payload_sha256"] = "z" * 64  # right length but not hexadecimal
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_bad_indices_canonical_raises():
    m = _good()
    m["indices_canonical"] = "yes"  # not a bool
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


# --- schema_version 6 (zstd) ---


def _base_zstd_manifest(value_dtype="float64"):
    return {
        "schema_version": 6,
        "layout": "cell",
        "codec": "zstd",
        "n_obs_total": 10,
        "n_vars": 5,
        "total_nnz": 20,
        "value_dtype_on_disk": value_dtype,
        "index_dtype_on_disk": "uint32",
        "x_data_dtype_min": value_dtype if value_dtype.startswith("float") else "uint8",
        "x_indices_dtype_min": "uint16",
        "pyfastpfor_codec": None,
        "payload_sha256": None,
    }


@pytest.mark.parametrize("value_dtype", ["uint32", "float32", "float64"])
def test_validate_cell_v6_ok(value_dtype):
    validate_manifest(_base_zstd_manifest(value_dtype))  # no raise


def test_supported_schema_versions_includes_6():
    assert 6 in SUPPORTED_SCHEMA_VERSIONS


def test_validate_cell_v6_rejects_bad_codec():
    m = _base_zstd_manifest()
    m["codec"] = "pfordelta"  # pfordelta must be v5, not v6
    with pytest.raises(IncompatibleSchemaError, match="schema_version 6 requires codec 'zstd'"):
        validate_manifest(m)


def test_validate_cell_v6_rejects_uint16_value():
    m = _base_zstd_manifest()
    m["value_dtype_on_disk"] = "uint16"  # zstd ints are always uint32
    with pytest.raises(IncompatibleSchemaError, match="value_dtype_on_disk"):
        validate_manifest(m)


def test_validate_cell_v6_rejects_uint16_index():
    m = _base_zstd_manifest()
    m["index_dtype_on_disk"] = "uint16"  # zstd indices are always uint32
    with pytest.raises(IncompatibleSchemaError, match="index_dtype_on_disk"):
        validate_manifest(m)


def test_validate_cell_v5_still_requires_pfordelta():
    m = _base_zstd_manifest()
    m["schema_version"] = 5  # v5 must be pfordelta with a string pyfastpfor_codec
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)
