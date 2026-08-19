"""manifest.py: schema_version refusal, row offset math, load/dump round-trip."""

import json

import pytest


def test_compute_row_offsets_even_split():
    from cellstream.manifest import compute_row_offsets

    offsets = compute_row_offsets(n_obs=100, n_shards=4)
    assert offsets == [0, 25, 50, 75, 100]


def test_compute_row_offsets_with_remainder():
    from cellstream.manifest import compute_row_offsets

    offsets = compute_row_offsets(n_obs=103, n_shards=4)
    # First 3 shards absorb the remainder (26 cells), last has 25.
    assert offsets == [0, 26, 52, 78, 103]


def test_compute_row_offsets_rejects_more_shards_than_cells():
    from cellstream.manifest import compute_row_offsets

    with pytest.raises(ValueError):
        compute_row_offsets(n_obs=3, n_shards=4)


def test_round_trip(tmp_path):
    from cellstream.manifest import dump_manifest, load_manifest

    m = {
        "schema_version": 1,
        "n_shards": 4,
        "n_obs_total": 100,
        "n_vars": 50,
        "row_offsets": [0, 25, 50, 75, 100],
        "nnz_per_shard": [10, 20, 30, 40],
        "total_nnz": 100,
        "codec": "blosc-zstd-1-u16u16",
        "x_data_dtype_on_disk": "uint16",
        "x_indices_dtype_on_disk": "uint16",
        "writer_fingerprint": {
            # the pre-rename key: a schema-v1 manifest could never have said anything else
            "shardad_version": "0.1.0",
            "n_workers_at_write": 8,
            "blosc_nthreads_per_worker": 4,
        },
        "created_at": "2026-05-25T10:00:00Z",
    }
    path = tmp_path / "manifest.json"
    dump_manifest(path, m)
    loaded = load_manifest(path)
    assert loaded == m


def test_unknown_schema_version_raises(tmp_path):
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import load_manifest

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 999, "n_shards": 1}))
    with pytest.raises(IncompatibleSchemaError) as exc:
        load_manifest(path)
    assert "999" in str(exc.value)


def test_bisect_shard_for_row():
    from cellstream.manifest import shard_for_row

    offsets = [0, 25, 50, 75, 100]
    assert shard_for_row(offsets, 0) == 0
    assert shard_for_row(offsets, 24) == 0
    assert shard_for_row(offsets, 25) == 1
    assert shard_for_row(offsets, 99) == 3


def test_shards_for_range():
    from cellstream.manifest import shards_for_range

    offsets = [0, 25, 50, 75, 100]
    # Range [30, 80) intersects shards 1, 2, 3.
    assert shards_for_range(offsets, 30, 80) == [1, 2, 3]
    # Range [25, 50) is entirely in shard 1.
    assert shards_for_range(offsets, 25, 50) == [1]
    # Empty range.
    assert shards_for_range(offsets, 50, 50) == []


def test_load_manifest_accepts_schema_v2(tmp_path):
    """v2 manifests load cleanly."""
    from cellstream.manifest import dump_manifest, load_manifest

    m = {
        "schema_version": 2,
        "shard_filename_template": "shard_{:04d}.shx",
        "n_shards": 1,
        "n_obs_total": 10,
        "n_vars": 5,
        "row_offsets": [0, 10],
        "nnz_per_shard": [50],
        "total_nnz": 50,
        "x_data_dtype_on_disk": "uint16",
        "x_indices_dtype_on_disk": "uint16",
        "codec": "zstd-1-u16u16-bitshuffle",
        "chunk_size_elements": 1048576,
        "writer_fingerprint": "test",
        "created_at": "2026-01-01T00:00:00Z",
    }
    p = tmp_path / "manifest.json"
    dump_manifest(p, m)
    loaded = load_manifest(p)
    assert loaded["schema_version"] == 2


def test_load_manifest_still_accepts_schema_v1(tmp_path):
    """v1 manifests keep working."""
    from cellstream.manifest import dump_manifest, load_manifest

    m = {
        "schema_version": 1,
        "n_shards": 1,
        "n_obs_total": 10,
        "n_vars": 5,
        "row_offsets": [0, 10],
        "nnz_per_shard": [50],
        "total_nnz": 50,
        "x_data_dtype_on_disk": "uint16",
        "x_indices_dtype_on_disk": "uint16",
        "codec": "blosc-zstd-1-u16u16",
        "writer_fingerprint": "test",
        "created_at": "2026-01-01T00:00:00Z",
    }
    p = tmp_path / "manifest.json"
    dump_manifest(p, m)
    assert load_manifest(p)["schema_version"] == 1


def test_load_manifest_rejects_unknown_schema_versions():
    """Anything outside {1, 2, 3} is loud."""
    import json
    import pathlib
    import tempfile

    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import load_manifest

    for bad in (0, 4, "v2", None):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "m.json"
            p.write_text(json.dumps({"schema_version": bad}))
            try:
                load_manifest(p)
            except IncompatibleSchemaError:
                pass
            else:
                raise AssertionError(f"expected raise on schema={bad!r}")


def test_shard_filename_template_default_for_v1():
    """v1 manifests don't carry shard_filename_template; the resolver fills
    in the implicit `shard_{:04d}.h5ad`."""
    from cellstream.manifest import shard_filename

    assert shard_filename({"schema_version": 1}, 7) == "shard_0007.h5ad"


def test_shard_filename_template_explicit_for_v2():
    """v2 manifests carry the template explicitly."""
    from cellstream.manifest import shard_filename

    m = {"schema_version": 2, "shard_filename_template": "shard_{:04d}.shx"}
    assert shard_filename(m, 7) == "shard_0007.shx"


# ---------------------------------------------------------------------------
# validate_manifest: row_offsets invariants (ultrareview L2)
# ---------------------------------------------------------------------------


def _valid_manifest():
    return {
        "schema_version": 2,
        "n_shards": 3,
        "n_obs_total": 100,
        "n_vars": 10,
        "row_offsets": [0, 40, 70, 100],
    }


def test_validate_manifest_accepts_valid():
    from cellstream.manifest import validate_manifest

    validate_manifest(_valid_manifest())  # must not raise


def test_validate_manifest_accepts_empty_shard_nondecreasing():
    """Grouped writes can emit empty shards -> equal adjacent offsets are OK."""
    from cellstream.manifest import validate_manifest

    m = _valid_manifest()
    m["row_offsets"] = [0, 40, 40, 100]  # middle shard is empty
    validate_manifest(m)  # must not raise


def test_validate_manifest_rejects_wrong_length():
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import validate_manifest

    m = _valid_manifest()
    m["row_offsets"] = [0, 40, 100]  # len 3, expected n_shards+1 == 4
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_validate_manifest_rejects_nonzero_first():
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import validate_manifest

    m = _valid_manifest()
    m["row_offsets"] = [1, 40, 70, 100]
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_validate_manifest_rejects_last_ne_n_obs():
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import validate_manifest

    m = _valid_manifest()
    m["row_offsets"] = [0, 40, 70, 99]  # last != n_obs_total (100)
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_validate_manifest_rejects_decreasing():
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import validate_manifest

    m = _valid_manifest()
    m["row_offsets"] = [0, 70, 40, 100]  # decreasing pair
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_validate_manifest_rejects_non_int_element():
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import validate_manifest

    m = _valid_manifest()
    m["row_offsets"] = [0, 40, 70.0, 100]  # float element
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_validate_manifest_rejects_bool_element():
    """bool is an int subclass; it must still be rejected as a row offset."""
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import validate_manifest

    m = _valid_manifest()
    m["row_offsets"] = [0, True, 70, 100]
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_validate_manifest_rejects_missing_row_offsets():
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import validate_manifest

    m = _valid_manifest()
    del m["row_offsets"]
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_validate_manifest_rejects_negative_n_shards():
    """n_shards=-1 with an empty row_offsets must raise a clean error, not an
    IndexError at offsets[0] (-1 + 1 == 0 == len([]))."""
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import validate_manifest

    m = _valid_manifest()
    m["n_shards"] = -1
    m["row_offsets"] = []
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_validate_manifest_rejects_negative_n_obs():
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import validate_manifest

    m = _valid_manifest()
    m["n_obs_total"] = -100
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)


def test_load_manifest_rejects_corrupt_row_offsets(tmp_path):
    """load_manifest runs validate_manifest after the schema check."""
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import dump_manifest, load_manifest

    m = _valid_manifest()
    m["row_offsets"] = [0, 40, 70, 200]  # last != n_obs_total
    p = tmp_path / "manifest.json"
    dump_manifest(p, m)
    with pytest.raises(IncompatibleSchemaError):
        load_manifest(p)


# ---------------------------------------------------------------------------
# validate_group_records: groups.json record invariants (ultrareview L1)
# ---------------------------------------------------------------------------


def _valid_records():
    return [
        {"shard": 0, "row_start": 0, "row_stop": 40, "label": "A", "role": "reference"},
        {"shard": 1, "row_start": 40, "row_stop": 70, "label": "B"},
        {"shard": 2, "row_start": 70, "row_stop": 100, "label": "C"},
    ]


def test_validate_group_records_accepts_valid():
    from cellstream.manifest import validate_group_records

    validate_group_records(_valid_records(), n_shards=3, n_obs=100)  # must not raise


def test_validate_group_records_rejects_missing_key():
    from cellstream.manifest import validate_group_records

    recs = _valid_records()
    del recs[1]["row_stop"]
    with pytest.raises(ValueError):
        validate_group_records(recs, n_shards=3, n_obs=100)


def test_validate_group_records_rejects_non_int_shard():
    from cellstream.manifest import validate_group_records

    recs = _valid_records()
    recs[1]["shard"] = "1"
    with pytest.raises(ValueError):
        validate_group_records(recs, n_shards=3, n_obs=100)


def test_validate_group_records_rejects_bool_shard():
    from cellstream.manifest import validate_group_records

    recs = _valid_records()
    recs[0]["shard"] = True  # bool is an int subclass; reject it
    with pytest.raises(ValueError):
        validate_group_records(recs, n_shards=3, n_obs=100)


def test_validate_group_records_rejects_non_str_label():
    from cellstream.manifest import validate_group_records

    recs = _valid_records()
    recs[1]["label"] = 123
    with pytest.raises(ValueError):
        validate_group_records(recs, n_shards=3, n_obs=100)


def test_validate_group_records_rejects_shard_out_of_range():
    from cellstream.manifest import validate_group_records

    recs = _valid_records()
    recs[2]["shard"] = 3  # n_shards == 3 -> valid range is [0, 3)
    with pytest.raises(ValueError):
        validate_group_records(recs, n_shards=3, n_obs=100)


def test_validate_group_records_rejects_negative_shard():
    from cellstream.manifest import validate_group_records

    recs = _valid_records()
    recs[0]["shard"] = -1
    with pytest.raises(ValueError):
        validate_group_records(recs, n_shards=3, n_obs=100)


def test_validate_group_records_rejects_start_gt_stop():
    from cellstream.manifest import validate_group_records

    recs = _valid_records()
    recs[1]["row_start"] = 71  # row_start > row_stop (70)
    with pytest.raises(ValueError):
        validate_group_records(recs, n_shards=3, n_obs=100)


def test_validate_group_records_rejects_stop_gt_n_obs():
    from cellstream.manifest import validate_group_records

    recs = _valid_records()
    recs[2]["row_stop"] = 101  # > n_obs (100)
    with pytest.raises(ValueError):
        validate_group_records(recs, n_shards=3, n_obs=100)


def test_validate_group_records_rejects_negative_start():
    from cellstream.manifest import validate_group_records

    recs = _valid_records()
    recs[0]["row_start"] = -1
    with pytest.raises(ValueError):
        validate_group_records(recs, n_shards=3, n_obs=100)
