"""Packing/opening a flat per-cell (layout=cell) SHPK archive."""

from __future__ import annotations

import json

import numpy as np

from cellstream.packed.reader import PackedArchive
from cellstream.packed.writer import _pack_directory as pack


def _hand_build_cell_dir(d):
    """Write the members a cell archive needs into directory ``d`` (no shards)."""
    payload = b"\x01\x02\x03\x04\x05\x06\x07\x08"  # opaque frame bytes
    offsets = np.array([0, 4, 8], dtype=np.int64)  # 2 cells
    indptr = np.array([0, 1, 2], dtype=np.int64)
    (d / "x_payload").write_bytes(payload)
    (d / "x_offsets").write_bytes(offsets.tobytes())
    (d / "x_indptr").write_bytes(indptr.tobytes())
    manifest = {
        "schema_version": 5,
        "layout": "cell",
        "codec": "pfordelta",
        "n_obs_total": 2,
        "n_vars": 3,
        "total_nnz": 2,
        "value_dtype_on_disk": "uint16",
        "index_dtype_on_disk": "uint16",
        "x_data_dtype_min": "uint16",
        "x_indices_dtype_min": "uint16",
        "pyfastpfor_codec": "simdfastpfor256",
        "payload_sha256": "0" * 64,
        "writer_fingerprint": {"cellstream_version": "x", "writer_kind": "cell"},
        "created_at": "2026-07-08T00:00:00Z",
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    return payload, offsets, indptr


def test_pack_and_open_cell_archive(tmp_path):
    src = tmp_path / "cell_src"
    src.mkdir()
    payload, offsets, indptr = _hand_build_cell_dir(src)
    dst = tmp_path / "cell.shad"
    pack(src, dst)

    pa = PackedArchive(dst)
    assert pa.manifest["layout"] == "cell"
    assert pa.n_shards == 0  # no shard members
    # member bytes round-trip byte-identical
    assert pa.member_bytes("x_payload") == payload
    np.testing.assert_array_equal(
        np.frombuffer(pa.member_bytes("x_offsets"), dtype=np.int64), offsets
    )
    np.testing.assert_array_equal(
        np.frombuffer(pa.member_bytes("x_indptr"), dtype=np.int64), indptr
    )


def test_member_location_payload_is_4k_aligned(tmp_path):
    src = tmp_path / "cell_src"
    src.mkdir()
    _hand_build_cell_dir(src)
    dst = tmp_path / "cell.shad"
    pack(src, dst)

    pa = PackedArchive(dst)
    path, base, length = pa.member_location("x_payload")
    assert path == str(dst)
    assert base % 4096 == 0
    assert length == 8
