"""SHX2 binary format spec tests."""

from __future__ import annotations

import json
import struct

import pytest


def test_magic_constants():
    from cellstream.v2.format import HEADER_PAD_BOUNDARY, SHX_MAGIC, SHX_RESERVED_V0_2

    assert SHX_MAGIC == b"SHX2"
    assert SHX_RESERVED_V0_2 == 0
    assert HEADER_PAD_BOUNDARY == 4096


def test_shx_header_is_json_round_trippable():
    from cellstream.v2.format import ChunkDescriptor, ShxHeader, StreamSpec

    h = ShxHeader(
        schema_version=2,
        shard_idx=7,
        n_rows=1000,
        n_vars=500,
        nnz=20_000,
        compression="zstd",
        shuffle="bitshuffle",
        compression_level=1,
        chunk_size_elements=1_048_576,
        data=StreamSpec("uint16", 1, [ChunkDescriptor(4096, 12_000, 40_000)]),
        indices=StreamSpec("uint16", 1, [ChunkDescriptor(16_096, 11_000, 40_000)]),
        indptr=StreamSpec("int64", 1, [ChunkDescriptor(27_096, 200, 8008)]),
    )
    js = h.to_json()
    parsed = json.loads(js)
    assert parsed["schema_version"] == 2
    assert parsed["nnz"] == 20_000
    h2 = ShxHeader.from_dict(parsed)
    assert h2.data.chunks[0].offset == 4096


def test_write_shx_file_writes_magic_at_offset_0(tmp_path):
    from cellstream.v2.format import ShxHeader, StreamSpec, write_shx_file

    p = tmp_path / "shard.shx"
    h = ShxHeader(
        schema_version=2,
        shard_idx=0,
        n_rows=0,
        n_vars=0,
        nnz=0,
        compression="zstd",
        shuffle="bitshuffle",
        compression_level=1,
        chunk_size_elements=1_048_576,
        data=StreamSpec("uint16", 0, []),
        indices=StreamSpec("uint16", 0, []),
        indptr=StreamSpec("int64", 0, []),
    )
    write_shx_file(p, h, chunk_streams=[b"", b"", b""])
    raw = p.read_bytes()
    assert raw[:4] == b"SHX2"
    reserved, hsize = struct.unpack("<HI", raw[4:10])
    assert reserved == 0
    assert hsize > 0
    parsed = json.loads(raw[10 : 10 + hsize].decode("utf-8"))
    assert parsed["schema_version"] == 2


def test_write_shx_file_pads_header_to_4k_boundary(tmp_path):
    from cellstream.v2.format import HEADER_PAD_BOUNDARY, ShxHeader, StreamSpec, write_shx_file

    p = tmp_path / "shard.shx"
    data_bytes = bytes(range(256)) * 16
    h = ShxHeader(
        schema_version=2,
        shard_idx=0,
        n_rows=0,
        n_vars=0,
        nnz=0,
        compression="zstd",
        shuffle="bitshuffle",
        compression_level=1,
        chunk_size_elements=1_048_576,
        data=StreamSpec("uint16", 1, []),
        indices=StreamSpec("uint16", 0, []),
        indptr=StreamSpec("int64", 0, []),
    )
    write_shx_file(p, h, chunk_streams=[data_bytes, b"", b""])
    raw = p.read_bytes()
    hsize = struct.unpack("<I", raw[6:10])[0]
    header_end = 10 + hsize
    padded = (header_end + HEADER_PAD_BOUNDARY - 1) & ~(HEADER_PAD_BOUNDARY - 1)
    assert raw[padded : padded + 16] == data_bytes[:16]


def test_parse_shx_header_round_trips(tmp_path):
    from cellstream.v2.format import (
        ChunkDescriptor,
        ShxHeader,
        StreamSpec,
        parse_shx_header,
        write_shx_file,
    )

    p = tmp_path / "shard.shx"
    h = ShxHeader(
        schema_version=2,
        shard_idx=3,
        n_rows=100,
        n_vars=50,
        nnz=500,
        compression="zstd",
        shuffle="bitshuffle",
        compression_level=1,
        chunk_size_elements=1_048_576,
        data=StreamSpec("uint16", 1, [ChunkDescriptor(4096, 200, 1000)]),
        indices=StreamSpec("uint16", 1, [ChunkDescriptor(4296, 180, 1000)]),
        indptr=StreamSpec("int64", 1, [ChunkDescriptor(4476, 50, 808)]),
    )
    write_shx_file(p, h, chunk_streams=[b"x" * 200, b"y" * 180, b"z" * 50])
    with p.open("rb") as fh:
        parsed = parse_shx_header(fh)
    assert parsed.schema_version == 2
    assert parsed.shard_idx == 3
    assert parsed.data.chunks[0].offset == 4096


def test_parse_shx_header_rejects_wrong_magic(tmp_path):
    p = tmp_path / "not_a_shard.bin"
    p.write_bytes(b"NOT_SHX2" + b"\x00" * 10000)
    from cellstream.v2.format import parse_shx_header

    with pytest.raises(ValueError, match="magic"):
        with p.open("rb") as fh:
            parse_shx_header(fh)


def test_parse_shx_header_rejects_unknown_major(tmp_path):
    p = tmp_path / "future.shx"
    raw = b"SHX3" + struct.pack("<HI", 0, 50) + b"{}" + b"\x00" * 10000
    p.write_bytes(raw)
    from cellstream.v2.format import parse_shx_header

    with pytest.raises(ValueError, match="major version"):
        with p.open("rb") as fh:
            parse_shx_header(fh)


def test_chunk_descriptor_offsets_match_file_layout(tmp_path):
    from cellstream.v2.format import (
        HEADER_PAD_BOUNDARY,
        ChunkDescriptor,
        ShxHeader,
        StreamSpec,
        parse_shx_header,
        write_shx_file,
    )

    p = tmp_path / "shard.shx"
    # write_shx_file(..., fill_chunk_offsets=True) requires the caller
    # to pre-populate chunks with at least decompressed_size; offsets
    # and compressed_size are filled by the writer.
    data_descs = [
        ChunkDescriptor(offset=0, compressed_size=0, decompressed_size=10),
        ChunkDescriptor(offset=0, compressed_size=0, decompressed_size=10),
    ]
    indices_descs = [
        ChunkDescriptor(offset=0, compressed_size=0, decompressed_size=10),
        ChunkDescriptor(offset=0, compressed_size=0, decompressed_size=10),
    ]
    indptr_descs = [
        ChunkDescriptor(offset=0, compressed_size=0, decompressed_size=56),
    ]
    h = ShxHeader(
        schema_version=2,
        shard_idx=0,
        n_rows=10,
        n_vars=5,
        nnz=20,
        compression="zstd",
        shuffle="bitshuffle",
        compression_level=1,
        chunk_size_elements=1_048_576,
        data=StreamSpec("uint16", 2, data_descs),
        indices=StreamSpec("uint16", 2, indices_descs),
        indptr=StreamSpec("int64", 1, indptr_descs),
    )
    streams = [b"DATA0DATA1", b"INDS0INDS1", b"INDPTR0"]
    write_shx_file(
        p,
        h,
        chunk_streams=streams,
        fill_chunk_offsets=True,
        per_stream_chunk_sizes=[(5, 5), (5, 5), (7,)],
    )
    with p.open("rb") as fh:
        parsed = parse_shx_header(fh)
    raw = p.read_bytes()
    d0_off = parsed.data.chunks[0].offset
    assert raw[d0_off : d0_off + 5] == b"DATA0"
    assert d0_off % HEADER_PAD_BOUNDARY == 0
