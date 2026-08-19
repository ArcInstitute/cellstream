"""Packed container binary format: head magic, JSON footer, fixed trailer.

Layout:
  [0, 4096)        head block: b"SHPK" + format_version u16, zero-padded
  [4096, sre)      shard region (4 KB-aligned, immutable)
  [sre, footer)    metadata members (header.h5ad, obs.parquet, manifest.json, groups.json)
  [footer]         UTF-8 JSON member index
  [trailer @ EOF]  fixed 64-byte binary, points back to the footer

The trailer is fixed-size so it is locatable by reading the last
TRAILER_SIZE bytes without knowing the layout (ZIP/Parquet pattern). All
extensibility lives in the JSON footer.
"""

from __future__ import annotations

import json
import struct
import zlib

from cellstream.errors import PackedArchiveError

HEAD_MAGIC = b"SHPK"
TRAILER_MAGIC = b"SHPKEND\x00"  # 8 bytes, distinct from head magic
CONTAINER_VERSION = 1
ALIGNMENT = 4096
HEAD_BLOCK_SIZE = 4096
TRAILER_SIZE = 64

# magic(8) ver(2) reserved(2) footer_off(8) footer_len(8) shard_region_end(8) crc(4) = 40 bytes
_TRAILER = struct.Struct("<8sHHQQQI")


def align_up(n: int, alignment: int = ALIGNMENT) -> int:
    return (n + alignment - 1) & ~(alignment - 1)


def pack_trailer(
    *,
    footer_offset,
    footer_length,
    shard_region_end,
    footer_crc32,
    version: int = CONTAINER_VERSION,
) -> bytes:
    raw = _TRAILER.pack(
        TRAILER_MAGIC,
        version,
        0,
        footer_offset,
        footer_length,
        shard_region_end,
        footer_crc32 & 0xFFFFFFFF,
    )
    return raw + b"\x00" * (TRAILER_SIZE - len(raw))


def parse_trailer(buf: bytes) -> dict:
    if len(buf) < TRAILER_SIZE:
        raise PackedArchiveError(f"trailer too short: {len(buf)} < {TRAILER_SIZE}")
    magic, version, _res, foff, flen, sre, crc = _TRAILER.unpack(buf[: _TRAILER.size])
    if magic != TRAILER_MAGIC:
        raise PackedArchiveError(
            f"bad packed trailer magic {magic!r}; file is not a packed archive "
            f"or its tail was truncated by an interrupted edit."
        )
    return {
        "format_version": version,
        "footer_offset": foff,
        "footer_length": flen,
        "shard_region_end": sre,
        "footer_crc32": crc,
    }


def serialize_footer(
    members, *, shard_region_end, alignment: int = ALIGNMENT, version: int = CONTAINER_VERSION
) -> bytes:
    # v4 (multi-matrix): shard members additionally carry an optional "matrix"
    # tag (family key) alongside "shard_idx"; var sidecars use role "var". The
    # footer is a plain JSON dict dump, so these extra fields round-trip as-is.
    return json.dumps(
        {
            "container_version": version,
            "alignment": alignment,
            "shard_region_end": shard_region_end,
            "members": members,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def parse_footer(buf: bytes) -> dict:
    return json.loads(buf.decode("utf-8"))


def footer_crc32(buf: bytes) -> int:
    return zlib.crc32(buf) & 0xFFFFFFFF
