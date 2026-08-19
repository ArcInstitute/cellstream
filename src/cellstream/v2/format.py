"""SHX2 binary format spec.

Layout (see v0.2 design spec):
  offset 0 : 4 bytes magic           b"SHX2"
  offset 4 : 2 bytes reserved u16    = 0 in v0.2
  offset 6 : 4 bytes header_size u32 length of UTF-8 JSON that follows
  offset 10: variable JSON header
  pad to 4 KB
  thereafter: concatenated compressed chunk bytes (addressed by header).

Versioning:
  major bump (incompatible)   -> bump magic bytes ("SHX2" -> "SHX3"...)
  minor bump (compatible add) -> bump reserved u16

All chunk descriptor sizes are byte counts (not element counts). Readers
compute n_elements via decompressed_size // dtype.itemsize.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

SHX_MAGIC = b"SHX2"
SHX_RESERVED_V0_2 = 0
HEADER_PAD_BOUNDARY = 4096
_MAX_HEADER_FIXEDPOINT_ITERS = (
    8  # cap for the header/offset fixed-point loop; expected ≤ 2-3 in practice
)
_FIXED_PREFIX_BYTES = 10

# Decompression-bomb / corrupt-header caps on a single chunk, mirroring the Rust
# decoder (rust/src/decode.rs MAX_CHUNK_{COMPRESSED,DECOMPRESSED}_BYTES). Legit
# chunks are <= ~8 MiB (CHUNK_ELEMS=1M x 8 bytes); cap 8x above that. Enforced at
# parse so a crafted header cannot drive an unbounded n_elements / allocation in
# the pure-Python decode fallback.
MAX_CHUNK_COMPRESSED_BYTES = 64 << 20  # 64 MiB
MAX_CHUNK_DECOMPRESSED_BYTES = 64 << 20  # 64 MiB


@dataclass
class ChunkDescriptor:
    offset: int  # byte offset within the .shx file
    compressed_size: int  # bytes
    decompressed_size: int  # bytes of on-disk dtype

    def to_dict(self) -> dict:
        return {
            "offset": self.offset,
            "compressed_size": self.compressed_size,
            "decompressed_size": self.decompressed_size,
        }

    @classmethod
    def from_dict(cls, d):
        # Surface every malformed-header failure as ValueError (a non-dict, a missing
        # key, or a wrong type must not leak KeyError/TypeError past the decode error
        # model). .get() turns a missing key into None, caught by the isinstance check.
        if not isinstance(d, dict):
            raise ValueError(
                f"ChunkDescriptor.from_dict expects a dict; got {type(d).__name__} "
                f"(archive may be corrupt)."
            )
        offset = d.get("offset")
        compressed_size = d.get("compressed_size")
        decompressed_size = d.get("decompressed_size")
        # Validate the untrusted chunk header at parse: non-negative ints within the
        # bomb caps, so n_elements (= decompressed_size // itemsize) stays bounded
        # for every stream and shuffle marker downstream.
        for name, value in (
            ("offset", offset),
            ("compressed_size", compressed_size),
            ("decompressed_size", decompressed_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"ChunkDescriptor.{name} must be a non-negative int; got {value!r} "
                    f"(archive may be corrupt)."
                )
        if compressed_size > MAX_CHUNK_COMPRESSED_BYTES:
            raise ValueError(
                f"ChunkDescriptor.compressed_size {compressed_size} exceeds the "
                f"{MAX_CHUNK_COMPRESSED_BYTES}-byte cap (archive may be corrupt)."
            )
        if decompressed_size > MAX_CHUNK_DECOMPRESSED_BYTES:
            raise ValueError(
                f"ChunkDescriptor.decompressed_size {decompressed_size} exceeds the "
                f"{MAX_CHUNK_DECOMPRESSED_BYTES}-byte cap (archive may be corrupt)."
            )
        return cls(offset, compressed_size, decompressed_size)


@dataclass
class StreamSpec:
    dtype: str
    n_chunks: int
    chunks: list[ChunkDescriptor] = field(default_factory=list)

    def to_dict(self):
        return {
            "dtype": self.dtype,
            "n_chunks": self.n_chunks,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["dtype"], d["n_chunks"], [ChunkDescriptor.from_dict(c) for c in d["chunks"]])


@dataclass
class ShxHeader:
    schema_version: int
    shard_idx: int
    n_rows: int
    n_vars: int
    nnz: int
    compression: str
    shuffle: str
    compression_level: int
    chunk_size_elements: int
    data: StreamSpec
    indices: StreamSpec
    indptr: StreamSpec

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "shard_idx": self.shard_idx,
                "n_rows": self.n_rows,
                "n_vars": self.n_vars,
                "nnz": self.nnz,
                "compression": self.compression,
                "shuffle": self.shuffle,
                "compression_level": self.compression_level,
                "chunk_size_elements": self.chunk_size_elements,
                "data": self.data.to_dict(),
                "indices": self.indices.to_dict(),
                "indptr": self.indptr.to_dict(),
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["schema_version"],
            d["shard_idx"],
            d["n_rows"],
            d["n_vars"],
            d["nnz"],
            d["compression"],
            d["shuffle"],
            d["compression_level"],
            d["chunk_size_elements"],
            StreamSpec.from_dict(d["data"]),
            StreamSpec.from_dict(d["indices"]),
            StreamSpec.from_dict(d["indptr"]),
        )


def _padded_header_end(header_json_len: int) -> int:
    raw_end = _FIXED_PREFIX_BYTES + header_json_len
    return (raw_end + HEADER_PAD_BOUNDARY - 1) & ~(HEADER_PAD_BOUNDARY - 1)


def write_shx_file(
    path: str | Path,
    header: ShxHeader,
    chunk_streams,
    *,
    fill_chunk_offsets: bool = False,
    per_stream_chunk_sizes=None,
) -> None:
    """Write a .shx file atomically (.tmp + os.replace).

    When fill_chunk_offsets=True, the caller must have pre-populated each
    stream's chunk descriptors with at least `decompressed_size` (the
    writer cannot infer it). offset and compressed_size are filled in by
    this function. Raises ValueError if the pre-populated chunk count
    doesn't match per_stream_chunk_sizes.

    Iterates offset-fill to a fixed point (typically 1-2 iterations:
    offsets grow as header grows, which grows offsets; the 4-KB padding
    boundary makes this converge quickly).
    """
    path = Path(path)
    if fill_chunk_offsets:
        if per_stream_chunk_sizes is None:
            raise ValueError("per_stream_chunk_sizes required when fill_chunk_offsets=True")
        for stream, sizes in zip(
            (header.data, header.indices, header.indptr),
            per_stream_chunk_sizes,
            strict=True,
        ):
            if len(stream.chunks) != len(sizes):
                raise ValueError(
                    f"fill_chunk_offsets=True requires header chunks to be "
                    f"pre-populated with decompressed_size; stream has "
                    f"{len(stream.chunks)} chunks vs {len(sizes)} expected sizes."
                )
        prev_data_start = -1
        converged = False
        for _ in range(_MAX_HEADER_FIXEDPOINT_ITERS):
            js = header.to_json()
            data_start = _padded_header_end(len(js.encode("utf-8")))
            if data_start == prev_data_start:
                converged = True
                break
            prev_data_start = data_start
            cursor = data_start
            for stream, sizes in zip(
                (header.data, header.indices, header.indptr),
                per_stream_chunk_sizes,
                strict=True,
            ):
                for c, sz in zip(stream.chunks, sizes, strict=True):
                    c.offset = cursor
                    c.compressed_size = sz
                    cursor += sz
        if not converged:
            raise RuntimeError(
                "write_shx_file: header/offset fixed-point did not converge in 8 iterations "
                "(unexpected: padded header should stabilize within 2-3 passes)"
            )
    js_bytes = header.to_json().encode("utf-8")
    pad_to = _padded_header_end(len(js_bytes))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(SHX_MAGIC)
        fh.write(struct.pack("<HI", SHX_RESERVED_V0_2, len(js_bytes)))
        fh.write(js_bytes)
        pad_len = pad_to - (_FIXED_PREFIX_BYTES + len(js_bytes))
        if pad_len:
            fh.write(b"\x00" * pad_len)
        for s in chunk_streams:
            fh.write(s)
    tmp.replace(path)


def write_shx_file_streaming(
    path: str | Path, header: ShxHeader, chunks_iter, *, per_stream_chunk_count
) -> None:
    """Streaming variant: drains chunks_iter once, writing each compressed
    chunk to a temp staging file as it arrives. Memory bounded by one
    chunk + the chunk descriptors list.

    chunks_iter yields (stream_idx, compressed_bytes, decompressed_size)
    tuples where stream_idx is 0 (data), 1 (indices), or 2 (indptr).

    per_stream_chunk_count is the expected count per stream; mismatched
    iterator length raises ValueError.

    NOTE: This function mutates `header.data.chunks`, `header.indices.chunks`,
    and `header.indptr.chunks` in place — they are cleared at start and
    re-populated as chunks arrive. Callers should not rely on chunk
    descriptors set before calling this function.
    """
    import shutil

    path = Path(path)
    staging = path.with_suffix(path.suffix + ".staging")
    for stream in (header.data, header.indices, header.indptr):
        stream.chunks = []
    with staging.open("wb") as sf:
        for stream_idx, compressed_bytes, decompressed_size in chunks_iter:
            stream = (header.data, header.indices, header.indptr)[stream_idx]
            stream.chunks.append(
                ChunkDescriptor(
                    offset=0,
                    compressed_size=len(compressed_bytes),
                    decompressed_size=decompressed_size,
                )
            )
            sf.write(compressed_bytes)
    for stream, expected in zip(
        (header.data, header.indices, header.indptr),
        per_stream_chunk_count,
        strict=True,
    ):
        if len(stream.chunks) != expected:
            staging.unlink(missing_ok=True)
            raise ValueError(
                f"stream chunk count mismatch: got {len(stream.chunks)}, expected {expected}"
            )
        stream.n_chunks = expected
    # Fixed-point: serialize, set offsets, re-check header length until stable.
    prev_data_start = -1
    converged = False
    for _ in range(_MAX_HEADER_FIXEDPOINT_ITERS):
        js = header.to_json()
        data_start = _padded_header_end(len(js.encode("utf-8")))
        if data_start == prev_data_start:
            converged = True
            break
        prev_data_start = data_start
        cursor = data_start
        for stream in (header.data, header.indices, header.indptr):
            for c in stream.chunks:
                c.offset = cursor
                cursor += c.compressed_size
    if not converged:
        staging.unlink(missing_ok=True)
        raise RuntimeError(
            "write_shx_file_streaming: header/offset fixed-point did not converge in 8 iterations"
        )
    js_bytes = header.to_json().encode("utf-8")
    pad_to = _padded_header_end(len(js_bytes))
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as fh:
            fh.write(SHX_MAGIC)
            fh.write(struct.pack("<HI", SHX_RESERVED_V0_2, len(js_bytes)))
            fh.write(js_bytes)
            pad_len = pad_to - (_FIXED_PREFIX_BYTES + len(js_bytes))
            if pad_len:
                fh.write(b"\x00" * pad_len)
            with staging.open("rb") as sf:
                shutil.copyfileobj(sf, fh, length=1 << 20)
        tmp.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def parse_shx_header(fh: BinaryIO) -> ShxHeader:
    """Read magic + JSON header from an open binary file handle."""
    raw_magic = fh.read(4)
    if raw_magic != SHX_MAGIC:
        if raw_magic[:3] == SHX_MAGIC[:3] and raw_magic[3:4] > SHX_MAGIC[3:4]:
            raise ValueError(
                f"unsupported SHX major version (file has {raw_magic!r}, this "
                f"cellstream supports {SHX_MAGIC!r}). Upgrade cellstream to read this file."
            )
        raise ValueError(f"bad SHX magic: got {raw_magic!r}, expected {SHX_MAGIC!r}")
    reserved, header_size = struct.unpack("<HI", fh.read(6))
    js = fh.read(header_size).decode("utf-8")
    return ShxHeader.from_dict(json.loads(js))
