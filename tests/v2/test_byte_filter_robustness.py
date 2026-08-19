"""byte_filter_decode fails loud (clear message) on a corrupt / size-mismatched frame."""

from __future__ import annotations

import numpy as np
import pytest
import zstandard as zstd


def test_byte_filter_decode_rejects_size_mismatch():
    """A frame whose decompressed size doesn't match dtype.itemsize * n_elements
    (e.g. a corrupt shard, or a bad n_elements from a corrupt header) raises a
    clear ValueError instead of a cryptic numpy reshape error."""
    from cellstream.v2.codec import byte_filter_decode, byte_filter_encode

    arr = np.arange(100, dtype=np.uint16)
    enc = byte_filter_encode(arr, role="data", level=1)
    # Decode claiming MORE elements than the frame actually holds.
    with pytest.raises(ValueError, match="corrupt"):
        byte_filter_decode(enc, dtype=np.uint16, n_elements=arr.size + 7, role="data")


def test_byte_filter_decode_rejects_truncated_frame():
    """A truncated/garbage compressed blob fails loud."""
    from cellstream.v2.codec import byte_filter_decode

    # L8: zstd.ZstdError is mapped to ValueError so corrupt input stays within the
    # documented {ValueError, OSError, LossyCastError} decode error model.
    with pytest.raises(ValueError):
        byte_filter_decode(b"not-a-zstd-frame", dtype=np.uint16, n_elements=10, role="data")


def test_byte_filter_decode_handles_no_content_size_frame():
    """A frame WITHOUT an embedded content size (declared == -1, e.g. externally
    produced) decodes when valid (bounded by max_output_size) and fails loud
    when it would exceed the expected size."""
    from cellstream.v2.codec import _byte_shuffle, byte_filter_decode

    arr = np.arange(64, dtype=np.uint16)
    shuffled = _byte_shuffle(arr)  # data role = SHUFFLE only
    enc = zstd.ZstdCompressor(level=1, write_content_size=False).compress(shuffled.tobytes())
    assert zstd.frame_content_size(enc) == -1

    dec = byte_filter_decode(enc, dtype=np.uint16, n_elements=arr.size, role="data")
    np.testing.assert_array_equal(arr, dec)

    # Claim fewer elements than the frame holds -> max_output_size bound trips
    # (now surfaced as ValueError, not the raw ZstdError).
    with pytest.raises(ValueError):
        byte_filter_decode(enc, dtype=np.uint16, n_elements=arr.size - 10, role="data")


def test_decode_rejects_decompression_bomb_over_cap():
    # M2: a chunk declaring more than the 64 MiB decompressed cap is rejected up
    # front (before any allocation), mirroring the Rust core. The compressed bytes
    # are irrelevant — the size guard trips first.
    from cellstream.v2.codec import (
        MAX_CHUNK_DECOMPRESSED_BYTES,
        byte_filter_decode,
        zstd_only_decode,
    )

    too_many = MAX_CHUNK_DECOMPRESSED_BYTES // 2 + 1  # uint16 -> expected > cap
    with pytest.raises(ValueError, match="cap"):
        byte_filter_decode(b"\x00", dtype=np.uint16, n_elements=too_many, role="data")
    with pytest.raises(ValueError, match="cap"):
        zstd_only_decode(b"\x00", dtype=np.uint16, n_elements=too_many)


def test_chunk_descriptor_rejects_oversized_and_negative_sizes():
    # M2 (source-level cap): the untrusted chunk header is validated at parse, so a
    # crafted decompressed_size/compressed_size cannot drive n_elements unbounded.
    from cellstream.v2.format import (
        MAX_CHUNK_COMPRESSED_BYTES,
        MAX_CHUNK_DECOMPRESSED_BYTES,
        ChunkDescriptor,
    )

    # Valid descriptor round-trips.
    ChunkDescriptor.from_dict({"offset": 0, "compressed_size": 10, "decompressed_size": 20})
    with pytest.raises(ValueError, match="cap"):
        ChunkDescriptor.from_dict(
            {
                "offset": 0,
                "compressed_size": 10,
                "decompressed_size": MAX_CHUNK_DECOMPRESSED_BYTES + 1,
            }
        )
    with pytest.raises(ValueError, match="cap"):
        ChunkDescriptor.from_dict(
            {
                "offset": 0,
                "compressed_size": MAX_CHUNK_COMPRESSED_BYTES + 1,
                "decompressed_size": 20,
            }
        )
    with pytest.raises(ValueError):
        ChunkDescriptor.from_dict({"offset": 0, "compressed_size": -1, "decompressed_size": 20})
    # Missing key / non-dict must surface as ValueError, not KeyError/TypeError.
    with pytest.raises(ValueError):
        ChunkDescriptor.from_dict({"offset": 0, "compressed_size": 10})  # no decompressed_size
    with pytest.raises(ValueError):
        ChunkDescriptor.from_dict(None)
    with pytest.raises(ValueError):
        ChunkDescriptor.from_dict({"offset": 0, "compressed_size": 1.5, "decompressed_size": 20})
