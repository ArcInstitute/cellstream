"""Reader _decode_stream dispatch on header.shuffle + per-array role."""

from __future__ import annotations

import mmap

import numpy as np
import pytest


def test_decode_stream_dispatches_byte_filter(tmp_path):
    """_decode_stream with shuffle=byte-filter marker + role decodes a
    byte-filter-encoded stream; with shuffle='bitshuffle' it decodes a
    bitshuffle stream — both in the same process (self-describing)."""
    from cellstream.v2 import codec
    from cellstream.v2.format import StreamSpec
    from cellstream.v2.reader_cpu import _decode_stream

    rng = np.random.default_rng(5)
    arr = rng.integers(0, 12000, size=4096, dtype=np.uint16)

    # byte-filter encode -> write bytes -> mmap -> decode via dispatch
    bf = codec.byte_filter_encode(arr, role="indices", level=1)
    p = tmp_path / "bf.bin"
    p.write_bytes(bf)
    with p.open("rb") as fh, mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
        spec = StreamSpec.from_dict(
            {
                "dtype": "uint16",
                "n_chunks": 1,
                "chunks": [
                    {"offset": 0, "compressed_size": len(bf), "decompressed_size": arr.nbytes}
                ],
            }
        )
        dec = _decode_stream(
            mm,
            spec,
            np.uint16,
            shuffle=codec.BYTE_FILTER_SHUFFLE_NAME,
            role="indices",
        )
    np.testing.assert_array_equal(arr, dec)


def test_decode_stream_bitshuffle_still_works(tmp_path):
    from cellstream.v2 import codec
    from cellstream.v2.format import StreamSpec
    from cellstream.v2.reader_cpu import _decode_stream

    rng = np.random.default_rng(6)
    arr = rng.integers(0, 1000, size=2048, dtype=np.uint16)
    blob = codec.bitshuffle_zstd_encode(arr, level=1)
    p = tmp_path / "bs.bin"
    p.write_bytes(blob)
    with p.open("rb") as fh, mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
        spec = StreamSpec.from_dict(
            {
                "dtype": "uint16",
                "n_chunks": 1,
                "chunks": [
                    {"offset": 0, "compressed_size": len(blob), "decompressed_size": arr.nbytes}
                ],
            }
        )
        dec = _decode_stream(mm, spec, np.uint16, shuffle="bitshuffle", role="data")
    np.testing.assert_array_equal(arr, dec)


def test_decode_stream_rawdata_marker_data_role(tmp_path):
    """#142: under the rawdata marker, the DATA stream decodes via zstd_only
    (raw float bytes, no shuffle)."""
    from cellstream.v2 import codec
    from cellstream.v2.format import StreamSpec
    from cellstream.v2.reader_cpu import _decode_stream

    rng = np.random.default_rng(7)
    arr = np.log1p(rng.integers(1, 50, size=4096)).astype(np.float32)
    blob = codec.zstd_only_encode(arr, level=1)
    p = tmp_path / "raw.bin"
    p.write_bytes(blob)
    with p.open("rb") as fh, mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
        spec = StreamSpec.from_dict(
            {
                "dtype": "float32",
                "n_chunks": 1,
                "chunks": [
                    {"offset": 0, "compressed_size": len(blob), "decompressed_size": arr.nbytes}
                ],
            }
        )
        dec = _decode_stream(
            mm,
            spec,
            np.float32,
            shuffle=codec.BYTE_FILTER_RAWDATA_SHUFFLE_NAME,
            role="data",
        )
    np.testing.assert_array_equal(arr, dec)


def test_decode_stream_rawdata_marker_indices_role(tmp_path):
    """Under the rawdata marker, indices/indptr still use the byte-filter
    SHUFFLE+BYTEDELTA chain (only the data stream goes raw)."""
    from cellstream.v2 import codec
    from cellstream.v2.format import StreamSpec
    from cellstream.v2.reader_cpu import _decode_stream

    rng = np.random.default_rng(8)
    arr = np.sort(rng.choice(18533, size=2000, replace=False)).astype(np.uint16)
    blob = codec.byte_filter_encode(arr, role="indices", level=1)
    p = tmp_path / "idx.bin"
    p.write_bytes(blob)
    with p.open("rb") as fh, mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
        spec = StreamSpec.from_dict(
            {
                "dtype": "uint16",
                "n_chunks": 1,
                "chunks": [
                    {"offset": 0, "compressed_size": len(blob), "decompressed_size": arr.nbytes}
                ],
            }
        )
        dec = _decode_stream(
            mm,
            spec,
            np.uint16,
            shuffle=codec.BYTE_FILTER_RAWDATA_SHUFFLE_NAME,
            role="indices",
        )
    np.testing.assert_array_equal(arr, dec)


def test_decode_stream_rejects_unknown_shuffle(tmp_path):
    from cellstream.v2 import codec
    from cellstream.v2.format import StreamSpec
    from cellstream.v2.reader_cpu import _decode_stream

    arr = np.arange(16, dtype=np.uint16)
    blob = codec.bitshuffle_zstd_encode(arr, level=1)
    p = tmp_path / "x.bin"
    p.write_bytes(blob)
    with p.open("rb") as fh, mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
        spec = StreamSpec.from_dict(
            {
                "dtype": "uint16",
                "n_chunks": 1,
                "chunks": [
                    {"offset": 0, "compressed_size": len(blob), "decompressed_size": arr.nbytes}
                ],
            }
        )
        with pytest.raises(ValueError, match="shuffle"):
            _decode_stream(mm, spec, np.uint16, shuffle="totally-unknown", role="data")
