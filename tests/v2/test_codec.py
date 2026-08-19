"""bitshuffle + zstd-1 CPU codec round-trip tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.uint16, np.int64])
def test_codec_round_trip_random(dtype):
    from cellstream.v2.codec import bitshuffle_zstd_decode, bitshuffle_zstd_encode

    rng = np.random.default_rng(7)
    arr = rng.integers(0, 1000, size=10_000, dtype=dtype)
    enc = bitshuffle_zstd_encode(arr, level=1)
    dec = bitshuffle_zstd_decode(enc, dtype=dtype, n_elements=arr.size)
    np.testing.assert_array_equal(arr, dec)


def test_codec_round_trip_zeros():
    from cellstream.v2.codec import bitshuffle_zstd_decode, bitshuffle_zstd_encode

    arr = np.zeros(1024, dtype=np.uint16)
    enc = bitshuffle_zstd_encode(arr, level=1)
    dec = bitshuffle_zstd_decode(enc, dtype=np.uint16, n_elements=arr.size)
    np.testing.assert_array_equal(arr, dec)
    assert len(enc) < arr.nbytes / 10


def test_codec_round_trip_partial_chunk():
    from cellstream.v2.codec import bitshuffle_zstd_decode, bitshuffle_zstd_encode

    arr = np.arange(13, dtype=np.uint16) * 7
    enc = bitshuffle_zstd_encode(arr, level=1)
    dec = bitshuffle_zstd_decode(enc, dtype=np.uint16, n_elements=arr.size)
    np.testing.assert_array_equal(arr, dec)


def test_codec_round_trip_empty():
    from cellstream.v2.codec import bitshuffle_zstd_decode, bitshuffle_zstd_encode

    arr = np.array([], dtype=np.uint16)
    enc = bitshuffle_zstd_encode(arr, level=1)
    dec = bitshuffle_zstd_decode(enc, dtype=np.uint16, n_elements=0)
    assert dec.size == 0
    assert dec.dtype == np.uint16


def test_compression_ratio_on_umi_shape():
    from cellstream.v2.codec import bitshuffle_zstd_encode

    rng = np.random.default_rng(11)
    arr = rng.integers(0, 16, size=100_000, dtype=np.uint16)
    enc = bitshuffle_zstd_encode(arr, level=1)
    ratio = arr.nbytes / len(enc)
    assert ratio > 3.0, f"expected >3x, got {ratio:.2f}x"
