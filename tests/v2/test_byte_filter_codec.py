"""Per-array byte-filter codec (SHUFFLE / SHUFFLE+BYTEDELTA) — unit round-trips."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("role", ["data", "indices", "indptr"])
def test_byte_filter_round_trip_uint16(role):
    from cellstream.v2.codec import byte_filter_decode, byte_filter_encode

    rng = np.random.default_rng(7)
    # Spread across the high byte too (values > 255) so the uint16 byte-combine
    # path (lo | hi<<8) is exercised, not just the low byte.
    arr = rng.integers(0, 13000, size=10_000, dtype=np.uint16)
    enc = byte_filter_encode(arr, role=role, level=1)
    dec = byte_filter_decode(enc, dtype=np.uint16, n_elements=arr.size, role=role)
    np.testing.assert_array_equal(arr, dec)
    assert dec.dtype == np.uint16


def test_byte_filter_round_trip_sorted_indices():
    """indices are sorted-within-row in real CSR — exercise SHUFFLE+BYTEDELTA on
    a monotonic-ish stream (where the win comes from)."""
    from cellstream.v2.codec import byte_filter_decode, byte_filter_encode

    # 50 rows of sorted column indices concatenated, like a CSR indices array.
    rng = np.random.default_rng(11)
    parts = [np.sort(rng.choice(18533, size=200, replace=False)) for _ in range(50)]
    arr = np.concatenate(parts).astype(np.uint16)
    enc = byte_filter_encode(arr, role="indices", level=1)
    dec = byte_filter_decode(enc, dtype=np.uint16, n_elements=arr.size, role="indices")
    np.testing.assert_array_equal(arr, dec)


def test_byte_filter_round_trip_int64_indptr_fallback():
    """indptr is int64 (itemsize 8) — exercises the transpose fallback in
    _byte_unshuffle (not the uint16 fast path)."""
    from cellstream.v2.codec import byte_filter_decode, byte_filter_encode

    arr = (np.arange(257, dtype=np.int64) * 7919).astype(np.int64)  # monotonic, > 2**32
    enc = byte_filter_encode(arr, role="indptr", level=1)
    dec = byte_filter_decode(enc, dtype=np.int64, n_elements=arr.size, role="indptr")
    np.testing.assert_array_equal(arr, dec)
    assert dec.dtype == np.int64


@pytest.mark.parametrize("role", ["data", "indices", "indptr"])
def test_byte_filter_round_trip_empty(role):
    from cellstream.v2.codec import byte_filter_decode, byte_filter_encode

    arr = np.array([], dtype=np.uint16)
    enc = byte_filter_encode(arr, role=role, level=1)
    dec = byte_filter_decode(enc, dtype=np.uint16, n_elements=0, role=role)
    assert dec.size == 0
    assert dec.dtype == np.uint16


def test_byte_filter_round_trip_partial_chunk():
    """Odd length (13) — last-partial-chunk symmetry."""
    from cellstream.v2.codec import byte_filter_decode, byte_filter_encode

    arr = (np.arange(13, dtype=np.uint16) * 257) % 60000
    enc = byte_filter_encode(arr, role="data", level=1)
    dec = byte_filter_decode(enc, dtype=np.uint16, n_elements=arr.size, role="data")
    np.testing.assert_array_equal(arr, dec)


def test_rawdata_marker_constant():
    from cellstream.v2.codec import BYTE_FILTER_RAWDATA_SHUFFLE_NAME

    assert BYTE_FILTER_RAWDATA_SHUFFLE_NAME == "byteshuffle-bytedelta-rawdata"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zstd_only_round_trip_float(dtype):
    from cellstream.v2.codec import zstd_only_decode, zstd_only_encode

    rng = np.random.default_rng(5)
    arr = rng.standard_normal(10_000).astype(dtype)
    enc = zstd_only_encode(arr, level=1)
    dec = zstd_only_decode(enc, dtype=dtype, n_elements=arr.size)
    np.testing.assert_array_equal(arr, dec)
    assert dec.dtype == dtype


def test_zstd_only_round_trip_empty():
    from cellstream.v2.codec import zstd_only_decode, zstd_only_encode

    arr = np.array([], dtype=np.float32)
    enc = zstd_only_encode(arr, level=1)
    dec = zstd_only_decode(enc, dtype=np.float32, n_elements=0)
    assert dec.size == 0
    assert dec.dtype == np.float32


def test_zstd_only_beats_byteshuffle_on_low_cardinality_float():
    """The #142 finding: byteshuffle scatters each float's bytes into separate
    planes and destroys the whole-value repeats zstd matches. On low-cardinality
    lognorm-shaped float data, raw zstd is dramatically smaller than the shuffled
    data-role byte-filter."""
    from cellstream.v2.codec import byte_filter_encode, zstd_only_encode

    rng = np.random.default_rng(13)
    # ~512 distinct lognorm-ish values, heavily repeated (low cardinality).
    vals = np.log1p(rng.integers(1, 50, size=512)).astype(np.float32)
    arr = vals[rng.integers(0, vals.size, size=200_000)].astype(np.float32)
    raw = len(zstd_only_encode(arr, level=1))
    shuf = len(byte_filter_encode(arr, role="data", level=1))
    assert raw < shuf, f"raw zstd {raw} should beat byteshuffle {shuf} on low-card float"


def test_zstd_only_decode_rejects_size_mismatch():
    """Bounded/fail-loud on a corrupt header n_elements (DoS guard, like
    byte_filter_decode)."""
    from cellstream.v2.codec import zstd_only_decode, zstd_only_encode

    arr = np.arange(100, dtype=np.float32)
    enc = zstd_only_encode(arr, level=1)
    with pytest.raises(ValueError):
        zstd_only_decode(enc, dtype=np.float32, n_elements=999)


def test_byte_filter_data_compresses_counts_well():
    """data role (SHUFFLE) compresses UMI-shaped counts well — the high byte is
    ~always zero. Anchor on an absolute compression-ratio floor (robust to
    zstd/bitshuffle library-version noise); also check byte-filter stays at
    least competitive with bitshuffle, with slack for that same noise."""
    from cellstream.v2.codec import bitshuffle_zstd_encode, byte_filter_encode

    rng = np.random.default_rng(3)
    arr = rng.integers(0, 16, size=200_000, dtype=np.uint16)  # 99%+ single-byte
    bf = len(byte_filter_encode(arr, role="data", level=1))
    bs = len(bitshuffle_zstd_encode(arr, level=1))
    ratio = arr.nbytes / bf
    assert ratio > 3.0, f"byte-filter ratio {ratio:.2f}x too low (expected >3x on counts)"
    assert bf <= bs * 1.05, f"byte-filter {bf} should be ~<= bitshuffle {bs} on counts"
