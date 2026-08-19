"""pfordelta codec round-trip + corruption tests (ported from cellstream)."""

from __future__ import annotations

import numpy as np
import pytest


def _roundtrip(indices, values, index_dtype, value_dtype):
    from cellstream.cell.codec import PforDeltaCodec

    c = PforDeltaCodec(index_dtype, value_dtype)
    enc = c.new_encoder()
    dec = c.new_decoder(70000)
    frame = enc.encode(np.asarray(indices), np.asarray(values))
    di, dv = dec.decode(frame, len(indices))
    return di, dv


@pytest.mark.parametrize("index_dtype", ["uint16", "uint32"])
@pytest.mark.parametrize("value_dtype", ["uint16", "uint32"])
def test_roundtrip_random_sorted(index_dtype, value_dtype):
    rng = np.random.default_rng(3)
    idx = np.sort(rng.choice(60000, size=200, replace=False)).astype(np.uint32)
    vmax = 60000 if value_dtype == "uint16" else 200000
    val = rng.integers(1, vmax, size=idx.size, dtype=np.uint32)
    di, dv = _roundtrip(idx, val, index_dtype, value_dtype)
    np.testing.assert_array_equal(di.astype(np.int64), idx.astype(np.int64))
    np.testing.assert_array_equal(dv.astype(np.int64), val.astype(np.int64))
    assert di.dtype == np.dtype(index_dtype)
    assert dv.dtype == np.dtype(value_dtype)


def test_roundtrip_empty_cell():
    di, dv = _roundtrip(np.empty(0, np.uint32), np.empty(0, np.uint32), "uint16", "uint16")
    assert di.size == 0 and dv.size == 0


def test_roundtrip_single_element():
    di, dv = _roundtrip(np.array([5]), np.array([7]), "uint16", "uint16")
    np.testing.assert_array_equal(di.astype(np.int64), [5])
    np.testing.assert_array_equal(dv.astype(np.int64), [7])


def test_encode_rejects_unsorted_indices():
    from cellstream.cell.codec import PforDeltaCodec

    enc = PforDeltaCodec("uint16", "uint16").new_encoder()
    with pytest.raises(ValueError, match="non-decreasing"):
        enc.encode(np.array([5, 2, 9]), np.array([1, 1, 1]))


def test_decode_short_frame_raises():
    from cellstream.cell.codec import PforDeltaCodec

    dec = PforDeltaCodec("uint16", "uint16").new_decoder(70000)
    with pytest.raises(ValueError, match="shorter than 4-byte length prefix"):
        dec.decode(b"\x00\x00", 3)


def test_decode_idx_len_exceeds_frame_raises():
    import struct

    from cellstream.cell.codec import PforDeltaCodec

    dec = PforDeltaCodec("uint16", "uint16").new_decoder(70000)
    bad = struct.pack("<I", 999) + b"\x00\x00\x00\x00"
    with pytest.raises(ValueError, match="idx blob length exceeds frame"):
        dec.decode(bad, 3)


def test_codec_for_rejects_unknown():
    from cellstream.cell.codec import codec_for

    with pytest.raises(ValueError, match="unsupported codec"):
        codec_for("bogus-codec", "uint16", "uint16")


# --- zstd codec (int + float32 + float64) ---


def _zstd_roundtrip(indices, values, index_dtype, value_dtype):
    from cellstream.cell.codec import ByteZstdCodec

    c = ByteZstdCodec(index_dtype, value_dtype)
    frame = c.new_encoder().encode(np.asarray(indices), np.asarray(values))
    di, dv = c.new_decoder(70000).decode(frame, len(indices))
    return di, dv, frame


@pytest.mark.parametrize("value_dtype", ["uint32", "float32", "float64"])
def test_zstd_roundtrip_sorted(value_dtype):
    rng = np.random.default_rng(11)
    idx = np.sort(rng.choice(50000, size=150, replace=False)).astype(np.uint32)
    if value_dtype == "uint32":
        val = rng.integers(1, 300000, size=idx.size, dtype=np.uint32)
    else:
        val = (rng.standard_normal(idx.size) * 100).astype(value_dtype)
    di, dv, _ = _zstd_roundtrip(idx, val, "uint32", value_dtype)
    np.testing.assert_array_equal(di.astype(np.int64), idx.astype(np.int64))
    assert dv.dtype == np.dtype(value_dtype)
    if value_dtype == "uint32":
        np.testing.assert_array_equal(dv, val)
    else:
        np.testing.assert_array_equal(dv, val.astype(value_dtype))  # bit-exact


def test_zstd_roundtrip_integer_value_dtype_uint16():
    """#223: an integer value_dtype != uint32 must round-trip. uint16 is the only such
    dtype the cell format supports (_DTYPES = {uint16, uint32, float32, float64}). The
    encoder byte-shuffled integer values at value_dtype width (2 bytes) while the decoder
    always reads uint32 width (4 bytes), so the uint16-shuffled value blob was rejected by
    the uint32 decode (size mismatch in _decompress_checked). On-disk integers are now
    always uint32; value_dtype is purely the output cast."""
    idx = np.array([0, 3, 7, 11, 42], dtype=np.uint32)
    val = np.array([1, 2, 65535, 5, 65534], dtype=np.uint32)  # all within uint16 range
    di, dv, _ = _zstd_roundtrip(idx, val, "uint32", "uint16")
    np.testing.assert_array_equal(di.astype(np.int64), idx.astype(np.int64))
    assert dv.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(dv.astype(np.int64), val.astype(np.int64))


def test_zstd_roundtrip_integer_value_dtype_uint16_truncates_out_of_range():
    """#223 (Codex review): for an integer value_dtype narrower than uint32, an on-disk
    value exceeding the output dtype truncates low-bits-first under the decoder's output
    .astype -- exactly like the always-uint32 index output cast. This is the intentional
    low-level primitive behavior; lossless-narrow enforcement is a reader-level concern
    (cf. cast_data / #218 wide_fits / #174), not the codec's. (Before this #223 fix the whole
    path raised a width-mismatch on any non-empty frame; the fix makes it behave like indices.)"""
    idx = np.array([0, 1], dtype=np.uint32)
    val = np.array([70000, 3], dtype=np.uint32)  # 70000 > uint16 max (65535)
    di, dv, _ = _zstd_roundtrip(idx, val, "uint32", "uint16")
    np.testing.assert_array_equal(di.astype(np.int64), idx.astype(np.int64))
    assert dv.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(dv, val.astype(np.uint16))  # low-bits-first, like np astype
    assert int(dv[0]) == 70000 & 0xFFFF  # == 4464


@pytest.mark.parametrize("value_dtype", ["float32", "float64"])
def test_zstd_roundtrip_nonfinite_bit_exact(value_dtype):
    idx = np.array([0, 3, 7, 9], dtype=np.uint32)
    val = np.array([np.nan, np.inf, -np.inf, 1.5], dtype=value_dtype)
    di, dv, _ = _zstd_roundtrip(idx, val, "uint32", value_dtype)
    # bit-exact: NaN stays NaN, Inf/-Inf preserved
    assert np.isnan(dv[0]) and np.isposinf(dv[1]) and np.isneginf(dv[2])
    assert dv[3] == np.array(1.5, dtype=value_dtype)


def test_zstd_roundtrip_empty_and_single():
    di, dv, _ = _zstd_roundtrip(
        np.empty(0, np.uint32), np.empty(0, np.float32), "uint32", "float32"
    )
    assert di.size == 0 and dv.size == 0
    di, dv, _ = _zstd_roundtrip(
        np.array([5], np.uint32), np.array([2.5], np.float64), "uint32", "float64"
    )
    np.testing.assert_array_equal(di.astype(np.int64), [5])
    assert dv[0] == 2.5


def test_zstd_encode_rejects_unsorted_indices():
    from cellstream.cell.codec import ByteZstdCodec

    enc = ByteZstdCodec("uint32", "float32").new_encoder()
    with pytest.raises(ValueError, match="non-decreasing"):
        enc.encode(np.array([5, 2, 9], np.uint32), np.array([1.0, 1.0, 1.0], np.float32))


def test_zstd_decode_short_and_overrun_frames():
    import struct

    from cellstream.cell.codec import ByteZstdCodec

    dec = ByteZstdCodec("uint32", "float32").new_decoder(70000)
    with pytest.raises(ValueError, match="shorter than 4-byte length prefix"):
        dec.decode(b"\x00\x00", 3)
    bad = struct.pack("<I", 999) + b"\x00\x00\x00\x00"
    with pytest.raises(ValueError, match="idx blob length exceeds frame"):
        dec.decode(bad, 3)


def test_codec_for_builds_zstd():
    from cellstream.cell.codec import ByteZstdCodec, codec_for

    assert isinstance(codec_for("zstd", "uint32", "float64"), ByteZstdCodec)


def test_pfordelta_decode_rejects_frame_shorter_than_declared_nnz():
    """#249: a short decode must RAISE, not return the previous row's scratch.

    ``_PforDecoder._unpack`` reuses one ``self._buf`` across every row and never zeroes it, so
    when a frame decodes fewer values than the sidecar's ``nnz`` claims, the tail of the returned
    slice used to be whatever the PREVIOUS cell left behind -- silently, with no error. pyfastpfor's
    ``decodeArray`` returns the real decoded count precisely so this is detectable; it was discarded.
    """
    from cellstream.cell.codec import PforDeltaCodec

    c = PforDeltaCodec("uint32", "uint32")
    enc, dec = c.new_encoder(), c.new_decoder(70000)

    # Decode a LONG row first so the reused scratch holds recognisable data to leak.
    long_idx = np.arange(1000, 1300, dtype=np.uint32)
    di, _ = dec.decode(enc.encode(long_idx, long_idx), long_idx.size)
    np.testing.assert_array_equal(di.astype(np.int64), long_idx.astype(np.int64))

    # A genuinely SHORT row whose declared nnz is the long row's -- i.e. a truncated payload or a
    # tampered sidecar. Must raise rather than return 300 values, 292 of them stale.
    short_idx = np.arange(7000, 7008, dtype=np.uint32)
    frame = enc.encode(short_idx, short_idx)
    with pytest.raises(ValueError, match="decoded 8 values, expected 300"):
        dec.decode(frame, long_idx.size)


def test_pfordelta_decode_short_value_blob_raises():
    """#249: the value blob is checked too -- the index blob alone being intact is not enough."""
    import struct

    from cellstream.cell.codec import PforDeltaCodec

    c = PforDeltaCodec("uint32", "uint32")
    enc, dec = c.new_encoder(), c.new_decoder(70000)

    n = 40
    idx = np.arange(n, dtype=np.uint32)
    frame = enc.encode(idx, idx)
    (idx_len,) = struct.unpack_from("<I", frame, 0)

    # Splice a SHORTER value blob (8 values) behind the intact 40-value index blob. The short
    # blob is taken from the PUBLIC encode() rather than the encoder's private _pack: the
    # value-blob tail of a frame is byte-identical to packing those values directly (verified),
    # so this pins the same bytes without coupling the test to an internal. (Copilot #255.)
    short = np.arange(8, dtype=np.uint32)
    short_frame = enc.encode(short, short)
    (short_idx_len,) = struct.unpack_from("<I", short_frame, 0)
    spliced = frame[: 4 + idx_len] + short_frame[4 + short_idx_len :]
    with pytest.raises(ValueError, match="decoded 8 values, expected 40"):
        dec.decode(spliced, n)


# --- #250: column index bounds ------------------------------------------------------------


def _forged_pfor_frame(indices, values):
    """A pfordelta frame carrying arbitrary absolute column indices.

    The ENCODER does not know n_vars, so it will happily pack a column index no writer would
    ever emit for the archive that decodes it. That is exactly the tampered-frame shape of
    #250, built without any byte surgery.
    """
    from cellstream.cell.codec import PforDeltaCodec

    enc = PforDeltaCodec("uint16", "uint16").new_encoder()
    return enc.encode(np.asarray(indices, dtype=np.uint32), np.asarray(values, dtype=np.uint32))


def test_pfordelta_decode_rejects_column_index_that_would_wrap():
    """#250: cumsum 70000 narrows to 70000-65536=4464 at uint16 -- a LEGAL column of an
    18533-gene archive. Pre-fix the caller got a silently wrong nonzero."""
    from cellstream.cell.codec import PforDeltaCodec

    frame = _forged_pfor_frame([70000], [7])
    dec = PforDeltaCodec("uint16", "uint16").new_decoder(18533)
    with pytest.raises(ValueError, match=r"column index 70000 out of range \[0, 18533\)"):
        dec.decode(frame, 1)


def test_pfordelta_decode_column_bounds_are_exclusive_at_n_vars():
    from cellstream.cell.codec import PforDeltaCodec

    ok = PforDeltaCodec("uint16", "uint16").new_decoder(18533)
    di, _ = ok.decode(_forged_pfor_frame([18532], [7]), 1)
    assert int(di[0]) == 18532
    with pytest.raises(ValueError, match=r"column index 18533 out of range"):
        ok.decode(_forged_pfor_frame([18533], [7]), 1)


def test_pfordelta_decode_reports_the_first_out_of_range_column():
    """Mirrors Rust write_col, which fails on the first bad index it reaches. The O(1) hot
    check only knows the max, so the message must come from a cold-path search."""
    from cellstream.cell.codec import PforDeltaCodec

    dec = PforDeltaCodec("uint16", "uint16").new_decoder(1000)
    with pytest.raises(ValueError, match=r"column index 1500 out of range \[0, 1000\)"):
        dec.decode(_forged_pfor_frame([10, 1500, 2000], [1, 1, 1]), 3)


def test_pfordelta_decode_empty_cell_skips_the_bounds_check():
    from cellstream.cell.codec import PforDeltaCodec

    dec = PforDeltaCodec("uint16", "uint16").new_decoder(18533)
    di, dv = dec.decode(_forged_pfor_frame([], []), 0)
    assert di.size == 0 and dv.size == 0


def test_zstd_decoder_accepts_and_ignores_n_vars():
    """zstd's on-disk index dtype is always uint32, so its .astype is a no-op and nothing
    wraps; the aggregate gather scan is what bounds zstd. The argument exists to keep
    new_decoder's signature uniform."""
    from cellstream.cell.codec import ByteZstdCodec

    c = ByteZstdCodec("uint32", "uint32")
    frame = c.new_encoder().encode(np.array([70000], np.uint32), np.array([7], np.uint32))
    di, _ = c.new_decoder(10).decode(frame, 1)
    assert int(di[0]) == 70000


def test_check_col_indices_helper_bounds_and_sign():
    """The scan _csr_direct_assign has always done, now shared with gather_rows (#250)."""
    from cellstream.cell.reader import _check_col_indices

    _check_col_indices(np.array([], dtype=np.int32), 10)  # empty: no-op
    _check_col_indices(np.array([0, 9], dtype=np.int32), 10)  # in range
    with pytest.raises(ValueError, match=r"column index out of range \[0, 10\) \(max=10\)"):
        _check_col_indices(np.array([0, 10], dtype=np.int32), 10)
    with pytest.raises(ValueError, match=r"column index out of range \[0, 10\) \(min=-1\)"):
        _check_col_indices(np.array([-1, 0], dtype=np.int32), 10)
    # unsigned cannot be negative -> the min() scan is skipped, only max is bounded
    _check_col_indices(np.array([0, 9], dtype=np.uint16), 10)
