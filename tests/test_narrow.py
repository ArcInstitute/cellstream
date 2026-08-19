"""narrow.lossless_cast: cast that round-trips identical or raises."""

import numpy as np
import pytest

from cellstream import narrow


def test_data_min_uint_dtype_picks_narrowest():
    # The narrowest unsigned dtype that holds the values, by scanned max.
    assert narrow._data_min_uint_dtype(np.array([0, 1, 200], dtype=np.float64)) == np.dtype(
        np.uint8
    )
    assert narrow._data_min_uint_dtype(np.array([0, 65535], dtype=np.float64)) == np.dtype(
        np.uint16
    )
    assert narrow._data_min_uint_dtype(np.array([0, 70000], dtype=np.float64)) == np.dtype(
        np.uint32
    )


def test_data_min_uint_dtype_fast_path_bool_uint8():
    assert narrow._data_min_uint_dtype(np.array([True, False, True])) == np.dtype(np.uint8)
    assert narrow._data_min_uint_dtype(np.array([0, 255, 7], dtype=np.uint8)) == np.dtype(np.uint8)


def test_data_min_uint_dtype_none_when_not_integer_routable():
    # Non-finite, non-integral, negative, or > 2**32-1 -> None (stored as native float).
    assert narrow._data_min_uint_dtype(np.array([0.0, 1.5, 3.0])) is None
    assert narrow._data_min_uint_dtype(np.array([-1.0, 2.0])) is None
    assert narrow._data_min_uint_dtype(np.array([np.nan, 1.0])) is None
    assert narrow._data_min_uint_dtype(np.array([np.inf, 1.0])) is None
    assert narrow._data_min_uint_dtype(np.array([0.0, 5e9])) is None


def test_data_min_uint_dtype_block_boundary():
    # A non-integral value in a late block must still be caught (block < size).
    rng = np.random.default_rng(0)
    data = rng.integers(0, 65535, size=10_000).astype(np.float64)
    data[9_999] = 1.25  # fractional in the last block
    assert narrow._data_min_uint_dtype(data, block=256) is None


def test_decide_on_disk_dtypes_on_disk_and_min():
    # Integer-routable counts -> on-disk uint32 (data + indices); min = narrowest data
    # uint + n_vars-derived index min.
    data = np.array([0, 1, 200], dtype=np.float64)
    od_d, od_i, min_d, min_i = narrow.decide_on_disk_dtypes(
        data, np.arange(data.size, dtype=np.int32), n_vars=100
    )
    assert (od_d, od_i) == (np.dtype(np.uint32), np.dtype(np.uint32))
    assert (min_d, min_i) == (np.dtype(np.uint8), np.dtype(np.uint16))


def test_decide_on_disk_dtypes_float_data_native():
    # Fractional float64 -> not integer-routable -> on-disk float64, min float64.
    data = np.array([0.0, 1.5, 3.0], dtype=np.float64)
    od_d, od_i, min_d, min_i = narrow.decide_on_disk_dtypes(
        data, np.arange(data.size, dtype=np.int32), n_vars=data.size
    )
    assert od_d == np.dtype(np.float64) and min_d == np.dtype(np.float64)
    assert od_i == np.dtype(np.uint32)


def test_decide_on_disk_dtypes_index_min_from_n_vars():
    data = np.array([0, 1, 2], dtype=np.uint16)
    assert narrow.decide_on_disk_dtypes(data, None, n_vars=60)[3] == np.dtype(np.uint16)
    assert narrow.decide_on_disk_dtypes(data, None, n_vars=70000)[3] == np.dtype(np.uint32)


def test_lossless_uint16_integer_data():
    from cellstream.narrow import lossless_cast

    arr = np.array([0, 1, 100, 65535], dtype=np.float32)
    out = lossless_cast(arr, np.uint16, where="X.data")
    assert out.dtype == np.uint16
    assert np.array_equal(out, [0, 1, 100, 65535])


def test_lossy_raises_with_min_max():
    from cellstream.errors import LossyCastError
    from cellstream.narrow import lossless_cast

    arr = np.array([0.5, 1.0, 70000.0], dtype=np.float32)
    with pytest.raises(LossyCastError) as exc:
        lossless_cast(arr, np.uint16, where="X.data")
    msg = str(exc.value)
    assert "X.data" in msg
    assert "70000" in msg or "0.5" in msg


def test_indices_int32_to_uint16_lossless():
    from cellstream.narrow import lossless_cast

    arr = np.array([0, 1, 18532], dtype=np.int32)
    out = lossless_cast(arr, np.uint16, where="X.indices")
    assert out.dtype == np.uint16
    assert np.array_equal(out.astype(np.int32), arr)


def test_indices_overflow_raises():
    from cellstream.errors import LossyCastError
    from cellstream.narrow import lossless_cast

    arr = np.array([0, 1, 70000], dtype=np.int32)
    with pytest.raises(LossyCastError):
        lossless_cast(arr, np.uint16, where="X.indices")


def test_nan_raises_without_warning():
    """NaN inputs raise LossyCastError without leaking a RuntimeWarning."""
    import warnings

    import numpy as np
    import pytest

    from cellstream.errors import LossyCastError
    from cellstream.narrow import lossless_cast

    arr = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # promote warnings to errors
        with pytest.raises(LossyCastError):
            lossless_cast(arr, np.uint16, where="X.data")


def test_nan_preserving_identity_is_lossless():
    """float32 -> float32 with NaN values should be considered lossless."""
    from cellstream.narrow import lossless_cast

    arr = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    out = lossless_cast(arr, np.float32, where="X.test")
    assert out.dtype == np.float32
    # The output mirrors the input including NaN positions.
    assert np.isnan(out[1])
    assert out[0] == 1.0 and out[2] == 2.0


# ---------------------------------------------------------------------------
# Comprehensive failure-mode matrix: every value/dtype combination that we
# expect lossless_cast(...) to reject when targeting uint16. There is no
# fallback in cellstream — if the cast fails here, the entire write fails;
# callers must pre-process X.data themselves (cast to int, clip, etc.).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, arr",
    [
        # Float with fractional parts — uint16 truncates to int
        ("float32_fractional", np.array([1.0, 2.5, 3.0], dtype=np.float32)),
        ("float64_fractional", np.array([0.1, 0.2, 0.3], dtype=np.float64)),
        # Positive integer overflow — values above uint16 max wrap
        ("int32_above_uint16_max", np.array([0, 65536], dtype=np.int32)),
        ("int64_above_uint16_max", np.array([0, 100_000], dtype=np.int64)),
        ("uint32_above_uint16_max", np.array([0, 70_000], dtype=np.uint32)),
        # Negative — uint16 can't hold them; wraps to large positive
        ("int8_negative", np.array([-1, 0, 1], dtype=np.int8)),
        ("int16_negative", np.array([-1, 0, 1], dtype=np.int16)),
        ("int32_negative", np.array([-1, 0, 1], dtype=np.int32)),
        ("float32_negative", np.array([-1.0, 0.0, 1.0], dtype=np.float32)),
        # Inf — cast to int wraps to platform-specific value, no round-trip
        ("float32_pos_inf", np.array([1.0, np.inf], dtype=np.float32)),
        ("float64_pos_inf", np.array([1.0, np.inf], dtype=np.float64)),
        ("float32_neg_inf", np.array([-np.inf, 1.0], dtype=np.float32)),
        # NaN — int conversion destroys NaN-ness (becomes 0); round-trip fails
        ("float32_nan", np.array([1.0, np.nan, 2.0], dtype=np.float32)),
        ("float64_nan", np.array([np.nan], dtype=np.float64)),
        # Mixed: invalid in the middle of an otherwise-valid array
        ("mixed_one_overflow", np.array([0, 1, 65536, 2, 3], dtype=np.int32)),
        ("mixed_one_negative", np.array([0, 1, -5, 2, 3], dtype=np.int32)),
        ("mixed_one_fractional", np.array([0.0, 1.0, 2.5, 3.0], dtype=np.float32)),
    ],
)
def test_lossy_cast_to_uint16_raises(label, arr):
    """Every entry here should fail lossless_cast(arr, uint16)."""
    import warnings

    from cellstream.errors import LossyCastError
    from cellstream.narrow import lossless_cast

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no RuntimeWarning leaks from internal astype
        with pytest.raises(LossyCastError) as exc:
            lossless_cast(arr, np.uint16, where=f"X.data[{label}]")
    msg = str(exc.value)
    assert label in msg  # diagnostic label propagated
    assert "uint16" in msg
    assert "source dtype" in msg


@pytest.mark.parametrize(
    "label, arr",
    [
        # Boundary: exact uint16 max value
        ("int_at_uint16_max", np.array([0, 65535], dtype=np.int32)),
        ("float_at_uint16_max", np.array([0.0, 65535.0], dtype=np.float32)),
        # Widening from a narrower unsigned dtype
        ("uint8_to_uint16", np.array([0, 128, 255], dtype=np.uint8)),
        # Integer values stored in float that round-trip via uint16
        ("float32_integer_values", np.array([0.0, 1.0, 32768.0, 65535.0], dtype=np.float32)),
        # Same-dtype no-op
        ("uint16_to_uint16", np.array([0, 65535], dtype=np.uint16)),
        # Empty array (degenerate but valid)
        ("empty", np.array([], dtype=np.int32)),
    ],
)
def test_lossless_cast_to_uint16_succeeds(label, arr):
    """Every entry here should round-trip cleanly to uint16."""
    from cellstream.narrow import lossless_cast

    out = lossless_cast(arr, np.uint16, where=f"X.data[{label}]")
    assert out.dtype == np.uint16
    # Round-trip equality (NaN-free here)
    assert np.array_equal(out.astype(arr.dtype), arr)


def test_one_off_value_above_max_raises():
    """65536 (uint16_max + 1) is the canonical boundary failure."""
    from cellstream.errors import LossyCastError
    from cellstream.narrow import lossless_cast

    arr = np.array([65536], dtype=np.int64)
    with pytest.raises(LossyCastError) as exc:
        lossless_cast(arr, np.uint16, where="X.data")
    assert "65536" in str(exc.value)


def test_one_off_value_at_max_succeeds():
    """65535 (uint16_max) is the canonical boundary success."""
    from cellstream.narrow import lossless_cast

    arr = np.array([65535], dtype=np.int64)
    out = lossless_cast(arr, np.uint16, where="X.data")
    assert out.dtype == np.uint16
    assert out[0] == 65535


def test_no_fallback_codec_for_lossy_data():
    """Documentation-as-test: cellstream ships only the uint16 codec preset.

    If a user has X.data that doesn't fit uint16, write_sharded raises
    LossyCastError before any disk write — there is no automatic fallback
    to a wider dtype. The caller must pre-process the data.
    """
    from cellstream import codecs

    presets = codecs.list_codecs()
    # The only preset in v0.2 stores uint16 data + uint16 indices.
    assert "blosc-zstd-1-u16u16" in presets
    # No 'u32u32' or 'i32u32' wider preset is shipped (yet).
    assert not any("u32" in p for p in presets), (
        f"unexpected wider codec preset(s): {presets}; "
        f"if a wider codec is added, this test (and its documentation) "
        f"should be updated"
    )


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1.5, 2.0, 3.0], dtype=np.float32),  # fractional -> None
        np.array([3.0, 100.0, 7.0], dtype=np.float32),  # integer-valued float -> uint8
        np.array([3, 250, 7], dtype=np.uint16),  # small counts -> uint8
        np.array([3, 70000, 7], dtype=np.int64),  # > uint16 -> uint16
        np.array([3, 5_000_000_000], dtype=np.int64),  # > uint32 -> None
        np.array([-1, 5, 3], dtype=np.int32),  # negative -> None
        np.array([np.inf, 1.0], dtype=np.float32),  # non-finite -> None
        np.array([], dtype=np.float32),  # empty -> uint8
        np.array([0, 1], dtype=bool),  # bool -> uint8
    ],
)
def test_streaming_matches_inmem(arr):
    """Streaming scanner returns the same dtype as the in-memory scanner, chunked at
    every size so chunk boundaries are exercised."""
    expected = narrow._data_min_uint_dtype(arr)
    for chunk in (1, 2, 3, 1024):
        blocks = [arr[i : i + chunk] for i in range(0, arr.size, chunk)] or [arr[:0]]
        got = narrow.data_min_uint_dtype_streaming(blocks)
        assert got == expected, f"chunk={chunk}: {got} != {expected}"


def test_streaming_early_exit_on_first_fractional():
    """A fractional value in the first block short-circuits to None without consuming
    later blocks."""
    consumed = []

    def gen():
        consumed.append(0)
        yield np.array([1.5], dtype=np.float32)  # fractional -> None immediately
        consumed.append(1)
        yield np.array([2.0], dtype=np.float32)

    assert narrow.data_min_uint_dtype_streaming(gen()) is None
    assert consumed == [0]  # second block never pulled
