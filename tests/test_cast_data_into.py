"""cast_data_into must apply cast_data's exact three-tier policy, writing into a caller
buffer instead of allocating (#244)."""

import numpy as np
import pytest

from cellstream.errors import LossyCastError
from cellstream.v2.materialize import cast_data, cast_data_into

CASES = [
    # (src_dtype, dst_dtype, values, allow_lossy, safe)
    ("uint32", "float32", [0, 1, 65535, 70000], False, False),
    ("uint32", "uint16", [0, 1, 65535], False, False),
    ("uint32", "uint16", [0, 1, 65535], False, True),
    ("uint32", "uint32", [0, 7, 4294967295], False, False),
    ("int64", "int32", [-5, 0, 5], False, False),
    ("float64", "float32", [0.5, 1.25], True, False),
    ("uint32", "uint16", [0, 70000], True, False),
]


@pytest.mark.parametrize("src,dst,vals,allow_lossy,safe", CASES)
def test_matches_cast_data(src, dst, vals, allow_lossy, safe):
    arr = np.array(vals, dtype=src)
    expected = cast_data(arr, np.dtype(dst), allow_lossy=allow_lossy, where="t", safe=safe)
    out = np.empty(arr.size, dtype=dst)
    cast_data_into(arr, out, allow_lossy=allow_lossy, where="t", safe=safe)
    assert out.dtype == np.dtype(dst)
    np.testing.assert_array_equal(out, expected.astype(dst))


def test_raises_like_cast_data():
    arr = np.array([0, 70000], dtype=np.uint32)
    out = np.empty(2, dtype=np.uint16)
    with pytest.raises(LossyCastError):
        cast_data(arr, np.uint16, allow_lossy=False, where="t")
    with pytest.raises(LossyCastError):
        cast_data_into(arr, out, allow_lossy=False, where="t")


def test_safe_integer_narrowing_still_verified():
    """#174: safe=True must NOT bypass verification on an int->int narrowing."""
    arr = np.array([0, 70000], dtype=np.uint32)
    out = np.empty(2, dtype=np.uint16)
    with pytest.raises(LossyCastError):
        cast_data_into(arr, out, allow_lossy=False, where="t", safe=True)


def test_writes_into_the_given_buffer():
    arr = np.array([1, 2, 3], dtype=np.uint32)
    buf = np.zeros(10, dtype=np.float32)
    cast_data_into(arr, buf[4:7], allow_lossy=False, where="t")
    np.testing.assert_array_equal(buf[4:7], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(buf[:4], 0.0)
    np.testing.assert_array_equal(buf[7:], 0.0)


def test_rejects_shape_mismatch():
    """np.copyto broadcasts a size-1 source, so a mis-sized out would be silently replicated
    rather than raising. cast_data returns an array shaped like its input; cast_data_into must
    refuse anything else. (Copilot #247 r4.)"""
    arr = np.array([5], dtype=np.uint32)
    out = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="must equal arr.shape"):
        cast_data_into(arr, out, allow_lossy=False, where="t")
    np.testing.assert_array_equal(out, 0.0)  # untouched
