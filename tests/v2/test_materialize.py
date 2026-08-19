"""cast_data three-tier policy + the safe= fast-path (the worker cast seam)."""

import numpy as np
import pytest

from cellstream.errors import LossyCastError
from cellstream.v2.materialize import cast_data


def test_cast_data_safe_bypasses_lossless_check():
    # uint32 -> float32 is NOT a numpy-"safe" cast, but safe=True (the caller proved
    # the values fit via the recorded min dtype) -> plain astype, no per-element check.
    arr = np.array([0, 5, 65535], dtype=np.uint32)
    out = cast_data(arr, np.float32, allow_lossy=False, where="t", safe=True)
    assert out.dtype == np.float32
    assert np.array_equal(out, arr.astype(np.float32))


def test_cast_data_without_safe_still_lossless_gated():
    # Out-of-range narrowing with neither safe nor allow_lossy must raise.
    big = np.array([70000], dtype=np.uint32)
    with pytest.raises(LossyCastError):
        cast_data(big, np.uint16, allow_lossy=False, where="t")


def test_cast_data_allow_lossy_unaffected_by_safe_default():
    big = np.array([70000], dtype=np.uint32)
    out = cast_data(big, np.uint16, allow_lossy=True, where="t")
    assert out.dtype == np.uint16  # wrapped, but allow_lossy permitted it
