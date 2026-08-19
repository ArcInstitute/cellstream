"""Byte-exact parity of the vendored FastPFor simdfastpfor256 ENCODE vs pyfastpfor."""

from __future__ import annotations

import numpy as np
import pytest

pyfastpfor = pytest.importorskip("pyfastpfor")
_rust = pytest.importorskip("cellstream.v2._rust")
pytestmark = pytest.mark.skipif(
    not hasattr(_rust, "_pfor_encode_words"), reason="Rust pfordelta encode kernel not built"
)


def _pyfastpfor_pack(arr_u32):
    from cellstream.cell.codec import _pfor_encode_capacity

    c = pyfastpfor.getCodec("simdfastpfor256")
    out = np.empty(_pfor_encode_capacity(arr_u32.size), dtype=np.uint32)
    wl = c.encodeArray(arr_u32, arr_u32.size, out, out.size)
    return np.ascontiguousarray(out[:wl])


@pytest.mark.parametrize("n", [1, 2, 63, 64, 65, 128, 200, 1000])
def test_rust_encode_matches_pyfastpfor(n):
    rng = np.random.default_rng(n)
    # include exception-triggering large values (PFOR stores high bits as exceptions)
    arr = rng.integers(0, 1_000_000, size=n, dtype=np.uint32)
    want = _pyfastpfor_pack(arr)
    got = _rust._pfor_encode_words(arr)
    np.testing.assert_array_equal(got, want)
    assert got.dtype == np.uint32


def test_rust_encode_empty_is_empty():
    got = _rust._pfor_encode_words(np.empty(0, dtype=np.uint32))
    assert got.size == 0  # mirrors codec.py _pack n==0 short-circuit (no codec call)


def test_rust_encode_decode_round_trip():
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 2_000_000, size=513, dtype=np.uint32)
    packed = _rust._pfor_encode_words(arr)
    back = _rust._pfor_decode_words(packed, arr.size)
    np.testing.assert_array_equal(back, arr)
