"""_validate_csr_structure: fail-loud parity with the inline block, one D2H sync.

GPU-gated via tests/v2/gpu/conftest.py (cupy + a visible device). cupy is imported
inside the test bodies so collection never fails on a CPU-only host.
"""

from __future__ import annotations

import numpy as np
import pytest


def _valid():
    # 3 rows, nnz=4, n_vars=10.
    indptr = np.array([0, 2, 3, 4], dtype=np.int64)
    indices = np.array([1, 5, 2, 9], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    return data, indices, indptr


def test_valid_passes():
    cp = pytest.importorskip("cupy")
    from cellstream.v2 import reader_gpu

    d, i, p = (cp.asarray(a) for a in _valid())
    reader_gpu._validate_csr_structure(d, i, p, nnz=4, n_rows=3, n_vars=10)  # no raise


@pytest.mark.parametrize("which", ["indptr0", "indptr_last", "nondecreasing", "index_oob"])
def test_corruption_raises(which):
    cp = pytest.importorskip("cupy")
    from cellstream.v2 import reader_gpu

    d, i, p = (cp.asarray(a) for a in _valid())
    if which == "indptr0":
        p[0] = 1  # indptr[0] != 0  -> [1,2,3,4]
    elif which == "indptr_last":
        p[-1] = 99  # indptr[-1] != nnz -> [0,2,3,99]
    elif which == "nondecreasing":
        p[2] = 1  # decreasing at idx 2 -> [0,2,1,4]
    elif which == "index_oob":
        i[0] = 10  # index == n_vars(10) -> out of bounds
    with pytest.raises(ValueError):
        reader_gpu._validate_csr_structure(d, i, p, nnz=4, n_rows=3, n_vars=10)


def test_negative_index_raises_for_signed_dtype():
    """A signed index dtype with a negative value must fail loud (byte-filter
    indices decode unsigned, but the check stays robust to a signed dtype)."""
    cp = pytest.importorskip("cupy")
    from cellstream.v2 import reader_gpu

    d, i, p = (cp.asarray(a) for a in _valid())
    i[0] = -1  # int32 -> signed branch of the check
    with pytest.raises(ValueError):
        reader_gpu._validate_csr_structure(d, i, p, nnz=4, n_rows=3, n_vars=10)
