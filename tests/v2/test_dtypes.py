"""_index_dtypes_for: single source of truth for index/indptr dtype dispatch."""

from __future__ import annotations

import numpy as np
import pytest


def test_cast_back_false_returns_uint16_int64():
    from cellstream.v2.dtypes import _index_dtypes_for

    indices, indptr = _index_dtypes_for(total_nnz=10_000, cast_back=False, target="cupy")
    assert indices == np.uint16
    assert indptr == np.int64


def test_cupy_cast_back_true_small_nnz_uses_int32_both():
    """cupy + cast_back=True + total_nnz < 2**31: both indices and indptr int32.
    Matches the spec's memory math: float32 data + int32 indices + int32 indptr."""
    from cellstream.v2.dtypes import _index_dtypes_for

    indices, indptr = _index_dtypes_for(total_nnz=10_000, cast_back=True, target="cupy")
    assert indices == np.int32
    assert indptr == np.int32


def test_cupy_cast_back_true_large_nnz_uses_int64():
    from cellstream.v2.dtypes import _index_dtypes_for

    indices, indptr = _index_dtypes_for(total_nnz=2**31 + 100, cast_back=True, target="cupy")
    assert indices == np.int64
    assert indptr == np.int64


def test_torch_cast_back_true_small_nnz_matches_int32():
    from cellstream.v2.dtypes import _index_dtypes_for

    indices, indptr = _index_dtypes_for(total_nnz=10_000, cast_back=True, target="torch")
    assert indices == np.int32
    assert indptr == np.int32


def test_torch_cast_back_true_large_nnz_matches_int64():
    from cellstream.v2.dtypes import _index_dtypes_for

    indices, indptr = _index_dtypes_for(total_nnz=2**31 + 100, cast_back=True, target="torch")
    assert indices == np.int64
    assert indptr == np.int64


def test_torch_cast_back_false_raises():
    from cellstream.v2.dtypes import _index_dtypes_for

    with pytest.raises(ValueError, match="uint16"):
        _index_dtypes_for(total_nnz=10_000, cast_back=False, target="torch")


def test_unknown_target_raises():
    from cellstream.v2.dtypes import _index_dtypes_for

    with pytest.raises(ValueError, match="target"):
        _index_dtypes_for(total_nnz=10_000, cast_back=True, target="bogus")
