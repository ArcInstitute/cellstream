import numpy as np
import pytest
import scipy.sparse as sp

from cellstream import narrow
from cellstream.v2 import _rust_dispatch

pytestmark = pytest.mark.skipif(
    not _rust_dispatch.rust_available(), reason="Rust core not built/enabled"
)


def _csr(data, indices, indptr, n_vars):
    return sp.csr_matrix(
        (np.asarray(data), np.asarray(indices), np.asarray(indptr)),
        shape=(len(indptr) - 1, n_vars),
    )


@pytest.mark.parametrize(
    "data_dtype,idx_dtype",
    [("float32", "int32"), ("float64", "int64"), ("uint16", "int32"), ("uint32", "int64")],
)
def test_decide_dtypes_matches_numpy(data_dtype, idx_dtype):
    # Integer-routable counts (max 3 -> min uint8), n_vars small (min index uint16).
    X = _csr(
        np.array([1, 2, 3], dtype=data_dtype),
        np.array([0, 1, 2], dtype=idx_dtype),
        np.array([0, 2, 3], dtype=np.int64),
        n_vars=8,
    )
    rust = _rust_dispatch.decide_dtypes(X, n_vars=8, n_threads=4)
    npy = narrow.decide_on_disk_dtypes(X.data, X.indices, n_vars=8)
    assert rust == npy
    assert rust == (np.dtype("uint32"), np.dtype("uint32"), np.dtype("uint8"), np.dtype("uint16"))


def test_decide_dtypes_float_not_narrowable():
    X = _csr(
        np.array([1.5, 2.0], dtype="float32"),
        np.array([0, 1], dtype="int32"),
        np.array([0, 2], dtype=np.int64),
        n_vars=4,
    )
    rust = _rust_dispatch.decide_dtypes(X, n_vars=4, n_threads=2)
    npy = narrow.decide_on_disk_dtypes(X.data, X.indices, n_vars=4)
    # fractional float32 -> on-disk float32 (min float32); index on-disk uint32, min uint16.
    expected = (np.dtype("float32"), np.dtype("uint32"), np.dtype("float32"), np.dtype("uint16"))
    assert rust == npy == expected


def test_decide_dtypes_dense_matches_numpy():
    from cellstream.v2.writer import _decide_dense_on_disk_dtypes

    D = np.array([[0, 3, 0], [7, 0, 2]], dtype="float32")  # max 7 -> min uint8
    rust = _rust_dispatch.decide_dtypes_dense(D, n_vars=3, n_threads=2)
    npy = _decide_dense_on_disk_dtypes(D, 3)
    assert rust == npy
    assert rust == (np.dtype("uint32"), np.dtype("uint32"), np.dtype("uint8"), np.dtype("uint16"))
