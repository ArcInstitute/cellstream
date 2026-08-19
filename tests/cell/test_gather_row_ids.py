"""#277 defect 1: gather_rows must not reinterpret a bool mask as integer row ids."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.cell.reader import CellStore
from cellstream.cell.writer import write_cell_archive


@pytest.fixture
def store(tmp_path):
    rows = np.array(
        [[1, 2, 3, 4], [0, 0, 0, 0], [5, 6, 7, 8], [0, 0, 0, 0], [0, 0, 0, 0], [9, 9, 9, 9]],
        dtype=np.uint16,
    )
    a = ad.AnnData(
        X=sp.csr_matrix(rows, dtype=np.uint16),
        obs=pd.DataFrame({"lbl": list("abcdef")}, index=[f"c{i}" for i in range(6)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(4)]),
    )
    p = tmp_path / "m.shad"
    write_cell_archive(a, p)
    s = CellStore(p)
    yield s, rows
    s.close()


def test_bool_mask_selects_masked_rows(store):
    s, rows = store
    mask = np.array([False, False, True, False, False, True])
    got = s.gather_rows(mask)
    assert got.shape == (2, 4)
    np.testing.assert_array_equal(got.toarray(), rows[[2, 5]].astype(np.float32))


def test_bool_mask_wrong_length_raises(store):
    s, _ = store
    with pytest.raises(ValueError, match="length n_obs=6"):
        s.gather_rows(np.array([True, False, True]))


def test_2d_bool_mask_raises_the_mask_error(store):
    """A 2-D bool array must get the MASK message, not the generic ndim one -- the bool
    check runs before the dimensionality check, as in read.py::_normalize_key."""
    s, _ = store
    with pytest.raises(ValueError, match="boolean mask must be 1-D"):
        s.gather_rows(np.zeros((2, 3), dtype=bool))


def test_object_dtype_row_ids_raise(store):
    """Deliberate compatibility change: an integer-valued object array used to be accepted
    (min/max/astype all work on it). It is rejected now, matching _normalize_key."""
    s, _ = store
    with pytest.raises(TypeError, match="integer dtype"):
        s.gather_rows(np.array([0, 2], dtype=object))


def test_empty_bool_mask_raises(store):
    """Deliberate compatibility change: a length-0 BOOL array used to fall through the
    size==0 exemption and select nothing. A zero-length mask cannot address a 6-row store,
    so it is now a malformed mask -- pass an empty INTEGER array to select nothing."""
    s, _ = store
    with pytest.raises(ValueError, match="boolean mask must be 1-D"):
        s.gather_rows(np.array([], dtype=bool))


def test_all_false_mask_selects_nothing(store):
    s, _ = store
    got = s.gather_rows(np.zeros(6, dtype=bool))
    assert got.shape == (0, 4)


def test_float_row_ids_raise(store):
    s, _ = store
    with pytest.raises(TypeError, match="integer dtype"):
        s.gather_rows(np.array([1.9, 2.9]))


def test_scalar_bool_raises(store):
    """gather_rows(True) silently returned row 1 (bool subclasses int). It is now a
    malformed mask -- shape () -- so it gets _normalize_key's mask ValueError."""
    s, _ = store
    with pytest.raises(ValueError, match="boolean mask must be 1-D"):
        s.gather_rows(True)


@pytest.mark.parametrize(
    "empty", [[], np.array([], dtype=np.int64), np.array([], dtype=np.float64)]
)
def test_empty_selection_still_works(store, empty):
    s, _ = store
    assert s.gather_rows(empty).shape == (0, 4)


def test_scalar_int_still_works(store):
    s, rows = store
    got = s.gather_rows(2)
    assert got.shape == (1, 4)
    np.testing.assert_array_equal(got.toarray(), rows[[2]].astype(np.float32))


def test_out_of_range_still_raises(store):
    s, _ = store
    with pytest.raises(IndexError):
        s.gather_rows(np.array([6]))
    with pytest.raises(IndexError):
        s.gather_rows(np.array([-1]))


def test_2d_still_raises(store):
    s, _ = store
    with pytest.raises(ValueError, match="1-dimensional"):
        s.gather_rows(np.array([[0, 1], [2, 3]]))


def test_anndata_view_mask(store):
    s, rows = store
    view = s.to_anndata(backed=True)
    got = view.X[np.array([False, False, True, False, False, True])]
    assert got.shape == (2, 4)
    np.testing.assert_array_equal(got.toarray(), rows[[2, 5]].astype(np.float32))


def test_gather_rows_adata_mask_aligns_obs(store):
    s, rows = store
    a = s.gather_rows_adata(np.array([False, False, True, False, False, True]))
    assert a.shape == (2, 4)
    assert list(a.obs["lbl"]) == ["c", "f"]
    np.testing.assert_array_equal(a.X.toarray(), rows[[2, 5]].astype(np.float32))
