"""zstd float reader-width tests (gather_rows + read_cell_bulk)."""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import make_cell_adata
from .test_zstd_writer import _float_adata, _write


def test_gather_rows_preserves_float64(tmp_path):
    from cellstream.cell.reader import open_cell

    a = _float_adata(np.float64)
    out = _write(tmp_path, a)
    store = open_cell(out)
    X = store.gather_rows(np.arange(a.n_obs))
    assert X.dtype == np.float64  # regression guard for the hard-float32 cast
    np.testing.assert_array_equal(X.toarray(), a.X.toarray())
    store.close()


def test_gather_rows_float32_and_int_stay_float32(tmp_path):
    from cellstream.cell.reader import open_cell

    a32 = _float_adata(np.float32)
    s32 = open_cell(_write(tmp_path / "f32", a32))
    assert s32.gather_rows(np.arange(a32.n_obs)).dtype == np.float32
    s32.close()

    aint = make_cell_adata()
    sint = open_cell(_write(tmp_path / "int", aint, codec="zstd"))
    assert sint.gather_rows(np.arange(aint.n_obs)).dtype == np.float32  # counts -> float32
    sint.close()


def test_read_cell_bulk_f64_downcast_gate(tmp_path):
    from cellstream.cell.reader import read_cell_bulk
    from cellstream.errors import LossyCastError

    a = _float_adata(np.float64)
    x = a.X.tocsr()
    x.data[0] = np.float64(1) / np.float64(3)
    a.X = x
    out = _write(tmp_path, a)  # stored float64 (default)
    # default read -> float64
    assert read_cell_bulk(out).X.dtype == np.float64
    # request float32 without allow_lossy -> raise (reuses resolve_plan/cast_data)
    with pytest.raises(LossyCastError):
        read_cell_bulk(out, data_dtype="float32")
    # forced
    assert read_cell_bulk(out, data_dtype="float32", allow_lossy=True).X.dtype == np.float32
