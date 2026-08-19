"""#277 defect 2: re-encoding a cell archive must not round counts through float32."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.cell.reader import CellStore, read_cell_bulk
from cellstream.cell.source import CellArchiveSource
from cellstream.cell.writer import write_cell_archive

# first uint32 not exactly representable in float32 (2**24 + 1)
BIG = 16777217


def _mk(values, dtype):
    n, m = values.shape
    return ad.AnnData(
        X=sp.csr_matrix(values, dtype=dtype),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(m)]),
    )


def test_uint32_survives_a_reencode_round_trip(tmp_path):
    a = _mk(np.array([[BIG, 1], [2, 3]], dtype=np.uint32), np.uint32)
    r1, r2 = tmp_path / "r1.shad", tmp_path / "r2.shad"
    write_cell_archive(a, r1)
    write_cell_archive(r1, r2)  # the documented codec-migration path

    for p in (r1, r2):
        s = CellStore(p)
        try:
            assert s.manifest["value_dtype_on_disk"] == "uint32"
        finally:
            s.close()  # else a failing assert leaks the mmap export into later tests
        got = read_cell_bulk(p, cast_back=True, data_dtype="uint32")
        assert int(got.X.toarray()[0, 0]) == BIG, f"{p.name} rounded {BIG}"


def test_source_reports_the_on_disk_dtype(tmp_path):
    a = _mk(np.array([[BIG, 1], [2, 3]], dtype=np.uint32), np.uint32)
    p = tmp_path / "s.shad"
    write_cell_archive(a, p)
    src = CellArchiveSource(CellStore(p))
    try:
        assert src.data_dtype() == np.dtype("uint32")
        blk = src.read_block(0, 2)
        assert blk.dtype == np.dtype("uint32")
        assert int(blk.toarray()[0, 0]) == BIG
    finally:
        src.close()


def test_float32_archive_round_trips_unchanged(tmp_path):
    vals = np.array([[1.5, 0.25], [0.0, 3.75]], dtype=np.float32)
    a = _mk(vals, np.float32)
    r1, r2 = tmp_path / "f1.shad", tmp_path / "f2.shad"
    write_cell_archive(a, r1)
    write_cell_archive(r1, r2)
    s = CellStore(r2)
    try:
        assert s.manifest["value_dtype_on_disk"] == "float32"
    finally:
        s.close()
    got = read_cell_bulk(r2, cast_back=True)
    np.testing.assert_array_equal(got.X.toarray(), vals)


def test_gather_rows_default_output_is_still_float32(tmp_path):
    """The dataloader contract must not change: no out_dtype -> float32 as before."""
    a = _mk(np.array([[1, 2], [3, 4]], dtype=np.uint32), np.uint32)
    p = tmp_path / "d.shad"
    write_cell_archive(a, p)
    s = CellStore(p)
    try:
        assert s.gather_rows(np.array([0, 1])).dtype == np.float32
        assert s.gather_rows(np.array([0, 1]), out_dtype=np.uint32).dtype == np.uint32
    finally:
        s.close()


@pytest.mark.parametrize("out_dtype", [">u4", ">f4", ">f8", ">u2"])
def test_non_native_out_dtype_is_rejected_by_scipy(tmp_path, out_dtype):
    """A non-native out_dtype raises, on the SAME scipy check the bulk path hits (#244).

    Two separate things happen here, and only the first is cellstream's:

    1. `dtype.name` drops byte order, so '>u4' reports "uint32" and would pass the Rust
       decode gate -- then Rust matches on str(dtype), sees '>u4', and raises "unsupported
       output dtypes" with no fallback (the #218 bug class). The gate requires isnative, so
       the request routes to the pure-Python branch instead of dying in Rust.
    2. That branch then builds its CSR via _csr_direct_assign, whose
       `csr_matrix(shape, dtype=)` rejects non-native dtypes outright -- so the call raises a
       clear scipy error rather than a confusing Rust one.

    Before #244 the pure-Python branch used the `(data, indices, indptr)` constructor, which
    skips that check, so gather_rows quietly returned a '>u4' CSR while read_cell_bulk
    rejected it. #244 removes that asymmetry: both paths now reject, consistently. The
    capability was a scipy loophole, not a supported contract -- scipy's own getdtype refuses
    these dtypes.
    """
    vals = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    a = _mk(vals, np.uint16)
    p = tmp_path / f"be_{out_dtype.strip('>')}.shad"
    write_cell_archive(a, p)
    s = CellStore(p)
    try:
        with pytest.raises(ValueError, match="does not support dtype"):
            s.gather_rows(np.array([0, 1]), out_dtype=out_dtype)
    finally:
        s.close()


def test_bulk_read_non_native_data_dtype_is_rejected_by_scipy(tmp_path):
    """The bulk route never reaches the Rust gate: scipy rejects the dtype first.

    resolve_plan preserves an explicit data_dtype verbatim, so read_cell_bulk('>u4') gets that
    far -- but _csr_direct_assign builds its CSR with `sp.csr_matrix(shape, dtype=...)`, and
    scipy rejects a non-native dtype outright. Fails loudly on both engines, before and after
    the isnative gate. Since #244 gather_rows behaves identically (see the test above), so this
    pins the two halves of ONE consistent contract rather than an asymmetry.
    """
    vals = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    a = _mk(vals, np.uint16)
    p = tmp_path / "bulk_be.shad"
    write_cell_archive(a, p)
    with pytest.raises(ValueError, match="does not support dtype"):
        read_cell_bulk(p, data_dtype=">u4")


@pytest.mark.parametrize("out_dtype", ["uint16", "uint32", "float32", "float64"])
def test_out_dtype_round_trips_values(tmp_path, out_dtype):
    vals = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    a = _mk(vals, np.uint16)
    p = tmp_path / f"o_{out_dtype}.shad"
    write_cell_archive(a, p)
    s = CellStore(p)
    try:
        got = s.gather_rows(np.array([0, 1]), out_dtype=out_dtype)
        assert got.dtype == np.dtype(out_dtype)
        np.testing.assert_array_equal(got.toarray().astype(np.int64), vals.astype(np.int64))
    finally:
        s.close()
