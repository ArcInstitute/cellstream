"""The zstd cell encoder must not write a frame its own decoder refuses (#282 item 9).

Over the cap the write used to SUCCEED and produce an archive neither engine could
read: 'decompressed chunk size 67109376 exceeds the 67108864-byte cap (corrupt file?)'
-- a message that blames the archive for a frame cellstream itself wrote.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cellstream
from cellstream.cell import writer as cell_writer
from cellstream.cell.archive_writer import CellArchiveWriter, concat_cell_archives
from cellstream.cell.writer import _validate_zstd_frame_bounds, write_cell_archive
from cellstream.v2.format import MAX_CHUNK_DECOMPRESSED_BYTES as CAP

N_VARS = 64


def _one_big_cell(n_values, n_vars=N_VARS):
    """One cell with n_values float64 entries, plus a tiny second cell.

    Duplicate indices keep n_vars at 64 so the archive is ~50 KB instead of the
    8.5M-column form the reproduction used; require_unique_indices=False is a
    documented public option and the frame-size arithmetic is identical either way.
    """
    indptr = np.array([0, n_values, n_values + 3], dtype=np.int64)
    head = np.sort(np.tile(np.arange(n_vars, dtype=np.int32), n_values // n_vars + 1)[:n_values])
    indices = np.concatenate([head, np.array([0, 1, 2], dtype=np.int32)])
    data = np.full(n_values + 3, 0.5, dtype=np.float64)
    x = sp.csr_matrix((data, indices, indptr), shape=(2, n_vars))
    return ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=["big", "small"]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )


OVER = CAP // 8 + 64  # float64 -> 8 bytes/value
UNDER = CAP // 8 - 64


# --- the predicate itself ----------------------------------------------------


def test_predicate_boundary_is_exact():
    _validate_zstd_frame_bounds(np.array([0, CAP // 8], dtype=np.int64), value_itemsize=8)
    with pytest.raises(ValueError):
        _validate_zstd_frame_bounds(np.array([0, CAP // 8 + 1], dtype=np.int64), value_itemsize=8)


def test_predicate_index_blob_binds_for_uint32():
    """Indices are ALWAYS uint32 for zstd, so a uint32-valued archive is bound by
    max(4, 4) = 4 bytes -- 16,777,216 values, not 8,388,608."""
    _validate_zstd_frame_bounds(np.array([0, CAP // 4], dtype=np.int64), value_itemsize=4)
    with pytest.raises(ValueError):
        _validate_zstd_frame_bounds(np.array([0, CAP // 4 + 1], dtype=np.int64), value_itemsize=4)


def test_predicate_reports_the_global_row():
    indptr = np.array([0, 1, CAP // 8 + 65], dtype=np.int64)
    with pytest.raises(ValueError, match=r"row 1001\b"):
        _validate_zstd_frame_bounds(indptr, value_itemsize=8, row_offset=1000)


def test_predicate_ignores_an_empty_indptr():
    _validate_zstd_frame_bounds(np.array([0], dtype=np.int64), value_itemsize=8)


# --- call site 1: in-memory --------------------------------------------------


def test_in_memory_write_over_the_cap_is_rejected(tmp_path):
    out = tmp_path / "over.shad"
    with pytest.raises(ValueError) as ei:
        write_cell_archive(
            _one_big_cell(OVER),
            out,
            codec="zstd",
            data_dtype="float64",
            require_unique_indices=False,
        )
    msg = str(ei.value)
    assert "row 0" in msg and str(CAP) in msg
    assert not out.exists()


@pytest.mark.parametrize("engine_env", [None, "1"], indirect=True)
def test_just_under_the_cap_round_trips(tmp_path, engine_env):
    out = tmp_path / "under.shad"
    write_cell_archive(
        _one_big_cell(UNDER),
        out,
        codec="zstd",
        data_dtype="float64",
        require_unique_indices=False,
    )
    st = cellstream.open(out)
    try:
        assert st.gather_rows(np.arange(2, dtype=np.int64)).nnz == UNDER + 3
    finally:
        st.close()


# --- call site 2: streaming source -------------------------------------------


def test_streaming_source_over_the_cap_is_rejected(tmp_path):
    """A backed .h5ad input goes through _encode_cell_streaming, a different call site."""
    src = tmp_path / "src.h5ad"
    _one_big_cell(OVER).write_h5ad(src)
    out = tmp_path / "stream.shad"
    with pytest.raises(ValueError, match=str(CAP)):
        write_cell_archive(
            src, out, codec="zstd", data_dtype="float64", require_unique_indices=False
        )
    assert not out.exists()


# --- call site 2b: the #216 integer -> zstd-float retry -----------------------


def test_integer_attempt_retry_to_zstd_float_is_also_guarded(tmp_path):
    """codec=None on non-integer float data encodes optimistically as pfordelta, gets
    NonIntegerCountsError, and RETRIES as zstd float. The guard must fire on the retry."""
    a = _one_big_cell(OVER)
    a.X.data[:] = 0.5  # genuinely non-integer -> forces the retry
    out = tmp_path / "retry.shad"
    with pytest.raises(ValueError, match=str(CAP)):
        write_cell_archive(a, out, require_unique_indices=False)
    assert not out.exists()


# --- call site 3: the push writer --------------------------------------------


def test_push_writer_over_the_cap_is_rejected(tmp_path):
    """encode_block_rows=1 is load-bearing: the defaults are 65,536 rows / 64M nnz, so a
    two-row 8.4M-nnz block would never flush and the guard would never run."""
    out = tmp_path / "push.shad"
    w = CellArchiveWriter(
        out,
        n_vars=N_VARS,
        value_dtype="float64",
        codec="zstd",
        require_unique_indices=False,
        encode_block_rows=1,
    )
    try:
        with pytest.raises(ValueError, match=str(CAP)):
            w.add_block(_one_big_cell(OVER).X)
    finally:
        w.abort()


# --- call site 4: concat republishing a legacy part --------------------------


def test_concat_rejects_a_part_with_an_over_cap_frame(tmp_path, monkeypatch):
    """concat byte-copies frames without decoding, so it cannot CREATE an over-cap
    frame -- but it must not republish one written by a pre-fix release. The part is
    built with the guard patched out, which is what a legacy archive looks like."""
    part = tmp_path / "legacy.shad"
    monkeypatch.setattr(cell_writer, "_validate_zstd_frame_bounds", lambda *a, **k: None)
    write_cell_archive(
        _one_big_cell(OVER),
        part,
        codec="zstd",
        data_dtype="float64",
        require_unique_indices=False,
    )
    monkeypatch.undo()

    out = tmp_path / "merged.shad"
    with pytest.raises(ValueError, match=str(CAP)):
        concat_cell_archives([part], out)
    assert not out.exists()
