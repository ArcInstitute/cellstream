"""#222 append-only CellArchiveWriter."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.cell import CellArchiveWriter, open_cell
from cellstream.cell import format as fmt
from cellstream.cell.writer import _resolve_push_codec, write_cell_archive
from cellstream.errors import ArchiveExistsError, LossyCastError, NonIntegerCountsError

from .conftest import assert_archives_equivalent, make_cell_adata
from .test_zstd_writer import _float_adata


@pytest.mark.parametrize(
    "value_dtype,codec,data_dtype,exp_codec,exp_on_disk,exp_schema,exp_pyfastpfor",
    [
        (
            "int32",
            None,
            None,
            fmt.CODEC_PFORDELTA,
            "uint32",
            fmt.SCHEMA_VERSION,
            fmt.PYFASTPFOR_CODEC,
        ),
        (
            "uint16",
            None,
            None,
            fmt.CODEC_PFORDELTA,
            "uint32",
            fmt.SCHEMA_VERSION,
            fmt.PYFASTPFOR_CODEC,
        ),
        (
            "int64",
            None,
            None,
            fmt.CODEC_PFORDELTA,
            "uint32",
            fmt.SCHEMA_VERSION,
            fmt.PYFASTPFOR_CODEC,
        ),
        ("int32", "zstd", None, fmt.CODEC_ZSTD, "uint32", fmt.SCHEMA_VERSION_ZSTD, None),
        ("float32", None, None, fmt.CODEC_ZSTD, "float32", fmt.SCHEMA_VERSION_ZSTD, None),
        ("float64", None, None, fmt.CODEC_ZSTD, "float64", fmt.SCHEMA_VERSION_ZSTD, None),
        ("float64", "zstd", "float32", fmt.CODEC_ZSTD, "float32", fmt.SCHEMA_VERSION_ZSTD, None),
    ],
)
def test_resolve_push_codec_table(
    value_dtype, codec, data_dtype, exp_codec, exp_on_disk, exp_schema, exp_pyfastpfor
):
    rc, on_disk, schema, pf = _resolve_push_codec(value_dtype, codec, data_dtype)
    assert (rc, on_disk, schema, pf) == (exp_codec, exp_on_disk, exp_schema, exp_pyfastpfor)


def test_resolve_push_codec_pfordelta_on_float_raises():
    with pytest.raises(ValueError, match="integer counts"):
        _resolve_push_codec("float32", "pfordelta", None)


def test_resolve_push_codec_data_dtype_on_int_raises():
    with pytest.raises(ValueError):
        _resolve_push_codec("int32", None, "float32")


def _blocks(adata, sizes):
    lo = 0
    for s in sizes:
        yield adata.X[lo : lo + s]
        lo += s


def test_writer_byte_identity_ungrouped(tmp_path):
    a = make_cell_adata()
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by=None)

    got = tmp_path / "got.shad"
    w = CellArchiveWriter(got, n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16")
    for blk in _blocks(a, [100, 40, 100]):
        w.add_block(sp.csr_matrix(blk))
    w.finalize(obs=a.obs, var=a.var)
    assert_archives_equivalent(ref, got)


def test_writer_finalize_bad_obs_count_cleans_tmpd(tmp_path):
    a = make_cell_adata()
    w = CellArchiveWriter(
        tmp_path / "o.shad", n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16"
    )
    w.add_block(sp.csr_matrix(a.X[:100]))
    tmpd = w._tmpd
    assert tmpd.exists()
    with pytest.raises(ValueError, match="rows"):
        w.finalize(obs=a.obs, var=a.var)  # 240 obs rows vs 100 added
    assert not tmpd.exists(), (
        "finalize() must clean up its staging tmpd even on the count-mismatch error"
    )


def test_writer_byte_identity_grouped_presorted_identity_fastpath(tmp_path):
    """Feed rows already in reference-first grouped order -> finalize perm is identity (no reorder),
    byte-identical to write_cell_archive over the SAME pre-sorted input."""
    a = make_cell_adata()
    from cellstream.cell.writer import _resolve_storage_order

    perm, _ = _resolve_storage_order(
        a.obs, np.diff(a.X.tocsr().indptr), "target_gene", ["t00"], None, None
    )
    a = a[perm].copy()  # pre-sort into write_cell_archive's storage order -> both see identity perm

    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by="target_gene", reference=["t00"])
    got = tmp_path / "got.shad"
    w = CellArchiveWriter(got, n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16")
    for blk in _blocks(a, [80, 80, 80]):
        w.add_block(sp.csr_matrix(blk))
    w.finalize(obs=a.obs, var=a.var, group_by="target_gene", reference=["t00"])
    assert_archives_equivalent(ref, got)


def test_writer_byte_identity_grouped_unsorted_reorder_path(tmp_path):
    """Feed rows in arbitrary order; finalize reorders to canonical -> byte-identical to
    write_cell_archive over the SAME unsorted input (which also reorders)."""
    a = make_cell_adata()
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by="target_gene", reference=["t00"])
    got = tmp_path / "got.shad"
    w = CellArchiveWriter(got, n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16")
    for blk in _blocks(a, [120, 120]):
        w.add_block(sp.csr_matrix(blk))
    w.finalize(obs=a.obs, var=a.var, group_by="target_gene", reference=["t00"])
    assert_archives_equivalent(ref, got)


def test_writer_byte_identity_categorical_group_by(tmp_path):
    """A categorical group_by sorts by CATEGORY order, not alphabetical -> the shared planner
    reproduces it exactly on both writers."""
    a = make_cell_adata()
    cats = sorted(set(a.obs["target_gene"]), reverse=True)  # non-alphabetical category order
    a.obs["target_gene"] = pd.Categorical(a.obs["target_gene"], categories=cats, ordered=True)
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by="target_gene", reference=[cats[0]])
    got = tmp_path / "got.shad"
    w = CellArchiveWriter(got, n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16")
    for blk in _blocks(a, [130, 110]):
        w.add_block(sp.csr_matrix(blk))
    w.finalize(obs=a.obs, var=a.var, group_by="target_gene", reference=[cats[0]])
    assert_archives_equivalent(ref, got)


def test_writer_byte_identity_zstd_int(tmp_path):
    """codec='zstd' over integer counts (spec SS10 gap: the byte-identity suite only exercised
    pfordelta before this fix)."""
    a = make_cell_adata()
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by=None, codec="zstd")
    got = tmp_path / "got.shad"
    w = CellArchiveWriter(got, n_vars=a.n_vars, codec="zstd", value_dtype="uint16")
    for blk in _blocks(a, [100, 40, 100]):
        w.add_block(sp.csr_matrix(blk))
    w.finalize(obs=a.obs, var=a.var)
    assert_archives_equivalent(ref, got)


def test_writer_byte_identity_zstd_float(tmp_path):
    """codec='zstd' over genuine (non-integer) float32 values -- the other zstd(#10) gap."""
    a = _float_adata(np.float32)  # n_obs=120, n_vars=40, genuine fractional values
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by=None, codec="zstd", data_dtype="float32")
    got = tmp_path / "got.shad"
    w = CellArchiveWriter(got, n_vars=a.n_vars, codec="zstd", value_dtype="float32")
    for blk in _blocks(a, [40, 40, 40]):
        w.add_block(sp.csr_matrix(blk))
    w.finalize(obs=a.obs, var=a.var)
    assert_archives_equivalent(ref, got)


def test_writer_zstd_float_narrowing_raises_lossycasterror(tmp_path):
    """[T6-M1] add_block on a float64 block that does not round-trip to a declared float32
    value_dtype must raise LossyCastError (the same cast_data gate write_cell_archive already
    enforces for its in-memory/streaming zstd-float paths -- see
    test_zstd_writer.py::test_data_dtype_float32_lossless_checked_raises)."""
    a = make_cell_adata()
    x = a.X.tocsr().astype(np.float64)
    x.data[0] = np.float64(1) / np.float64(3)  # not float32-exact
    blk = sp.csr_matrix(x[:100])
    # #233: coalescing defers encode-fused validation to the flush; encode_block_rows=1 forces the
    # flush (and the LossyCastError) to fire synchronously within add_block, as this test asserts.
    w = CellArchiveWriter(
        tmp_path / "o.shad",
        n_vars=a.n_vars,
        codec="zstd",
        value_dtype="float32",
        encode_block_rows=1,
    )
    with pytest.raises(LossyCastError):
        w.add_block(blk)
    w.abort()


def test_writer_zstd_float_narrowing_allow_lossy_forces_cast(tmp_path):
    """[T6-M1] the SAME lossy block, with allow_lossy=True, is accepted (forced cast) instead
    of raising -- companion to the raising case above."""
    a = make_cell_adata()
    x = a.X.tocsr().astype(np.float64)
    x.data[0] = np.float64(1) / np.float64(3)  # not float32-exact
    blk = sp.csr_matrix(x[:100])
    w = CellArchiveWriter(
        tmp_path / "o.shad",
        n_vars=a.n_vars,
        codec="zstd",
        value_dtype="float32",
        allow_lossy=True,
    )
    w.add_block(blk)  # forced cast -- must NOT raise
    w.finalize(obs=a.obs.iloc[:100], var=a.var)
    s = open_cell(tmp_path / "o.shad")
    try:
        assert s.manifest["value_dtype_on_disk"] == "float32"
    finally:
        s.close()


def _int32_fat_tail_adata():
    a = make_cell_adata()
    x = a.X.tocsr().astype(np.int32)
    x.data[:3] = [124_000, 190_000, 233_000]  # a few values > uint16 max, < uint32 max
    a.X = x
    return a


def test_writer_int32_fat_tail_roundtrips_uint32(tmp_path):
    a = _int32_fat_tail_adata()
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by=None)  # int32 source -> stored uint32
    got = tmp_path / "got.shad"
    w = CellArchiveWriter(got, n_vars=a.n_vars, codec="pfordelta", value_dtype="int32")
    for blk in _blocks(a, [120, 120]):
        w.add_block(sp.csr_matrix(blk))
    w.finalize(obs=a.obs, var=a.var)
    assert_archives_equivalent(ref, got)
    s = open_cell(got)
    try:
        assert s.manifest["value_dtype_on_disk"] == "uint32"
        assert s.manifest["x_data_dtype_min"] == "uint32"  # 233_000 needs uint32
    finally:
        s.close()


def test_writer_fractional_under_int_dtype_fails_loud(tmp_path):
    a = make_cell_adata()
    x = a.X.tocsr().astype(np.float32)
    x.data[0] = 1.5
    # #233: encode_block_rows=1 forces the flush (and NonIntegerCountsError) synchronously within
    # add_block; without it, coalescing would defer the raise to finalize.
    w = CellArchiveWriter(
        tmp_path / "o.shad",
        n_vars=a.n_vars,
        codec="pfordelta",
        value_dtype="int32",
        encode_block_rows=1,
    )
    with pytest.raises(NonIntegerCountsError):
        w.add_block(sp.csr_matrix(x[:100]))
    w.abort()


def test_writer_wrong_ncols_fails(tmp_path):
    a = make_cell_adata()
    w = CellArchiveWriter(
        tmp_path / "o.shad", n_vars=a.n_vars + 1, codec="pfordelta", value_dtype="uint16"
    )
    with pytest.raises(ValueError, match="expected n_vars"):
        w.add_block(sp.csr_matrix(a.X[:10]))
    w.abort()


def test_writer_obs_length_mismatch_fails(tmp_path):
    a = make_cell_adata()
    w = CellArchiveWriter(
        tmp_path / "o.shad", n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16"
    )
    w.add_block(sp.csr_matrix(a.X[:100]))
    with pytest.raises(ValueError, match="rows"):
        w.finalize(obs=a.obs, var=a.var)  # a.obs is 240 rows, only 100 added
    w.abort()


def test_writer_var_length_mismatch_fails(tmp_path):
    """finalize checks var length against n_vars, mirroring the existing obs-vs-_n_obs check
    (Copilot #222 review) -- AnnData would otherwise raise, but generically."""
    a = make_cell_adata()
    bad_var = make_cell_adata(n_vars=a.n_vars + 1).var
    w = CellArchiveWriter(
        tmp_path / "o.shad", n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16"
    )
    w.add_block(sp.csr_matrix(a.X))
    with pytest.raises(ValueError, match="var has"):
        w.finalize(obs=a.obs, var=bad_var)  # bad_var has n_vars+1 rows
    w.abort()


def test_writer_empty_archive(tmp_path):
    a = make_cell_adata()
    ref = tmp_path / "ref.shad"
    write_cell_archive(a[:0].copy(), ref, group_by=None)
    got = tmp_path / "got.shad"
    w = CellArchiveWriter(got, n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16")
    w.finalize(obs=a.obs.iloc[:0], var=a.var)
    assert_archives_equivalent(ref, got)


def test_writer_exists_guard(tmp_path):
    (tmp_path / "o.shad").write_bytes(b"x")
    with pytest.raises(ArchiveExistsError):
        CellArchiveWriter(tmp_path / "o.shad", n_vars=5, codec="pfordelta", value_dtype="uint16")


def test_writer_crash_leaves_no_shad(tmp_path):
    out = tmp_path / "crash.shad"
    script = textwrap.dedent(f"""
        import os, numpy as np, scipy.sparse as sp
        from cellstream.cell import CellArchiveWriter
        w = CellArchiveWriter({str(out)!r}, n_vars=10, codec="pfordelta", value_dtype="uint16")
        blk = sp.random(50, 10, density=0.2, format="csr").astype(np.uint16)
        w.add_block(blk)
        os._exit(9)  # hard kill mid-write, before finalize
    """)
    subprocess.run([sys.executable, "-c", script], check=False)
    assert not out.exists(), "a killed writer must not leave a .shad"
    # a scratch temp dir may remain (cellstream_cellw_*) but never the archive itself


def test_writer_accumulators_are_O_nobs_not_nnz(tmp_path):
    """Structural RAM proof (AC#5): the writer's accumulators are O(n_obs), never O(nnz).

    ``_frame_lens``/``_row_nnz`` each hold one int64 per ROW added so far (never one per nnz
    element), and ``_total_nnz`` is folded into a plain scalar as blocks stream in rather than
    collected into an array. This is the structural invariant behind "peak RAM O(block)+O(n_obs)"
    claimed in the class docstring -- an accidental switch to an O(nnz)-scale accumulator here
    would silently regress that guarantee without changing any byte-identity test's outcome.
    """
    a = make_cell_adata(n_obs=240, n_vars=60)
    # #233: encode_block_rows=1 flushes each add_block so the O(n_obs) accumulators populate as
    # blocks stream (with default coalescing they would buffer in _pending until finalize; the
    # _pending bound is covered by test_archive_writer_coalesce::test_coalesce_buffer_bounded_by_caps).
    w = CellArchiveWriter(
        tmp_path / "o.shad",
        n_vars=a.n_vars,
        codec="pfordelta",
        value_dtype="uint16",
        encode_block_rows=1,
    )
    for blk in _blocks(a, [80, 80, 80]):
        w.add_block(sp.csr_matrix(blk))
    n_frame_entries = sum(x.size for x in w._frame_lens)  # one int64 per ROW, never per nnz
    n_rownnz_entries = sum(x.size for x in w._row_nnz)
    assert n_frame_entries == a.n_obs
    assert n_rownnz_entries == a.n_obs
    assert isinstance(w._total_nnz, int)  # a scalar, not an array
    w.finalize(obs=a.obs, var=a.var)


def test_writer_terminal_state_after_failed_finalize(tmp_path):
    """A finalize() that raises (here: obs-count mismatch) must leave the writer TERMINAL: the
    finally deletes the staging tmpd, so a later add_block()/finalize() must raise a clear
    'already finalized or aborted' error rather than a confusing I/O error on the deleted staging
    path. (Copilot #222 review.)"""
    a = make_cell_adata()
    w = CellArchiveWriter(
        tmp_path / "o.shad", n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16"
    )
    w.add_block(sp.csr_matrix(a.X[:100]))
    with pytest.raises(ValueError, match="rows"):
        w.finalize(obs=a.obs, var=a.var)  # 240 obs vs 100 added -> raises
    with pytest.raises(RuntimeError, match="finalized or aborted"):
        w.add_block(sp.csr_matrix(a.X[:10]))
    with pytest.raises(RuntimeError, match="finalized or aborted"):
        w.finalize(obs=a.obs.iloc[:100], var=a.var)
