"""#222 concat_cell_archives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.cell import CellArchiveWriter, concat_cell_archives, open_cell
from cellstream.cell.writer import write_cell_archive

from .conftest import assert_archives_equivalent, make_cell_adata
from .test_zstd_writer import _float_adata


def _write_slices(adata, dst_dir, n_parts):
    """Byte-exact 'split': write each contiguous row-range of `adata` as its own UNGROUPED cell
    archive via the trusted `write_cell_archive`. A per-cell frame is a pure function of that one row
    (position-independent, deterministic codec, NO trained dict — spec §8), so a slice's frames are
    byte-identical to the same rows in a monolithic write for BOTH pfordelta and zstd — with no
    decode/re-encode round-trip (the flaw Copilot flagged: gather_rows() returns float32 for integer
    archives, so decode+re-encode would rely on cast + codec determinism, not the byte-copy invariant
    concat actually exercises). Parts are ungrouped; concat is told group_by explicitly (grouped
    tests) or inherits None (ungrouped)."""
    n = adata.n_obs
    bounds = [(k * n) // n_parts for k in range(n_parts + 1)]
    paths = []
    for k in range(n_parts):
        a, b = bounds[k], bounds[k + 1]
        p = dst_dir / f"part{k}.shad"
        write_cell_archive(adata[a:b].copy(), p, group_by=None)
        paths.append(p)
    return paths


def _write_slices_zstd_float(adata, dst_dir, n_parts, *, data_dtype="float32"):
    """Like `_write_slices`, but for zstd FLOAT storage: passes `codec='zstd'` + an explicit
    `data_dtype` through to `write_cell_archive` (so there is no optimistic-retry warning, same
    rationale as `_write_slices`'s pfordelta/zstd-int coverage -- a per-cell zstd frame is a pure
    function of that one row, so a slice's frames are byte-identical to the same rows in a
    monolithic write)."""
    n = adata.n_obs
    bounds = [(k * n) // n_parts for k in range(n_parts + 1)]
    paths = []
    for k in range(n_parts):
        a, b = bounds[k], bounds[k + 1]
        p = dst_dir / f"zfpart{k}.shad"
        write_cell_archive(adata[a:b].copy(), p, group_by=None, codec="zstd", data_dtype=data_dtype)
        paths.append(p)
    return paths


def test_concat_slices_equals_monolith_ungrouped(tmp_path):
    a = make_cell_adata()
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by=None)
    parts = _write_slices(a, tmp_path, 3)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out)  # parts ungrouped -> concat inherits group_by=None
    assert_archives_equivalent(ref, out)


def test_concat_single_grouped_part_inherits(tmp_path):
    """A single REAL grouped archive concatenated: concat inherits group_by/reference from its
    manifest and BYTE-COPIES its (grouped) payload, re-planning to identity -> byte-identical. Tests
    the inherit path + byte-copy of a real grouped payload, with no decode."""
    a = make_cell_adata()
    A = tmp_path / "A.shad"
    write_cell_archive(a, A, group_by="target_gene", reference=["t00"])
    out = tmp_path / "out.shad"
    concat_cell_archives([A], out)  # inherit group_by/reference from A's manifest
    assert_archives_equivalent(A, out)


def test_concat_inherits_multi_label_reference(tmp_path):
    """A part whose manifest `reference` is a LIST (multi-label reference) must be concatenable
    without _inherit crashing on a set-build over an unhashable list (regression)."""
    a = make_cell_adata()
    A = tmp_path / "A.shad"
    write_cell_archive(a, A, group_by="target_gene", reference=["t00", "t01"])
    out = tmp_path / "out.shad"
    concat_cell_archives([A], out)  # inherit the list-valued reference from A's manifest
    assert_archives_equivalent(A, out)


def test_concat_slices_equals_monolith_grouped(tmp_path):
    a = make_cell_adata()
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by="target_gene", reference=["t00"])
    parts = _write_slices(a, tmp_path, 4)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out, group_by="target_gene", reference=["t00"])  # explicit group_by
    assert_archives_equivalent(ref, out)


def test_concat_of_cellarchivewriter_parts(tmp_path):
    """End-to-end state3 flow: parts written by CellArchiveWriter (byte-identical to write_cell_archive
    per AC#1) concatenated -> byte-identical to the grouped monolith."""
    a = make_cell_adata()  # uint16 integer counts
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by="target_gene", reference=["t00"])
    n = a.n_obs
    bounds = [(k * n) // 3 for k in range(4)]
    parts = []
    for k in range(3):
        lo, hi = bounds[k], bounds[k + 1]
        p = tmp_path / f"w{k}.shad"
        w = CellArchiveWriter(p, n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16")
        w.add_block(sp.csr_matrix(a.X[lo:hi]))
        w.finalize(obs=a.obs.iloc[lo:hi], var=a.var)  # ungrouped parts
        parts.append(p)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out, group_by="target_gene", reference=["t00"])
    assert_archives_equivalent(ref, out)


def test_concat_reader_parity(tmp_path):
    a = make_cell_adata()
    A = tmp_path / "A.shad"
    write_cell_archive(a, A, group_by="target_gene", reference=["t00"])
    parts = _write_slices(a, tmp_path, 3)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out, group_by="target_gene", reference=["t00"])
    sa = open_cell(A)
    try:
        sb = open_cell(out)
        try:
            ids = np.array([5, 200, 1, 199, 100], dtype=np.int64)
            assert (sa.gather_rows(ids) != sb.gather_rows(ids)).nnz == 0
            for lbl in sa.group_labels():
                assert (sa.read_group(lbl) != sb.read_group(lbl)).nnz == 0
            assert (sa.read_reference() != sb.read_reference()).nnz == 0
        finally:
            sb.close()
    finally:
        sa.close()


def test_concat_categorical_roundtrip(tmp_path):
    a = make_cell_adata()
    cats = sorted(set(a.obs["target_gene"]), reverse=True)
    a.obs["target_gene"] = pd.Categorical(a.obs["target_gene"], categories=cats, ordered=True)
    A = tmp_path / "A.shad"
    write_cell_archive(a, A, group_by="target_gene", reference=[cats[0]])
    parts = _write_slices(a, tmp_path, 3)  # ungrouped slices preserve the categorical obs column
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out, group_by="target_gene", reference=[cats[0]])
    assert_archives_equivalent(A, out)


def test_concat_mixed_width_merges_widest(tmp_path):
    a = make_cell_adata()
    small = a[:120].copy()  # max stays < 65536 -> uint16
    big = a[120:].copy()
    xb = big.X.tocsr().astype(np.int32)
    xb.data[0] = 200_000  # -> uint32
    big.X = xb
    ps, pb = tmp_path / "s.shad", tmp_path / "b.shad"
    write_cell_archive(small, ps, group_by=None)
    write_cell_archive(big, pb, group_by=None)
    out = tmp_path / "out.shad"
    concat_cell_archives([ps, pb], out)
    s = open_cell(out)
    try:
        assert s.manifest["value_dtype_on_disk"] == "uint32"  # widened for the 200k value
        row = s.gather_rows(np.array([120], dtype=np.int64))  # the big-part first row
        assert int(row.data.max()) == 200_000  # decodes losslessly
    finally:
        s.close()


def test_concat_single_part(tmp_path):
    a = make_cell_adata()
    A = tmp_path / "A.shad"
    write_cell_archive(a, A, group_by=None)
    parts = _write_slices(a, tmp_path, 1)  # one part == the whole ungrouped archive
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out)
    assert_archives_equivalent(A, out)


def test_concat_empty_parts_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        concat_cell_archives([], tmp_path / "o.shad")


def test_concat_incompatible_nvars_raises(tmp_path):
    a = make_cell_adata(n_vars=60)
    b = make_cell_adata(n_vars=61)
    pa, pb = tmp_path / "a.shad", tmp_path / "b.shad"
    write_cell_archive(a, pa, group_by=None)
    write_cell_archive(b, pb, group_by=None)
    with pytest.raises(ValueError, match="n_vars"):
        concat_cell_archives([pa, pb], tmp_path / "o.shad")


def test_concat_zstd_float_slices_equals_monolith(tmp_path):
    """AC#2-style byte-identity for zstd FLOAT storage (spec SS10 gap): slices concatenated ==
    the monolithic zstd-float write, via the same byte-copy invariant _write_slices proves for
    pfordelta/zstd-int."""
    a = _float_adata(np.float32)  # n_obs=120, n_vars=40, genuine fractional values
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by=None, codec="zstd", data_dtype="float32")
    parts = _write_slices_zstd_float(a, tmp_path, 3)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out)
    assert_archives_equivalent(ref, out)


def test_concat_float_and_int_mix_raises(tmp_path):
    """concat must reject mixing a zstd-FLOAT part with a zstd-INTEGER part: same codec='zstd'
    on both, but the frames are not mutually decodable (spec SS6.2)."""
    a_int = make_cell_adata(n_vars=40)  # n_vars matches _float_adata so this reaches the
    a_float = _float_adata(np.float32)  # float/int check rather than an earlier n_vars reject
    p_int, p_float = tmp_path / "int.shad", tmp_path / "float.shad"
    write_cell_archive(a_int, p_int, group_by=None, codec="zstd")
    write_cell_archive(a_float, p_float, group_by=None, codec="zstd", data_dtype="float32")
    with pytest.raises(ValueError, match="mix float and integer"):
        concat_cell_archives([p_int, p_float], tmp_path / "out.shad")


def test_concat_float_width_mismatch_raises(tmp_path):
    """Two zstd-FLOAT parts of different on-disk widths (float32 vs float64) must reject --
    per spec SS6.2, float storage additionally requires an identical value_dtype_on_disk."""
    a32 = _float_adata(np.float32)
    a64 = _float_adata(np.float64)
    p32, p64 = tmp_path / "f32.shad", tmp_path / "f64.shad"
    write_cell_archive(a32, p32, group_by=None, codec="zstd", data_dtype="float32")
    write_cell_archive(a64, p64, group_by=None, codec="zstd", data_dtype="float64")
    with pytest.raises(ValueError, match="float parts must share"):
        concat_cell_archives([p32, p64], tmp_path / "out.shad")


def test_concat_different_uns_raises(tmp_path):
    """concat must fail loud on disagreeing uns rather than silently keeping parts[0]'s
    (spec SS6.2/SS13 require-equal, not first-wins -- the bug class this repo has hit before:
    CellArchiveSource.uns unconditionally returning {} sailed through every prior test)."""
    a = make_cell_adata()
    n = a.n_obs
    a1, a2 = a[: n // 2].copy(), a[n // 2 :].copy()
    a1.uns["k"] = "v1"
    a2.uns["k"] = "v2"
    p1, p2 = tmp_path / "p1.shad", tmp_path / "p2.shad"
    write_cell_archive(a1, p1, group_by=None)
    write_cell_archive(a2, p2, group_by=None)
    with pytest.raises(ValueError, match="uns"):
        concat_cell_archives([p1, p2], tmp_path / "out.shad")


def test_concat_same_nonempty_uns_succeeds(tmp_path):
    """Parts sharing the SAME non-empty uns must concat cleanly -- confirms the require-equal
    check actually passes on agreement and the merge reaches the byte-copy (not merely that it
    rejects disagreement)."""
    a = make_cell_adata()
    a.uns["k"] = "v"
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by=None)
    parts = _write_slices(a, tmp_path, 3)  # each slice keeps the whole (global) uns dict
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out)
    assert_archives_equivalent(ref, out)


def test_concat_crash_leaves_no_shad(tmp_path, monkeypatch):
    a = make_cell_adata()
    parts = _write_slices(a, tmp_path, 2)
    out = tmp_path / "out.shad"
    import cellstream.cell.archive_writer as aw

    def boom(*x, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(aw, "_reorder_and_finalize", boom)
    with pytest.raises(RuntimeError):
        concat_cell_archives(parts, out)
    assert not out.exists()


def test_concat_sidecars_are_O_nobs_not_nnz(tmp_path):
    """Structural RAM proof for concat, symmetric with
    test_archive_writer.py::test_writer_accumulators_are_O_nobs_not_nnz.

    concat_cell_archives is a plain function -- unlike CellArchiveWriter there is no
    persistent object whose internal in_offsets/row_nnz/obs accumulators could be introspected
    after the call returns, so this instead asserts the OUTPUT's shape (the fallback the task
    brief sanctions when internals aren't cleanly introspectable): x_offsets and x_indptr are
    each exactly n_obs+1 long -- one entry per ROW, never one per nonzero -- while the payload
    scales with nnz. make_cell_adata's n_obs=240 vs nnz=4000 (~16.7x apart, same fixture the
    writer-side test relies on for discriminating power) means a sidecar that regressed to
    nnz-scale would blow this comparison away, not coincidentally satisfy it.
    """
    a = make_cell_adata(n_obs=240, n_vars=60)
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by=None)
    parts = _write_slices(a, tmp_path, 3)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out)
    s = open_cell(out)
    try:
        assert len(s.offsets) == a.n_obs + 1
        assert len(s.indptr) == a.n_obs + 1
        total_nnz = int(s.indptr[-1])
        assert total_nnz == a.X.nnz  # sanity: the merge lost no rows
        assert len(s.offsets) < total_nnz  # sidecars O(n_obs); payload len tracks O(nnz)
    finally:
        s.close()


def test_concat_preserves_annotation_bearing_parts(tmp_path):
    """Was `test_concat_rejects_annotation_bearing_parts`: before #237 concat rebuilt the merged
    header from obs/var/uns only and failed loud on any annotation-bearing part (#232's guard).
    It now preserves them; the detailed coverage lives in test_concat_annotations.py."""
    a = make_cell_adata(annotations=True)
    p0, p1 = tmp_path / "p0.shad", tmp_path / "p1.shad"
    write_cell_archive(a, p0, group_by=None)
    write_cell_archive(a, p1, group_by=None)
    out = tmp_path / "out.shad"
    concat_cell_archives([p0, p1], out)
    s = open_cell(out)
    try:
        assert set(s.obsm) == {"X_pca"} and set(s.varm) == {"PCs"}
        assert set(s.obsp) == {"connectivities", "distances"} and set(s.varp) == {"corr"}
        assert s.obsm["X_pca"].shape == (2 * a.n_obs, 10)
        assert s.obsp["connectivities"].shape == (2 * a.n_obs, 2 * a.n_obs)
    finally:
        s.close()


def test_uns_equal_confirms_sparse_and_pandas_values():
    """_uns_equal's bool(va == vb) fallback RAISES for a DataFrame and for a sparse matrix, and
    the bare `except Exception: return False` then reports IDENTICAL values as unequal. #237
    reuses this helper for the varm/varp require-equal check, where sparse graphs and DataFrames
    are the two most common types -- so it must confirm them, not conservatively reject them."""
    from cellstream.cell.archive_writer import _uns_equal

    m = sp.random(6, 6, density=0.3, random_state=1, format="csr")
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    assert _uns_equal({"m": m, "d": df}, {"m": m.copy(), "d": df.copy()})

    m2 = m.copy()
    m2.data[0] += 1.0  # .data mutation, not [0, 0] -- avoids SparseEfficiencyWarning
    assert not _uns_equal({"m": m}, {"m": m2})
    assert not _uns_equal(
        {"m": m}, {"m": sp.random(5, 5, density=0.3, random_state=1, format="csr")}
    )
    assert not _uns_equal({"d": df}, {"d": pd.DataFrame({"a": [1, 9], "b": [3.0, 4.0]})})
    assert not _uns_equal({"m": m}, {"m": m.toarray()})  # sparse vs dense -> conservative False
    assert not _uns_equal({"d": df}, {"d": 3})  # pandas vs scalar -> conservative False
    assert _uns_equal({"n": {"deep": np.arange(4)}}, {"n": {"deep": np.arange(4)}})  # nested dict


def test_concat_same_dataframe_uns_succeeds(tmp_path):
    """Parts carrying an IDENTICAL DataFrame uns value must concat cleanly. They were rejected
    before #237 upgraded _uns_equal. Not compared via assert_archives_equivalent: that helper
    asserts `sa.uns == sb.uns`, and dict equality over DataFrame values raises."""
    a = make_cell_adata()
    a.uns["table"] = pd.DataFrame({"x": [1, 2, 3]})
    parts = _write_slices(a, tmp_path, 3)  # each slice keeps the whole (global) uns dict
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out)
    s = open_cell(out)
    try:
        pd.testing.assert_frame_equal(s.uns["table"], a.uns["table"])
    finally:
        s.close()


def test_concat_different_dataframe_uns_raises(tmp_path):
    """The upgrade must not loosen the require-equal policy: DIFFERING DataFrame uns still fails."""
    a = make_cell_adata()
    n = a.n_obs
    a1, a2 = a[: n // 2].copy(), a[n // 2 :].copy()
    a1.uns["table"] = pd.DataFrame({"x": [1, 2, 3]})
    a2.uns["table"] = pd.DataFrame({"x": [1, 2, 4]})
    p1, p2 = tmp_path / "u1.shad", tmp_path / "u2.shad"
    write_cell_archive(a1, p1, group_by=None)
    write_cell_archive(a2, p2, group_by=None)
    with pytest.raises(ValueError, match="uns"):
        concat_cell_archives([p1, p2], tmp_path / "out.shad")
