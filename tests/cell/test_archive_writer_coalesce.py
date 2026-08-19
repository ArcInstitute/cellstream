"""#233 CellArchiveWriter encode coalescing — byte-identity across granularities + edge cases."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.cell import CellArchiveWriter, open_cell
from cellstream.cell.writer import write_cell_archive
from cellstream.errors import LossyCastError

from .conftest import assert_archives_equivalent, make_cell_adata


def _feed(writer, X, step):
    for lo in range(0, X.shape[0], step):
        writer.add_block(sp.csr_matrix(X[lo : lo + step]))


@pytest.mark.parametrize(
    "codec,value_dtype,data_dtype",
    [("pfordelta", "uint16", None), ("zstd", "float32", "float32")],
)
def test_coalesce_byte_identity_across_granularities(tmp_path, codec, value_dtype, data_dtype):
    """Same rows/order encoded three ways must be byte-identical: per-row (cap=1, single-thread),
    coalesced (small cap -> multiple full-thread flushes), and bulk write_cell_archive."""
    a = make_cell_adata(n_obs=240, n_vars=60)
    if data_dtype == "float32":
        a.X = a.X.astype(np.float32)

    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by=None, codec=codec, data_dtype=data_dtype)

    per_row = tmp_path / "perrow.shad"
    w = CellArchiveWriter(
        per_row,
        n_vars=a.n_vars,
        codec=codec,
        value_dtype=value_dtype,
        data_dtype=data_dtype,
        encode_block_rows=1,
        encode_block_nnz=0,
    )
    _feed(w, a.X, step=1)
    w.finalize(obs=a.obs, var=a.var)

    coalesced = tmp_path / "coalesced.shad"
    w = CellArchiveWriter(
        coalesced,
        n_vars=a.n_vars,
        codec=codec,
        value_dtype=value_dtype,
        data_dtype=data_dtype,
        encode_block_rows=50,
        encode_block_nnz=0,
    )
    _feed(w, a.X, step=4)  # 4-row add_blocks coalesced into ~50-row flushes
    w.finalize(obs=a.obs, var=a.var)

    assert_archives_equivalent(ref, per_row)
    assert_archives_equivalent(ref, coalesced)


def test_coalesce_lossy_error_deferred_to_finalize(tmp_path):
    """Under default caps, a lossy block is buffered silently by add_block; the LossyCastError
    surfaces at the finalize tail-flush (fail-loud preserved, timing deferred — #233 S5a)."""
    from cellstream.errors import LossyCastError

    a = make_cell_adata(n_obs=120, n_vars=60)
    x = a.X.tocsr().astype(np.float64)
    x.data[0] = np.float64(1) / np.float64(3)  # not float32-exact
    w = CellArchiveWriter(tmp_path / "o.shad", n_vars=a.n_vars, codec="zstd", value_dtype="float32")
    w.add_block(sp.csr_matrix(x[:100]))  # buffered, no raise (default cap 65_536)
    with pytest.raises(LossyCastError):
        w.finalize(obs=a.obs.iloc[:100], var=a.var)


def test_coalesce_buffer_bounded_by_caps(tmp_path):
    """Pending buffer never exceeds the caps between add_block calls; a cap hit flushes to empty."""
    a = make_cell_adata(n_obs=240, n_vars=60)
    w = CellArchiveWriter(
        tmp_path / "o.shad",
        n_vars=a.n_vars,
        codec="pfordelta",
        value_dtype="uint16",
        encode_block_rows=50,
        encode_block_nnz=0,
    )
    for lo in range(0, a.n_obs, 4):
        w.add_block(sp.csr_matrix(a.X[lo : lo + 4]))
        assert w._pending_rows < 50, "buffer exceeded encode_block_rows"
        assert len(w._pending) * 4 <= 52
    w.finalize(obs=a.obs, var=a.var)
    assert w._pending == [] and w._pending_rows == 0


def test_coalesce_single_block_over_cap_flushes_directly(tmp_path):
    """A single add_block larger than the cap is one buffer entry, flushed immediately."""
    a = make_cell_adata(n_obs=240, n_vars=60)
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by=None, codec="pfordelta")
    got = tmp_path / "got.shad"
    w = CellArchiveWriter(
        got,
        n_vars=a.n_vars,
        codec="pfordelta",
        value_dtype="uint16",
        encode_block_rows=50,
        encode_block_nnz=0,
    )
    w.add_block(sp.csr_matrix(a.X))  # 240 rows >> cap 50 -> flush within this call
    assert w._pending == []
    w.finalize(obs=a.obs, var=a.var)
    assert_archives_equivalent(ref, got)


def test_coalesce_empty_writer(tmp_path):
    """No add_block at all -> finalize on an empty writer produces a valid 0-row archive."""
    a = make_cell_adata()
    got = tmp_path / "got.shad"
    w = CellArchiveWriter(got, n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16")
    w.finalize(obs=a.obs.iloc[:0], var=a.var)
    s = open_cell(got)
    try:
        assert s.n_obs == 0
    finally:
        s.close()


def test_coalesce_abort_mid_buffer_leaves_no_shad(tmp_path):
    """abort() with rows still buffered drops the buffer and writes no .shad."""
    a = make_cell_adata(n_obs=240, n_vars=60)
    out = tmp_path / "o.shad"
    w = CellArchiveWriter(out, n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16")
    w.add_block(sp.csr_matrix(a.X[:100]))  # buffered (default cap not reached)
    assert w._pending_rows == 100
    w.abort()
    assert w._pending == [] and not out.exists()


def test_reorder_finalize_identity_staging_size_mismatch_fails_loud(tmp_path):
    """#233 (Copilot #230): the identity fast-path in _reorder_and_finalize must fail loud when
    in_offsets[-1] != staging file size -- the validation reorder_frames does on the reorder path,
    which the identity branch skips. Guards a truncated/corrupt payload from silently producing a
    broken archive."""
    from cellstream.cell.writer import _reorder_and_finalize

    a = make_cell_adata()
    tmpd = tmp_path / "wt"
    tmpd.mkdir()
    staging = tmpd / "x_payload_staging"
    staging.write_bytes(b"\x00" * 10)  # 10 bytes on disk
    with pytest.raises(ValueError, match="staging file size"):
        _reorder_and_finalize(
            tmpd,
            staging,
            np.array([0, 5], dtype=np.int64),  # in_offsets[-1]=5 != 10 -> identity-path mismatch
            np.array([3], dtype=np.int64),  # row_nnz for 1 row -> identity perm
            obs=a.obs.iloc[:1],
            var=a.var,
            uns={},
            group_by=None,
            reference=None,
            sort_within_group=None,
            sort_order=None,
            codec_meta={"codec": "pfordelta", "schema_version": 5},
            n_vars=a.n_vars,
            out_path=tmp_path / "out.shad",
            overwrite=False,
            payload_checksum=False,
        )


def test_validate_in_offsets_direct():
    """Direct unit coverage of the shared canonical in_offsets guards (#234)."""
    from cellstream.cell.writer import _validate_in_offsets

    _validate_in_offsets(np.array([0, 3, 7], dtype=np.int64), 7)  # valid: no raise
    _validate_in_offsets(np.array([0, 3, 7], dtype=np.int64), 7, n_obs=2)  # valid with n_obs
    with pytest.raises(ValueError, match=r"in_offsets\[0\]"):
        _validate_in_offsets(np.array([1, 4], dtype=np.int64), 4)
    with pytest.raises(ValueError, match="non-decreasing"):
        _validate_in_offsets(np.array([0, 8, 5], dtype=np.int64), 5)
    with pytest.raises(ValueError, match="staging file size"):
        _validate_in_offsets(np.array([0, 5], dtype=np.int64), 10)
    with pytest.raises(ValueError, match=r"in_offsets length"):
        _validate_in_offsets(np.array([0, 5], dtype=np.int64), 5, n_obs=2)
    # int64-wrap adversarial input: np.diff would report a bogus +step and ACCEPT this; adjacent
    # comparison rejects it. (Gemini #217 bug class.)
    with pytest.raises(ValueError, match="non-decreasing"):
        _validate_in_offsets(np.array([0, 2**63 - 1, -2], dtype=np.int64), -2)
    # non-1D input is untrusted-structural: reject with a clean ValueError, not a leaked
    # IndexError/TypeError from int(in_offsets[0]). (Gemini #238.)
    with pytest.raises(ValueError, match="1D array"):
        _validate_in_offsets(np.asarray(5, dtype=np.int64), 0)  # 0D
    with pytest.raises(ValueError, match="1D array"):
        _validate_in_offsets(np.array([[0, 3], [3, 7]], dtype=np.int64), 7)  # 2D


def _identity_call(tmp_path, *, in_offsets, row_nnz, n_obs, staging_bytes):
    """Drive _reorder_and_finalize's identity fast-path with a corrupt offsets sidecar (#234)."""
    from cellstream.cell.writer import _reorder_and_finalize

    a = make_cell_adata()
    tmpd = tmp_path / "wt"
    tmpd.mkdir()
    staging = tmpd / "x_payload_staging"
    staging.write_bytes(b"\x00" * staging_bytes)
    _reorder_and_finalize(
        tmpd,
        staging,
        np.asarray(in_offsets, dtype=np.int64),
        np.asarray(row_nnz, dtype=np.int64),
        obs=a.obs.iloc[:n_obs],
        var=a.var,
        uns={},
        group_by=None,
        reference=None,
        sort_within_group=None,
        sort_order=None,
        codec_meta={"codec": "pfordelta", "schema_version": 5},
        n_vars=a.n_vars,
        out_path=tmp_path / "out.shad",
        overwrite=False,
        payload_checksum=False,
    )


def test_identity_rejects_nonzero_first_offset(tmp_path):
    with pytest.raises(ValueError, match=r"in_offsets\[0\]"):
        _identity_call(tmp_path, in_offsets=[1, 5], row_nnz=[3], n_obs=1, staging_bytes=5)


def test_identity_rejects_non_monotonic_offsets(tmp_path):
    with pytest.raises(ValueError, match="non-decreasing"):
        _identity_call(tmp_path, in_offsets=[0, 8, 5], row_nnz=[3, 2], n_obs=2, staging_bytes=5)


def test_identity_rejects_wrong_length_offsets(tmp_path):
    # [0]==0 ok, non-decreasing ok, staging size (5==5) ok, but size 2 != n_obs+1 (3).
    with pytest.raises(ValueError, match=r"in_offsets length"):
        _identity_call(tmp_path, in_offsets=[0, 5], row_nnz=[3, 2], n_obs=2, staging_bytes=5)


def test_coalesced_flush_error_names_buffered_rows(tmp_path):
    """@nick-youngblut #230: with coalescing on, a validation rejection surfaces at flush, not at
    the add_block that supplied the bad rows. The error must name the buffered add-order row span so
    the caller can locate the offending input. A negative count is rejected by BOTH encode engines
    (NonIntegerCountsError) and survives vstack/sort unchanged, so it is a deterministic trigger."""
    from cellstream.errors import NonIntegerCountsError

    out = tmp_path / "o.shad"
    w = CellArchiveWriter(out, n_vars=3, codec="pfordelta", value_dtype="uint16")
    clean = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int64))  # rows 0..1
    bad = sp.csr_matrix(np.array([[0, -5, 0]], dtype=np.int64))  # row 2: negative -> rejected
    w.add_block(clean)
    w.add_block(bad)
    assert w._pending_rows == 3 and len(w._pending) == 2  # both buffered (default cap not reached)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(3)])
    var = pd.DataFrame(index=["g0", "g1", "g2"])
    with pytest.raises(NonIntegerCountsError) as ei:
        w.finalize(obs=obs, var=var)
    notes = getattr(ei.value, "__notes__", [])
    assert any("[0, 3)" in n for n in notes), notes  # add-order row span of the flushed batch
    assert any("2 coalesced" in n for n in notes), notes  # buffered-block count


@pytest.mark.parametrize(
    "exc_type,msg,clause2_expected",
    [
        (OSError, "No space left on device", False),  # ENOSPC -- environment, not data
        (MemoryError, "out of memory during encode", False),
        (
            ValueError,
            "encode_cells: duplicate column index 1 within a cell",
            True,
        ),  # plain ValueError
        # LossyCastError is a CellstreamError, NOT a ValueError -- guard its explicit arm in the gate:
        (LossyCastError, "float64 -> float32 would lose information", True),
    ],
)
def test_flush_note_data_attribution_only_for_data_errors(
    tmp_path, monkeypatch, exc_type, msg, clause2_expected
):
    """#241 (@nick-youngblut): the flush note's positional row-span clause is attached to any
    error, but the 'offending value may be in any of those blocks' clause only to data errors
    (LossyCastError / ValueError), not to OSError (ENOSPC) / MemoryError."""
    import cellstream.cell._rust_dispatch as rd
    from cellstream.cell import archive_writer as aw

    a = make_cell_adata(n_obs=120, n_vars=60)
    w = CellArchiveWriter(
        tmp_path / "o.shad", n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16"
    )
    monkeypatch.setattr(rd, "use_rust_for_cell_encode", lambda *args, **kw: False)

    def boom(*args, **kwargs):
        raise exc_type(msg)

    monkeypatch.setattr(aw, "_stream_encode_block_python", boom)

    w.add_block(sp.csr_matrix(a.X[:100]))  # buffered (default cap not reached)
    with pytest.raises(exc_type) as ei:
        w.finalize(obs=a.obs.iloc[:100], var=a.var)  # tail-flush -> _flush -> boom

    notes = getattr(ei.value, "__notes__", [])
    assert any("add-order rows [0, 100)" in n for n in notes), notes  # clause 1: always present
    has_clause2 = any("offending value may be in" in n for n in notes)
    assert has_clause2 is clause2_expected, (msg, notes)
