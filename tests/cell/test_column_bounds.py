"""#250: the pure-Python read paths must reject a column index >= n_vars.

Two fixtures, because the two codecs fail differently:

* pfordelta -- the frame must be BYTE-PATCHED. Feeding column 70000 to CellArchiveWriter at
  n_vars=18533 does NOT produce a bad archive: writer.py's _encode_row_python does
  `idx = idx_raw.astype(idt)` with idt=uint16, so the writer stores 4464 and the archive is
  perfectly valid. So the test writes a normal archive holding 4464 and overwrites that one
  frame with a frame encoding 70000. The two pack to the same length -- 20 bytes on the current
  encoder, but the fixture asserts EQUALITY with the stored frame, never a literal size, since
  the packed length is an encoder/version detail -- so every offset stays valid. A default
  archive carries no payload checksum, so nothing else needs fixing up.

* zstd -- index_dtype_on_disk is always uint32, so the writer's cast preserves 70000. Until
  #264 that meant NO tampering was needed; the write-side guard now rejects it, so the fixture
  neutralizes `_validate_col_indices` to forge the archive. The READ-side assertions below are
  unchanged -- an archive like this can still arrive from an older cellstream or a foreign writer.

n_vars=18533 and column 70000 are load-bearing for pfordelta: 70000 narrows to 4464, which is
< 18533, so the bulk path's check_indices scan genuinely passes pre-fix on the NARROWED array.
A smaller n_vars would put 4464 out of range and the pre-existing scan would catch it.
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cellstream
from cellstream.cell import CellArchiveWriter, open_cell
from cellstream.cell.codec import PforDeltaCodec
from cellstream.cell.reader import read_cell_bulk
from cellstream.cell.writer import _validate_col_indices
from cellstream.packed.reader import PackedArchive
from cellstream.runtime import rust_available

N_VARS = 18533
BAD_COL = 70000
WRAPPED = BAD_COL - 65536  # 4464 -- a legal column of an 18533-gene archive


def _csr_with_column(col, dtype):
    """A 2-row CSR whose row 1 holds `col`, built by direct assignment so no scipy check runs."""
    X = sp.csr_matrix((2, N_VARS), dtype=dtype)
    X.data = np.array([3, 7], dtype=dtype)
    X.indices = np.array([5, col], dtype=np.int32)
    X.indptr = np.array([0, 1, 2], dtype=np.int64)
    return X


def _write(path, codec, value_dtype, dtype, col):
    w = CellArchiveWriter(path, n_vars=N_VARS, codec=codec, value_dtype=value_dtype)
    w.add_block(_csr_with_column(col, dtype))
    w.finalize(
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)]),
    )
    return path


@pytest.fixture
def pfor_archive(tmp_path, monkeypatch):
    """pfordelta .shad whose frame 1 is byte-patched to encode column 70000."""
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")  # force the Python encoder for the build
    path = tmp_path / "pfor_oob.shad"
    _write(path, "pfordelta", "uint16", np.uint16, WRAPPED)

    store = open_cell(path)
    try:
        off1, off2 = int(store.offsets[1]), int(store.offsets[2])
        old_len = off2 - off1
    finally:
        store.close()

    forged = (
        PforDeltaCodec("uint16", "uint16")
        .new_encoder()
        .encode(np.array([BAD_COL], np.uint32), np.array([7], np.uint32))
    )
    assert len(forged) == old_len, (
        f"forged frame is {len(forged)} bytes but the stored frame is {old_len}; patching it "
        "in place would corrupt every following offset. Rebuild the fixture."
    )
    _, base, _ = PackedArchive(path).member_location("x_payload")
    with open(path, "r+b") as fh:
        fh.seek(base + off1)
        fh.write(forged)

    # Prove the archive now really holds 70000 BEFORE asserting anything about the read paths.
    # Copy the bytes out and close BEFORE decoding, so a failed proof cannot leave the mmap
    # exported -- this repo has a history of that masking the real error as a BufferError.
    store = open_cell(path)
    try:
        raw = bytes(store._mm[off1:off2])
    finally:
        store.close()
    di, _ = PforDeltaCodec("uint32", "uint32").new_decoder(2**32 - 1).decode(raw, 1)
    assert int(di[0]) == BAD_COL, f"fixture did not take: stored frame holds {int(di[0])}"
    return path


@pytest.fixture
def zstd_archive(tmp_path, monkeypatch):
    """zstd .shad carrying column 70000.

    Until #264 this needed NO tampering -- index_dtype_on_disk is uint32, so the writer's cast
    preserved 70000 and add_block only checked csr.shape[1]. #264 closed that, so the fixture
    now neutralizes the write-side guard to forge the archive the READ side must still reject
    (such an archive can still arrive from an older cellstream or a foreign writer).

    Patch archive_writer's name, not writer's: archive_writer.py does `from .writer import
    _validate_col_indices`, so it holds its own module-level reference.
    """
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    monkeypatch.setattr(
        "cellstream.cell.archive_writer._validate_col_indices", lambda indices, n_vars: None
    )
    return _write(tmp_path / "zstd_oob.shad", "zstd", "uint32", np.uint32, BAD_COL)


def test_wrapped_column_is_in_range_so_the_narrowed_scan_cannot_catch_it():
    """Pins the premise of #250: the narrowed value is a LEGAL column, which is why
    check_indices on the narrowed array is not a fix."""
    assert WRAPPED == 4464
    assert 0 <= WRAPPED < N_VARS


@pytest.mark.parametrize("fixture_name", ["pfor_archive", "zstd_archive"])
def test_gather_rows_rejects_out_of_range_column_pure_python(fixture_name, request, monkeypatch):
    """Pre-fix, measured: pfordelta returned column 4464 and zstd returned column 70000, both
    with no error."""
    path = request.getfixturevalue(fixture_name)
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    store = open_cell(path)
    try:
        with pytest.raises(ValueError, match=r"out of range \[0, 18533\)"):
            store.gather_rows(np.array([0, 1], dtype=np.int64))
    finally:
        store.close()


@pytest.mark.parametrize("fixture_name", ["pfor_archive", "zstd_archive"])
def test_read_cell_bulk_rejects_out_of_range_column_pure_python(fixture_name, request, monkeypatch):
    """zstd already failed here pre-fix (check_indices sees the un-narrowed 70000); pfordelta
    did not, because check_indices scanned the narrowed 4464."""
    path = request.getfixturevalue(fixture_name)
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    with pytest.raises(ValueError, match=r"out of range \[0, 18533\)"):
        read_cell_bulk(path, n_workers=1)


@pytest.mark.parametrize("fixture_name", ["pfor_archive", "zstd_archive"])
def test_rust_and_python_agree_on_the_same_archive(fixture_name, request, monkeypatch):
    """Engine parity is the property the fix restores: Rust rejected these archives all along."""
    path = request.getfixturevalue(fixture_name)
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    from cellstream.cell._rust_dispatch import rust_available

    if not rust_available():
        pytest.skip("Rust extension not built")
    store = open_cell(path)
    try:
        with pytest.raises(ValueError, match=r"column index 70000 out of range \[0, 18533\)"):
            store.gather_rows(np.array([0, 1], dtype=np.int64))
    finally:
        store.close()


# ------------------------------------------------------------------ #264 (write side)


def _require_rust_for_default_engine(engine_env):
    """`engine_env`'s default variant DEGRADES to the Python encoder when no extension is built
    (it opts into CELLSTREAM_ALLOW_PYTHON_FALLBACK) rather than skipping, so "both engines" would
    silently be Python twice and a Rust pass would be reported that never happened. Skip that
    variant instead. Reading engine_env's value is a deliberate exception to its docstring."""
    if engine_env is None and not rust_available():
        pytest.skip("Rust extension not built: the default-engine variant would re-run Python")


def test_validate_col_indices_accepts_empty_and_the_top_boundary():
    """No false positives: an empty block and column n_vars-1 are both legal."""
    _validate_col_indices(np.array([], dtype=np.int32), N_VARS)
    _validate_col_indices(np.array([0, N_VARS - 1], dtype=np.int32), N_VARS)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32])
def test_validate_col_indices_rejects_the_first_out_of_range_column(dtype):
    with pytest.raises(ValueError, match=rf"out of range \[0, {N_VARS}\) \(max={N_VARS}\)"):
        _validate_col_indices(np.array([0, N_VARS], dtype=dtype), N_VARS)


def test_validate_col_indices_rejects_a_negative_column():
    with pytest.raises(ValueError, match=rf"out of range \[0, {N_VARS}\) \(min=-1\)"):
        _validate_col_indices(np.array([-1, 5], dtype=np.int32), N_VARS)


def test_validate_col_indices_reports_a_wide_unsigned_column_by_max():
    """An unsigned index whose SIGNED reading would be -1 is still an out-of-range MAX. (This
    does not prove the min() scan was skipped -- numpy preserves uint32, so its min is
    4294967295 and the `< 0` branch would decline it either way.)"""
    with pytest.raises(ValueError, match=rf"out of range \[0, {N_VARS}\) \(max=4294967295\)"):
        _validate_col_indices(np.array([4294967295], dtype=np.uint32), N_VARS)


@pytest.mark.parametrize(
    "codec,value_dtype,dtype",
    [("pfordelta", "uint16", np.uint16), ("zstd", "uint32", np.uint32)],
)
def test_cell_archive_writer_rejects_out_of_range_column(
    tmp_path, engine_env, codec, value_dtype, dtype
):
    """Pre-fix the write SUCCEEDED for every codec x engine combination here. What it produced
    differed by codec, which is why both are parametrized:

    * pfordelta -- Rust stored 70000 verbatim (an archive the reader then refused); the Python
      encoder's astype(uint16) narrowed it to 4464, a LEGAL column, and the archive read back
      clean at the wrong gene.
    * zstd -- index_dtype_on_disk is uint32, so BOTH engines stored 70000 and both archives
      were merely unreadable. No silent corruption on this codec.

    `engine_env` runs this under both engines."""
    _require_rust_for_default_engine(engine_env)
    with pytest.raises(ValueError, match=rf"out of range \[0, {N_VARS}\) \(max={BAD_COL}\)"):
        _write(tmp_path / "oob.shad", codec, value_dtype, dtype, BAD_COL)


def test_both_engines_reject_with_the_identical_message(tmp_path, monkeypatch):
    """Engine parity is the property this restores -- same input, same exception TEXT, not just
    the same type.

    pfordelta, NOT zstd: pfordelta is the codec where the two engines genuinely diverged pre-fix
    (Rust stored 70000, the Python encoder narrowed it to the legal column 4464, so the same
    input produced two different archives). zstd stores uint32 indices, so both engines wrote
    the identical bad archive and there was nothing to diverge.

    The Rust check is INSIDE the test, after the delenv -- a @skipif decorator is evaluated at
    IMPORT time, before anything clears the env, so an ambient CELLSTREAM_DISABLE_RUST=1 would skip
    this even on a machine where Rust works fine."""
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    if not rust_available():
        pytest.skip("Rust extension not built: the loop would compare Python against Python")
    msgs = []
    for disable in (None, "1"):
        if disable:
            monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", disable)
        else:
            monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
        with pytest.raises(ValueError) as ei:
            _write(tmp_path / f"parity_{disable}.shad", "pfordelta", "uint16", np.uint16, BAD_COL)
        msgs.append(str(ei.value))
    assert msgs[0] == msgs[1], msgs


def test_the_python_engine_no_longer_wraps_a_wide_column_into_a_legal_one(tmp_path, monkeypatch):
    """The exact silent-corruption case: BAD_COL narrows to WRAPPED, a LEGAL column of an
    18533-gene archive, so nothing downstream could ever have caught it."""
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    with pytest.raises(ValueError, match=rf"out of range \[0, {N_VARS}\) \(max={BAD_COL}\)"):
        _write(tmp_path / "wrap.shad", "pfordelta", "uint16", np.uint16, BAD_COL)


def test_flush_rejection_names_the_add_order_row_span(tmp_path):
    """#230's attribution note must survive: with coalescing the rejection surfaces at a flush,
    not at the add_block() that supplied the bad rows."""
    w = CellArchiveWriter(
        tmp_path / "note.shad",
        n_vars=N_VARS,
        codec="zstd",
        value_dtype="uint32",
        encode_block_rows=2,
    )
    with pytest.raises(ValueError) as ei:
        w.add_block(_csr_with_column(BAD_COL, np.uint32))
    assert any("add-order rows [0, 2)" in n for n in getattr(ei.value, "__notes__", []))


def test_cell_archive_writer_still_accepts_the_top_boundary_column(tmp_path, engine_env):
    """No false positive: column n_vars-1 writes and reads back, on both engines."""
    _require_rust_for_default_engine(engine_env)
    path = _write(tmp_path / "ok.shad", "zstd", "uint32", np.uint32, N_VARS - 1)
    store = open_cell(path)
    try:
        got = store.gather_rows(np.array([0, 1], dtype=np.int64))
    finally:
        store.close()
    assert got[1].indices.tolist() == [N_VARS - 1]


def _adata_with_column(col):
    """AnnData whose row 1 holds `col`, built by direct assignment so no scipy check runs."""
    return ad.AnnData(
        X=_csr_with_column(col, np.int32),
        obs=pd.DataFrame({"g": ["a", "b"]}, index=["c0", "c1"]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)]),
    )


def _write_facade(out, source, **kw):
    """cellstream.write_sharded(layout='cell'), with the experimental-format warning suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cellstream.write_sharded(source, out, layout="cell", overwrite=True, **kw)
    return out


@pytest.mark.parametrize("codec", ["pfordelta", "zstd"])
@pytest.mark.parametrize("source_form", ["in_memory", "h5ad_path"])
def test_write_sharded_cell_rejects_out_of_range_column(tmp_path, engine_env, codec, source_form):
    """The spec's rejection matrix, driven through the PUBLIC facade: codec x engine x source
    form. in_memory takes _encode_cell_payload; an .h5ad path takes _encode_cell_streaming
    (anndata writes the out-of-range CSR without complaint, so a bad file on disk is the
    realistic route)."""
    _require_rust_for_default_engine(engine_env)
    a = _adata_with_column(BAD_COL)
    if source_form == "h5ad_path":
        src = tmp_path / "src.h5ad"
        a.write_h5ad(src)
        a = src
    with pytest.raises(ValueError, match=rf"out of range \[0, {N_VARS}\) \(max={BAD_COL}\)"):
        _write_facade(tmp_path / "f.shad", a, codec=codec)


def test_write_sharded_cell_rejects_out_of_range_column_when_grouped(tmp_path):
    """Grouped streaming takes the reorder_frames tail; the guard must fire before it."""
    src = tmp_path / "grp.h5ad"
    _adata_with_column(BAD_COL).write_h5ad(src)
    with pytest.raises(ValueError, match=rf"out of range \[0, {N_VARS}\) \(max={BAD_COL}\)"):
        _write_facade(tmp_path / "g.shad", src, codec="zstd", group_by="g")


@pytest.mark.parametrize("source_form", ["in_memory", "h5ad_path"])
def test_write_sharded_cell_still_accepts_the_top_boundary_column(tmp_path, source_form):
    """No false positive on either encode arm."""
    a = _adata_with_column(N_VARS - 1)
    if source_form == "h5ad_path":
        src = tmp_path / f"ok_{source_form}.h5ad"
        a.write_h5ad(src)
        a = src
    p = _write_facade(tmp_path / f"ok_{source_form}.shad", a, codec="zstd")
    store = open_cell(p)
    try:
        assert store.gather_rows(np.array([1], dtype=np.int64)).indices.tolist() == [N_VARS - 1]
    finally:
        store.close()
