"""A rejected block must not poison a CellArchiveWriter (#282 item 10).

Before the fix the pure-Python arm appended the already-encoded rows of a rejected
block and never recorded their lengths, so every later flush still SUCCEEDED and the
whole write died at finalize() with
'reorder_frames: in_offsets[-1]=200 != staging file size 220'.
The Rust arm never did this; the fix is that both arms now agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cellstream
from cellstream.cell.archive_writer import CellArchiveWriter

N_VARS = 20


def _clean(n_rows, seed):
    rng = np.random.default_rng(seed)
    m = sp.random(n_rows, N_VARS, density=0.3, format="csr", random_state=rng)
    m.data = np.ceil(m.data * 10).astype(np.int32)
    return m


def _dup_block():
    """Row 0 clean, row 1 carries a duplicate column index (2 appears twice)."""
    indptr = np.array([0, 3, 7], dtype=np.int32)
    indices = np.array([0, 1, 5, 0, 2, 2, 9], dtype=np.int32)
    data = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    return sp.csr_matrix((data, indices, indptr), shape=(2, N_VARS))


def _build(out):
    """The reproduction sequence; returns the decoded (X, obs_names) after finalize."""
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32", encode_block_rows=1)

    w.add_block(_clean(4, 0))
    before = (
        w._staging.stat().st_size,
        len(w._frame_lens),
        len(w._row_nnz),
        w._total_nnz,
        w._max_value,
        w._n_obs,
    )

    with pytest.raises(ValueError, match="duplicate column index"):
        w.add_block(_dup_block())

    after = (
        w._staging.stat().st_size,
        len(w._frame_lens),
        len(w._row_nnz),
        w._total_nnz,
        w._max_value,
        w._n_obs,
    )
    assert after == before, f"the rejected block changed writer state: {before} -> {after}"

    w.add_block(_clean(4, 1))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(w._n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)])
    w.finalize(obs=obs, var=var)

    st = cellstream.open(out)
    try:
        x = st.gather_rows(np.arange(st.n_obs, dtype=np.int64))
        return x.toarray(), list(st.obs.index)
    finally:
        st.close()


@pytest.mark.parametrize("engine_env", [None, "1"], indirect=True)
def test_rejected_block_does_not_poison_the_writer(tmp_path, engine_env):
    dense, names = _build(tmp_path / "push.shad")
    assert dense.shape == (8, N_VARS)
    assert names == [f"c{i}" for i in range(8)]
    # the accepted blocks' VALUES, not just the shape
    expected = np.vstack([_clean(4, 0).toarray(), _clean(4, 1).toarray()])
    np.testing.assert_array_equal(dense, expected)


def test_both_engines_produce_the_same_archive(tmp_path, monkeypatch):
    """Parity is the point of the fix, so assert the two arms decode identically.

    The skip is load-bearing: clearing CELLSTREAM_DISABLE_RUST only selects Rust when the
    extension is COMPLETE. Without it, on a checkout with no (or a partial) extension
    the first _build silently runs Python too and the test compares Python against
    Python -- green, and testing nothing.
    """
    from cellstream.cell._rust_dispatch import rust_available

    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    if not rust_available():
        pytest.skip("cross-engine parity needs a complete Rust extension")
    rust_dense, rust_names = _build(tmp_path / "rust.shad")

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    assert not rust_available()
    py_dense, py_names = _build(tmp_path / "py.shad")

    np.testing.assert_array_equal(rust_dense, py_dense)
    assert rust_names == py_names


def test_rollback_survives_a_baseexception(tmp_path, monkeypatch):
    """Every naturally-occurring rejection here is a ValueError, so `except Exception`
    would leave the whole file green. Inject a KeyboardInterrupt from the row encoder
    AFTER it has already written a frame, which is the case that strands bytes."""
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")  # the arm that writes per-row
    from cellstream.cell import writer as cell_writer

    out = tmp_path / "bx.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32", encode_block_rows=1)
    try:
        w.add_block(_clean(4, 0))
        before = (w._staging.stat().st_size, len(w._frame_lens), w._total_nnz, w._n_obs)

        real = cell_writer._encode_row_python
        calls = {"n": 0}

        def boom(*a, **k):
            calls["n"] += 1
            if calls["n"] == 2:  # row 0 already written, row 1 interrupts
                raise KeyboardInterrupt("ctrl-c mid-encode")
            return real(*a, **k)

        monkeypatch.setattr(cell_writer, "_encode_row_python", boom)
        with pytest.raises(KeyboardInterrupt):
            w.add_block(_clean(4, 1))
        monkeypatch.undo()

        after = (w._staging.stat().st_size, len(w._frame_lens), w._total_nnz, w._n_obs)
        assert after == before, f"KeyboardInterrupt left state changed: {before} -> {after}"
    finally:
        if not w._finalized:
            w.abort()


def test_unrollbackable_flush_aborts_cleanly(tmp_path, monkeypatch):
    """If the truncate itself fails the writer is genuinely unusable -- it must say so
    immediately AND still release both the staging tmpdir and the archive lock, which
    is exactly what abort() owes.

    The injection has to be SELECTIVE. Removing the staging dir does not work: `mark =
    self._staging.stat()` runs before the try (it must, to be in scope in the handler),
    so the flush would die on FileNotFoundError without reaching the rollback at all.
    Making the file read-only does not work either: the Python arm's
    `open(staging, "ab")` would fail first and surface PermissionError instead of the
    duplicate-index ValueError. So: shadow `open` in the archive_writer module
    namespace (module globals win over builtins) and fail only on "r+b".
    """
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    import builtins

    from cellstream.cell import archive_writer as aw
    from cellstream.lock import acquire_write_lock

    out = tmp_path / "ur.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32", encode_block_rows=1)
    w.add_block(_clean(4, 0))
    tmpd = w._tmpd

    real_open = builtins.open

    def selective_open(file, mode="r", *a, **k):
        if mode == "r+b" and str(file) == str(w._staging):
            raise OSError("injected: cannot reopen staging to truncate")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(aw, "open", selective_open, raising=False)

    with pytest.raises(ValueError) as ei:
        w.add_block(_dup_block())
    monkeypatch.undo()
    assert "no longer usable" in "\n".join(getattr(ei.value, "__notes__", []))
    assert w._finalized is True
    assert not tmpd.exists()
    with acquire_write_lock(out.with_suffix(out.suffix + ".lock"), op="write"):
        pass  # the lock was released, not stranded
    with pytest.raises(RuntimeError, match="already finalized or aborted"):
        w.add_block(_clean(1, 9))


def test_unrollbackable_flush_survives_a_baseexception_from_the_truncate(tmp_path, monkeypatch):
    """The rollback handler is the LAST thing standing between a failed flush and a
    flock stranded for the process's lifetime, so it must not itself be skippable.

    `except OSError` around the truncate left exactly one hole: a KeyboardInterrupt or
    MemoryError raised by the reopen bypasses abort(), and the writer keeps both the
    staging tmpd and the lock. Same selective-`open` shadow as the OSError test, but
    raising a BaseException that `except OSError` cannot see. (codex, checkpoint 2.)
    """
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    import builtins

    from cellstream.cell import archive_writer as aw
    from cellstream.lock import acquire_write_lock

    out = tmp_path / "urbx.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32", encode_block_rows=1)
    w.add_block(_clean(4, 0))
    tmpd = w._tmpd

    real_open = builtins.open

    def selective_open(file, mode="r", *a, **k):
        if mode == "r+b" and str(file) == str(w._staging):
            raise KeyboardInterrupt("ctrl-c during the rollback")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(aw, "open", selective_open, raising=False)

    # The interrupt WINS over the duplicate-index ValueError -- abort() has already run,
    # so swallowing a Ctrl-C into a note on someone else's error would be the wrong trade.
    with pytest.raises(KeyboardInterrupt):
        w.add_block(_dup_block())
    monkeypatch.undo()

    assert w._finalized is True
    assert not tmpd.exists()
    with acquire_write_lock(out.with_suffix(out.suffix + ".lock"), op="write"):
        pass  # the lock was released, not stranded


def test_an_interrupt_in_the_scratch_sweep_cannot_skip_the_truncate(tmp_path, monkeypatch):
    """Ordering test: the best-effort scratch sweep must run AFTER the truncate.

    With the sweep first, an interrupt inside it skips the truncate entirely -- the
    bookkeeping is restored while the already-appended staging bytes stay, which is
    precisely the inconsistent writer #282 item 10 is about. The interrupt is allowed to
    win over the original error; it just must not win BEFORE the file is consistent.
    (codex, checkpoint 2.)
    """
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")  # the arm that appends per row
    from pathlib import Path as _Path

    out = tmp_path / "sweep.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32", encode_block_rows=1)
    try:
        w.add_block(_clean(4, 0))
        before = w._staging.stat().st_size

        real_glob = _Path.glob

        def boom(self, pattern, *a, **k):
            if pattern.endswith(".part*"):
                raise KeyboardInterrupt("ctrl-c during the scratch sweep")
            return real_glob(self, pattern, *a, **k)

        monkeypatch.setattr(_Path, "glob", boom)
        with pytest.raises(KeyboardInterrupt):
            w.add_block(_dup_block())  # row 0 encodes and appends, row 1 is rejected
        monkeypatch.undo()

        assert w._staging.stat().st_size == before, (
            "the interrupt in the sweep skipped the truncate -- staging was left inconsistent"
        )
    finally:
        if not w._finalized:
            w.abort()


def _dup_rows(n_rows):
    """Every row carries a duplicate index, so the RUST encoder is what rejects it.

    Column bounds are legal, so `_validate_col_indices` passes and the block reaches
    encode_cells_block -- which is the arm that spawns per-worker `.partN` files.
    """
    indptr = np.arange(0, 3 * (n_rows + 1), 3, dtype=np.int32)
    indices = np.tile(np.array([0, 2, 2], dtype=np.int32), n_rows)
    data = np.ones(3 * n_rows, dtype=np.int32)
    return sp.csr_matrix((data, indices, indptr), shape=(n_rows, N_VARS))


def test_rejected_rust_flush_leaves_no_worker_scratch(tmp_path, monkeypatch):
    """The Rust arm's per-worker scratch is part of the staging state the rollback owes.

    encode_cells_block writes `<staging>.partN` per worker and removes them only on its
    successful concat path, so a rejected Rust flush used to leave one coalesced block's
    worth of scratch sitting in the staging dir until finalize()/abort(). Verified before
    the fix: an 8-thread rejection left .part0..part7. (codex, checkpoint 2.)
    """
    from cellstream.cell._rust_dispatch import rust_available

    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    if not rust_available():
        pytest.skip("the .partN scratch only exists on the Rust arm")

    out = tmp_path / "parts.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32", encode_block_rows=1, threads=8)
    try:
        w.add_block(_clean(64, 0))
        assert not list(w._tmpd.glob("*.part*")), "clean flush already left scratch behind"

        with pytest.raises(ValueError, match="duplicate column index"):
            w.add_block(_dup_rows(64))

        stale = sorted(p.name for p in w._tmpd.glob("*.part*"))
        assert stale == [], f"the rejected Rust flush left worker scratch: {stale}"

        # ...and the writer still works, i.e. the sweep did not take anything it needed.
        w.add_block(_clean(4, 1))
        obs = pd.DataFrame(index=[f"c{i}" for i in range(w._n_obs)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)])
        w.finalize(obs=obs, var=var)
    finally:
        if not w._finalized:
            w.abort()

    st = cellstream.open(out)
    try:
        dense = st.gather_rows(np.arange(st.n_obs, dtype=np.int64)).toarray()
    finally:
        st.close()
    np.testing.assert_array_equal(
        dense, np.vstack([_clean(64, 0).toarray(), _clean(4, 1).toarray()])
    )


class _BoomOnAppend(list):
    """A list whose append() raises -- the injection point for a post-encode failure."""

    def append(self, item):
        raise MemoryError("injected: cannot record row nnz")


def test_failure_during_the_bookkeeping_rolls_the_staging_back(tmp_path, monkeypatch):
    """The bookkeeping lives INSIDE the try, and that half is what the other tests miss.

    Every naturally-occurring rejection here fails during the encode, before any
    bookkeeping runs -- so moving the five appends/increments back outside the try
    would leave the whole file green. This injects a failure AFTER a successful encode
    (which has already appended frames to staging) and at the second bookkeeping
    statement, so `_frame_lens` carries an entry that must be undone alongside the
    staging bytes. (codex, checkpoint 2.)
    """
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    out = tmp_path / "bk.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32", encode_block_rows=1)
    try:
        w.add_block(_clean(4, 0))
        before = (
            w._staging.stat().st_size,
            len(w._frame_lens),
            len(w._row_nnz),
            w._total_nnz,
            w._max_value,
            w._n_obs,
        )
        assert before[0] > 0, "the first block must have written frames, or this proves nothing"

        w._row_nnz = _BoomOnAppend(w._row_nnz)
        with pytest.raises(MemoryError):
            w.add_block(_clean(4, 1))
        w._row_nnz = list(w._row_nnz)

        after = (
            w._staging.stat().st_size,
            len(w._frame_lens),
            len(w._row_nnz),
            w._total_nnz,
            w._max_value,
            w._n_obs,
        )
        assert after == before, f"the bookkeeping failure was not rolled back: {before} -> {after}"

        # ...and the writer is genuinely still usable, which is the whole point of #282 item 10.
        w.add_block(_clean(4, 2))
        obs = pd.DataFrame(index=[f"c{i}" for i in range(w._n_obs)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)])
        w.finalize(obs=obs, var=var)
    finally:
        if not w._finalized:
            w.abort()

    st = cellstream.open(out)
    try:
        dense = st.gather_rows(np.arange(st.n_obs, dtype=np.int64)).toarray()
    finally:
        st.close()
    expected = np.vstack([_clean(4, 0).toarray(), _clean(4, 2).toarray()])
    np.testing.assert_array_equal(dense, expected)


@pytest.mark.parametrize("engine_env", [None, "1"], indirect=True)
def test_rejected_block_keeps_the_existing_row_span_note(tmp_path, engine_env):
    """#230's add-order attribution note must still attach after the rollback."""
    out = tmp_path / "push2.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32", encode_block_rows=1)
    w.add_block(_clean(4, 0))
    with pytest.raises(ValueError) as ei:
        w.add_block(_dup_block())
    notes = "\n".join(getattr(ei.value, "__notes__", []))
    assert "add-order rows [4, 6)" in notes
    w.abort()
