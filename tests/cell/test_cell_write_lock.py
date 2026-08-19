"""Concurrent layout='cell' writes to one output must fail loudly (#280 item 5).

Without the lock, two writers share one <dst>.tmp inode: the loser dies on a bare
FileNotFoundError naming an internal staging file, and in one observed race the
WINNER returned success while publishing the other writer's data.

flock is per OPEN FILE DESCRIPTION, not per process -- two separate open() calls in
one process DO contend. That is why these tests can hold the lock through
acquire_write_lock on their own fd and get exactly what a second process sees.
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
from cellstream.cell.writer import write_cell_archive
from cellstream.errors import ArchiveLockedError
from cellstream.lock import acquire_write_lock

N_VARS = 16


def _adata(n_obs=24, seed=5):
    rng = np.random.default_rng(seed)
    x = sp.csr_matrix(rng.integers(0, 6, size=(n_obs, N_VARS)).astype(np.int32))
    return ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)]),
    )


def _obs_var(n_obs):
    return (
        pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)]),
    )


def _held(out):
    return acquire_write_lock(out.with_suffix(out.suffix + ".lock"), op="write")


# --- contention -------------------------------------------------------------


def test_write_cell_archive_blocks_on_held_lock(tmp_path):
    out = tmp_path / "a.shad"
    with _held(out):
        with pytest.raises(ArchiveLockedError):
            write_cell_archive(_adata(), out)
    assert not out.exists()


def test_cell_archive_writer_blocks_on_held_lock(tmp_path):
    out = tmp_path / "b.shad"
    with _held(out):
        with pytest.raises(ArchiveLockedError):
            CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32")


def test_concat_blocks_on_held_lock(tmp_path):
    p1, p2 = tmp_path / "p1.shad", tmp_path / "p2.shad"
    write_cell_archive(_adata(seed=1), p1)
    write_cell_archive(_adata(seed=2), p2)
    out = tmp_path / "m.shad"
    with _held(out):
        with pytest.raises(ArchiveLockedError):
            concat_cell_archives([p1, p2], out)


def test_write_sharded_cell_honours_the_lock_by_default(tmp_path):
    out = tmp_path / "d.shad"
    with _held(out):
        with pytest.raises(ArchiveLockedError):
            cellstream.write_sharded(_adata(), out, layout="cell")


def test_existing_archive_still_reports_exists_not_locked(tmp_path):
    """Ordering matches write_packed: the existence check precedes the acquire, so a
    plainly-existing output keeps its ArchiveExistsError diagnostic."""
    out = tmp_path / "x.shad"
    write_cell_archive(_adata(), out)
    with pytest.raises(cellstream.errors.ArchiveExistsError):
        write_cell_archive(_adata(), out)


def test_existing_archive_reports_exists_even_while_the_lock_is_held(tmp_path):
    """The DISCRIMINATING form of the ordering check.

    The test above passes under BOTH orders -- a lock-first writer would acquire the free
    lock and then raise ArchiveExistsError just the same. Holding the lock while the output
    also exists is what separates them: existence-first raises ArchiveExistsError,
    lock-first raises ArchiveLockedError. (codex, checkpoint 2.)
    """
    part = tmp_path / "xpart.shad"
    write_cell_archive(_adata(seed=7), part)
    out = tmp_path / "xh.shad"
    write_cell_archive(_adata(), out)
    with _held(out):
        with pytest.raises(cellstream.errors.ArchiveExistsError):
            write_cell_archive(_adata(), out)
        with pytest.raises(cellstream.errors.ArchiveExistsError):
            CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32")
        with pytest.raises(cellstream.errors.ArchiveExistsError):
            concat_cell_archives([part], out)


# --- the escape hatch, on all three entry points -----------------------------


def test_write_cell_archive_unsafe_no_lock_bypasses(tmp_path):
    out = tmp_path / "u1.shad"
    with _held(out):
        write_cell_archive(_adata(), out, unsafe_no_lock=True)
    assert out.exists()


def test_cell_archive_writer_unsafe_no_lock_bypasses(tmp_path):
    out = tmp_path / "u2.shad"
    with _held(out):
        w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32", unsafe_no_lock=True)
        w.add_block(_adata(n_obs=4).X)
        obs, var = _obs_var(4)
        w.finalize(obs=obs, var=var)
    assert out.exists()


def test_concat_unsafe_no_lock_bypasses(tmp_path):
    p1 = tmp_path / "cp1.shad"
    write_cell_archive(_adata(seed=1), p1)
    out = tmp_path / "u3.shad"
    with _held(out):
        concat_cell_archives([p1], out, unsafe_no_lock=True)
    assert out.exists()


def test_write_sharded_forwards_unsafe_no_lock_to_cell(tmp_path, monkeypatch):
    """Assert the VALUE is forwarded, not merely that the write succeeded: the
    success-only form passes vacuously on the pre-fix baseline, where no cell lock
    exists at all. write_sharded has always accepted this knob and dropped it here
    -- the silently-ignored-knob shape #261 fixed.

    Patch cellstream.cell.writer, NOT cellstream.write: write_sharded does a
    function-local `from cellstream.cell.writer import write_cell_archive`
    (write.py:247), so a module attribute on cellstream.write is never read.
    """
    seen = {}
    real = cell_writer.write_cell_archive

    def spy(adata, out_path, **kw):
        seen.update(kw)
        return real(adata, out_path, **kw)

    monkeypatch.setattr(cell_writer, "write_cell_archive", spy)
    cellstream.write_sharded(_adata(), tmp_path / "f.shad", layout="cell", unsafe_no_lock=True)
    assert seen.get("unsafe_no_lock") is True


# --- lifecycle: released on every exit, held while live ----------------------


def test_lock_released_after_finalize(tmp_path):
    out = tmp_path / "e.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32")
    w.add_block(_adata(n_obs=4).X)
    obs, var = _obs_var(4)
    w.finalize(obs=obs, var=var)
    with _held(out):  # must not raise
        pass


def test_lock_released_after_abort(tmp_path):
    out = tmp_path / "g.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32")
    w.add_block(_adata(n_obs=4).X)
    w.abort()
    with _held(out):
        pass


def test_lock_released_after_exit_without_finalize(tmp_path):
    out = tmp_path / "h.shad"
    with pytest.raises(RuntimeError):
        with CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32") as w:
            w.add_block(_adata(n_obs=4).X)
            raise RuntimeError("caller blew up")
    with _held(out):
        pass


def test_lock_released_when_init_fails_after_acquire(tmp_path):
    """A failure later in __init__ -- here an unresolvable value_dtype -- must not
    strand the lock it already took.

    TypeError, not a blind `Exception` (ruff B017 rejects that): _resolve_push_codec
    resolves value_dtype through np.dtype(), which raises TypeError -- and it runs
    after the acquire, which is what makes this a stranded-lock case at all.

    __new__ + an explicit __init__ so the half-built object stays REFERENCED. The
    obvious `CellArchiveWriter(...)` form is vacuous: the failed instance becomes
    garbage immediately, CPython refcounting finalises the ExitStack's generator-backed
    flock context for us, and the lock is free even with the explicit close() deleted.
    (codex, checkpoint 2.)
    """
    out = tmp_path / "i.shad"
    w = CellArchiveWriter.__new__(CellArchiveWriter)
    with pytest.raises(TypeError):
        w.__init__(out, n_vars=N_VARS, value_dtype="not_a_dtype")
    assert w is not None  # keep the reference alive across the acquire below
    with _held(out):
        pass


def test_lock_released_after_a_failed_finalize(tmp_path):
    """finalize() marks the writer terminal on failure too, so it owes the lock back."""
    out = tmp_path / "ff.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32")
    w.add_block(_adata(n_obs=4).X)
    obs, var = _obs_var(9)  # deliberate row-count mismatch -> finalize raises
    with pytest.raises(ValueError):
        w.finalize(obs=obs, var=var)
    assert w._finalized is True
    with _held(out):
        pass


def test_abort_is_idempotent(tmp_path):
    """A second abort() -- and an __exit__ after one -- must not raise on the released
    ExitStack. contextlib.ExitStack.close() is idempotent; this pins that we rely on it."""
    out = tmp_path / "dab.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32")
    w.add_block(_adata(n_obs=4).X)
    w.abort()
    w.abort()
    with _held(out):
        pass


@pytest.mark.parametrize("method", ["abort", "finalize"])
def test_lock_released_even_if_the_staging_rmtree_is_interrupted(tmp_path, monkeypatch, method):
    """rmtree(ignore_errors=True) swallows OSError but NOT KeyboardInterrupt.

    Without the `finally` around the release, an interrupt landing in the staging removal
    strands the flock for the life of the process -- and the writer is still referenced
    here, so nothing else would free it. (codex, checkpoint 2.)
    """
    from cellstream.cell import archive_writer as aw

    out = tmp_path / f"rm_{method}.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32")
    w.add_block(_adata(n_obs=4).X)

    def boom(*a, **k):
        raise KeyboardInterrupt("ctrl-c during staging cleanup")

    monkeypatch.setattr(aw.shutil, "rmtree", boom)
    with pytest.raises(KeyboardInterrupt):
        if method == "abort":
            w.abort()
        else:
            obs, var = _obs_var(4)
            w.finalize(obs=obs, var=var)
    monkeypatch.undo()

    assert w is not None  # keep the writer referenced across the acquire below
    with _held(out):
        pass


def test_exit_after_finalize_does_not_double_release(tmp_path):
    """The `finally: self._stack.close()` in __exit__ is unconditional, so the ordinary
    `with ... finalize()` path closes the stack twice. It must stay clean."""
    out = tmp_path / "ef.shad"
    with CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32") as w:
        w.add_block(_adata(n_obs=4).X)
        obs, var = _obs_var(4)
        w.finalize(obs=obs, var=var)
    assert out.exists()
    with _held(out):
        pass


def test_lock_held_while_the_push_writer_is_live(tmp_path):
    """The whole point of the __init__..finalize span: a second writer cannot start
    against the same output midway through a long push write."""
    out = tmp_path / "j.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32")
    try:
        w.add_block(_adata(n_obs=4).X)
        with pytest.raises(ArchiveLockedError):
            write_cell_archive(_adata(), out)
    finally:
        w.abort()


def test_failed_write_releases_the_lock(tmp_path):
    """write_cell_archive raising must release the lock, not strand it."""
    out = tmp_path / "k.shad"
    bad = _adata(n_obs=4)
    bad.X = sp.csr_matrix(
        (
            np.array([1, 2], dtype=np.int32),
            np.array([0, N_VARS + 5], dtype=np.int32),  # out of range -> #264 rejects
            np.array([0, 2, 2, 2, 2], dtype=np.int32),
        ),
        shape=(4, N_VARS),
    )
    with pytest.raises(ValueError):
        write_cell_archive(bad, out)
    with _held(out):
        pass
