"""lock.py: fcntl.flock-based exclusive write lock with diagnostic body."""

import multiprocessing as mp
import os

import pytest


def _hold_lock(lock_path, started_evt, release_evt):
    """Helper for cross-process tests."""
    from cellstream.lock import acquire_write_lock

    with acquire_write_lock(lock_path, op="write"):
        started_evt.set()
        release_evt.wait(timeout=10)


def test_acquire_writes_diagnostic_body(tmp_path):
    from cellstream.lock import acquire_write_lock

    lock = tmp_path / ".cellstream.lock"
    with acquire_write_lock(lock, op="write"):
        body = lock.read_text()
    assert "host=" in body
    assert f"pid={os.getpid()}" in body
    assert "op=write" in body
    assert "started=" in body


def test_lock_released_after_context(tmp_path):
    """Second acquire after first context exits must succeed immediately."""
    from cellstream.lock import acquire_write_lock

    lock = tmp_path / ".cellstream.lock"
    with acquire_write_lock(lock, op="write"):
        pass
    with acquire_write_lock(lock, op="write"):
        pass


def test_concurrent_writers_fail_loud(tmp_path):
    from cellstream.errors import ArchiveLockedError
    from cellstream.lock import acquire_write_lock

    lock = tmp_path / ".cellstream.lock"
    started = mp.Event()
    release = mp.Event()
    p = mp.get_context("fork").Process(target=_hold_lock, args=(str(lock), started, release))
    p.start()
    try:
        assert started.wait(timeout=5)
        with pytest.raises(ArchiveLockedError) as exc:
            # op= is arbitrary diagnostic text; "update_obs" is a live one (the PACKED
            # mutators, packed/writer.py:309). It used to say "append", which named a v1
            # directory mutator deleted in #154 part A. Deliberately different from the
            # holder's "write" so the two assertions below cannot pass by accident.
            with acquire_write_lock(lock, op="update_obs"):
                pass
        # Diagnostic body from the holder shows up in the error message.
        assert f"pid={p.pid}" in str(exc.value)
        assert "op=write" in str(exc.value)
    finally:
        release.set()
        p.join(timeout=5)


# test_shared_lock_allows_multiple_holders and test_shared_lock_blocks_exclusive_lock stood
# here. They exercised acquire_write_lock's `shared=` (LOCK_SH) arm, which #154 removed: its
# only production caller was the v1->v2 converter, deleted in part A. Having no live caller is
# the whole justification -- the two tests passed, and the synchronization they asserted was
# genuinely correct (LOCK_SH blocked writers until every reader released).
#
# What was NOT multi-holder-safe, and is the reason not to restore the arm unchanged: the
# diagnostic body write is unconditional, so each holder truncates and rewrites the lock file.
# Harmless for LOCK_EX (one holder by definition); with N shared holders the file names only
# the last, and ArchiveLockedError reads that body to report who holds the lock. Measured, not
# inferred: two nested shared holders leave only the second's `op=` in the file. The converter
# was exposed to it too -- it took the shared lock on the v1 SOURCE, so two conversions from
# one source to different destinations would overlap.
