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


def test_lock_file_unlinked_after_release(tmp_path):
    """The sidecar is removed on release: a finished write leaves nothing beside its output (#2)."""
    from cellstream.lock import acquire_write_lock

    lock = tmp_path / "out.shad.lock"
    with acquire_write_lock(lock, op="write"):
        # Still present (and readable) while held -- the diagnostic body must not disappear early.
        assert lock.exists()
    assert not lock.exists()
    assert list(tmp_path.iterdir()) == []


def test_lock_file_unlinked_when_the_body_raises(tmp_path):
    """The unlink is in a finally, so an exception in the guarded block must not strand it."""
    from cellstream.lock import acquire_write_lock

    lock = tmp_path / "out.shad.lock"
    with pytest.raises(RuntimeError):
        with acquire_write_lock(lock, op="write"):
            raise RuntimeError("caller blew up")
    assert not lock.exists()


def test_unsafe_no_lock_creates_no_file(tmp_path):
    from cellstream.lock import acquire_write_lock

    lock = tmp_path / "out.shad.lock"
    with acquire_write_lock(lock, op="write", unsafe_no_lock=True):
        assert not lock.exists()
    assert not lock.exists()


def test_acquire_retries_when_the_locked_inode_is_already_dead(tmp_path, monkeypatch):
    """The unlink-vs-flock race, forced deterministically.

    A process that opened the lock file just before the previous holder unlinked it gets a
    descriptor to an inode no path names any more. ``flock`` on that dead inode SUCCEEDS (nobody
    else can even reach it), so only the post-flock inode check can notice -- without it, two
    writers would each "hold" a different inode and both proceed. Here the first ``open`` returns
    exactly that dead descriptor.
    """
    from cellstream import lock as lock_mod

    lock = tmp_path / "out.shad.lock"
    dead = tmp_path / "dead.lock"
    real_open = os.open
    calls = []

    def fake_open(path, flags, mode=0o777):
        calls.append(path)
        if len(calls) == 1:
            fd = real_open(dead, flags, mode)
            os.unlink(dead)  # live descriptor, no longer reachable through any path
            return fd
        return real_open(path, flags, mode)

    monkeypatch.setattr(lock_mod.os, "open", fake_open)
    with lock_mod.acquire_write_lock(lock, op="write") as _:
        assert len(calls) == 2
        assert lock.exists()
    assert not lock.exists()


def test_held_inode_is_current_tracks_replacement(tmp_path):
    """The guard itself, without the surrounding acquire: an inode is only 'current' while it is
    the file the path resolves to."""
    from cellstream.lock import _held_inode_is_current

    lock = tmp_path / "out.shad.lock"
    lock.write_text("held by nobody")
    fd = os.open(lock, os.O_RDWR)
    try:
        assert _held_inode_is_current(fd, lock)
        lock.unlink()
        assert not _held_inode_is_current(fd, lock)  # path gone
        lock.write_text("a different inode entirely")
        assert not _held_inode_is_current(fd, lock)  # path reused
    finally:
        os.close(fd)


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
