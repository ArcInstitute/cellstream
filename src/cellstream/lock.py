"""fcntl.flock-based exclusive write lock.

Held by the parent for the duration of a write operation. Workers do
NOT acquire the lock (the parent's lock covers them). On NFSv4 flock
is supported; on older NFS it may silently no-op — documented limitation.

The lock file body contains diagnostic info (host, pid, op, started)
which is read back into ArchiveLockedError when acquisition fails. Because the lock is
always EXCLUSIVE there is exactly one holder writing that body at a time — see the
``shared=`` note on ``acquire_write_lock`` for the multi-holder case that used to exist.

The lock file is a *sidecar*: it is unlinked when the holder releases, so a finished write
leaves nothing beside its output. Unlinking a file that others may be flock-ing is the classic
unlink-vs-flock race — a waiter can hold a descriptor to the unlinked inode and "acquire" a lock
no one else can see, while a third process creates a fresh file at the same path and locks that.
``acquire_write_lock`` therefore unlinks *while still holding* the lock and every acquirer
re-checks, after ``flock`` succeeds, that the inode it locked is still the file at the path;
on a mismatch it retries. See ArcInstitute/cellstream#2.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import os
import socket
from pathlib import Path

from cellstream.errors import ArchiveLockedError

# Retries for the unlink-vs-flock race documented in the module docstring. Each retry means some
# other process replaced the lock file between our open() and our flock(): transient by nature, so
# a small bound is enough to keep a pathological churn from spinning forever.
_ACQUIRE_RETRIES = 8


def _held_inode_is_current(fd: int, lock_path: Path) -> bool:
    """True when ``fd`` (already flock-ed) is still the file published at ``lock_path``.

    False covers both ways the race manifests: the path was unlinked by the previous holder
    (``FileNotFoundError``) or a new file was created at that path, i.e. a different inode.
    ``st_dev`` is compared as well because inode numbers are only unique per device.
    """
    try:
        path_stat = os.stat(lock_path)
    except FileNotFoundError:
        return False
    fd_stat = os.fstat(fd)
    return (fd_stat.st_dev, fd_stat.st_ino) == (path_stat.st_dev, path_stat.st_ino)


def _open_and_lock(lock_path: Path, *, op: str) -> int:
    """Open ``lock_path``, take the exclusive flock, and write the diagnostic body.

    Returns the locked descriptor, whose fd is guaranteed to still be the file at the path, so
    the caller may unlink it on release without racing a waiter onto a dead inode. Raises
    ``ArchiveLockedError`` when another process holds the lock (non-blocking, as before) or when
    ``_ACQUIRE_RETRIES`` attempts all landed on inodes that had already been replaced.
    """
    for _ in range(_ACQUIRE_RETRIES):
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = ""
            try:
                holder = lock_path.read_text().strip()
            except (OSError, UnicodeDecodeError):
                # A truncated/garbage body is diagnostics, never a reason to fail differently.
                pass
            os.close(fd)
            raise ArchiveLockedError(
                f"cannot acquire write lock on {lock_path}\n"
                f"  currently held by {holder or '<unknown>'}\n"
                f"  (the holding process may have crashed; if so, "
                f"manually delete the lock file)"
            ) from None
        except BaseException:
            os.close(fd)
            raise
        if _held_inode_is_current(fd, lock_path):
            # The diagnostic body is written while we hold the lock and the descriptor is still
            # the file at the path: exactly the state the old code wrote it in.
            body = (
                f"host={socket.gethostname()} pid={os.getpid()} "
                f"op={op} started={_dt.datetime.now(_dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )
            try:
                os.ftruncate(fd, 0)
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, body.encode())
                os.fsync(fd)
            except BaseException:
                os.close(fd)
                raise
            return fd
        # We locked an inode nobody else can reach by path -- the holder released between our
        # open() and our flock(). Holding it would not exclude anyone, so drop it and retry on
        # whatever file is at the path now.
        os.close(fd)
    raise ArchiveLockedError(
        f"cannot acquire write lock on {lock_path}\n"
        f"  the lock file was replaced {_ACQUIRE_RETRIES} times while acquiring\n"
        f"  (a concurrent writer is retrying this same path in a tight loop?)"
    )


def _release_and_unlink(fd: int, lock_path: Path) -> None:
    """Unlink the sidecar while the lock is still held, then unlock and close.

    Order matters: unlinking before the unlock means a waiter never acquires this inode and then
    believes it owns the path -- it either wakes on the unlinked inode (``_held_inode_is_current``
    reports the missing path and it retries) or it blocks on the new file. A failure to unlink
    (read-only directory, NFS oddity) must not turn an already-successful write into an error,
    so it is swallowed; the lock itself is still released.
    """
    try:
        os.unlink(lock_path)
    except FileNotFoundError:
        pass
    except OSError:
        pass
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


@contextlib.contextmanager
def acquire_write_lock(lock_path: str | Path, *, op: str, unsafe_no_lock: bool = False):
    """Acquire an EXCLUSIVE fcntl.flock on ``lock_path`` for the context's duration.

    Parameters
    ----------
    lock_path
        File path to use as the lock target. Created if missing, and **unlinked on release**, so
        a write leaves no ``*.lock`` sidecar beside its output (ArcInstitute/cellstream#2). If
        the directory is read-only the unlink fails and the file stays; that is deliberately not
        an error, because the write itself already succeeded.
    op
        Short string written into the diagnostic body for debugging contention. Live
        values: ``'write'`` (every writer), ``'update_obs'`` / ``'update_var'`` (the packed
        metadata mutators, ``packed/writer.py``), and ``'recover'`` (packed footer recovery,
        ``packed/reader.py``). Any string is accepted — it is diagnostics, not a mode.
    unsafe_no_lock
        Escape hatch: skip locking entirely. Documented for emergencies on
        filesystems where flock isn't honored. Default False.

    Always ``LOCK_EX``. A ``shared=True`` (``LOCK_SH``) arm existed for the v1->v2
    converter, which took a shared lock on the v1 *source* while holding an exclusive one on
    the v2 destination — blocking writers on the source during conversion without blocking
    concurrent readers. #154 removed the converter (part A) and then the parameter (part B).
    **Removal is justified by there being no live caller**, nothing more.

    The synchronization itself was correct: ``LOCK_SH`` did block writers until every reader
    released. What was not multi-holder-safe is the *contention diagnostic*: the body below is
    written **unconditionally**, so each holder truncates and rewrites the file. Harmless with
    one exclusive holder; with N shared holders the body names only the last, and
    ``ArchiveLockedError`` reports whatever it says — a writer blocked by several readers
    would be told about one of them, possibly one that has already gone. (The converter was
    exposed to this: two concurrent conversions from the SAME source to different destinations
    would both hold that source's shared lock.) So a future shared lock is welcome, but it
    should make the body multi-holder-aware rather than restore this as-is.
    """
    lock_path = Path(lock_path)
    if unsafe_no_lock:
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = _open_and_lock(lock_path, op=op)
    try:
        yield
    finally:
        _release_and_unlink(fd, lock_path)
