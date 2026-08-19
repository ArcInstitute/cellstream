"""fcntl.flock-based exclusive write lock.

Held by the parent for the duration of a write operation. Workers do
NOT acquire the lock (the parent's lock covers them). On NFSv4 flock
is supported; on older NFS it may silently no-op — documented limitation.

The lock file body contains diagnostic info (host, pid, op, started)
which is read back into ArchiveLockedError when acquisition fails. Because the lock is
always EXCLUSIVE there is exactly one holder writing that body at a time — see the
``shared=`` note on ``acquire_write_lock`` for the multi-holder case that used to exist.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import os
import socket
from pathlib import Path

from cellstream.errors import ArchiveLockedError


@contextlib.contextmanager
def acquire_write_lock(lock_path: str | Path, *, op: str, unsafe_no_lock: bool = False):
    """Acquire an EXCLUSIVE fcntl.flock on ``lock_path`` for the context's duration.

    Parameters
    ----------
    lock_path
        File path to use as the lock target. Created if missing.
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
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        flock_mode = fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(fd, flock_mode)
        except BlockingIOError:
            holder = ""
            try:
                holder = lock_path.read_text().strip()
            except Exception:
                pass
            raise ArchiveLockedError(
                f"cannot acquire write lock on {lock_path}\n"
                f"  currently held by {holder or '<unknown>'}\n"
                f"  (the holding process may have crashed; if so, "
                f"manually delete the lock file)"
            ) from None

        body = (
            f"host={socket.gethostname()} pid={os.getpid()} "
            f"op={op} started={_dt.datetime.now(_dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, body.encode())
        os.fsync(fd)

        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)
