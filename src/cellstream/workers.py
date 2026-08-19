"""Multiprocessing helpers shared by the v2 write and read worker families.

The v1 fork-pool machinery went with the schema-v1 writer in #154: the ``_PARENT_ADATA``
copy-on-write pattern and its setter/clearer, the Blosc encode/convert workers with their
``BLOSC_NTHREADS`` env window, the fork/spawn context helpers, and the ``require_linux``
gate. Their only caller was ``write_sharded``'s v1 path (the v1 READ workers it also served
went in part A of the same issue).

What remains is what both surviving worker families reach for: the ``OMP_NUM_THREADS`` env
window, its lock — shared with ``v2/reader_cpu`` so read and write serialize on ONE lock
rather than two — and the canonical test-only worker-death hook. Every remaining worker pool
is a **spawn** pool that builds its own context (``v2/writer.py``, ``v2/reader_cpu.py``).
"""

from __future__ import annotations

import contextlib
import os
import threading

# Serializes the OMP_NUM_THREADS env-mutation window in omp_nthreads_in_env(): two parent
# threads invoking a spawn-pool writer concurrently with different omp_nthreads must not
# race on os.environ.
# PUBLIC because the v2 read path (reader_cpu) reuses this SAME lock -- read and write both
# mutate the single global OMP_NUM_THREADS, so they must serialize on one lock, not two
# (ultrareview L10).
OMP_ENV_LOCK = threading.Lock()


@contextlib.contextmanager
def omp_nthreads_in_env(n_threads: int | None):
    """Set OMP_NUM_THREADS in the parent env for the executor's lifetime, then restore.

    bitshuffle reads OMP_NUM_THREADS at import time and sizes its OpenMP pool
    then. Spawn workers re-import bitshuffle, so the value must be set in the
    PARENT before any worker spawns (a Pool initializer runs too late — the
    initializer pickle imports the worker module, which imports bitshuffle,
    before the initializer body runs). Mirrors v2/reader_cpu.py's inline OMP
    dance.

    Held across the whole executor lifetime because ProcessPoolExecutor spawns
    workers lazily (each submit() can trigger another spawn).

    n_threads=None: no-op (still yields, so callers can use it unconditionally).
    """
    with OMP_ENV_LOCK:
        prev = os.environ.get("OMP_NUM_THREADS")
        should_modify = n_threads is not None and prev != str(n_threads)
        try:
            if should_modify:
                os.environ["OMP_NUM_THREADS"] = str(n_threads)
            yield
        finally:
            if should_modify:
                if prev is None:
                    os.environ.pop("OMP_NUM_THREADS", None)
                else:
                    os.environ["OMP_NUM_THREADS"] = prev


def _maybe_test_kill_self(shard_idx: int) -> None:
    """Canonical test-only worker-death hook, shared by every remaining worker.

    If ``CELLSTREAM_TEST_KILL_SHARD_IDX`` is a digit-string equal to this
    worker's shard_idx, SIGKILL the process before doing real work. Lets the
    fail-fast tests deterministically exercise the BrokenProcessPool path
    without needing an OOM or a real crash. Spawn workers capture the env var at
    process start.

    This is the single definition every worker entrypoint imports: the v2 writer
    (``writer._encode_one_shard`` and its ``_from_shm`` / ``_gather_local`` / ``_gather_shm``
    / ``_gather_backed`` variants) and the v2 reader
    (``reader_cpu._read_one_shard_into_shm`` and ``_scatter_one_shard_dense``). The v1 read
    workers it also used to serve were removed in #154, and the v1 write workers
    (``write_shard_worker`` / ``convert_shard_worker``) with the v1 writer in part B of the
    same issue.

    .isdigit() guard so a malformed env value doesn't ValueError.
    """
    _kill_idx = os.environ.get("CELLSTREAM_TEST_KILL_SHARD_IDX")
    if _kill_idx is not None and _kill_idx.isdigit() and int(_kill_idx) == shard_idx:
        import signal as _sig

        os.kill(os.getpid(), _sig.SIGKILL)
