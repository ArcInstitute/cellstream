"""Inject the WekaFS stale-``stat()`` fault that #305 turned out to be.

On Arc's WekaFS -- and on any attribute-caching network filesystem -- ``stat(path)`` can report a
stale-SHORT ``st_size`` immediately after an append through a different fd, while ``fstat`` on an
open fd, ``lseek(SEEK_END)`` and the bytes themselves are all already correct. No bytes are lost;
only the cached attribute lags, and it settles within a fraction of a second.

Measured 2026-08-20 with 1500 ``CellArchiveWriter`` writes per arm: 20 failures on Python 3.11 +
wekafs, 8 on 3.12, 12 on 3.13, and **0 on local ext4 for every version**. 3.11 was the worst arm,
which is why attributing those failures to CPython 3.13 was wrong -- the variable was the
filesystem, not the interpreter.

Injecting the fault beats waiting for it: any code that sizes a file it is about to read with a
path ``stat()`` fails here on ANY filesystem, including CI's local disk where the real staleness
never occurs. ``tests/test_stale_stat_harness.py`` is the negative control for this module and runs
ungated, so the callers below cannot pass vacuously.
"""

from __future__ import annotations

import contextlib
import os


class ShortStat:
    """A ``stat_result`` proxy that under-reports ``st_size`` and forwards everything else."""

    def __init__(self, real, size):
        self._real = real
        self.st_size = size

    def __getattr__(self, name):
        # object.__getattribute__ so a lookup arriving before _real is bound cannot recurse.
        return getattr(object.__getattribute__(self, "_real"), name)


@contextlib.contextmanager
def short_stat(basename, *, missing=8):
    """Make ``os.stat`` (and so ``Path.stat`` / ``os.path.getsize``) under-report ONE basename.

    ``missing`` bytes are subtracted, floored at 0, so a non-empty file always reports short and
    an empty one stays 0 -- mirroring the partial visibility actually observed (e.g. 132 reported
    for 536 real bytes).

    Scoped to a single basename on purpose: ``os.stat`` is load-bearing for a lot of machinery
    that runs during a write, and shrinking all of it would test something else entirely.
    """
    real = os.stat

    def fake(path, *args, **kwargs):
        st = real(path, *args, **kwargs)
        try:
            name = os.path.basename(os.fspath(path))
        except TypeError:  # an open fd, not a path -- never the target
            return st
        if name == basename and st.st_size > 0:
            return ShortStat(st, max(0, st.st_size - missing))
        return st

    os.stat = fake
    try:
        yield
    finally:
        os.stat = real
