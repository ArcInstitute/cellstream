"""Negative control for ``tests/_stale_stat.short_stat`` (#305).

Every stale-stat regression test in ``tests/cell`` and ``tests/packed`` is vacuous if the injection
does not actually reach ``pathlib.Path.stat`` on the running interpreter -- and ``Path.stat`` has
not always routed through the live ``os.stat`` (the pre-3.11 ``_NormalAccessor`` captured it at
import time). This module fails loudly if that ever changes again, instead of letting the tests
that depend on it turn green for the wrong reason.

It lives at the suite root, ungated: ``tests/cell`` is skipped entirely without the optional
``cell`` extra, and the control must not be skipped alongside the tests it guards.
"""

from __future__ import annotations

import os

from tests._stale_stat import short_stat


def test_short_stat_reaches_path_stat_and_getsize(tmp_path):
    p = tmp_path / "victim.bin"
    p.write_bytes(b"x" * 500)
    with short_stat("victim.bin", missing=8):
        assert p.stat().st_size == 492  # pathlib routes through the live os.stat
        assert os.path.getsize(p) == 492  # and so does os.path.getsize
        assert os.stat(p).st_size == 492
    assert p.stat().st_size == 500  # and the patch is fully undone


def test_fstat_and_lseek_are_not_fooled(tmp_path):
    """The fix's whole basis: an open fd knows the truth even when the path stat does not.

    If this ever failed, switching a size check from ``stat`` to ``fstat`` would fix nothing.
    """
    p = tmp_path / "victim.bin"
    p.write_bytes(b"x" * 500)
    with short_stat("victim.bin", missing=8):
        with open(p, "rb") as fh:
            assert os.fstat(fh.fileno()).st_size == 500
            assert fh.seek(0, os.SEEK_END) == 500
        assert len(p.read_bytes()) == 500


def test_short_stat_leaves_other_files_alone(tmp_path):
    """Scoped to one basename: a stray shrink of every stat would make the guarded tests pass or
    fail for reasons that have nothing to do with the file under test."""
    victim, other = tmp_path / "victim.bin", tmp_path / "other.bin"
    victim.write_bytes(b"x" * 500)
    other.write_bytes(b"y" * 500)
    with short_stat("victim.bin", missing=8):
        assert victim.stat().st_size == 492
        assert other.stat().st_size == 500


def test_empty_file_is_left_at_zero(tmp_path):
    """``max(0, size - missing)`` must not turn an empty file into a negative or a nonsense size:
    the cell writer creates its staging file empty and an n_obs=0 archive keeps it that way."""
    p = tmp_path / "victim.bin"
    p.write_bytes(b"")
    with short_stat("victim.bin", missing=8):
        assert p.stat().st_size == 0
