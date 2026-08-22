"""#305: the cell write path must not size its staging file with a path ``stat()``.

The fault injected here -- ``os.stat`` under-reports one basename while ``os.fstat`` stays truthful
-- is the WekaFS attribute-cache behaviour that #305 turned out to be; see ``tests/_stale_stat`` for
the measurements and ``tests/test_stale_stat_harness.py`` for the negative control that keeps these
tests from passing vacuously.

The error these reproduce is byte-identical to the one observed in the wild:
``ValueError: reorder_frames: in_offsets[-1]=168 != staging file size 36``.

⚠️ A Python-level ``os.stat`` patch cannot reach Rust's ``std::fs::metadata``, so the grouped test
runs on the pure-Python engine. The Rust guard's own change -- sizing the staging file from the
handle it opens to read it, instead of a separate ``stat`` on the path -- has no Python-injectable
test; ``test_reorder_frames.py`` covers that it still fires on a genuinely wrong ``in_offsets``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.cell.archive_writer import CellArchiveWriter
from cellstream.cell.reader import open_cell
from cellstream.cell.writer import _reorder_frames_python
from tests._stale_stat import short_stat

from .conftest import assert_archives_equivalent

STAGING = "x_payload_staging"
N_VARS = 16


def _block(n_obs, seed=5, n_vars=N_VARS):
    rng = np.random.default_rng(seed)
    return sp.csr_matrix(rng.integers(0, 6, size=(n_obs, n_vars)).astype(np.int32))


def _obs_var(n_obs):
    return (
        pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        pd.DataFrame(index=[f"g{i}" for i in range(N_VARS)]),
    )


def _write(out, obs, var, **finalize_kw):
    with CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32") as w:
        w.add_block(_block(4))
        w.finalize(obs=obs, var=var, **finalize_kw)


def test_identity_fast_path_survives_a_stale_short_stat(tmp_path):
    """``_reorder_and_finalize``'s identity fast-path validated ``in_offsets[-1]`` against
    ``Path(staging).stat().st_size``. That is the site the observed failures came from -- 12 of
    1500 writes on 3.13 + wekafs, 20 of 1500 on 3.11.

    Compared against a reference archive written WITHOUT the injection, so the contents are pinned
    too: a write that merely stopped raising while producing different bytes would fail here.
    """
    ref, victim = tmp_path / "ref.shad", tmp_path / "v.shad"
    obs, var = _obs_var(4)
    _write(ref, obs, var)
    with short_stat(STAGING):
        _write(victim, obs, var)
    assert_archives_equivalent(ref, victim)


def test_grouped_reorder_survives_a_stale_short_stat(tmp_path, monkeypatch):
    """The grouped path takes ``reorder_frames``, whose pure-Python arm sized the staging file with
    ``os.path.getsize``.

    Forced onto that arm deliberately: with the Rust extension live the guard is a
    ``std::fs::metadata`` call inside the extension, which no Python-level patch can reach -- so on
    the Rust engine this test would pass without exercising anything.
    """
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    ref, victim = tmp_path / "gref.shad", tmp_path / "gv.shad"
    obs, var = _obs_var(4)
    obs["target_gene"] = ["g1", "g0", "g1", "g0"]  # non-identity storage order
    _write(ref, obs, var, group_by="target_gene")
    with short_stat(STAGING):
        _write(victim, obs, var, group_by="target_gene")
    assert_archives_equivalent(ref, victim)


def test_flush_rollback_mark_survives_a_stale_short_stat(tmp_path):
    """The highest-severity site: ``_flush``'s ``mark = self._staging.stat().st_size`` is the
    rollback truncation point. A stale-short mark truncates away bytes belonging to blocks flushed
    EARLIER that are still counted in ``_frame_lens`` -- killing a good, ongoing push write instead
    of only discarding the block that failed.

    ``encode_block_rows=1`` flushes per ``add_block``, so the rejected block has a predecessor to
    destroy; without that predecessor the bug is unobservable.
    """
    out = tmp_path / "roll.shad"
    w = CellArchiveWriter(out, n_vars=N_VARS, value_dtype="int32", encode_block_rows=1)
    try:
        w.add_block(_block(4, seed=1))  # good, and flushed immediately
        bad = sp.csr_matrix(
            (
                np.array([1, 2], dtype=np.int32),
                np.array([0, N_VARS + 5], dtype=np.int32),  # out of range -> #264 rejects
                np.array([0, 2, 2, 2, 2], dtype=np.int32),
            ),
            shape=(4, N_VARS),
        )
        with short_stat(STAGING):
            with pytest.raises(ValueError):
                w.add_block(bad)
        # The good block must still be intact: the rollback may only undo the failed flush.
        obs, var = _obs_var(4)
        w.finalize(obs=obs, var=var)
    except BaseException:
        w.abort()
        raise
    s = open_cell(out)
    try:
        assert s.n_obs == 4
        got = s.gather_rows(np.arange(4))
        assert np.array_equal(got.toarray(), _block(4, seed=1).toarray())
    finally:
        s.close()


def test_reorder_frames_python_survives_a_stale_short_stat(tmp_path):
    """``_reorder_frames_python`` sized the staging file with ``os.path.getsize``. It is the
    pure-Python twin of the Rust guard, so leaving it out would make the two engines disagree about
    the same input on the same filesystem -- the parity the cell code keeps everywhere else.

    Frames are opaque byte strings here, exactly as in ``test_reorder_frames.py``.
    """
    frames = [b"aaa", b"bbbb", b"c", b"", b"dddddd", b"ee"]
    stg, dst = tmp_path / STAGING, tmp_path / "out.bin"
    with open(stg, "wb") as fh:
        for fr in frames:
            fh.write(fr)
    in_off = np.concatenate([[0], np.cumsum([len(f) for f in frames])]).astype(np.int64)
    perm = np.array([3, 0, 5, 1, 4, 2], dtype=np.int64)

    with short_stat(STAGING):
        out_off = _reorder_frames_python(stg, dst, in_off, perm)

    raw = dst.read_bytes()
    got = [raw[out_off[i] : out_off[i + 1]] for i in range(len(out_off) - 1)]
    assert got == [frames[i] for i in perm]
