"""pack() must leave no orphaned staging file on any failure, and must not reuse a
fixed staging name that two concurrent writers would share (#280 item 3).

Every write here passes ``codec="zstd"`` deliberately. The default codec for integer
counts is pfordelta, which needs the optional ``cell`` extra (pyfastpfor) -- and unlike
tests/cell, tests/packed has no conftest gate for it, so the default would turn a valid
``.[dev]``-only install into a collection error instead of a test of ``pack()``. zstd
rides on ``zstandard``, a hard dependency, and pack() is codec-blind either way.
(codex, checkpoint 2.)
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.cell.writer import write_cell_archive
from cellstream.packed import writer as packed_writer


def _adata(n_obs=32, n_vars=16, seed=3):
    rng = np.random.default_rng(seed)
    x = sp.csr_matrix(rng.integers(0, 6, size=(n_obs, n_vars)).astype(np.int32))
    return ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )


def _write(adata, out, **kw):
    return write_cell_archive(adata, out, codec="zstd", **kw)


def _tmp_leftovers(d):
    return sorted(p.name for p in d.iterdir() if p.name.endswith(".tmp"))


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt()])
def test_pack_failure_leaves_no_tmp(tmp_path, monkeypatch, exc):
    """A failure inside pack() must unlink the partial staging file.

    KeyboardInterrupt is parametrised deliberately: `except Exception` would let a
    Ctrl-C during a multi-GB pack strand an archive-sized file, which is the exact
    production case #280 item 3 describes.
    """
    real = packed_writer._write_member
    calls = {"n": 0}

    def boom(out, src_path, *, delete_after):
        calls["n"] += 1
        if calls["n"] == 2:
            raise exc
        return real(out, src_path, delete_after=delete_after)

    monkeypatch.setattr(packed_writer, "_write_member", boom)
    out = tmp_path / "a.shad"
    with pytest.raises(type(exc)):
        _write(_adata(), out)

    assert not out.exists()
    leftovers = _tmp_leftovers(tmp_path)
    assert leftovers == [], f"pack() orphaned staging file(s): {leftovers}"


def test_pack_removes_a_fully_written_staging_file_when_the_rename_fails(tmp_path, monkeypatch):
    """The other half of the cleanup: a failure at the FINAL os.replace.

    The mid-write case above never reaches the rename, so it does not cover the path
    where the staging file is complete -- and that one is the archive-sized orphan
    (e.g. an EXDEV / read-only-destination rename failure). (codex, checkpoint 2.)
    """
    out = tmp_path / "a.shad"
    real_replace = packed_writer.os.replace

    def spy(src, dst):
        if Path(dst) == out:
            raise OSError("injected: rename failed")
        return real_replace(src, dst)

    monkeypatch.setattr(packed_writer.os, "replace", spy)
    with pytest.raises(OSError, match="injected"):
        _write(_adata(), out)

    assert not out.exists()
    leftovers = _tmp_leftovers(tmp_path)
    assert leftovers == [], f"pack() orphaned a completed staging file: {leftovers}"


def test_pack_refuses_to_clobber_an_existing_staging_file(tmp_path, monkeypatch):
    """O_EXCL, not just a unique name.

    test_pack_staging_name_is_unique below would pass unchanged with the old "wb": it
    only proves the NAME is derived per call. This proves the MODE -- with the name
    forced to collide, "xb" raises instead of truncating, and the `created` guard leaves
    the other writer's file alone rather than unlinking a file this call never owned.
    (codex, checkpoint 2.)
    """
    fixed = uuid.UUID("00000000-0000-4000-8000-000000000000")
    monkeypatch.setattr(packed_writer.uuid, "uuid4", lambda: fixed)

    out = tmp_path / "a.shad"
    staged = tmp_path / f"{out.name}.{os.getpid()}.{fixed.hex}.tmp"
    sentinel = b"another writer's in-flight pack"
    staged.write_bytes(sentinel)

    with pytest.raises(FileExistsError):
        _write(_adata(), out)

    assert staged.read_bytes() == sentinel, "pack() truncated or unlinked a file it did not own"
    assert not out.exists()


def test_pack_staging_name_is_unique(tmp_path, monkeypatch):
    """Two packs to the same destination must not share a staging path.

    A fixed `<dst>.tmp` is what lets two concurrent writers open one inode; this
    asserts the name is derived per call, without racing two processes.

    The spy MUST filter on the destination: monkeypatching `packed_writer.os.replace`
    patches the shared `os` module, so header.write_obs_parquet and
    manifest.dump_manifest land in the same recording and an unfiltered count fails
    even after the fix.
    """
    out = tmp_path / "a.shad"
    seen = []
    real_replace = packed_writer.os.replace

    def spy(src, dst):
        if Path(dst) == out:
            seen.append(str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(packed_writer.os, "replace", spy)
    _write(_adata(seed=1), out)
    _write(_adata(seed=2), out, overwrite=True)

    assert len(seen) == 2, f"expected exactly the two pack() renames, got {seen}"
    assert seen[0] != seen[1], f"pack() reused a fixed staging name: {seen}"
    for name in seen:
        assert name.endswith(".tmp")
        assert name.startswith(str(out))
