"""#287 defect 1: a non-ValueError out of decode_cells must not mask the real error.

_rust_dispatch.decode_cells binds the caller's mmap-backed payload into a local and used to
drop that reference only on its `except ValueError` arm. For any other exception class the
traceback pins the frame, the mmap stays exported, and the caller's `finally: store.close()`
dies with BufferError("cannot close exported pointers exist") -- REPLACING the real error.

THE INJECTION TRAP (issue #287 warns about its own repro, correctly): a stub called as
`boom(*args)` keeps the payload alive in the STUB's own arg tuple, which the traceback pins
too, so its BufferError survives a correct fix. `_retaining` below is exactly that stub and is
kept as a POSITIVE CONTROL -- it must still raise BufferError on the fixed tree, which is what
proves the other tests' `RuntimeError` means "fixed" and not "detector went blind".
`_non_retaining` drops its own references first, which is what a real PyO3 panic does.
"""

from __future__ import annotations

import os

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellstream.cell._rust_dispatch import decode_cells, rust_available
from cellstream.cell.reader import CellStore, read_cell_bulk
from cellstream.cell.writer import write_cell_archive

rust_only = pytest.mark.skipif(not rust_available(), reason="Rust core not built/enabled")


def _archive(tmp_path):
    """A small zstd cell archive. zstd rather than the pfordelta default so the module leans on
    the always-present zstandard dep; the Rust decode gate accepts zstd + uint32 indices +
    uint32 values, which is what write_cell_archive produces here."""
    rng = np.random.default_rng(0)
    x = sp.csr_matrix((rng.random((64, 128)) < 0.3).astype(np.uint32) * 3)
    a = ad.AnnData(X=x)
    a.obs_names = [f"c{i}" for i in range(x.shape[0])]
    a.var_names = [f"g{j}" for j in range(x.shape[1])]
    p = tmp_path / "cells.shad"
    write_cell_archive(a, p, codec="zstd", overwrite=True)
    return p


def _retaining(*args, **kwargs):
    """POSITIVE CONTROL stub: its own frame keeps the payload alive, so BufferError is the
    correct outcome BEFORE and AFTER the fix."""
    raise RuntimeError("INJECTED")


def _non_retaining(*args, **kwargs):
    """Faithful proxy for a real native failure: drop our own references, then raise."""
    del args, kwargs
    raise RuntimeError("INJECTED")


class _Interrupted(BaseException):
    """A BaseException that is NOT an Exception -- i.e. the KeyboardInterrupt/SystemExit class.

    Custom rather than KeyboardInterrupt itself so nothing downstream (pytest's own handling,
    a signal-aware worker pool) can treat it specially and mask the measurement.
    """


def _non_retaining_base(*args, **kwargs):
    """Same, but raising outside the Exception hierarchy."""
    del args, kwargs
    raise _Interrupted("INJECTED-BASE")


@rust_only
@pytest.mark.parametrize("backing", ["heap", "shm"])
def test_read_cell_bulk_non_valueerror_reaches_the_caller(tmp_path, monkeypatch, backing):
    """read_cell_bulk's `finally: store.close()` must not turn RuntimeError into BufferError."""
    from cellstream.v2 import _rust

    path = _archive(tmp_path)
    monkeypatch.setattr(_rust, "decode_cells", _non_retaining)
    with pytest.raises(RuntimeError, match="INJECTED"):
        read_cell_bulk(path, output_backing=backing)


@rust_only
@pytest.mark.parametrize("backing", ["heap", "shm"])
def test_read_cell_bulk_baseexception_reaches_the_caller(tmp_path, monkeypatch, backing):
    """The outer arm must be a `finally`, NOT an `except Exception`.

    Every other injection in this module is an Exception subclass, so a mutant that wrote
    `except Exception: del ...; raise` instead would pass all of them -- and a
    KeyboardInterrupt/SystemExit would go on leaving the mmap exported and being replaced by
    BufferError, which is the whole defect. This is the test that pins the breadth.
    (codex, checkpoint 2.)"""
    from cellstream.v2 import _rust

    path = _archive(tmp_path)
    monkeypatch.setattr(_rust, "decode_cells", _non_retaining_base)
    with pytest.raises(_Interrupted, match="INJECTED-BASE"):
        read_cell_bulk(path, output_backing=backing)


@rust_only
def test_retaining_stub_still_raises_buffererror(tmp_path, monkeypatch):
    """POSITIVE CONTROL for the test above.

    A stub that RETAINS the payload in its own traceback-pinned frame still strands the mmap
    export, and no change inside decode_cells can help. If this ever stops raising BufferError
    the oracle has gone blind and the sibling tests' RuntimeError proves nothing. The
    __context__ assertion is what makes it a control rather than a coincidence: it proves this
    BufferError arose while the INJECTED error was in flight, i.e. that masking is what is
    being observed."""
    from cellstream.v2 import _rust

    path = _archive(tmp_path)
    monkeypatch.setattr(_rust, "decode_cells", _retaining)
    with pytest.raises(BufferError) as excinfo:
        read_cell_bulk(path, output_backing="heap")
    ctx = excinfo.value.__context__
    assert isinstance(ctx, RuntimeError) and "INJECTED" in str(ctx), (
        f"the BufferError did not mask the injected error (__context__={ctx!r})"
    )


@rust_only
def test_gather_rows_non_valueerror_reaches_the_caller(tmp_path, monkeypatch):
    """Same defect through the other public entry point, with the OWNER closing the store."""
    from cellstream.v2 import _rust

    path = _archive(tmp_path)
    monkeypatch.setattr(_rust, "decode_cells", _non_retaining)
    store = CellStore(path)
    with pytest.raises(RuntimeError, match="INJECTED"):
        try:
            store.gather_rows(np.arange(8, dtype=np.int64))
        finally:
            store.close()


def test_conversion_failure_reaches_the_caller(tmp_path):
    """The four ascontiguousarray conversions sit outside any try today.

    No monkeypatching: an object-dtype row_ids cannot become int64, so np.ascontiguousarray
    raises TypeError AFTER payload_u8 has already been rebound. Runs on both engines -- the
    conversion raises before _rust is ever touched."""
    path = _archive(tmp_path)
    store = CellStore(path)
    total = int(store.indptr[-1])
    with pytest.raises(TypeError):
        try:
            payload = np.frombuffer(store._mm, dtype=np.uint8)
            try:
                decode_cells(
                    payload,
                    store.offsets,
                    store.indptr,
                    np.array([object()], dtype=object),
                    np.empty(total, np.float32),
                    np.empty(total, np.int32),
                    np.empty(store.n_obs + 1, np.int64),
                    codec="zstd",
                    value_dtype_on_disk=store.manifest["value_dtype_on_disk"],
                    n_vars=store.n_vars,
                    allow_lossy=True,
                    n_threads=1,
                )
            finally:
                payload = None
        finally:
            store.close()


@rust_only
@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="no /dev/shm on this platform")
def test_shm_backing_unlinks_segments_on_decode_failure(tmp_path, monkeypatch):
    """The shm arm must leave no /dev/shm entry behind when decode raises.

    Deliberately uses _retaining and expects BufferError: this test's subject is the LEAK, not
    the masking, and propagation on the shm arm is already covered by the parametrized test
    above. Using _retaining keeps it green on both trees, so a failure here is unambiguously a
    leak rather than a re-run of the differential.

    Name-based, never count-based: a global /dev/shm inventory is machine-wide and races any
    concurrent test run. `assert names` is the control -- without it a spy that never fired
    would make the leak check vacuous."""
    import cellstream.shm as _shm
    from cellstream.v2 import _rust

    path = _archive(tmp_path)
    real_alloc = _shm.allocate_shm_buffers
    names = []

    def spy(**kw):
        bundle = real_alloc(**kw)
        names.extend([bundle.data.name, bundle.indices.name, bundle.indptr.name])
        return bundle

    monkeypatch.setattr(_shm, "allocate_shm_buffers", spy)
    monkeypatch.setattr(_rust, "decode_cells", _retaining)
    with pytest.raises(BufferError):
        read_cell_bulk(path, output_backing="shm")
    assert names, "control: the allocate_shm_buffers spy never fired -- shm arm not taken"
    leaked = [n for n in names if os.path.exists(f"/dev/shm/{n}")]
    assert not leaked, f"orphaned /dev/shm segments after a failed decode: {leaked}"


@pytest.mark.parametrize("fail_call", [1, 2, 3, 4])
def test_every_conversion_is_inside_the_protected_region(tmp_path, monkeypatch, fail_call):
    """Moving the outer `try` BELOW the first three conversions passes every other test here.

    test_conversion_failure_reaches_the_caller only fails the FOURTH conversion (row_ids), so
    it cannot tell "the try covers all four" from "the try starts just before row_ids" -- and
    an offsets/indptr failure would go on pinning payload_u8 and being masked. This fails each
    conversion in turn, with a BaseException so the breadth is pinned at the same time.
    (codex, checkpoint 2 round 2.)

    fail_call=1 is the case where payload_u8 has NOT yet been rebound: the parameter still
    holds the caller's array, this frame still pins it, and the masking is identical.
    """
    path = _archive(tmp_path)
    store = CellStore(path)
    total = int(store.indptr[-1])
    real_ascontiguous = np.ascontiguousarray
    seen = {"n": 0}

    def counting(*a, **kw):
        seen["n"] += 1
        if seen["n"] == fail_call:
            # THE INJECTION TRAP, again: on fail_call=1 the payload is in THIS stub's own arg
            # tuple, and the traceback pins this frame, so without dropping them the mmap stays
            # exported no matter what decode_cells does. Cases 2-4 never held it and passed
            # even without this -- which is exactly how the trap hides.
            del a, kw
            raise _Interrupted(f"INJECTED-CONV-{fail_call}")
        return real_ascontiguous(*a, **kw)

    with pytest.raises(_Interrupted, match=f"INJECTED-CONV-{fail_call}"):
        try:
            payload = np.frombuffer(store._mm, dtype=np.uint8)
            # Patched as late as possible so only decode_cells' own four calls are counted.
            monkeypatch.setattr(np, "ascontiguousarray", counting)
            try:
                decode_cells(
                    payload,
                    store.offsets,
                    store.indptr,
                    np.arange(4, dtype=np.int64),
                    np.empty(total, np.float32),
                    np.empty(total, np.int32),
                    np.empty(store.n_obs + 1, np.int64),
                    codec="zstd",
                    value_dtype_on_disk=store.manifest["value_dtype_on_disk"],
                    n_vars=store.n_vars,
                    allow_lossy=True,
                    n_threads=1,
                )
            finally:
                payload = None
        finally:
            store.close()
    assert seen["n"] == fail_call, f"control: conversion {fail_call} was never reached"
