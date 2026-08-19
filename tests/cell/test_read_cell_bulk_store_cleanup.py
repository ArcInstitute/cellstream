"""#287 defect 3: read_cell_bulk must not leak the open store when shm allocation fails.

The pure-Python arm allocated its bundle OUTSIDE the try whose `finally: store.close()` owns
the cleanup, so an OSError from a full /dev/shm left the archive's mmap and file descriptor
open until the traceback -- and with it the CellStore -- was released. Every other early-exit
path in the function closes it. Needs no interrupt: an ordinary OSError is enough.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import cellstream.cell.reader as _reader
import cellstream.shm as _shm
from cellstream.cell.reader import CellStore, read_cell_bulk
from cellstream.cell.writer import write_cell_archive


def _archive(tmp_path):
    rng = np.random.default_rng(1)
    x = sp.csr_matrix((rng.random((32, 64)) < 0.3).astype(np.uint32) * 3)
    a = ad.AnnData(X=x)
    a.obs_names = [f"c{i}" for i in range(x.shape[0])]
    a.var_names = [f"g{j}" for j in range(x.shape[1])]
    p = tmp_path / "cells.shad"
    write_cell_archive(a, p, codec="zstd", overwrite=True)
    return p


def _recording_cellstore(closed):
    class _Recording(CellStore):
        def close(self):
            closed.append(1)
            super().close()

    return _Recording


def test_recorder_fires_on_a_successful_read(tmp_path, monkeypatch):
    """POSITIVE CONTROL: prove read_cell_bulk really uses the patched CellStore, and that its
    close() is reached at all on this path. Without it, `assert closed` in the sibling test
    could be measuring nothing."""
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")  # force the pure-Python arm
    path = _archive(tmp_path)
    closed = []
    monkeypatch.setattr(_reader, "CellStore", _recording_cellstore(closed))
    adata = read_cell_bulk(path, n_workers=2)
    assert adata.shape == (32, 64)
    assert closed, "control: read_cell_bulk did not use the recording CellStore"


def test_allocation_failure_still_closes_the_store(tmp_path, monkeypatch):
    """The defect itself."""
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")  # force the pure-Python arm
    path = _archive(tmp_path)
    closed = []
    monkeypatch.setattr(_reader, "CellStore", _recording_cellstore(closed))

    def boom(**kw):
        raise OSError("INJECTED: /dev/shm exhausted")

    monkeypatch.setattr(_shm, "allocate_shm_buffers", boom)
    with pytest.raises(OSError, match="INJECTED"):
        read_cell_bulk(path, n_workers=2)
    assert closed, "the open store was leaked when shm allocation failed"
