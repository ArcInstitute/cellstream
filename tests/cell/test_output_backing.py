"""#277 defect 3: read_cell_bulk must validate output_backing and honour it on BOTH engines."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.cell._rust_dispatch import rust_available
from cellstream.cell.reader import read_cell_bulk
from cellstream.cell.writer import write_cell_archive

VALS = np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6], [7, 8, 0]], dtype=np.uint16)


@pytest.fixture
def archive(tmp_path):
    a = ad.AnnData(
        X=sp.csr_matrix(VALS, dtype=np.uint16),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(3)]),
    )
    p = tmp_path / "b.shad"
    write_cell_archive(a, p)
    return p


# `engine_env` (tests/cell/conftest.py) is a params=[None, "1"] fixture: naming it as a test
# argument runs that test under BOTH engines in one pytest invocation. It CLEARS
# CELLSTREAM_DISABLE_RUST on the default variant, so the Rust arm is really exercised even when
# the ambient env has it set -- which is what the old test at test_bulk.py:102 could not do.


def test_shm_backing_is_honoured(archive, engine_env):
    """The probe is hasattr, NOT `in adata.uns` -- attach_buffer_cleanup sets an ATTRIBUTE,
    so the uns form reads False on both engines and makes the assertion vacuous."""
    got = read_cell_bulk(archive, output_backing="shm", data_dtype="uint16")
    assert hasattr(got, "close_shared_memory"), (
        f"output_backing='shm' ignored (rust_available={rust_available()})"
    )
    np.testing.assert_array_equal(got.X.toarray().astype(np.int64), VALS.astype(np.int64))
    got.close_shared_memory()


def test_shm_backs_all_three_arrays(archive, engine_env):
    """ALL of data/indices/indptr must be genuine shm views, not just survive.

    indptr is the one that regresses silently: decode_cells writes int64 but resolve_plan
    picks int32 by default, so any astype-after-decode hands back a heap COPY while an
    unused segment stays attached -- and every value/lifetime assertion still passes.
    """
    from cellstream.shm import _ShmBackedArray

    got = read_cell_bulk(archive, output_backing="shm", data_dtype="uint16")
    try:
        for name in ("data", "indices", "indptr"):
            arr = getattr(got.X, name)
            assert isinstance(arr.base, _ShmBackedArray), (
                f"X.{name} is not shm-backed (base={type(arr.base).__name__}, "
                f"dtype={arr.dtype}, rust_available={rust_available()})"
            )
    finally:
        # getattr-guarded: if a regression makes output_backing="shm" return an AnnData with
        # no close_shared_memory -- the exact failure this test exists to catch -- an
        # unconditional call raises AttributeError from the finally and MASKS the informative
        # assertion above. (Copilot, checkpoint 2.)
        close = getattr(got, "close_shared_memory", None)
        if close is not None:
            close()


def test_heap_backing_has_no_shm_handle(archive, engine_env):
    got = read_cell_bulk(archive, output_backing="heap", data_dtype="uint16")
    assert not hasattr(got, "close_shared_memory")
    np.testing.assert_array_equal(got.X.toarray().astype(np.int64), VALS.astype(np.int64))


# NOTE: the #274 "extracted array outlives the AnnData" guard deliberately lives in
# tests/cell/test_shm_csr_dangling.py, NOT here. It must run in a SUBPROCESS with flushed
# markers: a regression is a use-after-free, so an in-process version would SIGSEGV the whole
# pytest run instead of failing one test. That file already covers data/indices/indptr under
# output_backing="shm" on both engines. (Copilot, checkpoint 2.)


def test_unknown_backing_raises(archive, engine_env):
    with pytest.raises(ValueError, match="must be 'heap' or 'shm'"):
        read_cell_bulk(archive, output_backing="banana", data_dtype="uint16")


def test_pinned_backing_raises_not_implemented(archive, engine_env):
    with pytest.raises(NotImplementedError, match="pinned"):
        read_cell_bulk(archive, output_backing="pinned", data_dtype="uint16")


def test_both_backings_agree(archive, engine_env):
    h = read_cell_bulk(archive, output_backing="heap", data_dtype="uint16")
    s = read_cell_bulk(archive, output_backing="shm", data_dtype="uint16")
    np.testing.assert_array_equal(h.X.toarray(), s.X.toarray())
    np.testing.assert_array_equal(h.X.indptr, s.X.indptr)
    np.testing.assert_array_equal(h.X.indices, s.X.indices)
    assert h.X.indptr.dtype == s.X.indptr.dtype
    s.close_shared_memory()
