import os
import stat
import sys

import numpy as np
import pytest

from cellstream.v2._rust_dispatch import rust_available

pytestmark = pytest.mark.skipif(not rust_available(), reason="Rust extension not built")

SHUFFLE = "byteshuffle-bytedelta"


def _rust():
    from cellstream.v2 import _rust

    return _rust


def test_encode_shards_inmem_row_index_out_of_range_raises_valueerror(tmp_path):
    # 3-row CSR (indptr length 4); a shard row list referencing row 5 is invalid.
    data = np.array([1, 2, 3], dtype=np.uint16)
    indices = np.array([0, 1, 0], dtype=np.int32)
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    bad_rows = [np.array([0, 5], dtype=np.int64)]
    with pytest.raises(ValueError, match="row index"):
        _rust().encode_shards_inmem(
            data,
            indices,
            indptr,
            bad_rows,
            str(tmp_path),
            n_vars=2,
            shuffle=SHUFFLE,
            level=1,
            chunk_size=1048576,
            n_threads=1,
            data_dtype=None,
            index_dtype=None,
        )


def test_encode_shards_inmem_non_decreasing_indptr_raises_valueerror(tmp_path):
    data = np.array([1], dtype=np.uint16)
    indices = np.array([0], dtype=np.int32)
    indptr = np.array([0, 2, 1], dtype=np.int64)
    rows = [np.array([0, 1], dtype=np.int64)]
    with pytest.raises(ValueError, match="non-decreasing"):
        _rust().encode_shards_inmem(
            data,
            indices,
            indptr,
            rows,
            str(tmp_path),
            n_vars=2,
            shuffle=SHUFFLE,
            level=1,
            chunk_size=1048576,
            n_threads=1,
            data_dtype=None,
            index_dtype=None,
        )


def test_encode_shards_inmem_indptr_exceeds_data_raises_valueerror(tmp_path):
    data = np.array([1, 2, 3], dtype=np.uint16)
    indices = np.array([0, 1, 0], dtype=np.int32)
    indptr = np.array([0, 1, 5], dtype=np.int64)
    rows = [np.array([0, 1], dtype=np.int64)]
    with pytest.raises(ValueError, match="exceeds data length"):
        _rust().encode_shards_inmem(
            data,
            indices,
            indptr,
            rows,
            str(tmp_path),
            n_vars=2,
            shuffle=SHUFFLE,
            level=1,
            chunk_size=1048576,
            n_threads=1,
            data_dtype=None,
            index_dtype=None,
        )


def test_encode_shards_inmem_data_indices_length_mismatch_raises_valueerror(tmp_path):
    data = np.array([1, 2, 3], dtype=np.uint16)
    indices = np.array([0, 1], dtype=np.int32)
    indptr = np.array([0, 1, 2], dtype=np.int64)
    rows = [np.array([0, 1], dtype=np.int64)]
    with pytest.raises(ValueError, match="!= indices length"):
        _rust().encode_shards_inmem(
            data,
            indices,
            indptr,
            rows,
            str(tmp_path),
            n_vars=2,
            shuffle=SHUFFLE,
            level=1,
            chunk_size=1048576,
            n_threads=1,
            data_dtype=None,
            index_dtype=None,
        )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.chmod does not prevent file creation in directories on Windows",
)
def test_encode_shards_inmem_readonly_outdir_raises_oserror(tmp_path):
    if sys.platform != "win32" and os.getuid() == 0:
        pytest.skip("read-only directory does not block writes for the root user")
    data = np.array([1, 2, 3], dtype=np.uint16)
    indices = np.array([0, 1, 0], dtype=np.int32)
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    rows = [np.array([0, 1, 2], dtype=np.int64)]
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)  # read+exec only: File::create fails
    try:
        with pytest.raises(OSError):  # PermissionError is an OSError; NOT PanicException
            _rust().encode_shards_inmem(
                data,
                indices,
                indptr,
                rows,
                str(ro),
                n_vars=2,
                shuffle=SHUFFLE,
                level=1,
                chunk_size=1048576,
                n_threads=1,
                data_dtype=None,
                index_dtype=None,
            )
    finally:
        os.chmod(ro, stat.S_IRWXU)  # restore so tmp cleanup can remove it
