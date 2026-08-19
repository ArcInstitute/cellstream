"""Capabilities unlocked by always-uint32 on-disk storage + recorded min dtypes:
counts > 65535 stored natively, n_var > 65536 indices, cast up/down from the
recorded min, and the reverse-compat fallback when *_min is absent."""

import json

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import cellstream
from cellstream.errors import LossyCastError
from cellstream.read import ShardedArchive
from tests._archive_helpers import pack_directory, write_directory


def _csr(arr):
    return ad.AnnData(X=sp.csr_matrix(np.asarray(arr)))


def _dense(a):
    return a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)


def test_counts_above_65535_stored_natively_uint32(tmp_path):
    # Old uint16 narrowing fell back to a wider float here; uint32 stores it natively.
    X = np.array([[0, 70000], [100000, 0]], dtype=np.uint32)
    out = tmp_path / "big"
    cellstream.write_sharded(_csr(X), out, n_shards=1)
    m = ShardedArchive(str(out)).manifest
    assert m["x_data_dtype_on_disk"] == "uint32"
    assert m["x_data_dtype_min"] == "uint32"  # 100000 needs uint32
    back = _dense(cellstream.read_h5ad(out))
    assert np.array_equal(back.astype(np.uint32), X)


def test_n_var_above_65536_uint32_indices(tmp_path):
    n_vars = 70000
    X = sp.csr_matrix(
        (
            np.array([1, 2], dtype=np.uint16),
            np.array([0, 69999], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
        ),
        shape=(2, n_vars),
    )
    out = tmp_path / "wide"
    cellstream.write_sharded(ad.AnnData(X=X), out, n_shards=1)
    m = ShardedArchive(str(out)).manifest
    assert m["x_indices_dtype_on_disk"] == "uint32"
    assert m["x_indices_dtype_min"] == "uint32"  # n_vars-1 > 65535
    assert ShardedArchive(str(out)).to_anndata().shape[1] == n_vars


def test_n_var_above_65536_narrow_read_is_lossless(tmp_path):
    # H1 (ultrareview): a cast_back=False (narrow) read of a wide archive must NOT
    # raise LossyCastError and must preserve column indices >= 65536 — the uint16
    # narrow default truncated/raised here. Mirrors the cast_back=True test above.
    n_vars = 70000
    X = sp.csr_matrix(
        (
            np.array([1, 2], dtype=np.uint16),
            np.array([0, 69999], dtype=np.int64),
            np.array([0, 1, 2], dtype=np.int64),
        ),
        shape=(2, n_vars),
    )
    out = tmp_path / "wide_narrow"
    cellstream.write_sharded(ad.AnnData(X=X), out, n_shards=1)
    a = ShardedArchive(str(out)).to_anndata(cast_back=False)  # narrow path
    assert a.shape == (2, n_vars)
    # Native on-disk index width is uint32; the narrow read must keep it (not uint16).
    assert a.X.indices.dtype == np.dtype(np.uint32)
    # Row 1's single nnz is the high column 69999 (would be truncated under uint16).
    assert int(a.X.indices[1]) == 69999
    assert int(a.X.data[1]) == 2
    assert int(a.X.indices[0]) == 0


def test_default_float32_read_is_lossless_and_cheap(tmp_path):
    X = np.array([[0, 3, 0], [5, 0, 200]], dtype=np.uint16)  # max 200 -> min uint8
    out = tmp_path / "f32"
    cellstream.write_sharded(_csr(X), out, n_shards=1)
    arch = ShardedArchive(str(out))
    got = _dense(arch.to_anndata())  # default cast_back -> float32 for counts
    assert got.dtype == np.float32 and np.array_equal(got, X.astype(np.float32))
    # min uint8 -> float32 is a numpy-safe cast, so the read takes the fast path.
    from cellstream.v2.reader_cpu import _read_min_dtypes

    md, _ = _read_min_dtypes(arch.manifest)
    assert np.can_cast(md, np.float32, "safe")


def test_lean_uint16_read_from_uint32_archive(tmp_path):
    X = np.array([[0, 3, 0], [5, 0, 200]], dtype=np.uint16)
    out = tmp_path / "lean"
    cellstream.write_sharded(_csr(X), out, n_shards=1)
    a = ShardedArchive(str(out)).to_anndata(data_dtype="uint16")  # cast down from uint32
    got = _dense(a)
    assert got.dtype == np.uint16 and np.array_equal(got, X)


def test_cast_down_to_uint8_requires_allow_lossy(tmp_path):
    X = np.array([[0, 300], [5, 0]], dtype=np.uint16)  # 300 > uint8 max; min uint16
    out = tmp_path / "u8"
    cellstream.write_sharded(_csr(X), out, n_shards=1)
    arch = ShardedArchive(str(out))
    with pytest.raises(LossyCastError):
        arch.to_anndata(data_dtype="uint8")  # allow_lossy defaults False
    a = arch.to_anndata(data_dtype="uint8", allow_lossy=True)
    assert _dense(a).dtype == np.uint8  # 300 wraps, but allow_lossy permitted it


def test_reverse_compat_fallback_when_min_absent(tmp_path):
    # Simulate a pre-min archive: strip the *_min keys; the reader falls back to the
    # on-disk dtype for cast safety and still reads correctly.
    X = np.array([[0, 3], [5, 0]], dtype=np.uint16)
    out = tmp_path / "legacy"
    # Tamper on a scaffolding DIRECTORY, then pack: rewriting a member in place changes
    # its byte length and would break a packed footer. The archive under test stays
    # packed, so this keeps exercising the PUBLIC reader (#154 part B).
    src = tmp_path / "legacy_src"
    write_directory(_csr(X), src, n_shards=1)
    mpath = src / "manifest.json"
    m = json.loads(mpath.read_text())
    m.pop("x_data_dtype_min", None)
    m.pop("x_indices_dtype_min", None)
    mpath.write_text(json.dumps(m))
    pack_directory(src, out)
    got = _dense(ShardedArchive(str(out)).to_anndata())
    assert np.array_equal(got.astype(np.uint16), X)
