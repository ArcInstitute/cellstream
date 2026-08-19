"""Import-guard + dispatch helpers for the optional Rust encode core.

The Rust extension (cellstream.v2._rust) is an acceleration layer. When it is
absent (no toolchain at build time) or disabled, the Python encoder runs
unchanged. Only the byte-filter codec is accelerated.
"""

from __future__ import annotations

import os

import numpy as np

from cellstream.errors import LossyCastError
from cellstream.v2.codec import BYTE_FILTER_RAWDATA_SHUFFLE_NAME, BYTE_FILTER_SHUFFLE_NAME

try:
    from cellstream.v2 import _rust  # type: ignore

    _HAVE_RUST = True
except ImportError:
    _rust = None
    _HAVE_RUST = False


def rust_available() -> bool:
    """True iff the Rust core should be used (built, not disabled by env)."""
    return _HAVE_RUST and os.environ.get("CELLSTREAM_DISABLE_RUST") != "1"


# Source dtype combos the Rust core handles ({u2,u4,f4,f8} data × {i4,i8} indices).
_RUST_DATA_DTYPES = {"uint16", "uint32", "float32", "float64"}
_RUST_INDEX_DTYPES = {"int32", "int64"}


def _dtypes_supported(X) -> bool:
    return str(X.data.dtype) in _RUST_DATA_DTYPES and str(X.indices.dtype) in _RUST_INDEX_DTYPES


def _dense_dtype_supported(arr) -> bool:
    return str(arr.dtype) in _RUST_DATA_DTYPES


def use_rust_for(shuffle: str, X=None) -> bool:
    """True iff the Rust core should handle this write: built, not disabled by env,
    byte-filter codec, and — when X is given — a source dtype combo Rust supports.
    An unsupported dtype falls back to the Python encoder (spec: unsupported dtype
    → fallback), so the Rust side's dtype error is only a belt-and-suspenders guard.
    """
    if not (rust_available() and shuffle == BYTE_FILTER_SHUFFLE_NAME):
        return False
    return X is None or _dtypes_supported(X)


def use_rust_for_dense(shuffle, dense) -> bool:
    """True iff the Rust dense entry should handle this write."""
    return (
        rust_available() and shuffle == BYTE_FILTER_SHUFFLE_NAME and _dense_dtype_supported(dense)
    )


def encode_inmem(
    X, out_dir, row_lists, n_vars, shuffle, n_threads, data_dtype=None, index_dtype=None
):
    """Call the Rust core; re-raise its lossy error as LossyCastError.

    data_dtype / index_dtype (np.dtype or None): when given, the Rust core skips
    its own narrow scan and writes that on-disk dtype (the decide-once path).
    """
    lists = [np.asarray(rl, dtype=np.int64) for rl in row_lists]
    try:
        nnz, ddt, idt = _rust.encode_shards_inmem(
            X.data,
            X.indices,
            X.indptr,
            lists,
            str(out_dir),
            n_vars=int(n_vars),
            shuffle=shuffle,
            n_threads=int(n_threads),
            data_dtype=(None if data_dtype is None else np.dtype(data_dtype).name),
            index_dtype=(None if index_dtype is None else np.dtype(index_dtype).name),
        )
        return nnz, ddt, idt
    except ValueError as e:
        if "lossy" in str(e):
            raise LossyCastError(str(e)) from e
        raise


def encode_inmem_dense(dense, out_dir, row_lists, n_vars, shuffle, n_threads, data_dtype=None):
    """Call the Rust dense core; re-raise lossy errors as LossyCastError.

    data_dtype (np.dtype or None): when given, the Rust core skips its data scan.
    """
    lists = [np.asarray(rl, dtype=np.int64) for rl in row_lists]
    try:
        nnz, ddt, idt = _rust.encode_shards_inmem_dense(
            np.ascontiguousarray(dense),
            lists,
            str(out_dir),
            n_vars=int(n_vars),
            shuffle=shuffle,
            n_threads=int(n_threads),
            data_dtype=(None if data_dtype is None else np.dtype(data_dtype).name),
        )
        return nnz, ddt, idt
    except ValueError as e:
        if "lossy" in str(e):
            raise LossyCastError(str(e)) from e
        raise


def encode_inmem_dense_via_csr(
    dense, out_dir, row_lists, n_vars, shuffle, n_threads, data_dtype=None
):
    """Call the fused Rust dense→CSR→encode core; re-raise lossy errors as LossyCastError.

    Mirrors encode_inmem_dense, but the Rust side converts dense→CSR in-process
    (parallel) and encodes via the shared CSR core, avoiding the scattered dense
    reads of the grouped direct path.
    """
    lists = [np.asarray(rl, dtype=np.int64) for rl in row_lists]
    try:
        nnz, ddt, idt = _rust.encode_shards_inmem_dense_via_csr(
            np.ascontiguousarray(dense),
            lists,
            str(out_dir),
            n_vars=int(n_vars),
            shuffle=shuffle,
            n_threads=int(n_threads),
            data_dtype=(None if data_dtype is None else np.dtype(data_dtype).name),
        )
        return nnz, ddt, idt
    except ValueError as e:
        if "lossy" in str(e):
            raise LossyCastError(str(e)) from e
        raise


def decide_dtypes(X, n_vars, n_threads):
    """Single parallel data scan for a CSR source; returns
    (on_disk_data, on_disk_index, min_data, min_index) as np.dtype, mirroring
    narrow.decide_on_disk_dtypes. Integer-routable data -> on-disk uint32; the data
    min is the narrowest lossless uint (or native float). Index dtypes derive from
    n_vars (no index scan): on-disk uint32, min uint16 if n_vars<=65536 else uint32.

    A cheap negativity-only guard on the n_vars > 65536 path fails loud on a malformed
    (negative) column index (the uint16-vs-uint32 width scan is gone; indices are always
    uint32 on disk). n_vars <= 65536 trusts the CSR invariant and never inspects indices.
    """
    if n_vars > 65536 and X.indices.size and int(X.indices.min()) < 0:
        raise LossyCastError(
            f"X.indices: column indices must be >= 0 for unsigned storage; "
            f"got min {int(X.indices.min())}."
        )
    od_data, min_data = _rust.decide_dtypes(X.data, n_threads=int(n_threads))
    on_disk_index = np.dtype(np.uint32)
    min_index = np.dtype(np.uint16) if n_vars <= 65536 else np.dtype(np.uint32)
    return np.dtype(od_data), on_disk_index, np.dtype(min_data), min_index


def decide_dtypes_dense(dense, n_vars, n_threads):
    """Single parallel data scan for a DENSE source; returns
    (on_disk_data, on_disk_index, min_data, min_index) as np.dtype. Index dtypes
    derive from n_vars (no index scan)."""
    od_data, min_data = _rust.decide_dtypes_dense(
        np.ascontiguousarray(dense), n_threads=int(n_threads)
    )
    on_disk_index = np.dtype(np.uint32)
    min_index = np.dtype(np.uint16) if n_vars <= 65536 else np.dtype(np.uint32)
    return np.dtype(od_data), on_disk_index, np.dtype(min_data), min_index


# (data, index) OUTPUT dtypes the Rust decode pyfns cover. MUST stay in sync with the
# match arms in rust/src/lib.rs: decode_v2_csr covers the data x index cross product and
# decode_v2_dense covers the data set. A dtype that passes this gate but is absent from a
# pyfn arm raises instead of falling back. float16 / int8 / int16 / uint8 are NOT covered
# -> Python fallback (spec sec.7); float16 in particular has no native Rust carrier.
_RUST_READ_DATA = frozenset({"float32", "float64", "uint16", "uint32", "int32", "int64"})
_RUST_READ_INDEX = frozenset({"uint16", "uint32", "int32", "int64"})


#: Per-shard shuffle markers the Rust decode core covers: the original byte-filter
#: (shuffled data) and the #142 rawdata variant (float data raw). Both are decoded
#: self-describingly per shard by the Rust core (rust/src/decode.rs::data_raw_from_marker).
_RUST_READ_SHUFFLE_MARKERS = frozenset({BYTE_FILTER_SHUFFLE_NAME, BYTE_FILTER_RAWDATA_SHUFFLE_NAME})


def use_rust_for_read(shuffle: str, plan) -> bool:
    """True iff the Rust decode core should handle this read: built, not disabled,
    a byte-filter codec marker (shuffled OR #142 rawdata), and a (container, data_dtype,
    index_dtype) combo the Rust core covers. Any uncovered case falls back to the Python
    reader (spec sec.7)."""
    if not (rust_available() and shuffle in _RUST_READ_SHUFFLE_MARKERS):
        return False
    # Dense: the scatter has no index stream in its output (column indices are used
    # internally as i64), so only the output data dtype gates here.
    if plan.container == "dense":
        return np.dtype(plan.data_dtype).name in _RUST_READ_DATA
    if plan.container != "csr":
        return False
    return (
        np.dtype(plan.data_dtype).name in _RUST_READ_DATA
        and np.dtype(plan.index_dtype).name in _RUST_READ_INDEX
    )


def decode_v2_csr(
    paths,
    bases,
    out_lens,
    row_lens,
    shard_rows,
    shard_nnz,
    sel_starts,
    sel_stops,
    row_idx_flat,
    out_data,
    out_indices,
    out_indptr,
    *,
    n_vars,
    row_gather=False,
    data_safe=True,
    index_safe=True,
    allow_lossy=False,
    n_threads,
    data_min_dtype=None,
    index_min_dtype=None,
):
    """Call the Rust CSR decoder (writes into out_* in place).

    out_lens/row_lens: realized per-shard output nnz / row counts (the selected
      subset). shard_rows/shard_nnz: each shard's FULL n_rows / nnz (trusted bounds
      used by Rust to validate the in-worker header before any allocation).
    Range mode (row_gather=False): sel_starts/sel_stops give each shard's local
      [lstart, lstop); row_idx_flat is empty. Rows mode (row_gather=True):
      row_idx_flat is the concatenated per-shard local row indices (sliced by
      row_lens); sel_starts/sel_stops are ignored.
    data_safe/index_safe: whether the on-disk -> target cast is numpy-safe (no check
      needed). allow_lossy: permit a lossy cast. When a cast is neither safe nor
      allow_lossy, Rust verifies each element losslessly and raises LossyCastError.
    """
    try:
        _rust.decode_v2_csr(
            [str(p) for p in paths],
            [int(b) for b in bases],
            [int(x) for x in out_lens],
            [int(x) for x in row_lens],
            [int(x) for x in shard_rows],
            [int(x) for x in shard_nnz],
            [int(x) for x in sel_starts],
            [int(x) for x in sel_stops],
            np.ascontiguousarray(row_idx_flat, dtype=np.int64),
            out_data,
            out_indices,
            out_indptr,
            n_vars=int(n_vars),
            row_gather=bool(row_gather),
            data_safe=bool(data_safe),
            index_safe=bool(index_safe),
            allow_lossy=bool(allow_lossy),
            n_threads=int(n_threads),
            data_min_dtype=(None if data_min_dtype is None else str(data_min_dtype)),
            index_min_dtype=(None if index_min_dtype is None else str(index_min_dtype)),
        )
    except ValueError as e:
        if "lossy" in str(e):
            raise LossyCastError(str(e)) from e
        raise


def decode_v2_dense(
    paths,
    bases,
    row_lens,
    shard_rows,
    shard_nnz,
    sel_starts,
    sel_stops,
    row_idx_flat,
    out_dense,
    *,
    n_vars,
    row_gather=False,
    data_safe=True,
    allow_lossy=False,
    n_threads,
):
    """Call the Rust dense decoder (scatters into out_dense in place).

    out_dense must be a pre-zeroed C-contiguous (n_rows_out, n_vars) array; row_lens
    are the realized per-shard output row counts (the selected subset).
    shard_rows/shard_nnz: each shard's FULL n_rows / nnz (trusted DoS bounds).
    Range mode (row_gather=False): sel_starts/sel_stops give each shard's local
    [lstart, lstop); row_idx_flat is empty. Rows mode (row_gather=True): row_idx_flat
    is the concatenated per-shard local row indices (sliced by row_lens).
    data_safe / allow_lossy: tier gate for the dense data cast (see decode_v2_csr).
    """
    try:
        _rust.decode_v2_dense(
            [str(p) for p in paths],
            [int(b) for b in bases],
            [int(x) for x in row_lens],
            [int(x) for x in shard_rows],
            [int(x) for x in shard_nnz],
            [int(x) for x in sel_starts],
            [int(x) for x in sel_stops],
            np.ascontiguousarray(row_idx_flat, dtype=np.int64),
            out_dense,
            n_vars=int(n_vars),
            row_gather=bool(row_gather),
            data_safe=bool(data_safe),
            allow_lossy=bool(allow_lossy),
            n_threads=int(n_threads),
        )
    except ValueError as e:
        if "lossy" in str(e):
            raise LossyCastError(str(e)) from e
        raise
