"""Narrow-dtype cast helpers.

The package writes X.data and X.indices as uint16 on disk when the values
fit; lossless_cast() is the single fail-loud gate that verifies the cast
is information-preserving before any disk write.
"""

from __future__ import annotations

import numpy as np

from cellstream.errors import LossyCastError


def lossless_cast(arr: np.ndarray, target_dtype, *, where: str) -> np.ndarray:
    """Cast ``arr`` to ``target_dtype``; raise LossyCastError if cast is lossy.

    Parameters
    ----------
    arr : numpy array
        Source data.
    target_dtype : dtype-like
        Target NumPy dtype (e.g. ``np.uint16``, ``"uint16"``).
    where : str
        Diagnostic label used in the error message (e.g. "X.data", "shard[3].X.indices").

    Returns
    -------
    numpy array with dtype == target_dtype, equal to arr when round-tripped.
    """
    target = np.dtype(target_dtype)
    # Semantic check BEFORE the bit-pattern check below: signed → unsigned
    # casts where the source bit width is ≤ target bit width
    # (e.g. int8/int16 → uint16) survive the round-trip equality check
    # even when negative values are destroyed: int8(-1) → uint16(65535)
    # → int8(-1) round-trips bit-for-bit, but the user's "-1 count" became
    # "65535 count" on disk. Catch the sign loss explicitly.
    if (
        np.issubdtype(target, np.unsignedinteger)
        and np.issubdtype(arr.dtype, np.signedinteger)
        and arr.size > 0
        and int(arr.min()) < 0
    ):
        raise LossyCastError(
            f"{where}: cast to {target} is lossy on this data "
            f"(source dtype={arr.dtype}, min={arr.min()}, max={arr.max()}; "
            f"negative values cannot be stored in an unsigned target)"
        )
    with np.errstate(invalid="ignore"):
        cast = arr.astype(target)
    if not np.array_equal(cast.astype(arr.dtype), arr, equal_nan=True):
        raise LossyCastError(
            f"{where}: cast to {target} is lossy on this data "
            f"(source dtype={arr.dtype}, min={arr.min()}, max={arr.max()})"
        )
    return cast


def _native_on_disk_data_dtype(values_dtype):
    """On-disk data dtype for non-integer-routable values: the native float width,
    else widen to float64 (e.g. signed/wide ints with negatives or out-of-uint32 range).

    Uses ``np.dtype(...).name`` (not ``str(...)``) so byte-order variants normalize
    (e.g. both ``>f4`` and ``<f4`` -> ``float32``).
    """
    return np.dtype(
        {
            "float32": np.float32,
            "float64": np.float64,
            "uint32": np.uint32,
            "uint16": np.uint16,
        }.get(np.dtype(values_dtype).name, np.float64)
    )


def _uint_dtype_from_max(hi: int):
    """Narrowest unsigned dtype (uint8/uint16/uint32) that holds a max value of ``hi``."""
    if hi <= 0xFF:
        return np.dtype(np.uint8)
    if hi <= 0xFFFF:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _fold_block_uint_max(block, hi):
    """Fold one data block into the running max ``hi`` for the narrowest-uint scan.

    Returns the updated max, or None if the block is NOT integer-routable (non-finite,
    fractional, negative, or > 2**32-1). An empty block leaves ``hi`` unchanged. This is
    the per-block core shared by ``_data_min_uint_dtype`` (in-memory) and
    ``data_min_uint_dtype_streaming`` (backed) so the two cannot drift.
    """
    if block.size == 0:
        return hi
    kind = block.dtype.kind
    is_float = kind == "f"
    is_unsigned = kind == "u"  # unsigned ints are never negative -> skip min()
    if is_float and not np.isfinite(block).all():
        return None
    if not is_unsigned and block.min() < 0:
        return None
    if is_float and not (block == np.floor(block)).all():
        return None
    bmax = int(block.max())
    if bmax > 0xFFFFFFFF:
        return None
    return bmax if bmax > hi else hi


def data_min_uint_dtype_streaming(blocks):
    """Streaming counterpart of ``_data_min_uint_dtype``.

    ``blocks`` is an iterable of numpy arrays (chunks of the data values, all one dtype).
    Returns the narrowest uint dtype (uint8/uint16/uint32) that losslessly holds every
    value, or None if ANY block is not integer-routable. An empty stream returns uint8
    (matching the in-memory empty case). Short-circuits on the first failing block.
    """
    hi = 0
    for b in blocks:
        hi = _fold_block_uint_max(np.asarray(b), hi)
        if hi is None:
            return None
    return _uint_dtype_from_max(hi)


def _data_min_uint_dtype(data, block: int = 1 << 20):
    """Narrowest unsigned-int dtype (uint8/uint16/uint32) that losslessly holds every
    value, or None when the data is NOT integer-routable (non-finite, non-integral,
    negative, or > 2**32-1 -> stored as native float).

    Block-processed (no full-array bool temporaries), early-exits on the first failing
    block, and tracks the running max to pick the narrowest width. Accepts N-D input;
    ``order="K"`` flattens C-/F-contiguous layouts as a view (no copy). Mirrors the Rust
    ``decide_data_dtypes`` scan (bound 2**32-1, min name uint8/uint16/uint32).
    """
    # bool / uint8: trivially integral, non-negative, <= 255 (covers empty too).
    if data.dtype.kind == "b" or (data.dtype.kind == "u" and data.dtype.itemsize == 1):
        return np.dtype(np.uint8)
    flat = np.ravel(data, order="K")
    if flat.size == 0:
        return np.dtype(np.uint8)
    hi = 0
    for i in range(0, flat.size, block):
        hi = _fold_block_uint_max(flat[i : i + block], hi)
        if hi is None:
            return None
    return _uint_dtype_from_max(hi)


def decide_on_disk_dtypes(data, indices, n_vars):
    """Mirror the Rust dtype policy for the Python fallback.

    Returns (on_disk_data, on_disk_index, min_data, min_index) as np.dtype.

    Storage: integer-routable data (finite, integral, non-negative, <= 2**32-1) ->
    on-disk uint32; otherwise native float (float32/float64; signed/wide int that
    doesn't fit -> float64). Indices -> on-disk uint32 always (trusts the CSR
    invariant 0 <= idx < n_vars; no index scan).

    Minimums (the narrowest dtype that losslessly holds the values, recorded so the
    reader can cast up/down cheaply): data -> the narrowest uint (uint8/uint16/uint32)
    or the native float; index -> uint16 if n_vars <= 65536 else uint32 (a scan-free
    upper bound from n_vars; never narrower than the true max column index).

    ``indices`` is used ONLY for a cheap negativity guard on the n_vars > 65536 path
    (no width scan — the on-disk index dtype is always uint32). For n_vars <= 65536 the
    CSR invariant 0 <= idx < n_vars <= 65536 is fully trusted (no index access), exactly
    as the #128 fast-path did.
    """
    # Fail loud on a malformed (negative) column index — same guard the old scan-based
    # n_vars > 65536 branch had, but negativity-only (the uint16-vs-uint32 WIDTH scan is
    # gone; indices are always uint32 on disk). The common n_vars <= 65536 path trusts
    # the invariant and never inspects indices.
    if n_vars > 65536 and indices is not None and indices.size and int(indices.min()) < 0:
        raise LossyCastError(
            f"X.indices: column indices must be >= 0 for unsigned storage; "
            f"got min {int(indices.min())}."
        )
    min_uint = _data_min_uint_dtype(data)
    if min_uint is not None:
        on_disk_data = np.dtype(np.uint32)
        min_data = min_uint
    else:
        on_disk_data = _native_on_disk_data_dtype(data.dtype)
        min_data = on_disk_data
    on_disk_index = np.dtype(np.uint32)
    min_index = np.dtype(np.uint16) if n_vars <= 65536 else np.dtype(np.uint32)
    return on_disk_data, on_disk_index, min_data, min_index
