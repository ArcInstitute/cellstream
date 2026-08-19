"""Single source of truth for index/indptr dtype dispatch in v0.2.

Every codepath that picks indices/indptr dtypes — full read, contiguous
subset, non-contiguous subset, kernel templating, CPU indptr prefix sum,
torch hand-off — calls _index_dtypes_for and uses its return verbatim.
This guarantees the PyTorch matching-dtype invariant holds everywhere.
"""

from __future__ import annotations

import numpy as np

_INT32_MAX = 2**31


def _index_dtypes_for(total_nnz: int, cast_back: bool, target: str) -> tuple[type, type]:
    """Return (indices_dtype, indptr_dtype) for the given (nnz, cast, target).

    Returns numpy dtype classes (np.uint16 / np.int32 / np.int64), NOT numpy.dtype
    instances — callers compare via `is` / `==`.

    Rules:
    - cast_back=False is the narrow path. Indices stay uint16, indptr stays int64.
      target='torch' with cast_back=False raises (PyTorch lacks a uint16 dtype).
    - cast_back=True (both cupy and torch targets): matching indices and indptr
      dtypes — both int32 when total_nnz < 2**31, both int64 otherwise. This
      mirrors the spec's memory math and satisfies torch's matching-dtype
      invariant in one helper.
    """
    if target not in ("cupy", "torch"):
        raise ValueError(f"target must be 'cupy' or 'torch', got {target!r}")
    if target == "torch" and not cast_back:
        raise ValueError(
            "target='torch' requires cast_back=True (PyTorch sparse tensors "
            "do not support uint16 data; use to_cupy_anndata(cast_back=False) "
            "for the narrow GPU path)."
        )
    if not cast_back:
        return np.uint16, np.int64
    matching = np.int32 if total_nnz < _INT32_MAX else np.int64
    return matching, matching
