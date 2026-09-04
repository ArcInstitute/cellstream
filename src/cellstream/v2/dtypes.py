"""Single source of truth for index/indptr dtype dispatch in v0.2.

Every codepath that picks indices/indptr dtypes — full read, contiguous
subset, non-contiguous subset, kernel templating, CPU indptr prefix sum,
torch hand-off — takes its DEFAULT from _index_dtypes_for, which is what
guarantees the PyTorch matching-dtype invariant. Two callers then treat that
default differently. ``v2.materialize.resolve_plan`` may OVERRIDE it (an explicit
``index_dtype``, or the uint32 narrow-read default above 65536 vars) and
re-checks the torch matching rule itself when it does. ``read._cupy_csr_to_torch``
re-applies it UNCONDITIONALLY on the DLPack hand-off, so on both live CSR
``to_torch`` paths an explicit ``index_dtype`` is not propagated and does not
CONTROL the tensor's index dtype -- it still matches whenever the request happens
to equal this module's answer. A dense ``to_torch`` bypasses that helper and has
no index/indptr dtype at all.
"""

from __future__ import annotations

import numpy as np

# EXCLUSIVE ceiling: the smallest ``total_nnz`` that no longer fits int32, whose max
# is ``2**31 - 1``. Always compared with ``<`` / ``>=``, never ``<=``.
_INT32_CEIL = 2**31


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
    matching = np.int32 if total_nnz < _INT32_CEIL else np.int64
    return matching, matching
