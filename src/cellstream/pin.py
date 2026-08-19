"""Host-memory pinning for the DLPack streaming handoff (#160).

`cudaHostRegister` page-locks a decoded CSR array in place so a GPU consumer's
async H2D (`torch.Tensor.to(device, non_blocking=True)`) overlaps compute. cupy
is imported lazily here so cellstream imports cleanly on CPU-only hosts; when cupy
is unavailable or registration fails, callers fall back to pageable memory.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

_PIN_WARNED = False


def _warn_once(msg: str) -> None:
    global _PIN_WARNED
    if not _PIN_WARNED:
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        _PIN_WARNED = True


def pin_array_inplace(arr) -> Callable[[], None] | None:
    """Page-lock a C-contiguous numpy array in place via cudaHostRegister.

    Returns an idempotent zero-arg unregister callable on success, or None if
    cupy is unavailable, the array is empty or non-contiguous, or registration
    fails (warns once per process in the unavailable/failed cases). The caller
    continues with pageable memory when None is returned.
    """
    if arr.nbytes == 0 or not arr.flags["C_CONTIGUOUS"]:
        return None
    try:
        import cupy as cp
    except Exception:
        _warn_once(
            "cellstream: cupy unavailable; pinned=True falls back to pageable host "
            "memory (async H2D will not overlap)."
        )
        return None
    ptr = arr.ctypes.data
    try:
        cp.cuda.runtime.hostRegister(ptr, arr.nbytes, 0)  # cudaHostRegisterDefault
    except Exception:
        _warn_once(
            "cellstream: cudaHostRegister failed; pinned=True falls back to pageable host memory."
        )
        return None

    done = [False]

    def _unregister() -> None:
        if done[0]:
            return
        done[0] = True
        try:
            cp.cuda.runtime.hostUnregister(ptr)
        except Exception:
            pass  # interpreter teardown / buffer already gone — best effort

    return _unregister
