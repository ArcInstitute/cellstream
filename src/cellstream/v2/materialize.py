"""Single source of truth for reader materialization: the (container, data dtype,
index dtype) matrix + fail-loud lossy policy.

resolve_plan() runs once in the parent (resolves + validates dtypes; lossy
enforcement is deferred to cast_data()). cast_data() runs in each decode worker
(the cast/scatter seam a future Rust pass will own)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cellstream.narrow import lossless_cast
from cellstream.v2.dtypes import _index_dtypes_for

_INT32_MAX = 2**31  # used by resolve_plan's indptr/torch sizing (wired in Task 2)

# dtypes we accept as an explicit data_dtype request.
_ALLOWED_DATA_KINDS = frozenset({"f", "i", "u"})  # float, signed int, unsigned int


@dataclass(frozen=True)
class MaterializePlan:
    container: str  # "csr" | "dense"
    data_dtype: np.dtype  # concrete X.data / dense-value dtype
    index_dtype: np.dtype  # column-indices dtype (CSR only)
    indptr_dtype: np.dtype  # follows cast_back policy; sized from manifest total_nnz
    allow_lossy: bool
    n_vars: int  # dense buffer width


def _validate_numeric_dtype(dt: np.dtype, *, where: str) -> None:
    if dt.kind not in _ALLOWED_DATA_KINDS:
        raise ValueError(
            f"{where}: unsupported dtype {dt!r}; expected a numeric float/int/uint dtype."
        )


def resolve_plan(
    manifest: dict, *, container, data_dtype, index_dtype, allow_lossy, cast_back, target
):
    """Resolve the four kwargs + manifest on-disk dtypes into a MaterializePlan.

    target in {"cpu", "cupy", "torch"}. The full index/indptr + torch logic is
    completed in Task 2; this task wires container + data_dtype + the dataclass.
    """
    if container not in ("csr", "dense"):
        raise ValueError(f"container must be 'csr' or 'dense'; got {container!r}")
    if target not in ("cpu", "cupy", "torch"):
        raise ValueError(f"target must be 'cpu', 'cupy', or 'torch'; got {target!r}")

    on_disk_data = np.dtype(manifest.get("x_data_dtype_on_disk", "uint16"))
    # The minimum losslessly-representable data dtype recorded at write (fallback to
    # on-disk for pre-min archives, where on-disk IS the minimum).
    min_data = np.dtype(manifest.get("x_data_dtype_min", on_disk_data))
    if data_dtype is not None:
        resolved_data = np.dtype(data_dtype)
        _validate_numeric_dtype(resolved_data, where="data_dtype")
    elif cast_back:
        # cast_back = "float32 for counts". Integer-routed data now stores as uint32
        # on disk (previously uint16); float32 it back ONLY when the recorded min fits
        # float32 losslessly (the common count case: min uint8/uint16). Large uint32
        # counts that are not float32-exact, and float on-disk data, keep their dtype.
        resolved_data = (
            np.dtype(np.float32)
            if on_disk_data.kind == "u" and np.can_cast(min_data, np.float32, "safe")
            else on_disk_data
        )
    else:
        resolved_data = on_disk_data

    # --- index/indptr resolution ---
    total_nnz = int(manifest["total_nnz"])
    n_vars = int(manifest["n_vars"])
    idx_default, indptr_default = _index_dtypes_for(
        total_nnz=total_nnz,
        cast_back=cast_back,
        target=("torch" if target == "torch" else "cupy"),
    )
    # The narrow path (cast_back=False) defaults column indices to uint16, but uint16
    # only holds them when n_vars <= 65536 (max col index n_vars-1 <= 65535). Wide
    # archives store uint32 indices on disk, so a uint16 narrow read raises
    # LossyCastError (or silently truncates under allow_lossy). Mirror the write side
    # (narrow.decide_on_disk_dtypes sets min_index = uint16 iff n_vars <= 65536, else
    # uint32) so the default stays lossless. cast_back=True is unaffected (int32/int64
    # hold any n_vars).
    if not cast_back and n_vars > 65536:
        idx_default = np.uint32
    resolved_index = np.dtype(index_dtype) if index_dtype is not None else np.dtype(idx_default)
    resolved_indptr = np.dtype(indptr_default)

    if target == "torch":
        # torch requires int32/int64 col indices and crow (indptr) of the SAME dtype.
        if resolved_index.kind != "i" or resolved_index.itemsize not in (4, 8):
            raise ValueError(
                f"target='torch' requires index_dtype int32 or int64; got {resolved_index}."
            )
        if resolved_index == np.dtype(np.int32) and total_nnz >= _INT32_MAX:
            raise ValueError(
                f"target='torch' index_dtype int32 cannot hold total_nnz={total_nnz} "
                f"(>= 2**31); use index_dtype='int64'."
            )
        resolved_indptr = resolved_index  # crow/col dtype must match

    return MaterializePlan(
        container=container,
        data_dtype=resolved_data,
        index_dtype=resolved_index,
        indptr_dtype=resolved_indptr,
        allow_lossy=bool(allow_lossy),
        n_vars=n_vars,
    )


def _cast_is_always_safe(src: np.dtype, dst: np.dtype) -> bool:
    """True iff numpy classifies src->dst as a 'safe' cast (no overflow/domain error; int->float may still lose precision above 2**53)."""
    return np.can_cast(src, dst, casting="safe")


def _is_integer_narrowing(src: np.dtype, dst: np.dtype) -> bool:
    """True iff dst is an integer target that cannot losslessly hold every value of the
    integer src dtype -- an int->int cast whose safety rests on the data, not the width."""
    return src.kind in "iu" and dst.kind in "iu" and not np.can_cast(src, dst, casting="safe")


def cast_data(arr: np.ndarray, dst_dtype, *, allow_lossy: bool, where: str, safe: bool = False):
    """Cast arr to dst_dtype with the three-tier lossy policy (the worker seam).

    0. ``safe=True`` -> plain astype. The caller proved the cast is lossless from the
       recorded min dtype (e.g. an always-uint32 on-disk stream whose values fit uint16),
       so skip the per-element check the dtype-pair test below would otherwise force
       (``np.can_cast(uint32, float32, "safe")`` is False). EXCEPTION (#174): when the
       cast is an integer->integer NARROWING, the min claim is the only thing permitting
       it, and a tampered/corrupt footer would silently truncate -- so verify per-element
       (lossless_cast) rather than trust ``safe`` on that seam. Float/upcast targets are
       unaffected (arr.dtype -> float is not an integer narrowing), keeping the default
       read fast.
    1. dtype-safe pair -> plain astype (no per-element check).
    2. not safe + allow_lossy -> plain astype (caller accepts loss).
    3. not safe + not allow_lossy -> value-level lossless_cast (raises LossyCastError
       unless the actual values happen to round-trip).
    """
    dst = np.dtype(dst_dtype)
    if arr.dtype == dst:
        return arr
    # #174: a `safe` verdict derived from the recorded x_*_dtype_min is only as
    # trustworthy as the footer. On an INTEGER-NARROWING cast (dst can't hold every value
    # of arr's decoded full width), a tampered/corrupt min that under-claims the true
    # range would make a blind astype silently truncate. Verify per-element on exactly
    # that seam -- honest archives pass; the O(nnz) scan runs only on an integer
    # narrow-target read, never the default float/upcast path. Python mirror of the Rust
    # high-plane zero check (#170 / PR4b).
    if safe and not allow_lossy and _is_integer_narrowing(arr.dtype, dst):
        return lossless_cast(arr, dst, where=where)
    if safe or _cast_is_always_safe(arr.dtype, dst):
        return arr.astype(dst)
    if allow_lossy:
        return arr.astype(dst)
    return lossless_cast(arr, dst, where=where)


def cast_data_into(
    arr: np.ndarray, out: np.ndarray, *, allow_lossy: bool, where: str, safe: bool = False
) -> None:
    """``cast_data``'s three-tier policy, writing into ``out`` instead of allocating (#244).

    The bulk cell fallback worker decodes row-by-row into preallocated shm slices; routing
    through ``cast_data`` there allocates one temporary per row and then copies it into the
    slice. This applies the SAME policy -- and raises the SAME LossyCastError -- then writes
    straight into ``out``. ``out.dtype`` is the destination dtype, so there is no dst_dtype
    parameter. The two validating tiers still allocate inside ``lossless_cast``; they are the
    rare paths and correctness beats an allocation there.
    """
    # np.copyto BROADCASTS, so a size-1 arr would silently fill a longer out instead of raising
    # (verified). Fail fast: this is the allocation-free equivalent of cast_data, and cast_data
    # returns an array whose shape always matches its input. (Copilot #247 r4.)
    if out.shape != arr.shape:
        raise ValueError(f"cast_data_into: out.shape {out.shape} must equal arr.shape {arr.shape}")
    dst = out.dtype
    if arr.dtype == dst:
        np.copyto(out, arr)
        return
    # #174: a `safe` verdict on an integer NARROWING rests on the footer's recorded min, which
    # a tampered archive controls -- verify per-element on exactly that seam (mirrors cast_data).
    if safe and not allow_lossy and _is_integer_narrowing(arr.dtype, dst):
        np.copyto(out, lossless_cast(arr, dst, where=where))
        return
    if safe or _cast_is_always_safe(arr.dtype, dst) or allow_lossy:
        # casting="unsafe" reproduces astype's truncation exactly for the accepted-loss tier.
        np.copyto(out, arr, casting="unsafe")
        return
    np.copyto(out, lossless_cast(arr, dst, where=where))
