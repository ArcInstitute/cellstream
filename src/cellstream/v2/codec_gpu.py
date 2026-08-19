"""Pure-cupy inverse of the v2 byte-filter codec (mirror of codec.py).

Operates on ALREADY-DECOMPRESSED device byte buffers (the output nvCOMP-RAW zstd
produces), so this module is unit-testable byte-exact against codec.py on any GPU
box without nvCOMP. cupy is imported inside functions so the module imports on
CPU-only hosts.

Reference: codec.byte_filter_decode / zstd_only_decode. On-disk stream layout is
plane-major (T_od, N): plane p is byte p of every element, laid out contiguously
(plane 0's N bytes, then plane 1's N bytes, ...). indices/indptr additionally get
a per-plane byte-delta (mod 256) after the shuffle; data is shuffle-only, and
FLOAT data (the "-rawdata" codec) is raw little-endian bytes (no shuffle/delta).
"""

from __future__ import annotations

import numpy as np

#: Roles that carry the per-plane byte-delta (mirror of codec._BYTEDELTA_ROLES).
_DELTA_ROLES = frozenset({"indices", "indptr"})


def _byte_undelta_gpu(planes_2d):
    """Inverse per-plane byte-delta: uint8 cumsum mod 256 along the element axis.

    codec._byte_undelta uses ``np.cumsum(dtype=uint8)`` (native wrap); cupy's uint8
    cumsum upcasts, so cumsum in int64 then mask to a byte — byte-identical.
    """
    import cupy as cp

    return (cp.cumsum(planes_2d.astype(cp.int64), axis=1) & 0xFF).astype(cp.uint8)


def _reassemble_gpu(planes_2d, out_dtype):
    """Combine the low byte-planes (T_eff, N) into (N,) of ``out_dtype``.

    Mirror of codec._byte_unshuffle's fast path: acc = p0 | p1<<8 | ... over the
    T_eff planes. The shift uses a Python int (NOT a numpy scalar) so the result
    keeps ``out_dtype`` — a np.uint32 shift scalar would promote uint16 -> uint32.
    """
    import cupy as cp

    odt = cp.dtype(out_dtype)
    acc = planes_2d[0].astype(odt)
    for p in range(1, planes_2d.shape[0]):
        acc = acc | (planes_2d[p].astype(odt) << (8 * p))
    return acc.astype(odt, copy=False)


def stream_inverse_gpu(
    decomp_chunks,
    *,
    on_disk_dtype,
    eff_dtype,
    role,
    is_raw,
    chunk_elems,
    check_high_planes=True,
):
    """Invert one stream's decompressed chunks into a 1-D cupy array of ``eff_dtype``.

    ``decomp_chunks``: list of 1-D cupy.uint8 arrays, one per chunk, each holding
    that chunk's decompressed on-disk bytes (plane-major, T_od * N bytes).

    ``eff_dtype`` may be narrower than ``on_disk_dtype`` (low-plane #161 fast path,
    u32-on-disk with a u16 min): only the low T_eff planes are reassembled and, when
    ``check_high_planes``, the discarded high planes are asserted all-zero (a delta-
    coded all-zero plane is itself all-zero, so the raw planes are checked directly).

    ``is_raw=True`` (float data, "-rawdata" codec): the decompressed bytes are the
    little-endian values — view as ``eff_dtype``, no shuffle/delta/reassemble.

    Full chunks (n == chunk_elems) are batched into one (n_full, T_eff, N) buffer
    for a single segmented cumsum + vectorized reassemble (the spike's transform_fused);
    the ragged tail chunk is handled separately. cellstream chunks a stream into
    chunk_elems pieces with only the LAST chunk partial, so [full..., tail] preserves
    element order.
    """
    import cupy as cp

    t_od = np.dtype(on_disk_dtype).itemsize
    t_eff = np.dtype(eff_dtype).itemsize
    odt = cp.dtype(eff_dtype)

    if is_raw:
        parts = [d.view(odt) for d in decomp_chunks if d.size]
        return cp.concatenate(parts) if parts else cp.empty(0, odt)

    is_delta = role in _DELTA_ROLES
    # cellstream chunks a stream as [full..., optional single trailing partial], so
    # batching all full chunks then the tail preserves element order. Reject a
    # malformed/foreign stream whose partial chunk is NOT the last non-empty one
    # (>1 partial, or a full chunk after a partial), which that batching would
    # silently reorder vs the CPU descriptor-order decode.
    non_empty = [d for d in decomp_chunks if d.size]
    partial_idx = [i for i, d in enumerate(non_empty) if d.size // t_od != chunk_elems]
    if partial_idx and (len(partial_idx) > 1 or partial_idx[0] != len(non_empty) - 1):
        raise ValueError(
            f"{role}: non-canonical chunk layout — expected [full..., optional trailing "
            f"partial], chunk_elems={chunk_elems}. Archive may be corrupt or foreign."
        )
    full = [d for d in non_empty if d.size // t_od == chunk_elems]
    tail = [d for d in non_empty if d.size // t_od != chunk_elems]
    parts = []

    if full:
        n = chunk_elems
        big = cp.empty((len(full), t_eff, n), cp.uint8)
        hi = None
        for j, d in enumerate(full):
            v = d.view(cp.uint8)[: t_od * n].reshape(t_od, n)
            if check_high_planes and t_eff < t_od:
                # Reduce each chunk's high planes to a 0-D bool on device (no (n,)
                # temporary); OR the scalars across chunks.
                hi = v[t_eff:].any() if hi is None else (hi | v[t_eff:].any())
            big[j] = v[:t_eff]
        if hi is not None and bool(hi):
            raise ValueError(
                f"{role}: high byte-plane is nonzero — on-disk {on_disk_dtype} does "
                f"not fit {eff_dtype} (archive/manifest inconsistent)."
            )
        if is_delta:
            big = (cp.cumsum(big.astype(cp.int64), axis=2) & 0xFF).astype(cp.uint8)
        acc = big[:, 0, :].astype(odt)
        for p in range(1, t_eff):
            acc = acc | (big[:, p, :].astype(odt) << (8 * p))
        parts.append(acc.astype(odt, copy=False).reshape(-1))

    for d in tail:
        n = d.size // t_od
        v = d.view(cp.uint8)[: t_od * n].reshape(t_od, n)
        if check_high_planes and t_eff < t_od and bool(v[t_eff:].any()):
            raise ValueError(
                f"{role}: high byte-plane is nonzero — on-disk {on_disk_dtype} does "
                f"not fit {eff_dtype} (archive/manifest inconsistent)."
            )
        pl = _byte_undelta_gpu(v[:t_eff]) if is_delta else v[:t_eff]
        parts.append(_reassemble_gpu(pl, eff_dtype))

    return cp.concatenate(parts) if parts else cp.empty(0, odt)
