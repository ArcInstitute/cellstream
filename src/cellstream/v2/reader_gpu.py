"""v2 GPU reader: nvCOMP-RAW zstd + pure-cupy byte-filter inverse -> device CSR.

Decodes ONE shard at a time. Reads cellstream's standard libzstd-1 frames as-is via
nvCOMP RAW bitstream (no format change). cupy/nvcomp import inside functions so the
module imports on CPU-only hosts. Output dtypes come from ``materialize.resolve_plan``
(the same source of truth the CPU reader uses). See
docs/superpowers/specs/2026-07-02-gpu-device-resident-decode-design.md.
"""

from __future__ import annotations

import threading

import numpy as np

from cellstream.errors import GpuOutOfMemoryError, LossyCastError
from cellstream.manifest import shards_for_range
from cellstream.v2 import codec_gpu
from cellstream.v2.codec import BYTE_FILTER_RAWDATA_SHUFFLE_NAME, BYTE_FILTER_SHUFFLE_NAME
from cellstream.v2.format import parse_shx_header
from cellstream.v2.materialize import resolve_plan

_SUPPORTED_SHUFFLE = frozenset({BYTE_FILTER_SHUFFLE_NAME, BYTE_FILTER_RAWDATA_SHUFFLE_NAME})
_GPU_IMPORT_HINT = (
    "GPU decode needs cupy + nvCOMP. Install the GPU extra: pip install 'cellstream[gpu]' "
    "(cupy-cuda12x + nvidia-nvcomp-cu12)."
)


def _import_gpu():
    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(_GPU_IMPORT_HINT) from e
    try:
        try:
            from nvidia import nvcomp
        except Exception:
            import nvcomp  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(_GPU_IMPORT_HINT) from e
    codec = nvcomp.Codec(algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW)
    return cp, nvcomp, codec


def _check_codec(header):
    """Fail fast (import-free) on any non-byte-filter codec, matching #40 scope."""
    if header.shuffle not in _SUPPORTED_SHUFFLE:
        raise NotImplementedError(
            f"GPU decode supports v2-byte-filter only (got shuffle={header.shuffle!r}); "
            f"use the CPU reader: ShardedArchive.to_anndata()."
        )


_CSR_FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def require_csr_float(data_dtype):
    """A device (or host) sparse CSR requires float32/float64 data.

    ``cupyx.scipy.sparse`` accepts only bool/float32/float64/complex data, and
    ``scipy.sparse`` additionally rejects float16 — so the narrow (uint16) and
    float16 paths cannot be a CSR at all. Fail with actionable guidance instead
    of a cryptic cupyx/scipy ValueError. (Narrow/float16 are ``container='dense'``
    only.)
    """
    if np.dtype(data_dtype) not in _CSR_FLOAT_DTYPES:
        raise NotImplementedError(
            f"a device CSR requires float32/float64 data — cupyx/scipy sparse CSR does "
            f"not support {np.dtype(data_dtype)} (narrow uint16 / float16). Use "
            f"cast_back=True (float32), a float32/float64 data_dtype, or "
            f"container='dense' for a narrow/float16 device array."
        )


class _ReusableStaging:
    """Grow-on-demand pinned host + device uint8 buffers, reused across shards.

    Reuse is load-bearing: a per-call pinned alloc + host memcpy inflated H2D
    68 -> 275 ms in the spike. GPU decode is single-shard-serial-per-device by
    design; the lock guards the shared buffers so an accidental concurrent call
    fails safe (blocks) rather than overwriting an in-flight decode's input.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self._n = 0
        self._pin = None
        self._dev = None

    def get(self, nbytes, cp):
        if self._n < nbytes:
            self._pin = cp.cuda.alloc_pinned_memory(nbytes)
            self._dev = cp.empty(nbytes, cp.uint8)
            self._n = nbytes
        return self._pin, self._dev


_DEFAULT_STAGING = _ReusableStaging()


def _eff_dtype(on_disk, min_dt):
    """Low-plane #161: u32-on-disk with a u16/u8 min decodes to the min dtype."""
    return min_dt if (on_disk == "uint32" and min_dt in ("uint8", "uint16")) else on_disk


def _archive_locator(archive_path):
    """Return (get_loc, manifest) for a packed archive."""
    from cellstream.read import ShardedArchive

    sh = ShardedArchive(archive_path)

    def get_loc(idx):
        path, base, _len = sh._packed.shard_location(idx)
        return str(path), int(base)

    return get_loc, sh.manifest


def _cast_device(arr, dst_dtype, *, allow_lossy, safe, where):
    """Cast a device array mirroring materialize.cast_data / narrow.lossless_cast.

    ``safe`` (from np.can_cast(min_dtype, dst, "safe")) or ``allow_lossy`` -> plain
    astype (the hot path: no device value scan). Otherwise a value-level lossless
    check: reject signed->unsigned negatives and require an equal_nan roundtrip.
    """
    import cupy as cp

    dst = cp.dtype(dst_dtype)
    if arr.dtype == dst:
        return arr
    if safe or allow_lossy:
        return arr.astype(dst)
    if dst.kind == "u" and arr.dtype.kind == "i" and bool((arr < 0).any()):
        raise LossyCastError(f"{where}: negative values cannot be cast to unsigned {dst}.")
    cast = arr.astype(dst)
    back = cast.astype(arr.dtype)
    eq = back == arr
    if arr.dtype.kind == "f":
        eq = eq | (cp.isnan(arr) & cp.isnan(back))
    if not bool(eq.all()):
        raise LossyCastError(
            f"{where}: casting {arr.dtype} -> {dst} loses data; pass allow_lossy=True."
        )
    return cast


def _validate_csr_structure(data, indices, indptr, *, nnz, n_rows, n_vars):
    """Structural CSR validation (CPU fail-loud parity), batched to ONE host sync.

    Every check is computed as a device scalar/bool, stacked, and read back with a
    single ``.get()`` — vs the prior ~5 separate ``bool()`` / ``int()`` D2H syncs.
    Semantics are identical: raises ValueError on any structural violation.
    """
    import cupy as cp

    if data.shape[0] != nnz or indices.shape[0] != nnz:
        raise ValueError(
            f"decoded nnz mismatch: data={data.shape[0]} indices={indices.shape[0]} "
            f"vs header.nnz={nnz} (shard may be corrupt)."
        )
    if indptr.shape[0] != n_rows + 1:
        raise ValueError(
            f"decoded indptr length {indptr.shape[0]} vs n_rows+1={n_rows + 1} "
            f"(shard may be corrupt)."
        )
    # All device reductions first, then a single stacked D2H sync.
    ip0 = indptr[0] != 0
    ipm = indptr[-1] != nnz
    nondec = (indptr[1:] < indptr[:-1]).any() if n_rows else cp.asarray(False)
    if nnz:
        idx_oob = (indices >= n_vars).any()
        # Byte-filter indices decode unsigned; stay robust to a signed index dtype.
        idx_neg = (indices < 0).any() if indices.dtype.kind != "u" else cp.asarray(False)
    else:
        idx_oob = cp.asarray(False)
        idx_neg = cp.asarray(False)
    # cp.stack (not cp.asarray) — a list of cupy 0-D scalars can't be built by
    # asarray (object dtype -> ValueError); stack concatenates on-device. Unpack the
    # numpy bool array directly (numpy.bool_ is truthy-equivalent to bool).
    ip0_bad, ipm_bad, nondec_bad, oob_bad, neg_bad = cp.stack(
        [ip0, ipm, nondec, idx_oob, idx_neg]
    ).get()
    if ip0_bad or ipm_bad:
        raise ValueError(
            f"indptr[0]/indptr[-1] inconsistent with nnz={nnz} (shard may be corrupt)."
        )
    if nondec_bad:
        raise ValueError("indptr is not non-decreasing (shard may be corrupt).")
    if oob_bad:
        raise ValueError(f"a column index is >= n_vars={n_vars} (shard may be corrupt).")
    if neg_bad:
        raise ValueError("a column index is negative (shard may be corrupt).")


def decode_shard_to_device(path, base, header, *, manifest, plan, staging=None):
    """Decode one shard to (d_data, d_indices, d_indptr) cupy arrays in plan dtypes.

    Byte-exact vs the CPU decode. Validates every chunk's decompressed size against
    the header (fail-loud, matching codec._decompress_checked) and the final stream
    lengths against nnz / n_rows+1.
    """
    _check_codec(header)
    cp, nv, codec = _import_gpu()
    staging = staging or _DEFAULT_STAGING

    is_raw_data = header.shuffle == BYTE_FILTER_RAWDATA_SHUFFLE_NAME
    data_min = manifest.get("x_data_dtype_min") or header.data.dtype
    idx_min = manifest.get("x_indices_dtype_min") or header.indices.dtype
    data_eff = _eff_dtype(header.data.dtype, data_min)
    idx_eff = _eff_dtype(header.indices.dtype, idx_min)
    streams = [
        ("data", header.data, header.data.dtype, data_eff, is_raw_data),
        ("indices", header.indices, header.indices.dtype, idx_eff, False),
        ("indptr", header.indptr, header.indptr.dtype, header.indptr.dtype, False),
    ]
    # Live chunks = decompressed_size > 0 (matches codec's n_elements==0 early-out).
    live = [
        (role, s, on_disk, eff, is_raw, [c for c in s.chunks if c.decompressed_size > 0])
        for role, s, on_disk, eff, is_raw in streams
    ]
    all_live = [c for *_rest, chunks in live for c in chunks]
    if not all_live:  # empty shard: only valid when it has no rows and no nnz
        if header.nnz != 0 or header.n_rows != 0:
            raise ValueError(
                f"shard has no non-empty chunks but header.nnz={header.nnz}, "
                f"n_rows={header.n_rows} (shard may be corrupt)."
            )
        return (
            cp.empty(0, cp.dtype(plan.data_dtype)),
            cp.empty(0, cp.dtype(plan.index_dtype)),
            cp.zeros(header.n_rows + 1, cp.dtype(plan.indptr_dtype)),
        )
    for c in all_live:
        if c.compressed_size <= 0:
            raise ValueError(
                f"chunk has decompressed_size={c.decompressed_size} but "
                f"compressed_size={c.compressed_size} (shard may be corrupt)."
            )
    lo = min(c.offset for c in all_live)
    hi = max(c.offset + c.compressed_size for c in all_live)

    out = {}
    with staging.lock:  # held across the whole decode: nvCOMP consumes `dev` async
        nb = hi - lo
        pin, dev = staging.get(nb, cp)
        hv = np.frombuffer(pin, np.uint8, nb)
        # Read the compressed span STRAIGHT into pinned host memory: readinto avoids
        # a redundant multi-hundred-MB host->host memcpy (profiled at ~55 ms/shard,
        # the dominant "other" cost) that a fresh fh.read() + `hv[:] = ...` incurred.
        with open(path, "rb") as fh:
            fh.seek(base + lo)
            got = fh.readinto(hv)
        if got != nb:
            raise ValueError(
                f"short read: got {got} of {nb} bytes at offset {base + lo} "
                f"(shard may be truncated)."
            )
        dev[:nb].set(hv)  # pinned H2D
        nv_in = [
            nv.as_array(dev[c.offset - lo : c.offset - lo + c.compressed_size]) for c in all_live
        ]
        nv_out = codec.decode(nv_in)
        k = 0
        for role, _s, on_disk, eff, is_raw, chunks in live:
            t_od = np.dtype(on_disk).itemsize
            decs = []
            for j, c in enumerate(chunks):
                d = cp.asarray(nv_out[k + j])
                if d.nbytes != c.decompressed_size:
                    raise ValueError(
                        f"{role} chunk {j}: nvCOMP decompressed {d.nbytes} bytes, "
                        f"expected {c.decompressed_size} (shard may be corrupt)."
                    )
                if c.decompressed_size % t_od != 0:
                    raise ValueError(
                        f"{role} chunk {j}: decompressed_size {c.decompressed_size} "
                        f"not divisible by itemsize {t_od} (shard may be corrupt)."
                    )
                decs.append(d)
            k += len(chunks)
            out[role] = codec_gpu.stream_inverse_gpu(
                decs,
                on_disk_dtype=on_disk,
                eff_dtype=eff,
                role=role,
                is_raw=is_raw,
                chunk_elems=header.chunk_size_elements,
            )
        cp.cuda.runtime.deviceSynchronize()  # ensure `dev` consumed before releasing lock

    _validate_csr_structure(
        out["data"],
        out["indices"],
        out["indptr"],
        nnz=header.nnz,
        n_rows=header.n_rows,
        n_vars=header.n_vars,
    )

    data_safe = bool(np.can_cast(np.dtype(data_eff), np.dtype(plan.data_dtype), casting="safe"))
    idx_safe = bool(np.can_cast(np.dtype(idx_eff), np.dtype(plan.index_dtype), casting="safe"))
    d_data = _cast_device(
        out["data"], plan.data_dtype, allow_lossy=plan.allow_lossy, safe=data_safe, where="data"
    )
    d_indices = _cast_device(
        out["indices"],
        plan.index_dtype,
        allow_lossy=plan.allow_lossy,
        safe=idx_safe,
        where="indices",
    )
    d_indptr = out["indptr"].astype(cp.dtype(plan.indptr_dtype), copy=False)
    return d_data, d_indices, d_indptr


def _make_csr(data, indices, indptr, shape):
    """Build a cupyx CSR preserving the array dtypes (assign, don't reconstruct)."""
    import cupy as cp
    import cupyx.scipy.sparse as cusp

    out = cusp.csr_matrix(shape, dtype=cp.dtype(data.dtype))
    out.data = data
    out.indices = indices
    out.indptr = indptr
    return out


def _stitch(blocks, n_vars, plan):
    """Concatenate per-shard (data, indices, indptr) blocks into one device CSR.

    Globalizes indptr (each shard's indptr is shard-local). dtype-preserving (avoids
    cusp.vstack canonicalization).
    """
    import cupy as cp

    data = cp.concatenate([b[0] for b in blocks])
    indices = cp.concatenate([b[1] for b in blocks])
    total_rows = sum(b[2].shape[0] - 1 for b in blocks)
    indptr = cp.empty(total_rows + 1, cp.dtype(plan.indptr_dtype))
    indptr[0] = 0
    row = nnz = 0
    for bd, _bi, bp in blocks:
        r = bp.shape[0] - 1
        indptr[row + 1 : row + r + 1] = bp[1:] + nnz
        row += r
        nnz += len(bd)  # == bp[-1], host-side -> no per-block D2H sync (Gemini)
    return _make_csr(data, indices, indptr, (total_rows, n_vars))


def read_v2_shard_to_device(
    archive_path,
    shard_idx,
    *,
    cast_back=True,
    data_dtype=None,
    index_dtype=None,
    allow_lossy=False,
    staging=None,
):
    """Decode a single shard into a device-resident cupy CSR."""
    get_loc, man = _archive_locator(archive_path)
    path, base = get_loc(shard_idx)
    with open(path, "rb") as fh:
        fh.seek(base)
        header = parse_shx_header(fh)
    _check_codec(header)  # import-free fail-fast on unsupported codec
    plan = resolve_plan(
        man,
        container="csr",
        data_dtype=data_dtype,
        index_dtype=index_dtype,
        allow_lossy=allow_lossy,
        cast_back=cast_back,
        target="cupy",
    )
    require_csr_float(plan.data_dtype)
    blk = decode_shard_to_device(path, base, header, manifest=man, plan=plan, staging=staging)
    return _stitch([blk], man["n_vars"], plan)


def read_v2_to_device(
    archive_dir,
    *,
    cast_back=True,
    n_streams=4,
    data_dtype=None,
    index_dtype=None,
    allow_lossy=False,
    _override_projected_bytes=None,
):
    """Read a full v2 archive into one device CSR (preflight-gated, one shard at a time)."""
    get_loc, man = _archive_locator(archive_dir)
    # Fail fast (import-free) on an unsupported codec: a bitshuffle archive should
    # say "use the CPU reader", not "install cupy".
    p0, b0 = get_loc(0)
    with open(p0, "rb") as fh:
        fh.seek(b0)
        _check_codec(parse_shx_header(fh))
    plan = resolve_plan(
        man,
        container="csr",
        data_dtype=data_dtype,
        index_dtype=index_dtype,
        allow_lossy=allow_lossy,
        cast_back=cast_back,
        target="cupy",
    )
    require_csr_float(plan.data_dtype)
    cp, _nv, _codec = _import_gpu()
    projected = _project_total_bytes(man, plan)
    _enforce_preflight(
        _override_projected_bytes if _override_projected_bytes is not None else projected
    )

    n_obs, total_nnz, n_vars = man["n_obs_total"], man["total_nnz"], man["n_vars"]
    d_data = cp.empty(total_nnz, cp.dtype(plan.data_dtype))
    d_indices = cp.empty(total_nnz, cp.dtype(plan.index_dtype))
    d_indptr = cp.empty(n_obs + 1, cp.dtype(plan.indptr_dtype))
    d_indptr[0] = 0
    staging = _ReusableStaging()
    cum_nnz = cum_rows = 0
    for si in range(man["n_shards"]):
        path, base = get_loc(si)
        with open(path, "rb") as fh:
            fh.seek(base)
            header = parse_shx_header(fh)
        bd, bi, bp = decode_shard_to_device(
            path, base, header, manifest=man, plan=plan, staging=staging
        )
        s_nnz, s_rows = bd.shape[0], header.n_rows
        d_data[cum_nnz : cum_nnz + s_nnz] = bd
        d_indices[cum_nnz : cum_nnz + s_nnz] = bi
        d_indptr[cum_rows + 1 : cum_rows + s_rows + 1] = bp[1:] + cum_nnz
        cum_nnz += s_nnz
        cum_rows += s_rows
    if cum_nnz != total_nnz or cum_rows != n_obs:
        raise ValueError(
            f"decoded totals nnz={cum_nnz}/{total_nnz}, rows={cum_rows}/{n_obs} do not "
            f"match the manifest (archive may be corrupt); preallocated buffers may be "
            f"partially unfilled."
        )
    cp.cuda.runtime.deviceSynchronize()
    return _make_csr(d_data, d_indices, d_indptr, (n_obs, n_vars))


def read_v2_subset_to_device(
    archive_dir,
    *,
    cast_back=True,
    n_streams=4,
    row_start=None,
    row_stop=None,
    row_mask=None,
    row_indices=None,
    data_dtype=None,
    index_dtype=None,
    allow_lossy=False,
):
    """Decode a contiguous, SHARD-ALIGNED row range on the GPU.

    Non-contiguous (mask / int array) and non-shard-aligned partial slices raise
    NotImplementedError (#150); small subsets take the CPU-decode+upload path in
    read.py before reaching here.
    """
    get_loc, man = _archive_locator(archive_dir)
    n_obs, n_vars, row_offsets = man["n_obs_total"], man["n_vars"], man["row_offsets"]

    # All the "can't do this on the GPU" gates run BEFORE _import_gpu so they raise
    # the right actionable error even on a host without cupy (R1.9).
    if row_mask is not None or row_indices is not None:
        raise NotImplementedError(
            "on-GPU non-contiguous subset decode is #150; use a shard-aligned "
            "contiguous slice, sh.to_anndata() (CPU), or a small archive."
        )
    start = row_start or 0
    stop = row_stop if row_stop is not None else n_obs
    affected = shards_for_range(row_offsets, start, stop)
    if affected and (start != row_offsets[affected[0]] or stop != row_offsets[affected[-1] + 1]):
        raise NotImplementedError(
            "on-GPU sub-shard slicing is #150; this slice is not shard-aligned. "
            "Use sh.to_anndata() (CPU) or a small archive (CPU-upload)."
        )
    if affected:
        p0, b0 = get_loc(affected[0])
        with open(p0, "rb") as fh:
            fh.seek(b0)
            _check_codec(parse_shx_header(fh))

    plan = resolve_plan(
        man,
        container="csr",
        data_dtype=data_dtype,
        index_dtype=index_dtype,
        allow_lossy=allow_lossy,
        cast_back=cast_back,
        target="cupy",
    )
    require_csr_float(plan.data_dtype)
    cp, _nv, _codec = _import_gpu()
    if not affected:
        empty = (
            cp.empty(0, cp.dtype(plan.data_dtype)),
            cp.empty(0, cp.dtype(plan.index_dtype)),
            cp.zeros(stop - start + 1, cp.dtype(plan.indptr_dtype)),
        )
        return _make_csr(*empty, (stop - start, n_vars))
    staging = _ReusableStaging()
    blocks = []
    for si in affected:
        path, base = get_loc(si)
        with open(path, "rb") as fh:
            fh.seek(base)
            header = parse_shx_header(fh)
        _check_codec(header)
        blocks.append(
            decode_shard_to_device(path, base, header, manifest=man, plan=plan, staging=staging)
        )
    return _stitch(blocks, n_vars, plan)


def _project_total_bytes(manifest, plan):
    total_nnz, n_obs = manifest["total_nnz"], manifest["n_obs_total"]
    chunk = manifest.get("chunk_size_elements", 1 << 20)
    out = (
        total_nnz * plan.data_dtype.itemsize
        + total_nnz * plan.index_dtype.itemsize
        + (n_obs + 1) * plan.indptr_dtype.itemsize
    )
    # One shard's transient working set (compressed span + nvCOMP output + transform
    # batch); freed between shards, so add one shard's worth as headroom.
    max_shard_nnz = max(manifest.get("nnz_per_shard") or [total_nnz])
    working = max_shard_nnz * 8 + chunk * 8 * 4
    return int(out + working)


def _enforce_preflight(projected_bytes):
    import cupy as cp

    pool = cp.get_default_memory_pool()
    physical_free, _ = cp.cuda.runtime.memGetInfo()
    effective_free = physical_free + (pool.total_bytes() - pool.used_bytes())
    if projected_bytes > effective_free:
        raise GpuOutOfMemoryError(
            f"projected {projected_bytes / 1e9:.1f} GB > effective free "
            f"{effective_free / 1e9:.1f} GB. Stream per shard (GroupShard.x_cupy / "
            f"iter_group_shards) or use the CPU reader (sh.to_anndata())."
        )
