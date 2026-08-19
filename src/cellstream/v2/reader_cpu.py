"""v2 CPU reader: parallel mmap + decode -> numpy CSR streams.

Uses the v0.1 SharedMemory bundle pattern: parent pre-allocates one shm
segment per stream sized for the full output. Workers attach by name and
write directly into pre-sliced views -- no per-shard numpy arrays cross
process boundaries.

Subset support:
- Full archive (no kwargs).
- Contiguous slice: pass row_start / row_stop.
- Bool mask: pass row_mask (length n_obs).
- Int array: pass row_indices (any order; reader sorts internally).
"""

from __future__ import annotations

import mmap
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cellstream import workers as _workers
from cellstream.manifest import load_manifest, shard_filename, shards_for_range
from cellstream.v2 import codec
from cellstream.v2.format import parse_shx_header
from cellstream.v2.materialize import cast_data, resolve_plan

# Serializes the OMP_NUM_THREADS env-mutation window inside read_v2_to_host.
# Without this, two threads in the parent process calling read_v2_to_host
# concurrently with different omp_nthreads values would race on os.environ
# and the second-spawned Pool could inherit the first call's restored value.
# The race window is the few lines between the env set and Pool() returning;
# the lock cost is negligible relative to the actual read.
#
# Gemini suggested instead using a Pool initializer defined in a "clean"
# module that doesn't transitively import bitshuffle, so the worker bootstrap
# wouldn't trigger bitshuffle's import-time OMP init before the initializer
# runs. Empirically rejected: cellstream/__init__.py eagerly imports
# cellstream.read -> cellstream.v2.reader_cpu -> bitshuffle, so ANY cellstream.*
# module's import path drags bitshuffle in. A clean initializer would need to
# live outside the cellstream package altogether — a larger restructure deferred
# to a future PR. For now the parent-env approach + lock is the empirically-
# validated fix.
# Reuse the SAME lock instance the write path uses (ultrareview L10): read and
# write both mutate the single global os.environ["OMP_NUM_THREADS"], so they must
# serialize on one lock — two separate locks let a concurrent read + write clobber
# each other's save/restore. (`_workers` is already imported above; workers.py
# imports nothing from cellstream.v2, so there is no cycle.)
_OMP_ENV_LOCK = _workers.OMP_ENV_LOCK


class _BytesIOWrap:
    """Minimal cursor-style wrapper around an mmap, used by parse_shx_header."""

    def __init__(self, mm):
        self.mm = mm
        self.pos = 0

    def read(self, n):
        out = self.mm[self.pos : self.pos + n]
        self.pos += n
        return out


def _default_dir_locator(archive_dir, manifest):
    """Locator for directory archives: each shard is a standalone file (base 0, whole file)."""
    archive_dir = Path(archive_dir)

    def _loc(idx):
        return (str(archive_dir / shard_filename(manifest, idx)), 0, 0)

    return _loc


def _open_shard_mmap(shard_file, base, length):
    """Open a read-only mmap over a shard. base/length==0 maps the whole file
    (directory archive); a nonzero (base, length) maps a 4 KB-aligned window
    inside a packed file. Returns (file_handle, mmap).

    A SHARD is never 0 bytes (it always carries an SHX header), so the overload is
    unambiguous here. For a NAMED MEMBER, which may legitimately be empty, use
    :func:`_open_member_mmap` instead (#277)."""
    fh = open(shard_file, "rb")
    try:
        if length == 0:
            mm = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
        else:
            mm = mmap.mmap(fh.fileno(), length, offset=base, prot=mmap.PROT_READ)
    except BaseException:
        fh.close()
        raise
    return fh, mm


class _EmptyMapping(bytes):
    """Zero-length stand-in for an mmap over an empty member.

    ``mmap`` cannot express an empty window — ``mmap(fd, 0, offset=b)`` maps ``b``→EOF — so a
    genuinely 0-byte member needs a buffer object instead. ``bytes`` already satisfies every
    use of a payload mapping (``np.frombuffer``, slicing, ``hashlib.update``, ``len``); this
    adds the ``close()`` the mmap contract requires.
    """

    def close(self):
        pass


def _open_member_mmap(member_file, base, length):
    """Open a read-only mmap over a NAMED MEMBER window. Returns (file_handle, mapping).

    Unlike :func:`_open_shard_mmap`, ``length`` here is always EXACT: a member may legitimately
    be 0 bytes (a layout='cell' archive with n_obs == 0 has an empty ``x_payload``), so ``0``
    maps an EMPTY window, never the whole file. Reusing the shard helper made ``CellStore._mm``
    a mapping of the entire ``.shad``, so ``verify_payload()`` hashed the container against the
    manifest's SHA-256 of the empty payload and reported an intact archive as corrupt (#277).
    """
    fh = open(member_file, "rb")
    try:
        if length == 0:
            return fh, _EmptyMapping()
        mm = mmap.mmap(fh.fileno(), length, offset=base, prot=mmap.PROT_READ)
    except BaseException:
        fh.close()
        raise
    return fh, mm


def _dispatch_workers(fn, inputs, n_workers, omp_nthreads):
    """Run ``fn`` over ``inputs`` across spawn workers (or serially for a single
    shard / n_workers<=1). ``fn`` writes into the shm bundle by side effect; its
    return value is ignored.

    Uses spawn (not fork): bitshuffle initialises OMP thread pools on import, and
    forking after that deadlocks the child. OMP_NUM_THREADS is set in the *parent*
    so spawn workers inherit it from the OS env on first import (a Pool
    ``initializer`` is too late — pickling it re-imports reader_cpu -> bitshuffle
    first, which inits OMP to os.cpu_count(); 16x192=3072 threads on a 64-CPU box
    is a ~30x collapse). ProcessPoolExecutor spawns LAZILY, so OMP_NUM_THREADS must
    stay set (and _OMP_ENV_LOCK held) across the whole submit + as_completed window.
    Manual executor lifecycle (not ``with``) keeps fail-fast: a SIGKILLed/OOM worker
    surfaces as BrokenProcessPool promptly instead of hanging (#48/#49); on ANY
    error we shutdown(wait=False, cancel_futures=True) and re-raise so the caller
    can release its shm bundle."""
    if not (n_workers > 1 and len(inputs) > 1):
        for i in inputs:
            fn(i)
        return
    with _OMP_ENV_LOCK:
        _prev_omp = os.environ.get("OMP_NUM_THREADS")
        _should_modify = omp_nthreads is not None and _prev_omp != str(omp_nthreads)
        try:
            if _should_modify:
                os.environ["OMP_NUM_THREADS"] = str(omp_nthreads)
            ctx = mp.get_context("spawn")
            ex = ProcessPoolExecutor(max_workers=min(n_workers, len(inputs)), mp_context=ctx)
            try:
                futures = [ex.submit(fn, i) for i in inputs]
                for fut in as_completed(futures):
                    fut.result()
                ex.shutdown(wait=True)
            except BaseException:
                # BrokenProcessPool (worker SIGKILL/OOM), a worker-raised exception
                # re-raised by fut.result(), or KeyboardInterrupt: fast-shutdown so
                # we never block on survivors, then propagate for shm cleanup.
                ex.shutdown(wait=False, cancel_futures=True)
                raise
        finally:
            if _should_modify:
                if _prev_omp is None:
                    os.environ.pop("OMP_NUM_THREADS", None)
                else:
                    os.environ["OMP_NUM_THREADS"] = _prev_omp


def _read_min_dtypes(manifest):
    """Minimum losslessly-representable (data, index) dtypes recorded at write.

    Archives written before min recording lack these keys; for them the on-disk dtype
    IS the minimum (the uint16 era), so fall back to it. Cast safety is judged from the
    MINIMUM, not the storage width: an always-uint32 count archive records
    x_data_dtype_min='uint16', so a default float32 read is can_cast(uint16, float32,
    'safe')=True rather than the can_cast(uint32, float32, 'safe')=False it would
    otherwise hit (which would force per-element lossless verification on every read).
    """
    on_disk_data = np.dtype(manifest.get("x_data_dtype_on_disk", "uint16"))
    on_disk_index = np.dtype(manifest.get("x_indices_dtype_on_disk", "uint16"))
    return (
        np.dtype(manifest.get("x_data_dtype_min", on_disk_data)),
        np.dtype(manifest.get("x_indices_dtype_min", on_disk_index)),
    )


def read_v2_to_host(
    archive_dir,
    *,
    cast_back=True,
    n_workers=8,
    omp_nthreads=4,
    row_start=None,
    row_stop=None,
    row_mask=None,
    row_indices=None,
    manifest=None,
    shard_locator=None,
    plan=None,
    output_backing="heap",
    _return_shm_bundle=False,
):
    """Read a v2 archive into (data, indices, indptr) numpy arrays.

    Default path (`_return_shm_bundle=False`): the caller receives plain numpy
    arrays with no lifetime management required. The pure-Python engine gets
    there by decoding into a SharedMemory bundle, copying out, and releasing it;
    the Rust engine with output_backing="heap" decodes straight into heap arrays
    and never touches /dev/shm.

    Zero-copy path (`_return_shm_bundle=True`): returns
    `(data, indices, indptr, bundle)` for a CSR read or `(dense_2d, bundle)` for
    a dense one. Whether `bundle` is None depends on the ENGINE, not only on
    output_backing — the Rust path honors "heap" literally, the pure-Python path
    always allocates shm because its workers must write somewhere shared:

    - CSR + bundle (Rust output_backing="shm", or ANY pure-Python CSR read): the
      three arrays come from `cellstream.shm.array_for_shared_memory` and each keeps
      its own segment mapped (#274), so they stay valid on their own. The caller
      still attaches the bundle (`cellstream.shm.attach_shm_cleanup`) so the /dev/shm
      names get unlinked and `close_shared_memory()` works.
    - dense + bundle (Rust output_backing="shm", or ANY pure-Python dense read): a
      BARE view. Its caller MUST keep the bundle alive, or re-wrap it with
      `cellstream.shm.dense_array_for_bundle` as `read._x_from_result` does.
    - CSR or dense + None (Rust engine, output_backing="heap"): ordinary heap
      arrays owning their buffers; no lifetime management needed.

    Used by `read.ShardedArchiveView._read_x_v2` to hand off shm-backed arrays
    into an AnnData or a bare matrix.

    Parameters
    ----------
    omp_nthreads : int, default 4
        OpenMP thread count per spawn worker.  Passed to workers by
        setting ``OMP_NUM_THREADS`` in the *parent* process immediately
        before the Pool is created; spawn workers inherit the env var
        from the OS and bitshuffle reads it on first import.

        Note: using a Pool ``initializer`` to set the env var is
        insufficient — pickling the initializer causes the worker to
        import ``reader_cpu.py`` (and hence ``bitshuffle``) before the
        initializer can run, so bitshuffle initialises its OpenMP pool
        to ``os.cpu_count()`` regardless.

        The default of 4 matches v0.1's ``blosc_nthreads`` default
        and keeps total thread count (n_workers × omp_nthreads) ≈ 64
        (the requested CPUs on cpu_high_mem nodes), preventing severe
        oversubscription (e.g. 16 × 192 = 3072 threads on a 64-CPU
        allocation causes ~30× throughput collapse).
    """
    from cellstream import shm as _shm

    archive_dir = Path(archive_dir) if archive_dir is not None else None
    # Non-directory callers (e.g. packed archives) must supply BOTH manifest and
    # shard_locator; otherwise archive_dir is required. Guard before the resolution
    # below so a misuse is a clear ValueError, not a Path(None) TypeError.
    if archive_dir is None and (manifest is None or shard_locator is None):
        raise ValueError(
            "read_v2_to_host requires either archive_dir, or both manifest and "
            "shard_locator (for non-directory sources such as packed archives)."
        )
    if manifest is None:
        manifest = load_manifest(archive_dir / "manifest.json")
    if shard_locator is None:
        shard_locator = _default_dir_locator(archive_dir, manifest)
    n_obs = manifest["n_obs_total"]
    # Validate: at most one row-selector kwarg.
    n_selectors = sum(x is not None for x in (row_mask, row_indices))
    if n_selectors > 0 and (row_start is not None or row_stop is not None):
        raise ValueError("Pass at most one of {row_mask, row_indices, row_start/row_stop}.")
    if row_mask is not None and row_indices is not None:
        raise ValueError("Pass at most one of {row_mask, row_indices, row_start/row_stop}.")
    rows = _resolve_row_selector(n_obs, row_start, row_stop, row_mask, row_indices)
    contiguous = isinstance(rows, slice)

    if output_backing not in ("heap", "shm"):
        if output_backing == "pinned":
            raise NotImplementedError(
                "output_backing='pinned' is reserved for the GPU H2D path (#40) "
                "and is not implemented yet."
            )
        raise ValueError(f"output_backing must be 'heap' or 'shm'; got {output_backing!r}")

    if contiguous:
        affected = shards_for_range(manifest["row_offsets"], rows.start, rows.stop)
    else:
        # Vectorized: O(len(rows) * log(n_shards)) via numpy searchsorted,
        # vs. a Python-loop linear scan via manifest.shard_for_row.
        offsets = np.asarray(manifest["row_offsets"])
        per_row_shard = np.searchsorted(offsets, rows, side="right") - 1
        affected = sorted(set(per_row_shard.tolist()))

    # Resolve the materialization plan once in the parent. A None plan means the
    # caller used the legacy cast_back flag -> build a CSR plan from it, which
    # reproduces the previous dtype behavior exactly (back-compat).
    if plan is None:
        plan = resolve_plan(
            manifest,
            container="csr",
            data_dtype=None,
            index_dtype=None,
            allow_lossy=False,
            cast_back=cast_back,
            target="cpu",
        )
    data_dtype = plan.data_dtype
    indices_dtype = plan.index_dtype
    indptr_dtype = plan.indptr_dtype
    allow_lossy = plan.allow_lossy

    # Pre-pass: decode each affected shard's indptr stream (small) to compute
    # per-shard (nnz_out, n_rows_out) for the output. Lets us size the shm
    # bundle exactly before any worker decodes data/indices.
    per_shard_plan = _plan_shards(shard_locator, manifest, affected, rows, contiguous)
    total_nnz_out = sum(p["nnz_out"] for p in per_shard_plan)
    total_rows_out = sum(p["n_rows_out"] for p in per_shard_plan)

    from cellstream.v2 import _rust_dispatch
    from cellstream.v2.writer import BYTE_FILTER_CODEC_NAME

    # The Rust decode core (and use_rust_for_read) gate on the SHUFFLE marker each
    # shard self-describes (ShxHeader.shuffle). Source it from the actual per-shard
    # header (recorded by _plan_shards above) — NOT the manifest codec — so a float
    # archive's rawdata marker (BYTE_FILTER_RAWDATA_SHUFFLE_NAME) correctly routes to
    # the Python decoder, while a pre-#142 shuffled-float archive (header marker still
    # BYTE_FILTER_SHUFFLE_NAME) keeps the Rust path. The marker is uniform per archive,
    # so the first affected shard is authoritative. (#142; Gemini PR #143 r1.)
    # No affected shards (empty selection) -> nothing to decode; the manifest-codec
    # proxy is a harmless fallback.
    if per_shard_plan:
        shuffle_marker = per_shard_plan[0]["shuffle"]
    else:
        codec_name = manifest.get("codec")
        shuffle_marker = (
            codec.BYTE_FILTER_SHUFFLE_NAME if codec_name == BYTE_FILTER_CODEC_NAME else codec_name
        )
    # Per-shard trusted full dimensions (DoS/integrity bounds for the Rust decode
    # core) + the row Selection, shared by the CSR and dense Rust branches. A read is
    # EITHER contiguous (Range per shard) OR a row gather (Rows per shard) -- never
    # mixed. local_sel is ("range", lstart, lstop) for a slice and ("rows",
    # local_rows_int64) for a mask/int-array read.
    shard_rows = [p["full_rows"] for p in per_shard_plan]
    shard_nnz = [p["full_nnz"] for p in per_shard_plan]
    if contiguous:
        row_gather = False
        sel_starts = [p["local_sel"][1] for p in per_shard_plan]
        sel_stops = [p["local_sel"][2] for p in per_shard_plan]
        row_idx_flat = np.empty(0, dtype=np.int64)
    else:
        row_gather = True
        sel_starts = []
        sel_stops = []
        row_idx_flat = (
            np.concatenate([p["local_sel"][1] for p in per_shard_plan])
            if per_shard_plan
            else np.empty(0, dtype=np.int64)
        ).astype(np.int64, copy=False)
    # Tier gate for the Rust decode lossy policy: is the on-disk -> target cast numpy-safe?
    # (mirrors materialize._cast_is_always_safe). When NOT safe and NOT allow_lossy, the
    # Rust core verifies each element losslessly and raises LossyCastError.
    # Judge cast safety from the recorded MINIMUM dtype (fallback to on-disk for
    # pre-min archives, where on-disk IS the minimum). This keeps the default float32
    # read of an always-uint32 count archive cheap: min uint16 -> float32 is numpy-safe,
    # whereas the on-disk uint32 -> float32 cast is not.
    min_data_dtype, min_index_dtype = _read_min_dtypes(manifest)
    data_safe = bool(np.can_cast(min_data_dtype, data_dtype, casting="safe"))
    index_safe = bool(np.can_cast(min_index_dtype, indices_dtype, casting="safe"))
    if plan.container == "csr" and _rust_dispatch.use_rust_for_read(shuffle_marker, plan):
        # Single-process threaded Rust decode. The Rust driver carves disjoint
        # per-shard sub-slices via split_at_mut using the realized output lengths.
        # output_backing selects WHERE the output X is allocated: plain heap numpy
        # arrays (default, no /dev/shm) or a /dev/shm-backed ShmBundle (cross-process
        # / GPU-staging handoff).
        out_lens = [p["nnz_out"] for p in per_shard_plan]  # realized nnz per shard
        row_lens = [p["n_rows_out"] for p in per_shard_plan]  # output rows per shard
        paths = [shard_locator(s)[0] for s in affected]
        bases = [shard_locator(s)[1] for s in affected]
        if output_backing == "shm":
            # /dev/shm-backed output. Allocate a ShmBundle sized to the FINAL output
            # dtypes (matching the pure-Python fallback's bundle layout below), build
            # numpy views over its buffers, and let Rust write the nnz-sized data /
            # indices streams straight into shm (shm is kernel zero-filled, so the
            # &mut views handed across the FFI are sound). The indptr is special: the
            # Rust pyfn writes it as int64 unconditionally, so decode it into a tiny
            # int64 heap scratch (O(n_rows), negligible vs the nnz-sized streams) and
            # cast that into the indptr_dtype shm view -- keeping all three returned
            # arrays genuine shm-backed views in the plan's dtypes (resolve_plan picks
            # int32 indptr for small archives, int64 once nnz exceeds the int32 range).
            bundle = _shm.allocate_shm_buffers(
                data_nbytes=max(1, total_nnz_out * np.dtype(data_dtype).itemsize),
                indices_nbytes=max(1, total_nnz_out * np.dtype(indices_dtype).itemsize),
                indptr_nbytes=(total_rows_out + 1) * np.dtype(indptr_dtype).itemsize,
            )
            try:
                # array_for_shared_memory (not a bare np.ndarray view): each returned
                # plain ndarray keeps its own segment mapped via its .base chain, so an
                # extracted X.data outlives the temporary CSR/AnnData (#274).
                out_data = _shm.array_for_shared_memory(
                    bundle.data, (total_nnz_out,), np.dtype(data_dtype)
                )
                out_indices = _shm.array_for_shared_memory(
                    bundle.indices, (total_nnz_out,), np.dtype(indices_dtype)
                )
                indptr_view = _shm.array_for_shared_memory(
                    bundle.indptr, (total_rows_out + 1,), np.dtype(indptr_dtype)
                )
                indptr_scratch = np.zeros(total_rows_out + 1, dtype=np.int64)
                _rust_dispatch.decode_v2_csr(
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
                    indptr_scratch,
                    n_vars=int(manifest["n_vars"]),
                    row_gather=row_gather,
                    data_safe=data_safe,
                    index_safe=index_safe,
                    allow_lossy=allow_lossy,
                    n_threads=n_workers,
                    data_min_dtype=min_data_dtype.name,
                    index_min_dtype=min_index_dtype.name,
                )
                indptr_view[:] = indptr_scratch  # int64 -> indptr_dtype into shm
                if _return_shm_bundle:
                    return out_data, out_indices, indptr_view, bundle
                # Default path: copy out of shm then release it.
                d = np.array(out_data)
                i = np.array(out_indices)
                p = np.array(indptr_view)
                bundle.close_and_unlink()
                return d, i, p
            except BaseException:
                bundle.close_and_unlink()
                raise
        # output_backing == "heap": plain numpy arrays, no /dev/shm.
        # np.zeros (not np.empty): the Rust decoder writes every element before any
        # read, but handing it a &mut slice over uninitialized memory across the FFI
        # is a soundness grey area (Gemini #132 r9). At scale this is free -- large
        # allocations are mmap'd and zero-filled by the kernel on first touch (the
        # decode write faults them in either way), so zeros == empty in cost.
        out_data = np.zeros(total_nnz_out, dtype=data_dtype)
        out_indices = np.zeros(total_nnz_out, dtype=indices_dtype)
        # The Rust pyfn writes the indptr as int64 unconditionally; allocate an int64
        # buffer for the decode, then cast to the plan's indptr dtype so the returned
        # arrays match the Python reader exactly.
        out_indptr = np.zeros(total_rows_out + 1, dtype=np.int64)
        _rust_dispatch.decode_v2_csr(
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
            n_vars=int(manifest["n_vars"]),
            row_gather=row_gather,
            data_safe=data_safe,
            index_safe=index_safe,
            allow_lossy=allow_lossy,
            n_threads=n_workers,
            data_min_dtype=min_data_dtype.name,
            index_min_dtype=min_index_dtype.name,
        )
        out_indptr = out_indptr.astype(indptr_dtype, copy=False)
        if _return_shm_bundle:
            return out_data, out_indices, out_indptr, None
        return out_data, out_indices, out_indptr

    if plan.container == "dense":
        # Dense output: one (total_rows_out, n_vars) shm segment; workers scatter
        # their disjoint row-block directly (no CSR intermediate). Returns either
        # the dense ndarray (default) or (dense, bundle) for the zero-copy path.
        n_vars = plan.n_vars
        if _rust_dispatch.use_rust_for_read(shuffle_marker, plan):
            # Single-process threaded Rust dense scatter. The Rust driver carves
            # disjoint per-shard row-blocks via split_at_mut and scatters only the
            # nonzero cells, so the output buffer MUST be pre-zeroed (np.zeros for
            # heap; a fresh /dev/shm segment is kernel zero-filled). output_backing
            # selects WHERE the dense X lives: plain heap numpy (default, no
            # /dev/shm) or a /dev/shm-backed bundle (cross-process / GPU-staging).
            paths = [shard_locator(s)[0] for s in affected]
            bases = [shard_locator(s)[1] for s in affected]
            row_lens = [p["n_rows_out"] for p in per_shard_plan]  # output rows per shard
            if output_backing == "shm":
                dense_bundle = _shm.allocate_dense_shm_buffer(
                    nbytes=max(1, total_rows_out * n_vars * plan.data_dtype.itemsize)
                )
                try:
                    dense_view = np.ndarray(
                        (total_rows_out, n_vars),
                        dtype=plan.data_dtype,
                        buffer=dense_bundle.array.buf,
                    )
                    _rust_dispatch.decode_v2_dense(
                        paths,
                        bases,
                        row_lens,
                        shard_rows,
                        shard_nnz,
                        sel_starts,
                        sel_stops,
                        row_idx_flat,
                        dense_view,
                        n_vars=n_vars,
                        row_gather=row_gather,
                        data_safe=data_safe,
                        allow_lossy=allow_lossy,
                        n_threads=n_workers,
                    )
                    if _return_shm_bundle:
                        return dense_view, dense_bundle
                    out = np.array(dense_view)
                    dense_bundle.close_and_unlink()
                    return out
                except BaseException:
                    dense_bundle.close_and_unlink()
                    raise
            # output_backing == "heap": plain numpy, no /dev/shm. np.zeros both
            # zero-inits (the scatter only writes nonzero cells) and is free at
            # scale -- large allocations are mmap'd and the kernel zero-fills pages
            # on first touch (the decode write faults them in either way).
            dense_view = np.zeros((total_rows_out, n_vars), dtype=plan.data_dtype)
            _rust_dispatch.decode_v2_dense(
                paths,
                bases,
                row_lens,
                shard_rows,
                shard_nnz,
                sel_starts,
                sel_stops,
                row_idx_flat,
                dense_view,
                n_vars=n_vars,
                row_gather=row_gather,
                data_safe=data_safe,
                allow_lossy=allow_lossy,
                n_threads=n_workers,
            )
            if _return_shm_bundle:
                return dense_view, None
            return dense_view
        dense_bundle = _shm.allocate_dense_shm_buffer(
            nbytes=max(1, total_rows_out * n_vars * plan.data_dtype.itemsize)
        )
        try:
            dense_inputs = []
            running_rows = 0
            for shard_idx, p in zip(affected, per_shard_plan, strict=True):
                shard_file, base, length = shard_locator(shard_idx)
                dense_inputs.append(
                    (
                        shard_file,
                        base,
                        length,
                        shard_idx,
                        p,
                        dense_bundle.array.name,
                        total_rows_out,
                        n_vars,
                        running_rows,
                        plan.data_dtype.name,
                        plan.allow_lossy,
                        data_safe,
                    )
                )
                running_rows += p["n_rows_out"]
            _dispatch_workers(_scatter_one_shard_dense, dense_inputs, n_workers, omp_nthreads)
            dense_view = np.ndarray(
                (total_rows_out, n_vars), dtype=plan.data_dtype, buffer=dense_bundle.array.buf
            )
            if _return_shm_bundle:
                return dense_view, dense_bundle
            out = np.array(dense_view)
            dense_bundle.close_and_unlink()
            return out
        except BaseException:
            dense_bundle.close_and_unlink()
            raise

    bundle = _shm.allocate_shm_buffers(
        data_nbytes=max(1, total_nnz_out * np.dtype(data_dtype).itemsize),
        indices_nbytes=max(1, total_nnz_out * np.dtype(indices_dtype).itemsize),
        indptr_nbytes=(total_rows_out + 1) * np.dtype(indptr_dtype).itemsize,
    )

    try:
        inputs = []
        running_nnz = 0
        running_rows = 0
        for i, (shard_idx, plan) in enumerate(zip(affected, per_shard_plan, strict=True)):
            is_first = i == 0
            shard_file, base, length = shard_locator(shard_idx)
            inputs.append(
                (
                    shard_file,
                    base,
                    length,
                    shard_idx,
                    plan,
                    bundle.data.name,
                    bundle.indices.name,
                    bundle.indptr.name,
                    running_nnz,
                    running_rows,
                    np.dtype(data_dtype).name,
                    np.dtype(indices_dtype).name,
                    np.dtype(indptr_dtype).name,
                    is_first,
                    allow_lossy,
                    data_safe,
                    index_safe,
                    int(manifest["n_vars"]),
                )
            )
            running_nnz += plan["nnz_out"]
            running_rows += plan["n_rows_out"]

        _dispatch_workers(_read_one_shard_into_shm, inputs, n_workers, omp_nthreads)

        # Pinned views (#274): this branch returns shm-backed arrays whenever
        # _return_shm_bundle is set -- including output_backing="heap", the default, which
        # the pure-Python fallback documents as "always uses /dev/shm" -- so the segment
        # pin is what keeps an extracted X.data valid after the CSR is dropped.
        data_view = _shm.array_for_shared_memory(
            bundle.data, (total_nnz_out,), np.dtype(data_dtype)
        )
        indices_view = _shm.array_for_shared_memory(
            bundle.indices, (total_nnz_out,), np.dtype(indices_dtype)
        )
        indptr_view = _shm.array_for_shared_memory(
            bundle.indptr, (total_rows_out + 1,), np.dtype(indptr_dtype)
        )
        # SharedMemory is not guaranteed zero-initialised; if `inputs` is
        # empty (empty subset, all-False bool mask, empty row_indices), no
        # worker writes indptr_view[0], leaving it as garbage that breaks
        # the resulting CSR. Explicitly initialise to 0. For non-empty
        # inputs the first worker overwrites this anyway.
        indptr_view[0] = 0
        if _return_shm_bundle:
            # Zero-copy path. The three arrays each pin their own segment (#274), so
            # they remain valid without the bundle; the caller still owns the bundle,
            # for unlinking the /dev/shm names and for close_shared_memory().
            return data_view, indices_view, indptr_view, bundle
        # Default path: copy out of shm then release it. The shm bundle is an
        # implementation detail — callers just get regular numpy arrays.
        data = np.array(data_view)
        indices = np.array(indices_view)
        indptr = np.array(indptr_view)
        bundle.close_and_unlink()
        return data, indices, indptr
    except BaseException:
        # Failure path (worker death / exception, view-build exception,
        # KeyboardInterrupt, or any other error after bundle allocation): the
        # executor (if any) was already fast-shut-down inside _dispatch_workers;
        # here we close+unlink the shm segments so no /dev/shm entries leak. Catch
        # BaseException not Exception so Ctrl+C and SystemExit also trigger cleanup
        # before the interrupt propagates.
        bundle.close_and_unlink()
        raise


def _resolve_row_selector(n_obs, row_start, row_stop, row_mask, row_indices):
    if row_mask is not None:
        return np.asarray(np.flatnonzero(row_mask), dtype=np.int64)
    if row_indices is not None:
        return np.sort(np.asarray(row_indices, dtype=np.int64))
    if row_start is not None or row_stop is not None:
        return slice(0 if row_start is None else row_start, n_obs if row_stop is None else row_stop)
    return slice(0, n_obs)


def _local_row_range_for_shard(row_offsets, shard_idx, rows, contiguous):
    rs, re = row_offsets[shard_idx], row_offsets[shard_idx + 1]
    if contiguous:
        local_start = max(0, rows.start - rs)
        local_stop = min(re - rs, rows.stop - rs)
        return ("range", local_start, local_stop)
    local_rows = rows[(rows >= rs) & (rows < re)] - rs
    return ("rows", local_rows.astype(np.int64))


def _plan_shards(shard_locator, manifest, affected, rows, contiguous):
    """Pre-pass: per affected shard, decode the indptr stream + compute
    (local_sel, nnz_out, n_rows_out).

    shard_indptr is intentionally NOT stored in the plan dict: it is a
    large array (~n_rows × 8 bytes, a few MB on a real archive) that would be
    pickled and piped to each spawn worker, causing the parent's
    posix.write() calls to block for hundreds of seconds waiting for
    workers to drain the OS pipe buffer.  Workers re-decode the indptr
    stream cheaply from the already-open mmap instead.

    Output indptr size = sum(n_rows_out) + 1; output data/indices size = sum(nnz_out).
    """
    plans = []
    for shard_idx in affected:
        local = _local_row_range_for_shard(manifest["row_offsets"], shard_idx, rows, contiguous)
        shard_file, base, length = shard_locator(shard_idx)
        fh, mm = _open_shard_mmap(shard_file, base, length)
        try:
            header = parse_shx_header(_BytesIOWrap(mm))
            shard_indptr = _decode_stream(
                mm, header.indptr, np.int64, shuffle=header.shuffle, role="indptr"
            )
        finally:
            mm.close()
            fh.close()
        # Validate the decoded (untrusted) indptr before indexing it below: a
        # truncated stream would IndexError at shard_indptr[lstop]; a
        # non-monotonic one yields a negative nnz_out / wrong shm sizing. The
        # authoritative row count is the manifest's row_offsets span (this runs
        # in the parent for BOTH the Rust and Python read paths).
        rs, re = manifest["row_offsets"][shard_idx], manifest["row_offsets"][shard_idx + 1]
        _validate_indptr(shard_indptr, int(re) - int(rs), shard_idx)
        if local[0] == "range":
            _, lstart, lstop = local
            nnz_out = int(shard_indptr[lstop]) - int(shard_indptr[lstart])
            n_rows_out = lstop - lstart
        else:
            _, local_rows = local
            if local_rows.size == 0:
                nnz_out = 0
                n_rows_out = 0
            else:
                lens = shard_indptr[local_rows + 1] - shard_indptr[local_rows]
                nnz_out = int(lens.sum())
                n_rows_out = int(local_rows.size)
        plans.append(
            {
                "local_sel": local,
                "nnz_out": nnz_out,
                "n_rows_out": n_rows_out,
                "full_rows": len(shard_indptr) - 1,
                "full_nnz": int(shard_indptr[-1]),
                # Authoritative self-describing codec marker for this shard. The Rust
                # decode gate keys on this (NOT the manifest codec), so a float
                # rawdata archive routes to the Python decoder while old shuffled
                # archives stay on Rust. Uniform per archive. (#142)
                "shuffle": header.shuffle,
            }
        )
    return plans


def _validate_indptr(indptr, expected_rows, shard_idx):
    """Validate a decoded (untrusted) indptr stream before it is indexed.

    Mirrors the integrity the Rust decode core enforces internally: the indptr
    must have exactly ``expected_rows + 1`` entries (a truncated stream would
    otherwise IndexError in ``_plan_shards`` at ``shard_indptr[lstop]``), start
    at 0, and be non-decreasing (a non-monotonic indptr yields a negative slice
    length / silently corrupt CSR). Raises ``ValueError`` on violation
    (consistent with the ``DecodeError::Corrupt -> ValueError`` error model)."""
    if len(indptr) != expected_rows + 1:
        raise ValueError(
            f"shard[{shard_idx}]: corrupt indptr (length {len(indptr)} != "
            f"expected_rows+1 = {expected_rows + 1})"
        )
    if int(indptr[0]) != 0:
        raise ValueError(f"shard[{shard_idx}]: corrupt indptr (indptr[0] = {int(indptr[0])} != 0)")
    # indptr[1:] >= indptr[:-1] (views, no copy) avoids np.diff's int64 temporary
    # and the subtraction entirely (also overflow-free). (Gemini PR #168 r2.)
    if not bool(np.all(indptr[1:] >= indptr[:-1])):
        raise ValueError(f"shard[{shard_idx}]: corrupt indptr (not non-decreasing)")


def _validate_decoded_shard(indptr, data, indices, n_vars, shard_idx, expected_rows):
    """Validate one shard's decoded (untrusted) CSR streams before slicing or
    scattering. Mirrors the Rust decode core's integrity guarantees so the
    pure-Python fallback can't be tricked by a malformed shard into silent CSR
    corruption (bad indptr), an out-of-bounds dense write (``col >= n_vars``), or
    a silent negative-index wrap (``col < 0``). Raises ``ValueError`` on
    violation."""
    _validate_indptr(indptr, expected_rows, shard_idx)
    if int(indptr[-1]) != len(data) or len(data) != len(indices):
        raise ValueError(
            f"shard[{shard_idx}]: inconsistent CSR stream lengths "
            f"(indptr[-1]={int(indptr[-1])}, len(data)={len(data)}, "
            f"len(indices)={len(indices)})"
        )
    if indices.size:
        # Unsigned index dtypes (the common uint16/uint32 case) can't be
        # negative, so skip the indices.min() O(N) scan and only bound the max;
        # signed dtypes still get the < 0 check. (Gemini PR #168 r3.)
        if indices.dtype.kind != "u":
            cmin = int(indices.min())
            if cmin < 0:
                raise ValueError(
                    f"shard[{shard_idx}]: column index out of range [0, {n_vars}) (min={cmin})"
                )
        cmax = int(indices.max())
        if cmax >= n_vars:
            raise ValueError(
                f"shard[{shard_idx}]: column index out of range [0, {n_vars}) (max={cmax})"
            )


def _read_one_shard_into_shm(args):
    """Decode one shard's data + indices into the parent's shm buffers."""
    (
        shard_file,
        base,
        length,
        shard_idx,
        plan,
        data_name,
        indices_name,
        indptr_name,
        nnz_offset,
        rows_offset,
        data_dtype_name,
        indices_dtype_name,
        indptr_dtype_name,
        is_first,
        allow_lossy,
        data_safe,
        index_safe,
        n_vars,
    ) = args
    _workers._maybe_test_kill_self(shard_idx)
    from multiprocessing import shared_memory

    data_dtype = np.dtype(data_dtype_name)
    indices_dtype = np.dtype(indices_dtype_name)
    indptr_dtype = np.dtype(indptr_dtype_name)
    d_shm = shared_memory.SharedMemory(name=data_name)
    i_shm = shared_memory.SharedMemory(name=indices_name)
    p_shm = shared_memory.SharedMemory(name=indptr_name)
    try:
        nnz_out = plan["nnz_out"]
        n_rows_out = plan["n_rows_out"]
        if nnz_out > 0:
            out_data = np.ndarray(
                (nnz_out,),
                dtype=data_dtype,
                buffer=d_shm.buf,
                offset=nnz_offset * data_dtype.itemsize,
            )
            out_idx = np.ndarray(
                (nnz_out,),
                dtype=indices_dtype,
                buffer=i_shm.buf,
                offset=nnz_offset * indices_dtype.itemsize,
            )
        else:
            out_data = np.empty(0, dtype=data_dtype)
            out_idx = np.empty(0, dtype=indices_dtype)
        # First shard of the read owns the leading 0 of the indptr; later
        # shards skip the boundary slot. is_first is passed from the parent
        # as a positional index flag (i == 0) to avoid the rows_offset==0
        # ambiguity when an early shard has zero selected rows.
        out_indptr_full = np.ndarray(
            (n_rows_out + 1,),
            dtype=indptr_dtype,
            buffer=p_shm.buf,
            offset=rows_offset * indptr_dtype.itemsize,
        )
        fh, mm = _open_shard_mmap(shard_file, base, length)
        try:
            header = parse_shx_header(_BytesIOWrap(mm))
            data_full = _decode_stream(
                mm,
                header.data,
                np.dtype(header.data.dtype),
                shuffle=header.shuffle,
                role="data",
            )
            indices_full = _decode_stream(
                mm,
                header.indices,
                np.dtype(header.indices.dtype),
                shuffle=header.shuffle,
                role="indices",
            )
            # Decode indptr here in the worker rather than receiving it via
            # pickle from the parent.  The parent's _plan_shards used indptr
            # only to compute nnz_out/n_rows_out (now in plan).  Sending the
            # full ~1.67 MB shard_indptr array over the IPC pipe caused
            # posix.write() to block for hundreds of seconds because the OS
            # pipe buffer filled up faster than oversubscribed workers could
            # drain it.  Re-decoding is cheap: indptr is typically 1 chunk.
            indptr_full = _decode_stream(
                mm, header.indptr, np.int64, shuffle=header.shuffle, role="indptr"
            )
        finally:
            mm.close()
            fh.close()
        # Reject a malformed shard before slicing the decoded streams: a
        # non-monotonic / wrong-length indptr or an out-of-range column index
        # would otherwise silently corrupt the output CSR.
        _validate_decoded_shard(
            indptr_full, data_full, indices_full, n_vars, shard_idx, plan["full_rows"]
        )
        if plan["local_sel"][0] == "range":
            _, lstart, lstop = plan["local_sel"]
            nnz_start = int(indptr_full[lstart])
            nnz_stop = int(indptr_full[lstop])
            if nnz_out > 0:
                # Cast the selected (contiguous) nnz slice ONCE: precise lossy gate
                # (only selected data) + no per-row cast overhead.
                out_data[:] = cast_data(
                    data_full[nnz_start:nnz_stop],
                    data_dtype,
                    allow_lossy=allow_lossy,
                    where=f"shard[{shard_idx}].X.data",
                    safe=data_safe,
                )
                out_idx[:] = cast_data(
                    indices_full[nnz_start:nnz_stop],
                    indices_dtype,
                    allow_lossy=allow_lossy,
                    where=f"shard[{shard_idx}].X.indices",
                    safe=index_safe,
                )
            local_indptr = (indptr_full[lstart : lstop + 1] - nnz_start).astype(indptr_dtype)
        else:
            _, local_rows = plan["local_sel"]
            lens = indptr_full[local_rows + 1] - indptr_full[local_rows]
            # Gather the selected nnz into a NATIVE-dtype temp via the per-row
            # cursor (cheap memcpy, no per-row cast), then cast ONCE -> precise
            # lossy gate + avoids thousands of cast_data calls.
            tmp_data = np.empty(nnz_out, dtype=data_full.dtype)
            tmp_idx = np.empty(nnz_out, dtype=indices_full.dtype)
            cursor = 0
            for r, ln in zip(local_rows.tolist(), lens.tolist(), strict=True):
                if ln > 0:
                    s = int(indptr_full[r])
                    tmp_data[cursor : cursor + ln] = data_full[s : s + ln]
                    tmp_idx[cursor : cursor + ln] = indices_full[s : s + ln]
                cursor += ln
            if nnz_out > 0:
                out_data[:] = cast_data(
                    tmp_data,
                    data_dtype,
                    allow_lossy=allow_lossy,
                    where=f"shard[{shard_idx}].X.data",
                    safe=data_safe,
                )
                out_idx[:] = cast_data(
                    tmp_idx,
                    indices_dtype,
                    allow_lossy=allow_lossy,
                    where=f"shard[{shard_idx}].X.indices",
                    safe=index_safe,
                )
            local_indptr = np.concatenate(([0], np.cumsum(lens))).astype(indptr_dtype)
        # Shift by the parent's running nnz offset, then write to shm.
        local_indptr = local_indptr + np.array(nnz_offset, dtype=indptr_dtype)
        if is_first:
            out_indptr_full[:] = local_indptr
        else:
            out_indptr_full[1:] = local_indptr[1:]
    finally:
        d_shm.close()
        i_shm.close()
        p_shm.close()


def _scatter_one_shard_dense(args):
    """Decode one shard and scatter its selected rows into the dense output buffer.

    Each worker owns a disjoint contiguous row-block ``[rows_offset, rows_offset +
    n_rows_out)`` of the ``(total_rows, n_vars)`` dense segment, so writes never
    race. Unwritten cells stay zero (the segment is demand-zeroed by the kernel).
    The selected data is cast ONCE per shard (not per row)."""
    (
        shard_file,
        base,
        length,
        shard_idx,
        plan_d,
        dense_name,
        total_rows,
        n_vars,
        rows_offset,
        data_dtype_name,
        allow_lossy,
        data_safe,
    ) = args
    _workers._maybe_test_kill_self(shard_idx)
    from multiprocessing import shared_memory

    data_dtype = np.dtype(data_dtype_name)
    d_shm = shared_memory.SharedMemory(name=dense_name)
    try:
        dense_view = np.ndarray((total_rows, n_vars), dtype=data_dtype, buffer=d_shm.buf)
        fh, mm = _open_shard_mmap(shard_file, base, length)
        try:
            header = parse_shx_header(_BytesIOWrap(mm))
            data_full = _decode_stream(
                mm, header.data, np.dtype(header.data.dtype), shuffle=header.shuffle, role="data"
            )
            indices_full = _decode_stream(
                mm,
                header.indices,
                np.dtype(header.indices.dtype),
                shuffle=header.shuffle,
                role="indices",
            )
            indptr_full = _decode_stream(
                mm, header.indptr, np.int64, shuffle=header.shuffle, role="indptr"
            )
        finally:
            mm.close()
            fh.close()
        # Reject a malformed shard before scattering: an out-of-range column
        # index would IndexError (positive) or silently wrap to the wrong cell
        # (negative) in the dense_view[..., indices] assignment below.
        _validate_decoded_shard(
            indptr_full, data_full, indices_full, n_vars, shard_idx, plan_d["full_rows"]
        )
        if plan_d["local_sel"][0] == "range":
            _, lstart, lstop = plan_d["local_sel"]
            nnz_start = int(indptr_full[lstart])
            data_cast = cast_data(
                data_full[nnz_start : int(indptr_full[lstop])],
                data_dtype,
                allow_lossy=allow_lossy,
                where=f"shard[{shard_idx}].X.data",
                safe=data_safe,
            )
            for out_row, r in enumerate(range(lstart, lstop)):
                s = int(indptr_full[r])
                e = int(indptr_full[r + 1])
                if e > s:
                    dense_view[rows_offset + out_row, indices_full[s:e]] = data_cast[
                        s - nnz_start : e - nnz_start
                    ]
        else:
            _, lr = plan_d["local_sel"]
            lens = indptr_full[lr + 1] - indptr_full[lr]
            tmp = np.empty(int(lens.sum()), dtype=data_full.dtype)
            cursor = 0
            for r, ln in zip(lr.tolist(), lens.tolist(), strict=True):
                if ln > 0:
                    s = int(indptr_full[r])
                    tmp[cursor : cursor + ln] = data_full[s : s + ln]
                cursor += ln
            data_cast = cast_data(
                tmp,
                data_dtype,
                allow_lossy=allow_lossy,
                where=f"shard[{shard_idx}].X.data",
                safe=data_safe,
            )
            cursor = 0
            for out_row, (r, ln) in enumerate(zip(lr.tolist(), lens.tolist(), strict=True)):
                if ln > 0:
                    s = int(indptr_full[r])
                    dense_view[rows_offset + out_row, indices_full[s : s + ln]] = data_cast[
                        cursor : cursor + ln
                    ]
                cursor += ln
    finally:
        d_shm.close()


def _decode_stream(mm, stream_spec, dtype, shuffle="bitshuffle", role="data"):
    """Decode one stream's chunks. ``shuffle`` (from the shard header) selects the
    codec; ``role`` (data/indices/indptr) selects the byte-filter sub-chain.
    Self-describing: legacy bitshuffle and new byte-filter shards both decode
    here. Unknown shuffle markers fail loud."""
    if shuffle == codec.BYTE_FILTER_SHUFFLE_NAME:

        def _decode_chunk(compressed, n_elements):
            return codec.byte_filter_decode(
                compressed, dtype=dtype, n_elements=n_elements, role=role
            )
    elif shuffle == codec.BYTE_FILTER_RAWDATA_SHUFFLE_NAME:
        # #142: float-data archives store the DATA stream raw (zstd only); the
        # indices/indptr streams keep the byte-filter SHUFFLE+BYTEDELTA chain.
        if role == "data":

            def _decode_chunk(compressed, n_elements):
                return codec.zstd_only_decode(compressed, dtype=dtype, n_elements=n_elements)
        else:

            def _decode_chunk(compressed, n_elements):
                return codec.byte_filter_decode(
                    compressed, dtype=dtype, n_elements=n_elements, role=role
                )
    elif shuffle == "bitshuffle":

        def _decode_chunk(compressed, n_elements):
            return codec.bitshuffle_zstd_decode(compressed, dtype=dtype, n_elements=n_elements)
    else:
        raise ValueError(
            f"Unrecognized shard shuffle marker {shuffle!r}; expected "
            f"'bitshuffle', {codec.BYTE_FILTER_SHUFFLE_NAME!r}, or "
            f"{codec.BYTE_FILTER_RAWDATA_SHUFFLE_NAME!r}. The archive may "
            f"have been written by a newer cellstream version."
        )

    parts = []
    for c in stream_spec.chunks:
        if c.decompressed_size == 0:
            continue
        compressed = bytes(mm[c.offset : c.offset + c.compressed_size])
        n_elements = c.decompressed_size // np.dtype(dtype).itemsize
        parts.append(_decode_chunk(compressed, n_elements))
    if not parts:
        return np.array([], dtype=dtype)
    return np.concatenate(parts)
