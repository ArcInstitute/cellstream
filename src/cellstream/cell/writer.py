"""Writer for layout=cell: storage permutation + single-core streaming encode + pack.

pyfastpfor holds the GIL, so phase-1 encode is single-core streaming (bounded RAM:
~one resident cell frame + the O(n_obs) offset/indptr/obs sidecars). Parallel encode
arrives with the Rust port. The storage permutation (reference-first, group_by,
sort_within_group) reuses cellstream's target-aware planner (gather-by-index, never
adata[perm].copy()); grouping only changes order, not per-cell addressability.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import mmap
import os
import tempfile
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellstream import __version__ as _cellstream_version
from cellstream import grouping, header
from cellstream import lock as _lock
from cellstream.errors import ArchiveExistsError, NonIntegerCountsError
from cellstream.manifest import dump_manifest
from cellstream.narrow import (
    _fold_block_uint_max,
    _uint_dtype_from_max,
    data_min_uint_dtype_streaming,
)
from cellstream.packed.writer import _pack_directory
from cellstream.v2.format import MAX_CHUNK_DECOMPRESSED_BYTES
from cellstream.v2.materialize import cast_data

from . import format as fmt
from .codec import _import_pyfastpfor, codec_for
from .format import _plan_blocks

# Chunk for the pure-Python validation pass. UNCHUNKED, np.rint allocates a full nnz-sized
# float copy (26 GB at 6.5B nnz) plus bool masks -- the #216 review finding. 16M elements is
# 64 MB at float32. Rust needs none of this: CountU32 folds the check into the encode loop.
_VALIDATE_CHUNK = 1 << 24


def _has_non_uint_value(data) -> bool:
    """True if ANY value is non-integer, negative, or non-finite -- i.e. not losslessly
    castable to uint32. Chunked (bounded temporaries) and early-exits on the first bad chunk.

    Branches on dtype to skip checks that are intrinsically satisfied: unsigned ints cannot
    be negative or fractional; a signed int can only fail via its minimum.
    """
    if data.size == 0:
        return False
    kind = data.dtype.kind
    if kind == "u":
        return False
    if kind == "i":
        return bool(data.min() < 0)
    for lo in range(0, data.shape[0], _VALIDATE_CHUNK):
        c = data[lo : lo + _VALIDATE_CHUNK]
        if not (np.all(np.isfinite(c)) and np.all(c >= 0) and np.all(c == np.rint(c))):
            return True
    return False


def _default_encode_threads(n_tasks: int) -> int:
    """Worker count sized to the CPU allocation: len(os.sched_getaffinity(0)) (respects Slurm
    --cpus-per-task / container pinning), falling back to os.cpu_count() where sched_getaffinity
    is unavailable or blocked by seccomp. Capped to n_tasks, floored at 1."""
    if hasattr(os, "sched_getaffinity"):
        try:
            n = len(os.sched_getaffinity(0))
        except OSError:
            n = os.cpu_count() or 1
    else:
        n = os.cpu_count() or 1
    return max(1, min(n, n_tasks))


def _file_size(path) -> int:
    """Bytes on disk, measured through an OPEN FD -- never a path ``stat()`` (#305).

    On an attribute-caching network filesystem -- WekaFS, which backs every Arc store, and NFS --
    ``stat(path)`` can report a stale-SHORT ``st_size`` immediately after an append through a
    different fd, while ``fstat`` on an open fd, ``lseek(SEEK_END)`` and the bytes themselves are
    all already correct. No bytes are lost; only the cached attribute lags, and it settles within a
    fraction of a second.

    Sizing a staging file with a path stat therefore REJECTED perfectly good writes: 20 of 1500
    ``CellArchiveWriter`` writes on Python 3.11 + wekafs, 8 on 3.12, 12 on 3.13, and 0 of 1500 on
    local ext4 for every version -- which is why those failures were misread as a CPython 3.13
    defect. Opening the file is also what refreshes the attribute, so this is the honest
    measurement rather than a retry.

    ⚠️ This helper carries NO time-of-check guarantee: the fd is closed again on return, so whatever
    the caller does next is a fresh lookup. Three of the eight sites in this fix DO size and read
    through one handle -- Rust's ``reorder_frames``, ``packed.reader._trailer_valid`` and
    ``_update_metadata_packed``'s tail backup -- and there the path cannot be swapped between the
    two operations; even those are not protected against a concurrent writer mutating the same
    inode. What this helper buys is a size that is not a stale cached attribute. That is the whole
    bug, and it is all it buys. (codex, checkpoint 2 rounds 1-2.)

    ``tests/_stale_stat.py`` injects that exact divergence (``os.stat`` short, ``os.fstat``
    truthful) so the callers are covered on any filesystem, CI's local disk included.

    ``os.fstat`` and ``lseek(SEEK_END)`` are equally immune (verified: both correct in all 26
    observed mismatches), so the choice between them follows one rule in this codebase -- **fstat
    when the handle is only being measured, seek when it is also being read from.** `_file_size` and
    ``packed.reader._apply_recovery`` measure and close, so they fstat and leave the file position
    alone; ``_trailer_valid`` and ``_update_metadata_packed``'s tail backup have to move the position
    anyway, so they seek and spend no second call. (Gemini suggested seek in both places; the mix is
    deliberate, not drift.)
    """
    with open(path, "rb") as fh:
        return os.fstat(fh.fileno()).st_size


def _validate_in_offsets(in_offsets, staged_size, *, n_obs=None):
    """Canonical in_offsets guards, shared by _reorder_frames_python and _reorder_and_finalize's
    identity fast-path (which skips reorder_frames and so would otherwise skip these). The three
    VALUE guards -- in_offsets[0]==0, non-decreasing, in_offsets[-1]==staged_size -- have messages
    byte-identical to the Rust reorder_frames (rust/src/cell.rs); the tested parity substrings are
    "in_offsets[0]", "non-decreasing", and "staging file size". The STRUCTURAL guards have NO Rust
    counterpart (Rust enforces them at its typed FFI boundary, not with these messages): the 1D-array
    check (Rust's reorder_frames takes a PyReadonlyArray1) and the optional identity-path-only n_obs
    length check (Rust uses the perm-length check, which the identity branch has no perm to run).

    in_offsets is UNTRUSTED (this IS the corrupt-input guard). Monotonicity is checked by ADJACENT
    COMPARISON, never np.diff: numpy int64 subtraction WRAPS silently -- for [0, 2**63-1, -2] the
    true step is negative but np.diff yields +2**63-1, so `np.any(np.diff(x) < 0)` ACCEPTS a
    non-monotonic array. A direct comparison does no arithmetic and cannot wrap. Same bug class
    Gemini caught in #217, which is why cell.rs guards its indptr with checked_sub. (Gemini, PR #221.)
    """
    in_offsets = np.asarray(in_offsets, dtype=np.int64)
    if in_offsets.ndim != 1:
        raise ValueError("reorder_frames: in_offsets must be a 1D array")
    if in_offsets.size < 1 or int(in_offsets[0]) != 0:
        raise ValueError("reorder_frames: in_offsets[0] must be 0")
    if np.any(in_offsets[1:] < in_offsets[:-1]):
        raise ValueError("reorder_frames: in_offsets must be non-decreasing")
    if int(in_offsets[-1]) != int(staged_size):
        raise ValueError(
            f"reorder_frames: in_offsets[-1]={int(in_offsets[-1])} != staging file size "
            f"{int(staged_size)}"
        )
    if n_obs is not None and in_offsets.size != n_obs + 1:
        raise ValueError(
            f"reorder_frames: in_offsets length {in_offsets.size} != n_obs+1 ({n_obs + 1})"
        )


def _validate_col_indices(indices, n_vars):
    """Bound every column index to [0, n_vars) BEFORE any cell encode.

    Runs ahead of the engine dispatch so Rust and the pure-Python fallback reject the SAME
    inputs. Without it the two engines write DIFFERENT archives from one CSR: Rust stores an
    out-of-range index verbatim (the reader then rejects the whole archive), while
    _encode_row_python's `idx_raw.astype(idt)` WRAPS it into a legal column -- 70000 -> 4464 at
    n_vars <= 65536 -- so the archive reads back clean at the WRONG GENE, with no error at
    write or read.

    Write-side companion to reader._check_col_indices (#250), which guards the read. The
    message deliberately differs: there the ARCHIVE is corrupt, here the caller's INPUT is
    invalid. scipy cannot catch this for us -- csr_matrix(...) runs
    check_format(full_check=False) and the column-bounds scan lives under full_check=True.

    Unconditional, with no opt-out flag: unlike duplicate (cell, gene) entries
    (require_unique_indices), an out-of-range column is never a legitimate input. Unsigned
    index dtypes cannot be negative -> skip the min() scan. Cost is one O(nnz) min+max:
    22.5 ms on a 64M-nnz block, ~4% of that block's encode. (#264)
    """
    if indices.size == 0:
        return
    if indices.dtype.kind != "u":
        cmin = int(indices.min())
        if cmin < 0:
            raise ValueError(
                f"invalid CSR block: column index out of range [0, {n_vars}) (min={cmin})"
            )
    cmax = int(indices.max())
    if cmax >= n_vars:
        raise ValueError(f"invalid CSR block: column index out of range [0, {n_vars}) (max={cmax})")


def _validate_zstd_frame_bounds(indptr, *, value_itemsize, row_offset=0):
    """Reject a cell whose zstd frame would exceed the decoder's per-chunk cap (#282 item 9).

    A zstd cell frame carries TWO independent zstd chunks: the index blob, always
    ``nnz * 4`` (zstd stores uint32 indices unconditionally), and the value blob,
    ``nnz * value_itemsize``. Neither encoder splits -- byte_filter_encode and
    zstd_only_encode each emit ONE zstd frame for the whole array -- and the decoder caps
    each chunk's declared decompressed size at MAX_CHUNK_DECOMPRESSED_BYTES on both
    engines (v2/codec.py::_decompress_checked, rust/src/decode.rs:374). So the binding
    width is the wider of the two, and without this check a large enough single cell
    writes successfully and is then unreadable by EVERYTHING, with a reader error that
    blames the archive ("corrupt file?") for a frame cellstream itself wrote.

    Runs ahead of the engine dispatch, like _validate_col_indices, so Rust and the
    pure-Python fallback reject the same inputs; deliberately NOT fused into the Rust
    encoder, which would buy a second implementation to keep in sync while the
    pure-Python arm still needs its own (the #264 rationale).

    O(n_obs), not O(nnz): one np.diff().max() over the indptr.
    """
    indptr = np.asarray(indptr)
    if indptr.size < 2:
        return
    per_row = np.diff(indptr.astype(np.int64))
    itemsize = max(4, int(value_itemsize))
    max_nnz = int(per_row.max())
    if max_nnz * itemsize <= MAX_CHUNK_DECOMPRESSED_BYTES:
        return
    r = int(per_row.argmax())
    raise ValueError(
        f"invalid CSR block: row {row_offset + r} has {max_nnz} stored values, whose zstd "
        f"frame would declare {max_nnz * itemsize} decompressed bytes and exceed the "
        f"{MAX_CHUNK_DECOMPRESSED_BYTES}-byte per-chunk cap -- the archive would be written "
        f"successfully and then be unreadable by both engines. Store at most "
        f"{MAX_CHUNK_DECOMPRESSED_BYTES // itemsize} values in a single cell, or use "
        f"codec='pfordelta' (integer counts only), which is not chunk-capped."
    )


def _reorder_frames_python(staging_path, out_path, in_offsets, perm):
    """Pure-Python reorder_frames (CELLSTREAM_DISABLE_RUST=1 / non-x86). Byte-identical to Rust's by
    construction -- it is a byte permutation, so there is nothing to differ on. The in_offsets guards
    ("in_offsets[0]", "non-decreasing", "staging") live in the shared _validate_in_offsets; the perm
    guards ("perm length", "out of range") stay here."""
    in_offsets = np.asarray(in_offsets, dtype=np.int64)
    perm = np.asarray(perm, dtype=np.int64)
    staged = _file_size(staging_path)
    _validate_in_offsets(in_offsets, staged)
    n_src = in_offsets.size - 1
    # perm must define a full reorder of every source frame -- a shorter perm silently drops
    # frames, a longer one (with repeated valid indices) silently duplicates them. Both would
    # otherwise return OK with no other signal, so this is checked explicitly rather than left
    # to the out-of-range guard below (which only catches invalid *values*, not a wrong *count*
    # of otherwise-valid ones). (#219 Task 5 review finding 2.)
    if perm.size != n_src:
        raise ValueError(f"reorder_frames: perm length {perm.size} != {n_src} frames")
    if perm.size and (int(perm.min()) < 0 or int(perm.max()) >= n_src):
        raise ValueError(f"reorder_frames: perm entry out of range for {n_src} frames")

    lens = in_offsets[perm + 1] - in_offsets[perm] if perm.size else np.empty(0, dtype=np.int64)
    out_offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    with open(staging_path, "rb") as sf, open(out_path, "wb") as of:
        mm = mmap.mmap(sf.fileno(), 0, access=mmap.ACCESS_READ) if staged else None
        try:
            for i in perm:
                start, end = int(in_offsets[i]), int(in_offsets[i + 1])
                if start == end:
                    # Zero-length frame: skip before touching mm. When EVERY frame is
                    # zero-length the staging file is 0 bytes, so mm is None -- indexing it
                    # would raise TypeError even though there is nothing to copy. Mirrors
                    # Rust's `if len == 0 { continue; }`. (#219 Task 5 review finding 1.)
                    continue
                of.write(mm[start:end])
        finally:
            if mm is not None:
                mm.close()
    return out_offsets


def _sort_key(col, sort_order, *, group_label=False):
    """Ordering key for a sort column, matching the shard writer's ``_key`` exactly.

    ``group_label=True`` is passed for the ``group_by`` column ONLY -- see the
    ``alphabetical`` arm below.

    ``.values``, NOT ``.to_numpy()``. For a categorical column ``.values`` returns the
    ``Categorical``, which ``plan_group_shards`` sorts by its DECLARED CATEGORY ORDER;
    ``.to_numpy()`` returns an ndarray of the category VALUES, which sorts alphabetically and
    cannot be recovered downstream.

    Using ``.to_numpy()`` for ``sort_within_group`` made ``sort_order`` a silent no-op for
    categorical keys and diverged ``layout='cell'`` from ``layout='shard'`` for identical
    parameters: the shard writer has always applied this same transform (``v2/writer.py``).
    ``group_by`` was unaffected in the default mode -- ``plan_group_shards`` extracts ``.values``
    from a Series itself -- but did not honour ``sort_order='alphabetical'``; routing it through
    here fixes that too, and is a no-op for the default. (#252)
    """
    if sort_order != "alphabetical":
        return col.values
    if group_label:
        # The group_by column's alphabetical sort key must use the SAME missing-value
        # normalization the label view uses, or 'None' and '<NA>' sort apart and one logical
        # group lands in two non-adjacent records (#294, #282 item 5). A sort_within_group key
        # is not a label -- it names no group and bounds no block -- so it stays .astype(str).
        return grouping.stringify_group_labels(col)
    return col.astype(str).values


def _resolve_storage_order(obs, row_nnz, group_by, reference, sort_within_group, sort_order):
    """Return (permutation int64[n_obs], groups_record dict|None). Reuses grouping.plan_group_shards.

    Takes the obs DataFrame + a precomputed per-row nnz array (int64) instead of a CellBlockSource,
    so CellArchiveWriter.finalize and concat_cell_archives can call it with accumulated primitives.
    Ungrouped -> identity permutation, no groups record."""
    # FIRST -- before the group_by is None early return below, or a sort knob passed WITHOUT
    # group_by is never checked. This one call covers every cell writer: _resolve_storage_order's
    # only callers are write_cell_archive and _reorder_and_finalize, the shared tail of
    # CellArchiveWriter.finalize and concat_cell_archives (#261).
    grouping.validate_sort_params(
        group_by=group_by, sort_within_group=sort_within_group, sort_order=sort_order
    )
    import types

    n = len(obs)
    if group_by is None:
        return np.arange(n, dtype=np.int64), None
    if group_by not in obs.columns:
        raise ValueError(f"group_by column {group_by!r} not found in obs.")
    # Preflight, before any destructive step on the DESTINATION (the os.replace / reorder /
    # unlink / pack all live further down in _reorder_and_finalize): a column holding BOTH a
    # literal 'nan' and missing values normalizes to ONE merged group, so read_group('nan')
    # would silently return both populations. Deliberately here and not in plan_group_shards --
    # the shard writers reach the planner only after overwrite=True has already deleted the
    # prior archive, and the planner does not know the column's name.
    grouping.validate_group_label_collision(obs[group_by], where=f"obs[{group_by!r}]")
    ref_src = types.SimpleNamespace(
        obs=obs, n_obs=n
    )  # grouping.resolve_reference needs .obs/.n_obs
    ref_mask = grouping.resolve_reference(reference, ref_src, group_by)
    per_row_nnz = np.asarray(row_nnz, dtype=np.int64)
    if per_row_nnz.shape[0] != n:
        raise ValueError(
            f"_resolve_storage_order: row_nnz length {per_row_nnz.shape[0]} != n_obs {n}"
        )
    secondary = None
    if sort_within_group is not None:
        cols = (
            [sort_within_group] if isinstance(sort_within_group, str) else list(sort_within_group)
        )
        for c in cols:
            if c not in obs.columns:
                raise ValueError(f"sort_within_group column {c!r} not found in obs.")
        keys = [_sort_key(obs[c], sort_order) for c in cols]
        secondary = keys if len(keys) > 1 else keys[0]
    plan = grouping.plan_group_shards(
        group_labels=_sort_key(obs[group_by], sort_order, group_label=True),
        per_row_nnz=per_row_nnz,
        reference_mask=ref_mask,
        target_shard_bytes=1 << 62,
        bytes_per_nnz=4,
        secondary_key=secondary,
    )
    ref_labels = sorted({g["label"] for g in plan.groups if g["role"] == "reference"})
    groups_record = {
        "reference": (ref_labels[0] if len(ref_labels) == 1 else (ref_labels or None)),
        "groups": [
            {"label": g["label"], "start": int(g["row_start"]), "stop": int(g["row_stop"])}
            for g in plan.groups
        ],
    }
    return plan.permutation, groups_record


def _may_exceed_uint32(dtype) -> bool:
    """True if a value of ``dtype`` can exceed the uint32 maximum (0xFFFFFFFF), so the
    overflow check needs an O(nnz) ``.max()`` scan. False for <=32-bit integer dtypes
    (uint32's max IS 0xFFFFFFFF; signed/narrower are strictly smaller) -> the scan is
    skippable (Gemini PR #216). Floats always need the scan (they hold values >> 2**32)."""
    dt = np.dtype(dtype)
    return dt.kind == "f" or dt.itemsize > 4


def _resolve_float_value_dtype(src_dtype, data_dtype) -> str:
    """On-disk float width for genuine-float data. None -> preserve input
    (f64 -> float64, else float32); explicit 'float32'/'float64' -> that width
    (the f64->f32 lossy gate is enforced later by cast_data)."""
    if data_dtype is None:
        return "float64" if np.dtype(src_dtype).itemsize >= 8 else "float32"
    if data_dtype not in ("float32", "float64"):
        raise ValueError(
            f"data_dtype must be 'float32', 'float64', or None for layout='cell' "
            f"float storage; got {data_dtype!r}."
        )
    return data_dtype


def _plan_cell_codec(src_dtype, *, codec, data_dtype):
    """Plan the codec ATTEMPTS from the source DTYPE alone -- never the data (#216 review).

    Returns a list of 1 or 2 ``(codec, value_dtype)`` attempts. ``value_dtype`` is None for
    pfordelta (derived in-path from the observed max) and the zstd on-disk dtype otherwise.

    THE POINT: the old ``_resolve_cell_codec`` took X (or a scan oracle) and scanned every value
    (isfinite/>=0/==rint) to decide whether float storage was required. That scan was 86% of the
    write and +32 GB of temporaries at 6.5B nnz -- to answer a question the encoder ALREADY
    answers per value, for free, fused into its loop (pfor.rs CountU32). So we no longer ask: we
    try the integer codec, and if the encoder rejects the values, write_cell_archive warns and
    retries with the second attempt (float storage). Taking a dtype rather than a matrix makes the
    scan impossible BY CONSTRUCTION.

    A 2nd attempt exists ONLY for float input with no explicit data_dtype. Explicit
    codec='pfordelta' gets NO fallback -- the user asked for it, so it fails loud.
    """
    src_dtype = np.dtype(src_dtype)
    is_float_input = src_dtype.kind == "f"

    if codec is not None and codec not in (fmt.CODEC_PFORDELTA, fmt.CODEC_ZSTD):
        raise ValueError(f"layout='cell' supports codec 'pfordelta' or 'zstd'; got {codec!r}.")

    # An explicit float data_dtype on float input FORCES float storage on its own. The old
    # code computed has_fractional EAGERLY and only then OR'd it with this free predicate, so
    # it paid a full O(nnz) scan and DISCARDED the answer (262.5s -> 25.2s; #216 review).
    if is_float_input and data_dtype is not None:
        if codec == fmt.CODEC_PFORDELTA:
            raise ValueError(
                "layout='cell' codec='pfordelta' requires integer counts; "
                "pass codec='zstd' to store floats."
            )
        return [(fmt.CODEC_ZSTD, _resolve_float_value_dtype(src_dtype, data_dtype))]

    if data_dtype is not None:  # non-float input + data_dtype
        if codec == fmt.CODEC_ZSTD:
            raise ValueError(
                "data_dtype requires float input; integer counts store as uint32 "
                "(cast X to float32/float64 to store floats)."
            )
        raise ValueError(
            "data_dtype is only valid for codec='zstd' float storage; "
            "pfordelta stores integer counts as uint."
        )

    first = (fmt.CODEC_ZSTD, "uint32") if codec == fmt.CODEC_ZSTD else (fmt.CODEC_PFORDELTA, None)
    # Integer input can never need float storage; explicit pfordelta must fail loud, not
    # silently switch codec behind the user's back.
    if not is_float_input or codec == fmt.CODEC_PFORDELTA:
        return [first]
    return [first, (fmt.CODEC_ZSTD, _resolve_float_value_dtype(src_dtype, None))]


def _resolve_push_codec(value_dtype, codec, data_dtype):
    """Resolve a DECLARED input value dtype to a single (codec, on_disk_value, schema, pyfastpfor)
    plan for the push writer -- NO optimistic fallback. Reuses _plan_cell_codec's validation;
    a declared float with codec=None resolves to zstd float storage (the last attempt).

    _plan_cell_codec is optimistic BY DESIGN (#216 review): given a genuine data source, an
    explicit codec='pfordelta' on float input is still ATTEMPTED (a float column may hold only
    integer-valued floats) and only fails loud once the encoder discovers an actual fractional
    value at encode time. The push writer has no source to attempt against here -- only a
    declared dtype -- so that deferral can never fire, and the same request would otherwise
    silently resolve to a pfordelta plan for float data. Reject it upfront instead, same message
    _plan_cell_codec/write_cell_archive already raise once their deferred attempt fails.
    """
    value_dtype = np.dtype(value_dtype)
    if value_dtype.kind == "f" and codec == fmt.CODEC_PFORDELTA:
        raise ValueError(
            "layout='cell' codec='pfordelta' requires integer counts; "
            "pass codec='zstd' to store floats."
        )
    attempts = _plan_cell_codec(value_dtype, codec=codec, data_dtype=data_dtype)
    resolved_codec, on_disk = attempts[-1]
    if resolved_codec == fmt.CODEC_PFORDELTA:
        return resolved_codec, "uint32", fmt.SCHEMA_VERSION, fmt.PYFASTPFOR_CODEC
    return resolved_codec, on_disk, fmt.SCHEMA_VERSION_ZSTD, None


def _hash_file(path):
    """SHA-256 of the FINAL storage-order payload, in one sequential pass. hashlib releases the
    GIL, so ~30 s for 40 GB. ONE implementation for both engines and both orders -- no cross-FFI
    hasher state. payload_checksum defaults to False (#214), so this usually never runs."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode_row_python(enc, idx_raw, vals, idt, require_unique_indices) -> bytes:
    """Encode ONE cell to a frame. Extracted from _stream_encode_python so the streaming block
    encoder and the in-memory reference encoder run THE SAME CODE -- which is also what makes
    their byte-identity trivially true rather than a coincidence to be tested for.

    Do NOT re-inline this. The duplicate-index check and the astype(idt) width are exactly the
    two things that must not drift between the two paths.
    """
    idx = idx_raw.astype(idt)
    if require_unique_indices and idx.size > 1:
        # the caller ran sort_indices() -> a duplicate (cell, gene) is an adjacent equal.
        eq = idx[1:] == idx[:-1]
        if eq.any():
            k = int(np.argmax(eq))
            raise ValueError(
                f"encode_cells: duplicate column index {int(idx[k + 1])} "
                f"within a cell (require_unique_indices=True); pass "
                f"require_unique_indices=False to permit duplicate (cell, gene) entries"
            )
    return enc.encode(idx, vals)


def _stream_encode_python(
    n_obs,
    perm,
    X,
    enc,
    idt,
    data_cast,
    *,
    require_unique_indices,
    payload_checksum,
    offsets_path,
    indptr_path,
    payload_path,
):
    """Single-core streaming encode shared by the pfordelta Python fallback and zstd.

    ``data_cast`` is the whole X.data pre-cast to the on-disk value dtype (so per-cell
    slicing is a view). Byte-identical to the prior pfordelta fallback loop. Returns the
    payload SHA-256 hex (or None when payload_checksum=False)."""
    offsets = [0]
    indptr = [0]
    hasher = hashlib.sha256() if payload_checksum else None
    with open(payload_path, "wb") as pf:
        for new_row in range(n_obs):
            i = int(perm[new_row])
            s, e = X.indptr[i], X.indptr[i + 1]
            frame = _encode_row_python(
                enc, X.indices[s:e], data_cast[s:e], idt, require_unique_indices
            )
            pf.write(frame)
            if hasher is not None:
                hasher.update(frame)
            offsets.append(offsets[-1] + len(frame))
            indptr.append(indptr[-1] + int(e - s))
    np.asarray(offsets, dtype=np.int64).tofile(offsets_path)
    np.asarray(indptr, dtype=np.int64).tofile(indptr_path)
    return hasher.hexdigest() if hasher is not None else None


def _stream_encode_block_python(
    Xb, enc, idt, value_np_dtype, *, require_unique_indices, payload_fh, allow_lossy
):
    """Pure-Python encode of ONE contiguous input-order block, appended to an open payload file.
    Returns (frame_lens list[int], block_nnz, block_max).

    The per-BLOCK data_cast is this path's RAM win: the old code pre-cast the WHOLE X.data
    (nnz x 4|8 bytes) up front. Frame bytes are identical -- same encoder, same per-row inputs.

    Validates + casts per BLOCK exactly as the in-memory Python fallback validates the whole
    array up front -- blocking the input does not weaken "fail loud, unchanged" (design doc
    Sec 7.2), it just fires on block k instead of on the whole array. codec.py's encoders do NOT
    validate (_PforEncoder / _ByteZstdEncoder.encode() just blindly astype()); the check has to
    live here, same as it lives in write_cell_archive today for the in-memory Python fallback.
    The Rust branch needs none of this: encode_cells_block shares encode_cells's already-proven,
    already-validated core.

    An unsigned on-disk dtype (pfordelta always stores uint32 on the streaming Python fallback;
    zstd integer counts) validates via _fold_block_uint_max -- the SAME finite/non-negative/
    integral/<=uint32-max predicate the narrow-dtype scan already uses elsewhere (imported, not
    reimplemented). On failure, a follow-up diagnostic re-runs the checks separately to raise the
    EXACT exception TYPE _encode_cell_payload's in-memory uint arm raises, so the #216 optimistic-
    encode retry in write_cell_archive fires on the pure-Python engine exactly as it does on Rust:
    a non-integer / negative / non-finite value is the RETRYABLE NonIntegerCountsError (the writer
    discards the pass and re-encodes as float storage); a uint32 OVERFLOW is retryable only for a
    FLOAT source (float holds it exactly) and a HARD ValueError for an INT source (an i64 > 2^53
    cast to float is lossy). The messages stay Rust's shared CountU32 text word-for-word
    (encode_cells_block routes through the same trait), distinguishing negative-vs-overflow, which
    _fold_block_uint_max's single fused None used to conflate (PR4 review finding 3). NOT
    LossyCastError: that type is reserved for the float narrowing gate below and must NOT be
    retryable, exactly as on the in-memory path.

    block_max is computed ONLY for an unsigned on-disk dtype. `int(np.max(...))` RAISES
    ValueError("cannot convert float NaN to integer") on float data containing NaN -- and NaN is
    a FIRST-CLASS supported value in a float cell archive (PR3's lossy gate uses equal_nan, so
    NaN -> NaN is lossless). Float storage never consumes max_value anyway: x_data_dtype_min is
    just the float width there, and choose_value_dtype is pfordelta's rule alone. So this is not
    merely a guard -- the computation is meaningless on that path. (Gemini, PR #221 r2.)

    A float on-disk dtype (zstd float storage) routes through cast_data -- the SAME f64->f32
    lossy gate (LossyCastError unless allow_lossy=True) the in-memory Python fallback applies via
    cast_data(X.data, ..., where="cell.data"). A raw .astype() here would silently truncate
    precision instead of raising -- defeating a gate the in-memory path already enforces.
    """
    if value_np_dtype.kind == "u":
        block_max = _fold_block_uint_max(Xb.data, 0)
        if block_max is None:
            # _fold_block_uint_max folds finite/non-negative/integral/<=uint32-max into ONE
            # None result -- re-run the two checks separately (same predicates AND exception
            # types as _encode_cell_payload's uint arms) so the retry loop fires on exactly the
            # same cases on the pure-Python engine as it does on Rust: a non-integer / negative /
            # non-finite value is the RETRYABLE NonIntegerCountsError, while a uint32 OVERFLOW is
            # retryable only for a FLOAT source (float storage holds it exactly) and a HARD
            # ValueError for an INT source (an i64 > 2^53 cast to float is lossy). The messages
            # stay Rust's shared CountU32 text word-for-word (encode_cells_block routes through
            # the same trait), so the streaming Python and Rust engines remain message-identical.
            if _has_non_uint_value(Xb.data):
                raise NonIntegerCountsError(
                    "layout='cell' requires non-negative integer counts; got non-integer, "
                    "negative, or non-finite values."
                )
            exc = NonIntegerCountsError if Xb.data.dtype.kind == "f" else ValueError
            raise exc(
                f"layout='cell' stores counts as uint32; value {int(Xb.data.max())} "
                f"exceeds the uint32 maximum {0xFFFFFFFF}."
            )
        data_cast = Xb.data.astype(value_np_dtype)
    else:
        block_max = 0
        data_cast = cast_data(Xb.data, value_np_dtype, allow_lossy=allow_lossy, where="cell.data")
    lens = []
    for r in range(Xb.shape[0]):
        s, e = Xb.indptr[r], Xb.indptr[r + 1]
        frame = _encode_row_python(
            enc, Xb.indices[s:e], data_cast[s:e], idt, require_unique_indices
        )
        payload_fh.write(frame)
        lens.append(len(frame))
    return lens, int(Xb.nnz), block_max


def _clear_cell_staging(tmpd) -> None:
    """Discard a rejected attempt's staging before retrying with another codec (#216 review).

    Sweeps every x_payload* artifact: the payload itself, per-worker `.part{N}` files (the Rust
    encoders remove those only on their SUCCESS path -- a rejected value returns Err before that
    cleanup), and the streaming reorder staging `x_payload_staging(.part*)` -- a rejected streaming
    attempt appends blocks and leaves them behind. Not archive corruption (packed/writer.py
    hardcodes the layout='cell' member list; it does not glob) but, at 6.5B nnz, GBs of dead
    staging that would otherwise sit on disk for the whole re-encode. The worker count is not
    knowable here, so the artifacts are globbed rather than enumerated.
    """
    tmpd = Path(tmpd)
    for extra in tmpd.glob(f"{fmt.PAYLOAD}*"):
        extra.unlink(missing_ok=True)
    for name in (fmt.OFFSETS, fmt.INDPTR):
        (tmpd / name).unlink(missing_ok=True)


def _encode_cell_payload(
    X,
    perm,
    n_obs,
    n_vars,
    tmpd,
    *,
    resolved_codec,
    zstd_value_dtype,
    allow_lossy,
    payload_checksum,
    require_unique_indices,
):
    """Encode an in-memory X into tmpd/{x_payload,x_offsets,x_indptr} for ONE codec attempt.

    Extracted so write_cell_archive can call it TWICE (via _encode_cell_source): the optimistic
    integer attempt, then -- only on NonIntegerCountsError -- the zstd float retry (#216 review).
    It must therefore be safe to call on a tmpd whose staging files have been cleared: every
    output is (re)created truncating (``open(..., "wb")`` / ``ndarray.tofile`` / Rust
    ``File::create``), so a retry overwrites rather than appends.

    Returns dict(codec, schema_version, pyfastpfor_codec, index_dtype, value_dtype,
                 x_data_dtype_min, total_nnz, payload_sha256). ``codec`` is ``resolved_codec``
    echoed back so the caller has a single source of truth for the manifest: this function only
    ever returns on the attempt that SUCCEEDED (a rejected attempt raises before reaching here).
    """
    from ._rust_dispatch import encode_cells as _rust_encode_cells
    from ._rust_dispatch import use_rust_for_cell_encode

    index_dtype = fmt.choose_index_dtype(n_vars)
    # BEFORE the engine dispatch in either codec arm, so both engines reject the same input
    # and neither can write a frame holding a column no reader will accept (#264).
    _validate_col_indices(X.indices, n_vars)
    if resolved_codec == fmt.CODEC_ZSTD:
        _validate_zstd_frame_bounds(
            X.indptr, value_itemsize=fmt.np_dtype(zstd_value_dtype).itemsize
        )

    if resolved_codec == fmt.CODEC_PFORDELTA:
        schema_version = fmt.SCHEMA_VERSION
        pyfastpfor_codec = fmt.PYFASTPFOR_CODEC
        if use_rust_for_cell_encode(resolved_codec, X.data.dtype, X.indices.dtype, "uint32"):
            # One Rust pass: gather-by-perm, fold validation (finite/int/non-neg/<=u32max),
            # byte-exact frames -> x_payload, native int64 offsets/indptr, streamed sha256,
            # and the observed max value.
            offsets_arr, indptr_arr, total_nnz, max_value, payload_sha256 = _rust_encode_cells(
                X.data,
                X.indices,
                X.indptr,
                perm,
                tmpd / fmt.PAYLOAD,
                n_threads=_default_encode_threads(n_obs),
                compute_checksum=payload_checksum,
                check_duplicates=require_unique_indices,
            )
            offsets_arr.tofile(tmpd / fmt.OFFSETS)
            indptr_arr.tofile(tmpd / fmt.INDPTR)
            total_nnz = int(total_nnz)
            max_value = int(max_value)
            value_dtype = fmt.choose_value_dtype(max_value)
            if total_nnz:
                x_data_dtype_min = _uint_dtype_from_max(max_value).name
            else:
                x_data_dtype_min = np.dtype(np.uint8).name
        else:
            # Pure-Python streaming fallback (CELLSTREAM_DISABLE_RUST=1 / non-x86 / unsupported).
            if _has_non_uint_value(X.data):
                raise NonIntegerCountsError(
                    "layout='cell' requires non-negative integer counts; got non-integer, "
                    "negative, or non-finite values."
                )
            if X.nnz and _may_exceed_uint32(X.data.dtype) and int(X.data.max()) > 0xFFFFFFFF:
                # FLOAT source -> NonIntegerCountsError so the writer retries with float storage
                # (which holds the value exactly); INT source -> hard ValueError, since i64 > 2^53
                # cast to float is lossy. Mirrors pfor.rs count_u32_{float,int}. (#217 lossy test.)
                exc = NonIntegerCountsError if X.data.dtype.kind == "f" else ValueError
                raise exc(
                    f"layout='cell' stores counts as uint32; value {int(X.data.max())} "
                    f"exceeds the uint32 maximum {0xFFFFFFFF}."
                )
            max_value = int(X.data.max()) if X.nnz else 0
            value_dtype = fmt.choose_value_dtype(max_value)
            min_uint = data_min_uint_dtype_streaming([X.data]) if X.nnz else np.dtype(np.uint8)
            x_data_dtype_min = min_uint.name if min_uint is not None else value_dtype
            total_nnz = int(X.nnz)
            enc = codec_for(resolved_codec, index_dtype, value_dtype).new_encoder()
            data_cast = X.data.astype(fmt.np_dtype(value_dtype))
            payload_sha256 = _stream_encode_python(
                n_obs,
                perm,
                X,
                enc,
                fmt.np_dtype(index_dtype),
                data_cast,
                require_unique_indices=require_unique_indices,
                payload_checksum=payload_checksum,
                offsets_path=tmpd / fmt.OFFSETS,
                indptr_path=tmpd / fmt.INDPTR,
                payload_path=tmpd / fmt.PAYLOAD,
            )
    else:  # zstd
        schema_version = fmt.SCHEMA_VERSION_ZSTD
        pyfastpfor_codec = None
        index_dtype = "uint32"  # zstd always stores uint32 indices
        total_nnz = int(X.nnz)
        value_dtype = zstd_value_dtype  # FIXED on disk: uint32 / float32 / float64
        if use_rust_for_cell_encode(
            resolved_codec, X.data.dtype, X.indices.dtype, zstd_value_dtype
        ):
            # One Rust pass: gather-by-perm, fold validation (int path: finite / integral /
            # non-negative / <= u32max; float path: the f64->f32 lossless gate), byte-exact
            # frames -> x_payload, native int64 offsets/indptr, streamed sha256, observed max.
            #
            # NOTE: no data_cast. Rust reads X.data in its SOURCE dtype and casts per frame, so
            # the full-size pre-cast copy (nnz x 4|8 bytes) is never allocated -- the same way the
            # pfordelta Rust arm above already works. On the optimistic zstd-uint32 attempt Rust
            # raises NonIntegerCounts on fractional / overflowing floats, so the writer's retry
            # loop falls back to zstd float storage (the #216 optimistic-encode contract).
            offsets_arr, indptr_arr, total_nnz, max_value, payload_sha256 = _rust_encode_cells(
                X.data,
                X.indices,
                X.indptr,
                perm,
                tmpd / fmt.PAYLOAD,
                n_threads=_default_encode_threads(n_obs),
                compute_checksum=payload_checksum,
                check_duplicates=require_unique_indices,
                codec=fmt.CODEC_ZSTD,
                value_dtype_on_disk=zstd_value_dtype,
                allow_lossy=allow_lossy,
            )
            offsets_arr.tofile(tmpd / fmt.OFFSETS)
            indptr_arr.tofile(tmpd / fmt.INDPTR)
            total_nnz = int(total_nnz)
            if zstd_value_dtype == "uint32":
                # max_value feeds x_data_dtype_min ONLY. zstd's on-disk value_dtype is FIXED at
                # uint32 -- it must NEVER be narrowed by choose_value_dtype (pfordelta's rule).
                x_data_dtype_min = (
                    _uint_dtype_from_max(int(max_value)).name
                    if total_nnz
                    else np.dtype(np.uint8).name
                )
            else:
                x_data_dtype_min = zstd_value_dtype
        else:
            # Pure-Python streaming fallback (CELLSTREAM_DISABLE_RUST=1 / non-x86 / unsupported).
            # The REFERENCE implementation the cross-engine decode-identity test compares against.
            if zstd_value_dtype == "uint32":
                # Validate at the astype(uint32) boundary so a negative/fractional value can
                # never silently wrap (e.g. -5 -> 4294967291). Branch on dtype to skip scans
                # that are intrinsically satisfied: float -> full re-check (self-contained, not
                # relying on the codec planner's routing); signed int -> only its minimum can be
                # negative; unsigned -> intrinsically non-negative integers, no scan.
                if _has_non_uint_value(X.data):
                    raise NonIntegerCountsError(
                        "layout='cell' codec='zstd' integer counts require non-negative "
                        "integers; got non-integer, negative, or non-finite values."
                    )
                if X.nnz and _may_exceed_uint32(X.data.dtype) and int(X.data.max()) > 0xFFFFFFFF:
                    # See the pfordelta arm above: FLOAT overflow is retryable (-> float storage),
                    # INT overflow is a hard error (i64 > 2^53 -> float is lossy).
                    exc = NonIntegerCountsError if X.data.dtype.kind == "f" else ValueError
                    raise exc(
                        f"layout='cell' codec='zstd' stores integer counts as uint32; "
                        f"value {int(X.data.max())} exceeds the uint32 maximum {0xFFFFFFFF}."
                    )
                min_uint = data_min_uint_dtype_streaming([X.data]) if X.nnz else np.dtype(np.uint8)
                x_data_dtype_min = min_uint.name if min_uint is not None else "uint32"
                data_cast = X.data.astype(np.uint32)
            else:  # float32 / float64
                x_data_dtype_min = zstd_value_dtype
                # cast_data enforces the f64->f32 lossy gate (raises LossyCastError without
                # allow_lossy; forced with). NaN/Inf round-trip (equal_nan in lossless_cast).
                data_cast = cast_data(
                    X.data,
                    fmt.np_dtype(zstd_value_dtype),
                    allow_lossy=allow_lossy,
                    where="cell.data",
                )
            enc = codec_for(fmt.CODEC_ZSTD, index_dtype, value_dtype).new_encoder()
            payload_sha256 = _stream_encode_python(
                n_obs,
                perm,
                X,
                enc,
                fmt.np_dtype(index_dtype),
                data_cast,
                require_unique_indices=require_unique_indices,
                payload_checksum=payload_checksum,
                offsets_path=tmpd / fmt.OFFSETS,
                indptr_path=tmpd / fmt.INDPTR,
                payload_path=tmpd / fmt.PAYLOAD,
            )

    return {
        "codec": resolved_codec,
        "schema_version": schema_version,
        "pyfastpfor_codec": pyfastpfor_codec,
        "index_dtype": index_dtype,
        "value_dtype": value_dtype,
        "x_data_dtype_min": x_data_dtype_min,
        "total_nnz": total_nnz,
        "payload_sha256": payload_sha256,
    }


def _encode_cell_streaming(
    source,
    perm,
    groups_record,
    n_obs,
    n_vars,
    tmpd,
    *,
    resolved_codec,
    zstd_value_dtype,
    block_size,
    allow_lossy,
    payload_checksum,
    require_unique_indices,
):
    """Streaming (block-append) encode of a NON-in-memory source for ONE codec attempt (#219).

    Extracted from write_cell_archive's streaming branch so the #216 retry loop can call it TWICE
    (optimistic integer attempt, then -- only on NonIntegerCountsError -- the zstd float retry),
    exactly like _encode_cell_payload does for the in-memory path. It re-runs _plan_blocks from
    block 0 every call and TRUNCATES enc_path up front, so a retry after _clear_cell_staging
    restarts cleanly rather than appending to the rejected attempt's frames.

    Encoding always proceeds in INPUT row order; the storage permutation is applied afterwards to
    the encoded FRAMES (reorder_frames), never to the source rows. Returns the SAME 8-key dict
    _encode_cell_payload returns, so both paths feed write_cell_archive one manifest source.
    """
    from ._rust_dispatch import encode_cells_block as _rust_encode_cells_block
    from ._rust_dispatch import reorder_frames as _rust_reorder_frames
    from ._rust_dispatch import use_rust_for_cell_encode

    index_dtype = fmt.choose_index_dtype(n_vars)
    if resolved_codec == fmt.CODEC_ZSTD:
        index_dtype = "uint32"  # zstd always stores uint32 indices
    on_disk_value = zstd_value_dtype if resolved_codec == fmt.CODEC_ZSTD else "uint32"

    grouped = groups_record is not None
    staging = tmpd / "x_payload_staging"
    enc_path = staging if grouped else (tmpd / fmt.PAYLOAD)

    src_dt, src_it = source.data_dtype(), source.index_dtype()
    itemsize = src_dt.itemsize + src_it.itemsize
    use_rust = use_rust_for_cell_encode(resolved_codec, src_dt, src_it, on_disk_value)

    # TRUNCATE the payload before the loop. Two reasons, both load-bearing:
    #  (a) encode_cells_block opens with .append(true).create(true) -- if enc_path already held
    #      bytes (a rejected prior attempt's frames), block 0 would append to STALE DATA. The
    #      retry loop's _clear_cell_staging removes them too, but the append contract must not
    #      depend on that having run.
    #  (b) n_obs == 0 -> _plan_blocks returns [] -> encode_cells_block is NEVER CALLED ->
    #      tmpd/x_payload would not exist -> pack() fails on a missing member and _hash_file
    #      raises FileNotFoundError. The empty archive MUST still produce an empty payload.
    enc_path.write_bytes(b"")

    lens_parts, total_nnz, max_value = [], 0, 0
    py_enc = py_fh = None
    if not use_rust:
        # pfordelta's on-disk value width normally comes from the observed max, which a streaming
        # write does not know up front -- so the Python fallback encodes at uint32 (the widest
        # pfordelta ever stores) and lets choose_value_dtype narrow the MANIFEST field afterwards.
        # This EMITS BYTE-IDENTICAL FRAMES: _PforEncoder.encode() forces uint32 regardless of the
        # input width, so encoding at uint32 here and at uint16 on the in-memory path packs the
        # identical u32 words. (zstd is safe because its value_dtype is FIXED by _plan_cell_codec,
        # so both paths pass the same one.)
        py_enc = codec_for(resolved_codec, index_dtype, on_disk_value).new_encoder()
        py_fh = open(enc_path, "wb")
    try:
        for lo, hi in _plan_blocks(source.row_nnz(), block_size, itemsize):
            Xb = source.read_block(lo, hi)
            Xb.sort_indices()
            _validate_col_indices(Xb.indices, n_vars)  # before the engine dispatch (#264)
            if resolved_codec == fmt.CODEC_ZSTD:
                _validate_zstd_frame_bounds(
                    Xb.indptr,
                    value_itemsize=fmt.np_dtype(on_disk_value).itemsize,
                    row_offset=lo,  # block-local -> input-order global row
                )
            if use_rust:
                blens, bnnz, bmax = _rust_encode_cells_block(
                    Xb.data,
                    Xb.indices,
                    Xb.indptr,
                    enc_path,
                    n_threads=_default_encode_threads(hi - lo),
                    check_duplicates=require_unique_indices,
                    codec=resolved_codec,
                    value_dtype_on_disk=on_disk_value,
                    allow_lossy=allow_lossy,
                )
            else:
                blens, bnnz, bmax = _stream_encode_block_python(
                    Xb,
                    py_enc,
                    fmt.np_dtype(index_dtype),
                    fmt.np_dtype(on_disk_value),
                    require_unique_indices=require_unique_indices,
                    payload_fh=py_fh,
                    allow_lossy=allow_lossy,
                )
            lens_parts.append(np.asarray(blens, dtype=np.int64))
            total_nnz += int(bnnz)
            max_value = max(max_value, int(bmax))
    finally:
        if py_fh is not None:
            py_fh.close()

    all_lens = np.concatenate(lens_parts) if lens_parts else np.empty(0, dtype=np.int64)
    in_offsets = np.concatenate([[0], np.cumsum(all_lens)]).astype(np.int64)

    if grouped:
        offsets_arr = _rust_reorder_frames(staging, tmpd / fmt.PAYLOAD, in_offsets, perm)
        staging.unlink()
    else:
        offsets_arr = in_offsets  # staging IS the payload; identity perm

    offsets_arr.astype(np.int64).tofile(tmpd / fmt.OFFSETS)
    np.concatenate([[0], np.cumsum(source.row_nnz()[perm])]).astype(np.int64).tofile(
        tmpd / fmt.INDPTR
    )
    payload_sha256 = _hash_file(tmpd / fmt.PAYLOAD) if payload_checksum else None

    if resolved_codec == fmt.CODEC_PFORDELTA:
        schema_version = fmt.SCHEMA_VERSION
        pyfastpfor_codec = fmt.PYFASTPFOR_CODEC
        value_dtype = fmt.choose_value_dtype(max_value)
        x_data_dtype_min = (
            _uint_dtype_from_max(max_value).name if total_nnz else np.dtype(np.uint8).name
        )
    else:
        schema_version = fmt.SCHEMA_VERSION_ZSTD
        pyfastpfor_codec = None
        # zstd's on-disk value_dtype is FIXED -- it must NEVER be narrowed by choose_value_dtype
        # (that is pfordelta's rule). Only x_data_dtype_min uses the observed max.
        value_dtype = zstd_value_dtype
        x_data_dtype_min = (
            (_uint_dtype_from_max(max_value).name if total_nnz else np.dtype(np.uint8).name)
            if zstd_value_dtype == "uint32"
            else zstd_value_dtype
        )

    return {
        "codec": resolved_codec,
        "schema_version": schema_version,
        "pyfastpfor_codec": pyfastpfor_codec,
        "index_dtype": index_dtype,
        "value_dtype": value_dtype,
        "x_data_dtype_min": x_data_dtype_min,
        "total_nnz": int(total_nnz),
        "payload_sha256": payload_sha256,
    }


def _encode_cell_source(
    source,
    perm,
    groups_record,
    n_obs,
    n_vars,
    tmpd,
    *,
    resolved_codec,
    zstd_value_dtype,
    block_size,
    allow_lossy,
    payload_checksum,
    require_unique_indices,
):
    """Encode ONE codec attempt from ``source``, routing in-memory vs streaming.

    Both arms return the SAME 8-key dict, so write_cell_archive's retry loop is agnostic to which
    ran. In-memory uses the single-shot _encode_cell_payload; a backed/archive source streams via
    _encode_cell_streaming. Either raises NonIntegerCountsError to trigger the #216 retry.
    """
    if source.is_in_memory:
        X = source.X
        X.sort_indices()
        return _encode_cell_payload(
            X,
            perm,
            n_obs,
            n_vars,
            tmpd,
            resolved_codec=resolved_codec,
            zstd_value_dtype=zstd_value_dtype,
            allow_lossy=allow_lossy,
            payload_checksum=payload_checksum,
            require_unique_indices=require_unique_indices,
        )
    return _encode_cell_streaming(
        source,
        perm,
        groups_record,
        n_obs,
        n_vars,
        tmpd,
        resolved_codec=resolved_codec,
        zstd_value_dtype=zstd_value_dtype,
        block_size=block_size,
        allow_lossy=allow_lossy,
        payload_checksum=payload_checksum,
        require_unique_indices=require_unique_indices,
    )


def _build_cell_manifest(
    *,
    schema_version,
    codec,
    n_obs,
    n_vars,
    total_nnz,
    value_dtype,
    index_dtype,
    x_data_dtype_min,
    pyfastpfor_codec,
    group_by,
    reference,
    sort_within_group,
    sort_order,
    payload_sha256,
    indices_canonical,
):
    """The layout=cell manifest schema — ONE source of truth for write_cell_archive,
    CellArchiveWriter, and concat. writer_kind stays 'cell' for all (byte-identity)."""
    return {
        "schema_version": schema_version,
        "layout": fmt.LAYOUT_CELL,
        "codec": codec,
        "n_obs_total": int(n_obs),
        "n_vars": int(n_vars),
        "total_nnz": int(total_nnz),
        "value_dtype_on_disk": value_dtype,
        "index_dtype_on_disk": index_dtype,
        "x_data_dtype_min": x_data_dtype_min,
        "x_indices_dtype_min": "uint16" if n_vars <= 65536 else "uint32",
        "pyfastpfor_codec": pyfastpfor_codec,
        "group_by": group_by,
        "reference": reference,
        "sort_within_group": sort_within_group,
        "sort_order": sort_order,
        "payload_sha256": payload_sha256,
        "indices_canonical": bool(indices_canonical),
        "writer_fingerprint": {"cellstream_version": _cellstream_version, "writer_kind": "cell"},
        "created_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _write_cell_members_and_pack(
    tmpd,
    *,
    obs_storage,
    var,
    uns,
    groups_record,
    manifest,
    out_path,
    overwrite,
    obsm=None,
    varm=None,
    obsp=None,
    varp=None,
):
    """Write obs.parquet + header.h5ad + groups.json + manifest.json into tmpd, then pack().
    Assumes tmpd already holds x_payload / x_offsets / x_indptr. pack() is atomic (tmp->rename).
    obsm/varm/obsp/varp (cell-axis fields already permuted to storage order by the caller, #229) are
    stored in header.h5ad; the push (CellArchiveWriter) and concat writers pass none -> empty/None is
    byte-identical to omitting them, so those writers still store X/obs/var/uns only."""
    header.write_obs_parquet(obs_storage, tmpd / fmt.OBS)
    n_obs, n_vars = int(manifest["n_obs_total"]), int(manifest["n_vars"])
    hdr = ad.AnnData(
        X=sp.csr_matrix((n_obs, n_vars), dtype="float32"),
        obs=obs_storage,
        var=var,
        uns=uns,
        obsm=(obsm or None),
        varm=(varm or None),
    )
    for k, v in (obsp or {}).items():
        hdr.obsp[k] = v
    for k, v in (varp or {}).items():
        hdr.varp[k] = v
    header.write_header(hdr, tmpd / fmt.HEADER)
    if groups_record is not None:
        (tmpd / fmt.GROUPS).write_text(json.dumps(groups_record))
    dump_manifest(tmpd / fmt.MANIFEST, manifest)
    _pack_directory(tmpd, out_path, overwrite=overwrite, _delete_sources=True)


def _reorder_and_finalize(
    tmpd,
    staging,
    in_offsets,
    row_nnz,
    *,
    obs,
    var,
    uns,
    group_by,
    reference,
    sort_within_group,
    sort_order,
    codec_meta,
    n_vars,
    out_path,
    overwrite,
    payload_checksum,
    obsm=None,
    varm=None,
    obsp=None,
    varp=None,
):
    """Shared tail for CellArchiveWriter.finalize and concat: plan storage order from obs, reorder
    frames (or identity os.replace), write offsets/indptr, then build manifest + write members + pack.
    RAM O(n_obs); reorder_frames streams over the mmap. Byte-identical to write_cell_archive by
    reusing the same planner + reorder_frames + manifest/members writers.

    Cell-axis annotations (``obsm``/``obsp``), when given, are permuted to storage order here -- the
    one place ``perm`` exists; ``varm``/``varp`` (feature axis) pass through untouched. Callers that
    store none (CellArchiveWriter) leave all four ``None``, which is byte-identical to omitting
    them (#237)."""
    from ._rust_dispatch import reorder_frames

    in_offsets = np.asarray(in_offsets, dtype=np.int64)
    row_nnz = np.asarray(row_nnz, dtype=np.int64)
    perm, groups_record = _resolve_storage_order(
        obs, row_nnz, group_by, reference, sort_within_group, sort_order
    )
    payload = tmpd / fmt.PAYLOAD
    n = len(obs)
    is_identity = perm.size == n and bool(np.array_equal(perm, np.arange(n, dtype=np.int64)))
    if is_identity:
        # #234 (Copilot/Gemini #230): the identity fast-path skips reorder_frames -- and therefore
        # its in_offsets guards. Enforce the SAME canonical guards here so concat of a part with a
        # corrupt offsets sidecar (non-monotonic / not starting at 0 / wrong length / truncated
        # payload) fails loud instead of byte-copying into a broken archive. n_obs pins the frame
        # count the identity path can't get from reorder_frames' perm-length check.
        staged_len = _file_size(staging)
        _validate_in_offsets(in_offsets, staged_len, n_obs=n)
        if Path(staging) != payload:
            os.replace(staging, payload)
        offsets_arr = in_offsets
    else:
        offsets_arr = reorder_frames(staging, payload, in_offsets, perm)
        Path(staging).unlink(missing_ok=True)
    offsets_arr.astype(np.int64).tofile(tmpd / fmt.OFFSETS)
    np.concatenate([[0], np.cumsum(row_nnz[perm])]).astype(np.int64).tofile(tmpd / fmt.INDPTR)
    payload_sha256 = _hash_file(payload) if payload_checksum else None

    manifest = _build_cell_manifest(
        schema_version=codec_meta["schema_version"],
        codec=codec_meta["codec"],
        n_obs=n,
        n_vars=n_vars,
        total_nnz=codec_meta["total_nnz"],
        value_dtype=codec_meta["value_dtype"],
        index_dtype=codec_meta["index_dtype"],
        x_data_dtype_min=codec_meta["x_data_dtype_min"],
        pyfastpfor_codec=codec_meta["pyfastpfor_codec"],
        group_by=group_by,
        reference=(groups_record or {}).get("reference"),
        sort_within_group=sort_within_group,
        sort_order=sort_order,
        payload_sha256=payload_sha256,
        indices_canonical=codec_meta["indices_canonical"],
    )
    # Cell-axis annotations reach storage order HERE -- the one place `perm` exists (#237).
    # is_identity is ALREADY computed above for the payload fast path, so this gate is free, and it
    # is strictly stronger than write_cell_archive's `groups_record is None` (#240): it also skips
    # the no-op reindex when a GROUPED write happens to resolve to the identity perm.
    # permute_cell_axis_fields(_, _, None) returns shallow dict copies -> byte-identical output.
    obsm_perm, obsp_perm = header.permute_cell_axis_fields(
        obsm or {}, obsp or {}, None if is_identity else perm
    )
    _write_cell_members_and_pack(
        tmpd,
        obs_storage=obs.iloc[perm].copy(),
        var=var,
        uns=uns,
        obsm=obsm_perm,
        varm=varm or {},
        obsp=obsp_perm,
        varp=varp or {},
        groups_record=groups_record,
        manifest=manifest,
        out_path=out_path,
        overwrite=overwrite,
    )


def write_cell_archive(
    adata,
    out_path,
    *,
    group_by=None,
    reference=None,
    sort_within_group=None,
    sort_order=None,
    codec: str | None = None,
    data_dtype: str | None = None,
    allow_lossy: bool = False,
    overwrite: bool = False,
    payload_checksum: bool = False,
    require_unique_indices: bool = True,
    unsafe_no_lock: bool = False,
    block_size: int | None = None,
    tmp_dir: str | Path | None = None,
) -> None:
    """Write a cell source to a layout=cell .shad file.

    ``adata`` may be an in-memory :class:`~anndata.AnnData`, a path to a backed-readable
    ``.h5ad`` (**CSR** X streams via ``BackedH5adSource``; a **dense** X -- any HDF5 2-D
    Dataset, including files without ``encoding-type`` -- streams via ``DenseH5adSource``,
    #225), or an existing layout=cell ``.shad`` path /
    :class:`~cellstream.cell.reader.CellStore` (the codec-migration / re-encode case). The
    streaming sources read contiguous input-order row blocks instead of materialising the
    whole matrix, so peak RAM is O(block) rather than O(nnz). An in-memory ``AnnData`` always
    uses the original single-shot path; a CSC or unknown-encoding ``.h5ad`` falls back to a
    full in-memory read.

    codec
        ``None`` (default) auto-selects: integer counts -> ``"pfordelta"`` (the fast
        default), genuine non-integer floats -> ``"zstd"``. ``"pfordelta"`` on float data
        raises. ``"zstd"`` is opt-in for integer counts (smaller on dense cells) and stores
        floats. pfordelta archives are ``schema_version`` 5; zstd archives are 6.
    data_dtype
        On-disk **float** width for ``codec="zstd"`` genuine-float data:
        ``None`` (default) preserves the input width (f64 -> ``"float64"``, else
        ``"float32"``); ``"float32"``/``"float64"`` force that width. Ignored/invalid for
        integer counts (which always store as ``uint32``).
    allow_lossy
        Gate for a lossy ``data_dtype="float32"`` down-cast of float64 input: ``False``
        (default) raises ``LossyCastError`` unless every value round-trips exactly; ``True``
        forces the cast. NaN/Inf always round-trip.
    payload_checksum
        When True, compute + store the streamed payload SHA-256 (enables
        ``CellStore.verify_payload()``). Default False -> ~15% faster cell writes.
    require_unique_indices
        When True (default), verify that no cell has a duplicate ``(cell, gene)`` entry and
        **raise** on the first one; the archive is then marked canonical so the reader can
        cache ``has_canonical_format``. When False, skip the check (duplicates permitted;
        archive not marked canonical). Never collapses duplicates (no ``sum_duplicates``).
    unsafe_no_lock
        Escape hatch: skip the archive write lock entirely. Same meaning and same default
        as ``write_sharded``/``write_packed``. Documented for filesystems where ``flock``
        is not honoured or errors; concurrent writers to one output are then unprotected.
    block_size
        Row-count override for the streaming input plan. Ignored for an in-memory
        ``AnnData`` (the single-shot path never blocks). ``None`` (default) plans
        NNZ-balanced blocks under an internal ~2 GiB per-block budget; an explicit row
        count uses fixed-size blocks instead. Encoding always proceeds in *input* row
        order; the storage permutation (when ``group_by`` is set) is applied afterwards to
        the encoded frames, never to the source rows -- so a streaming source only ever
        needs contiguous reads.
    tmp_dir
        Scratch directory for the write staging area. Defaults to ``out_path.parent``
        (**not** ``TMPDIR``/``/tmp``): a grouped streaming write holds staging + final
        payload (~2x payload) there, and the output filesystem must hold the finished
        archive anyway, whereas ``/tmp`` typically cannot hold even 1x payload.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise ArchiveExistsError(f"{out_path} already exists. Pass overwrite=True.")

    # Same shape and same ORDER as write_packed (packed/writer.py:200-206): existence check,
    # then acquire. pack() re-checks existence under the lock, so the check above is a fast
    # fail and not the authority -- and keeping it first means a plainly-existing output
    # still reports ArchiveExistsError rather than ArchiveLockedError, i.e. this adds no
    # diagnostic change at all. acquire_write_lock mkdir -ps the lock file's parent, so a
    # not-yet-created out_path.parent needs no explicit mkdir here. (#280 item 5)
    with _lock.acquire_write_lock(
        out_path.with_suffix(out_path.suffix + ".lock"), op="write", unsafe_no_lock=unsafe_no_lock
    ):
        from .source import open_cell_source

        source = open_cell_source(adata)
        try:
            present = source.dropped_fields()
            if present:
                warnings.warn(
                    f"layout='cell' stores X + obs/var/uns/obsm/varm/obsp/varp; dropping {present}. "
                    f"Multi-matrix (.layers) is deferred.",
                    stacklevel=2,
                )
            if source.raw is not None:
                warnings.warn("layout='cell' does not store .raw; dropping it.", stacklevel=2)

            n_obs, n_vars = source.n_obs, source.n_vars
            row_nnz = source.row_nnz() if group_by is not None else None
            perm, groups_record = _resolve_storage_order(
                source.obs, row_nnz, group_by, reference, sort_within_group, sort_order
            )

            attempts = _plan_cell_codec(source.data_dtype(), codec=codec, data_dtype=data_dtype)

            # Fail early ONLY when the pure-Python encoder is the one that will run. The Rust
            # encoder has its own vendored FastPFor and never calls pyfastpfor, so the old
            # unconditional check made a plain `pip install cellstream` unable to write the DEFAULT
            # cell codec -- pyfastpfor has no linux wheel, and codec=None plans pfordelta first
            # for integer counts. zstd never needed it on either engine (Codex #217).
            #
            # `rust_available()` is coarser than the per-block `use_rust_for_cell_encode` gate,
            # which can still decline on dtype grounds. That case loses its early failure: the
            # encoder construction inside _encode_cell_block raises the same message from
            # codec._import_pyfastpfor instead, one step later and after a temp dir exists (which
            # the writer removes on the error path). Accepted deliberately -- the alternative is
            # keeping a check that is wrong for every ordinary install.
            from ._rust_dispatch import rust_available

            if not rust_available() and any(c == fmt.CODEC_PFORDELTA for c, _ in attempts):
                _import_pyfastpfor()

            index_dtype = fmt.choose_index_dtype(n_vars)

            # tmp_dir defaults to out_path.parent (spec §7.4): the output filesystem must hold the
            # final archive anyway, whereas /tmp typically cannot hold even one payload. But that dir
            # may not exist yet -- pack() creates it on the way out, and callers legitimately pass a
            # not-yet-created output dir (tests/cell/test_zstd_reader.py::
            # test_gather_rows_float32_and_int_stay_float32 calls test_zstd_writer.py::
            # _write(tmp_path / "f32", ...), whose out_path.parent -- tmp_path / "f32" -- pytest
            # never created).
            # mkdtemp(dir=...) requires it to EXIST, so create it first. The old default (/tmp) always
            # existed, which is why this only surfaces now.
            tmp_base = Path(tmp_dir) if tmp_dir is not None else out_path.parent
            tmp_base.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="cellstream_cell_", dir=str(tmp_base)) as tmpd:
                tmpd = Path(tmpd)

                # Encode OPTIMISTICALLY as the integer codec and let the encoder be the oracle:
                # it already determines integrality per value, for free, fused into its loop. Asking
                # up front instead cost an O(nnz) scan -- 86% of the write (146.8s -> 16.0s) and
                # +32 GB peak RSS at 6.51B nnz -- to answer a question the encode already answers.
                #
                # ONLY NonIntegerCountsError is retryable. A uint32 overflow of an INT source, a
                # duplicate index, a lossy cast, an I/O error must every one fail loud -- a genuine
                # overflow silently becoming a float archive is precisely the bug this must not
                # introduce. NonIntegerCountsError SUBCLASSES ValueError, so catching bare ValueError
                # here would swallow all of those and retry on them.
                for i, (try_codec, try_value_dtype) in enumerate(attempts):
                    try:
                        encoded = _encode_cell_source(
                            source,
                            perm,
                            groups_record,
                            n_obs,
                            n_vars,
                            tmpd,
                            resolved_codec=try_codec,
                            zstd_value_dtype=try_value_dtype,
                            block_size=block_size,
                            allow_lossy=allow_lossy,
                            payload_checksum=payload_checksum,
                            require_unique_indices=require_unique_indices,
                        )
                        break
                    except NonIntegerCountsError:
                        if i == len(attempts) - 1:
                            # No fallback left. An EXPLICIT codec='pfordelta' keeps its actionable
                            # message rather than leaking the encoder's -- the user chose the codec.
                            if codec == fmt.CODEC_PFORDELTA and source.data_dtype().kind == "f":
                                raise ValueError(
                                    "layout='cell' codec='pfordelta' requires integer counts; "
                                    "pass codec='zstd' to store floats."
                                ) from None
                            raise
                        nxt_codec, nxt_dtype = attempts[i + 1]
                        warnings.warn(
                            f"layout='cell': the values are not non-negative integers, so "
                            f"codec={try_codec!r} integer storage is impossible. Discarding that pass "
                            f"and re-encoding with codec='zstd' float storage ({nxt_dtype}). Pass "
                            f"data_dtype={nxt_dtype!r} to skip the wasted pass.",
                            UserWarning,
                            stacklevel=3,
                        )
                        _clear_cell_staging(tmpd)

                resolved_codec = encoded["codec"]
                schema_version = encoded["schema_version"]
                pyfastpfor_codec = encoded["pyfastpfor_codec"]
                index_dtype = encoded["index_dtype"]
                value_dtype = encoded["value_dtype"]
                x_data_dtype_min = encoded["x_data_dtype_min"]
                total_nnz = encoded["total_nnz"]
                payload_sha256 = encoded["payload_sha256"]

                # metadata tail — obs.parquet (storage order) + header.h5ad
                # (var + uns + obsm/varm/obsp/varp; cell-axis fields permuted to storage order, #229) +
                # manifest, routed through the shared helpers so write_cell_archive, CellArchiveWriter,
                # and concat all build/write the identical layout=cell schema (ONE source of truth).
                obs_perm = source.obs.iloc[perm].copy()
                # Ungrouped writes resolve to an identity perm (groups_record is None,
                # _resolve_storage_order writer.py:166); permuting obsm/obsp would be a semantic no-op
                # (obsp[perm,:][:,perm] reproduces the input). Pass None so permute_cell_axis_fields
                # short-circuits to shallow dict copies -- byte-identical output, zero reindex. (#240)
                cell_perm = None if groups_record is None else perm
                obsm_perm, obsp_perm = header.permute_cell_axis_fields(
                    source.obsm, source.obsp, cell_perm
                )
                manifest = _build_cell_manifest(
                    schema_version=schema_version,
                    codec=resolved_codec,
                    n_obs=n_obs,
                    n_vars=n_vars,
                    total_nnz=total_nnz,
                    value_dtype=value_dtype,
                    index_dtype=index_dtype,
                    x_data_dtype_min=x_data_dtype_min,
                    pyfastpfor_codec=pyfastpfor_codec,
                    group_by=group_by,
                    reference=(groups_record or {}).get("reference"),
                    sort_within_group=sort_within_group,
                    sort_order=sort_order,
                    payload_sha256=payload_sha256,
                    indices_canonical=require_unique_indices,
                )
                _write_cell_members_and_pack(
                    tmpd,
                    obs_storage=obs_perm,
                    var=source.var.copy(),
                    uns=source.uns,
                    obsm=obsm_perm,
                    varm=dict(source.varm),
                    obsp=obsp_perm,
                    varp=source.varp,
                    groups_record=groups_record,
                    manifest=manifest,
                    out_path=out_path,
                    overwrite=overwrite,
                )
        finally:
            source.close()
