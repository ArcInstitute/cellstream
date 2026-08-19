"""Rust dispatch for the per-cell pfordelta codec (mirrors cellstream.v2._rust_dispatch)."""

from __future__ import annotations

import os
import warnings

import numpy as np

from cellstream.errors import LossyCastError, NonIntegerCountsError

# EVERY _rust.* symbol this module dereferences. encode_cells_block (:309) and
# reorder_frames (:343) arrived in #219, after decode_cells/encode_cells -- an extension
# built in between reported rust_available() == True and then AttributeError'd, and
# reorder_frames' own `if not rust_available(): <python fallback>` could never fire.
# The set MUST stay in lockstep with the call sites below. (#282 item 4)
_CELL_ENTRY_POINTS = ("decode_cells", "encode_cells", "encode_cells_block", "reorder_frames")

try:
    from cellstream.v2 import _rust

    _HAVE = all(hasattr(_rust, _n) for _n in _CELL_ENTRY_POINTS)
except ImportError:  # pragma: no cover
    _rust = None
    _HAVE = False

# The Rust-side exception (subclasses ValueError). A tuple that is empty when the extension
# predates it makes `except _NonIntegerCounts` a no-op rather than an AttributeError.
_NonIntegerCounts = (
    (_rust.NonIntegerCounts,) if _HAVE and hasattr(_rust, "NonIntegerCounts") else ()
)

_RUST_SRC_DATA = frozenset({"float32", "float64", "uint16", "uint32", "int32", "int64"})
# indices int32/int64 ONLY: scipy CSR never uses unsigned index dtypes, and the Rust
# encode_cells match handles only int32/int64 indices. Keeping uint here would make
# use_rust_for_cell_encode return True and then RAISE in Rust instead of falling back to
# Python — the sets MUST match the Rust match arms (Gemini #212 r2).
_RUST_SRC_INDEX = frozenset({"int32", "int64"})
_RUST_ON_DISK = frozenset({"uint16", "uint32"})


def rust_available() -> bool:
    return _HAVE and os.environ.get("CELLSTREAM_DISABLE_RUST") != "1"


# Reasons the cell read path fell back to pure Python. One UserWarning per reason per process,
# so a long-running dataloader does not spam, but a benchmark log still records the path taken.
_FALLBACK_WARNED: set[str] = set()

# Covers BOTH ways _HAVE goes false -- it requires every name in _CELL_ENTRY_POINTS,
# so a stale extension that imports fine but predates the cell work lands here too. Saying only
# "built without the Rust extension" would send that user hunting for a build problem they do not
# have. rust_status() is what tells the two apart. (Copilot #247 checkpoint 2.)
_MISSING_EXT_MSG = (
    "the Rust extension (cellstream.v2._rust) is unavailable -- either it was not built, or it was "
    f"built without the full set of cell entry points ({'/'.join(_CELL_ENTRY_POINTS)}), "
    "which is what a wheel "
    "predating the cell work looks like. The pure-Python cell gather is far slower on this path "
    "(issue #244 measured ~72% of a scoring workload's wall clock inside it).\n"
    "Install or rebuild a wheel with the current extension, or set "
    "CELLSTREAM_ALLOW_PYTHON_FALLBACK=1 to proceed anyway (CELLSTREAM_DISABLE_RUST=1 also permits it, "
    "and forces the Python engine everywhere).\n"
    'Diagnostics -- distinguishes "not built" from "built but incomplete": '
    'python -c "import cellstream; print(cellstream.rust_status())"'
)

_REASON_TEXT = {
    "disabled_by_env": "CELLSTREAM_DISABLE_RUST=1 is set",
    "extension_missing": "the Rust extension (cellstream.v2._rust) is not available",
    "gate_declined": "this archive's codec/dtype combination is not one the Rust decoder can produce",
}


def _reset_fallback_warnings() -> None:
    """Test seam: forget which reasons have already warned."""
    _FALLBACK_WARNED.clear()


def check_cell_python_fallback(where: str) -> None:
    """Gate a cell-layout read that is about to run on the pure-Python decoder (#244).

    Classifies WHY the fallback was chosen and applies the policy:

    * ``disabled_by_env`` -- allow, warn once. Checked FIRST and deliberately:
      CELLSTREAM_DISABLE_RUST=1 is already an explicit request for the Python engine, it drives a
      whole job in .github/workflows/test.yml and the ``engine_env`` fixture in
      tests/cell/conftest.py, and it must never raise regardless of whether the extension is built.
    * ``extension_missing`` -- RAISE unless CELLSTREAM_ALLOW_PYTHON_FALLBACK=1. This is the silent
      degradation #244 was filed about, and it is fixable by installing a wheel.
    * ``gate_declined`` -- allow, warn once, NEVER raise. Rust genuinely cannot produce this
      dtype/codec combination, so Python is the correct answer, not a degradation, and no
      install would change it.
    """
    if os.environ.get("CELLSTREAM_DISABLE_RUST") == "1":
        reason = "disabled_by_env"
    elif not _HAVE:
        reason = "extension_missing"
    else:
        reason = "gate_declined"
    if reason == "extension_missing" and os.environ.get("CELLSTREAM_ALLOW_PYTHON_FALLBACK") != "1":
        raise RuntimeError(f"{where}: {_MISSING_EXT_MSG}")
    if reason not in _FALLBACK_WARNED:
        _FALLBACK_WARNED.add(reason)
        warnings.warn(
            f"{where}: falling back to the pure-Python cell decoder because "
            f"{_REASON_TEXT[reason]}. This path is substantially slower; see issue #244. "
            'Diagnostics: python -c "import cellstream; print(cellstream.rust_status())"',
            UserWarning,
            stacklevel=3,
        )


# zstd on-disk value dtypes the Rust frame decoder handles. Indices are ALWAYS uint32 for
# zstd (PR1). This MUST stay in lockstep with cell.rs FrameCodec::from_py's match arms: if
# Python lets a dtype through that Rust rejects, this returns True and then RAISES in Rust
# instead of falling back to Python (the Gemini #212 r2 failure mode).
_RUST_ZSTD_ON_DISK_VALUE = frozenset({"uint32", "float32", "float64"})

# Output-dtype sets the Rust cell decoder (cell.rs decode_cells match) can PRODUCE. These MUST
# stay in lockstep with cell.rs's match arms: a dtype Python lets through that Rust rejects
# returns True and then RAISES with no fallback (bug #218 -- the same shape as the encode gate,
# which gates on the on-disk value dtype it would otherwise never see).
_RUST_OUT_DATA = frozenset({"uint16", "uint32", "float32", "float64"})
_RUST_OUT_INDEX = frozenset({"int32", "int64", "uint16", "uint32"})


def use_rust_for_cell_decode(codec, on_disk_index, on_disk_value, out_data, out_index) -> bool:
    if not rust_available():
        return False
    # #218: the failure depends on the OUTPUT dtypes (from resolve_plan), not just on-disk. Reject
    # anything the Rust dispatch cannot produce -> Python fallback (correct) instead of a gate that
    # promises Rust and then RAISES with no fallback. Normalize via np.dtype(...).name so callers
    # may pass a dtype, a numpy scalar type, or a name string.
    #
    # isnative FIRST: `.name` drops byte order, so a non-native dtype like '>u4' reports "uint32"
    # and passes the set test -- but Rust matches on str(dtype), sees ">u4", and RAISES
    # "unsupported output dtypes" with no fallback. That is exactly the #218 bug class this gate
    # exists to prevent. resolve_plan does NOT normalize byte order (it preserves an explicit
    # data_dtype verbatim), but read_cell_bulk never actually reaches this: _csr_direct_assign
    # builds its CSR with sp.csr_matrix(shape, dtype=...), and scipy rejects a non-native dtype
    # outright. gather_rows(out_dtype=...) is the route that reaches it (#277) -- its
    # pure-Python branch uses the (data, indices, indptr) constructor, which has no such check
    # and casts correctly. So falling back here gives the caller the dtype they asked for
    # instead of an exception. See tests/cell/test_reencode_dtype.py for both halves.
    if not (np.dtype(out_data).isnative and np.dtype(out_index).isnative):
        return False
    if np.dtype(out_data).name not in _RUST_OUT_DATA:
        return False
    if np.dtype(out_index).name not in _RUST_OUT_INDEX:
        return False
    if codec == "pfordelta":
        return str(on_disk_index) in _RUST_ON_DISK and str(on_disk_value) in _RUST_ON_DISK
    if codec == "zstd":
        return str(on_disk_index) == "uint32" and str(on_disk_value) in _RUST_ZSTD_ON_DISK_VALUE
    return False


def use_rust_for_cell_encode(codec, src_data_dtype, src_index_dtype, on_disk_value_dtype) -> bool:
    """Gate the Rust cell encoder.

    Takes the ON-DISK value dtype, not just the source dtypes. That is the shape of bug #218:
    a gate that decides from dtypes it can see, while the failure depends on a dtype it never
    sees. The writer knows this one (``zstd_value_dtype``, from ``_plan_cell_codec``) -- so
    pass it.

    The accepted sets MUST match the Rust match arms EXACTLY. If Python lets through a case
    Rust rejects, this returns True and Rust then RAISES instead of falling back to the Python
    encoder (the Gemini #212 r2 failure mode).
    """
    if not rust_available():
        return False
    if str(src_data_dtype) not in _RUST_SRC_DATA or str(src_index_dtype) not in _RUST_SRC_INDEX:
        return False
    if codec == "pfordelta":
        # FastPFor always emits u32 words; the on-disk width is only a narrowing hint.
        return True
    if codec == "zstd":
        if str(on_disk_value_dtype) not in _RUST_ZSTD_ON_DISK_VALUE:
            return False
        # Float on-disk requires a FLOAT SOURCE. Rust's encode_cells rejects integer-source +
        # float-on-disk (mirroring _plan_cell_codec, which raises "data_dtype requires float
        # input"), so accepting it here would return True and then make Rust RAISE instead of
        # falling back to the Python encoder -- the #212 r2 failure mode this gate exists to
        # prevent. Unreachable through write_cell_archive today (float storage implies float
        # input), but the gate's contract IS "the two engines' accept-sets are identical", and
        # a latent violation becomes live the moment _plan_cell_codec changes (e.g. #219).
        # (Codex, PR #220 code review.)
        if str(on_disk_value_dtype) in ("float32", "float64"):
            return str(src_data_dtype) in ("float32", "float64")
        return True
    return False


def encode_cells(
    x_data,
    x_indices,
    x_indptr,
    perm,
    payload_out_path,
    *,
    n_threads,
    compute_checksum=True,
    check_duplicates=False,
    codec="pfordelta",
    value_dtype_on_disk="uint32",
    allow_lossy=False,
):
    # compute_checksum / check_duplicates default to the Rust-level defaults (checksum on,
    # dup-check off) so direct callers keep their prior behavior; write_cell_archive always
    # passes explicit values resolved from payload_checksum / require_unique_indices.
    # The Rust core reads data/indices/indptr via as_slice() (needs C-contiguous). CSR arrays
    # from scipy are contiguous, but perm can arrive strided; ascontiguousarray is a no-op for
    # already-contiguous inputs. indptr keeps its native int32/int64 (indptr_to_i64 widens).
    x_data = np.ascontiguousarray(x_data)
    x_indices = np.ascontiguousarray(x_indices)
    x_indptr = np.ascontiguousarray(x_indptr)
    perm = np.ascontiguousarray(perm, dtype=np.int64)
    try:
        return _rust.encode_cells(
            x_data,
            x_indices,
            x_indptr,
            perm,
            str(payload_out_path),
            n_threads,
            compute_checksum,
            check_duplicates,
            str(codec),
            str(value_dtype_on_disk),
            allow_lossy,
        )
    except _NonIntegerCounts as e:
        raise NonIntegerCountsError(str(e)) from e
    except ValueError as e:
        if "lossy" in str(e):
            raise LossyCastError(str(e)) from e
        raise


def decode_cells(
    payload_u8,
    offsets,
    indptr,
    row_ids,
    out_data,
    out_indices,
    out_indptr,
    *,
    codec,
    value_dtype_on_disk,
    n_vars,
    allow_lossy,
    n_threads,
):
    # The Rust core reads these via as_slice(), which requires C-contiguous arrays. row_ids in
    # particular can arrive non-contiguous from `X[rows]` with a reversed/strided/fancy index
    # (the pure-Python fallback iterated and didn't care). payload (mmap view) and offsets/indptr
    # (frombuffer) are already contiguous, so ascontiguousarray is a no-op for them; only a
    # non-contiguous row_ids gets copied. Output arrays are caller-allocated contiguous and
    # written in place, so they are NOT re-materialized here.
    #
    # The conversions are INSIDE the try: payload_u8 is rebound on the first line, so a failure
    # converting offsets/indptr/row_ids already has the mmap export pinned by this frame.
    try:
        payload_u8 = np.ascontiguousarray(payload_u8, dtype=np.uint8)
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        row_ids = np.ascontiguousarray(row_ids, dtype=np.int64)
        try:
            _rust.decode_cells(
                payload_u8,
                offsets,
                indptr,
                row_ids,
                out_data,
                out_indices,
                out_indptr,
                str(codec),
                str(value_dtype_on_disk),
                n_vars,
                allow_lossy,
                n_threads,
            )
        except ValueError as e:
            # decode never raises NonIntegerCounts (that is an encode-only signal), so there is
            # no NonIntegerCounts arm here -- a ValueError subclass would only reach this branch.
            if "lossy" in str(e):
                raise LossyCastError(str(e)) from e
            raise
    finally:
        # Drop EVERY reference this frame holds to caller-owned buffers, on EVERY exit path.
        #
        # payload_u8 IS the caller's mmap-backed view (ascontiguousarray is a no-op on an
        # already-contiguous uint8 array), and a raised exception's traceback pins THIS frame --
        # which keeps the mmap buffer exported. The caller's `finally: store.close()`
        # (reader.py read_cell_bulk / a gather_rows owner) then dies with
        # BufferError("cannot close exported pointers exist"), REPLACING the very error we are
        # propagating. This used to be a `del payload_u8` on the `except ValueError` arm only,
        # so a PyO3 panic, a MemoryError, a KeyboardInterrupt, or a failed conversion above all
        # still masked. (#287; the ValueError half was #220.)
        #
        # offsets/indptr are frombuffer views over `bytes` and the out_* arrays are
        # caller-allocated, so listing all seven is uniformity rather than a second fix -- it
        # costs nothing and removes the question of which argument happens to be buffer-backed
        # in a future release. It specifically does NOT make SharedMemory.close() succeed on
        # the output_backing="shm" error path: read_cell_bulk still holds data/indices/ip_view
        # in its own pinned frame, and the out_indptr it passes is a heap int64 scratch, not
        # the shm view. The segment NAME is unlinked either way. (codex, checkpoint 1.)
        #
        # `del` (not `= None`) because all seven are parameters -- always bound, so
        # UnboundLocalError is impossible -- and because `x = None` trips ruff F841.
        del payload_u8, offsets, indptr, row_ids, out_data, out_indices, out_indptr


def encode_cells_block(
    x_data,
    x_indices,
    x_indptr,
    payload_out_path,
    *,
    n_threads,
    check_duplicates=False,
    codec="pfordelta",
    value_dtype_on_disk="uint32",
    allow_lossy=False,
):
    """Encode ONE contiguous input-order block, APPENDING to payload_out_path (#219).
    Returns (frame_lens int64[block_rows], block_nnz, block_max)."""
    x_data = np.ascontiguousarray(x_data)
    x_indices = np.ascontiguousarray(x_indices)
    x_indptr = np.ascontiguousarray(x_indptr)
    try:
        return _rust.encode_cells_block(
            x_data,
            x_indices,
            x_indptr,
            str(payload_out_path),
            n_threads,
            check_duplicates,
            str(codec),
            str(value_dtype_on_disk),
            allow_lossy,
        )
    except _NonIntegerCounts as e:
        # Translate the raw rust.NonIntegerCounts into the engine-independent
        # cellstream.errors.NonIntegerCountsError, exactly as encode_cells does. write_cell_archive's
        # #216 optimistic-encode retry keys off that TYPE, and a STREAMING write that optimistically
        # tries the integer codec reaches Rust through THIS entry point (not encode_cells) -- so
        # without the translation a fractional/overflowing float in a streamed source would escape
        # as the untranslated rust type and the retry-to-float-storage would never fire.
        raise NonIntegerCountsError(str(e)) from e
    except ValueError as e:
        if "lossy" in str(e):
            raise LossyCastError(str(e)) from e
        raise


def reorder_frames(staging_path, out_path, in_offsets, perm):
    """Permute encoded frames input-order -> storage-order (#219). Codec-agnostic.
    Routes to the pure-Python implementation when Rust is unavailable."""
    in_offsets = np.ascontiguousarray(in_offsets, dtype=np.int64)
    perm = np.ascontiguousarray(perm, dtype=np.int64)
    if not rust_available():
        from .writer import _reorder_frames_python

        return _reorder_frames_python(staging_path, out_path, in_offsets, perm)
    return _rust.reorder_frames(str(staging_path), str(out_path), in_offsets, perm)
