"""#222 append-only CellArchiveWriter + concat_cell_archives (layout=cell)."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellstream import header
from cellstream import lock as _lock
from cellstream.errors import ArchiveExistsError, LossyCastError
from cellstream.narrow import _uint_dtype_from_max

from . import format as fmt
from .codec import _import_pyfastpfor, codec_for
from .writer import (
    _default_encode_threads,
    _file_size,
    _reorder_and_finalize,
    _resolve_push_codec,
    _stream_encode_block_python,
    _validate_col_indices,
    _validate_zstd_frame_bounds,
)


class CellArchiveWriter:
    """Append-only, push-model writer for one layout=cell .shad. Rows must arrive in ascending
    global order; finalize plans the storage permutation (identity fast-path when already ordered).
    Peak RAM O(block)+O(n_obs). The .shad's existence is the completion marker (pack tmp->rename).

    `finalize()` is the ONLY commit. `add_block` only encodes into a private tmp staging
    directory -- nothing under `path` exists yet. `finalize` plans the storage permutation from
    `obs`, reorders the staged frames (identity fast-path is a rename when already in final
    order), writes obs/header/manifest, and packs the result to `path` via the usual
    tmp-dir-then-atomic-rename. `abort()` -- and an un-finalized `__exit__`, clean or
    exceptional -- discards the staging directory instead; it never produces a `.shad`.

    Ordering contract (producer's responsibility, unverified): blocks given to `add_block` must
    arrive in ascending global row order, and `obs` given to `finalize` must be in that same
    order -- obs row i is the i-th row ever added. Only the total row COUNT is checked; an
    out-of-order caller silently gets a wrong archive. Each flushed (coalesced) block has its
    column indices sorted (`csr.sort_indices()`), matching `write_cell_archive`'s convention.

    Validation timing (#233): per-block validation is fused into the encoder, and coalescing moves
    the encode from `add_block` to the flush. So `LossyCastError` / `NonIntegerCountsError` /
    duplicate-index rejection surface at the flush that encodes the offending row -- a cap-triggered
    flush inside `add_block`, or the tail flush in `finalize` -- not necessarily synchronously from
    `add_block`. The `n_vars` column-count check and the "already finalized" guard remain synchronous
    in `add_block`. Fail-loud is preserved; only the timing of encode-fused errors moves.

    Parameters
    ----------
    path
        Output ``.shad`` path. Must not already exist unless ``overwrite=True``.
    n_vars
        Feature (column) count. Every block passed to ``add_block`` must have exactly this
        many columns.
    value_dtype
        Declared dtype of the values this writer will receive (e.g. an integer dtype for
        counts, ``float32``/``float64`` for floats). Resolved ONCE, up front, to a single
        (codec, on-disk dtype) plan -- unlike ``write_cell_archive``, there is no source
        matrix here to optimistically re-attempt against, so an incompatible
        ``codec``/``value_dtype`` pairing (e.g. ``codec="pfordelta"`` with a float
        ``value_dtype``) raises immediately instead of retrying.
    codec
        ``None`` (default) auto-selects from ``value_dtype``: integer -> ``"pfordelta"``,
        float -> ``"zstd"``. Or force ``"pfordelta"``/``"zstd"`` explicitly. Same semantics
        as ``write_cell_archive``'s ``codec`` parameter.
    data_dtype
        On-disk float width, for a float ``value_dtype`` stored via ``codec="zstd"``. Same
        semantics as ``write_cell_archive``'s ``data_dtype`` parameter.
    threads
        Encode thread count used by each flush. ``None`` (default) sizes to the flushed
        (coalesced) block's row count and the CPU allocation (``_default_encode_threads``).
    encode_block_rows, encode_block_nnz
        #233 internal coalescing caps. ``add_block`` buffers incoming CSRs and encodes only when
        the buffer reaches ``encode_block_rows`` rows OR ``encode_block_nnz`` nnz (whichever
        first), so many small pushes are encoded as one large, fully-parallel block (recovering
        ``write_cell_archive`` throughput) while peak RAM stays ~O(cap). Defaults 65_536 rows /
        64_000_000 nnz; ``None``/0 disables that cap. Byte-identical to encoding each pushed block
        separately.
    allow_lossy
        Gate for a lossy float down-cast (e.g. ``float64`` -> ``float32`` ``data_dtype``).
        Same semantics as ``write_cell_archive``'s ``allow_lossy`` parameter.
    overwrite
        When True, permit ``path`` to already exist (replaced atomically when ``finalize``
        packs). Default False raises ``ArchiveExistsError`` up front, in the constructor.
    payload_checksum
        When True, compute + store the streamed payload SHA-256 (enables
        ``CellStore.verify_payload()``). Default False is faster.
    require_unique_indices
        When True (default), reject a duplicate ``(cell, gene)`` entry within any block
        (raised from ``add_block``) and mark the finished archive canonical. Same semantics
        as ``write_cell_archive``'s parameter of the same name.
    unsafe_no_lock
        Escape hatch: skip the archive write lock. Same meaning and default as
        ``write_cell_archive``.
    tmp_dir
        Scratch directory for the write staging area, used by both ``add_block`` (per-block
        encode output) and ``finalize`` (reorder). Defaults to ``path.parent``, not
        ``TMPDIR``/``/tmp`` -- same rationale as ``write_cell_archive``'s ``tmp_dir``.
    """

    def __init__(
        self,
        path,
        *,
        n_vars,
        value_dtype,
        codec=None,
        data_dtype=None,
        threads=None,
        encode_block_rows=65_536,
        encode_block_nnz=64_000_000,
        allow_lossy=False,
        overwrite=False,
        payload_checksum=False,
        require_unique_indices=True,
        unsafe_no_lock=False,
        tmp_dir=None,
    ):
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise ArchiveExistsError(f"{self.path} already exists. Pass overwrite=True.")
        # A push writer owns its output for its whole lifetime, so the lock spans __init__ ->
        # finalize()/abort()/__exit__ rather than just the final pack. Held in an ExitStack so
        # every exit path releases it with one idempotent close(). (#280 item 5)
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(
            _lock.acquire_write_lock(
                self.path.with_suffix(self.path.suffix + ".lock"),
                op="write",
                unsafe_no_lock=unsafe_no_lock,
            )
        )
        try:
            self.n_vars = int(n_vars)
            self._overwrite = overwrite
            self._allow_lossy = allow_lossy
            self._payload_checksum = payload_checksum
            self._require_unique = require_unique_indices
            self._threads = threads
            # #233: coalesce small add_block calls into large parallel encode units. 0/None = disabled.
            self._encode_block_rows = int(encode_block_rows) if encode_block_rows else 0
            self._encode_block_nnz = int(encode_block_nnz) if encode_block_nnz else 0
            self._pending, self._pending_rows, self._pending_nnz = [], 0, 0
            self._codec, self._on_disk_value, self._schema, self._pyfastpfor = _resolve_push_codec(
                value_dtype, codec, data_dtype
            )
            self._index_dtype = (
                "uint32" if self._codec == fmt.CODEC_ZSTD else fmt.choose_index_dtype(n_vars)
            )
            # Fail early only when the pure-Python encoder will actually run -- same reasoning
            # as write_cell_archive: the Rust encoder never calls pyfastpfor, and requiring the
            # `cell` extra (no linux wheel) to push-write the default codec is a bug, not a
            # safety net.
            from ._rust_dispatch import rust_available

            if not rust_available() and self._codec == fmt.CODEC_PFORDELTA:
                _import_pyfastpfor()
            # Built on first use by the `_py_enc` property; see it for why nothing is validated
            # away by deferring.
            self._py_enc_cache = None
            tmp_base = Path(tmp_dir) if tmp_dir is not None else self.path.parent
            tmp_base.mkdir(parents=True, exist_ok=True)
            self._tmpd = Path(tempfile.mkdtemp(prefix="cellstream_cellw_", dir=str(tmp_base)))
            self._staging = self._tmpd / "x_payload_staging"
            self._staging.write_bytes(b"")
            self._frame_lens, self._row_nnz = [], []
            self._total_nnz, self._max_value, self._n_obs = 0, 0, 0
            self._finalized = False
        except BaseException:
            # A failure after the acquire -- the (now engine-gated) pyfastpfor check, mkdtemp,
            # an unresolvable codec -- must not strand the lock for the process's lifetime.
            self._stack.close()
            raise

    @property
    def _py_enc(self):
        """The pure-Python block encoder, built on first use and then cached.

        Only ``_flush``'s fallback branch touches it, so on the Rust path it is never built and
        pyfastpfor is never imported. Cached because the pfordelta encoder owns a growable
        scratch buffer reused across blocks -- rebuilding per flush would reallocate it.

        Deferring validates nothing away: ``_resolve_push_codec`` has already checked the codec
        name and resolved ``_on_disk_value`` to one of the allowed dtype names, and
        ``_index_dtype`` comes from ``choose_index_dtype`` / the zstd constant -- so
        ``codec_for(...).new_encoder()`` cannot fail on a bad argument here, only on a missing
        pyfastpfor, which is the whole point.
        """
        if self._py_enc_cache is None:
            self._py_enc_cache = codec_for(
                self._codec, self._index_dtype, self._on_disk_value
            ).new_encoder()
        return self._py_enc_cache

    def add_block(self, csr_block):
        if self._finalized:
            raise RuntimeError(
                "CellArchiveWriter already finalized or aborted; create a new writer."
            )
        csr = sp.csr_matrix(csr_block)
        if csr.shape[1] != self.n_vars:
            raise ValueError(f"block has {csr.shape[1]} columns, expected n_vars={self.n_vars}.")
        self._pending.append(csr)
        self._pending_rows += csr.shape[0]
        self._pending_nnz += csr.nnz
        if (self._encode_block_rows and self._pending_rows >= self._encode_block_rows) or (
            self._encode_block_nnz and self._pending_nnz >= self._encode_block_nnz
        ):
            self._flush()

    def _flush(self):
        """Encode all buffered blocks as ONE coalesced block at full n_threads, appended to staging.

        Byte-identical to encoding each pushed block separately: the encoder emits one independent
        frame per row in row order with no block header, so coalescing changes only rows-per-encode
        and the thread count -- never the frame bytes or their order (#233 spec S4)."""
        if not self._pending:
            return
        from ._rust_dispatch import encode_cells_block, use_rust_for_cell_encode

        block = sp.vstack(self._pending, format="csr")
        block.sort_indices()
        # Capture attribution BEFORE freeing _pending (below): with coalescing a validation
        # rejection surfaces here at flush, not at the add_block() that supplied the bad rows, so
        # the error must name the buffered add-order row span. (@nick-youngblut, PR #230.)
        n_buffered = len(self._pending)
        row_start, row_end = self._n_obs, self._n_obs + self._pending_rows
        # Free the buffered inputs BEFORE encoding: vstack already copied into `block`, so keeping
        # `_pending` alive through the encode would double peak RAM to ~2x the cap (#233 spec S6).
        self._pending, self._pending_rows, self._pending_nnz = [], 0, 0
        threads = (
            self._threads if self._threads is not None else _default_encode_threads(block.shape[0])
        )
        # Transaction marks. The pure-Python arm writes each frame inside its per-row loop, so a
        # per-row rejection leaves the already-encoded rows appended while the except path
        # re-raises WITHOUT recording their lengths -- in_offsets[-1] can then never again equal
        # the staging size and finalize() dies, potentially hours later (#282 item 10). The
        # bookkeeping moves INSIDE the try as well, so a failure between the encode and the
        # append (realistically MemoryError) cannot leave the file and the metadata disagreeing.
        # Engine-agnostic on purpose: a no-op on the Rust arm (which encodes into per-worker
        # .partN files and concatenates only on success), and writing it once is what makes the
        # two arms observably identical -- the parity contract _validate_col_indices states below.
        # Through an fd, never a path stat: a stale-short mark would truncate away bytes
        # belonging to blocks flushed EARLIER that are still counted in _frame_lens, killing
        # a good push write instead of only discarding the block that failed (#305).
        mark = _file_size(self._staging)
        snap = (
            len(self._frame_lens),
            len(self._row_nnz),
            self._total_nnz,
            self._max_value,
            self._n_obs,
        )
        try:
            # BEFORE the engine dispatch: Rust and the Python fallback must reject the same
            # inputs. Inside the try so #230's add-order row-span note attaches (#264).
            _validate_col_indices(block.indices, self.n_vars)
            if self._codec == fmt.CODEC_ZSTD:
                _validate_zstd_frame_bounds(
                    block.indptr,
                    value_itemsize=fmt.np_dtype(self._on_disk_value).itemsize,
                    row_offset=row_start,  # add-order global row
                )
            if use_rust_for_cell_encode(
                self._codec, block.data.dtype, block.indices.dtype, self._on_disk_value
            ):
                blens, bnnz, bmax = encode_cells_block(
                    block.data,
                    block.indices,
                    block.indptr,
                    self._staging,
                    n_threads=threads,
                    check_duplicates=self._require_unique,
                    codec=self._codec,
                    value_dtype_on_disk=self._on_disk_value,
                    allow_lossy=self._allow_lossy,
                )
            else:
                with open(self._staging, "ab") as fh:
                    blens, bnnz, bmax = _stream_encode_block_python(
                        block,
                        self._py_enc,
                        fmt.np_dtype(self._index_dtype),
                        fmt.np_dtype(self._on_disk_value),
                        require_unique_indices=self._require_unique,
                        payload_fh=fh,
                        allow_lossy=self._allow_lossy,
                    )
            self._frame_lens.append(np.asarray(blens, dtype=np.int64))
            self._row_nnz.append(np.diff(block.indptr).astype(np.int64))
            self._total_nnz += int(bnnz)
            self._max_value = max(self._max_value, int(bmax))
            self._n_obs += block.shape[0]
        except BaseException as e:
            # BaseException, not Exception: a KeyboardInterrupt landing inside the encode would
            # otherwise skip the rollback entirely -- the same gap #287 documents elsewhere.
            del self._frame_lens[snap[0] :]
            del self._row_nnz[snap[1] :]
            self._total_nnz, self._max_value, self._n_obs = snap[2], snap[3], snap[4]
            try:
                with open(self._staging, "r+b") as fh:
                    fh.truncate(mark)
            except BaseException as rollback_exc:
                # Cannot roll back -> the writer really is unusable. abort() is exactly the right
                # terminal action and does all three things this needs: sets _finalized (so
                # add_block()/finalize() give the existing "already finalized or aborted; create a
                # new writer" message instead of a confusing offsets mismatch at finalize()),
                # rmtree's the staging tmpd, and closes the ExitStack. Setting _finalized by hand
                # instead would make __exit__ skip abort() and leak BOTH the flock and the tmpd.
                #
                # BaseException, not OSError: this rollback is the LAST thing standing between a
                # failed flush and a flock stranded for the life of the process, so a
                # KeyboardInterrupt or MemoryError landing in these two syscalls must not be the
                # single path that skips abort(). (codex, checkpoint 2.)
                self.abort()
                e.add_note(
                    "cellstream CellArchiveWriter: the staging file could not be rolled back "
                    f"after this error ({rollback_exc!r}); this writer is no longer usable. "
                    "Create a new writer."
                )
                if isinstance(rollback_exc, (KeyboardInterrupt, SystemExit)):
                    # ...but do not SWALLOW an interrupt into a note on someone else's error.
                    # abort() has already released the flock and the staging tmpd, so letting the
                    # interrupt win over `e` costs nothing and keeps Ctrl-C meaning Ctrl-C.
                    raise
            # The Rust arm encodes into per-worker `<staging>.partN` files and removes them only
            # on its successful concat path (rust/src/cell.rs), so a REJECTED Rust flush leaves
            # one coalesced block's worth of scratch behind until finalize()/abort() rmtrees the
            # staging dir. Verified: an 8-thread rejected flush leaves .part0..part7.
            #
            # Best-effort, and deliberately NOT abort()-on-failure the way a failed truncate is:
            # a stale part is never READ again (the concat loop only ever opens the parts of its
            # own call, and reorder_frames reads x_payload_staging alone), so this is a space
            # leak, not a correctness one -- and tearing down a multi-hour push write over an
            # undeletable scratch file would be the worse trade.
            #
            # AFTER the truncate, never before: the truncate is the load-bearing half of the
            # rollback, so nothing that can raise may run ahead of it -- an interrupt in the sweep
            # would otherwise restore the bookkeeping while leaving the appended staging bytes,
            # which is exactly the inconsistent writer this whole transaction exists to prevent.
            # The outer suppress covers the directory enumeration, the inner one each unlink.
            # (codex, checkpoint 2.)
            with contextlib.suppress(OSError):
                for _part in self._staging.parent.glob(self._staging.name + ".part*"):
                    with contextlib.suppress(OSError):
                        _part.unlink()
            # add_note preserves the exact type/message/__cause__/traceback (unlike reconstructing
            # the exception). add-order rows = the order add_block() was called, which the caller can
            # map back to its own input; NOT the final storage order (decided by the perm at
            # finalize). Notes show in the traceback and via e.__notes__.
            note = (
                f"cellstream CellArchiveWriter: this error surfaced while flushing {n_buffered} "
                f"coalesced add_block() call(s) spanning add-order rows [{row_start}, {row_end})."
            )
            # The row span is true for ANY error and always kept. The "offending value" attribution
            # is only true for a bad value in the buffered rows: within this encode the reachable
            # data errors are exactly LossyCastError plus the two ValueError-typed ones --
            # NonIntegerCountsError (subclasses ValueError) and the duplicate-index rejection (a
            # plain ValueError, both engines). So (LossyCastError, ValueError) attaches it for those
            # and NOT for OSError (ENOSPC/PermissionError), MemoryError, or a threading RuntimeError,
            # which would otherwise be pointed at the user's data. (#241, @nick-youngblut)
            if isinstance(e, (LossyCastError, ValueError)):
                note += (
                    " With encode-block coalescing the offending value may be in any of those "
                    "buffered blocks, not only the most recent add_block()."
                )
            e.add_note(note)
            raise

    def finalize(
        self,
        *,
        obs,
        var,
        uns=None,
        group_by=None,
        reference=None,
        sort_within_group=None,
        sort_order=None,
    ):
        if self._finalized:
            raise RuntimeError(
                "CellArchiveWriter already finalized or aborted; create a new writer."
            )
        try:
            self._flush()
            if len(obs) != self._n_obs:
                raise ValueError(f"obs has {len(obs)} rows but {self._n_obs} were added.")
            if len(var) != self.n_vars:
                raise ValueError(f"var has {len(var)} rows but n_vars={self.n_vars}.")
            all_lens = (
                np.concatenate(self._frame_lens) if self._frame_lens else np.empty(0, np.int64)
            )
            in_offsets = np.concatenate([[0], np.cumsum(all_lens)]).astype(np.int64)
            row_nnz = np.concatenate(self._row_nnz) if self._row_nnz else np.empty(0, np.int64)
            if self._codec == fmt.CODEC_PFORDELTA:
                value_dtype = fmt.choose_value_dtype(self._max_value)
                x_data_dtype_min = (
                    _uint_dtype_from_max(self._max_value).name
                    if self._total_nnz
                    else np.dtype(np.uint8).name
                )
            else:
                value_dtype = self._on_disk_value
                x_data_dtype_min = (
                    (
                        _uint_dtype_from_max(self._max_value).name
                        if self._total_nnz
                        else np.dtype(np.uint8).name
                    )
                    if self._on_disk_value == "uint32"
                    else self._on_disk_value
                )
            codec_meta = {
                "schema_version": self._schema,
                "codec": self._codec,
                "pyfastpfor_codec": self._pyfastpfor,
                "index_dtype": self._index_dtype,
                "value_dtype": value_dtype,
                "x_data_dtype_min": x_data_dtype_min,
                "total_nnz": self._total_nnz,
                "indices_canonical": self._require_unique,
            }
            _reorder_and_finalize(
                self._tmpd,
                self._staging,
                in_offsets,
                row_nnz,
                obs=obs,
                var=var,
                uns=dict(uns) if uns else {},
                group_by=group_by,
                reference=reference,
                sort_within_group=sort_within_group,
                sort_order=sort_order,
                codec_meta=codec_meta,
                n_vars=self.n_vars,
                out_path=self.path,
                overwrite=self._overwrite,
                payload_checksum=self._payload_checksum,
            )
        finally:
            # Mark terminal on BOTH success and failure: the finally deletes the staging tmpd, so a
            # failed finalize (obs/var mismatch, or a _reorder_and_finalize error) leaves the writer
            # unusable. Flagging it here makes a subsequent add_block()/finalize() raise a clear
            # "already finalized or aborted" error instead of a confusing I/O error on the now-deleted
            # staging path. (Copilot, PR #230.)
            self._finalized = True
            self._discard_staging_and_release()

    def _discard_staging_and_release(self):
        """Terminal cleanup: drop the staging tmpd, then ALWAYS release the archive lock.

        The release sits in a `finally` because rmtree's ignore_errors=True swallows OSError
        but NOT KeyboardInterrupt / SystemExit / MemoryError -- and without it, an interrupt
        landing in the removal would strand the flock for the life of the process on a writer
        the caller still holds a reference to. __exit__ already had this shape; finalize() and
        abort() now match it. close() is idempotent. (codex, checkpoint 2.)
        """
        try:
            shutil.rmtree(self._tmpd, ignore_errors=True)
        finally:
            self._stack.close()

    def abort(self):
        self._pending, self._pending_rows, self._pending_nnz = [], 0, 0
        self._finalized = True
        self._discard_staging_and_release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Abort on ANY exit when not finalized (clean OR exception): a forgotten finalize() must not
        # leak the staging tmpd. abort() discards, never commits -- no silent .shad. (Gemini, PR #230.)
        try:
            if not self._finalized:
                self.abort()
        finally:
            self._stack.close()  # unconditional: _finalized can be set without releasing
        return False


def _ann_value_equal(va, vb):
    """Best-effort equality for ONE uns/varm/varp value; conservatively False on anything it cannot
    confirm equal.

    Sparse and pandas need explicit arms: `bool(va == vb)` RAISES for both (sparse `==` yields a
    sparse boolean matrix, DataFrame `==` an elementwise frame), the fallback's bare `except` then
    swallows it, and IDENTICAL values get reported unequal -- which would make #237's varm/varp
    require-equal check unusable for the two most common varm/varp types. The arms are mutually
    exclusive (a sparse matrix has no `.equals`, a pandas object is not `issparse`), so their order
    is not load-bearing. Sparse is compared with `!=`, NOT `==`: scipy emits a
    SparseEfficiencyWarning for sparse `==` sparse and none for `!=`."""
    if isinstance(va, dict) and isinstance(vb, dict):
        return _uns_equal(va, vb)
    if sp.issparse(va) or sp.issparse(vb):
        if not (sp.issparse(va) and sp.issparse(vb)) or va.shape != vb.shape:
            return False
        try:
            return int((va != vb).nnz) == 0
        except Exception:
            return False
    if hasattr(va, "equals") or hasattr(vb, "equals"):  # pandas DataFrame/Series/Index
        if not (hasattr(va, "equals") and hasattr(vb, "equals")):
            return False
        try:
            return bool(va.equals(vb))
        except Exception:
            return False
    if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(va), np.asarray(vb)))
        except Exception:
            return False
    try:
        return bool(va == vb)
    except Exception:
        return False


def _uns_equal(a, b):
    """Best-effort deep equality for two uns/varm/varp dicts; conservatively False (-> fail loud) on
    anything it cannot confirm equal (nested dict / sparse / pandas / ndarray / scalar)."""
    if set(a.keys()) != set(b.keys()):
        return False
    return all(_ann_value_equal(a[k], b[k]) for k in a)


def _widest(names):
    """Widest of a set of on-disk dtype names (uint8<uint16<uint32; floats must already match)."""
    order = {"uint8": 0, "uint16": 1, "uint32": 2}
    ns = list(names)
    if any(n not in order for n in ns):  # float widths: require agreement
        if len(set(ns)) != 1:
            raise ValueError(f"incompatible float value dtypes across parts: {sorted(set(ns))}")
        return ns[0]
    return max(ns, key=lambda n: order[n])


def _inherited_reference_mask(stores, reference, obs, group_by):
    """Preserve reference MEMBERSHIP positionally when the reference was inherited.

    Only for the INHERITED reference -- the label the input archives recorded, written by
    whatever version produced them. Re-resolving that historical label against the concatenated
    obs is unsafe across the #294 naming change, in two distinct ways, both measured:

    * a pre-#294 archive named a missing label by dtype (``'None'`` for object+None, ``'<NA>'``
      for nullable-string+pd.NA). Those names no longer exist, so the label matched nothing and
      concat failed with "Reference labels not found" on archives that concatenated fine before;
    * worse, and SILENT: a legacy archive could hold a literal ``'<NA>'`` *and* real ``pd.NA``
      under one reference label ``'<NA>'``. That label still resolves -- to the literal rows
      only -- so the missing rows quietly stop being reference. Measured on two 24-row parts,
      the reference fell from 48 rows to 32 with no error.

    Each part's own group record is the ground truth: validation guarantees the reference
    records are a leading prefix ``[0, R_i)``, so a positional mask over the concatenated input
    order reproduces exactly what each part recorded, independent of how labels are spelled.
    ``resolve_reference`` already accepts a boolean mask of length ``n_obs``, so this needs no
    change downstream; ``_resolve_storage_order`` applies it before the sort.

    Applies ONLY when the output is grouped on the same column the sources were. A caller can
    override ``group_by`` while leaving ``reference`` to inherit, and the old membership then
    means nothing on the new axis: sources grouped by ``old=[r,r,x,x]`` with reference ``r``
    carry the mask ``[T,T,F,F]``, but concatenating them with ``group_by="new"`` where
    ``new=[a,a,r,r]`` must make ``r`` the reference, not ``a`` -- measured, without this guard
    the output's leading group was ``a``. On a different axis the inherited LABEL is the only
    meaningful instruction, so fall back to resolving it.

    Does NOT fail open. Once the reference is inherited and non-null, every valid part must be
    able to report a validated prefix count; a part whose group record does not validate raises
    out of ``reference_row_count()``, and swallowing that would let a corrupt source be
    republished whenever its label happened to resolve. A caller-supplied ``reference`` never
    reaches here and is still resolved by label.

    The ``n_obs == 0`` branch below is defensive, not a live path: a zero-row archive cannot
    carry a reference at all (``resolve_reference`` has no labels to match), so it records
    ``reference=None`` and ``_inherit`` rejects it alongside a referenced part before this
    runs. Measured, not assumed.
    """
    if any(s.manifest.get("group_by") != group_by for s in stores):
        return reference
    counts = []
    for s in stores:
        c = s.reference_row_count()  # propagates IncompatibleSchemaError -- see the docstring
        if c is None:
            if int(s.n_obs) != 0:
                raise ValueError(
                    f"concat: a part records reference={reference!r} in its manifest but its "
                    f"group record carries no reference prefix; the part is inconsistent."
                )
            c = 0
        counts.append(int(c))
    # ONE allocation for the whole mask, filled in place: the reference is a leading prefix
    # [0, c) within each part, so nothing needs a per-part array and a concatenate -- which
    # would transiently hold both the per-part arrays and the joined result.
    mask = np.zeros(sum(int(s.n_obs) for s in stores), dtype=bool)
    base = 0
    for s, c in zip(stores, counts, strict=True):
        mask[base : base + c] = True
        base += int(s.n_obs)
    if mask.shape[0] != len(obs):
        raise ValueError(
            f"concat: reference mask length {mask.shape[0]} != concatenated obs {len(obs)}; "
            f"the parts' row counts and their obs disagree."
        )
    return mask


def concat_cell_archives(
    parts,
    out,
    *,
    group_by=None,
    reference=None,
    sort_within_group=None,
    sort_order=None,
    overwrite=False,
    payload_checksum=False,
    unsafe_no_lock=False,
    tmp_dir=None,
    annotations="preserve",
):
    """Merge N self-describing cell archives into one canonical archive by BYTE-COPYING frames --
    no decode, no re-encode. Group/sort params default to the parts' manifests (must agree).
    Peak RAM O(block)+O(n_obs), plus the annotation cost below when annotations are preserved.

    The output is committed atomically, same as ``CellArchiveWriter.finalize``: nothing under
    ``out`` exists until the final ``pack()`` (inside the shared ``_reorder_and_finalize`` tail)
    succeeds, via the usual tmp-dir-then-atomic-rename. A failure partway through -- a manifest
    disagreement, a truncated payload, an aborted byte-copy -- leaves any partial state in a temp
    directory that is removed in a ``finally``, never at ``out``.

    Parameters
    ----------
    parts
        Archives to merge, each either a filesystem path (opened via ``open_cell`` and closed
        again before returning) or an already-open ``CellStore`` (left open -- ownership stays
        with the caller). Must agree on ``codec``, ``n_vars``, ``pyfastpfor_codec``,
        ``schema_version``, integer-vs-float value storage, and ``var`` (feature space); float
        parts must additionally share ``value_dtype_on_disk``. Must be non-empty.
    out
        Output ``.shad`` path for the merged archive. Must not already exist unless
        ``overwrite=True``.
    group_by, reference, sort_within_group, sort_order
        Storage-order controls, same meaning as ``write_cell_archive``'s parameters of the
        same names. ``None`` (default) inherits the value from the parts' manifests, which
        must all agree; pass explicitly to override, or to merge parts whose manifests
        disagree on that setting.
    overwrite
        When True, permit ``out`` to already exist (replaced atomically by the final pack).
        Default False raises ``ArchiveExistsError`` up front.
    payload_checksum
        When True, compute + store the merged payload's SHA-256. Default False.
    unsafe_no_lock
        Escape hatch: skip the archive write lock. Same meaning and default as
        ``write_cell_archive``.
    tmp_dir
        Scratch directory for the merge staging area (byte-copied payload + any reorder).
        Defaults to ``out.parent``; same rationale as ``write_cell_archive``'s ``tmp_dir``.
    annotations
        ``"preserve"`` (default) merges the parts' ``obsm``/``varm``/``obsp``/``varp`` into the
        output header: ``obsm`` is row-concatenated, ``obsp`` is assembled **block-diagonal**
        (``anndata.concat(pairwise=True)`` semantics -- graphs computed per part have no
        cross-part edges), and ``varm``/``varp`` are inherited from part 0 after a require-equal
        check across the rest (all parts share ``var``). Every part must carry the same keys, the
        same container kind per key, and matching widths; anything else raises. Value dtypes may
        differ and promote.

        ``"drop"`` merges ``X``/``obs``/``var``/``uns`` only and never reads the parts'
        annotations -- the escape hatch from the extra cost: preservation adds
        O(n_obs x sum of obsm widths) + O(sum of nnz(obsp)) to the peak RAM above, and a **dense**
        ``obsp`` is the sharp edge, since the block-diagonal result is (sum n_i)^2 -- N times the
        parts' combined dense size for N parts.
    """
    from .reader import CellStore, open_cell

    out = Path(out)
    if out.exists() and not overwrite:
        raise ArchiveExistsError(f"{out} already exists. Pass overwrite=True.")
    # Existence check first, then acquire -- write_packed's order, and the same rationale as
    # write_cell_archive's: a plainly-existing output keeps its ArchiveExistsError. The lock
    # spans the whole merge, because the byte-copy publishes at the tail pack(). (#280 item 5)
    with _lock.acquire_write_lock(
        out.with_suffix(out.suffix + ".lock"), op="write", unsafe_no_lock=unsafe_no_lock
    ):
        if len(parts) == 0:
            raise ValueError("concat_cell_archives: parts is empty.")
        if annotations not in ("preserve", "drop"):
            raise ValueError(
                f"concat: annotations must be 'preserve' or 'drop', got {annotations!r}."
            )

        stores, owned = [], []
        tmpd = None
        try:
            for p in parts:  # inside try: a mid-loop open_cell() failure still closes prior stores
                if isinstance(p, CellStore):
                    stores.append(p)
                    owned.append(False)
                else:
                    stores.append(open_cell(p))
                    owned.append(True)
            m0 = stores[0].manifest
            for s in stores[1:]:
                m = s.manifest
                for key in ("codec", "n_vars", "pyfastpfor_codec", "schema_version"):
                    if m.get(key) != m0.get(key):
                        raise ValueError(
                            f"concat: parts disagree on {key!r} ({m.get(key)} != {m0.get(key)})."
                        )
                # Reject mixing float and integer parts explicitly (same codec 'zstd' can be either, and
                # their frames are NOT mutually decodable) -- fail fast + clear, BEFORE the byte-copy,
                # rather than falling through to _widest's confusing message. (Gemini, PR #230.)
                is_float0 = m0["value_dtype_on_disk"] in ("float32", "float64")
                is_float = m["value_dtype_on_disk"] in ("float32", "float64")
                if is_float0 != is_float:
                    raise ValueError("concat: cannot mix float and integer parts.")
                if is_float0 and m["value_dtype_on_disk"] != m0["value_dtype_on_disk"]:
                    raise ValueError("concat: float parts must share value_dtype_on_disk.")
                if not stores[0].var.equals(s.var):
                    raise ValueError("concat: parts have different var (feature space).")
                # Require-equal, not first-wins: silently keeping stores[0].uns and never checking
                # the rest is a known bug class here (CellArchiveSource.uns unconditionally returning
                # {} sailed through every existing test undetected -- see conftest.assert_archives_
                # equivalent's docstring). uns may hold numpy arrays / nested dicts, so a naive `!=`
                # can raise -- _uns_equal is array-aware and conservatively False (fail loud) on
                # anything it cannot confirm equal. (spec #222 SS6.2/SS13.)
                if not _uns_equal(stores[0].uns, s.uns):
                    raise ValueError(
                        "concat: parts have different uns; concat requires identical uns "
                        "(drop or align uns before concatenating)."
                    )

            # obsm/varm/obsp/varp live in a part's header.h5ad (#229). Assemble them HERE, in the
            # pre-flight: a part's annotations are a metadata-only read, so a cross-part mismatch fails
            # in milliseconds instead of after a multi-GB byte-copy. (#237, replacing #232's fail-loud
            # placeholder guard.)
            if annotations == "drop":
                # Explicit: do not even READ the parts' annotations, and do not warn -- the caller asked
                # for X/obs/var/uns only. This is also the escape hatch from preservation's extra RAM
                # cost (see the docstring).
                obsm_cat, varm_cat, obsp_cat, varp_cat = {}, {}, {}, {}
            else:
                obsm_cat, obsp_cat = header.concat_cell_axis_fields(
                    [s.obsm for s in stores], [s.obsp for s in stores]
                )
                # Feature axis: all parts share var (checked above), so varm/varp are INHERITED from
                # part 0 after a require-equal check across the rest -- the same require-equal-not-
                # first-wins policy already applied to uns, and anndata.concat(merge="same") semantics.
                # _uns_equal is array-, pandas- and sparse-aware and conservatively False on anything it
                # cannot confirm equal.
                for field in ("varm", "varp"):
                    ref_field = getattr(stores[0], field)
                    for i, s in enumerate(stores[1:], start=1):
                        if not _uns_equal(ref_field, getattr(s, field)):
                            raise ValueError(
                                f"concat: parts have different {field} (part {i} differs from part 0); "
                                f"concat requires identical {field} because all parts share var "
                                f'(align them, or pass annotations="drop").'
                            )
                varm_cat, varp_cat = dict(stores[0].varm), dict(stores[0].varp)

            def _inherit(v, key):
                if v is not None:
                    return v
                vals = [s.manifest.get(key) for s in stores]
                if any(x != vals[0] for x in vals[1:]):
                    raise ValueError(f"concat: parts disagree on {key!r}; pass it explicitly.")
                return vals[0]

            group_by = _inherit(group_by, "group_by")
            caller_supplied_reference = reference is not None
            reference = _inherit(reference, "reference")
            sort_within_group = _inherit(sort_within_group, "sort_within_group")
            sort_order = _inherit(sort_order, "sort_order")

            tmp_base = Path(tmp_dir) if tmp_dir is not None else out.parent
            tmp_base.mkdir(parents=True, exist_ok=True)
            tmpd = Path(tempfile.mkdtemp(prefix="cellstream_cellcat_", dir=str(tmp_base)))
            staging = tmpd / "x_payload_staging"

            offset_parts = []
            base_off = 0
            base_row = 0
            row_nnz_parts, obs_parts, total_nnz = [], [], 0
            with open(staging, "wb") as sf:
                for s in stores:
                    # BEFORE the byte-copy: concat byte-copies frames without decoding, so it
                    # cannot CREATE an over-cap frame -- but a part written by a pre-fix release
                    # can carry one, and merging it would republish an archive neither engine can
                    # read (#282 item 9). Checking first means a bad part is rejected without
                    # writing GBs into staging. The manifest key is value_dtype_on_disk (there is
                    # no "value_dtype" key), matching the require-equal checks above.
                    if m0.get("codec") == fmt.CODEC_ZSTD:
                        _validate_zstd_frame_bounds(
                            s.indptr,
                            value_itemsize=fmt.np_dtype(m0["value_dtype_on_disk"]).itemsize,
                            row_offset=base_row,
                        )
                    base_row += len(s.indptr) - 1
                    p_path, base, length = s._packed.member_location(fmt.PAYLOAD)
                    with open(p_path, "rb") as pf:  # byte-copy exactly `length` payload bytes
                        pf.seek(base)
                        remaining = length
                        while remaining:
                            chunk = pf.read(min(1 << 20, remaining))
                            if not chunk:
                                raise ValueError(
                                    f"concat: truncated payload in a part ({remaining} left)."
                                )
                            sf.write(chunk)
                            remaining -= len(chunk)
                    # s.offsets is ALREADY int64 (CellStore reads it via np.frombuffer(dtype=int64)),
                    # so no per-part astype copy is needed -- just rebase and stash the array; a
                    # single np.concatenate below avoids materializing ~n_obs boxed Python ints.
                    offset_parts.append(s.offsets[1:] + base_off)
                    base_off += int(s.offsets[-1])
                    row_nnz_parts.append(np.diff(s.indptr).astype(np.int64))
                    # PER-PART length check, not just the aggregate below: the inherited
                    # reference mask is built from each part's n_obs, so two parts whose
                    # declared and actual obs lengths differ in compensating directions (4/4
                    # declared, 3/5 actual) would sum correctly while the mask and the payload
                    # kept 4/4 boundaries against obs's 3/5 -- a canonical-looking archive with
                    # silently misassociated obs rows and reference membership.
                    part_obs = s.obs
                    if len(part_obs) != int(s.n_obs):
                        raise ValueError(
                            f"concat: a part declares n_obs={int(s.n_obs)} but its obs has "
                            f"{len(part_obs)} rows; the part is inconsistent."
                        )
                    obs_parts.append(part_obs)
                    total_nnz += int(s.indptr[-1])

            in_offsets = (
                np.concatenate([np.array([0], dtype=np.int64), *offset_parts])
                if offset_parts
                else np.array([0], dtype=np.int64)
            )
            row_nnz = np.concatenate(row_nnz_parts) if row_nnz_parts else np.empty(0, np.int64)
            obs = (
                pd.concat(obs_parts) if obs_parts else stores[0].obs.iloc[:0]
            )  # preserve index labels

            codec_meta = {
                "schema_version": m0["schema_version"],
                "codec": m0["codec"],
                "pyfastpfor_codec": m0["pyfastpfor_codec"],
                "index_dtype": _widest(s.manifest["index_dtype_on_disk"] for s in stores),
                "value_dtype": _widest(s.manifest["value_dtype_on_disk"] for s in stores),
                "x_data_dtype_min": _widest(s.manifest["x_data_dtype_min"] for s in stores),
                "total_nnz": total_nnz,
                "indices_canonical": all(
                    bool(s.manifest.get("indices_canonical", False)) for s in stores
                ),
            }
            if not caller_supplied_reference and reference is not None and group_by is not None:
                reference = _inherited_reference_mask(stores, reference, obs, group_by)
            _reorder_and_finalize(
                tmpd,
                staging,
                in_offsets,
                row_nnz,
                obs=obs,
                var=stores[0].var,
                uns=stores[0].uns,
                group_by=group_by,
                reference=reference,
                sort_within_group=sort_within_group,
                sort_order=sort_order,
                codec_meta=codec_meta,
                n_vars=stores[0].n_vars,
                out_path=out,
                overwrite=overwrite,
                payload_checksum=payload_checksum,
                obsm=obsm_cat,
                varm=varm_cat,
                obsp=obsp_cat,
                varp=varp_cat,
            )
        finally:
            if tmpd is not None:
                shutil.rmtree(tmpd, ignore_errors=True)
            for s, own in zip(stores, owned, strict=True):
                if own:
                    s.close()
