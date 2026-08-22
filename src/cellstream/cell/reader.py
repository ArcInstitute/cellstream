"""Reader for layout=cell: lazy handle, O(1) scattered gather, subset reads.

open_cell() reads the SHPK footer, loads x_offsets/x_indptr into RAM (int64,
~8*(n_obs+1) bytes), and mmaps the 4 KB-aligned x_payload member. gather_rows is
cellstream's per-cell gather verbatim over that mmap: single-threaded in the calling
process (parallelism across sets comes from DataLoader workers, so the pyfastpfor GIL
is a non-issue). Bulk read (spawn pool over row-ranges) lives in read_cell_bulk.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellstream.errors import IncompatibleSchemaError
from cellstream.packed.reader import PackedArchive
from cellstream.v2.reader_cpu import _open_member_mmap

from . import format as fmt
from .codec import codec_for
from .view import AnnDataView

_I32MAX = int(np.iinfo(np.int32).max)


def _check_col_indices(indices, n_vars):
    """Bound every column index to [0, n_vars). Raises ValueError naming any out-of-range index
    -- the minimum if a signed dtype holds a negative, otherwise the maximum; not the first by
    array position.

    Shared by _csr_direct_assign (the bulk path, under check_indices) and gather_rows (which
    had NO column-bounds check at all: it builds its CSR with the scipy constructor, whose
    check_format(full_check=False) skips the bounds scan entirely). One scan per call, not per
    row -- measured at 0.44-1.5% of a pure-Python 1024-row gather. The Rust decode path
    validates every index inline and does not call this. (#250)
    """
    if indices.size == 0:
        return
    # unsigned index dtypes can't be negative -> skip the min() scan (repo
    # guideline); only bound the max. Signed dtypes still get the < 0 check.
    if indices.dtype.kind != "u":
        cmin = int(indices.min())
        if cmin < 0:
            raise ValueError(f"corrupt CSR: column index out of range [0, {n_vars}) (min={cmin})")
    cmax = int(indices.max())
    if cmax >= n_vars:
        raise ValueError(f"corrupt CSR: column index out of range [0, {n_vars}) (max={cmax})")


def _csr_direct_assign(
    data, indices, indptr, *, n_obs, n_vars, check_indices=False, canonical=False
):
    """Assemble a CSR by empty-construct + direct ``.data/.indices/.indptr``
    assignment, skipping scipy ``check_format``'s O(nnz) scan (column-index
    bounds + sortedness + dtype coercion) on the hot bulk-read path. On a
    multi-billion-nnz archive that scan costs ~10 s; this mirrors the v2 shard
    reader (read.py) which bypasses it the same way.

    Structural integrity is re-checked cheaply here as defense-in-depth against
    a corrupt archive:
      * the O(n_obs) indptr checks (length, indptr[0]==0, indptr[-1]==nnz,
        non-decreasing) always run;
      * the O(nnz) column-index bounds check runs only when ``check_indices``
        is True. The Rust ``decode_cells`` validates every column index against
        [0, n_vars) inline while decoding (rust/src/pfor.rs), so the Rust bulk
        path leaves ``check_indices=False`` and never re-scans the indices; the
        pure-Python fallback worker (``_decode_cell_range_into_shm``) does NOT
        validate, so those sites pass ``check_indices=True`` to preserve the
        safety scipy ``check_format`` used to provide.

    scipy's ``has_sorted_indices`` is always cached (pfordelta emits per-cell indices sorted, so
    sortedness is intrinsic). ``has_canonical_format`` is cached only when ``canonical=True`` —
    passed by the bulk reader when the manifest's ``indices_canonical`` flag says the writer
    verified duplicate-free (``require_unique_indices=True``); otherwise it is left uncached so
    scipy scans lazily on demand.
    """
    if len(indptr) != n_obs + 1:
        raise ValueError(f"corrupt CSR: len(indptr)={len(indptr)} != n_obs+1={n_obs + 1}")
    if int(indptr[0]) != 0:
        raise ValueError(f"corrupt CSR: indptr[0]={int(indptr[0])} must be 0")
    if not (int(indptr[-1]) == len(data) == len(indices)):
        raise ValueError(
            f"corrupt CSR: indptr[-1]={int(indptr[-1])} must equal "
            f"len(data)={len(data)} == len(indices)={len(indices)}"
        )
    # arr[1:] >= arr[:-1] avoids a np.diff temporary (repo perf guideline)
    if not np.all(indptr[1:] >= indptr[:-1]):
        raise ValueError("corrupt CSR: indptr must be non-decreasing")
    if check_indices:
        _check_col_indices(indices, n_vars)
    X = sp.csr_matrix((n_obs, n_vars), dtype=data.dtype)
    X.data = data
    X.indices = indices
    X.indptr = indptr
    # pfordelta emits per-cell indices sorted (write-time sort + delta codec) -> sortedness is
    # intrinsic and always safe to cache. Cache canonical only when the writer verified it, so
    # scipy skips its O(nnz) canonical scan on the first sparse op over a large materialized CSR.
    X.has_sorted_indices = True
    if canonical:
        X.has_canonical_format = True
    return X


def _normalize_row_ids(row_ids, n_obs):
    """Resolve a gather selector into a validated 1-D int64 array of row ids.

    Mirrors ``read.py::_normalize_key``'s ARRAY handling, so a selector means the same thing on
    both layouts (#277). It is not a full equivalent: ``_normalize_key`` also takes slices, and
    it wraps a scalar int before converting, so a 0-D integer array is accepted here and not
    there. For arrays the rules are:

    - a bool ARRAY of length n_obs is a MASK -> np.flatnonzero. Without this it fell through
      to the signed-integer branch below, where min()==False==0 and max()==True==1 pass for
      any n_obs >= 2, and the astype(int64) turned the mask into the index array
      [0,1,0,0,1,...] -- silently returning n_obs rows of the wrong data.
    - a bool SCALAR / a 2-D bool array / an EMPTY bool array on a nonempty store is a MALFORMED
      mask -> ValueError (Python's bool subclasses int, so gather_rows(True) otherwise read as
      row 1). The empty-bool case is a deliberate compatibility change: it used to fall through
      the size==0 exemption below and select nothing. A length-0 mask cannot address a nonempty
      store, so it is now an error -- pass an empty INTEGER array for an empty selection.
    - a non-integer dtype -- float, and also object -- raises TypeError rather than
      truncating 1.9 -> 1. An integer-valued object array used to be accepted here; it is
      rejected now, deliberately, to match _normalize_key.
      An EMPTY array is exempt: np.asarray([]) is float64 and a zero-length selection is
      always valid.
    """
    arr = np.asarray(row_ids)
    # Bool FIRST, on the UN-promoted array, exactly as _normalize_key orders it -- so a 2-D
    # mask and a SCALAR bool both get the mask-specific message, and the np.atleast_1d below
    # cannot turn a 0-d bool into a bogus length-1 "mask".
    if arr.dtype == bool:
        if arr.ndim != 1 or arr.shape[0] != n_obs:
            raise ValueError(
                f"boolean mask must be 1-D with length n_obs={n_obs}, got shape {arr.shape}"
            )
        return np.flatnonzero(arr).astype(np.int64, copy=False)
    row_ids = np.atleast_1d(arr)
    if row_ids.ndim != 1:
        raise ValueError(f"row_ids must be 1-dimensional, got {row_ids.ndim}D")
    if row_ids.size:
        if not np.issubdtype(row_ids.dtype, np.integer):
            raise TypeError(f"row_ids must have an integer dtype, got {row_ids.dtype}")
        if row_ids.dtype.kind == "u":
            if int(row_ids.max()) >= n_obs:
                raise IndexError("row id out of range")
        elif int(row_ids.min()) < 0 or int(row_ids.max()) >= n_obs:
            raise IndexError("row id out of range")
    return row_ids.astype(np.int64, copy=False)


def _normalize_reference(ref):
    """Order- and shape-insensitive form of a ``reference`` value, for comparing the manifest's
    against groups.json's. ``None`` stays ``None``; a bare string becomes a 1-tuple. Both sides
    are type-checked before this runs (`_validate_group_manifest` / `_validate_groups_record`),
    so it never sees a non-string-iterable."""
    if ref is None:
        return None
    return tuple(sorted([ref] if isinstance(ref, str) else list(ref)))


class GroupSpan(NamedTuple):
    """One group's contiguous storage-order row range, ``[start, stop)``.

    Returned by :meth:`CellStore.group_spans`. A plain tuple, so it unpacks
    (``for label, start, stop in store.group_spans()``) and cannot be mutated into the
    store's cached record.
    """

    label: str
    start: int
    stop: int

    @property
    def n_rows(self) -> int:
        """Rows in this span -- what a caller sizes a read (or ``n_threads``) from."""
        return self.stop - self.start


class CellStore:
    """Lazy handle over a layout=cell .shad archive."""

    def __init__(self, path):
        self.path = Path(path)
        self._packed = PackedArchive(self.path)
        self.manifest = self._packed.manifest
        if self.manifest.get("layout") != fmt.LAYOUT_CELL:
            raise IncompatibleSchemaError(
                f"{self.path}: not a layout='cell' archive "
                f"(layout={self.manifest.get('layout')!r}, "
                f"schema_version={self.manifest.get('schema_version')!r}). "
                f"Use cellstream.ShardedArchive / cellstream.read_h5ad for shard-layout archives."
            )
        self.n_obs = int(self.manifest["n_obs_total"])
        self.n_vars = int(self.manifest["n_vars"])
        self._indices_canonical = bool(self.manifest.get("indices_canonical", False))
        # BEFORE reading either sidecar: member_bytes(x_offsets) on an archive missing it raises
        # PackedArchiveError("member not found"), which pre-empts the aggregated
        # IncompatibleSchemaError this check exists to produce -- so a missing sidecar reported
        # the low-level error instead of "this archive is incomplete". Roles are checked too, not
        # just names: docs/format.md sec 3 specifies the name->role map, and a footer that swapped
        # the offsets/indptr roles satisfied a names-only check while violating the spec.
        # (codex, checkpoint 2 round 2.)
        #
        # groups.json is deliberately NOT checked here. Its correspondence with the manifest's
        # group_by is validated when groups are ACCESSED, which keeps open cheap and is pinned by
        # tests/cell/test_group_spans.py::test_groups_json_on_an_ungrouped_manifest_is_rejected
        # (it asserts the archive opens and group_spans() raises). See _load_groups.
        roles = {m["name"]: m.get("role") for m in self._packed.members}
        required = (
            (fmt.PAYLOAD, fmt.ROLE_PAYLOAD),
            (fmt.OFFSETS, fmt.ROLE_OFFSETS),
            (fmt.INDPTR, fmt.ROLE_INDPTR),
            (fmt.HEADER, "header"),
            (fmt.OBS, "obs"),
            (fmt.MANIFEST, "manifest"),
        )
        missing = [n for n, _ in required if n not in roles]
        if missing:
            raise IncompatibleSchemaError(
                f"{self.path}: corrupt cell archive — missing required member(s) {missing!r}."
            )
        wrong = [(n, roles[n], r) for n, r in required if roles[n] != r]
        if wrong:
            raise IncompatibleSchemaError(
                f"{self.path}: corrupt cell archive — member role mismatch "
                + ", ".join(f"{n!r} has role {got!r}, expected {want!r}" for n, got, want in wrong)
                + "."
            )
        expected = self.n_obs + 1
        self.offsets = np.frombuffer(self._packed.member_bytes(fmt.OFFSETS), dtype=np.int64)
        self.indptr = np.frombuffer(self._packed.member_bytes(fmt.INDPTR), dtype=np.int64)
        if self.offsets.size != expected or self.indptr.size != expected:
            raise IncompatibleSchemaError(
                f"{self.path}: corrupt cell archive — offsets/indptr size "
                f"({self.offsets.size}/{self.indptr.size}) != n_obs+1 ({expected})."
            )
        # Every GLOBAL sidecar invariant, at open, before any of them is used to size a read.
        # Previously only the two SIZES and offsets[0] were checked, which left several
        # tampered/corrupt archives able to return data: adding a constant to every indptr entry
        # preserves the per-row differences, so gather_rows happily served rows from an archive
        # whose indptr[0] != 0 and whose indptr[-1] != total_nnz, and corruption confined to rows
        # the caller did not select was never noticed at all. These are O(n_obs) compares over
        # two arrays already resident in RAM, once per open. (codex, checkpoint 2.)
        #
        # Missing members are checked here for the same reason: the matrix members alone are
        # enough to gather, so an archive with no header.h5ad / obs.parquet used to read fine and
        # then fail much later, on an attribute access, with a PackedArchiveError about a member
        # name rather than "this archive is incomplete".
        for arr, who in ((self.offsets, fmt.OFFSETS), (self.indptr, fmt.INDPTR)):
            if arr.size and arr[0] != 0:
                raise IncompatibleSchemaError(
                    f"{self.path}: corrupt cell archive — {who}[0]={int(arr[0])} != 0."
                )
            # Comparison, never np.diff: these are int64 and UNTRUSTED here, so a subtraction
            # could wrap and report a decreasing array as increasing.
            if arr.size > 1 and not np.all(arr[1:] >= arr[:-1]):
                raise IncompatibleSchemaError(
                    f"{self.path}: corrupt cell archive — {who} is not non-decreasing."
                )
        # Indexing [-1] needs no length guard: the size check above already raised unless both
        # arrays are exactly n_obs+1 long, and n_obs+1 >= 1 always. Making these endpoint checks
        # conditional on that length instead would SKIP them when it does not hold -- turning a
        # guaranteed raise into a silent pass. (Gemini round 3, declined for that reason; the
        # int() casts it flagged in the conditions were redundant and are gone. The casts inside
        # the messages stay: numpy 2 reprs a scalar as "np.int64(5)".)
        total_nnz = int(self.manifest["total_nnz"])
        if self.indptr.size and self.indptr[-1] != total_nnz:
            raise IncompatibleSchemaError(
                f"{self.path}: corrupt cell archive — {fmt.INDPTR}[-1]={int(self.indptr[-1])} "
                f"!= manifest total_nnz={total_nnz}."
            )
        p, base, length = self._packed.member_location(fmt.PAYLOAD)
        if self.offsets.size and self.offsets[-1] != length:
            raise IncompatibleSchemaError(
                f"{self.path}: corrupt cell archive — {fmt.OFFSETS}[-1]="
                f"{int(self.offsets[-1])} != {fmt.PAYLOAD} length {length}."
            )
        self._pf, self._mm = _open_member_mmap(p, base, length)
        self._codec = codec_for(
            self.manifest["codec"],
            self.manifest["index_dtype_on_disk"],
            self.manifest["value_dtype_on_disk"],
        )
        # CSR `data` output width for the gather path on BOTH engines: preserve float64
        # archives; everything else -> float32 (integer counts and float32 archives).
        self._x_out_dtype = (
            np.float64 if self.manifest["value_dtype_on_disk"] == "float64" else np.float32
        )
        # pyfastpfor is deliberately NOT imported here, and the pure-Python decoder is
        # deliberately NOT built here. The Rust decoder never calls pyfastpfor, so doing either
        # at OPEN time made the `cell` extra -- no linux wheel, compiles C++ from an sdist -- a
        # hard requirement for reading an archive this machine's wheel decodes without it.
        # `_decoder` below builds on first use and raises codec._import_pyfastpfor's actionable
        # message if the pure-Python path is genuinely the one running.
        #
        # `_codec` itself is safe to build eagerly: codec_for only stores dtype names, and
        # neither PforDeltaCodec nor ByteZstdCodec imports anything until new_encoder /
        # new_decoder is called.
        # `_decoder_n_vars` is n_vars AS OF OPEN, captured so the lazy build is observationally
        # identical to the eager `new_decoder(self.n_vars)` it replaces. A caller can reassign
        # `store.n_vars` between open and the first gather -- nothing in the library does, but
        # tests/cell/test_fallback_equivalence.py does it to fabricate an out-of-range column,
        # and reading the live attribute at build time would move the rejection from the CSR
        # assembly's bounds scan to the pfordelta frame check, changing which error a tampered
        # store raises. Deferring the CONSTRUCTION is the point of this change; silently
        # rebinding what it is constructed FROM is not.
        self._decoder_cache = None
        self._decoder_n_vars = self.n_vars
        self._obs = None
        self._var = None
        self._uns = None
        self._hdr = None
        self._obsm = None
        self._varm = None
        self._obsp = None
        self._varp = None

    @property
    def _decoder(self):
        """The pure-Python frame decoder, built on first use.

        Only ``gather_rows``' fallback branch touches it, and that branch runs after
        ``check_cell_python_fallback`` -- so on the Rust path it is never constructed and
        pyfastpfor is never imported. Cached, because the pfordelta decoder owns growable
        scratch buffers that are reused across rows.
        """
        if self._decoder_cache is None:
            self._decoder_cache = self._codec.new_decoder(self._decoder_n_vars)
        return self._decoder_cache

    def _load_groups(self):
        if getattr(self, "_groups", None) is None:
            self._validate_group_manifest()
            grouped = self.manifest.get("group_by") is not None
            if not self._packed.has_member(fmt.GROUPS):
                # Every cell writer emits groups.json whenever group_by is set, so a missing
                # member on a grouped archive is damage -- not "ungrouped". Left unchecked it
                # would silently report no groups and no reference for an archive that has
                # both (#248).
                if grouped:
                    raise IncompatibleSchemaError(
                        f"{self.path}: corrupt cell archive — manifest declares "
                        f"group_by={self.manifest['group_by']!r} but the archive has "
                        f"no groups.json member."
                    )
                self._groups = {"reference": None, "groups": []}
            else:
                if not grouped:
                    raise IncompatibleSchemaError(
                        f"{self.path}: corrupt cell archive — the archive has a groups.json "
                        f"member but the manifest declares no group_by."
                    )
                raw = self._packed.member_bytes(fmt.GROUPS)
                try:
                    rec = json.loads(raw)
                except (ValueError, UnicodeDecodeError) as e:
                    # The contract is IncompatibleSchemaError, not a raw JSONDecodeError.
                    raise IncompatibleSchemaError(
                        f"{self.path}: corrupt cell archive — groups.json is not valid JSON ({e})."
                    ) from e
                self._validate_groups_record(rec)
                self._groups = rec
        return self._groups

    def _validate_group_manifest(self) -> None:
        """Type the manifest's own grouping fields before anything reads them.

        ``_validate_groups_record`` compares groups.json's ``reference`` against the manifest's,
        and ``_normalize_reference`` would raise a bare ``TypeError`` on a non-iterable value --
        or quietly normalise a dict like ``{"t00": 1}`` to the same tuple as the string
        ``"t00"``. The contract on a corrupt archive is ``IncompatibleSchemaError``, so the
        manifest side is checked first (#248).
        """
        where = f"{self.path}: corrupt cell archive — manifest"
        group_by = self.manifest.get("group_by")
        if not (group_by is None or isinstance(group_by, str)):
            raise IncompatibleSchemaError(
                f"{where} 'group_by' must be null or a string, got {group_by!r}."
            )
        ref = self.manifest.get("reference")
        if not (
            ref is None
            or isinstance(ref, str)
            or (isinstance(ref, list) and ref and all(isinstance(x, str) for x in ref))
        ):
            raise IncompatibleSchemaError(
                f"{where} 'reference' must be null, a string, or a non-empty list of strings, "
                f"got {ref!r}."
            )
        if group_by is None and ref is not None:
            raise IncompatibleSchemaError(
                f"{where} declares no 'group_by' but names reference {ref!r}."
            )

    def _validate_groups_record(self, rec) -> None:
        """Reject a corrupt groups.json before a caller sizes a read (or an allocation) off it.

        Enforces exactly the invariants every cellstream cell writer produces -- verified
        2026-08-02 across write_cell_archive (plain / single-, multi- and last-sorting
        reference / colliding labels / a colliding label AS the reference),
        CellArchiveWriter.finalize and concat_cell_archives: non-empty records that are ordered
        and tile [0, n_obs) -- which for a grouped archive with n_obs == 0 means the EMPTY list,
        the one case where `groups: []` is legal and accepted -- reference-labelled records
        forming a LEADING PREFIX, and a
        `reference` value identical to the manifest's (write_cell_archive takes one from the
        other, writer.py:1144).

        read_reference() and reference_row_count() both rely on that prefix -- the reference
        block is [0, R) -- so without these checks a record list such as
        ``x:[0,10), r:[10,20)`` with reference "r" would pass bounds validation and serve ten
        non-reference rows as reference, and a `reference` deleted to null would silently
        report "no reference" for an archive that has one (#248).

        Check ORDER is load-bearing and ends with the manifest cross-check: a reference naming
        a label no record carries is ALSO a disagreement with the manifest, so cross-checking
        earlier would collapse distinct corruptions into one message.

        DUPLICATE LABELS ARE LEGAL and must not be rejected here: the writer can emit several
        records sharing a stringified label, including inside the reference prefix. read_group
        raises on such a label; group_spans() returns them all.
        """
        where = f"{self.path}: corrupt cell archive — groups.json"
        if not isinstance(rec, dict) or not isinstance(rec.get("groups"), list):
            raise IncompatibleSchemaError(f"{where} is not an object with a 'groups' list.")

        prev_stop = 0
        for i, g in enumerate(rec["groups"]):
            if not isinstance(g, dict) or not isinstance(g.get("label"), str):
                raise IncompatibleSchemaError(f"{where} record {i} has no string 'label'.")
            start, stop = g.get("start"), g.get("stop")
            # bool is a subclass of int -- exclude it, or True/False would pass as 1/0.
            if (
                not isinstance(start, int)
                or not isinstance(stop, int)
                or isinstance(start, bool)
                or isinstance(stop, bool)
            ):
                raise IncompatibleSchemaError(
                    f"{where} record {i} ({g['label']!r}) has non-integer start/stop "
                    f"({start!r}, {stop!r})."
                )
            if not 0 <= start <= stop <= self.n_obs:
                raise IncompatibleSchemaError(
                    f"{where} record {i} ({g['label']!r}) has span [{start}, {stop}) outside "
                    f"[0, n_obs={self.n_obs}]."
                )
            if start == stop:
                # plan_group_shards derives blocks from label-change boundaries, so every
                # record it emits holds at least one row. A zero-row record would give a
                # leading reference block a row count of 0.
                raise IncompatibleSchemaError(
                    f"{where} record {i} ({g['label']!r}) is empty ([{start}, {stop})); no "
                    f"writer emits a zero-row group."
                )
            if start != prev_stop:
                raise IncompatibleSchemaError(
                    f"{where} records must tile [0, n_obs) in storage order: record {i} "
                    f"({g['label']!r}) starts at {start}, expected {prev_stop}."
                )
            prev_stop = stop
        if prev_stop != self.n_obs:
            raise IncompatibleSchemaError(
                f"{where} records must tile [0, n_obs): they cover [0, {prev_stop}) but "
                f"n_obs={self.n_obs}."
            )

        ref = rec.get("reference")
        if not (
            ref is None
            or isinstance(ref, str)
            or (isinstance(ref, list) and ref and all(isinstance(x, str) for x in ref))
        ):
            # A writer canonicalizes "no reference" to null (writer.py:297), never [].
            raise IncompatibleSchemaError(
                f"{where} 'reference' must be null, a string, or a non-empty list of "
                f"strings, got {ref!r}."
            )
        if ref is not None:
            ref_labels = set([ref] if isinstance(ref, str) else ref)
            missing = sorted(ref_labels - {g["label"] for g in rec["groups"]})
            if missing:
                raise IncompatibleSchemaError(
                    f"{where} names reference label(s) {missing!r} that no group record carries."
                )
            hit = [i for i, g in enumerate(rec["groups"]) if g["label"] in ref_labels]
            if hit != list(range(len(hit))):
                raise IncompatibleSchemaError(
                    f"{where} reference records must be the leading block: labels "
                    f"{sorted(ref_labels)!r} occupy record positions {hit!r}, not "
                    f"{list(range(len(hit)))!r}."
                )
        # LAST -- after membership and prefix (see the docstring).
        if _normalize_reference(ref) != _normalize_reference(self.manifest.get("reference")):
            raise IncompatibleSchemaError(
                f"{where} 'reference' ({ref!r}) disagrees with the manifest's "
                f"({self.manifest.get('reference')!r}); every writer stores the same value in "
                f"both."
            )

    @property
    def is_grouped(self) -> bool:
        return self.manifest.get("group_by") is not None

    def group_labels(self) -> list[str]:
        return [g["label"] for g in self._load_groups()["groups"]]

    def group_spans(self) -> list[GroupSpan]:
        """Every group's ``[start, stop)`` storage-order row range, in storage order.

        One entry per record in the archive's ``groups.json`` -- ``[]`` for an ungrouped
        archive (and for a grouped archive with ``n_obs == 0``). The public form of the record
        ``read_group`` addresses, exact by construction: deriving spans from
        ``obs[group_by].value_counts()`` instead diverges, because the writer stringifies
        labels: in an archive written by this version every missing value (``None`` / ``NaN`` /
        ``NaT`` / ``pd.NA``) becomes the single group ``'nan'`` regardless of the column's dtype
        (#294); archives written before that keep whatever their writer produced (``'None'`` /
        ``'<NA>'``), and unused categorical categories are absent entirely.

        Returns a **list**, not a label-keyed mapping: under the default ``sort_order``,
        distinct raw obs values that are equal once stringified CAN produce several records
        carrying the same label -- whenever the raw sort order separates their runs; adjacent
        runs merge into one block instead -- and a mapping would silently drop all but one of
        them (#248). ``read_group`` raises on such a label; use the spans here to address them
        by position via :meth:`gather_rows`.
        """
        return [GroupSpan(g["label"], g["start"], g["stop"]) for g in self._load_groups()["groups"]]

    def read_group(self, label: str, n_threads: int = 1) -> sp.csr_matrix:
        return self.gather_rows(self._group_rows(label), n_threads=n_threads)

    def _group_rows(self, label: str):
        """Validated storage-order row range [start, stop) for a group label (difflib near-matches
        on a miss). Extracted from read_group so read_group_adata reuses the SAME validated lookup
        without rebuilding recs/arange; read_group becomes a one-liner over it.

        Scans the records instead of building a ``{label: record}`` dict: under the default
        sort order, distinct raw obs values that are equal once stringified produce SEVERAL
        records with the same label, and the dict kept only the last of them -- so read_group
        silently returned part of the group (#248). Such a label is now ambiguous and raises
        rather than guessing.
        """
        recs = self._load_groups()["groups"]
        hits = [g for g in recs if g["label"] == label]
        if not hits:
            import difflib

            uniq = list(dict.fromkeys(g["label"] for g in recs))
            near = difflib.get_close_matches(label, uniq, n=5)
            hint = f" Near matches: {near!r}." if near else ""
            raise KeyError(f"Group label {label!r} not found in archive.{hint}")
        if len(hits) > 1:
            spans = ", ".join(f"[{g['start']}, {g['stop']})" for g in hits)
            raise ValueError(
                f"Group label {label!r} is ambiguous: {len(hits)} records carry it ({spans}). "
                f"This archive's group column held distinct values that are equal once "
                f"stringified. Use group_spans() to address them by position."
            )
        g = hits[0]
        return np.arange(g["start"], g["stop"], dtype=np.int64)

    def reference_row_count(self) -> int | None:
        """Rows in the reference pool, or ``None`` when the archive has none.

        ``None`` exactly when :meth:`read_reference` returns ``None`` (ungrouped, grouped with
        no reference, or ``n_obs == 0``); otherwise equal to ``read_reference().shape[0]`` --
        that call is implemented in terms of this one, so the two cannot drift.

        Reference rows lead contiguously from row 0 (``plan_group_shards`` sorts on
        ``~reference_mask`` first), so the count IS the reference prefix's last ``stop``.
        ``_validate_groups_record`` enforces that prefix, that every listed reference label
        occurs, and that the value agrees with the manifest -- so this cannot silently count
        non-reference rows or silently report "no reference".
        """
        rec = self._load_groups()
        ref = rec.get("reference")
        if ref is None:
            return None
        ref_labels = set([ref] if isinstance(ref, str) else ref)
        # max over EVERY record carrying a reference label -- not a {label: record} dict,
        # which would drop all but the last of several records sharing a stringified
        # label (#248). Validation has already established the set is non-empty.
        return max(g["stop"] for g in rec["groups"] if g["label"] in ref_labels)

    def read_reference(self, n_threads: int = 1):
        stop = self.reference_row_count()
        if stop is None:
            return None
        # reference groups lead contiguously from row 0, so the block is [0, stop).
        return self.gather_rows(np.arange(0, stop, dtype=np.int64), n_threads=n_threads)

    # --- scattered gather (the dataloader path) ---
    def gather_rows(self, row_ids, n_threads: int = 1, out_dtype=None) -> sp.csr_matrix:
        """Gather ``row_ids`` into a CSR matrix.

        ``row_ids`` is a 1-D integer array (an INTEGER scalar is promoted to one row) or a
        1-D bool MASK of length ``n_obs`` (#277). Non-integer dtypes raise TypeError; a bool
        scalar is a malformed mask, not a row id, and raises ValueError. See
        :func:`_normalize_row_ids`.

        ``n_threads`` parallelises the Rust ``decode_cells`` path across the selected rows
        (issue #244). It is clamped to between ``1`` and the selected-row count (an empty gather
        uses ``1``), and the default of ``1`` keeps existing callers serial. It has no effect on
        the pure-Python fallback (decoded serially per row).

        ``out_dtype`` overrides the CSR ``data`` output width. ``None`` (the default) uses the
        archive's ``_x_out_dtype`` -- float64 for a float64 archive, float32 for everything
        else. Pass the archive's own ``value_dtype_on_disk`` for a lossless read (#277); the
        default float32 rounds integer counts above 2**24.
        """
        row_ids = _normalize_row_ids(row_ids, self.n_obs)
        # out_dtype=None keeps self._x_out_dtype -- the documented float32 dataloader output.
        # An explicit width is the LOSSLESS re-encode path (#277): _x_out_dtype rounds integer
        # counts above 2**24 (16777217 -> 16777216), and CellArchiveSource used to report that
        # float32 as the SOURCE dtype, so the writer stamped the rounded values uint32.
        out_dt = np.dtype(self._x_out_dtype if out_dtype is None else out_dtype)
        # n_vars above INT32_MAX was never uniformly safe on EITHER engine. The Python fallback
        # cast decoded indices through int32, wrapping any column >= 2**31 to a negative, and
        # the Rust branch still hard-codes int32 output indices. That wrap is the ONE case where
        # this rewrite could not be bit-identical: decoding uint32 -> int64 directly would
        # silently "fix" it into a positive value. Reject the shape instead -- 2**31 columns
        # is not a reachable gene-expression matrix, and this turns latent silent corruption
        # into a clean error on both engines. With it in place, every remaining index path is
        # provably identical to today's. (#244)
        if self.n_vars > _I32MAX:
            raise ValueError(
                f"layout='cell' gather does not support n_vars={self.n_vars} > {_I32MAX} "
                f"(column indices are materialised as int32)."
            )
        from ._rust_dispatch import (
            check_cell_python_fallback,
            decode_cells,
            use_rust_for_cell_decode,
        )

        if use_rust_for_cell_decode(
            self.manifest["codec"],
            self.manifest["index_dtype_on_disk"],
            self.manifest["value_dtype_on_disk"],
            out_dt,
            np.int32,
        ):
            sel_nnz = self.indptr[row_ids + 1] - self.indptr[row_ids]
            total = int(sel_nnz.sum())
            # Must match the pure-Python branch below (out_dt). Hard-coding float32
            # here was safe ONLY while the Rust gate was pfordelta+integer-only; PR2 opens it
            # to zstd/float64, which would otherwise SILENTLY truncate -- allow_lossy=True
            # below means Rust would not even raise.
            data = np.empty(total, dtype=out_dt)
            indices = np.empty(total, dtype=np.int32)
            indptr = np.empty(row_ids.size + 1, dtype=np.int64)
            # Clamp to the selected-row count so an over-large request (or an empty gather)
            # never asks the decoder for more workers than there is work. (#244)
            nw = max(1, min(int(n_threads), int(row_ids.size)))
            payload = None
            try:
                payload = np.frombuffer(self._mm, dtype=np.uint8)  # zero-copy over the mmap
                decode_cells(
                    payload,
                    self.offsets,
                    self.indptr,
                    row_ids,
                    data,
                    indices,
                    indptr,
                    codec=self.manifest["codec"],
                    value_dtype_on_disk=self.manifest["value_dtype_on_disk"],
                    n_vars=self.n_vars,
                    allow_lossy=True,
                    n_threads=nw,
                )
            finally:
                # Drop OUR reference to the mmap-backed view before a decode_cells exception can
                # propagate. decode_cells (_rust_dispatch.py) already `del`s its OWN local
                # payload_u8 on a ValueError, but that only drops ITS reference -- this frame's
                # `payload` is a SECOND reference to the SAME array, and the in-flight traceback
                # pins THIS frame too, keeping the mmap exported. Without this, CellStore.close()
                # -> self._mm.close() raises BufferError("cannot close exported pointers exist"),
                # MASKING the real error. Mirrors read_cell_bulk's `finally: payload = None`
                # below, which already has this discipline. (PR4 final review, Critical.)
                payload = None
            return sp.csr_matrix((data, indices, indptr), shape=(row_ids.size, self.n_vars))

        # #244: a silent pure-Python gather is the bug this issue was filed about. Raise unless
        # the caller opted in; warn once otherwise.
        check_cell_python_fallback("CellStore.gather_rows")
        # Preallocate from indptr and decode each row into a slice. The old path built two
        # lists, concatenated both, and cast the whole block -- three materializations of the
        # same data, and the top four frames of the #244 profile (~414 s of 636 s).
        sel_nnz = self.indptr[row_ids + 1] - self.indptr[row_ids]
        total = int(sel_nnz.sum())
        # Reproduce the index dtype the scipy triple-form ctor would have chosen. It does NOT
        # keep the dtypes it is handed: it calls _get_index_dtype((indices, indptr),
        # maxval=max(shape), check_contents=True) and recasts BOTH. Preallocating int64 and
        # direct-assigning would silently change X.indptr.dtype on every ordinary gather --
        # and diverge from the Rust branch, which still goes through that ctor. indices <
        # n_vars and indptr <= total, so these scalars bound exactly what check_contents
        # would have scanned for; no O(nnz) pass is needed.
        # The `... else None` is NOT a shortcut -- it mirrors scipy verbatim. _compressed.py:
        #     maxval = None
        #     if shape is not None and 0 not in shape:
        #         maxval = max(shape)
        # i.e. scipy itself passes maxval=None whenever a dimension is 0, and the distinction is
        # load-bearing: _get_index_dtype(..., maxval=None) -> int32 where maxval=2**31+5 ->
        # int64. Computing max(shape) unconditionally here would DIVERGE from the ctor on an
        # empty gather. Do not "simplify" this.
        maxval = max(row_ids.size, self.n_vars) if (row_ids.size and self.n_vars) else None
        fits32 = (maxval is None or maxval <= _I32MAX) and total <= _I32MAX
        idx_dtype = np.int32 if fits32 else np.int64
        data = np.empty(total, dtype=out_dt)
        indices = np.empty(total, dtype=idx_dtype)
        indptr = np.empty(row_ids.size + 1, dtype=idx_dtype)
        indptr[0] = 0
        # Accumulate straight into indptr[1:] at idx_dtype -- no temporary at all.
        #
        # Safety rests on fits32, NOT on the accumulator width: idx_dtype is int32 only when
        # total <= _I32MAX, and the cumsum is non-decreasing with final value exactly total, so
        # every partial sum is bounded by a value that fits. total comes from sel_nnz.sum() on
        # this same int64 array, so the bound cannot disagree with the data.
        #
        # An earlier revision accumulated in int64 and let the assignment narrow, on the theory
        # that a wide accumulator prevented a mid-accumulation wrap. It does not: the narrowing
        # assignment wraps identically. Verified -- for sel_nnz = [2**30]*3 into an int32 indptr,
        # both forms give [0, 1073741824, -2147483648, -1073741824], bit-for-bit. So the wide
        # accumulator bought nothing and cost an O(n_rows) allocation. (Gemini #247 r3; supersedes
        # the defensive form added for Copilot #247 r2.)
        np.cumsum(sel_nnz, dtype=idx_dtype, out=indptr[1:])
        pos = 0
        for i in row_ids:
            n = int(self.indptr[i + 1] - self.indptr[i])
            # Called even when n == 0: decode() validates the frame prefix and index-blob
            # bound BEFORE looking at nnz, so skipping empty rows would silently accept a
            # malformed zero-nnz frame that raises today.
            self._decoder.decode_into(
                self._mm[self.offsets[i] : self.offsets[i + 1]],
                n,
                indices[pos : pos + n],
                data[pos : pos + n],
            )
            pos += n
        # _csr_direct_assign, not the scipy ctor: skips check_format's O(nnz) scan and
        # _prune_array (9.5% of the #244 profile).
        #
        # check_indices=True is a DELIBERATE behaviour change, not preservation: the ctor only
        # runs check_format(full_check=False), which does not bounds-check columns at all (a
        # column index of 99 in a 5-column matrix passes silently today). Rust's decode_cells
        # validates every column inline while the Python decoder does not, so without this the
        # fallback would be strictly less safe than the fast path. Corrupt archives now raise
        # instead of yielding a quietly wrong matrix; valid archives are unaffected.
        #
        # canonical= is deliberately NOT passed. The scipy ctor this replaced never cached
        # has_canonical_format, and caching it here would be a FOURTH corrupt-input deviation
        # beyond the three this rewrite is allowed: a tampered manifest claiming
        # indices_canonical=true while rows hold duplicate columns would make sum_duplicates()
        # a silent no-op (verified: nnz stays 3 instead of collapsing to 2). The bulk reader
        # still passes it -- that is pre-existing and out of scope here. Leaving it unset only
        # costs scipy a lazy canonical scan on the first op that needs one.
        # (codex checkpoint-2 review, CRITICAL.)
        return _csr_direct_assign(
            data,
            indices,
            indptr,
            n_obs=row_ids.size,
            n_vars=self.n_vars,
            check_indices=True,
        )

    def _subset_adata(self, X, rows):
        """AnnData for a cell subset: X + sliced obs/obsm, induced obsp subgraph, whole
        var/varm/varp/uns. Loads the full header annotations then slices -> O(n_obs*width),
        NOT O(subset) (there is no partial read of a header.h5ad blob). Opt-in (#229)."""
        import anndata as _ad

        from cellstream.header import permute_cell_axis_fields

        hdr = self._header()
        # reuse the write-path helper: per-type row slice (DataFrame obsm via .iloc) + induced obsp
        # subgraph. `rows` here is a subset selection, not a full permutation -- both work.
        obsm_sub, obsp_sub = permute_cell_axis_fields(hdr.obsm, hdr.obsp, rows)
        a = _ad.AnnData(
            X=X,
            obs=self.obs.iloc[rows],
            var=hdr.var,
            uns=dict(hdr.uns),
            obsm=obsm_sub,
            varm=dict(hdr.varm),
        )
        for k, v in obsp_sub.items():
            a.obsp[k] = v
        for k, v in hdr.varp.items():
            a.varp[k] = v
        return a

    def read_group_adata(self, label: str, n_threads: int = 1):
        """Like :meth:`read_group` but returns an AnnData carrying the group's obs/obsm and the
        induced obsp subgraph (opt-in; see :meth:`_subset_adata` for the memory note)."""
        rows = self._group_rows(label)
        return self._subset_adata(self.gather_rows(rows, n_threads=n_threads), rows)

    def gather_rows_adata(self, row_ids, n_threads: int = 1):
        """Like :meth:`gather_rows` but returns an AnnData carrying sliced obs/obsm and the
        induced obsp subgraph (opt-in; see :meth:`_subset_adata` for the memory note)."""
        # Normalize FIRST so _subset_adata's obsm/obsp fancy-indexing gets the same 1-D int64
        # ids the matrix was built from -- a scalar row id must become a 1-D array (Gemini
        # #232), and a bool mask must become row ids, not be handed on as a mask (#277).
        row_ids = _normalize_row_ids(row_ids, self.n_obs)
        return self._subset_adata(self.gather_rows(row_ids, n_threads=n_threads), row_ids)

    def to_anndata(self, backed: bool = True):
        if not backed:
            raise NotImplementedError(
                "CellStore.to_anndata(backed=False) is deferred; use cellstream.read_h5ad(path) "
                "for a materialized AnnData, or backed=True for the lazy view."
            )
        return AnnDataView(self)

    @property
    def has_payload_checksum(self) -> bool:
        """True if a payload SHA-256 was stored at write time (``payload_checksum=True``).

        When False, :meth:`verify_payload` returns True vacuously (nothing to verify)."""
        return self.manifest.get("payload_sha256") is not None

    def verify_payload(self) -> bool:
        """Recompute the payload SHA-256 and compare to the manifest (O(payload) I/O).

        Returns True when no checksum is stored (default writes set ``payload_checksum=False``) —
        i.e. "nothing to verify", NOT "verified intact". Use :attr:`has_payload_checksum` to tell
        the two apart."""
        want = self.manifest.get("payload_sha256")
        if not want:
            return True
        h = hashlib.sha256()
        h.update(self._mm)  # buffer-protocol: hash the mmap in place, no bytes() copy
        return h.hexdigest() == want

    @property
    def obs(self) -> pd.DataFrame:
        if self._obs is None:
            self._obs = pd.read_parquet(io.BytesIO(self._packed.member_bytes(fmt.OBS)))
        return self._obs

    def _header(self):
        # header_adata() is uncached (re-opens + mmaps + re-reads each call); cache the one
        # AnnData so var/uns/obsm/varm/obsp/varp don't each re-materialize the member (#229).
        if self._hdr is None:
            self._hdr = self._packed.header_adata()
        return self._hdr

    @property
    def var(self) -> pd.DataFrame:
        if self._var is None:
            # header.h5ad member -> var DataFrame (via the packed archive's header AnnData)
            self._var = self._header().var
        return self._var

    @property
    def uns(self) -> dict:
        if self._uns is None:
            # header.h5ad member -> uns dict (via the packed archive's header AnnData).
            # Mirrors `var` above exactly -- same source member, same lazy cache. Before this,
            # CellArchiveSource.uns unconditionally returned {} (PR4 review finding 1):
            # write_cell_archive(some.shad, out.shad, ...) -- the codec-migration / re-encode
            # workflow PR4 exists for -- silently dropped uns on every re-encode.
            self._uns = dict(self._header().uns)
        return self._uns

    @property
    def obsm(self) -> dict:
        if self._obsm is None:
            self._obsm = dict(self._header().obsm)
        return self._obsm

    @property
    def varm(self) -> dict:
        if self._varm is None:
            self._varm = dict(self._header().varm)
        return self._varm

    @property
    def obsp(self) -> dict:
        if self._obsp is None:
            self._obsp = dict(self._header().obsp)
        return self._obsp

    @property
    def varp(self) -> dict:
        if self._varp is None:
            self._varp = dict(self._header().varp)
        return self._varp

    @property
    def shape(self):
        return (self.n_obs, self.n_vars)

    def close(self) -> None:
        # NOT a try/finally: if _mm.close() raises (BufferError -- a live exported view, e.g. a
        # gather_rows caller whose in-flight exception is still pinning a payload reference), the
        # mapping is still valid and _pf must stay open too. Unconditionally closing _pf here used
        # to turn that recoverable BufferError into a live mmap + numpy view over a closed fd --
        # the next heap operation (a stray gc.collect()) could then abort the process instead of
        # just failing this call. Only close the backing file once the mmap itself actually
        # closed. (PR4 final review, Critical.)
        self._mm.close()
        self._pf.close()


def _decode_cell_range_into_shm(task):
    """Spawn-worker: decode rows [row_start, row_stop) of a cell payload into shm.

    task = (payload_path, base, length, row_start, row_stop, offsets_rel(bytes),
            indptr_rel(bytes), data_name, indices_name, indptr_name, nnz_offset,
            rows_offset, data_dtype_name, index_dtype_name, indptr_dtype_name,
            codec_name, on_disk_index_dtype, on_disk_value_dtype, allow_lossy, n_vars)
    Writes data[nnz_offset:...], indices[nnz_offset:...], and indptr[rows_offset+1:...]
    (crow entries; the global indptr[0]=0 is written once by the parent).
    """
    from multiprocessing import shared_memory as _shm

    import numpy as _np

    from cellstream.v2.materialize import cast_data_into
    from cellstream.v2.reader_cpu import _open_member_mmap

    from .codec import codec_for

    (
        payload_path,
        base,
        length,
        row_start,
        row_stop,
        offsets_b,
        indptr_b,
        data_name,
        indices_name,
        indptr_name,
        nnz_offset,
        rows_offset,
        data_dtype_name,
        index_dtype_name,
        indptr_dtype_name,
        codec_name,
        on_disk_index,
        on_disk_value,
        allow_lossy,
        n_vars,
    ) = task

    offsets = _np.frombuffer(offsets_b, dtype=_np.int64)  # len (n_rows+1), payload-relative
    indptr_rel = _np.frombuffer(indptr_b, dtype=_np.int64)  # len (n_rows+1), CSR cumulative

    fh, mm = None, None
    d_shm, i_shm, p_shm = None, None, None
    try:
        fh, mm = _open_member_mmap(payload_path, base, length)
        d_shm = _shm.SharedMemory(name=data_name)
        i_shm = _shm.SharedMemory(name=indices_name)
        p_shm = _shm.SharedMemory(name=indptr_name)
        data_dt = _np.dtype(data_dtype_name)
        idx_dt = _np.dtype(index_dtype_name)
        ip_dt = _np.dtype(indptr_dtype_name)
        data_buf = _np.ndarray((d_shm.size // data_dt.itemsize,), dtype=data_dt, buffer=d_shm.buf)
        idx_buf = _np.ndarray((i_shm.size // idx_dt.itemsize,), dtype=idx_dt, buffer=i_shm.buf)
        ip_buf = _np.ndarray((p_shm.size // ip_dt.itemsize,), dtype=ip_dt, buffer=p_shm.buf)
        dec = codec_for(codec_name, on_disk_index, on_disk_value).new_decoder(n_vars)
        write_pos = int(nnz_offset)
        n_rows = row_stop - row_start
        # Per-row scratch at the DECODER's own output dtypes; cast_data_into then applies the
        # lossy policy straight into the shm slice. The old path allocated one temporary per
        # row per array via cast_data and then copied it into the slice. (#244)
        # decode_into is only a WIN where the decoder can write into a caller buffer without
        # allocating -- that is pfordelta (it unpacks into its own scratch). The zstd
        # primitives allocate their own output, so routing through decode_into there would add
        # one copy per row versus today; feed decode()'s arrays straight to cast_data_into.
        zero_alloc = getattr(dec, "decode_into_avoids_alloc", False)
        scratch_i = _np.empty(0, dtype=dec._idt)
        scratch_v = _np.empty(0, dtype=dec._vdt)
        for j in range(n_rows):
            nnz = int(indptr_rel[j + 1] - indptr_rel[j])
            frame = mm[offsets[j] : offsets[j + 1]]
            # Decoded unconditionally, including nnz == 0: both decoders validate the frame
            # prefix and index-blob bound before looking at nnz, and today's loop calls
            # decode() for every row. Guarding on nnz would silently accept a malformed
            # zero-nnz frame that raises today.
            if zero_alloc:
                if scratch_i.size < nnz:
                    scratch_i = _np.empty(nnz, dtype=dec._idt)
                    scratch_v = _np.empty(nnz, dtype=dec._vdt)
                dec.decode_into(frame, nnz, scratch_i[:nnz], scratch_v[:nnz])
                src_i, src_v = scratch_i[:nnz], scratch_v[:nnz]
            else:
                src_i, src_v = dec.decode(frame, nnz)
            if nnz:
                cast_data_into(
                    src_i,
                    idx_buf[write_pos : write_pos + nnz],
                    allow_lossy=allow_lossy,
                    where="cell.indices",
                )
                cast_data_into(
                    src_v,
                    data_buf[write_pos : write_pos + nnz],
                    allow_lossy=allow_lossy,
                    where="cell.data",
                )
            write_pos += nnz
            # crow entry for global row (row_start + j + 1). indptr_rel is a slice of the
            # store's ABSOLUTE cumulative-nnz array, so indptr_rel[j+1] is already the
            # global crow value — assign it directly (do NOT re-add indptr[row_start]).
            ip_buf[rows_offset + j + 1] = int(indptr_rel[j + 1])
    finally:
        if d_shm is not None:
            d_shm.close()
        if i_shm is not None:
            i_shm.close()
        if p_shm is not None:
            p_shm.close()
        if mm is not None:
            mm.close()
        if fh is not None:
            fh.close()


def read_cell_bulk(
    path,
    *,
    cast_back: bool = True,
    container: str = "csr",
    data_dtype=None,
    index_dtype=None,
    allow_lossy: bool = False,
    n_workers: int = 16,
    output_backing: str = "heap",
):
    """Materialize a whole layout=cell archive into an AnnData (spawn pool over row-ranges)."""
    import anndata as _ad

    from cellstream.shm import allocate_shm_buffers, attach_buffer_cleanup
    from cellstream.shm import array_for_shared_memory as _shm_array_for_shared_memory
    from cellstream.v2.materialize import resolve_plan
    from cellstream.v2.reader_cpu import _dispatch_workers

    def _assemble(X, store, hdr):
        a = _ad.AnnData(
            X=X,
            obs=store.obs,
            var=hdr.var,
            uns=dict(hdr.uns),
            obsm=dict(hdr.obsm),
            varm=dict(hdr.varm),
        )
        for k, v in hdr.obsp.items():
            a.obsp[k] = v
        for k, v in hdr.varp.items():
            a.varp[k] = v
        return a

    if container != "csr":
        raise NotImplementedError(
            "layout='cell' bulk read supports container='csr' in phase 1 (dense is deferred)."
        )
    # Validate BEFORE opening the store so a bad value cannot leak its mmap/file handle.
    # Same contract as read_v2_to_host (v2/reader_cpu.py) -- the cell path used to accept
    # anything, including output_backing="banana" (#277).
    if output_backing not in ("heap", "shm"):
        if output_backing == "pinned":
            raise NotImplementedError(
                "output_backing='pinned' is reserved for the GPU H2D path (#40) "
                "and is not implemented yet."
            )
        raise ValueError(f"output_backing must be 'heap' or 'shm'; got {output_backing!r}")
    store = CellStore(path)
    from ._rust_dispatch import (
        check_cell_python_fallback,
        decode_cells,
        use_rust_for_cell_decode,
    )

    # store is already open; the manifest reads or resolve_plan (a bad user dtype) can raise HERE,
    # before the branch-local try/finally blocks that close store take over. Close it on that error
    # path so a failed read does not leak the mmap/file handle. (Copilot #226 r4.)
    try:
        man = store.manifest
        n_obs, n_vars, total_nnz = store.n_obs, store.n_vars, int(man["total_nnz"])
        synth = {
            "x_data_dtype_on_disk": man["value_dtype_on_disk"],
            "x_indices_dtype_on_disk": man["index_dtype_on_disk"],
            "x_data_dtype_min": man["x_data_dtype_min"],
            "total_nnz": total_nnz,
            "n_vars": n_vars,
        }
        # NOTE: resolve_plan (materialize.py:76-90) derives the destination index dtype from
        # n_vars/cast_back and does NOT read x_indices_dtype_on_disk — so this key is a faithful
        # mirror of the on-disk manifest (future-proofing), not a correctness requirement. The
        # actual on-disk index dtype is passed separately to the worker (man["index_dtype_on_disk"]
        # -> on_disk_index) to build the decoder.
        plan = resolve_plan(
            synth,
            container="csr",
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
            cast_back=cast_back,
            target="cpu",
        )
    except BaseException:
        store.close()
        raise

    # Gate on the RESOLVED output dtypes (#218): use_rust_for_cell_decode falls back to the
    # pure-Python worker for any (data, index) pair the Rust dispatch cannot produce, instead of
    # promising Rust and then raising with no fallback.
    if use_rust_for_cell_decode(
        man["codec"],
        man["index_dtype_on_disk"],
        man["value_dtype_on_disk"],
        plan.data_dtype,
        plan.index_dtype,
    ):
        payload = None
        bundle = None
        try:
            if output_backing == "shm":
                # #277: this branch used to allocate np.empty unconditionally and RETURN before
                # the heap/shm split further down, so output_backing was silently ignored on a
                # Rust build. Mirror the pure-Python branch: shm segments, each wrapped by
                # array_for_shared_memory so the array it backs pins its OWN segment (#274).
                # max(1, ...) because a zero-byte SharedMemory is invalid.
                bundle = allocate_shm_buffers(
                    data_nbytes=max(1, total_nnz) * plan.data_dtype.itemsize,
                    indices_nbytes=max(1, total_nnz) * plan.index_dtype.itemsize,
                    indptr_nbytes=(n_obs + 1) * plan.indptr_dtype.itemsize,
                )
                data = _shm_array_for_shared_memory(
                    bundle.data, (total_nnz,), np.dtype(plan.data_dtype)
                )
                indices = _shm_array_for_shared_memory(
                    bundle.indices, (total_nnz,), np.dtype(plan.index_dtype)
                )
                # decode_cells writes indptr as int64 UNCONDITIONALLY, but resolve_plan picks
                # int32 for any archive under 2**31 nnz -- i.e. the DEFAULT. Decoding into the
                # shm view directly (or casting afterwards with astype) would hand back a heap
                # COPY as X.indptr while an unused int64 segment stayed attached. So decode
                # into a tiny int64 heap scratch (O(n_obs), negligible next to the nnz-sized
                # streams) and cast it INTO the plan-dtype shm view -- keeping all three
                # returned arrays genuine shm views. This is exactly what the v2 Rust read path
                # already does (v2/reader_cpu.py, the output_backing == "shm" arm).
                ip_view = _shm_array_for_shared_memory(
                    bundle.indptr, (n_obs + 1,), np.dtype(plan.indptr_dtype)
                )
                indptr = np.empty(n_obs + 1, dtype=np.int64)
            else:
                ip_view = None
                data = np.empty(total_nnz, dtype=plan.data_dtype)
                indices = np.empty(total_nnz, dtype=plan.index_dtype)
                indptr = np.empty(n_obs + 1, dtype=np.int64)
            row_ids = np.arange(n_obs, dtype=np.int64)
            payload = np.frombuffer(store._mm, dtype=np.uint8)
            nw = max(1, min(n_workers, n_obs)) if n_obs else 1
            decode_cells(
                payload,
                store.offsets,
                store.indptr,
                row_ids,
                data,
                indices,
                indptr,
                codec=man["codec"],
                value_dtype_on_disk=man["value_dtype_on_disk"],
                n_vars=n_vars,
                allow_lossy=allow_lossy,
                n_threads=nw,
            )
            if ip_view is not None:
                ip_view[:] = indptr  # int64 scratch -> plan-dtype shm view
                out_indptr = ip_view
            else:
                # copy=False: `indptr` is this frame's private int64 scratch, so when the plan
                # dtype is also int64 (cast_back=False, or an archive at/over 2**31 nnz) there
                # is nothing to copy for.
                out_indptr = indptr.astype(plan.indptr_dtype, copy=False)
            X = _csr_direct_assign(
                data,
                indices,
                out_indptr,
                n_obs=n_obs,
                n_vars=n_vars,
                canonical=store._indices_canonical,
            )
            hdr = store._header()
            adata = _assemble(X, store, hdr)
            if bundle is not None:
                attach_buffer_cleanup(adata, bundle)
            return adata
        except BaseException:
            if bundle is not None:
                bundle.close_and_unlink()
            raise
        finally:
            payload = None
            store.close()

    # #244: same policy as gather_rows. Checked in the PARENT, not the spawned worker -- the
    # worker runs once per row-range and would warn N times for one logical read. read_cell_bulk
    # holds an OPEN store here, so a raise must not leak it.
    try:
        check_cell_python_fallback("read_cell_bulk")
    except BaseException:
        store.close()
        raise

    data_dt, idx_dt, ip_dt = plan.data_dtype, plan.index_dtype, plan.indptr_dtype

    # bundle=None + allocation INSIDE the try: allocating above it meant an OSError from a full
    # /dev/shm skipped the `finally: store.close()` below and leaked the archive's mmap and fd
    # until the traceback was released -- every other early-exit path in this function closes
    # it. (#287, found auditing the callers.)
    bundle = None
    try:
        bundle = allocate_shm_buffers(
            data_nbytes=max(1, total_nnz) * data_dt.itemsize,
            indices_nbytes=max(1, total_nnz) * idx_dt.itemsize,
            indptr_nbytes=(n_obs + 1) * ip_dt.itemsize,
        )
        offsets, indptr = store.offsets, store.indptr
        # partition rows into ~n_workers contiguous ranges
        nw = max(1, min(n_workers, n_obs)) if n_obs else 1
        bounds = [(k * n_obs) // nw for k in range(nw + 1)]
        tasks = []
        p_path, base, length = store._packed.member_location(fmt.PAYLOAD)
        for k in range(nw):
            a, b = bounds[k], bounds[k + 1]
            if a == b:
                continue
            tasks.append(
                (
                    p_path,
                    base,
                    length,
                    a,
                    b,
                    offsets[a : b + 1].tobytes(),  # payload-relative frame offsets
                    indptr[a : b + 1].tobytes(),  # CSR cumulative nnz
                    bundle.data.name,
                    bundle.indices.name,
                    bundle.indptr.name,
                    int(indptr[a]),  # nnz write offset
                    a,  # rows_offset (crow index base)
                    data_dt.name,
                    idx_dt.name,
                    ip_dt.name,
                    man["codec"],
                    man["index_dtype_on_disk"],
                    man["value_dtype_on_disk"],
                    allow_lossy,
                    n_vars,
                )
            )
        # write global indptr[0] = 0 in the parent before dispatch
        ip_view0 = np.ndarray((n_obs + 1,), dtype=ip_dt, buffer=bundle.indptr.buf)
        ip_view0[0] = 0
        _dispatch_workers(_decode_cell_range_into_shm, tasks, nw, omp_nthreads=None)

        # Pinned views (#274): on the output_backing="shm" path these become
        # adata.X.data/.indices/.indptr, so each array keeping its OWN segment mapped is
        # what makes an extracted X.data valid after the AnnData is dropped. (ip_view0
        # above stays a bare view -- it is parent-side scratch that never escapes.)
        data_view = _shm_array_for_shared_memory(bundle.data, (total_nnz,), np.dtype(data_dt))
        idx_view = _shm_array_for_shared_memory(bundle.indices, (total_nnz,), np.dtype(idx_dt))
        ip_view = _shm_array_for_shared_memory(bundle.indptr, (n_obs + 1,), np.dtype(ip_dt))
        hdr = store._header()

        if output_backing == "heap":
            data = data_view.copy()
            indices = idx_view.copy()
            indptr = ip_view.copy()
            # pure-Python fallback worker does not validate col bounds -> re-scan
            X = _csr_direct_assign(
                data,
                indices,
                indptr,
                n_obs=n_obs,
                n_vars=n_vars,
                check_indices=True,
                canonical=store._indices_canonical,
            )
            adata = _assemble(X, store, hdr)
            bundle.close_and_unlink()
            return adata

        # direct-assign (not the scipy ctor) so the CSR references the shm views
        # with no copy/recast — the intent of output_backing="shm". check_indices:
        # pure-Python fallback worker does not validate col bounds -> re-scan.
        X = _csr_direct_assign(
            data_view,
            idx_view,
            ip_view,
            n_obs=n_obs,
            n_vars=n_vars,
            check_indices=True,
            canonical=store._indices_canonical,
        )
        adata = _assemble(X, store, hdr)
        attach_buffer_cleanup(adata, bundle)
        return adata
    except BaseException:
        if bundle is not None:
            bundle.close_and_unlink()
        raise
    finally:
        store.close()


def open_cell(path) -> CellStore:
    return CellStore(path)
