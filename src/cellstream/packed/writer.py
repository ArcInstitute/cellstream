"""Write-side: pack a directory archive into one file, and the reverse."""

from __future__ import annotations

import contextlib
import os
import shutil
import struct
import tempfile
import uuid
from pathlib import Path

from cellstream import header as _header
from cellstream import lock
from cellstream.errors import ArchiveExistsError
from cellstream.manifest import load_manifest, shard_filename
from cellstream.packed import format as fmt
from cellstream.packed.reader import PackedArchive, _apply_recovery

# Metadata members, in tail order. groups.json is optional.
_METADATA_ORDER = ("header.h5ad", "obs.parquet", "manifest.json", "groups.json")
# Roles that live in the immutable (pre-shard_region_end) region and are deleted
# when packing writer-native temp dirs: v2 shards, plus the cell layout's payload/
# offsets/indptr data members.
_IMMUTABLE_ROLES = frozenset({"shard", "payload", "offsets", "indptr"})


def _enumerate_members(dir_path: Path):
    """Return an ordered list of (name, role, shard_idx_or_None, matrix_or_None,
    source_path).

    Shards first (immutable; per-matrix families for v4), then metadata
    (header, obs, manifest, groups).
    """
    man = load_manifest(dir_path / "manifest.json")
    out = []
    if man.get("layout") == "cell":
        # Flat per-cell store: three immutable data members, then the metadata tail.
        for name, role in (
            ("x_payload", "payload"),
            ("x_offsets", "offsets"),
            ("x_indptr", "indptr"),
        ):
            out.append((name, role, None, None, dir_path / name))
        roles = {
            "header.h5ad": "header",
            "obs.parquet": "obs",
            "manifest.json": "manifest",
            "groups.json": "groups",
        }
        for name in _METADATA_ORDER:
            p = dir_path / name
            if p.exists():
                out.append((name, roles[name], None, None, p))
        return out
    if man.get("schema_version") == 4:
        for key, md in man["matrices"].items():
            for i in range(man["n_shards"]):
                rel = md["shard_filename_template"].format(i)
                out.append((rel, "shard", i, key, dir_path / rel))
        # per-matrix var sidecars (own-var families only). Dedup by filename:
        # today x/raw own distinct var/<dir>.parquet sidecars, but if a future
        # extension ever pointed two matrices at the same var_filename, appending
        # it twice would put a duplicate member name in the footer (corruption).
        seen_vfs: set[str] = set()
        for md in man["matrices"].values():
            vf = md.get("var_filename")
            if vf and vf not in seen_vfs and (dir_path / vf).exists():
                seen_vfs.add(vf)
                out.append((vf, "var", None, None, dir_path / vf))
        # per-family metadata sidecars (obsm/varm/uns). Same dedup discipline
        # as var sidecars.
        seen_mfs: set[str] = set()
        for md in man["matrices"].values():
            mf = md.get("meta_filename")
            if mf and mf not in seen_mfs and (dir_path / mf).exists():
                seen_mfs.add(mf)
                out.append((mf, "meta", None, None, dir_path / mf))
    else:
        for i in range(man["n_shards"]):
            fn = shard_filename(man, i)
            out.append((fn, "shard", i, None, dir_path / fn))
    roles = {
        "header.h5ad": "header",
        "obs.parquet": "obs",
        "manifest.json": "manifest",
        "groups.json": "groups",
    }
    for name in _METADATA_ORDER:
        p = dir_path / name
        if p.exists():
            out.append((name, roles[name], None, None, p))
    return out


def _write_member(out, src_path: Path, *, delete_after: bool) -> tuple[int, int]:
    """Pad ``out`` to alignment, copy ``src_path`` into it, return (offset, length)."""
    pos = out.tell()
    pad = fmt.align_up(pos) - pos
    if pad:
        out.write(b"\x00" * pad)
    offset = out.tell()
    with src_path.open("rb") as fh:
        shutil.copyfileobj(fh, out, length=1 << 20)
    # The bytes ACTUALLY copied, never src_path.stat().st_size (#305). copyfileobj's `length` is a
    # buffer size, not a byte limit, so it always copies the whole file -- while a path stat on an
    # attribute-caching network filesystem (WekaFS on every Arc store, and NFS) can read a
    # stale-SHORT st_size right after the member was written. Recording that value writes a footer
    # entry SHORTER than the member really is: a packed archive that reports a successful write and
    # is wrong on disk. Deriving it from the stream cannot disagree with what was written.
    #
    # No syscall claim: `BufferedWriter.tell()` asks the raw stream for its position, so this trades
    # a stat() for an lseek() rather than saving one. (Copilot.)
    length = out.tell() - offset
    if delete_after:
        src_path.unlink()
    return offset, length


def _pack_directory(
    src_dir, dst_file, *, overwrite: bool = False, _delete_sources: bool = False
) -> None:
    """Assemble a v2 directory layout into a single packed file (pure byte concat).

    INTERNAL. The public ``pack``/``unpack`` entry points and the ``cellstream pack``
    CLI were removed with the directory container in #154; this is the assembly
    step ``write_packed`` / ``write_packed_multi`` run over their own temp dir.

    ``_delete_sources``: delete each shard source file immediately after it is
    appended (used by the writer-native path on a temp dir to bound disk use).
    """
    src_dir = Path(src_dir)
    dst_file = Path(dst_file)
    if dst_file.exists() and not overwrite:
        raise ArchiveExistsError(f"{dst_file} already exists. Pass overwrite=True.")
    # Create the destination parent (e.g. an output path /new/path/a.shad),
    # matching header.write_header / v2.writer.write_v2_archive.
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    members = _enumerate_members(src_dir)
    # Per-call staging name. A fixed "<dst>.tmp" is shared by any two writers targeting the
    # same output: both open it "wb", interleave, and whichever renames first publishes an
    # inode the other is still writing into -- observed to publish one job's data under the
    # other job's name (#280 item 5). pid + a FULL uuid4 (not a 32-bit prefix) makes that
    # structurally impossible, and "xb" (O_EXCL) turns any residual collision into a raise
    # instead of the silent truncation this is fixing. Explicit name rather than
    # tempfile.mkstemp: mkstemp creates 0600 and os.replace would carry that mode onto the
    # published archive, whereas "xb" creates with 0o666 & ~umask exactly like "wb".
    tmp = dst_file.parent / f"{dst_file.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    footer_members = []
    shard_region_end = None
    import struct as _struct

    # BaseException, not Exception: a KeyboardInterrupt mid-pack is exactly the case that
    # strands an archive-sized file, and no caller cleans it up -- write_packed/
    # write_packed_multi rmtree only their staging tmpdir, write_cell_archive only its
    # TemporaryDirectory. `created` guards against unlinking a file this call does not own
    # (an "xb" collision raises FileExistsError before we own anything). suppress(OSError)
    # so a cleanup failure cannot replace the real error. (#280 item 3)
    created = False
    try:
        with tmp.open("xb") as out:
            created = True
            out.write(fmt.HEAD_MAGIC)
            out.write(_struct.pack("<H", fmt.CONTAINER_VERSION))
            out.write(b"\x00" * (fmt.HEAD_BLOCK_SIZE - out.tell()))
            for name, role, shard_idx, matrix, srcpath in members:
                offset, length = _write_member(
                    out, srcpath, delete_after=(_delete_sources and role in _IMMUTABLE_ROLES)
                )
                entry = {"name": name, "role": role, "offset": offset, "length": length}
                if shard_idx is not None:
                    entry["shard_idx"] = shard_idx
                if matrix is not None:
                    entry["matrix"] = matrix
                footer_members.append(entry)
                if role not in _IMMUTABLE_ROLES and shard_region_end is None:
                    shard_region_end = offset
            if shard_region_end is None:
                shard_region_end = out.tell()
            footer_offset = out.tell()
            fbuf = fmt.serialize_footer(footer_members, shard_region_end=shard_region_end)
            out.write(fbuf)
            out.write(
                fmt.pack_trailer(
                    footer_offset=footer_offset,
                    footer_length=len(fbuf),
                    shard_region_end=shard_region_end,
                    footer_crc32=fmt.footer_crc32(fbuf),
                )
            )
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, dst_file)
    except BaseException:
        if created:
            with contextlib.suppress(OSError):
                tmp.unlink()
        raise


def write_packed(source, dst_file, *, overwrite=False, unsafe_no_lock=False, **v2_kwargs):
    """Writer-native pack: encode shards to a temp dir (existing v2 writer),
    then concat into ``dst_file`` deleting shard temps as appended (≈1× disk)."""
    from cellstream.v2.writer import write_v2_archive

    dst_file = Path(dst_file)
    if dst_file.exists() and not overwrite:
        raise ArchiveExistsError(f"{dst_file} already exists. Pass overwrite=True.")
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dst_file.with_suffix(dst_file.suffix + ".lock")
    with lock.acquire_write_lock(lock_path, op="write", unsafe_no_lock=unsafe_no_lock):
        tmpdir = Path(tempfile.mkdtemp(prefix=dst_file.name + ".packtmp.", dir=dst_file.parent))
        try:
            write_v2_archive(source, tmpdir, overwrite=True, unsafe_no_lock=True, **v2_kwargs)
            _pack_directory(tmpdir, dst_file, overwrite=overwrite, _delete_sources=True)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def write_packed_multi(source, dst_file, *, overwrite=False, unsafe_no_lock=False, **multi_kwargs):
    """Writer-native multi-matrix pack: encode a v4 directory to a temp dir
    (write_v2_multi), then concat into ``dst_file`` deleting shard temps."""
    from cellstream.v2.writer import write_v2_multi

    dst_file = Path(dst_file)
    if dst_file.exists() and not overwrite:
        raise ArchiveExistsError(f"{dst_file} already exists. Pass overwrite=True.")
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dst_file.with_suffix(dst_file.suffix + ".lock")
    with lock.acquire_write_lock(lock_path, op="write", unsafe_no_lock=unsafe_no_lock):
        tmpdir = Path(tempfile.mkdtemp(prefix=dst_file.name + ".packtmp.", dir=dst_file.parent))
        try:
            write_v2_multi(source, tmpdir, overwrite=True, unsafe_no_lock=True, **multi_kwargs)
            _pack_directory(tmpdir, dst_file, overwrite=overwrite, _delete_sources=True)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _adata_header_bytes(header_adata) -> bytes:
    """Serialize a header AnnData (X empty) to header.h5ad bytes via a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tf:
        tmpname = tf.name
    try:
        _header.write_header(header_adata, tmpname)
        return Path(tmpname).read_bytes()
    finally:
        Path(tmpname).unlink(missing_ok=True)


def _parquet_obs_index(pa):
    """The index of ``pa``'s obs.parquet -- what a reader's obs returns -- read without its
    columns, through a window over the archive rather than a copy of the whole member."""
    import pandas as pd

    with pa._member_window("obs.parquet") as w:
        return pd.read_parquet(w, columns=[]).index


def _parquet_obs_column(pa, name):
    """One column of ``pa``'s obs.parquet, read alone through a window over the archive."""
    import pandas as pd

    with pa._member_window("obs.parquet") as w:
        return pd.read_parquet(w, columns=[name])[name]


def _group_labels(col):
    """Each row's group label as the writer names it (stringify_group_labels), except that a
    missing value is None rather than the "nan" sentinel, so it is told apart from that literal.

    To a writer of this version a missing label and the literal "nan" are one group (#294), but a
    writer before it named missing rows 'None' or '<NA>', so swapping the literal in for them
    would give those rows a label no record carries -- and a column holding both is one
    validate_group_label_collision refuses to write at all. Missing flavours still compare equal
    to each other: None, NaN and pd.NA are the same missing label here.
    """
    from cellstream.grouping import stringify_group_labels

    labels = stringify_group_labels(col)
    labels[col.isna().to_numpy()] = None
    return labels


def _shown(label) -> str:
    return "missing" if label is None else repr(label)


def _same_partition(a, b) -> bool:
    """True if label arrays ``a`` and ``b`` group the rows alike: two rows share a label in one
    exactly when they share one in the other."""
    pairs = set(zip(a.tolist(), b.tolist(), strict=True))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def _refuse_group_by_change(pa, group_by, header_obs, new_obs) -> None:
    """Raise if ``new_obs`` drops the ``group_by`` column or gives any row another group label.

    The tail rewrite carries groups.json over byte-for-byte, and with it the spans read_group()
    and group_spans() serve, so obs must keep the labels those spans were planned from. Accepted
    before this check, a relabelled or dropped column left them serving the stored groups for rows
    obs labelled otherwise, or not at all (#339).

    Labels are compared the way the writer names groups (see _group_labels), not value for value,
    so a dtype change that keeps every label moves no row and is accepted. They are compared with
    the column as the archive stores it, in EITHER copy: the header's obs, already read, or
    obs.parquet's -- what a reader's obs returns -- read only when the header's does not match.
    The two can name the same groups differently: on pandas 2 an integer categorical with a
    missing value stays categorical in the h5ad ('1') but reads back from parquet as float64
    ('1.0'), so an unchanged column must pass from either -- obs.parquet's while it groups the rows
    as the header's does. Not against groups.json's labels: an archive written before #294 can
    name a missing label 'None' or '<NA>' there.
    """
    if group_by not in new_obs.columns:
        raise ValueError(
            f"new_obs has no {group_by!r} column, but the archive is grouped by it: groups.json, "
            "which read_group() and group_spans() serve, is kept as written, so obs must keep "
            "that column. Regrouping needs a re-write."
        )
    new = _group_labels(new_obs[group_by])
    copies = {}
    if group_by in header_obs.columns:
        header = _group_labels(header_obs[group_by])
        if (header == new).all():
            return
        copies["header.h5ad"] = header
    if pa.has_member("obs.parquet") and group_by in pa.obs_columns():
        parquet = _group_labels(_parquet_obs_column(pa, group_by))
        # A copy to match only if it groups the rows as the header does. A read-back can be lossy
        # enough to merge groups the header and groups.json keep apart -- integer categories
        # 2**53 and 2**53 + 1 come back from parquet as one float64 -- and matching it would
        # write that merge into both copies. (codex.)
        if "header.h5ad" not in copies or _same_partition(copies["header.h5ad"], parquet):
            if (parquet == new).all():
                return
            copies["obs.parquet"] = parquet
    if not copies:
        raise ValueError(
            f"the archive is grouped by {group_by!r}, but its stored obs has no {group_by!r} "
            "column (an update_obs made before #339 could drop it), so new_obs cannot be checked "
            "against the labels groups.json was written from. Re-write the archive."
        )
    # Reported against the copy the new column is closest to, which is named: measured against
    # the other, rows the edit never touched can differ too ('1' against '1.0'). (codex.)
    mismatch = {name: labels != new for name, labels in copies.items()}
    name = min(mismatch, key=lambda k: int(mismatch[k].sum()))
    changed = mismatch[name].nonzero()[0]
    i = int(changed[0])
    raise ValueError(
        f"new_obs[{group_by!r}] changes the group label of {changed.size} of {len(new)} rows "
        f"against the stored column in {name} (first: row {i}, {new_obs.index[i]!r}, "
        f"{_shown(copies[name][i])} -> {_shown(new[i])}), but the archive is grouped by that "
        "column: groups.json, which read_group() and group_spans() serve, is kept as written. "
        "Relabelling needs a re-write."
    )


def _obs_parquet_bytes(obs) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
        tmpname = tf.name
    try:
        _header.write_obs_parquet(obs, tmpname)
        return Path(tmpname).read_bytes()
    finally:
        Path(tmpname).unlink(missing_ok=True)


# The exact member NAMES _update_metadata_packed's tail rewrite re-emits; every member stored
# below shard_region_end is carried over verbatim alongside them. Keep in sync with the
# metadata_blobs list in _update_metadata_packed -- _unpreservable_members reads this to decide
# what the rewrite would destroy.
#
# NAMES, not roles. A role allowlist is not sufficient: the rewrite emits fixed names, so a
# member with an ALLOWED role but a different name (role="manifest", name="manifest2.json")
# passes a role check and is still silently dropped. Not reachable from today's
# _enumerate_members, whose role map is 1:1 with these names -- but the whole point of this
# guard is to fail closed on members it was not written for. (Copilot #259.)
_REWRITTEN_METADATA_NAMES = frozenset(
    {"header.h5ad", "obs.parquet", "manifest.json", "groups.json"}
)


def _below_region_end(member, shard_region_end) -> bool:
    """True if ``member`` lies wholly below ``shard_region_end``. The tail rewrite truncates
    there, so these bytes survive it unchanged and the member's footer entry stays valid as is."""
    return member["offset"] + member["length"] <= shard_region_end


def _unpreservable_members(members, shard_region_end):
    """Names in ``members`` that _update_metadata_packed's tail rewrite would NOT carry over.

    The rewrite keeps every member stored below ``shard_region_end`` -- bytes and footer entry
    unchanged -- and re-emits _REWRITTEN_METADATA_NAMES; anything else is deleted. A member below
    the boundary under one of those names is reported too: the rewrite would emit the name a
    second time, and a footer with a duplicate member name does not open.

    By position, not by role (#338). The packer writes the data members first -- v2 shards, the
    cell layout's payload / offsets / indptr -- and starts shard_region_end at the first member of
    any other role, so on every archive today's writers produce the two rules agree; but the
    position is what the truncation actually preserves. The role rule this replaces kept only
    role == "shard" and so refused every cell archive, and one widened to _IMMUTABLE_ROLES would
    still keep the entry of a data member stored PAST the boundary, whose bytes the truncation
    destroys.

    Split out from the guard so the rule is unit-testable without building an archive.
    """
    kept = {m["name"] for m in members if _below_region_end(m, shard_region_end)}
    lost = {m["name"] for m in members} - kept - _REWRITTEN_METADATA_NAMES
    return sorted(lost | (kept & _REWRITTEN_METADATA_NAMES))


def _finalize_tail(fh, kept_entries, metadata_blobs, shard_region_end):
    """Append metadata members (4 KB-aligned) + footer + trailer to ``fh`` (already
    seeked/truncated to shard_region_end). ``kept_entries`` are the footer entries of the
    members stored below shard_region_end, re-emitted unchanged; ``metadata_blobs`` is a list
    of (name, role, bytes). Returns nothing; fh is fsynced by the caller."""
    footer_members = list(kept_entries)
    for name, role, blob in metadata_blobs:
        pos = fh.tell()
        pad = fmt.align_up(pos) - pos
        if pad:
            fh.write(b"\x00" * pad)
        offset = fh.tell()
        fh.write(blob)
        footer_members.append({"name": name, "role": role, "offset": offset, "length": len(blob)})
    footer_offset = fh.tell()
    fbuf = fmt.serialize_footer(footer_members, shard_region_end=shard_region_end)
    fh.write(fbuf)
    fh.write(
        fmt.pack_trailer(
            footer_offset=footer_offset,
            footer_length=len(fbuf),
            shard_region_end=shard_region_end,
            footer_crc32=fmt.footer_crc32(fbuf),
        )
    )


def _update_metadata_packed(file, *, new_obs=None, new_var=None, unsafe_no_lock=False):
    file = Path(file)
    lock_path = file.with_suffix(file.suffix + ".lock")
    op = "update_var" if new_var is not None else "update_obs"
    # Acquire the lock BEFORE parsing: a concurrent writer fast-fails with
    # ArchiveLockedError (not a confusing torn-tail CRC error), and parse→rewrite
    # happen atomically under the lock so concurrent edits can't lose updates.
    with lock.acquire_write_lock(lock_path, op=op, unsafe_no_lock=unsafe_no_lock):
        # We hold the lock: finish any prior interrupted edit, then parse the
        # recovered state with open-time recovery disabled (it would re-enter
        # this same flock).
        _apply_recovery(file)
        pa = PackedArchive(file, _recover=False)
        # Fail before the tail rewrite. The rewrite below keeps the members stored below
        # shard_region_end and re-emits a FIXED list of four metadata blobs BY NAME, so any other
        # member is dropped when the tail is truncated -- and the .tailbak that could restore it
        # is unlinked on the success path, so the loss is unrecoverable.
        #
        # Reproduced on a v4 multi-matrix archive: update_obs() returned SUCCESSFULLY,
        # destroyed var/x.parquet and var/raw.parquet, removed the .tailbak, and left the
        # archive raising PackedArchiveError: member 'var/raw.parquet' not found. (#254)
        #
        # Carrying the sidecars through metadata_blobs is the real fix and can land when
        # multi-matrix is actually in use; refusing is what makes the destructive path
        # unreachable in the meantime.
        #
        # Scope of the guarantee, precisely: this runs AFTER _apply_recovery and after the
        # lock file is created, so it is not 'no bytes written since call entry'. Recovery
        # completing a previously-interrupted update is correct and happens under this same
        # exclusive flock. What is guaranteed is that the REQUESTED update does not rewrite the
        # tail. On a clean archive -- the case that matters -- the packed archive file itself is
        # not modified, though the .lock sidecar is still written. (codex review of #254.)
        sre = pa.shard_region_end
        unpreservable = _unpreservable_members(pa.members, sre)
        if unpreservable:
            raise NotImplementedError(
                f"{file}: update_obs/update_var does not preserve {unpreservable}. The tail "
                "rewrite keeps only the members stored below shard_region_end and re-emits "
                "header.h5ad, obs.parquet, manifest.json and groups.json by name, so any other "
                "member would be deleted permanently (and one stored below the boundary under one "
                "of those names would be emitted twice). Multi-matrix (v4) archives' per-family "
                "var/ and meta/ sidecars are the known case (project issue #254). The requested "
                "update did not rewrite the tail."
            )
        header_adata = pa.header_adata()
        if new_obs is not None:
            if len(new_obs) != header_adata.n_obs:
                raise ValueError(
                    f"new_obs has {len(new_obs)} rows but archive has n_obs={header_adata.n_obs}"
                )
            # Match the directory update path: the index (obs_names / row order)
            # must be preserved — we don't reorder cells, and reordering would
            # misalign obs with the stored X rows.
            #
            # Exactly, against either index the archive stores: the header's obs_names, or
            # obs.parquet's -- what a reader's obs returns. They differ when a writer was handed a
            # non-string index: anndata stringifies the header's names, while obs.parquet keeps
            # the index as given (the push writer stores a default integer index as int64), so
            # `store.obs` plus a column matched neither until it was compared with obs.parquet's
            # (#338). Never a string cast of both: it makes the label "nan" equal to a missing
            # label, which let a swap of those two rows through. (codex.)
            if not new_obs.index.equals(header_adata.obs.index) and not (
                pa.has_member("obs.parquet") and new_obs.index.equals(_parquet_obs_index(pa))
            ):
                raise ValueError(
                    "new_obs index does not match the archive's obs index; "
                    "obs row order must be preserved (we don't reorder cells)."
                )
            # After the index check, so the stored and new rows line up.
            if pa.manifest.get("group_by") is not None:
                _refuse_group_by_change(pa, pa.manifest["group_by"], header_adata.obs, new_obs)
            header_adata.obs = new_obs
        if new_var is not None:
            if len(new_var) != header_adata.n_vars:
                raise ValueError(
                    f"new_var has {len(new_var)} rows but archive has n_vars={header_adata.n_vars}"
                )
            if not new_var.index.equals(header_adata.var.index):
                raise ValueError(
                    "new_var index does not match the archive's var index; "
                    "var order must be preserved (we don't reorder genes)."
                )
            header_adata.var = new_var

        # Every member below sre keeps its footer entry: the truncation never touches its bytes.
        # That is v2 shards, and a cell archive's x_payload / x_offsets / x_indptr -- which the
        # role == "shard" filter this replaces left out, so the guard refused every cell archive
        # (#338). Footer order is kept.
        kept_entries = [m for m in pa.members if _below_region_end(m, sre)]
        if new_obs is not None:
            # From the frame as given, like the writers do -- not from the header's copy, whose
            # index anndata may have stringified.
            obs_blob = _obs_parquet_bytes(new_obs)
        elif pa.has_member("obs.parquet"):
            # An update_var leaves obs alone, so obs.parquet is carried over byte-for-byte.
            # Rebuilding it from header_adata.obs -- the HEADER's copy -- changed its dtypes:
            # anndata stores repeated strings in an h5ad as categoricals, so a string column of
            # obs came back as `category` from an edit that never touched obs (#338).
            obs_blob = pa.member_bytes("obs.parquet")
        else:
            obs_blob = _obs_parquet_bytes(header_adata.obs)
        metadata_blobs = [
            ("header.h5ad", "header", _adata_header_bytes(header_adata)),
            ("obs.parquet", "obs", obs_blob),
            ("manifest.json", "manifest", pa.member_bytes("manifest.json")),
        ]
        if pa.has_member("groups.json"):
            metadata_blobs.append(("groups.json", "groups", pa.member_bytes("groups.json")))

        # Back up the old tail (prefixed with its shard_region_end u64) for rollback. Sized through
        # the fd that reads it, never `file.stat()`: a stale-short size on a network filesystem
        # would back up a TRUNCATED tail, so a crash between here and the rewrite below would
        # recover the archive to a short one (#305). `seek(END)` returns the same number fstat
        # would, on the handle that is being opened anyway -- and unlike `_file_size` elsewhere,
        # this one IS the handle the read uses, so the path cannot be swapped between sizing it and
        # reading it. (Not a guarantee against a concurrent writer on the same inode; nothing here
        # claims one.)
        with file.open("rb") as rfh:
            size = rfh.seek(0, os.SEEK_END)
            rfh.seek(sre)
            old_tail = rfh.read(size - sre)
        tailbak = file.with_suffix(file.suffix + ".tailbak")
        with tailbak.open("wb") as bf:
            bf.write(struct.pack("<Q", sre))
            bf.write(old_tail)
            bf.flush()
            os.fsync(bf.fileno())
        # Truncate the tail and rewrite it.
        with file.open("r+b") as fh:
            fh.seek(sre)
            fh.truncate(sre)
            _finalize_tail(fh, kept_entries, metadata_blobs, sre)
            fh.flush()
            os.fsync(fh.fileno())
        tailbak.unlink(missing_ok=True)


def update_obs_packed(file, new_obs, *, unsafe_no_lock=False):
    _update_metadata_packed(file, new_obs=new_obs, unsafe_no_lock=unsafe_no_lock)


def update_var_packed(file, new_var, *, unsafe_no_lock=False):
    _update_metadata_packed(file, new_var=new_var, unsafe_no_lock=unsafe_no_lock)
