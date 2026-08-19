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
    length = src_path.stat().st_size
    with src_path.open("rb") as fh:
        shutil.copyfileobj(fh, out, length=1 << 20)
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


def _obs_parquet_bytes(obs) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
        tmpname = tf.name
    try:
        _header.write_obs_parquet(obs, tmpname)
        return Path(tmpname).read_bytes()
    finally:
        Path(tmpname).unlink(missing_ok=True)


# The exact non-shard member NAMES _update_metadata_packed's tail rewrite re-emits; shard
# members are carried over verbatim alongside them. Keep in sync with the metadata_blobs
# list in _update_metadata_packed -- _unpreservable_members reads this to decide what the
# rewrite would destroy.
#
# NAMES, not roles. A role allowlist is not sufficient: the rewrite emits fixed names, so a
# member with an ALLOWED role but a different name (role="manifest", name="manifest2.json")
# passes a role check and is still silently dropped. Not reachable from today's
# _enumerate_members, whose role map is 1:1 with these names -- but the whole point of this
# guard is to fail closed on members it was not written for. (Copilot #259.)
_REWRITTEN_METADATA_NAMES = frozenset(
    {"header.h5ad", "obs.parquet", "manifest.json", "groups.json"}
)


def _unpreservable_members(members):
    """Names in ``members`` that _update_metadata_packed's tail rewrite would NOT re-emit.

    Split out from the guard so the rule is unit-testable without building an archive.
    """
    preserved = {m["name"] for m in members if m.get("role") == "shard"}
    preserved |= _REWRITTEN_METADATA_NAMES
    return sorted({m["name"] for m in members if m["name"] not in preserved})


def _finalize_tail(fh, shard_entries, metadata_blobs, shard_region_end):
    """Append metadata members (4 KB-aligned) + footer + trailer to ``fh`` (already
    seeked/truncated to shard_region_end). ``metadata_blobs`` is a list of
    (name, role, bytes). Returns nothing; fh is fsynced by the caller."""
    footer_members = list(shard_entries)  # shard entries unchanged (offset < sre)
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
        # Fail before the tail rewrite. The rewrite below re-emits role=="shard" members plus
        # a FIXED list of four metadata blobs BY NAME, so any other member is dropped when the
        # tail is truncated -- and the .tailbak that could restore it is unlinked on the
        # success path, so the loss is unrecoverable.
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
        unpreservable = _unpreservable_members(pa.members)
        if unpreservable:
            raise NotImplementedError(
                f"{file}: update_obs/update_var does not preserve {unpreservable}, and "
                "rewriting the archive tail would delete them permanently. Multi-matrix "
                "(v4) archives carry per-family var/ and meta/ sidecars that this path "
                "cannot round-trip yet (project issue #254). The requested update did not rewrite "
                "the tail."
            )
        sre = pa.shard_region_end
        header_adata = pa.header_adata()
        if new_obs is not None:
            if len(new_obs) != header_adata.n_obs:
                raise ValueError(
                    f"new_obs has {len(new_obs)} rows but archive has n_obs={header_adata.n_obs}"
                )
            # Match the directory update path: the index (obs_names / row order)
            # must be preserved — we don't reorder cells, and reordering would
            # misalign obs with the stored X rows.
            if not new_obs.index.equals(header_adata.obs.index):
                raise ValueError(
                    "new_obs index does not match the archive's obs index; "
                    "obs row order must be preserved (we don't reorder cells)."
                )
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

        shard_entries = [m for m in pa.members if m["role"] == "shard"]
        metadata_blobs = [
            ("header.h5ad", "header", _adata_header_bytes(header_adata)),
            ("obs.parquet", "obs", _obs_parquet_bytes(header_adata.obs)),
            ("manifest.json", "manifest", pa.member_bytes("manifest.json")),
        ]
        if pa.has_member("groups.json"):
            metadata_blobs.append(("groups.json", "groups", pa.member_bytes("groups.json")))

        # Back up the old tail (prefixed with its shard_region_end u64) for rollback.
        size = file.stat().st_size
        with file.open("rb") as rfh:
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
            _finalize_tail(fh, shard_entries, metadata_blobs, sre)
            fh.flush()
            os.fsync(fh.fileno())
        tailbak.unlink(missing_ok=True)


def update_obs_packed(file, new_obs, *, unsafe_no_lock=False):
    _update_metadata_packed(file, new_obs=new_obs, unsafe_no_lock=unsafe_no_lock)


def update_var_packed(file, new_var, *, unsafe_no_lock=False):
    _update_metadata_packed(file, new_var=new_var, unsafe_no_lock=unsafe_no_lock)
