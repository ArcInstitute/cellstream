"""Read-side accessor for a packed (.shad) archive."""

from __future__ import annotations

import json
from pathlib import Path

from cellstream.errors import IncompatibleSchemaError, PackedArchiveError
from cellstream.manifest import SUPPORTED_SCHEMA_VERSIONS, validate_manifest
from cellstream.packed import format as fmt


def is_packed_file(path) -> bool:
    """True if ``path`` is a regular file starting with the SHPK head magic."""
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("rb") as fh:
        return fh.read(4) == fmt.HEAD_MAGIC


def _is_safe_member_name(name) -> bool:
    """A member name is either a single bare filename or a two-segment
    ``<dir>/<file>`` family path (v4 multi-matrix: ``x/shard_0000.shx``,
    ``var/raw.parquet``). Each segment must be a safe bare name — no parent
    refs, not absolute, no backslashes — so unpack() can't be tricked into
    writing outside its destination (Zip-Slip)."""
    if not isinstance(name, str) or not name:
        return False
    if "\\" in name:
        return False
    segments = name.split("/")
    if len(segments) > 2:
        return False
    for seg in segments:
        if not seg or seg in (".", ".."):
            return False
        p = Path(seg)
        if p.is_absolute() or seg != p.name:
            return False
    return True


def _index_shard_members(members: list[dict], manifest: dict) -> dict[str | None, list[dict]]:
    """Group shard members for lookup. v1/2/3 -> {None: [members by shard_idx]};
    v4 -> {matrix_key: [members by shard_idx]}. Validates exactly n_shards per
    family, numbered {0..n_shards-1} with no gap/dup."""
    if manifest.get("layout") == "cell":
        shard_members = [m["name"] for m in members if m.get("role") == "shard"]
        if shard_members:
            raise PackedArchiveError(
                f"cell layout archive has unexpected shard members: {shard_members!r}."
            )
        return {None: []}
    n = manifest["n_shards"]
    is_v4 = manifest.get("schema_version") == 4
    buckets: dict = {}
    if not is_v4:
        # Ensure the completeness check runs even for a corrupt legacy archive
        # with zero shard members (n>0); otherwise the loop below is skipped and
        # the missing shards go undetected.
        buckets.setdefault(None, [])
    for m in members:
        if m.get("role") != "shard":
            continue
        key = m.get("matrix") if is_v4 else None
        buckets.setdefault(key, []).append(m)
    for key, shs in buckets.items():
        shs.sort(key=lambda m: m["shard_idx"])
        idxs = [m["shard_idx"] for m in shs]
        if idxs != list(range(n)):
            raise PackedArchiveError(
                f"footer shard_idx values {idxs} for matrix {key!r} are not exactly "
                f"{{0..{n - 1}}} (gap, duplicate, or wrong count) — corrupt archive."
            )
    if is_v4:
        expected = set(manifest.get("matrices", {}))
        if set(buckets) != expected:
            raise PackedArchiveError(
                f"footer shard families {set(buckets)} != manifest matrices {expected}."
            )
    return buckets


def _trailer_valid(path: Path) -> bool:
    try:
        size = path.stat().st_size
        if size < fmt.TRAILER_SIZE:
            return False
        with path.open("rb") as fh:
            fh.seek(size - fmt.TRAILER_SIZE)
            t = fmt.parse_trailer(fh.read(fmt.TRAILER_SIZE))
            foff, flen = t["footer_offset"], t["footer_length"]
            # Bounds-check before reading so a corrupt/oversized footer_length
            # can't drive a huge read; the footer must lie before the trailer.
            if foff < fmt.HEAD_BLOCK_SIZE or flen < 0 or foff + flen > size - fmt.TRAILER_SIZE:
                return False
            fh.seek(foff)
            fbuf = fh.read(flen)
        if len(fbuf) != flen or fmt.footer_crc32(fbuf) != t["footer_crc32"]:
            return False
        # The footer/trailer shard_region_end cross-check MUST happen here, not only in
        # PackedArchive.__init__. _apply_recovery deletes the .tailbak the moment this returns
        # true, so a valid-CRC tail whose two SRE copies disagree would lose its good backup and
        # only THEN fail to open -- turning an unopenable archive into an unrecoverable one. The
        # cross-check was added to __init__ first and introduced exactly that window.
        # (codex, checkpoint 2 round 2.)
        footer = fmt.parse_footer(fbuf)
        if not isinstance(footer, dict):
            return False
        return footer.get("shard_region_end") == t["shard_region_end"]
    except Exception:
        return False


def _apply_recovery(path: Path) -> None:
    """Finish an interrupted edit. **Caller MUST already hold the write lock.**

    If a <file>.tailbak exists: when the current trailer is already valid the
    edit completed before its unlink (drop the stale backup); otherwise roll the
    tail back to the backed-up previous state. No-op when no .tailbak exists.
    """
    import os
    import struct

    tailbak = path.with_suffix(path.suffix + ".tailbak")
    if not tailbak.exists():
        return
    if _trailer_valid(path):
        tailbak.unlink()  # edit completed before unlink; backup is stale
        return
    raw = tailbak.read_bytes()
    # Sanity-check the backup before truncating: a corrupt/tampered .tailbak must
    # NOT be able to irreversibly truncate the archive (e.g. sre=0). Fail loud.
    if len(raw) < 8:
        raise PackedArchiveError(
            f"{path}: .tailbak is too short to be valid; refusing to recover. "
            f"Inspect/remove {tailbak} manually."
        )
    (sre,) = struct.unpack("<Q", raw[:8])
    size = path.stat().st_size
    if not (fmt.HEAD_BLOCK_SIZE <= sre <= size) or sre % fmt.ALIGNMENT != 0:
        raise PackedArchiveError(
            f"{path}: .tailbak shard_region_end {sre} is out of range or misaligned; "
            f"refusing to truncate the archive. Inspect/remove {tailbak} manually."
        )
    old_tail = raw[8:]
    with path.open("r+b") as fh:
        fh.seek(sre)
        fh.truncate(sre)
        fh.write(old_tail)
        fh.flush()
        os.fsync(fh.fileno())
    tailbak.unlink()


def _recover_if_needed(path: Path) -> None:
    """If a <file>.tailbak from an interrupted edit exists, recover UNDER THE
    ARCHIVE'S WRITE LOCK.

    Recovery truncates/rewrites the tail, so it must hold the same exclusive
    lock writers use — otherwise a reader opening the file mid-edit could roll
    back a live writer's in-progress rewrite and corrupt the archive. If a live
    writer currently holds the lock, the .tailbak belongs to that in-progress
    edit (not a crash), so we skip recovery: the writer will complete it, or its
    crash will free the lock (flock auto-releases on death) for a later open to
    recover. The no-.tailbak fast path takes no lock, so normal reads are free.
    """
    from cellstream import lock as _lock
    from cellstream.errors import ArchiveLockedError

    tailbak = path.with_suffix(path.suffix + ".tailbak")
    if not tailbak.exists():
        return
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        with _lock.acquire_write_lock(lock_path, op="recover"):
            _apply_recovery(path)  # re-checks .tailbak under the lock
    except ArchiveLockedError:
        # A live writer holds the lock and owns the in-progress edit — don't touch.
        return


class _MmapWindow:
    """0-based read-only seekable file-like over [base, base+length) of an mmap.

    Lets h5py read header.h5ad's datasets directly from the packed file's
    mmap with no intermediate full-member bytes copy.
    """

    def __init__(self, mm, base: int, length: int):
        self._mm = mm
        self._base = base
        self._len = length
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        # Clamp to the bytes actually available; a caller (e.g. h5py) may have
        # seeked at or past EOF, in which case `avail` is 0 and we return b""
        # like a normal stream (never a negative-length slice).
        avail = max(0, self._len - self._pos)
        if n is None or n < 0 or n > avail:
            n = avail
        start = self._base + self._pos
        out = bytes(self._mm[start : start + n])
        self._pos += len(out)
        return out

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            new = offset
        elif whence == 1:
            new = self._pos + offset
        elif whence == 2:
            new = self._len + offset
        else:
            raise ValueError(f"invalid whence: {whence!r}")
        # Seeking past EOF is allowed (subsequent reads return b""); a negative
        # absolute position is not.
        if new < 0:
            raise ValueError(f"negative seek position {new}")
        self._pos = new
        return self._pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False


class PackedArchive:
    """Parses a packed file's trailer + footer and locates members by name."""

    def __init__(self, path, *, _recover: bool = True):
        # _recover=False: caller already holds the write lock and has applied
        # recovery itself (the writer path), so skip the lock-acquiring open-time
        # recovery to avoid re-entering the same flock.
        self.path = Path(path)
        if _recover:
            _recover_if_needed(self.path)
        with self.path.open("rb") as fh:
            if fh.read(4) != fmt.HEAD_MAGIC:
                raise PackedArchiveError(f"{self.path}: bad head magic; not a packed archive")
            fh.seek(0, 2)
            size = fh.tell()
            if size < fmt.TRAILER_SIZE:
                raise PackedArchiveError(f"{self.path}: file too small to be a packed archive")
            fh.seek(size - fmt.TRAILER_SIZE)
            self.trailer = fmt.parse_trailer(fh.read(fmt.TRAILER_SIZE))
            ver = self.trailer["format_version"]
            if ver > fmt.CONTAINER_VERSION:
                raise PackedArchiveError(
                    f"{self.path}: packed container version {ver} is newer than this "
                    f"cellstream supports ({fmt.CONTAINER_VERSION}); upgrade cellstream to read it."
                )
            # Validate the footer slice lies entirely before the trailer and
            # within the file before reading it, so a corrupt/huge footer_length
            # fails loud instead of reading garbage / over-reading.
            foff = self.trailer["footer_offset"]
            flen = self.trailer["footer_length"]
            if foff < fmt.HEAD_BLOCK_SIZE or foff + flen > size - fmt.TRAILER_SIZE:
                raise PackedArchiveError(
                    f"{self.path}: footer (offset={foff}, length={flen}) is out of bounds "
                    f"for a {size}-byte file — corrupt trailer."
                )
            fh.seek(foff)
            fbuf = fh.read(flen)
        if fmt.footer_crc32(fbuf) != self.trailer["footer_crc32"]:
            raise PackedArchiveError(
                f"{self.path}: footer CRC mismatch — corrupt file or an interrupted edit."
            )
        self.footer = fmt.parse_footer(fbuf)
        if not isinstance(self.footer, dict):
            raise PackedArchiveError(f"{self.path}: footer is not a JSON object — corrupt footer.")
        members = self.footer.get("members")
        sre = self.footer.get("shard_region_end")
        if not isinstance(members, list) or not isinstance(sre, int):
            raise PackedArchiveError(
                f"{self.path}: footer is missing or has an invalid 'members' list / "
                f"'shard_region_end' — corrupt footer."
            )
        self.members = members
        # The trailer and the footer each carry shard_region_end, and they must agree: the
        # trailer's copy is what a recovery path reads WITHOUT parsing the footer, so a
        # disagreement means one of the two is stale or crafted, and picking either silently
        # would put the truncation boundary somewhere the other half of the file does not expect.
        # Cheap to compare, and the write path emits both from one variable. (codex, checkpoint 2.)
        if sre != self.trailer["shard_region_end"]:
            raise PackedArchiveError(
                f"{self.path}: footer shard_region_end {sre} disagrees with the trailer's "
                f"{self.trailer['shard_region_end']} — corrupt file."
            )
        self.shard_region_end = sre
        # Validate footer-provided names + offsets against the file bounds BEFORE
        # any member is read / mmap'd / used to truncate. A valid-CRC but bogus
        # footer (crafted or corrupt) must not drive huge reads, bad mmap windows,
        # path traversal in unpack(), or dangerous truncation offsets.
        self._validate_footer(foff)
        self._by_name = {m["name"]: m for m in self.members}
        self._manifest = json.loads(self.member_bytes("manifest.json").decode("utf-8"))
        # Apply the same schema-compatibility guard as manifest.load_manifest so a
        # packed archive with an unsupported schema_version fails loud here, not
        # in a confusing downstream way.
        v = self._manifest.get("schema_version")
        if v not in SUPPORTED_SCHEMA_VERSIONS:
            raise IncompatibleSchemaError(
                f"{self.path}: manifest schema_version {v!r} is not supported "
                f"(this cellstream supports {SUPPORTED_SCHEMA_VERSIONS}). "
                f"Either upgrade cellstream or regenerate the archive."
            )
        # row_offsets shape invariants (ultrareview L2) — same check directory
        # archives get via manifest.load_manifest; raises IncompatibleSchemaError.
        validate_manifest(self._manifest)
        # Cross-check the footer's shard members against the manifest (ultrareview
        # L3): exactly n_shards shards numbered {0..n_shards-1}, no gaps/dups —
        # per-matrix for v4, flat (None bucket) for v1/2/3.
        self._shards_by_matrix = _index_shard_members(self.members, self._manifest)
        # v1/2/3 keep a flat _shards list (matrix=None bucket) for the unchanged path.
        self._shards = self._shards_by_matrix.get(None, [])
        self.n_shards = len(self._shards)

    def _validate_footer(self, footer_offset: int) -> None:
        """Reject a footer whose members / shard_region_end are unsafe or out of
        bounds, before any of them is used to read, mmap, or truncate the file."""
        sre = self.shard_region_end
        # sre is used as a truncation boundary on update/recovery, and shards/header
        # are mmap'd at page-aligned offsets — so it must be in range AND aligned.
        if not (fmt.HEAD_BLOCK_SIZE <= sre <= footer_offset) or sre % fmt.ALIGNMENT != 0:
            raise PackedArchiveError(
                f"{self.path}: shard_region_end {sre!r} out of range "
                f"[{fmt.HEAD_BLOCK_SIZE}, {footer_offset}] or not {fmt.ALIGNMENT}-byte "
                f"aligned — corrupt footer."
            )
        seen_names: set[str] = set()
        for m in self.members:
            if not isinstance(m, dict):
                raise PackedArchiveError(f"{self.path}: footer member is not an object: {m!r}")
            name = m.get("name")
            if not _is_safe_member_name(name):
                raise PackedArchiveError(
                    f"{self.path}: unsafe member name {name!r} (must be a bare filename)."
                )
            # Member names must be unique — self._by_name would otherwise silently
            # drop a duplicate (last-wins), masking a corrupt/crafted footer.
            if name in seen_names:
                raise PackedArchiveError(f"{self.path}: duplicate member name {name!r} in footer.")
            seen_names.add(name)
            role = m.get("role")
            off, length = m.get("offset"), m.get("length")
            if not isinstance(off, int) or not isinstance(length, int) or length < 0:
                raise PackedArchiveError(f"{self.path}: member {name!r} has invalid offset/length.")
            if off < fmt.HEAD_BLOCK_SIZE or off + length > footer_offset:
                raise PackedArchiveError(
                    f"{self.path}: member {name!r} (offset={off}, length={length}) is out of "
                    f"bounds for the body region [{fmt.HEAD_BLOCK_SIZE}, {footer_offset}]."
                )
            # shard/header/payload members are mmap'd at their base, which requires page
            # alignment; a misaligned offset is a corrupt footer.
            if role in ("shard", "header", "payload") and off % fmt.ALIGNMENT != 0:
                raise PackedArchiveError(
                    f"{self.path}: member {name!r} (role={role!r}) offset {off} is not "
                    f"{fmt.ALIGNMENT}-byte aligned — corrupt footer."
                )
            # shard members are sorted by shard_idx, so it must be a valid int.
            # Reject bool too (it's an int subclass): a crafted footer with
            # shard_idx true/false would otherwise pass here AND pass the
            # [False, True] == [0, 1] cross-check in _index_shard_members.
            si = m.get("shard_idx")
            if role == "shard" and (not isinstance(si, int) or isinstance(si, bool)):
                raise PackedArchiveError(
                    f"{self.path}: shard member {name!r} is missing an integer 'shard_idx'."
                )
            # "matrix" (v4 family key) is used as a dict key in _index_shard_members
            # (buckets.setdefault(key, [])); a non-str/non-None value (e.g. a list)
            # would pass footer JSON parsing but raw-crash there with a TypeError.
            matrix = m.get("matrix")
            if matrix is not None and not isinstance(matrix, str):
                raise PackedArchiveError(
                    f"{self.path}: member {name!r} has a non-string 'matrix' field: {matrix!r}."
                )

    @property
    def manifest(self) -> dict:
        return self._manifest

    def member_bytes(self, name: str) -> bytes:
        if name not in self._by_name:
            raise PackedArchiveError(f"{self.path}: member {name!r} not found")
        m = self._by_name[name]
        with self.path.open("rb") as fh:
            fh.seek(m["offset"])
            data = fh.read(m["length"])
        if len(data) != m["length"]:
            raise PackedArchiveError(
                f"{self.path}: short read for member {name!r} (got {len(data)} of "
                f"{m['length']} bytes — truncated or corrupt file)."
            )
        return data

    def has_member(self, name: str) -> bool:
        return name in self._by_name

    def member_location(self, name: str):
        """Return (file_path:str, base_offset:int, length:int) for a named member.

        The by-name analogue of ``shard_location`` — for members mmap'd by name
        (e.g. the cell layout's ``x_payload``). ``base_offset`` is 4 KB-aligned for
        roles the writer aligns (payload/header/shard).
        """
        if name not in self._by_name:
            raise PackedArchiveError(f"{self.path}: member {name!r} not found")
        m = self._by_name[name]
        return (str(self.path), m["offset"], m["length"])

    def shard_location(self, idx: int, *, matrix: str | None = None):
        """Return (file_path:str, base_offset:int, length:int) for shard ``idx``.

        v4 archives require ``matrix=<key>``; v1/2/3 use the flat family."""
        if self._manifest.get("schema_version") == 4:
            if matrix is None:
                raise PackedArchiveError(
                    f"{self.path}: this is a multi-matrix (v4) archive; "
                    f"shard_location requires matrix=<key>."
                )
            if matrix not in self._shards_by_matrix:
                raise PackedArchiveError(
                    f"{self.path}: unknown matrix {matrix!r}; this archive has "
                    f"matrices {list(self._shards_by_matrix)}."
                )
            shards = self._shards_by_matrix[matrix]
        else:
            shards = self._shards
        # Bounds-check so a negative idx can't wrap (Python negative indexing) to
        # the wrong shard and an out-of-range idx raises a clear PackedArchiveError
        # rather than a bare IndexError.
        if not (0 <= idx < len(shards)):
            raise PackedArchiveError(
                f"{self.path}: shard index {idx} out of range [0, {len(shards)})"
            )
        m = shards[idx]
        return (str(self.path), m["offset"], m["length"])

    def var_bytes(self, key: str) -> bytes:
        """Raw parquet bytes of a v4 own-var family's ``var/<dir>.parquet``."""
        from cellstream.matrices import var_member_name

        return self.member_bytes(var_member_name(key))

    def meta_bytes(self, key: str) -> bytes:
        """Raw h5ad bytes of a v4 family's ``meta/<dir>.h5ad`` metadata sidecar."""
        from cellstream.matrices import meta_member_name

        return self.member_bytes(meta_member_name(key))

    def header_adata(self):
        """Return the header AnnData (X empty), read directly from an mmap window.

        anndata.read_h5ad accepts a binary file-like and opens it with h5py
        internally (VERIFIED on anndata 0.12.16 / h5py 3.16.0); it does NOT
        accept an already-open ``h5py.File`` (raises TypeError). Falls back to a
        temp-file extraction if the file-like path fails on an older anndata.
        """
        import mmap

        import anndata as ad

        if "header.h5ad" not in self._by_name:
            raise PackedArchiveError(f"{self.path}: required member 'header.h5ad' is missing")
        m = self._by_name["header.h5ad"]
        fh = self.path.open("rb")
        try:
            mm = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
            try:
                return ad.read_h5ad(_MmapWindow(mm, m["offset"], m["length"]))
            except (OSError, ValueError, TypeError):
                import tempfile

                # Capture tmpname before the try so a failure DURING the chunked copy
                # (e.g. a truncated member) still unlinks the delete=False temp file
                # rather than leaking it (ultrareview L14). tmpname stays None if
                # NamedTemporaryFile itself fails, so the finally can't NameError and
                # mask that error.
                tmpname = None
                try:
                    fh.seek(m["offset"])
                    remaining = m["length"]
                    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tf:
                        tmpname = tf.name
                        while remaining:
                            chunk = fh.read(min(1 << 20, remaining))
                            if not chunk:
                                raise PackedArchiveError(
                                    f"{self.path}: truncated header.h5ad member "
                                    f"(EOF with {remaining} bytes unread)"
                                ) from None
                            tf.write(chunk)
                            remaining -= len(chunk)
                    return ad.read_h5ad(tmpname)
                finally:
                    if tmpname is not None:
                        # Suppress unlink errors: a raise in finally would discard the
                        # primary exception from the try (Gemini). missing_ok handles
                        # the absent-file case; this guards the rest (perms, locks).
                        try:
                            Path(tmpname).unlink(missing_ok=True)
                        except OSError:
                            pass
            finally:
                mm.close()
        finally:
            fh.close()

    def obs(self):
        """Return the obs DataFrame, preferring obs.parquet (matches _v1)."""
        import io

        import pandas as pd

        if self.has_member("obs.parquet"):
            return pd.read_parquet(io.BytesIO(self.member_bytes("obs.parquet")))
        return self.header_adata().obs

    def var(self):
        return self.header_adata().var
