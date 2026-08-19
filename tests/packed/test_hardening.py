"""Hardening tests for the packed container (Copilot code-review findings)."""

from __future__ import annotations

import pytest

from cellstream.packed.reader import _MmapWindow, is_packed_file
from cellstream.packed.writer import _pack_directory as pack
from cellstream.v2.reader_cpu import read_v2_to_host


def test_mmapwindow_read_past_eof_returns_empty():
    # _MmapWindow slices its backing object, so a bytes buffer works as the "mm".
    w = _MmapWindow(b"hello world", base=0, length=5)  # window = b"hello"
    assert w.read() == b"hello"
    assert w.read() == b""  # at EOF
    w.seek(100)  # seeking past EOF is allowed
    assert w.read(10) == b""  # reads past EOF return b"", never a negative-length slice


def test_mmapwindow_seek_validation():
    w = _MmapWindow(b"hello", base=0, length=5)
    with pytest.raises(ValueError):
        w.seek(-1)  # negative absolute position
    with pytest.raises(ValueError):
        w.seek(0, 9)  # invalid whence


def test_mmapwindow_base_offset():
    # window over bytes [6, 11) of the buffer = b"world"
    w = _MmapWindow(b"hello world", base=6, length=5)
    assert w.read() == b"world"


def test_read_v2_to_host_requires_dir_or_locator():
    with pytest.raises(ValueError, match="archive_dir"):
        read_v2_to_host(None)
    with pytest.raises(ValueError, match="archive_dir"):
        # manifest supplied but no archive_dir and no shard_locator
        read_v2_to_host(None, manifest={"n_obs_total": 1})


def test_pack_creates_parent_dirs(grouped_dir, tmp_path):
    dst = tmp_path / "new" / "nested" / "a.shardpack"  # parents do not exist
    pack(grouped_dir, dst)
    assert is_packed_file(dst)


def test_trailer_version_too_new_raises(grouped_dir, tmp_path):
    """A packed file whose trailer format_version is newer than we support fails loud."""
    from cellstream.errors import PackedArchiveError
    from cellstream.packed import format as fmt
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    data = bytearray(dst.read_bytes())
    t = fmt.parse_trailer(bytes(data[-fmt.TRAILER_SIZE :]))
    newer = fmt.pack_trailer(
        footer_offset=t["footer_offset"],
        footer_length=t["footer_length"],
        shard_region_end=t["shard_region_end"],
        footer_crc32=t["footer_crc32"],
        version=fmt.CONTAINER_VERSION + 1,
    )
    data[-fmt.TRAILER_SIZE :] = newer
    dst.write_bytes(bytes(data))
    with pytest.raises(PackedArchiveError, match="newer than this cellstream"):
        PackedArchive(dst)


def test_corrupt_footer_offset_raises(grouped_dir, tmp_path):
    """A trailer whose footer (offset,length) is out of bounds fails loud."""
    from cellstream.errors import PackedArchiveError
    from cellstream.packed import format as fmt
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    data = bytearray(dst.read_bytes())
    t = fmt.parse_trailer(bytes(data[-fmt.TRAILER_SIZE :]))
    bad = fmt.pack_trailer(
        footer_offset=10**12,  # way past EOF
        footer_length=t["footer_length"],
        shard_region_end=t["shard_region_end"],
        footer_crc32=t["footer_crc32"],
    )
    data[-fmt.TRAILER_SIZE :] = bad
    dst.write_bytes(bytes(data))
    with pytest.raises(PackedArchiveError, match="out of bounds"):
        PackedArchive(dst)


def _rewrite_footer(path, mutate):
    """Read a packed file's footer JSON, apply mutate(footer_dict), rewrite the
    footer + trailer in place (footer_offset is unchanged; shards untouched)."""
    from cellstream.packed import format as fmt

    data = bytearray(path.read_bytes())
    t = fmt.parse_trailer(bytes(data[-fmt.TRAILER_SIZE :]))
    foff = t["footer_offset"]
    footer = fmt.parse_footer(bytes(data[foff : foff + t["footer_length"]]))
    mutate(footer)
    fbuf = fmt.serialize_footer(
        footer["members"],
        shard_region_end=footer["shard_region_end"],
        alignment=footer.get("alignment", fmt.ALIGNMENT),
    )
    new = (
        bytes(data[:foff])
        + fbuf
        + fmt.pack_trailer(
            footer_offset=foff,
            footer_length=len(fbuf),
            shard_region_end=footer["shard_region_end"],
            footer_crc32=fmt.footer_crc32(fbuf),
        )
    )
    path.write_bytes(new)


def test_is_safe_member_name():
    from cellstream.packed.reader import _is_safe_member_name

    # Bare names stay valid.
    assert _is_safe_member_name("manifest.json")
    assert _is_safe_member_name("shard_0000.shx")
    # v4 multi-matrix: a two-segment ``<dir>/<file>`` family path is now valid
    # (each segment a safe bare name).
    assert _is_safe_member_name("x/shard_0000.shx")
    assert _is_safe_member_name("var/raw.parquet")
    # Zip-Slip / traversal vectors stay rejected, including cases that the
    # two-segment relaxation must NOT open up: 3+ segments, parent refs inside
    # a segment, absolute two-segment, and empty segments.
    for bad in [
        "../evil",
        "/etc/passwd",
        "..",
        ".",
        "",
        "a\\b",
        123,
        None,
        "a/b/c",
        "x/..",
        "../x/y",
        "a/../b",
        "/a/b",
        "a//b",
    ]:
        assert not _is_safe_member_name(bad)


def test_unsafe_member_name_rejected(grouped_dir, tmp_path):
    from cellstream.errors import PackedArchiveError
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    _rewrite_footer(dst, lambda f: f["members"][0].__setitem__("name", "../evil"))
    with pytest.raises(PackedArchiveError, match="unsafe member name"):
        PackedArchive(dst)


def test_out_of_bounds_member_rejected(grouped_dir, tmp_path):
    from cellstream.errors import PackedArchiveError
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    _rewrite_footer(dst, lambda f: f["members"][0].__setitem__("offset", 10**12))
    with pytest.raises(PackedArchiveError, match="out of bounds"):
        PackedArchive(dst)


def test_corrupt_tailbak_refuses_to_truncate(sample_adata, tmp_path):
    """A torn trailer + a corrupt .tailbak must NOT truncate the archive."""
    import struct

    from cellstream.errors import PackedArchiveError
    from cellstream.packed import format as fmt
    from cellstream.packed.reader import PackedArchive
    from cellstream.write import write_sharded

    f = tmp_path / "a.shardpack"
    write_sharded(sample_adata, f, format="v2", n_shards=2)
    data = bytearray(f.read_bytes())
    data[-fmt.TRAILER_SIZE :] = b"\x00" * fmt.TRAILER_SIZE  # invalidate the trailer
    f.write_bytes(bytes(data))
    size = f.stat().st_size
    # bogus backup: sre=0 (would truncate the archive to nothing if applied)
    f.with_suffix(f.suffix + ".tailbak").write_bytes(struct.pack("<Q", 0) + b"junk")
    with pytest.raises(PackedArchiveError, match="out of range or misaligned"):
        PackedArchive(f)
    assert f.stat().st_size == size  # archive was NOT truncated


def _rewrite_footer_raw(path, footer_dict, *, sre=None):
    """Write an arbitrary dict as the footer (bypassing serialize_footer) so
    malformed footers can be tested. Recomputes CRC + trailer."""
    import json

    from cellstream.packed import format as fmt

    data = bytearray(path.read_bytes())
    t = fmt.parse_trailer(bytes(data[-fmt.TRAILER_SIZE :]))
    foff = t["footer_offset"]
    fbuf = json.dumps(footer_dict, separators=(",", ":")).encode("utf-8")
    new = (
        bytes(data[:foff])
        + fbuf
        + fmt.pack_trailer(
            footer_offset=foff,
            footer_length=len(fbuf),
            shard_region_end=(t["shard_region_end"] if sre is None else sre),
            footer_crc32=fmt.footer_crc32(fbuf),
        )
    )
    path.write_bytes(new)


def test_footer_missing_members_rejected(grouped_dir, tmp_path):
    from cellstream.errors import PackedArchiveError
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    _rewrite_footer_raw(dst, {"container_version": 1, "alignment": 4096})  # no members/sre
    with pytest.raises(PackedArchiveError, match="missing or has an invalid"):
        PackedArchive(dst)


def test_shard_member_missing_idx_rejected(grouped_dir, tmp_path):
    from cellstream.errors import PackedArchiveError
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    _rewrite_footer(dst, lambda f: f["members"][0].pop("shard_idx", None))  # members[0] is a shard
    with pytest.raises(PackedArchiveError, match="shard_idx"):
        PackedArchive(dst)


# ---------------------------------------------------------------------------
# ultrareview L2 (packed manifest invariants) + L3 (footer cross-checks)
# ---------------------------------------------------------------------------


def _rewrite_member_bytes(path, name, new_bytes):
    """Overwrite a member's bytes in place, padding with spaces to its recorded
    length (the footer CRC covers only the footer, not member bytes, and members
    are not individually CRC'd — so offsets/footer/trailer stay valid)."""
    from cellstream.packed import format as fmt

    data = bytearray(path.read_bytes())
    t = fmt.parse_trailer(bytes(data[-fmt.TRAILER_SIZE :]))
    foff = t["footer_offset"]
    footer = fmt.parse_footer(bytes(data[foff : foff + t["footer_length"]]))
    m = next(mm for mm in footer["members"] if mm["name"] == name)
    off, length = m["offset"], m["length"]
    assert len(new_bytes) <= length, "test corruption must fit the member slot"
    data[off : off + length] = new_bytes + b" " * (length - len(new_bytes))
    path.write_bytes(bytes(data))


def test_packed_unsupported_schema_version_rejected(sample_adata, tmp_path):
    """The packed reader rejects an unsupported embedded manifest version."""
    import json

    from cellstream.errors import IncompatibleSchemaError
    from cellstream.packed.reader import PackedArchive
    from cellstream.write import write_sharded

    f = tmp_path / "a.shardpack"
    write_sharded(sample_adata, f, format="v2", n_shards=2)
    manifest = json.loads(PackedArchive(f).member_bytes("manifest.json").decode())
    assert manifest["schema_version"] == 2
    manifest["schema_version"] = 9
    _rewrite_member_bytes(f, "manifest.json", json.dumps(manifest).encode("utf-8"))

    with pytest.raises(IncompatibleSchemaError):
        PackedArchive(f)


def test_packed_corrupt_manifest_row_offsets_rejected(sample_adata, tmp_path):
    """A packed archive whose manifest violates row_offsets invariants fails loud
    (PackedArchive.__init__ runs validate_manifest)."""
    import json

    from cellstream.errors import IncompatibleSchemaError
    from cellstream.packed.reader import PackedArchive
    from cellstream.write import write_sharded

    f = tmp_path / "a.shardpack"
    write_sharded(sample_adata, f, format="v2", n_shards=2)
    # Original manifest -> corrupt the last offset so it != n_obs_total. Choose a
    # value of equal-or-shorter length so it fits the member slot (padded).
    orig = json.loads(PackedArchive(f).member_bytes("manifest.json").decode())
    corrupt = dict(orig)
    corrupt["row_offsets"] = list(orig["row_offsets"])
    corrupt["row_offsets"][-1] = 0  # "0" is shorter than the true last offset
    _rewrite_member_bytes(f, "manifest.json", json.dumps(corrupt).encode("utf-8"))
    with pytest.raises(IncompatibleSchemaError):
        PackedArchive(f)


def test_packed_duplicate_member_name_rejected(grouped_dir, tmp_path):
    from cellstream.errors import PackedArchiveError
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    # Give member[1] the same name as member[0] -> duplicate.
    _rewrite_footer(dst, lambda f: f["members"][1].__setitem__("name", f["members"][0]["name"]))
    with pytest.raises(PackedArchiveError, match="duplicate member name"):
        PackedArchive(dst)


def test_packed_shard_idx_gap_rejected(grouped_dir, tmp_path):
    """Footer shard set must be exactly {0..n_shards-1}; a duplicate idx (=> gap)
    is rejected by the members-vs-manifest cross-check."""
    from cellstream.errors import PackedArchiveError
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)

    def dup_idx(f):
        shards = [m for m in f["members"] if m.get("role") == "shard"]
        assert len(shards) >= 2, "fixture must have >=2 shards"
        shards[0]["shard_idx"] = shards[1]["shard_idx"]  # duplicate -> leaves a gap

    _rewrite_footer(dst, dup_idx)
    with pytest.raises(PackedArchiveError, match="shard"):
        PackedArchive(dst)


def test_packed_bool_shard_idx_rejected(sample_adata, tmp_path):
    """A crafted footer with boolean shard_idx (JSON true/false) must be rejected:
    bool is an int subclass, so [False, True] would otherwise pass both the
    isinstance(..., int) check and the [False, True] == [0, 1] cross-check."""
    from cellstream.errors import PackedArchiveError
    from cellstream.packed.reader import PackedArchive
    from cellstream.write import write_sharded

    f = tmp_path / "a.shardpack"
    write_sharded(sample_adata, f, format="v2", n_shards=2)

    def boolify(footer):
        shards = sorted(
            (m for m in footer["members"] if m.get("role") == "shard"),
            key=lambda m: m["shard_idx"],
        )
        assert len(shards) == 2
        shards[0]["shard_idx"] = False  # JSON false -> Python bool, == 0
        shards[1]["shard_idx"] = True  # JSON true  -> Python bool, == 1

    _rewrite_footer(f, boolify)
    with pytest.raises(PackedArchiveError):
        PackedArchive(f)


def test_shard_location_out_of_range_raises(grouped_dir, tmp_path):
    from cellstream.errors import PackedArchiveError
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    pa = PackedArchive(dst)
    n = pa.n_shards
    with pytest.raises(PackedArchiveError):
        pa.shard_location(-1)  # negative index must NOT wrap to the last shard
    with pytest.raises(PackedArchiveError):
        pa.shard_location(n)  # out of range


def test_header_adata_fallback_unlinks_tempfile_on_error(grouped_dir, tmp_path, monkeypatch):
    """L14: when header_adata falls back to the temp-file path and the chunked copy
    fails (truncated member), the NamedTemporaryFile(delete=False) must NOT leak."""
    import tempfile as _tempfile
    from pathlib import Path

    import anndata as _ad

    from cellstream.errors import PackedArchiveError
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    dst = tmp_path / "a.shad"
    pack(grouped_dir, dst)
    pa = PackedArchive(dst)

    # Force the temp-file fallback by making the mmap-window read raise.
    def _raise(*a, **k):
        raise OSError("forced mmap-window failure")

    monkeypatch.setattr(_ad, "read_h5ad", _raise)

    # Corrupt the in-memory header member length so the chunked copy hits EOF and
    # raises PackedArchiveError mid-copy (the leak window).
    hdr = dict(pa._by_name["header.h5ad"])
    hdr["length"] = hdr["length"] + (1 << 30)
    pa._by_name["header.h5ad"] = hdr

    created: list[str] = []
    real_ntf = _tempfile.NamedTemporaryFile

    def _spy_ntf(*a, **k):
        tf = real_ntf(*a, **k)
        created.append(tf.name)
        return tf

    monkeypatch.setattr(_tempfile, "NamedTemporaryFile", _spy_ntf)

    with pytest.raises(PackedArchiveError):
        pa.header_adata()
    assert created, "fallback temp file was not created — test setup broken"
    leaked = [n for n in created if Path(n).exists()]
    assert not leaked, f"temp file leaked on the error path: {leaked}"


def test_packed_bad_head_magic_rejected(tmp_path):
    """L15: a file without the SHPK head magic fails loud."""
    from cellstream.errors import PackedArchiveError
    from cellstream.packed.reader import PackedArchive

    f = tmp_path / "notpacked.shad"
    f.write_bytes(b"XXXX" + b"\x00" * 200)
    with pytest.raises(PackedArchiveError, match="bad head magic"):
        PackedArchive(f)


def test_packed_file_too_small_rejected(tmp_path):
    """L15: a file with the magic but smaller than the trailer fails loud."""
    from cellstream.errors import PackedArchiveError
    from cellstream.packed import format as fmt
    from cellstream.packed.reader import PackedArchive

    f = tmp_path / "tiny.shad"
    data = fmt.HEAD_MAGIC + b"\x00" * 8  # has magic, total 12 bytes < TRAILER_SIZE (64)
    assert len(data) < fmt.TRAILER_SIZE
    f.write_bytes(data)
    with pytest.raises(PackedArchiveError, match="too small"):
        PackedArchive(f)
