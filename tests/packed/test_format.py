import pytest

from cellstream.errors import PackedArchiveError
from cellstream.packed import format as fmt


def test_align_up():
    assert fmt.align_up(0) == 0
    assert fmt.align_up(1) == 4096
    assert fmt.align_up(4096) == 4096
    assert fmt.align_up(4097) == 8192


def test_trailer_roundtrip():
    raw = fmt.pack_trailer(
        footer_offset=8192, footer_length=123, shard_region_end=4096, footer_crc32=0xDEADBEEF
    )
    assert len(raw) == fmt.TRAILER_SIZE
    t = fmt.parse_trailer(raw)
    assert t["footer_offset"] == 8192
    assert t["footer_length"] == 123
    assert t["shard_region_end"] == 4096
    assert t["footer_crc32"] == 0xDEADBEEF
    assert t["format_version"] == fmt.CONTAINER_VERSION


def test_trailer_bad_magic_raises():
    bad = b"\x00" * fmt.TRAILER_SIZE
    with pytest.raises(PackedArchiveError):
        fmt.parse_trailer(bad)


def test_footer_roundtrip_and_crc():
    members = [
        {"name": "shard_0000.shx", "role": "shard", "shard_idx": 0, "offset": 4096, "length": 10},
        {"name": "header.h5ad", "role": "header", "offset": 8192, "length": 20},
    ]
    buf = fmt.serialize_footer(members, shard_region_end=8192)
    crc = fmt.footer_crc32(buf)
    parsed = fmt.parse_footer(buf)
    assert parsed["members"] == members
    assert parsed["shard_region_end"] == 8192
    assert parsed["alignment"] == fmt.ALIGNMENT
    assert isinstance(crc, int)
