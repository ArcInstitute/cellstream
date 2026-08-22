"""#305: a packed member's recorded length must be the bytes copied, not a path ``stat()``.

``_write_member`` took ``length = src_path.stat().st_size`` and wrote it into the footer entry,
while ``shutil.copyfileobj`` copied the whole file regardless (its ``length=`` argument is a buffer
size, not a byte limit). Under the WekaFS stale-stat fault that #305 turned out to be, that records
a member SHORTER than the bytes actually present -- a packed archive that reports a successful
write and is wrong on disk. This is the shard AND cell layouts' shared packer.

See ``tests/_stale_stat`` for the measurements and ``tests/test_stale_stat_harness.py`` for the
negative control.
"""

from __future__ import annotations

import pathlib
import struct

import pandas as pd

from cellstream.manifest import SUPPORTED_SCHEMA_VERSIONS
from cellstream.packed import format as fmt
from cellstream.packed.reader import PackedArchive, _apply_recovery, _trailer_valid
from cellstream.packed.reader import PackedArchive as _PA
from cellstream.packed.writer import _write_member, update_obs_packed
from tests._archive_helpers import write_archive
from tests._stale_stat import short_stat

PAYLOAD = b"".join(bytes([i % 251]) for i in range(500))


def test_member_length_is_the_bytes_copied_not_a_path_stat(tmp_path):
    src = tmp_path / "x_offsets"
    src.write_bytes(PAYLOAD)
    dst = tmp_path / "packed.bin"
    with dst.open("wb") as out:
        out.write(b"H" * 10)  # unaligned head, so the padding branch runs too
        with short_stat("x_offsets"):
            offset, length = _write_member(out, src, delete_after=False)

    assert length == len(PAYLOAD)
    raw = dst.read_bytes()
    # The recorded (offset, length) must address the whole member, not a truncation of it.
    assert raw[offset : offset + length] == PAYLOAD
    assert offset % fmt.ALIGNMENT == 0


def test_member_length_is_unchanged_without_the_fault(tmp_path):
    """Control: the same call with no injection, so a fix that broke the ordinary path -- an
    off-by-one in the new length, say -- cannot hide behind the injected test alone."""
    src = tmp_path / "x_offsets"
    src.write_bytes(PAYLOAD)
    dst = tmp_path / "packed.bin"
    with dst.open("wb") as out:
        out.write(b"H" * 10)
        offset, length = _write_member(out, src, delete_after=False)

    assert length == len(PAYLOAD)
    assert dst.read_bytes()[offset : offset + length] == PAYLOAD


def test_member_length_of_an_empty_member(tmp_path):
    """An n_obs=0 archive packs a zero-byte ``x_payload``; the length must stay 0 and the offset
    must still be the aligned position, not the pre-padding one."""
    src = tmp_path / "x_payload"
    src.write_bytes(b"")
    dst = tmp_path / "packed.bin"
    with dst.open("wb") as out:
        out.write(b"H" * 10)
        with short_stat("x_payload"):
            offset, length = _write_member(out, src, delete_after=False)

    assert length == 0
    assert offset % fmt.ALIGNMENT == 0


def test_trailer_stays_valid_under_a_stale_short_stat(tmp_path, sample_adata):
    """``_trailer_valid`` seeked to ``size - TRAILER_SIZE`` with ``size`` from a path stat, so a
    stale-short value made it parse the wrong bytes and report a perfectly good archive as invalid.
    Reachable by the most ordinary pattern there is: write an archive, then open it.
    """
    out = tmp_path / "a.shad"
    write_archive(sample_adata, out, n_shards=2)
    assert _trailer_valid(out) is True  # control: valid without the fault
    with short_stat("a.shad"):
        assert _trailer_valid(out) is True
        # And the whole read path, not just the predicate.
        assert PackedArchive(out).manifest["schema_version"] in SUPPORTED_SCHEMA_VERSIONS


def test_trailer_is_still_rejected_when_it_really_is_short(tmp_path):
    """The negative half: the guard must keep rejecting a file that is genuinely too small to hold
    a trailer, or the fix would have replaced a false negative with a false positive."""
    tiny = tmp_path / "tiny.shad"
    tiny.write_bytes(b"SHPK" + b"\x00" * 8)
    assert _trailer_valid(tiny) is False


def test_tail_backup_is_complete_under_a_stale_short_stat(tmp_path, sample_adata, monkeypatch):
    """``_update_metadata_packed`` backs up the old tail before rewriting it, and sized that read
    with a path stat. A stale-short size backs up a TRUNCATED tail, so a crash before the rewrite
    completed would recover the archive to a short one.

    The archive itself comes out correct either way, so asserting on the archive would be vacuous.
    This keeps the ``.tailbak`` instead (its unlink is the last step on the success path) and
    measures it: 8 bytes of ``shard_region_end`` plus every byte from ``sre`` to EOF.
    """
    out = tmp_path / "u.shad"
    write_archive(sample_adata, out, n_shards=2)
    real_size = out.stat().st_size
    sre = _PA(out).footer["shard_region_end"]

    kept = {}
    real_unlink = pathlib.Path.unlink

    def keep_tailbak(self, *a, **kw):
        if self.name.endswith(".tailbak"):
            kept["bytes"] = self.read_bytes()
            kept["path"] = self
            return
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(pathlib.Path, "unlink", keep_tailbak)
    with short_stat("u.shad"):
        update_obs_packed(out, pd.DataFrame(index=sample_adata.obs_names))

    raw = kept["bytes"]
    assert struct.unpack("<Q", raw[:8])[0] == sre
    assert len(raw) - 8 == real_size - sre, "the backed-up tail is short"


def test_recovery_bound_accepts_a_valid_sre_under_a_stale_short_stat(tmp_path, sample_adata):
    """``_apply_recovery`` bounds the backed-up ``shard_region_end`` with ``sre <= size``, sized by
    a path stat. A stale-short size puts a VALID sre out of range and refuses a recovery that
    should have succeeded, leaving the archive stranded behind a manual-inspection error.

    ``missing`` is large here on purpose: the injected shortfall has to reach past ``sre``, which
    sits well before EOF, or the bound would still hold and the test would prove nothing.
    """
    out = tmp_path / "r.shad"
    write_archive(sample_adata, out, n_shards=2)
    size = out.stat().st_size
    sre = _PA(out).footer["shard_region_end"]
    tail = out.read_bytes()[sre:]

    # A valid .tailbak plus a trailer that no longer validates -> the roll-back branch.
    (tmp_path / "r.shad.tailbak").write_bytes(struct.pack("<Q", sre) + tail)
    with out.open("r+b") as fh:
        fh.seek(size - fmt.TRAILER_SIZE)
        fh.write(b"\x00" * fmt.TRAILER_SIZE)
    assert _trailer_valid(out) is False

    with short_stat("r.shad", missing=size - sre + 1):
        _apply_recovery(out)

    assert _trailer_valid(out) is True
    assert not (tmp_path / "r.shad.tailbak").exists()
    assert _PA(out).footer["shard_region_end"] == sre
