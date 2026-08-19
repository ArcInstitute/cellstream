"""PR4 (#219): reorder_frames -- the codec-agnostic byte permutation of encoded frames.

It NEVER parses a frame, so it cannot be wrong about a codec, an on-disk dtype, or a lossy cast.
These tests treat frames as OPAQUE BYTE STRINGS, which is exactly the contract. They run on both
engines (the pure-Python fallback is a byte permutation too, so there is nothing to differ on).
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstream.cell._rust_dispatch import reorder_frames

FRAMES = [b"aaa", b"bbbb", b"c", b"", b"dddddd", b"ee"]


def _write_frames(path, frames):
    with open(path, "wb") as f:
        for fr in frames:
            f.write(fr)
    lens = [len(fr) for fr in frames]
    return np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


def _read_frames(path, offsets):
    raw = path.read_bytes()
    return [raw[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


@pytest.mark.parametrize(
    "perm",
    [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [3, 0, 5, 1, 4, 2]],
    ids=["identity", "reverse", "arbitrary"],
)
def test_reorder_permutes_frames(tmp_path, perm):
    stg, out = tmp_path / "s.bin", tmp_path / "o.bin"
    in_off = _write_frames(stg, FRAMES)
    perm = np.asarray(perm, dtype=np.int64)
    out_off = reorder_frames(stg, out, in_off, perm)
    assert _read_frames(out, out_off) == [FRAMES[i] for i in perm]
    # offsets stay cumulative + contiguous -- the format's core invariant.
    # Compare adjacent elements rather than np.diff: see _reorder_frames_python's guard for why
    # (np.diff WRAPS on adversarial int64 and would silently accept a non-monotonic array).
    assert out_off[0] == 0
    assert out_off[-1] == sum(len(f) for f in FRAMES)
    assert np.all(out_off[1:] >= out_off[:-1])


def test_reorder_single_row(tmp_path):
    stg, out = tmp_path / "s.bin", tmp_path / "o.bin"
    in_off = _write_frames(stg, [b"zzz"])
    out_off = reorder_frames(stg, out, in_off, np.array([0], dtype=np.int64))
    assert _read_frames(out, out_off) == [b"zzz"]


def test_reorder_empty(tmp_path):
    stg, out = tmp_path / "s.bin", tmp_path / "o.bin"
    in_off = _write_frames(stg, [])
    out_off = reorder_frames(stg, out, in_off, np.array([], dtype=np.int64))
    assert out.read_bytes() == b""
    np.testing.assert_array_equal(out_off, [0])


@pytest.mark.parametrize("perm", [[0, 1, 9], [0, 1, -1]], ids=["too_big", "negative"])
def test_reorder_rejects_bad_perm(tmp_path, perm):
    stg, out = tmp_path / "s.bin", tmp_path / "o.bin"
    in_off = _write_frames(stg, FRAMES[:3])
    with pytest.raises(ValueError, match="out of range"):
        reorder_frames(stg, out, in_off, np.asarray(perm, dtype=np.int64))


def test_reorder_rejects_offsets_past_end_of_file(tmp_path):
    """A truncated / mismatched staging file must fail LOUD, never read OOB."""
    stg, out = tmp_path / "s.bin", tmp_path / "o.bin"
    _write_frames(stg, FRAMES[:2])
    bad = np.array([0, 3, 999], dtype=np.int64)  # claims more bytes than the file holds
    with pytest.raises(ValueError, match="staging"):
        reorder_frames(stg, out, bad, np.array([0, 1], dtype=np.int64))


def test_reorder_rejects_non_monotonic_offsets(tmp_path):
    stg, out = tmp_path / "s.bin", tmp_path / "o.bin"
    _write_frames(stg, FRAMES[:3])
    bad = np.array([0, 5, 3, 8], dtype=np.int64)
    with pytest.raises(ValueError, match="non-decreasing"):
        reorder_frames(stg, out, bad, np.array([0, 1, 2], dtype=np.int64))


def test_reorder_rejects_nonzero_first_offset(tmp_path):
    stg, out = tmp_path / "s.bin", tmp_path / "o.bin"
    _write_frames(stg, FRAMES[:2])
    bad = np.array([1, 4, 7], dtype=np.int64)
    with pytest.raises(ValueError, match=r"in_offsets\[0\]"):
        reorder_frames(stg, out, bad, np.array([0, 1], dtype=np.int64))


def test_reorder_all_empty_frames(tmp_path):
    """Every frame zero-length (e.g. a streamed block where every cell has nnz=0 --
    decode_rows_into's `if nnz == 0 { continue }` means the encoder legitimately emits these).
    The staging file is 0 bytes, so a naive Python mmap-only implementation has `mm is None`
    and crashes subscripting it; Rust already skips zero-length copies (#219 Task 5 review
    finding 1). Both engines must return the all-zero out_offsets and write an empty file."""
    stg, out = tmp_path / "s.bin", tmp_path / "o.bin"
    in_off = _write_frames(stg, [b"", b"", b""])
    perm = np.array([2, 0, 1], dtype=np.int64)
    out_off = reorder_frames(stg, out, in_off, perm)
    np.testing.assert_array_equal(out_off, [0, 0, 0, 0])
    assert out.read_bytes() == b""


def test_reorder_rejects_perm_too_short(tmp_path):
    """A perm shorter than n_src silently truncates the output (frames dropped) unless both
    engines validate the length up front (#219 Task 5 review finding 2)."""
    stg, out = tmp_path / "s.bin", tmp_path / "o.bin"
    in_off = _write_frames(stg, FRAMES[:3])
    with pytest.raises(ValueError, match="perm length"):
        reorder_frames(stg, out, in_off, np.array([0, 1], dtype=np.int64))


def test_reorder_rejects_perm_too_long(tmp_path):
    """A perm longer than n_src, with repeated but individually-valid indices, silently
    duplicates frames unless both engines validate the length up front (#219 Task 5 review
    finding 2). Every entry is in-range, so only a length check (not the out-of-range guard)
    can catch this."""
    stg, out = tmp_path / "s.bin", tmp_path / "o.bin"
    in_off = _write_frames(stg, FRAMES[:3])
    with pytest.raises(ValueError, match="perm length"):
        reorder_frames(stg, out, in_off, np.array([0, 1, 2, 0], dtype=np.int64))
