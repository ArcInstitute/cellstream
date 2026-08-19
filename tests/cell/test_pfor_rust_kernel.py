"""Byte-exact parity of the vendored FastPFor simdfastpfor256 kernel vs pyfastpfor."""

from __future__ import annotations

import numpy as np
import pytest

pyfastpfor = pytest.importorskip("pyfastpfor")
_rust = pytest.importorskip("cellstream.v2._rust")
pytestmark = pytest.mark.skipif(
    not hasattr(_rust, "_pfor_decode_words"), reason="Rust pfordelta kernel not built"
)


def _pyfastpfor_pack(arr_u32):
    from cellstream.cell.codec import _pfor_encode_capacity

    c = pyfastpfor.getCodec("simdfastpfor256")
    out = np.empty(_pfor_encode_capacity(arr_u32.size), dtype=np.uint32)
    wl = c.encodeArray(arr_u32, arr_u32.size, out, out.size)
    return np.ascontiguousarray(out[:wl])


@pytest.mark.parametrize("n", [1, 2, 63, 64, 65, 128, 200, 1000])
def test_rust_decode_matches_pyfastpfor(n):
    rng = np.random.default_rng(n)
    # include exception-triggering large values (PFOR stores high bits as exceptions)
    arr = rng.integers(0, 1_000_000, size=n, dtype=np.uint32)
    packed = _pyfastpfor_pack(arr)
    got = _rust._pfor_decode_words(packed, n)
    np.testing.assert_array_equal(got, arr)
    assert got.dtype == np.uint32


def test_rust_decode_rejects_corrupt_truncated_blob_cleanly():
    bad = np.array([0xDEADBEEF], dtype=np.uint32)
    with pytest.raises(ValueError, match="corrupt pfordelta frame"):
        _rust._pfor_decode_words(bad, 1000)


def test_rust_decode_rejects_empty_blob_for_non_empty_output_cleanly():
    bad = np.empty(0, dtype=np.uint32)
    with pytest.raises(ValueError, match="empty codec blob"):
        _rust._pfor_decode_words(bad, 5)


# SIMDFastPFor's BlockSize for the simdfastpfor256 codec.
_BLOCK = 256


def test_rust_decode_rejects_a_count_that_is_not_a_blocksize_multiple():
    """`CompositeCodec` only ever hands stage 1 a BlockSize-aligned prefix, so a header claiming
    otherwise would make the page loop advance `out` past what `__decodeArray` wrote."""
    with pytest.raises(ValueError, match="corrupt pfordelta frame"):
        _rust._pfor_decode_words(np.array([1, 0, 0, 0], dtype=np.uint32), _BLOCK)


@pytest.mark.parametrize("n", [_BLOCK, 1000, 100_000])
def test_rust_decode_rejects_poisoned_wheremeta_without_reading_oob(n):
    """#269 defect B: `wheremeta` is an unchecked displacement INTO THE INPUT.

    The stage-1 count MUST be a nonzero multiple of BlockSize or `wheremeta` is never examined
    and this vector tests nothing -- the checkpoint-2 review caught exactly that mistake in the
    first draft, where every vector started with a count of 1. Two mechanisms, not one: a
    non-multiple hits the divisibility throw, while zero passes that check (0 % 256 == 0) and
    instead leaves the page loop with nothing to do.

    Parametrised over n because the fault is input-side: pre-fix this SIGSEGVs regardless of the
    output size, so a fix that merely enlarged the OUTPUT buffer cannot pass.
    """
    poisoned = np.array([_BLOCK, 0x20000000, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
    with pytest.raises(ValueError, match="corrupt pfordelta frame"):
        _rust._pfor_decode_words(poisoned, n)


@pytest.mark.parametrize(
    "words",
    [
        [_BLOCK, 1, 0xFFFFFFFF, 0],  # exception byte block runs off the end
        [_BLOCK, 1, 0],  # metadata truncated before the bitmap
        [_BLOCK, 1, 0, 2, 0xFFFFFFFF],  # exception-stream esize enormous
        [_BLOCK, 0, 0, 0],  # wheremeta == 0
    ],
    ids=["bytesize", "truncated", "esize", "wheremeta-zero"],
)
def test_rust_decode_rejects_poisoned_metadata(words):
    """#269 defect B: every displacement taken from the blob must be bounds-checked.

    Each vector carries a BlockSize-aligned stage-1 count so it actually reaches
    `__decodeArray`; each was confirmed against the patched header to trip a *different* bound
    (byte block / bitmap / exception count / wheremeta), not the same one four times.
    """
    with pytest.raises(ValueError, match="corrupt pfordelta frame"):
        _rust._pfor_decode_words(np.array(words, dtype=np.uint32), _BLOCK)


def _valid_block_frame():
    """A well-formed single-block frame carrying real exceptions, plus its derived layout.

    Offsets are computed FROM the frame, never hardcoded, and the structural preconditions are
    asserted: if a future encoder (or `pyfastpfor` wheel) produces a differently shaped frame,
    these tests must fail loudly rather than quietly stop exercising the bounds they pin.
    """
    arr = np.random.default_rng(4).integers(0, 900, size=_BLOCK).astype(np.uint32)
    # A single block puts ALL its exceptions in the one stream indexed by `maxbits - b`, so
    # these force exception VALUES, not two different streams.
    arr[10] = 0xFFFFFFFF
    arr[77] = 5000
    packed = _pyfastpfor_pack(arr)
    wheremeta = int(packed[1])
    bytesize = int(packed[wheremeta + 1])
    bc = (wheremeta + 2) * 4  # byte offset of the per-block [b, cexcept, maxbits, pos...] bytes
    raw = packed.view(np.uint8)
    b, cexcept, maxbits = int(raw[bc]), int(raw[bc + 1]), int(raw[bc + 2])
    bitmap_idx = (wheremeta + 2) + (bytesize + 3) // 4
    bitmap = int(packed[bitmap_idx])
    diff = maxbits - b
    assert int(packed[0]) == _BLOCK, f"stage-1 count {packed[0]} is not one full block"
    assert 1 <= wheremeta < packed.size, wheremeta
    assert cexcept > 0, "frame has no exceptions; the maxbits/stream bounds would go untested"
    assert 0 < b < maxbits <= 32, (b, maxbits)
    # diff == 1 is encoded as a bare set-bit with NO exception stream, so the decoder takes a
    # branch that never consults `present[]` -- `bitmap-zero` below would then silently stop
    # testing the stream-presence bound.
    assert diff > 1, f"maxbits - b == {diff}; bitmap-zero would not reach the presence check"
    assert bitmap & (1 << (diff - 1)), f"stream {diff} absent from bitmap {bitmap:#x}"
    # single block => byte container is exactly [b, cexcept, maxbits] + cexcept position bytes
    assert bytesize == 3 + cexcept, (bytesize, cexcept)
    assert cexcept < 255, cexcept  # so setting it to 255 is genuinely an over-claim
    layout = {"bytesize_idx": wheremeta + 1, "bitmap_idx": bitmap_idx, "bc": bc, "diff": diff}
    return packed, arr, layout


def test_valid_block_frame_fixture_round_trips():
    """The mutation fixture must decode cleanly UNMUTATED, or the mutation tests below prove
    nothing -- they would pass on a frame that was already broken."""
    packed, arr, _ = _valid_block_frame()
    np.testing.assert_array_equal(_rust._pfor_decode_words(packed, _BLOCK), arr)


@pytest.mark.parametrize(
    "kind",
    [
        "wheremeta-huge",  # displacement past the input
        "bytesize-huge",  # exception byte block past the input
        "bytesize-small",  # bitmap lands inside the byte block -> bogus esize
        "bit-width-33",  # b > 32
        "packed-block-overrun",  # legal b, but the packed walk runs past packedend
        "cexcept-255",  # exception bytes exhausted
        "maxbits-below-b",  # `maxbits - b` would index unpackpointers[] out of range
        "bitmap-zero",  # the block names an exception stream the page does not carry
        "truncated",  # metadata cut short
    ],
)
def test_rust_decode_rejects_structurally_mutated_frame(kind):
    """#269 defect B, reached through the paths a synthetic 4-word vector cannot: `b`, `maxbits`,
    exception-stream presence, exception-byte supply and packed-block length are all read from a
    per-block byte container that only exists once a frame has real blocks in it.

    Measured against a pre-fix rebuild -- `rust/src/pfor.rs`, `rust/src/cell.rs` and the vendored
    `simdfastpfor.h` restored to 20768f3d, extension rebuilt, one test per process. Seven of the
    nine killed the interpreter, and those seven are the evidence for the new bounds::

        wheremeta-huge 139   bytesize-huge 139       bytesize-small 139   bitmap-zero 139
        maxbits-below-b 139  packed-block-overrun 139  truncated 134
        (139 = SIGSEGV, 134 = SIGABRT)

    The two that did NOT crash are ``bit-width-33`` and ``cexcept-255``, and they differ:

    - ``bit-width-33`` already raised pre-fix, via the pre-existing `got != expected` count
      check. It pins the public contract ("reject, do not return wrong data") rather than
      proving a new bound -- do not cite it as evidence for the bounds themselves.
    - ``cexcept-255`` pre-fix DID NOT RAISE AT ALL. Its out-of-bounds read stayed inside the
      heap, decode still reported the expected count, and the frame was accepted: silent data
      corruption, strictly worse than a crash and invisible to any exit-code test. It is the
      case that justifies the sanitizer gate -- see `rust/asan/run.sh`.
    """
    packed, _, layout = _valid_block_frame()
    m = packed.copy()
    raw = m.view(np.uint8)
    bc = layout["bc"]
    if kind == "wheremeta-huge":
        m[1] = 0x20000000
    elif kind == "bytesize-huge":
        m[layout["bytesize_idx"]] = 0xFFFFFFFF
    elif kind == "bytesize-small":
        m[layout["bytesize_idx"]] = 3
    elif kind == "bit-width-33":
        raw[bc] = 33
    elif kind == "packed-block-overrun":
        raw[bc] = 32
    elif kind == "cexcept-255":
        raw[bc + 1] = 255
    elif kind == "maxbits-below-b":
        raw[bc + 2] = 1
    elif kind == "bitmap-zero":
        m[layout["bitmap_idx"]] = 0
    elif kind == "truncated":
        m = np.ascontiguousarray(m[:-1])
    else:  # pragma: no cover - guards against a typo in the parametrisation
        raise AssertionError(kind)
    with pytest.raises(ValueError, match="corrupt pfordelta frame"):
        _rust._pfor_decode_words(m, _BLOCK)


def test_rust_decode_still_handles_a_single_high_exception():
    """Regression guard for the FALSE-REJECT risk: 255 zeroes + one UINT32_MAX produces a
    one-element, 32-bit exception stream — exactly the shape the first draft's
    `1 + ceil(size/32)*bit` accounting would have rejected. Correct is `1 + ceil(size*bit/32)`."""
    arr = np.zeros(256, dtype=np.uint32)
    arr[100] = 0xFFFFFFFF
    packed = _pyfastpfor_pack(arr)
    np.testing.assert_array_equal(_rust._pfor_decode_words(packed, 256), arr)


def test_rust_decode_still_handles_a_multi_page_frame():
    """PageSize is 65536; a frame above it exercises the page loop, whose per-page metadata is
    now bounds-checked. Must round-trip unchanged."""
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 1_000_000, size=70_000, dtype=np.uint32)
    packed = _pyfastpfor_pack(arr)
    np.testing.assert_array_equal(_rust._pfor_decode_words(packed, 70_000), arr)
