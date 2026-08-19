"""layout=cell error handling (spec §11)."""

from __future__ import annotations

import gc

import numpy as np
import pytest

import cellstream
from cellstream.cell.writer import _has_non_uint_value, write_cell_archive
from tests.cell.conftest import make_cell_adata


def test_verify_payload_true_on_fresh_archive(tmp_path):
    # default write stores NO checksum -> verify_payload is True ("nothing to verify")
    a = make_cell_adata()
    out = tmp_path / "u.shad"
    write_cell_archive(a, out)
    store = cellstream.open(out)
    assert store.has_payload_checksum is False
    assert store.verify_payload() is True


def test_verify_payload_matches_when_enabled(tmp_path):
    a = make_cell_adata()
    out = tmp_path / "c.shad"
    write_cell_archive(a, out, payload_checksum=True)
    store = cellstream.open(out)
    assert store.has_payload_checksum is True
    assert store.verify_payload() is True


def test_verify_payload_detects_corruption(tmp_path):
    from cellstream.packed.reader import PackedArchive

    a = make_cell_adata()
    out = tmp_path / "c.shad"
    write_cell_archive(a, out, payload_checksum=True)
    # flip a byte inside the payload member (precise offset via member_location)
    _, base, length = PackedArchive(out).member_location("x_payload")
    data = bytearray(out.read_bytes())
    data[base + length // 2] ^= 0xFF
    out.write_bytes(bytes(data))
    assert cellstream.open(out).verify_payload() is False


def test_write_value_over_uint32_raises(tmp_path):
    a = make_cell_adata()
    a.X = a.X.tocsr()
    a.X.data = a.X.data.astype(np.int64)
    a.X.data[0] = 2**32 + 5  # exceeds uint32 -> not integer-routable to unsigned store
    with pytest.raises(ValueError):
        write_cell_archive(a, tmp_path / "x.shad")


def test_corrupt_frame_raises_on_gather(tmp_path):
    from cellstream.cell.codec import PforDeltaCodec

    dec = PforDeltaCodec("uint16", "uint16").new_decoder(70000)
    with pytest.raises(ValueError, match="corrupt pfordelta frame"):
        dec.decode(b"\x08\x00\x00\x00", 3)  # idx_len=8 exceeds a 4-byte frame body


@pytest.mark.parametrize("disable_rust", [False, True])
def test_streaming_write_from_corrupt_source_raises_valueerror(tmp_path, monkeypatch, disable_rust):
    """C1 (PR4 final review, Critical): a corrupt source archive hit mid-stream through
    write_cell_archive used to surface as BufferError on the Rust engine instead of the real
    ValueError -- gather_rows' `payload` frombuffer view stayed alive (pinned by the in-flight
    traceback) past CellStore.close(), whose unconditional `finally: self._pf.close()` then
    closed the backing fd out from under a still-exported mmap (heap corruption -> SIGABRT on a
    later gc.collect(), in the reviewer's repro). Both engines must raise a clean ValueError."""
    if disable_rust:
        monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")

    a = make_cell_adata()
    src_path = tmp_path / "src.shad"
    write_cell_archive(a, src_path)

    from cellstream.packed.reader import PackedArchive

    _, base, _length = PackedArchive(src_path).member_location("x_payload")
    # Byte 3 of the payload is the MSB of frame 0's little-endian idx_len length-prefix (no
    # group_by -> identity permutation -> row 0 is written first -> offsets[0] == 0). Flipping
    # it blows idx_len up by a multiple of 2**24, guaranteeing "idx blob length exceeds frame"
    # regardless of how small the real frame is.
    with open(src_path, "r+b") as f:
        f.seek(base + 3)
        b = f.read(1)
        f.seek(base + 3)
        f.write(bytes([b[0] ^ 0xFF]))

    out_path = tmp_path / "out.shad"
    with pytest.raises(ValueError, match="corrupt") as exc_info:
        write_cell_archive(src_path, out_path)
    assert not isinstance(exc_info.value, BufferError)

    # The process must still be healthy: pre-fix, this is where the reviewer's repro SIGABRTs.
    gc.collect()
    reopened = cellstream.open(src_path)
    assert reopened.n_obs == a.n_obs
    reopened.close()


def test_open_non_cell_archive_message(tmp_path):
    from cellstream.write import write_sharded

    a = make_cell_adata()
    p = tmp_path / "shard.shad"
    write_sharded(a, p, format="v2", n_shards=2)
    with pytest.raises(Exception, match="cell"):
        cellstream.open(p)


def test_rust_raises_distinct_non_integer_counts_type(tmp_path):
    """The fallback in write_cell_archive keys off this TYPE, not a message substring.
    For a FLOAT source, a value > u32::MAX IS this type: the value is a finite non-negative
    integer that simply doesn't fit uint32, and the float fallback stores it exactly, so it
    must be retryable. (The INT path is the contrast -- see
    test_rust_int64_over_uint32_is_value_error_not_non_integer_counts. #217 lossy-policy test.)

    The _rust_dispatch.encode_cells wrapper translates the raw rust.NonIntegerCounts into
    cellstream.errors.NonIntegerCountsError (the engine-independent type -- see
    test_non_integer_counts_error_is_engine_independent), so this test asserts the
    Rust-level type via __cause__ rather than expecting it to escape the wrapper directly."""
    rust = pytest.importorskip("cellstream.v2._rust")
    assert issubclass(rust.NonIntegerCounts, ValueError)

    import numpy as np

    from cellstream.cell._rust_dispatch import encode_cells, rust_available
    from cellstream.errors import NonIntegerCountsError

    if not rust_available():
        pytest.skip("Rust unavailable")

    # encode_cells_typed's per-worker encode_block creates a sibling `<out>.part{N}`
    # staging file (pfor.rs:677) before any value is validated, so the output path
    # must be a real writable location -- /dev/null.part0 would fail with an unrelated
    # PermissionError (/dev is not writable) before ever reaching count_u32().
    out = str(tmp_path / "payload.bin")

    # one fractional value -> NonIntegerCounts, translated to NonIntegerCountsError
    data = np.array([1.0, 2.5], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    indptr = np.array([0, 2], dtype=np.int32)
    perm = np.array([0], dtype=np.int64)
    with pytest.raises(NonIntegerCountsError) as ei_nic:
        encode_cells(
            data,
            indices,
            indptr,
            perm,
            out,
            n_threads=1,
            compute_checksum=False,
            check_duplicates=False,
        )
    assert isinstance(ei_nic.value.__cause__, rust.NonIntegerCounts)

    # a FLOAT value > u32::MAX is ALSO NonIntegerCounts: integer storage is impossible, and the
    # float fallback holds it exactly, so it must be retryable (contrast the int path below).
    big = np.array([1.0, 5e9], dtype=np.float64)
    with pytest.raises(NonIntegerCountsError) as ei_big:
        encode_cells(
            big,
            indices,
            indptr,
            perm,
            out,
            n_threads=1,
            compute_checksum=False,
            check_duplicates=False,
        )
    assert isinstance(ei_big.value.__cause__, rust.NonIntegerCounts)
    assert "exceeds the uint32 maximum" in str(ei_big.value)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_rust_negative_int_raises_non_integer_counts_type(tmp_path, dtype):
    """count_u32_int! (i32/i64) must ALSO map a negative value to NonIntegerCounts.
    test_rust_raises_distinct_non_integer_counts_type above only exercises the FLOAT arm
    (count_u32_float!) -- encode_cells routes real int32/int64 source dtypes too (writer.py's
    ("int64","int32")/("int64","int64")/("int32","int32")/("int32","int64") combos), so the
    int arm needs its own regression test for this type distinction. (#216 review.)

    Like test_rust_raises_distinct_non_integer_counts_type above, the _rust_dispatch wrapper
    translates rust.NonIntegerCounts into cellstream.errors.NonIntegerCountsError, so the
    Rust-level type is checked via __cause__."""
    rust = pytest.importorskip("cellstream.v2._rust")
    from cellstream.cell._rust_dispatch import encode_cells, rust_available
    from cellstream.errors import NonIntegerCountsError

    if not rust_available():
        pytest.skip("Rust unavailable")

    # see test_rust_raises_distinct_non_integer_counts_type above for why tmp_path, not /dev/null
    out = str(tmp_path / "payload.bin")
    data = np.array([1, -2], dtype=dtype)
    indices = np.array([0, 1], dtype=np.int32)
    indptr = np.array([0, 2], dtype=np.int32)
    perm = np.array([0], dtype=np.int64)
    with pytest.raises(NonIntegerCountsError) as ei_nic:
        encode_cells(
            data,
            indices,
            indptr,
            perm,
            out,
            n_threads=1,
            compute_checksum=False,
            check_duplicates=False,
        )
    assert isinstance(ei_nic.value.__cause__, rust.NonIntegerCounts)


def test_non_integer_counts_error_is_engine_independent():
    """Rust and the pure-Python encoder MUST raise the same type -- the retry in
    write_cell_archive is engine-independent. It subclasses ValueError so every existing
    pytest.raises(ValueError) stays green."""
    from cellstream.errors import CellstreamError, NonIntegerCountsError

    assert issubclass(NonIntegerCountsError, ValueError)
    assert issubclass(NonIntegerCountsError, CellstreamError)


def test_rust_int64_over_uint32_is_value_error_not_non_integer_counts(tmp_path):
    """count_u32_int!'s > u32::MAX arm must stay a plain ValueError, NOT NonIntegerCounts.
    test_write_value_over_uint32_raises above only asserts pytest.raises(ValueError), which
    would still PASS even if this arm were accidentally flipped to NonIntegerCounts (it IS a
    ValueError) -- so that test does not actually guard the type distinction for the int path.
    A later retry fires on exactly the NonIntegerCounts type: zstd-int cannot store an
    out-of-range value either, so silently retrying would hide a genuine overflow bug.
    (#216 review.)"""
    rust = pytest.importorskip("cellstream.v2._rust")
    from cellstream.cell._rust_dispatch import encode_cells, rust_available

    if not rust_available():
        pytest.skip("Rust unavailable")

    out = str(tmp_path / "payload.bin")
    data = np.array([1, 2**32 + 5], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int32)
    indptr = np.array([0, 2], dtype=np.int32)
    perm = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError) as ei:
        encode_cells(
            data,
            indices,
            indptr,
            perm,
            out,
            n_threads=1,
            compute_checksum=False,
            check_duplicates=False,
        )
    assert not isinstance(ei.value, rust.NonIntegerCounts)
    assert "exceeds the uint32 maximum" in str(ei.value)


@pytest.mark.parametrize("codec", ["pfordelta", "zstd"])
def test_python_encoder_raises_retryable_type(tmp_path, monkeypatch, codec):
    """CELLSTREAM_DISABLE_RUST=1 must raise the SAME type Rust does, or the retry only works
    on one engine.

    Uses an int64 matrix with a negative value, not a fractional float value: under the
    current attempt-list design (#216 review) there is no up-front scan, and FLOAT input
    would not reliably surface NonIntegerCountsError to this test for either codec.
    codec='pfordelta' on float input DOES reach _encode_cell_payload's Python-fallback arm
    and raises NonIntegerCountsError there -- but write_cell_archive's last-attempt
    handling then converts it to a plain ValueError ("requires integer counts...")
    precisely because the dtype is float (see
    test_explicit_pfordelta_on_fractional_still_raises), so
    pytest.raises(NonIntegerCountsError) would not see it. codec='zstd' on float input
    with no explicit data_dtype gets a SECOND attempt (float storage) from
    _plan_cell_codec, and a merely negative/fractional value is perfectly valid there
    (that branch never calls _has_non_uint_value) -- so the write would silently SUCCEED
    on retry instead of raising. A negative INT64 value sidesteps both: _plan_cell_codec
    gives non-float input exactly one attempt for either codec (no fallback to retry into,
    and the float-only ValueError conversion above never triggers since
    X.data.dtype.kind != 'f'), so _has_non_uint_value's negative-value check raises
    NonIntegerCountsError and it propagates straight out of write_cell_archive for both
    codec='pfordelta' and codec='zstd' (confirmed via _plan_cell_codec directly: negative
    int64 -> [('pfordelta', None)] and [('zstd', 'uint32')], i.e. a single attempt into
    the arm, for both codecs)."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    from cellstream.cell.writer import write_cell_archive
    from cellstream.errors import NonIntegerCountsError

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    X = sp.csr_matrix(np.array([[1, -2]], dtype=np.int64))
    a = ad.AnnData(X=X)
    a.var_names = ["g0", "g1"]
    with pytest.raises(NonIntegerCountsError):
        write_cell_archive(a, tmp_path / "x.shad", codec=codec, overwrite=True)


@pytest.mark.parametrize("codec", ["pfordelta", "zstd"])
def test_int_uint32_overflow_is_a_hard_error(tmp_path, codec):
    """An INT source value > u32::MAX stays a plain ValueError (NOT the retryable
    NonIntegerCountsError), for both codecs. This is THE surviving safety property of the
    overflow guard: casting an int64 > 2^53 to float is LOSSY, so an out-of-range integer must
    fail loud, never silently become a float archive. (The float source is the opposite case --
    see test_float_uint32_overflow_falls_back_to_float. #217 lossy-policy test.)

    Covers both codecs' independent ``> 0xFFFFFFFF`` guards inside _encode_cell_payload
    (pfordelta's and zstd's). codec='zstd' never dispatches to Rust ("zstd (Python-only in
    PR1)"), so this is the only test reaching that arm; the two messages are textually distinct
    ("stores counts as uint32" vs "codec='zstd' stores integer counts as uint32"), so each
    assertion hits its own codec's guard. Anchor on the function + the "exceeds the uint32
    maximum" message; line numbers drift."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    from cellstream.cell.writer import write_cell_archive
    from cellstream.errors import NonIntegerCountsError

    X = sp.csr_matrix(np.array([[1, 2**32 + 5]], dtype=np.int64))
    a = ad.AnnData(X=X)
    a.var_names = ["g0", "g1"]
    with pytest.raises(ValueError) as ei:
        write_cell_archive(a, tmp_path / "x.shad", codec=codec, overwrite=True)
    assert not isinstance(ei.value, NonIntegerCountsError)
    assert "exceeds the uint32 maximum" in str(ei.value)


@pytest.mark.parametrize("engine_env", [None, "1"])
def test_float_uint32_overflow_falls_back_to_float(tmp_path, monkeypatch, engine_env):
    """A FLOAT source value > u32::MAX means integer storage is impossible, and the float
    fallback stores it EXACTLY -- so codec=None must WARN and fall back to float storage, not
    hard-fail. This is the fix for #217's test_rust_zstd_lossy_policy: a log1p archive
    (fractional) that also held one large value hard-failed because the optimistic encoder could
    hit the large value before a fractional one (thread-order-dependent, so nondeterministic).
    Runs on BOTH engines."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    from cellstream.cell.reader import open_cell
    from cellstream.cell.writer import write_cell_archive

    if engine_env:
        monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", engine_env)

    # integral, but 5e9 overflows uint32; float64 stores it exactly (no loss)
    X = sp.csr_matrix(np.array([[1.0, 5e9]], dtype=np.float64))
    a = ad.AnnData(X=X)
    a.var_names = ["g0", "g1"]
    out = tmp_path / "x.shad"
    with pytest.warns(UserWarning, match="re-encoding with codec='zstd'"):
        write_cell_archive(a, out, overwrite=True)  # codec=None -> optimistic pfordelta, falls back
    s = open_cell(str(out))
    try:
        assert s.manifest["codec"] == "zstd"
        assert s.manifest["value_dtype_on_disk"] == "float64"
    finally:
        s.close()


def test_explicit_pfordelta_on_float_overflow_still_raises(tmp_path):
    """codec='pfordelta' explicit on a float source that overflows uint32 gets NO fallback --
    the user chose the integer codec, so it fails loud (with the actionable pfordelta message)
    rather than silently switching to float storage."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    from cellstream.cell.writer import write_cell_archive

    X = sp.csr_matrix(np.array([[1.0, 5e9]], dtype=np.float64))
    a = ad.AnnData(X=X)
    a.var_names = ["g0", "g1"]
    with pytest.raises(ValueError):
        write_cell_archive(a, tmp_path / "x.shad", codec="pfordelta", overwrite=True)


@pytest.mark.parametrize(
    "data, expected",
    [
        pytest.param(np.array([], dtype=np.float64), False, id="empty-float"),
        pytest.param(np.array([], dtype=np.int64), False, id="empty-int"),
        pytest.param(np.array([0, 1, 65535], dtype=np.uint16), False, id="uint16-no-scan"),
        pytest.param(np.array([0, 1, 2**32 - 1], dtype=np.uint32), False, id="uint32-no-scan"),
        pytest.param(np.array([1, -2, 3], dtype=np.int64), True, id="int-negative"),
        pytest.param(np.array([0, 1, 2**32 + 5], dtype=np.int64), False, id="int-non-negative"),
        pytest.param(np.array([1.0, 5.0, 0.0], dtype=np.float64), False, id="float-integral"),
        pytest.param(np.array([1.0, 2.5], dtype=np.float64), True, id="float-fractional"),
        pytest.param(np.array([1.0, -1.0], dtype=np.float64), True, id="float-negative"),
        pytest.param(np.array([1.0, np.nan], dtype=np.float64), True, id="float-nan"),
        pytest.param(np.array([1.0, np.inf], dtype=np.float64), True, id="float-inf"),
    ],
)
def test_has_non_uint_value_truth_table(data, expected):
    """Direct unit test of the chunked predicate (writer.py:41-59, #216 review) -- no
    AnnData/write_cell_archive scaffolding needed. The uint16/uint32 cases assert the
    dtype-branch short-circuit (kind=='u' -> False with no scan, writer.py:51-52); uint32
    uses the largest representable uint32 value (2**32-1) to show the short-circuit holds
    even at that dtype's own boundary. The int-non-negative case deliberately uses a value
    that EXCEEDS the uint32 maximum (2**32+5, representable in int64) on a SIGNED dtype to
    prove this predicate truly ignores magnitude there too (kind=='i' only checks
    .min() < 0, writer.py:53-54) -- range-checking is a separate concern, each caller's own
    >0xFFFFFFFF guard (see test_uint32_overflow_is_not_retryable above)."""
    assert _has_non_uint_value(data) is expected


def test_has_non_uint_value_chunk_boundaries(monkeypatch):
    """An off-by-one in the chunk range/slice (writer.py:55-58) would silently SKIP values
    sitting at a chunk boundary -- a correctness hole (in production, values near a
    16M-element _VALIDATE_CHUNK boundary would go unvalidated). Shrinks _VALIDATE_CHUNK to
    4 via monkeypatch so this stays fast and does not allocate GB, and exercises 3 chunks
    of sizes (4, 4, 2) over a 10-element array: [0:4], [4:8], [8:10].

    Places a bad value at each of the three boundary-risk positions in turn: the last
    element of the FIRST chunk (index 3), the first element of the SECOND chunk (index 4),
    and the very last element of the array (index 9, inside the final PARTIAL chunk) --
    each on an otherwise-all-valid array, isolating that exact position as the sole cause
    of a True result."""
    from cellstream.cell import writer as writer_mod

    monkeypatch.setattr(writer_mod, "_VALIDATE_CHUNK", 4)

    baseline = np.ones(10, dtype=np.float64)
    assert writer_mod._has_non_uint_value(baseline) is False  # sanity: 3 valid chunks -> False

    last_of_first_chunk = baseline.copy()
    last_of_first_chunk[3] = 0.5
    assert writer_mod._has_non_uint_value(last_of_first_chunk) is True

    first_of_second_chunk = baseline.copy()
    first_of_second_chunk[4] = 0.5
    assert writer_mod._has_non_uint_value(first_of_second_chunk) is True

    last_of_array = baseline.copy()
    last_of_array[-1] = 0.5
    assert writer_mod._has_non_uint_value(last_of_array) is True


def test_pfordelta_rejects_oversized_blob_before_the_codec_runs():
    """#269 layer 2: an oversized blob is rejected BEFORE the codec is called.

    The stub is load-bearing. Pre-fix, a real 70 KB blob reaches pyfastpfor and can overrun the
    4096-element buffer, so an in-process red run would crash pytest itself. Asserting the codec
    is never called is also a stronger claim than asserting the message.
    """
    from cellstream.cell.codec import PforDeltaCodec

    dec = PforDeltaCodec("uint16", "uint16").new_decoder(1000)

    class _Exploded(Exception):
        pass

    class _StubCodec:
        def decodeArray(self, *a, **k):
            raise _Exploded("guard let an oversized blob reach the codec")

    dec._c = _StubCodec()
    blob = b"\x00" * 70_000  # > 8*1 + 65536
    # The WHOLE string, not a substring: both engines must emit character-identical guard text
    # (#269). The twin assertion is rust/src/pfor.rs
    # bounds_tests::decode_blob_rejects_blob_larger_than_the_bound -- with the same nnz and blob
    # size, so a divergence in either the message or the bound fails one of the two.
    expected = (
        "corrupt pfordelta frame: codec blob is 70000 bytes, exceeds the maximum 65544 for nnz=1"
    )
    with pytest.raises(ValueError) as excinfo:
        dec._unpack(blob, 1)
    assert str(excinfo.value) == expected
