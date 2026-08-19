"""Rust zstd cell ENCODE kernel (PR3), driven through _rust.encode_cells directly.

The gate is DECODE-identity, not byte-identity: zstd payload bytes are compressor output
and depend on libzstd (spec §3). These tests assert the Rust-encoded frames decode to
exactly what PR1's pure-Python encoder would have produced.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from cellstream.cell._rust_dispatch import rust_available
from cellstream.cell.codec import codec_for

pytestmark = pytest.mark.skipif(
    not rust_available(), reason="Rust extension unavailable / CELLSTREAM_DISABLE_RUST=1"
)


def _rust():
    from cellstream.v2 import _rust

    return _rust


def _csr(rng, n_obs=64, n_vars=200, density=0.1, dtype=np.float32):
    X = sp.random(n_obs, n_vars, density=density, format="csr", random_state=rng)
    X.data = np.rint(X.data * 100).astype(dtype)
    X.sort_indices()
    return X


def _decode_payload(path, offsets, indptr, n_obs, codec, value_dtype):
    """Decode every frame with the pure-Python decoder — the reference."""
    dec = codec_for(codec, "uint32", value_dtype).new_decoder(70000)
    blob = path.read_bytes()
    out = []
    for r in range(n_obs):
        frame = blob[offsets[r] : offsets[r + 1]]
        nnz = int(indptr[r + 1] - indptr[r])
        out.append(dec.decode(frame, nnz))
    return out


def test_rust_zstd_uint32_frames_decode_correctly(tmp_path):
    rng = np.random.default_rng(0)
    X = _csr(rng)
    perm = np.arange(X.shape[0], dtype=np.int64)
    p = tmp_path / "payload.bin"

    offsets, indptr, total, _mx, _sha = _rust().encode_cells(
        X.data, X.indices, X.indptr, perm, str(p), 4, True, False, "zstd", "uint32", False
    )

    assert total == X.nnz
    got = _decode_payload(p, offsets, indptr, X.shape[0], "zstd", "uint32")
    for r, (idx, val) in enumerate(got):
        s, e = X.indptr[r], X.indptr[r + 1]
        np.testing.assert_array_equal(idx, X.indices[s:e].astype(np.uint32))
        np.testing.assert_array_equal(val, X.data[s:e].astype(np.uint32))


def test_rust_zstd_float64_frames_decode_bit_exact(tmp_path):
    rng = np.random.default_rng(1)
    X = sp.random(32, 150, density=0.2, format="csr", random_state=rng, dtype=np.float64)
    X.sort_indices()
    perm = np.arange(X.shape[0], dtype=np.int64)
    p = tmp_path / "payload.bin"

    offsets, indptr, _t, _mx, _sha = _rust().encode_cells(
        X.data, X.indices, X.indptr, perm, str(p), 4, False, False, "zstd", "float64", False
    )

    got = _decode_payload(p, offsets, indptr, X.shape[0], "zstd", "float64")
    for r, (_idx, val) in enumerate(got):
        s, e = X.indptr[r], X.indptr[r + 1]
        np.testing.assert_array_equal(val, X.data[s:e])  # bit-exact, no tolerance


def test_rust_zstd_f64_to_f32_lossy_raises_without_allow_lossy(tmp_path):
    X = sp.csr_matrix(np.array([[0.1, 0.0, 2.5]], dtype=np.float64))  # 0.1 is not f32-exact
    X.sort_indices()
    perm = np.arange(1, dtype=np.int64)
    with pytest.raises(ValueError, match="lossy"):
        _rust().encode_cells(
            X.data,
            X.indices,
            X.indptr,
            perm,
            str(tmp_path / "p.bin"),
            1,
            False,
            False,
            "zstd",
            "float32",
            False,
        )


def test_rust_zstd_f64_to_f32_lossy_forced_with_allow_lossy(tmp_path):
    X = sp.csr_matrix(np.array([[0.1, 0.0, 2.5]], dtype=np.float64))
    X.sort_indices()
    perm = np.arange(1, dtype=np.int64)
    p = tmp_path / "p.bin"
    offsets, indptr, _t, _mx, _sha = _rust().encode_cells(
        X.data, X.indices, X.indptr, perm, str(p), 1, False, False, "zstd", "float32", True
    )
    ((_idx, val),) = _decode_payload(p, offsets, indptr, 1, "zstd", "float32")
    np.testing.assert_array_equal(val, X.data.astype(np.float32))


def test_rust_zstd_nan_inf_f64_to_f32_is_NOT_lossy(tmp_path):
    """equal_nan semantics (narrow.py::lossless_cast): NaN->NaN is LOSSLESS regardless of
    payload, and +-Inf survives. A bitwise implementation would wrongly raise here."""
    vals = np.array([np.nan, np.inf, -np.inf, 1.5], dtype=np.float64)
    X = sp.csr_matrix(vals.reshape(1, -1))
    X.sort_indices()
    perm = np.arange(1, dtype=np.int64)
    p = tmp_path / "p.bin"

    offsets, indptr, _t, _mx, _sha = _rust().encode_cells(  # must NOT raise
        X.data, X.indices, X.indptr, perm, str(p), 1, False, False, "zstd", "float32", False
    )
    ((_idx, val),) = _decode_payload(p, offsets, indptr, 1, "zstd", "float32")
    np.testing.assert_array_equal(val, vals.astype(np.float32))


def test_rust_zstd_f64_overflowing_f32_range_raises(tmp_path):
    """A finite f64 that overflows f32 becomes inf, so back != v -> LOSSY. numpy agrees."""
    X = sp.csr_matrix(np.array([[1e300, 1.0]], dtype=np.float64))
    X.sort_indices()
    perm = np.arange(1, dtype=np.int64)
    with pytest.raises(ValueError, match="lossy"):
        _rust().encode_cells(
            X.data,
            X.indices,
            X.indptr,
            perm,
            str(tmp_path / "p.bin"),
            1,
            False,
            False,
            "zstd",
            "float32",
            False,
        )


def test_rust_zstd_empty_cell_frame_is_22_bytes(tmp_path):
    """An nnz=0 cell yields TWO zstd-of-empty blobs (9 B each) + the 4-byte length prefix.
    pfordelta's empty frame is 4 bytes; zstd's is 22. Both engines must agree."""
    X = sp.csr_matrix((3, 10), dtype=np.float32)  # all rows empty
    perm = np.arange(3, dtype=np.int64)
    offsets, _indptr, total, _mx, _sha = _rust().encode_cells(
        X.data,
        X.indices,
        X.indptr,
        perm,
        str(tmp_path / "p.bin"),
        1,
        False,
        False,
        "zstd",
        "uint32",
        False,
    )
    assert total == 0
    assert list(np.diff(offsets)) == [22, 22, 22]


def test_rust_zstd_unsorted_indices_raise(tmp_path):
    X = sp.csr_matrix(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    X.indices = np.array([2, 0, 1], dtype=np.int32)  # deliberately unsorted
    perm = np.arange(1, dtype=np.int64)
    with pytest.raises(ValueError, match="non-decreasing"):
        _rust().encode_cells(
            X.data,
            X.indices,
            X.indptr,
            perm,
            str(tmp_path / "p.bin"),
            1,
            False,
            False,
            "zstd",
            "uint32",
            False,
        )


def test_rust_zstd_duplicate_indices_raise_when_checked(tmp_path):
    X = sp.csr_matrix(np.array([[1.0, 2.0]], dtype=np.float32))
    X.indices = np.array([1, 1], dtype=np.int32)  # duplicate (cell, gene); lengths still match
    perm = np.arange(1, dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate column index"):
        _rust().encode_cells(
            X.data,
            X.indices,
            X.indptr,
            perm,
            str(tmp_path / "p.bin"),
            1,
            False,
            True,
            "zstd",
            "uint32",
            False,
        )


def test_rust_zstd_integer_source_with_float_on_disk_is_rejected(tmp_path):
    """Both engines must have IDENTICAL accept-sets (the Gemini #212 r2 invariant).

    _plan_cell_codec raises "data_dtype requires float input" for integer input, so Python
    can never produce this combination -- and the Rust kernel must refuse it too rather than
    quietly doing a lossy int->float cast. (An "honest" cast is a trap: the check is blind for
    i64 above 2^53, because `self as f64` already rounds. PR #220 r1.)
    """
    X = sp.random(8, 40, density=0.3, format="csr", random_state=0)
    X.data = np.rint(X.data * 10).astype(np.uint16)  # INTEGER source
    X.sort_indices()
    perm = np.arange(X.shape[0], dtype=np.int64)
    with pytest.raises(ValueError, match="requires float input"):
        _rust().encode_cells(
            X.data,
            X.indices,
            X.indptr,
            perm,
            str(tmp_path / "p.bin"),
            1,
            False,
            False,
            "zstd",
            "float32",
            False,
        )


def test_pfordelta_default_args_unchanged(tmp_path):
    """The three new args default so every pre-PR3 caller keeps working."""
    rng = np.random.default_rng(3)
    X = _csr(rng)
    perm = np.arange(X.shape[0], dtype=np.int64)
    p = tmp_path / "p.bin"
    offsets, indptr, total, _mx, _sha = _rust().encode_cells(
        X.data, X.indices, X.indptr, perm, str(p), 4
    )
    assert total == X.nnz
    got = _decode_payload(p, offsets, indptr, X.shape[0], "pfordelta", "uint32")
    for r, (idx, _val) in enumerate(got):
        s, e = X.indptr[r], X.indptr[r + 1]
        np.testing.assert_array_equal(idx, X.indices[s:e].astype(np.uint32))


@pytest.mark.parametrize("value_dtype", ["float32", "float64", "uint32"])
def test_rust_zstd_frames_are_byte_exact_vs_python_encoder(tmp_path, value_dtype):
    """NON-GATING byte-exactness at the FRAME level -- the strongest check that Rust emits
    PR1's exact bytes, and the one the archive-level bonus test cannot give for FLOATS
    (it only exercises the uint32 path). Skip on libzstd drift: Rust's `zstd` crate and
    Python's `zstandard` wrap different libzstd builds, so identical bytes are a property of
    the current versions, not a contract (spec §3). (Codex, PR #220 code review.)
    """
    rng = np.random.default_rng(41)
    if value_dtype == "uint32":
        X = _csr(rng, n_obs=24, n_vars=120, dtype=np.float32)
    else:
        X = sp.random(24, 120, density=0.2, format="csr", random_state=rng, dtype=np.float64)
        X.sort_indices()
        if value_dtype == "float32":
            X.data = X.data.astype(np.float32)  # f32 source -> f32 on disk, no lossy gate
    perm = np.arange(X.shape[0], dtype=np.int64)
    p = tmp_path / "p.bin"

    offsets, _indptr, _t, _mx, _sha = _rust().encode_cells(
        X.data, X.indices, X.indptr, perm, str(p), 4, False, False, "zstd", value_dtype, False
    )

    # The reference: PR1's pure-Python encoder, frame by frame.
    py_enc = codec_for("zstd", "uint32", value_dtype).new_encoder()
    vdt = (
        np.float32
        if value_dtype == "float32"
        else (np.float64 if value_dtype == "float64" else np.uint32)
    )
    blob = p.read_bytes()
    for r in range(X.shape[0]):
        s0, e0 = X.indptr[r], X.indptr[r + 1]
        want = py_enc.encode(X.indices[s0:e0].astype(np.uint32), X.data[s0:e0].astype(vdt))
        got = blob[offsets[r] : offsets[r + 1]]
        if got != want:
            pytest.skip("libzstd byte-output drifted; decode-identity still holds")


def test_rust_zstd_empty_cell_frame_is_byte_exact_vs_python(tmp_path):
    """The empty cell is the subtle one: Python's byte_filter_encode returns compress(b"")
    for size 0, and Rust's encode_chunk returns zstd1(&[]). Both must agree -> a 22-byte frame
    (4-byte prefix + two 9-byte zstd-of-empty blobs), vs pfordelta's 4."""
    X = sp.csr_matrix((2, 8), dtype=np.float32)  # all rows empty
    perm = np.arange(2, dtype=np.int64)
    p = tmp_path / "p.bin"
    offsets, _i, _t, _m, _s = _rust().encode_cells(
        X.data, X.indices, X.indptr, perm, str(p), 1, False, False, "zstd", "uint32", False
    )
    py_enc = codec_for("zstd", "uint32", "uint32").new_encoder()
    want = py_enc.encode(np.empty(0, np.uint32), np.empty(0, np.uint32))
    assert len(want) == 22
    blob = p.read_bytes()
    for r in range(2):
        assert blob[offsets[r] : offsets[r + 1]] == want
