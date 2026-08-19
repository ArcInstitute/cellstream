"""encode_cells parity: full payload + offsets/indptr byte-identical to the Python encoder."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import scipy.sparse as sp

_rust = pytest.importorskip("cellstream.v2._rust")
pytestmark = pytest.mark.skipif(
    not hasattr(_rust, "encode_cells"), reason="Rust encode_cells not built"
)


def _python_encode(csr, perm, index_dtype, value_dtype):
    """Reference: exactly the writer.py streaming loop, returning (payload, offsets, indptr, max)."""
    from cellstream.cell.codec import codec_for

    enc = codec_for("pfordelta", index_dtype, value_dtype).new_encoder()
    idt = np.dtype(index_dtype)
    vdt = np.dtype(value_dtype)
    payload, offsets, indptr = bytearray(), [0], [0]
    mx = 0
    for new_row in range(csr.shape[0]):
        i = int(perm[new_row])
        s, e = csr.indptr[i], csr.indptr[i + 1]
        idx = csr.indices[s:e].astype(idt)
        val = csr.data[s:e].astype(vdt)
        if val.size:
            mx = max(mx, int(val.max()))
        frame = enc.encode(idx, val)
        payload += frame
        offsets.append(offsets[-1] + len(frame))
        indptr.append(indptr[-1] + int(e - s))
    return bytes(payload), np.asarray(offsets, np.int64), np.asarray(indptr, np.int64), mx


def _make_csr(n_obs, n_vars, density, seed, dtype):
    rng = np.random.default_rng(seed)
    csr = sp.random(n_obs, n_vars, density=density, random_state=rng, dtype=np.float64)
    csr.data = rng.integers(1, 500, size=csr.nnz).astype(dtype)
    csr = csr.tocsr()
    csr.sort_indices()
    return csr


def _run_rust(csr, perm, tmp_path, n_threads):
    ppath = str(tmp_path / "x_payload")
    off, ip, total, mx, sha = _rust.encode_cells(
        np.ascontiguousarray(csr.data),
        np.ascontiguousarray(csr.indices),
        np.ascontiguousarray(csr.indptr),
        np.ascontiguousarray(perm, dtype=np.int64),
        ppath,
        n_threads,
    )
    with open(ppath, "rb") as f:
        payload = f.read()
    return payload, off, ip, int(total), int(mx), sha


@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("data_dtype", ["float32", "float64", "uint16", "uint32", "int64"])
def test_encode_cells_byte_identical(tmp_path, n_threads, data_dtype):
    n_obs, n_vars = 200, 500
    csr = _make_csr(n_obs, n_vars, 0.1, 7, data_dtype)
    perm = np.arange(n_obs, dtype=np.int64)
    index_dtype = "uint16"  # n_vars < 65536
    value_dtype = "uint16"  # values < 500
    p_payload, p_off, p_ip, p_mx = _python_encode(csr, perm, index_dtype, value_dtype)
    r_payload, r_off, r_ip, r_total, r_mx, r_sha = _run_rust(csr, perm, tmp_path, n_threads)
    assert r_payload == p_payload
    np.testing.assert_array_equal(r_off, p_off)
    np.testing.assert_array_equal(r_ip, p_ip)
    assert r_total == int(p_ip[-1])
    assert r_mx == p_mx
    assert r_sha == hashlib.sha256(p_payload).hexdigest()


def test_encode_cells_grouped_perm_byte_identical(tmp_path):
    """Non-identity perm (grouped storage order) still byte-identical."""
    n_obs, n_vars = 300, 400
    csr = _make_csr(n_obs, n_vars, 0.12, 3, "float32")
    rng = np.random.default_rng(99)
    perm = rng.permutation(n_obs).astype(np.int64)
    p_payload, p_off, p_ip, _ = _python_encode(csr, perm, "uint16", "uint16")
    r_payload, r_off, r_ip, _, _, r_sha = _run_rust(csr, perm, tmp_path, 4)
    assert r_payload == p_payload
    np.testing.assert_array_equal(r_off, p_off)
    np.testing.assert_array_equal(r_ip, p_ip)
    assert r_sha == hashlib.sha256(p_payload).hexdigest()


def test_encode_cells_edge_shapes_byte_identical(tmp_path):
    """Empty cells (nnz=0), single-element cells, and a high-value exception run."""
    import struct

    n_vars = 300
    rows = [
        (np.array([], np.int64), np.array([], np.int64)),  # empty cell
        (np.array([5], np.int64), np.array([1], np.int64)),  # single element
        (
            np.array([0, 1, 2, 299], np.int64),
            np.array([1, 65535, 70000, 3], np.int64),
        ),  # u32 vals + exceptions
    ]
    indptr = np.zeros(len(rows) + 1, np.int64)
    data_parts, idx_parts = [], []
    for k, (idx, val) in enumerate(rows):
        indptr[k + 1] = indptr[k] + idx.size
        idx_parts.append(idx)
        data_parts.append(val)
    csr = sp.csr_matrix(
        (
            np.concatenate(data_parts).astype(np.uint32) if data_parts else np.array([], np.uint32),
            np.concatenate(idx_parts).astype(np.int32) if idx_parts else np.array([], np.int32),
            indptr,
        ),
        shape=(len(rows), n_vars),
    )
    csr.sort_indices()
    perm = np.arange(len(rows), dtype=np.int64)
    # value_dtype uint32 because 70000 > 65535
    p_payload, p_off, p_ip, p_mx = _python_encode(csr, perm, "uint16", "uint32")
    r_payload, r_off, r_ip, _, r_mx, r_sha = _run_rust(csr, perm, tmp_path, 2)
    assert r_payload == p_payload
    np.testing.assert_array_equal(r_off, p_off)
    np.testing.assert_array_equal(r_ip, p_ip)
    assert r_mx == p_mx == 70000
    # empty cell frame is exactly the 4-byte zero length prefix
    assert p_payload[: p_off[1]] == struct.pack("<I", 0)


def test_encode_cells_rejects_non_integer(tmp_path):
    csr = sp.csr_matrix(np.array([[1.5, 0.0, 2.0]], dtype=np.float32))
    csr.sort_indices()
    with pytest.raises(ValueError, match="non-negative integer"):
        _run_rust(csr, np.array([0], np.int64), tmp_path, 1)


def test_encode_cells_rejects_negative(tmp_path):
    csr = sp.csr_matrix(np.array([[3.0, -1.0, 2.0]], dtype=np.float32))
    csr.sort_indices()
    with pytest.raises(ValueError, match="non-negative integer"):
        _run_rust(csr, np.array([0], np.int64), tmp_path, 1)


def test_dispatch_gating():
    """PR3 gave the gate a 4th parameter (the ON-DISK value dtype) and made zstd Rust-eligible.

    The zstd assertion here used to be `not ...`: PR2 shipped no Rust zstd ENCODER, so the gate
    had to refuse it. PR3 adds one, so it now accepts -- that inversion is the whole point of
    the PR. pfordelta is unchanged: FastPFor always emits u32 words, so its on-disk width is
    only a narrowing hint and the gate ignores it.
    """
    _rd = pytest.importorskip("cellstream.cell._rust_dispatch")
    if not _rd.rust_available():
        pytest.skip("Rust core not built/enabled")
    # pfordelta -- unchanged
    assert _rd.use_rust_for_cell_encode(
        "pfordelta", np.dtype("float32"), np.dtype("int32"), "uint32"
    )
    assert _rd.use_rust_for_cell_encode(
        "pfordelta", np.dtype("uint16"), np.dtype("int64"), "uint32"
    )
    assert not _rd.use_rust_for_cell_encode(
        "pfordelta", np.dtype("float16"), np.dtype("int32"), "uint32"
    )
    # zstd -- NOW Rust-eligible (PR3), gated on the on-disk value dtype
    assert _rd.use_rust_for_cell_encode("zstd", np.dtype("float32"), np.dtype("int32"), "uint32")
    assert _rd.use_rust_for_cell_encode("zstd", np.dtype("float64"), np.dtype("int32"), "float64")
    # an on-disk dtype Rust's FrameEnc::from_py rejects must NOT be waved through: the gate
    # would return True and Rust would then RAISE instead of falling back (Gemini #212 r2).
    assert not _rd.use_rust_for_cell_encode(
        "zstd", np.dtype("float32"), np.dtype("int32"), "uint16"
    )


def test_wrapper_matches_python(tmp_path):
    _rd = pytest.importorskip("cellstream.cell._rust_dispatch")
    if not _rd.rust_available():
        pytest.skip("Rust core not built/enabled")
    csr = _make_csr(120, 400, 0.1, 4, "float32")
    perm = np.arange(120, dtype=np.int64)
    p_payload, p_off, p_ip, p_mx = _python_encode(csr, perm, "uint16", "uint16")
    ppath = str(tmp_path / "x_payload")
    # pass a NON-contiguous perm to exercise ascontiguousarray in the wrapper
    off, ip, total, mx, sha = _rd.encode_cells(
        csr.data, csr.indices, csr.indptr, perm[::-1][::-1], ppath, n_threads=3
    )
    with open(ppath, "rb") as f:
        assert f.read() == p_payload
    np.testing.assert_array_equal(off, p_off)
    np.testing.assert_array_equal(ip, p_ip)
    assert int(total) == int(p_ip[-1]) and int(mx) == p_mx


def _dup_csr():
    """One cell (row 0) with a duplicate column index (2, 2), already sorted."""
    n_obs, n_vars = 3, 10
    x = sp.csr_matrix((n_obs, n_vars), dtype=np.uint16)
    x.data = np.array([1, 1, 5], dtype=np.uint16)
    x.indices = np.array([2, 2, 7], dtype=np.int32)
    x.indptr = np.array([0, 3, 3, 3], dtype=np.int64)
    return x


def test_encode_cells_no_checksum_returns_none(tmp_path):
    n_obs, n_vars = 200, 500
    csr = _make_csr(n_obs, n_vars, 0.1, 7, "uint16")
    perm = np.arange(n_obs, dtype=np.int64)
    ppath = str(tmp_path / "x_payload")
    off, ip, total, mx, sha = _rust.encode_cells(
        np.ascontiguousarray(csr.data),
        np.ascontiguousarray(csr.indices),
        np.ascontiguousarray(csr.indptr),
        perm,
        ppath,
        4,
        False,  # compute_checksum
        False,  # check_duplicates
    )
    assert sha is None
    with open(ppath, "rb") as f:
        payload = f.read()
    # payload is still correct even though the hash was skipped
    p_payload, _, _, _ = _python_encode(csr, perm, "uint16", "uint16")
    assert payload == p_payload


def test_encode_cells_check_duplicates_raises(tmp_path):
    x = _dup_csr()
    perm = np.arange(x.shape[0], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate column index"):
        _rust.encode_cells(
            np.ascontiguousarray(x.data),
            np.ascontiguousarray(x.indices),
            np.ascontiguousarray(x.indptr),
            perm,
            str(tmp_path / "p"),
            2,
            True,  # compute_checksum
            True,  # check_duplicates
        )


def test_encode_cells_duplicates_allowed_when_unchecked(tmp_path):
    x = _dup_csr()
    perm = np.arange(x.shape[0], dtype=np.int64)
    off, ip, total, mx, sha = _rust.encode_cells(
        np.ascontiguousarray(x.data),
        np.ascontiguousarray(x.indices),
        np.ascontiguousarray(x.indptr),
        perm,
        str(tmp_path / "p"),
        1,
        True,
        False,  # check_duplicates off -> a zero gap encodes fine, no raise
    )
    assert int(total) == 3
