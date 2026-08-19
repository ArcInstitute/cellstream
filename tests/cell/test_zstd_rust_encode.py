"""Rust zstd cell ENCODE, end-to-end through write_cell_archive (PR3).

THE GATE is DECODE-identity (spec §3): an archive written by Rust and one written by the
pure-Python encoder must decode to identical values. zstd payload bytes are compressor output
and depend on libzstd, so byte-identity is a non-gating bonus
(test_zstd_rust_encode_byte_identity_bonus), not a contract.
"""

from __future__ import annotations

import contextlib
import os

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellstream.cell._rust_dispatch import rust_available, use_rust_for_cell_encode
from cellstream.cell.reader import open_cell
from cellstream.cell.writer import write_cell_archive
from cellstream.errors import LossyCastError

pytestmark = pytest.mark.skipif(
    not rust_available(), reason="Rust extension unavailable / CELLSTREAM_DISABLE_RUST=1"
)


@contextlib.contextmanager
def _no_rust():
    """Force the pure-Python encode path (rust_available() re-reads os.environ per call)."""
    prev = os.environ.get("CELLSTREAM_DISABLE_RUST")
    os.environ["CELLSTREAM_DISABLE_RUST"] = "1"
    try:
        assert not rust_available()
        yield
    finally:
        if prev is None:
            os.environ.pop("CELLSTREAM_DISABLE_RUST", None)
        else:
            os.environ["CELLSTREAM_DISABLE_RUST"] = prev


def _adata(rng, n_obs=80, n_vars=300, density=0.15, dtype=np.float32, fractional=False):
    X = sp.random(n_obs, n_vars, density=density, format="csr", random_state=rng)
    X.data = X.data.astype(dtype) if fractional else np.rint(X.data * 50).astype(dtype)
    X.sort_indices()
    a = ad.AnnData(X=X)
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.var_names = [f"g{i}" for i in range(n_vars)]
    return a


def _read_all(path):
    s = open_cell(path)
    try:
        return s.gather_rows(np.arange(s.n_obs)).toarray()
    finally:
        s.close()


def test_gate_accepts_zstd_and_needs_the_on_disk_dtype():
    assert use_rust_for_cell_encode("zstd", "float32", "int32", "uint32")
    assert use_rust_for_cell_encode("zstd", "float64", "int32", "float64")
    assert use_rust_for_cell_encode("pfordelta", "uint16", "int32", "uint32")
    # dtypes Rust's FrameEnc::from_py rejects must NOT be waved through (Gemini #212 r2):
    # the gate would return True and Rust would RAISE instead of falling back to Python.
    assert not use_rust_for_cell_encode("zstd", "float32", "int32", "uint16")
    assert not use_rust_for_cell_encode("zstd", "float32", "int64", "float16")
    assert not use_rust_for_cell_encode("zstd", "float16", "int32", "uint32")  # bad source
    # Float ON-DISK requires a float SOURCE. Rust's encode_cells rejects integer-source +
    # float-on-disk; if this gate said True, Rust would RAISE instead of falling back to the
    # Python encoder -- the very #212 r2 failure mode the gate exists to prevent.
    # (Codex, PR #220 code review: the gate DID say True here.)
    assert not use_rust_for_cell_encode("zstd", "uint16", "int32", "float32")
    assert not use_rust_for_cell_encode("zstd", "int32", "int32", "float64")
    assert not use_rust_for_cell_encode("zstd", "uint32", "int64", "float32")
    # ...but an integer source is still fine for the uint32 on-disk path.
    assert use_rust_for_cell_encode("zstd", "uint16", "int32", "uint32")


@pytest.mark.parametrize(
    "dtype,data_dtype,fractional",
    [
        (np.float32, None, False),  # integer counts -> uint32 on disk
        (np.float32, "float32", False),  # explicit float storage
        (np.float64, "float64", True),  # genuine floats, f64 on disk
        (np.float32, None, True),  # fractional -> auto float storage
    ],
)
def test_rust_and_python_encoders_decode_identically(tmp_path, dtype, data_dtype, fractional):
    """THE GATE. Same AnnData, two engines, identical decoded values."""
    rng = np.random.default_rng(7)
    a = _adata(rng, dtype=dtype, fractional=fractional)

    ru, py = tmp_path / "rust.shad", tmp_path / "py.shad"
    write_cell_archive(a, ru, codec="zstd", data_dtype=data_dtype)
    with _no_rust():
        write_cell_archive(a, py, codec="zstd", data_dtype=data_dtype)

    np.testing.assert_array_equal(_read_all(ru), _read_all(py))


def test_rust_zstd_roundtrip_is_exact(tmp_path):
    rng = np.random.default_rng(11)
    a = _adata(rng, dtype=np.float64, fractional=True)
    p = tmp_path / "a.shad"
    write_cell_archive(a, p, codec="zstd", data_dtype="float64")
    np.testing.assert_array_equal(_read_all(p), a.X.toarray())  # bit-exact


def test_rust_lossy_f64_to_f32_raises(tmp_path):
    a = _adata(np.random.default_rng(13), dtype=np.float64, fractional=True)
    with pytest.raises(LossyCastError):
        write_cell_archive(a, tmp_path / "a.shad", codec="zstd", data_dtype="float32")


def test_rust_lossy_f64_to_f32_forced_with_allow_lossy(tmp_path):
    a = _adata(np.random.default_rng(13), dtype=np.float64, fractional=True)
    p = tmp_path / "a.shad"
    write_cell_archive(a, p, codec="zstd", data_dtype="float32", allow_lossy=True)
    np.testing.assert_allclose(_read_all(p), a.X.toarray().astype(np.float32))


def test_rust_manifest_matches_python_manifest(tmp_path):
    """x_data_dtype_min now comes from Rust's max_value instead of a Python O(nnz) scan --
    it must land on the same answer. And value_dtype_on_disk must NOT be narrowed by it."""
    rng = np.random.default_rng(17)
    a = _adata(rng, dtype=np.float32)
    ru, py = tmp_path / "r.shad", tmp_path / "p.shad"
    write_cell_archive(a, ru, codec="zstd")
    with _no_rust():
        write_cell_archive(a, py, codec="zstd")

    sr, sp_ = open_cell(ru), open_cell(py)
    try:
        for key in (
            "value_dtype_on_disk",
            "index_dtype_on_disk",
            "x_data_dtype_min",
            "total_nnz",
            "schema_version",
            "codec",
        ):
            assert sr.manifest[key] == sp_.manifest[key], key
        # zstd's on-disk width is FIXED; max_value feeds x_data_dtype_min ONLY.
        assert sr.manifest["value_dtype_on_disk"] == "uint32"
        assert sr.manifest["schema_version"] == 6  # no schema bump
    finally:
        sr.close()
        sp_.close()


def test_rust_payload_checksum_matches_python(tmp_path):
    rng = np.random.default_rng(19)
    a = _adata(rng, dtype=np.float32)
    ru, py = tmp_path / "r.shad", tmp_path / "p.shad"
    write_cell_archive(a, ru, codec="zstd", payload_checksum=True)
    with _no_rust():
        write_cell_archive(a, py, codec="zstd", payload_checksum=True)
    sr, sp_ = open_cell(ru), open_cell(py)
    try:
        assert sr.verify_payload()
        assert sp_.verify_payload()
    finally:
        sr.close()
        sp_.close()


def test_rust_zstd_empty_archive(tmp_path):
    a = ad.AnnData(X=sp.csr_matrix((0, 5), dtype=np.float32))
    a.var_names = [f"g{i}" for i in range(5)]
    p = tmp_path / "e.shad"
    write_cell_archive(a, p, codec="zstd")
    s = open_cell(p)
    try:
        assert s.n_obs == 0
    finally:
        s.close()


def test_rust_zstd_duplicate_indices_raise(tmp_path):
    # Build the CSR directly with MATCHING array lengths. Mutating .indices on a matrix built
    # from a dense array leaves data/indices length-mismatched, and validate_csr_structure then
    # raises a STRUCTURE error before the duplicate check ever runs -- the test would pass for
    # the wrong reason. (Codex, PR #220 r1.)
    X = sp.csr_matrix(
        (
            np.array([1.0, 2.0], dtype=np.float32),  # data
            np.array([1, 1], dtype=np.int32),  # indices: duplicate (cell, gene)
            np.array([0, 2], dtype=np.int32),  # indptr
        ),
        shape=(1, 3),
    )
    a = ad.AnnData(X=X)
    a.var_names = ["g0", "g1", "g2"]
    with pytest.raises(ValueError, match="duplicate column index"):
        write_cell_archive(a, tmp_path / "d.shad", codec="zstd", require_unique_indices=True)


def test_rust_grouped_zstd_write_matches_python(tmp_path):
    """The permutation path: group_by reorders rows, so perm is a real permutation."""
    rng = np.random.default_rng(23)
    a = _adata(rng, n_obs=60)
    a.obs["target_gene"] = ["ctrl" if i % 3 == 0 else f"t{i % 5}" for i in range(60)]
    ru, py = tmp_path / "r.shad", tmp_path / "p.shad"
    write_cell_archive(a, ru, codec="zstd", group_by="target_gene", reference="ctrl")
    with _no_rust():
        write_cell_archive(a, py, codec="zstd", group_by="target_gene", reference="ctrl")
    np.testing.assert_array_equal(_read_all(ru), _read_all(py))


def test_zstd_rust_encode_byte_identity_bonus(tmp_path):
    """NON-GATING: the Rust and Python zstd encoders should emit byte-identical payloads
    (same libzstd-1 over the same LE bytes). Skip on libzstd drift -- decode-identity
    (test_rust_and_python_encoders_decode_identically) is the real guarantee. See spec §3.

    Deliberately NaN-FREE (integer-valued float32). Rust's `as f32` and numpy's
    astype(float32) are not contractually required to emit the same NaN PAYLOAD bits, so a NaN
    in the data could break byte-identity without anything actually being wrong. NaN
    correctness is covered by the decode-identity + equal_nan tests. (Codex, PR #220 r1.)

    Compares the PAYLOAD, not the whole file: the packed .shad manifest embeds a `created_at`
    timestamp, so two sequential writes that straddle a second boundary yield different FILES
    with identical payloads -- a whole-file filecmp would skip spuriously and the test would
    silently stop testing anything. A skip must mean libzstd drifted, nothing else.
    (Gemini, PR #220 r2.)
    """
    rng = np.random.default_rng(29)
    a = _adata(rng, dtype=np.float32)  # integer-valued -> uint32 on disk, no NaN
    ru, py = tmp_path / "r.shad", tmp_path / "p.shad"
    write_cell_archive(a, ru, codec="zstd")
    with _no_rust():
        write_cell_archive(a, py, codec="zstd")

    s_ru, s_py = open_cell(ru), open_cell(py)
    pay_ru = pay_py = None
    try:
        # s._mm IS the payload member (not the whole file) and offsets are 0-based.
        pay_ru = np.frombuffer(s_ru._mm, dtype=np.uint8)[s_ru.offsets[0] : s_ru.offsets[-1]]
        pay_py = np.frombuffer(s_py._mm, dtype=np.uint8)[s_py.offsets[0] : s_py.offsets[-1]]
        equal = np.array_equal(pay_ru, pay_py)
    finally:
        # Drop the views BEFORE closing, and do it in `finally` -- a live frombuffer view PINS
        # the mmap, so close() raises BufferError("cannot close exported pointers exist"). If an
        # exception escaped the try block with the views still live, that BufferError would
        # replace it and MASK the real failure. Exactly the bug PR2 fixed in
        # _rust_dispatch.decode_cells (`del payload_u8` in its except). (Gemini, PR #220.)
        del pay_ru, pay_py
        s_ru.close()
        s_py.close()

    if not equal:
        pytest.skip("libzstd byte-output drifted; decode-identity still holds")
