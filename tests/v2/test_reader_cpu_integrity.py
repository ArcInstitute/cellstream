"""Integrity validation of decoded (untrusted) shard streams in the pure-Python
decode fallback (ultrareview M1).

The Rust decode core validates a shard's decoded CSR streams (non-decreasing
indptr; column < n_vars). These tests assert the Python fallback in
``reader_cpu`` enforces the same invariants and raises a clean ``ValueError`` on
a corrupt/hostile shard instead of producing silent CSR corruption or an
uncontrolled numpy ``IndexError``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstream.v2 import reader_cpu


# --------------------------------------------------------------------------
# _validate_indptr (direct unit tests)
#   signature: _validate_indptr(indptr, expected_rows, shard_idx)
# --------------------------------------------------------------------------
def test_validate_indptr_accepts_valid():
    reader_cpu._validate_indptr(np.array([0, 2, 5, 5, 8], dtype=np.int64), 4, shard_idx=0)


def test_validate_indptr_accepts_empty_shard_indptr():
    # A single-row shard with zero nnz decodes to [0, 0]; must not trip.
    reader_cpu._validate_indptr(np.array([0, 0], dtype=np.int64), 1, shard_idx=0)


def test_validate_indptr_rejects_truncated_length():
    # len 3 != expected_rows(5) + 1 -- this is the truncation that would
    # otherwise IndexError in _plan_shards at shard_indptr[lstop].
    with pytest.raises(ValueError, match="indptr"):
        reader_cpu._validate_indptr(np.array([0, 2, 4], dtype=np.int64), 5, shard_idx=3)


def test_validate_indptr_rejects_empty():
    with pytest.raises(ValueError, match="indptr"):
        reader_cpu._validate_indptr(np.array([], dtype=np.int64), 2, shard_idx=3)


def test_validate_indptr_rejects_nonzero_start():
    with pytest.raises(ValueError, match="indptr"):
        reader_cpu._validate_indptr(np.array([1, 2, 3], dtype=np.int64), 2, shard_idx=3)


def test_validate_indptr_rejects_descending():
    with pytest.raises(ValueError, match="indptr"):
        reader_cpu._validate_indptr(np.array([0, 5, 4, 9], dtype=np.int64), 3, shard_idx=3)


# --------------------------------------------------------------------------
# _validate_decoded_shard (direct unit tests)
#   signature: _validate_decoded_shard(indptr, data, indices, n_vars,
#                                       shard_idx, expected_rows)
# --------------------------------------------------------------------------
def _valid_streams():
    indptr = np.array([0, 2, 4], dtype=np.int64)  # 2 rows
    data = np.array([1, 2, 3, 4], dtype=np.uint16)
    indices = np.array([0, 2, 1, 3], dtype=np.uint16)
    return indptr, data, indices


def test_validate_decoded_shard_accepts_valid():
    indptr, data, indices = _valid_streams()
    reader_cpu._validate_decoded_shard(indptr, data, indices, 4, shard_idx=0, expected_rows=2)


def test_validate_decoded_shard_accepts_empty_nnz():
    reader_cpu._validate_decoded_shard(
        np.array([0, 0, 0], dtype=np.int64),
        np.array([], dtype=np.uint16),
        np.array([], dtype=np.uint16),
        4,
        shard_idx=0,
        expected_rows=2,
    )


def test_validate_decoded_shard_rejects_nonmonotonic_indptr():
    indptr = np.array([0, 4, 2], dtype=np.int64)
    data = np.array([1, 2, 3, 4], dtype=np.uint16)
    indices = np.array([0, 2, 1, 3], dtype=np.uint16)
    with pytest.raises(ValueError, match="indptr"):
        reader_cpu._validate_decoded_shard(indptr, data, indices, 4, shard_idx=0, expected_rows=2)


def test_validate_decoded_shard_rejects_indptr_tail_mismatch():
    indptr = np.array([0, 2, 3], dtype=np.int64)  # tail 3 != len(data) 4
    data = np.array([1, 2, 3, 4], dtype=np.uint16)
    indices = np.array([0, 2, 1, 3], dtype=np.uint16)
    with pytest.raises(ValueError, match="length"):
        reader_cpu._validate_decoded_shard(indptr, data, indices, 4, shard_idx=0, expected_rows=2)


def test_validate_decoded_shard_rejects_data_indices_length_mismatch():
    indptr = np.array([0, 2, 4], dtype=np.int64)
    data = np.array([1, 2, 3, 4], dtype=np.uint16)
    indices = np.array([0, 2, 1], dtype=np.uint16)  # len 3 != len(data) 4
    with pytest.raises(ValueError, match="length"):
        reader_cpu._validate_decoded_shard(indptr, data, indices, 4, shard_idx=0, expected_rows=2)


def test_validate_decoded_shard_rejects_col_ge_nvars():
    indptr = np.array([0, 2, 4], dtype=np.int64)
    data = np.array([1, 2, 3, 4], dtype=np.uint16)
    indices = np.array([0, 2, 1, 4], dtype=np.uint16)  # col 4 >= n_vars 4
    with pytest.raises(ValueError, match="column|range"):
        reader_cpu._validate_decoded_shard(indptr, data, indices, 4, shard_idx=2, expected_rows=2)


def test_validate_decoded_shard_rejects_negative_col():
    # On-disk index dtypes are unsigned, but the < 0 branch is defensive against
    # a signed dtype / the dense numpy negative-wrap footgun.
    indptr = np.array([0, 2, 4], dtype=np.int64)
    data = np.array([1, 2, 3, 4], dtype=np.uint16)
    indices = np.array([0, 2, 1, -1], dtype=np.int32)
    with pytest.raises(ValueError, match="column|range"):
        reader_cpu._validate_decoded_shard(indptr, data, indices, 4, shard_idx=2, expected_rows=2)


# --------------------------------------------------------------------------
# Integration: corruption injected through the real Python read path
# (bitshuffle archive -> always Python; n_workers=1 -> in-process so the
# _decode_stream monkeypatch reaches the worker / _plan_shards).
# --------------------------------------------------------------------------
def _corrupt_decode_stream(role_to_corrupt, transform):
    """Wrap reader_cpu._decode_stream, transforming only the chosen stream role.
    ``transform`` takes the decoded array and returns the (possibly resized)
    corrupted array."""
    real = reader_cpu._decode_stream

    def fake(mm, spec, dtype, shuffle="bitshuffle", role="data"):
        arr = real(mm, spec, dtype, shuffle=shuffle, role=role)
        if role == role_to_corrupt and arr.size:
            arr = transform(arr.copy())
        return arr

    return fake


def test_csr_python_read_rejects_out_of_range_col(tmp_path, tiny_csr, monkeypatch):
    from tests.v2.test_reader_cpu import _hand_build_v2_archive

    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, tiny_csr, n_shards=1)
    n_vars = tiny_csr.shape[1]

    def _oob(a):
        a[0] = n_vars  # out of range [0, n_vars)
        return a

    monkeypatch.setattr(reader_cpu, "_decode_stream", _corrupt_decode_stream("indices", _oob))
    with pytest.raises(ValueError, match="column|range"):
        reader_cpu.read_v2_to_host(arch, cast_back=False, n_workers=1)


def test_dense_python_read_rejects_out_of_range_col(tmp_path, tiny_csr, monkeypatch):
    from cellstream.v2.materialize import resolve_plan
    from tests._archive_helpers import archive_manifest
    from tests.v2.test_reader_cpu import _hand_build_v2_archive

    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, tiny_csr, n_shards=1)
    n_vars = tiny_csr.shape[1]
    # #154 part B3a, route B: the subject is the PYTHON decode fallback's validation, and the
    # fixture must stay a hand-built DIRECTORY -- _hand_build_v2_archive is shared with
    # test_reader_cpu.py (which has no census failures at all) and test_dispatcher.py, and the
    # three sibling tests here already read it through read_v2_to_host. Only the incidental
    # public-reader call for the manifest moves, to the container-agnostic accessor.
    plan = resolve_plan(
        archive_manifest(arch),
        container="dense",
        data_dtype="float32",
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )

    def _oob(a):
        a[0] = n_vars
        return a

    monkeypatch.setattr(reader_cpu, "_decode_stream", _corrupt_decode_stream("indices", _oob))
    with pytest.raises(ValueError, match="column|range"):
        reader_cpu.read_v2_to_host(arch, plan=plan, n_workers=1)


def test_python_read_rejects_nonmonotonic_indptr(tmp_path, tiny_csr, monkeypatch):
    from tests.v2.test_reader_cpu import _hand_build_v2_archive

    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, tiny_csr, n_shards=1)

    def _break(a):
        # Make a middle entry exceed the following one -> non-monotonic, while
        # keeping the length and tail intact so the monotonicity check is what fires.
        a[1] = int(a[2]) + 1
        return a

    monkeypatch.setattr(reader_cpu, "_decode_stream", _corrupt_decode_stream("indptr", _break))
    with pytest.raises(ValueError, match="indptr"):
        reader_cpu.read_v2_to_host(arch, cast_back=False, n_workers=1)


def test_python_read_rejects_truncated_indptr(tmp_path, tiny_csr, monkeypatch):
    # A truncated indptr stream would IndexError in _plan_shards at
    # shard_indptr[lstop]; the length check turns it into a clean ValueError.
    from tests.v2.test_reader_cpu import _hand_build_v2_archive

    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, tiny_csr, n_shards=1)

    monkeypatch.setattr(
        reader_cpu, "_decode_stream", _corrupt_decode_stream("indptr", lambda a: a[:-1])
    )
    with pytest.raises(ValueError, match="indptr"):
        reader_cpu.read_v2_to_host(arch, cast_back=False, n_workers=1)
