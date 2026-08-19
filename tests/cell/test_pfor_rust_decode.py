"""decode_cells parity vs the pure-Python _PforDecoder, direct-to-heap."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

_rd = pytest.importorskip("cellstream.cell._rust_dispatch")
pytestmark = pytest.mark.skipif(not _rd.rust_available(), reason="Rust core not built/enabled")


def _build_frames(csr, index_dtype, value_dtype):
    """Encode each row with the pure-Python codec; return (payload, offsets, indptr)."""
    from cellstream.cell.codec import codec_for

    enc = codec_for("pfordelta", index_dtype, value_dtype).new_encoder()
    payload, offsets, indptr = bytearray(), [0], [0]
    for i in range(csr.shape[0]):
        s, e = csr.indptr[i], csr.indptr[i + 1]
        frame = enc.encode(csr.indices[s:e].astype(index_dtype), csr.data[s:e].astype(value_dtype))
        payload += frame
        offsets.append(len(payload))
        indptr.append(int(e))
    return (
        np.frombuffer(bytes(payload), dtype=np.uint8),
        np.asarray(offsets, np.int64),
        np.asarray(indptr, np.int64),
    )


@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("data_dtype,index_dtype", [("float32", "int32"), ("uint16", "int32")])
def test_decode_cells_matches_python(n_threads, data_dtype, index_dtype):
    rng = np.random.default_rng(11)
    n_obs, n_vars = 50, 300
    csr = sp.random(n_obs, n_vars, density=0.1, random_state=rng, dtype=np.float64)
    csr.data = rng.integers(1, 500, size=csr.nnz).astype(np.float64)
    csr = csr.tocsr()
    csr.sort_indices()
    payload, offsets, indptr = _build_frames(csr, "uint16", "uint16")

    row_ids = np.arange(n_obs, dtype=np.int64)
    total = int(indptr[-1])
    out_data = np.empty(total, dtype=data_dtype)
    out_indices = np.empty(total, dtype=index_dtype)
    out_indptr = np.empty(n_obs + 1, dtype=np.int64)
    _rd.decode_cells(
        payload,
        offsets,
        indptr,
        row_ids,
        out_data,
        out_indices,
        out_indptr,
        codec="pfordelta",
        value_dtype_on_disk="uint16",
        n_vars=n_vars,
        allow_lossy=False,
        n_threads=n_threads,
    )
    got = sp.csr_matrix((out_data, out_indices, out_indptr), shape=(n_obs, n_vars))
    np.testing.assert_array_equal(got.toarray(), csr.toarray())


def test_decode_cells_gather_subset_and_reorder():
    rng = np.random.default_rng(5)
    n_obs, n_vars = 40, 120
    csr = sp.random(n_obs, n_vars, density=0.15, random_state=rng, dtype=np.float64)
    csr.data = rng.integers(1, 200, size=csr.nnz).astype(np.float64)
    csr = csr.tocsr()
    csr.sort_indices()
    payload, offsets, indptr = _build_frames(csr, "uint16", "uint16")
    row_ids = np.array([7, 7, 3, 39, 0], dtype=np.int64)  # dup + out-of-order + last row
    sel_nnz = indptr[row_ids + 1] - indptr[row_ids]
    total = int(sel_nnz.sum())
    od, oi = np.empty(total, np.float32), np.empty(total, np.int32)
    op = np.empty(row_ids.size + 1, np.int64)
    _rd.decode_cells(
        payload,
        offsets,
        indptr,
        row_ids,
        od,
        oi,
        op,
        codec="pfordelta",
        value_dtype_on_disk="uint16",
        n_vars=n_vars,
        allow_lossy=False,
        n_threads=1,
    )
    got = sp.csr_matrix((od, oi, op), shape=(row_ids.size, n_vars))
    np.testing.assert_array_equal(got.toarray(), csr.toarray()[row_ids])


def test_decode_cells_rejects_out_of_range_column():
    """A decoded column index >= n_vars (corrupt/mismatched frame) must raise, not crash."""
    rng = np.random.default_rng(9)
    n_obs, n_vars = 10, 300
    csr = sp.random(n_obs, n_vars, density=0.2, random_state=rng, dtype=np.float64)
    csr.data = rng.integers(1, 100, size=csr.nnz).astype(np.float64)
    csr = csr.tocsr()
    csr.sort_indices()
    payload, offsets, indptr = _build_frames(csr, "uint16", "uint16")
    row_ids = np.arange(n_obs, dtype=np.int64)
    total = int(indptr[-1])
    od, oi, op = (
        np.empty(total, np.float32),
        np.empty(total, np.int32),
        np.empty(n_obs + 1, np.int64),
    )
    # decode with a too-small n_vars so a valid column index trips the [0, n_vars) guard
    with pytest.raises(ValueError, match="out of range"):
        _rd.decode_cells(
            payload,
            offsets,
            indptr,
            row_ids,
            od,
            oi,
            op,
            codec="pfordelta",
            value_dtype_on_disk="uint16",
            n_vars=5,
            allow_lossy=False,
            n_threads=1,
        )
