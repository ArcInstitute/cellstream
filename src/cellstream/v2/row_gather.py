"""Row-gather primitives for the lean target-aware (grouped) v2 writer (#76).

Two ways to assemble a shard's rows into a local CSR, in a requested output
order:
  - gather_csr_rows: from in-RAM CSR arrays (the shm provider + serial fallback).
  - read_backed_csr_rows: from a backed h5ad on disk (the path provider).
Both return (data, indices, indptr) for the given rows IN THE GIVEN ORDER;
indptr is rebased to start at 0. The encode is identical downstream, so output
bytes do not depend on which provider produced the rows.
"""

from __future__ import annotations

import numpy as np


def gather_csr_rows(data, indices, indptr, row_indices):
    """Gather `row_indices` (in output order) from a CSR into a local CSR.

    data/indices: full CSR value/column arrays (numpy). indptr: full int64 row
    pointers. row_indices: int64 original-row indices in the desired output
    order (may be unsorted / contain duplicates). Returns (out_data,
    out_indices, out_indptr) with out_indptr rebased to 0.
    """
    oi = np.asarray(row_indices, dtype=np.int64)
    if oi.size == 0:
        return (
            np.empty(0, dtype=data.dtype),
            np.empty(0, dtype=indices.dtype),
            np.zeros(1, dtype=np.int64),
        )
    starts = indptr[oi]
    row_lens = indptr[oi + 1] - starts
    out_indptr = np.zeros(oi.size + 1, dtype=np.int64)
    np.cumsum(row_lens, out=out_indptr[1:])
    total = int(out_indptr[-1])
    out_data = np.empty(total, dtype=data.dtype)
    out_indices = np.empty(total, dtype=indices.dtype)
    for j in range(oi.size):
        s = int(starts[j])
        e = s + int(row_lens[j])
        os_ = int(out_indptr[j])
        oe = int(out_indptr[j + 1])
        out_data[os_:oe] = data[s:e]
        out_indices[os_:oe] = indices[s:e]
    return out_data, out_indices, out_indptr


def read_backed_csr_rows(h5ad_path, row_indices):
    """Read `row_indices` (in output order) from a backed h5ad's X into a local CSR.

    Opens the source read-only, wraps X as a backed CSR via
    anndata.io.sparse_dataset, and fancy-indexes the rows. anndata performs the
    efficient sorted read + reorder internally and returns a scipy CSR in
    `row_indices` order. Returns (data, indices, indptr) in source dtypes;
    the caller validates + casts. Concurrent read-only access from multiple
    processes is safe -- the source file is immutable for the duration of the write.
    (This used to cite v1's ``_write_sharded_from_path`` workers as the precedent; that
    writer was removed in #154.)
    """
    import h5py
    import scipy.sparse as sp
    from anndata.io import sparse_dataset

    oi = np.asarray(row_indices, dtype=np.int64)
    with h5py.File(str(h5ad_path), "r") as f:
        ds = sparse_dataset(f["X"])
        if oi.size == 0:
            sub = sp.csr_matrix((0, int(ds.shape[1])), dtype=ds.dtype)
        else:
            # Read strictly-sorted, unique rows from HDF5 (forward-only seeks —
            # the spec's "sorted read" intent), then reorder to oi order in
            # memory via fast scipy CSR row-indexing. Robust across anndata
            # versions (older SparseDataset may reject unsorted/dup fancy index).
            # Verified equal to ds[oi] and to X[oi] on anndata 0.12.16.
            uniq, inverse = np.unique(oi, return_inverse=True)
            sub = ds[uniq][inverse]
    return (
        np.ascontiguousarray(sub.data),
        np.ascontiguousarray(sub.indices),
        sub.indptr.astype(np.int64),
    )
