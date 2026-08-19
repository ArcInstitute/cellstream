from __future__ import annotations

import numpy as np
import pytest

from tests._archive_helpers import archive_manifest, shard_paths_and_bases, write_archive


def _write_small_archive(tmp_path):
    import anndata as ad
    import scipy.sparse as sp

    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16)
    indices = np.array([0, 5, 9, 1, 7, 3], dtype=np.int32)
    indptr = np.array([0, 2, 3, 3, 4, 5, 6], dtype=np.int32)
    X = sp.csr_matrix((data, indices, indptr), shape=(6, 10), dtype=np.uint16)
    out = tmp_path / "csr_cols"
    # Packed (#154 part B3a, route A): every test here reads through the public
    # ShardedArchive and none inspects loose members, so the container is incidental --
    # the subject is CSR column/indptr validation in materialize + the Rust decoder.
    write_archive(ad.AnnData(X=X), out, n_shards=2)
    return out


def test_x_from_result_rejects_out_of_range_csr_column(tmp_path):
    from cellstream import ShardedArchive
    from cellstream.v2.materialize import resolve_plan

    out = _write_small_archive(tmp_path)
    ar = ShardedArchive(out)
    plan = resolve_plan(
        ar.manifest,
        container="csr",
        data_dtype="float32",
        index_dtype="int32",
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    result = (
        np.array([1, 2], dtype=np.float32),
        np.array([0, ar.n_vars], dtype=np.int32),
        np.array([0, 2], dtype=np.int64),
        None,
    )
    with pytest.raises(ValueError, match="out of range"):
        ar._x_from_result(result, plan, n_obs_out=1)


def test_rust_decode_v2_csr_rejects_out_of_range_column(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream.v2 import _rust_dispatch
    from cellstream.v2._rust_dispatch import rust_available

    if not rust_available():
        pytest.skip("Rust core disabled")

    # PACKED, via shard_paths_and_bases (route J). The prior revision of this test kept a
    # DIRECTORY fixture that B3a itself had introduced, on the reasoning that driving
    # decode_v2_csr with paths+bases "is the directory layout by definition". It is not: a packed
    # archive gives the same file n times at each shard's 4 KB-aligned offset, which is exactly
    # what production builds (v2/reader_cpu.py:385-386). Surfaced by Copilot noticing the stale
    # "_dir" leaf name next door.
    out = _write_small_archive(tmp_path)
    m = archive_manifest(out)
    n_shards = int(m["n_shards"])
    row_offsets = m["row_offsets"]
    row_lens = [int(row_offsets[i + 1] - row_offsets[i]) for i in range(n_shards)]
    nnz_per_shard = list(map(int, m["nnz_per_shard"]))
    total_nnz = int(m["total_nnz"])
    n_obs = int(m["n_obs_total"])
    paths, bases = shard_paths_and_bases(out)
    assert len(set(paths)) == 1 and all(b > 0 for b in bases)  # really packed, nonzero offsets
    args = dict(
        paths=paths,
        bases=bases,
        out_lens=nnz_per_shard,
        row_lens=row_lens,
        shard_rows=row_lens,
        shard_nnz=nnz_per_shard,
        sel_starts=[0] * n_shards,
        sel_stops=row_lens,
        row_idx_flat=np.empty(0, dtype=np.int64),
        out_data=np.empty(total_nnz, dtype=np.float32),
        out_indices=np.empty(total_nnz, dtype=np.int32),
        out_indptr=np.zeros(n_obs + 1, dtype=np.int64),
        n_vars=4,
        n_threads=2,
    )
    with pytest.raises(ValueError, match="column index"):
        _rust_dispatch.decode_v2_csr(**args)


# ---------------------------------------------------------------------------
# ultrareview L4 — CSR-from-shm assembly consistency in _x_from_result
# ---------------------------------------------------------------------------


def _csr_plan(ar):
    from cellstream.v2.materialize import resolve_plan

    return resolve_plan(
        ar.manifest,
        container="csr",
        data_dtype="float32",
        index_dtype="int32",
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )


def test_x_from_result_rejects_indptr_length_mismatch(tmp_path):
    from cellstream import ShardedArchive

    out = _write_small_archive(tmp_path)
    ar = ShardedArchive(out)
    result = (
        np.array([1, 2], dtype=np.float32),  # data (len 2)
        np.array([0, 1], dtype=np.int32),  # indices in range
        np.array([0, 1, 2], dtype=np.int64),  # indptr len 3 != n_obs_out+1 (== 2)
        None,
    )
    with pytest.raises(ValueError, match="indptr"):
        ar._x_from_result(result, _csr_plan(ar), n_obs_out=1)


def test_x_from_result_rejects_indptr_tail_mismatch(tmp_path):
    from cellstream import ShardedArchive

    out = _write_small_archive(tmp_path)
    ar = ShardedArchive(out)
    result = (
        np.array([1, 2], dtype=np.float32),  # data (len 2)
        np.array([0, 1], dtype=np.int32),  # indices in range
        np.array([0, 5], dtype=np.int64),  # indptr[-1]=5 != len(data)=2
        None,
    )
    with pytest.raises(ValueError, match="indptr"):
        ar._x_from_result(result, _csr_plan(ar), n_obs_out=1)


def test_x_from_result_rejects_data_indices_length_mismatch(tmp_path):
    from cellstream import ShardedArchive

    out = _write_small_archive(tmp_path)
    ar = ShardedArchive(out)
    result = (
        np.array([1, 2, 3], dtype=np.float32),  # data (len 3)
        np.array([0, 1], dtype=np.int32),  # indices (len 2) != len(data)
        np.array([0, 3], dtype=np.int64),  # indptr[-1]=3 == len(data)
        None,
    )
    with pytest.raises(ValueError, match="indptr|indices"):
        ar._x_from_result(result, _csr_plan(ar), n_obs_out=1)


def test_x_from_result_rejects_nonzero_indptr_start(tmp_path):
    from cellstream import ShardedArchive

    out = _write_small_archive(tmp_path)
    ar = ShardedArchive(out)
    result = (
        np.array([1, 2], dtype=np.float32),
        np.array([0, 1], dtype=np.int32),
        np.array([1, 2], dtype=np.int64),  # indptr[0]=1 != 0
        None,
    )
    with pytest.raises(ValueError, match="indptr"):
        ar._x_from_result(result, _csr_plan(ar), n_obs_out=1)


def test_x_from_result_rejects_decreasing_indptr(tmp_path):
    from cellstream import ShardedArchive

    out = _write_small_archive(tmp_path)
    ar = ShardedArchive(out)
    result = (
        np.array([5], dtype=np.float32),  # data len 1
        np.array([0], dtype=np.int32),  # indices len 1
        np.array([0, 2, 1], dtype=np.int64),  # decreasing at the end; indptr[-1]=1==len(data)
        None,
    )
    with pytest.raises(ValueError, match="indptr"):
        ar._x_from_result(result, _csr_plan(ar), n_obs_out=2)
