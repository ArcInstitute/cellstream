"""v2.reader_cpu — round-trip a synthetic v2 archive built by hand."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _hand_build_v2_archive(out_dir, csr, *, n_shards=2, chunk_size=1024):
    from cellstream.manifest import dump_manifest
    from cellstream.v2 import codec
    from cellstream.v2.format import ChunkDescriptor, ShxHeader, StreamSpec, write_shx_file

    n_obs, n_vars = csr.shape
    out_dir.mkdir(parents=True, exist_ok=True)
    row_offsets = [int(round(i * n_obs / n_shards)) for i in range(n_shards + 1)]
    nnz_per_shard = []
    for s in range(n_shards):
        r0, r1 = row_offsets[s], row_offsets[s + 1]
        sub = csr[r0:r1]
        streams_bytes, streams_sizes = [], []
        n_chunks_per_stream = []
        descs_per_stream = []
        for arr, dtype in [(sub.data, np.uint16), (sub.indices, np.uint16), (sub.indptr, np.int64)]:
            arr = arr.astype(dtype)
            chunks_bytes, sizes, descs = [], [], []
            if arr.size == 0:
                enc = codec.bitshuffle_zstd_encode(arr, level=1)
                chunks_bytes.append(enc)
                sizes.append(len(enc))
                descs.append(ChunkDescriptor(0, len(enc), 0))
            else:
                for i in range(0, arr.size, chunk_size):
                    slab = arr[i : i + chunk_size]
                    enc = codec.bitshuffle_zstd_encode(slab, level=1)
                    chunks_bytes.append(enc)
                    sizes.append(len(enc))
                    descs.append(ChunkDescriptor(0, len(enc), int(slab.size * slab.dtype.itemsize)))
            streams_bytes.append(b"".join(chunks_bytes))
            streams_sizes.append(sizes)
            n_chunks_per_stream.append(len(chunks_bytes))
            descs_per_stream.append(descs)

        header = ShxHeader(
            schema_version=2,
            shard_idx=s,
            n_rows=int(r1 - r0),
            n_vars=int(n_vars),
            nnz=int(sub.nnz),
            compression="zstd",
            shuffle="bitshuffle",
            compression_level=1,
            chunk_size_elements=chunk_size,
            data=StreamSpec("uint16", n_chunks_per_stream[0], descs_per_stream[0]),
            indices=StreamSpec("uint16", n_chunks_per_stream[1], descs_per_stream[1]),
            indptr=StreamSpec("int64", n_chunks_per_stream[2], descs_per_stream[2]),
        )
        write_shx_file(
            out_dir / f"shard_{s:04d}.shx",
            header,
            chunk_streams=streams_bytes,
            fill_chunk_offsets=True,
            per_stream_chunk_sizes=streams_sizes,
        )
        nnz_per_shard.append(int(sub.nnz))

    dump_manifest(
        out_dir / "manifest.json",
        {
            "schema_version": 2,
            "shard_filename_template": "shard_{:04d}.shx",
            "n_shards": n_shards,
            "n_obs_total": n_obs,
            "n_vars": n_vars,
            "row_offsets": row_offsets,
            "nnz_per_shard": nnz_per_shard,
            "total_nnz": int(csr.nnz),
            "x_data_dtype_on_disk": "uint16",
            "x_indices_dtype_on_disk": "uint16",
            "codec": "zstd-1-u16u16-bitshuffle",
            "chunk_size_elements": chunk_size,
            "writer_fingerprint": "test_hand_built",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )


def test_full_read_round_trip(tmp_path, mid_csr):
    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=4, chunk_size=4096)
    from cellstream.v2.reader_cpu import read_v2_to_host

    data, indices, indptr = read_v2_to_host(arch, cast_back=False, n_workers=2)
    np.testing.assert_array_equal(data, mid_csr.data)
    np.testing.assert_array_equal(indices, mid_csr.indices)
    np.testing.assert_array_equal(indptr, mid_csr.indptr)


def test_cast_back_true_widens_to_float32_and_int32(tmp_path, mid_csr):
    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=2)
    from cellstream.v2.reader_cpu import read_v2_to_host

    data, indices, _ = read_v2_to_host(arch, cast_back=True, n_workers=2)
    assert data.dtype == np.float32
    assert indices.dtype == np.int32
    np.testing.assert_array_equal(data, mid_csr.data.astype(np.float32))
    np.testing.assert_array_equal(indices.astype(np.int64), mid_csr.indices.astype(np.int64))


def test_subset_contiguous(tmp_path, mid_csr):
    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=4, chunk_size=2048)
    from cellstream.v2.reader_cpu import read_v2_to_host

    data, indices, indptr = read_v2_to_host(
        arch, cast_back=False, n_workers=2, row_start=100, row_stop=350
    )
    expected = mid_csr[100:350]
    np.testing.assert_array_equal(data, expected.data)
    np.testing.assert_array_equal(indices, expected.indices)
    np.testing.assert_array_equal(indptr, expected.indptr)


def test_subset_bool_mask(tmp_path, mid_csr):
    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=2)
    rng = np.random.default_rng(3)
    mask = rng.random(mid_csr.shape[0]) > 0.7
    from cellstream.v2.reader_cpu import read_v2_to_host

    data, indices, indptr = read_v2_to_host(arch, cast_back=False, n_workers=2, row_mask=mask)
    expected = mid_csr[mask]
    np.testing.assert_array_equal(data, expected.data)
    np.testing.assert_array_equal(indices, expected.indices)
    np.testing.assert_array_equal(indptr, expected.indptr)


def test_indptr_int64_avoids_uint16_overflow(tmp_path):
    rng = np.random.default_rng(1)
    n_obs, n_vars = 1000, 200
    nnz = 90_000
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, 5, size=nnz, dtype=np.uint16)
    csr_raw = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    csr = sp.csr_matrix(csr_raw.shape, dtype=np.uint16)
    csr.data = csr_raw.data.astype(np.uint16)
    csr.indices = csr_raw.indices.astype(np.uint16)
    csr.indptr = csr_raw.indptr.astype(np.int64)
    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, csr, n_shards=2, chunk_size=8192)
    from cellstream.v2.reader_cpu import read_v2_to_host

    _, _, indptr = read_v2_to_host(arch, cast_back=False, n_workers=2)
    assert indptr.dtype == np.int64
    assert indptr[-1] == csr.nnz


def test_subset_int_array(tmp_path, mid_csr):
    """sh[int_array] equivalent (unsorted input — reader sorts internally)."""
    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=2)
    rng = np.random.default_rng(11)
    idx = rng.choice(mid_csr.shape[0], size=100, replace=False)
    from cellstream.v2.reader_cpu import read_v2_to_host

    data, indices, indptr = read_v2_to_host(arch, cast_back=False, n_workers=2, row_indices=idx)
    expected = mid_csr[np.sort(idx)]
    np.testing.assert_array_equal(data, expected.data)
    np.testing.assert_array_equal(indices, expected.indices)
    np.testing.assert_array_equal(indptr, expected.indptr)


def test_n_workers_1_single_process_path(tmp_path, mid_csr):
    """n_workers=1 takes the no-pool branch."""
    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=4)
    from cellstream.v2.reader_cpu import read_v2_to_host

    data, indices, indptr = read_v2_to_host(arch, cast_back=False, n_workers=1)
    np.testing.assert_array_equal(data, mid_csr.data)
    np.testing.assert_array_equal(indices, mid_csr.indices)
    np.testing.assert_array_equal(indptr, mid_csr.indptr)


def test_return_shm_bundle_keeps_zero_copy_views(tmp_path, mid_csr):
    """_return_shm_bundle=True returns live views; close_and_unlink invalidates."""
    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=2)
    from cellstream.v2.reader_cpu import read_v2_to_host

    data, indices, indptr, bundle = read_v2_to_host(
        arch, cast_back=False, n_workers=2, _return_shm_bundle=True
    )
    try:
        np.testing.assert_array_equal(data, mid_csr.data)
        np.testing.assert_array_equal(indices, mid_csr.indices)
        np.testing.assert_array_equal(indptr, mid_csr.indptr)
    finally:
        bundle.close_and_unlink()


def test_conflicting_row_selectors_raise(tmp_path, mid_csr):
    """Passing both row_mask and row_start raises."""
    arch = tmp_path / "arch"
    _hand_build_v2_archive(arch, mid_csr, n_shards=2)
    from cellstream.v2.reader_cpu import read_v2_to_host

    mask = np.zeros(mid_csr.shape[0], dtype=bool)
    mask[:10] = True
    import pytest

    with pytest.raises(ValueError, match="at most one"):
        read_v2_to_host(arch, cast_back=False, n_workers=1, row_mask=mask, row_start=0, row_stop=50)


def test_read_min_dtypes_prefers_min_fields():
    from cellstream.v2.reader_cpu import _read_min_dtypes

    m = {
        "x_data_dtype_on_disk": "uint32",
        "x_indices_dtype_on_disk": "uint32",
        "x_data_dtype_min": "uint16",
        "x_indices_dtype_min": "uint16",
    }
    md, mi = _read_min_dtypes(m)
    assert md == np.dtype("uint16") and mi == np.dtype("uint16")
    # the recorded min is what makes the default float32 read a safe (cheap) cast:
    assert np.can_cast(md, np.float32, "safe") is True
    # whereas the raw on-disk uint32 would NOT be float32-safe:
    assert np.can_cast(np.dtype(m["x_data_dtype_on_disk"]), np.float32, "safe") is False


def test_read_min_dtypes_falls_back_to_on_disk():
    from cellstream.v2.reader_cpu import _read_min_dtypes

    # Legacy archive: no *_min keys -> on-disk dtype IS the minimum (uint16 era).
    m = {"x_data_dtype_on_disk": "uint16", "x_indices_dtype_on_disk": "uint16"}
    md, mi = _read_min_dtypes(m)
    assert md == np.dtype("uint16") and mi == np.dtype("uint16")


def test_read_min_dtypes_defaults_when_absent():
    from cellstream.v2.reader_cpu import _read_min_dtypes

    md, mi = _read_min_dtypes({})
    assert md == np.dtype("uint16") and mi == np.dtype("uint16")
