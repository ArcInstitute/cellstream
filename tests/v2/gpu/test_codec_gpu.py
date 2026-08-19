"""codec_gpu: pure-cupy byte-filter inverse, byte-exact vs codec.py (no nvCOMP).

GPU-gated via tests/v2/gpu/conftest.py (cupy + a visible device). cupy is imported
inside the test bodies so collection never fails on a CPU-only host.
"""

from __future__ import annotations

import numpy as np
import pytest
import zstandard as zstd

from cellstream.v2 import codec


def _raw_planes(arr, role):
    """The exact decompressed bytes nvCOMP would emit for this stream/role.

    role: "data" (shuffle-only), "indices"/"indptr" (shuffle+bytedelta),
    "data-raw" (float data: raw zstd, no shuffle/delta).
    """
    if role == "data-raw":
        comp = codec.zstd_only_encode(arr)
    else:
        comp = codec.byte_filter_encode(arr, role=role)
    return zstd.ZstdDecompressor().decompress(comp)


@pytest.mark.parametrize(
    "role,dtype,is_raw",
    [
        ("data", np.uint16, False),
        ("indices", np.uint16, False),
        ("indptr", np.int64, False),
        ("data-raw", np.float32, True),
    ],
)
def test_stream_inverse_matches_codec(role, dtype, is_raw):
    cp = pytest.importorskip("cupy")
    from cellstream.v2 import codec_gpu

    rng = np.random.default_rng(0)
    if is_raw:
        arr = (rng.random(4096) * 10).astype(dtype)
    else:
        arr = rng.integers(0, 500, size=4096).astype(dtype)
    raw = _raw_planes(arr, role)
    d_chunks = [cp.asarray(np.frombuffer(raw, np.uint8))]
    codec_role = "data" if role == "data-raw" else role
    got = codec_gpu.stream_inverse_gpu(
        d_chunks,
        on_disk_dtype=np.dtype(dtype).name,
        eff_dtype=np.dtype(dtype).name,
        role=codec_role,
        is_raw=is_raw,
        chunk_elems=1 << 20,
    )
    assert got.dtype == cp.dtype(dtype)
    np.testing.assert_array_equal(got.get(), arr)


def test_multi_full_chunk_batched_path():
    """Exercise the batched full-chunk path: several full chunks + a tail.

    A small chunk_elems makes 4096 elems span multiple 'full' chunks so the
    (n_full, T_eff, N) batched cumsum+reassemble is used, not just the tail.
    """
    cp = pytest.importorskip("cupy")
    from cellstream.v2 import codec_gpu

    rng = np.random.default_rng(3)
    chunk_elems = 1000
    arr = rng.integers(0, 60000, size=4096).astype(np.uint16)  # 4 full + 96 tail
    # Encode per-chunk exactly as the writer chunks a stream (indices role).
    d_chunks = []
    for start in range(0, arr.size, chunk_elems):
        piece = arr[start : start + chunk_elems]
        raw = _raw_planes(piece, "indices")
        d_chunks.append(cp.asarray(np.frombuffer(raw, np.uint8)))
    got = codec_gpu.stream_inverse_gpu(
        d_chunks,
        on_disk_dtype="uint16",
        eff_dtype="uint16",
        role="indices",
        is_raw=False,
        chunk_elems=chunk_elems,
    )
    np.testing.assert_array_equal(got.get(), arr)


def test_low_plane_u32_to_u16_and_high_plane_guard():
    cp = pytest.importorskip("cupy")
    from cellstream.v2 import codec_gpu

    arr = (np.arange(4096, dtype=np.uint32) * 3).astype(np.uint32)  # fits u16
    raw = _raw_planes(arr, "indices")
    d = [cp.asarray(np.frombuffer(raw, np.uint8))]
    got = codec_gpu.stream_inverse_gpu(
        d,
        on_disk_dtype="uint32",
        eff_dtype="uint16",
        role="indices",
        is_raw=False,
        chunk_elems=1 << 20,
    )
    assert got.dtype == cp.uint16
    np.testing.assert_array_equal(got.get().astype(np.uint32), arr)

    big = np.array([0, 70000, 1], dtype=np.uint32)  # 70000 sets a high plane
    raw2 = _raw_planes(big, "indices")
    d2 = [cp.asarray(np.frombuffer(raw2, np.uint8))]
    with pytest.raises(ValueError):
        codec_gpu.stream_inverse_gpu(
            d2,
            on_disk_dtype="uint32",
            eff_dtype="uint16",
            role="indices",
            is_raw=False,
            chunk_elems=1 << 20,
        )


def test_reassembly_equivalence_large_multichunk_u32_low():
    """Batched full-chunk u32->u16 low-plane path across several full chunks + a
    tail; guards the high-plane 0-D .any() refactor stays byte-exact."""
    cp = pytest.importorskip("cupy")
    from cellstream.v2 import codec_gpu

    rng = np.random.default_rng(7)
    chunk_elems = 1024
    arr = rng.integers(0, 60000, size=5000).astype(np.uint32)  # fits u16; 4 full + tail
    d_chunks = []
    for start in range(0, arr.size, chunk_elems):
        raw = _raw_planes(arr[start : start + chunk_elems], "indices")
        d_chunks.append(cp.asarray(np.frombuffer(raw, np.uint8)))
    got = codec_gpu.stream_inverse_gpu(
        d_chunks,
        on_disk_dtype="uint32",
        eff_dtype="uint16",
        role="indices",
        is_raw=False,
        chunk_elems=chunk_elems,
    )
    assert got.dtype == cp.uint16
    np.testing.assert_array_equal(got.get().astype(np.uint32), arr)
