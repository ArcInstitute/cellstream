"""Native v2 writer via write_sharded(adata_or_path, out, format='v2')."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellstream.v2 import _rust_dispatch
from cellstream.v2 import writer as _w
from cellstream.v2.writer import write_v2_archive


def _make_test_adata(seed=42, n_obs=1000, n_vars=500, density=0.05):
    rng = np.random.default_rng(seed)
    nnz = int(n_obs * n_vars * density)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, 50, size=nnz, dtype=np.uint16)
    csr_raw = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    csr = sp.csr_matrix(csr_raw.shape, dtype=np.float32)
    csr.data = csr_raw.data.astype(np.float32)
    csr.indices = csr_raw.indices.astype(np.int32)
    csr.indptr = csr_raw.indptr.astype(np.int64)
    return ad.AnnData(X=csr)


def test_bytes_per_nnz_keeps_4_for_u16u16():
    assert _w._bytes_per_nnz(np.uint16, np.uint16) == 4


def test_bytes_per_nnz_real_widths_otherwise():
    assert _w._bytes_per_nnz(np.float32, np.uint16) == 6  # 4 + 2
    assert _w._bytes_per_nnz(np.float64, np.uint16) == 10  # 8 + 2
    assert _w._bytes_per_nnz(np.float64, np.uint32) == 12  # 8 + 4
    assert _w._bytes_per_nnz(np.uint16, np.uint32) == 6  # 2 + 4


def test_decide_dense_dtypes_integral_on_disk_uint32_min_uint16():
    D = np.array([[0, 1, 2], [3, 0, 65535]], dtype=np.float32)  # max 65535 -> min uint16
    od_d, od_i, min_d, min_i = _w._decide_dense_on_disk_dtypes(D, n_vars=3)
    assert (od_d, od_i) == (np.dtype(np.uint32), np.dtype(np.uint32))
    assert (min_d, min_i) == (np.dtype(np.uint16), np.dtype(np.uint16))


def test_decide_dense_dtypes_fractional_keeps_native_and_wide_index():
    D = np.array([[0.0, 1.5], [2.0, 0.0]], dtype=np.float64)
    od_d, od_i, min_d, min_i = _w._decide_dense_on_disk_dtypes(D, n_vars=100_000)
    assert (od_d, min_d) == (np.dtype(np.float64), np.dtype(np.float64))
    assert od_i == np.dtype(np.uint32)
    assert min_i == np.dtype(np.uint32)  # n_vars-1 > 65535


def _f64_csr_adata(n_obs=2000, n_var=50, density=0.5, seed=1):
    X = sp.random(n_obs, n_var, density=density, format="csr", random_state=seed)
    X.data = (X.data + 0.123).astype(np.float64)  # fractional -> forces float64 on disk
    return ad.AnnData(X=X)


def test_f64_csr_shards_sized_by_real_bytes_per_nnz(tmp_path):
    a = _f64_csr_adata()
    nnz = int(a.X.nnz)
    target = 50_000  # small target to force multiple shards
    out = tmp_path / "f64"
    write_v2_archive(a, out, target_shard_bytes=target, codec="v2-byte-filter")
    import json
    import math

    man = json.loads((out / "manifest.json").read_text())
    assert man["x_data_dtype_on_disk"] == "float64"
    assert man["n_shards"] == max(1, math.ceil(nnz * 10 / target))  # 8 (f64) + 2 (u16 idx)


def test_integer_csr_shard_count_unchanged(tmp_path):
    # Integral, in-range counts -> uint32 on disk, but sized by the uint16 proxy
    # (bytes_per_nnz stays 4) so the shard count is unchanged vs the legacy uint16 era.
    X = sp.random(2000, 50, density=0.5, format="csr", random_state=2)
    X.data = np.floor(X.data * 100).astype(np.float32)  # integral 0..99
    a = ad.AnnData(X=X)
    nnz = int(X.nnz)
    target = 50_000
    out = tmp_path / "ints"
    write_v2_archive(a, out, target_shard_bytes=target, codec="v2-byte-filter")
    import json
    import math

    man = json.loads((out / "manifest.json").read_text())
    assert man["x_data_dtype_on_disk"] == "uint32"
    assert man["x_data_dtype_min"] == "uint8"  # max 99 -> uint8 (sizing still uses 4)
    assert man["n_shards"] == max(1, math.ceil(nnz * 4 / target))


@pytest.mark.skipif(not _rust_dispatch.rust_available(), reason="needs rust .so")
def test_dense_f64_shards_sized_by_real_bytes_per_nnz(tmp_path):
    rng = np.random.default_rng(3)
    D = rng.random((2000, 50), dtype=np.float64)  # fractional -> float64 on disk
    D[D < 0.5] = 0.0  # ~50% zeros for the dense encoder
    a = ad.AnnData(X=D)
    nnz = int(np.count_nonzero(D))
    target = 50_000
    out = tmp_path / "dense_f64"
    write_v2_archive(a, out, target_shard_bytes=target, codec="v2-byte-filter")
    import json
    import math

    man = json.loads((out / "manifest.json").read_text())
    assert man["x_data_dtype_on_disk"] == "float64"
    assert man["n_shards"] == max(1, math.ceil(nnz * 10 / target))


def test_grouped_inmem_f64_planner_uses_real_bytes_per_nnz(tmp_path):
    n_obs, n_var = 4000, 40
    X = sp.random(n_obs, n_var, density=0.4, format="csr", random_state=4)
    X.data = (X.data + 0.7).astype(np.float64)  # fractional
    obs = __import__("pandas").DataFrame({"grp": np.arange(n_obs).astype(str)})
    a = ad.AnnData(X=X, obs=obs)
    out_small = tmp_path / "g_small"
    # With one group per row, group alignment cannot cap the f64 shard count
    # below the dtype-aware working-set lower bound.
    write_v2_archive(
        a,
        out_small,
        group_by="grp",
        target_shard_bytes=20_000,
        codec="v2-byte-filter",
    )
    import json
    import math

    man = json.loads((out_small / "manifest.json").read_text())
    nnz = int(X.nnz)
    expected_lower_bound = max(1, math.ceil(nnz * 10 / 20_000))
    assert man["x_data_dtype_on_disk"] == "float64"
    assert man["n_shards"] >= expected_lower_bound


def test_grouped_inmem_csc_input_sizes_without_crash(tmp_path):
    # A CSC source.X must not break the dtype-aware grouped sizing: a CSC matrix's
    # .indices are row indices and COO has no .indices, so the planner must coerce
    # to CSR before reading .data/.indices (Gemini #122). The encode path already
    # coerces, so the full grouped write round-trips to a float64 archive.
    n_obs, n_var = 600, 40
    X = sp.random(n_obs, n_var, density=0.4, format="csr", random_state=7)
    X.data = (X.data + 0.5).astype(np.float64)  # fractional -> float64 on disk
    obs = __import__("pandas").DataFrame({"grp": (np.arange(n_obs) % 5).astype(str)})
    a = ad.AnnData(X=X.tocsc(), obs=obs)  # CSC input
    out = tmp_path / "g_csc"
    write_v2_archive(a, out, group_by="grp", target_shard_bytes=20_000, codec="v2-byte-filter")
    import json

    man = json.loads((out / "manifest.json").read_text())
    assert man["x_data_dtype_on_disk"] == "float64"


def test_write_sharded_v2_round_trips(tmp_path):
    """write_sharded(adata, out, format='v2') → ShardedArchive(out).to_anndata() == adata."""
    adata = _make_test_adata(seed=1)
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "v2_arch"
    write_sharded(adata, out, format="v2", n_shards=4, n_workers=2)
    sh = ShardedArchive(out)
    assert sh.schema_version == 2
    back = sh.to_anndata(cast_back=True, n_workers=2)
    np.testing.assert_array_equal(back.X.toarray(), adata.X.toarray())


def test_write_sharded_v2_path_input(tmp_path):
    """write_sharded(src_path, out, format='v2') accepts a source .h5ad path.

    Not a streaming test despite the old wording: with no group_by, write_sharded resolves the
    source through its ``source = ad.read_h5ad(adata)`` arm before calling the writer; only a
    GROUPED path source is handed through as a path and read backed (#76).
    See tests/test_write_sharded.py::test_write_sharded_path_input for the measurement.
    """
    adata = _make_test_adata(seed=2)
    src = tmp_path / "src.h5ad"
    adata.write_h5ad(src)
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "v2_arch"
    write_sharded(str(src), out, format="v2", n_shards=2, n_workers=2)
    back = ShardedArchive(out).to_anndata(cast_back=True, n_workers=2)
    np.testing.assert_array_equal(back.X.toarray(), adata.X.toarray())


def test_format_default_is_v2(tmp_path):
    """As of v0.3.0, write_sharded(adata, out) with no format= defaults to v2."""
    adata = _make_test_adata(seed=3, n_obs=200, n_vars=80)
    from cellstream import ShardedArchive, write_sharded

    out = tmp_path / "default_arch"
    write_sharded(adata, out, n_shards=2, n_workers=2)
    sh = ShardedArchive(out)
    assert sh.schema_version == 2


def test_lossy_cast_fires_before_any_disk_write(tmp_path):
    """A malformed (negative) column index raises before disk is touched.

    Note: out-of-uint16-range *data* (e.g. 70000.0) is no longer an error — it
    is stored at the source's native float/wide width (narrow-if-safe). The
    only remaining fail-loud condition is a malformed index on the scan-based
    n_vars > 65536 branch (negative, or beyond uint32), which must still fire
    BEFORE any shard/header/obs is written.
    """
    import scipy.sparse as sp

    adata = _make_test_adata(seed=4, n_obs=10, n_vars=100_000, density=0.00001)
    adata.X = sp.csr_matrix(adata.X)
    adata.X.indices[0] = -1  # malformed CSR index -> fail loud before any write
    from cellstream import write_sharded
    from cellstream.errors import LossyCastError

    out = tmp_path / "v2_arch"
    with pytest.raises(LossyCastError):
        write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    # No disk artifacts should exist — the index check fires BEFORE
    # write_header and write_obs_parquet. A packed archive is a single file, so its
    # absence IS the whole assertion; the old per-member checks were directory-shaped
    # and would silently pass against a packed path that has no members (#154 part B).
    assert not out.exists()


def test_writer_existing_archive_blocks_without_overwrite(tmp_path):
    """Writing into an existing v2 archive without overwrite=True raises."""
    adata = _make_test_adata(seed=5)
    from cellstream import write_sharded
    from cellstream.errors import ArchiveExistsError

    out = tmp_path / "v2_arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    with pytest.raises(ArchiveExistsError):
        write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)


def test_writer_emits_codec_string_matching_spec(tmp_path):
    """Manifest's `codec` field reads the default v2 codec (`v2-byte-filter`
    as of v0.3.0) and the shard template / fingerprint match the spec.
    """
    adata = _make_test_adata(seed=6, n_obs=100, n_vars=50)
    from cellstream import write_sharded

    out = tmp_path / "v2_arch"
    write_sharded(adata, out, format="v2", n_shards=2, n_workers=2)
    from tests._archive_helpers import archive_manifest

    m = archive_manifest(out)
    assert m["codec"] == "v2-byte-filter"
    assert m["shard_filename_template"] == "shard_{:04d}.shx"
    # writer_fingerprint is a dict (matches v0.1 shape)
    assert isinstance(m["writer_fingerprint"], dict)
    assert m["writer_fingerprint"]["writer_kind"] == "v2-native"


def test_default_container_is_packed_for_v2(tmp_path):
    """v0.4.0: write_sharded with no container= defaults to packed for v2 (the
    default format) — a single .shardpack file, auto-detected on read."""
    from cellstream import ShardedArchive, write_sharded
    from cellstream.packed.reader import is_packed_file

    adata = _make_test_adata(seed=9, n_obs=100, n_vars=40)
    out = tmp_path / "default_v2"
    write_sharded(adata, out, n_shards=2, n_workers=2)  # NO format, NO container
    assert is_packed_file(out)  # single packed file, not a dir
    assert not (out / "manifest.json").exists()  # not a directory archive
    back = ShardedArchive(out).to_anndata(cast_back=True, n_workers=2)
    np.testing.assert_array_equal(back.X.toarray(), adata.X.toarray())


# test_default_container_is_directory_for_v1 lived here: the container None-sentinel used
# to resolve PER FORMAT (v2 -> packed, v1 -> directory, because packed is v2-only). With
# the v1 writer removed in #154 the sentinel has one arm, covered by the test above.
