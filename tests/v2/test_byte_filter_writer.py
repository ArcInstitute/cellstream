"""Writer codec dispatch: codec= -> ShxHeader.shuffle marker, byte-exact round-trip.

#154 part B3a, route D: ``write_v2_archive`` is the SUBJECT of every test here, so none of
these calls may become ``write_archive`` -- that would silently move the subject to the public
writer, which is B1's own recurring defect (four consecutive codex rounds to close). The five
tests that additionally read through the PUBLIC reader now ``pack_directory`` the directory the
internal writer produced and read the packed copy. ``_pack_directory`` is a pure byte concat
(``packed/writer.py:111``), so the shards under test are byte-identical inside the packed
container: the internal writer stays the subject AND the public reader stays in the round trip,
which reading through ``read_v2_to_host`` instead would have given up.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from tests._archive_helpers import pack_directory


def _make_adata(seed=1, n_obs=600, n_vars=400, density=0.06):
    rng = np.random.default_rng(seed)
    nnz = int(n_obs * n_vars * density)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, 50, size=nnz, dtype=np.uint16)
    raw = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    csr = sp.csr_matrix(raw.shape, dtype=np.float32)
    csr.data = raw.data.astype(np.float32)
    csr.indices = raw.indices.astype(np.int32)
    csr.indptr = raw.indptr.astype(np.int64)
    return ad.AnnData(X=csr)


def _make_float_adata(seed=1, n_obs=600, n_vars=400, density=0.06):
    """Genuinely non-integral float32 CSR — log1p values cannot narrow to uint16,
    so the on-disk data dtype stays float32 (exercises the #142 rawdata path)."""
    rng = np.random.default_rng(seed)
    nnz = int(n_obs * n_vars * density)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = np.log1p(rng.integers(1, 50, size=nnz)).astype(np.float32)
    raw = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    raw.data = raw.data.astype(np.float32)
    raw.indices = raw.indices.astype(np.int32)
    raw.indptr = raw.indptr.astype(np.int64)
    return ad.AnnData(X=raw)


def test_write_v2_float_python_uses_rawdata_marker(tmp_path, monkeypatch):
    """#142: a float32 archive written by the Python encoder carries the rawdata
    shuffle marker (data = raw zstd) and keeps float32 on disk."""
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    from cellstream.v2.codec import BYTE_FILTER_RAWDATA_SHUFFLE_NAME
    from cellstream.v2.format import parse_shx_header
    from cellstream.v2.writer import write_v2_archive

    adata = _make_float_adata(seed=2)
    out = tmp_path / "f32_arch"
    write_v2_archive(adata, out, n_shards=4, n_workers=2, codec="v2-byte-filter")

    shards = sorted(out.glob("shard_*.shx"))
    assert shards
    for shard in shards:
        with shard.open("rb") as fh:
            hdr = parse_shx_header(fh)
        assert hdr.shuffle == BYTE_FILTER_RAWDATA_SHUFFLE_NAME
        assert hdr.data.dtype == "float32"
        # indices/indptr streams keep their byte-filter dtypes (unchanged)
        assert hdr.indptr.dtype == "int64"


def test_write_v2_integer_python_keeps_old_marker(tmp_path, monkeypatch):
    """Integer-valued data routes to the integer stream -> keeps
    BYTE_FILTER_SHUFFLE_NAME (the rawdata gate is float-only). Integer data is now
    stored as uint32 on disk."""
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    from cellstream.v2.codec import BYTE_FILTER_SHUFFLE_NAME
    from cellstream.v2.format import parse_shx_header
    from cellstream.v2.writer import write_v2_archive

    adata = _make_adata(seed=2)  # integer-valued float32 -> uint32 on disk
    out = tmp_path / "int_arch"
    write_v2_archive(adata, out, n_shards=4, n_workers=2, codec="v2-byte-filter")
    for shard in sorted(out.glob("shard_*.shx")):
        with shard.open("rb") as fh:
            hdr = parse_shx_header(fh)
        assert hdr.shuffle == BYTE_FILTER_SHUFFLE_NAME
        assert hdr.data.dtype == "uint32"


def test_write_v2_float_python_round_trip(tmp_path, monkeypatch):
    """#142: a float32 rawdata archive round-trips byte-exact through the pure
    Python encode+decode path (exercises the _decode_stream rawdata branch)."""
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    from cellstream import ShardedArchive
    from cellstream.v2.writer import write_v2_archive

    adata = _make_float_adata(seed=2)
    out = tmp_path / "f32_rt"
    write_v2_archive(adata, out, n_shards=4, n_workers=2, codec="v2-byte-filter")
    back = ShardedArchive(pack_directory(out, tmp_path / "f32_rt.shad")).to_anndata(n_workers=2)
    np.testing.assert_array_equal(back.X.toarray(), adata.X.toarray())
    assert back.X.dtype == np.float32


def test_float_rawdata_archive_reads_with_rust_enabled(tmp_path, monkeypatch):
    """#142 gate fix (Gemini PR #143 r1): a rawdata archive (written by the
    Python encoder) must read correctly with the Rust core ENABLED — the read
    gate keys on the per-shard header marker, so it routes to the Python decoder
    instead of handing rawdata shards to a Rust core that can't decode them."""
    import cellstream.v2._rust_dispatch as rd
    from cellstream import ShardedArchive
    from cellstream.v2.writer import write_v2_archive

    # Write with the Python encoder so the archive carries the rawdata marker.
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    adata = _make_float_adata(seed=3)
    out = tmp_path / "f32_rustread"
    write_v2_archive(adata, out, n_shards=4, n_workers=2, codec="v2-byte-filter")

    # Pack BEFORE re-enabling Rust: _pack_directory is a byte concat, so the rawdata
    # marker the read gate keys on is carried through unchanged.
    packed = pack_directory(out, tmp_path / "f32_rustread.shad")

    # Re-enable Rust for the read; skip if the Rust core isn't built (CI fallback).
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    if not rd.rust_available():
        pytest.skip("Rust core not built; gate fix is only meaningful with Rust enabled")
    back = ShardedArchive(packed).to_anndata(n_workers=2)
    np.testing.assert_array_equal(back.X.toarray(), adata.X.toarray())
    assert back.X.dtype == np.float32


def test_resolve_v2_shuffle_maps_names():
    from cellstream.v2.codec import BYTE_FILTER_SHUFFLE_NAME
    from cellstream.v2.writer import _resolve_v2_shuffle

    assert _resolve_v2_shuffle("v2-byte-filter") == BYTE_FILTER_SHUFFLE_NAME
    assert _resolve_v2_shuffle("bitshuffle") == "bitshuffle"
    assert _resolve_v2_shuffle("blosc-zstd-1-u16u16") == "bitshuffle"
    assert _resolve_v2_shuffle("blosc-zstd-1-u16u16-datashuf") == "bitshuffle"
    # the v0.2-documented v2 codec name + the manifest codec value must resolve
    # to bitshuffle, not regress to ValueError (Copilot design review)
    assert _resolve_v2_shuffle("zstd-1-u16u16-bitshuffle") == "bitshuffle"
    with pytest.raises(ValueError, match="Unknown v2 codec"):
        _resolve_v2_shuffle("nope-not-a-codec")


def test_write_v2_archive_byte_filter_round_trips(tmp_path):
    from cellstream import ShardedArchive
    from cellstream.v2.codec import BYTE_FILTER_SHUFFLE_NAME
    from cellstream.v2.format import parse_shx_header
    from cellstream.v2.writer import write_v2_archive

    adata = _make_adata(seed=2)
    out = tmp_path / "bf_arch"
    write_v2_archive(adata, out, n_shards=4, n_workers=2, codec="v2-byte-filter")

    # round-trip byte-exact, through the public reader on a packed copy
    back = ShardedArchive(pack_directory(out, tmp_path / "bf_arch.shad")).to_anndata(
        cast_back=True, n_workers=2
    )
    np.testing.assert_array_equal(back.X.toarray(), adata.X.toarray())

    # every shard self-describes as byte-filter. parse_shx_header takes any
    # binary file handle with .read(n), so a plain open() works — no need to
    # import the private mmap wrapper (Copilot design review).
    for shard in sorted(out.glob("shard_*.shx")):
        with shard.open("rb") as fh:
            hdr = parse_shx_header(fh)
        assert hdr.shuffle == BYTE_FILTER_SHUFFLE_NAME

    # manifest codec reflects the byte-filter codec (informational; reader does
    # NOT depend on it — it dispatches on header.shuffle). The stored name is the
    # public preset 'v2-byte-filter' (per 29f7f0f).
    m = json.loads((out / "manifest.json").read_text())
    assert m["codec"] == "v2-byte-filter"


def test_write_v2_archive_default_unchanged(tmp_path):
    """Default codec still writes bitshuffle shards + the legacy manifest codec."""
    from cellstream.v2.writer import write_v2_archive

    adata = _make_adata(seed=3)
    out = tmp_path / "default_arch"
    write_v2_archive(adata, out, n_shards=2, n_workers=2)  # no codec= -> default
    m = json.loads((out / "manifest.json").read_text())
    assert m["codec"] == "zstd-1-u16u16-bitshuffle"


def test_write_v2_archive_unknown_codec_raises(tmp_path):
    from cellstream.v2.writer import write_v2_archive

    adata = _make_adata(seed=4)
    with pytest.raises(ValueError, match="Unknown v2 codec"):
        write_v2_archive(adata, tmp_path / "x", n_shards=2, n_workers=1, codec="bogus")
    # fail-loud BEFORE writing: no manifest left behind
    assert not (tmp_path / "x" / "manifest.json").exists()


def test_byte_filter_and_bitshuffle_read_in_same_process(tmp_path):
    """Two archives — one byte-filter, one default — both round-trip in the same
    process (the self-describing property)."""
    from cellstream import ShardedArchive
    from cellstream.v2.writer import write_v2_archive

    adata = _make_adata(seed=6)
    bf = tmp_path / "bf"
    bs = tmp_path / "bs"
    write_v2_archive(adata, bf, n_shards=3, n_workers=2, codec="v2-byte-filter")
    write_v2_archive(adata, bs, n_shards=3, n_workers=2)
    back_bf = ShardedArchive(pack_directory(bf, tmp_path / "bf.shad")).to_anndata(
        cast_back=True, n_workers=2
    )
    back_bs = ShardedArchive(pack_directory(bs, tmp_path / "bs.shad")).to_anndata(
        cast_back=True, n_workers=2
    )
    np.testing.assert_array_equal(back_bf.X.toarray(), adata.X.toarray())
    np.testing.assert_array_equal(back_bs.X.toarray(), adata.X.toarray())


def test_byte_filter_multichunk_round_trips(tmp_path):
    """A shard with > CHUNK_SIZE_ELEMENTS (1,048,576) nnz forces multi-chunk
    encode + decode-and-concatenate; per-chunk BYTEDELTA on indices must
    round-trip across chunk boundaries (the codec/reader gap the small-array
    tests miss)."""
    from cellstream import ShardedArchive
    from cellstream.v2.writer import CHUNK_SIZE_ELEMENTS, write_v2_archive

    n_obs, per_row, n_vars = 1200, 1000, 2000
    nnz = n_obs * per_row
    assert nnz > CHUNK_SIZE_ELEMENTS  # data + indices each span > 1 chunk

    rng = np.random.default_rng(21)
    # One sorted per-row template tiled across rows: a valid CSR (sorted-within
    # -row) indices array, far cheaper than a per-row Python loop. The row->row
    # wrap (high last col -> low first col) still exercises the BYTEDELTA mod-256
    # wraparound, and 1.2M elements still cross the chunk boundary. (Copilot review)
    template = np.sort(rng.choice(n_vars, size=per_row, replace=False)).astype(np.int32)
    indices = np.tile(template, n_obs)
    indptr = np.arange(0, nnz + 1, per_row, dtype=np.int64)
    data = rng.integers(1, 5000, size=nnz).astype(np.float32)
    X = sp.csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))
    adata = ad.AnnData(X=X)

    out = tmp_path / "mc"
    write_v2_archive(adata, out, n_shards=1, n_workers=1, codec="v2-byte-filter")
    back = ShardedArchive(pack_directory(out, tmp_path / "mc.shad")).to_anndata(
        cast_back=True, n_workers=1
    )
    np.testing.assert_array_equal(back.X.toarray(), adata.X.toarray())
