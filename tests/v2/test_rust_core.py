"""Differential + smoke tests for the Rust encode core (cellstream.v2._rust).

#154 part B3a, route D: the ENCODER is the subject, so every write here stays on
``write_v2_archive`` -- including the one at ``_encode_via``, which lives inside a subprocess
script STRING and is therefore invisible to an AST call count (the plan records this as the
inventory's string-sweep case). The three tests that read through the public reader
``pack_directory()`` the encoder's output first; ``_pack_directory`` is a pure byte concat, so
the encoded shard bytes being compared are unchanged. The two non-gating byte-diff tests and
the glob-based shard-header assertions need no public read and stay on the directories.
"""

from __future__ import annotations

import os
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from tests._archive_helpers import pack_directory


def test_rust_extension_imports_and_selftests():
    rust = pytest.importorskip("cellstream.v2._rust")
    assert rust._selftest() == "ok"


def _toy_adata(n_obs=12, n_vars=5, seed=0, density=0.4):
    rng = np.random.default_rng(seed)
    X = sp.random(n_obs, n_vars, density=density, format="csr", random_state=seed)
    X.data = (rng.integers(1, 50, size=X.nnz)).astype(np.float32)
    obs = {"target_gene": [f"g{i % 3}" for i in range(n_obs)]}
    return ad.AnnData(X=X, obs=pd.DataFrame(obs, index=[f"c{i}" for i in range(n_obs)]))


def _toy_float_adata(n_obs=12, n_vars=5, seed=0, density=0.4):
    """Non-integral float32 CSR — stays float32 on disk (#142 rawdata path)."""
    rng = np.random.default_rng(seed)
    X = sp.random(n_obs, n_vars, density=density, format="csr", random_state=seed)
    X.data = np.log1p(rng.integers(1, 50, size=X.nnz)).astype(np.float32)
    obs = {"target_gene": [f"g{i % 3}" for i in range(n_obs)]}
    return ad.AnnData(X=X, obs=pd.DataFrame(obs, index=[f"c{i}" for i in range(n_obs)]))


def _decode_shx(path):
    """Decode a standalone .shx shard to (data, indices, indptr) via the public
    format + codec helpers — no manifest needed."""
    from cellstream.v2 import codec
    from cellstream.v2 import format as v2fmt

    with open(path, "rb") as fh:
        hdr = v2fmt.parse_shx_header(fh)
        fh.seek(0)
        blob = fh.read()

    def _stream(spec, dtype, role):
        out = []
        for c in spec.chunks:
            comp = blob[c.offset : c.offset + c.compressed_size]
            n = c.decompressed_size // np.dtype(dtype).itemsize
            out.append(codec.byte_filter_decode(comp, dtype=dtype, n_elements=n, role=role))
        return np.concatenate(out) if out else np.array([], dtype=dtype)

    return (
        _stream(hdr.data, np.dtype(hdr.data.dtype), "data"),
        _stream(hdr.indices, np.dtype(hdr.indices.dtype), "indices"),
        _stream(hdr.indptr, np.int64, "indptr"),
    )


def test_rust_encode_shards_inmem_decode_identical(tmp_path):
    rust = pytest.importorskip("cellstream.v2._rust")
    adata = _toy_adata()
    X = adata.X.tocsr()
    rust_dir = tmp_path / "rust"
    rust_dir.mkdir()
    # one shard = all rows in original order (self-contained: compare to source CSR)
    row_lists = [np.arange(adata.n_obs, dtype=np.int64)]
    nnz, _ddt, _idt = rust.encode_shards_inmem(
        X.data,
        X.indices,
        X.indptr,
        row_lists,
        str(rust_dir),
        n_vars=adata.n_vars,
        shuffle="byteshuffle-bytedelta",
        n_threads=2,
    )
    assert nnz == [X.nnz]
    d, i, p = _decode_shx(rust_dir / "shard_0000.shx")
    np.testing.assert_array_equal(d.astype(np.float32), X.data.astype(np.float32))
    np.testing.assert_array_equal(i.astype(np.int64), X.indices.astype(np.int64))
    np.testing.assert_array_equal(p.astype(np.int64), X.indptr.astype(np.int64))


def test_dispatch_uses_rust_when_available(tmp_path, monkeypatch):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream.v2 import _rust_dispatch

    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    assert _rust_dispatch.use_rust_for("byteshuffle-bytedelta") is True
    assert _rust_dispatch.use_rust_for("bitshuffle") is False  # legacy codec → Python
    # Unsupported source dtype → fall back to the Python encoder, not a hard error.
    xi16 = sp.random(4, 3, density=0.5, format="csr", random_state=0).astype(np.int16)
    assert _rust_dispatch.use_rust_for("byteshuffle-bytedelta", xi16) is False


def test_dispatch_disabled_by_env(monkeypatch):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream.v2 import _rust_dispatch

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    assert _rust_dispatch.use_rust_for("byteshuffle-bytedelta") is False


def test_write_sharded_rust_path_roundtrips(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream import ShardedArchive
    from cellstream.v2.writer import write_v2_archive

    adata = _toy_adata(n_obs=30, n_vars=6, seed=3)
    out = tmp_path / "arc"
    write_v2_archive(adata, out, group_by="target_gene", codec="v2-byte-filter")
    arc = ShardedArchive(pack_directory(out, tmp_path / "arc.shad"))
    back = arc.to_anndata()
    # row order is permuted by grouping; compare on counts
    assert back.n_obs == adata.n_obs
    assert int(back.X.nnz) == int(adata.X.nnz)


def _encode_via(env_disable, adata, out_dir):
    """Encode adata to out_dir in a subprocess with CELLSTREAM_DISABLE_RUST set or not."""
    src_h5ad = out_dir.parent / f"src_{out_dir.name}.h5ad"
    adata.write_h5ad(src_h5ad)
    script = (
        "import anndata as ad; from cellstream.v2.writer import write_v2_archive;"
        f"a=ad.read_h5ad(r'{src_h5ad}');"
        f"write_v2_archive(a, r'{out_dir}', group_by='target_gene', codec='v2-byte-filter')"
    )
    env = dict(**os.environ)
    if env_disable:
        env["CELLSTREAM_DISABLE_RUST"] = "1"
    else:
        env.pop("CELLSTREAM_DISABLE_RUST", None)
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_rust_vs_python_decode_identical(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream import ShardedArchive

    adata = _toy_adata(n_obs=40, n_vars=8, seed=7)
    py_dir = tmp_path / "py"
    py_dir.mkdir()
    ru_dir = tmp_path / "ru"
    ru_dir.mkdir()
    _encode_via(True, adata, py_dir)  # Python encoder
    _encode_via(False, adata, ru_dir)  # Rust encoder

    py = ShardedArchive(pack_directory(py_dir, tmp_path / "py.shad")).to_anndata()
    ru = ShardedArchive(pack_directory(ru_dir, tmp_path / "ru.shad")).to_anndata()
    # Same grouping → same row order → arrays decode-identical.
    np.testing.assert_array_equal(py.X.toarray(), ru.X.toarray())
    assert list(py.obs_names) == list(ru.obs_names)


def test_rust_vs_python_float_rawdata_identical(tmp_path):
    """#142: a float32 archive encoded by the Rust core self-describes with the
    rawdata marker, decodes identically to the Python encoder's output, and is
    lossless — proving the Rust data stream is raw (not shuffled)."""
    pytest.importorskip("cellstream.v2._rust")
    from cellstream import ShardedArchive
    from cellstream.v2.codec import BYTE_FILTER_RAWDATA_SHUFFLE_NAME
    from cellstream.v2.format import parse_shx_header

    adata = _toy_float_adata(n_obs=40, n_vars=8, seed=7)
    py_dir = tmp_path / "py"
    py_dir.mkdir()
    ru_dir = tmp_path / "ru"
    ru_dir.mkdir()
    _encode_via(True, adata, py_dir)  # Python encoder
    _encode_via(False, adata, ru_dir)  # Rust encoder

    for d in (py_dir, ru_dir):
        shards = sorted(d.glob("shard_*.shx"))
        assert shards
        for shard in shards:
            with shard.open("rb") as fh:
                hdr = parse_shx_header(fh)
            assert hdr.shuffle == BYTE_FILTER_RAWDATA_SHUFFLE_NAME
            assert hdr.data.dtype == "float32"

    py = ShardedArchive(pack_directory(py_dir, tmp_path / "py.shad")).to_anndata()
    ru = ShardedArchive(pack_directory(ru_dir, tmp_path / "ru.shad")).to_anndata()
    # same grouping input -> identical row order; data decodes identically.
    assert list(py.obs_names) == list(ru.obs_names)
    np.testing.assert_array_equal(py.X.toarray(), ru.X.toarray())
    # lossless vs source — align the grouped output back to the source obs order.
    ru_aligned = ru[list(adata.obs_names)]
    np.testing.assert_array_equal(ru_aligned.X.toarray(), adata.X.toarray())
    assert ru.X.dtype == np.float32


def test_rust_vs_python_float_byte_diff_bonus(tmp_path):
    """Non-gating: float rawdata .shx bytes should be byte-identical Rust vs
    Python (same libzstd-1 over the same raw LE bytes). Skip on libzstd drift —
    decode-identity (above) is the real guarantee."""
    pytest.importorskip("cellstream.v2._rust")
    import filecmp

    adata = _toy_float_adata(n_obs=40, n_vars=8, seed=7)
    py_dir = tmp_path / "py"
    py_dir.mkdir()
    ru_dir = tmp_path / "ru"
    ru_dir.mkdir()
    _encode_via(True, adata, py_dir)
    _encode_via(False, adata, ru_dir)
    for shard in sorted(p.name for p in py_dir.glob("shard_*.shx")):
        if not filecmp.cmp(py_dir / shard, ru_dir / shard, shallow=False):
            pytest.skip("libzstd byte-output drifted; decode-identity still holds")


def test_rust_byte_diff_bonus(tmp_path):
    """Non-gating: if libzstd matches, .shx bytes are identical. Skip on mismatch."""
    pytest.importorskip("cellstream.v2._rust")
    adata = _toy_adata(n_obs=40, n_vars=8, seed=7)
    py_dir = tmp_path / "py"
    py_dir.mkdir()
    ru_dir = tmp_path / "ru"
    ru_dir.mkdir()
    _encode_via(True, adata, py_dir)
    _encode_via(False, adata, ru_dir)
    import filecmp

    for shard in sorted(p.name for p in py_dir.glob("shard_*.shx")):
        if not filecmp.cmp(py_dir / shard, ru_dir / shard, shallow=False):
            pytest.skip("libzstd byte-output drifted; decode-identity still holds")


def test_lossy_cast_error_parity(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream.errors import LossyCastError
    from cellstream.v2.writer import write_v2_archive

    adata = _toy_adata(n_obs=10, n_vars=100_000, seed=1, density=0.00001)
    adata.X = adata.X.tocsr()
    adata.X.indices[0] = -1  # malformed CSR index must fail before writing
    with pytest.raises(LossyCastError):
        write_v2_archive(adata, tmp_path / "bad", group_by="target_gene", codec="v2-byte-filter")


def test_profile_entry_point_returns_buckets(tmp_path):
    rust = pytest.importorskip("cellstream.v2._rust")
    # small nnz keeps CI fast; one (partial) chunk still yields zstd_s > 0, and
    # bucket attribution is scale-invariant.
    out = rust._profile_encode_single_shard(
        nnz=200_000, nnz_per_row=6_000, mode="scalar", out_dir=str(tmp_path)
    )
    assert set(out) == {
        "decide_s",
        "decide_indices_s",
        "feed_s",
        "shuffle_s",
        "delta_s",
        "zstd_s",
        "staging_s",
        "nnz",
    }
    assert out["nnz"] == 200_000
    assert out["zstd_s"] > 0.0
    assert all(
        out[k] >= 0.0
        for k in ("decide_s", "decide_indices_s", "feed_s", "shuffle_s", "delta_s", "staging_s")
    )


def test_profile_entry_point_rejects_bad_out_dir():
    rust = pytest.importorskip("cellstream.v2._rust")
    # a non-existent out_dir must raise a clean Python error, not panic in Rust
    with pytest.raises(ValueError):
        rust._profile_encode_single_shard(
            nnz=1000, nnz_per_row=100, mode="scalar", out_dir="/nonexistent/cellstream/profile/xyz"
        )
