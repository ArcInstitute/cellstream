"""Min-aware low-plane decode fast-path: a uint32-on-disk stream whose recorded
min fits uint16 is decoded from its 2 low byte-planes. The optimization must be
byte-identical to both the source and the full-gather (pure-Python) decode."""

import json

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import cellstream
from cellstream.read import ShardedArchive
from cellstream.v2 import _rust_dispatch
from tests._archive_helpers import pack_directory, write_directory

pytestmark = pytest.mark.skipif(
    not _rust_dispatch.rust_available(),
    reason="fast-path is a Rust decode optimization; nothing to compare without the Rust core",
)


def _csr(arr, dtype):
    return ad.AnnData(X=sp.csr_matrix(np.asarray(arr, dtype=dtype)))


def _dense(a):
    return a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)


def _read(out, monkeypatch, *, disable_rust, **kw):
    """Read `out` with the Rust core enabled or force-disabled (pure-Python full gather)."""
    if disable_rust:
        monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    else:
        monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    return _dense(ShardedArchive(str(out)).to_anndata(**kw))


def test_counts_both_streams_narrowed_byte_identical(tmp_path, monkeypatch):
    # Counts archive: uint32 on disk for data + indices, all values <= 65535,
    # n_var <= 65536 -> data min uint16, index min uint16 -> both fast-path.
    rng = np.random.default_rng(0)
    X = (rng.random((40, 50)) < 0.3) * rng.integers(1, 5000, (40, 50))
    out = tmp_path / "counts"
    cellstream.write_sharded(_csr(X, np.uint16), out, n_shards=2)
    m = ShardedArchive(str(out)).manifest
    assert m["x_data_dtype_on_disk"] == "uint32" and m["x_data_dtype_min"] in ("uint8", "uint16")
    assert m["x_indices_dtype_on_disk"] == "uint32" and m["x_indices_dtype_min"] == "uint16"
    # Default float32 read: fast path (Rust) == full gather (Python) == source.
    fast = _read(out, monkeypatch, disable_rust=False)
    full = _read(out, monkeypatch, disable_rust=True)
    assert fast.dtype == np.float32
    assert np.array_equal(fast, X.astype(np.float32))
    assert np.array_equal(fast, full)
    # uint16 output read: same byte-identity.
    fast16 = _read(out, monkeypatch, disable_rust=False, data_dtype="uint16")
    full16 = _read(out, monkeypatch, disable_rust=True, data_dtype="uint16")
    assert fast16.dtype == np.uint16 and np.array_equal(fast16, X.astype(np.uint16))
    assert np.array_equal(fast16, full16)


def test_genuine_uint32_does_not_trigger_fast_path(tmp_path, monkeypatch):
    # A value > 65535 -> data min uint32 -> the trigger must NOT fire (full 4-plane gather).
    X = np.array([[0, 70000], [100000, 0]], dtype=np.uint32)
    out = tmp_path / "big"
    cellstream.write_sharded(_csr(X, np.uint32), out, n_shards=1)
    assert ShardedArchive(str(out)).manifest["x_data_dtype_min"] == "uint32"
    fast = _read(out, monkeypatch, disable_rust=False)
    full = _read(out, monkeypatch, disable_rust=True)
    assert np.array_equal(fast.astype(np.uint32), X)
    assert np.array_equal(fast, full)


def test_index_only_narrowing_lognorm_style(tmp_path, monkeypatch):
    # Float32 data (raw, never narrowed) + uint32 indices with n_var <= 65536
    # -> index min uint16 -> index fast-path engages, data stream untouched.
    rng = np.random.default_rng(1)
    base = (rng.random((30, 40)) < 0.3) * rng.random((30, 40)).astype(np.float32) * 9.0
    out = tmp_path / "lognorm"
    cellstream.write_sharded(_csr(base, np.float32), out, n_shards=2)
    m = ShardedArchive(str(out)).manifest
    assert m["x_indices_dtype_on_disk"] == "uint32" and m["x_indices_dtype_min"] == "uint16"
    fast = _read(out, monkeypatch, disable_rust=False)
    full = _read(out, monkeypatch, disable_rust=True)
    assert np.array_equal(fast, base.astype(np.float32))
    assert np.array_equal(fast, full)


def test_reverse_compat_min_keys_absent(tmp_path, monkeypatch):
    # Pre-#158 archive (no *_min keys): reader falls back to on-disk for the min,
    # so a uint32-on-disk stream gets min uint32 -> no fast-path -> still correct.
    X = np.array([[0, 3, 0], [5, 0, 200]], dtype=np.uint16)
    out = tmp_path / "legacy"
    # Tamper on a scaffolding DIRECTORY, then pack -- see #154 part B.
    src = tmp_path / "legacy_src"
    write_directory(_csr(X, np.uint16), src, n_shards=1)
    mpath = src / "manifest.json"
    m = json.loads(mpath.read_text())
    m.pop("x_data_dtype_min", None)
    m.pop("x_indices_dtype_min", None)
    mpath.write_text(json.dumps(m))
    pack_directory(src, out)
    fast = _read(out, monkeypatch, disable_rust=False)
    full = _read(out, monkeypatch, disable_rust=True)
    assert np.array_equal(fast.astype(np.uint16), X)
    assert np.array_equal(fast, full)
