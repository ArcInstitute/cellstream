"""Pinned-host CSR + DLPack per-shard streaming handoff tests (#160).

CPU-only (run in CI). The pinned registration path is GPU-gated and lives in
tests/v2/gpu/test_dlpack_pinned.py; here pinned=True falls back to pageable on a
CPU-only host (cupy absent) but still exercises the API + view-lifetime paths.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

import cellstream
from tests.v2.test_prefetch_stream import _write_grouped

# ---------------------------------------------------------------------------
# pin.py — host pinning helper fallback (Task 1)
# ---------------------------------------------------------------------------


def test_pin_array_inplace_no_cupy_returns_none_and_warns_once(monkeypatch):
    import builtins

    import cellstream.pin as pinmod

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("no cupy")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(pinmod, "_PIN_WARNED", False)

    arr = np.arange(16, dtype=np.float32)
    with pytest.warns(RuntimeWarning, match="pageable"):
        assert pinmod.pin_array_inplace(arr) is None
    # second call: still None, but no second warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test error
        assert pinmod.pin_array_inplace(arr) is None


def test_pin_array_inplace_empty_array_returns_none(monkeypatch):
    import cellstream.pin as pinmod

    monkeypatch.setattr(pinmod, "_PIN_WARNED", False)
    assert pinmod.pin_array_inplace(np.empty(0, dtype=np.float32)) is None


# ---------------------------------------------------------------------------
# GroupShard.dlpack() (Task 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("disable_rust", [False, True], ids=["rust", "python"])
@pytest.mark.parametrize("cast_back", [True, False], ids=["f32", "u16"])
def test_dlpack_roundtrip_byte_identical(tmp_path, monkeypatch, disable_rust, cast_back):
    if disable_rust:
        monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    out, _ = _write_grouped(tmp_path, [f"g{i}" for i in range(40)])
    arc = cellstream.ShardedArchive(out)
    seen = 0
    for gs in arc.iter_group_shards(prefetch=2, n_workers=2, cast_back=cast_back):
        x = gs.x()
        caps = gs.dlpack()
        assert caps.shape == x.shape
        np.testing.assert_array_equal(np.from_dlpack(caps.data), x.data)
        np.testing.assert_array_equal(np.from_dlpack(caps.indices), x.indices)
        np.testing.assert_array_equal(np.from_dlpack(caps.indptr), x.indptr)
        assert caps.data.dtype == x.data.dtype  # f32 or uint16 preserved
        assert caps.indices.dtype == x.indices.dtype
        seen += 1
    assert seen > 0


def test_dlpack_dense_raises(tmp_path):
    out, _ = _write_grouped(tmp_path, [f"g{i}" for i in range(40)])
    arc = cellstream.ShardedArchive(out)
    gs = next(arc.iter_group_shards(prefetch=2, n_workers=2, container="dense"))
    with pytest.raises(ValueError, match="CSR-only"):
        gs.dlpack()


# ---------------------------------------------------------------------------
# iter_group_shards(pinned=) guards (Task 3)
# ---------------------------------------------------------------------------


def test_pinned_requires_prefetch(tmp_path):
    out, _ = _write_grouped(tmp_path, [f"g{i}" for i in range(40)])
    arc = cellstream.ShardedArchive(out)
    with pytest.raises(ValueError, match="prefetch"):
        list(arc.iter_group_shards(prefetch=0, pinned=True))


def test_pinned_requires_csr(tmp_path):
    out, _ = _write_grouped(tmp_path, [f"g{i}" for i in range(40)])
    arc = cellstream.ShardedArchive(out)
    with pytest.raises(ValueError, match="CSR"):
        list(arc.iter_group_shards(prefetch=2, n_workers=2, pinned=True, container="dense"))


# ---------------------------------------------------------------------------
# Pinned producer path — CPU fallback is byte-identical (Task 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("disable_rust", [False, True], ids=["rust", "python"])
def test_pinned_true_cpu_fallback_byte_identical(tmp_path, monkeypatch, disable_rust):
    if disable_rust:
        monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    import cellstream.pin as pinmod

    monkeypatch.setattr(pinmod, "_PIN_WARNED", False)
    # Force the pageable path so the test is deterministic on GPU dev hosts too
    # (the producer does `from cellstream.pin import pin_array_inplace` at call time).
    monkeypatch.setattr(pinmod, "pin_array_inplace", lambda arr: None)

    out, _ = _write_grouped(tmp_path, [f"g{i}" for i in range(40)])
    arc = cellstream.ShardedArchive(out)
    rows_pinned = [
        (gs.x().data.copy(), gs.x().indptr.copy())
        for gs in arc.iter_group_shards(prefetch=2, n_workers=2, pinned=True)
    ]
    rows_plain = [
        (gs.x().data.copy(), gs.x().indptr.copy())
        for gs in arc.iter_group_shards(prefetch=2, n_workers=2)
    ]
    assert len(rows_pinned) == len(rows_plain) > 0
    for (d1, p1), (d2, p2) in zip(rows_pinned, rows_plain, strict=True):
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Use-after-free regression (Task 5) — keep ONLY the dlpack arrays, drop the
# GroupShard + iterator, force GC; views must stay valid (no munmap/unregister).
# Runs in a child process (a UAF crashes the whole interpreter).
# ---------------------------------------------------------------------------

_UAF_BODY = """
import gc, numpy as np, scipy.sparse as sp, anndata as ad, cellstream
from cellstream.write import write_sharded
import atexit, shutil, tempfile, pathlib
tmp = pathlib.Path(tempfile.mkdtemp()); atexit.register(shutil.rmtree, tmp, ignore_errors=True)
rng = np.random.default_rng(0)
labels = [f"g{i}" for i in range(60)]
nnz = int(len(labels) * 50 * 0.2)
csr = sp.coo_matrix((rng.integers(1, 50, nnz, dtype=np.uint16),
                     (rng.integers(0, len(labels), nnz), rng.integers(0, 50, nnz))),
                    shape=(len(labels), 50)).tocsr().astype("float32")
csr.indices = csr.indices.astype(np.int32); csr.indptr = csr.indptr.astype(np.int64)
out = tmp / "arc"
write_sharded(ad.AnnData(X=csr, obs={"gene": labels}), out, format="v2",
              group_by="gene", target_shard_bytes=4000)
arc = cellstream.ShardedArchive(out)
kept = []
for gs in arc.iter_group_shards(prefetch=2, n_workers=2, pinned=True):
    caps = gs.dlpack()
    kept.append((np.from_dlpack(caps.data), float(gs.x().data.sum())))
    del gs, caps
    gc.collect()
total = 0.0
for arr, expected in kept:
    s = float(arr.sum()); total += s
    assert abs(s - expected) < 1e-3, (s, expected)
print("OK", len(kept), total)
"""


@pytest.mark.parametrize("disable_rust", [False, True], ids=["rust", "python"])
def test_dlpack_use_after_free_no_segfault(disable_rust):
    env = dict(os.environ)
    if disable_rust:
        env["CELLSTREAM_DISABLE_RUST"] = "1"
    p = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_UAF_BODY)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert p.returncode == 0, f"child crashed: rc={p.returncode}\nSTDERR:\n{p.stderr}"
    assert "OK" in p.stdout, p.stdout
