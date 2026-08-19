"""REPORTS on-GPU decode vs CPU decode of one v2 archive. Asserts no performance threshold.

The old ``>= 2x`` gate was **removed** rather than re-tuned: it had been validated against a
configuration that never ran on-GPU decode at all, and #154 had no GPU hardware on which to
measure a replacement. Restoring a gate needs a real run on a GPU node; until then this reports,
and a bad ratio is for a human to read rather than a red test.

This file's git history holds the four earlier configurations, each of which measured something
other than what it claimed. That forensic account lived here through several review rounds and was
itself where each round's new false statement appeared, so it now lives in the log instead --
which is the honest place for it, and leaves this docstring with almost nothing that can go stale.

Fixed CPU-then-GPU order with a warm cache; not counterbalanced.
"""

from __future__ import annotations

import importlib
import os
import statistics
import time

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp


def _make_test_adata(n_obs=50_000, n_vars=3000, density=0.014, seed=77):
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


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    not (os.environ.get("CELLSTREAM_BENCH") == "1" and os.environ.get("CELLSTREAM_GPU") == "1"),
    reason="CELLSTREAM_BENCH=1 CELLSTREAM_GPU=1 required to run GPU benchmarks",
)
def test_report_v2_gpu_vs_cpu_decode(tmp_path):
    cp = pytest.importorskip("cupy")
    # Skip cleanly rather than let read_v2_to_device raise ImportError mid-benchmark
    # (_import_gpu, reader_gpu.py:31). Note the reader does NOT silently fall back to CPU on a
    # missing nvCOMP — it raises — so this is a legibility guard, not a correctness one. Both
    # module locations the reader accepts are checked, so an environment carrying only
    # top-level `nvcomp` is not skipped for a dependency it has.
    if not (_importable("nvidia.nvcomp") or _importable("nvcomp")):
        pytest.skip("nvCOMP not importable as either 'nvidia.nvcomp' or 'nvcomp'")
    adata = _make_test_adata()
    from cellstream import ShardedArchive, write_sharded

    arc = tmp_path / "v2"
    write_sharded(adata, arc, format="v2", n_shards=4, n_workers=4)

    NW = 4
    # _gpu_threshold_bytes=0 forces on-device decode; the default 5 GB (read.py:1018) sends
    # this ~17 MB fixture down the CPU-decode + upload path instead. Of the other two knobs:
    # n_workers controls only that CPU fallback (read.py:1025) and is inert on the forced GPU
    # path, which does not receive it (read.py:1038); n_streams is inert EVERYWHERE -- the
    # fallback never passes it, and v2/reader_gpu.py declares it (:396, :461) without either
    # body reading it, decoding shards serially at :433. n_workers is passed so this call still
    # matches the CPU baseline if the threshold ever routes the fixture back to the fallback;
    # n_streams is passed only because it is part of the public signature -- being inert, it
    # cannot make anything comparable to anything.
    gpu_kw = dict(cast_back=True, n_streams=4, n_workers=NW, _gpu_threshold_bytes=0)

    def _read_gpu():
        out = ShardedArchive(arc).to_cupy_anndata(**gpu_kw)
        # read_v2_to_device already deviceSynchronize()s before returning (reader_gpu.py:453),
        # so this is redundant on the forced path — kept so the timer stays honest if that
        # internal sync moves, or if the call routes to the async upload path instead.
        cp.cuda.Device().synchronize()
        return out

    # Warm BOTH paths (page cache, CUDA context, nvcomp codec init) before timing either.
    ShardedArchive(arc).to_anndata(cast_back=True, n_workers=NW)
    _read_gpu()

    cpu, gpu = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        ShardedArchive(arc).to_anndata(cast_back=True, n_workers=NW)
        cpu.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _read_gpu()
        gpu.append(time.perf_counter() - t0)

    t_cpu, t_gpu = statistics.median(cpu), statistics.median(gpu)
    # Asserted BEFORE the ratio below, which is the only thing it protects: a zero t_gpu would
    # otherwise surface as a ZeroDivisionError inside the f-string. It does NOT establish that
    # either path decoded anything — a no-op that burns a microsecond passes it.
    assert t_cpu > 0 and t_gpu > 0, f"non-positive timings: cpu={cpu!r} gpu={gpu!r}"
    print(
        f"\nCPU decode (n_workers={NW}: {NW} Rust threads, or up to {NW} Python workers "
        f"without the Rust core) vs forced on-device decode (serial across shards; n_streams "
        f"is inert), same archive, median of 3: {t_cpu / t_gpu:.2f}x  "
        f"(t_cpu={t_cpu:.2f}s, t_gpu={t_gpu:.2f}s)"
    )
    # Deliberately NO `assert speedup >= 2.0` -- see the module docstring.
