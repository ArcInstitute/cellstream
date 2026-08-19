"""Regression tests for issue #182: a temporary dense AnnData/matrix from
to_anndata(container="dense", output_backing="shm") (or .x()) must not munmap its
shm backing while a .X view is live (use-after-free segfault).

The repro crashes the whole interpreter, so it runs in a CHILD process and each
test asserts a clean exit (pre-fix: SIGSEGV / non-zero returncode).
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest


def _run(body: str, *, disable_rust: bool) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if disable_rust:
        env["CELLSTREAM_DISABLE_RUST"] = "1"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
    )


# Build a tiny packed v2 archive and compute the expected dense sum from a BOUND
# (safe) AnnData, then run a loop of the temporary-AnnData repro.
_BUILD = """
import gc
import numpy as np, scipy.sparse as sp, anndata as ad
import cellstream
from cellstream.write import write_sharded
import atexit, shutil, tempfile, pathlib
tmp = pathlib.Path(tempfile.mkdtemp())
atexit.register(shutil.rmtree, tmp, ignore_errors=True)  # clean up on child exit
csr = sp.random(60, 50, density=0.2, format="csr", dtype="float32",
                random_state=np.random.default_rng(0))
out = tmp / "arc"
write_sharded(ad.AnnData(X=csr), out, format="v2")
arc = cellstream.ShardedArchive(out)
bound = arc.to_anndata(container="dense", output_backing="shm")  # bound = safe
expected = float(np.asarray(bound.X).sum())
del bound; gc.collect()
"""


@pytest.mark.parametrize("disable_rust", [False, True], ids=["rust", "python"])
def test_to_anndata_dense_shm_temporary_no_segfault(disable_rust):
    p = _run(
        _BUILD
        + """
for _ in range(8):
    v = np.asarray(arc.to_anndata(container="dense", output_backing="shm").X)
    gc.collect()
    assert abs(float(v.sum()) - expected) < 1e-6, (float(v.sum()), expected)
print("OK")
""",
        disable_rust=disable_rust,
    )
    assert p.returncode == 0, f"rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    assert "OK" in p.stdout


@pytest.mark.parametrize("disable_rust", [False, True], ids=["rust", "python"])
def test_view_x_dense_shm_temporary_no_segfault(disable_rust):
    p = _run(
        _BUILD
        + """
n = arc.n_obs
for _ in range(8):
    v = np.asarray(arc[0:n].x(container="dense", output_backing="shm"))
    gc.collect()
    assert abs(float(v.sum()) - expected) < 1e-6, (float(v.sum()), expected)
print("OK")
""",
        disable_rust=disable_rust,
    )
    assert p.returncode == 0, f"rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    assert "OK" in p.stdout


def test_dense_array_for_bundle_pins_through_asarray():
    """The helper returns a buffer-owning _ShmDenseArray whose bundle pin
    survives np.asarray + dropping every direct segment ref + gc. Safe in-process
    (the pin is on the buffer-owner, so no UAF can occur)."""
    import gc
    from multiprocessing import shared_memory

    from cellstream.shm import DenseShmBundle, dense_array_for_bundle

    seg = shared_memory.SharedMemory(create=True, size=4 * 100)
    name = seg.name
    try:
        bundle = DenseShmBundle(array=seg)
        X = dense_array_for_bundle(bundle, (100,), np.float32)
        X[:] = np.arange(100, dtype=np.float32)
        assert X._cellstream_shm_bundle is bundle
        v = np.asarray(X)
        del X, bundle, seg  # only v's .base chain keeps the segment mapped
        gc.collect()
        np.testing.assert_array_equal(v, np.arange(100, dtype=np.float32))
        del v
        gc.collect()
    finally:  # guarantee the /dev/shm segment is unlinked even if an assert fails
        try:
            shm_temp = shared_memory.SharedMemory(name=name)
            shm_temp.close()  # release this transient handle's fd/mapping
            shm_temp.unlink()
        except FileNotFoundError:
            pass
