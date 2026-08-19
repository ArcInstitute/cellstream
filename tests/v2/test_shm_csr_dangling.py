"""Regression tests for issue #274: shm-backed CSR arrays must stay valid after their
container (csr_matrix / AnnData) is collected.

Extracting ``X.data`` and dropping the container used to munmap the segment the array
still points at, because the segment was referenced only from the container. Pre-fix
this SIGSEGVs on the pure-Python engine with DEFAULT arguments, and on both engines with
``output_backing="shm"``.

Each case runs in a CHILD process (an in-process use-after-free is indistinguishable
from a crashed test runner) and prints two flushed markers -- MARKER-REACHED before the
touch and TOUCH-SUCCEEDED after it -- so "died before the touch" (a broken fixture, which
proves nothing), "died AT the touch" (the #274 crash), and "read fine but failed a value
assertion" are three distinguishable outcomes.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from cellstream.v2._rust_dispatch import rust_available

_TIMEOUT = 300

# The "rust" leg is meaningless when the extension is not built -- and under the
# CELLSTREAM_DISABLE_RUST=1 CI leg it would silently re-run the Python engine, making the
# documented pre-fix split wrong. rust_available() is False in both cases, so skip.
_ENGINES = [
    pytest.param(
        False,
        id="rust",
        marks=pytest.mark.skipif(
            not rust_available(), reason="Rust extension unavailable or disabled"
        ),
    ),
    pytest.param(True, id="python"),
]

# Two shards + n_workers=2 so the pure-Python engine actually takes its multiprocess
# branch (_dispatch_workers runs inline when len(inputs) == 1).
#
# The archive lives in a PARENT-owned tmp_path, not tempfile.mkdtemp() + atexit: these
# children are EXPECTED to SIGSEGV pre-fix, and atexit handlers do not run after a
# signal death, so an atexit cleanup would leak an archive per crashed case.
_BUILD = """
import gc, pathlib
import numpy as np, scipy.sparse as sp, anndata as ad
import cellstream
from cellstream.write import write_sharded
tmp = pathlib.Path(TMP_DIR)
csr = sp.random(200, 50, density=0.2, format="csr", dtype="float32",
                random_state=np.random.default_rng(0))
out = tmp / "arc"
write_sharded(ad.AnnData(X=csr), out, format="v2", n_shards=2)
arc = cellstream.ShardedArchive(out)
n = arc.n_obs
# Reference checksums, taken from a read whose container is still BOUND (safe on both
# engines pre- and post-fix). Each CSR array needs its own expected value -- .indices
# and .indptr do not sum to the data sum.
_b = arc[0:n].x(n_workers=2)
EXP = {
    "data": float(np.asarray(_b.data, dtype="float64").sum()),
    "indices": float(np.asarray(_b.indices, dtype="float64").sum()),
    "indptr": float(np.asarray(_b.indptr, dtype="float64").sum()),
}
assert abs(EXP["data"] - float(csr.data.astype("float64").sum())) < 1e-3
del _b
gc.collect()
"""

# `kind` selects which reference checksum to compare against.
_TOUCH = """
gc.collect()
gc.collect()
print("MARKER-REACHED", flush=True)
got = float(np.asarray(d, dtype="float64").sum())
print("TOUCH-SUCCEEDED", flush=True)
assert abs(got - EXP[kind]) < 1e-3, (kind, got, EXP[kind])
print("OK", flush=True)
"""


def _run(body: str, *, tmp, disable_rust: bool) -> subprocess.CompletedProcess:
    # Only ever SET the env var, never unset it -- mirrors
    # tests/v2/test_dense_shm_temporary.py. Popping it would force Rust back on during
    # the CELLSTREAM_DISABLE_RUST=1 CI leg; the _ENGINES skipif handles that case instead.
    env = dict(os.environ)
    if disable_rust:
        env["CELLSTREAM_DISABLE_RUST"] = "1"
    prelude = f"TMP_DIR = {str(tmp)!r}\n"
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
        timeout=_TIMEOUT,
    )


def _check(p):
    ctx = f"rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    assert "MARKER-REACHED" in p.stdout, f"child died BEFORE the touch -- fixture bug\n{ctx}"
    assert "TOUCH-SUCCEEDED" in p.stdout, f"child died AT the touch -- #274 dangle\n{ctx}"
    assert p.returncode == 0, ctx
    assert "OK" in p.stdout, ctx


# (case id, kind, the statement producing the extracted array `d`)
_CASES = [
    ("x-data", "data", "d = arc[0:n].x(n_workers=2).data"),
    ("x-indices", "indices", "d = arc[0:n].x(n_workers=2).indices"),
    ("x-indptr", "indptr", "d = arc[0:n].x(n_workers=2).indptr"),
    ("x-shm-data", "data", 'd = arc[0:n].x(n_workers=2, output_backing="shm").data'),
    ("x-shm-indices", "indices", 'd = arc[0:n].x(n_workers=2, output_backing="shm").indices'),
    ("x-shm-indptr", "indptr", 'd = arc[0:n].x(n_workers=2, output_backing="shm").indptr'),
    ("anndata-data", "data", "d = arc[0:n].to_anndata(n_workers=2).X.data"),
    (
        "anndata-shm-data",
        "data",
        'd = arc[0:n].to_anndata(n_workers=2, output_backing="shm").X.data',
    ),
    (
        "full-anndata-shm-data",
        "data",
        'd = arc.to_anndata(n_workers=2, output_backing="shm").X.data',
    ),
]


@pytest.mark.parametrize("disable_rust", _ENGINES)
@pytest.mark.parametrize(
    ("kind", "case"), [(k, c) for _, k, c in _CASES], ids=[i for i, _, _ in _CASES]
)
def test_extracted_csr_array_survives_container_gc(tmp_path, kind, case, disable_rust):
    body = _BUILD + f"kind = {kind!r}\n" + case + "\n" + _TOUCH
    _check(_run(body, tmp=tmp_path, disable_rust=disable_rust))


# iter_group_shards() calls ShardedArchive._require_grouped(), so it needs an archive
# written with group_by= -- the ungrouped _BUILD archive would raise before the marker.
# write_sharded REJECTS n_shards together with group_by, so the shard count is steered
# with target_shard_bytes instead and asserted.
_BUILD_GROUPED = """
import gc, pathlib
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
import cellstream
from cellstream.write import write_sharded
tmp = pathlib.Path(TMP_DIR)
csr = sp.random(200, 50, density=0.2, format="csr", dtype="float32",
                random_state=np.random.default_rng(0))
adata = ad.AnnData(X=csr)
adata.obs["target_gene"] = pd.Categorical([f"g{i % 4}" for i in range(200)])
out = tmp / "garc"
write_sharded(adata, out, format="v2",
              group_by="target_gene", target_shard_bytes=4096)
arc = cellstream.ShardedArchive(out)
assert arc.n_shards >= 2, arc.n_shards
expected = float(csr.data.astype("float64").sum())
"""


@pytest.mark.parametrize("disable_rust", _ENGINES)
def test_prefetch_per_shard_data_survives_carrier_gc(tmp_path, disable_rust):
    """prefetch._split_batch's per-shard views inherit the pin from the batch arrays.

    Each per-shard matrix is a slice view of the batch's CSR arrays, so dropping the
    GroupShard carriers must not invalidate the extracted .data arrays.
    """
    p = _run(
        _BUILD_GROUPED
        + """
kept = []
for shard in arc.iter_group_shards(prefetch=2, n_workers=2):
    kept.append(shard.x().data)
del shard
gc.collect()
gc.collect()
print("MARKER-REACHED", flush=True)
total = sum(float(np.asarray(k, dtype="float64").sum()) for k in kept)
print("TOUCH-SUCCEEDED", flush=True)
# No reference= was passed, so the group shards cover every row exactly once.
assert abs(total - expected) < 1e-3, (total, expected)
print("OK", flush=True)
""",
        tmp=tmp_path,
        disable_rust=disable_rust,
    )
    _check(p)
