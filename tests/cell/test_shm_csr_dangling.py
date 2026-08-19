"""Issue #274, layout=cell arm: read_cell_bulk(output_backing="shm") must return CSR
arrays that stay valid after the AnnData is collected.

Narrower than the layout=shard arm: read_cell_bulk honors output_backing="heap" (it
copies out and closes the bundle), so only output_backing="shm" dangles pre-fix.

As of #277 the Rust path is no longer heap-direct -- it honors output_backing="shm" and
allocates a real bundle -- so the "rust" leg below now genuinely exercises the shm arm on
both engines rather than silently testing heap arrays twice.

Two flushed markers per child, as in tests/v2/test_shm_csr_dangling.py.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from cellstream.cell._rust_dispatch import rust_available

_TIMEOUT = 300

# Skip the "rust" leg when the extension is unavailable or globally disabled -- see the
# same guard in tests/v2/test_shm_csr_dangling.py.
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

# Parent-owned tmp_path, not tempfile.mkdtemp() + atexit: these children are EXPECTED to
# SIGSEGV pre-fix, and atexit handlers do not run after a signal death.
_BUILD = """
import gc, warnings, pathlib
import numpy as np, scipy.sparse as sp, anndata as ad
import cellstream
from cellstream.cell.reader import read_cell_bulk
tmp = pathlib.Path(TMP_DIR)
csr = sp.random(200, 50, density=0.2, format="csr", dtype="float32",
                random_state=np.random.default_rng(0))
csr.data = np.ceil(csr.data * 100).astype("float32")
out = tmp / "cellarc"
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cellstream.write_sharded(ad.AnnData(X=csr), out, layout="cell")
# Reference checksums from a read whose AnnData is still BOUND (safe pre- and post-fix).
# Each array needs its own expected value -- .indices/.indptr do not sum to the data sum.
_b = read_cell_bulk(out, n_workers=2, output_backing="heap")
EXP = {
    "data": float(np.asarray(_b.X.data, dtype="float64").sum()),
    "indices": float(np.asarray(_b.X.indices, dtype="float64").sum()),
    "indptr": float(np.asarray(_b.X.indptr, dtype="float64").sum()),
}
assert abs(EXP["data"] - float(csr.data.astype("float64").sum())) < 1e-3
del _b
gc.collect()
"""

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
    # Only ever SET the env var, never unset it (see the v2 sibling test).
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


_CASES = [
    ("shm-data", "data", 'd = read_cell_bulk(out, n_workers=2, output_backing="shm").X.data'),
    (
        "shm-indices",
        "indices",
        'd = read_cell_bulk(out, n_workers=2, output_backing="shm").X.indices',
    ),
    (
        "shm-indptr",
        "indptr",
        'd = read_cell_bulk(out, n_workers=2, output_backing="shm").X.indptr',
    ),
    ("heap-data", "data", 'd = read_cell_bulk(out, n_workers=2, output_backing="heap").X.data'),
]


@pytest.mark.parametrize("disable_rust", _ENGINES)
@pytest.mark.parametrize(
    ("kind", "case"), [(k, c) for _, k, c in _CASES], ids=[i for i, _, _ in _CASES]
)
def test_cell_extracted_csr_array_survives_container_gc(tmp_path, kind, case, disable_rust):
    body = _BUILD + f"kind = {kind!r}\n" + case + "\n" + _TOUCH
    _check(_run(body, tmp=tmp_path, disable_rust=disable_rust))
