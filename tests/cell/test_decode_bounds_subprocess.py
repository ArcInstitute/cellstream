"""#269: the pfordelta decode must not read or write out of bounds on a corrupt archive.

These run in a SUBPROCESS on purpose. Both engines already raise a ValueError today; the bug is
that the raise happens AFTER the codec has scribbled the heap, so the process dies later -- at
gc or at interpreter shutdown. An in-process `pytest.raises` therefore passes against the
unfixed code. Only the child's exit status distinguishes fixed from broken.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# 16 full 256-value blocks + a 255-value VariableByte tail. See the plan for why this size.
_ROW_NNZ = 16 * 256 + 255
_TAMPERED_NNZ = 16 * 256 + 1
# Exactly two full blocks, so the stage-1 count is a nonzero multiple of BlockSize and word 1
# of the codec blob really is `wheremeta`.
_POISON_ROW_NNZ = 2 * 256

_CHILD = r"""
import gc, pathlib, sys
import numpy as np, scipy.sparse as sp, anndata as ad
import cellstream
from cellstream.cell.writer import write_cell_archive
from cellstream.packed.reader import PackedArchive

out = pathlib.Path(sys.argv[1]); mode = sys.argv[2]
row_nnz = int(sys.argv[3]); tampered = int(sys.argv[4])

n_vars = 20000
rng = np.random.default_rng(0)
# row 0 gets exactly row_nnz entries; the rest are small.
rows, cols = [], []
cols0 = rng.choice(n_vars, size=row_nnz, replace=False); cols0.sort()
rows += [0] * row_nnz; cols += list(cols0)
for r in range(1, 40):
    c = rng.choice(n_vars, size=50, replace=False); c.sort()
    rows += [r] * 50; cols += list(c)
vals = rng.integers(1, 50, size=len(rows)).astype(np.uint32)
X = sp.csr_matrix((vals, (rows, cols)), shape=(40, n_vars))
a = ad.AnnData(X=X)
a.obs_names = [f"c{i}" for i in range(X.shape[0])]
a.var_names = [f"g{i}" for i in range(n_vars)]
write_cell_archive(a, out)

pa = PackedArchive(out)
_, pbase, plen = pa.member_location("x_payload")
member = "x_offsets" if mode == "offsets" else "x_indptr"
_, mbase, mlen = pa.member_location(member)
raw = bytearray(out.read_bytes())
arr = np.frombuffer(bytes(raw[mbase:mbase + mlen]), dtype=np.int64).copy()
if mode == "offsets":
    arr[1] = plen           # row 0's frame spans the WHOLE payload
else:
    arr[1] = tampered       # understate row 0's nnz so the tail overruns
tampered_bytes = bytearray(raw)
tampered_bytes[mbase:mbase + mlen] = arr.tobytes()
tampered_path = out.with_suffix(".tampered.shad")
tampered_path.write_bytes(bytes(tampered_bytes))

# LAYER 1, added when CellStore gained open-time sidecar validation: the corrupt archive should
# not reach the decoder at all. An `offsets` tamper makes the array non-monotonic, so open
# rejects it; an `indptr` tamper of this shape stays non-decreasing with an unchanged endpoint,
# so open cannot see it and layer 2 is the only guard. Report which happened; the parent asserts
# the right one per mode.
_probe = None
try:
    _probe = cellstream.open(tampered_path)
    print("RESULT open-accepted")
except Exception as e:
    print("RESULT open-rejected " + type(e).__name__)
finally:
    # On the `indptr` shape open SUCCEEDS, so this leaves a live store; closing it explicitly
    # keeps the child's exit status -- the only measurement in this file -- free of any argument
    # about when a discarded temporary gets collected. (Copilot, suppressed comment.)
    if _probe is not None:
        _probe.close()
    _probe = None

# Report the engine the CHILD actually dispatched to. Clearing CELLSTREAM_DISABLE_RUST in the
# parent only expresses an intent, and `rust_available()` only says the extension imported --
# reader.py gates on use_rust_for_cell_decode(), which also depends on the resolved dtypes. Spy
# on that real gate, so a fallback cannot leave both parametrisations testing one engine.
import cellstream.cell._rust_dispatch as _rd
_gate = []
_orig_gate = _rd.use_rust_for_cell_decode
def _spy_gate(*a, **k):
    r = _orig_gate(*a, **k)
    _gate.append(bool(r))
    return r
_rd.use_rust_for_cell_decode = _spy_gate

# LAYER 2 -- the guarantee #269 is about. Open the PRISTINE archive and inject the same
# corruption into the in-memory sidecar, so the decoder receives exactly the array the tampered
# file carried. Reaching it through the file would now be stopped by layer 1, and the decode
# bound this test exists for would stop being exercised at all: a check that makes another check
# untestable removes coverage rather than adding it.
_store = None
try:
    _store = cellstream.open(out)
    if mode == "offsets":
        _store.offsets = arr
    else:
        _store.indptr = arr
    _store.gather_rows(np.array([0], dtype=np.int64))
    print("RESULT no-raise")
except ValueError:
    print("RESULT valueerror")
except Exception as e:
    print(f"RESULT other {type(e).__name__}")
finally:
    # Close it, and DROP the reference, before the gc.collect() below. This block used to be
    # `shardad.open(out).gather_rows(...)` -- an unreferenced temporary -- and naming the store
    # to inject the corrupt sidecar kept the mmap alive in this frame, so the collect() that
    # precedes "RESULT survived" no longer collected the thing the test is watching. Since the
    # whole measurement here is the child's EXIT STATUS, a live export at interpreter shutdown is
    # exactly the failure mode being tested for. A close() that itself raises is reported rather
    # than allowed to kill the child, so the parent's other assertions still run and the failure
    # is visible instead of a bare nonzero exit. (Gemini, round 2.)
    if _store is not None:
        try:
            _store.close()
        except Exception as e:
            print(f"RESULT close-failed {type(e).__name__}")
        _store = None
print("RESULT engine=" + ("rust" if any(_gate) else "python" if _gate else "unknown"))
gc.collect()
print("RESULT survived")
"""


@pytest.mark.parametrize("disable_rust", [False, True], ids=["rust", "python"])
@pytest.mark.parametrize("mode", ["offsets", "indptr"])
def test_corrupt_sidecar_does_not_corrupt_the_heap(tmp_path, disable_rust, mode):
    from cellstream.cell._rust_dispatch import rust_available

    if not disable_rust and not rust_available():
        pytest.skip("Rust extension not available")
    env = dict(os.environ)
    if disable_rust:
        env["CELLSTREAM_DISABLE_RUST"] = "1"
    else:
        env.pop("CELLSTREAM_DISABLE_RUST", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            str(tmp_path / f"{mode}.shad"),
            mode,
            str(_ROW_NNZ),
            str(_TAMPERED_NNZ),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    # the assertion that matters: a clean exit. Pre-fix this is 139 (SIGSEGV) or 134 (SIGABRT).
    assert proc.returncode == 0, (
        f"child exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # the child must have run the engine this parametrisation is named after
    expected_engine = "python" if disable_rust else "rust"
    assert f"RESULT engine={expected_engine}" in proc.stdout, proc.stdout
    assert "RESULT valueerror" in proc.stdout, proc.stdout
    assert "RESULT survived" in proc.stdout, proc.stdout
    # Layer 1 vs layer 2, asserted per mode rather than "either is fine". If a future open-time
    # check started rejecting the indptr shape too, this says so instead of quietly leaving the
    # decode bound untested.
    if mode == "offsets":
        assert "RESULT open-rejected IncompatibleSchemaError" in proc.stdout, proc.stdout
    else:
        assert "RESULT open-accepted" in proc.stdout, proc.stdout
    # The store must close cleanly after the failed gather. A BufferError here would mean the
    # raised exception's traceback still pins the payload mmap -- the #287 shape -- which the
    # child reports rather than dying on, so assert its absence instead of reading it out of an
    # exit status. Turns the round-2 hygiene point into an actual oracle.
    assert "RESULT close-failed" not in proc.stdout, proc.stdout


_CHILD_POISON = r"""
import gc, pathlib, struct, sys
import numpy as np, scipy.sparse as sp, anndata as ad
import cellstream
from cellstream.cell.writer import write_cell_archive
from cellstream.packed.reader import PackedArchive

out = pathlib.Path(sys.argv[1]); row_nnz = int(sys.argv[2])

n_vars = 4000
rng = np.random.default_rng(3)
# Row 0 needs at least one FULL 256-value block, otherwise CompositeCodec hands stage 1
# nothing, the stage-1 count is 0 and word 1 of the blob is VariableByte data rather than
# `wheremeta` -- the poison would never reach __decodeArray. (Checkpoint-2 review: the first
# draft used a density that gave row 0 only 210 values and tested nothing.)
rows, cols = [], []
c0 = rng.choice(n_vars, size=row_nnz, replace=False); c0.sort()
rows += [0] * row_nnz; cols += list(c0)
for r in range(1, 40):
    c = rng.choice(n_vars, size=50, replace=False); c.sort()
    rows += [r] * 50; cols += list(c)
vals = rng.integers(1, 50, size=len(rows)).astype(np.uint32)
X = sp.csr_matrix((vals, (rows, cols)), shape=(40, n_vars))
a = ad.AnnData(X=X)
a.obs_names = [f"c{i}" for i in range(X.shape[0])]
a.var_names = [f"g{i}" for i in range(n_vars)]
write_cell_archive(a, out)

# see _CHILD: spy on the real decode gate, not on rust_available()
import cellstream.cell._rust_dispatch as _rd
_gate = []
_orig_gate = _rd.use_rust_for_cell_decode
def _spy_gate(*a, **k):
    r = _orig_gate(*a, **k)
    _gate.append(bool(r))
    return r
_rd.use_rust_for_cell_decode = _spy_gate

pa = PackedArchive(out)
_, pbase, plen = pa.member_location("x_payload")
_, obase, olen = pa.member_location("x_offsets")
raw = bytearray(out.read_bytes())
offs = np.frombuffer(bytes(raw[obase:obase + olen]), dtype=np.int64)
f0 = int(offs[0])
# frame = [u32 idx_len][idx blob ...]. Word 0 of the idx blob is SIMDFastPFor's stage-1 count
# and word 1 is `wheremeta`. Verify the count is a nonzero multiple of BlockSize BEFORE
# poisoning, so a future writer change that shortens row 0 fails the test loudly instead of
# silently reverting it to a no-op.
stage1 = struct.unpack_from("<I", raw, pbase + f0 + 4)[0]
print(f"RESULT stage1={stage1}")
if stage1 == 0 or stage1 % 256 != 0:
    raise SystemExit("setup failed: row 0 does not carry a whole number of full blocks")
struct.pack_into("<I", raw, pbase + f0 + 4 + 4, 0x20000000)
out.write_bytes(bytes(raw))

try:
    cellstream.open(out).gather_rows(np.array([0], dtype=np.int64))
    print("RESULT no-raise")
except ValueError:
    print("RESULT valueerror")
except Exception as e:
    print(f"RESULT other {type(e).__name__}")
print("RESULT engine=" + ("rust" if any(_gate) else "python" if _gate else "unknown"))
gc.collect()
print("RESULT survived")
"""


def test_poisoned_wheremeta_in_an_archive_does_not_crash_rust(tmp_path):
    """#269 defect B end-to-end. Rust engine only -- the pure-Python engine decodes through the
    pyfastpfor wheel's own FastPFor, which we cannot patch (#270)."""
    from cellstream.cell._rust_dispatch import rust_available

    if not rust_available():
        pytest.skip("Rust extension not available")
    env = dict(os.environ)
    env.pop("CELLSTREAM_DISABLE_RUST", None)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_POISON, str(tmp_path / "poison.shad"), str(_POISON_ROW_NNZ)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert proc.returncode == 0, (
        f"child exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "RESULT engine=rust" in proc.stdout, proc.stdout
    # the poisoned word must really have been `wheremeta`, not VariableByte payload
    assert f"RESULT stage1={_POISON_ROW_NNZ}" in proc.stdout, proc.stdout
    assert "RESULT valueerror" in proc.stdout, proc.stdout
