"""#273: a single cell above ~512k incompressible values must not corrupt the heap.

One case per process: the pre-fix failure is SIGSEGV / SIGABRT, which an in-process test
cannot observe (it would take the runner with it).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("pyfastpfor")

# 600k > the ~512k threshold at which n + 1024 words of headroom is exceeded.
NNZ = 600_000
N_VARS = NNZ + 100_000

ROUNDTRIP = textwrap.dedent(
    """
    import numpy as np
    from cellstream.cell.codec import _pfor_encode_capacity, _PforEncoder
    import pyfastpfor

    n = {nnz}
    rng = np.random.default_rng(7)
    arr = rng.integers(1, 2**32 - 1, size=n, dtype=np.uint64).astype(np.uint32)
    blob = _PforEncoder("simdfastpfor256")._pack(arr)
    words = np.frombuffer(blob, dtype=np.uint32)
    # Deliberately oversized rather than exactly n: simdfastpfor256's VariableByte tail emits
    # one u32 per varint in the INPUT and ignores the declared capacity, so a decode buffer of
    # exactly `expected` is the #269 defect-A overflow. The encode capacity is a convenient
    # generous bound here; 4.8 MB is immaterial next to this test's payload. (Copilot #284 cp2
    # suggested tightening this to n -- rejected for that reason.)
    out = np.empty(_pfor_encode_capacity(n), dtype=np.uint32)
    got = pyfastpfor.getCodec("simdfastpfor256").decodeArray(words, words.size, out, n)
    assert got == n, f"decoded {{got}} of {{n}}"
    np.testing.assert_array_equal(out[:n], arr)
    print("OK")
    """
).format(nnz=NNZ)

E2E = textwrap.dedent(
    """
    import tempfile, pathlib, warnings
    import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
    warnings.filterwarnings("ignore")
    import cellstream
    import cellstream.cell._rust_dispatch as _rd
    from cellstream.cell._rust_dispatch import rust_available

    # Report the engine the writer ACTUALLY dispatched to. `rust_available()` only says the
    # extension imported; write.py gates on use_rust_for_cell_encode(), which can still fall
    # back on a dtype combination it does not accept -- which would leave BOTH parametrisations
    # quietly testing the Python encoder. Spy on the real gate, mirroring
    # test_decode_bounds_subprocess.py (Copilot, PR #284 checkpoint 2).
    _gate = []
    _orig_gate = _rd.use_rust_for_cell_encode
    def _spy_gate(*a, **k):
        r = _orig_gate(*a, **k)
        _gate.append(bool(r))
        return r
    _rd.use_rust_for_cell_encode = _spy_gate

    if __name__ == "__main__":
        import sys
        want_rust = sys.argv[1] == "rust"
        # Fail loudly rather than silently testing the Python engine twice.
        assert rust_available() == want_rust, (
            f"engine mismatch: rust_available()={{rust_available()}}, wanted {{want_rust}}"
        )
        nnz, n_vars = {nnz}, {n_vars}
        rng = np.random.default_rng(0)
        cols = np.sort(rng.choice(n_vars, size=nnz, replace=False)).astype(np.int64)
        # Full-entropy uint32 VALUES are what reaches the bound; index gaps cannot.
        vals = rng.integers(1, 2**32 - 1, size=nnz, dtype=np.uint64).astype(np.uint32)
        X = sp.csr_matrix(
            (vals.astype(np.float64), cols, np.array([0, nnz, nnz], dtype=np.int64)),
            shape=(2, n_vars),
        )
        a = ad.AnnData(
            X=X,
            obs=pd.DataFrame(index=["c0", "c1"]),
            # np.arange().astype(str) rather than a 700k-element list comprehension.
            var=pd.DataFrame(index=np.arange(n_vars).astype(str)),
        )
        t = pathlib.Path(tempfile.mkdtemp())
        p = t / "big.shad"
        # n_workers=1 keeps the encode IN THIS PROCESS, so the gate spy above can observe it;
        # with spawned workers the gate is called in the child and the spy records nothing.
        # Pin the codec explicitly -- layout='cell' also supports zstd and auto-selects.
        cellstream.write_sharded(a, str(p), layout="cell", codec="pfordelta", n_workers=1)
        assert _gate, "use_rust_for_cell_encode was never called -- the spy is not wired in"
        assert any(_gate) == want_rust, (
            f"writer dispatched to the wrong engine: gate calls={{_gate}}, wanted "
            f"rust={{want_rust}}"
        )
        back = cellstream.read_h5ad(str(p))
        assert back.X.nnz == nnz, f"nnz {{back.X.nnz}} != {{nnz}}"
        assert (back.X != X).nnz == 0, "round-trip mismatch"
        print("OK gate=", _gate)
    """
).format(nnz=NNZ, n_vars=N_VARS)


def _env(disable_rust):
    env = dict(os.environ)
    if disable_rust:
        env["CELLSTREAM_DISABLE_RUST"] = "1"
    else:
        env.pop("CELLSTREAM_DISABLE_RUST", None)
    return env


def _why(rc):
    """subprocess.run reports a signal death as NEGATIVE (-11/-6), not the shell's 139/134."""
    if rc < 0:
        try:
            return f"killed by {signal.Signals(-rc).name}"
        except ValueError:
            return f"killed by signal {-rc}"
    return f"exit {rc}"


def _assert_clean(proc):
    # SINGLE backslashes: this is module-level test code, not one of the textwrap.dedent
    # subprocess templates above -- \\n here would print a literal "\n" (Gemini, PR #284).
    assert proc.returncode == 0, (
        f"{_why(proc.returncode)} (returncode={proc.returncode}; the pre-fix heap "
        f"corruption shows as SIGSEGV or SIGABRT)\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"
    )
    assert "OK" in proc.stdout


def test_large_cell_pack_round_trips():
    """_pack allocates exactly cap, so pre-fix this aborts the process."""
    _assert_clean(
        subprocess.run(
            [sys.executable, "-c", ROUNDTRIP],
            capture_output=True,
            text=True,
            env=_env(True),
            timeout=900,
        )
    )


@pytest.mark.parametrize("engine", ["rust", "python"])
def test_large_incompressible_cell_encodes_without_corrupting_the_heap(engine):
    if engine == "rust":
        from cellstream.cell._rust_dispatch import rust_available

        if not rust_available():
            pytest.skip("Rust extension not built")
    _assert_clean(
        subprocess.run(
            [sys.executable, "-c", E2E, engine],
            capture_output=True,
            text=True,
            env=_env(engine == "python"),
            timeout=900,
        )
    )
