"""#273: the pfordelta ENCODE buffer must be sized from a provable bound.

The bound itself is proven by the derivation from the vendored source in
rust/src/pfor.rs; a finite sweep of guard-band cases cannot prove it for all inputs.
What the guard-band tests here do is REGRESSION-PIN that derivation against
representative adversarial cases -- because the vendored encoder enforces no output
capacity, the only observable symptom of a wrong bound is a write past the end.

Large-n round trips live in the subprocess module -- pre-fix, an in-process _pack above
~512k corrupts the heap and aborts the runner.
"""

from __future__ import annotations

import numpy as np
import pytest

pyfastpfor = pytest.importorskip("pyfastpfor")

# _USIZE_MAX is IMPORTED, not redefined as 2**64 - 1: codec.py derives it from
# ctypes.sizeof(c_size_t), and Rust saturates at the platform usize. Hardcoding 64 bits
# would make the saturation and parity tests wrong on a 32-bit platform (Copilot, PR #284).
from cellstream.cell.codec import _USIZE_MAX, _pfor_encode_capacity, _PforEncoder  # noqa: E402

GUARD = 4096
SENTINEL = np.uint32(0xDEADBEEF)


def _worst_case_values(n, seed):
    """Full-entropy uint32: block width 32, no exceptions -- the worst case for the
    dominant term. Compressible input cannot reach the bound and would pass pre-fix."""
    rng = np.random.default_rng(seed)
    return rng.integers(1, 2**32 - 1, size=n, dtype=np.uint64).astype(np.uint32)


def test_capacity_formula():
    assert _pfor_encode_capacity(0) == 4096
    assert _pfor_encode_capacity(1000) == 6096


def test_capacity_saturates_like_rust():
    """Python ints are arbitrary precision; the clamp must mirror Rust's saturation."""
    assert _pfor_encode_capacity(_USIZE_MAX) == _USIZE_MAX
    assert _pfor_encode_capacity(_USIZE_MAX // 2) == _USIZE_MAX


@pytest.mark.parametrize("n", [256, 65_536, 262_144, 524_288, 600_000])
def test_encode_never_writes_past_capacity(n):
    """Pre-fix (cap = n + 1024) this fails at n = 524_288 and 600_000 only; the three
    smaller sizes fit under the old bound too and are coverage, not proof."""
    arr = _worst_case_values(n, seed=n)
    cap = _pfor_encode_capacity(n)
    out = np.full(cap + GUARD, SENTINEL, dtype=np.uint32)
    codec = pyfastpfor.getCodec("simdfastpfor256")
    wl = codec.encodeArray(arr, n, out, cap)
    breached = int((out[cap:] != SENTINEL).sum())
    # Sentinel FIRST: pre-fix wl > cap, so asserting wl first would mask this evidence.
    assert breached == 0, f"n={n}: codec wrote {breached} words past cap {cap}"
    assert wl <= cap, f"n={n}: wl {wl} exceeded capacity {cap}"


def test_pack_actually_uses_the_capacity_helper(monkeypatch):
    """Wiring pin: the guard-band test calls pyfastpfor directly, so it would still pass
    if the helper were correct but _pack kept using n + 1024. Assert _pack's own scratch.

    Two things keep this from silently stopping to test anything:

    * ``enc._out`` is reset to empty first. Otherwise the assertion would lean on
      ``_PforEncoder.__init__``'s 4096-word starting buffer merely *happening* to be
      smaller than the required capacity -- raise that constant above 6096 and a _pack
      still using ``n + 1024`` would sail through (codex, checkpoint 2).
    * a spy pins that ``_pack`` calls the helper at all, rather than arriving at a
      big-enough buffer by some other route.
    """
    seen = []
    real = _pfor_encode_capacity
    monkeypatch.setattr(
        "cellstream.cell.codec._pfor_encode_capacity",
        lambda n: (seen.append(n), real(n))[1],
    )
    enc = _PforEncoder("simdfastpfor256")
    n = 1000
    enc._out = np.empty(0, dtype=np.uint32)  # do not inherit __init__'s head start
    enc._pack(np.arange(n, dtype=np.uint32))
    assert seen == [n], f"_pack did not call _pfor_encode_capacity(n); calls={seen}"
    assert enc._out.size >= _pfor_encode_capacity(n), (
        f"_pack sized its scratch at {enc._out.size}, below the required "
        f"{_pfor_encode_capacity(n)} -- it is not using _pfor_encode_capacity"
    )


@pytest.mark.parametrize(
    "n", [0, 1, 255, 256, 257, 1000, 524_288, 600_000, (_USIZE_MAX - 4096) // 2, _USIZE_MAX]
)
def test_capacity_matches_rust(n):
    """Drift guard. Do NOT skip merely because the attribute is missing -- a forgotten
    pyo3 registration must fail, not silently skip."""
    from cellstream.cell._rust_dispatch import rust_available

    if not rust_available():
        pytest.skip("Rust extension not built or disabled")
    _rust = pytest.importorskip("cellstream.v2._rust")
    assert hasattr(_rust, "_pfor_encode_capacity"), (
        "the Rust extension is present but _pfor_encode_capacity is not registered -- "
        "check the wrap_pyfunction! line in rust/src/lib.rs"
    )
    assert _pfor_encode_capacity(n) == _rust._pfor_encode_capacity(n)


def test_encoded_bytes_are_independent_of_the_declared_capacity():
    """Byte-exactness pin: capacity must not alter the produced bytes.

    The reference MUST use the OLD capacity (n + 1024). Sizing it with
    _pfor_encode_capacity too would make this tautological -- both sides would receive the
    same new capacity, and the test would pass even if the output did depend on it.

    n = 300 is small and compressible, so it fits under the old bound: safe pre-fix.
    """
    arr = np.arange(300, dtype=np.uint32)
    reference = pyfastpfor.getCodec("simdfastpfor256")
    old_cap = arr.size + 1024  # the pre-#273 capacity, deliberately hardcoded
    scratch = np.empty(old_cap, dtype=np.uint32)
    wl = reference.encodeArray(arr, arr.size, scratch, old_cap)
    assert wl <= old_cap, "fixture escaped the old bound; pick a smaller/more compressible n"
    expected = scratch[:wl].tobytes()

    assert _PforEncoder("simdfastpfor256")._pack(arr) == expected


def test_rust_encode_bytes_match_the_python_reference():
    """Cross-engine byte pin: the two engines must still agree after the capacity change."""
    from cellstream.cell._rust_dispatch import rust_available

    if not rust_available():
        pytest.skip("Rust extension not built or disabled")
    _rust = pytest.importorskip("cellstream.v2._rust")
    arr = np.arange(300, dtype=np.uint32)
    reference = pyfastpfor.getCodec("simdfastpfor256")
    scratch = np.empty(arr.size + 1024, dtype=np.uint32)
    wl = reference.encodeArray(arr, arr.size, scratch, scratch.size)
    assert _rust._pfor_encode_words(arr).tobytes() == scratch[:wl].tobytes()
