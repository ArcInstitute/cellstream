# Local patches to vendored third-party sources

Anything listed here MUST be re-applied when the vendored source is updated. A bump that drops one
of these silently reverts a memory-safety fix.

## FastPFor — `headers/simdfastpfor.h`

**Applied:** 2026-07-28 · **Issue:** #269 (pre-rename tracker)

`SIMDFastPFor::__decodeArray` took its displacements straight out of the compressed blob and bounded
none of them against the input. A single poisoned word in an otherwise well-formed blob read far past
the end — `_pfor_decode_words([1, 0x20000000, 0, ...], n)` segfaulted at every `n`, confirming the
fault is input-side and independent of output sizing. (That original repro no longer reaches
`__decodeArray`: change 1 below rejects a leading count of `1` first. The regression test uses `256`.)

Upstream cannot see the problem from where it stands: `__decodeArray` has no input-length parameter
at all (its `length` is an *out* parameter reporting words consumed), and `decodeArray` left its
`length` **unnamed under `NDEBUG`**, using it only for a trailing `assert`.

**Changes:**
1. `decodeArray`: `length` always named, `initin` hoisted out of the `NDEBUG` guard, `inend`
   computed and passed down. Removing the `#ifdef` is deliberate: upstream named `length` only under
   `#ifndef NDEBUG` and used it solely for a trailing `assert`, so `-DNDEBUG` silently dropped the
   only available bound. Here `length` feeds `inend`, which is ordinary code — the bound survives
   `-DNDEBUG` rather than the build refusing to compile. (It compiles fine under `-DNDEBUG`; the gate
   below exercises both.) Also rejects a stage-one count that is not a
   multiple of `BlockSize` (it would advance `out` past what was written) and replaces the
   out-of-object `PageSize + out` comparison with a difference.
2. `__decodeArray`: takes `const uint32_t *const inend`. Bounds **packed** walks against
   `packedend = headerin + wheremeta` and **metadata** walks against `inend` — two different
   regions. Validates `wheremeta`, `bytesize`, each exception stream's size, `b`/`maxbits`,
   exception positions, and per-stream consumption. The trailing `assert(in == headerin +
   wheremeta)` becomes a throw.
3. `unpackmesimd`: zero-initialise the partial `buffer`, and advance `in` directly by
   `ceil(remaining*bit/32)` instead of advancing by `bit` and then subtracting. The two are
   algebraically identical (`bit - floor((32-r)*bit/32) == ceil(r*bit/32)` for every `r` in
   [0,32)), so this is not a behaviour change — but the original transiently formed an
   out-of-object pointer, which is UB even though nothing dereferences it.
4. `#include <stdexcept>`.

**Accounting note — do not "simplify" this.** `unpackmesimd` consumes `1 + ceil(size*bit/32)` words,
NOT `1 + ceil(size/32)*bit`. The latter over-counts (`size=1, bit=32` → 33 vs 2) and would reject
valid archives: a 256-value block of 255 zeroes and one `UINT32_MAX` produces exactly that stream.
`tests/cell/test_pfor_rust_kernel.py::test_rust_decode_still_handles_a_single_high_exception` pins it.

The throws are caught by `rust/csrc/pfor_shim.cpp`'s existing `catch (...)`, which returns 0, and
`pfor.rs::ffi_decode` reports that as a catchable corrupt-frame `ValueError`. Valid frames never trip
the bounds, so decode output is byte-identical — the Rust-vs-pyfastpfor parity tests are the net.

**Verification gate — `rust/asan/run.sh`. Re-run this after any FastPFor bump.** The pytest suite can
only observe an out-of-bounds access that leaves mapped memory and kills the process. Defect B is a
*read*, and a read that stays inside the heap is invisible to it — not hypothetically: pre-fix, the
`cexcept-255` mutation walked off the end of the exception byte container, its first invalid read
landing one byte past the input buffer. That byte was mapped, so the decode still reported the
expected value count and the frame was **accepted**. Silent data corruption, no crash, no error.
(ASan halts at that first invalid byte; unsanitized, the walk carries on past it.)

The gate builds `csrc/pfor_shim.cpp` plus these vendored sources with `-fsanitize=address,undefined`
(no Rust nightly needed) and makes 38,701 counted decodes — exhaustive single-word *and* single-byte
mutation of valid frames, truncation fuzz over every prefix, multi-page frames — plus two valid
round-trip checks that call the decoder directly. It runs the current tree twice, with assertions
live (what `rust/build.rs` ships) and under `-DNDEBUG`; both must report `HARNESS CLEAN`, exit 0.

Always run the negative control alongside it:

```bash
bash rust/asan/run.sh                       # expect: HARNESS CLEAN ×2, exit 0
bash rust/asan/run.sh --prefix-ref 20768f3d # expect: heap-buffer-overflow in __decodeArray
```

The control is the part that makes the clean run mean anything, and it is deliberately **not** the
full sweep. Against the pre-patch header the sweep dies on the first poisoned `wheremeta` — a wild
pointer that segfaults with or without a sanitizer, which proves nothing about reads that stay inside
mapped memory. So the control runs one isolated case, `gate cexcept-overclaim`, the mutation that
pre-fix neither crashed nor raised:

```
MUTATED DECODE REACHED
ERROR: AddressSanitizer: heap-buffer-overflow ... READ of size 1
0 bytes to the right of 356-byte region
    #0 FastPForLib::SIMDFastPFor<8u>::__decodeArray
    #3 fastpfor256_decode csrc/pfor_shim.cpp:21
```

`run.sh` **fails** the control on a clean run or a harness fault (exit 2/3), and then requires all
three of `MUTATED DECODE REACHED`, `heap-buffer-overflow` and `__decodeArray`. Each rules out a
different false pass: the marker proves execution got past the unmutated precondition decode;
`heap-buffer-overflow` excludes an `AddressSanitizer:DEADLYSIGNAL`, i.e. a plain segfault that an
exit-code test would already have caught; and `__decodeArray` places it in the function under test.
So a FastPFor bump cannot quietly turn the gate vacuous. Self-tested by pointing `--prefix-ref` at a
ref that is *not* pre-fix, which correctly reports `VACUOUS GATE` and exits 1.

**Not fixed here:** the same defect is live in the `pyfastpfor` wheel used by the pure-Python engine,
which compiles its own copy of FastPFor. See #270.

## Encode path: NOT patched — capacity is the caller's responsibility (#273)

`SIMDFastPFor::encodeArray` / `__encodeArray` enforce **no** output capacity. `__encodeArray`
never reads its `nvalue` argument; it only assigns `nvalue = out - initout` after writing, and
`SIMDFastPFor::encodeArray`'s only reaction to an overrun is a `fprintf(stderr, ...)` after the
fact. `CompositeCodec::_encodeArray`'s `nvalue1 > nvalue` throw is guarded by
`if (roundedlength < length)`, so it is skipped entirely when the length is a multiple of
`BlockSize`.

We therefore size the buffer ourselves: `pfor_encode_capacity()` in `rust/src/pfor.rs`, mirrored
by `_pfor_encode_capacity()` in `src/cellstream/cell/codec.py`.

**A FastPFor bump must re-derive that formula.** It does not depend only on `__encodeArray`'s
write sites — every one of these is load-bearing, and a change to any of them can invalidate the
bound silently:

| what | why the bound depends on it |
|---|---|
| `__encodeArray`'s write sites — `packblockupsimd`, the `bytescontainer` memcpy, `packmeupwithoutmasksimd` | the per-page and per-block terms, incl. the ≤ `31*(1+31)` = 992 words of exception-container padding |
| `CompositeCodec`'s `roundedlength` split | decides how many values go to the block codec vs the tail |
| `BlockSize` (256) | the `bytescontainer` ≤ 2 bytes/block term |
| `PageSize` (65536) | the per-page 996-word term; a smaller page multiplies it |
| `VariableByte`'s worst case (5 bytes per `uint32`) | the ≤ 64-word tail premium for `r = n mod 256` values |
| `getBestBFromData`'s cost model | the "never worse than plain packing at `b = maxb`" argument that caps packed data at ≤ 1 word/value |

It is pinned by `encode_never_writes_past_pfor_encode_capacity` in `bounds_tests`, which encodes
into a sentinel-filled guard band — but note that a finite sweep regression-tests the derivation,
it does not prove it.
