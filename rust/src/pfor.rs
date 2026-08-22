//! Per-cell pfordelta (simdfastpfor256) frame codec, byte-exact via vendored FastPFor.
//! The codec-agnostic cell scaffolding (row loop, threading, output writing) lives in
//! cell.rs -- for BOTH decode and encode; this module owns the FastPFor ffi,
//! `decode_blob_u32`, and the pfordelta frame body (`encode_frame`).
#![cfg(target_arch = "x86_64")]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// EncodeError / encode_err_to_py / CountU32 / IdxU32 / CellScratch moved to cell.rs with the
// rest of the codec-agnostic encode scaffolding. encode_err_to_py is used by
// _pfor_encode_words below (Codex #220 r1: omitting it from this list breaks the build).
use crate::cell::{encode_err_to_py, CellScratch, CountU32, EncodeError, IdxU32};
use crate::decode::DecodeError;

extern "C" {
    fn fastpfor256_decode(inp: *const u32, in_len: usize, out: *mut u32, n: usize) -> usize;
    fn fastpfor256_encode(inp: *const u32, n: usize, out: *mut u32, out_cap: usize) -> usize;
}

/// Output capacity (in u32 words) to hand `fastpfor256_encode` for `n` input values.
///
/// The vendored encoder enforces NO capacity: `CompositeCodec::_encodeArray`'s
/// `nvalue1 > nvalue` throw is skipped whenever `length` is an exact multiple of
/// `BlockSize`, `SIMDFastPFor::encodeArray` passes `thisnvalue(0)` down instead of the
/// remaining space, and `__encodeArray` never reads `nvalue` at all -- it only assigns
/// `nvalue = out - initout` on the way out, after writing. The caller must therefore
/// supply a sufficient buffer; the old `n + 1024` was not one, and a single cell above
/// ~512k incompressible values corrupted the heap (#273).
///
/// Worst case, derived from the vendored source. `simdfastpfor256` is
/// `CompositeCodec<SIMDFastPFor<8>, VariableByte>`: the block codec handles
/// `roundedlength = floor(n/256)*256` values and `VariableByte` handles the `r = n mod 256`
/// tail. Every term is rounded UP -- an earlier draft of this comment dropped the tail and
/// the ceilings and was therefore not a proof.
///   packed data + exceptions <= 1 word/value over the blocked region  (`getBestBFromData`
///     minimises against `bestcost = maxb*BlockSize`, and `overheadofeachexcept = 8` bits
///     charges the per-exception `bytescontainer` byte, so the chosen encoding never costs
///     more than plain packing at b = maxb, i.e. <= maxb*256 <= 8192 bits = 256 words per
///     256-value block)
///   + `bytescontainer` <= 2 bytes/block                   -> <= ceil(n/512) words
///   + 3 words/page (page header, bytescontainersize, bitmap) + 1 word for rounding the
///     page's byte container up to a word boundary;  PageSize = 65536
///   + `packmeupwithoutmasksimd` padding: each exception container is resized up to a
///     multiple of 32 and it is called for up to 31 bit-widths -> <= 31*(1+31) = 992
///     words/page
///   + the `VariableByte` tail: <= 5 bytes per uint32, i.e. ceil(5r/4) words for r values
///     where the `n` term above already charges r -- a premium of <= 64 words (r <= 255)
///   + 1 global length word
///
/// Total = n + ceil(n/512) + ceil(n/PageSize)*996 + 65 -- i.e. ~1.017n for large n, and
/// <= 1318 words at n=256. (The blank line above is load-bearing: the `+ ...` lines are a
/// markdown list, so without it clippy's doc_lazy_continuation rejects this as an
/// unindented list continuation.)
///
/// `2n + 4096` clears that at every n with a wide margin (an 85-case adversarial guard-band
/// sweep peaked at 49.9% of this capacity). Deliberately loose, exactly like
/// `max_valid_blob_bytes`: a tight formula mirroring vendored internals is load-bearing
/// arithmetic that a FastPFor bump could invalidate silently. The cost is a reused scratch
/// buffer at 2x nnz words, not a per-cell allocation.
///
/// Capacity never changes the words written -- only whether the codec throws -- so this
/// cannot alter on-disk bytes.
///
/// MUST match codec.py `_pfor_encode_capacity`.
pub(crate) fn pfor_encode_capacity(n: usize) -> usize {
    n.saturating_mul(2).saturating_add(4096)
}

/// Pack `inp` into `out` (a REUSED scratch buffer) as simdfastpfor256 words, byte-exact vs
/// pyfastpfor. Empty input clears `out` and returns 0 WITHOUT calling the C++ codec, mirroring
/// codec.py `_pack` (n==0 -> b""). On return `out.len()` == the word count. Reusing `out` across
/// cells eliminates the per-cell heap allocation in the hot loop (mirrors codec.py's reused
/// `self._out` and decode_rows_into's reused `scratch`; Gemini #212 r3).
///
/// The `truncate` below means the next cell's `resize(cap, 0)` re-zeroes ~`cap - wl` words, which
/// #273's larger capacity grew from O(1024) to O(n). Codex (checkpoint 2) proposed dropping the
/// truncate and returning `wl` as the only valid length. MEASURED and REJECTED: end-to-end
/// `write_sharded(layout='cell')` came out at 0.198s vs 0.205s for 4000x3000-nnz cells and 0.435s
/// vs 0.426s for 200x120k-nnz cells -- within run-to-run noise both ways. Not worth it, because
/// `out.len() != wl` would become a silent archive-corrupting trap for any future caller that
/// slices by `.len()` (which `encode_frame` did until it was rewritten for the experiment).
pub fn ffi_encode_into(inp: &[u32], out: &mut Vec<u32>) -> Result<usize, EncodeError> {
    if inp.is_empty() {
        out.clear();
        return Ok(0);
    }
    let cap = pfor_encode_capacity(inp.len());
    out.resize(cap, 0); // reuses capacity across cells; fastpfor overwrites [0, wl)
    let wl = unsafe { fastpfor256_encode(inp.as_ptr(), inp.len(), out.as_mut_ptr(), cap) };
    if wl > cap {
        // With a correct capacity this is unreachable for a well-behaved codec; its
        // remaining job is the shim's SIZE_MAX (== usize::MAX) throw sentinel. (#273)
        return Err(EncodeError::Value(
            "pfordelta encode failed (codec error or output overrun)".into(),
        ));
    }
    out.truncate(wl);
    Ok(wl)
}

/// Allocating convenience wrapper (test hook / non-hot-path use only).
pub fn ffi_encode(inp: &[u32]) -> Result<Vec<u32>, EncodeError> {
    let mut out = Vec::new();
    ffi_encode_into(inp, &mut out)?;
    Ok(out)
}

/// Test hook: pack a u32 array to simdfastpfor256 words (byte-exact-parity probe).
#[pyfunction]
pub fn _pfor_encode_words(
    py: Python<'_>,
    arr: PyReadonlyArray1<u32>,
) -> PyResult<Py<PyArray1<u32>>> {
    let inp = arr.as_slice()?;
    let out = py
        .allow_threads(|| ffi_encode(inp))
        .map_err(encode_err_to_py)?;
    Ok(out.into_pyarray_bound(py).unbind())
}

/// Test hook: the encode capacity for `n` values, so the Python mirror can be pinned to it.
#[pyfunction]
pub fn _pfor_encode_capacity(n: usize) -> usize {
    pfor_encode_capacity(n)
}

/// The largest codec blob a VALID pfordelta frame can carry for `nnz` values.
///
/// Deliberately loose -- 2x raw u32 storage plus 64 KiB, against a real encoding of ~2-3
/// bytes/value -- so it cannot reject a valid frame. Its only job is to bound the decode
/// allocation in `decode_blob_u32`, so a corrupt/tampered `x_offsets` that makes one frame span
/// the whole payload cannot turn `expected + blob.len()` into an allocation DoS. Correctness
/// stays with the `got != expected` check.
///
/// Independent of `pfor_encode_capacity`: the encode-output capacity and the valid-decode-blob
/// bound serve separate purposes, and coupling them would make one formula load-bearing in two
/// independent places.
///
/// MUST match codec.py `_max_valid_blob_bytes`.
pub(crate) fn max_valid_blob_bytes(nnz: usize) -> usize {
    nnz.saturating_mul(8).saturating_add(65_536)
}

/// Decode `expected` values from the packed `inp` words into `out`.
///
/// `out` is a REUSED scratch buffer that this function sizes itself -- callers must not
/// pre-slice it. `expected` is both the capacity declared to the codec and the count the
/// decode must produce.
///
/// The split is the fix for #269 defect A. `simdfastpfor256` is
/// `CompositeCodec<SIMDFastPFor, VariableByte>`: stage 1 honours the declared capacity (it
/// throws `NotEnoughStorage`), but the `VariableByte` tail emits one u32 per varint in the
/// INPUT, writing at `out + k` with no regard for capacity. So the reachable bound is
/// `expected + input_bytes`, and handing the codec a buffer sized exactly `expected` let the
/// tail write past the end -- an 88-slot buffer took 80,135 values in the filed repro. Sizing
/// the ALLOCATION to that bound makes the write in-bounds by construction; `got != expected`
/// then reports the corruption instead of merely observing its aftermath.
pub fn ffi_decode(inp: &[u32], out: &mut Vec<u32>, expected: usize) -> Result<(), DecodeError> {
    if expected == 0 {
        return Ok(());
    }
    if inp.is_empty() {
        return Err(DecodeError::Corrupt(
            "corrupt pfordelta frame: empty codec blob for non-empty output".into(),
        ));
    }
    // provable upper bound on what the composite codec can write; grow-only across rows
    let need = expected.saturating_add(inp.len().saturating_mul(4));
    if out.len() < need {
        out.resize(need, 0);
    }
    let got = unsafe { fastpfor256_decode(inp.as_ptr(), inp.len(), out.as_mut_ptr(), expected) };
    if got != expected {
        return Err(DecodeError::Corrupt(format!(
            "corrupt pfordelta frame: decoded {got} of {expected} expected values"
        )));
    }
    Ok(())
}

pub fn decode_blob_u32(
    blob: &[u8],
    out: &mut Vec<u32>,
    expected: usize,
) -> Result<(), DecodeError> {
    if expected == 0 {
        return Ok(());
    }
    if blob.is_empty() || !blob.len().is_multiple_of(4) {
        return Err(DecodeError::Corrupt(
            "corrupt pfordelta frame: codec blob is empty or not 4-byte aligned".into(),
        ));
    }
    // #269 layer 2: cap the blob BEFORE ffi_decode sizes the allocation from it.
    let cap = max_valid_blob_bytes(expected);
    if blob.len() > cap {
        return Err(DecodeError::Corrupt(format!(
            "corrupt pfordelta frame: codec blob is {} bytes, exceeds the maximum {} for nnz={}",
            blob.len(),
            cap,
            expected
        )));
    }
    if (blob.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u32>()) {
        let words =
            unsafe { std::slice::from_raw_parts(blob.as_ptr() as *const u32, blob.len() / 4) };
        ffi_decode(words, out, expected)
    } else {
        // `as_chunks::<4>()` rather than `chunks_exact(4)`: identical semantics (both yield
        // blob.len()/4 groups and drop the same remainder) but it hands back `&[u8; 4]` directly,
        // so the infallible-yet-panicking `try_into().unwrap()` goes away. Also what clippy 1.98's
        // new `chunks_exact_to_as_chunks` lint demands -- CI's `stable` toolchain floats, so a new
        // lint on untouched code turns every PR red; pinning it is #310.
        let words: Vec<u32> = blob
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        ffi_decode(&words, out, expected)
    }
}

/// Test hook: decode a single simdfastpfor256 blob to a fresh u32 array (byte-exact-parity probe).
#[pyfunction]
pub fn _pfor_decode_words(
    py: Python<'_>,
    blob: PyReadonlyArray1<u32>,
    n: usize,
) -> PyResult<Py<PyArray1<u32>>> {
    let inp = blob.as_slice()?;
    let mut out: Vec<u32> = Vec::new();
    py.allow_threads(|| ffi_decode(inp, &mut out, n))
        .map_err(crate::decode_err_to_py)?;
    // ffi_decode over-allocates by design (#269); the hook returns exactly `n`.
    out.truncate(n);
    Ok(out.into_pyarray_bound(py).unbind())
}

/// Assemble one cell's frame into `scratch.frame` (cleared first); returns the frame length.
/// Byte-exact with codec.py `_PforEncoder.encode`: [u32 idx_blob_len][idx_blob][val_blob],
/// idx_blob = pack(delta-gaps of sorted u32 indices), val_blob = pack(u32 values). Tracks the
/// max value in `max_v` during the cast loop. All buffers live in `scratch` and are reused.
pub(crate) fn encode_frame<D: CountU32, I: IdxU32>(
    idx_src: &[I],
    val_src: &[D],
    scratch: &mut CellScratch,
    max_v: &mut u64,
    check_duplicates: bool,
) -> Result<usize, EncodeError> {
    let nnz = idx_src.len();
    scratch.gaps.clear();
    scratch.gaps.reserve(nnz);
    // delta-gaps with the codec.py sortedness guard (idx[k] >= idx[k-1] on u32)
    let mut prev: u32 = 0;
    for (k, &raw) in idx_src.iter().enumerate() {
        let cur = raw.idx_u32();
        if k == 0 {
            scratch.gaps.push(cur);
        } else {
            if cur < prev {
                return Err(EncodeError::Value(
                    "pfordelta requires non-decreasing (sorted) indices".into(),
                ));
            }
            // A duplicate (cell, gene) entry is exactly a zero gap on sorted indices; fold the
            // fail-loud check into the delta loop (near-zero cost). Off => a zero gap encodes fine.
            if check_duplicates && cur == prev {
                return Err(EncodeError::Value(format!(
                    "encode_cells: duplicate column index {cur} within a cell \
                     (require_unique_indices=True); pass require_unique_indices=False \
                     to permit duplicate (cell, gene) entries"
                )));
            }
            scratch.gaps.push(cur - prev);
        }
        prev = cur;
    }
    scratch.vals_u32.clear();
    scratch.vals_u32.reserve(nnz);
    for &v in val_src {
        let vu = v.count_u32()?;
        if vu as u64 > *max_v {
            *max_v = vu as u64;
        }
        scratch.vals_u32.push(vu);
    }
    // disjoint field borrows: &scratch.gaps (read) + &mut scratch.idx_words (write) are
    // distinct fields, so the borrow checker permits both simultaneously.
    ffi_encode_into(&scratch.gaps, &mut scratch.idx_words)?;
    ffi_encode_into(&scratch.vals_u32, &mut scratch.val_words)?;
    scratch.frame.clear();
    let idx_blob_len = (scratch.idx_words.len() * 4) as u32;
    scratch.frame.extend_from_slice(&idx_blob_len.to_le_bytes());
    for w in &scratch.idx_words {
        scratch.frame.extend_from_slice(&w.to_le_bytes());
    }
    for w in &scratch.val_words {
        scratch.frame.extend_from_slice(&w.to_le_bytes());
    }
    Ok(scratch.frame.len())
}

#[cfg(test)]
mod bounds_tests {
    use super::*;

    /// `decode_blob_u32` has two arms: a zero-copy reinterpret when the blob is 4-byte aligned, and
    /// a byte-regrouping fallback when it is not. NOTHING in the Python suite reaches the fallback --
    /// a numpy-backed payload is aligned -- so the two arms are proven equal here instead. Written
    /// when the fallback was rewritten from `chunks_exact(4).map(try_into().unwrap())` to
    /// `as_chunks::<4>()`; an untested rewrite in a decode hot path is how #269 and #273 happened.
    #[test]
    fn decode_blob_unaligned_fallback_matches_the_aligned_arm() {
        let src: Vec<u32> = (0..300u32).map(|i| i * 7 + 1).collect();
        let packed = ffi_encode(&src).unwrap_or_else(|_| panic!("encode failed"));
        let aligned: Vec<u8> = packed.iter().flat_map(|w| w.to_le_bytes()).collect();
        let align = std::mem::align_of::<u32>();
        assert!((aligned.as_ptr() as usize).is_multiple_of(align));

        // A single leading pad byte is NOT enough: a Vec<u8> is only 1-byte aligned, so
        // `&padded[1..]` lands on a 4-byte boundary whenever the allocation itself did -- the first
        // draft of this test did exactly that, and its own guard assertion caught it. Instead
        // over-allocate, read the real base address, and pick the offset that provably is NOT a
        // multiple of 4.
        let mut padded = vec![0u8; aligned.len() + align];
        let base = padded.as_ptr() as usize;
        let off = (align + 1 - (base % align)) % align;
        padded[off..off + aligned.len()].copy_from_slice(&aligned);
        let unaligned = &padded[off..off + aligned.len()];
        assert!(
            !(unaligned.as_ptr() as usize).is_multiple_of(align),
            "the slice came out aligned; this test would prove nothing"
        );

        let mut from_aligned = Vec::new();
        decode_blob_u32(&aligned, &mut from_aligned, src.len()).unwrap();
        let mut from_unaligned = Vec::new();
        decode_blob_u32(unaligned, &mut from_unaligned, src.len()).unwrap();

        // `out` is deliberately left LONGER than `expected` (#269 defect A -- see
        // decode_blob_allocates_beyond_expected), so compare the decoded prefix, then require the
        // two arms to agree on the whole buffer including that slack.
        assert_eq!(&from_aligned[..src.len()], &src[..]);
        assert_eq!(&from_unaligned[..src.len()], &src[..]);
        assert_eq!(from_unaligned, from_aligned);
    }

    #[test]
    fn pfor_encode_capacity_is_generous_and_saturating() {
        assert_eq!(pfor_encode_capacity(0), 4096);
        assert_eq!(pfor_encode_capacity(1000), 6096);
        // saturating at both the multiply and the add
        assert_eq!(pfor_encode_capacity(usize::MAX), usize::MAX);
        assert_eq!(pfor_encode_capacity(usize::MAX / 2), usize::MAX);
    }

    /// The load-bearing test: the vendored encoder enforces NO output capacity
    /// (`SIMDFastPFor::__encodeArray` never reads its `nvalue` argument), so the bound
    /// can only be proven by watching for writes past it. Encode into a sentinel-filled
    /// guard band and assert nothing beyond `cap` was touched.
    ///
    /// Pre-fix (cap = n + 1024) this fails at n = 524_288 and n = 600_000 only; the three
    /// smaller sizes fit under the old bound too and are coverage, not proof.
    #[test]
    fn encode_never_writes_past_pfor_encode_capacity() {
        const GUARD: usize = 4096;
        const SENTINEL: u32 = 0xDEAD_BEEF;
        // Full-entropy values: block width 32, no exceptions -> worst case for the
        // dominant term. Compressible input cannot reach the bound.
        let mut state: u64 = 0x2026_0729;
        for &n in &[256usize, 65_536, 262_144, 524_288, 600_000] {
            let inp: Vec<u32> = (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state >> 32) as u32
                })
                .collect();
            let cap = pfor_encode_capacity(n);
            let mut buf = vec![SENTINEL; cap + GUARD];
            let wl = unsafe { fastpfor256_encode(inp.as_ptr(), n, buf.as_mut_ptr(), cap) };
            let breached = buf[cap..].iter().filter(|&&w| w != SENTINEL).count();
            assert_eq!(
                breached, 0,
                "n={n}: codec wrote {breached} words past cap {cap}"
            );
            assert!(wl <= cap, "n={n}: wl {wl} exceeded capacity {cap}");
        }
    }

    #[test]
    fn max_valid_blob_bytes_is_generous_and_saturating() {
        assert_eq!(max_valid_blob_bytes(0), 65_536);
        assert_eq!(max_valid_blob_bytes(100), 66_336);
        // must not overflow on an absurd nnz
        assert_eq!(max_valid_blob_bytes(usize::MAX), usize::MAX);
    }

    #[test]
    fn decode_blob_rejects_blob_larger_than_the_bound() {
        // 70_000 bytes of blob for nnz=1 -> over 8*1 + 65536
        let blob = vec![0u8; 70_000];
        let mut out: Vec<u32> = Vec::new();
        let err = decode_blob_u32(&blob, &mut out, 1).unwrap_err();
        // Equality on the payload, not `contains` on the Debug form: a substring match still
        // passes if the message grows a prefix or suffix, so it does not actually pin
        // character-identical parity. The twin assertion is
        // tests/cell/test_errors.py::test_pfordelta_rejects_oversized_blob_before_the_codec_runs
        // -- same nnz, same blob size, so a divergence in either the text or the bound fails one
        // of the two.
        let DecodeError::Corrupt(actual) = err else {
            panic!("expected DecodeError::Corrupt, got {err:?}");
        };
        assert_eq!(
            actual,
            "corrupt pfordelta frame: codec blob is 70000 bytes, \
             exceeds the maximum 65544 for nnz=1"
        );
    }

    #[test]
    fn decode_blob_allocates_beyond_expected() {
        // A well-formed blob for 300 values: the allocation must exceed `expected`
        // so the VariableByte tail cannot write past the end (#269 defect A).
        let src: Vec<u32> = (0..300u32).collect();
        let packed = ffi_encode(&src).unwrap_or_else(|_| panic!("encode failed"));
        let blob: Vec<u8> = packed.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut out: Vec<u32> = Vec::new();
        decode_blob_u32(&blob, &mut out, 300).unwrap();
        assert!(
            out.len() >= 300 + blob.len(),
            "allocation {} too small",
            out.len()
        );
        assert_eq!(&out[..300], &src[..]);
    }
}
