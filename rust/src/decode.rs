//! Fused v2 decode kernel: the inverse of rust/src/shx.rs (zstd -> un-bytedelta
//! -> un-byteshuffle) fused with a cast straight into the caller's output. Byte
//! layout is plane-major: T planes of N elements, raw[p*N + e] = element e's
//! byte p. Mirrors src/cellstream/v2/codec.byte_filter_decode.

use crate::narrow::NarrowError;
use crate::shx_read::StreamSpec;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Decode-path error model (#133). Threaded through the whole decode chain; the FFI
/// boundary (rust/src/lib.rs) maps each variant to a typed Python exception.
#[derive(Debug)]
pub enum DecodeError {
    /// I/O failure reading a shard (open/seek/read/EOF). Maps to the OSError family.
    Io(std::io::Error),
    /// Malformed/corrupt archive content or an unsupported on-disk dtype (foreign
    /// archive). Maps to ValueError. Message MUST NOT contain "lossy".
    Corrupt(String),
    /// Tier-3 lossy cast: a decoded value does not fit the requested dtype losslessly.
    /// Maps to LossyCastError (text unchanged).
    Lossy(NarrowError),
}

impl From<std::io::Error> for DecodeError {
    fn from(e: std::io::Error) -> Self {
        // parse_header (shx_read.rs) already encodes CONTENT errors (bad magic,
        // oversized header_size, serde parse) as ErrorKind::InvalidData; genuine I/O
        // (NotFound, UnexpectedEof on truncation, seek failures) keeps its own kind.
        // One impl routes both correctly through `?`.
        match e.kind() {
            std::io::ErrorKind::InvalidData => DecodeError::Corrupt(e.to_string()),
            _ => DecodeError::Io(e),
        }
    }
}

// DoS guard: a corrupt chunk descriptor could claim a huge compressed_size and OOM
// the per-chunk `comp` allocation (which happens before read_exact could fail on
// EOF). Legit chunks are <= ~8 MiB (1M elements x 8 bytes, compressed <=
// decompressed); cap well above that.
const MAX_CHUNK_COMPRESSED_BYTES: u64 = 64 << 20; // 64 MiB

// Decompression-bomb guard: zstd_decompress allocates `decompressed_size` bytes up front,
// so a tiny compressed frame declaring a huge decompressed size could OOM the process.
// Cap the DECOMPRESSED chunk bytes at the same 64 MiB ceiling (8x the ~8 MiB legit max).
// Bounds the pub decode_chunk_into for direct callers too (Gemini PR #139).
const MAX_CHUNK_DECOMPRESSED_BYTES: usize = 64 << 20; // 64 MiB

// Self-describing per-shard codec markers the Rust decoder handles (must match
// encode.rs / codec.py). The data stream is byte-shuffled under the first and raw
// (zstd only) under the second; indices/indptr are byte-shuffle+delta in both (#142/#144).
const SHUFFLE_BYTEFILTER: &str = "byteshuffle-bytedelta";
const SHUFFLE_RAWDATA: &str = "byteshuffle-bytedelta-rawdata";

/// Map a shard's self-describing `shuffle` marker to whether its DATA stream is raw.
/// Fail loud on any other marker (the Rust decoder covers only the byte-filter codec;
/// bitshuffle archives never reach here — the Python gate routes them away).
fn data_raw_from_marker(shuffle: &str) -> Result<bool, DecodeError> {
    match shuffle {
        SHUFFLE_RAWDATA => Ok(true),
        SHUFFLE_BYTEFILTER => Ok(false),
        other => Err(DecodeError::Corrupt(format!(
            "unsupported shard shuffle marker {other:?}; the Rust decoder handles only \
             the byte-filter codec ({SHUFFLE_BYTEFILTER:?} / {SHUFFLE_RAWDATA:?})"
        ))),
    }
}

/// True iff a recorded min dtype name fits in uint16 (uint8/uint16), enabling the
/// low-plane gather for a uint32-on-disk stream. None / uint32 / float / anything
/// else -> false (full-width gather). The min is the manifest's
/// x_data_dtype_min / x_indices_dtype_min, already trusted for cast-safety.
fn min_fits_u16(min_dtype: Option<&str>) -> bool {
    matches!(min_dtype, Some("uint8") | Some("uint16"))
}

/// zstd-decompress one chunk into a fresh Vec sized to the declared length.
pub fn zstd_decompress(
    compressed: &[u8],
    decompressed_size: usize,
) -> Result<Vec<u8>, DecodeError> {
    if decompressed_size == 0 {
        return Ok(Vec::new());
    }
    zstd::bulk::decompress(compressed, decompressed_size)
        .map_err(|e| DecodeError::Corrupt(format!("zstd decompress failed: {e}")))
}

/// In-place inverse of shx::byte_delta: per plane, uint8 cumsum mod 256 along
/// the element axis. `planes` is plane-major (T*N bytes); operates per plane.
/// Mirrors codec._byte_undelta (np.cumsum axis=1 dtype=uint8).
pub fn byte_undelta_inplace(planes: &mut [u8], n: usize, t: usize) {
    if n == 0 {
        return;
    }
    assert!(
        planes.len() >= n * t,
        "planes slice too small for n * t elements"
    );
    // chunks_exact_mut(n).take(t) carves the t planes as disjoint &mut [u8] of len
    // n; the compiler elides bounds checks in the per-element cumsum inner loop.
    for plane in planes.chunks_exact_mut(n).take(t) {
        let mut acc = plane[0];
        for slot in &mut plane[1..] {
            acc = acc.wrapping_add(*slot);
            *slot = acc;
        }
    }
}

/// Reads element `e` (little-endian) and casts straight to `Dst`. `read` is the
/// plane-major (byte-shuffled) layout; `read_raw` is element-major (unshuffled/raw).
/// `Dst: Cast` is the target output element type.
pub trait OnDiskRead {
    const ITEMSIZE: usize;
    /// Read element e from plane-major (shuffled) bytes; return as the wide native value.
    fn read(planes: &[u8], n: usize, e: usize) -> WideVal;
    /// Read element e from element-major (raw) bytes: bytes[e*T .. e*T+T] little-endian (#142/#144).
    fn read_raw(bytes: &[u8], e: usize) -> WideVal;
}

/// Widest lossless carrier for a decoded on-disk value before the cast to Dst.
#[derive(Clone, Copy)]
pub enum WideVal {
    Int(i64),
    Float(f64),
}

/// Per-stream on-disk byte layout (#142/#144). Replaces a `delta: bool` so the
/// invalid "raw + delta" combination is unrepresentable (Gemini PR #144).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DecodeFilter {
    /// Element-major raw little-endian bytes (zstd only): float DATA streams (#142).
    Raw,
    /// Plane-major byte-shuffle, no delta: integer DATA streams.
    Shuffle,
    /// Plane-major byte-shuffle + per-plane byte-delta: indices/indptr streams.
    ShuffleDelta,
}

macro_rules! impl_on_disk_read {
    ($marker:ident, $prim:ty, $sz:expr, $variant:ident) => {
        pub struct $marker;
        impl OnDiskRead for $marker {
            const ITEMSIZE: usize = $sz;
            #[inline]
            fn read(planes: &[u8], n: usize, e: usize) -> WideVal {
                let mut b = [0u8; $sz];
                for p in 0..$sz {
                    b[p] = planes[p * n + e];
                }
                WideVal::$variant(<$prim>::from_le_bytes(b).into())
            }
            #[inline]
            fn read_raw(bytes: &[u8], e: usize) -> WideVal {
                let start = e * $sz;
                // The slice is exactly $sz bytes by construction (caller guards
                // bytes.len() >= n*$sz), so try_into never fails (Gemini PR #144).
                let b: [u8; $sz] = bytes[start..start + $sz].try_into().unwrap();
                WideVal::$variant(<$prim>::from_le_bytes(b).into())
            }
        }
    };
}
impl_on_disk_read!(OnDiskU16, u16, 2, Int);
impl_on_disk_read!(OnDiskU32, u32, 4, Int);
impl_on_disk_read!(OnDiskF32, f32, 4, Float);
impl_on_disk_read!(OnDiskF64, f64, 8, Float);
impl_on_disk_read!(OnDiskI64, i64, 8, Int);

/// Target element types the decoder can write.
pub trait Cast: Copy {
    fn from_wide(v: WideVal) -> Self;
}

macro_rules! impl_cast_int {
    ($t:ty) => {
        impl Cast for $t {
            #[inline]
            fn from_wide(v: WideVal) -> Self {
                match v {
                    WideVal::Int(i) => i as $t,
                    WideVal::Float(f) => f as $t,
                }
            }
        }
    };
}

macro_rules! impl_cast_float {
    ($t:ty) => {
        impl Cast for $t {
            #[inline]
            fn from_wide(v: WideVal) -> Self {
                match v {
                    WideVal::Int(i) => i as $t,
                    WideVal::Float(f) => f as $t,
                }
            }
        }
    };
}
impl_cast_int!(i32);
impl_cast_int!(i64);
impl_cast_int!(u16);
impl_cast_int!(u32);
impl_cast_float!(f32);
impl_cast_float!(f64);

/// Column-index bounds predicate for the CSR decode integrity check (M4).
/// On-disk index dtypes are unsigned (never negative); i32/i64 output dtypes
/// could in principle be negative, so the signed impls reject < 0 too.
pub trait ColIndex: Copy {
    fn col_in_bounds(self, n_vars: usize) -> bool;
}

macro_rules! impl_col_index_signed {
    ($t:ty) => {
        impl ColIndex for $t {
            #[inline]
            fn col_in_bounds(self, n_vars: usize) -> bool {
                self >= 0 && (self as u64) < n_vars as u64
            }
        }
    };
}

macro_rules! impl_col_index_unsigned {
    ($t:ty) => {
        impl ColIndex for $t {
            #[inline]
            fn col_in_bounds(self, n_vars: usize) -> bool {
                (self as u64) < n_vars as u64
            }
        }
    };
}

impl_col_index_signed!(i32);
impl_col_index_signed!(i64);
impl_col_index_unsigned!(u16);
impl_col_index_unsigned!(u32);

/// Round-trip lossless predicate for the three-tier lossy read policy (Task 16). A
/// decoded on-disk wide value "fits" the target `Self` iff casting it to `Self` and
/// back to the wide carrier is lossless -- mirroring numpy's lossless_cast
/// (src/cellstream/narrow.lossless_cast): finite + integral + in-range for an integer
/// target; an exact round-trip for a narrower float; and NaN counts as fitting for a
/// float target (numpy compares with equal_nan=True). On-disk sources are only u16/u32
/// (-> WideVal::Int, always >= 0) or f32/f64 (-> WideVal::Float), so the
/// signed-source -> unsigned sign-loss rule never applies on the read path.
pub trait CastBack: Cast {
    fn wide_fits(v: WideVal) -> bool;
}

// i64::MAX (2^63 - 1) is NOT exactly representable in f32/f64 -- it rounds UP to 2^63.
// So a naive `(f as i64) as f64 == f` round-trip FALSE-PASSES the out-of-range value
// 2^63: `f as i64` saturates to i64::MAX, which rounds back to 2^63 and compares equal
// (verified, Gemini PR #137). Every float<->i64 cast below is therefore guarded by this
// exclusive [-2^63, 2^63) range so `as i64` never saturates. The smaller int targets
// (i32/u16/u32) need no guard: their MAX is < 2^53 and exactly f64-representable, so any
// out-of-range float saturates to a value that differs from the original (the round-trip
// catches it).
const I64_F64_LOWER: f64 = -9_223_372_036_854_775_808.0; // -2^63 == i64::MIN, exact
const I64_F64_UPPER: f64 = 9_223_372_036_854_775_808.0; // 2^63, one past i64::MAX

/// True iff `f` is a finite integer in i64's range, so `f as i64` is exact (the cast
/// neither saturates nor truncates). Used both for the i64 target (float -> int) and to
/// guard the int -> float -> i64 round-trip of the f32/f64 targets.
#[inline]
fn float_fits_i64(f: f64) -> bool {
    f.is_finite() && f.fract() == 0.0 && (I64_F64_LOWER..I64_F64_UPPER).contains(&f)
}

macro_rules! impl_castback_int {
    ($t:ty) => {
        impl CastBack for $t {
            #[inline]
            fn wide_fits(v: WideVal) -> bool {
                match v {
                    // Round-trips iff in the target's range (a wrap/truncation changes
                    // the value, so the back-cast to i64 no longer equals the original).
                    WideVal::Int(i) => (i as $t) as i64 == i,
                    // finite + integral + in-range; NaN / inf / non-integral all fail
                    // (their cast-and-back never equals the original float). Safe for
                    // these < 2^53-MAX targets: an out-of-range float saturates to a
                    // value != f. i64 is excluded (its MAX is not f64-exact, see above).
                    WideVal::Float(f) => (f as $t) as f64 == f,
                }
            }
        }
    };
}
impl_castback_int!(i32);
impl_castback_int!(u16);
impl_castback_int!(u32);

// i64 gets a manual impl: the macro's `(f as i64) as f64 == f` round-trip false-passes
// 2^63 via saturation (see I64_F64_UPPER). On-disk Int sources are only u16/u32, so the
// Int arm trivially fits i64; the Float arm guards the cast with float_fits_i64.
impl CastBack for i64 {
    #[inline]
    fn wide_fits(v: WideVal) -> bool {
        match v {
            WideVal::Int(_) => true,
            WideVal::Float(f) => float_fits_i64(f),
        }
    }
}

impl CastBack for f32 {
    #[inline]
    fn wide_fits(v: WideVal) -> bool {
        match v {
            WideVal::Int(i) => {
                // Guard the back-cast against f32 -> i64 saturation (see I64_F64_UPPER):
                // only inside i64's float range is `f as i64 == i` a sound exactness check.
                let f = i as f32;
                float_fits_i64(f as f64) && f as i64 == i
            }
            // NaN round-trips as NaN (numpy equal_nan=True); else require an exact
            // f32 -> f64 round-trip (no precision loss).
            WideVal::Float(f) => f.is_nan() || (f as f32) as f64 == f,
        }
    }
}

impl CastBack for f64 {
    #[inline]
    fn wide_fits(v: WideVal) -> bool {
        // Widening to f64 from any on-disk source is a numpy-safe cast (u16/u32 < 2^53;
        // f32/f64 exact), so this target is never tier-3-checked in practice; impl it
        // correctly for completeness (the generic decode path still monomorphizes it).
        match v {
            WideVal::Int(i) => {
                // Same f64 -> i64 saturation guard as f32 (see I64_F64_UPPER).
                let f = i as f64;
                float_fits_i64(f) && f as i64 == i
            }
            WideVal::Float(_) => true,
        }
    }
}

/// Decode one chunk: zstd -> (undelta) -> unshuffle+cast into out[..n]. When `check`
/// is set (tier-3 lossy policy), each element is verified to round-trip losslessly
/// through `Dst` BEFORE the write; the first failing element aborts with a NarrowError
/// (its value carried as min/max). `decompressed_size` is the chunk's on-disk byte
/// count (= n * Src::ITEMSIZE).
pub fn decode_chunk_into<Src: OnDiskRead, Gather: OnDiskRead, Dst: Cast + CastBack>(
    compressed: &[u8],
    n: usize,
    decompressed_size: usize,
    filter: DecodeFilter,
    check: bool,
    out: &mut [Dst],
) -> Result<(), DecodeError> {
    if n == 0 {
        return Ok(());
    }
    let t = Src::ITEMSIZE;
    // n*t is computed on an untrusted element count (n derives from a header-declared
    // decompressed_size for direct/pub callers), so guard the multiply against usize
    // overflow before it could wrap past the size checks below (Gemini PR #139).
    let total_bytes = n.checked_mul(t).ok_or_else(|| {
        DecodeError::Corrupt(
            "element count * item size overflowed usize (corrupt chunk?)".to_string(),
        )
    })?;
    // Absolute decompression-bomb cap BEFORE the zstd allocation: bound the per-chunk
    // decompressed bytes regardless of the caller-supplied n (legit chunks are <= 8 MiB).
    if total_bytes > MAX_CHUNK_DECOMPRESSED_BYTES {
        return Err(DecodeError::Corrupt(format!(
            "decompressed chunk size {total_bytes} exceeds the {MAX_CHUNK_DECOMPRESSED_BYTES}-byte cap (corrupt file?)"
        )));
    }
    // Validate the DECLARED decompressed size against n*t BEFORE allocating, so a corrupt
    // chunk descriptor cannot drive a huge zstd output allocation (OOM/DoS) -- decode_chunk_into
    // is pub, so direct callers get this guard too.
    if decompressed_size != total_bytes {
        return Err(DecodeError::Corrupt(format!(
            "declared decompressed size {decompressed_size} != n*t {total_bytes} (corrupt chunk?)"
        )));
    }
    let mut planes = zstd_decompress(compressed, decompressed_size)?;
    // The ACTUAL decompressed byte count must also fill exactly: zstd may return fewer
    // bytes than the declared capacity (a short/truncated frame), which would leave planes
    // too small for the n element reads below.
    if planes.len() != total_bytes {
        return Err(DecodeError::Corrupt(format!(
            "decompressed plane size {} != n*t {total_bytes} (corrupt chunk?)",
            planes.len()
        )));
    }
    // Raw (#142): element-major LE bytes, no shuffle, no delta. Shuffle/ShuffleDelta:
    // plane-major, with the per-plane byte-delta inverse only for ShuffleDelta.
    let raw = filter == DecodeFilter::Raw;
    // Gather width <= on-disk width; for raw/float streams there is no plane
    // narrowing (read_raw is element-major), so Gather must equal Src there.
    debug_assert!(
        Gather::ITEMSIZE <= t,
        "gather width must not exceed on-disk width"
    );
    debug_assert!(
        !raw || Gather::ITEMSIZE == t,
        "raw layout requires Gather == Src (float/raw streams are never narrowed)"
    );
    // Min-aware low-plane fast-path: a uint32 stream whose recorded min fits uint16 has
    // zero high planes, so we un-delta + gather only the low Gather::ITEMSIZE planes over
    // the SAME plane-major buffer (stride n). When Gather == Src this is the full path.
    if filter == DecodeFilter::ShuffleDelta {
        byte_undelta_inplace(&mut planes, n, Gather::ITEMSIZE);
    }
    // L5: on the narrowing low-plane path the gather reads only the low
    // Gather::ITEMSIZE planes and discards the high planes -- sound only if the
    // recorded min dtype is honest (then the high planes are all zero). Verify it
    // here so a tampered/corrupt min claiming a narrower width than the data
    // actually holds is caught instead of silently truncating. The high-plane
    // bytes are already in `planes`; an all-zero original stays all-zero under both
    // Shuffle (identity) and ShuffleDelta (byte-delta of zeros is zeros, and a
    // nonzero original yields a nonzero delta byte), so they can be tested directly
    // without un-delta. Raw never narrows (debug_assert Gather == Src above).
    if Gather::ITEMSIZE < t && planes[Gather::ITEMSIZE * n..].iter().any(|&b| b != 0) {
        return Err(DecodeError::Corrupt(
            "narrowed stream has nonzero high byte-planes: on-disk values exceed the \
             recorded min dtype width (corrupt or tampered min dtype?)"
                .to_string(),
        ));
    }
    for (e, dst) in out.iter_mut().take(n).enumerate() {
        let w = if raw {
            Src::read_raw(&planes, e)
        } else {
            // Gather::read uses stride n (the Src element count) and reads only the low
            // Gather::ITEMSIZE planes: planes[0*n+e], planes[1*n+e], ...
            Gather::read(&planes, n, e)
        };
        if check && !Dst::wide_fits(w) {
            let v = match w {
                WideVal::Int(i) => i as f64,
                WideVal::Float(f) => f,
            };
            return Err(DecodeError::Lossy(NarrowError {
                min: v,
                max: v,
                reason: "value not losslessly representable in the requested dtype",
            }));
        }
        *dst = Dst::from_wide(w);
    }
    Ok(())
}

/// Decode one stream's chunks into `out` (already sized to the stream's element
/// count). `filter` selects the on-disk layout (Raw / Shuffle / ShuffleDelta); `check`
/// enables the tier-3 per-element lossless verification. Returns the first chunk's NarrowError.
fn decode_stream_into<Src: OnDiskRead, Gather: OnDiskRead, Dst: Cast + CastBack>(
    file: &mut File,
    base: u64,
    spec: &StreamSpec,
    filter: DecodeFilter,
    check: bool,
    out: &mut [Dst],
) -> Result<(), DecodeError> {
    let t = Src::ITEMSIZE;
    let mut elem_off = 0usize;
    for c in &spec.chunks {
        if c.decompressed_size == 0 {
            continue;
        }
        if c.compressed_size > MAX_CHUNK_COMPRESSED_BYTES {
            return Err(DecodeError::Corrupt(format!(
                "compressed chunk size {} exceeds the {MAX_CHUNK_COMPRESSED_BYTES}-byte cap (corrupt file?)",
                c.compressed_size
            )));
        }
        let mut comp = vec![0u8; c.compressed_size as usize];
        // Positioned read via seek + read_exact (cross-platform; each thread owns
        // its File handle so the cursor is not shared). Avoids unix-only pread.
        // checked_add: c.offset is an untrusted header value; in a release build (debug
        // overflow checks off) base + c.offset would wrap silently (Gemini PR #139).
        let seek_pos = base
            .checked_add(c.offset)
            .ok_or_else(|| DecodeError::Corrupt("chunk offset overflowed u64".to_string()))?;
        file.seek(SeekFrom::Start(seek_pos))?;
        file.read_exact(&mut comp)?;
        // usize::try_from (not `as`): c.decompressed_size is u64; on a 32-bit target `as
        // usize` would silently truncate (Gemini PR #139; mirrors the PR #130 hardening).
        let decompressed_size = usize::try_from(c.decompressed_size).map_err(|_| {
            DecodeError::Corrupt(
                "decompressed chunk size exceeds the platform pointer width".to_string(),
            )
        })?;
        let n = decompressed_size / t;
        // checked_add: n derives from an untrusted decompressed_size; on a 32-bit target
        // elem_off + n could wrap and bypass this bounds check (Gemini PR #139). Reuse the
        // validated `end` for the slice so the sum is computed once.
        let end = elem_off
            .checked_add(n)
            .filter(|&e| e <= out.len())
            .ok_or_else(|| {
                DecodeError::Corrupt(
                    "decompressed stream elements exceed the output slice (corrupt/malformed shard?)"
                        .to_string(),
                )
            })?;
        decode_chunk_into::<Src, Gather, Dst>(
            &comp,
            n,
            decompressed_size,
            filter,
            check,
            &mut out[elem_off..end],
        )?;
        elem_off = end;
    }
    if elem_off != out.len() {
        return Err(DecodeError::Corrupt(format!(
            "decoded stream element count {elem_off} does not fill the output {} (truncated/malformed shard?)",
            out.len()
        )));
    }
    Ok(())
}

/// Which rows of a shard to decode. Lifetime borrows the caller's local row-index
/// buffer for the Rows variant.
pub enum Selection<'a> {
    /// Contiguous local row range [lstart, lstop).
    Range(usize, usize),
    /// Arbitrary local row indices (shard-local, ascending). Each must be < n_rows.
    Rows(&'a [i64]),
}

/// Decode one shard's CSR streams for the given row Selection into the caller's
/// global output sub-slices. `full_rows`/`full_nnz` are this shard's TRUSTED full
/// dimensions (from the Python manifest pre-pass); the in-worker header (untrusted) is
/// validated against them BEFORE any header-sized allocation (DoS). `check_data`/
/// `check_index` enable the tier-3 lossless verification of the data / index casts.
/// Same INTERIOR-only indptr contract as decode_all_csr. Returns realized nnz on Ok.
#[allow(clippy::too_many_arguments)]
pub fn decode_one_shard_csr_sel<Dst: Cast + CastBack, IdxDst: Cast + CastBack + ColIndex>(
    path: &Path,
    base: u64,
    sel: &Selection,
    full_rows: usize,
    full_nnz: usize,
    n_vars: usize,
    nnz_offset: i64,
    check_data: bool,
    check_index: bool,
    data_min_dtype: Option<&str>,
    index_min_dtype: Option<&str>,
    out_data: &mut [Dst],   // length = realized nnz_out
    out_idx: &mut [IdxDst], // length = realized nnz_out
    out_indptr: &mut [i64], // length = rows_out (INTERIOR slots only)
) -> Result<usize, DecodeError> {
    // Zero selected rows: skip the open / header parse / full-stream decode entirely --
    // they would only produce 0 nnz. The reader's `affected` set normally excludes
    // empty shards, so this guards direct callers and any future empty-shard dispatch
    // (Gemini PR #137). out_indptr.len() == the selected row count by the driver's
    // split_at_mut, so emptiness here means 0 selected rows.
    if out_indptr.is_empty() {
        return Ok(0);
    }
    let mut file = File::open(path)?;
    // Parse this shard's header in the worker thread (one open per shard; parallel
    // JSON parse). The v2 format always writes the indptr stream as int64.
    let header = crate::shx_read::parse_header(&mut file, base)?;
    if header.indptr.dtype != "int64" {
        return Err(DecodeError::Corrupt(format!(
            "indptr stream must be int64 on disk, got {}",
            header.indptr.dtype
        )));
    }
    // DoS/integrity: validate the untrusted in-worker header dims against the trusted
    // manifest full_rows/full_nnz BEFORE any header-sized allocation below.
    if header.n_rows != full_rows {
        return Err(DecodeError::Corrupt(format!(
            "header.n_rows {} disagrees with the manifest shard row count {full_rows}",
            header.n_rows
        )));
    }
    if header.nnz != full_nnz {
        return Err(DecodeError::Corrupt(format!(
            "header.nnz {} disagrees with the manifest shard nnz {full_nnz}",
            header.nnz
        )));
    }
    // #142/#144: float archives store the data stream raw (self-describing marker).
    let data_raw = data_raw_from_marker(&header.shuffle)?;
    let nnz_out = match *sel {
        Selection::Range(lstart, lstop) if lstart == 0 && lstop == full_rows => {
            // Whole-shard fast path: decode straight into the caller's slices.
            assert_eq!(
                out_indptr.len(),
                full_rows,
                "out_indptr length must equal full_rows interior slots"
            );
            decode_data_stream::<Dst>(
                &mut file,
                base,
                &header.data,
                data_raw,
                check_data,
                data_min_dtype,
                out_data,
            )?;
            decode_index_stream::<IdxDst>(
                &mut file,
                base,
                &header.indices,
                check_index,
                index_min_dtype,
                out_idx,
            )?;
            let mut local = vec![0i64; full_rows + 1];
            decode_stream_into::<OnDiskI64, OnDiskI64, i64>(
                &mut file,
                base,
                &header.indptr,
                DecodeFilter::ShuffleDelta,
                false,
                &mut local,
            )?;
            if local[0] != 0 {
                return Err(DecodeError::Corrupt(
                    "local indptr must start at 0".to_string(),
                ));
            }
            if local[full_rows] as usize != out_data.len() {
                return Err(DecodeError::Corrupt(format!(
                    "realized NNZ {} from indptr does not match the output data buffer length {}",
                    local[full_rows],
                    out_data.len()
                )));
            }
            // Non-decreasing check, matching the partial Range / Rows branches: a decreasing
            // decoded indptr would otherwise produce an invalid CSR in the fast path
            // (Gemini PR #139). O(n_rows), negligible vs the O(nnz) stream decode.
            if !local.windows(2).all(|w| w[0] <= w[1]) {
                return Err(DecodeError::Corrupt(
                    "indptr must be non-decreasing".to_string(),
                ));
            }
            for (out, &v) in out_indptr.iter_mut().zip(&local) {
                *out = v + nnz_offset;
            }
            local[full_rows] as usize
        }
        Selection::Range(lstart, lstop) => {
            // Partial contiguous range: decode the full streams into native-width
            // temporaries (bounded by the trusted full_rows/full_nnz), then copy the
            // selected contiguous nnz slice and build the local interior indptr.
            assert!(
                lstart <= lstop && lstop <= full_rows,
                "range [{lstart}, {lstop}) out of shard bounds (full_rows {full_rows})"
            );
            assert_eq!(
                out_indptr.len(),
                lstop - lstart,
                "out_indptr length must equal the selected row count"
            );
            let mut full_indptr = vec![0i64; full_rows + 1];
            decode_stream_into::<OnDiskI64, OnDiskI64, i64>(
                &mut file,
                base,
                &header.indptr,
                DecodeFilter::ShuffleDelta,
                false,
                &mut full_indptr,
            )?;
            if full_indptr[0] != 0 {
                return Err(DecodeError::Corrupt(
                    "local indptr must start at 0".to_string(),
                ));
            }
            if full_indptr[full_rows] as usize != full_nnz {
                return Err(DecodeError::Corrupt(format!(
                    "realized NNZ {} from indptr does not match full_nnz {full_nnz}",
                    full_indptr[full_rows]
                )));
            }
            if !full_indptr.windows(2).all(|w| w[0] <= w[1]) {
                return Err(DecodeError::Corrupt(
                    "indptr must be non-decreasing".to_string(),
                ));
            }
            let nnz_start = full_indptr[lstart] as usize;
            let nnz_stop = full_indptr[lstop] as usize;
            let mut data_full = vec![Dst::from_wide(WideVal::Int(0)); full_nnz];
            let mut idx_full = vec![IdxDst::from_wide(WideVal::Int(0)); full_nnz];
            decode_data_stream::<Dst>(
                &mut file,
                base,
                &header.data,
                data_raw,
                check_data,
                data_min_dtype,
                &mut data_full,
            )?;
            decode_index_stream::<IdxDst>(
                &mut file,
                base,
                &header.indices,
                check_index,
                index_min_dtype,
                &mut idx_full,
            )?;
            // Explicit length checks: copy_from_slice panics cryptically on a length
            // mismatch; assert the domain invariant (out buffers == selected nnz) first.
            if out_data.len() != nnz_stop - nnz_start || out_idx.len() != nnz_stop - nnz_start {
                return Err(DecodeError::Corrupt(format!(
                    "selected nnz {} does not match output buffer lengths (data {}, idx {})",
                    nnz_stop - nnz_start,
                    out_data.len(),
                    out_idx.len()
                )));
            }
            out_data.copy_from_slice(&data_full[nnz_start..nnz_stop]);
            out_idx.copy_from_slice(&idx_full[nnz_start..nnz_stop]);
            for (out, &val) in out_indptr.iter_mut().zip(&full_indptr[lstart..lstop]) {
                *out = val - nnz_start as i64 + nnz_offset;
            }
            nnz_stop - nnz_start
        }
        Selection::Rows(local_rows) => {
            // Arbitrary local row gather: decode the full streams into temporaries
            // (bounded by the trusted dims), then copy each selected row's nnz slice
            // in order. Interior slot j = cumulative nnz BEFORE selected row j.
            assert_eq!(
                out_indptr.len(),
                local_rows.len(),
                "out_indptr length must equal the selected row count"
            );
            let mut full_indptr = vec![0i64; full_rows + 1];
            decode_stream_into::<OnDiskI64, OnDiskI64, i64>(
                &mut file,
                base,
                &header.indptr,
                DecodeFilter::ShuffleDelta,
                false,
                &mut full_indptr,
            )?;
            if full_indptr[0] != 0 {
                return Err(DecodeError::Corrupt(
                    "local indptr must start at 0".to_string(),
                ));
            }
            if full_indptr[full_rows] as usize != full_nnz {
                return Err(DecodeError::Corrupt(format!(
                    "realized NNZ {} from indptr does not match full_nnz {full_nnz}",
                    full_indptr[full_rows]
                )));
            }
            if !full_indptr.windows(2).all(|w| w[0] <= w[1]) {
                return Err(DecodeError::Corrupt(
                    "indptr must be non-decreasing".to_string(),
                ));
            }
            // Pre-pass: validate every selected row index and that the output buffers
            // exactly hold the gathered nnz, BEFORE the copy loop -- a too-small buffer
            // then fails with a clear message instead of a cryptic mid-loop slice panic
            // (mirrors the Range branch's length asserts).
            let mut expected_nnz = 0usize;
            for &r in local_rows {
                assert!(
                    r >= 0 && (r as usize) < full_rows,
                    "selected row index {r} out of shard bounds (full_rows {full_rows})"
                );
                let r = r as usize;
                // checked_add: duplicate selected rows could in principle sum past
                // usize::MAX (physically impossible at real scale, but fail loud).
                expected_nnz = expected_nnz
                    .checked_add((full_indptr[r + 1] - full_indptr[r]) as usize)
                    .expect("gathered nnz sum overflowed usize");
            }
            if out_data.len() != expected_nnz || out_idx.len() != expected_nnz {
                return Err(DecodeError::Corrupt(format!(
                    "gathered nnz {expected_nnz} does not match output buffer lengths (data {}, idx {})",
                    out_data.len(),
                    out_idx.len()
                )));
            }
            let mut data_full = vec![Dst::from_wide(WideVal::Int(0)); full_nnz];
            let mut idx_full = vec![IdxDst::from_wide(WideVal::Int(0)); full_nnz];
            decode_data_stream::<Dst>(
                &mut file,
                base,
                &header.data,
                data_raw,
                check_data,
                data_min_dtype,
                &mut data_full,
            )?;
            decode_index_stream::<IdxDst>(
                &mut file,
                base,
                &header.indices,
                check_index,
                index_min_dtype,
                &mut idx_full,
            )?;
            // Row indices + buffer lengths validated in the pre-pass above.
            let mut cursor = 0i64;
            for (out_ptr, &r) in out_indptr.iter_mut().zip(local_rows) {
                let r = r as usize;
                let a = full_indptr[r] as usize;
                let b = full_indptr[r + 1] as usize;
                let ln = b - a;
                *out_ptr = cursor + nnz_offset;
                let c = cursor as usize;
                out_data[c..c + ln].copy_from_slice(&data_full[a..b]);
                out_idx[c..c + ln].copy_from_slice(&idx_full[a..b]);
                cursor += ln as i64;
            }
            cursor as usize
        }
    };
    if out_idx.iter().any(|&col| !col.col_in_bounds(n_vars)) {
        return Err(DecodeError::Corrupt(format!(
            "column index out of range [0, {n_vars}) (corrupt indices?)"
        )));
    }
    Ok(nnz_out)
}

/// Dispatch the data stream's on-disk dtype to the typed decoder. `data_raw` selects the
/// element-major raw layout (#142 float data) vs plane-major byte-shuffle (integer data); the
/// data stream never carries byte-delta. `check` enables the tier-3 lossless cast verification.
/// `min_dtype` is the manifest's recorded minimum dtype for this stream (e.g. `x_data_dtype_min`);
/// when the on-disk dtype is uint32 and the min fits uint16, the low-2-plane fast-path is used.
fn decode_data_stream<Dst: Cast + CastBack>(
    file: &mut File,
    base: u64,
    spec: &StreamSpec,
    data_raw: bool,
    check: bool,
    min_dtype: Option<&str>,
    out: &mut [Dst],
) -> Result<(), DecodeError> {
    let f = if data_raw {
        DecodeFilter::Raw
    } else {
        DecodeFilter::Shuffle
    };
    match spec.dtype.as_str() {
        "uint16" => {
            decode_stream_into::<OnDiskU16, OnDiskU16, Dst>(file, base, spec, f, check, out)
        }
        // Min-aware low-plane fast-path: uint32 on disk but min fits uint16 -> gather the
        // 2 low planes only. `!data_raw` is belt-and-suspenders (integer data is never raw).
        "uint32" if !data_raw && min_fits_u16(min_dtype) => {
            decode_stream_into::<OnDiskU32, OnDiskU16, Dst>(file, base, spec, f, check, out)
        }
        "uint32" => {
            decode_stream_into::<OnDiskU32, OnDiskU32, Dst>(file, base, spec, f, check, out)
        }
        "float32" => {
            decode_stream_into::<OnDiskF32, OnDiskF32, Dst>(file, base, spec, f, check, out)
        }
        "float64" => {
            decode_stream_into::<OnDiskF64, OnDiskF64, Dst>(file, base, spec, f, check, out)
        }
        other => Err(DecodeError::Corrupt(format!(
            "unsupported on-disk data dtype {other}"
        ))),
    }
}

/// Dispatch the index stream's on-disk dtype (u16 or u32) to the typed decoder. Index streams
/// are always plane-major byte-shuffle + byte-delta. `check` enables the tier-3 lossless cast
/// verification of the on-disk -> Dst cast. `min_dtype` is the manifest's recorded minimum dtype
/// for this stream (e.g. `x_indices_dtype_min`); when the on-disk dtype is uint32 and the min
/// fits uint16, the low-2-plane fast-path is used.
fn decode_index_stream<Dst: Cast + CastBack>(
    file: &mut File,
    base: u64,
    spec: &StreamSpec,
    check: bool,
    min_dtype: Option<&str>,
    out: &mut [Dst],
) -> Result<(), DecodeError> {
    match spec.dtype.as_str() {
        "uint16" => decode_stream_into::<OnDiskU16, OnDiskU16, Dst>(
            file,
            base,
            spec,
            DecodeFilter::ShuffleDelta,
            check,
            out,
        ),
        "uint32" if min_fits_u16(min_dtype) => decode_stream_into::<OnDiskU32, OnDiskU16, Dst>(
            file,
            base,
            spec,
            DecodeFilter::ShuffleDelta,
            check,
            out,
        ),
        "uint32" => decode_stream_into::<OnDiskU32, OnDiskU32, Dst>(
            file,
            base,
            spec,
            DecodeFilter::ShuffleDelta,
            check,
            out,
        ),
        other => Err(DecodeError::Corrupt(format!(
            "unsupported on-disk index dtype {other}"
        ))),
    }
}

type ShardOutput<'a, Dst, IdxDst> = (usize, &'a mut [Dst], &'a mut [IdxDst], &'a mut [i64]);

/// Decode all shards' CSR into one output triple, threaded, with a STATIC contiguous
/// shard partition (split_at_mut, provably disjoint). `check_data`/`check_index` enable
/// the tier-3 lossless verification. On a lossy value, the first worker error is
/// captured (AtomicBool fast-stop + Mutex<Option<NarrowError>>, like encode) and
/// returned as Err; the output is then abandoned by the caller.
#[allow(clippy::too_many_arguments)]
pub fn decode_all_csr<Dst: Cast + CastBack + Send, IdxDst: Cast + CastBack + ColIndex + Send>(
    paths: &[std::path::PathBuf],
    bases: &[u64],
    selections: &[Selection],
    shard_rows: &[usize],
    shard_nnz: &[usize],
    out_lens: &[usize],
    row_lens: &[usize],
    n_vars: usize,
    n_threads: usize,
    check_data: bool,
    check_index: bool,
    data_min_dtype: Option<&str>,
    index_min_dtype: Option<&str>,
    out_data: &mut [Dst],
    out_idx: &mut [IdxDst],
    out_indptr: &mut [i64],
) -> Result<(), DecodeError> {
    let n_shards = paths.len();
    // Per-shard parallel arrays must all match paths (the FFI layer also checks
    // this; asserted here too for direct Rust callers, since the driver indexes [s]).
    assert_eq!(
        bases.len(),
        n_shards,
        "bases length must equal paths length"
    );
    assert_eq!(
        selections.len(),
        n_shards,
        "selections length must equal paths length"
    );
    assert_eq!(
        shard_rows.len(),
        n_shards,
        "shard_rows length must equal paths length"
    );
    assert_eq!(
        shard_nnz.len(),
        n_shards,
        "shard_nnz length must equal paths length"
    );
    assert_eq!(
        out_lens.len(),
        n_shards,
        "out_lens length must equal paths length"
    );
    assert_eq!(
        row_lens.len(),
        n_shards,
        "row_lens length must equal paths length"
    );
    if n_shards == 0 {
        if !out_indptr.is_empty() {
            out_indptr[0] = 0;
        }
        return Ok(());
    }
    // Upfront buffer-size check: the per-shard split_at_mut peels sum(out_lens) /
    // sum(row_lens) slots; assert the buffers hold them so a too-small buffer fails
    // with a clear message rather than a cryptic split_at_mut panic.
    let total_out_len: usize = out_lens.iter().sum();
    let total_row_len: usize = row_lens.iter().sum();
    assert!(
        out_data.len() >= total_out_len,
        "out_data length {} < total expected NNZ {total_out_len}",
        out_data.len()
    );
    assert!(
        out_idx.len() >= total_out_len,
        "out_idx length {} < total expected NNZ {total_out_len}",
        out_idx.len()
    );
    assert!(
        out_indptr.len() > total_row_len,
        "out_indptr length {} < total expected rows + 1 ({})",
        out_indptr.len(),
        total_row_len + 1
    );
    // Per-shard headers are parsed inside the worker threads (decode_one_shard_csr_sel),
    // so the main thread neither pre-parses nor opens the shard files twice.

    // Per-shard global nnz base = running prefix sum of out_lens.
    let mut nnz_bases: Vec<i64> = Vec::with_capacity(n_shards);
    let mut acc: i64 = 0;
    for &l in out_lens {
        nnz_bases.push(acc);
        acc += l as i64;
    }
    let total_nnz = acc;

    // Partition the interior [0, total_row_len) per shard and write the single
    // final boundary at out_indptr[total_row_len]. Use the ROW-COUNT SUM (not the
    // buffer length) so an over-sized output buffer cannot misplace the boundary or
    // leave an interior gap unwritten. out_indptr.len() > total_row_len is asserted
    // above, so the tail has at least the one boundary slot. ALL safe split_at_mut,
    // no raw pointers, no provenance laundering.
    let total_rows = total_row_len;
    let (indptr_interior, indptr_tail) = out_indptr.split_at_mut(total_rows);
    let mut d_rest = &mut out_data[..];
    let mut i_rest = &mut out_idx[..];
    let mut p_rest = &mut indptr_interior[..];
    let mut per_shard: Vec<ShardOutput<'_, Dst, IdxDst>> = Vec::with_capacity(n_shards);
    for s in 0..n_shards {
        let (d_s, d_tail) = d_rest.split_at_mut(out_lens[s]);
        let (i_s, i_tail) = i_rest.split_at_mut(out_lens[s]);
        let (p_s, p_tail) = p_rest.split_at_mut(row_lens[s]);
        per_shard.push((s, d_s, i_s, p_s));
        d_rest = d_tail;
        i_rest = i_tail;
        p_rest = p_tail;
    }

    let n_workers = n_threads.max(1).min(n_shards);
    let per = n_shards.div_ceil(n_workers);
    let nnz_bases = &nnz_bases;
    // First-error capture across worker threads (mirrors encode::scan_parallel's
    // fail-fast). The AtomicBool lets a worker skip its remaining shards once any
    // thread has recorded a lossy value; the Mutex holds the first NarrowError seen.
    let err_flag = std::sync::atomic::AtomicBool::new(false);
    let err_slot: std::sync::Mutex<Option<DecodeError>> = std::sync::Mutex::new(None);
    {
        let err_flag = &err_flag;
        let err_slot = &err_slot;
        std::thread::scope(|scope| {
            let mut it = per_shard.into_iter();
            for _ in 0..n_workers {
                // Move a contiguous group of (s, owned disjoint &mut slices) into this
                // thread. paths/bases/selections/shard_rows/shard_nnz/nnz_bases are
                // shared &refs (Copy); the &mut slices are exclusive + provably disjoint
                // (split_at_mut). Each worker parses its own shard header.
                let group: Vec<ShardOutput<'_, Dst, IdxDst>> = it.by_ref().take(per).collect();
                if group.is_empty() {
                    break;
                }
                scope.spawn(move || {
                    for (s, d_s, i_s, p_s) in group {
                        if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
                            return;
                        }
                        if let Err(e) = decode_one_shard_csr_sel::<Dst, IdxDst>(
                            &paths[s],
                            bases[s],
                            &selections[s],
                            shard_rows[s],
                            shard_nnz[s],
                            n_vars,
                            nnz_bases[s],
                            check_data,
                            check_index,
                            data_min_dtype,
                            index_min_dtype,
                            d_s,
                            i_s,
                            p_s,
                        ) {
                            err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            let mut slot = err_slot.lock().unwrap();
                            if slot.is_none() {
                                *slot = Some(e);
                            }
                            return;
                        }
                    }
                });
            }
        });
    }
    if let Some(e) = err_slot.into_inner().unwrap() {
        return Err(e);
    }

    // The single global final boundary slot = total nnz. (Empty read: handled by
    // the n_shards==0 early return.)
    if let Some(slot) = indptr_tail.first_mut() {
        *slot = total_nnz;
    }
    Ok(())
}

/// Decode one shard and scatter the rows in `sel` into a dense row-block `out_block`
/// (length == selected_rows * n_vars). `full_rows`/`full_nnz` are TRUSTED full dims.
/// `check_data` enables the tier-3 lossless verification of the data cast. The caller
/// MUST pre-zero `out_block` (the scatter only writes nonzero cells).
#[allow(clippy::too_many_arguments)]
pub fn decode_one_shard_dense_sel<Dst: Cast + CastBack>(
    path: &Path,
    base: u64,
    sel: &Selection,
    full_rows: usize,
    full_nnz: usize,
    n_vars: usize,
    check_data: bool,
    out_block: &mut [Dst],
) -> Result<(), DecodeError> {
    // Zero selected rows (out_block empty == sel_rows * n_vars == 0): skip the open /
    // header parse / full-stream decode -- nothing to scatter. Mirrors the CSR guard;
    // the reader's `affected` set normally excludes empty shards (Gemini PR #137).
    if out_block.is_empty() {
        return Ok(());
    }
    let mut file = File::open(path)?;
    let header = crate::shx_read::parse_header(&mut file, base)?;
    if header.indptr.dtype != "int64" {
        return Err(DecodeError::Corrupt(format!(
            "indptr stream must be int64 on disk, got {}",
            header.indptr.dtype
        )));
    }
    // DoS/integrity: validate the untrusted in-worker header dims against the trusted
    // manifest full_rows/full_nnz BEFORE any header-sized allocation below.
    if header.n_rows != full_rows {
        return Err(DecodeError::Corrupt(format!(
            "header.n_rows {} disagrees with the manifest shard row count {full_rows}",
            header.n_rows
        )));
    }
    if header.nnz != full_nnz {
        return Err(DecodeError::Corrupt(format!(
            "header.nnz {} disagrees with the manifest shard nnz {full_nnz}",
            header.nnz
        )));
    }
    // #142/#144: float archives store the data stream raw (self-describing marker).
    let data_raw = data_raw_from_marker(&header.shuffle)?;
    // The output block holds exactly the SELECTED rows. Bound the selected count and
    // assert the block size via checked_mul so a pathological n_vars cannot wrap.
    let sel_rows = match *sel {
        Selection::Range(lstart, lstop) => {
            assert!(
                lstart <= lstop && lstop <= full_rows,
                "range [{lstart}, {lstop}) out of shard bounds (full_rows {full_rows})"
            );
            lstop - lstart
        }
        Selection::Rows(local_rows) => local_rows.len(),
    };
    let expected_len = sel_rows
        .checked_mul(n_vars)
        .expect("selected_rows * n_vars overflowed usize");
    assert_eq!(
        out_block.len(),
        expected_len,
        "out_block length {} must equal selected_rows * n_vars ({})",
        out_block.len(),
        expected_len
    );
    // Decode the full shard streams into temporaries (bounded by the trusted dims).
    let mut data = vec![Dst::from_wide(WideVal::Int(0)); full_nnz];
    let mut idx = vec![0i64; full_nnz];
    decode_data_stream::<Dst>(
        &mut file,
        base,
        &header.data,
        data_raw,
        check_data,
        None,
        &mut data,
    )?;
    decode_index_stream::<i64>(&mut file, base, &header.indices, false, None, &mut idx)?;
    let mut indptr = vec![0i64; full_rows + 1];
    decode_stream_into::<OnDiskI64, OnDiskI64, i64>(
        &mut file,
        base,
        &header.indptr,
        DecodeFilter::ShuffleDelta,
        false,
        &mut indptr,
    )?;
    // Integrity (mirrors the CSR path): start at 0, end at full_nnz, non-decreasing,
    // so every a..b row slice below stays within data/idx bounds.
    if indptr[0] != 0 {
        return Err(DecodeError::Corrupt(
            "local indptr must start at 0".to_string(),
        ));
    }
    if indptr[full_rows] as usize != full_nnz {
        return Err(DecodeError::Corrupt(format!(
            "realized NNZ {} from indptr does not match full_nnz {full_nnz}",
            indptr[full_rows]
        )));
    }
    if !indptr.windows(2).all(|w| w[0] <= w[1]) {
        return Err(DecodeError::Corrupt(
            "indptr must be non-decreasing".to_string(),
        ));
    }
    // Scatter each selected source row r into output row `out_row`.
    let scatter = |out_row: usize, r: usize, out_block: &mut [Dst]| -> Result<(), DecodeError> {
        let a = indptr[r] as usize;
        let b = indptr[r + 1] as usize;
        let row_base = out_row * n_vars;
        for k in a..b {
            // A corrupt column index would otherwise silently scatter into a later
            // row of this flat row-block (only the TOTAL length is bounds-checked).
            let col = idx[k] as usize;
            if col >= n_vars {
                return Err(DecodeError::Corrupt(format!(
                    "column index {col} exceeds n_vars {n_vars} (corrupt indices?)"
                )));
            }
            out_block[row_base + col] = data[k];
        }
        Ok(())
    };
    match *sel {
        Selection::Range(lstart, lstop) => {
            for (out_row, r) in (lstart..lstop).enumerate() {
                scatter(out_row, r, out_block)?;
            }
        }
        Selection::Rows(local_rows) => {
            for (out_row, &r) in local_rows.iter().enumerate() {
                assert!(
                    r >= 0 && (r as usize) < full_rows,
                    "selected row index {r} out of shard bounds (full_rows {full_rows})"
                );
                scatter(out_row, r as usize, out_block)?;
            }
        }
    }
    Ok(())
}

/// Decode all shards into one dense (total_rows, n_vars) buffer per the per-shard
/// Selection, threaded with a STATIC contiguous row-block partition. `check_data`
/// enables the tier-3 lossless verification. First worker error captured + returned as
/// Err. The caller MUST pre-zero `dense` (the scatter only writes nonzero cells).
#[allow(clippy::too_many_arguments)]
pub fn decode_all_dense<Dst: Cast + CastBack + Send>(
    paths: &[std::path::PathBuf],
    bases: &[u64],
    selections: &[Selection],
    shard_rows: &[usize],
    shard_nnz: &[usize],
    row_lens: &[usize],
    n_vars: usize,
    n_threads: usize,
    check_data: bool,
    dense: &mut [Dst],
) -> Result<(), DecodeError> {
    let n_shards = paths.len();
    assert_eq!(
        bases.len(),
        n_shards,
        "bases length must equal paths length"
    );
    assert_eq!(
        selections.len(),
        n_shards,
        "selections length must equal paths length"
    );
    assert_eq!(
        shard_rows.len(),
        n_shards,
        "shard_rows length must equal paths length"
    );
    assert_eq!(
        shard_nnz.len(),
        n_shards,
        "shard_nnz length must equal paths length"
    );
    assert_eq!(
        row_lens.len(),
        n_shards,
        "row_lens length must equal paths length"
    );
    if n_shards == 0 {
        return Ok(());
    }
    // Upfront buffer-size check so a too-small buffer fails with a clear message
    // rather than a cryptic split_at_mut panic (mirrors decode_all_csr).
    let total_row_len: usize = row_lens.iter().sum();
    let expected_cells = total_row_len
        .checked_mul(n_vars)
        .expect("total_row_len * n_vars overflowed usize");
    assert!(
        dense.len() >= expected_cells,
        "dense buffer length {} < total expected cells {expected_cells}",
        dense.len()
    );
    // Peel disjoint contiguous row-blocks (row_lens[s] * n_vars) -- all SAFE.
    let mut rest = &mut dense[..];
    let mut blocks: Vec<(usize, &mut [Dst])> = Vec::with_capacity(n_shards);
    for (s, &row_len) in row_lens.iter().enumerate().take(n_shards) {
        let (blk, tail) = rest.split_at_mut(row_len * n_vars);
        blocks.push((s, blk));
        rest = tail;
    }

    let n_workers = n_threads.max(1).min(n_shards);
    let per = n_shards.div_ceil(n_workers);
    // First-error capture across worker threads (mirrors decode_all_csr).
    let err_flag = std::sync::atomic::AtomicBool::new(false);
    let err_slot: std::sync::Mutex<Option<DecodeError>> = std::sync::Mutex::new(None);
    {
        let err_flag = &err_flag;
        let err_slot = &err_slot;
        std::thread::scope(|scope| {
            let mut it = blocks.into_iter();
            for _ in 0..n_workers {
                let group: Vec<(usize, &mut [Dst])> = it.by_ref().take(per).collect();
                if group.is_empty() {
                    break;
                }
                // Each worker parses its own shard headers (single open per shard,
                // parallel JSON parse -- mirrors decode_all_csr); no main-thread
                // pre-parse, no double open.
                scope.spawn(move || {
                    for (s, blk) in group {
                        if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
                            return;
                        }
                        if let Err(e) = decode_one_shard_dense_sel::<Dst>(
                            &paths[s],
                            bases[s],
                            &selections[s],
                            shard_rows[s],
                            shard_nnz[s],
                            n_vars,
                            check_data,
                            blk,
                        ) {
                            err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            let mut slot = err_slot.lock().unwrap();
                            if slot.is_none() {
                                *slot = Some(e);
                            }
                            return;
                        }
                    }
                });
            }
        });
    }
    if let Some(e) = err_slot.into_inner().unwrap() {
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shx::{byte_delta, byte_shuffle};

    #[test]
    fn undelta_inverts_byte_delta() {
        // single plane, exercises mod-256 wrap (matches shx.rs delta_roundtrips)
        let orig = vec![10u8, 13, 13, 9];
        let mut d = orig.clone();
        byte_delta(&mut d, 4, 1);
        byte_undelta_inplace(&mut d, 4, 1);
        assert_eq!(d, orig);
    }

    #[test]
    fn undelta_inverts_multiplane_u16() {
        let bytes: Vec<u8> = [1u16, 2, 3, 1000, 65535]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let n = 5;
        let mut s = byte_shuffle(&bytes, 2); // plane-major
        byte_delta(&mut s, n, 2);
        let pre_delta = byte_shuffle(&bytes, 2);
        byte_undelta_inplace(&mut s, n, 2);
        assert_eq!(s, pre_delta);
    }

    #[test]
    fn zstd_roundtrips_empty_and_data() {
        assert!(zstd_decompress(&zstd::bulk::compress(&[], 1).unwrap(), 0)
            .unwrap()
            .is_empty());
        let raw = vec![7u8; 33];
        let c = zstd::bulk::compress(&raw, 1).unwrap();
        assert_eq!(zstd_decompress(&c, raw.len()).unwrap(), raw);
    }

    #[test]
    fn decode_chunk_u16_to_f32_matches_spike() {
        // data stream: u16 on disk, shuffle ONLY (no delta), cast to f32.
        let elems: Vec<u16> = vec![0, 1, 1000, 65535, 42];
        let bytes: Vec<u8> = elems.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = crate::shx::encode_chunk(&bytes, 2, false);
        let mut out = vec![0f32; elems.len()];
        decode_chunk_into::<OnDiskU16, OnDiskU16, f32>(
            &compressed,
            elems.len(),
            2 * elems.len(),
            DecodeFilter::Shuffle,
            false,
            &mut out,
        )
        .unwrap();
        let expected: Vec<f32> = elems.iter().map(|&v| v as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn decode_chunk_raw_f32_element_major() {
        // #142 read: a RAW data chunk is zstd of the LE bytes verbatim (element-major,
        // NO byte-shuffle, NO delta). decode_chunk_into with DecodeFilter::Raw must read
        // it element-major and cast — the read-side twin of shx::encode_chunk_raw.
        let elems: Vec<f32> = vec![1.5, -2.0, 3.25, 0.0, 65535.5];
        let bytes: Vec<u8> = elems.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = crate::shx::encode_chunk_raw(&bytes);
        let mut out = vec![0f32; elems.len()];
        decode_chunk_into::<OnDiskF32, OnDiskF32, f32>(
            &compressed,
            elems.len(),
            4 * elems.len(),
            DecodeFilter::Raw,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, elems);
    }

    #[test]
    fn decode_chunk_raw_f64_with_cast_to_f32() {
        // raw f64 data decoded element-major + cast to f32 (lossless for these values).
        let elems: Vec<f64> = vec![0.0, 1.5, -8.0, 1024.5];
        let bytes: Vec<u8> = elems.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = crate::shx::encode_chunk_raw(&bytes);
        let mut out = vec![0f32; elems.len()];
        decode_chunk_into::<OnDiskF64, OnDiskF64, f32>(
            &compressed,
            elems.len(),
            8 * elems.len(),
            DecodeFilter::Raw,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, vec![0.0f32, 1.5, -8.0, 1024.5]);
    }

    #[test]
    fn decode_chunk_u16_indices_with_delta_to_i32() {
        // indices stream: u16 on disk, shuffle + delta, cast to i32.
        let elems: Vec<u16> = vec![3, 5, 5, 9, 65535];
        let bytes: Vec<u8> = elems.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = crate::shx::encode_chunk(&bytes, 2, true); // delta=true
        let mut out = vec![0i32; elems.len()];
        decode_chunk_into::<OnDiskU16, OnDiskU16, i32>(
            &compressed,
            elems.len(),
            2 * elems.len(),
            DecodeFilter::ShuffleDelta,
            false,
            &mut out,
        )
        .unwrap();
        let expected: Vec<i32> = elems.iter().map(|&v| v as i32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn decode_chunk_i64_indptr_with_delta() {
        let elems: Vec<i64> = vec![0, 2, 2, 7];
        let bytes: Vec<u8> = elems.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = crate::shx::encode_chunk(&bytes, 8, true);
        let mut out = vec![0i64; elems.len()];
        decode_chunk_into::<OnDiskI64, OnDiskI64, i64>(
            &compressed,
            elems.len(),
            8 * elems.len(),
            DecodeFilter::ShuffleDelta,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, elems);
    }

    #[test]
    fn decode_one_shard_csr_roundtrips_u16_to_f32() {
        use crate::encode::encode_all;

        // 3-row CSR, 4 cols: r0=[5@0,7@2]; r1=[]; r2=[9@1,1@3]
        let data: Vec<u16> = vec![5, 7, 9, 1];
        let indices: Vec<i32> = vec![0, 2, 1, 3];
        let indptr: Vec<i64> = vec![0, 2, 2, 4];
        let rows: Vec<i64> = vec![0, 1, 2];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_shard_test");
        let _ = std::fs::create_dir_all(&dir);
        let nnz = encode_all(&data, &indices, &indptr, &lists, 4, 1, &dir).unwrap();
        assert_eq!(nnz, vec![4]);

        let path = dir.join("shard_0000.shx");

        let mut out_data = vec![0f32; 4];
        let mut out_idx = vec![0i32; 4];
        let mut out_indptr = vec![0i64; 3]; // n_rows INTERIOR slots (boundary is the driver's job)
        let nnz = decode_one_shard_csr_sel::<f32, i32>(
            &path,
            0,
            &Selection::Range(0, 3),
            3, // full_rows
            4, // full_nnz
            4, // n_vars
            0, // nnz_offset
            false,
            false,
            None,
            None,
            &mut out_data,
            &mut out_idx,
            &mut out_indptr,
        )
        .unwrap();
        assert_eq!(out_data, vec![5.0, 7.0, 9.0, 1.0]);
        assert_eq!(out_idx, vec![0, 2, 1, 3]);
        assert_eq!(out_indptr, vec![0, 2, 2]); // interior slots; global boundary (4) set by the driver
        assert_eq!(nnz, 4);
    }

    #[test]
    fn data_raw_from_marker_maps_and_rejects() {
        assert!(!data_raw_from_marker(SHUFFLE_BYTEFILTER).unwrap());
        assert!(data_raw_from_marker(SHUFFLE_RAWDATA).unwrap());
        assert!(matches!(
            data_raw_from_marker("bitshuffle"),
            Err(DecodeError::Corrupt(_))
        ));
    }

    #[test]
    fn decode_one_shard_csr_rawdata_f32_roundtrips() {
        use crate::encode::encode_all_generic;
        use crate::narrow::OnDiskDtype;
        // Float data -> rawdata marker on disk; the decoder must read the data stream
        // element-major (raw) while indices/indptr stay shuffle+delta. Exercises a
        // negative, a fractional, and an explicit stored 0.0.
        let data: Vec<f32> = vec![1.5, -2.0, 3.25, 0.0];
        let indices: Vec<i32> = vec![0, 2, 1, 3];
        let indptr: Vec<i64> = vec![0, 2, 2, 4]; // r0=[0,2]; r1=[]; r2=[1,3]
        let rows: Vec<i64> = vec![0, 1, 2];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_rawdata_csr");
        let _ = std::fs::create_dir_all(&dir);
        let (_nnz, ddt, _idt) = encode_all_generic(
            &data,
            &indices,
            &indptr,
            &lists,
            4,
            1,
            &dir,
            OnDiskDtype::F32,
            OnDiskDtype::U16,
        )
        .unwrap();
        assert_eq!(ddt, OnDiskDtype::F32);
        let path = dir.join("shard_0000.shx");
        let bytes = std::fs::read(&path).unwrap();
        let hsize = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
        let hdr = std::str::from_utf8(&bytes[10..10 + hsize]).unwrap();
        assert!(
            hdr.contains("\"shuffle\":\"byteshuffle-bytedelta-rawdata\""),
            "hdr: {hdr}"
        );

        let mut out_data = vec![0f32; 4];
        let mut out_idx = vec![0i32; 4];
        let mut out_indptr = vec![0i64; 3];
        let n = decode_one_shard_csr_sel::<f32, i32>(
            &path,
            0,
            &Selection::Range(0, 3),
            3,
            4,
            4,
            0,
            false,
            false,
            None,
            None,
            &mut out_data,
            &mut out_idx,
            &mut out_indptr,
        )
        .unwrap();
        assert_eq!(n, 4);
        assert_eq!(out_data, vec![1.5, -2.0, 3.25, 0.0]);
        assert_eq!(out_idx, vec![0, 2, 1, 3]);
        assert_eq!(out_indptr, vec![0, 2, 2]);
    }

    #[test]
    fn decode_one_shard_dense_rawdata_f32_scatters() {
        use crate::encode::encode_all_generic;
        use crate::narrow::OnDiskDtype;
        // 2 rows, 3 vars: r0=[1.5@0, -2.0@2]; r1=[3.25@1].
        let data: Vec<f32> = vec![1.5, -2.0, 3.25];
        let indices: Vec<i32> = vec![0, 2, 1];
        let indptr: Vec<i64> = vec![0, 2, 3];
        let rows: Vec<i64> = vec![0, 1];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_rawdata_dense");
        let _ = std::fs::create_dir_all(&dir);
        encode_all_generic(
            &data,
            &indices,
            &indptr,
            &lists,
            3,
            1,
            &dir,
            OnDiskDtype::F32,
            OnDiskDtype::U16,
        )
        .unwrap();
        let path = dir.join("shard_0000.shx");
        let mut out_block = vec![0f32; 2 * 3];
        decode_one_shard_dense_sel::<f32>(
            &path,
            0,
            &Selection::Range(0, 2),
            2,
            3,
            3,
            false,
            &mut out_block,
        )
        .unwrap();
        // row0 = [1.5, 0, -2.0]; row1 = [0, 3.25, 0]
        assert_eq!(out_block, vec![1.5, 0.0, -2.0, 0.0, 3.25, 0.0]);
    }

    #[test]
    fn decode_one_shard_csr_sel_partial_range() {
        use crate::encode::encode_all;
        // 3-row CSR, 4 cols: r0=[5@0,7@2]; r1=[9@1,1@3]; r2=[2@0]
        let data: Vec<u16> = vec![5, 7, 9, 1, 2];
        let indices: Vec<i32> = vec![0, 2, 1, 3, 0];
        let indptr: Vec<i64> = vec![0, 2, 4, 5];
        let rows: Vec<i64> = vec![0, 1, 2];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_sel_range_test");
        let _ = std::fs::create_dir_all(&dir);
        encode_all(&data, &indices, &indptr, &lists, 4, 1, &dir).unwrap();
        let path = dir.join("shard_0000.shx");

        // Select local rows [1, 3): rows 1,2 -> nnz 3.
        let mut out_data = vec![0f32; 3];
        let mut out_idx = vec![0i32; 3];
        let mut out_indptr = vec![0i64; 2]; // 2 selected rows, INTERIOR slots only
        let nnz = decode_one_shard_csr_sel::<f32, i32>(
            &path,
            0,
            &Selection::Range(1, 3),
            3, // full_rows
            5, // full_nnz
            4, // n_vars
            0,
            false,
            false,
            None,
            None,
            &mut out_data,
            &mut out_idx,
            &mut out_indptr,
        )
        .unwrap();
        assert_eq!(out_data, vec![9.0, 1.0, 2.0]);
        assert_eq!(out_idx, vec![1, 3, 0]);
        assert_eq!(out_indptr, vec![0, 2]); // interior; global boundary (3) is the driver's job
        assert_eq!(nnz, 3);
    }

    #[test]
    fn decode_one_shard_csr_sel_rows_gather() {
        use crate::encode::encode_all;
        // Same 3-row shard; gather rows [0, 2] (skip row 1).
        let data: Vec<u16> = vec![5, 7, 9, 1, 2];
        let indices: Vec<i32> = vec![0, 2, 1, 3, 0];
        let indptr: Vec<i64> = vec![0, 2, 4, 5];
        let rows: Vec<i64> = vec![0, 1, 2];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_sel_rows_test");
        let _ = std::fs::create_dir_all(&dir);
        encode_all(&data, &indices, &indptr, &lists, 4, 1, &dir).unwrap();
        let path = dir.join("shard_0000.shx");

        let gather: Vec<i64> = vec![0, 2];
        let mut out_data = vec![0f32; 3]; // row0 nnz 2 + row2 nnz 1
        let mut out_idx = vec![0i32; 3];
        let mut out_indptr = vec![0i64; 2];
        let nnz = decode_one_shard_csr_sel::<f32, i32>(
            &path,
            0,
            &Selection::Rows(&gather),
            3,
            5,
            4,
            0,
            false,
            false,
            None,
            None,
            &mut out_data,
            &mut out_idx,
            &mut out_indptr,
        )
        .unwrap();
        assert_eq!(out_data, vec![5.0, 7.0, 2.0]);
        assert_eq!(out_idx, vec![0, 2, 0]);
        assert_eq!(out_indptr, vec![0, 2]);
        assert_eq!(nnz, 3);
    }

    #[test]
    fn decode_one_shard_csr_sel_rejects_out_of_range_column() {
        use crate::encode::encode_all;
        let data: Vec<u16> = vec![5, 7];
        let indices: Vec<i32> = vec![0, 4];
        let indptr: Vec<i64> = vec![0, 2];
        let rows: Vec<i64> = vec![0];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_csr_bad_col_test");
        let _ = std::fs::create_dir_all(&dir);
        encode_all(&data, &indices, &indptr, &lists, 5, 1, &dir).unwrap();
        let path = dir.join("shard_0000.shx");

        let mut out_data = vec![0f32; 2];
        let mut out_idx = vec![0i32; 2];
        let mut out_indptr = vec![0i64; 1];
        let err = decode_one_shard_csr_sel::<f32, i32>(
            &path,
            0,
            &Selection::Range(0, 1),
            1,
            2,
            4,
            0,
            false,
            false,
            None,
            None,
            &mut out_data,
            &mut out_idx,
            &mut out_indptr,
        )
        .unwrap_err();
        match err {
            DecodeError::Corrupt(msg) => assert!(msg.contains("column index")),
            other => panic!("expected corrupt column error, got {other:?}"),
        }
    }

    #[test]
    fn decode_chunk_into_narrowing_rejects_nonzero_high_plane() {
        // One u32 element 70000 = 0x0001_1170: plane-major LE bytes put 0x01 in
        // plane 2 (a high plane the u16 low-plane gather discards). Shuffle layout
        // (no delta); for n==1 plane-major == the value's LE bytes. The narrowing
        // gather would silently yield 0x1170 (4464); the high-plane check rejects it.
        let planes: Vec<u8> = vec![0x70, 0x11, 0x01, 0x00];
        let compressed = zstd::bulk::compress(&planes, 1).unwrap();
        let mut out = vec![0f32; 1];
        let err = decode_chunk_into::<OnDiskU32, OnDiskU16, f32>(
            &compressed,
            1,
            4,
            DecodeFilter::Shuffle,
            false,
            &mut out,
        )
        .unwrap_err();
        match err {
            DecodeError::Corrupt(msg) => assert!(msg.contains("high byte-plane")),
            other => panic!("expected corrupt high-plane error, got {other:?}"),
        }
    }

    #[test]
    fn decode_chunk_into_narrowing_accepts_zero_high_plane() {
        // 4464 = 0x0000_1170: high planes are zero -> the gather is lossless and the
        // check must NOT fire (the honest-archive case).
        let planes: Vec<u8> = vec![0x70, 0x11, 0x00, 0x00];
        let compressed = zstd::bulk::compress(&planes, 1).unwrap();
        let mut out = vec![0f32; 1];
        decode_chunk_into::<OnDiskU32, OnDiskU16, f32>(
            &compressed,
            1,
            4,
            DecodeFilter::Shuffle,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(out[0], 4464.0);
    }

    #[test]
    fn decode_chunk_into_rejects_oversized_decompressed() {
        // n * itemsize exceeds the 64 MiB decompressed-bomb cap -> Corrupt, BEFORE any
        // zstd allocation (ultrareview L15). `out`/`compressed` are never touched.
        let n = MAX_CHUNK_DECOMPRESSED_BYTES / 4 + 1; // u32 itemsize == 4
        let mut out: Vec<f32> = Vec::new();
        let err = decode_chunk_into::<OnDiskU32, OnDiskU32, f32>(
            &[],
            n,
            0,
            DecodeFilter::Shuffle,
            false,
            &mut out,
        )
        .unwrap_err();
        match err {
            DecodeError::Corrupt(msg) => {
                assert!(msg.contains("decompressed chunk size") && msg.contains("cap"))
            }
            other => panic!("expected corrupt decompressed-cap error, got {other:?}"),
        }
    }

    #[test]
    fn decode_stream_into_rejects_oversized_compressed() {
        use crate::shx_read::ChunkDesc;
        // A chunk whose declared compressed_size exceeds the 64 MiB cap is rejected
        // BEFORE the file read (ultrareview L15), so the empty dummy file is never read.
        let dir = std::env::temp_dir().join("dec_compcap_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("dummy.bin");
        std::fs::File::create(&path).unwrap();
        let mut file = std::fs::File::open(&path).unwrap();
        let spec = StreamSpec {
            dtype: "uint32".to_string(),
            chunks: vec![ChunkDesc {
                offset: 0,
                compressed_size: MAX_CHUNK_COMPRESSED_BYTES + 1,
                decompressed_size: 4,
            }],
        };
        let mut out = vec![0f32; 1];
        let err = decode_stream_into::<OnDiskU32, OnDiskU32, f32>(
            &mut file,
            0,
            &spec,
            DecodeFilter::Shuffle,
            false,
            &mut out,
        )
        .unwrap_err();
        match err {
            DecodeError::Corrupt(msg) => {
                assert!(msg.contains("compressed chunk size") && msg.contains("cap"))
            }
            other => panic!("expected corrupt compressed-cap error, got {other:?}"),
        }
    }

    #[test]
    fn decode_one_shard_dense_scatters_rows() {
        use crate::encode::encode_all;
        // 2 rows, 4 cols: r0=[5@0,7@2]; r1=[9@1]
        let data: Vec<u16> = vec![5, 7, 9];
        let indices: Vec<i32> = vec![0, 2, 1];
        let indptr: Vec<i64> = vec![0, 2, 3];
        let rows: Vec<i64> = vec![0, 1];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_dense_test");
        let _ = std::fs::create_dir_all(&dir);
        encode_all(&data, &indices, &indptr, &lists, 4, 1, &dir).unwrap();
        let path = dir.join("shard_0000.shx");
        let n_vars = 4usize;
        let total_rows = 2usize;
        let mut dense = vec![0f32; total_rows * n_vars];
        decode_one_shard_dense_sel::<f32>(
            &path,
            0,
            &Selection::Range(0, 2),
            2, // full_rows
            3, // full_nnz
            n_vars,
            false,
            &mut dense,
        )
        .unwrap();
        assert_eq!(
            dense,
            vec![5.0, 0.0, 7.0, 0.0, /* r0 */ 0.0, 9.0, 0.0, 0.0 /* r1 */]
        );
    }

    #[test]
    fn decode_one_shard_dense_sel_partial_range() {
        use crate::encode::encode_all;
        // 3 rows, 4 cols: r0=[5@0,7@2]; r1=[9@1,1@3]; r2=[2@0]
        let data: Vec<u16> = vec![5, 7, 9, 1, 2];
        let indices: Vec<i32> = vec![0, 2, 1, 3, 0];
        let indptr: Vec<i64> = vec![0, 2, 4, 5];
        let rows: Vec<i64> = vec![0, 1, 2];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_dense_sel_range_test");
        let _ = std::fs::create_dir_all(&dir);
        encode_all(&data, &indices, &indptr, &lists, 4, 1, &dir).unwrap();
        let path = dir.join("shard_0000.shx");
        let n_vars = 4usize;
        // Select rows [1, 3): rows 1,2 -> 2 output rows.
        let mut dense = vec![0f32; 2 * n_vars];
        decode_one_shard_dense_sel::<f32>(
            &path,
            0,
            &Selection::Range(1, 3),
            3,
            5,
            n_vars,
            false,
            &mut dense,
        )
        .unwrap();
        assert_eq!(
            dense,
            vec![0.0, 9.0, 0.0, 1.0, /* r1 */ 2.0, 0.0, 0.0, 0.0 /* r2 */]
        );
    }

    #[test]
    fn decode_one_shard_dense_sel_rows_gather() {
        use crate::encode::encode_all;
        let data: Vec<u16> = vec![5, 7, 9, 1, 2];
        let indices: Vec<i32> = vec![0, 2, 1, 3, 0];
        let indptr: Vec<i64> = vec![0, 2, 4, 5];
        let rows: Vec<i64> = vec![0, 1, 2];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_dense_sel_rows_test");
        let _ = std::fs::create_dir_all(&dir);
        encode_all(&data, &indices, &indptr, &lists, 4, 1, &dir).unwrap();
        let path = dir.join("shard_0000.shx");
        let n_vars = 4usize;
        // Gather rows [0, 2] (skip row 1).
        let gather: Vec<i64> = vec![0, 2];
        let mut dense = vec![0f32; 2 * n_vars];
        decode_one_shard_dense_sel::<f32>(
            &path,
            0,
            &Selection::Rows(&gather),
            3,
            5,
            n_vars,
            false,
            &mut dense,
        )
        .unwrap();
        assert_eq!(
            dense,
            vec![5.0, 0.0, 7.0, 0.0, /* r0 */ 2.0, 0.0, 0.0, 0.0 /* r2 */]
        );
    }

    #[test]
    fn decode_all_csr_two_shards_threaded() {
        use crate::encode::encode_all;
        let data: Vec<u16> = vec![5, 7, 9, 1];
        let indices: Vec<i32> = vec![0, 1, 0, 1];
        let indptr: Vec<i64> = vec![0, 1, 2, 4, 4];
        let s0: Vec<i64> = vec![0, 1]; // shard 0: rows 0,1 -> nnz 2
        let s1: Vec<i64> = vec![2, 3]; // shard 1: rows 2,3 -> nnz 2
        let lists: Vec<&[i64]> = vec![&s0, &s1];
        let dir = std::env::temp_dir().join("dec_all_test");
        let _ = std::fs::create_dir_all(&dir);
        let nnz = encode_all(&data, &indices, &indptr, &lists, 2, 2, &dir).unwrap();
        assert_eq!(nnz, vec![2, 2]);

        let paths = vec![dir.join("shard_0000.shx"), dir.join("shard_0001.shx")];
        let bases = vec![0u64, 0u64];
        let out_lens = vec![2usize, 2usize]; // realized nnz per shard
        let row_lens = vec![2usize, 2usize]; // output rows per shard
        let selections = vec![Selection::Range(0, 2), Selection::Range(0, 2)];
        let shard_rows = vec![2usize, 2usize];
        let shard_nnz = vec![2usize, 2usize];
        let mut out_data = vec![0f32; 4];
        let mut out_idx = vec![0i32; 4];
        let mut out_indptr = vec![0i64; 5]; // total_rows+1
        decode_all_csr::<f32, i32>(
            &paths,
            &bases,
            &selections,
            &shard_rows,
            &shard_nnz,
            &out_lens,
            &row_lens,
            2,
            2,
            false,
            false,
            None,
            None,
            &mut out_data,
            &mut out_idx,
            &mut out_indptr,
        )
        .unwrap();
        assert_eq!(out_data, vec![5.0, 7.0, 9.0, 1.0]);
        // global rows: r0=[5@0] r1=[7@1] r2=[9@0,1@1] r3=[] -> nnz 1,1,2,0
        assert_eq!(out_indptr, vec![0, 1, 2, 4, 4]);
    }

    #[test]
    fn decode_all_csr_tolerates_oversized_indptr_buffer() {
        // Regression for Gemini PR #132 r7: the final boundary must land at
        // out_indptr[sum(row_lens)], NOT out_indptr[len-1], so an over-sized output
        // buffer cannot misplace the boundary or leave an interior gap.
        use crate::encode::encode_all;
        let data: Vec<u16> = vec![5, 7, 9, 1];
        let indices: Vec<i32> = vec![0, 1, 0, 1];
        let indptr: Vec<i64> = vec![0, 1, 2, 4, 4];
        let s0: Vec<i64> = vec![0, 1];
        let s1: Vec<i64> = vec![2, 3];
        let lists: Vec<&[i64]> = vec![&s0, &s1];
        let dir = std::env::temp_dir().join("dec_all_oversized_test");
        let _ = std::fs::create_dir_all(&dir);
        encode_all(&data, &indices, &indptr, &lists, 2, 2, &dir).unwrap();

        let paths = vec![dir.join("shard_0000.shx"), dir.join("shard_0001.shx")];
        let bases = vec![0u64, 0u64];
        let out_lens = vec![2usize, 2usize];
        let row_lens = vec![2usize, 2usize]; // sum = 4 -> boundary at index 4
        let selections = vec![Selection::Range(0, 2), Selection::Range(0, 2)];
        let shard_rows = vec![2usize, 2usize];
        let shard_nnz = vec![2usize, 2usize];
        let mut out_data = vec![0f32; 4];
        let mut out_idx = vec![0i32; 4];
        // Over-sized by one slot; the -1 sentinel proves the boundary lands at
        // index 4 (sum(row_lens)) and the trailing slot is left untouched.
        let mut out_indptr = vec![-1i64; 6];
        decode_all_csr::<f32, i32>(
            &paths,
            &bases,
            &selections,
            &shard_rows,
            &shard_nnz,
            &out_lens,
            &row_lens,
            2,
            2,
            false,
            false,
            None,
            None,
            &mut out_data,
            &mut out_idx,
            &mut out_indptr,
        )
        .unwrap();
        assert_eq!(&out_indptr[..5], &[0, 1, 2, 4, 4]);
        assert_eq!(out_indptr[4], 4, "final boundary must be at sum(row_lens)");
        assert_eq!(
            out_indptr[5], -1,
            "trailing over-sized slot must be untouched"
        );
    }

    #[test]
    fn wide_fits_matches_lossless_policy() {
        use WideVal::{Float, Int};
        // integer target range: u16 holds 0..=65535; i32 holds it all.
        assert!(<u16 as CastBack>::wide_fits(Int(65535)));
        assert!(!<u16 as CastBack>::wide_fits(Int(70000)));
        assert!(<i32 as CastBack>::wide_fits(Int(70000)));
        // float -> int: integral in-range fits; non-integral / negative / NaN do not.
        assert!(<i32 as CastBack>::wide_fits(Float(42.0)));
        assert!(!<i32 as CastBack>::wide_fits(Float(2.5)));
        assert!(!<u16 as CastBack>::wide_fits(Float(-1.0)));
        assert!(!<i32 as CastBack>::wide_fits(Float(f64::NAN)));
        // narrower float: exact f32 round-trip required; NaN counts as fitting.
        assert!(<f32 as CastBack>::wide_fits(Float(0.5)));
        assert!(!<f32 as CastBack>::wide_fits(Float(0.1)));
        assert!(<f32 as CastBack>::wide_fits(Float(f64::NAN)));
        // i64 target: 2^63 overflows i64 (max 2^63 - 1) and must NOT false-pass via
        // float -> int saturation (Gemini PR #137); the largest f64 below 2^63 fits.
        assert!(!<i64 as CastBack>::wide_fits(Float(
            9_223_372_036_854_775_808.0
        )));
        assert!(<i64 as CastBack>::wide_fits(Float(
            9_223_372_036_854_774_784.0
        )));
        assert!(<i64 as CastBack>::wide_fits(Float(42.0)));
        assert!(!<i64 as CastBack>::wide_fits(Float(2.5)));
    }

    #[test]
    fn decode_one_shard_csr_sel_lossy_data_check() {
        use crate::encode::encode_all_generic;
        use crate::narrow::OnDiskDtype;
        // Non-integral float64 data -> stored as float64 on disk; casting to i32 is lossy.
        let data: Vec<f64> = vec![5.5, 7.5, 9.5, 1.5];
        let indices: Vec<i32> = vec![0, 2, 1, 3];
        let indptr: Vec<i64> = vec![0, 2, 2, 4];
        let rows: Vec<i64> = vec![0, 1, 2];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_lossy_check_test");
        let _ = std::fs::create_dir_all(&dir);
        encode_all_generic(
            &data,
            &indices,
            &indptr,
            &lists,
            4,
            1,
            &dir,
            OnDiskDtype::F64,
            OnDiskDtype::U16,
        )
        .unwrap();
        let path = dir.join("shard_0000.shx");
        // check_data = true -> the float64 5.5 does not fit i32 losslessly -> Err.
        let mut out_data = vec![0i32; 4];
        let mut out_idx = vec![0i32; 4];
        let mut out_indptr = vec![0i64; 3];
        let err = decode_one_shard_csr_sel::<i32, i32>(
            &path,
            0,
            &Selection::Range(0, 3),
            3,
            4,
            4,
            0,
            true,
            false,
            None,
            None,
            &mut out_data,
            &mut out_idx,
            &mut out_indptr,
        );
        assert!(err.is_err());
        // check_data = false -> truncating cast succeeds (no per-element check).
        let ok = decode_one_shard_csr_sel::<i32, i32>(
            &path,
            0,
            &Selection::Range(0, 3),
            3,
            4,
            4,
            0,
            false,
            false,
            None,
            None,
            &mut out_data,
            &mut out_idx,
            &mut out_indptr,
        );
        assert!(ok.is_ok());
    }

    #[test]
    fn empty_selection_skips_io() {
        // Zero selected rows must early-return WITHOUT opening the shard file: a
        // nonexistent path would panic in File::open's .expect otherwise (Gemini PR #137).
        let bogus = Path::new("/nonexistent/cellstream/shard.shx");
        let mut d: Vec<f32> = vec![];
        let mut i: Vec<i32> = vec![];
        let mut p: Vec<i64> = vec![];
        let nnz = decode_one_shard_csr_sel::<f32, i32>(
            bogus,
            0,
            &Selection::Range(2, 2),
            5,
            9,
            4,
            0,
            false,
            false,
            None,
            None,
            &mut d,
            &mut i,
            &mut p,
        )
        .unwrap();
        assert_eq!(nnz, 0);
        let mut blk: Vec<f32> = vec![];
        decode_one_shard_dense_sel::<f32>(
            bogus,
            0,
            &Selection::Range(2, 2),
            5,
            9,
            4,
            false,
            &mut blk,
        )
        .unwrap();
    }

    #[test]
    fn decode_missing_file_returns_io_err() {
        // Non-empty selection on a nonexistent path: File::open fails -> Io.
        let bogus = Path::new("/nonexistent/cellstream/missing.shx");
        let mut d = vec![0f32; 1];
        let mut i = vec![0i32; 1];
        let mut p = vec![0i64; 1];
        let r = decode_one_shard_csr_sel::<f32, i32>(
            bogus,
            0,
            &Selection::Range(0, 1),
            1,
            1,
            4,
            0,
            false,
            false,
            None,
            None,
            &mut d,
            &mut i,
            &mut p,
        );
        assert!(matches!(r, Err(DecodeError::Io(_))));
    }

    #[test]
    fn decode_truncated_file_returns_io_err() {
        use crate::encode::encode_all;
        let data: Vec<u16> = vec![5, 7, 9, 1];
        let indices: Vec<i32> = vec![0, 2, 1, 3];
        let indptr: Vec<i64> = vec![0, 2, 2, 4];
        let rows: Vec<i64> = vec![0, 1, 2];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_trunc_test");
        let _ = std::fs::create_dir_all(&dir);
        encode_all(&data, &indices, &indptr, &lists, 4, 1, &dir).unwrap();
        let path = dir.join("shard_0000.shx");
        // Truncate to 4 bytes: parse_header's read_exact(10-byte prefix) hits EOF.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(4)
            .unwrap();
        let mut d = vec![0f32; 4];
        let mut i = vec![0i32; 4];
        let mut p = vec![0i64; 3];
        let r = decode_one_shard_csr_sel::<f32, i32>(
            &path,
            0,
            &Selection::Range(0, 3),
            3,
            4,
            4,
            0,
            false,
            false,
            None,
            None,
            &mut d,
            &mut i,
            &mut p,
        );
        assert!(matches!(r, Err(DecodeError::Io(_))));
    }

    #[test]
    fn decode_bad_magic_returns_corrupt_err() {
        // A full-size file with a wrong magic: parse_header returns InvalidData -> Corrupt.
        let dir = std::env::temp_dir().join("dec_badmagic_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("shard_0000.shx");
        {
            use std::io::Write;
            let mut fh = std::fs::File::create(&path).unwrap();
            fh.write_all(b"XXXX").unwrap(); // wrong magic (not SHX2)
            fh.write_all(&0u16.to_le_bytes()).unwrap();
            fh.write_all(&0u32.to_le_bytes()).unwrap(); // header_size 0
        }
        let mut d = vec![0f32; 1];
        let mut i = vec![0i32; 1];
        let mut p = vec![0i64; 1];
        let r = decode_one_shard_csr_sel::<f32, i32>(
            &path,
            0,
            &Selection::Range(0, 1),
            1,
            1,
            4,
            0,
            false,
            false,
            None,
            None,
            &mut d,
            &mut i,
            &mut p,
        );
        assert!(matches!(r, Err(DecodeError::Corrupt(_))));
    }

    #[test]
    fn decode_header_dims_mismatch_returns_corrupt_err() {
        use crate::encode::encode_all;
        // Valid 3-row shard, but the caller passes the WRONG full_rows (4) -> the
        // header.n_rows(3) != full_rows(4) integrity check fires -> Corrupt.
        let data: Vec<u16> = vec![5, 7, 9, 1];
        let indices: Vec<i32> = vec![0, 2, 1, 3];
        let indptr: Vec<i64> = vec![0, 2, 2, 4];
        let rows: Vec<i64> = vec![0, 1, 2];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_dims_test");
        let _ = std::fs::create_dir_all(&dir);
        encode_all(&data, &indices, &indptr, &lists, 4, 1, &dir).unwrap();
        let path = dir.join("shard_0000.shx");
        let mut d = vec![0f32; 4];
        let mut i = vec![0i32; 4];
        let mut p = vec![0i64; 3];
        let r = decode_one_shard_csr_sel::<f32, i32>(
            &path,
            0,
            &Selection::Range(0, 3),
            4, // wrong full_rows
            4,
            4,
            0,
            false,
            false,
            None,
            None,
            &mut d,
            &mut i,
            &mut p,
        );
        assert!(matches!(r, Err(DecodeError::Corrupt(_))));
    }

    #[test]
    fn decode_chunk_rejects_oversized_decompressed_size() {
        // Decompression-bomb guard: a chunk declaring n*t above the 64 MiB cap must error
        // BEFORE allocating, not OOM. n*t = 80 MiB (t=2 for u16). compressed/out are never
        // touched (we return at the cap check), so a tiny out slice is fine.
        let n = 40_000_000usize; // n*t = 80 MiB > 64 MiB cap
        let mut out = vec![0f32; 1];
        let r = decode_chunk_into::<OnDiskU16, OnDiskU16, f32>(
            &[],
            n,
            n * 2,
            DecodeFilter::Shuffle,
            false,
            &mut out,
        );
        assert!(matches!(r, Err(DecodeError::Corrupt(_))));
    }

    #[test]
    fn low_plane_u16_gather_over_u32_data_matches_full() {
        // DATA stream stored uint32 (Shuffle, no delta) with every value <= 65535:
        // the two high byte-planes are zero, so a 2-plane (u16) gather over the
        // uint32 buffer must reconstruct the exact value -- byte-identical to the
        // full 4-plane gather. This is the spec's verified mechanism.
        let elems: Vec<u32> = vec![0, 1, 1000, 65535, 42];
        let bytes: Vec<u8> = elems.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = crate::shx::encode_chunk(&bytes, 4, false); // itemsize 4, no delta
        let n = elems.len();
        let expected: Vec<f32> = elems.iter().map(|&v| v as f32).collect();

        // Full 4-plane gather (today's path): Src == Gather == OnDiskU32.
        let mut full = vec![0f32; n];
        decode_chunk_into::<OnDiskU32, OnDiskU32, f32>(
            &compressed,
            n,
            4 * n,
            DecodeFilter::Shuffle,
            false,
            &mut full,
        )
        .unwrap();
        assert_eq!(full, expected);

        // Low-2-plane gather (fast path): Src = OnDiskU32 (sizing), Gather = OnDiskU16 (read).
        let mut low = vec![0f32; n];
        decode_chunk_into::<OnDiskU32, OnDiskU16, f32>(
            &compressed,
            n,
            4 * n,
            DecodeFilter::Shuffle,
            false,
            &mut low,
        )
        .unwrap();
        assert_eq!(low, expected);
        assert_eq!(
            low, full,
            "low-plane gather must be byte-identical to full gather"
        );
    }

    #[test]
    fn low_plane_u16_gather_over_u32_indices_matches_full() {
        // INDEX stream stored uint32 (Shuffle + per-plane byte-delta) with values
        // <= 65535. The byte-delta is per-plane and independent, so un-deltaing only
        // the 2 low planes and reading them reconstructs the exact value.
        let elems: Vec<u32> = vec![3, 5, 5, 9, 65535];
        let bytes: Vec<u8> = elems.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = crate::shx::encode_chunk(&bytes, 4, true); // itemsize 4, delta=true
        let n = elems.len();
        let expected: Vec<i64> = elems.iter().map(|&v| v as i64).collect();

        let mut full = vec![0i64; n];
        decode_chunk_into::<OnDiskU32, OnDiskU32, i64>(
            &compressed,
            n,
            4 * n,
            DecodeFilter::ShuffleDelta,
            false,
            &mut full,
        )
        .unwrap();
        assert_eq!(full, expected);

        let mut low = vec![0i64; n];
        decode_chunk_into::<OnDiskU32, OnDiskU16, i64>(
            &compressed,
            n,
            4 * n,
            DecodeFilter::ShuffleDelta,
            false,
            &mut low,
        )
        .unwrap();
        assert_eq!(low, expected);
        assert_eq!(
            low, full,
            "low-plane index gather must be byte-identical to full gather"
        );
    }

    #[test]
    fn min_fits_u16_classifies_names() {
        assert!(min_fits_u16(Some("uint8")));
        assert!(min_fits_u16(Some("uint16")));
        assert!(!min_fits_u16(Some("uint32")));
        assert!(!min_fits_u16(Some("float32")));
        assert!(!min_fits_u16(None));
    }

    #[test]
    fn decode_one_shard_csr_u32_archive_low_plane_matches_full() {
        use crate::encode::encode_all_generic;
        use crate::narrow::OnDiskDtype;
        // uint32 on disk for BOTH data and indices, every value <= 65535.
        let data: Vec<u32> = vec![5, 7, 9, 1];
        let indices: Vec<i32> = vec![0, 2, 1, 3];
        let indptr: Vec<i64> = vec![0, 2, 2, 4]; // r0=[0,2]; r1=[]; r2=[1,3]
        let rows: Vec<i64> = vec![0, 1, 2];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("dec_u32_lowplane_csr");
        let _ = std::fs::create_dir_all(&dir);
        let (_nnz, ddt, idt) = encode_all_generic(
            &data,
            &indices,
            &indptr,
            &lists,
            4,
            1,
            &dir,
            OnDiskDtype::U32,
            OnDiskDtype::U32,
        )
        .unwrap();
        assert_eq!(ddt, OnDiskDtype::U32);
        assert_eq!(idt, OnDiskDtype::U32);
        let path = dir.join("shard_0000.shx");

        // Decode three ways; the output must be byte-identical across all of them.
        let decode = |dmin: Option<&str>, imin: Option<&str>| {
            let mut od = vec![0f32; 4];
            let mut oi = vec![0i32; 4];
            let mut op = vec![0i64; 3];
            decode_one_shard_csr_sel::<f32, i32>(
                &path,
                0,
                &Selection::Range(0, 3),
                3,
                4,
                4,
                0,
                false,
                false,
                dmin,
                imin,
                &mut od,
                &mut oi,
                &mut op,
            )
            .unwrap();
            (od, oi, op)
        };
        let full = decode(Some("uint32"), Some("uint32")); // full 4-plane gather
        let none = decode(None, None); // no min recorded -> full gather
        let fast = decode(Some("uint16"), Some("uint16")); // low-2-plane fast path
        assert_eq!(full.0, vec![5.0, 7.0, 9.0, 1.0]);
        assert_eq!(full.1, vec![0, 2, 1, 3]);
        assert_eq!(full.2, vec![0, 2, 2]);
        assert_eq!(fast, full, "fast path must equal full gather");
        assert_eq!(none, full, "no-min fallback must equal full gather");
    }
}
