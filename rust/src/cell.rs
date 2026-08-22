//! Codec-agnostic per-cell decode: row loop, frame framing, `n_vars` validation,
//! threading, and output writing. The per-frame index/value decode is delegated to the
//! owning codec module (`pfor.rs` / `cell_zstd.rs`) via the `FrameCodec` discriminant.
//!
//! Extracted from `pfor.rs`, which was named for one codec but owned the shared
//! machinery. The pfordelta path is bit-identical to that original: `FrameCodec::name()`
//! renders `"pfordelta"`, so every corrupt-frame message is reproduced verbatim.
//!
//! x86_64-gated to match `pfor.rs` (the vendored FastPFor is x86-only); on other targets
//! Python keeps its pure-Python fallback for BOTH codecs.
#![cfg(target_arch = "x86_64")]
// pyo3 0.22's create_exception! macro (below, for NonIntegerCounts) trips rustc's check-cfg
// `unexpected_cfgs` lint under -D warnings (a false positive: the macro expands to code gated
// on a `gil-refs` cfg this crate never declares). A #[allow] on the invocation does not reach
// the macro-expanded span, so suppress at module scope.
#![allow(unexpected_cfgs)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sha2::{Digest, Sha256};
use std::fs::File;
// BufWriter buffers encode_block's many small per-frame writes; the serial concat uses raw
// File since it already reads/writes in 1 MiB chunks (BufReader dropped -- Gemini #212 r1).
// Seek/SeekFrom: reorder_frames does random-access reads of the staging file (#219 Task 5).
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};

use crate::decode::{Cast, CastBack, ColIndex, DecodeError, WideVal};
use crate::narrow::NarrowError;

/// On-disk value encoding of a zstd cell frame. (pfordelta has no analogue: FastPFor
/// always emits u32 words, so its on-disk value dtype is only a narrowing hint.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ValueEnc {
    U32,
    F32,
    F64,
}

/// Which frame codec an archive uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameCodec {
    Pfor,
    Zstd(ValueEnc),
}

impl FrameCodec {
    /// Codec name used in corrupt-frame messages. `Pfor` MUST render `"pfordelta"` so the
    /// pre-existing error strings stay byte-for-byte unchanged (tests assert on them).
    pub fn name(self) -> &'static str {
        match self {
            FrameCodec::Pfor => "pfordelta",
            FrameCodec::Zstd(_) => "zstd",
        }
    }

    /// Build from the manifest's `codec` + `value_dtype_on_disk`.
    ///
    /// For `pfordelta`, `value_dtype_on_disk` is ACCEPTED AND IGNORED — FastPFor always
    /// emits u32 words regardless of the recorded on-disk width.
    ///
    /// The accepted zstd value dtypes MUST stay in lockstep with
    /// `_rust_dispatch._RUST_ZSTD_ON_DISK_VALUE`; if Python lets through a dtype this
    /// rejects, the gate returns True and then RAISES here instead of falling back to
    /// Python (the Gemini #212 r2 failure mode).
    pub fn from_py(codec: &str, value_dtype_on_disk: &str) -> Result<Self, DecodeError> {
        match codec {
            "pfordelta" => Ok(FrameCodec::Pfor),
            "zstd" => match value_dtype_on_disk {
                "uint32" => Ok(FrameCodec::Zstd(ValueEnc::U32)),
                "float32" => Ok(FrameCodec::Zstd(ValueEnc::F32)),
                "float64" => Ok(FrameCodec::Zstd(ValueEnc::F64)),
                other => Err(DecodeError::Corrupt(format!(
                    "decode_cells: unsupported zstd on-disk value dtype {other:?}"
                ))),
            },
            other => Err(DecodeError::Corrupt(format!(
                "decode_cells: unsupported codec {other:?}"
            ))),
        }
    }
}

/// Which frame codec an ENCODE writes. Mirror of `FrameCodec` (the decode discriminant).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameEnc {
    Pfor,
    Zstd(ValueEnc),
}

impl FrameEnc {
    /// `Pfor` MUST render `"pfordelta"` so the pre-existing encode error strings stay
    /// byte-for-byte unchanged (tests assert on them).
    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            FrameEnc::Pfor => "pfordelta",
            FrameEnc::Zstd(_) => "zstd",
        }
    }

    /// Build from the writer's resolved codec + on-disk value dtype.
    ///
    /// For `pfordelta`, `value_dtype_on_disk` is ACCEPTED AND IGNORED — FastPFor always emits
    /// u32 words regardless of the recorded width (same contract as `FrameCodec::from_py`).
    ///
    /// The accepted zstd value dtypes MUST stay in lockstep with
    /// `_rust_dispatch._RUST_ZSTD_ON_DISK_VALUE`; if Python lets through a dtype this rejects,
    /// the gate returns True and then RAISES here instead of falling back to Python (the
    /// Gemini #212 r2 failure mode).
    pub fn from_py(codec: &str, value_dtype_on_disk: &str) -> Result<Self, EncodeError> {
        match codec {
            "pfordelta" => Ok(FrameEnc::Pfor),
            "zstd" => match value_dtype_on_disk {
                "uint32" => Ok(FrameEnc::Zstd(ValueEnc::U32)),
                "float32" => Ok(FrameEnc::Zstd(ValueEnc::F32)),
                "float64" => Ok(FrameEnc::Zstd(ValueEnc::F64)),
                other => Err(EncodeError::Value(format!(
                    "encode_cells: unsupported zstd on-disk value dtype {other:?}"
                ))),
            },
            other => Err(EncodeError::Value(format!(
                "encode_cells: unsupported codec {other:?}"
            ))),
        }
    }
}

/// The EncodeError a rejected float narrowing produces. The message MUST contain "lossy" --
/// `_rust_dispatch.encode_cells` maps exactly that substring to LossyCastError.
#[inline]
pub(crate) fn lossy_encode_err(v: f64) -> EncodeError {
    EncodeError::Value(format!(
        "encode_cells: lossy cast of value {v} to the requested on-disk float dtype \
         (pass allow_lossy=True to force it)"
    ))
}

/// The error the integer arms of `FloatVal` raise. Wording mirrors `_resolve_cell_codec`'s so
/// both engines fail loud identically.
#[inline]
pub(crate) fn float_from_int_err() -> EncodeError {
    EncodeError::Value(
        "encode_cells: zstd float storage requires float input; integer counts store as uint32"
            .into(),
    )
}

/// Cast a source value to an on-disk FLOAT width, folding the lossless gate.
///
/// Mirrors `narrow.py::lossless_cast`, which compares with
/// `np.array_equal(cast.astype(src), src, equal_nan=True)`. So **NaN -> NaN is LOSSLESS
/// regardless of payload** -- do NOT compare bit patterns here (an f64 NaN payload does not
/// survive narrowing to f32, and a bitwise check would raise where numpy does not). +-Inf is
/// likewise lossless. A finite f64 that overflows f32 range becomes `inf`, so `back != v` and
/// it correctly raises.
///
/// INTEGER sources FAIL LOUD on the float arms. Float storage implies float input
/// (`_resolve_cell_codec` raises for integer input + `data_dtype`), and `encode_cells` rejects
/// the combination up front, so these arms are unreachable -- they exist only because
/// `encode_cells_typed<D, I>` instantiates every source dtype.
///
/// An "honest" int->float cast here is a TRAP, and both reviewers found half of it (PR #220 r1):
/// a `(narrowed as f64) != (self as f64)` check is BLIND for i64 above 2^53 -- `self as f64`
/// ALREADY rounds, so both sides land on the same value and the check passes on a lossy cast
/// (verified: 2^53+1 and 2^63-1 both slip through) -- and an unchecked `to_f64` loses int64
/// silently too. Rejecting removes the bug class instead of making a dead path precise.
pub(crate) trait FloatVal: Copy {
    fn to_f32_checked(self, allow_lossy: bool) -> Result<f32, EncodeError>;
    fn to_f64_checked(self) -> Result<f64, EncodeError>;
}

impl FloatVal for f64 {
    #[inline]
    fn to_f32_checked(self, allow_lossy: bool) -> Result<f32, EncodeError> {
        let narrowed = self as f32;
        if !allow_lossy {
            let back = narrowed as f64;
            if back != self && !(back.is_nan() && self.is_nan()) {
                return Err(lossy_encode_err(self));
            }
        }
        Ok(narrowed)
    }
    #[inline]
    fn to_f64_checked(self) -> Result<f64, EncodeError> {
        Ok(self)
    }
}

impl FloatVal for f32 {
    #[inline]
    fn to_f32_checked(self, _allow_lossy: bool) -> Result<f32, EncodeError> {
        Ok(self)
    }
    #[inline]
    fn to_f64_checked(self) -> Result<f64, EncodeError> {
        Ok(self as f64) // widening: always lossless
    }
}

macro_rules! float_val_rejects_int {
    ($t:ty) => {
        impl FloatVal for $t {
            #[inline]
            fn to_f32_checked(self, _allow_lossy: bool) -> Result<f32, EncodeError> {
                Err(float_from_int_err())
            }
            #[inline]
            fn to_f64_checked(self) -> Result<f64, EncodeError> {
                Err(float_from_int_err())
            }
        }
    };
}
float_val_rejects_int!(u16);
float_val_rejects_int!(u32);
float_val_rejects_int!(i32);
float_val_rejects_int!(i64);

/// The `DecodeError::Lossy` a rejected value cast produces. Shared by both codecs so the
/// lossy contract is identical whichever frame decoder ran.
#[inline]
pub(crate) fn lossy_err(w: WideVal) -> DecodeError {
    let v = match w {
        WideVal::Int(i) => i as f64,
        WideVal::Float(f) => f,
    };
    DecodeError::Lossy(NarrowError {
        min: v,
        max: v,
        reason: "value not losslessly representable in the requested dtype",
    })
}

/// Like `lossy_err`, but for a rejected COLUMN INDEX narrow (#218). Maps to LossyCastError the
/// same way: `decode_lossy_err` supplies the shared "X: lossy cast on read -- a value does not
/// fit ..." prefix and the offending value; this only sets the trailing `reason` to name the
/// column index, so the full message ends "...; column index does not fit the requested index
/// dtype." rather than the generic value reason.
#[inline]
fn lossy_col_err(acc: u64) -> DecodeError {
    DecodeError::Lossy(NarrowError {
        min: acc as f64,
        max: acc as f64,
        reason: "column index does not fit the requested index dtype",
    })
}

/// Validate an ABSOLUTE column index against `[0, n_vars)` on the wide (u64) value BEFORE
/// narrowing to `Idx`. A post-cast check could wrap (e.g. 70000 -> 4464 as u16) and let an
/// out-of-range index slip past. The `u64 -> i64` widening is CHECKED: for a well-formed archive
/// `acc < n_vars <= 2^32` always fits, but a corrupt manifest could claim `n_vars > 2^63` (past
/// which `acc as i64` would silently wrap), so fail loud there -- consistent with the checked_add
/// / checked_sub corrupt-archive guards elsewhere in this module.
/// Then apply the per-element lossless-narrow policy (mirroring the value path in
/// decode_rows_into): reject -- unless `allow_lossy` -- a column index that does not fit `Idx`, so
/// uint16/uint32 output indices are byte-identical to the Python `cast_data` path (#218). The
/// signed int32/int64 arms accept any realistic column index, so the check only bites the narrow
/// unsigned outputs -- materially uint16 when a column index >= 65536.
#[inline]
fn write_col<Idx: Cast + ColIndex + CastBack>(
    acc: u64,
    n_vars: usize,
    codec: &str,
    allow_lossy: bool,
    slot: &mut Idx,
) -> Result<(), DecodeError> {
    if acc >= n_vars as u64 {
        return Err(DecodeError::Corrupt(format!(
            "corrupt {codec} frame: column index {acc} out of range [0, {n_vars})"
        )));
    }
    let Ok(acc_i64) = i64::try_from(acc) else {
        return Err(DecodeError::Corrupt(format!(
            "corrupt {codec} frame: column index {acc} exceeds i64::MAX"
        )));
    };
    let w = WideVal::Int(acc_i64);
    if !allow_lossy && !Idx::wide_fits(w) {
        return Err(lossy_col_err(acc));
    }
    *slot = Idx::from_wide(w);
    Ok(())
}

/// Decode selected rows into out_data/out_idx (both pre-sized). Single-thread body.
/// Reuses decode.rs traits UNMODIFIED (Cast::from_wide / CastBack::wide_fits).
#[allow(clippy::too_many_arguments)]
fn decode_rows_into<Dst: Cast + CastBack, Idx: Cast + ColIndex + CastBack>(
    codec: FrameCodec,
    payload: &[u8],
    offsets: &[i64],
    indptr: &[i64],
    row_ids: &[i64],
    // ABSOLUTE prefix sum of selected nnz for THIS block (len row_ids.len()+1). Element [0]
    // is the block's base, so the write position rebases against it -- which lets the threaded
    // caller pass a plain sub-slice of the global prefix sum with NO per-block allocation.
    // (Single-threaded caller passes the whole vector, whose [0] is 0, so this is a no-op there.)
    row_out_off: &[i64],
    out_data: &mut [Dst],
    out_idx: &mut [Idx],
    allow_lossy: bool,
    n_vars: usize,
    rows: std::ops::Range<usize>, // which selected rows this call handles
) -> Result<(), DecodeError> {
    let cname = codec.name();
    let mut scratch: Vec<u32> = Vec::new();
    for k in rows {
        let r = row_ids[k] as usize;
        // guard against a corrupt/out-of-range row id causing an OOB panic across the FFI boundary
        let Some(r_next) = r.checked_add(1) else {
            return Err(DecodeError::Corrupt(format!(
                "corrupt {cname} frame: row index {r} out of bounds"
            )));
        };
        if r_next >= indptr.len() || r_next >= offsets.len() {
            return Err(DecodeError::Corrupt(format!(
                "corrupt {cname} frame: row index {r} out of bounds"
            )));
        }
        // checked_sub: indptr comes from the archive, so an adversarial/corrupt pair could
        // overflow i64 -- which panics in debug and WRAPS in release, potentially yielding a
        // bogus positive nnz that sails past the < 0 check below (Gemini #217).
        let Some(nnz_i64) = indptr[r_next].checked_sub(indptr[r]) else {
            return Err(DecodeError::Corrupt(format!(
                "corrupt {cname} frame: indptr is not non-decreasing"
            )));
        };
        if nnz_i64 < 0 {
            return Err(DecodeError::Corrupt(format!(
                "corrupt {cname} frame: indptr is not non-decreasing"
            )));
        }
        let nnz = nnz_i64 as usize;
        if nnz == 0 {
            continue;
        } // empty cell: contributes no elements; skip frame parse/decode
        let wpos = (row_out_off[k] - row_out_off[0]) as usize;
        // defensively bound-check the frame offsets before slicing (corrupt-archive guard)
        let (start, end) = (offsets[r] as usize, offsets[r_next] as usize);
        if start > end || end > payload.len() {
            return Err(DecodeError::Corrupt(format!(
                "corrupt {cname} frame: offsets out of bounds"
            )));
        }
        let frame = &payload[start..end];
        // parse [u32 idx_len][idx_blob][val_blob] -- identical framing for BOTH codecs,
        // mirroring codec.py's corrupt guards.
        if frame.len() < 4 {
            return Err(DecodeError::Corrupt(format!(
                "corrupt {cname} frame: shorter than 4-byte length prefix"
            )));
        }
        let idx_len = u32::from_le_bytes(frame[0..4].try_into().unwrap()) as usize;
        // frame.len() >= 4 here, so `frame.len() - 4` cannot underflow and this form cannot overflow.
        if idx_len > frame.len() - 4 {
            return Err(DecodeError::Corrupt(format!(
                "corrupt {cname} frame: idx blob length exceeds frame"
            )));
        }
        let (idx_blob, val_blob) = (&frame[4..4 + idx_len], &frame[4 + idx_len..]);

        // --- indices -> scratch[..nnz] as ABSOLUTE u32 column indices, then validate+narrow ---
        // grow-only: the pfor arm (below) may size `scratch` well past nnz for its provable
        // decode bound (#269), and an unconditional resize(nnz) would discard that every row.
        if scratch.len() < nnz {
            scratch.resize(nnz, 0);
        }
        match codec {
            FrameCodec::Pfor => {
                // pfordelta stores VALUE-WISE GAPS: cumsum in u64 (matches
                // np.cumsum(dtype=uint64).astype(idt)), validating the TRUE cumulative
                // index before it is narrowed.
                crate::pfor::decode_blob_u32(idx_blob, &mut scratch, nnz)?;
                let mut acc: u64 = 0;
                for j in 0..nnz {
                    acc += scratch[j] as u64;
                    write_col(acc, n_vars, cname, allow_lossy, &mut out_idx[wpos + j])?;
                }
            }
            FrameCodec::Zstd(_) => {
                // zstd byte-plane-deltas the SHUFFLED planes, so the inverse reconstructs
                // the index VALUES directly -- there is NO value-wise cumsum here.
                crate::cell_zstd::decode_indices(idx_blob, nnz, &mut scratch[..nnz])?;
                for j in 0..nnz {
                    write_col(
                        scratch[j] as u64,
                        n_vars,
                        cname,
                        allow_lossy,
                        &mut out_idx[wpos + j],
                    )?;
                }
            }
        }

        // --- values ---
        match codec {
            FrameCodec::Pfor => {
                // Cast to Dst via the existing traits, honoring the lossy policy exactly as
                // decode.rs does so the "lossy" error contract is preserved.
                crate::pfor::decode_blob_u32(val_blob, &mut scratch, nnz)?;
                for j in 0..nnz {
                    let w = WideVal::Int(scratch[j] as i64);
                    if !allow_lossy && !Dst::wide_fits(w) {
                        return Err(lossy_err(w));
                    }
                    out_data[wpos + j] = Dst::from_wide(w);
                }
            }
            FrameCodec::Zstd(enc) => {
                // Values land DIRECTLY in out_data with the cast fused -- one pass, no scratch.
                crate::cell_zstd::decode_values::<Dst>(
                    enc,
                    val_blob,
                    nnz,
                    allow_lossy,
                    &mut out_data[wpos..wpos + nnz],
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_cells_typed<Dst: Cast + CastBack + Send, Idx: Cast + ColIndex + CastBack + Send>(
    codec: FrameCodec,
    payload: &[u8],
    offsets: &[i64],
    indptr: &[i64],
    row_ids: &[i64],
    out_data: &mut [Dst],
    out_idx: &mut [Idx],
    out_indptr: &mut [i64],
    allow_lossy: bool,
    n_vars: usize,
    n_threads: usize,
) -> Result<(), DecodeError> {
    let cname = codec.name();
    let n = row_ids.len();
    if out_indptr.len() < n + 1 {
        return Err(DecodeError::Corrupt(format!(
            "decode_cells: out_indptr length {} < selected rows + 1 ({})",
            out_indptr.len(),
            n + 1
        )));
    }

    let mut row_out_off = Vec::with_capacity(n + 1);
    row_out_off.push(0i64);
    let mut total: i64 = 0;
    for &row_id in row_ids {
        if row_id < 0 {
            return Err(DecodeError::Corrupt(format!(
                "corrupt {cname} frame: row index {row_id} out of bounds"
            )));
        }
        let r = row_id as usize;
        let Some(r_next) = r.checked_add(1) else {
            return Err(DecodeError::Corrupt(format!(
                "corrupt {cname} frame: row index {r} out of bounds"
            )));
        };
        if r_next >= indptr.len() || r_next >= offsets.len() {
            return Err(DecodeError::Corrupt(format!(
                "corrupt {cname} frame: row index {r} out of bounds"
            )));
        }
        // checked_sub: see decode_rows_into -- a corrupt indptr pair could wrap in release.
        let Some(nnz) = indptr[r_next].checked_sub(indptr[r]) else {
            return Err(DecodeError::Corrupt(format!(
                "corrupt {cname} frame: indptr is not non-decreasing"
            )));
        };
        if nnz < 0 {
            return Err(DecodeError::Corrupt(format!(
                "corrupt {cname} frame: indptr is not non-decreasing"
            )));
        }
        total = total.checked_add(nnz).ok_or_else(|| {
            DecodeError::Corrupt("decode_cells: selected nnz prefix sum overflowed i64".into())
        })?;
        row_out_off.push(total);
    }
    out_indptr[..n + 1].copy_from_slice(&row_out_off);

    let total_usize = total as usize;
    if out_data.len() < total_usize || out_idx.len() < total_usize {
        return Err(DecodeError::Corrupt(format!(
            "decode_cells: output buffers smaller than selected nnz {total_usize} (data {}, idx {})",
            out_data.len(),
            out_idx.len()
        )));
    }

    if n == 0 {
        return Ok(());
    }
    if n_threads <= 1 {
        return decode_rows_into(
            codec,
            payload,
            offsets,
            indptr,
            row_ids,
            &row_out_off,
            out_data,
            out_idx,
            allow_lossy,
            n_vars,
            0..n,
        );
    }

    let n_workers = n_threads.min(n);
    let per = n.div_ceil(n_workers);
    let err_flag = std::sync::atomic::AtomicBool::new(false);
    let err_slot: std::sync::Mutex<Option<DecodeError>> = std::sync::Mutex::new(None);

    {
        let err_flag = &err_flag;
        let err_slot = &err_slot;
        let mut row_start = 0usize;
        let mut data_rest = &mut out_data[..total_usize];
        let mut idx_rest = &mut out_idx[..total_usize];
        std::thread::scope(|scope| {
            while row_start < n {
                let row_end = (row_start + per).min(n);
                let nnz_start = row_out_off[row_start] as usize;
                let nnz_end = row_out_off[row_end] as usize;
                let block_nnz = nnz_end - nnz_start;
                let (data_block, data_tail) = data_rest.split_at_mut(block_nnz);
                let (idx_block, idx_tail) = idx_rest.split_at_mut(block_nnz);
                data_rest = data_tail;
                idx_rest = idx_tail;

                let row_block = &row_ids[row_start..row_end];
                // Zero-allocation: hand the worker a sub-slice of the global prefix sum and let
                // decode_rows_into rebase against its [0], instead of collecting a rebased Vec
                // per block (Gemini #217 r2).
                let block_off = &row_out_off[row_start..=row_end];
                scope.spawn(move || {
                    if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    if let Err(e) = decode_rows_into(
                        codec,
                        payload,
                        offsets,
                        indptr,
                        row_block,
                        block_off,
                        data_block,
                        idx_block,
                        allow_lossy,
                        n_vars,
                        0..row_block.len(),
                    ) {
                        err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        let mut slot = err_slot.lock().unwrap();
                        if slot.is_none() {
                            *slot = Some(e);
                        }
                    }
                });
                row_start = row_end;
            }
        });
    }

    if let Some(e) = err_slot.into_inner().unwrap() {
        return Err(e);
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn decode_cells(
    py: Python<'_>,
    payload: PyReadonlyArray1<u8>,
    offsets: PyReadonlyArray1<i64>,
    indptr: PyReadonlyArray1<i64>,
    row_ids: PyReadonlyArray1<i64>,
    out_data: &Bound<'_, PyAny>,
    out_indices: &Bound<'_, PyAny>,
    out_indptr: &Bound<'_, PyAny>,
    codec: &str,
    value_dtype_on_disk: &str,
    n_vars: usize,
    allow_lossy: bool,
    n_threads: usize,
) -> PyResult<()> {
    let codec = FrameCodec::from_py(codec, value_dtype_on_disk).map_err(crate::decode_err_to_py)?;
    let payload = payload
        .as_slice()
        .map_err(|_| PyValueError::new_err("decode_cells: payload must be contiguous"))?;
    let offsets = offsets
        .as_slice()
        .map_err(|_| PyValueError::new_err("decode_cells: offsets must be contiguous"))?;
    let indptr = indptr
        .as_slice()
        .map_err(|_| PyValueError::new_err("decode_cells: indptr must be contiguous"))?;
    let row_ids = row_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("decode_cells: row_ids must be contiguous"))?;

    let data_name = out_data.getattr("dtype")?.str()?.to_string();
    let idx_name = out_indices.getattr("dtype")?.str()?.to_string();

    macro_rules! run {
        ($d:ty, $i:ty) => {{
            let mut d_rw = out_data.extract::<PyReadwriteArray1<$d>>()?;
            let mut i_rw = out_indices.extract::<PyReadwriteArray1<$i>>()?;
            let mut p_rw = out_indptr.extract::<PyReadwriteArray1<i64>>()?;
            let d_s = d_rw.as_slice_mut().map_err(|_| {
                PyValueError::new_err("decode_cells: output data array must be contiguous")
            })?;
            let i_s = i_rw.as_slice_mut().map_err(|_| {
                PyValueError::new_err("decode_cells: output indices array must be contiguous")
            })?;
            let p_s = p_rw.as_slice_mut().map_err(|_| {
                PyValueError::new_err("decode_cells: output indptr array must be contiguous")
            })?;
            py.allow_threads(|| {
                decode_cells_typed::<$d, $i>(
                    codec,
                    payload,
                    offsets,
                    indptr,
                    row_ids,
                    d_s,
                    i_s,
                    p_s,
                    allow_lossy,
                    n_vars,
                    n_threads,
                )
            })
            .map_err(crate::decode_err_to_py)?;
            Ok(())
        }};
    }

    match (data_name.as_str(), idx_name.as_str()) {
        ("uint16", "int32") => run!(u16, i32),
        ("uint16", "int64") => run!(u16, i64),
        ("uint32", "int32") => run!(u32, i32),
        ("uint32", "int64") => run!(u32, i64),
        ("float32", "int32") => run!(f32, i32),
        ("float32", "int64") => run!(f32, i64),
        ("float64", "int32") => run!(f64, i32),
        ("float64", "int64") => run!(f64, i64),
        // #218: unsigned output-index arms (cast_back=False / explicit uint index_dtype). Kept in
        // lockstep with _rust_dispatch._RUST_OUT_INDEX; write_col enforces the per-element check.
        ("uint16", "uint16") => run!(u16, u16),
        ("uint16", "uint32") => run!(u16, u32),
        ("uint32", "uint16") => run!(u32, u16),
        ("uint32", "uint32") => run!(u32, u32),
        ("float32", "uint16") => run!(f32, u16),
        ("float32", "uint32") => run!(f32, u32),
        ("float64", "uint16") => run!(f64, u16),
        ("float64", "uint32") => run!(f64, u32),
        (d, i) => Err(PyValueError::new_err(format!(
            "decode_cells: unsupported output dtypes data={d}, indices={i}"
        ))),
    }
}

// ===========================================================================
// ENCODE -- codec-agnostic scaffolding, moved verbatim from pfor.rs (PR3 Task 1).
// The per-frame body is delegated to the owning codec module via crate::pfor::encode_frame.
// ===========================================================================

/// Encode failure at the cell-encode boundary. `Value` -> PyValueError (matches codec.py /
/// writer.py messages); `Io` -> the OSError family via pyo3's io::Error conversion.
pub enum EncodeError {
    Value(String),
    /// Integer (uint32) storage of these values is IMPOSSIBLE, so write_cell_archive should
    /// retry with zstd FLOAT storage. DISTINCT from Value precisely because the retry keys on
    /// this variant and MUST NOT retry on any other.
    ///
    /// Two conditions produce it. "non-integer / negative / non-finite" always does, for int
    /// and float sources alike. "value > u32::MAX" does ONLY for a FLOAT source: a float that
    /// overflows uint32 (e.g. 1e300) is still a finite non-negative integer that simply doesn't
    /// fit uint32, and the float fallback stores it EXACTLY, so it must trigger the retry rather
    /// than hard-fail. For an INT source, overflow stays `Value` -- casting an i64 > 2^53 to
    /// float is LOSSY, so falling back would silently corrupt data, a genuine error.
    /// (#216 review, nick-youngblut; float-overflow split found by #217's lossy-policy test.)
    NonIntegerCounts(String),
    Io(std::io::Error),
}
impl From<std::io::Error> for EncodeError {
    fn from(e: std::io::Error) -> Self {
        EncodeError::Io(e)
    }
}

pyo3::create_exception!(
    _rust,
    NonIntegerCounts,
    PyValueError,
    "Integer (uint32) storage is impossible for these values -- retry with float storage."
);

pub(crate) fn encode_err_to_py(e: EncodeError) -> PyErr {
    match e {
        EncodeError::Value(m) => PyValueError::new_err(m),
        EncodeError::NonIntegerCounts(m) => NonIntegerCounts::new_err(m),
        EncodeError::Io(io) => PyErr::from(io),
    }
}

/// Cast a source count value to a u32 for pfordelta, folding the writer.py validation
/// (finite, non-negative, integer, <= u32::MAX). Mirrors `.astype(uint32)` for accepted
/// values (truncation == identity on integers). Returns the value or an EncodeError whose
/// message matches writer.py so both engines fail loud identically.
pub(crate) trait CountU32: Copy {
    fn count_u32(self) -> Result<u32, EncodeError>;
}
macro_rules! count_u32_uint {
    ($t:ty) => {
        impl CountU32 for $t {
            #[inline]
            fn count_u32(self) -> Result<u32, EncodeError> {
                Ok(self as u32) // u16/u32: always non-negative and in range
            }
        }
    };
}
count_u32_uint!(u16);
count_u32_uint!(u32);
macro_rules! count_u32_int {
    ($t:ty) => {
        impl CountU32 for $t {
            #[inline]
            fn count_u32(self) -> Result<u32, EncodeError> {
                if self < 0 {
                    return Err(EncodeError::NonIntegerCounts(
                        "layout='cell' requires non-negative integer counts; got non-integer, \
                         negative, or non-finite values."
                            .into(),
                    ));
                }
                if (self as u64) > u32::MAX as u64 {
                    return Err(EncodeError::Value(format!(
                        "layout='cell' stores counts as uint32; value {self} exceeds the uint32 \
                         maximum {}.",
                        u32::MAX
                    )));
                }
                Ok(self as u32)
            }
        }
    };
}
count_u32_int!(i32);
count_u32_int!(i64);
macro_rules! count_u32_float {
    ($t:ty) => {
        impl CountU32 for $t {
            #[inline]
            fn count_u32(self) -> Result<u32, EncodeError> {
                // reject non-finite / negative / non-integer with the writer.py message;
                // trunc() == the integer for accepted values, matching .astype(uint32).
                if !self.is_finite() || self < 0 as $t || self != self.trunc() {
                    return Err(EncodeError::NonIntegerCounts(
                        "layout='cell' requires non-negative integer counts; got non-integer, \
                         negative, or non-finite values."
                            .into(),
                    ));
                }
                // Compare in f64: `u32::MAX as f32` rounds UP to 4294967296.0, so the check
                // must widen to f64 (exact for both f32 and f64) to reject exactly 2^32, matching
                // Python's `int(v) > 0xFFFFFFFF` (Gemini PR #212 r1). f32->f64 widening is lossless.
                //
                // NonIntegerCounts, NOT Value: for a FLOAT source, a value too big for uint32 is
                // still integer storage being impossible, and the float fallback stores it exactly
                // -- so this must trigger the retry, not hard-fail (the int macro above keeps Value:
                // an i64 > 2^53 cast to float is lossy, a genuine error). #217 lossy-policy test.
                if self as f64 > u32::MAX as f64 {
                    return Err(EncodeError::NonIntegerCounts(format!(
                        "layout='cell' stores integer counts as uint32; value {self} exceeds the \
                         uint32 maximum {} -- store as float instead.",
                        u32::MAX
                    )));
                }
                Ok(self as u32)
            }
        }
    };
}
count_u32_float!(f32);
count_u32_float!(f64);

/// The int32/int64 CSR index dtypes -> u32 column index. Mirrors `.astype(uint32)` exactly,
/// including the (pathological) wrap on a negative index, so byte-parity holds even on
/// malformed input. Only i32/i64 are impl'd — the encode_cells match never routes uint indices
/// (scipy CSR indices are always signed; the Python gate restricts to int32/int64 to match).
pub(crate) trait IdxU32: Copy {
    fn idx_u32(self) -> u32;
}
macro_rules! idx_u32 {
    ($t:ty) => {
        impl IdxU32 for $t {
            #[inline]
            fn idx_u32(self) -> u32 {
                self as u32
            }
        }
    };
}
idx_u32!(i32);
idx_u32!(i64);

/// Per-worker REUSED scratch — eliminates all per-cell heap allocation in the hot loop
/// (mirrors codec.py's reused `self._out` and decode_rows_into's reused `scratch`).
///
/// `vals_u32` is shared by pfordelta and zstd's U32 arm; the float arms, the absolute-index
/// buffer and the LE staging buffer are zstd-only. (`encode_chunk`/`encode_chunk_raw` allocate
/// their own return Vec — inherent to their signature, and the same pattern
/// `encode.rs::flush_chunk` already uses in production for the shard writer. Two mallocs are
/// noise next to a zstd call over thousands of values.)
#[derive(Default)]
pub(crate) struct CellScratch {
    // pfordelta
    pub(crate) gaps: Vec<u32>,
    pub(crate) idx_words: Vec<u32>,
    pub(crate) val_words: Vec<u32>,
    // pfordelta values (FastPFor consumes a &[u32], so this one must materialize)
    pub(crate) vals_u32: Vec<u32>,
    // zstd: LE element bytes, fed straight to encode_chunk / encode_chunk_raw. zstd needs NO
    // intermediate typed Vec -- indices and values are validated and emitted as bytes in a
    // single pass (Gemini, PR #220), so idx_abs / vals_f32 / vals_f64 are gone.
    pub(crate) le_buf: Vec<u8>,
    // shared output
    pub(crate) frame: Vec<u8>,
}

/// Assemble one cell's frame into `scratch.frame`; returns the frame length. Delegates the
/// per-frame body to the owning codec module via the `FrameEnc` discriminant.
#[allow(clippy::too_many_arguments)]
fn encode_one_frame<D: CountU32 + FloatVal, I: IdxU32>(
    enc: FrameEnc,
    idx_src: &[I],
    val_src: &[D],
    scratch: &mut CellScratch,
    max_v: &mut u64,
    check_duplicates: bool,
    allow_lossy: bool,
) -> Result<usize, EncodeError> {
    match enc {
        FrameEnc::Pfor => {
            crate::pfor::encode_frame(idx_src, val_src, scratch, max_v, check_duplicates)
        }
        FrameEnc::Zstd(v) => crate::cell_zstd::encode_frame(
            v,
            idx_src,
            val_src,
            scratch,
            max_v,
            check_duplicates,
            allow_lossy,
        ),
    }
}

/// Encode a contiguous block of output rows [row_lo, row_hi) into `staging_path`, in row order.
/// Returns (per-row frame byte lengths, local max value). `perm=Some` gathers source rows via the
/// storage permutation (`encode_cells`); `perm=None` walks `row_lo..row_hi` directly -- the block
/// is already in input/identity order, so no gather is needed (`encode_cells_block`, #219 Task 5).
#[allow(clippy::too_many_arguments)]
fn encode_block<D: CountU32 + FloatVal, I: IdxU32>(
    enc: FrameEnc,
    data: &[D],
    indices: &[I],
    indptr: &[i64],
    perm: Option<&[i64]>,
    row_lo: usize,
    row_hi: usize,
    staging_path: &std::path::Path,
    check_duplicates: bool,
    allow_lossy: bool,
) -> Result<(Vec<i64>, u64), EncodeError> {
    let mut w = BufWriter::new(File::create(staging_path)?);
    let mut lens = Vec::with_capacity(row_hi - row_lo);
    let mut scratch = CellScratch::default();
    let mut max_v: u64 = 0;
    // perm bounds validated in the typed entry before spawning; index directly.
    match perm {
        Some(perm) => {
            for &p in &perm[row_lo..row_hi] {
                let i = p as usize;
                let (s, e) = (indptr[i] as usize, indptr[i + 1] as usize);
                let len = encode_one_frame(
                    enc,
                    &indices[s..e],
                    &data[s..e],
                    &mut scratch,
                    &mut max_v,
                    check_duplicates,
                    allow_lossy,
                )?;
                w.write_all(&scratch.frame)?;
                lens.push(len as i64);
            }
        }
        None => {
            for i in row_lo..row_hi {
                let (s, e) = (indptr[i] as usize, indptr[i + 1] as usize);
                let len = encode_one_frame(
                    enc,
                    &indices[s..e],
                    &data[s..e],
                    &mut scratch,
                    &mut max_v,
                    check_duplicates,
                    allow_lossy,
                )?;
                w.write_all(&scratch.frame)?;
                lens.push(len as i64);
            }
        }
    }
    w.flush()?;
    Ok((lens, max_v))
}

/// (x_offsets, x_indptr, total_nnz, max_value, payload_sha256_hex_opt, frame_lens) — the
/// encode_cells_typed result. The sha is `None` when `compute_checksum` is false. frame_lens is
/// the flat per-row encoded-frame byte length, in output-row order (#219 Task 5) --
/// `encode_cells_block`'s primary output; `encode_cells` computes it too but ignores it (its
/// `x_offsets` already carries the same information, cumulatively, for that caller).
type EncodeCellsOut = (Vec<i64>, Vec<i64>, u64, u64, Option<String>, Vec<i64>);
/// The Python-facing encode_cells return: numpy int64 offsets/indptr + total_nnz/max/opt-sha (str|None).
type EncodeCellsPyOut = (
    Py<PyArray1<i64>>,
    Py<PyArray1<i64>>,
    u64,
    u64,
    Option<String>,
);

#[allow(clippy::too_many_arguments)]
fn encode_cells_typed<D: CountU32 + FloatVal + Sync, I: IdxU32 + Sync>(
    enc: FrameEnc,
    data: &[D],
    indices: &[I],
    indptr: &[i64],
    perm: Option<&[i64]>,
    payload_out_path: &str,
    n_threads: usize,
    append: bool,
    compute_checksum: bool,
    check_duplicates: bool,
    allow_lossy: bool,
) -> Result<EncodeCellsOut, EncodeError> {
    let n_src_rows = indptr.len().saturating_sub(1);
    let n_obs = perm.map_or(n_src_rows, |p| p.len());

    // out x_indptr (cumulative nnz in storage order) + perm bounds validation, single pass
    // O(n_obs). `perm=None` (encode_cells_block, #219 Task 5) means identity over all
    // n_src_rows rows -- trivially in range, so the perm-bounds check is skipped.
    let mut out_indptr = Vec::with_capacity(n_obs + 1);
    out_indptr.push(0i64);
    let mut total: i64 = 0;
    match perm {
        Some(perm) => {
            for &p in perm {
                if p < 0 || (p as usize) >= n_src_rows {
                    return Err(EncodeError::Value(format!(
                        "encode_cells: perm entry {p} out of range for {n_src_rows} source rows"
                    )));
                }
                let i = p as usize;
                let nnz = indptr[i + 1] - indptr[i];
                if nnz < 0 {
                    return Err(EncodeError::Value(
                        "encode_cells: source indptr is not non-decreasing".into(),
                    ));
                }
                total = total.checked_add(nnz).ok_or_else(|| {
                    EncodeError::Value("encode_cells: total nnz prefix sum overflowed i64".into())
                })?;
                out_indptr.push(total);
            }
        }
        None => {
            for i in 0..n_src_rows {
                let nnz = indptr[i + 1] - indptr[i];
                if nnz < 0 {
                    return Err(EncodeError::Value(
                        "encode_cells: source indptr is not non-decreasing".into(),
                    ));
                }
                total = total.checked_add(nnz).ok_or_else(|| {
                    EncodeError::Value("encode_cells: total nnz prefix sum overflowed i64".into())
                })?;
                out_indptr.push(total);
            }
        }
    }
    let total_nnz = total as u64;

    let payload_path = std::path::PathBuf::from(payload_out_path);
    let part_path = |b: usize| -> std::path::PathBuf {
        std::path::PathBuf::from(format!("{payload_out_path}.part{b}"))
    };

    // Empty archive/block: (re)create the payload when NOT appending; when appending, only
    // ensure it exists -- an empty encode_cells_block call must never truncate a prior block's
    // bytes. sha256("") for a whole non-appended archive; trivial offsets/indptr/frame_lens.
    if n_obs == 0 {
        if append {
            std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&payload_path)?;
        } else {
            File::create(&payload_path)?;
        }
        let sha = compute_checksum.then(|| format!("{:x}", Sha256::digest([])));
        return Ok((vec![0i64], out_indptr, 0, 0, sha, Vec::new()));
    }

    // div_ceil can over-partition (e.g. n_obs=5, workers=4 -> per=2 -> only 3 non-empty
    // chunks, and a spurious w=3 with row_lo=6 > row_hi=5 whose row_hi-row_lo underflows
    // usize). Recompute the worker count FROM `per` so every worker owns a non-empty range:
    // n_workers = div_ceil(n_obs, per) guarantees (n_workers-1)*per < n_obs, i.e. row_lo <
    // row_hi for the last worker. n_obs >= 1 here (n_obs==0 returned above) => per >= 1 =>
    // n_workers >= 1 (Gemini #212 r4).
    let per = n_obs.div_ceil(n_threads.max(1).min(n_obs));
    let n_workers = n_obs.div_ceil(per);

    // Fixed contiguous partition (mirrors decode_cells_typed): worker w owns [w*per, ..).
    // Each writes its own staging file, so no shared-output aliasing; join order == row order.
    let block_out: Vec<(Vec<i64>, u64)> =
        std::thread::scope(|scope| -> Result<Vec<(Vec<i64>, u64)>, EncodeError> {
            let mut handles = Vec::with_capacity(n_workers);
            for w in 0..n_workers {
                let row_lo = w * per;
                let row_hi = ((w + 1) * per).min(n_obs);
                let sp = part_path(w);
                handles.push(scope.spawn(move || {
                    encode_block(
                        enc,
                        data,
                        indices,
                        indptr,
                        perm,
                        row_lo,
                        row_hi,
                        &sp,
                        check_duplicates,
                        allow_lossy,
                    )
                }));
            }
            let mut collected = Vec::with_capacity(n_workers);
            for h in handles {
                collected.push(h.join().expect("encode worker panicked")?);
            }
            Ok(collected)
        })?;

    // Serial concat in block order -> final payload (APPENDING when `append`); build x_offsets +
    // frame_lens; optionally stream SHA-256.
    let mut offsets = Vec::with_capacity(n_obs + 1);
    offsets.push(0i64);
    let mut frame_lens = Vec::with_capacity(n_obs);
    let mut acc: i64 = 0;
    let mut hasher = compute_checksum.then(Sha256::new);
    let mut max_v: u64 = 0;
    {
        let mut out = if append {
            std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&payload_path)?
        } else {
            File::create(&payload_path)?
        };
        let mut buf = vec![0u8; 1 << 20];
        for (w, (lens, mx)) in block_out.iter().enumerate() {
            if *mx > max_v {
                max_v = *mx;
            }
            let sp = part_path(w);
            let mut r = File::open(&sp)?;
            loop {
                let n = r.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n])?;
                if let Some(h) = hasher.as_mut() {
                    h.update(&buf[..n]);
                }
            }
            drop(r);
            std::fs::remove_file(&sp)?;
            for &len in lens {
                acc += len;
                offsets.push(acc);
                frame_lens.push(len);
            }
        }
        out.flush()?;
    }
    if acc != *offsets.last().unwrap() || offsets.len() != n_obs + 1 {
        return Err(EncodeError::Value(
            "encode_cells: internal offset accounting mismatch".into(),
        ));
    }
    let sha = hasher.map(|h| format!("{:x}", h.finalize()));
    Ok((offsets, out_indptr, total_nnz, max_v, sha, frame_lens))
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, perm, payload_out_path, n_threads,
    compute_checksum=true, check_duplicates=false,
    codec="pfordelta", value_dtype_on_disk="uint32", allow_lossy=false
))]
#[allow(clippy::too_many_arguments)]
pub fn encode_cells(
    py: Python<'_>,
    x_data: &Bound<'_, PyAny>,
    x_indices: &Bound<'_, PyAny>,
    x_indptr: &Bound<'_, PyAny>,
    perm: PyReadonlyArray1<i64>,
    payload_out_path: String,
    n_threads: usize,
    compute_checksum: bool,
    check_duplicates: bool,
    codec: &str,
    value_dtype_on_disk: &str,
    allow_lossy: bool,
) -> PyResult<EncodeCellsPyOut> {
    let enc = FrameEnc::from_py(codec, value_dtype_on_disk).map_err(encode_err_to_py)?;
    let indptr = crate::indptr_to_i64(x_indptr)?;
    // Guard CSR structure (indptr[0]==0, non-decreasing, indptr[-1] <= data_len == indices_len)
    // BEFORE encoding so a malformed indptr cannot slice data/indices out of bounds and panic in
    // a worker thread (opaque PanicException across FFI). Mirrors encode_shards_inmem (Gemini
    // PR #212 r1). Cheap: O(n_src_rows+1), not nnz-scaled. perm bounds are checked separately below.
    crate::validate_csr_structure(&indptr, x_data.len()?, x_indices.len()?)?;
    let perm_s = perm
        .as_slice()
        .map_err(|_| PyValueError::new_err("encode_cells: perm must be contiguous"))?;

    let data_name = x_data.getattr("dtype")?.str()?.to_string();
    let idx_name = x_indices.getattr("dtype")?.str()?.to_string();

    // Reject integer source + float on-disk, mirroring _resolve_cell_codec (writer.py:175:
    // "data_dtype requires float input; integer counts store as uint32"). Python can never route
    // here -- float storage implies float input -- so this cannot change any reachable behavior.
    // It keeps BOTH engines' accept-sets IDENTICAL (the invariant the Gemini #212 r2 gate bug was
    // about) and makes FloatVal's integer arms genuinely dead rather than quietly lossy: an
    // "honest" int->float cast there is BLIND for i64 above 2^53 (PR #220 r1).
    if matches!(
        enc,
        FrameEnc::Zstd(ValueEnc::F32) | FrameEnc::Zstd(ValueEnc::F64)
    ) && !matches!(data_name.as_str(), "float32" | "float64")
    {
        return Err(encode_err_to_py(float_from_int_err()));
    }

    macro_rules! run {
        ($d:ty, $i:ty) => {{
            let df = x_data.extract::<PyReadonlyArray1<$d>>()?;
            let idf = x_indices.extract::<PyReadonlyArray1<$i>>()?;
            let ds = df
                .as_slice()
                .map_err(|_| PyValueError::new_err("encode_cells: data must be contiguous"))?;
            let is = idf
                .as_slice()
                .map_err(|_| PyValueError::new_err("encode_cells: indices must be contiguous"))?;
            let (offsets, out_indptr, total, mx, sha, _frame_lens) = py
                .allow_threads(|| {
                    encode_cells_typed::<$d, $i>(
                        enc,
                        ds,
                        is,
                        &indptr,
                        Some(perm_s),
                        &payload_out_path,
                        n_threads,
                        false, // append: encode_cells always (re)writes a fresh payload
                        compute_checksum,
                        check_duplicates,
                        allow_lossy,
                    )
                })
                .map_err(encode_err_to_py)?;
            Ok((
                offsets.into_pyarray_bound(py).unbind(),
                out_indptr.into_pyarray_bound(py).unbind(),
                total,
                mx,
                sha,
            ))
        }};
    }

    match (data_name.as_str(), idx_name.as_str()) {
        ("float32", "int32") => run!(f32, i32),
        ("float32", "int64") => run!(f32, i64),
        ("float64", "int32") => run!(f64, i32),
        ("float64", "int64") => run!(f64, i64),
        ("uint16", "int32") => run!(u16, i32),
        ("uint16", "int64") => run!(u16, i64),
        ("uint32", "int32") => run!(u32, i32),
        ("uint32", "int64") => run!(u32, i64),
        ("int64", "int32") => run!(i64, i32),
        ("int64", "int64") => run!(i64, i64),
        ("int32", "int32") => run!(i32, i32),
        ("int32", "int64") => run!(i32, i64),
        (d, i) => Err(PyValueError::new_err(format!(
            "encode_cells: unsupported source dtypes data={d}, indices={i}"
        ))),
    }
}

/// (frame_lens, block_nnz, block_max) — the encode_cells_block result.
type EncodeBlockPyOut = (Py<PyArray1<i64>>, u64, u64);

/// Encode ONE contiguous input-order block, APPENDING to the payload (#219 Task 5). No `perm`
/// (the block IS in input order -- the storage permutation is paid later by `reorder_frames`),
/// and no SHA (the writer hashes the FINAL storage-order payload in one pass, `_hash_file`).
///
/// `x_data`/`x_indices`/`x_indptr` describe a SELF-CONTAINED block CSR: `x_indptr[0] == 0`, sized
/// to exactly this block's rows (the caller slices it, e.g. `source.read_block(lo, hi)`). Mirrors
/// `encode_cells`, including BOTH of its up-front guards -- `crate::validate_csr_structure(...)`
/// and the integer-source + float-on-disk rejection via `float_from_int_err()` -- so the two
/// engines' accept-sets stay identical (the invariant the Gemini #212 r2 gate bug was about).
#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, payload_out_path, n_threads,
    check_duplicates=false, codec="pfordelta", value_dtype_on_disk="uint32", allow_lossy=false
))]
#[allow(clippy::too_many_arguments)]
pub fn encode_cells_block(
    py: Python<'_>,
    x_data: &Bound<'_, PyAny>,
    x_indices: &Bound<'_, PyAny>,
    x_indptr: &Bound<'_, PyAny>,
    payload_out_path: String,
    n_threads: usize,
    check_duplicates: bool,
    codec: &str,
    value_dtype_on_disk: &str,
    allow_lossy: bool,
) -> PyResult<EncodeBlockPyOut> {
    let enc = FrameEnc::from_py(codec, value_dtype_on_disk).map_err(encode_err_to_py)?;
    let indptr = crate::indptr_to_i64(x_indptr)?;
    // Guard CSR structure BEFORE encoding, exactly as encode_cells does (see its comment): a
    // malformed indptr must not slice data/indices out of bounds and panic in a worker thread.
    crate::validate_csr_structure(&indptr, x_data.len()?, x_indices.len()?)?;

    let data_name = x_data.getattr("dtype")?.str()?.to_string();
    let idx_name = x_indices.getattr("dtype")?.str()?.to_string();

    // Reject integer source + float on-disk -- see encode_cells's identical guard for the full
    // rationale. Keeping both entry points' checks identical is the point: a future change to
    // _resolve_cell_codec that lets this combination reach ONE of the two engines but not the
    // other reproduces the #212 r2 failure mode.
    if matches!(
        enc,
        FrameEnc::Zstd(ValueEnc::F32) | FrameEnc::Zstd(ValueEnc::F64)
    ) && !matches!(data_name.as_str(), "float32" | "float64")
    {
        return Err(encode_err_to_py(float_from_int_err()));
    }

    macro_rules! run {
        ($d:ty, $i:ty) => {{
            let df = x_data.extract::<PyReadonlyArray1<$d>>()?;
            let idf = x_indices.extract::<PyReadonlyArray1<$i>>()?;
            let ds = df.as_slice().map_err(|_| {
                PyValueError::new_err("encode_cells_block: data must be contiguous")
            })?;
            let is = idf.as_slice().map_err(|_| {
                PyValueError::new_err("encode_cells_block: indices must be contiguous")
            })?;
            let (_offsets, _out_indptr, block_nnz, block_max, _sha, frame_lens) = py
                .allow_threads(|| {
                    encode_cells_typed::<$d, $i>(
                        enc,
                        ds,
                        is,
                        &indptr,
                        None, // perm: the block is already in input order
                        &payload_out_path,
                        n_threads,
                        true,  // append
                        false, // compute_checksum: the writer hashes the whole payload once, later
                        check_duplicates,
                        allow_lossy,
                    )
                })
                .map_err(encode_err_to_py)?;
            Ok((
                frame_lens.into_pyarray_bound(py).unbind(),
                block_nnz,
                block_max,
            ))
        }};
    }

    match (data_name.as_str(), idx_name.as_str()) {
        ("float32", "int32") => run!(f32, i32),
        ("float32", "int64") => run!(f32, i64),
        ("float64", "int32") => run!(f64, i32),
        ("float64", "int64") => run!(f64, i64),
        ("uint16", "int32") => run!(u16, i32),
        ("uint16", "int64") => run!(u16, i64),
        ("uint32", "int32") => run!(u32, i32),
        ("uint32", "int64") => run!(u32, i64),
        ("int64", "int32") => run!(i64, i32),
        ("int64", "int64") => run!(i64, i64),
        ("int32", "int32") => run!(i32, i32),
        ("int32", "int64") => run!(i32, i64),
        (d, i) => Err(PyValueError::new_err(format!(
            "encode_cells_block: unsupported source dtypes data={d}, indices={i}"
        ))),
    }
}

/// Permute encoded frames from INPUT order into STORAGE order (#219 Task 5). Codec-agnostic: it
/// NEVER parses a frame, so it cannot be wrong about a codec, an on-disk dtype, or a lossy cast.
/// Frame r of the output is `staging[in_offsets[perm[r]] .. in_offsets[perm[r]+1]]`.
///
/// Guards mirror decode_rows_into's: a corrupt in_offsets/perm must fail LOUD with a ValueError,
/// never an OOB panic across the FFI boundary. The message substrings are load-bearing -- the
/// pure-Python fallback (`writer._reorder_frames_python`) reproduces them and the tests match on
/// them: "in_offsets[0]", "non-decreasing", "staging", "perm length", "out of range".
///
/// MONOTONICITY IS CHECKED BY COMPARISON, NEVER SUBTRACTION: `in_offsets[i+1] < in_offsets[i]`.
/// in_offsets is UNTRUSTED here (this IS the corrupt-input guard) and an i64 subtraction of an
/// adversarial pair OVERFLOWS -- which panics in debug and WRAPS in release, yielding a bogus
/// POSITIVE step that sails past a `< 0` check. Exactly why decode_rows_into guards its indptr
/// with checked_sub (Gemini #217). Use checked_sub for the frame LENGTH too:
/// `in_off[perm[r]+1].checked_sub(in_off[perm[r]])`. (Gemini, PR #221.)
///
/// v1 is single-threaded (sequential write). out_offsets is computable UP FRONT -- frame len for
/// storage row r is `in_off[perm[r]+1] - in_off[perm[r]]` -- so a parallel pwrite into disjoint
/// ranges is available later. NOT in this PR (unmeasured).
#[pyfunction]
pub fn reorder_frames(
    py: Python<'_>,
    staging_path: String,
    out_path: String,
    in_offsets: PyReadonlyArray1<i64>,
    perm: PyReadonlyArray1<i64>,
) -> PyResult<Py<PyArray1<i64>>> {
    let in_offsets = in_offsets
        .as_slice()
        .map_err(|_| PyValueError::new_err("reorder_frames: in_offsets must be contiguous"))?;
    let perm = perm
        .as_slice()
        .map_err(|_| PyValueError::new_err("reorder_frames: perm must be contiguous"))?;

    match in_offsets.first() {
        Some(&0) => {}
        _ => {
            return Err(PyValueError::new_err(
                "reorder_frames: in_offsets[0] must be 0",
            ))
        }
    }
    // Compare adjacent elements -- NEVER subtract. See the doc comment above: in_offsets is
    // UNTRUSTED here (this IS the corrupt-input guard), and a direct comparison does no
    // arithmetic, so it cannot wrap.
    if !in_offsets.windows(2).all(|w| w[0] <= w[1]) {
        return Err(PyValueError::new_err(
            "reorder_frames: in_offsets must be non-decreasing",
        ));
    }
    let n_src = in_offsets.len() - 1;

    // Open the staging file ONCE, here, and size it from THAT HANDLE (#305). Never
    // `std::fs::metadata(path)`: on an attribute-caching network filesystem -- WekaFS, which backs
    // every Arc store, and NFS -- a path stat can report a stale-SHORT st_size immediately after an
    // append through a different fd, while fstat on an open fd, lseek(SEEK_END) and the bytes
    // themselves are all already correct. No bytes are lost; only the cached attribute lags. Sizing
    // the staging file with a path stat therefore REJECTED perfectly good writes (measured: 20 of
    // 1500 cell writes on Python 3.11 + wekafs, 0 of 1500 on local ext4 -- which is why those
    // failures were misread as a CPython 3.13 defect).
    //
    // The handle is then USED BY the copy loop below, so the file read is the file measured: the
    // path cannot be replaced between the size check and the read. That is the whole guarantee -- a
    // concurrent writer on the same inode could still change the bytes after `metadata()`, and
    // nothing here claims otherwise. (codex, checkpoint 2 round 3.)
    // `File::metadata` is fstat(2), so this is open+fstat where it used to be stat+open -- the same
    // two syscalls, just in the order that makes the answer trustworthy.
    //
    // The open happens INSIDE allow_threads: on exactly the network filesystems that motivate this
    // change, open() can block for a while, and it used to be off the GIL because it lived in the
    // copy closure. Keeping it there costs nothing and keeps other Python threads running.
    // (codex, checkpoint 2 finding 2.)
    //
    // No Python-level test can inject this (the stale stat lives below the FFI boundary); the
    // guard's own behaviour on a genuinely wrong in_offsets is covered by tests/cell/
    // test_reorder_frames.py, and the Python twin's fix by test_stale_stat_size.py.
    let (mut sf, staged_len) = py.allow_threads(|| -> std::io::Result<(File, u64)> {
        let f = File::open(&staging_path)?;
        let len = f.metadata()?.len();
        Ok((f, len))
    })?;
    let last = in_offsets[n_src];
    // last >= 0 is already guaranteed (in_offsets[0] == 0 and non-decreasing), so `as u64` exact.
    if last as u64 != staged_len {
        return Err(PyValueError::new_err(format!(
            "reorder_frames: in_offsets[-1]={last} != staging file size {staged_len}"
        )));
    }

    // perm must define a full reorder of every source frame -- a shorter perm silently drops
    // frames, a longer one (with repeated valid indices) silently duplicates them. Both would
    // otherwise return Ok with no other signal, so this is checked explicitly rather than left
    // to the per-entry bounds check below (which only catches invalid *values*, not a wrong
    // *count* of otherwise-valid ones). (#219 Task 5 review finding 2.)
    if perm.len() != n_src {
        return Err(PyValueError::new_err(format!(
            "reorder_frames: perm length {} != {n_src} frames",
            perm.len()
        )));
    }

    // Perm-bounds validation + out_offsets, single pass O(n_dst). checked_sub for the frame
    // length -- see the doc comment above: in_offsets is still UNTRUSTED at this point.
    let mut out_offsets = Vec::with_capacity(perm.len() + 1);
    out_offsets.push(0i64);
    let mut acc: i64 = 0;
    for &p in perm {
        if p < 0 || (p as usize) >= n_src {
            return Err(PyValueError::new_err(format!(
                "reorder_frames: perm entry {p} out of range for {n_src} frames"
            )));
        }
        let i = p as usize;
        let len = in_offsets[i + 1]
            .checked_sub(in_offsets[i])
            .ok_or_else(|| {
                PyValueError::new_err("reorder_frames: in_offsets must be non-decreasing")
            })?;
        acc = acc.checked_add(len).ok_or_else(|| {
            PyValueError::new_err("reorder_frames: output offset prefix sum overflowed i64")
        })?;
        out_offsets.push(acc);
    }

    // Sequential copy loop, GIL released. Random-access reads of the staging file (perm can be
    // any order); sequential writes of the output. `out_offsets` was fully computed above from
    // the VALIDATED in_offsets/perm, so the lengths used here need no further checking.
    py.allow_threads(|| -> std::io::Result<()> {
        let mut of = BufWriter::new(File::create(&out_path)?);
        let mut buf: Vec<u8> = Vec::new();
        for (r, &p) in perm.iter().enumerate() {
            let len = (out_offsets[r + 1] - out_offsets[r]) as usize;
            if len == 0 {
                continue;
            }
            let i = p as usize;
            buf.resize(len, 0u8);
            sf.seek(SeekFrom::Start(in_offsets[i] as u64))?;
            sf.read_exact(&mut buf)?;
            of.write_all(&buf)?;
        }
        of.flush()
    })?;

    Ok(out_offsets.into_pyarray_bound(py).unbind())
}
