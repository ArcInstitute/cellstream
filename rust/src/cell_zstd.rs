//! zstd per-cell frame codec -- BOTH directions.
//!
//! Writes NO new codec kernels in either direction. DECODE: `decode::decode_chunk_into`
//! already fuses
//! `zstd -> un-bytedelta -> un-byteshuffle -> cast`, and its `DecodeFilter` variants map
//! 1:1 onto the three roles of PR1's frame (`src/cellstream/cell/codec.py::_ByteZstdEncoder`,
//! which itself calls the shard `v2/codec.py` primitives verbatim):
//!
//! | PR1 Python                          | here                                |
//! |-------------------------------------|-------------------------------------|
//! | `byte_filter_encode(role="indices")`| `Src=OnDiskU32`, `ShuffleDelta`     |
//! | `byte_filter_encode(role="data")`   | `Src=OnDiskU32`, `Shuffle`          |
//! | `zstd_only_encode` (float)          | `Src=OnDiskF32/F64`, `Raw`          |
//!
//! (`_BYTEDELTA_ROLES = {"indices", "indptr"}` in v2/codec.py -- so indices get the
//! byte-plane delta and integer DATA does not.)
//!
//! `Gather` is ALWAYS `Src`: the min-aware low-plane gather (#161) never applies to cell
//! frames -- they record no min dtype -- and `decode_chunk_into` debug-asserts
//! `Gather == Src` for `Raw`. With `Gather == Src` the high-plane zero check is skipped and
//! the full width is read, which is what we want.
//!
//! `decompressed_size` is DERIVED (`nnz * itemsize`), never stored, so nothing about the
//! on-disk format changes.
//!
//! ENCODE (PR3) is the exact mirror: `shx::encode_chunk` / `encode_chunk_raw` -- the same
//! primitives the shard writer already uses (`encode.rs::flush_chunk`) -- cover the same
//! three roles:
//!
//! | PR1 Python                          | here                              |
//! |-------------------------------------|-----------------------------------|
//! | `byte_filter_encode(role="indices")`| `encode_chunk(le, t=4, delta=true)`  |
//! | `byte_filter_encode(role="data")`   | `encode_chunk(le, t=4, delta=false)` |
//! | `zstd_only_encode` (float)          | `encode_chunk_raw(le)`            |
#![cfg(target_arch = "x86_64")]

use crate::cell::{CellScratch, CountU32, EncodeError, FloatVal, IdxU32, ValueEnc};
use crate::decode::{
    decode_chunk_into, Cast, CastBack, DecodeError, DecodeFilter, OnDiskF32, OnDiskF64, OnDiskU32,
};
use crate::shx::{encode_chunk, encode_chunk_raw};

/// Decode a frame's index blob into ABSOLUTE u32 column indices in `out[..nnz]`.
///
/// zstd byte-plane-deltas the SHUFFLED planes, so the inverse reconstructs the index
/// VALUES directly -- there is NO value-wise cumsum (unlike pfordelta, which stores gaps).
/// The caller still validates each index against `[0, n_vars)` before narrowing.
pub fn decode_indices(blob: &[u8], nnz: usize, out: &mut [u32]) -> Result<(), DecodeError> {
    decode_chunk_into::<OnDiskU32, OnDiskU32, u32>(
        blob,
        nnz,
        nnz * 4,
        DecodeFilter::ShuffleDelta,
        false, // u32 -> u32 can never be lossy
        out,
    )
}

/// Decode a frame's value blob straight into `out[..nnz]`, casting to `Dst` with the lossy
/// policy fused (`check = !allow_lossy`, matching decode.rs's tier-3 contract).
///
/// No scratch buffer: unlike pfordelta -- which unpacks to a u32 scratch and then runs a
/// separate cast loop -- this is a single pass over the values.
pub fn decode_values<Dst: Cast + CastBack>(
    enc: ValueEnc,
    blob: &[u8],
    nnz: usize,
    allow_lossy: bool,
    out: &mut [Dst],
) -> Result<(), DecodeError> {
    let check = !allow_lossy;
    match enc {
        ValueEnc::U32 => decode_chunk_into::<OnDiskU32, OnDiskU32, Dst>(
            blob,
            nnz,
            nnz * 4,
            DecodeFilter::Shuffle,
            check,
            out,
        ),
        ValueEnc::F32 => decode_chunk_into::<OnDiskF32, OnDiskF32, Dst>(
            blob,
            nnz,
            nnz * 4,
            DecodeFilter::Raw,
            check,
            out,
        ),
        ValueEnc::F64 => decode_chunk_into::<OnDiskF64, OnDiskF64, Dst>(
            blob,
            nnz,
            nnz * 8,
            DecodeFilter::Raw,
            check,
            out,
        ),
    }
}

/// Encode one zstd cell frame into `scratch.frame`; returns the frame length.
///
/// Byte-exact with `cell/codec.py::_ByteZstdEncoder.encode`:
///   `[u32 LE idx_blob_len][idx_blob][val_blob]`
/// where
///   idx_blob = `byte_filter_encode(indices, role="indices")` = shuffle + BYTEDELTA + zstd-1
///   val_blob = `byte_filter_encode(values,  role="data")`    = shuffle +            zstd-1  (U32)
///            | `zstd_only_encode(values)`                    = raw                  zstd-1  (F32/F64)
///
/// Writes NO new codec kernels: `shx::encode_chunk` / `encode_chunk_raw` are the same
/// primitives the shard writer already uses in production (`encode.rs::flush_chunk`).
///
/// Indices are stored ABSOLUTE, not as gaps: zstd byte-plane-deltas the SHUFFLED planes, so the
/// frame holds index VALUES (unlike pfordelta, which stores value-wise gaps).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_frame<D: CountU32 + FloatVal, I: IdxU32>(
    enc: ValueEnc,
    idx_src: &[I],
    val_src: &[D],
    scratch: &mut CellScratch,
    max_v: &mut u64,
    check_duplicates: bool,
    allow_lossy: bool,
) -> Result<usize, EncodeError> {
    let nnz = idx_src.len();

    // --- indices: ABSOLUTE u32, with codec.py's sortedness guard folded into the gather ---
    // Single pass: validate and emit the LE bytes straight into le_buf. There is no
    // intermediate Vec<u32> -- encode_chunk consumes bytes, so materializing the values first
    // would be a second pass over nnz for nothing. (Gemini, PR #220.)
    scratch.le_buf.clear();
    scratch.le_buf.reserve(nnz * 4);
    let mut prev: u32 = 0;
    for (k, &raw) in idx_src.iter().enumerate() {
        let cur = raw.idx_u32();
        if k > 0 {
            if cur < prev {
                return Err(EncodeError::Value(
                    "zstd codec requires non-decreasing (sorted) indices".into(),
                ));
            }
            // A duplicate (cell, gene) entry is exactly an equal adjacent index on sorted input.
            if check_duplicates && cur == prev {
                return Err(EncodeError::Value(format!(
                    "encode_cells: duplicate column index {cur} within a cell \
                     (require_unique_indices=True); pass require_unique_indices=False \
                     to permit duplicate (cell, gene) entries"
                )));
            }
        }
        scratch.le_buf.extend_from_slice(&cur.to_le_bytes());
        prev = cur;
    }
    let idx_blob = encode_chunk(&scratch.le_buf, 4, true);

    // --- values: cast + fold-validate per on-disk width, again in a SINGLE pass into le_buf ---
    scratch.le_buf.clear();
    let val_blob = match enc {
        ValueEnc::U32 => {
            scratch.le_buf.reserve(nnz * 4);
            for &v in val_src {
                let vu = v.count_u32()?;
                if vu as u64 > *max_v {
                    *max_v = vu as u64;
                }
                scratch.le_buf.extend_from_slice(&vu.to_le_bytes());
            }
            // role="data" -> NO bytedelta (_BYTEDELTA_ROLES = {"indices", "indptr"})
            encode_chunk(&scratch.le_buf, 4, false)
        }
        ValueEnc::F32 => {
            scratch.le_buf.reserve(nnz * 4);
            for &v in val_src {
                scratch
                    .le_buf
                    .extend_from_slice(&v.to_f32_checked(allow_lossy)?.to_le_bytes());
            }
            encode_chunk_raw(&scratch.le_buf) // zstd_only_encode
        }
        ValueEnc::F64 => {
            scratch.le_buf.reserve(nnz * 8);
            for &v in val_src {
                scratch
                    .le_buf
                    .extend_from_slice(&v.to_f64_checked()?.to_le_bytes());
            }
            encode_chunk_raw(&scratch.le_buf) // zstd_only_encode
        }
    };

    // --- frame framing: identical to pfordelta's ---
    scratch.frame.clear();
    scratch
        .frame
        .extend_from_slice(&(idx_blob.len() as u32).to_le_bytes());
    scratch.frame.extend_from_slice(&idx_blob);
    scratch.frame.extend_from_slice(&val_blob);
    Ok(scratch.frame.len())
}
