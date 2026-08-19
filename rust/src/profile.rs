//! Single-thread encode micro-bench used to attribute per-thread time across
//! {feed, shuffle, delta, zstd, staging}. Built into the cdylib and reached
//! from Python via lib::_profile_encode_single_shard. NOT on the production
//! write path; it times the real crate primitives on a synthetic CSR so the
//! attribution generalizes (per-element costs are scale-invariant).

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::narrow::{decide_data_dtypes, OnDiskDtype, SrcKind};
use crate::shx::{byte_delta, byte_shuffle, zstd1, CHUNK_ELEMS};

/// Per-process monotonic counter so each profile call gets a distinct staging
/// file even when several run concurrently in one process (same PID).
static STAGING_SEQ: AtomicU64 = AtomicU64::new(0);

/// Per-bucket wall-time (seconds) for one single-thread encode pass.
#[derive(Clone, Debug, Default)]
pub struct Breakdown {
    pub decide_s: f64,
    pub decide_indices_s: f64,
    pub feed_s: f64,
    pub shuffle_s: f64,
    pub delta_s: f64,
    pub zstd_s: f64,
    pub staging_s: f64,
    pub nnz: usize,
}

/// Which feed implementation to time. `Scalar` = the current production
/// per-element `extend_from_slice(to_le_bytes())`; `Bulk` = the Phase-2
/// vectorized narrow into a u16 chunk buffer (added in Phase 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedMode {
    Scalar,
    Bulk,
}

/// Deterministic synthetic CSR payload: `nnz` f32 data values (small integers,
/// u16-safe) and `nnz` i32 indices (u16-safe), shaped as ~`nnz_per_row` per row.
/// Returns (data, indices). indptr is implied (contiguous spans) — the per-element
/// feed cost does not depend on row boundaries, so a flat stream is faithful.
pub fn synth_csr(nnz: usize, nnz_per_row: usize) -> (Vec<f32>, Vec<i32>) {
    let n_vars: i32 = 20_000;
    let mut data = Vec::with_capacity(nnz);
    let mut indices = Vec::with_capacity(nnz);
    for k in 0..nnz {
        data.push(((k % 50) + 1) as f32);
        // stride so within a row the indices increase then wrap (realistic-ish)
        let col = ((k % nnz_per_row.max(1)) as i32 * 3) % n_vars;
        indices.push(col);
    }
    (data, indices)
}

/// Time a single-thread encode of a synthetic CSR, attributing wall-time to the
/// five buckets at chunk (CHUNK_ELEMS) granularity. Each chunk's compressed bytes
/// are appended sequentially to one staging file in `out_dir` (removed at the
/// end) so the staging bucket times realistic sequential writes.
pub fn profile_single_shard(
    nnz: usize,
    nnz_per_row: usize,
    mode: FeedMode,
    out_dir: &Path,
) -> Breakdown {
    use std::fs::File;
    use std::io::Write;

    let (data, indices) = synth_csr(nnz, nnz_per_row);
    let mut b = Breakdown {
        nnz,
        ..Default::default()
    };

    // decide buckets: production runs these BEFORE encoding to pick the on-disk
    // dtype; timed here at n_threads=1 for per-element attribution consistent with
    // the other single-thread buckets. Synthetic data is integer-routable, so this
    // times the full-scan (success) path.
    let t_dec = Instant::now();
    let _ddt = decide_data_dtypes(&data, SrcKind::F32, 1);
    b.decide_s = t_dec.elapsed().as_secs_f64();
    let t_deci = Instant::now();
    let _idt = OnDiskDtype::U32;
    b.decide_indices_s = t_deci.elapsed().as_secs_f64();

    // Unique staging name (PID + per-call seq) so concurrent profiler runs
    // sharing one out_dir — across processes (the Python driver defaults to /tmp)
    // or threads in one process — don't clobber each other's file.
    let seq = STAGING_SEQ.fetch_add(1, Ordering::Relaxed);
    let staging_path = out_dir.join(format!(
        "profile_staging_{}_{}.bin",
        std::process::id(),
        seq
    ));
    let mut staging = File::create(&staging_path).expect("create staging");

    let mut pos = 0usize;
    // reused buffers (bounded at CHUNK_ELEMS — the flat-RAM invariant)
    let mut data_le: Vec<u8> = Vec::with_capacity(CHUNK_ELEMS * 2);
    let mut idx_le: Vec<u8> = Vec::with_capacity(CHUNK_ELEMS * 2);
    let mut data_u16: Vec<u16> = Vec::with_capacity(CHUNK_ELEMS);
    let mut idx_u16: Vec<u16> = Vec::with_capacity(CHUNK_ELEMS);

    while pos < nnz {
        let end = (pos + CHUNK_ELEMS).min(nnz);
        let n = end - pos;

        // --- feed bucket: narrow + produce the per-chunk staged form ---
        let t0 = Instant::now();
        let (data_planes, idx_planes_unshuffled): (Vec<u8>, Vec<u8>) = match mode {
            FeedMode::Scalar => {
                data_le.clear();
                idx_le.clear();
                for k in pos..end {
                    data_le.extend_from_slice(&(data[k] as u16).to_le_bytes());
                    idx_le.extend_from_slice(&(indices[k] as u16).to_le_bytes());
                }
                // hand the interleaved LE bytes to the shuffle bucket below
                (std::mem::take(&mut data_le), std::mem::take(&mut idx_le))
            }
            FeedMode::Bulk => {
                data_u16.clear();
                idx_u16.clear();
                data_u16.extend(data[pos..end].iter().map(|&v| v as u16));
                idx_u16.extend(indices[pos..end].iter().map(|&v| v as u16));
                // re-encode to LE bytes only to share one shuffle code path in the
                // profiler; Phase 2 production skips this (shuffle reads u16 directly).
                let d: Vec<u8> = data_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
                let i: Vec<u8> = idx_u16.iter().flat_map(|v| v.to_le_bytes()).collect();
                (d, i)
            }
        };
        b.feed_s += t0.elapsed().as_secs_f64();

        // --- shuffle bucket ---
        let t1 = Instant::now();
        let data_shuf = byte_shuffle(&data_planes, 2);
        let mut idx_shuf = byte_shuffle(&idx_planes_unshuffled, 2);
        b.shuffle_s += t1.elapsed().as_secs_f64();

        // --- delta bucket (indices only) ---
        let t2 = Instant::now();
        byte_delta(&mut idx_shuf, n, 2);
        b.delta_s += t2.elapsed().as_secs_f64();

        // --- zstd bucket ---
        let t3 = Instant::now();
        let cd = zstd1(&data_shuf);
        let ci = zstd1(&idx_shuf);
        b.zstd_s += t3.elapsed().as_secs_f64();

        // --- staging bucket ---
        let t4 = Instant::now();
        staging.write_all(&cd).expect("write staging");
        staging.write_all(&ci).expect("write staging");
        b.staging_s += t4.elapsed().as_secs_f64();

        // recycle the LE byte capacity in scalar mode
        if mode == FeedMode::Scalar {
            data_le = data_planes;
            idx_le = idx_planes_unshuffled;
            data_le.clear();
            idx_le.clear();
        }

        pos = end;
    }
    // close the handle before unlinking so cleanup also works on non-Unix
    drop(staging);
    let _ = std::fs::remove_file(&staging_path);
    b
}

/// Dense element dtype for the throughput probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseDtype {
    U16,
    F32,
}

/// Dense codec variant under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseCodec {
    Plain,
    Shuffle,
    ShuffleDelta,
}

/// Per-bucket wall-time + compressed size for one single-thread DENSE encode pass.
#[derive(Clone, Debug, Default)]
pub struct DenseBreakdown {
    pub feed_s: f64,
    pub shuffle_s: f64,
    pub delta_s: f64,
    pub zstd_s: f64,
    pub staging_s: f64,
    pub n_elements: usize, // n_rows * n_var (dense, includes zeros)
    pub nnz: usize,        // logical nonzeros (≈ n_elements * density)
    pub compressed_bytes: usize,
}

/// Deterministic synthetic DENSE block as raw row-major LE bytes at `itemsize`.
/// ~`density` of elements are small nonzero counts; the rest are zero.
pub fn synth_dense_bytes(n_rows: usize, n_var: usize, density: f64, dt: DenseDtype) -> Vec<u8> {
    // density > 1 keeps writes in bounds but yields nnz > n_elements (collisions),
    // which is a meaningless probe input — assert the invariant for Rust callers
    // (the pyo3 wrapper already rejects it with a PyValueError). debug-only: zero
    // cost in the release build that produces the throughput numbers.
    debug_assert!(
        (0.0..=1.0).contains(&density),
        "synth_dense_bytes: density must be in [0, 1], got {density}"
    );
    let total = n_rows * n_var;
    let isz = match dt {
        DenseDtype::U16 => 2,
        DenseDtype::F32 => 4,
    };
    let mut v = vec![0u8; total * isz];
    // Place EXACTLY round(total*density) nonzeros at evenly-spread positions.
    // (A float `step = (1/density) as usize` truncates — e.g. density=0.35 ->
    // step=2 -> ~50% nonzeros — which would benchmark the wrong density.)
    let n_nonzero = ((total as f64) * density).round() as usize;
    for i in 0..n_nonzero {
        // strictly-increasing distinct index for i in 0..n_nonzero (<= total)
        let k = ((i as u128 * total as u128) / n_nonzero as u128) as usize;
        let val = ((i % 50) + 1) as u32;
        let le: [u8; 4] = match dt {
            DenseDtype::U16 => {
                let b = (val as u16).to_le_bytes();
                [b[0], b[1], 0, 0]
            }
            DenseDtype::F32 => (val as f32).to_le_bytes(),
        };
        v[k * isz..k * isz + isz].copy_from_slice(&le[..isz]);
    }
    v
}

/// Time a single-thread DENSE encode: chunk the row-major block, apply the chosen
/// transform, zstd-1, stage. Returns buckets + total compressed bytes. NOT on the
/// production write path; mirrors `profile_single_shard` for the CSR/dense A/B.
pub fn profile_dense_block(
    n_rows: usize,
    n_var: usize,
    density: f64,
    dt: DenseDtype,
    codec: DenseCodec,
    out_dir: &Path,
) -> DenseBreakdown {
    use std::fs::File;
    use std::io::Write;

    let isz = match dt {
        DenseDtype::U16 => 2,
        DenseDtype::F32 => 4,
    };
    let block = synth_dense_bytes(n_rows, n_var, density, dt);
    let total_elems = n_rows * n_var;
    let mut b = DenseBreakdown {
        n_elements: total_elems,
        nnz: ((total_elems as f64) * density).round() as usize, // matches synth_dense_bytes
        ..Default::default()
    };

    let seq = STAGING_SEQ.fetch_add(1, Ordering::Relaxed);
    let staging_path = out_dir.join(format!(
        "profile_dense_staging_{}_{}.bin",
        std::process::id(),
        seq
    ));
    let mut staging = File::create(&staging_path).expect("create dense staging");

    let chunk_bytes = CHUNK_ELEMS * isz;
    let mut pos = 0usize; // byte position
    while pos < block.len() {
        let end = (pos + chunk_bytes).min(block.len());
        let n = (end - pos) / isz; // elements in this chunk

        // feed_s stays 0: dense has NO gather and NO narrow — the contiguous
        // chunk is borrowed, not copied (the CSR probe's feed cost is exactly the
        // gather+narrow this avoids). Borrowing here keeps the A/B honest; a
        // per-chunk `.to_vec()` would invent a copy cost dense does not pay.
        let src = &block[pos..end];

        // --- shuffle bucket ---
        let t1 = Instant::now();
        let shuffled: Option<Vec<u8>> = match codec {
            DenseCodec::Plain => None,
            DenseCodec::Shuffle | DenseCodec::ShuffleDelta => Some(byte_shuffle(src, isz)),
        };
        b.shuffle_s += t1.elapsed().as_secs_f64();

        // --- delta bucket ---
        let t2 = Instant::now();
        let mut owned = shuffled;
        if codec == DenseCodec::ShuffleDelta {
            byte_delta(owned.as_mut().expect("shuffle ran for delta codec"), n, isz);
        }
        b.delta_s += t2.elapsed().as_secs_f64();

        // --- zstd bucket ---
        let t3 = Instant::now();
        let to_compress: &[u8] = owned.as_deref().unwrap_or(src);
        let c = zstd1(to_compress);
        b.zstd_s += t3.elapsed().as_secs_f64();
        b.compressed_bytes += c.len();

        // --- staging bucket ---
        let t4 = Instant::now();
        staging.write_all(&c).expect("write dense staging");
        b.staging_s += t4.elapsed().as_secs_f64();

        pos = end;
    }
    drop(staging);
    let _ = std::fs::remove_file(&staging_path);
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_generator_yields_requested_nnz() {
        let (data, indices) = synth_csr(50_000, 6_000);
        assert_eq!(data.len(), 50_000);
        assert_eq!(indices.len(), 50_000);
        // values fit u16 losslessly (so the narrow is the real production narrow)
        assert!(data
            .iter()
            .all(|&v| v.fract() == 0.0 && (0.0..=65535.0).contains(&v)));
        assert!(indices.iter().all(|&v| (0..=65535).contains(&v)));
    }

    #[test]
    fn profile_scalar_returns_positive_buckets() {
        let dir = std::env::temp_dir().join("cellstream_profile_test");
        let _ = std::fs::create_dir_all(&dir);
        // small nnz keeps the debug `cargo test` fast; one (partial) chunk still
        // exercises shuffle/delta/zstd so zstd_s > 0. Attribution is scale-invariant.
        let b = profile_single_shard(200_000, 6_000, FeedMode::Scalar, &dir);
        assert_eq!(b.nnz, 200_000);
        assert!(b.feed_s >= 0.0 && b.shuffle_s >= 0.0 && b.zstd_s > 0.0 && b.staging_s >= 0.0);
        // decide buckets populated (the pre-encode scans ran); synthetic data is u16-safe.
        assert!(b.decide_s >= 0.0 && b.decide_indices_s >= 0.0);
    }

    #[test]
    fn dense_probe_returns_positive_buckets_and_bytes() {
        let dir = std::env::temp_dir();
        let b = profile_dense_block(1000, 2000, 0.35, DenseDtype::U16, DenseCodec::Shuffle, &dir);
        assert_eq!(b.n_elements, 1000 * 2000);
        assert!(b.nnz > 0 && b.nnz < b.n_elements);
        assert!(b.compressed_bytes > 0);
        // sparse dense (35% small ints) must compress well below raw 2 bytes/elem
        assert!(b.compressed_bytes < b.n_elements * 2);
        assert!(b.zstd_s >= 0.0 && b.shuffle_s >= 0.0);
    }
}
