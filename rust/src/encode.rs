//! In-memory encode core: stream each shard's rows from the resident source,
//! route integer-like data to uint32 in-thread, encode to .shx with N scoped threads. Generic over
//! source data/index dtypes; indptr is pre-widened to i64.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::narrow::{scan_u32, NarrowError, OnDiskDtype, ToIntScan};
use crate::shx::{assemble_shx_streaming, encode_chunk, encode_chunk_raw, ChunkDesc, CHUNK_ELEMS};

const SHUFFLE: &str = "byteshuffle-bytedelta";
/// #142: float DATA streams store raw (zstd-only) under the byte-filter codec.
const SHUFFLE_RAWDATA: &str = "byteshuffle-bytedelta-rawdata";

/// Source element that can be emitted as little-endian on-disk bytes.
pub trait ToOnDisk: Copy {
    fn as_u16_le(self) -> [u8; 2];
    fn as_u32_le(self) -> [u8; 4];
    fn as_f32_le(self) -> [u8; 4];
    fn as_f64_le(self) -> [u8; 8];
    fn as_i64(self) -> i64;
    #[allow(dead_code)]
    fn is_zero(self) -> bool;
}

macro_rules! impl_to_on_disk {
    ($t:ty) => {
        impl ToOnDisk for $t {
            #[inline]
            fn as_u16_le(self) -> [u8; 2] {
                (self as u16).to_le_bytes()
            }
            #[inline]
            fn as_u32_le(self) -> [u8; 4] {
                (self as u32).to_le_bytes()
            }
            #[inline]
            fn as_f32_le(self) -> [u8; 4] {
                (self as f32).to_le_bytes()
            }
            #[inline]
            fn as_f64_le(self) -> [u8; 8] {
                (self as f64).to_le_bytes()
            }
            #[inline]
            fn as_i64(self) -> i64 {
                self as i64
            }
            #[inline]
            fn is_zero(self) -> bool {
                self == 0 as $t
            }
        }
    };
}
impl_to_on_disk!(f32);
impl_to_on_disk!(f64);
impl_to_on_disk!(i32);
impl_to_on_disk!(i64);
impl_to_on_disk!(u16);
impl_to_on_disk!(u32);

fn extend_on_disk<T: ToOnDisk>(buf: &mut Vec<u8>, value: T, dtype: OnDiskDtype) {
    match dtype {
        OnDiskDtype::U16 => buf.extend_from_slice(&value.as_u16_le()),
        OnDiskDtype::U32 => buf.extend_from_slice(&value.as_u32_le()),
        OnDiskDtype::F32 => buf.extend_from_slice(&value.as_f32_le()),
        OnDiskDtype::F64 => buf.extend_from_slice(&value.as_f64_le()),
    }
}

pub trait RowSource {
    type Data: ToOnDisk;

    fn for_each_in_row<F: FnMut(Self::Data, i64)>(&self, r: usize, f: F);
}

pub struct CsrRowSource<'a, D, I> {
    data: &'a [D],
    indices: &'a [I],
    indptr: &'a [i64],
}

impl<D: ToOnDisk, I: ToOnDisk> RowSource for CsrRowSource<'_, D, I> {
    type Data = D;

    fn for_each_in_row<F: FnMut(D, i64)>(&self, r: usize, mut f: F) {
        let a = self.indptr[r] as usize;
        let b = self.indptr[r + 1] as usize;
        for k in a..b {
            f(self.data[k], self.indices[k].as_i64());
        }
    }
}

#[allow(dead_code)]
pub struct DenseRowSource<'a, D> {
    dense: &'a [D],
    n_vars: usize,
}

impl<'a, D: ToOnDisk> DenseRowSource<'a, D> {
    #[allow(dead_code)]
    pub fn new(dense: &'a [D], n_vars: usize) -> Self {
        Self { dense, n_vars }
    }
}

impl<D: ToOnDisk> RowSource for DenseRowSource<'_, D> {
    type Data = D;

    fn for_each_in_row<F: FnMut(D, i64)>(&self, r: usize, mut f: F) {
        let start = r * self.n_vars;
        let end = start + self.n_vars;
        for (col, &value) in self.dense[start..end].iter().enumerate() {
            if !value.is_zero() {
                f(value, col as i64);
            }
        }
    }
}

fn flush_chunk(
    staging: &mut File,
    buf: &[u8],
    t: usize,
    delta: bool,
    raw: bool,
) -> io::Result<ChunkDesc> {
    // raw (#142): zstd only, no shuffle/delta — for the float data stream.
    let compressed = if raw {
        encode_chunk_raw(buf)
    } else {
        encode_chunk(buf, t, delta)
    };
    staging.write_all(&compressed)?;
    Ok(ChunkDesc {
        offset: 0,
        compressed_size: compressed.len(),
        decompressed_size: buf.len(),
    })
}

/// Encode one shard: stream-compress rolling chunks + assemble. Returns nnz.
fn encode_one_shard<S: RowSource>(
    shard_idx: usize,
    row_list: &[i64],
    source: &S,
    n_vars: usize,
    out_dir: &Path,
    data_dtype: OnDiskDtype,
    indices_dtype: OnDiskDtype,
) -> io::Result<usize> {
    let path = out_dir.join(format!("shard_{:04}.shx", shard_idx));
    let data_staging_path = out_dir.join(format!("shard_{:04}.shx.d.staging", shard_idx));
    let indices_staging_path = out_dir.join(format!("shard_{:04}.shx.i.staging", shard_idx));
    let indptr_staging_path = out_dir.join(format!("shard_{:04}.shx.p.staging", shard_idx));

    // RAII cleanup: remove the staging files on EVERY exit path — success OR an
    // early `?` I/O error — so a failed encode doesn't leak `.staging` files (#111).
    struct StagingCleanup<'a> {
        data_path: &'a Path,
        indices_path: &'a Path,
        indptr_path: &'a Path,
    }
    impl Drop for StagingCleanup<'_> {
        fn drop(&mut self) {
            let _ = fs::remove_file(self.data_path);
            let _ = fs::remove_file(self.indices_path);
            let _ = fs::remove_file(self.indptr_path);
        }
    }
    let _staging_cleanup = StagingCleanup {
        data_path: &data_staging_path,
        indices_path: &indices_staging_path,
        indptr_path: &indptr_staging_path,
    };

    let mut data_descs = Vec::new();
    let mut indices_descs = Vec::new();
    let mut indptr_descs = Vec::new();
    let mut indptr_local: Vec<i64> = Vec::with_capacity(row_list.len() + 1);
    indptr_local.push(0);

    // #142: a float on-disk data dtype stores the DATA stream raw (zstd only, no
    // shuffle/delta) and self-describes with the rawdata marker; integer data +
    // the index/indptr streams keep SHUFFLE(+DELTA). The data dtype is uniform per
    // archive, so this is decided once per shard from the resolved dtype.
    let data_raw = matches!(data_dtype, OnDiskDtype::F32 | OnDiskDtype::F64);
    let shuffle = if data_raw { SHUFFLE_RAWDATA } else { SHUFFLE };

    let nnz;
    {
        let mut data_staging = File::create(&data_staging_path)?;
        let mut indices_staging = File::create(&indices_staging_path)?;
        let t_d = data_dtype.itemsize();
        let t_i = indices_dtype.itemsize();
        let mut data_buf: Vec<u8> = Vec::with_capacity(CHUNK_ELEMS * t_d);
        let mut idx_buf: Vec<u8> = Vec::with_capacity(CHUNK_ELEMS * t_i);
        let mut chunk_elems = 0;
        let mut cum: i64 = 0;

        for &r in row_list {
            let r = r as usize;
            let before = cum;
            let mut write_err: io::Result<()> = Ok(());
            source.for_each_in_row(r, |value, col| {
                if write_err.is_err() {
                    return;
                }
                extend_on_disk(&mut data_buf, value, data_dtype);
                extend_on_disk(&mut idx_buf, col, indices_dtype);
                chunk_elems += 1;
                if chunk_elems == CHUNK_ELEMS {
                    debug_assert_eq!(data_buf.len(), CHUNK_ELEMS * t_d);
                    debug_assert_eq!(idx_buf.len(), CHUNK_ELEMS * t_i);
                    let data_desc =
                        match flush_chunk(&mut data_staging, &data_buf, t_d, false, data_raw) {
                            Ok(desc) => desc,
                            Err(e) => {
                                write_err = Err(e);
                                return;
                            }
                        };
                    let indices_desc =
                        match flush_chunk(&mut indices_staging, &idx_buf, t_i, true, false) {
                            Ok(desc) => desc,
                            Err(e) => {
                                write_err = Err(e);
                                return;
                            }
                        };
                    data_descs.push(data_desc);
                    indices_descs.push(indices_desc);
                    data_buf.clear();
                    idx_buf.clear();
                    chunk_elems = 0;
                }
                cum += 1;
            });
            write_err?;
            debug_assert!(cum >= before);
            indptr_local.push(cum);
        }
        nnz = cum as usize;

        if !data_buf.is_empty() {
            debug_assert_eq!(data_buf.len(), chunk_elems * t_d);
            debug_assert_eq!(idx_buf.len(), chunk_elems * t_i);
            data_descs.push(flush_chunk(
                &mut data_staging,
                &data_buf,
                t_d,
                false,
                data_raw,
            )?);
            indices_descs.push(flush_chunk(
                &mut indices_staging,
                &idx_buf,
                t_i,
                true,
                false,
            )?);
        } else if data_descs.is_empty() {
            data_descs.push(flush_chunk(&mut data_staging, &[], t_d, false, data_raw)?);
            indices_descs.push(flush_chunk(&mut indices_staging, &[], t_i, true, false)?);
        }

        data_staging.flush()?;
        indices_staging.flush()?;
    }

    {
        let mut indptr_staging = File::create(&indptr_staging_path)?;
        let indptr_bytes: Vec<u8> = indptr_local.iter().flat_map(|v| v.to_le_bytes()).collect();
        let indptr_chunk_bytes = CHUNK_ELEMS * 8;
        if indptr_bytes.is_empty() {
            indptr_descs.push(flush_chunk(&mut indptr_staging, &[], 8, true, false)?);
        } else {
            for slab in indptr_bytes.chunks(indptr_chunk_bytes) {
                indptr_descs.push(flush_chunk(&mut indptr_staging, slab, 8, true, false)?);
            }
        }
        indptr_staging.flush()?;
    }

    let n_rows = row_list.len();
    assemble_shx_streaming(
        &path,
        shard_idx,
        n_rows,
        n_vars,
        nnz,
        shuffle,
        [
            (
                data_dtype.name(),
                data_staging_path.as_path(),
                &mut data_descs,
            ),
            (
                indices_dtype.name(),
                indices_staging_path.as_path(),
                &mut indices_descs,
            ),
            ("int64", indptr_staging_path.as_path(), &mut indptr_descs),
        ],
    )?;
    // staging files are removed by `_staging_cleanup`'s Drop on scope exit.
    Ok(nnz)
}

/// Preflight: validate the FULL data + indices before writing any shard, in
/// parallel across `n_threads` scoped threads (read-only, no allocation). The
/// first failing chunk's NarrowError is returned (its min/max is chunk-local,
/// which is fine — the LossyCastError parity contract is the raise, not the
/// exact diagnostic values).
pub fn preflight<D: ToIntScan + Sync, I: ToIntScan + Sync>(
    data: &[D],
    indices: &[I],
    n_threads: usize,
) -> Result<(), (&'static str, NarrowError)> {
    scan_parallel(data, n_threads).map_err(|e| ("X.data", e))?;
    scan_parallel(indices, n_threads).map_err(|e| ("X.indices", e))?;
    Ok(())
}

/// Scan one array across `n_threads` chunks; returns the first chunk's error.
fn scan_parallel<T: ToIntScan + Sync>(arr: &[T], n_threads: usize) -> Result<(), NarrowError> {
    if arr.is_empty() {
        return Ok(());
    }
    let chunk = arr.len().div_ceil(n_threads.max(1));
    std::thread::scope(|s| {
        let handles: Vec<_> = arr
            .chunks(chunk)
            .map(|c| s.spawn(move || scan_u32(c)))
            .collect();
        for h in handles {
            h.join().unwrap()?;
        }
        Ok(())
    })
}

fn encode_all_over_source<S: RowSource + Sync>(
    source: &S,
    row_lists: &[&[i64]],
    n_vars: usize,
    n_threads: usize,
    out_dir: &Path,
    data_dtype: OnDiskDtype,
    indices_dtype: OnDiskDtype,
) -> io::Result<(Vec<usize>, OnDiskDtype, OnDiskDtype)> {
    let n_shards = row_lists.len();
    let nnz: Vec<AtomicUsize> = (0..n_shards).map(|_| AtomicUsize::new(0)).collect();
    let next = AtomicUsize::new(0);
    // On the first worker I/O error, signal the others to stop claiming new shards
    // (bounds wasted work to in-flight shards; cheap relaxed load on the hot path).
    let stop_flag = AtomicBool::new(false);
    let n_workers = n_threads.max(1).min(n_shards.max(1));
    std::thread::scope(|scope| -> io::Result<()> {
        let handles: Vec<_> = (0..n_workers)
            .map(|_| {
                let nnz = &nnz;
                let next = &next;
                let stop_flag = &stop_flag;
                scope.spawn(move || -> io::Result<()> {
                    loop {
                        if stop_flag.load(Ordering::Relaxed) {
                            break;
                        }
                        let s = next.fetch_add(1, Ordering::Relaxed);
                        if s >= n_shards {
                            break;
                        }
                        let got = match encode_one_shard(
                            s,
                            row_lists[s],
                            source,
                            n_vars,
                            out_dir,
                            data_dtype,
                            indices_dtype,
                        ) {
                            Ok(g) => g,
                            Err(e) => {
                                stop_flag.store(true, Ordering::Relaxed);
                                return Err(e);
                            }
                        };
                        nnz[s].store(got, Ordering::Relaxed);
                    }
                    Ok(())
                })
            })
            .collect();
        // join all (scope waits regardless); `.unwrap()` re-raises a worker PANIC
        // unchanged, `?` propagates the first io::Error as a clean error.
        for h in handles {
            h.join().unwrap()?;
        }
        Ok(())
    })?;
    Ok((
        nnz.iter().map(|a| a.load(Ordering::Relaxed)).collect(),
        data_dtype,
        indices_dtype,
    ))
}

/// Encode all shards with N scoped threads. Returns nnz per shard (index-ordered).
/// Caller must have run `preflight` first.
#[allow(clippy::too_many_arguments)]
pub fn encode_all_generic<D: ToOnDisk + Sync, I: ToOnDisk + Sync>(
    data: &[D],
    indices: &[I],
    indptr: &[i64],
    row_lists: &[&[i64]],
    n_vars: usize,
    n_threads: usize,
    out_dir: &Path,
    data_dtype: OnDiskDtype,
    indices_dtype: OnDiskDtype,
) -> io::Result<(Vec<usize>, OnDiskDtype, OnDiskDtype)> {
    let source = CsrRowSource {
        data,
        indices,
        indptr,
    };
    encode_all_over_source(
        &source,
        row_lists,
        n_vars,
        n_threads,
        out_dir,
        data_dtype,
        indices_dtype,
    )
}

#[allow(dead_code)]
pub fn encode_all_dense_generic<D: ToOnDisk + Sync>(
    dense: &[D],
    n_vars: usize,
    row_lists: &[&[i64]],
    n_threads: usize,
    out_dir: &Path,
    data_dtype: OnDiskDtype,
    indices_dtype: OnDiskDtype,
) -> io::Result<(Vec<usize>, OnDiskDtype, OnDiskDtype)> {
    let source = DenseRowSource::new(dense, n_vars);
    encode_all_over_source(
        &source,
        row_lists,
        n_vars,
        n_threads,
        out_dir,
        data_dtype,
        indices_dtype,
    )
}

/// Encode all shards with N scoped threads. Returns nnz per shard (index-ordered).
/// Caller must have run `preflight` first.
pub fn encode_all<D: ToOnDisk + Sync, I: ToOnDisk + Sync>(
    data: &[D],
    indices: &[I],
    indptr: &[i64],
    row_lists: &[&[i64]],
    n_vars: usize,
    n_threads: usize,
    out_dir: &Path,
) -> io::Result<Vec<usize>> {
    Ok(encode_all_generic(
        data,
        indices,
        indptr,
        row_lists,
        n_vars,
        n_threads,
        out_dir,
        OnDiskDtype::U32,
        OnDiskDtype::U32,
    )?
    .0)
}

/// Index carrier the intermediate CSR uses: u16 when n_vars<=65536, else u32.
/// u32 (not i32): column indices are non-negative and reach n_vars-1 <= u32::MAX.
pub trait IndexCarrier: Copy + Send + Sync + Default {
    fn from_col(col: usize) -> Self;
}
impl IndexCarrier for u16 {
    #[inline]
    fn from_col(col: usize) -> Self {
        col as u16
    }
}
impl IndexCarrier for u32 {
    #[inline]
    fn from_col(col: usize) -> Self {
        col as u32
    }
}

/// Convert a C-contiguous row-major dense matrix to CSR (source row order),
/// two-pass and parallel over disjoint row blocks. Pass 1 counts per-row nnz
/// directly into indptr[1..] (-> prefix sum in place); pass 2 fills preallocated
/// data/indices in their disjoint output slices. Zeros are dropped via
/// D::is_zero (matching DenseRowSource). No concatenation copy.
pub fn dense_to_csr_core<D, I>(
    dense: &[D],
    n_vars: usize,
    n_threads: usize,
) -> (Vec<D>, Vec<I>, Vec<i64>)
where
    D: ToOnDisk + Default + Sync + Send,
    I: IndexCarrier,
{
    if n_vars > 0 {
        debug_assert_eq!(
            dense.len() % n_vars,
            0,
            "dense length must be a multiple of n_vars"
        );
    }
    // checked_div yields None when n_vars == 0 -> n_rows 0 (also covers empty dense).
    let n_rows = dense.len().checked_div(n_vars).unwrap_or(0);
    if n_rows == 0 {
        return (Vec::new(), Vec::new(), vec![0i64]);
    }

    // Row-block partition (contiguous row ranges covering [0, n_rows)); n_rows >= 1 here.
    let n_blocks = n_threads.max(1).min(n_rows);
    let block = n_rows.div_ceil(n_blocks);
    let blocks: Vec<(usize, usize)> = (0..n_rows)
        .step_by(block)
        .map(|r0| (r0, (r0 + block).min(n_rows)))
        .collect();

    // Pass 1: per-row nnz written directly into indptr[1..], parallel over
    // disjoint slices (no separate counts buffer).
    let mut indptr = vec![0i64; n_rows + 1];
    {
        let mut rest: &mut [i64] = &mut indptr[1..];
        let mut subs: Vec<(&mut [i64], usize)> = Vec::with_capacity(blocks.len());
        for &(r0, r1) in &blocks {
            let (head, tail) = rest.split_at_mut(r1 - r0);
            rest = tail;
            subs.push((head, r0));
        }
        std::thread::scope(|s| {
            for (csub, r0) in subs {
                s.spawn(move || {
                    // chunks_exact over the contiguous block: no per-element bounds check.
                    let row_slice = &dense[r0 * n_vars..(r0 + csub.len()) * n_vars];
                    for (c, row) in csub.iter_mut().zip(row_slice.chunks_exact(n_vars)) {
                        let mut cnt = 0i64;
                        for &v in row {
                            if !v.is_zero() {
                                cnt += 1;
                            }
                        }
                        *c = cnt;
                    }
                });
            }
        });
    }

    // Prefix sum in-place: each entry (from index 1) holds its row's nnz; indptr[0]=0.
    let mut acc = 0i64;
    for c in indptr.iter_mut().skip(1) {
        acc += *c;
        *c = acc;
    }
    let total_nnz = acc as usize;

    // Pass 2: fill preallocated data/indices in disjoint, indptr-aligned slices.
    let mut data: Vec<D> = vec![D::default(); total_nnz];
    let mut indices: Vec<I> = vec![I::default(); total_nnz];
    {
        let mut d_rest: &mut [D] = &mut data[..];
        let mut i_rest: &mut [I] = &mut indices[..];
        let mut subs: Vec<(&mut [D], &mut [I], usize, usize)> = Vec::with_capacity(blocks.len());
        for &(r0, r1) in &blocks {
            let len = (indptr[r1] - indptr[r0]) as usize;
            let (d_head, d_tail) = d_rest.split_at_mut(len);
            let (i_head, i_tail) = i_rest.split_at_mut(len);
            d_rest = d_tail;
            i_rest = i_tail;
            subs.push((d_head, i_head, r0, r1));
        }
        std::thread::scope(|s| {
            for (d_sub, i_sub, r0, r1) in subs {
                s.spawn(move || {
                    let mut k = 0usize;
                    // chunks_exact over the contiguous block: no per-element bounds check.
                    let row_slice = &dense[r0 * n_vars..r1 * n_vars];
                    for row in row_slice.chunks_exact(n_vars) {
                        for (col, &v) in row.iter().enumerate() {
                            if !v.is_zero() {
                                d_sub[k] = v;
                                i_sub[k] = I::from_col(col);
                                k += 1;
                            }
                        }
                    }
                    debug_assert_eq!(k, d_sub.len());
                });
            }
        });
    }

    (data, indices, indptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_rowsource_matches_csr_decomposition() {
        // 2x3 dense, row-major. row0 = [0, 2.5, 0]; row1 = [NaN, 0, -0.0]
        let n_vars = 3usize;
        let dense: Vec<f64> = vec![0.0, 2.5, 0.0, f64::NAN, 0.0, -0.0];
        let src = DenseRowSource::new(&dense, n_vars);
        let mut out: Vec<(f64, i64)> = Vec::new();
        src.for_each_in_row(0, |v, c| out.push((v, c)));
        assert_eq!(out, vec![(2.5, 1)]);
        out.clear();
        src.for_each_in_row(1, |v, c| out.push((v, c)));
        assert_eq!(out.len(), 1);
        assert!(out[0].0.is_nan() && out[0].1 == 0);
    }

    #[test]
    fn dense_u16_byte_identical_to_csr_u16() {
        // Integral, non-negative dense -> u16 path must match scipy-style CSR decomposition.
        let n_vars = 4usize;
        let dense: Vec<f32> = vec![
            0.0, 3.0, 0.0, 5.0, //
            0.0, 0.0, 0.0, 0.0, //
            7.0, 0.0, 2.0, 0.0,
        ];
        let csr_data: Vec<f32> = vec![3.0, 5.0, 7.0, 2.0];
        let csr_idx: Vec<i32> = vec![1, 3, 0, 2];
        let csr_indptr: Vec<i64> = vec![0, 2, 2, 4];
        let rows: Vec<i64> = vec![2, 0, 1];
        let lists: Vec<&[i64]> = vec![&rows];

        let d_dir = std::env::temp_dir().join("rs_dense");
        let c_dir = std::env::temp_dir().join("rs_csr");
        let _ = std::fs::create_dir_all(&d_dir);
        let _ = std::fs::create_dir_all(&c_dir);
        let (dn, _, _) = encode_all_dense_generic(
            &dense,
            n_vars,
            &lists,
            2,
            &d_dir,
            OnDiskDtype::U16,
            OnDiskDtype::U16,
        )
        .unwrap();
        let (cn, _, _) = encode_all_generic(
            &csr_data,
            &csr_idx,
            &csr_indptr,
            &lists,
            n_vars,
            2,
            &c_dir,
            OnDiskDtype::U16,
            OnDiskDtype::U16,
        )
        .unwrap();
        assert_eq!(dn, cn);
        let db = std::fs::read(d_dir.join("shard_0000.shx")).unwrap();
        let cb = std::fs::read(c_dir.join("shard_0000.shx")).unwrap();
        assert_eq!(
            db, cb,
            "dense->u16 shard bytes must equal csr->u16 shard bytes"
        );
    }

    #[test]
    fn encodes_two_shards_from_scattered_rows() {
        // 4-row CSR, 2 cols. rows: r0=[5@0], r1=[7@1], r2=[9@0,1@1], r3=[]
        let data: Vec<f32> = vec![5.0, 7.0, 9.0, 1.0];
        let indices: Vec<i32> = vec![0, 1, 0, 1];
        let indptr: Vec<i64> = vec![0, 1, 2, 4, 4];
        let s0: Vec<i64> = vec![2, 0]; // grouped, scattered
        let s1: Vec<i64> = vec![1, 3];
        let lists: Vec<&[i64]> = vec![&s0, &s1];
        let dir = std::env::temp_dir().join("enc_core_test");
        let _ = std::fs::create_dir_all(&dir);
        preflight(&data, &indices, 2).unwrap();
        let nnz = encode_all(&data, &indices, &indptr, &lists, 2, 2, &dir).unwrap();
        assert_eq!(nnz, vec![3, 1]); // shard0 = rows 2,0 → 2+1 nnz; shard1 = rows 1,3 → 1+0
        assert!(dir.join("shard_0000.shx").exists());
        assert!(dir.join("shard_0001.shx").exists());
        let bytes = std::fs::read(dir.join("shard_0000.shx")).unwrap();
        let hsize = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
        let hdr = std::str::from_utf8(&bytes[10..10 + hsize]).unwrap();
        assert!(hdr.contains("\"dtype\":\"uint32\""));
    }

    #[test]
    fn preflight_rejects_lossy_data() {
        let data: Vec<f32> = vec![1.0, 2.5];
        let indices: Vec<i32> = vec![0, 1];
        assert!(preflight(&data, &indices, 2).is_err());
    }

    #[test]
    fn u16_path_is_byte_identical_to_legacy() {
        let data: Vec<f32> = vec![5.0, 7.0, 9.0, 1.0];
        let indices: Vec<i32> = vec![0, 1, 0, 1];
        let indptr: Vec<i64> = vec![0, 1, 2, 4, 4];
        let rows: Vec<i64> = vec![0, 1, 2, 3];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("enc_u16_ident");
        let _ = std::fs::create_dir_all(&dir);
        let (nnz, ddt, idt) = encode_all_generic(
            &data,
            &indices,
            &indptr,
            &lists,
            2,
            2,
            &dir,
            OnDiskDtype::U16,
            OnDiskDtype::U16,
        )
        .unwrap();
        assert_eq!(nnz, vec![4]);
        assert_eq!((ddt, idt), (OnDiskDtype::U16, OnDiskDtype::U16));
        let bytes = std::fs::read(dir.join("shard_0000.shx")).unwrap();
        let hsize = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
        let hdr = std::str::from_utf8(&bytes[10..10 + hsize]).unwrap();
        assert!(hdr.contains("\"dtype\":\"uint16\""));
    }

    #[test]
    fn float_data_uses_rawdata_marker() {
        // #142: a float on-disk data dtype -> the shard self-describes with the
        // rawdata marker (data stored raw, no shuffle/delta).
        let data: Vec<f32> = vec![1.5, 2.5, 1.5, 2.5];
        let indices: Vec<i32> = vec![0, 1, 2, 3];
        let indptr: Vec<i64> = vec![0, 4];
        let rows: Vec<i64> = vec![0];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("enc_f32_rawmarker");
        let _ = std::fs::create_dir_all(&dir);
        let (_nnz, ddt, _idt) = encode_all_generic(
            &data,
            &indices,
            &indptr,
            &lists,
            4,
            2,
            &dir,
            OnDiskDtype::F32,
            OnDiskDtype::U16,
        )
        .unwrap();
        assert_eq!(ddt, OnDiskDtype::F32);
        let bytes = std::fs::read(dir.join("shard_0000.shx")).unwrap();
        let hsize = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
        let hdr = std::str::from_utf8(&bytes[10..10 + hsize]).unwrap();
        assert!(
            hdr.contains("\"shuffle\":\"byteshuffle-bytedelta-rawdata\""),
            "float archive must carry the rawdata marker; hdr: {hdr}"
        );
    }

    #[test]
    fn integer_data_keeps_shuffle_marker() {
        // u16 on-disk data dtype -> the old byteshuffle-bytedelta marker (unchanged).
        let data: Vec<f32> = vec![5.0, 7.0, 9.0, 1.0];
        let indices: Vec<i32> = vec![0, 1, 0, 1];
        let indptr: Vec<i64> = vec![0, 1, 2, 4, 4];
        let rows: Vec<i64> = vec![0, 1, 2, 3];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("enc_u16_keepmarker");
        let _ = std::fs::create_dir_all(&dir);
        encode_all_generic(
            &data,
            &indices,
            &indptr,
            &lists,
            2,
            2,
            &dir,
            OnDiskDtype::U16,
            OnDiskDtype::U16,
        )
        .unwrap();
        let bytes = std::fs::read(dir.join("shard_0000.shx")).unwrap();
        let hsize = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
        let hdr = std::str::from_utf8(&bytes[10..10 + hsize]).unwrap();
        assert!(
            hdr.contains("\"shuffle\":\"byteshuffle-bytedelta\""),
            "hdr: {hdr}"
        );
        assert!(
            !hdr.contains("rawdata"),
            "integer archive must not use the rawdata marker"
        );
    }

    #[test]
    fn float_data_emits_float_stream() {
        let data: Vec<f64> = vec![1.5, -2.0, 3.25];
        let indices: Vec<i32> = vec![0, 1, 2];
        let indptr: Vec<i64> = vec![0, 3];
        let rows: Vec<i64> = vec![0];
        let lists: Vec<&[i64]> = vec![&rows];
        let dir = std::env::temp_dir().join("enc_f64");
        let _ = std::fs::create_dir_all(&dir);
        let (_nnz, ddt, _idt) = encode_all_generic(
            &data,
            &indices,
            &indptr,
            &lists,
            3,
            2,
            &dir,
            OnDiskDtype::F64,
            OnDiskDtype::U16,
        )
        .unwrap();
        assert_eq!(ddt, OnDiskDtype::F64);
        let bytes = std::fs::read(dir.join("shard_0000.shx")).unwrap();
        let hsize = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
        let hdr = std::str::from_utf8(&bytes[10..10 + hsize]).unwrap();
        assert!(hdr.contains("\"dtype\":\"float64\""));
    }

    #[test]
    fn dense_to_csr_core_matches_decomposition() {
        // 3x4 dense, row-major. row0=[0,3,0,5] row1=[0,0,0,0] row2=[7,0,2,0]
        let n_vars = 4usize;
        let dense: Vec<f32> = vec![
            0.0, 3.0, 0.0, 5.0, //
            0.0, 0.0, 0.0, 0.0, //
            7.0, 0.0, 2.0, 0.0,
        ];
        let (data, indices, indptr) = dense_to_csr_core::<f32, u16>(&dense, n_vars, 2);
        assert_eq!(data, vec![3.0, 5.0, 7.0, 2.0]);
        assert_eq!(indices, vec![1u16, 3, 0, 2]);
        assert_eq!(indptr, vec![0i64, 2, 2, 4]);
    }

    #[test]
    fn dense_to_csr_core_single_eq_multi_thread() {
        let n_vars = 5usize;
        let dense: Vec<f64> = (0..40)
            .map(|i| if i % 3 == 0 { 0.0 } else { (i % 7) as f64 })
            .collect();
        let a = dense_to_csr_core::<f64, u32>(&dense, n_vars, 1);
        let b = dense_to_csr_core::<f64, u32>(&dense, n_vars, 4);
        assert_eq!(a, b);
    }

    #[test]
    fn dense_to_csr_core_all_zero_and_empty() {
        // all-zero rows -> empty data/indices, indptr all zeros
        let (d, i, p) = dense_to_csr_core::<u16, u16>(&[0u16, 0, 0, 0], 2, 2);
        assert!(d.is_empty() && i.is_empty());
        assert_eq!(p, vec![0i64, 0, 0]);
        // zero rows (n_vars==0 or empty dense) -> just [0]
        let (d2, i2, p2) = dense_to_csr_core::<f32, u16>(&[], 0, 2);
        assert!(d2.is_empty() && i2.is_empty());
        assert_eq!(p2, vec![0i64]);
    }

    #[test]
    fn dense_via_csr_byte_identical_to_direct() {
        // Integral non-negative dense -> u16 on-disk. Scattered grouped row list.
        let n_vars = 4usize;
        let dense: Vec<f32> = vec![
            0.0, 3.0, 0.0, 5.0, //
            0.0, 0.0, 0.0, 0.0, //
            7.0, 0.0, 2.0, 0.0,
        ];
        let rows: Vec<i64> = vec![2, 0, 1]; // grouped, scattered
        let lists: Vec<&[i64]> = vec![&rows];

        let direct = std::env::temp_dir().join("via_csr_direct");
        let u16carrier = std::env::temp_dir().join("via_csr_u16");
        let u32carrier = std::env::temp_dir().join("via_csr_u32");
        for d in [&direct, &u16carrier, &u32carrier] {
            let _ = std::fs::create_dir_all(d);
        }

        // Direct dense path (the baseline).
        encode_all_dense_generic(
            &dense,
            n_vars,
            &lists,
            2,
            &direct,
            OnDiskDtype::U16,
            OnDiskDtype::U16,
        )
        .unwrap();
        // Via-CSR with a u16 index carrier.
        let (du, iu, pu) = dense_to_csr_core::<f32, u16>(&dense, n_vars, 2);
        encode_all_generic(
            &du,
            &iu,
            &pu,
            &lists,
            n_vars,
            2,
            &u16carrier,
            OnDiskDtype::U16,
            OnDiskDtype::U16,
        )
        .unwrap();
        // Via-CSR with a u32 index carrier (same on-disk dtype).
        let (di, ii, pi) = dense_to_csr_core::<f32, u32>(&dense, n_vars, 2);
        encode_all_generic(
            &di,
            &ii,
            &pi,
            &lists,
            n_vars,
            2,
            &u32carrier,
            OnDiskDtype::U16,
            OnDiskDtype::U16,
        )
        .unwrap();

        let b_direct = std::fs::read(direct.join("shard_0000.shx")).unwrap();
        let b_u16 = std::fs::read(u16carrier.join("shard_0000.shx")).unwrap();
        let b_u32 = std::fs::read(u32carrier.join("shard_0000.shx")).unwrap();
        assert_eq!(
            b_u16, b_direct,
            "via-CSR (u16 carrier) must equal direct dense bytes"
        );
        assert_eq!(
            b_u32, b_direct,
            "via-CSR (u32 carrier) must equal direct dense bytes"
        );
    }
}
