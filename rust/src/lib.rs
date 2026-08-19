// pyo3 0.22's #[pyfunction] macro expansion trips clippy::useless_conversion on
// the generated return-path conversion (a false positive). A function-level allow
// does not reach the macro-expanded span, so it must be suppressed at module scope.
#![allow(clippy::useless_conversion)]

pub mod cell;
pub mod cell_zstd;
pub mod decode;
pub mod encode;
pub mod narrow;
pub mod pfor;
mod profile;
mod shx;
mod shx_read;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::path::Path;
use std::path::PathBuf;

use encode::{dense_to_csr_core, encode_all_dense_generic, encode_all_generic};
use narrow::{decide_data_dtypes, OnDiskDtype, SrcKind};
use profile::{profile_single_shard, FeedMode};

/// Smoke probe so Python can confirm the extension built + imports.
#[pyfunction]
fn _selftest() -> &'static str {
    "ok"
}

/// Widen an i4/i8 numpy indptr array to Vec<i64> (small: n_obs+1, not nnz-scaled).
pub(crate) fn indptr_to_i64(arr: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let nc = || PyValueError::new_err("indptr array must be contiguous");
    if let Ok(a) = arr.extract::<PyReadonlyArray1<i64>>() {
        return Ok(a.as_slice().map_err(|_| nc())?.to_vec());
    }
    if let Ok(a) = arr.extract::<PyReadonlyArray1<i32>>() {
        return Ok(a
            .as_slice()
            .map_err(|_| nc())?
            .iter()
            .map(|&v| v as i64)
            .collect());
    }
    Err(PyTypeError::new_err("indptr must be int32 or int64"))
}

fn decode_lossy_err(e: &narrow::NarrowError) -> PyErr {
    // Tier-3 read cast failed a per-element lossless round-trip. Carries the word
    // "lossy" so the Python dispatch shim re-raises it as cellstream.errors.LossyCastError.
    PyValueError::new_err(format!(
        "X: lossy cast on read -- a value does not fit the requested dtype \
         (offending value approx {}; {}). Pass allow_lossy=True to permit it.",
        e.min, e.reason
    ))
}

/// Map a decode-path `DecodeError` (#133) to a typed Python exception.
pub(crate) fn decode_err_to_py(e: decode::DecodeError) -> PyErr {
    use decode::DecodeError;
    match e {
        // PyO3 auto-maps std::io::Error -> the OSError family (FileNotFoundError, etc.).
        DecodeError::Io(io) => PyErr::from(io),
        // Plain ValueError. The Python shim re-raises any ValueError whose text contains
        // the (case-sensitive, lowercase) substring "lossy" as LossyCastError; our own
        // Corrupt format strings never contain it, but the From<io::Error> path echoes
        // serde/io messages, so neutralize a stray "lossy" defensively (Gemini PR #139).
        DecodeError::Corrupt(m) => {
            let safe = if m.contains("lossy") {
                m.replace("lossy", "invalid")
            } else {
                m
            };
            PyValueError::new_err(safe)
        }
        // Unchanged tier-3 lossy text -> LossyCastError on the Python side.
        DecodeError::Lossy(ne) => decode_lossy_err(&ne),
    }
}

fn validate_encode_options(level: i32, chunk_size: usize, shuffle: &str) -> PyResult<()> {
    if level != 1 {
        return Err(PyValueError::new_err(
            "rust core: only zstd compression level 1 is supported",
        ));
    }
    if chunk_size != 1_048_576 {
        return Err(PyValueError::new_err(
            "rust core: chunk_size is pinned to 1048576 elements",
        ));
    }
    if shuffle != "byteshuffle-bytedelta" {
        return Err(PyValueError::new_err(
            "rust core supports only the byte-filter codec",
        ));
    }
    Ok(())
}

/// Validate that every row index in every shard's row list is in `[0, n_rows)`.
/// Belt-and-suspenders for the internal encode API (the Python writer always
/// passes valid `np.arange`/permutation row lists). Raises `ValueError` instead
/// of letting an out-of-range index reach an unchecked `indptr[r]` (panic).
fn check_row_index_bounds(row_lists: &[&[i64]], n_rows: usize) -> PyResult<()> {
    for rl in row_lists {
        for &r in *rl {
            // Compare in u64 so a large i64 index can't truncate to usize on 32-bit targets.
            if r < 0 || (r as u64) >= (n_rows as u64) {
                return Err(PyValueError::new_err(format!(
                    "row index {r} out of range for {n_rows} rows"
                )));
            }
        }
    }
    Ok(())
}

/// Validate an in-memory CSR's structural invariants before encoding so a
/// malformed indptr can't panic across the FFI (ultrareview L6).
pub(crate) fn validate_csr_structure(
    indptr: &[i64],
    data_len: usize,
    indices_len: usize,
) -> PyResult<()> {
    if data_len != indices_len {
        return Err(PyValueError::new_err(format!(
            "CSR data length {data_len} != indices length {indices_len}"
        )));
    }
    match indptr.first() {
        Some(&0) => {}
        _ => return Err(PyValueError::new_err("CSR indptr must start at 0")),
    }
    if !indptr.windows(2).all(|w| w[0] <= w[1]) {
        return Err(PyValueError::new_err("CSR indptr must be non-decreasing"));
    }
    let last = *indptr.last().unwrap();
    // Compare as u64, not `as usize`: on a 32-bit platform `indptr[-1] > u32::MAX`
    // would truncate and silently bypass the bound. (Gemini PR #169 r1.)
    if last < 0 || last as u64 > data_len as u64 {
        return Err(PyValueError::new_err(format!(
            "CSR indptr[-1] {last} exceeds data length {data_len}"
        )));
    }
    Ok(())
}

fn default_index_dtype(n_vars: usize) -> PyResult<OnDiskDtype> {
    if n_vars as u64 <= (u32::MAX as u64) + 1 {
        Ok(OnDiskDtype::U32)
    } else {
        Err(PyValueError::new_err("n_vars too large for uint32 indices"))
    }
}

#[pyfunction]
#[pyo3(signature = (data, indices, indptr, shard_row_lists, out_dir, *,
                    n_vars, shuffle, level=1, chunk_size=1048576, n_threads=8,
                    data_dtype=None, index_dtype=None))]
#[allow(clippy::too_many_arguments)]
fn encode_shards_inmem(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    shard_row_lists: &Bound<'_, PyList>,
    out_dir: &str,
    n_vars: usize,
    shuffle: &str,
    level: i32,
    chunk_size: usize,
    n_threads: usize,
    data_dtype: Option<&str>,
    index_dtype: Option<&str>,
) -> PyResult<(Vec<usize>, String, String)> {
    validate_encode_options(level, chunk_size, shuffle)?;
    let out = Path::new(out_dir);
    let indptr_i64 = indptr_to_i64(indptr)?;
    validate_csr_structure(&indptr_i64, data.len()?, indices.len()?)?;

    // Borrow each shard's row list as &[i64]; keep guards alive on this thread.
    let row_guards: Vec<PyReadonlyArray1<i64>> = shard_row_lists
        .iter()
        .map(|o| o.extract())
        .collect::<PyResult<_>>()?;
    let row_lists: Vec<&[i64]> = row_guards
        .iter()
        .map(|g| {
            g.as_slice()
                .map_err(|_| PyValueError::new_err("row list array must be contiguous"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    // #111: reject out-of-range internal row indices with a clean ValueError.
    check_row_index_bounds(&row_lists, indptr_i64.len().saturating_sub(1))?;

    // Dispatch on (data dtype, index dtype). The borrow guards (df/idf) are held
    // on this thread; the &[T] slices are Sync and shared across scoped threads.
    macro_rules! run {
        ($d:ty, $i:ty, $src_kind:expr) => {{
            let df = data.extract::<PyReadonlyArray1<$d>>()?;
            let idf = indices.extract::<PyReadonlyArray1<$i>>()?;
            let ds = df
                .as_slice()
                .map_err(|_| PyValueError::new_err("data array must be contiguous"))?;
            let is = idf
                .as_slice()
                .map_err(|_| PyValueError::new_err("indices array must be contiguous"))?;
            let ddt = match data_dtype {
                Some(n) => OnDiskDtype::from_name(n)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown data_dtype '{n}'")))?,
                None => decide_data_dtypes(ds, $src_kind, n_threads).0,
            };
            let idt = match index_dtype {
                Some(n) => OnDiskDtype::from_name(n)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown index_dtype '{n}'")))?,
                None => default_index_dtype(n_vars)?,
            };
            let encoded = py.allow_threads(|| {
                encode_all_generic(
                    ds,
                    is,
                    &indptr_i64,
                    &row_lists,
                    n_vars,
                    n_threads,
                    out,
                    ddt,
                    idt,
                )
            });
            let nnz = encoded?.0;
            Ok((nnz, ddt.name().to_string(), idt.name().to_string()))
        }};
    }

    let data_dtype = data.getattr("dtype")?.str()?.to_string();
    let indices_dtype = indices.getattr("dtype")?.str()?.to_string();
    match (data_dtype.as_str(), indices_dtype.as_str()) {
        ("uint16", "int32") => run!(u16, i32, SrcKind::U16),
        ("uint16", "int64") => run!(u16, i64, SrcKind::U16),
        ("uint32", "int32") => run!(u32, i32, SrcKind::U32),
        ("uint32", "int64") => run!(u32, i64, SrcKind::U32),
        ("float32", "int32") => run!(f32, i32, SrcKind::F32),
        ("float32", "int64") => run!(f32, i64, SrcKind::F32),
        ("float64", "int32") => run!(f64, i32, SrcKind::F64),
        ("float64", "int64") => run!(f64, i64, SrcKind::F64),
        _ => {
            // Unreachable in practice: the Python dispatch pre-filters unsupported
            // dtypes to the fallback (see _rust_dispatch.use_rust_for). Belt-and-suspenders.
            Err(PyTypeError::new_err(format!(
                "rust core: unsupported dtype combination data='{}' indices='{}'",
                data_dtype, indices_dtype
            )))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (dense, shard_row_lists, out_dir, *,
                    n_vars, shuffle, level=1, chunk_size=1048576, n_threads=8,
                    data_dtype=None))]
#[allow(clippy::too_many_arguments)]
fn encode_shards_inmem_dense(
    py: Python<'_>,
    dense: &Bound<'_, PyAny>,
    shard_row_lists: &Bound<'_, PyList>,
    out_dir: &str,
    n_vars: usize,
    shuffle: &str,
    level: i32,
    chunk_size: usize,
    n_threads: usize,
    data_dtype: Option<&str>,
) -> PyResult<(Vec<usize>, String, String)> {
    validate_encode_options(level, chunk_size, shuffle)?;
    let out = Path::new(out_dir);

    let row_guards: Vec<PyReadonlyArray1<i64>> = shard_row_lists
        .iter()
        .map(|o| o.extract())
        .collect::<PyResult<_>>()?;
    let row_lists: Vec<&[i64]> = row_guards
        .iter()
        .map(|g| {
            g.as_slice()
                .map_err(|_| PyValueError::new_err("row list array must be contiguous"))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let idt = default_index_dtype(n_vars)?;

    macro_rules! run_dense {
        ($d:ty, $src_kind:expr) => {{
            let arr = dense.extract::<PyReadonlyArray2<$d>>()?;
            // #111: bounds-check before building/encoding; n_rows = dense row count.
            let n_rows = arr.shape()[0];
            check_row_index_bounds(&row_lists, n_rows)?;
            let s = arr
                .as_slice()
                .map_err(|_| PyValueError::new_err("dense array must be C-contiguous"))?;
            let ddt = match data_dtype {
                Some(n) => OnDiskDtype::from_name(n)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown data_dtype '{n}'")))?,
                None => decide_data_dtypes(s, $src_kind, n_threads).0,
            };
            let encoded = py.allow_threads(|| {
                encode_all_dense_generic(s, n_vars, &row_lists, n_threads, out, ddt, idt)
            });
            let nnz = encoded?.0;
            Ok((nnz, ddt.name().to_string(), idt.name().to_string()))
        }};
    }

    let data_dtype = dense.getattr("dtype")?.str()?.to_string();
    match data_dtype.as_str() {
        "uint16" => run_dense!(u16, SrcKind::U16),
        "uint32" => run_dense!(u32, SrcKind::U32),
        "float32" => run_dense!(f32, SrcKind::F32),
        "float64" => run_dense!(f64, SrcKind::F64),
        _ => Err(PyTypeError::new_err(format!(
            "rust core: unsupported dense dtype data='{}'",
            data_dtype
        ))),
    }
}

/// Fast grouped-dense write: convert the dense matrix to CSR in-process (parallel,
/// two-pass) and encode via the shared CSR core (encode_all_generic). Avoids the
/// scattered dense[r,:] reads of the grouped direct encoder. Same call shape and
/// return as encode_shards_inmem_dense. The intermediate CSR uses a u16 index
/// carrier when n_vars <= 65536, else u32 (column indices are non-negative and
/// reach n_vars-1 <= u32::MAX); the on-disk index dtype is always uint32.
#[pyfunction]
#[pyo3(signature = (dense, shard_row_lists, out_dir, *,
                    n_vars, shuffle, level=1, chunk_size=1048576, n_threads=8,
                    data_dtype=None))]
#[allow(clippy::too_many_arguments)]
fn encode_shards_inmem_dense_via_csr(
    py: Python<'_>,
    dense: &Bound<'_, PyAny>,
    shard_row_lists: &Bound<'_, PyList>,
    out_dir: &str,
    n_vars: usize,
    shuffle: &str,
    level: i32,
    chunk_size: usize,
    n_threads: usize,
    data_dtype: Option<&str>,
) -> PyResult<(Vec<usize>, String, String)> {
    validate_encode_options(level, chunk_size, shuffle)?;
    let out = Path::new(out_dir);

    let row_guards: Vec<PyReadonlyArray1<i64>> = shard_row_lists
        .iter()
        .map(|o| o.extract())
        .collect::<PyResult<_>>()?;
    let row_lists: Vec<&[i64]> = row_guards
        .iter()
        .map(|g| {
            g.as_slice()
                .map_err(|_| PyValueError::new_err("row list array must be contiguous"))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let idt = default_index_dtype(n_vars)?;

    macro_rules! run_via_csr {
        ($d:ty, $src_kind:expr) => {{
            let arr = dense.extract::<PyReadonlyArray2<$d>>()?;
            // #111: bounds-check before building/encoding; n_rows = dense row count.
            let n_rows = arr.shape()[0];
            check_row_index_bounds(&row_lists, n_rows)?;
            let s = arr
                .as_slice()
                .map_err(|_| PyValueError::new_err("dense array must be C-contiguous"))?;
            let ddt = match data_dtype {
                Some(n) => OnDiskDtype::from_name(n)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown data_dtype '{n}'")))?,
                None => decide_data_dtypes(s, $src_kind, n_threads).0,
            };
            let encoded = py.allow_threads(|| {
                if n_vars <= 65_536 {
                    let (data, indices, indptr) =
                        dense_to_csr_core::<$d, u16>(s, n_vars, n_threads);
                    encode_all_generic(
                        &data, &indices, &indptr, &row_lists, n_vars, n_threads, out, ddt, idt,
                    )
                } else {
                    let (data, indices, indptr) =
                        dense_to_csr_core::<$d, u32>(s, n_vars, n_threads);
                    encode_all_generic(
                        &data, &indices, &indptr, &row_lists, n_vars, n_threads, out, ddt, idt,
                    )
                }
            });
            let nnz = encoded?.0;
            Ok((nnz, ddt.name().to_string(), idt.name().to_string()))
        }};
    }

    let dense_dtype = dense.getattr("dtype")?.str()?.to_string();
    match dense_dtype.as_str() {
        "uint16" => run_via_csr!(u16, SrcKind::U16),
        "uint32" => run_via_csr!(u32, SrcKind::U32),
        "float32" => run_via_csr!(f32, SrcKind::F32),
        "float64" => run_via_csr!(f64, SrcKind::F64),
        _ => Err(PyTypeError::new_err(format!(
            "rust core: unsupported dense dtype data='{}'",
            dense_dtype
        ))),
    }
}

/// Decide the on-disk data dtype and minimum losslessly-representable data dtype
/// for a CSR source WITHOUT encoding.
#[pyfunction]
#[pyo3(signature = (data, *, n_threads=8))]
fn decide_dtypes(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    n_threads: usize,
) -> PyResult<(String, String)> {
    macro_rules! decide {
        ($d:ty, $src_kind:expr) => {{
            let df = data.extract::<PyReadonlyArray1<$d>>()?;
            let ds = df
                .as_slice()
                .map_err(|_| PyValueError::new_err("data array must be contiguous"))?;
            let (ddt, min_name) = py.allow_threads(|| decide_data_dtypes(ds, $src_kind, n_threads));
            Ok((ddt.name().to_string(), min_name.to_string()))
        }};
    }

    let data_dtype = data.getattr("dtype")?.str()?.to_string();
    match data_dtype.as_str() {
        "uint16" => decide!(u16, SrcKind::U16),
        "uint32" => decide!(u32, SrcKind::U32),
        "float32" => decide!(f32, SrcKind::F32),
        "float64" => decide!(f64, SrcKind::F64),
        _ => Err(PyTypeError::new_err(format!(
            "rust core: unsupported data dtype '{data_dtype}'"
        ))),
    }
}

/// Decide the on-disk data dtype and minimum losslessly-representable data dtype
/// for a DENSE source WITHOUT encoding.
#[pyfunction]
#[pyo3(signature = (dense, *, n_threads=8))]
fn decide_dtypes_dense(
    py: Python<'_>,
    dense: &Bound<'_, PyAny>,
    n_threads: usize,
) -> PyResult<(String, String)> {
    macro_rules! decide_dense {
        ($d:ty, $src_kind:expr) => {{
            let arr = dense.extract::<PyReadonlyArray2<$d>>()?;
            let s = arr
                .as_slice()
                .map_err(|_| PyValueError::new_err("dense array must be C-contiguous"))?;
            let (ddt, min_name) = py.allow_threads(|| decide_data_dtypes(s, $src_kind, n_threads));
            Ok((ddt.name().to_string(), min_name.to_string()))
        }};
    }

    let data_dtype = dense.getattr("dtype")?.str()?.to_string();
    match data_dtype.as_str() {
        "uint16" => decide_dense!(u16, SrcKind::U16),
        "uint32" => decide_dense!(u32, SrcKind::U32),
        "float32" => decide_dense!(f32, SrcKind::F32),
        "float64" => decide_dense!(f64, SrcKind::F64),
        _ => Err(PyTypeError::new_err(format!(
            "rust core: unsupported dense dtype data='{data_dtype}'"
        ))),
    }
}

/// Internal profiling entry point (NOT public API). Times a single-thread
/// encode of a synthetic CSR and returns per-bucket seconds. `mode` is
/// "scalar" (current production feed) or "bulk" (Phase-2 vectorized feed).
#[pyfunction]
#[pyo3(signature = (*, nnz, nnz_per_row, mode, out_dir))]
fn _profile_encode_single_shard(
    py: Python<'_>,
    nnz: usize,
    nnz_per_row: usize,
    mode: &str,
    out_dir: &str,
) -> PyResult<std::collections::HashMap<String, f64>> {
    let feed_mode = match mode {
        "scalar" => FeedMode::Scalar,
        "bulk" => FeedMode::Bulk,
        _ => return Err(PyValueError::new_err("mode must be 'scalar' or 'bulk'")),
    };
    let out = Path::new(out_dir);
    if !out.is_dir() {
        // fail with a Python exception rather than panicking inside File::create
        return Err(PyValueError::new_err(format!(
            "out_dir is not an existing directory: {out_dir}"
        )));
    }
    let b = py.allow_threads(|| profile_single_shard(nnz, nnz_per_row, feed_mode, out));
    let mut m = std::collections::HashMap::new();
    m.insert("decide_s".to_string(), b.decide_s);
    m.insert("decide_indices_s".to_string(), b.decide_indices_s);
    m.insert("feed_s".to_string(), b.feed_s);
    m.insert("shuffle_s".to_string(), b.shuffle_s);
    m.insert("delta_s".to_string(), b.delta_s);
    m.insert("zstd_s".to_string(), b.zstd_s);
    m.insert("staging_s".to_string(), b.staging_s);
    m.insert("nnz".to_string(), b.nnz as f64);
    Ok(m)
}

#[pyfunction]
#[pyo3(signature = (*, n_rows, n_var, density, dtype, codec, out_dir))]
fn _profile_dense_encode_block(
    py: Python<'_>,
    n_rows: usize,
    n_var: usize,
    density: f64,
    dtype: &str,
    codec: &str,
    out_dir: &str,
) -> PyResult<std::collections::HashMap<String, f64>> {
    // density must be a fraction in [0, 1]. The block writes stay in bounds for
    // any density (k = i*total/n_nonzero <= total-1), but density > 1 makes the
    // probe meaningless: round(total*density) > total, so the reported nnz
    // exceeds n_elements and the "distinct position per nonzero" invariant breaks.
    if !(0.0..=1.0).contains(&density) {
        return Err(PyValueError::new_err(format!(
            "density must be in [0.0, 1.0], got {density}"
        )));
    }
    let dt = match dtype {
        "uint16" => profile::DenseDtype::U16,
        "float32" => profile::DenseDtype::F32,
        _ => return Err(PyValueError::new_err("dtype must be 'uint16' or 'float32'")),
    };
    let cd = match codec {
        "plain" => profile::DenseCodec::Plain,
        "shuffle" => profile::DenseCodec::Shuffle,
        "shuffle_delta" => profile::DenseCodec::ShuffleDelta,
        _ => {
            return Err(PyValueError::new_err(
                "codec must be 'plain', 'shuffle', or 'shuffle_delta'",
            ))
        }
    };
    let out = Path::new(out_dir);
    if !out.is_dir() {
        return Err(PyValueError::new_err(format!(
            "out_dir is not an existing directory: {out_dir}"
        )));
    }
    let b = py.allow_threads(|| profile::profile_dense_block(n_rows, n_var, density, dt, cd, out));
    let mut m = std::collections::HashMap::new();
    m.insert("feed_s".to_string(), b.feed_s);
    m.insert("shuffle_s".to_string(), b.shuffle_s);
    m.insert("delta_s".to_string(), b.delta_s);
    m.insert("zstd_s".to_string(), b.zstd_s);
    m.insert("staging_s".to_string(), b.staging_s);
    m.insert("n_elements".to_string(), b.n_elements as f64);
    m.insert("nnz".to_string(), b.nnz as f64);
    m.insert("compressed_bytes".to_string(), b.compressed_bytes as f64);
    Ok(m)
}

/// Decode a full v2 archive's CSR into caller-provided output arrays.
/// `out_data`/`out_indices`/`out_indptr` are pre-sized numpy arrays (heap or
/// shm-backed). `out_lens[i]`/`row_lens[i]` are shard i's realized output nnz /
/// rows (full read: header.nnz / header.n_rows). Only the SAFE-cast default path
/// (Phase B); lossy + arbitrary dtypes land in later tasks. Writes in place.
#[pyfunction]
#[pyo3(signature = (paths, bases, out_lens, row_lens, shard_rows, shard_nnz,
                    sel_starts, sel_stops, row_idx_flat,
                    out_data, out_indices, out_indptr, *,
                    n_vars, row_gather=false, data_safe=true, index_safe=true,
                    allow_lossy=false, n_threads=8,
                    data_min_dtype=None, index_min_dtype=None))]
#[allow(clippy::too_many_arguments)]
fn decode_v2_csr(
    py: Python<'_>,
    paths: Vec<String>,
    bases: Vec<u64>,
    out_lens: Vec<usize>,
    row_lens: Vec<usize>,
    shard_rows: Vec<usize>,
    shard_nnz: Vec<usize>,
    sel_starts: Vec<i64>,
    sel_stops: Vec<i64>,
    row_idx_flat: PyReadonlyArray1<i64>,
    out_data: &Bound<'_, PyAny>,
    out_indices: &Bound<'_, PyAny>,
    out_indptr: &Bound<'_, PyAny>,
    n_vars: usize,
    row_gather: bool,
    data_safe: bool,
    index_safe: bool,
    allow_lossy: bool,
    n_threads: usize,
    data_min_dtype: Option<String>,
    index_min_dtype: Option<String>,
) -> PyResult<()> {
    use decode::{decode_all_csr, Selection};
    // Validate the per-shard parallel arrays match before any indexing, so a
    // caller mistake returns a clean PyValueError instead of a thread panic.
    let n_shards = paths.len();
    if bases.len() != n_shards
        || out_lens.len() != n_shards
        || row_lens.len() != n_shards
        || shard_rows.len() != n_shards
        || shard_nnz.len() != n_shards
    {
        return Err(PyValueError::new_err(
            "decode_v2_csr: paths, bases, out_lens, row_lens, shard_rows, shard_nnz must have equal length",
        ));
    }
    // Build the per-shard Selection. A read is EITHER contiguous (Range, every shard)
    // OR a row gather (Rows, every shard) -- never mixed -- so a single row_gather flag
    // drives the mode. In rows mode `row_idx_flat` holds the concatenated per-shard
    // local row indices, sliced by row_lens; in range mode it is empty and
    // sel_starts/sel_stops give each shard's [lstart, lstop).
    // Borrow the row-index buffer directly (zero-copy); it lives as long as
    // row_idx_flat, which outlives the synchronous decode call below.
    let row_flat = row_idx_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("decode_v2_csr: row_idx_flat must be contiguous"))?;
    // Reject negative indices at the FFI boundary: `as usize` would wrap them into
    // huge values, surfacing later as a cryptic panic instead of a clean error.
    if row_flat.iter().any(|&r| r < 0) {
        return Err(PyValueError::new_err(
            "decode_v2_csr: row_idx_flat must contain only non-negative indices",
        ));
    }
    let selections: Vec<Selection> = if row_gather {
        let total_rows: usize = row_lens.iter().sum();
        if row_flat.len() != total_rows {
            return Err(PyValueError::new_err(
                "decode_v2_csr: row_idx_flat length must equal sum(row_lens)",
            ));
        }
        let mut sels = Vec::with_capacity(n_shards);
        let mut off = 0usize;
        for s in 0..n_shards {
            let sub = &row_flat[off..off + row_lens[s]];
            // Upfront upper-bound check (negatives already rejected above): a row
            // index >= the shard's row count would panic in a worker thread, which
            // pyo3 surfaces only as an opaque PanicException -- fail clean here.
            if sub.iter().any(|&r| r as usize >= shard_rows[s]) {
                return Err(PyValueError::new_err(format!(
                    "decode_v2_csr: row index out of bounds for shard {s} (shard_rows {})",
                    shard_rows[s]
                )));
            }
            sels.push(Selection::Rows(sub));
            off += row_lens[s];
        }
        sels
    } else {
        if sel_starts.len() != n_shards || sel_stops.len() != n_shards {
            return Err(PyValueError::new_err(
                "decode_v2_csr: sel_starts/sel_stops must have length n_shards in range mode",
            ));
        }
        if sel_starts.iter().chain(sel_stops.iter()).any(|&v| v < 0) {
            return Err(PyValueError::new_err(
                "decode_v2_csr: sel_starts and sel_stops must be non-negative",
            ));
        }
        // Upfront range validation: an out-of-bounds or inconsistent range would
        // otherwise panic in a worker thread (opaque PanicException). Validate against
        // the trusted shard_rows / row_lens before spawning threads.
        for s in 0..n_shards {
            // sel_starts/sel_stops validated non-negative above -> compare in usize
            // (avoids casting the usize shard_rows to i64).
            let start = sel_starts[s] as usize;
            let stop = sel_stops[s] as usize;
            let limit = shard_rows[s];
            if start > stop {
                return Err(PyValueError::new_err(format!(
                    "decode_v2_csr: sel_start {start} > sel_stop {stop} for shard {s}"
                )));
            }
            if stop > limit {
                return Err(PyValueError::new_err(format!(
                    "decode_v2_csr: sel_stop {stop} exceeds shard_rows {limit} for shard {s}"
                )));
            }
            if stop - start != row_lens[s] {
                return Err(PyValueError::new_err(format!(
                    "decode_v2_csr: row_lens[{s}] ({}) does not match range length {}",
                    row_lens[s],
                    stop - start
                )));
            }
        }
        (0..n_shards)
            .map(|s| Selection::Range(sel_starts[s] as usize, sel_stops[s] as usize))
            .collect()
    };

    let path_bufs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
    let total_out_len: usize = out_lens.iter().sum();
    let total_row_len: usize = row_lens.iter().sum();

    // indptr is always int64 in the output buffer.
    let mut indptr_rw = out_indptr.extract::<numpy::PyReadwriteArray1<i64>>()?;
    let indptr_slice = indptr_rw.as_slice_mut()?;

    let data_name = out_data.getattr("dtype")?.str()?.to_string();
    let idx_name = out_indices.getattr("dtype")?.str()?.to_string();

    // Three-tier lossy policy (Task 16): tier 1 (numpy-safe) and tier 2 (allow_lossy)
    // both use the fused `as` cast with no per-element check; tier 3
    // (!safe && !allow_lossy) verifies each element round-trips losslessly and raises.
    let check_data = !data_safe && !allow_lossy;
    let check_index = !index_safe && !allow_lossy;
    let data_min = data_min_dtype.as_deref();
    let index_min = index_min_dtype.as_deref();

    macro_rules! run {
        ($d:ty, $i:ty) => {{
            let mut d_rw = out_data.extract::<numpy::PyReadwriteArray1<$d>>()?;
            let mut i_rw = out_indices.extract::<numpy::PyReadwriteArray1<$i>>()?;
            let ds = d_rw.as_slice_mut()?;
            let is = i_rw.as_slice_mut()?;
            if ds.len() < total_out_len || is.len() < total_out_len {
                return Err(PyValueError::new_err(
                    "decode_v2_csr: output data/indices array smaller than sum(out_lens)",
                ));
            }
            if indptr_slice.len() < total_row_len + 1 {
                return Err(PyValueError::new_err(
                    "decode_v2_csr: output indptr array smaller than sum(row_lens) + 1",
                ));
            }
            py.allow_threads(|| {
                decode_all_csr::<$d, $i>(
                    &path_bufs,
                    &bases,
                    &selections,
                    &shard_rows,
                    &shard_nnz,
                    &out_lens,
                    &row_lens,
                    n_vars,
                    n_threads,
                    check_data,
                    check_index,
                    data_min,
                    index_min,
                    ds,
                    is,
                    indptr_slice,
                )
            })
            .map_err(decode_err_to_py)?;
            Ok(())
        }};
    }

    match (data_name.as_str(), idx_name.as_str()) {
        ("float32", "uint16") => run!(f32, u16),
        ("float32", "uint32") => run!(f32, u32),
        ("float32", "int32") => run!(f32, i32),
        ("float32", "int64") => run!(f32, i64),
        ("float64", "uint16") => run!(f64, u16),
        ("float64", "uint32") => run!(f64, u32),
        ("float64", "int32") => run!(f64, i32),
        ("float64", "int64") => run!(f64, i64),
        ("uint16", "uint16") => run!(u16, u16),
        ("uint16", "uint32") => run!(u16, u32),
        ("uint16", "int32") => run!(u16, i32),
        ("uint16", "int64") => run!(u16, i64),
        ("uint32", "uint16") => run!(u32, u16),
        ("uint32", "uint32") => run!(u32, u32),
        ("uint32", "int32") => run!(u32, i32),
        ("uint32", "int64") => run!(u32, i64),
        ("int32", "uint16") => run!(i32, u16),
        ("int32", "uint32") => run!(i32, u32),
        ("int32", "int32") => run!(i32, i32),
        ("int32", "int64") => run!(i32, i64),
        ("int64", "uint16") => run!(i64, u16),
        ("int64", "uint32") => run!(i64, u32),
        ("int64", "int32") => run!(i64, i32),
        ("int64", "int64") => run!(i64, i64),
        (d, i) => Err(PyValueError::new_err(format!(
            "decode_v2_csr: unsupported output dtype combo data={d} indices={i}"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (paths, bases, row_lens, shard_rows, shard_nnz,
                    sel_starts, sel_stops, row_idx_flat, out_dense,
                    *, n_vars, row_gather=false, data_safe=true,
                    allow_lossy=false, n_threads=8))]
#[allow(clippy::too_many_arguments)]
fn decode_v2_dense(
    py: Python<'_>,
    paths: Vec<String>,
    bases: Vec<u64>,
    row_lens: Vec<usize>,
    shard_rows: Vec<usize>,
    shard_nnz: Vec<usize>,
    sel_starts: Vec<i64>,
    sel_stops: Vec<i64>,
    row_idx_flat: PyReadonlyArray1<i64>,
    out_dense: &Bound<'_, PyAny>,
    n_vars: usize,
    row_gather: bool,
    data_safe: bool,
    allow_lossy: bool,
    n_threads: usize,
) -> PyResult<()> {
    use decode::{decode_all_dense, Selection};
    // Validate the per-shard parallel arrays match before any indexing, so a
    // caller mistake returns a clean PyValueError instead of a thread panic.
    let n_shards = paths.len();
    if bases.len() != n_shards
        || row_lens.len() != n_shards
        || shard_rows.len() != n_shards
        || shard_nnz.len() != n_shards
    {
        return Err(PyValueError::new_err(
            "decode_v2_dense: paths, bases, row_lens, shard_rows, shard_nnz must have equal length",
        ));
    }
    // Build the per-shard Selection. A read is EITHER contiguous (Range, every shard)
    // OR a row gather (Rows, every shard) -- never mixed. In rows mode `row_idx_flat`
    // holds the concatenated per-shard local row indices, sliced by row_lens.
    // Borrow the row-index buffer directly (zero-copy); it lives as long as
    // row_idx_flat, which outlives the synchronous decode call below.
    let row_flat = row_idx_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("decode_v2_dense: row_idx_flat must be contiguous"))?;
    // Reject negative indices at the FFI boundary: `as usize` would wrap them into
    // huge values, surfacing later as a cryptic panic instead of a clean error.
    if row_flat.iter().any(|&r| r < 0) {
        return Err(PyValueError::new_err(
            "decode_v2_dense: row_idx_flat must contain only non-negative indices",
        ));
    }
    let selections: Vec<Selection> = if row_gather {
        let total_rows: usize = row_lens.iter().sum();
        if row_flat.len() != total_rows {
            return Err(PyValueError::new_err(
                "decode_v2_dense: row_idx_flat length must equal sum(row_lens)",
            ));
        }
        let mut sels = Vec::with_capacity(n_shards);
        let mut off = 0usize;
        for s in 0..n_shards {
            let sub = &row_flat[off..off + row_lens[s]];
            // Upfront upper-bound check (negatives already rejected above): a row
            // index >= the shard's row count would panic in a worker thread, which
            // pyo3 surfaces only as an opaque PanicException -- fail clean here.
            if sub.iter().any(|&r| r as usize >= shard_rows[s]) {
                return Err(PyValueError::new_err(format!(
                    "decode_v2_dense: row index out of bounds for shard {s} (shard_rows {})",
                    shard_rows[s]
                )));
            }
            sels.push(Selection::Rows(sub));
            off += row_lens[s];
        }
        sels
    } else {
        if sel_starts.len() != n_shards || sel_stops.len() != n_shards {
            return Err(PyValueError::new_err(
                "decode_v2_dense: sel_starts/sel_stops must have length n_shards in range mode",
            ));
        }
        if sel_starts.iter().chain(sel_stops.iter()).any(|&v| v < 0) {
            return Err(PyValueError::new_err(
                "decode_v2_dense: sel_starts and sel_stops must be non-negative",
            ));
        }
        // Upfront range validation: an out-of-bounds or inconsistent range would
        // otherwise panic in a worker thread (opaque PanicException). Validate against
        // the trusted shard_rows / row_lens before spawning threads.
        for s in 0..n_shards {
            // sel_starts/sel_stops validated non-negative above -> compare in usize
            // (avoids casting the usize shard_rows to i64).
            let start = sel_starts[s] as usize;
            let stop = sel_stops[s] as usize;
            let limit = shard_rows[s];
            if start > stop {
                return Err(PyValueError::new_err(format!(
                    "decode_v2_dense: sel_start {start} > sel_stop {stop} for shard {s}"
                )));
            }
            if stop > limit {
                return Err(PyValueError::new_err(format!(
                    "decode_v2_dense: sel_stop {stop} exceeds shard_rows {limit} for shard {s}"
                )));
            }
            if stop - start != row_lens[s] {
                return Err(PyValueError::new_err(format!(
                    "decode_v2_dense: row_lens[{s}] ({}) does not match range length {}",
                    row_lens[s],
                    stop - start
                )));
            }
        }
        (0..n_shards)
            .map(|s| Selection::Range(sel_starts[s] as usize, sel_stops[s] as usize))
            .collect()
    };

    let path_bufs: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
    let total_row_len: usize = row_lens.iter().sum();
    // checked_mul so a pathological row_lens/n_vars cannot wrap the buffer-size
    // check (which would otherwise pass a too-small buffer to the Rust scatter).
    let expected_cells = total_row_len.checked_mul(n_vars).ok_or_else(|| {
        PyValueError::new_err("decode_v2_dense: total_row_len * n_vars overflowed usize")
    })?;
    let name = out_dense.getattr("dtype")?.str()?.to_string();
    // Tier-3 lossy gate for the dense data cast (the dense path has no output index
    // stream; column indices are decoded to i64 internally, always a safe cast).
    let check_data = !data_safe && !allow_lossy;
    macro_rules! run_dense {
        ($d:ty) => {{
            let mut rw = out_dense.extract::<numpy::PyReadwriteArray2<$d>>()?;
            let s = rw.as_slice_mut().map_err(|_| {
                PyValueError::new_err("decode_v2_dense: output dense array must be C-contiguous")
            })?;
            if s.len() < expected_cells {
                return Err(PyValueError::new_err(
                    "decode_v2_dense: output dense array smaller than sum(row_lens) * n_vars",
                ));
            }
            py.allow_threads(|| {
                decode_all_dense::<$d>(
                    &path_bufs,
                    &bases,
                    &selections,
                    &shard_rows,
                    &shard_nnz,
                    &row_lens,
                    n_vars,
                    n_threads,
                    check_data,
                    s,
                )
            })
            .map_err(decode_err_to_py)?;
            Ok(())
        }};
    }
    match name.as_str() {
        "float32" => run_dense!(f32),
        "float64" => run_dense!(f64),
        "uint16" => run_dense!(u16),
        "uint32" => run_dense!(u32),
        "int32" => run_dense!(i32),
        "int64" => run_dense!(i64),
        d => Err(PyValueError::new_err(format!(
            "decode_v2_dense: unsupported output dtype {d}"
        ))),
    }
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_selftest, m)?)?;
    m.add_function(wrap_pyfunction!(encode_shards_inmem, m)?)?;
    m.add_function(wrap_pyfunction!(encode_shards_inmem_dense, m)?)?;
    m.add_function(wrap_pyfunction!(encode_shards_inmem_dense_via_csr, m)?)?;
    m.add_function(wrap_pyfunction!(decide_dtypes, m)?)?;
    m.add_function(wrap_pyfunction!(decide_dtypes_dense, m)?)?;
    m.add_function(wrap_pyfunction!(_profile_encode_single_shard, m)?)?;
    m.add_function(wrap_pyfunction!(_profile_dense_encode_block, m)?)?;
    m.add_function(wrap_pyfunction!(decode_v2_csr, m)?)?;
    m.add_function(wrap_pyfunction!(decode_v2_dense, m)?)?;
    #[cfg(target_arch = "x86_64")]
    m.add_function(wrap_pyfunction!(pfor::_pfor_decode_words, m)?)?;
    #[cfg(target_arch = "x86_64")]
    m.add_function(wrap_pyfunction!(cell::decode_cells, m)?)?;
    #[cfg(target_arch = "x86_64")]
    m.add_function(wrap_pyfunction!(pfor::_pfor_encode_words, m)?)?;
    #[cfg(target_arch = "x86_64")]
    m.add_function(wrap_pyfunction!(pfor::_pfor_encode_capacity, m)?)?;
    #[cfg(target_arch = "x86_64")]
    m.add_function(wrap_pyfunction!(cell::encode_cells, m)?)?;
    #[cfg(target_arch = "x86_64")]
    m.add_function(wrap_pyfunction!(cell::encode_cells_block, m)?)?;
    #[cfg(target_arch = "x86_64")]
    m.add_function(wrap_pyfunction!(cell::reorder_frames, m)?)?;
    #[cfg(target_arch = "x86_64")]
    m.add(
        "NonIntegerCounts",
        m.py().get_type_bound::<cell::NonIntegerCounts>(),
    )?;
    Ok(())
}
