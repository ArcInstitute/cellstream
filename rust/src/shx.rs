//! SHX2 encode kernel — reproduces src/cellstream/v2/{codec,format,writer}.py byte
//! surface: per-stream 1 MiB-element chunks; byte-shuffle (+ byte-delta for
//! indices/indptr); zstd level 1; SHX2 container with fixed-point JSON header.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

const HEADER_PAD: usize = 4096;
const FIXED_PREFIX: usize = 10; // 4 magic + 2 reserved + 4 header_size
const MAX_FIXEDPOINT_ITERS: usize = 8;

/// Byte-shuffle: N elements of T bytes -> (T planes x N) plane-major.
/// out[p*n + e] = bytes[e*T + p]  (matches numpy view(u8).reshape(-1,T).T.ascontiguous)
pub(crate) fn byte_shuffle(bytes: &[u8], t: usize) -> Vec<u8> {
    let n = bytes.len() / t;
    let mut out = vec![0u8; bytes.len()];
    if t == 2 {
        // u16 fast path (data + indices): deinterleave lo/hi planes — sequential
        // reads + sequential writes, auto-vectorizes far better than the strided
        // generic transpose.
        let (lo, hi) = out.split_at_mut(n);
        for e in 0..n {
            lo[e] = bytes[2 * e];
            hi[e] = bytes[2 * e + 1];
        }
    } else {
        // Generic transpose, plane-outer so writes are sequential per plane.
        for p in 0..t {
            let dst = &mut out[p * n..(p + 1) * n];
            for e in 0..n {
                dst[e] = bytes[e * t + p];
            }
        }
    }
    out
}

/// Per-plane byte-delta (mod 256) along the element axis, applied AFTER shuffle.
/// d[p,0] = s[p,0]; d[p,e] = s[p,e] - s[p,e-1]. In-place, reverse order so the
/// previous element is still the original value when we subtract.
pub(crate) fn byte_delta(planes: &mut [u8], n: usize, t: usize) {
    if n == 0 {
        return;
    }
    for p in 0..t {
        let base = p * n;
        for e in (1..n).rev() {
            planes[base + e] = planes[base + e].wrapping_sub(planes[base + e - 1]);
        }
    }
}

pub(crate) fn zstd1(bytes: &[u8]) -> Vec<u8> {
    zstd::bulk::compress(bytes, 1).expect("zstd compress")
}

/// Encode one chunk: shuffle, (delta), zstd-1. `elem_bytes` is the raw
/// little-endian bytes of `n` elements of `t` bytes each (may be empty).
pub fn encode_chunk(elem_bytes: &[u8], t: usize, delta: bool) -> Vec<u8> {
    if elem_bytes.is_empty() {
        return zstd1(&[]);
    }
    let n = elem_bytes.len() / t;
    let mut s = byte_shuffle(elem_bytes, t);
    if delta {
        byte_delta(&mut s, n, t);
    }
    zstd1(&s)
}

/// Encode one chunk RAW: zstd-1 only, no shuffle/delta. Used for the float DATA
/// stream (#142) where byteshuffle scatters each value's bytes into separate
/// planes and destroys the whole-value repeats zstd matches. Byte-identical to
/// the Python `codec.zstd_only_encode` (same libzstd-1 over the same LE bytes).
pub fn encode_chunk_raw(elem_bytes: &[u8]) -> Vec<u8> {
    zstd1(elem_bytes)
}

#[derive(Clone, Debug)]
pub(crate) struct ChunkDesc {
    pub(crate) offset: usize,
    pub(crate) compressed_size: usize,
    pub(crate) decompressed_size: usize,
}

struct Stream {
    dtype: &'static str,
    chunks: Vec<ChunkDesc>,
}

pub(crate) const CHUNK_ELEMS: usize = 1_048_576;

/// Chunk a stream's bytes into <=1 MiB-element compressed chunks. Empty stream
/// -> exactly one (empty) chunk, matching the Python writer.
#[allow(dead_code)]
fn chunk_stream(
    elem_bytes: &[u8],
    t: usize,
    delta: bool,
    dtype: &'static str,
) -> (Stream, Vec<u8>) {
    let mut payload = Vec::new();
    let mut chunks = Vec::new();
    let chunk_bytes = CHUNK_ELEMS * t;
    if elem_bytes.is_empty() {
        let c = encode_chunk(&[], t, delta);
        chunks.push(ChunkDesc {
            offset: 0,
            compressed_size: c.len(),
            decompressed_size: 0,
        });
        payload.extend_from_slice(&c);
    } else {
        let mut i = 0;
        while i < elem_bytes.len() {
            let end = (i + chunk_bytes).min(elem_bytes.len());
            let slab = &elem_bytes[i..end];
            let c = encode_chunk(slab, t, delta);
            chunks.push(ChunkDesc {
                offset: 0,
                compressed_size: c.len(),
                decompressed_size: slab.len(),
            });
            payload.extend_from_slice(&c);
            i = end;
        }
    }
    (Stream { dtype, chunks }, payload)
}

fn round_up(x: usize, m: usize) -> usize {
    (x + m - 1) & !(m - 1)
}

/// Serialize the SHX2 JSON header EXACTLY as ShxHeader.to_json (compact
/// separators, fixed key order) so byte-identity is achievable.
fn header_json(
    shard_idx: usize,
    n_rows: usize,
    n_vars: usize,
    nnz: usize,
    shuffle: &str,
    streams: &[&Stream; 3],
) -> String {
    fn stream_json(s: &Stream) -> String {
        let mut chunks = String::from("[");
        for (i, c) in s.chunks.iter().enumerate() {
            if i > 0 {
                chunks.push(',');
            }
            chunks.push_str(&format!(
                "{{\"offset\":{},\"compressed_size\":{},\"decompressed_size\":{}}}",
                c.offset, c.compressed_size, c.decompressed_size
            ));
        }
        chunks.push(']');
        format!(
            "{{\"dtype\":\"{}\",\"n_chunks\":{},\"chunks\":{}}}",
            s.dtype,
            s.chunks.len(),
            chunks
        )
    }
    format!(
        "{{\"schema_version\":2,\"shard_idx\":{},\"n_rows\":{},\"n_vars\":{},\"nnz\":{},\
\"compression\":\"zstd\",\"shuffle\":\"{}\",\"compression_level\":1,\
\"chunk_size_elements\":1048576,\"data\":{},\"indices\":{},\"indptr\":{}}}",
        shard_idx,
        n_rows,
        n_vars,
        nnz,
        shuffle,
        stream_json(streams[0]),
        stream_json(streams[1]),
        stream_json(streams[2]),
    )
}

fn header_bytes_with_offsets(
    shard_idx: usize,
    n_rows: usize,
    n_vars: usize,
    nnz: usize,
    shuffle: &str,
    streams: &mut [Stream; 3],
) -> Vec<u8> {
    // Fixed-point: header length depends on offsets, offsets depend on header length.
    let mut prev_data_start: i64 = -1;
    let mut converged = false;
    for _ in 0..MAX_FIXEDPOINT_ITERS {
        let js = header_json(
            shard_idx,
            n_rows,
            n_vars,
            nnz,
            shuffle,
            &[&streams[0], &streams[1], &streams[2]],
        );
        let data_start = round_up(FIXED_PREFIX + js.len(), HEADER_PAD);
        if data_start as i64 == prev_data_start {
            converged = true;
            break;
        }
        prev_data_start = data_start as i64;
        let mut cursor = data_start;
        for s in streams.iter_mut() {
            for c in s.chunks.iter_mut() {
                c.offset = cursor;
                cursor += c.compressed_size;
            }
        }
    }
    assert!(converged, "header/offset fixed-point did not converge");

    header_json(
        shard_idx,
        n_rows,
        n_vars,
        nnz,
        shuffle,
        &[&streams[0], &streams[1], &streams[2]],
    )
    .into_bytes()
}

fn write_header_prefix(mut fh: impl Write, js_bytes: &[u8]) -> io::Result<()> {
    let pad_to = round_up(FIXED_PREFIX + js_bytes.len(), HEADER_PAD);
    let pad_len = pad_to - (FIXED_PREFIX + js_bytes.len());

    fh.write_all(b"SHX2")?;
    // reserved u16 = 0, header_size u32 = len(js), little-endian
    fh.write_all(&0u16.to_le_bytes())?;
    fh.write_all(&(js_bytes.len() as u32).to_le_bytes())?;
    fh.write_all(js_bytes)?;
    if pad_len > 0 {
        fh.write_all(&vec![0u8; pad_len])?;
    }
    Ok(())
}

/// Assemble + write one .shx file (atomic: .tmp then rename). `streams` are
/// (dtype, raw little-endian element bytes, itemsize, delta) in data/indices/indptr order.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn assemble_shx(
    path: &Path,
    shard_idx: usize,
    n_rows: usize,
    n_vars: usize,
    nnz: usize,
    shuffle: &str,
    data: (&'static str, &[u8], usize, bool),
    indices: (&'static str, &[u8], usize, bool),
    indptr: (&'static str, &[u8], usize, bool),
) -> io::Result<()> {
    let (sd, pd) = chunk_stream(data.1, data.2, data.3, data.0);
    let (si, pi) = chunk_stream(indices.1, indices.2, indices.3, indices.0);
    let (sp, pp) = chunk_stream(indptr.1, indptr.2, indptr.3, indptr.0);
    let mut streams = [sd, si, sp];
    let js_bytes = header_bytes_with_offsets(shard_idx, n_rows, n_vars, nnz, shuffle, &mut streams);

    let tmp = path.with_extension("shx.tmp");
    {
        let mut fh = fs::File::create(&tmp)?;
        write_header_prefix(&mut fh, &js_bytes)?;
        fh.write_all(&pd)?;
        fh.write_all(&pi)?;
        fh.write_all(&pp)?;
        fh.sync_all().ok();
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Stream-assemble: each staging file already holds its stream's compressed
/// chunks (in order). `streams` is (dtype, staging_path, chunk_descs) for
/// data, indices, indptr. chunk_descs have compressed_size + decompressed_size
/// set; this fn fills `offset` (fixed-point), writes the header, then io::copy's
/// each staging file's bytes into the final .shx — no payload held in RAM.
#[allow(clippy::too_many_arguments)]
pub fn assemble_shx_streaming(
    path: &Path,
    shard_idx: usize,
    n_rows: usize,
    n_vars: usize,
    nnz: usize,
    shuffle: &str,
    streams: [(&'static str, &Path, &mut [ChunkDesc]); 3],
) -> io::Result<()> {
    let [(data_dtype, data_path, data_chunks), (indices_dtype, indices_path, indices_chunks), (indptr_dtype, indptr_path, indptr_chunks)] =
        streams;
    let mut stream_values = [
        Stream {
            dtype: data_dtype,
            chunks: data_chunks.to_vec(),
        },
        Stream {
            dtype: indices_dtype,
            chunks: indices_chunks.to_vec(),
        },
        Stream {
            dtype: indptr_dtype,
            chunks: indptr_chunks.to_vec(),
        },
    ];
    let js_bytes =
        header_bytes_with_offsets(shard_idx, n_rows, n_vars, nnz, shuffle, &mut stream_values);

    for (dst, src) in data_chunks.iter_mut().zip(stream_values[0].chunks.iter()) {
        dst.offset = src.offset;
    }
    for (dst, src) in indices_chunks
        .iter_mut()
        .zip(stream_values[1].chunks.iter())
    {
        dst.offset = src.offset;
    }
    for (dst, src) in indptr_chunks.iter_mut().zip(stream_values[2].chunks.iter()) {
        dst.offset = src.offset;
    }

    let tmp = path.with_extension("shx.tmp");
    {
        let mut out = fs::File::create(&tmp)?;
        write_header_prefix(&mut out, &js_bytes)?;
        for staging_path in [data_path, indices_path, indptr_path] {
            let mut staging = fs::File::open(staging_path)?;
            io::copy(&mut staging, &mut out)?;
        }
        out.sync_all().ok();
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Inverse of byte_shuffle for test verification only.
    fn byte_unshuffle(planes: &[u8], t: usize) -> Vec<u8> {
        let n = planes.len() / t;
        let mut out = vec![0u8; planes.len()];
        for e in 0..n {
            for p in 0..t {
                out[e * t + p] = planes[p * n + e];
            }
        }
        out
    }

    #[test]
    fn shuffle_roundtrips_u16() {
        let elems: Vec<u16> = vec![0x0102, 0xF0FF, 0x0000, 0xABCD];
        let bytes: Vec<u8> = elems.iter().flat_map(|v| v.to_le_bytes()).collect();
        let s = byte_shuffle(&bytes, 2);
        // lo plane then hi plane
        assert_eq!(&s[..4], &[0x02, 0xFF, 0x00, 0xCD]);
        assert_eq!(&s[4..], &[0x01, 0xF0, 0x00, 0xAB]);
        assert_eq!(byte_unshuffle(&s, 2), bytes);
    }

    #[test]
    fn delta_roundtrips() {
        // single plane (t=1) for a clear check of mod-256 wrap
        let mut s = vec![10u8, 13, 13, 9];
        byte_delta(&mut s, 4, 1);
        assert_eq!(s, vec![10u8, 3, 0, 252]); // 9-13 = -4 = 252 mod 256
                                              // undelta (cumsum mod 256) recovers original
        let mut acc = 0u8;
        let recovered: Vec<u8> = s
            .iter()
            .map(|&d| {
                acc = acc.wrapping_add(d);
                acc
            })
            .collect();
        assert_eq!(recovered, vec![10u8, 13, 13, 9]);
    }

    #[test]
    fn encode_chunk_zstd_decompresses_to_shuffled_bytes() {
        let elems: Vec<u16> = vec![1, 2, 3, 1000, 65535];
        let bytes: Vec<u8> = elems.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = encode_chunk(&bytes, 2, false);
        let raw = zstd::bulk::decompress(&compressed, 1 << 20).unwrap();
        assert_eq!(raw, byte_shuffle(&bytes, 2));
    }

    #[test]
    fn empty_chunk_is_valid_zstd_frame() {
        let compressed = encode_chunk(&[], 2, false);
        let raw = zstd::bulk::decompress(&compressed, 1 << 20).unwrap();
        assert!(raw.is_empty());
    }

    #[test]
    fn encode_chunk_raw_decompresses_to_unshuffled_bytes() {
        // #142: raw chunk = zstd of the LE bytes verbatim (no byte-shuffle).
        let elems: Vec<f32> = vec![1.5, 2.5, 1.5, 2.5];
        let bytes: Vec<u8> = elems.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = encode_chunk_raw(&bytes);
        let raw = zstd::bulk::decompress(&compressed, 1 << 20).unwrap();
        assert_eq!(raw, bytes);
        // and it must differ from the shuffled encoding of the same bytes
        assert_ne!(compressed, encode_chunk(&bytes, 4, false));
    }

    #[test]
    fn encode_chunk_raw_empty_is_valid_frame() {
        let compressed = encode_chunk_raw(&[]);
        let raw = zstd::bulk::decompress(&compressed, 1 << 20).unwrap();
        assert!(raw.is_empty());
        // an empty data chunk is identical whether "raw" or shuffled (both zstd of empty).
        assert_eq!(compressed, encode_chunk(&[], 4, false));
    }

    #[test]
    fn assemble_shx_writes_parseable_file() {
        let dir = std::env::temp_dir().join("shx_asm_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("shard_0000.shx");

        let data: Vec<u16> = vec![5, 7, 9];
        let indices: Vec<u16> = vec![0, 2, 4];
        let indptr: Vec<i64> = vec![0, 2, 3]; // 2 rows
        let db: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ib: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
        let pb: Vec<u8> = indptr.iter().flat_map(|v| v.to_le_bytes()).collect();

        assemble_shx(
            &path,
            0,
            2,
            10,
            3,
            "byteshuffle-bytedelta",
            ("uint16", &db, 2, false),
            ("uint16", &ib, 2, true),
            ("int64", &pb, 8, true),
        )
        .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..4], b"SHX2");
        // header_size is the u32 at offset 6 (little-endian)
        let hsize = u32::from_le_bytes(bytes[6..10].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + hsize]).unwrap();
        assert!(header.contains("\"nnz\":3"));
        assert!(header.contains("\"shuffle\":\"byteshuffle-bytedelta\""));
        assert!(header.contains("\"chunk_size_elements\":1048576"));
    }
}
