//! SHX2 header reader: the parse-side mirror of src/cellstream/v2/format.py
//! (parse_shx_header) and the byte layout rust/src/shx.rs writes. Reads the
//! 10-byte fixed prefix + the UTF-8 JSON header via serde; chunk offsets/sizes
//! address compressed chunks later read with positioned reads.

#![allow(dead_code)]

use serde::Deserialize;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

pub const SHX_MAGIC: &[u8; 4] = b"SHX2";
const FIXED_PREFIX: usize = 10; // 4 magic + 2 reserved + 4 header_size
                                // DoS guard: a corrupt/hostile header_size could request a huge allocation
                                // (Gemini PR #131 security finding). Real SHX2 headers are KBs (4 KB-padded,
                                // up to ~hundreds of KB for many-chunk shards); cap well above any legit size.
const MAX_HEADER_BYTES: usize = 64 << 20; // 64 MiB

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkDesc {
    pub offset: u64,
    pub compressed_size: u64,
    pub decompressed_size: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamSpec {
    pub dtype: String,
    pub chunks: Vec<ChunkDesc>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ShxHeader {
    pub shard_idx: usize,
    pub n_rows: usize,
    pub n_vars: usize,
    pub nnz: usize,
    pub shuffle: String,
    pub data: StreamSpec,
    pub indices: StreamSpec,
    pub indptr: StreamSpec,
}

/// Parse the SHX2 header from an open file at the given mmap base (0 for a
/// directory shard; the 4 KB-aligned shard start for a packed archive).
pub fn parse_header(file: &mut File, base: u64) -> io::Result<ShxHeader> {
    file.seek(SeekFrom::Start(base))?;
    let mut prefix = [0u8; FIXED_PREFIX];
    file.read_exact(&mut prefix)?;
    if &prefix[0..4] != SHX_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad SHX magic: {:?}", &prefix[0..4]),
        ));
    }
    let header_size = u32::from_le_bytes(prefix[6..10].try_into().unwrap()) as usize;
    if header_size > MAX_HEADER_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("SHX2 header_size {header_size} exceeds the {MAX_HEADER_BYTES}-byte cap (corrupt file?)"),
        ));
    }
    let mut js = vec![0u8; header_size];
    file.read_exact(&mut js)?;
    serde_json::from_slice(&js).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_a_minimal_header() {
        // Hand-build the 10-byte prefix + compact JSON exactly as shx.rs writes it.
        let js = r#"{"schema_version":2,"shard_idx":3,"n_rows":2,"n_vars":10,"nnz":3,"compression":"zstd","shuffle":"byteshuffle-bytedelta","compression_level":1,"chunk_size_elements":1048576,"data":{"dtype":"uint16","n_chunks":1,"chunks":[{"offset":4096,"compressed_size":12,"decompressed_size":6}]},"indices":{"dtype":"uint16","n_chunks":1,"chunks":[{"offset":4108,"compressed_size":11,"decompressed_size":6}]},"indptr":{"dtype":"int64","n_chunks":1,"chunks":[{"offset":4119,"compressed_size":10,"decompressed_size":24}]}}"#;
        let dir = std::env::temp_dir().join("shx_read_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("h.shx");
        {
            let mut fh = File::create(&path).unwrap();
            fh.write_all(SHX_MAGIC).unwrap();
            fh.write_all(&0u16.to_le_bytes()).unwrap();
            fh.write_all(&(js.len() as u32).to_le_bytes()).unwrap();
            fh.write_all(js.as_bytes()).unwrap();
        }
        let mut fh = File::open(&path).unwrap();
        let h = parse_header(&mut fh, 0).unwrap();
        assert_eq!(h.shard_idx, 3);
        assert_eq!(h.n_rows, 2);
        assert_eq!(h.nnz, 3);
        assert_eq!(h.shuffle, "byteshuffle-bytedelta");
        assert_eq!(h.data.dtype, "uint16");
        assert_eq!(h.indptr.dtype, "int64");
        assert_eq!(h.data.chunks[0].decompressed_size, 6);
        assert_eq!(h.indptr.chunks[0].offset, 4119);
    }

    #[test]
    fn rejects_oversized_header_size() {
        // A corrupt header_size beyond the cap must error, not attempt a huge alloc.
        let dir = std::env::temp_dir().join("shx_read_dos_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad.shx");
        {
            let mut fh = File::create(&path).unwrap();
            fh.write_all(SHX_MAGIC).unwrap();
            fh.write_all(&0u16.to_le_bytes()).unwrap();
            fh.write_all(&u32::MAX.to_le_bytes()).unwrap(); // ~4 GiB header_size
        }
        let mut fh = File::open(&path).unwrap();
        assert!(parse_header(&mut fh, 0).is_err());
    }
}
