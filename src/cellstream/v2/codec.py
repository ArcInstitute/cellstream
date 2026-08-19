"""Pure-function CPU codec for SHX2.

bitshuffle + zstd-1 on write; zstd decompress + bitshuffle inverse on
read. The same compressed bytes are decoded on GPU by v2.reader_gpu
(via nvCOMP for zstd + our CUDA kernel for the inverse).
"""

from __future__ import annotations

import bitshuffle
import numpy as np
import zstandard as zstd

from cellstream.v2.format import MAX_CHUNK_DECOMPRESSED_BYTES

# python-zstandard's frame_content_size() returns a SIGNED -1 (unknown) / -2 (error)
# on the cext backend, but the module constants are the UNSIGNED 2**64-1 / 2**64-2;
# other backends/platforms may return either form. Accept both so the unknown/error
# detection is portable (Gemini, PR #167).
_CONTENTSIZE_UNKNOWN = getattr(zstd, "CONTENTSIZE_UNKNOWN", 2**64 - 1)
_CONTENTSIZE_ERROR = getattr(zstd, "CONTENTSIZE_ERROR", 2**64 - 2)


def bitshuffle_zstd_encode(arr: np.ndarray, *, level: int = 1) -> bytes:
    """Bitshuffle preprocess then zstd-compress. Returns the compressed bytes."""
    if arr.size == 0:
        return zstd.ZstdCompressor(level=level).compress(b"")
    shuf = bitshuffle.bitshuffle(np.ascontiguousarray(arr))
    return zstd.ZstdCompressor(level=level).compress(shuf.tobytes())


def bitshuffle_zstd_decode(compressed: bytes, *, dtype: type, n_elements: int) -> np.ndarray:
    """zstd-decompress then bitshuffle inverse to recover ``arr``."""
    if n_elements == 0:
        return np.array([], dtype=dtype)
    raw = _decompress_checked(
        compressed,
        expected=np.dtype(dtype).itemsize * n_elements,
        where="bitshuffle_zstd_decode",
    )
    # np.frombuffer + reshape produces a C-contiguous view of the bytes,
    # so passing `shuf` directly to bitunshuffle is safe — no ascontiguousarray needed.
    shuf = np.frombuffer(raw, dtype=dtype).reshape(n_elements)
    return bitshuffle.bitunshuffle(shuf)


# ---------------------------------------------------------------------------
# Per-array byte-filter codec (the ~38% CSR-payload win).
#   data            -> SHUFFLE              + zstd
#   indices/indptr  -> SHUFFLE + BYTEDELTA  + zstd
# Validated byte-exact vs blosc2 on two real count matrices.
# Inverses are simple (byte-plane transpose; uint8 cumsum mod 256) so they are
# GPU-kernelable later (deferred follow-on, ties into reader_gpu.py #40).
# ---------------------------------------------------------------------------

#: On-disk ShxHeader.shuffle marker for the byte-filter codec (self-describing).
BYTE_FILTER_SHUFFLE_NAME = "byteshuffle-bytedelta"

#: On-disk ShxHeader.shuffle marker for the byte-filter codec when the DATA stream
#: is stored RAW (zstd only, no shuffle/delta) — used for float data, where the
#: byteshuffle scatters each value's bytes into separate planes and destroys the
#: whole-value repeats zstd relies on (#142). Self-describing: data = raw zstd;
#: indices/indptr = SHUFFLE + BYTEDELTA (unchanged). Backward-compatible — old float
#: archives keep BYTE_FILTER_SHUFFLE_NAME and decode their shuffled data exactly as before.
BYTE_FILTER_RAWDATA_SHUFFLE_NAME = "byteshuffle-bytedelta-rawdata"

#: Array roles that get the BYTEDELTA step after SHUFFLE. `data` uses SHUFFLE only
#: (BYTEDELTA hurts counts: 3.40 < 4.41 bits/value on real data).
_BYTEDELTA_ROLES = frozenset({"indices", "indptr"})


def _byte_shuffle(a: np.ndarray) -> np.ndarray:
    """Byte-shuffle: (N elems, T bytes) -> (T, N) byte planes."""
    T = a.itemsize
    return np.ascontiguousarray(a.view(np.uint8).reshape(-1, T).T)


def _byte_unshuffle(s2d: np.ndarray, dtype) -> np.ndarray:
    """Inverse of _byte_shuffle. s2d is (T, N) byte planes -> (N,) of `dtype`."""
    # uint16 fast path: vectorized byte-combine (lo | hi<<8) avoids the strided
    # transpose copy -> decode FASTER than bitshuffle (microbench: data 990 vs
    # 767 MB/s). data + indices are uint16; fall back to transpose for indptr.
    dt = np.dtype(dtype)
    if dt.itemsize == 2:
        return (s2d[0].astype(np.uint16) | (s2d[1].astype(np.uint16) << 8)).view(dtype)
    return np.ascontiguousarray(s2d.T).view(dtype).ravel()


def _byte_delta(s2d: np.ndarray) -> np.ndarray:
    """Per-plane byte delta (mod 256) along the element axis — applied AFTER shuffle."""
    d = s2d.copy()
    d[:, 1:] = s2d[:, 1:] - s2d[:, :-1]  # uint8 wraps mod 256
    return d


def _byte_undelta(d2d: np.ndarray) -> np.ndarray:
    """Inverse of _byte_delta. uint8 cumsum wraps mod 256 natively (no int64 temp)."""
    return np.cumsum(d2d, axis=1, dtype=np.uint8)


def byte_filter_encode(arr: np.ndarray, *, role: str, level: int = 1) -> bytes:
    """SHUFFLE (+ BYTEDELTA for indices/indptr) then zstd-compress."""
    if arr.size == 0:
        return zstd.ZstdCompressor(level=level).compress(b"")
    s = _byte_shuffle(np.ascontiguousarray(arr))
    if role in _BYTEDELTA_ROLES:
        s = _byte_delta(s)
    return zstd.ZstdCompressor(level=level).compress(s.tobytes())


def _decompress_checked(compressed: bytes, *, expected: int, where: str) -> bytes:
    """zstd-decompress with a decompression-bomb guard and a clean error model.

    Bounds the allocation at ``expected`` and rejects an oversized declared size up
    front (mirroring the Rust core's 64 MiB cap, format.MAX_CHUNK_DECOMPRESSED_BYTES),
    so a tiny crafted frame cannot OOM the process. Our compressor always embeds the
    content size, so frame_content_size() == expected for valid archives; we reject a
    mismatch before decompressing (zstd's max_output_size does NOT cap an embedded-size
    frame, so this pre-check is the effective bound there). A frame WITHOUT an embedded
    size (== -1) is externally-produced/corrupt and is bounded by max_output_size.
    zstd's ZstdError is mapped to ValueError so corrupt input stays within the
    documented {ValueError, OSError, LossyCastError} decode error model.
    """
    if expected <= 0:
        raise ValueError(
            f"{where}: non-positive expected decompressed size {expected} (archive may be corrupt)."
        )
    if expected > MAX_CHUNK_DECOMPRESSED_BYTES:
        raise ValueError(
            f"{where}: chunk declares {expected} decompressed bytes, exceeding the "
            f"{MAX_CHUNK_DECOMPRESSED_BYTES}-byte cap (archive may be corrupt)."
        )
    try:
        declared = zstd.frame_content_size(compressed)
        if declared in (-1, _CONTENTSIZE_UNKNOWN):
            raw = zstd.ZstdDecompressor().decompress(compressed, max_output_size=expected)
        elif declared in (-2, _CONTENTSIZE_ERROR):
            raise ValueError(f"{where}: invalid or corrupt zstd frame header.")
        elif declared != expected:
            raise ValueError(
                f"{where}: frame declares {declared} bytes, expected {expected}; "
                f"shard may be corrupt."
            )
        else:
            raw = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as e:
        raise ValueError(f"{where}: zstd decode failed ({e}); shard may be corrupt.") from e
    if len(raw) != expected:
        raise ValueError(
            f"{where}: decompressed {len(raw)} bytes, expected {expected}; shard may be corrupt."
        )
    return raw


def byte_filter_decode(compressed: bytes, *, dtype: type, n_elements: int, role: str) -> np.ndarray:
    """zstd-decompress then (un-BYTEDELTA) + un-SHUFFLE to recover ``arr``."""
    if n_elements == 0:
        return np.array([], dtype=dtype)
    T = np.dtype(dtype).itemsize
    raw = _decompress_checked(compressed, expected=T * n_elements, where="byte_filter_decode")
    s = np.frombuffer(raw, np.uint8).reshape(T, n_elements)
    if role in _BYTEDELTA_ROLES:
        s = _byte_undelta(s)
    return _byte_unshuffle(s, dtype)


# ---------------------------------------------------------------------------
# Raw-data variant of the byte-filter codec: zstd ONLY (no shuffle, no delta).
# Used for the DATA stream of FLOAT archives (#142) — byteshuffle scatters each
# float's bytes into separate planes and destroys the whole-value repeats zstd
# matches, inflating low-cardinality lognorm data ~2.8x. Plain zstd on the raw
# little-endian bytes preserves those repeats. indices/indptr keep SHUFFLE+DELTA
# (byte_filter_encode/decode). Byte-identical to the Rust core (same libzstd-1).
# ---------------------------------------------------------------------------


def zstd_only_encode(arr: np.ndarray, *, level: int = 1) -> bytes:
    """zstd-compress the raw little-endian bytes of ``arr`` (no shuffle/delta).

    ``ndarray.tobytes()`` already returns a C-contiguous byte copy regardless of
    the array's layout, so no ``ascontiguousarray`` pre-pass is needed (Gemini)."""
    if arr.size == 0:
        return zstd.ZstdCompressor(level=level).compress(b"")
    return zstd.ZstdCompressor(level=level).compress(arr.tobytes())


def zstd_only_decode(compressed: bytes, *, dtype: type, n_elements: int) -> np.ndarray:
    """zstd-decompress raw bytes back to ``arr`` (inverse of zstd_only_encode).

    Shares byte_filter_decode's fail-loud / bounded-allocation guard: reject a
    frame whose declared/decompressed size disagrees with ``n_elements`` (a
    corrupt shard or a bad header), and cap a no-embedded-size frame at the
    expected length so a crafted input cannot drive an unbounded allocation."""
    if n_elements == 0:
        return np.array([], dtype=dtype)
    T = np.dtype(dtype).itemsize
    raw = _decompress_checked(compressed, expected=T * n_elements, where="zstd_only_decode")
    return np.frombuffer(raw, dtype=dtype)
