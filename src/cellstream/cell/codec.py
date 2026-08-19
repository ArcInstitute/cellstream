"""Per-cell pfordelta codec (delta+PFOR indices, PFOR values, via pyfastpfor).

Ported from cellstream. A frame is::

    [ uint32 LE : idx_blob length in bytes ]
    [ idx_blob ]  # simdfastpfor256 pack of per-cell DELTA-gaps of sorted gene indices
    [ val_blob ]  # simdfastpfor256 pack of the raw values; runs to end of frame

The Codec is immutable shared config; ``new_encoder()``/``new_decoder()`` mint
per-thread/per-process mutable state (a pyfastpfor codec + a scratch buffer),
mirroring cellstream's per-worker codec pattern.
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np

from cellstream.v2.codec import (
    byte_filter_decode,
    byte_filter_encode,
    zstd_only_decode,
    zstd_only_encode,
)

from . import format as fmt

_USIZE_MAX = 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1
_PFOR_BLOB_SLACK_BYTES = 65536


def _import_pyfastpfor():
    """Import pyfastpfor, or raise with the install instruction.

    THE one place that message lives. Callers used to guard the import eagerly -- at
    ``CellStore.__init__``, at ``write_cell_archive``'s codec planning, and in
    ``CellArchiveWriter.__init__`` -- so that a slim install failed early with something
    actionable. But the Rust engine has its own vendored FastPFor and never calls pyfastpfor
    (verified with a stub module whose codec raises on any use: with Rust active a write, a
    gather and a bulk read all completed bit-exact and the stub was never touched). Those
    eager guards therefore made the ``cell`` extra -- which has NO linux wheel and compiles
    C++ from an sdist -- a hard requirement for the default cell codec even on a machine whose
    wheel could do the whole job.

    Routing every construction site through here is what lets the eager guards go: the
    pure-Python encoder/decoder is built only where it is used, and when it genuinely is
    needed the user still gets this message instead of a bare ModuleNotFoundError from inside
    a decode loop. The text is unchanged from the guards it replaces -- error strings are API.
    """
    try:
        import pyfastpfor
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "layout='cell' with codec='pfordelta' requires the 'pyfastpfor' package. "
            "Install the extra: pip install 'cellstream[cell]'."
        ) from e
    return pyfastpfor


def _pfor_encode_capacity(n):
    """Output capacity (in uint32 words) to declare to ``encodeArray`` for ``n`` values (#273).

    The vendored FastPFor encoder enforces NO capacity -- ``SIMDFastPFor::__encodeArray``
    never reads its ``nvalue`` argument, and its caller's only reaction to an overrun is an
    after-the-fact ``fprintf`` -- so the caller must supply a sufficient buffer. The old
    ``n + 1024`` was not one: worst-case output is bounded by
    ``n + ceil(n/512) + ceil(n/65536)*996 + 65`` words, which crosses 1024 words of headroom
    at n ~ 512k and corrupted the heap on both engines.

    ``2n + 4096`` clears that worst case at every n by a wide margin (an 85-case adversarial
    guard-band sweep peaked at 49.9% of it). Deliberately loose for the same reason as
    ``_max_valid_blob_bytes``: a tight formula mirroring vendored internals would silently
    break on a FastPFor bump. Full derivation in rust/src/pfor.rs.

    Capacity never changes the words written -- only whether the codec throws -- so this
    cannot alter on-disk bytes.

    MUST match rust/src/pfor.rs ``pfor_encode_capacity``.
    """
    # clamp to the platform usize max so the value matches Rust's saturating arithmetic --
    # Python ints are arbitrary precision and would otherwise diverge at absurd n.
    return min(2 * n + 4096, _USIZE_MAX)


def _max_valid_blob_bytes(n):
    """Largest codec blob a VALID pfordelta frame can carry for ``n`` values (#269).

    Deliberately loose -- 2x raw uint32 storage plus 64 KiB, against a real encoding of ~2-3
    bytes/value -- so it cannot reject a valid frame. Its only job is to bound the decode
    buffer sized from it below. MUST match rust/src/pfor.rs ``max_valid_blob_bytes``.
    """
    # clamp to the platform usize max so the value matches Rust's saturating arithmetic --
    # Python ints are arbitrary precision and would otherwise diverge at absurd n.
    return min(8 * n + _PFOR_BLOB_SLACK_BYTES, _USIZE_MAX)


def _check_col_bounds(cs, n_vars, codec_name):
    """Reject an absolute column index >= ``n_vars`` on the WIDE value, before narrowing (#250).

    ``cs`` is the uint64 cumsum of the per-cell delta gaps, i.e. the absolute column indices at
    full width. Narrowing it to the on-disk index dtype first WRAPS an out-of-range column into
    a legal one (70000 -> 4464 at uint16), and nothing downstream can then tell: the bulk path's
    ``_csr_direct_assign(check_indices=True)`` scans the NARROWED array and sees 4464. Rust
    checks the wide value in ``write_col`` (rust/src/cell.rs) and raises; measured pre-fix, the
    pure-Python path returned column 4464 with no error on the same archive. The message form is
    copied from write_col deliberately.

    O(1), not O(nnz), for nnz <= 2**32+1: the gaps come out of a uint32 buffer so every gap is
    >= 0, so as long as the uint64 accumulator does not wrap, ``cs`` is non-decreasing on any
    input -- tampered or not -- and ``cs[-1]`` IS the maximum. That is a property of the
    arithmetic, not an assumption about the data, which is what makes a single scalar compare a
    sound bound for the whole row.

    KNOWN LIMIT, deliberately not enforced: numpy's uint64 cumsum is MODULAR, so that proviso
    -- and with it the whole O(1) argument -- holds only for nnz <= (2**64-1)//(2**32-1) =
    2**32+1 elements in ONE cell. Past that a wrapped accumulator could hide a large index
    behind a small ``cs[-1]``. Reaching it needs ~4 billion nonzeros in a single cell, and
    ``_unpack`` would try to allocate a 16 GB uint32 buffer for that row first. Stated rather
    than checked so the O(1) argument above rests on a named premise, not an unstated one.

    The searchsorted that recovers the FIRST offender (matching Rust, which fails on the first
    bad index it reaches) runs only on the failure path.
    """
    if cs.size and int(cs[-1]) >= n_vars:
        first = int(cs[int(np.searchsorted(cs, n_vars, side="left"))])
        raise ValueError(
            f"corrupt {codec_name} frame: column index {first} out of range [0, {n_vars})"
        )


def _out_size_error(nnz, out_indices, out_values):
    """Build the error for an out buffer that is not exactly ``nnz`` long (#244, Copilot #247 r4).

    ``np.copyto`` BROADCASTS, so a size-1 source silently fills a longer destination rather than
    raising -- verified: decode_into(frame, nnz=1, out of size 5) filled all five slots. Mismatched
    sizes above 1 do raise, so the hole is specifically the size-1 case, which is exactly the row a
    slicing bug is most likely to produce. Both call sites slice to exactly nnz today, so this
    guards the CONTRACT of a new low-level primitive rather than a live bug -- but silent
    corruption of a caller's preallocated array is the failure mode this whole issue is about.
    Two integer compares per row; unmeasurable next to the decode.
    """
    return ValueError(
        f"decode_into: output buffers must be exactly nnz={nnz} long, got "
        f"indices={out_indices.size}, values={out_values.size}"
    )


def _copy_via(out, src, via, scratch):
    """Write ``src`` into ``out``, preserving the legacy intermediate cast through ``via``.

    The old fallback cast through the ON-DISK dtype (uint32 scratch -> uint16 -> float32).
    Collapsing that into one copy is identical for every valid archive but differs on a
    corrupt one, where the uint16 step truncates -- and #174 was specifically about narrow-read
    tamper hardening. So apply the intermediate iff ``via`` is NARROWER than the destination;
    when it is not narrower the two-step and one-step results are bit-identical and the
    intermediate is elided.

    NOT a general-purpose casting rule. It is stated for exactly the dtype pairs this codebase
    produces -- pfordelta uint16/uint32 on disk against a float32/float64 output -- where the
    only way the intermediate can change a value is by truncating a wider unsigned integer. A
    same-width signed<->unsigned or float intermediate would need different reasoning, which is
    why this lives here next to the pfordelta decoder and not in a shared utility.

    ``scratch`` is a caller-owned buffer at ``via``, exactly ``out.size`` long, so the two-step
    path allocates nothing per row.

    KNOWN LIMIT, deliberately not fixed: for the triple (src uint64, via uint32, out float32) the
    itemsize rule elides the intermediate (4 < 4 is False), so a source value above 2**32 lands as
    ~4.29e9 instead of wrapping the way the legacy two-step did. That triple is UNREACHABLE here --
    the uint64 source is only ever the index cumsum, whose destination is int32/int64 in the gather
    and the decoder's own uint16/uint32 scratch in the bulk worker, never a float; the float
    destinations only ever receive the uint32 value stream, which cannot exceed 2**32. Widening the
    rule to cover it would add a memcpy per row to the live pfordelta-uint32-values -> float32 path
    for a case that cannot occur. If you ever call this with a float ``out`` on the INDEX path,
    revisit this first. (codex checkpoint-2 review, adjudicated.)
    """
    if via.itemsize < out.dtype.itemsize:
        np.copyto(scratch, src, casting="unsafe")
        np.copyto(out, scratch, casting="unsafe")
    else:
        np.copyto(out, src, casting="unsafe")


class _PforEncoder:
    def __init__(self, codec_name):
        pyfastpfor = _import_pyfastpfor()
        self._c = pyfastpfor.getCodec(codec_name)
        self._out = np.empty(4096, dtype=np.uint32)

    def _pack(self, arr_u32) -> bytes:
        n = arr_u32.size
        if n == 0:
            return b""
        cap = _pfor_encode_capacity(n)
        if self._out.size < cap:
            self._out = np.empty(cap, dtype=np.uint32)
        wl = self._c.encodeArray(arr_u32, n, self._out, self._out.size)
        return self._out[:wl].tobytes()

    def encode(self, indices, values) -> bytes:
        idx = np.ascontiguousarray(indices, dtype=np.uint32)
        gaps = idx.copy()
        if idx.size > 1:
            # check sortedness before subtracting: idx is uint32, so idx[1:] - idx[:-1] would
            # wrap on a decrease; the >= guard rejects that first, making the uint32 subtraction safe.
            if not np.all(idx[1:] >= idx[:-1]):
                raise ValueError("pfordelta requires non-decreasing (sorted) indices")
            gaps[1:] = idx[1:] - idx[:-1]
        idx_blob = self._pack(gaps)
        val_blob = self._pack(np.ascontiguousarray(values, dtype=np.uint32))
        return struct.pack("<I", len(idx_blob)) + idx_blob + val_blob


class _PforDecoder:
    decode_into_avoids_alloc = True  # _unpack writes into this decoder's own scratch

    def __init__(self, codec_name, index_dtype, value_dtype, n_vars):
        pyfastpfor = _import_pyfastpfor()
        self._c = pyfastpfor.getCodec(codec_name)
        self._idt = fmt.np_dtype(index_dtype)
        self._vdt = fmt.np_dtype(value_dtype)
        self._n_vars = int(n_vars)
        self._buf = np.empty(4096, dtype=np.uint32)
        # decode_into scratch: cumsum accumulator + per-dtype staging for the narrowing step.
        self._cs = np.empty(4096, dtype=np.uint64)
        self._si = np.empty(4096, dtype=self._idt)
        self._sv = np.empty(4096, dtype=self._vdt)

    def _grow(self, n):
        # Allocate into locals and publish all three together: a MemoryError on the 2nd or 3rd
        # allocation would otherwise leave _cs already enlarged while _si/_sv stayed short, and
        # the next call -- keyed on _cs.size -- would skip the grow and hand decode_into
        # undersized staging buffers. Guard on the MINIMUM so any partial state self-heals.
        # (codex checkpoint-2 review.)
        # Grow by DOUBLING, not to exactly n: strictly ascending row sizes (4097, 4098, ...)
        # would otherwise reallocate all three buffers on every single row. Measured on a
        # 3000-row ascending-nnz decode: 80.4 ms exact-growth vs 77.8 ms doubling. Costs at
        # most 2x a scratch that is 16 bytes per element of the LARGEST row, and a random row
        # order reallocs only O(log n) times either way. (Gemini #247 r2.)
        cur = min(self._cs.size, self._si.size, self._sv.size)
        if cur < n:
            m = max(cur * 2, n)
            cs = np.empty(m, dtype=np.uint64)
            si = np.empty(m, dtype=self._idt)
            sv = np.empty(m, dtype=self._vdt)
            self._cs, self._si, self._sv = cs, si, sv

    def _unpack(self, blob, n):
        if n == 0:
            return np.empty(0, dtype=np.uint32)
        if len(blob) == 0 or len(blob) % 4 != 0:
            raise ValueError("corrupt pfordelta frame: codec blob is empty or not 4-byte aligned")
        # #269 layer 2: cap the blob BEFORE sizing the buffer from it, so a corrupt x_offsets
        # that makes one frame span the whole payload cannot turn `n + len(blob)` into an
        # allocation DoS. Message text matches rust/src/pfor.rs decode_blob_u32.
        cap = _max_valid_blob_bytes(n)
        if len(blob) > cap:
            raise ValueError(
                f"corrupt pfordelta frame: codec blob is {len(blob)} bytes, "
                f"exceeds the maximum {cap} for nnz={n}"
            )
        # #269 layer 1: simdfastpfor256's VariableByte tail emits one u32 per varint in the
        # INPUT and ignores the declared capacity, so the reachable bound is n + len(blob)
        # values. Size the ALLOCATION to that; `n` stays the capacity handed to decodeArray, so
        # its own semantics and the `got != n` check below are unchanged.
        need = n + len(blob)
        if self._buf.size < need:
            self._buf = np.empty(need, dtype=np.uint32)
        words = np.frombuffer(blob, dtype=np.uint32)
        got = self._c.decodeArray(words, words.size, self._buf, n)
        if got != n:
            # decodeArray's return IS the number of values it actually produced. self._buf is
            # reused across every row and never zeroed, so returning self._buf[:n] after a short
            # decode hands back the PREVIOUS cell's tail as if it were this cell's data --
            # silently, and only on a corrupt/truncated frame, i.e. exactly when it must not.
            # A bare int compare, not a helper call: this runs once per ROW (#249).
            raise ValueError(f"corrupt pfordelta frame: decoded {got} values, expected {n}")
        return self._buf[:n]

    def decode(self, frame, nnz):
        if len(frame) < 4:
            raise ValueError("corrupt pfordelta frame: shorter than 4-byte length prefix")
        (idx_len,) = struct.unpack_from("<I", frame, 0)
        if 4 + idx_len > len(frame):
            raise ValueError("corrupt pfordelta frame: idx blob length exceeds frame")
        idx_blob = frame[4 : 4 + idx_len]
        val_blob = frame[4 + idx_len :]
        gaps = self._unpack(idx_blob, nnz)
        # cumsum copies out of the reused buffer before the next _unpack overwrites it
        cs = np.cumsum(gaps, dtype=np.uint64)
        # bounds-check the WIDE value: the .astype below would wrap an out-of-range column into
        # a legal one, and no downstream scan could then detect it (#250).
        _check_col_bounds(cs, self._n_vars, fmt.CODEC_PFORDELTA)
        indices = cs.astype(self._idt)
        # the pre-fix one-liner dropped the uint64 cumsum at the end of the .astype expression;
        # naming it would otherwise keep 8 bytes/nnz alive through the value decode below.
        del cs
        values = self._unpack(val_blob, nnz).astype(self._vdt)
        return indices, values

    def decode_into(self, frame, nnz, out_indices, out_values) -> None:
        """Decode one frame straight into caller-provided slices (#244 ask #3).

        Same framing and validation as :meth:`decode`; the difference is that nothing is
        allocated per row. ORDER MATTERS: ``_unpack`` reuses one scratch buffer, so the index
        cumsum must be copied out before the value ``_unpack`` overwrites it.
        """
        # Inlined rather than a helper call: this runs once per ROW, and a Python-level call
        # per row measured ~1 ms over a 4096-row gather. See _out_size_error for the why.
        if out_indices.size != nnz or out_values.size != nnz:
            raise _out_size_error(nnz, out_indices, out_values)
        if len(frame) < 4:
            raise ValueError("corrupt pfordelta frame: shorter than 4-byte length prefix")
        (idx_len,) = struct.unpack_from("<I", frame, 0)
        if 4 + idx_len > len(frame):
            raise ValueError("corrupt pfordelta frame: idx blob length exceeds frame")
        if nnz == 0:
            # Validation above already ran -- that is the point. decode() validates every frame
            # including empty ones today, so the CALLER must not skip this method for nnz==0.
            return
        self._grow(nnz)
        gaps = self._unpack(frame[4 : 4 + idx_len], nnz)
        np.cumsum(gaps, dtype=np.uint64, out=self._cs[:nnz])
        # Bound the WIDE cumsum before _copy_via narrows it, exactly as decode() does (#250).
        # decode_into is the pure-Python gather's default path after #244, so omitting this
        # reinstated the bug #250 fixed: an out-of-range column reached the CSR unreported.
        # It MUST be checked here, on uint64, not after narrowing -- that is the whole point of
        # #250: the narrowing step is what destroys the evidence.
        _check_col_bounds(self._cs[:nnz], self._n_vars, fmt.CODEC_PFORDELTA)
        _copy_via(out_indices, self._cs[:nnz], self._idt, self._si[:nnz])
        _copy_via(out_values, self._unpack(frame[4 + idx_len :], nnz), self._vdt, self._sv[:nnz])


class PforDeltaCodec:
    """delta+PFOR indices, PFOR values (pyfastpfor). No trained dictionary."""

    name = fmt.CODEC_PFORDELTA

    def __init__(self, index_dtype, value_dtype, codec_name=fmt.PYFASTPFOR_CODEC):
        self.index_dtype, self.value_dtype, self._cname = index_dtype, value_dtype, codec_name

    def new_encoder(self):
        return _PforEncoder(self._cname)

    def new_decoder(self, n_vars):
        return _PforDecoder(self._cname, self.index_dtype, self.value_dtype, n_vars)


class _ByteZstdEncoder:
    """byteshuffle+bytedelta+zstd indices; zstd-only floats / byteshuffle+zstd int values.

    Reuses the shard v2 primitives verbatim so the frames are byte-identical to the
    Rust port (cell_zstd.rs). Indices AND integer values are always uint32 on disk; the
    value branch is chosen from ``value_dtype`` (float32/float64 -> zstd_only at that
    width; else byteshuffle+zstd of uint32). For integers ``value_dtype`` is purely the
    decoder's output ``.astype`` cast, never the on-disk width: an integer ``value_dtype``
    other than uint32 round-trips for representable values and truncates larger ones
    low-bits-first -- exactly like the always-uint32 index output cast -- instead of
    corrupting (#223). Lossless-narrow enforcement, when wanted, is a reader-level concern
    (cf. cast_data / #218 wide_fits), not the codec's."""

    def __init__(self, value_dtype):
        self._vdt = fmt.np_dtype(value_dtype)
        self._value_is_float = self._vdt.kind == "f"

    def encode(self, indices, values) -> bytes:
        idx = np.ascontiguousarray(indices, dtype=np.uint32)
        if idx.size > 1 and not np.all(idx[1:] >= idx[:-1]):
            raise ValueError("zstd codec requires non-decreasing (sorted) indices")
        idx_blob = byte_filter_encode(idx, role="indices")
        if self._value_is_float:
            # float on-disk width == value_dtype (float32/float64); zstd-only (no shuffle).
            vals = np.ascontiguousarray(values, dtype=self._vdt)
            val_blob = zstd_only_encode(vals)
        else:
            # #223: on-disk integer values are ALWAYS uint32 (symmetric with the decoder,
            # which reads uint32 then .astype(value_dtype), and with the always-uint32 index
            # path). value_dtype is purely the OUTPUT cast for integers, never the on-disk
            # width -- byteshuffle crushes the zero high planes so a narrower on-disk width
            # buys nothing, and a mismatch here vs the uint32 decode corrupts the round-trip.
            vals = np.ascontiguousarray(values, dtype=np.uint32)
            val_blob = byte_filter_encode(vals, role="data")
        return struct.pack("<I", len(idx_blob)) + idx_blob + val_blob


class _ByteZstdDecoder:
    decode_into_avoids_alloc = False  # byte_filter_decode/zstd_only_decode allocate their own

    def __init__(self, index_dtype, value_dtype):
        self._idt = fmt.np_dtype(index_dtype)
        self._vdt = fmt.np_dtype(value_dtype)
        self._value_is_float = self._vdt.kind == "f"

    def decode(self, frame, nnz):
        if len(frame) < 4:
            raise ValueError("corrupt zstd frame: shorter than 4-byte length prefix")
        (idx_len,) = struct.unpack_from("<I", frame, 0)
        if 4 + idx_len > len(frame):
            raise ValueError("corrupt zstd frame: idx blob length exceeds frame")
        idx_blob = frame[4 : 4 + idx_len]
        val_blob = frame[4 + idx_len :]
        indices = byte_filter_decode(
            idx_blob, dtype=np.uint32, n_elements=nnz, role="indices"
        ).astype(self._idt, copy=False)
        if self._value_is_float:
            values = zstd_only_decode(val_blob, dtype=self._vdt, n_elements=nnz)
        else:
            values = byte_filter_decode(
                val_blob, dtype=np.uint32, n_elements=nnz, role="data"
            ).astype(self._vdt, copy=False)
        return indices, values

    def decode_into(self, frame, nnz, out_indices, out_values) -> None:
        """Decode one frame straight into caller-provided slices (#244 ask #3).

        byte_filter_decode / zstd_only_decode allocate their own output, so the saving here is
        the removal of the whole-block concatenate + astype in the caller, not a per-row one.
        """
        # Inlined rather than a helper call: this runs once per ROW, and a Python-level call
        # per row measured ~1 ms over a 4096-row gather. See _out_size_error for the why.
        if out_indices.size != nnz or out_values.size != nnz:
            raise _out_size_error(nnz, out_indices, out_values)
        indices, values = self.decode(frame, nnz)
        np.copyto(out_indices, indices, casting="unsafe")
        np.copyto(out_values, values, casting="unsafe")


class ByteZstdCodec:
    """byteshuffle/bytedelta+zstd indices, zstd-only floats / byteshuffle+zstd ints.

    Same frame framing + interface as :class:`PforDeltaCodec`; gives layout=cell float
    storage and a smaller integer option. On-disk indices and integer values are always
    uint32; ``index_dtype`` / integer ``value_dtype`` are the decoder's output casts. Float
    ``value_dtype`` (float32/float64) is both the on-disk and output width.

    ``new_decoder`` takes ``n_vars`` for signature parity with :class:`PforDeltaCodec` and
    drops it -- ``_ByteZstdDecoder`` itself never sees it. On-disk zstd indices are always
    uint32, so the decoder's ``.astype`` never narrows and an out-of-range column cannot WRAP
    the way #250 describes for pfordelta -- but it does survive unchanged, so zstd is bounded
    one level up, by the aggregate scan in ``reader.gather_rows`` and by
    ``_csr_direct_assign(check_indices=True)`` on the bulk path. A per-row check here was
    measured at ~4x the aggregate scan for the same coverage and rejected; the cost is that an
    out-of-range zstd column is reported by the reader rather than per-frame the way Rust
    reports it."""

    name = fmt.CODEC_ZSTD

    def __init__(self, index_dtype, value_dtype):
        self.index_dtype, self.value_dtype = index_dtype, value_dtype

    def new_encoder(self):
        return _ByteZstdEncoder(self.value_dtype)

    def new_decoder(self, n_vars):  # noqa: ARG002 - see the class docstring
        return _ByteZstdDecoder(self.index_dtype, self.value_dtype)


def codec_for(name, index_dtype, value_dtype):
    """Build a Codec for the given codec name + dtypes (read: from manifest; write: params)."""
    if name == fmt.CODEC_PFORDELTA:
        return PforDeltaCodec(index_dtype, value_dtype)
    if name == fmt.CODEC_ZSTD:
        return ByteZstdCodec(index_dtype, value_dtype)
    raise ValueError(f"unsupported codec {name!r}; supported: {sorted(fmt.SUPPORTED_CODECS)}")
