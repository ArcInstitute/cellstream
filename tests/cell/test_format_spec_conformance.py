"""Conformance gate for docs/format.md.

A reader implemented from the SPEC TEXT ALONE must recover the same matrix the library
does. Nothing below imports cellstream's container or codec helpers on the parse side --
``_SpecReader`` is written against docs/format.md and re-derives the framing, the byte
filters and the delta/cumsum arithmetic itself. If the two disagree, either the code
changed without the document or the document was wrong; both are bugs.

Section references in the assertions point at docs/format.md.

The negative controls at the bottom are what keep the positive tests honest: each one
damages a specific byte the spec calls load-bearing and asserts the LIBRARY refuses the
archive. A green positive test with no negative control cannot tell "the invariant holds"
from "nothing checks it".
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zstandard as zstd

import cellstream
from cellstream.errors import IncompatibleSchemaError, PackedArchiveError
from cellstream.write import write_sharded

# --------------------------------------------------------------------------- spec reader

HEAD_MAGIC = b"SHPK"  # sec 2.1
TRAILER_MAGIC = b"SHPKEND\x00"  # sec 2.2
HEAD_BLOCK = 4096
ALIGNMENT = 4096
TRAILER_SIZE = 64
_TRAILER = struct.Struct("<8sHHQQQI")  # sec 2.2: magic ver res foff flen sre crc

# Sec 1: the format is little-endian, so a reader written from the document uses EXPLICIT LE
# dtypes rather than the native ones the reference implementation happens to use. On x86_64 the
# two agree, which is exactly why this must be spelled out: a native-dtype reader would agree
# with a native-dtype writer on any host, including one where both are wrong.
_LE = {"uint32": "<u4", "float32": "<f4", "float64": "<f8"}


class _SpecReader:
    """docs/format.md, sections 2-9, implemented from the document."""

    def __init__(self, path):
        self.raw = path.read_bytes()
        raw = self.raw
        # sec 2.1 -- head block
        assert raw[:4] == HEAD_MAGIC
        (self.head_version,) = struct.unpack_from("<H", raw, 4)
        assert raw[6:HEAD_BLOCK] == b"\x00" * (HEAD_BLOCK - 6)
        # sec 2.2 -- trailer
        tr = raw[-TRAILER_SIZE:]
        magic, ver, res, foff, flen, sre, crc = _TRAILER.unpack(tr[: _TRAILER.size])
        assert magic == TRAILER_MAGIC
        assert tr[_TRAILER.size :] == b"\x00" * (TRAILER_SIZE - _TRAILER.size)
        # `res` is DELIBERATELY not asserted: sec 2.2 says readers MUST ignore the reserved
        # field, because rejecting a nonzero value would make the first extension that uses it
        # unreadable by every existing reader. That writers emit 0 is asserted separately, in
        # test_container_invariants. (codex, checkpoint 2 round 2.)
        self.reserved = res
        assert foff >= HEAD_BLOCK and foff + flen <= len(raw) - TRAILER_SIZE
        self.trailer_version, self.footer_offset, self.footer_length = ver, foff, flen
        self.shard_region_end = sre
        # sec 2.3 -- footer, CRC-verified
        fbuf = raw[foff : foff + flen]
        assert zlib.crc32(fbuf) & 0xFFFFFFFF == crc, "footer CRC mismatch"
        self.footer = json.loads(fbuf.decode("utf-8"))
        self.members = {m["name"]: m for m in self.footer["members"]}
        self.member_order = [m["name"] for m in self.footer["members"]]
        # sec 6
        self.manifest = json.loads(self.member("manifest.json"))
        assert self.manifest["layout"] == "cell"
        assert self.manifest["schema_version"] in (5, 6)
        self.n_obs = int(self.manifest["n_obs_total"])
        self.n_vars = int(self.manifest["n_vars"])
        # sec 4 -- raw little-endian i64, n_obs+1 each
        self.offsets = np.frombuffer(self.member("x_offsets"), dtype="<i8")
        self.indptr = np.frombuffer(self.member("x_indptr"), dtype="<i8")
        self.payload = self.member("x_payload")

    def member(self, name) -> bytes:
        m = self.members[name]
        return self.raw[m["offset"] : m["offset"] + m["length"]]

    def has(self, name) -> bool:
        return name in self.members

    # sec 5.2 -- byte filters
    @staticmethod
    def _un_byte_shuffle(planes: np.ndarray, dtype) -> np.ndarray:
        return np.frombuffer(np.ascontiguousarray(planes.T).tobytes(), dtype=dtype)

    @staticmethod
    def _un_byte_delta(planes: np.ndarray) -> np.ndarray:
        return np.cumsum(planes, axis=1, dtype=np.uint8)  # u8 wraps mod 256

    def _zstd(self, blob: bytes, *, expect: int) -> bytes:
        """Sec 5.2, all four bounds -- including the no-embedded-size case.

        Requiring an embedded content size would narrow the contract: the spec says a frame
        WITHOUT one is externally produced or corrupt and must be bounded by max_output_size,
        not rejected. Our own compressor always embeds it, so this arm is unreachable for an
        archive we wrote -- which is why asserting equality looked fine and was still wrong.
        """
        assert expect <= 64 << 20, "sec 5.2: chunk above the 64 MiB cap"
        declared = zstd.frame_content_size(blob)
        unknown = (-1, getattr(zstd, "CONTENTSIZE_UNKNOWN", 2**64 - 1))
        if declared in unknown:
            # max(1, ...) because python-zstandard reads max_output_size=0 as "no maximum
            # supplied" and then refuses a frame with no embedded size -- so the expect==0 case
            # (an empty row's blob) would raise instead of decoding. The exact-length assertion
            # below is what actually bounds the result. (codex, checkpoint 2 round 2.)
            out = zstd.ZstdDecompressor().decompress(blob, max_output_size=max(1, expect))
        else:
            assert declared == expect, f"sec 5.2: frame declares {declared}, expected {expect}"
            out = zstd.ZstdDecompressor().decompress(blob)
        assert len(out) == expect
        return out

    # sec 5.1
    def _pfor(self, blob: bytes, n: int) -> np.ndarray:
        import pyfastpfor

        if n == 0:
            return np.zeros(0, np.uint32)
        assert len(blob) and len(blob) % 4 == 0, "sec 5.1: blob empty or not 4-byte aligned"
        assert len(blob) <= 8 * n + 65536, "sec 10.16: blob above the loose bound"
        codec = pyfastpfor.getCodec("simdfastpfor256")
        # The blob words are LE on disk; pyfastpfor needs a NATIVE uint32 buffer, so convert at
        # that boundary rather than reading native bytes off the file (sec 1).
        words = np.ascontiguousarray(np.frombuffer(blob, dtype=_LE["uint32"]), dtype=np.uint32)
        out = np.empty(n + len(blob), np.uint32)  # sec 5.1: VariableByte output bound
        got = codec.decodeArray(words, words.size, out, n)
        assert got == n, f"sec 10.17: decoded {got}, expected {n}"
        return out[:n].copy()

    # sec 9
    def read_row(self, r: int):
        frame = self.payload[self.offsets[r] : self.offsets[r + 1]]
        nnz = int(self.indptr[r + 1] - self.indptr[r])
        assert len(frame) >= 4, "sec 10.14: frame shorter than the length prefix"
        (idx_len,) = struct.unpack_from("<I", frame, 0)
        assert 4 + idx_len <= len(frame), "sec 10.14: idx_len exceeds the frame"
        idx_blob, val_blob = frame[4 : 4 + idx_len], frame[4 + idx_len :]
        if self.manifest["codec"] == "pfordelta":
            gaps = self._pfor(idx_blob, nnz)
            cols = np.cumsum(gaps, dtype=np.uint64)
            assert not cols.size or int(cols[-1]) < self.n_vars, "sec 10.18: column out of range"
            vals = self._pfor(val_blob, nnz)
        else:
            planes = np.frombuffer(self._zstd(idx_blob, expect=4 * nnz), np.uint8)
            cols = self._un_byte_shuffle(self._un_byte_delta(planes.reshape(4, nnz)), _LE["uint32"])
            vdt = np.dtype(_LE[self.manifest["value_dtype_on_disk"]])
            if vdt.kind == "f":
                vals = np.frombuffer(self._zstd(val_blob, expect=vdt.itemsize * nnz), vdt)
            else:  # sec 5.2: integer values are always u32 on disk, no byte-delta
                vp = np.frombuffer(self._zstd(val_blob, expect=4 * nnz), np.uint8)
                vals = self._un_byte_shuffle(vp.reshape(4, nnz), _LE["uint32"])
        assert len(cols) == len(vals) == nnz
        return cols, vals

    def to_csr(self) -> sp.csr_matrix:
        cols, vals = [], []
        for r in range(self.n_obs):
            c, v = self.read_row(r)
            cols.append(c.astype(np.int64))
            vals.append(v.astype(np.float64))
        ind = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        dat = np.concatenate(vals) if vals else np.zeros(0, np.float64)
        return sp.csr_matrix(
            (dat, ind, self.indptr.astype(np.int64)), shape=(self.n_obs, self.n_vars)
        )


# ------------------------------------------------------------------------------ fixtures

N_OBS, N_VARS = 40, 300


def _counts(seed=3, n_obs=N_OBS, n_vars=N_VARS, density=0.08):
    d = sp.random(n_obs, n_vars, density=density, format="csr", random_state=seed)
    x = sp.csr_matrix((np.ceil(d.data * 50).astype("int32"), d.indices, d.indptr), shape=d.shape)
    x.eliminate_zeros()
    return x


def _floats(seed=5, n_obs=N_OBS, n_vars=N_VARS, density=0.08):
    """float64 values that do NOT survive a float32 round trip.

    An earlier version narrowed to float32 first, so the float64 case could not have detected
    precision loss anywhere in the chain -- the values it compared were float32-representable to
    begin with. `x + 1/3` at ~1e0 needs more than 24 bits of mantissa, so a float32 detour shows
    up as a mismatch. Kept as float64 here; the float32 CASES entry casts on the way in, which is
    what the writer would do anyway. (codex, checkpoint 2.)
    """
    d = sp.random(n_obs, n_vars, density=density, format="csr", random_state=seed)
    data = d.data.astype(np.float64) * 3.7 + (1.0 / 3.0)
    return sp.csr_matrix((data, d.indices, d.indptr), shape=d.shape)


def _floats32(seed=5, n_obs=N_OBS, n_vars=N_VARS, density=0.08):
    x = _floats(seed, n_obs, n_vars, density)
    return sp.csr_matrix((x.data.astype(np.float32), x.indices, x.indptr), shape=x.shape)


def _wide_counts(seed=11, n_obs=N_OBS, n_vars=70_000, density=0.002):
    """n_vars > 65536 AND values > 65535, so pfordelta picks uint32 for BOTH output casts.

    Without this every pfordelta case ran at uint16/uint16, so the on-disk-dtype-is-only-an-
    output-cast claim was only ever exercised in its narrow form.
    """
    d = sp.random(n_obs, n_vars, density=density, format="csr", random_state=seed)
    data = (d.data * 200_000).astype("int64") + 70_000
    x = sp.csr_matrix((data, d.indices, d.indptr), shape=d.shape)
    x.eliminate_zeros()
    return x


def _adata(x, *, grouped=False):
    obs = pd.DataFrame(index=[f"c{i}" for i in range(x.shape[0])])
    if grouped:
        rng = np.random.default_rng(7)
        obs["grp"] = rng.choice(["ctrl", "a", "b"], x.shape[0])
    return ad.AnnData(
        X=x.copy(), obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(x.shape[1])])
    )


_GROUPED = {"group_by": "grp", "reference": "ctrl"}

CASES = {
    # name: (X builder, write kwargs, expected schema_version, expected on-disk value dtype)
    # Both codecs x {narrow ints, wide ints, float32, float64} x {ungrouped, grouped}, minus the
    # combinations the writer cannot produce (pfordelta stores integers only).
    "pfordelta": (_counts, {"codec": "pfordelta"}, 5, "uint16"),
    "pfordelta_grouped": (_counts, {"codec": "pfordelta", **_GROUPED}, 5, "uint16"),
    "pfordelta_wide": (_wide_counts, {"codec": "pfordelta"}, 5, "uint32"),
    "pfordelta_wide_grouped": (_wide_counts, {"codec": "pfordelta", **_GROUPED}, 5, "uint32"),
    "zstd_int": (_counts, {"codec": "zstd"}, 6, "uint32"),
    "zstd_int_wide": (_wide_counts, {"codec": "zstd"}, 6, "uint32"),
    "zstd_int_wide_grouped": (_wide_counts, {"codec": "zstd", **_GROUPED}, 6, "uint32"),
    "zstd_grouped": (_counts, {"codec": "zstd", **_GROUPED}, 6, "uint32"),
    "zstd_float32": (_floats32, {"codec": "zstd", "data_dtype": "float32"}, 6, "float32"),
    "zstd_float64": (_floats, {"codec": "zstd", "data_dtype": "float64"}, 6, "float64"),
    "zstd_float32_grouped": (
        _floats32,
        {"codec": "zstd", "data_dtype": "float32", **_GROUPED},
        6,
        "float32",
    ),
    "zstd_float64_grouped": (
        _floats,
        {"codec": "zstd", "data_dtype": "float64", **_GROUPED},
        6,
        "float64",
    ),
}


def _write(tmp_path, name):
    build, kwargs, _, _ = CASES[name]
    p = tmp_path / f"{name}.csad"
    write_sharded(
        _adata(build(), grouped="group_by" in kwargs), p, layout="cell", overwrite=True, **kwargs
    )
    return p


# ------------------------------------------------------------------- the conformance check


@pytest.mark.parametrize("name", sorted(CASES))
def test_spec_reader_matches_the_library(tmp_path, name):
    """A reader built from docs/format.md alone decodes what the library decodes."""
    path = _write(tmp_path, name)
    spec = _SpecReader(path)
    _, _, want_sv, want_vdt = CASES[name]
    assert spec.manifest["schema_version"] == want_sv
    assert spec.manifest["value_dtype_on_disk"] == want_vdt

    store = cellstream.open(path)
    try:
        # gather_rows indexes STORAGE rows -- the same rows the spec reader walks (sec 1).
        lib = store.gather_rows(np.arange(store.n_obs))
    finally:
        store.close()
    mine = spec.to_csr()
    assert mine.shape == lib.shape
    # float64 both sides: casting the library's result to float64 is lossless from float32 and
    # exact from float64, so the comparison can be EXACT rather than tolerance-based. The
    # float64 fixture carries values that are not float32-representable, so a float32 detour
    # anywhere in either chain shows up here as a nonzero difference.
    if want_vdt == "float64":
        assert lib.dtype == np.float64, f"{name}: library returned {lib.dtype}, not float64"
    delta = abs(mine - sp.csr_matrix(lib, dtype="float64"))
    assert delta.nnz == 0, f"{name}: spec decode differs from the library at {delta.nnz} entries"


@pytest.mark.parametrize("name", sorted(CASES))
def test_container_invariants(tmp_path, name):
    """Sections 2.3, 2.4 and 3: roles, alignment, NUL padding, region boundary."""
    spec = _SpecReader(_write(tmp_path, name))
    assert spec.head_version == spec.trailer_version == 1
    assert spec.reserved == 0, "writers emit 0 in the reserved trailer field (sec 2.2)"
    assert spec.footer["container_version"] == 1
    assert spec.footer["alignment"] == ALIGNMENT
    # sec 3 -- the three data members come first, in this order, and no shard member exists
    assert spec.member_order[:3] == ["x_payload", "x_offsets", "x_indptr"]
    roles = [m["role"] for m in spec.footer["members"]]
    assert roles[:3] == ["payload", "offsets", "indptr"]
    assert "shard" not in roles
    for req in ("x_payload", "x_offsets", "x_indptr", "header.h5ad", "obs.parquet"):
        assert spec.has(req), req
    # sec 2.4 -- every member 4 KB-aligned; every gap NUL; footer abuts the trailer unaligned
    ordered = sorted(spec.footer["members"], key=lambda m: m["offset"])
    prev_end = HEAD_BLOCK
    for m in ordered:
        assert m["offset"] % ALIGNMENT == 0, m["name"]
        assert m["offset"] >= prev_end, m["name"]
        assert set(spec.raw[prev_end : m["offset"]]) <= {0}, f"non-NUL pad before {m['name']}"
        prev_end = m["offset"] + m["length"]
    assert set(spec.raw[prev_end : spec.footer_offset]) <= {0}
    assert spec.footer_offset + spec.footer_length + TRAILER_SIZE == len(spec.raw)
    # sec 2.4 -- shard_region_end is the first non-immutable member, and the two copies agree
    first_meta = next(m for m in ordered if m["role"] not in ("payload", "offsets", "indptr"))
    assert spec.shard_region_end == first_meta["offset"] == spec.footer["shard_region_end"]
    assert spec.shard_region_end % ALIGNMENT == 0


@pytest.mark.parametrize("name", sorted(CASES))
def test_offsets_and_indptr(tmp_path, name):
    """Section 4: sizes, monotonicity, and the payload-covering identity."""
    spec = _SpecReader(_write(tmp_path, name))
    n = spec.n_obs
    for arr, who in ((spec.offsets, "x_offsets"), (spec.indptr, "x_indptr")):
        assert arr.size == n + 1, who
        assert spec.members[who]["length"] == 8 * (n + 1), who
        assert arr[0] == 0, who
        assert np.all(arr[1:] >= arr[:-1]), who
    assert int(spec.offsets[-1]) == len(spec.payload) == spec.members["x_payload"]["length"]
    assert int(spec.indptr[-1]) == int(spec.manifest["total_nnz"])


@pytest.mark.parametrize("name", sorted(CASES))
def test_indices_sorted_within_each_row(tmp_path, name):
    """Section 5: column indices are ascending within a row."""
    spec = _SpecReader(_write(tmp_path, name))
    for r in range(spec.n_obs):
        cols, _ = spec.read_row(r)
        assert np.all(cols[1:] >= cols[:-1]), f"row {r} is not sorted"
        assert not cols.size or int(cols.max()) < spec.n_vars


def test_pfordelta_ignores_the_on_disk_dtype_fields(tmp_path):
    """Section 5.1 / 6.2: pfordelta streams are physically u32; *_on_disk is an output cast.

    ``_SpecReader._pfor`` never reads ``value_dtype_on_disk``. The manifest here says
    ``uint16``, so if that were the physical width the decode above would be wrong.
    """
    path = _write(tmp_path, "pfordelta")
    spec = _SpecReader(path)
    assert spec.manifest["value_dtype_on_disk"] == "uint16"
    assert spec.manifest["index_dtype_on_disk"] == "uint16"
    store = cellstream.open(path)
    try:
        lib = store.gather_rows(np.arange(store.n_obs))
    finally:
        store.close()
    assert abs(spec.to_csr() - sp.csr_matrix(lib, dtype="float64")).nnz == 0


@pytest.mark.parametrize(
    ("codec", "kwargs", "want_size"),
    # pfordelta's 4 IS normative: packing zero values yields an empty blob, whatever the FastPFor
    # build. zstd's 22 is NOT -- sec 5.3 says so explicitly, since it is two copies of whatever
    # this zstd emits for b"". So the zstd case asserts only the framing minimum and that the row
    # decodes to nothing. (codex, checkpoint 2.)
    [("pfordelta", {"codec": "pfordelta"}, 4), ("zstd", {"codec": "zstd"}, None)],
)
def test_empty_row_frame_size(tmp_path, codec, kwargs, want_size):
    """Section 5.3: an nnz==0 row still has a frame; its size is codec-dependent."""
    x = _counts().tolil()
    x[3, :] = 0
    x[7, :] = 0
    x = x.tocsr()
    x.eliminate_zeros()
    p = tmp_path / f"empty_{codec}.csad"
    write_sharded(_adata(x), p, layout="cell", overwrite=True, **kwargs)
    spec = _SpecReader(p)
    empties = [r for r in range(spec.n_obs) if spec.indptr[r + 1] == spec.indptr[r]]
    assert empties == [3, 7]
    for r in empties:
        span = int(spec.offsets[r + 1] - spec.offsets[r])
        if want_size is None:
            assert span >= 4, f"row {r}: {span} bytes cannot hold the u32 length prefix"
        else:
            assert span == want_size, f"row {r}: {span} bytes, expected {want_size}"
        # _SpecReader validates the prefix even for an empty row -- permitted, not required:
        # sec 5.3 records that the pure-Python decoder does and the Rust decoder does not. What
        # IS required is that nnz comes from x_indptr, so an empty row yields empty arrays.
        cols, vals = spec.read_row(r)
        assert cols.size == vals.size == 0
    store = cellstream.open(p)
    try:
        lib = store.gather_rows(np.arange(store.n_obs))
    finally:
        store.close()
    assert abs(spec.to_csr() - sp.csr_matrix(lib, dtype="float64")).nnz == 0


def test_payload_sha256_covers_exactly_the_payload_member(tmp_path):
    """Section 6.4: the digest is over the x_payload member bytes, not the file."""
    p = tmp_path / "checksum.csad"
    write_sharded(
        _adata(_counts()), p, layout="cell", codec="zstd", payload_checksum=True, overwrite=True
    )
    spec = _SpecReader(p)
    want = spec.manifest["payload_sha256"]
    assert isinstance(want, str) and len(want) == 64
    assert hashlib.sha256(spec.payload).hexdigest() == want
    assert hashlib.sha256(spec.raw).hexdigest() != want  # not the whole file
    store = cellstream.open(p)
    try:
        assert store.verify_payload() is True
    finally:
        store.close()


def test_default_write_stores_no_checksum(tmp_path):
    """Section 6.4: null means 'nothing to verify', and it is the default."""
    spec = _SpecReader(_write(tmp_path, "zstd_int"))
    assert spec.manifest["payload_sha256"] is None


@pytest.mark.parametrize("name", ["pfordelta_grouped", "zstd_grouped"])
def test_groups_json_shape_and_invariants(tmp_path, name):
    """Section 8: object (not array), tiling, and the reference leading prefix."""
    spec = _SpecReader(_write(tmp_path, name))
    assert spec.manifest["group_by"] == "grp"
    assert spec.has("groups.json")
    rec = json.loads(spec.member("groups.json"))
    assert isinstance(rec, dict) and isinstance(rec["groups"], list)
    prev = 0
    for g in rec["groups"]:
        assert isinstance(g["label"], str)
        assert isinstance(g["start"], int) and not isinstance(g["start"], bool)
        assert isinstance(g["stop"], int) and not isinstance(g["stop"], bool)
        assert 0 <= g["start"] < g["stop"] <= spec.n_obs
        assert g["start"] == prev, "records must tile [0, n_obs) in storage order"
        prev = g["stop"]
    assert prev == spec.n_obs
    # reference: agrees with the manifest and forms a leading prefix
    ref = rec["reference"]
    assert ref == spec.manifest["reference"] == "ctrl"
    hit = [i for i, g in enumerate(rec["groups"]) if g["label"] == ref]
    assert hit == list(range(len(hit))) and hit, "reference records must lead"
    # the reference span really is the archive's first rows
    assert rec["groups"][0]["start"] == 0


def test_ungrouped_archive_has_no_groups_member(tmp_path):
    """Section 6.5 / 8: group_by null <=> groups.json absent."""
    spec = _SpecReader(_write(tmp_path, "zstd_int"))
    assert spec.manifest["group_by"] is None
    assert not spec.has("groups.json")


def test_header_h5ad_is_a_real_h5ad_with_an_empty_x(tmp_path):
    """Section 7: h5ad with an empty CSR X; obs/var present; storage-order obs."""
    import io

    import h5py
    from anndata.io import read_elem

    spec = _SpecReader(_write(tmp_path, "pfordelta_grouped"))
    with h5py.File(io.BytesIO(spec.member("header.h5ad")), "r") as f:
        assert f.attrs["encoding-type"] == "anndata"
        assert f["X/data"].shape == (0,) and f["X/indices"].shape == (0,)
        assert f["X/indptr"].shape == (spec.n_obs + 1,)
        # obs/var go through anndata's read_elem, NOT h5py slicing. The `_index` element is a
        # plain dataset on some anndata versions and a nullable-string-array GROUP on others --
        # CI's 3.12 job hits the group form, where `f["obs/_index"].shape` raises
        # AttributeError: 'Group' object has no attribute 'shape'. read_elem decodes both, and is
        # the encoding-agnostic reader the format's own header member is written by. (The X
        # datasets above are the CSR encoding, which is stable, so they stay direct.)
        header_obs = read_elem(f["obs"])
        header_var = read_elem(f["var"])
        assert len(header_obs) == spec.n_obs
        assert len(header_var) == spec.n_vars
        header_obs_names = list(header_obs.index)
    obs = pd.read_parquet(io.BytesIO(spec.member("obs.parquet")))
    assert len(obs) == spec.n_obs
    assert list(obs.index) == header_obs_names, "obs.parquet and header.h5ad must agree"
    # storage order, not input order: the reference group leads
    assert obs["grp"].iloc[0] == "ctrl"


def test_every_writer_satisfies_the_open_time_invariants(tmp_path):
    """The other side of the open-time validation: no VALID archive may be rejected.

    Section 4 / 10.5-10.7 are enforced at open now, so every writer in the package has to
    satisfy them -- including the two that do not go through write_cell_archive's encode loop.
    `concat_cell_archives` byte-copies frames and rebuilds both sidecars; `CellArchiveWriter`
    stages frames and reorders them at finalize; and an empty archive is the degenerate case
    where every array has length 1. Checked here rather than trusted, because a check that
    rejects real output is worse than the gap it closed.
    """
    from cellstream.cell.archive_writer import CellArchiveWriter, concat_cell_archives

    built = {}
    x, a = _counts(), _adata(_counts(), grouped=True)
    built["one-shot"] = _write(tmp_path, "pfordelta")
    built["one-shot grouped"] = _write(tmp_path, "pfordelta_grouped")
    built["one-shot zstd"] = _write(tmp_path, "zstd_int")

    push = tmp_path / "push.csad"
    w = CellArchiveWriter(push, n_vars=x.shape[1], codec="pfordelta", value_dtype="uint16")
    w.add_block(sp.csr_matrix(x[:15]))
    w.add_block(sp.csr_matrix(x[15:]))
    w.finalize(obs=a.obs, var=a.var)
    built["push"] = push

    cat = tmp_path / "concat.csad"
    concat_cell_archives([built["one-shot"], push], cat, overwrite=True)
    built["concat"] = cat

    empty = tmp_path / "empty.csad"
    write_sharded(
        ad.AnnData(
            X=sp.csr_matrix((0, x.shape[1]), dtype="uint16"),
            obs=pd.DataFrame(index=[]),
            var=a.var,
        ),
        empty,
        layout="cell",
        codec="pfordelta",
        overwrite=True,
    )
    built["empty"] = empty

    for label, path in built.items():
        spec = _SpecReader(path)
        pay_len = spec.members["x_payload"]["length"]
        for arr, who in ((spec.offsets, "x_offsets"), (spec.indptr, "x_indptr")):
            assert arr.size == spec.n_obs + 1, f"{label}: {who} size"
            assert not arr.size or int(arr[0]) == 0, f"{label}: {who}[0]"
            assert arr.size < 2 or np.all(arr[1:] >= arr[:-1]), f"{label}: {who} monotonicity"
        assert not spec.indptr.size or int(spec.indptr[-1]) == int(spec.manifest["total_nnz"]), (
            f"{label}: indptr endpoint vs total_nnz"
        )
        assert not spec.offsets.size or int(spec.offsets[-1]) == pay_len, (
            f"{label}: offsets endpoint vs x_payload length"
        )
        assert spec.shard_region_end == spec.footer["shard_region_end"], f"{label}: sre copies"
        store = cellstream.open(path)  # must not raise
        try:
            assert store.n_obs == spec.n_obs
        finally:
            store.close()


def test_cell_metadata_is_immutable(tmp_path):
    """Section 2.4: the tail-rewrite editing path refuses a cell archive rather than dropping
    its data members, and CellStore exposes no editing surface at all.

    The rewrite re-emits members by role and by a fixed list of metadata names; `payload` /
    `offsets` / `indptr` are in neither set, so a successful rewrite would delete them and
    unlink the .tailbak that could restore them. The guard is what makes that unreachable.
    """
    from cellstream.packed.writer import update_obs_packed

    path = _write(tmp_path, "pfordelta")
    store = cellstream.open(path)
    try:
        new_obs = store.obs.copy()
        assert not hasattr(store, "update_obs")
        assert not hasattr(store, "update_var")
    finally:
        store.close()
    new_obs["added"] = 1
    before = path.read_bytes()
    with pytest.raises(NotImplementedError, match="x_payload"):
        update_obs_packed(path, new_obs)
    assert path.read_bytes() == before, "the refused update must not have touched the archive"


def test_appendix_a_numbers(tmp_path):
    """Appendix A: the structural numbers of the worked example."""
    spec = _SpecReader(_write(tmp_path, "pfordelta_grouped"))
    assert (spec.n_obs, spec.n_vars) == (40, 300)
    assert int(spec.manifest["total_nnz"]) == 960
    assert spec.members["x_payload"]["offset"] == 4096
    assert spec.members["x_offsets"]["offset"] == 8192
    assert spec.members["x_indptr"]["offset"] == 12288
    assert spec.members["x_offsets"]["length"] == spec.members["x_indptr"]["length"] == 328
    assert spec.shard_region_end == 16384
    assert int(spec.offsets[-1]) == len(spec.payload)
    rec = json.loads(spec.member("groups.json"))
    assert [(g["label"], g["start"], g["stop"]) for g in rec["groups"]] == [
        ("ctrl", 0, 10),
        ("a", 10, 23),
        ("b", 23, 40),
    ]


# ------------------------------------------------------------------- negative controls
# Each damages one byte range the spec calls load-bearing and asserts the LIBRARY refuses
# the archive. Without these the assertions above cannot be distinguished from no-ops.


def _patch(path, offset, blob):
    with path.open("r+b") as fh:
        fh.seek(offset)
        fh.write(blob)


def test_rewrite_member_is_itself_lossless(tmp_path):
    """Positive control for the tamper helper below.

    Every negative control damages an archive through ``_rewrite_member``. If that helper
    corrupted an archive on its own, each of them would still "pass" on an incidental error
    rather than on the invariant it names. Rewriting a member with its OWN bytes must leave
    the archive byte-identical and fully readable.
    """
    p = _write(tmp_path, "pfordelta_grouped")
    before = p.read_bytes()
    spec = _SpecReader(p)
    _rewrite_member(p, "manifest.json", spec.member("manifest.json"))
    assert p.read_bytes() == before
    _rewrite_member(p, "groups.json", spec.member("groups.json"))
    assert p.read_bytes() == before
    store = cellstream.open(p)
    try:
        assert store.group_spans()  # the exact call the group negative controls make
        lib = store.gather_rows(np.arange(store.n_obs))
    finally:
        store.close()
    assert abs(_SpecReader(p).to_csr() - sp.csr_matrix(lib, dtype="float64")).nnz == 0


def test_rewrite_member_handles_a_different_length(tmp_path):
    """The other half of the tamper helper's positive control.

    The identity rewrite exercises none of the paths semantic tampering uses: member length
    unchanged, footer length unchanged, trailer not relocated. This adds an ADDITIVE manifest key
    -- which section 6 says a reader MUST ignore -- so the member and footer both grow, the
    trailer moves, and the archive must still open and decode identically. (codex, checkpoint 2.)
    """
    path = _write(tmp_path, "pfordelta")
    spec = _SpecReader(path)
    before_len = spec.members["manifest.json"]["length"]
    before_footer = spec.footer_length
    before_footer_offset = spec.footer_offset
    expected = spec.to_csr()

    man = dict(spec.manifest)
    man["a_key_a_conforming_reader_must_ignore"] = "y" * 64
    _rewrite_member(path, "manifest.json", json.dumps(man).encode())

    after = _SpecReader(path)
    assert after.members["manifest.json"]["length"] > before_len
    assert after.footer_offset > before_footer_offset, "the footer must have moved"
    assert after.footer_length >= before_footer
    assert after.footer_offset + after.footer_length + TRAILER_SIZE == len(after.raw)
    store = cellstream.open(path)
    try:
        lib = store.gather_rows(np.arange(store.n_obs))
    finally:
        store.close()
    assert abs(after.to_csr() - sp.csr_matrix(lib, dtype="float64")).nnz == 0
    assert abs(after.to_csr() - expected).nnz == 0


def test_reject_bad_head_magic(tmp_path):
    """Section 10.1."""
    p = _write(tmp_path, "zstd_int")
    _patch(p, 0, b"XXXX")
    with pytest.raises(PackedArchiveError, match="head magic"):
        cellstream.open(p)


def test_reject_bad_trailer_magic(tmp_path):
    """Section 10.1."""
    p = _write(tmp_path, "zstd_int")
    _patch(p, p.stat().st_size - TRAILER_SIZE, b"XXXXXXXX")
    with pytest.raises(PackedArchiveError, match="trailer magic"):
        cellstream.open(p)


def test_reject_footer_crc_mismatch(tmp_path):
    """Section 10.2 -- flip one byte inside the footer, leaving the CRC stale."""
    p = _write(tmp_path, "zstd_int")
    spec = _SpecReader(p)
    off = spec.footer_offset + spec.footer_length - 2  # inside the JSON, before the closing brace
    orig = spec.raw[off : off + 1]
    _patch(p, off, bytes([orig[0] ^ 0x01]))
    with pytest.raises(PackedArchiveError, match="CRC"):
        cellstream.open(p)


def test_reject_unsupported_schema_version(tmp_path):
    """Section 10.8 / sec 11 -- the version gate is monotonic, not best-effort."""
    p = _write(tmp_path, "zstd_int")
    spec = _SpecReader(p)
    man = dict(spec.manifest)
    man["schema_version"] = 99
    blob = json.dumps(man).encode()
    assert len(blob) <= spec.members["manifest.json"]["length"] + 512
    _rewrite_member(p, "manifest.json", blob)
    with pytest.raises(IncompatibleSchemaError, match="schema_version"):
        cellstream.open(p)


def test_reject_shifted_indptr(tmp_path):
    """Section 10.7 -- the endpoint check, not just the per-row differences.

    Adding a constant to every entry preserves every row's nnz, so a reader that only looked at
    differences served rows happily from an archive whose indptr[0] != 0 and whose indptr[-1] !=
    total_nnz. This is the exact shape codex identified; same length, so the member is replaced
    in place.
    """
    path = _write(tmp_path, "pfordelta")
    spec = _SpecReader(path)
    shifted = (spec.indptr + 5).astype("<i8")
    _rewrite_member(path, "x_indptr", shifted.tobytes())
    with pytest.raises(IncompatibleSchemaError, match=r"x_indptr\[0\]"):
        cellstream.open(path)


def test_reject_wrong_indptr_endpoint(tmp_path):
    """Section 10.7, the ENDPOINT specifically -- not reachable via the first-element check.

    test_reject_shifted_indptr adds a constant to every entry, so it trips indptr[0] first and
    would still pass with the endpoint check removed. Raising only the LAST element keeps
    indptr[0] == 0 and the array non-decreasing, so nothing but `indptr[-1] == total_nnz` can
    object. (codex, checkpoint 2 round 2.)
    """
    path = _write(tmp_path, "pfordelta")
    spec = _SpecReader(path)
    broken = np.array(spec.indptr, dtype="<i8")
    broken[-1] += 1
    _rewrite_member(path, "x_indptr", broken.tobytes())
    with pytest.raises(IncompatibleSchemaError, match="total_nnz"):
        cellstream.open(path)


def test_reject_wrong_offsets_endpoint(tmp_path):
    """Section 10.5, the ENDPOINT -- offsets[-1] must equal the x_payload member's length."""
    path = _write(tmp_path, "pfordelta")
    spec = _SpecReader(path)
    broken = np.array(spec.offsets, dtype="<i8")
    broken[-1] += 1
    _rewrite_member(path, "x_offsets", broken.tobytes())
    with pytest.raises(IncompatibleSchemaError, match="x_payload length"):
        cellstream.open(path)


def test_reject_swapped_sidecar_roles(tmp_path):
    """Section 3 -- the required name->role map, which a names-only check would let through."""
    path = _write(tmp_path, "pfordelta")

    def swap(members):
        for entry in members:
            if entry["name"] == "x_offsets":
                entry["role"] = "indptr"
            elif entry["name"] == "x_indptr":
                entry["role"] = "offsets"

    _rewrite_footer(path, swap)
    with pytest.raises(IncompatibleSchemaError, match="member role mismatch"):
        cellstream.open(path)


def test_nonzero_reserved_trailer_field_is_accepted(tmp_path):
    """Section 2.2 -- readers MUST IGNORE the reserved field.

    Rejecting a nonzero value would make the first extension that uses it unreadable by every
    reader already deployed, so this asserts tolerance rather than rejection.
    """
    path = _write(tmp_path, "pfordelta")
    spec = _SpecReader(path)
    at = len(spec.raw) - TRAILER_SIZE + 10  # the reserved u16
    with path.open("r+b") as fh:
        fh.seek(at)
        fh.write(struct.pack("<H", 0xBEEF))
    store = cellstream.open(path)  # must NOT raise
    try:
        assert store.n_obs == spec.n_obs
    finally:
        store.close()
    assert _SpecReader(path).reserved == 0xBEEF


def test_reject_non_monotonic_offsets(tmp_path):
    """Section 10.6 -- monotonicity is checked at open, not only when a frame is sliced."""
    path = _write(tmp_path, "pfordelta")
    spec = _SpecReader(path)
    broken = np.array(spec.offsets, dtype="<i8")
    broken[2] = broken[1] - 1  # one decreasing pair, endpoint untouched
    _rewrite_member(path, "x_offsets", broken.tobytes())
    with pytest.raises(IncompatibleSchemaError, match="not non-decreasing"):
        cellstream.open(path)


def test_reject_truncated_offsets(tmp_path):
    """Section 10.5 -- x_offsets shorter than n_obs+1."""
    p = _write(tmp_path, "zstd_int")
    spec = _SpecReader(p)
    _rewrite_member(p, "x_offsets", spec.member("x_offsets")[:-8])
    with pytest.raises(IncompatibleSchemaError, match="offsets/indptr size"):
        cellstream.open(p)


@pytest.mark.parametrize("engine", ["rust", "python"])
def test_reject_out_of_range_column_before_narrowing(tmp_path, monkeypatch, engine):
    """Section 5.1 / 10.18 -- the bound is checked on the WIDE value, before any narrowing cast.

    The fixture is built so that narrowing FIRST would hide the violation instead of merely
    reporting it late. A real column at 70000 is written with n_vars=100000 (so the index output
    cast is uint32); the manifest is then tampered to n_vars=65535 with
    index_dtype_on_disk=uint16, which makes 70000 narrow to 4464 -- a perfectly legal column that
    the aggregate CSR bounds scan cannot object to. Only a check on the uint64 cumsum can catch
    it, and the assertion pins the REPORTED column so a wrapped 4464 cannot pass for a pass.

    The previous fixture moved n_vars from 300 to 8. Columns 8..299 do not wrap when narrowed to
    uint16, so the aggregate scan rejected them anyway and the control passed with the wide check
    removed; it also ran only on whichever engine the machine had. (codex, checkpoint 2.)
    """
    if engine == "python":
        monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    else:
        from cellstream.cell._rust_dispatch import rust_available

        monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
        if not rust_available():
            pytest.skip("Rust extension not available")

    n_vars = 100_000
    rows = [np.array([1, 70_000], dtype=np.int64)] * 10
    indptr = np.arange(0, 2 * len(rows) + 1, 2, dtype=np.int64)
    x = sp.csr_matrix(
        (np.full(2 * len(rows), 7, dtype=np.int64), np.concatenate(rows), indptr),
        shape=(len(rows), n_vars),
    )
    path = tmp_path / "wide.csad"
    write_sharded(_adata(x), path, layout="cell", codec="pfordelta", overwrite=True)
    spec = _SpecReader(path)
    assert spec.manifest["index_dtype_on_disk"] == "uint32"

    man = dict(spec.manifest)
    man["n_vars"] = 65_535
    man["index_dtype_on_disk"] = "uint16"  # 70000 -> 4464 if narrowing happens first
    _rewrite_member(path, "manifest.json", json.dumps(man).encode())

    store = cellstream.open(path)
    try:
        with pytest.raises(ValueError, match=r"70000 out of range") as excinfo:
            store.gather_rows(np.arange(store.n_obs))
        assert "4464" not in str(excinfo.value), (
            "the column was narrowed before the bound was checked -- 70000 wrapped to 4464"
        )
    finally:
        store.close()


def test_reject_broken_group_tiling(tmp_path):
    """Section 10.21 -- a gap in the group tiling."""
    p = _write(tmp_path, "pfordelta_grouped")
    spec = _SpecReader(p)
    rec = json.loads(spec.member("groups.json"))
    rec["groups"][1]["start"] += 1  # leaves a one-row hole
    _rewrite_member(p, "groups.json", json.dumps(rec).encode())
    store = cellstream.open(p)
    try:
        with pytest.raises(IncompatibleSchemaError, match="tile"):
            store.group_spans()
    finally:
        store.close()


def test_reject_reference_not_leading(tmp_path):
    """Section 10.21 -- reference records must be the leading block."""
    p = _write(tmp_path, "pfordelta_grouped")
    spec = _SpecReader(p)
    rec = json.loads(spec.member("groups.json"))
    labels = [g["label"] for g in rec["groups"]]
    rec["reference"] = labels[-1]
    man = dict(spec.manifest)
    man["reference"] = labels[-1]
    _rewrite_member(p, "groups.json", json.dumps(rec).encode())
    _rewrite_member(p, "manifest.json", json.dumps(man).encode())
    store = cellstream.open(p)
    try:
        with pytest.raises(IncompatibleSchemaError, match="leading block"):
            store.group_spans()
    finally:
        store.close()


def test_reject_unsafe_member_name(tmp_path):
    """Section 10.4a -- a member name is joined to a path by any extractor (Zip-Slip)."""
    p = _write(tmp_path, "zstd_int")
    _rewrite_footer(p, lambda ms: ms[0].__setitem__("name", "../escape"))
    with pytest.raises(PackedArchiveError, match="unsafe member name"):
        cellstream.open(p)


def test_reject_duplicate_member_name(tmp_path):
    """Section 10.4a -- by-name lookup would resolve last-wins and hide a crafted entry."""
    p = _write(tmp_path, "zstd_int")
    _rewrite_footer(p, lambda ms: ms.append(dict(ms[0])))
    with pytest.raises(PackedArchiveError, match="duplicate member name"):
        cellstream.open(p)


def test_reject_member_out_of_bounds(tmp_path):
    """Section 10.4b -- a member range must stay inside the body region."""
    p = _write(tmp_path, "zstd_int")
    _rewrite_footer(p, lambda ms: ms[0].__setitem__("length", 1 << 40))
    with pytest.raises(PackedArchiveError, match="out of bounds"):
        cellstream.open(p)


def test_reject_misaligned_payload_member(tmp_path):
    """Section 10.4c -- x_payload is mmap'd at its base, so its offset must be page-aligned."""
    p = _write(tmp_path, "zstd_int")
    _rewrite_footer(p, lambda ms: ms[0].__setitem__("offset", ms[0]["offset"] + 1))
    with pytest.raises(PackedArchiveError, match="not 4096-byte aligned"):
        cellstream.open(p)


def _rewrite_member(path, name, blob):
    """Replace one member's bytes in place, fixing its footer length, the footer CRC and
    the trailer.

    A replacement may exceed the member's original length in two situations, and no others:
    it still fits the 4 KB alignment padding before the NEXT member, or the target is the LAST
    member, in which case the footer simply moves later (the footer is unaligned by design).
    Growing a middle member past its padding would have to shift every following member and is
    refused rather than half-implemented.
    """
    spec = _SpecReader(path)
    m = spec.members[name]
    ordered = sorted(spec.footer["members"], key=lambda x: x["offset"])
    nxt = next((x for x in ordered if x["offset"] > m["offset"]), None)
    room = (nxt["offset"] if nxt else spec.footer_offset) - m["offset"]
    new_footer_offset = None
    if len(blob) > room:
        assert nxt is None, f"{name}: {len(blob)} bytes does not fit in {room} and is not last"
        new_footer_offset = m["offset"] + len(blob)
        room = len(blob)

    def _set_length(members):
        for entry in members:
            if entry["name"] == name:
                entry["length"] = len(blob)

    with path.open("r+b") as fh:
        fh.seek(m["offset"])
        fh.write(blob)
        fh.write(b"\x00" * (room - len(blob)))
    # `spec` is the PRE-WRITE snapshot, deliberately. _rewrite_footer would otherwise re-parse
    # the file, and between the member write and the footer fix the member's declared length is
    # still the old one -- so a shorter replacement reads back JSON with trailing NULs and a
    # longer one reads back truncated JSON. Either way _SpecReader's manifest parse would raise
    # from inside the tamper helper. (Caught by three existing negative controls when this was
    # first split out.)
    _rewrite_footer(path, _set_length, spec=spec, footer_offset=new_footer_offset)


def _rewrite_footer(path, mutate, spec=None, footer_offset=None):
    """Re-serialize the footer after ``mutate(members)``, fixing its CRC and the trailer.

    Split out of ``_rewrite_member`` so a test can tamper with the member INDEX (a name, a
    duplicate, an offset) without touching member bytes. Serialization matches
    ``packed.format.serialize_footer`` exactly -- compact separators, the same three scalar keys
    in the same order -- which is what lets the positive control assert byte-identity.

    ``spec`` lets a caller supply a snapshot parsed BEFORE it wrote member bytes; see the note in
    ``_rewrite_member``. ``footer_offset`` moves the footer, for a last-member replacement that
    grew past its old extent.
    """
    spec = spec if spec is not None else _SpecReader(path)
    foff = spec.footer_offset if footer_offset is None else footer_offset
    members = [dict(x) for x in spec.footer["members"]]
    mutate(members)
    fbuf = json.dumps(
        {
            "container_version": spec.footer["container_version"],
            "alignment": spec.footer["alignment"],
            "shard_region_end": spec.footer["shard_region_end"],
            "members": members,
        },
        separators=(",", ":"),
    ).encode()
    with path.open("r+b") as fh:
        fh.seek(foff)
        fh.write(fbuf)
        fh.write(
            struct.pack(
                "<8sHHQQQI",
                TRAILER_MAGIC,
                1,
                0,
                foff,
                len(fbuf),
                spec.shard_region_end,
                zlib.crc32(fbuf) & 0xFFFFFFFF,
            ).ljust(TRAILER_SIZE, b"\x00")
        )
        fh.truncate(foff + len(fbuf) + TRAILER_SIZE)
