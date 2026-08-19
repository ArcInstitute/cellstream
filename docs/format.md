# Archive format specification

**Status:** normative for `layout="cell"` archives with `schema_version` 5 and 6.
**Container version:** `SHPK` 1.
**Reference implementation:** `src/cellstream/packed/` (container) and `src/cellstream/cell/`
(cell layout), with a byte-identical Rust core in `rust/src/`.
**Conformance test:** `tests/cell/test_format_spec_conformance.py` re-implements the reader
described here from this document alone and asserts it recovers the same matrix as the library.
Its matrix is both codecs × {narrow integers, wide integers (`n_vars > 65536` and values above
65535, so both output casts are `uint32`), `float32`, `float64`} × {grouped, ungrouped}, minus
the combinations the writer cannot produce — pfordelta stores integers only. The `float64`
fixture carries values that are *not* `float32`-representable, so a float32 detour anywhere in
either chain fails the comparison rather than passing it.

This document specifies the on-disk bytes well enough to write an independent reader. It does
**not** specify the Python API (see the README) or the legacy `layout="shard"` archive
([Appendix B](#appendix-b--the-legacy-shard-layout)).

---

## 1. Conventions

* **Byte order is little-endian** everywhere. All fixed-width integers, the `x_offsets` /
  `x_indptr` arrays, and every codec word are little-endian. The format has no big-endian
  encoding and no endianness marker.

  **The reference implementation is little-endian-ONLY, and does not enforce this.** It writes
  and reads the sidecars and codec words with *native* numpy dtypes (`np.int64`, `np.uint32`) and
  never byte-swaps, so on a big-endian host it would emit big-endian bytes — a non-conforming
  archive that its own reader would then read back "correctly" while nothing else could. No
  big-endian platform is supported (wheels are linux x86_64), so this is a documented limitation
  rather than a live bug; an independent reader should use explicit little-endian dtypes
  (`<i8`, `<u4`, `<f4`, `<f8`) so it is correct on any host.
* **MUST / MUST NOT / SHOULD** carry their usual normative force. "The writer emits" describes
  what the reference implementation produces; "a reader MUST accept" is the wider contract.
* `u16` / `u32` / `i64` denote unsigned 16-bit, unsigned 32-bit and signed 64-bit
  little-endian integers.
* **nnz** = number of *stored* values (a stored value may be zero if the caller stored one).
* **Storage order** is the physical row order inside the archive, which for a grouped write is
  *not* the caller's input order. Row indices in this document are always storage-order
  indices. `obs.parquet` and `header.h5ad`'s `obs` are stored in the same storage order, so the
  archive is internally consistent and self-describing; the input order is not recorded.
* JSON members are UTF-8 and MUST be parsed as JSON, not scanned textually. The writer emits
  the footer with compact separators (`,` / `:`) and the other JSON members with
  `indent=2`, but neither is normative — a reader MUST NOT depend on JSON whitespace.

### 1.1 File extension

The extension is a **caller convention with no effect on the bytes**: `.shad` for the shard
layout, `.csad` for the cell layout. Identification is by the 4-byte head magic (§2.1) and then
by `manifest.json`'s `layout` field. A reader MUST NOT infer the layout from the filename.

---

## 2. The `SHPK` container

A cell archive is one regular file. It is a byte-concatenation of *members* — independently
addressable byte ranges — bracketed by a fixed-size head block and a fixed-size trailer:

```
 offset 0        ┌──────────────────────────────────────────┐
                 │ head block: "SHPK" + u16 version, then   │  4096 bytes, fixed
                 │ NUL padding to 4096                      │
 offset 4096     ├──────────────────────────────────────────┤
                 │ immutable data region                    │
                 │   x_payload   (4 KB-aligned)             │
                 │   x_offsets   (4 KB-aligned)             │
                 │   x_indptr    (4 KB-aligned)             │
 shard_region_end├──────────────────────────────────────────┤
                 │ metadata region                          │
                 │   header.h5ad   (4 KB-aligned)           │
                 │   obs.parquet   (4 KB-aligned)           │
                 │   manifest.json (4 KB-aligned)           │
                 │   groups.json   (4 KB-aligned, optional) │
 footer_offset   ├──────────────────────────────────────────┤
                 │ footer: UTF-8 JSON member index          │  NOT aligned
 EOF − 64        ├──────────────────────────────────────────┤
                 │ trailer: 64 bytes, points back           │
 EOF             └──────────────────────────────────────────┘
```

The trailer is fixed-size and last, so it is locatable by reading the final 64 bytes without
knowing anything else about the file (the ZIP / Parquet pattern). All extensibility lives in
the JSON footer.

### 2.1 Head block

| offset | size | value |
|---|---|---|
| 0 | 4 | `"SHPK"` (`0x53 0x48 0x50 0x4B`) |
| 4 | 2 | u16 container version, currently `1` |
| 6 | 4090 | NUL |

The head block is exactly 4096 bytes, so the first member starts at 4096 with no padding
arithmetic. A reader MUST verify the magic. The head's version field duplicates the trailer's;
they are written from the same constant.

### 2.2 Trailer

The last 64 bytes of the file. The first 40 are the C-struct `<8sHHQQQI`:

| offset | size | field |
|---|---|---|
| 0 | 8 | `"SHPKEND\0"` — trailer magic, deliberately distinct from the head magic |
| 8 | 2 | u16 container version (`1`) |
| 10 | 2 | reserved — writers MUST write 0; **readers MUST ignore it** |
| 12 | 8 | u64 `footer_offset` |
| 20 | 8 | u64 `footer_length` |
| 28 | 8 | u64 `shard_region_end` |
| 36 | 4 | u32 CRC-32 (zlib) of the footer bytes |
| 40 | 24 | NUL padding to 64 |

**Reader algorithm.** Read the last 64 bytes; check the magic; read
`[footer_offset, footer_offset + footer_length)`; verify its CRC-32 against the trailer. A
mismatch means the tail was truncated or torn by an interrupted edit and the archive MUST be
rejected.

Bound-check before reading, so a corrupt `footer_length` cannot drive a huge allocation:
`footer_offset >= 4096` and `footer_offset + footer_length <= filesize - 64`. That `<=` is
what the reference implementation enforces; every writer in fact produces exact abutment
(`footer_offset + footer_length == filesize - 64`), so a reader may assert the stricter form,
but the format only requires the bound.

A trailer whose `container version` is **greater** than the reader supports MUST be refused
rather than parsed optimistically.

The reserved field is for future use, so a reader MUST NOT reject a nonzero value — doing so
would make the first extension that uses it unreadable by every existing reader. Writers emit 0.

The trailer's `shard_region_end` MUST equal the footer's (§2.3). The trailer's copy is what a
recovery path reads *without* parsing the footer, so a disagreement means one of the two is
stale or crafted, and silently preferring either would put the truncation boundary somewhere the
other half of the file does not expect.

### 2.3 Footer

UTF-8 JSON:

```json
{"container_version":1,"alignment":4096,"shard_region_end":16384,"members":[
  {"name":"x_payload",    "role":"payload",  "offset":4096,  "length":2536},
  {"name":"x_offsets",    "role":"offsets",  "offset":8192,  "length":328},
  {"name":"x_indptr",     "role":"indptr",   "offset":12288, "length":328},
  {"name":"header.h5ad",  "role":"header",   "offset":16384, "length":36120},
  {"name":"obs.parquet",  "role":"obs",      "offset":53248, "length":2163},
  {"name":"manifest.json","role":"manifest", "offset":57344, "length":599},
  {"name":"groups.json",  "role":"groups",   "offset":61440, "length":158}
]}
```

* `alignment` is 4096 and `container_version` is 1 for every archive this spec covers.
* `shard_region_end` MUST equal the trailer's field of the same name.
* Each member entry carries `name`, `role`, `offset`, `length`. The shard layout adds
  `shard_idx` and (multi-matrix) `matrix`; a cell archive MUST NOT contain any member with
  `role == "shard"`. Unknown keys in a member entry MUST be ignored, not rejected.
* Members are looked up **by name**, so a name MUST be unique within the footer — a duplicate
  would otherwise resolve last-wins and hide a crafted entry. Order in the array is the physical
  write order and is informative, not normative, but a cell writer always emits `x_payload`,
  `x_offsets`, `x_indptr` first, in that order.
* **A member name MUST be path-safe**, because an extractor writes it to disk: either a single
  bare filename or a two-segment `<dir>/<file>` (the shard layout's `x/shard_0000.shx`,
  `var/raw.parquet`). No segment may be empty, `.` or `..`; no backslashes; nothing absolute.
  This is a Zip-Slip guard, not cosmetics — validate it before joining the name to any path.
* Every member's `[offset, offset + length)` MUST lie inside the body region
  `[4096, footer_offset)`, and `length` MUST be non-negative. A read that returns fewer bytes
  than `length` means the file is truncated and MUST be rejected, not zero-filled.

### 2.4 Alignment, padding, and the two regions

* Every member offset a writer emits is a multiple of 4096. Gaps between members are NUL-filled,
  as is the gap between the last member and the footer. The 4 KB alignment is what makes
  `x_payload` directly `mmap`-able at page granularity. A reader MUST enforce it for the roles it
  `mmap`s — the reference implementation checks `shard`, `header` and `payload`, and tolerates a
  misaligned `offsets` / `indptr` / metadata member, which it reads with ordinary `seek`+`read`.
* The **footer itself is not aligned** — it starts immediately after the last member's bytes.
* `shard_region_end` is the offset of the first member whose role is *not* one of `payload`,
  `offsets`, `indptr` — i.e. the boundary between the immutable data region and the metadata
  region. It is 4 KB-aligned because member offsets are.

The split exists so metadata can be edited without rewriting the data: `update_obs` /
`update_var` truncate the file at `shard_region_end` and re-append the metadata region, footer
and trailer, leaving the payload untouched.

**That editing path is not available for a cell archive.** The rewrite re-emits members by
ROLE and by a fixed list of metadata NAMES, and a cell archive's three data members
(`x_payload` / `x_offsets` / `x_indptr`) are in neither set — so it refuses rather than
silently dropping them. Measured: `update_obs_packed` on a `layout="cell"` archive raises
`NotImplementedError: … does not preserve ['x_indptr', 'x_offsets', 'x_payload'] …`, and
`CellStore` exposes no `update_obs` / `update_var` at all. Cell metadata is therefore
effectively immutable once written; rewrite the archive to change it. The region split still
matters to a reader for the reasons below.

Consequences a reader should know about:

* The metadata region's byte offsets are **not stable** across a metadata edit. Never cache
  them across a reopen.
* Mid-edit crash safety uses a sidecar `<file>.tailbak`, holding a `u64` `shard_region_end`
  followed by the previous tail bytes. Only a tail rewrite creates one, so for a cell archive
  this path is dormant — but the recovery check runs on every open, so a reader must still
  handle the sidecar correctly if one is present. On open, if a `.tailbak` exists and the current trailer
  does *not* validate, the tail is rolled back from the backup; if the trailer *does* validate,
  the edit completed and the backup is stale. Recovery takes the archive's write lock
  (`<file>.lock`, an `flock`), so a reader cannot roll back a live writer's in-progress edit.
  A `.tailbak` whose recorded `shard_region_end` is out of range or not 4 KB-aligned MUST be
  refused rather than used to truncate the file.
* An interrupted *initial* write never publishes a partial archive: the writer assembles into
  `<name>.<pid>.<uuid4>.tmp` opened `O_EXCL`, `fsync`s, then `os.replace`s into place.

---

## 3. Members of a cell archive

| member | role | required | contents |
|---|---|---|---|
| `x_payload` | `payload` | yes | concatenated per-cell frames, storage order (§5) |
| `x_offsets` | `offsets` | yes | `i64[n_obs+1]` byte offsets into `x_payload` (§4) |
| `x_indptr` | `indptr` | yes | `i64[n_obs+1]` CSR row pointer, cumulative nnz (§4) |
| `header.h5ad` | `header` | yes | an h5ad holding everything except `X` (§7) |
| `obs.parquet` | `obs` | yes | `obs` as parquet, storage order (§7) |
| `manifest.json` | `manifest` | yes | the schema, dtypes and codec (§6) |
| `groups.json` | `groups` | grouped only | group label → storage row span (§8) |

A cell archive has no shards, no `n_shards` and no `row_offsets`: it is flat and addressed
per cell.

---

## 4. `x_offsets` and `x_indptr`

Both are raw little-endian `i64` arrays of exactly `n_obs + 1` elements, i.e. each member's
`length` is `8 * (n_obs + 1)` bytes. Neither is compressed — they are loaded whole into RAM at
open, which is what makes a scattered read `O(rows requested)` rather than `O(n_obs)`.

* `x_offsets[r]` is the byte offset of storage row `r`'s frame within `x_payload`;
  `x_offsets[r+1]` is its end. `x_offsets[0]` MUST be 0 and the array MUST be non-decreasing.
  Writers emit contiguous frames, so `x_offsets[n_obs] == len(x_payload)`.
* `x_indptr` is the CSR row pointer of the matrix in storage order: `x_indptr[0] == 0`,
  non-decreasing, and `x_indptr[n_obs] == manifest["total_nnz"]`. Row `r` holds
  `nnz_r = x_indptr[r+1] - x_indptr[r]` stored values.

`nnz_r` is **the** length authority for decoding row `r` — a frame does not record its own
element count. A frame that decodes to a different count MUST be rejected (a short decode from
a truncated frame would otherwise hand back a previous row's buffer tail).

An empty row (`nnz_r == 0`) still has a frame; see §5.3.

---

## 5. `x_payload` — the per-cell frames

`x_payload` is the concatenation of one **frame** per storage row, in storage order. Row `r`'s
frame is exactly `x_payload[x_offsets[r] : x_offsets[r+1]]`. Frames are independently
decodable: decoding row `r` requires no other row.

Every frame, in both codecs, has the same framing:

```
┌────────────────────┬──────────────────────┬───────────────────────────────┐
│ u32 LE idx_len     │ index blob (idx_len) │ value blob (to end of frame)  │
└────────────────────┴──────────────────────┴───────────────────────────────┘
```

The value blob carries **no length field** — it runs to the end of the frame, whose extent
comes from `x_offsets`. A reader MUST check `len(frame) >= 4` and `4 + idx_len <= len(frame)`
before slicing.

Which codec produced the blobs is given by `manifest["codec"]` for the whole archive; there is
no per-frame codec marker. Both blob streams decode to exactly `nnz_r` elements.

Within one row, column indices are **sorted ascending**. Both encoders reject a
non-monotonic index array, and the pfordelta codec's delta representation cannot express one.
Duplicate columns within a row are representable; whether the writer verified their absence is
recorded in `manifest["indices_canonical"]` (§6.4).

### 5.1 `codec = "pfordelta"` (`schema_version` 5) — integer counts

Both blobs are FastPFor `simdfastpfor256` packings of a `u32` array, i.e. the output of
`encodeArray`, and are therefore whole numbers of `u32` words: **a blob length that is not a
multiple of 4 is corrupt.**

* **index blob** packs the per-row *delta gaps* of the sorted column indices:
  `gaps[0] = idx[0]`, `gaps[k] = idx[k] - idx[k-1]`. Decoding is `cumsum` over `u64`.
* **value blob** packs the values as `u32`, **not** delta-coded.

Physical width is **always `u32`** for both streams. `manifest["value_dtype_on_disk"]` and
`["index_dtype_on_disk"]` are the decoder's *output cast*, not the stored width — a reader MUST
NOT use them to size or slice a pfordelta blob. (The conformance test decodes pfordelta while
ignoring both fields, and matches the library bit-for-bit.)

`simdfastpfor256` is `CompositeCodec<SIMDFastPFor<8>, VariableByte>`: the block codec handles
`floor(n/256)*256` values and `VariableByte` the `n mod 256` tail. Byte-exact parity is against
**pyfastpfor 1.4.0**, whose FastPFor sources are vendored at `rust/vendor/FastPFor`
(provenance in `rust/vendor/FastPFor/PROVENANCE.md`). Two implementation notes that a
re-implementer will hit:

* The vendored encoder does **not** enforce the output capacity it is handed — the caller must
  supply a sufficient buffer. Worst-case output for `n` values is
  `n + ceil(n/512) + ceil(n/65536)*996 + 65` words; both engines declare `2n + 4096`. Capacity
  never changes the bytes written, only whether the codec throws, so it cannot affect the
  on-disk form.
* On decode, `VariableByte` emits one `u32` per varint in the *input*, ignoring the declared
  capacity, so the reachable output bound is `n + len(blob)` values. Size the destination
  buffer to that, then require `decodeArray` to return exactly `n`.

Decoding a pfordelta index stream, the column bound MUST be checked on the **wide** (`u64`)
cumsum, before any narrowing cast: narrowing first wraps an out-of-range column into a legal
one (70000 → 4464 at `u16`) and destroys the evidence. Because gaps come out of a `u32` buffer,
the cumsum is non-decreasing on any input, so `cumsum[-1] < n_vars` is a sound `O(1)` bound for
the whole row (for `nnz_r <= 2**32 + 1`, past which a modular `u64` accumulator could wrap).

### 5.2 `codec = "zstd"` (`schema_version` 6) — integers or floats

Every blob is exactly one zstd frame, produced at **level 1** with the content size embedded.
No blob is split across frames.

* **index blob** — indices are always `u32` on disk. Pipeline:
  `byte-shuffle → per-plane byte-delta → zstd`.
* **value blob** — depends on `manifest["value_dtype_on_disk"]`:
  * `float32` / `float64`: **plain zstd of the raw little-endian values**, no shuffle, no
    delta. Byte-shuffling a float scatters each value's bytes into separate planes and destroys
    the whole-value repeats zstd relies on, inflating low-cardinality log-normalized data
    ~2.8×.
  * `uint32`: `byte-shuffle → zstd` (no byte-delta: delta hurts counts).

  For an integer archive the on-disk value width is **always `u32`**, regardless of
  `value_dtype_on_disk`, exactly as for indices. Float archives store at the declared width, so
  there `value_dtype_on_disk` *is* the physical width.

**Byte-shuffle.** For an array of `N` elements of `T` bytes, view it as `(N, T)` bytes and
transpose to `(T, N)`: all first bytes, then all second bytes, and so on. The inverse
transposes back.

**Byte-delta.** Applied *after* the shuffle, along the element axis, independently per plane,
in `u8` arithmetic (**wrapping mod 256**): `d[p][0] = s[p][0]`,
`d[p][k] = s[p][k] - s[p][k-1]`. The inverse is a `u8` cumulative sum, which wraps identically.
Byte-delta applies to the **index** stream only.

**Decode order** is therefore: zstd-decompress → un-byte-delta (indices only) → un-byte-shuffle
→ reinterpret as `u32` (or as the declared float type, which needs neither inverse).

Every zstd decompression MUST be bounded. The expected decompressed size is known exactly —
`4 * nnz_r` for indices, `itemsize * nnz_r` for values — and both engines:

1. reject an expected size above the **64 MiB per-chunk cap**;
2. reject a frame whose *embedded* content size disagrees with the expected size, before
   decompressing (zstd's `max_output_size` does not cap a frame that embeds its size);
3. cap a frame *without* an embedded size at the expected length;
4. reject a decompressed length that differs from the expected size.

The 64 MiB cap is a **hard per-cell limit** for this codec: at most
`64 MiB / max(4, value_itemsize)` stored values in one cell — 16 777 216 for `u32`/`float32`,
8 388 608 for `float64`. The writer refuses such a cell up front rather than writing an archive
nothing can read. `pfordelta` has no equivalent cap.

Unlike pfordelta, the zstd codec does **not** bound column indices per frame — its indices are
`u32` on disk, so an out-of-range column survives a decode unchanged rather than wrapping. It
is bounded one level up, by the reader's aggregate index scan when it materializes a CSR. A
conforming reader MUST reject an index `>= n_vars` at some point before returning data; doing
it per frame is permitted and is what the Rust core does.

### 5.3 Empty rows

A row with `nnz_r == 0` still occupies a frame, and its size differs by codec — because each
codec's "encode nothing" is different, not because the framing changes:

| codec | empty frame | why |
|---|---|---|
| `pfordelta` | **4 bytes**: `idx_len = 0`, then nothing | packing 0 values yields an empty blob |
| `zstd` | **22 bytes**: `idx_len = 9`, then two 9-byte zstd frames | `compress(b"")` is a real, non-empty zstd frame |

Byte-for-byte, an empty zstd frame observed on the current build is

```
09 00 00 00  28 b5 2f fd 20 00 01 00 00  28 b5 2f fd 20 00 01 00 00
└ idx_len=9  └ zstd frame of b"" ──────┘  └ zstd frame of b"" ──────┘
```

The 22 is descriptive of this zstd build, not normative — a zstd version bump could change the
empty-frame encoding. **The normative rule is that `nnz_r` comes from `x_indptr` and never from
a frame's size.** A reader MUST NOT special-case an empty frame by *size*: take `nnz_r`, and if
it is 0, produce empty index/value arrays.

**Whether an empty frame's CONTENTS are validated is engine-dependent in the reference
implementation, and the spec does not require either behaviour.** Measured on both engines with
an `nnz == 0` frame whose `idx_len` was clobbered to `0xffffffff`:

* the **pure-Python** decoder validates the prefix before it looks at `nnz`, and raises
  `corrupt <codec> frame: idx blob length exceeds frame`;
* the **Rust** decoder skips the frame parse entirely for `nnz == 0`
  (`rust/src/cell.rs`: `if nnz == 0 { continue; }`), so the same corruption is accepted and the
  row simply contributes no elements.

So a conforming reader MAY validate an empty frame or ignore it; what it MUST NOT do is derive
the element count from the frame. Do not rely on an empty frame's bytes being checked — and note
that this asymmetry means a corrupt empty frame is a case where the two engines disagree, which
is a defect in the implementation rather than a property of the format.

---

## 6. `manifest.json`

One JSON object. Example (pfordelta, ungrouped):

```json
{
 "schema_version": 5,
 "layout": "cell",
 "codec": "pfordelta",
 "n_obs_total": 40,
 "n_vars": 300,
 "total_nnz": 960,
 "value_dtype_on_disk": "uint16",
 "index_dtype_on_disk": "uint16",
 "x_data_dtype_min": "uint8",
 "x_indices_dtype_min": "uint16",
 "pyfastpfor_codec": "simdfastpfor256",
 "group_by": null,
 "reference": null,
 "sort_within_group": null,
 "sort_order": null,
 "payload_sha256": null,
 "indices_canonical": true,
 "writer_fingerprint": {"cellstream_version": "0.8.1", "writer_kind": "cell"},
 "created_at": "2026-08-19T08:37:53Z"
}
```

### 6.1 Identification and shape

| field | type | meaning |
|---|---|---|
| `schema_version` | int | `5` for `pfordelta`, `6` for `zstd`. A reader MUST reject any other value for a cell archive. |
| `layout` | `"cell"` | Selects this specification. Absent or different → not a cell archive. |
| `codec` | `"pfordelta"` \| `"zstd"` | MUST agree with `schema_version`: 5 ⇒ `pfordelta`, 6 ⇒ `zstd`. |
| `n_obs_total` | int ≥ 0 | Number of cells (rows). Fixes the length of `x_offsets` / `x_indptr` at `n_obs_total + 1`. |
| `n_vars` | int ≥ 0 | Number of features (columns). Every decoded column index MUST lie in `[0, n_vars)`. |
| `total_nnz` | int ≥ 0 | MUST equal `x_indptr[n_obs_total]`. |

`schema_version` is a monotonic compatibility gate, not a feature flag: a reader that does not
recognize a version MUST refuse the archive rather than guess.

### 6.2 Dtype fields

| field | schema 5 | schema 6 | meaning |
|---|---|---|---|
| `value_dtype_on_disk` | `uint16` \| `uint32` | `uint32` \| `float32` \| `float64` | see below |
| `index_dtype_on_disk` | `uint16` \| `uint32` | `uint32` | see below |
| `x_data_dtype_min` | string | string | narrowest dtype that holds every stored value losslessly |
| `x_indices_dtype_min` | string | string | narrowest dtype that holds every column index |

**`*_on_disk` is the decoder's output cast, and only sometimes the physical width.** It is the
physical width for **zstd float values** and nothing else: pfordelta always packs `u32` words,
and zstd always stores `u32` indices and `u32` integer values. The names are historical.
Concretely, the writer picks them as:

* `index_dtype_on_disk` = `uint16` if `n_vars < 65536` else `uint32` (pfordelta);
  always `uint32` for zstd.
* `value_dtype_on_disk` = `uint16` if `max_value < 65536` else `uint32` (pfordelta integer
  counts); `uint32` for zstd integers; the declared float width for zstd floats.

**The `*_min` fields exist to make a cast decision cheap, not to describe layout.** An archive
storing counts as `u32` can still record `x_data_dtype_min: "uint8"`, which lets a reader judge
`can_cast(uint8, float32, "safe") == True` and skip a per-element lossless check on a default
`float32` read. For an **integer** archive `x_data_dtype_min` is the narrowest unsigned dtype
holding the observed maximum (`uint8` … `uint32`), computed from the data. For a **float**
archive it is simply the stored float width — there is no narrower lossless float claim to
make, so it equals `value_dtype_on_disk`. `x_indices_dtype_min` is `uint16` when `n_vars <= 65536` — note `<=`, one wider
than the `<` used to choose `index_dtype_on_disk`, and both are right: at exactly
`n_vars == 65536` the largest possible column index is 65535, which `u16` holds, while the
storage choice is deliberately conservative. A reader treating `*_min` as advisory (falling back
to the `*_on_disk` value when absent) is conforming.

### 6.3 Codec parameters

| field | meaning |
|---|---|
| `pyfastpfor_codec` | schema 5: the FastPFor codec name, `"simdfastpfor256"` (a string, required). Schema 6: `null` (or a string; ignored — zstd uses no FastPFor). |

### 6.4 Integrity

| field | meaning |
|---|---|
| `payload_sha256` | `null`, or the lowercase 64-hex SHA-256 of **exactly the `x_payload` member bytes** — not the whole file, and not including the alignment padding. `null` when the archive was written with `payload_checksum=False`, the default. `null` therefore means *nothing to verify*, **not** *verified intact*. |
| `indices_canonical` | bool. `true` iff the writer verified that no row holds a duplicate column. Optional (absent on old archives) and then treated as `false`. A reader MAY use `true` to skip a canonical-form scan; asserting canonical form when it is false silently breaks duplicate summation, so a reader MUST NOT assume `true` by default. |

### 6.5 Storage order

| field | meaning |
|---|---|
| `group_by` | `obs` column name the storage order was grouped by, or `null`. Non-`null` ⇒ `groups.json` MUST be present; `null` ⇒ it MUST be absent. |
| `reference` | the reference label(s) — `null`, a string, or a non-empty list of strings. MUST equal `groups.json`'s `reference` (§8). |
| `sort_within_group` | secondary sort column(s) within a group: `null`, a string, or a list of strings. Informative provenance. |
| `sort_order` | `null`, `"natural"`, or `"alphabetical"`. Informative provenance. |

### 6.6 Provenance

| field | meaning |
|---|---|
| `writer_fingerprint` | `{"cellstream_version": <str>, "writer_kind": "cell"}`. `writer_kind` is `"cell"` for all three cell writers (one-shot, push, concat) so their output stays byte-identical. **Archives written before the `shardad` → `cellstream` rename carry `shardad_version` instead**; both are informative, so a reader MUST NOT require either key. |
| `created_at` | UTC `"%Y-%m-%dT%H:%M:%SZ"`. |

Both are informative. A reader MUST NOT gate behaviour on them.

Unknown top-level manifest keys MUST be ignored.

---

## 7. `header.h5ad` and `obs.parquet`

**`header.h5ad`** is a standard h5ad file — readable by `anndata` as-is — carrying everything
except the matrix. `X` is present but **empty**: a CSR group whose `indptr` has `n_obs + 1`
entries and whose `data` / `indices` are zero-length, so the shape round-trips without storing
values. `obs`, `var`, `uns`, `obsm`, `varm`, `obsp`, `varp` are stored normally, with
`obs`/`obsm`/`obsp` already permuted into storage order (`obsp`, being cell × cell, on both
axes). It is the reference for everything except `obs` — `var`, `uns`, `obsm`, `varm`, `obsp`,
`varp` live only here, so for those there is nothing to disagree with. For `obs` see below.

Because it is a real h5ad, a reader can hand the member's byte range to HDF5 directly — the
reference implementation exposes a seekable window over the archive's `mmap` and never copies
the member out.

**`obs.parquet`** is `obs` written as parquet, in the same storage order, with `obs_names` as
the index. It exists so `obs` can be read by tool-agnostic readers (polars, duckdb, pandas)
without an HDF5 dependency, and it is what the reference reader actually uses: `CellStore.obs`
reads the parquet member, never the h5ad's `obs`.

**Neither member overrides the other.** Both are written from the same DataFrame in the same
pass, so a disagreement is not a precedence question — it means the archive is invalid, and a
reader that silently preferred one would return different metadata from the reference API for the
same file. If a reader validates only one, `obs.parquet` is the one the reference implementation
reads.

Two fields the cell layout does **not** store, both dropped with a warning at write time:
`layers`, and `.raw`.

---

## 8. `groups.json`

Present **iff** `manifest["group_by"]` is non-`null`. A grouped write sorts cells by the group
column — reference groups first — so that every group occupies one contiguous storage-row span,
and records those spans here. This is what makes "read just this perturbation" a single
contiguous range read.

Note that the cell layout's `groups.json` is a JSON **object**; the legacy shard layout's is a
JSON *array* of a different shape (Appendix B). They are not interchangeable.

```json
{"reference": "ctrl",
 "groups": [{"label": "ctrl", "start": 0,  "stop": 10},
            {"label": "a",    "start": 10, "stop": 23},
            {"label": "b",    "start": 23, "stop": 40}]}
```

`groups` is a list of records in storage order, each with a string `label` and integer
`start` / `stop` giving the half-open storage-row span `[start, stop)`. Required invariants,
all of which a reader MUST enforce before sizing a read from them:

1. `0 <= start < stop <= n_obs` — every record holds at least one row; no writer emits an
   empty group.
2. The records **tile** `[0, n_obs)`: the first starts at 0, each starts where the previous
   stopped, and the last stops at `n_obs`. For a grouped archive with `n_obs == 0` the list is
   empty — the only case where `"groups": []` is legal.
3. `reference` is `null`, a string, or a non-empty list of strings — never `[]`, which a writer
   canonicalizes to `null`.
4. Every reference label MUST be carried by some record, and the reference records MUST form a
   **leading prefix** of the list (positions `0..k-1`). Readers rely on the reference block
   being `[0, R)`; without this check a record list `x:[0,10), r:[10,20)` with reference `"r"`
   would pass bounds validation and serve ten non-reference rows as reference.
5. `reference` MUST agree with `manifest["reference"]` (comparing a single-element list to the
   equivalent string). Every writer stores the same value in both.

**Duplicate labels are legal** and MUST NOT be rejected: a writer can emit several records
sharing a label, including inside the reference prefix. A by-label lookup should then report
the ambiguity; a span enumeration returns them all.

**The missing-value label is the literal string `"nan"`.** Group labels are stringified, and
every flavour of missing (`None`, `NaN`, `<NA>`) maps to that one sentinel, so one logical
"missing" is one group. A group column holding *both* missing values and a literal `"nan"`
label is rejected at write time, because normalizing would fuse two distinct populations under
one name.

---

## 9. Reading one cell — the whole algorithm

```
open(path):
    assert file[0:4] == b"SHPK"
    t = parse_trailer(file[-64:])                  # magic, footer_offset, footer_length, crc
    footer = json(file[t.footer_offset : t.footer_offset + t.footer_length])
    assert crc32(those bytes) == t.footer_crc32
    manifest = json(member(footer, "manifest.json"))
    assert manifest["layout"] == "cell" and manifest["schema_version"] in (5, 6)
    offsets = i64[]  from member(footer, "x_offsets")     # n_obs+1, offsets[0] == 0
    indptr  = i64[]  from member(footer, "x_indptr")      # n_obs+1, indptr[0] == 0
    payload = mmap over member(footer, "x_payload")       # 4 KB-aligned, so page-aligned

read_row(r):
    frame = payload[offsets[r] : offsets[r+1]]
    nnz   = indptr[r+1] - indptr[r]
    idx_len = u32_le(frame[0:4])                          # requires len(frame) >= 4
    assert 4 + idx_len <= len(frame)
    idx_blob, val_blob = frame[4 : 4+idx_len], frame[4+idx_len :]
    if manifest["codec"] == "pfordelta":
        gaps = simdfastpfor256_decode(idx_blob, nnz)      # exactly nnz u32 values
        cols = cumsum_u64(gaps);  assert nnz == 0 or cols[-1] < n_vars
        vals = simdfastpfor256_decode(val_blob, nnz)
    else:
        cols = un_byte_shuffle(un_byte_delta(zstd(idx_blob, expect=4*nnz)), u32)
        vals = (zstd(val_blob, expect=itemsize*nnz)               if float
                else un_byte_shuffle(zstd(val_blob, expect=4*nnz), u32))
    assert len(cols) == len(vals) == nnz
    return cols, vals                                    # cols ascending within the row
```

A whole-matrix read is this loop over `r in [0, n_obs)`, writing into a CSR whose `indptr` is
`x_indptr` verbatim. An arbitrary subset is the same loop over the requested rows with a
recomputed `indptr`; nothing else in the archive needs to be touched, which is the point of the
layout.

---

## 10. Validation summary

A conforming reader MUST reject an archive that violates any of these.
`tests/cell/test_format_spec_conformance.py` carries **negative controls** — a deliberately
damaged archive, and an assertion that the library refuses it — for items 1, 2, 4a, 4b, 4c, 5, 6, 7,
8, 18 and 21. Item 18's control is deliberately built so that narrowing before the check would
*hide* the violation rather than merely report it late, and it runs on both engines. The rest are
enforced by the reference implementation and covered elsewhere under `tests/` (notably
`tests/cell/test_column_bounds.py`, `test_cell_zstd_frame_cap.py`, `test_manifest_cell.py` and
`test_decode_bounds_subprocess.py`, whose sidecar-corruption cases pin both the open-time checks
and the decode-level bound).

**Container**

1. Head magic is not `SHPK`, or trailer magic is not `SHPKEND\0`.
2. Footer CRC-32 mismatch, or a footer range outside `[4096, filesize-64]`.
3. A required member is missing, or a member with `role == "shard"` is present.
4. A `.tailbak` whose recorded `shard_region_end` is out of range or not 4 KB-aligned.
4a. A member name that is not path-safe (§2.3), or a duplicate member name.
4b. A member whose `[offset, offset + length)` leaves `[4096, footer_offset)`, a negative
    `length`, or a short read against a declared `length`.
4c. `shard_region_end` outside `[4096, footer_offset]` or not 4 KB-aligned; a member offset that
    is not 4 KB-aligned for a role the reader `mmap`s.

**Shape**

5. `len(x_offsets) != n_obs+1`, `len(x_indptr) != n_obs+1`, or `x_offsets[-1]` disagreeing with
   the `x_payload` member's length. All are checked at open.
6. `x_offsets[0] != 0`, or `x_offsets` decreasing. Both are checked at open. (Monotonicity
   *also* fails later, when a decreasing pair makes a frame slice fail the §5 framing checks —
   but only for a row the reader actually parses, which is why the open-time scan matters.)
7. `x_indptr[0] != 0`, `x_indptr` decreasing, or `x_indptr[n_obs] != total_nnz`. All three are
   checked at open — adding a constant to every entry preserves the per-row differences, so
   without the endpoint check a shifted `x_indptr` served rows from a corrupt archive.

**Manifest**

8. `schema_version` not in `{5, 6}`, or disagreeing with `codec`.
9. `n_obs_total` / `n_vars` / `total_nnz` not a non-negative int.
10. `value_dtype_on_disk` / `index_dtype_on_disk` outside the per-version allow-list (§6.2).
11. `payload_sha256` present, non-`null`, and not 64 hex characters.
12. `indices_canonical` present and not a bool.
13. `pyfastpfor_codec` not a string on schema 5.

**Frame**

14. `len(frame) < 4`, or `4 + idx_len > len(frame)` — for any row the reader parses. Rows with
    `nnz_r == 0` are exempt on the Rust engine, which does not parse them at all (§5.3).
15. A pfordelta blob that is empty for `nnz > 0`, or whose length is not a multiple of 4.
16. A pfordelta blob longer than `8*nnz + 65536` bytes — a loose bound (real encodings run
    2–3 bytes/value) whose only job is to stop a corrupt `x_offsets` from making one frame span
    the payload and drive an allocation sized from it.
17. A decode producing a count other than `nnz`.
18. A column index `>= n_vars` (checked on the wide value, before narrowing).
19. A zstd chunk declaring more than 64 MiB, disagreeing with its expected size, or
    decompressing to a different length.

**Groups**

20. `groups.json` present without `group_by`, or `group_by` without `groups.json`.
21. Any §8 invariant: an empty record, a gap or overlap in the tiling, coverage short of
    `n_obs`, a reference label no record carries, reference records not forming a leading
    prefix, or a `reference` disagreeing with the manifest's.

---

## 11. Compatibility policy

* `schema_version` is a **monotonic gate**. New versions get a new number; a reader that does
  not know a version refuses the archive instead of guessing. Version 5 (pfordelta) and 6
  (zstd) are the cell layout's only versions.
* Additive metadata — a new manifest key, a new footer member entry key — does not bump the
  version. Readers MUST ignore keys they do not know, which is what makes that safe.
* The container (`SHPK` version 1) is shared by both layouts and has not changed.
* `layout="cell"` is currently marked **experimental** by the writer, which warns that the
  on-disk layout may change in a future release. This document specifies what versions 5 and 6
  are; it is not yet a stability promise that no version 7 will exist.
* Compression output is **not** a contract. The guarantee is **decode identity**, not byte
  identity of a re-encode: a zstd or FastPFor version bump may legitimately change the bytes
  for the same input.

  Within one build, the two engines *are* byte-identical where it matters. Verified: writing
  the same input with the Rust core and with `CELLSTREAM_DISABLE_RUST=1` produces
  `x_payload`, `x_offsets` and `x_indptr` with identical SHA-256s, on all three of
  pfordelta / zstd / grouped. **The whole file is not identical**, because
  `manifest.json` carries a wall-clock `created_at`. So a whole-file digest is the wrong
  reproducibility oracle for an archive — compare the data members, or `payload_sha256`.

---

## Appendix A — worked example

A 40 × 300 integer archive, `codec="pfordelta"`, grouped on a 3-label `obs` column with `ctrl`
as reference. Written by the reference implementation, and reproduced by
`tests/cell/test_format_spec_conformance.py`.

The **offsets** and the **structural** numbers below are what the test enforces. The two
byte *sizes* that depend on a third party are informative: `x_payload` on the FastPFor build,
`header.h5ad` on the installed `h5py` / `anndata`.

```
size 62206 bytes                            offset  length
  head block                                     0    4096   "SHPK" + u16 1, NUL-padded
  x_payload                                   4096    2536   40 frames, storage order
  x_offsets                                   8192     328   = 8 * 41
  x_indptr                                   12288     328   = 8 * 41 ; [40] == 960 == total_nnz
--- shard_region_end = 16384 ---------------------------------------------------------------
  header.h5ad                                16384   36120   empty CSR X, obs/var in storage order
  obs.parquet                                53248    2163
  manifest.json                              57344     599
  groups.json                                61440     158   ctrl:[0,10) a:[10,23) b:[23,40)
  footer (JSON, unaligned)                   61598     544
  trailer                                    62142      64   "SHPKEND\0", CRC-32 of the footer
```

`x_offsets[40] == 2536 == len(x_payload)`: frames are contiguous and cover the member exactly.

## Appendix B — the legacy shard layout

`layout="shard"` (`.shad`, `schema_version` 2–4) predates the cell layout and stores the matrix
as row-sharded `SHX2` binary members inside the same `SHPK` container, with a
`bitshuffle`-or-byte-filter + zstd chunk codec, a per-shard row partition (`row_offsets`,
`nnz_per_shard`), and optional multi-matrix families (`layers`, `.raw`, modalities). Its
`groups.json` is a JSON *array* of `{label, shard, row_start, row_stop, role}` records — a
different shape from the cell layout's object.

The code still reads and writes it and existing archives keep working, but it is **legacy and
unsupported**, and it is deliberately not specified here. New data should use `layout="cell"`.
