# cellstream

**Low-RAM, highly compressed, condition-grouped storage for large single-cell AnnData — built for
scattered random access.**

cellstream stores an `AnnData` count matrix as one flat file of **independently addressable per-cell
frames**. Reading an arbitrary set of cells decodes only those cells, so an `X[rows]` draw costs
roughly the same whether the archive holds a hundred thousand cells or ten million. Cells that share
a label — typically a perturbation (the target gene or guide in a CRISPR screen, or any `obs`
column) — are stored contiguously, so one condition can be read directly.

For integer count matrices a cellstream archive is several times smaller than a gzipped `.h5ad`, and
nothing about a read requires the whole matrix to be resident.

```python
import cellstream

# write a per-cell store, grouped by perturbation
cellstream.write_sharded(
    adata, "screen.csad", layout="cell",
    group_by="target_gene", reference=["non-targeting"],
)

store = cellstream.open("screen.csad")
myc = store.read_group("MYC")                    # just the MYC cells      -> CSR
few = store.gather_rows([7, 42, 1000, 500_001])  # 4 arbitrary cells       -> CSR
ref = store.read_reference()                     # the non-targeting block -> CSR
```

---

## Why cellstream

- **Scattered reads are cheap** *(the distinguishing feature)* — every cell is its own
  independently-decodable frame behind an `int64` offset index, so `gather_rows(row_ids)` touches
  only the frames it needs. There is no shard or chunk to decode around the cells you asked for.
- **Grouped by condition** — cells for one perturbation (or any `obs` label, set via `group_by`)
  are stored contiguously, with an optional reference group leading the file. `read_group(label)`
  returns just those cells; `group_spans()` exposes the exact `[start, stop)` layout so a caller
  can size or stream a read before issuing one.
- **Low RAM on both sides.** Reads materialize only the requested rows. Writes stream: a backed
  `.h5ad` source (CSR or dense) is consumed in row blocks, so peak memory is `O(block)` rather than
  `O(nnz)`, and `CellArchiveWriter` lets you push blocks as you produce them.
- **Small on disk** — integer counts go through a pfordelta codec (FastPFor over the delta-coded
  gene indices, and over the values); genuine floats through a byte-planed zstd codec. See
  [Benchmarks](#benchmarks).
- **Fast** — encode and decode both run through a native Rust core with a vendored FastPFor.
- **Annotations round-trip** — `obs`, `var`, `uns`, `obsm`, `varm`, `obsp`, `varp` are all
  preserved, and `read_h5ad(path)` rebuilds the object. ⚠️ `.layers` and `.raw` are **not** stored
  by the per-cell layout: the writer warns and drops them.

---

## Install

```bash
pip install cellstream
```

```bash
uv pip install cellstream        # or, inside a uv project: uv add cellstream
```

**Wheels cover Linux x86_64 with SSE4.1 and glibc ≥ 2.28, on CPython 3.11 and 3.12** (they are
tagged `manylinux_2_28`). They carry the prebuilt Rust extension, so nothing compiles at install
time and every dependency resolves to a wheel. This is the supported path, and the only
configuration this release claims — see
[Platform support](#platform-support-and-building-from-source). Check which one you are on:

```bash
cellstream runtime
# cellstream 0.9.1 | rust=yes | extension present and enabled
```

**conda / mamba** — create the environment with conda/mamba, then install cellstream with pip; all
of its dependencies are on conda-forge. cellstream itself is not yet packaged for conda-forge.

```bash
mamba create -n cellstream -c conda-forge python=3.11
mamba activate cellstream
pip install cellstream
```

### Optional extras

| extra | pulls in | when you need it |
|---|---|---|
| `cell` | `pyfastpfor` | Only for the **pure-Python** pfordelta codec, and only on x86_64. The Rust extension carries its own vendored FastPFor and never calls it, so a wheel install does not need this at all; reach for it if you run with `CELLSTREAM_DISABLE_RUST=1`. It is an extra rather than a dependency because `pyfastpfor` publishes no Linux wheel — installing it compiles C++. It is **not** an aarch64 workaround: `pyfastpfor` binds the same x86 FastPFor. |
| `test` | `pytest`, `pytest-benchmark`, `ruff` | running the test suite. |
| `dev` | the `test` set plus `mypy` | contributing. |
| `mudata` | `mudata` | the [legacy shard layout](#legacy-the-shard-layout) only — multi-modal reads are a feature of that layout, not of the per-cell store. |
| `gpu` | `cupy-cuda12x`, `nvidia-nvcomp-cu12` | likewise legacy-layout only: on-GPU decode of shard archives. |

⚠️ Neither `test` nor `dev` pulls `pyfastpfor`, so a bare `.[dev]` **silently skips** every
per-cell test on the pure-Python engine — install `.[dev,cell]`. CI installs `.[test,cell]` for
exactly this reason. `pre-commit` is not an extra at all: `pip install pre-commit` separately.

### Platform support, and building from source

**Supported: Linux x86_64 with SSE4.1, glibc ≥ 2.28, on CPython 3.11 or 3.12.** That is what the
wheels cover and what CI tests, and it is the only configuration this release claims.

The SSE4.1 requirement is real but **not expressed by the wheel tag** — `manylinux_2_28_x86_64` is
a glibc statement. The vendored FastPFor is compiled with `-msse4.1` and the shim asks for
`simdfastpfor256`, so on a CPU without SSE4.1 the wheel installs and the default pfordelta path
faults. Most current x86_64 hardware has it, but not all x86_64 hardware ever made does — AMD's
K10 / Phenom II line has SSE4a instead, and AMD gained SSE4.1 only with Bulldozer in 2011. Check
rather than assume:

```bash
grep -q sse4_1 /proc/cpuinfo && echo "SSE4.1: yes" || echo "SSE4.1: NO — the wheels will fault"
```

On a supported Python (3.11 or 3.12), where no wheel is published for your platform,
`pip install cellstream` falls back to the sdist and builds from source.

**On 3.13 there is no supported version — and `pip install cellstream` does not simply fail.**
0.9.1's `requires-python` excludes 3.13, so an unpinned install backtracks to an older release
that still accepts it: 0.9.0, or behind it the 0.0.1 name placeholder, unless those have been
yanked. Whatever lands there is unsupported — see the CHANGELOG's 0.9.1 entry for what goes wrong.
`pip install cellstream==0.9.1` on 3.13 fails outright, which is the intended behaviour.

Note what a source build does **not** buy you: a wheel tag cannot express an ISA
requirement, so on an x86_64 machine without SSE4.1 pip takes the *wheel* and the fault above still
applies — and forcing a source build would not help either, since `rust/build.rs` compiles with
`-msse4.1` regardless. Building from source lowers the glibc floor, not the ISA floor.

That build needs **two** toolchains present at install time — Rust, from [rustup.rs](https://rustup.rs) or
conda-forge, **and a C++11 compiler**, because `rust/build.rs` drives `cc` over the vendored
FastPFor sources. Both, not just Rust:

```bash
mamba install -c conda-forge rust cxx-compiler
```

(On a distro, the system compiler is usually already there — `build-essential` on Debian/Ubuntu,
`gcc-c++` on Fedora/RHEL.)

With no Rust toolchain it **fails loudly** rather than silently producing an install whose per-cell
reads are much slower with no indication — the extension is declared non-optional on purpose.
`CELLSTREAM_NO_RUST=1` at install time asks for a pure-Python build deliberately.

But a source build is **not** a portability story, and two platforms are worth naming here rather
than leaving you to discover them:

| platform | state |
|---|---|
| Linux or macOS **x86_64 with SSE4.1** | **Works.** The sdist builds the full Rust core. CI builds the sdist, installs it on Linux/CPython 3.11 and asserts the extension is present and complete; the full suite runs against a source checkout on 3.11 and 3.12 under both engines. macOS is not in CI. |
| **aarch64**, including Apple Silicon | **Unsupported.** The crate compiles, but the per-cell code path does not exist there: `rust/src/cell.rs` is `#![cfg(target_arch = "x86_64")]`, because the vendored FastPFor is x86 SIMD. So `rust_available()` is `False` and a per-cell read raises unless you set `CELLSTREAM_ALLOW_PYTHON_FALLBACK=1`. With that opt-in, a **zstd** archive does decode in pure Python and needs no extra — but **pfordelta**, the default codec for integer counts, has no working backend: its pure-Python path needs `pyfastpfor`, which is itself a binding to the same x86 FastPFor. |
| **Windows** | **Unsupported**, and not merely untested: `import cellstream` fails outright, because `cellstream/lock.py` imports `fcntl` at module scope. |

**Development install**

```bash
git clone https://github.com/ArcInstitute/cellstream.git
cd cellstream
pip install -e ".[dev,cell,mudata]"     # or: uv pip install -e ".[dev,cell,mudata]"
pytest tests/
```

Core dependencies: `anndata`, `h5py`, `hdf5plugin`, `numpy`, `scipy`, `pandas`, `pyarrow`,
`bitshuffle`, `zstandard`.

### Checking the runtime from Python

```python
import cellstream
assert cellstream.rust_available(), cellstream.rust_status()
```

`cellstream.rust_available()` is the conservative check: it requires the extension to be importable
**and** complete. The lower-level `cellstream.v2._rust_dispatch.rust_available()` only asks whether
`cellstream.v2._rust` imported, so it returns `True` for an extension built before the per-cell
entry points existed — exactly the half-working install the top-level check exists to catch.

At runtime a per-cell read on a build with no extension **raises**. The escape hatches are
deliberate and named: `CELLSTREAM_ALLOW_PYTHON_FALLBACK=1` proceeds anyway, and
`CELLSTREAM_DISABLE_RUST=1` forces the Python engine everywhere.

---

## Quick start

```python
import cellstream

# --- write -------------------------------------------------------------------
cellstream.write_sharded(adata, "cells.csad", layout="cell", overwrite=True)

# a path source streams in row blocks -- peak RAM is O(block), not O(nnz)
cellstream.write_sharded("big.h5ad", "big.csad", layout="cell", overwrite=True)

# --- read --------------------------------------------------------------------
store = cellstream.open("cells.csad")

store.shape                                    # (n_obs, n_vars)
store.gather_rows([7, 42, 1000])               # 3 arbitrary cells -> CSR
store.gather_rows(range(0, 5000), n_threads=8) # a contiguous block, 8-way parallel

lazy = store.to_anndata(backed=True)           # lazy AnnData view; X decodes on indexing
lazy.X[[7, 42, 1000]]                          # -> CSR of just those rows

full = cellstream.read_h5ad("cells.csad")      # the whole object, materialized

# --- metadata without decoding X ---------------------------------------------
store.obs, store.var, store.uns
store.obsm, store.varm, store.obsp, store.varp
```

`.csad` is a **convention**, not a requirement: nothing in the writer inspects the extension, and
`open()` identifies an archive from its container magic and manifest. `open()` is per-cell only and
raises a clear error on any other archive; `read_h5ad()` is the dispatching entry point that accepts
either layout (and a plain `.h5ad`).

---

## Condition grouping

Pass `group_by` (and optionally `reference`) at write time and the archive records where every group
lives:

```python
cellstream.write_sharded(
    adata, "screen.csad", layout="cell",
    group_by="target_gene", reference=["non-targeting"], overwrite=True,
)

store = cellstream.open("screen.csad")
store.is_grouped              # True  (a property, not a method)
store.group_labels()          # the available group labels
store.read_group("MYC")       # one perturbation's cells   -> CSR
store.read_reference()        # the reference block         -> CSR
store.reference_row_count()   # rows read_reference() would return; None if no reference
```

`sort_within_group=` and `sort_order=` set a secondary ordering inside each group (for example by
guide `assignment`).

### Sizing a read before issuing it

`group_spans()` returns each group's `[start, stop)` range in storage order — the reference block
leads, from row 0 — so you can stream group by group with a bounded working set:

```python
for span in store.group_spans():
    X = store.gather_rows(range(span.start, span.stop), n_threads=min(8, span.n_rows))
```

The spans are exact by construction. Deriving them from `obs[group_by].value_counts()` instead
**diverges**, because the writer stringifies labels: every missing value (`None` / `NaN` / `NaT` /
`pd.NA`) becomes the single group `'nan'` regardless of the column's dtype, archives written by
older versions keep whatever their writer produced (`'None'` / `'<NA>'`), and unused categorical
categories never reach the archive. Two distinct `obs` values that are equal once stringified **can**
yield several records sharing one label, whenever the raw sort order separates their runs — which is
why this returns a list. `read_group(label)` is the convenience form and requires the label to be
unique, so it raises on an ambiguous one and points you back here to address the spans by position.

---

## Writing incrementally, and concatenating

`CellArchiveWriter` is an append-only push writer: hand it CSR blocks as you produce them and it
encodes each one straight into the archive, never holding the full matrix.

```python
from cellstream.cell import CellArchiveWriter, concat_cell_archives

with CellArchiveWriter(
    "out.csad", n_vars=adata.n_vars, value_dtype="uint32", overwrite=True
) as w:
    for block in produce_csr_blocks():
        w.add_block(block)
    w.finalize(obs=obs_df, var=var_df)

# N-way concatenation, by byte-copy -- frames are never re-encoded
concat_cell_archives(["a.csad", "b.csad"], "ab.csad", annotations="preserve")
```

`X`, `obs`, `var` and `uns` are **always** merged. `annotations` controls only the four
annotation matrices: `"preserve"` (the default) carries `obsm` row-concatenated, `obsp` assembled
**block-diagonal** (per-part graphs have no cross-part edges), and `varm`/`varp` inherited from
part 0 after a require-equal check across the rest.

`"drop"` omits those four from the **output**. It is the escape hatch from the *merge* cost — a
dense `obsp` assembled block-diagonal is (Σnᵢ)² — but it is not a promise that they are never read:
the require-equal checks on `var` and `uns` load each part's `header.h5ad` in full, so a part
carrying a large dense pairwise annotation is still materialized while those checks run.

---

## Integrity

| Knob | Default | Effect |
|---|---|---|
| `payload_checksum` | `False` | Store a per-payload checksum. `store.verify_payload()` then re-reads and verifies every frame. Off by default because it costs roughly 21% of write time. ⚠️ Check `store.has_payload_checksum` first: with no stored checksum, `verify_payload()` returns `True` **vacuously** — there is nothing to verify. |
| `require_unique_indices` | `True` | Reject a duplicate `(cell, gene)` entry rather than writing a frame that decodes to something else. Column indices are **sorted** first, so unsorted input is accepted — this catches duplicates, not disorder — and a clean archive is marked canonical so the reader can skip scipy's O(nnz) validation. |

Narrowing casts fail loud with `LossyCastError` unless `allow_lossy=True`. Every
**package-specific** exception class inherits from `cellstream.CellstreamError`; ordinary argument
validation still raises built-in `ValueError` / `TypeError` / `FileNotFoundError`.

---

## What's in a `.csad` archive

A per-cell archive is a single offset-indexed container bundling:

| Member | Contents |
|--------|----------|
| `manifest.json` | Schema version, codec, `n_obs`/`n_vars`, on-disk dtypes, writer fingerprint — the index that ties everything together. |
| `header.h5ad` | An empty-`X` AnnData carrying `obs`, `var`, `uns`, `obsm`, `varm`, `obsp`, `varp` and the encoding info. `.layers` and `.raw` are not stored. |
| `obs.parquet` | The `obs` DataFrame again, as the reader's fast path: `store.obs` reads this member and never touches `header.h5ad`. |
| the payload | One independently-decodable frame per cell, plus the `int64` offset index that addresses them. |
| `groups.json` | *(grouped archives only)* each group label → its row range, with `role: reference \| group`. Powers `read_group` / `read_reference` / `group_spans`. |

The container and the per-cell layout are specified byte-for-byte in
[docs/format.md](https://github.com/ArcInstitute/cellstream/blob/main/docs/format.md) — enough to
write an independent reader. That document is not prose *about* the format:
`tests/cell/test_format_spec_conformance.py` implements a reader from its text alone, importing none
of the library's container or codec helpers, and checks it against the real one across every
supported codec/dtype combination — both codecs × {narrow ints, wide ints, float32, float64} ×
{grouped, ungrouped}, minus the combinations the writer cannot produce (pfordelta stores integers
only) — plus negative controls. So a change to the library that departs from the documented layout
turns that test red. It is a drift detector, not a proof: the independent reader is hand-written
*from* the prose rather than generated from it, so an error in the prose that nobody encoded is
still an error in the prose.

---

## API reference

### Writing — `write_sharded(adata, out, layout="cell", ...)`

The signature default is `layout="shard"`; every call in this README passes `layout="cell"`
explicitly.

`adata` is an in-memory `AnnData`, a path to a source `.h5ad` (CSR or dense `X`, streamed in row
blocks), or an existing per-cell archive (the re-encode / codec-migration case). `out` is the file
path.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `layout` | `"shard"` | Pass `"cell"` for the per-cell store. (`"shard"` is the [legacy layout](#legacy-the-shard-layout).) |
| `group_by` | `None` | `obs` column to group by: cells are sorted by this label and stored contiguously. |
| `reference` | `None` | Reference cells placed first — a bool mask of length `n_obs`, or label values present in `obs[group_by]`. Requires `group_by`. |
| `sort_within_group` | `None` | Secondary `obs` column(s) to sort cells within each group. |
| `sort_order` | `None` | `"natural"` (native-dtype order; the default) or `"alphabetical"`. |
| `codec` | `None` | Auto-selects: integer counts → `"pfordelta"`, genuine non-integer floats → `"zstd"`. For `pfordelta` — not for `zstd`, which stores genuine floats — every value must be **finite, non-negative, integral and representable as `uint32`**. An explicit `"pfordelta"` on float input is therefore still *attempted*, because a float column may hold only integer-valued floats, and fails loud at encode time if **any** value breaks one of those four conditions (fractional, negative, non-finite, or above the `uint32` maximum); `CellArchiveWriter`, which sees only a declared dtype, rejects a float `value_dtype` up front instead. `"zstd"` is opt-in for integer counts (smaller on dense cells). pfordelta archives are `schema_version` 5, zstd archives 6. |
| `data_dtype` | `None` | On-disk float width for `codec="zstd"` float data. `None` preserves the input width. Integer counts always store as `uint32`. |
| `allow_lossy` | `False` | Permit a narrowing cast instead of raising `LossyCastError`. |
| `block_size` | `None` | Rows per streamed input block, for a path or archive source. |
| `payload_checksum` | `False` | See [Integrity](#integrity). |
| `require_unique_indices` | `True` | See [Integrity](#integrity). |
| `overwrite` | `False` | Replace an existing archive. |
| `tmp_dir` | `None` | Staging directory for the write. |

`cellstream.cell.write_cell_archive()` is the same writer without the layout dispatch.

### Reading — `cellstream.open(path)` → `CellStore`

| Method | Returns | Notes |
|--------|---------|-------|
| `.gather_rows(row_ids, n_threads=1, out_dtype=None)` | `csr_matrix` | Arbitrary row ids, in the order given. The core read. |
| `.gather_rows_adata(row_ids, n_threads=1)` | `AnnData` | The same, with the matching `obs`/`var` attached. |
| `.read_group(label, n_threads=1)` | `csr_matrix` | *(grouped)* one group's cells. Raises if `label` is ambiguous. |
| `.read_group_adata(label, n_threads=1)` | `AnnData` | The same, with annotations. |
| `.read_reference(n_threads=1)` | `csr_matrix` \| `None` | *(grouped)* the reference block. |
| `.group_labels()` / `.group_spans()` / `.reference_row_count()` | — | Group layout, readable before any read. `.is_grouped` is a **property**. |
| `.to_anndata(backed=True)` | `AnnDataView` | Lazy, AnnData-*like* view (`cellstream.cell.AnnDataView`, not an `anndata.AnnData`): `X` decodes on indexing. `backed=False` raises — use `read_h5ad(path)` for a materialized object. |
| `.obs` / `.var` / `.uns` / `.obsm` / `.varm` / `.obsp` / `.varp` | — | Annotations, without decoding `X`. |
| `.shape` | — | `(n_obs, n_vars)`. (`.n_rows` is a `GroupSpan` property, not a store one.) |
| `.has_payload_checksum` (**property**) / `.verify_payload()` | `bool` | See [Integrity](#integrity). |
| `.close()` | `None` | Release the mmap and the file handle. |

Top-level helpers: `read_h5ad(path)`, `read_obs(path)`, `read_var(path)` — all auto-detect the
container.

---

## How it works

1. **One frame per cell.** Each cell's CSR row is compressed on its own and addressed through an
   `int64` offset index, which is what makes a scattered read proportional to the cells requested
   rather than to the archive.

2. **Codec chosen by the data.** Integer counts use **pfordelta**: a frame is a FastPFor pack of
   the delta-gaps between that cell's sorted gene indices, followed by a FastPFor pack of its
   values. There is **no zstd stage** on this path — the frame-of-reference delta coding is where
   the compression comes from. Genuine floats use a separate **zstd** codec with a byte-plane
   filter, because byte-shuffling float values destroys the whole-value repeats zstd exploits. Each
   archive self-describes: `manifest["codec"]` and `manifest["schema_version"]` name the codec for
   the whole archive — there is deliberately **no per-frame marker**, since a per-row tag would
   cost bytes on every cell to express something uniform — and a reader that does not recognize
   the name **raises** rather than guessing, so the format is forward-safe.

3. **Native encode and decode.** Both directions run through a Rust core that vendors FastPFor and
   decodes across worker threads straight into the output buffer. The pure-Python fallback is
   byte-identical — the Rust `zstd` crate vendors the same libzstd as `python-zstandard` — so
   nothing about the output depends on which path runs. Reads and writes are tested under both
   engines.

4. **Grouping is a storage order, not an index.** `group_by` sorts cells before writing and records
   each group's row range, so a grouped read is a contiguous frame range — no per-cell lookup, and
   no cost for ungrouped archives.

---

## Benchmarks

⚠️ **The published sweep below measures the legacy shard layout, not the per-cell layout.** It is
kept because it establishes the compression and the `.h5ad` baselines. An equivalent end-to-end
sweep for `layout="cell"` has not been run; when it is, this section will be replaced rather than
extended.

Two production CRISPR perturbation screens, referred to by coded IDs:

- **CCL_1** — 1,209,053 cells × 18,533 genes, **7.39 B nnz**; uncompressed `.h5ad` 89.1 GB.
- **CCL_2** — 5,540,000 cells × 18,533 genes, **36.08 B nnz**; uncompressed `.h5ad` 434.9 GB.

The `.h5ad` baselines are the realistic file a user actually has — a standard scipy CSR (float32
`X.data`, native int indices) written via `anndata.write_h5ad` at no compression and at
`compression="gzip"` (anndata's default level 4). Two representations are shown: **counts**
(integers) and **log-norm** (`normalize_total` + `log1p`, float32). Reads are **warm**, 16 workers,
Rust core.

**CCL_1 · 1.21 M cells · 7.39 B nnz** — cells show **counts / log-norm**:

| | uncompressed `.h5ad` | gzip-4 `.h5ad` | **cellstream (shard)** |
|---|---:|---:|---:|
| **Size on disk** | 89.1 / 89.1 GB | 17.6 / 17.3 GB | **6.3 / 9.3 GB** |
| **Write** | 61 / 54 s | 1218 / 1082 s | **61 / 71 s** |
| **Read → AnnData (warm)** | 35 / 36 s | 247 / 231 s | **4.7 / 8.4 s** |

**CCL_2 · 5.54 M cells · 36.1 B nnz** — cells show **counts / log-norm**:

| | uncompressed `.h5ad` | gzip-4 `.h5ad` | **cellstream (shard)** |
|---|---:|---:|---:|
| **Size on disk** | 435 / 435 GB | 87 / 86 GB | **31 / 47 GB** |
| **Write** | 523 / 663 s | 6759 / 6042 s | **306 / 320 s** |
| **Read → AnnData (warm)** | 492 / 484 s | 1230 / 1148 s | **57 / 40 s** |

The count archive is **~14× smaller than an uncompressed `.h5ad`** and **~2.8× smaller than a
gzip-4 one**, written in about the time the uncompressed write takes and read back several times
faster. Cold and remote reads benefit most — there are far fewer bytes to move.

---

## Legacy: the shard layout

`layout="shard"` (still the `write_sharded` default) co-packs cells into ~GB shards and is read
through `ShardedArchive`. It predates the per-cell layout, and it is **legacy and unsupported**: the
code ships, stays importable and stays tested, but it is not documented here and new work should use
`layout="cell"`.

---

## Format stability

See [docs/format.md](https://github.com/ArcInstitute/cellstream/blob/main/docs/format.md) for the
on-disk specification of the container and the per-cell layout, including its compatibility policy.

The v2 shard format is **frozen**: every release keeps reading v2 archives written by v0.3.0+,
enforced by committed golden fixtures and a reverse-compatibility test. The per-cell format
(`schema_version` 5 and 6) is **experimental and may change**.

Both are self-describing: a per-cell archive names its codec and schema version in
`manifest.json`, archive-wide, and a reader that does not recognize either one **raises** instead
of guessing. So an incompatible future change would take a new codec name or schema version and
leave existing archives readable.

---

## Contributing, and what is open

Issues and pull requests are welcome. `pip install -e ".[dev,cell,mudata]"` and `pytest tests/` is
the whole loop; `pip install pre-commit && pre-commit install` adds the `ruff` and `cargo fmt`
gates CI enforces (`pre-commit` is not one of the extras). The test
suite runs under both engines — the Rust core and, with `CELLSTREAM_DISABLE_RUST=1`, the pure-Python
fallback — and a change to either path is expected to keep both green.

cellstream first became public in 0.9.0, after a substantial private development history, so
expect open ends: the per-cell format is marked experimental for a reason, and correctness work on
the layout continues. Bug reports with a reproducing archive are the most useful thing you can send.

---

## License

MIT — see [LICENSE](https://github.com/ArcInstitute/cellstream/blob/main/LICENSE).
Copyright (c) 2026 Arc Research Institute.
