# Changelog

All notable changes to **cellstream** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **A path `stat()` is no longer used to decide how many bytes are on disk.** On a filesystem that
  caches file attributes -- a distributed filesystem such as WekaFS, or NFS -- `stat(path)` can
  report a stale-SHORT `st_size` immediately after an append through a different fd, while `fstat`
  on an open fd, `lseek(SEEK_END)` and the bytes themselves are all already correct. No data is
  lost; only the cached attribute lags, and it settles within a fraction of a second.

  Eight call sites sized a file that way. Seven now measure through an fd rather than the path, and
  the eighth derives its number from the write it just performed. Three of them -- the storage-order
  step in Rust, the archive-open trailer check and the tail backup -- size and read through ONE
  handle, so for those the path cannot be swapped in between; the rest close the fd again. What the
  change buys everywhere is a size that is not a stale cached attribute, which is the whole bug. It
  is not a claim about a concurrent writer on the same inode.

  - the per-cell writer's storage-order step, both on its identity fast-path and in
    `reorder_frames` on both engines. This is where the `reorder_frames: in_offsets[-1] != staging
    file size` failures described under 0.9.1 came from: the staging file was intact and complete,
    and the guard was reading a stale size for it.
  - the push writer's rollback point after a rejected block. A stale-short value truncated away
    bytes belonging to blocks written EARLIER, so one rejected block could destroy a long, healthy
    push write rather than only itself.
  - the packed writer's member length, which went into the archive footer while the copy loop wrote
    the whole file regardless. A stale-short value recorded a member SHORTER than the bytes present
    -- an archive that reported a successful write and was wrong on disk. Both the per-cell and the
    sharded layout go through this packer.
  - the in-place tail-edit backup, where a stale-short size backed up a truncated tail, and the
    archive-open trailer check, where it made a perfectly good, freshly written archive look
    invalid.

  The guards themselves are unchanged: a genuinely wrong offset table still fails loudly. All seven
  Python sites have regression tests that inject the divergence directly -- `os.stat` short,
  `os.fstat` truthful -- so they run anywhere, including on the local disks CI uses, where the real
  staleness never occurs. The Rust site cannot be injected from Python; it is covered by its
  existing corrupt-input tests plus verification against a real distributed filesystem.

- **`MatrixSpec.__eq__` returns a bool again on CPython 3.13.** 3.13 generates a pairwise field
  comparison for dataclasses instead of comparing the two field tuples, which loses the identity
  shortcut and the bool coercion that `PyObject_RichCompareBool` provided -- so with a sparse
  matrix or a DataFrame in a field, `==` returned the elementwise result instead of `True`, and
  anything asking for its truth value raised. Present in 0.9.0 and 0.9.1. This is the "failure in
  an unrelated subsystem" mentioned under 0.9.1, and it is unrelated to the write path.

- **Two error messages no longer point at a repository you cannot reach.** Opening a legacy
  v1/directory archive, and passing `format="v1"` to `write_sharded`, both answered with a
  `pip install` of a git URL for this project's pre-rename repository at tag v0.4.0. That
  instruction could not work for anyone reading it: the repository is not public, and v0.4.0 was
  never uploaded to PyPI either. Both messages now say what is true instead -- no installable
  release can read such an archive, so re-encode from the original source data with the current
  writer -- and `PackedOnlyError`'s docstring no longer promises an install command it does not
  carry. The release tooling gained a check that refuses to publish if a non-public repository URL
  reappears in a shipped file, including one a Python source file assembles from adjacent string
  literals so that no single line contains it.

### Changed

- **The CPython 3.13 exclusion is lifted; `requires-python` is `>=3.11` again.** 0.9.1's cap rested
  on the intermittent write failures above being a 3.13 defect. They were not: with the dependency
  stack held constant and 1500 writes per version, **3.11 failed 20 times, 3.12 8 and 3.13 12, and
  a local disk zero times on every version** -- 3.11 was the worst arm. Two clean 3.11/3.12 runs at
  roughly one failure per thousand tests were luck, not a control.

  3.13 has no `bitshuffle` wheel, so installing there builds that sdist; that is a build cost, not
  a failure. **CI now runs 3.11, 3.12 and 3.13** and the classifiers claim all three; the published
  **wheels** stay cp311/cp312, which is a distribution fact rather than a support one.

  **The yanks on 0.9.0 and the 0.0.1 placeholder stay**, for a better reason than the one they were
  applied for: every *functional* release before this fix carries the stale-`stat()` bug on a
  distributed filesystem, whatever the Python version. (0.0.1 is a name placeholder with no
  working code, so it stays yanked for being unusable rather than for the bug.)

## [0.9.1] — 2026-08-19

### Fixed

- **`requires-python` now excludes CPython 3.13** (`>=3.11,<3.13`). 0.9.0 declared `>=3.11`, so
  pip installed it on 3.13 -- where both this package and `bitshuffle` build from source,
  `import cellstream` succeeds, and `cellstream runtime` reports
  `rust=yes | extension present and enabled`. The test suite then fails **non-deterministically**:
  three runs on 3.13 each failed a different one or two tests, every one of which passes in
  isolation. Two shared a signature in the per-cell write path --
  `ValueError: reorder_frames: in_offsets[-1]=… != staging file size …`, the staged file's size
  disagreeing with the offsets the encoder had accumulated -- and a third run failed in an
  unrelated subsystem, so it is not confined to that path. **The root cause is not diagnosed.**

  One failure seen in that third run has been *excluded* from this evidence rather than counted:
  `tests/test_out_of_shm.py` asserts on machine-global `/dev/shm` state, and it fails from
  contention with any other process on the same machine. It failed identically on **3.12**, and
  passes in isolation on both, so it is environmental and says nothing about 3.13.

  The resolved dependency *versions* do not explain it, which was checked rather than assumed:
  CPython **3.12** with the *identical* resolved stack (pandas 3.0.5, anndata 0.13.2, numpy 2.5.2,
  scipy 1.18.0) runs the per-cell suite clean, and so does 3.11 both from the published wheel and
  from a source build of this package's sdist. One difference is **not** controlled: 3.12 installed
  `bitshuffle` from a wheel, while 3.13 has none and had to build its sdist -- and a build that
  succeeds does not by itself establish runtime correctness. So the supportable statement is
  narrower than "it is 3.13": the failures have been observed only in the 3.13 environments
  tested, and the resolved dependency versions do not account for them.

  A package that installs, imports, and reports a working native extension before intermittently
  failing a write is the wrong thing to leave installable. Nothing changes for 3.11 or 3.12, and
  support can return once the failures are understood -- this is a deliberate stop, not a claim
  that 3.13 is unsupportable.

  **What this bound does and does not do.** Compatibility metadata is per release, so it applies to
  0.9.1 alone. On its own that only makes 3.13 skip 0.9.1 and backtrack to whatever older release
  still accepts 3.13: 0.9.0 (`>=3.11`), and behind that the 0.0.1 name placeholder (`>=3.10`),
  which contains no working code. **Both have therefore been yanked**, so a 3.13 resolver now has
  nothing to fall back to and an unpinned `pip install cellstream` there fails outright, which is
  the intended behaviour. A yank is a soft signal rather than a deletion: an *exact* pin such as
  `pip install cellstream==0.9.0` still installs, and is unsupported.

### Changed

- The *Known limitations* note on CPython 3.13 claimed it was blocked "on a dependency ... rather
  than on anything in this package". **That claim was incorrect when written**, not merely
  outdated, and has been corrected in place.

## [0.9.0] — 2026-08-19

**First public release.** cellstream has a substantial development history in a private
repository, where it was previously named `shardad`; that history is not reproduced here, which
is why this changelog starts partway up rather than at 0.1.0. The version number continues the
internal series (0.8.x), so nothing published under this name goes backwards. A `cellstream 0.0.1`
placeholder was uploaded to PyPI in July 2026 to reserve the name and contains no working code —
0.9.0 is the first real release.

### Added

- **`layout="cell"` — the per-cell store.** An `AnnData` count matrix stored as one flat file of
  independently addressable per-cell frames behind an `int64` offset index, so reading an
  arbitrary set of rows decodes only those rows. `cellstream.open(path)` returns a `CellStore`
  with `gather_rows`, `read_group`, `read_reference`, `group_spans` and lazy
  `to_anndata(backed=True)`; `cellstream.read_h5ad(path)` materializes the whole object and also
  accepts a plain `.h5ad`. `.csad` is the file-name convention; the reader identifies an archive
  from its container magic and manifest, not its extension.
- **Condition grouping as a storage order.** `group_by` (plus an optional `reference`) sorts cells
  before writing and records each group's `[start, stop)` row range, so a grouped read is a
  contiguous frame range with no per-cell lookup and no cost for ungrouped archives.
  `sort_within_group` / `sort_order` set a secondary ordering.
- **Two codecs, chosen by the data.** Integer counts use pfordelta (FastPFor over the delta-coded
  gene indices, then over the values, with no zstd stage); genuine floats use zstd behind a
  byte-plane filter. Each archive names its codec and schema version in `manifest.json`,
  archive-wide — there is deliberately no per-frame marker — and a reader that does not recognize
  either one raises instead of guessing. pfordelta archives are `schema_version` 5, zstd archives 6.
- **A native Rust core** for both encode and decode, vendoring FastPFor, decoding across worker
  threads straight into the output buffer. The pure-Python fallback is byte-identical, and the
  suite runs under both engines.
- **Streaming and incremental writes.** A backed `.h5ad` source is consumed in row blocks, so peak
  memory is `O(block)` rather than `O(nnz)`. `CellArchiveWriter` is an append-only push writer;
  `concat_cell_archives` joins archives N-way by byte-copy, without re-encoding a single frame.
- **Annotation round-trip** for `obs`, `var`, `uns`, `obsm`, `varm`, `obsp` and `varp`. `.layers`
  and `.raw` are **not** stored by the per-cell layout — the writer warns and drops them.
- **Integrity controls.** `require_unique_indices=True` (default) rejects a duplicate
  `(cell, gene)` entry instead of writing a frame that decodes to something else; opt-in
  `payload_checksum=True` plus `store.verify_payload()` re-reads and verifies every frame.
  Narrowing casts raise `LossyCastError` unless `allow_lossy=True`.
- **[`docs/format.md`](docs/format.md) — the on-disk format specification**, container and
  per-cell layout byte for byte: the head block, 4 KB-aligned members, the JSON footer and its
  CRC-32, the trailer, the offset index, both codec framings, the manifest field by field, and a
  compatibility policy. Enough to write an independent reader.
- **A conformance gate for that document.** `tests/cell/test_format_spec_conformance.py`
  implements a reader from the specification text alone — importing none of the library's
  container or codec helpers — and checks it against the real one across every supported
  codec/dtype combination: both codecs × {narrow ints, wide ints, float32, float64} × {grouped,
  ungrouped}, minus the combinations the writer cannot produce (pfordelta stores integers only),
  with negative controls. A change to the library that departs from the documented layout turns
  that test red. It is a drift detector rather than a proof: the independent reader is written
  *from* the prose, not generated from it.
- **`cellstream` console script**: `cellstream runtime` prints one line describing the Rust
  runtime, `cellstream info` prints a packed file's member table.
- **Linux x86_64 wheels for CPython 3.11 and 3.12** (`manylinux_2_28`, so glibc ≥ 2.28; the
  vendored FastPFor also needs SSE4.1, which the wheel tag cannot express — check
  `grep sse4_1 /proc/cpuinfo` rather than assuming, since AMD's K10 line has SSE4a instead),
  carrying the prebuilt
  Rust extension, plus an sdist. Every dependency of a wheel install resolves to a wheel —
  enforced in CI with `PIP_ONLY_BINARY=:all:`, not just intended. See *Known limitations* for what
  the sdist does and does not make possible.

### Known limitations

- **The per-cell format (`schema_version` 5 and 6) is experimental and may change.** An archive
  names its codec and schema version in `manifest.json` and a reader raises on either being
  unrecognized, so an incompatible change would take a new name or version and leave existing
  archives readable — but the format is not yet frozen.
- **Linux x86_64 with SSE4.1 and glibc ≥ 2.28, CPython 3.11 or 3.12, is the only supported
  configuration.** An sdist ships and does build a working, Rust-enabled install on Linux or
  macOS **x86_64 with SSE4.1**, given both a Rust toolchain and a C++11 compiler (`rust/build.rs`
  drives `cc` over the vendored FastPFor). Building from source does not lower the ISA floor —
  `-msse4.1` is applied either way — and it is not a portability story; the gaps are in the
  package, not in the packaging:
  - **aarch64, including Apple Silicon:** the crate compiles, but `rust/src/cell.rs` is
    `#![cfg(target_arch = "x86_64")]` (the vendored FastPFor is x86 SIMD), so the per-cell entry
    points are absent. `rust_available()` is `False` and a per-cell read raises unless the caller
    opts into the pure-Python fallback. With that opt-in a **zstd** archive decodes, and needs no
    extra; **pfordelta** — the default for integer counts — does not, because its pure-Python path
    needs `pyfastpfor`, itself a binding to the same x86 FastPFor.
  - **Windows:** `import cellstream` fails — `cellstream/lock.py` imports `fcntl` at module scope.
  - **CPython 3.13** is not supported. This entry originally said it was blocked "on a dependency
    (`bitshuffle` publishes no cp313 wheel) rather than on anything in this package", which was
    incorrect: bitshuffle's sdist builds on 3.13, as does this package's, and the result imports
    and reports a working Rust extension. The real blocker is that the suite then fails
    non-deterministically -- see the 0.9.1 entry. Note that **0.9.0's own `requires-python` does
    not exclude 3.13**; 0.9.1 is the release that does.
  Making aarch64 or Windows work is ordinary future work, not a packaging switch.
- **`layout="shard"` is legacy and unsupported.** It remains the `write_sharded` default for
  backwards compatibility, still ships, still imports and is still tested, but it is not
  documented and new work should pass `layout="cell"`. The GPU decode path and multi-modal
  (`mudata`) reads belong to that layout, not to the per-cell store.
- **`.layers` and `.raw` are not stored** by the per-cell layout.

[0.9.1]: https://github.com/ArcInstitute/cellstream/releases/tag/v0.9.1
[0.9.0]: https://github.com/ArcInstitute/cellstream/releases/tag/v0.9.0
