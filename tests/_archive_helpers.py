"""Shared archive builders for the test suite (#154 part B).

Two builders, one set of accessors:

* ``write_archive`` -- the PUBLIC packed writer. **The default.** Use it unless the
  test genuinely needs loose members on disk.
* ``write_directory`` -- the INTERNAL v2 writer, for the minority of tests that need
  an on-disk directory (byte tampering, member deletion, member paths handed to a
  path-taking API, directory goldens).

**Two defaults differ between the public and internal writers, and both are observable
in the archive** -- so a fixture built on the internal writer's own defaults would
silently differ from its packed twin: green, and testing something else.

==========================  ==========================  ==============================
knob                        public (``write_sharded``)  internal (``write_v2_archive``)
==========================  ==========================  ==============================
``codec=None``              ``"v2-byte-filter"``        ``CODEC_NAME`` (bitshuffle)
``n_workers``               ``16``                      ``8`` -- and it is recorded in
                                                        the manifest as
                                                        ``writer_fingerprint.n_workers_at_write``
==========================  ==========================  ==============================

``target_shard_bytes`` is a third case but not a differing default: both sides default
to 2 GiB. What differs is an **explicit** ``target_shard_bytes=None``, which many call
sites pass -- ``write_sharded`` turns it into 2 GiB, while the internal writer takes it
straight into ``if target_shard_bytes <= 0`` and dies on ``NoneType <= int``. So an
unnormalized ``None`` fails loudly rather than resizing anything -- the value that WOULD
slip through unnoticed is a wrong-but-positive one (1 GiB where the public writer uses
2 GiB is invisible to any shard-count check on a small fixture). ``write_directory``
normalizes it, so a migrated call site can keep passing the ``None`` it always passed.

``write_directory`` is **NOT** a drop-in for ``write_sharded(container="directory")``.
It reproduces the writer's defaults and source resolution, but not the public API's
surface: see ``UNSUPPORTED_KWARGS`` (rejected outright) and ``UNMIRRORED_VALIDATIONS``
(accepted, but without the public preflight). A test that asserts a public preflight
error -- or that a public knob is *forwarded* -- belongs on ``write_archive``, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad

from cellstream.manifest import load_groups, load_manifest, shard_filename
from cellstream.packed.reader import PackedArchive, is_packed_file

#: The codec the PUBLIC writer resolves ``codec=None`` to (write.py).
PUBLIC_V2_CODEC = "v2-byte-filter"
#: The PUBLIC writer's n_workers default. The internal writer's is 8, and the value
#: lands in the manifest under writer_fingerprint.n_workers_at_write.
PUBLIC_N_WORKERS = 16
#: What write_sharded turns ``target_shard_bytes=None`` into before calling the writer.
DEFAULT_TARGET_SHARD_BYTES = 2 * 2**30

#: Public ``write_sharded`` keywords the internal writer does not accept AT ALL: passing
#: any of them through ``write_directory`` raises ``TypeError: write_v2_archive() got an
#: unexpected keyword argument``, not the public error a test would be asserting on.
#: (Fail-loud, not silent -- but the wrong failure.) One exception: a non-empty
#: ``modalities`` is caught by the multi-matrix guard below and gets that diagnostic
#: instead, which is the intended message rather than an incidental one.
#: ``blosc_nthreads`` was in this list until #154 part B removed it from ``write_sharded``
#: with the v1 writer -- it is no longer a public write keyword at all, and
#: ``test_the_two_keyword_lists_match_the_real_signatures`` checks every name here IS one.
UNSUPPORTED_KWARGS = (
    "format",
    "layout",
    "modalities",
    "primary_modality",
    "payload_checksum",
    "require_unique_indices",
    "data_dtype",
    "allow_lossy",
    "block_size",
    "tmp_dir",
)

#: Public keywords removed outright rather than mirrored by the internal writer.
REMOVED_KWARGS = ("container",)

#: Keywords the internal writer DOES accept, but whose public preflight lives in
#: ``write_sharded`` and is therefore skipped here:
#:   * ``write_nthreads > 0``
#:   * ``sort_within_group`` / ``sort_order`` -- the internal writer ignores sort knobs
#:     without ``group_by`` and treats an unknown ``sort_order`` as natural
#:   * ``dense_grouped_via_csr`` constraints
#: The internal writer DOES duplicate: unknown-codec, positive target size,
#: ``n_shards``+``group_by``, and ``reference``+``group_by``.
UNMIRRORED_VALIDATIONS = (
    "write_nthreads",
    "sort_within_group",
    "sort_order",
    "dense_grouped_via_csr",
)

#: A test asserting a public preflight error, or that ``write_sharded`` FORWARDS one of
#: these knobs, is testing the PUBLIC API and must stay on ``write_archive``.
NOT_EQUIVALENT = UNSUPPORTED_KWARGS + UNMIRRORED_VALIDATIONS


def write_archive(adata, path, **kw) -> Path:
    """Packed archive via the public writer. The default for migrated fixtures.

    ``path`` needs no extension -- packed archives are detected by SHPK magic, so a
    fixture keeps whatever path it already used.
    """
    from cellstream.write import write_sharded

    # setdefault alone is not a promise. The guard predates #154 part B3b, when the suite
    # ran with packed-only enforcement disabled and container="directory" would quietly hand
    # back a DIRECTORY from a builder whose whole contract is "packed". Both escapes are now
    # closed by write_sharded itself (container= is a TypeError, format="v1" a ValueError),
    # so this is belt-and-braces -- kept because it is this builder's own contract, keeps the
    # message specific to write_archive, and covers whatever a future format value would do.
    if kw.setdefault("format", "v2") != "v2":
        raise ValueError(
            f"write_archive builds a v2 PACKED archive and accepts only format='v2'; "
            f"got {kw['format']!r}. Since #154 part B, 'v2' is the ONLY format write_sharded "
            "accepts either -- so drop the argument rather than looking for another builder."
        )
    if "container" in kw:
        raise ValueError(
            "write_archive is always packed, and #154 part B3 removes the container "
            "parameter entirely -- drop it from the call."
        )
    write_sharded(adata, path, **kw)
    return Path(path)


def write_directory(
    adata, out_dir, *, codec=None, n_workers=PUBLIC_N_WORKERS, target_shard_bytes=None, **kw
) -> Path:
    """On-disk directory archive via the internal writer.

    Only for tests that genuinely need loose members. Reproduces the public writer's
    defaults and source resolution so a directory fixture and a packed one differ in
    container only -- but NOT its validation (see ``NOT_EQUIVALENT``).

    Multi-matrix inputs are rejected: ``write_sharded`` routes ``.layers`` / ``.raw`` /
    MuData to the multi-matrix packed writer, so accepting them here would silently
    write X only.
    """
    from cellstream.v2.writer import write_v2_archive

    # Reuse the writer's own detector: it peeks a PATH source with h5py, handles MuData,
    # and routes through real_layer_names (the anndata-0.13 layers[None] case). A check
    # written here that only looked at an in-memory AnnData would miss a path source
    # carrying layers/raw and silently write X only.
    from cellstream.write import _source_is_multimatrix

    # modalities= is a PUBLIC multi-matrix source shape too, and it is what the public
    # writer feeds the detector as its second argument. Without it the guard is bypassed
    # -- the call still fails, but on write_v2_archive's "unexpected keyword argument"
    # rather than the deliberate diagnostic below.
    if _source_is_multimatrix(adata, kw.get("modalities")):
        raise TypeError(
            "write_directory writes a single-matrix v2 archive. write_sharded routes "
            "MuData / .layers / .raw inputs to the multi-matrix packed writer; a "
            "directory fixture cannot stand in for that surface."
        )
    # Source resolution, mirroring write_sharded: a grouped write hands the path to the
    # writer to read backed; a non-grouped write loads it first.
    if not isinstance(adata, ad.AnnData) and kw.get("group_by") is None:
        adata = ad.read_h5ad(adata)
    write_v2_archive(
        adata,
        out_dir,
        codec=(PUBLIC_V2_CODEC if codec is None else codec),
        n_workers=n_workers,
        target_shard_bytes=(
            DEFAULT_TARGET_SHARD_BYTES if target_shard_bytes is None else target_shard_bytes
        ),
        **kw,
    )
    return Path(out_dir)


def pack_directory(src_dir, dst_file, *, overwrite: bool = True) -> Path:
    """Assemble a directory archive into a packed file.

    The "tamper-then-pack" seam: build a directory, tamper a member on disk, pack it,
    and read the result through the PUBLIC reader -- so a tamper test does not have to
    drop to the internal reader.
    """
    from cellstream.packed.writer import _pack_directory

    _pack_directory(src_dir, dst_file, overwrite=overwrite)
    return Path(dst_file)


def _is_packed(p: Path) -> bool:
    return p.is_file() and is_packed_file(p)


def archive_manifest(path) -> dict:
    """The manifest dict, for either container."""
    p = Path(path)
    if _is_packed(p):
        return PackedArchive(p).manifest
    return load_manifest(p / "manifest.json")


def archive_groups(path) -> list[dict]:
    """The groups.json records, for either container.

    The list-type check mirrors what ``load_groups`` and ``ShardedArchive._load_groups``
    both do -- without it the packed arm would silently accept a malformed member that
    the directory arm rejects.
    """
    p = Path(path)
    if _is_packed(p):
        data = json.loads(PackedArchive(p).member_bytes("groups.json").decode("utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"groups.json must contain a JSON array, got {type(data).__name__!r}.")
        return data
    return load_groups(p / "groups.json")


def archive_member_bytes(path, name: str) -> bytes:
    """Raw bytes of a named member, for either container."""
    p = Path(path)
    if _is_packed(p):
        return PackedArchive(p).member_bytes(name)
    return (p / name).read_bytes()


def archive_has_member(path, name: str) -> bool:
    """Whether a named member is present, for either container.

    Replaces ``(out / "header.h5ad").exists()`` -- in a packed archive there is no such
    path, so an ``exists()`` check silently becomes False and the assertion stops
    testing anything.
    """
    p = Path(path)
    if _is_packed(p):
        return PackedArchive(p).has_member(name)
    return (p / name).exists()


def archive_obs(path):
    """The obs DataFrame, for either container (replaces reading obs.parquet directly).

    Both arms prefer ``obs.parquet`` and fall back to ``header.h5ad`` -- which is what
    ``PackedArchive.obs()`` and the internal directory parity accessor both do. Because of that
    fallback a successful call does NOT prove ``obs.parquet`` exists; assert that with
    ``archive_has_member`` instead.
    """
    p = Path(path)
    if _is_packed(p):
        return PackedArchive(p).obs()
    from cellstream import h5ad_io

    return h5ad_io.read_obs(p)


def archive_var(path):
    """The var DataFrame, for either container.

    No parquet/header split here: ``PackedArchive.var()`` reads the header
    unconditionally (there is no ``var.parquet`` sidecar for the X family), so the two
    arms are equivalent without the fallback ``archive_obs`` needs.
    """
    p = Path(path)
    if _is_packed(p):
        return PackedArchive(p).var()
    from cellstream import h5ad_io

    return h5ad_io.read_var(p)


def _reject_multimatrix(manifest: dict, path: Path, who: str) -> None:
    """Fail loud, and IDENTICALLY per container, on a v4 (multi-matrix) archive.

    Without this the two helpers are asymmetric, which codex flagged. The packed arm already fails
    clearly: ``PackedArchive.shard_location`` requires ``matrix=<key>`` for v4 and says so. The
    directory arm did not.

    ⚠️ codex's proposed mechanism did not reproduce, and the corrected one is worth keeping straight.
    It predicted ``shard_filename`` would fall back to the v0.1 ``shard_{:04d}.h5ad`` convention and
    raise ``FileNotFoundError``. Measured on a real v4 archive: the manifest *does* carry a top-level
    ``shard_filename_template`` -- the PRIMARY family's, ``'x/shard_{:04d}.shx'`` -- so the filename
    step neither falls back nor errors, and the directory arm **can silently treat a v4 archive as
    primary-X-only**. It is not guaranteed to get that far cleanly either: a v4 manifest carries no
    top-level ``x_data_dtype_on_disk`` / ``x_indices_dtype_on_disk`` / ``nnz_per_shard`` (those are
    per-family), so ``resolve_plan`` would be working from defaults and could mis-materialize or fail
    downstream instead. Either way the answer is to refuse here rather than to guess.

    ``read_v2_to_host`` is single-matrix by construction -- the multi-matrix reader goes through
    ``ShardedArchive._matrix_shard_locator`` -- so there is nothing for these helpers to migrate a
    v4 caller TO. Rejecting is the honest answer, not a limitation to paper over.

    Takes the manifest rather than loading it: both callers already have one, and building a
    ``PackedArchive`` re-parses the trailer, re-CRCs the footer and re-parses the manifest every
    time. (Gemini; codex made the same observation about this redundant parse in its round 3.)
    """
    if manifest.get("schema_version") == 4:
        raise ValueError(
            f"{who} is single-matrix only; {path} is a multi-matrix (v4) archive. read_v2_to_host "
            "has no multi-matrix surface -- read a family through ShardedArchive(path).to_anndata() "
            "(the primary), .layer(name).to_anndata(), .modality(name).to_anndata(), or a view's "
            "matrix= argument. (ShardedArchive(path)[key] selects ROWS, not a matrix family.)"
        )


def read_v2_host(path, **kw):
    """``read_v2_to_host`` over EITHER container -- so a fixture can go packed without
    being split into a directory twin.

    The internal reader takes a directory positionally, or ``manifest=`` **and**
    ``shard_locator=`` for any other source; that pair is exactly what
    ``ShardedArchive._read_v2_source_kwargs`` hands it for a packed archive,
    so a migrated test drives the same locator production does rather than a
    test-only shim. Supplying only one of the pair raises a dedicated ``ValueError``
    (``v2/reader_cpu.py:266``), so a half-done migration fails loud.

    Returns whatever ``read_v2_to_host`` returns for the given kwargs: ``(data,
    indices, indptr)`` for a CSR read, a 2-D array for a dense plan, and the 4-/2-tuple
    forms under ``_return_shm_bundle=True``.

    Single-matrix only, and rejected the SAME WAY for both containers -- see
    ``_reject_multimatrix``.
    """
    from cellstream.v2.reader_cpu import read_v2_to_host

    p = Path(path)
    if _is_packed(p):
        ar = PackedArchive(p)
        _reject_multimatrix(ar.manifest, p, "read_v2_host")
        # ar is kept alive by the bound method in shard_locator; PackedArchive holds no
        # open fd after __init__ anyway (it opens and closes inside a `with`).
        return read_v2_to_host(None, manifest=ar.manifest, shard_locator=ar.shard_location, **kw)
    _reject_multimatrix(archive_manifest(p), p, "read_v2_host")
    return read_v2_to_host(p, **kw)


def shard_paths_and_bases(path):
    """``(paths, bases)`` for every shard, for EITHER container.

    For the tests that drive a low-level shard consumer -- ``_rust.decode_v2_csr`` and
    friends -- directly, which is why they were on a directory fixture: they need one
    ``(file, offset)`` per shard. A directory gives ``[<one file per shard>]`` with
    ``bases`` all zero; a packed archive gives the SAME file n times with the shard's
    4 KB-aligned offset as its base. Production builds these two lists exactly this way
    (``v2/reader_cpu.py:385-386``), so the packed form is the one that ships.

    ⚠️ ``PackedArchive.shard_location`` returns a **3-tuple** ``(path, offset, length)``; so does
    the directory locator, as ``(path, 0, 0)`` -- 0 length means "map the whole file"
    (``v2/reader_cpu.py:_default_dir_locator``). Only the first two are returned here.
    """
    p = Path(path)
    if _is_packed(p):
        # ONE PackedArchive for the whole call: it parses the trailer, CRCs the footer and parses the
        # manifest in __init__, and this used to do that three times over -- once via
        # _reject_multimatrix, once via archive_manifest, and once per shard inside the
        # comprehension. (Gemini, over two rounds.)
        ar = PackedArchive(p)
        _reject_multimatrix(ar.manifest, p, "shard_paths_and_bases")
        n_shards = int(ar.manifest["n_shards"])
        locs = [ar.shard_location(i) for i in range(n_shards)]
        return [loc[0] for loc in locs], [loc[1] for loc in locs]
    m = archive_manifest(p)
    _reject_multimatrix(m, p, "shard_paths_and_bases")
    n_shards = int(m["n_shards"])
    return [str(p / shard_filename(m, i)) for i in range(n_shards)], [0] * n_shards


def shard_member_bytes(path, idx: int) -> bytes:
    """Raw bytes of shard ``idx``, resolved through the manifest's filename template."""
    return archive_member_bytes(path, shard_filename(archive_manifest(path), idx))
