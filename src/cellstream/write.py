"""Write-side public API: ``write_sharded`` and ``write_h5ad_narrow``.

``write_sharded`` is a router, not a writer: it validates the public parameter surface,
resolves the sentinels (codec and target_shard_bytes), and dispatches to the packed shard,
multi-matrix, or cell writer. The ``_append``/``_update`` mutators this docstring used to
name went with the v1 directory surface in #154, and the schema-v1 writer that lived here
went with it in part B of the same issue.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from cellstream import grouping, shard_io
from cellstream.errors import LEGACY_V1_REMEDY
from cellstream.matrices import real_layer_names

ADataOrPath = ad.AnnData | str | Path


def _source_is_multimatrix(adata: ADataOrPath, modalities=None) -> bool:
    """True if the source has >1 matrix (non-empty .layers, a .raw, extra
    modalities, or a MuData input).

    Handles both in-memory AnnData and a path source (peeked with h5py so the
    check never loads X), so ``write_sharded``'s multi-matrix pre-flight fires
    before any shard is written regardless of input kind.
    """
    if modalities:
        return True
    # MuData input (duck-typed via .mod) is multi-matrix and must route to the
    # mudata adapter; check it BEFORE the h5py path (str(MuData) is not a file).
    if not isinstance(adata, ad.AnnData) and hasattr(adata, "mod"):
        return True
    if isinstance(adata, ad.AnnData):
        return len(real_layer_names(adata)) > 0 or adata.raw is not None
    import h5py

    with h5py.File(str(adata), "r") as f:
        has_layers = "layers" in f and len(f["layers"].keys()) > 0
        has_raw = "raw" in f
    return has_layers or has_raw


def write_h5ad_narrow(
    adata: ad.AnnData,
    path: str | Path,
    *,
    codec: str = "blosc-zstd-1-u16u16-datashuf",
) -> None:
    """Write ``adata`` as a single .h5ad with narrow-dtype X.

    Thin wrapper around shard_io.write_one_h5ad with a stable default codec.
    """
    shard_io.write_one_h5ad(adata, path, codec=codec, where="X")


def write_sharded(
    adata: ADataOrPath,
    out_dir: str | Path,
    *,
    format: str = "v2",
    layout: str = "shard",
    n_shards: int | None = None,
    n_workers: int = 16,
    write_nthreads: int | None = None,
    codec: str | None = None,
    overwrite: bool = False,
    unsafe_no_lock: bool = False,
    group_by: str | None = None,
    reference=None,
    modalities=None,
    primary_modality=None,
    target_shard_bytes: int | None = None,
    sort_within_group: str | list[str] | None = None,
    sort_order: str | None = None,
    dense_grouped_via_csr: bool | None = None,
    payload_checksum: bool | None = None,
    require_unique_indices: bool | None = None,
    data_dtype: str | None = None,
    allow_lossy: bool = False,
    block_size: int | None = None,
    tmp_dir: str | Path | None = None,
) -> None:
    """Write an AnnData (or source h5ad path) as one packed ``.shad`` archive.

    Parameters
    ----------
    adata
        An in-memory AnnData, a str/Path to a source
        `.h5ad` file, or an in-memory ``MuData`` (optional ``mudata`` package;
        routed via ``mudata_adapter.mudata_to_native`` to the multi-matrix
        packed writer — the primary modality becomes ``x``, every other
        modality becomes a ``modality:*`` family, and MuData's global
        obs/obsm/uns become the shared header; see ``ShardedArchive.to_mudata()``
        for the reverse). A path source with ``group_by`` is read lazily
        (backed) and each shard's rows are gathered on demand — the full
        matrix is never resident (#76); other path writes load into RAM.
        For very large target-aware writes, prefer a path source.
    out_dir
        Output path — the ``.shad`` **file** to write (only its parent
        directory is created). Reads auto-detect the archive.
    format
        ``'v2'``, the only accepted value (SHX2 binary format, narrow-dtype
        byte-filter+zstd compression, mmap-friendly). Read on the CPU with
        ``ShardedArchive.to_anndata`` or on the device with
        ``ShardedArchive.to_cupy_anndata`` — on-GPU decode of v2 landed with #40
        (v0.5.5); it is used above ``_gpu_threshold_bytes`` and is CSR-only, so
        ``ShardedArchive.to_cupy_anndata(container="dense")`` on that path is what raises
        NotImplementedError. (That read-side ``container`` selects a materialization
        SHAPE; it is unrelated to the removed write-side archive container, #154 B3b.)
        Anything other than ``'v2'`` raises ``ValueError``: the legacy ``'v1'``
        h5ad-shard writer was removed in #154, having raised
        ``PackedOnlyError`` since v0.5.
    n_shards
        Number of shards. Default ``None`` → byte-driven count
        (``target_shard_bytes`` controls the split).  Explicit integer
        overrides the byte-driven logic.  Cannot be combined with
        ``group_by`` (pass ``target_shard_bytes`` instead).
    n_workers
        Concurrency knob: 16 parallel encode workers by default (fastest for
        in-memory writes in the rw-bench sweep, and the reader default). For
        very large **path-source** writes, a smaller ``n_workers`` can reduce
        memory-bandwidth contention. See HANDOFF.md.
    write_nthreads
        OMP encode-thread count (bitshuffle/zstd). Default None → 4. See #76.
        The v1-only ``blosc_nthreads`` *write* knob was removed with the v1
        writer in #154; the **read**-side ``blosc_nthreads``
        (``ShardedArchive.to_anndata`` / ``read_h5ad``) is a different, live
        knob and is unaffected.
    codec
        Codec preset name. Default ``None`` → 'v2-byte-filter' (data→SHUFFLE,
        indices/indptr→SHUFFLE+BYTEDELTA; ~38% smaller CSR payload,
        byte-exact). The legacy v1 preset names are still accepted and map to
        bitshuffle. Unknown codec names raise ValueError.
    overwrite
        If True, replace an existing archive at ``out_dir``.
    unsafe_no_lock
        Skip write locking. Documented escape hatch for filesystems where
        flock isn't honored.
    group_by
        obs column name to use for group-aware sharding. When set, cells are
        sorted by this column, groups are packed into byte-sized shards
        aligned to group edges, and reference cells (if any) sort first so
        they occupy shard 0. Cannot be combined with ``n_shards``.
    reference
        Identifies reference cells for the dedicated reference shard.
        Either a boolean array of length n_obs (True = reference), or a
        list of label values present in ``obs[group_by]``. Requires
        ``group_by``.
    modalities
        Extra aligned modalities as a ``dict[str, AnnData] | None`` (default
        ``None``), keyed by name, each in the **same cells/obs order** as the
        primary ``adata``. Each becomes its own ``modality:<name>`` family
        with its own var, layers, and obsm/varm/uns, sharing the primary's
        cell partition (Plan 2 multi-matrix). Passing a non-empty
        ``modalities`` (or a ``.raw``/non-empty ``.layers`` on ``adata``)
        routes the write to the multi-matrix (``schema_version=4``) packed
        writer. A ``MuData`` passed as ``adata`` is also accepted — it is
        converted to a primary + ``modalities`` pair via the optional
        ``mudata`` adapter (``mudata_adapter.mudata_to_native``) rather than
        being built by the caller here.
    primary_modality
        The primary modality's name as a ``str | None`` (default ``None``),
        stored in the manifest so ``ShardedArchive.to_mudata()`` can name it
        on read-back. Resolves to ``"rna"`` when ``modalities`` is given and
        this is left ``None``; unused (no modality name is stored at all)
        when ``modalities`` is empty/``None``.
    target_shard_bytes
        Working-set-bytes target per shard. Default 2 GiB.
    sort_within_group
        An ``obs`` column name, or list of column names, to order cells
        **within** each group (e.g. by guide). Default ``None`` (group
        order only). Requires ``group_by``.
        Changes the on-disk byte layout (not byte-identical to an unsorted
        write) and tends to improve compression locality. See #87.
    sort_order
        How the primary ``group_by`` order and every ``sort_within_group``
        key are compared. ``None`` (default) ≡
        ``"natural"``: each column by its native dtype — a categorical by its
        **category order** (e.g. cell-cycle G1<S<G2M), numeric numerically,
        plain strings alphabetically. ``"alphabetical"`` forces every key to a
        string (lexicographic). Requires ``group_by``.
        The ``"natural"`` default re-orders a categorical
        ``group_by`` with a non-alphabetical category order vs. the pre-#87
        alphabetical layout (intentional). See #87.
    dense_grouped_via_csr
        For an **in-memory dense** ``X`` with ``group_by`` set, choose
        the grouped-write strategy. ``None`` (default) ≡ ``False``: the direct
        encoder — **dense-only RAM** (no intermediate copy). ``True`` opts in to
        converting dense→CSR in parallel (Rust) and encoding via the CSR core:
        on balance only ~1.15× faster (and noisy — the dense grouped write is
        memory-bandwidth-bound, so node contention swamps the difference) while
        peaking at roughly the dense array **plus** the intermediate CSR
        (~+48% peak RAM, ~+30 GB at the measured dataset's scale). Default-off keeps the
        lower-RAM path the default (a post-merge A/B found the speedup did not
        justify the RAM as a default; see the 2026-06-18 flip spec). Requires
        ``group_by``. No-op for CSR/path sources (those never take the dense
        branch).
    block_size
        cell-only (``layout='cell'``). Rows per streaming block when the source is a **path**
        or an open ``CellStore``; an in-memory ``AnnData`` is already resident and uses the
        single-shot path. ``None`` (default) auto-plans **nnz-balanced** blocks under a 2 GiB
        budget — a fixed cell count is a bad default because nnz/cell varies by orders of
        magnitude across datasets. Peak RAM is O(block), not O(nnz) (#219).
    tmp_dir
        cell-only. Where the writer stages members before packing. **Defaults to
        ``out_dir.parent``** — the output filesystem must hold the final archive anyway,
        whereas ``/tmp`` typically cannot hold even one payload. A *grouped* streaming write
        stages ~2× the payload here (the frame reorder); ungrouped stages 1×.
    """
    if layout not in ("shard", "cell"):
        raise ValueError(f"layout must be 'shard' or 'cell'; got {layout!r}.")
    if layout == "cell":
        import warnings as _warnings

        from cellstream.cell.writer import write_cell_archive

        _bad = [
            name
            for name, val in (
                ("n_shards", n_shards),
                ("target_shard_bytes", target_shard_bytes),
                ("write_nthreads", write_nthreads),
                ("modalities", modalities),
                ("primary_modality", primary_modality),
                ("dense_grouped_via_csr", dense_grouped_via_csr),
            )
            if val is not None
        ]
        if format != "v2":
            _bad.append(f"format={format!r}")
        if _bad:
            raise ValueError(
                f"layout='cell' is a flat single-file store and does not accept "
                f"shard-mode parameters: {_bad}. Cell archives are always packed .shad."
            )
        _warnings.warn(
            "layout='cell' is an experimental flat per-cell format (schema_version 5 for "
            "pfordelta, 6 for zstd). The on-disk layout may change in a future release.",
            UserWarning,
            stacklevel=2,
        )
        write_cell_archive(
            adata,
            out_dir,
            group_by=group_by,
            reference=reference,
            sort_within_group=sort_within_group,
            sort_order=sort_order,
            codec=codec,  # None -> auto (pfordelta int / zstd float); explicit passes through
            data_dtype=data_dtype,
            allow_lossy=allow_lossy,
            overwrite=overwrite,
            payload_checksum=(False if payload_checksum is None else payload_checksum),
            require_unique_indices=(
                True if require_unique_indices is None else require_unique_indices
            ),
            # write_sharded has always accepted unsafe_no_lock and dropped it here. Now that
            # the cell path locks, dropping it would silently ignore a load-bearing knob --
            # the shape #261 fixed. (#280 item 5)
            unsafe_no_lock=unsafe_no_lock,
            block_size=block_size,
            tmp_dir=tmp_dir,
        )
        return

    # payload_checksum / require_unique_indices are cell-only (layout='cell'); reject them for a
    # shard write, where they would otherwise be silently ignored (mirror the cell-branch _bad gate).
    _cell_only = [
        name
        for name, val in (
            ("payload_checksum", payload_checksum),
            ("require_unique_indices", require_unique_indices),
            ("data_dtype", data_dtype),
            # allow_lossy is a plain bool (default False); only flag it when truthy so the
            # common shard path is unaffected.
            ("allow_lossy", allow_lossy if allow_lossy else None),
            ("block_size", block_size),
            ("tmp_dir", tmp_dir),
        )
        if val is not None
    ]
    if _cell_only:
        raise ValueError(
            f"{_cell_only} are cell-only parameters (layout='cell') and are not valid "
            f"for layout='shard'."
        )

    # v2 is the only shard format (#154): the legacy v1 h5ad-shard writer is gone, and it
    # had raised PackedOnlyError since v0.5 anyway. Checked BEFORE codec resolution so a
    # bad format reports itself rather than an incidental error further down.
    if format != "v2":
        raise ValueError(
            f"format must be 'v2', got {format!r}. The legacy v1 (h5ad-shard) format was "
            f"removed in #154 and has been unsupported since v0.5; {LEGACY_V1_REMEDY}"
        )

    if codec is None:
        codec = "v2-byte-filter"  # the v0.3 default

    # --- Cross-parameter validation ---
    if reference is not None and group_by is None:
        raise ValueError(
            "reference requires group_by. Pass group_by=<obs column> together with reference."
        )
    if n_shards is not None and group_by is not None:
        raise ValueError(
            "n_shards and group_by cannot be used together. With group_by, shard "
            "count is governed by target_shard_bytes (default 2 GiB; a smaller budget "
            "yields more shards, roughly halving the budget doubles the count) and is "
            "floored by the number of groups (one shard per group minimum; tiny groups "
            "merge, large groups split), so an exact n_shards cannot be honored. Pass "
            "target_shard_bytes to control shard size instead."
        )
    # The two blosc_nthreads/write_nthreads cross-checks that stood here went with the v1
    # writer (#154): blosc_nthreads is no longer a parameter of this function at all (it was
    # v1-only, and the v2 path rejected rather than forwarded it), so passing it is now a
    # TypeError; and write_nthreads has no format left to be wrong for.
    if write_nthreads is not None and write_nthreads <= 0:
        raise ValueError(f"write_nthreads must be a positive integer, got {write_nthreads}.")
    # IN PLACE, not hoisted above the layout='cell' early return: moving this up would put it
    # ahead of the cell _bad gate, the format guard and the reference/n_shards
    # checks, silently changing which error a doubly-invalid call reports. The cell layout gets
    # the same rules from cell/writer.py::_resolve_storage_order instead, which also covers the
    # three direct cellstream.cell writers this function never sees (#261).
    grouping.validate_sort_params(
        group_by=group_by,
        sort_within_group=sort_within_group,
        sort_order=sort_order,
    )
    # None (default) and False both mean the direct path; only True opts into
    # via-CSR. Resolve the sentinel FIRST so an explicit False (== the documented
    # default) is a no-op and never trips the group_by guard below — only an
    # actual via-CSR opt-in (True) is constrained to grouped writes.
    _dense_via_csr = bool(dense_grouped_via_csr)
    if _dense_via_csr and group_by is None:
        raise ValueError(
            "dense_grouped_via_csr=True requires group_by (it only affects grouped "
            "dense writes). Pass group_by=<obs column>, or drop dense_grouped_via_csr."
        )
    # The v1-only "target_shard_bytes without group_by" rejection stood here (v1 sized by
    # n_shards); every write is byte-driven now, so there is nothing to reject.

    # obsp/varp are now stored (Bucket 3), so there is no top-level unsupported-field
    # pre-flight; the only remaining fail-loud at this stage is a modality's own .raw
    # (checked per-modality inside the multi-matrix writer / mudata_to_native).

    # Multi-matrix (v4) pre-flight: a source with >1 matrix (non-empty layers
    # and/or a .raw) is stored via the packed multi-matrix writer. Single-matrix
    # writes never take this branch and fall through unchanged below.
    if _source_is_multimatrix(adata, modalities):
        primary_meta = None
        if not isinstance(adata, ad.AnnData) and hasattr(adata, "mod"):
            from cellstream.mudata_adapter import mudata_to_native

            adata, modalities, primary_meta, primary_modality = mudata_to_native(
                adata, primary_modality=primary_modality
            )
        if not isinstance(adata, ad.AnnData):
            adata = ad.read_h5ad(adata)  # multi path is in-memory
        from cellstream.packed.writer import write_packed_multi

        write_packed_multi(
            adata,
            out_dir,
            overwrite=overwrite,
            unsafe_no_lock=unsafe_no_lock,
            n_shards=n_shards,
            n_workers=n_workers,
            group_by=group_by,
            reference=reference,
            modalities=modalities,
            primary_modality=(primary_modality or "rna") if modalities else None,
            primary_meta=primary_meta,
            target_shard_bytes=(2 * 2**30 if target_shard_bytes is None else target_shard_bytes),
            codec=codec,
            write_nthreads=write_nthreads,
            sort_within_group=sort_within_group,
            sort_order=sort_order,
        )
        return

    from cellstream.v2.writer import _resolve_v2_shuffle

    # Fail loud on an unknown v2 codec BEFORE any (potentially huge) source
    # load. The packed staging writer re-resolves as the single source of truth.
    _resolve_v2_shuffle(codec)

    if group_by is not None and not isinstance(adata, ad.AnnData):
        source = adata  # path; grouped writer reads it backed
    elif not isinstance(adata, ad.AnnData):
        source = ad.read_h5ad(adata)
    else:
        source = adata
    from cellstream.packed.writer import write_packed

    write_packed(
        source,
        out_dir,
        overwrite=overwrite,
        unsafe_no_lock=unsafe_no_lock,
        n_shards=n_shards,
        n_workers=n_workers,
        group_by=group_by,
        reference=reference,
        target_shard_bytes=(2 * 2**30 if target_shard_bytes is None else target_shard_bytes),
        codec=codec,
        write_nthreads=write_nthreads,
        sort_within_group=sort_within_group,
        sort_order=sort_order,
        dense_grouped_via_csr=_dense_via_csr,
    )
