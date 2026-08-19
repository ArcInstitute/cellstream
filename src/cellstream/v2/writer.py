"""Native v2 writer: stages a shard DIRECTORY, for the packed writer to pack.

INTERNAL as of #154 part B3b: no longer called DIRECTLY by write_sharded, only
reached through it. For layout="shard", write_sharded routes to
packed.writer.write_packed / write_packed_multi, which call this to stage a shard
directory and then _pack_directory to fold it into one .shad file. (layout="cell"
has its own writer and never comes here.) The public directory container is gone --
this is the staging step that outlived it.

Reuses v0.1's lock / manifest / header machinery via cellstream.lock,
cellstream.manifest, cellstream.header. The per-shard encode step is the only
new code — it streams the shard's data/indices/indptr through
v2.codec.bitshuffle_zstd_encode in chunk-sized blocks and emits an .shx
file via v2.format.write_shx_file_streaming.

Its worker pools are **spawn** pools, built here with mp.get_context("spawn"); from
cellstream.workers it takes only omp_nthreads_in_env (bitshuffle's OMP_NUM_THREADS window,
whose lock is shared with the v2 reader) and _maybe_test_kill_self. The v0.1 *fork*-pool
machinery this docstring used to claim it reused went with the v1 writer in #154.

The ENCODER is incremental: ``_encode_shard_streams`` works in
``O(CHUNK_SIZE_ELEMENTS × itemsize)`` and retains no chunk history. That is a scaling
statement, not an exact figure — each chunk also costs codec-dependent temporaries (for
bitshuffle+zstd: the shuffled array, its ``tobytes()`` copy, and zstd's own workspace) plus
the compressed output. And it bounds only the encode step: the gather arms materialize a whole
shard first, so it says nothing about total per-worker memory.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np

from cellstream import __version__ as _cellstream_version
from cellstream import grouping, narrow
from cellstream import header as v1_header
from cellstream import lock as _lock
from cellstream import manifest as _manifest
from cellstream import shm as _shm
from cellstream import workers as _workers
from cellstream.errors import ArchiveExistsError, LossyCastError
from cellstream.v2 import codec
from cellstream.v2.codec import BYTE_FILTER_SHUFFLE_NAME
from cellstream.v2.format import ShxHeader, StreamSpec, write_shx_file_streaming
from cellstream.v2.row_gather import gather_csr_rows, read_backed_csr_rows

logger = logging.getLogger(__name__)

CHUNK_SIZE_ELEMENTS = 1_048_576
CODEC_NAME = "zstd-1-u16u16-bitshuffle"
SCHEMA_VERSION = 2

#: Manifest `codec` value when the byte-filter codec is used (informational —
#: the reader dispatches on the per-shard ShxHeader.shuffle marker, not this).
#: Aligned with the public write_sharded(codec=) name; the reader never
#: dispatches on this field (it reads ShxHeader.shuffle per shard).
BYTE_FILTER_CODEC_NAME = "v2-byte-filter"

#: User-facing codec name for the byte-filter path (the public write_sharded default
#: as of v0.3.0; pass this to write_sharded(codec=...) to select it explicitly).
V2_BYTE_FILTER_CODEC = "v2-byte-filter"

#: Codec names that select the bitshuffle path. Must include EVERY name
#: already documented/written as a v2 (or pre-v0.3.0 default) codec so no existing
#: caller regresses to ValueError: CODEC_NAME ("zstd-1-u16u16-bitshuffle") is the
#: v0.2-documented v2 codec name and the manifest `codec` value; the v1 preset
#: names were write_sharded's pre-v0.3 default (and the v2 path historically ignored
#: codec); "bitshuffle" is the plain self-descriptive name.
_V2_BITSHUFFLE_CODECS = frozenset(
    {CODEC_NAME, "bitshuffle", "blosc-zstd-1-u16u16", "blosc-zstd-1-u16u16-datashuf"}
)


#: Bounded OpenMP thread count for v2 spawn encode workers. Internal default
#: (not a public param — a public write-time thread knob is issue #2). Matches
#: the read path's omp_nthreads default and v1's blosc_nthreads default; keeps
#: n_workers × OMP threads from oversubscribing (the read side measured ~30×
#: collapse from unbounded OMP under spawn).
_V2_WRITE_OMP_NTHREADS = 4


@dataclass
class _GroupedMeta:
    """Metadata needed to plan + write a grouped archive WITHOUT loading X.

    indptr is the source CSR row-pointer array (int64), enough for per-row nnz.
    obs/var are DataFrames; uns/obsm/varm are dicts. n_obs/n_vars are ints.
    """

    obs: object
    var: object
    uns: dict
    obsm: dict
    varm: dict
    obsp: dict
    varp: dict
    indptr: np.ndarray
    n_obs: int
    n_vars: int


def _dense_per_row_nnz(D, block: int = 4096) -> np.ndarray:
    """Count nonzeros per row of a dense 2-D array, block-wise (no full bool mask)."""
    n = D.shape[0]
    out = np.empty(n, dtype=np.int64)
    for i in range(0, n, block):
        out[i : i + block] = np.count_nonzero(D[i : i + block], axis=1)
    return out


def _bytes_per_nnz(ddt, idt) -> int:
    """Working-set bytes/nnz for shard sizing (#120).

    Keep the historical 4 for the u16/u16 layout so existing shard counts (and the
    golden byte-identity tests) are unchanged; otherwise use the real on-disk widths.
    """
    ddt, idt = np.dtype(ddt), np.dtype(idt)
    if ddt == np.dtype(np.uint16) and idt == np.dtype(np.uint16):
        return 4
    return ddt.itemsize + idt.itemsize


def _sizing_bytes_per_nnz(min_data, min_index) -> int:
    """Shard-sizing proxy from the MIN dtypes. Integer data compresses ~the same at any
    width (uint8/uint16/uint32 are byte-planed to the same payload), so size it as the
    historical uint16 proxy regardless of the true min uint width -- keeping default
    shard counts unchanged AND making in-mem and backed encoders size identically (so
    their shard bytes stay byte-identical). Float data uses its native width."""
    d = np.dtype(np.uint16) if np.dtype(min_data).kind == "u" else np.dtype(min_data)
    return _bytes_per_nnz(d, min_index)


def _min_index_dtype_for_n_vars(n_vars):
    """Minimum index dtype from n_vars (max possible column = n_vars-1): uint16 if
    n_vars <= 65536 else uint32. The on-disk index dtype is always uint32; this is the
    scan-free min recorded so the reader can cast indices down losslessly."""
    return np.dtype(np.uint16) if (n_vars - 1) <= 65535 else np.dtype(np.uint32)


def _decide_dense_on_disk_dtypes(D, n_vars):
    """Predict (on_disk_data, on_disk_index, min_data, min_index) for a DENSE input,
    without building CSR. Integer-routable data -> on-disk uint32 (min = narrowest
    uint); otherwise native float. Index on-disk uint32; min from n_vars. Sizing /
    manifest only: the Rust dense encoder decides its own (matching) on-disk dtype.

    Defers to the element-blocked, RAM-bounded ``narrow._data_min_uint_dtype``, which
    flattens N-D input with ``order="K"`` (a no-copy view for C-/F-contiguous arrays).
    """
    D = np.asarray(D)
    min_uint = narrow._data_min_uint_dtype(D)
    if min_uint is not None:
        on_disk_data, min_data = np.dtype(np.uint32), min_uint
    else:
        on_disk_data = narrow._native_on_disk_data_dtype(D.dtype)
        min_data = on_disk_data
    return on_disk_data, np.dtype(np.uint32), min_data, _min_index_dtype_for_n_vars(n_vars)


def _decide_backed_on_disk_data_dtype(src_path, block=1 << 20):
    """Decide the uniform on-disk data dtype for a BACKED grouped write.

    Streams the source's 1-D ``X/data`` once (values only -- no indices, no indptr, no
    CSR materialization), mirroring ``narrow.decide_on_disk_dtypes``: integer-routable
    values -> ``(uint32, narrowest-uint)``; otherwise ``(native-float, native-float)``.
    Early-exits on the first non-integer-routable block, so fractional data costs ~one
    block. Reading the full data column is only paid when the data IS integer-routable --
    the same scan the in-memory path already does. Returns ``(on_disk_data, min_data)``.
    """
    import h5py

    with h5py.File(str(src_path), "r") as f:
        ddset = f["X/data"]
        src_dtype = ddset.dtype
        n = int(ddset.shape[0])

        def _blocks():
            for i in range(0, n, block):
                yield ddset[i : i + block]

        min_uint = narrow.data_min_uint_dtype_streaming(_blocks())
    if min_uint is not None:
        return np.dtype(np.uint32), min_uint
    native = narrow._native_on_disk_data_dtype(src_dtype)
    return native, native


def _resolve_grouped_metadata(source) -> _GroupedMeta:
    """From an in-mem AnnData or a path, return _GroupedMeta with NO X load.

    In-mem: read CSR indptr directly, or count dense row nnz block-wise. Path:
    read obs/var/uns/obsm/varm via read_elem + X/indptr via h5py — X values
    are never read.
    """
    import scipy.sparse as _sp

    if isinstance(source, ad.AnnData):
        X = source.X
        if _sp.issparse(X):
            if X.format != "csr":
                X = _sp.csr_matrix(X)
            indptr = np.asarray(X.indptr, dtype=np.int64)
        else:
            nnz_per_row = _dense_per_row_nnz(np.asarray(X))
            indptr = np.concatenate(([0], np.cumsum(nnz_per_row))).astype(np.int64)
        return _GroupedMeta(
            obs=source.obs,
            var=source.var,
            uns=dict(source.uns),
            obsm=dict(source.obsm),
            varm=dict(source.varm),
            obsp=dict(source.obsp),
            varp=dict(source.varp),
            indptr=indptr,
            n_obs=int(source.n_obs),
            n_vars=int(source.n_vars),
        )
    import h5py
    from anndata.io import read_elem

    with h5py.File(str(source), "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        uns = read_elem(f["uns"]) if "uns" in f else {}
        obsm = read_elem(f["obsm"]) if "obsm" in f else {}
        varm = read_elem(f["varm"]) if "varm" in f else {}
        # obsp/varp (pairwise graphs) — metadata-sized; X values still never read.
        obsp = read_elem(f["obsp"]) if "obsp" in f else {}
        varp = read_elem(f["varp"]) if "varp" in f else {}
        # Fail loud if X isn't CSR: a CSC source's indptr is length n_vars+1,
        # which would otherwise mis-plan downstream with a confusing error.
        enc = f["X"].attrs.get("encoding-type")
        if enc != "csr_matrix":
            raise ValueError(
                f"v2 grouped write requires a CSR-encoded source X; got "
                f"encoding-type={enc!r}. Re-save the source with CSR X "
                f"(adata.X = scipy.sparse.csr_matrix(adata.X)) before writing."
            )
        indptr = f["X/indptr"][:].astype(np.int64)
        n_vars = int(f["X"].attrs["shape"][1])
    return _GroupedMeta(
        obs=obs,
        var=var,
        uns=uns,
        obsm=obsm,
        varm=varm,
        obsp=obsp,
        varp=varp,
        indptr=indptr,
        n_obs=len(obs),
        n_vars=n_vars,
    )


def _write_permuted_metadata(out_dir, meta: _GroupedMeta, perm: np.ndarray) -> None:
    """Write header.h5ad + obs.parquet in permuted order, with no X copy.

    obs/obsm permuted by `perm`; var/varm/uns unchanged. Adds the _orig_row
    column (= perm). Guards the reserved column.
    """
    import scipy.sparse as _sp

    if "_orig_row" in meta.obs.columns:
        warnings.warn(
            "Replacing the existing reserved obs column '_orig_row' with group-aware "
            "row bookkeeping. This is expected when re-sharding an archive read back "
            "via to_anndata(); rename the column first if it held unrelated data.",
            UserWarning,
            # user -> write_sharded -> write_v2_archive -> _write_v2_grouped ->
            # _write_permuted_metadata -> warn: 5 frames to the public call site
            # (matches the stacklevel=5 convention on the existing warn at line ~658).
            stacklevel=5,
        )
    obs_perm = meta.obs.iloc[perm].copy()
    # Overwrite any stale incoming _orig_row with the new permutation.
    obs_perm["_orig_row"] = perm.astype("int64")
    # Permute each obsm value by type: preserve DataFrames (columns/index) and
    # sparse matrices (don't densify) — np.asarray would force-densify sparse and
    # drop DataFrame metadata. Matches what the old adata[perm].copy() path did.
    obsm_perm = {}
    for k, v in meta.obsm.items():
        if hasattr(v, "iloc"):
            obsm_perm[k] = v.iloc[perm]
        elif isinstance(v, np.ndarray) or _sp.issparse(v):
            obsm_perm[k] = v[perm]
        else:
            obsm_perm[k] = np.asarray(v)[perm]
    # obsp (cell×cell) is permuted on BOTH axes; varp (feature axis) is unchanged.
    obsp_perm = {k: v[perm][:, perm] for k, v in meta.obsp.items()}
    h = ad.AnnData(
        X=_sp.csr_matrix((meta.n_obs, meta.n_vars), dtype="float32"),
        obs=obs_perm,
        var=meta.var,
        uns=meta.uns,
        obsm=obsm_perm,
        varm=meta.varm,
    )
    # Attach obsp/varp only when non-empty (no empty groups on any anndata version).
    for k, v in obsp_perm.items():
        h.obsp[k] = v
    for k, v in meta.varp.items():
        h.varp[k] = v
    v1_header.write_header(h, out_dir / "header.h5ad")
    v1_header.write_obs_parquet(obs_perm, out_dir / "obs.parquet")


def _can_use_shm(*, total_nnz: int, n_indptr: int, data_dtype, indices_dtype) -> bool:
    """True iff the shared-memory source handoff is usable: Linux AND /dev/shm
    has room for the on-disk source bundle (data_dtype data + indices_dtype
    indices + int64 indptr). Otherwise the caller falls back to the serial path.

    Any filesystem error while probing /dev/shm (missing, inaccessible, etc.)
    is treated as 'cannot use shm' (returns False), so the documented serial
    fallback holds instead of crashing the write.
    """
    import platform

    from cellstream import shm as _shm

    if platform.system() != "Linux":
        return False
    try:
        required = _shm._required_bytes(
            total_nnz=total_nnz,
            n_indptr=n_indptr,
            data_dtype=np.dtype(data_dtype),
            indices_dtype=np.dtype(indices_dtype),
        )
        return required <= _shm._free_bytes(_shm.SHM_DIR)
    except OSError:
        return False


def _resolve_v2_shuffle(codec_name: str) -> str:
    """Map a user-facing v2 codec name to the on-disk ShxHeader.shuffle marker.

    ``"v2-byte-filter"`` -> the byte-filter marker; bitshuffle preset names ->
    ``"bitshuffle"``. Unknown names raise ValueError (fail-loud).
    """
    if codec_name == V2_BYTE_FILTER_CODEC:
        return BYTE_FILTER_SHUFFLE_NAME
    if codec_name in _V2_BITSHUFFLE_CODECS:
        return "bitshuffle"
    raise ValueError(
        f"Unknown v2 codec {codec_name!r}. Valid: {V2_BYTE_FILTER_CODEC!r} (per-array "
        f"byte-filter) or one of {sorted(_V2_BITSHUFFLE_CODECS)} (bitshuffle)."
    )


def write_v2_archive(
    adata,
    out_dir,
    *,
    n_shards=None,
    n_workers=8,
    overwrite=False,
    unsafe_no_lock=False,
    group_by=None,
    reference=None,
    target_shard_bytes=2 * 2**30,
    codec=CODEC_NAME,
    write_nthreads=None,
    sort_within_group=None,
    sort_order=None,
    dense_grouped_via_csr=False,
):
    """Native v2 writer; same call shape as v0.1's write_sharded.

    Parameters
    ----------
    n_shards
        Number of shards. ``None`` (default) → byte-driven count.
        Cannot be combined with ``group_by``.
    codec
        v2 codec name. This internal function's own default is ``CODEC_NAME``
        ("zstd-1-u16u16-bitshuffle"), which resolves to the bitshuffle path.
        The public ``write_sharded`` API supplies ``"v2-byte-filter"`` by
        default (as of v0.3.0), so callers going through ``write_sharded``
        get the byte-filter codec unless they pass an explicit ``codec=``.
        Pass ``"v2-byte-filter"`` for the per-array byte-filter codec
        (data→SHUFFLE, indices/indptr→SHUFFLE+BYTEDELTA; ~38% smaller CSR
        payload, byte-exact). The v1 preset names also resolve to bitshuffle;
        unknown names raise ValueError.
    group_by
        obs column name for group-aware sharding.  When set, cells are
        sorted by the column, packed into byte-sized group-aligned shards,
        and reference cells (if any) sort first so they occupy shard 0.
    reference
        Bool array (True = reference cell) or list of label values in
        ``obs[group_by]``.
    target_shard_bytes
        Working-set-bytes target per shard (default 2 GiB). Must be > 0.

    Notes
    -----
    Grouped (``group_by``) writes accept an in-memory AnnData **or** a source
    h5ad path: a path source is read lazily (backed) and each shard's rows are
    gathered on demand, so the full matrix is never resident (#76). Non-grouped
    v2 writes operate on an in-memory AnnData (a path is loaded by
    ``write_sharded``). For very large target-aware writes, prefer a path source.

    Parallel writes (``n_workers > 1`` and more than one shard, on Linux with
    enough ``/dev/shm``) hand the source CSR to spawn workers through a single
    shared-memory bundle (cast to the on-disk dtypes once), instead of pickling
    each shard's arrays — see issue #76. This adds a bounded transient of
    ~``4 * total_nnz`` bytes in ``/dev/shm`` on top of the in-RAM source. If the
    handoff is unavailable (non-Linux or insufficient ``/dev/shm``), the write
    falls back to the serial path with a warning.
    """
    # --- Validation (before acquiring the write lock) ---
    # Resolve codec -> on-disk shuffle marker; fail loud on an unknown name
    # before any disk write.
    shuffle = _resolve_v2_shuffle(codec)
    write_nthreads = _V2_WRITE_OMP_NTHREADS if write_nthreads is None else int(write_nthreads)
    if target_shard_bytes <= 0:
        raise ValueError(f"target_shard_bytes must be > 0, got {target_shard_bytes}.")
    if n_shards is not None and group_by is not None:
        raise ValueError(
            "n_shards and group_by cannot be used together. With group_by, shard "
            "count is governed by target_shard_bytes (default 2 GiB; a smaller budget "
            "yields more shards, roughly halving the budget doubles the count) and is "
            "floored by the number of groups (one shard per group minimum; tiny groups "
            "merge, large groups split), so an exact n_shards cannot be honored. Pass "
            "target_shard_bytes to control shard size instead."
        )
    if reference is not None and group_by is None:
        raise ValueError(
            "reference requires group_by. Pass group_by=<obs column> together with reference."
        )
    if isinstance(adata, ad.AnnData):
        if group_by is not None and group_by not in adata.obs.columns:
            raise ValueError(
                f"group_by={group_by!r} is not a column in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        if reference is not None and group_by is not None:
            grouping.validate_reference_labels(adata, group_by, reference)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with _lock.acquire_write_lock(
        out_dir / ".cellstream.lock", op="write", unsafe_no_lock=unsafe_no_lock
    ):
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists():
            if not overwrite:
                raise ArchiveExistsError(
                    f"manifest.json already exists at {out_dir}. Pass overwrite=True to replace."
                )
            for name in ("manifest.json", "header.h5ad", "obs.parquet", "groups.json"):
                p = out_dir / name
                if p.exists():
                    p.unlink()
            for p in out_dir.glob("shard_*.shx*"):
                p.unlink()
        if group_by is not None:
            _write_v2_grouped(
                adata,
                out_dir,
                n_workers=n_workers,
                reference=reference,
                group_by=group_by,
                target_shard_bytes=target_shard_bytes,
                shuffle=shuffle,
                write_nthreads=write_nthreads,
                sort_within_group=sort_within_group,
                sort_order=sort_order,
                dense_grouped_via_csr=dense_grouped_via_csr,
            )
        else:
            _write_v2_inmem(
                adata,
                out_dir,
                n_shards=n_shards,
                n_workers=n_workers,
                target_shard_bytes=target_shard_bytes,
                shuffle=shuffle,
                write_nthreads=write_nthreads,
            )


def write_v2_multi(
    adata,
    out_dir,
    *,
    modalities=None,
    primary_meta=None,
    primary_modality=None,
    n_shards=None,
    n_workers=8,
    overwrite=False,
    unsafe_no_lock=False,
    group_by=None,
    reference=None,
    target_shard_bytes=2 * 2**30,
    codec=CODEC_NAME,
    write_nthreads=None,
    sort_within_group=None,
    sort_order=None,
):
    """Write a v4 (multi-matrix) directory archive from an in-memory AnnData.

    The cell partition is computed ONCE on the anchor X and reused verbatim for
    every family (x, each layer, raw), guaranteeing the shared-partition
    invariant by construction. Each family's shards are encoded with the SAME
    per-shard row lists as the anchor, so the x-family is byte-identical to a
    single-matrix write of X.

    ``modalities`` (optional) is a ``{name: AnnData}`` map of additional
    modalities (e.g. an ATAC anchor + its layers) sharing the SAME cell
    partition as the primary; each is validated as row-aligned to ``adata``
    (same ``obs_names``, same order) before its own family/var/meta sidecar are
    written. ``primary_meta`` (optional) is the primary's own metadata sidecar
    (e.g. ``obsm["X_umap"]``), written next to the ``x`` family when the
    archive also carries ``modalities`` (mirrors each modality's sidecar for
    the primary). ``primary_modality`` (optional) names the primary modality
    (defaults to ``"rna"``) recorded on the ``x`` family's manifest entry when
    ``modalities`` is non-empty. Every per-modality/``primary_meta`` sidecar's
    ``obsm`` is permuted to the archive's shared cell order (see
    ``header.write_modality_meta``'s ``perm``), so it lines up row-for-row with
    its family's on-disk shards even under ``group_by``.
    """
    import scipy.sparse as _sp

    from cellstream import matrices as _mx

    shuffle = _resolve_v2_shuffle(codec)
    write_nthreads = _V2_WRITE_OMP_NTHREADS if write_nthreads is None else int(write_nthreads)
    if not isinstance(adata, ad.AnnData):
        raise TypeError("write_v2_multi requires an in-memory AnnData (multi-matrix path).")
    for _mname, _madata in (modalities or {}).items():
        v1_header._reject_unsupported_modality_fields(_madata)
    specs = _mx.decompose(adata, modalities=modalities)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with _lock.acquire_write_lock(
        out_dir / ".cellstream.lock", op="write", unsafe_no_lock=unsafe_no_lock
    ):
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists() and not overwrite:
            raise ArchiveExistsError(
                f"manifest.json already exists at {out_dir}. Pass overwrite=True to replace."
            )
        if manifest_path.exists():
            import json as _json
            import shutil as _shutil

            # Collect the PRIOR archive's family keys before deleting anything: a
            # prior write may have had families (e.g. a layer) the NEW adata does
            # not, and those family dirs must still be removed or they'd be left
            # orphaned — unreferenced by the freshly-written manifest. Any failure
            # reading/parsing the old manifest (corrupt, pre-v4, etc.) must not
            # block the overwrite, so this degrades to an empty set.
            # Derive the PRIOR family DIRS from the old manifest's stored
            # shard_filename_template (its resolved "<dir>/..." token), NOT by
            # re-deriving matrix_dir(key): a prior key written by a newer writer
            # (e.g. a Plan-2 "modality:*") may not be a key matrix_dir() recognises,
            # and matrix_dir() would raise ValueError and crash the overwrite. The
            # manifest already stores the resolved path (design decision: readers
            # never re-derive dirs), so using it cleans any prior family robustly.
            try:
                prior_manifest = _json.loads(manifest_path.read_text())
                prior_fam_dirs = {
                    md["shard_filename_template"].split("/", 1)[0]
                    for md in prior_manifest.get("matrices", {}).values()
                    if "/" in md.get("shard_filename_template", "")
                }
            except Exception:
                prior_fam_dirs = set()

            for name in ("manifest.json", "header.h5ad", "obs.parquet", "groups.json"):
                p = out_dir / name
                if p.exists():
                    p.unlink()
            for fam_dir in {_mx.matrix_dir(s.key) for s in specs} | prior_fam_dirs:
                fam = out_dir / fam_dir
                if fam.exists():
                    _shutil.rmtree(fam)
            if (out_dir / "var").exists():
                _shutil.rmtree(out_dir / "var")
            if (out_dir / "meta").exists():
                _shutil.rmtree(out_dir / "meta")
            # A prior single-matrix v2/v3 directory write places shards at the
            # TOP level (not in a family dir); overwriting such a dir with a v4
            # archive must not leave them orphaned. A v4 write never places
            # shards at the top level, so this only ever removes true leftovers.
            for p in out_dir.glob("shard_*.shx"):
                p.unlink()

        # --- Partition: computed ONCE on the anchor x. ---
        meta = _resolve_grouped_metadata(adata)  # obs/var/uns/obsm/varm + x indptr
        # Shard sizing uses the anchor (primary x) dtypes, like the single-matrix
        # writers (_write_v2_grouped / _write_v2_inmem) — NOT a hardcoded 4, so a
        # float or dense anchor sizes correctly.
        _anchor_X = adata.X
        if not _sp.issparse(_anchor_X) or _anchor_X.format != "csr":
            _anchor_X = _sp.csr_matrix(_anchor_X)
        _, _, _min_d, _min_i = narrow.decide_on_disk_dtypes(
            _anchor_X.data, _anchor_X.indices, int(adata.n_vars)
        )
        _bpn = _sizing_bytes_per_nnz(_min_d, _min_i)
        if group_by is not None:
            obs_adata = ad.AnnData(obs=meta.obs)
            ref_mask = grouping.resolve_reference(reference, obs_adata, group_by)
            mode = "natural" if sort_order is None else sort_order

            def _key(col, *, group_label=False):
                # group_label=True ONLY for the group_by column: its key must use the same
                # missing-value normalization resolve_reference uses, or the two disagree and
                # a reference on a label that collides with a missing value raises
                # "reference_mask splits label" (#294). A sort_within_group key names no
                # group, so it stays .astype(str).
                if mode != "alphabetical":
                    return col.values
                return (
                    grouping.stringify_group_labels(col) if group_label else col.astype(str).values
                )

            secondary_keys = None
            if sort_within_group is not None:
                cols = (
                    [sort_within_group]
                    if isinstance(sort_within_group, str)
                    else list(sort_within_group)
                )
                secondary_keys = [_key(meta.obs[c]) for c in cols]
            per_row_nnz = np.diff(meta.indptr).astype(np.int64)
            plan = grouping.plan_group_shards(
                group_labels=_key(meta.obs[group_by], group_label=True),
                per_row_nnz=per_row_nnz,
                reference_mask=ref_mask,
                target_shard_bytes=target_shard_bytes,
                bytes_per_nnz=_bpn,
                secondary_key=secondary_keys,
            )
            perm = plan.permutation
            row_offsets = plan.row_offsets
            reference_shard = plan.reference_shard
            groups = plan.groups
        else:
            n_obs = meta.n_obs
            ns = n_shards
            if ns is None:
                total_nnz = int(meta.indptr[-1])
                ns = max(1, math.ceil(total_nnz * _bpn / target_shard_bytes))
            ns = min(ns, max(n_obs, 1))
            perm = np.arange(n_obs, dtype=np.int64)
            row_offsets = [int(round(i * n_obs / ns)) for i in range(ns + 1)]
            reference_shard = None
            groups = None

        n_sh = len(row_offsets) - 1
        oi_lists = [perm[row_offsets[s] : row_offsets[s + 1]] for s in range(n_sh)]

        # --- Shared metadata (once). Permuted by the anchor perm. ---
        _write_permuted_metadata(out_dir, meta, perm)

        # --- Per-family encode + var sidecars. ---
        matrices_map: dict = {}
        for spec in specs:
            fam_dir = out_dir / _mx.matrix_dir(spec.key)
            fam_dir.mkdir(parents=True, exist_ok=True)
            fam_X = spec.X
            if not _sp.issparse(fam_X) or fam_X.format != "csr":
                fam_X = _sp.csr_matrix(fam_X)
            fam_adata = ad.AnnData(X=fam_X)
            nnz_per_shard, ddt, idt = _grouped_encode_inmem(
                fam_adata, fam_dir, oi_lists, spec.n_vars, shuffle, n_workers, write_nthreads
            )
            _, _, min_d, min_i = narrow.decide_on_disk_dtypes(
                fam_X.data, fam_X.indices, spec.n_vars
            )
            template = f"{_mx.matrix_dir(spec.key)}/shard_{{:04d}}.shx"
            entry = {
                "role": spec.role,
                "owner": spec.owner,
                "n_vars": int(spec.n_vars),
                "nnz_per_shard": nnz_per_shard,
                "total_nnz": int(sum(nnz_per_shard)),
                "x_data_dtype_on_disk": ddt,
                "x_indices_dtype_on_disk": idt,
                "x_data_dtype_min": np.dtype(min_d).name,
                "x_indices_dtype_min": np.dtype(min_i).name,
                "codec": (
                    BYTE_FILTER_CODEC_NAME if shuffle == BYTE_FILTER_SHUFFLE_NAME else CODEC_NAME
                ),
                "chunk_size_elements": CHUNK_SIZE_ELEMENTS,
                "shard_filename_template": template,
            }
            if spec.var is not None:  # own-var families (x, raw, modality anchors)
                vname = _mx.var_member_name(spec.key)
                v1_header.write_var_parquet(spec.var, out_dir / vname)
                entry["var_filename"] = vname
            if spec.role.startswith(_mx.MODALITY_PREFIX):
                mname = spec.role[len(_mx.MODALITY_PREFIX) :]
                entry["modality_name"] = mname
                mmeta = out_dir / _mx.meta_member_name(spec.key)
                # perm: obsm is cell-axis and must follow the SAME shared
                # permutation as this modality's on-disk X shards (identity
                # when group_by is None), or sidecar row i and shard row i
                # would silently refer to different cells under group_by.
                v1_header.write_modality_meta(modalities[mname], mmeta, perm=perm)
                entry["meta_filename"] = _mx.meta_member_name(spec.key)
            elif spec.key == _mx.PRIMARY_KEY and modalities:
                entry["modality_name"] = primary_modality or "rna"
                if primary_meta is not None:
                    mmeta = out_dir / _mx.meta_member_name(spec.key)
                    v1_header.write_modality_meta(primary_meta, mmeta, perm=perm)
                    entry["meta_filename"] = _mx.meta_member_name(spec.key)
            matrices_map[spec.key] = entry

        if groups is not None:
            _manifest.dump_groups(out_dir / "groups.json", groups)
        manifest_dict = _manifest.build_v4_manifest(
            n_obs_total=meta.n_obs,
            row_offsets=row_offsets,
            matrices=matrices_map,
            group_by=group_by,
            reference_shard=reference_shard,
            writer_fingerprint={
                "cellstream_version": _cellstream_version,
                "n_workers_at_write": n_workers,
                "n_shards_at_write": n_sh,
                "writer_kind": "v2-multi",
            },
            created_at=_dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        _manifest.dump_manifest(out_dir / "manifest.json", manifest_dict)


def _write_v2_grouped(
    source,
    out_dir,
    *,
    n_workers,
    reference,
    group_by,
    target_shard_bytes,
    shuffle,
    write_nthreads=_V2_WRITE_OMP_NTHREADS,
    sort_within_group=None,
    sort_order=None,
    dense_grouped_via_csr=False,
):
    """Lean grouped v2 write from in-memory AnnData or a source h5ad path.

    Plans from obs + CSR indptr metadata, writes permuted metadata, and gathers
    each shard's original rows on demand. The downstream encoder is unchanged,
    so shard bytes match the legacy eager ``adata[perm].copy()`` path.
    """
    out_dir = Path(out_dir)
    meta = _resolve_grouped_metadata(source)
    is_inmem = isinstance(source, ad.AnnData)
    import scipy.sparse as _sp

    if group_by not in meta.obs.columns:
        raise ValueError(
            f"group_by={group_by!r} is not a column in obs. "
            f"Available columns: {list(meta.obs.columns)}"
        )

    # sort_order controls how the primary group_by order AND every
    # sort_within_group key are compared. "natural" (default) keeps each column's
    # native dtype (categorical -> category order, numeric numerically, str
    # alphabetically); "alphabetical" forces every key to a string. Keys come from
    # obs (loaded for both providers), so in-mem and path order identically. #87.
    mode = "natural" if sort_order is None else sort_order

    def _key(col, *, group_label=False):
        # positional .values drops the obs index; .astype(str) for alphabetical.
        # group_label=True ONLY for the group_by column -- see the same guard in the streaming
        # path above: its key must share resolve_reference's missing-value normalization (#294).
        if mode != "alphabetical":
            return col.values
        return grouping.stringify_group_labels(col) if group_label else col.astype(str).values

    secondary_keys = None
    if sort_within_group is not None:
        sort_cols = (
            [sort_within_group] if isinstance(sort_within_group, str) else list(sort_within_group)
        )
        missing = [c for c in sort_cols if c not in meta.obs.columns]
        if missing:
            raise ValueError(
                f"sort_within_group column(s) {missing} not in obs. "
                f"Available columns: {list(meta.obs.columns)}"
            )
        secondary_keys = [_key(meta.obs[c]) for c in sort_cols]

    obs_adata = ad.AnnData(obs=meta.obs)
    if reference is not None:
        grouping.validate_reference_labels(obs_adata, group_by, reference)
    ref_mask = grouping.resolve_reference(reference, obs_adata, group_by)

    per_row_nnz = np.diff(meta.indptr).astype(np.int64)
    labels = _key(meta.obs[group_by], group_label=True)
    _fwd_ddt = _fwd_idt = None  # forwarded into the encoder (CSR + Rust only)
    if is_inmem:
        from cellstream.v2 import _rust_dispatch  # not imported elsewhere in this fn

        _X = source.X
        if _sp.issparse(_X):
            if _X.format not in ("csr", "csc", "coo"):
                _X = _sp.csr_matrix(_X)
            if _X.format == "csr":
                if _rust_dispatch.use_rust_for(shuffle, _X):
                    # #129: single parallel scan, reused by sizing AND the encoder.
                    _ddt_pre, _idt_pre, _min_d, _min_i = _rust_dispatch.decide_dtypes(
                        _X, meta.n_vars, n_workers
                    )
                    _fwd_ddt, _fwd_idt = _ddt_pre, _idt_pre
                else:
                    _ddt_pre, _idt_pre, _min_d, _min_i = narrow.decide_on_disk_dtypes(
                        _X.data, _X.indices, meta.n_vars
                    )
            else:
                # CSC/COO: .indices are NOT column indices, so do not scan them
                # (data-narrow is order-independent). Not forwarded: the encoder
                # coerces to CSR and re-decides the (matching) on-disk dtype.
                _min_uint = narrow._data_min_uint_dtype(_X.data)
                if _min_uint is not None:
                    _ddt_pre, _min_d = np.dtype(np.uint32), _min_uint
                else:
                    _ddt_pre = narrow._native_on_disk_data_dtype(_X.data.dtype)
                    _min_d = _ddt_pre
                _idt_pre = np.dtype(np.uint32)
                _min_i = _min_index_dtype_for_n_vars(meta.n_vars)
        else:
            _ddt_pre, _idt_pre, _min_d, _min_i = _decide_dense_on_disk_dtypes(_X, meta.n_vars)
            if _rust_dispatch.use_rust_for_dense(shuffle, np.asarray(_X)):
                _fwd_ddt = _ddt_pre  # dense index dtype is n_vars-derived in both places
        _bpn = _sizing_bytes_per_nnz(_min_d, _min_i)  # size by min width (compressed-equivalent)
    else:
        # Backed grouped writes decide the on-disk data dtype from a single streaming
        # scan of X/data (mirrors the in-mem decide_on_disk_dtypes): integer-routable
        # -> uint32; fractional / negative / overflow -> native float. min_d is
        # authoritative (archive-wide) so no per-shard aggregation is needed.
        _backed_on_disk_data, _min_d = _decide_backed_on_disk_data_dtype(source)
        _min_i = _min_index_dtype_for_n_vars(meta.n_vars)
        _bpn = _sizing_bytes_per_nnz(_min_d, _min_i)
    plan = grouping.plan_group_shards(
        group_labels=labels,
        per_row_nnz=per_row_nnz,
        reference_mask=ref_mask,
        target_shard_bytes=target_shard_bytes,
        bytes_per_nnz=_bpn,
        secondary_key=secondary_keys,
    )

    perm = plan.permutation
    row_offsets = plan.row_offsets
    n_shards = len(row_offsets) - 1
    logger.info(
        "grouped v2 write: %d shard(s) at target_shard_bytes=%d bytes",
        n_shards,
        target_shard_bytes,
    )

    _write_permuted_metadata(out_dir, meta, perm)
    oi_lists = [perm[row_offsets[s] : row_offsets[s + 1]] for s in range(n_shards)]

    if is_inmem:
        nnz_per_shard, ddt, idt = _grouped_encode_inmem(
            source,
            out_dir,
            oi_lists,
            meta.n_vars,
            shuffle,
            n_workers,
            write_nthreads,
            data_dtype=_fwd_ddt,
            index_dtype=_fwd_idt,
            dense_grouped_via_csr=dense_grouped_via_csr,
        )
    else:
        nnz_per_shard, ddt, idt = _grouped_encode_backed(
            str(source),
            out_dir,
            oi_lists,
            meta.n_vars,
            shuffle,
            n_workers,
            write_nthreads,
            _backed_on_disk_data,
        )
        # _min_d is authoritative from the upstream pre-scan; no post-encode override.

    _manifest.dump_groups(out_dir / "groups.json", plan.groups)
    total_nnz = int(sum(nnz_per_shard))
    manifest_dict = {
        "schema_version": 3,
        "shard_filename_template": "shard_{:04d}.shx",
        "n_shards": n_shards,
        "n_obs_total": meta.n_obs,
        "n_vars": meta.n_vars,
        "row_offsets": row_offsets,
        "nnz_per_shard": nnz_per_shard,
        "total_nnz": total_nnz,
        "x_data_dtype_on_disk": ddt,
        "x_indices_dtype_on_disk": idt,
        "x_data_dtype_min": np.dtype(_min_d).name,
        "x_indices_dtype_min": np.dtype(_min_i).name,
        "codec": (BYTE_FILTER_CODEC_NAME if shuffle == BYTE_FILTER_SHUFFLE_NAME else CODEC_NAME),
        "chunk_size_elements": CHUNK_SIZE_ELEMENTS,
        "group_by": group_by,
        "reference_shard": plan.reference_shard,
        "writer_fingerprint": {
            "cellstream_version": _cellstream_version,
            "n_workers_at_write": n_workers,
            "n_shards_at_write": n_shards,
            "writer_kind": "v2-native-lean",
        },
        "created_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _manifest.dump_manifest(out_dir / "manifest.json", manifest_dict)


def _grouped_encode_inmem(
    adata,
    out_dir,
    oi_lists,
    n_vars,
    shuffle,
    n_workers,
    write_nthreads,
    data_dtype=None,
    index_dtype=None,
    dense_grouped_via_csr=False,
):
    """Encode grouped shards from an in-memory AnnData using row gather."""
    import scipy.sparse as _sp

    from cellstream.v2 import _rust_dispatch

    X = adata.X
    if not _sp.issparse(X) and _rust_dispatch.use_rust_for_dense(shuffle, np.asarray(X)):
        if dense_grouped_via_csr:
            # Convert dense→CSR in parallel (Rust) and encode via the CSR core,
            # avoiding the scattered dense[r,:] reads of the direct encoder.
            return _rust_dispatch.encode_inmem_dense_via_csr(
                np.ascontiguousarray(X),
                out_dir,
                oi_lists,
                n_vars,
                shuffle,
                n_workers,
                data_dtype=data_dtype,
            )
        return _rust_dispatch.encode_inmem_dense(
            np.ascontiguousarray(X),
            out_dir,
            oi_lists,
            n_vars,
            shuffle,
            n_workers,
            data_dtype=data_dtype,
        )

    if not _sp.issparse(X) or X.format != "csr":
        X = _sp.csr_matrix(X)

    if _rust_dispatch.use_rust_for(shuffle, X):
        # Rust validates losslessly in-thread (scan_u16, zero-alloc); skip the
        # Python lossless preflight whose astype round-trip is the nnz-scaled
        # transient #95 targets.
        return _rust_dispatch.encode_inmem(
            X,
            out_dir,
            oi_lists,
            n_vars,
            shuffle,
            n_workers,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
        )

    ddt, idt, _, _ = narrow.decide_on_disk_dtypes(X.data, X.indices, n_vars)

    n_shards = len(oi_lists)
    total_source_nnz = int(X.data.size)
    n_indptr = X.shape[0] + 1
    use_parallel = n_workers > 1 and n_shards > 1
    if use_parallel and not _can_use_shm(
        total_nnz=total_source_nnz, n_indptr=n_indptr, data_dtype=ddt, indices_dtype=idt
    ):
        warnings.warn(
            "cellstream: v2 grouped parallel write falling back to serial gather — "
            "shared-memory source handoff unavailable (non-Linux or /dev/shm too "
            "small). The write still avoids the permuted-copy; it is just serial.",
            # user -> write_sharded -> write_v2_archive -> _write_v2_grouped ->
            # _grouped_encode_inmem -> warn: 5 frames to the public call site.
            stacklevel=5,
        )
        use_parallel = False

    bundle = None
    if use_parallel:
        try:
            bundle = _shm.allocate_shm_buffers(
                data_nbytes=max(1, total_source_nnz * ddt.itemsize),
                indices_nbytes=max(1, total_source_nnz * idt.itemsize),
                indptr_nbytes=max(1, n_indptr * 8),
            )
        except OSError as e:
            # TOCTOU: /dev/shm can fill between the _can_use_shm preflight and here.
            # Fall back to the serial path the preflight would have chosen rather
            # than crashing the write (ultrareview L11). allocate_shm_buffers cleans
            # up any partial segments before re-raising, so nothing leaks.
            warnings.warn(
                "cellstream: v2 grouped parallel write falling back to serial gather — "
                f"shared-memory allocation failed after the preflight ({e}). The write "
                "still avoids the permuted-copy; it is just serial.",
                stacklevel=5,
            )
            use_parallel = False

    if use_parallel:
        ex = None
        try:
            dv = np.ndarray((total_source_nnz,), dtype=ddt, buffer=bundle.data.buf)
            iv = np.ndarray((total_source_nnz,), dtype=idt, buffer=bundle.indices.buf)
            pv = np.ndarray((n_indptr,), dtype=np.int64, buffer=bundle.indptr.buf)
            # Cast happens elementwise on assignment into the already-typed views
            # (no full temporary array — matters at large nnz for f64/u32).
            dv[:] = X.data
            iv[:] = X.indices
            pv[:] = X.indptr
            inputs = [
                (
                    out_dir,
                    s,
                    bundle.data.name,
                    bundle.indices.name,
                    bundle.indptr.name,
                    total_source_nnz,
                    n_indptr,
                    oi_lists[s],
                    int(n_vars),
                    shuffle,
                    ddt.name,
                    idt.name,
                )
                for s in range(n_shards)
            ]
            ctx = mp.get_context("spawn")
            with _workers.omp_nthreads_in_env(write_nthreads):
                ex = ProcessPoolExecutor(max_workers=min(n_workers, n_shards), mp_context=ctx)
                fut_to_idx = {ex.submit(_encode_one_shard_gather_shm, a): a[1] for a in inputs}
                results_by_idx = {}
                for fut in as_completed(fut_to_idx):
                    results_by_idx[fut_to_idx[fut]] = fut.result()
                ex.shutdown(wait=True)
            results = [results_by_idx[s] for s in range(n_shards)]
        finally:
            # Nested finally: even if ex.shutdown raises (e.g. KeyboardInterrupt),
            # the shm bundle must still be unlinked or it leaks in /dev/shm.
            try:
                if ex is not None:
                    ex.shutdown(wait=True, cancel_futures=True)
            finally:
                bundle.close_and_unlink()
    else:
        results = []
        with _workers.omp_nthreads_in_env(write_nthreads):
            for s in range(n_shards):
                results.append(
                    _encode_one_shard_gather_local(
                        (
                            out_dir,
                            s,
                            X.data,
                            X.indices,
                            X.indptr,
                            oi_lists[s],
                            int(n_vars),
                            shuffle,
                            ddt.name,
                            idt.name,
                        )
                    )
                )
    return [r["nnz"] for r in sorted(results, key=lambda r: r["shard_idx"])], ddt.name, idt.name


def _grouped_encode_backed(
    src_path, out_dir, oi_lists, n_vars, shuffle, n_workers, write_nthreads, on_disk_data
):
    """Encode grouped shards from a backed h5ad source without loading full X.

    ``on_disk_data`` is the archive-wide data dtype decided upstream by
    ``_decide_backed_on_disk_data_dtype`` (uint32 for integer-routable data, else the
    native float width). Returns ``(nnz_per_shard, on_disk_data_name, "uint32")``.
    """
    n_shards = len(oi_lists)
    dd = np.dtype(on_disk_data).name
    inputs = [
        (out_dir, s, src_path, oi_lists[s], int(n_vars), shuffle, dd) for s in range(n_shards)
    ]
    use_parallel = n_workers > 1 and n_shards > 1
    if use_parallel:
        ctx = mp.get_context("spawn")
        ex = None
        try:
            with _workers.omp_nthreads_in_env(write_nthreads):
                ex = ProcessPoolExecutor(max_workers=min(n_workers, n_shards), mp_context=ctx)
                fut_to_idx = {ex.submit(_encode_one_shard_gather_backed, a): a[1] for a in inputs}
                results_by_idx = {}
                for fut in as_completed(fut_to_idx):
                    results_by_idx[fut_to_idx[fut]] = fut.result()
                ex.shutdown(wait=True)
            results = [results_by_idx[s] for s in range(n_shards)]
        finally:
            if ex is not None:
                ex.shutdown(wait=True, cancel_futures=True)
    else:
        with _workers.omp_nthreads_in_env(write_nthreads):
            results = [_encode_one_shard_gather_backed(a) for a in inputs]
    results_sorted = sorted(results, key=lambda r: r["shard_idx"])
    nnz_per_shard = [r["nnz"] for r in results_sorted]
    return nnz_per_shard, dd, "uint32"


def _write_v2_inmem(
    adata,
    out_dir,
    *,
    n_shards,
    n_workers,
    target_shard_bytes=2 * 2**30,
    shuffle="bitshuffle",
    write_nthreads=_V2_WRITE_OMP_NTHREADS,
):
    out_dir = Path(out_dir)
    n_obs, n_vars = adata.shape

    # No group_by: byte-driven even split or explicit n_shards.
    # Dense byte-filter writes can go straight to Rust; fallback paths still
    # coerce to CSR so downstream code can assume .data/.indices/.indptr.
    import scipy.sparse as _sp

    from cellstream.v2 import _rust_dispatch

    _X = adata.X
    _dense_rust = (
        not _sp.issparse(_X)
        and n_obs > 0
        and _rust_dispatch.use_rust_for_dense(shuffle, np.asarray(_X))
    )
    if _dense_rust:
        D = np.ascontiguousarray(_X)
        # Decide on-disk + min dtypes ONCE (parallel scan), reused for sizing, the
        # encoder (forwarded -> skips its own scan), and the manifest min.
        _ddt_pre, _idt_pre, _min_d, _min_i = _rust_dispatch.decide_dtypes_dense(
            D, n_vars, n_workers
        )
        if n_shards is None:
            total_nnz = int(np.count_nonzero(D))
            n_shards = max(
                1,
                math.ceil(total_nnz * _sizing_bytes_per_nnz(_min_d, _min_i) / target_shard_bytes),
            )
        n_shards = min(n_shards, n_obs)
        row_offsets = [int(round(i * n_obs / n_shards)) for i in range(n_shards + 1)]
        row_lists = [
            np.arange(row_offsets[s], row_offsets[s + 1], dtype=np.int64) for s in range(n_shards)
        ]
        # Dense index dtype is n_vars-derived in both decide and encode, so only
        # data_dtype is forwarded (the encoder re-derives the uint32 index dtype).
        nnz_per_shard, ddt, idt = _rust_dispatch.encode_inmem_dense(
            D, out_dir, row_lists, n_vars, shuffle, n_workers, data_dtype=_ddt_pre
        )
        v1_header.write_header(adata, out_dir / "header.h5ad")
        v1_header.write_obs_parquet(adata.obs, out_dir / "obs.parquet")
    else:
        if not _sp.issparse(_X) or _X.format != "csr":
            adata = adata.copy()
            adata.X = _sp.csr_matrix(_X)

        use_rust = _rust_dispatch.use_rust_for(shuffle, adata.X) and n_obs > 0
        # Decide on-disk + min dtypes ONCE (parallel Rust scan when Rust will encode;
        # numpy otherwise), reused for sizing, the encoder, and the manifest min. No
        # extra scan: when forwarded, the encoder skips its own scan.
        if use_rust:
            _ddt_pre, _idt_pre, _min_d, _min_i = _rust_dispatch.decide_dtypes(
                adata.X, n_vars, n_workers
            )
        else:
            _ddt_pre, _idt_pre, _min_d, _min_i = narrow.decide_on_disk_dtypes(
                adata.X.data, adata.X.indices, n_vars
            )
        if n_obs == 0:
            # Empty AnnData: produce a single empty shard.
            n_shards = 1
            row_offsets = [0, 0]
        else:
            if n_shards is None:
                total_nnz = int(adata.X.indptr[-1])
                n_shards = max(
                    1,
                    math.ceil(
                        total_nnz * _sizing_bytes_per_nnz(_min_d, _min_i) / target_shard_bytes
                    ),
                )
            # Classic guard: never more shards than obs.
            n_shards = min(n_shards, n_obs)
            row_offsets = [int(round(i * n_obs / n_shards)) for i in range(n_shards + 1)]

        if not use_rust:
            ddt, idt = _ddt_pre, _idt_pre
        if use_rust:
            # Rust's scan_u16 preflight validates (and raises LossyCastError) before
            # writing any shard, so write the header/obs sidecars only AFTER a
            # successful encode — preserving the "no disk artifact on a lossy cast"
            # guarantee without the Python lossless_cast transient (#95).
            row_lists = [
                np.arange(row_offsets[s], row_offsets[s + 1], dtype=np.int64)
                for s in range(n_shards)
            ]
            nnz_per_shard, ddt, idt = _rust_dispatch.encode_inmem(
                adata.X,
                out_dir,
                row_lists,
                n_vars,
                shuffle,
                n_workers,
                data_dtype=_ddt_pre,
                index_dtype=_idt_pre,
            )
            v1_header.write_header(adata, out_dir / "header.h5ad")
            v1_header.write_obs_parquet(adata.obs, out_dir / "obs.parquet")
        else:
            # Pre-flight (above) already validated before any disk write. Match
            # v0.1's call shape exactly: write_header(adata, path), then encode.
            v1_header.write_header(adata, out_dir / "header.h5ad")
            v1_header.write_obs_parquet(adata.obs, out_dir / "obs.parquet")
            # Size the shm views by data.size (== indptr[-1] for canonical CSR, >= it
            # otherwise) so the fill `view[:] = X.data` always matches; workers slice
            # [nnz0:nnz1] with nnz1 <= indptr[-1] <= data.size, so the bound holds.
            total_source_nnz = int(adata.X.data.size)
            n_indptr = n_obs + 1
            use_parallel = n_workers > 1 and n_shards > 1
            if use_parallel and not _can_use_shm(
                total_nnz=total_source_nnz, n_indptr=n_indptr, data_dtype=ddt, indices_dtype=idt
            ):
                warnings.warn(
                    "cellstream: v2 parallel write falling back to serial — the shared-memory "
                    "source handoff is unavailable (non-Linux or /dev/shm too small for the "
                    f"~{total_source_nnz * (ddt.itemsize + idt.itemsize) + n_indptr * 8:,}-byte source). The write "
                    "will be slower; free/grow /dev/shm or reduce the source to parallelize.",
                    # 4 frames up: warn -> _write_v2_inmem -> write_v2_archive ->
                    # write_sharded -> user call site (the public API boundary).
                    stacklevel=4,
                )
                use_parallel = False

            bundle = None
            if use_parallel:
                # Parallel path: hand the FULL source CSR to spawn workers via one shm
                # bundle (cast to the on-disk dtypes once, here), so no per-shard arrays
                # are pickled through the IPC pipe (issue #76). Workers attach by name,
                # slice their row range, and encode. The encode is unchanged, so .shx
                # bytes are byte-identical to the serial path.
                try:
                    bundle = _shm.allocate_shm_buffers(
                        data_nbytes=max(1, total_source_nnz * ddt.itemsize),
                        indices_nbytes=max(1, total_source_nnz * idt.itemsize),
                        indptr_nbytes=max(1, n_indptr * 8),
                    )
                except OSError as e:
                    # TOCTOU: /dev/shm can fill between the _can_use_shm preflight and
                    # here. Fall back to the serial path rather than crashing the write
                    # (ultrareview L11). allocate_shm_buffers cleans up partial segments
                    # before re-raising, so nothing leaks.
                    warnings.warn(
                        "cellstream: v2 parallel write falling back to serial — shared-memory "
                        f"allocation failed after the preflight ({e}). The write will be "
                        "slower; free/grow /dev/shm to parallelize.",
                        stacklevel=4,
                    )
                    use_parallel = False

            if use_parallel:
                ex = None
                try:
                    # Fill once (assignment casts float32->uint16 / int32->uint16 /
                    # int->int64; the cast is already proven lossless by the preflight
                    # scan above, so the implicit setitem cast is safe).
                    data_view = np.ndarray((total_source_nnz,), dtype=ddt, buffer=bundle.data.buf)
                    indices_view = np.ndarray(
                        (total_source_nnz,), dtype=idt, buffer=bundle.indices.buf
                    )
                    indptr_view = np.ndarray((n_indptr,), dtype=np.int64, buffer=bundle.indptr.buf)
                    data_view[:] = adata.X.data
                    indices_view[:] = adata.X.indices
                    indptr_view[:] = adata.X.indptr
                    inputs = []
                    for s in range(n_shards):
                        r0, r1 = row_offsets[s], row_offsets[s + 1]
                        inputs.append(
                            (
                                out_dir,
                                s,
                                bundle.data.name,
                                bundle.indices.name,
                                bundle.indptr.name,
                                total_source_nnz,
                                n_indptr,
                                int(r0),
                                int(r1),
                                int(n_vars),
                                shuffle,
                                ddt.name,
                                idt.name,
                            )
                        )
                    # spawn (not fork): bitshuffle/OpenMP deadlocks on fork-after-import.
                    ctx = mp.get_context("spawn")
                    # Bound OMP_NUM_THREADS in the parent for the executor's lifetime so
                    # spawn workers don't oversubscribe (held across submit+as_completed
                    # because the pool spawns workers lazily).
                    with _workers.omp_nthreads_in_env(write_nthreads):
                        # Cap workers at n_shards (== len(inputs)): a write with
                        # n_workers > n_shards shouldn't spawn idle interpreters
                        # (~0.5-1 s startup + a bitshuffle import each). Mirrors
                        # reader_cpu's max_workers=min(n_workers, len(inputs)).
                        ex = ProcessPoolExecutor(
                            max_workers=min(n_workers, n_shards), mp_context=ctx
                        )
                        # as_completed (not ex.map): surfaces the FIRST worker death
                        # promptly; results re-sorted by shard_idx below.
                        future_to_idx = {
                            ex.submit(_encode_one_shard_from_shm, i): i[1] for i in inputs
                        }
                        results_by_idx: dict = {}
                        for fut in as_completed(future_to_idx):
                            results_by_idx[future_to_idx[fut]] = fut.result()
                        ex.shutdown(wait=True)
                    results = [results_by_idx[s] for s in range(n_shards)]
                finally:
                    # wait=True (not False): this runs inside the archive write lock, so
                    # we must not unwind while a surviving worker is mid-.shx-write (that
                    # would release the lock and let a concurrent writer corrupt the
                    # archive). On a healthy run shutdown already happened; this is the
                    # error/interrupt path. cancel_futures drops not-yet-started shards.
                    if ex is not None:
                        ex.shutdown(wait=True, cancel_futures=True)
                    # Parent owns the bundle: unlink on success AND failure (no /dev/shm
                    # leak). close_and_unlink is idempotent.
                    bundle.close_and_unlink()
            else:
                # Serial path (n_workers<=1, n_shards<=1, or shm unavailable): slice
                # in-process and encode. No shm; behavior identical to pre-#76.
                inputs = []
                for s in range(n_shards):
                    r0, r1 = row_offsets[s], row_offsets[s + 1]
                    sub = adata.X[r0:r1]
                    inputs.append(
                        (
                            out_dir,
                            s,
                            sub.data,
                            sub.indices,
                            sub.indptr,
                            int(r1 - r0),
                            int(n_vars),
                            shuffle,
                            ddt.name,
                            idt.name,
                        )
                    )
                results = [_encode_one_shard(i) for i in inputs]
            nnz_per_shard = [r["nnz"] for r in results]
            ddt, idt = ddt.name, idt.name
    total_nnz = int(sum(nnz_per_shard))

    manifest_dict = {
        "schema_version": SCHEMA_VERSION,
        "shard_filename_template": "shard_{:04d}.shx",
        "n_shards": n_shards,
        "n_obs_total": n_obs,
        "n_vars": n_vars,
        "row_offsets": row_offsets,
        "nnz_per_shard": nnz_per_shard,
        "total_nnz": total_nnz,
        "x_data_dtype_on_disk": ddt,
        "x_indices_dtype_on_disk": idt,
        "x_data_dtype_min": np.dtype(_min_d).name,
        "x_indices_dtype_min": np.dtype(_min_i).name,
        "codec": (BYTE_FILTER_CODEC_NAME if shuffle == BYTE_FILTER_SHUFFLE_NAME else CODEC_NAME),
        "chunk_size_elements": CHUNK_SIZE_ELEMENTS,
        "writer_fingerprint": {
            "cellstream_version": _cellstream_version,
            "n_workers_at_write": n_workers,
            "n_shards_at_write": n_shards,
            "writer_kind": "v2-native",
        },
        "created_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _manifest.dump_manifest(out_dir / "manifest.json", manifest_dict)


def _encode_one_shard(args):
    """Serial-path entry: slice already done in parent; cast then encode.

    args = (out_dir, shard_idx, data, indices, indptr, n_rows, n_vars, shuffle,
            data_dtype_name, indices_dtype_name)
    where data/indices/indptr are the parent's per-shard slices (source dtypes).
    """
    (
        out_dir,
        shard_idx,
        data,
        indices,
        indptr,
        n_rows,
        n_vars,
        shuffle,
        data_dtype_name,
        indices_dtype_name,
    ) = args
    _workers._maybe_test_kill_self(shard_idx)
    data_arr = data.astype(np.dtype(data_dtype_name), copy=False)
    indices_arr = indices.astype(np.dtype(indices_dtype_name), copy=False)
    indptr_i64 = indptr.astype(np.int64)
    return _encode_shard_streams(
        out_dir, shard_idx, data_arr, indices_arr, indptr_i64, n_rows, n_vars, shuffle
    )


def _encode_one_shard_from_shm(args):
    """Parallel-path worker: slice this shard's rows from the shared-memory
    source bundle (no pickled arrays cross the process boundary), then encode.

    args = (out_dir, shard_idx, data_name, indices_name, indptr_name,
            total_nnz, n_indptr, r0, r1, n_vars, shuffle, data_dtype_name,
            indices_dtype_name)
    The three segments hold the FULL source CSR already cast to the on-disk
    dtypes. Mirrors
    reader_cpu._read_one_shard_into_shm: attach by name, build views, slice,
    close (only the parent unlinks).
    """
    (
        out_dir,
        shard_idx,
        data_name,
        indices_name,
        indptr_name,
        total_nnz,
        n_indptr,
        r0,
        r1,
        n_vars,
        shuffle,
        data_dtype_name,
        indices_dtype_name,
    ) = args
    _workers._maybe_test_kill_self(shard_idx)
    from multiprocessing import shared_memory

    d_shm = shared_memory.SharedMemory(name=data_name)
    i_shm = shared_memory.SharedMemory(name=indices_name)
    p_shm = shared_memory.SharedMemory(name=indptr_name)
    try:
        data_view = np.ndarray((total_nnz,), dtype=np.dtype(data_dtype_name), buffer=d_shm.buf)
        indices_view = np.ndarray(
            (total_nnz,), dtype=np.dtype(indices_dtype_name), buffer=i_shm.buf
        )
        indptr_view = np.ndarray((n_indptr,), dtype=np.int64, buffer=p_shm.buf)
        nnz0 = int(indptr_view[r0])
        nnz1 = int(indptr_view[r1])
        n_rows = r1 - r0
        data_arr = data_view[nnz0:nnz1]
        indices_arr = indices_view[nnz0:nnz1]
        # Rebase this shard's indptr to 0 (matches the serial path's sub.indptr).
        indptr_i64 = (indptr_view[r0 : r1 + 1] - nnz0).astype(np.int64)
        return _encode_shard_streams(
            out_dir, shard_idx, data_arr, indices_arr, indptr_i64, n_rows, n_vars, shuffle
        )
    finally:
        # Workers only close() their mappings; the PARENT owns unlink() (it
        # calls bundle.close_and_unlink() after the pool drains). Same contract
        # as reader_cpu / shm.py — do not unlink here.
        d_shm.close()
        i_shm.close()
        p_shm.close()


def _encode_one_shard_gather_local(args):
    """Serial / in-parent gather worker: gather scattered rows from in-RAM CSR
    arrays (source dtypes), cast, encode. Used for the shm-unavailable fallback.

    args = (out_dir, shard_idx, data, indices, indptr, row_indices, n_vars, shuffle,
            data_dtype_name, indices_dtype_name)
    """
    (
        out_dir,
        shard_idx,
        data,
        indices,
        indptr,
        row_indices,
        n_vars,
        shuffle,
        data_dtype_name,
        indices_dtype_name,
    ) = args
    _workers._maybe_test_kill_self(shard_idx)
    d, i, p = gather_csr_rows(data, indices, indptr, row_indices)
    data_arr = d.astype(np.dtype(data_dtype_name), copy=False)
    indices_arr = i.astype(np.dtype(indices_dtype_name), copy=False)
    return _encode_shard_streams(
        out_dir, shard_idx, data_arr, indices_arr, p, len(row_indices), n_vars, shuffle
    )


def _encode_one_shard_gather_shm(args):
    """Parallel shm gather worker: attach the shm bundle holding the ORIGINAL
    CSR (already cast to on-disk dtypes), gather scattered rows, encode.

    args = (out_dir, shard_idx, data_name, indices_name, indptr_name,
            total_nnz, n_indptr, row_indices, n_vars, shuffle, data_dtype_name,
            indices_dtype_name)
    """
    (
        out_dir,
        shard_idx,
        data_name,
        indices_name,
        indptr_name,
        total_nnz,
        n_indptr,
        row_indices,
        n_vars,
        shuffle,
        data_dtype_name,
        indices_dtype_name,
    ) = args
    _workers._maybe_test_kill_self(shard_idx)
    from multiprocessing import shared_memory

    d_shm = shared_memory.SharedMemory(name=data_name)
    i_shm = shared_memory.SharedMemory(name=indices_name)
    p_shm = shared_memory.SharedMemory(name=indptr_name)
    try:
        data_view = np.ndarray((total_nnz,), dtype=np.dtype(data_dtype_name), buffer=d_shm.buf)
        indices_view = np.ndarray(
            (total_nnz,), dtype=np.dtype(indices_dtype_name), buffer=i_shm.buf
        )
        indptr_view = np.ndarray((n_indptr,), dtype=np.int64, buffer=p_shm.buf)
        d, i, p = gather_csr_rows(data_view, indices_view, indptr_view, row_indices)
        # views are already at the target on-disk dtypes; gather preserves dtype.
        return _encode_shard_streams(out_dir, shard_idx, d, i, p, len(row_indices), n_vars, shuffle)
    finally:
        d_shm.close()
        i_shm.close()
        p_shm.close()


def _encode_one_shard_gather_backed(args):
    """Parallel backed gather worker: read scattered rows from a backed h5ad, cast to the
    archive-wide on-disk data dtype decided upstream, encode.

    args = (out_dir, shard_idx, src_path, row_indices, n_vars, shuffle, data_dtype_name)
    """
    out_dir, shard_idx, src_path, row_indices, n_vars, shuffle, data_dtype_name = args
    _workers._maybe_test_kill_self(shard_idx)
    d, i, p = read_backed_csr_rows(src_path, row_indices)
    target = np.dtype(data_dtype_name)
    if target.kind == "u":
        # Integer-routable per the upstream pre-scan; lossless_cast is a cheap
        # defense-in-depth re-validation (fires only if the file changed between the
        # scan and this read).
        data_arr = narrow.lossless_cast(d, target, where=f"shard[{shard_idx}].X.data")
    else:
        # Native float on-disk (fractional / negative / > uint32); the pre-scan already
        # ruled out an integer representation. _encode_shard_streams stores float raw.
        data_arr = d.astype(target, copy=False)
    # Indices are trusted (CSR invariant 0 <= idx < n_vars), matching the in-mem path —
    # except, like that path (decide_dtypes), guard against a negative column index when
    # n_vars > 65536 (it would silently wrap to a huge uint32). Skip the min() scan for
    # unsigned index dtypes, which can't be negative (ultrareview L12).
    if n_vars > 65536 and i.size and i.dtype.kind != "u" and int(i.min()) < 0:
        raise LossyCastError(
            f"shard[{shard_idx}].X.indices: column indices must be >= 0 for unsigned "
            f"storage; got min {int(i.min())}."
        )
    indices_u32 = i.astype(np.uint32)
    return _encode_shard_streams(
        out_dir, shard_idx, data_arr, indices_u32, p, len(row_indices), n_vars, shuffle
    )


def _encode_shard_streams(
    out_dir, shard_idx, data_arr, indices_arr, indptr_i64, n_rows, n_vars, shuffle
):
    """Encode one shard's three (already-typed, already-rebased) streams to .shx.

    data_arr/indices_arr are at their on-disk dtypes; indptr_i64 is int64 rebased to 0 for this
    shard. THIS FUNCTION is incremental in ``O(CHUNK_SIZE_ELEMENTS × itemsize)`` and retains no
    chunk history — a scaling statement, since each chunk also costs codec-dependent
    temporaries and the compressed output. Not a bound on the worker's total memory either: the
    arrays arrive already materialized and the gather arms build a whole shard first.
    Shared by the serial path (_encode_one_shard) and the parallel shm worker
    (_encode_one_shard_from_shm) so both produce byte-identical output.
    """
    streams = [data_arr, indices_arr, indptr_i64]

    def _n_chunks_for(arr):
        return max(1, (arr.size + CHUNK_SIZE_ELEMENTS - 1) // CHUNK_SIZE_ELEMENTS)

    per_stream_chunk_count = [_n_chunks_for(a) for a in streams]

    use_byte_filter = shuffle == codec.BYTE_FILTER_SHUFFLE_NAME
    _roles = ("data", "indices", "indptr")

    # #142: under the byte-filter codec, a FLOAT data stream is stored RAW (zstd
    # only, no shuffle/delta) — byteshuffle scatters each float's bytes into
    # separate planes and destroys the whole-value repeats zstd matches, inflating
    # low-cardinality lognorm data ~2.8x. Integer data + index/indptr streams keep
    # SHUFFLE(+DELTA). The shard self-describes via the rawdata marker; the data
    # dtype is uniform per archive, so a per-shard marker suffices.
    raw_data = use_byte_filter and data_arr.dtype.kind == "f"
    effective_shuffle = codec.BYTE_FILTER_RAWDATA_SHUFFLE_NAME if raw_data else shuffle

    def _encode(slab, stream_idx):
        if use_byte_filter:
            if raw_data and stream_idx == 0:  # float data stream -> raw zstd
                return codec.zstd_only_encode(slab, level=1)
            return codec.byte_filter_encode(slab, role=_roles[stream_idx], level=1)
        return codec.bitshuffle_zstd_encode(slab, level=1)

    def _chunks_iter():
        for stream_idx, arr in enumerate(streams):
            if arr.size == 0:
                yield stream_idx, _encode(arr, stream_idx), 0
                continue
            for i in range(0, arr.size, CHUNK_SIZE_ELEMENTS):
                slab = arr[i : i + CHUNK_SIZE_ELEMENTS]
                yield stream_idx, _encode(slab, stream_idx), int(slab.size * slab.dtype.itemsize)

    nnz = int(data_arr.size)
    header = ShxHeader(
        schema_version=SCHEMA_VERSION,
        shard_idx=shard_idx,
        n_rows=int(n_rows),
        n_vars=int(n_vars),
        nnz=nnz,
        compression="zstd",
        shuffle=effective_shuffle,
        compression_level=1,
        chunk_size_elements=CHUNK_SIZE_ELEMENTS,
        data=StreamSpec(data_arr.dtype.name, 0, []),
        indices=StreamSpec(indices_arr.dtype.name, 0, []),
        indptr=StreamSpec("int64", 0, []),
    )
    write_shx_file_streaming(
        out_dir / f"shard_{shard_idx:04d}.shx",
        header,
        chunks_iter=_chunks_iter(),
        per_stream_chunk_count=per_stream_chunk_count,
    )
    return {"shard_idx": shard_idx, "nnz": nnz}
