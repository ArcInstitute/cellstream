"""Read-side public dispatcher.

ShardedArchive(path) parses manifest.json::schema_version and reads through
the v2 reader (cellstream.v2.reader_cpu). Archives are packed (SHPK) single
files; the v1 h5ad-shard reader and its __getattr__ proxy were removed in
#154, along with the ShardedH5ad alias.

Single-file .h5ad helpers (read_narrow_h5ad, read_h5ad, read_obs, read_var)
live in cellstream.h5ad_io and are re-exported here.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellstream import h5ad_io
from cellstream import shm as _shm
from cellstream.errors import reject_non_packed
from cellstream.manifest import validate_group_records
from cellstream.v2 import reader_cpu

_V1_MATERIALIZATION_MSG = (
    "materialization knobs (container/data_dtype/index_dtype) are v2-only; "
    "re-write the archive as v2 (write_sharded(..., format='v2'))."
)


def _materialization_kwargs_set(container, data_dtype, index_dtype):
    """True iff the caller requested any non-default materialization knob."""
    return container != "csr" or data_dtype is not None or index_dtype is not None


@dataclass(frozen=True)
class ShardCSRDLPack:
    """A streamed shard's CSR arrays for a GPU consumer (#160).

    Each array implements the DLPack protocol; wrap zero-copy with
    ``torch.from_dlpack(caps.data)`` (etc.) then ``.to(device, non_blocking=True)``.
    When produced by ``iter_group_shards(prefetch>=1, pinned=True)`` the arrays
    are page-locked so the async H2D overlaps compute; otherwise they are
    pageable. ``shape`` is the shard's ``(n_obs_shard, n_vars)``.
    """

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple


class _HandoffArray(np.ndarray):
    """A view handed out by ``GroupShard.dlpack()`` that pins its source CSR (and
    thus any shm bundle) alive via ``_cellstream_dlpack_keepalive``.

    A bare ndarray rejects attributes, so a consumer that keeps only the raw
    array (or a ``np.from_dlpack`` / ``torch.from_dlpack`` view of it) would let
    the source CSR — and the shm segments backing it in the Python decode path —
    be freed while the view is still live (use-after-free; the #182 class of
    bug). DLPack/``from_dlpack`` keep this view object alive via the capsule
    deleter, so the keepalive ref transitively holds the bundle open.
    """


class GroupShard:
    """Lightweight handle for a single non-reference shard in a grouped archive.

    I/O is deferred until ``.to_anndata()`` / ``.to_cupy_anndata()`` is called.

    Attributes
    ----------
    archive : ShardedArchive
        The parent archive.
    shard_index : int
        Zero-based index of this shard in the archive.
    global_start : int
        First global row (inclusive) belonging to this shard.
    global_stop : int
        First global row (exclusive) after this shard.
    groups : dict[str, slice]
        Mapping from group label to a *shard-local* slice selecting the rows
        of that group within the data returned by ``.to_anndata()``.
    """

    def __init__(
        self,
        archive: ShardedArchive,
        shard_index: int,
        global_start: int,
        global_stop: int,
        groups: dict[str, slice],
        _decoded=None,
        matrix: str = "x",
    ) -> None:
        self.archive = archive
        self.shard_index = shard_index
        self.global_start = global_start
        self.global_stop = global_stop
        self.groups = groups
        # Set by iter_group_shards(prefetch>=1) to a prefetch._DecodedShard carrier
        # (raw matrix + deferred AnnData); None in lazy mode (prefetch=0).
        self._decoded = _decoded
        # Family selector (Task 8); "x" (default) is the primary matrix. Carried
        # so the shared-partition streaming API (iter_group_shards(matrix=...))
        # can read one family per shard while shard i stays the same cells
        # across families.
        self.matrix = matrix

    @property
    def labels(self) -> list[str]:
        """Group labels present in this shard."""
        return list(self.groups.keys())

    def to_anndata(self, **kw) -> ad.AnnData:
        """Read this shard and return an AnnData (deferred I/O).

        When this ``GroupShard`` was produced by ``iter_group_shards(prefetch>=1)``
        the AnnData is already decoded and is returned directly; passing decode
        kwargs here is then an error (they were applied at the iterator call).
        """
        if self._decoded is not None:
            if kw:
                raise ValueError(
                    "this GroupShard is pre-decoded by iter_group_shards(prefetch=...); "
                    "its AnnData is already materialized. Pass decode kwargs to "
                    "iter_group_shards(prefetch=..., **kwargs), not to .to_anndata() "
                    f"(got {sorted(kw)})."
                )
            return self._decoded.to_anndata()
        return self.archive[self.global_start : self.global_stop].to_anndata(
            matrix=self.matrix, **kw
        )

    def x(self, **kw):
        """Return this shard's raw X matrix (scipy CSR or dense ndarray), no AnnData.

        The per-group row slices are on ``self.groups``; this returns only the
        matrix. The result is byte-identical to ``self.to_anndata(**kw).X`` for the
        same knobs, but skips the AnnData wrapper (and, in lazy mode, the per-shard
        header read).

        When this ``GroupShard`` is pre-decoded by ``iter_group_shards(prefetch>=1)``
        the matrix is already materialized and is returned directly; passing decode
        kwargs here is then an error (they were applied at the iterator call,
        exactly as for ``to_anndata()``).
        """
        if self._decoded is not None:
            if kw:
                raise ValueError(
                    "this GroupShard is pre-decoded by iter_group_shards(prefetch=...); "
                    "its X is already materialized. Pass decode kwargs to "
                    "iter_group_shards(prefetch=..., **kwargs), not to .x() "
                    f"(got {sorted(kw)})."
                )
            return self._decoded.x
        return self.archive[self.global_start : self.global_stop].x(matrix=self.matrix, **kw)

    def x_cupy(self, **kw):
        """Return this shard's raw X as a device-resident cupy CSR (no AnnData).

        Device sibling of ``x()`` for GPU consumers (e.g. gpudge's streaming DE):
        decodes this shard on the GPU (nvCOMP-RAW zstd + cupy byte-filter inverse)
        and returns a ``cupyx.scipy.sparse.csr_matrix``. On a pre-decoded
        (``prefetch>=1``) GroupShard this raises ``ValueError`` — the batch was
        decoded to host, so a device materialization must be a lazy read (mirrors
        ``to_cupy_anndata`` / ``.x()`` on a pre-decoded shard).
        """
        if self._decoded is not None:
            raise ValueError(
                "x_cupy() is not supported on a pre-decoded GroupShard "
                "(iter_group_shards(prefetch=...) decodes to host); iterate without "
                "prefetch for a device materialization."
            )
        return self.archive[self.global_start : self.global_stop].x_cupy(**kw)

    def dlpack(self) -> ShardCSRDLPack:
        """Return this shard's CSR arrays as a DLPack-exportable struct (#160).

        For a GPU consumer: ``t = torch.from_dlpack(caps.data); t.to(device,
        non_blocking=True)``. The async copy overlaps compute only when this
        GroupShard came from ``iter_group_shards(prefetch>=1, pinned=True)``
        (page-locked); otherwise the memory is pageable. CSR only — a dense
        shard raises ``ValueError``.
        """
        x = self.x()
        if not sp.issparse(x):
            raise ValueError(
                "GroupShard.dlpack() is CSR-only; this shard's X is dense. Use "
                "iter_group_shards(container='csr') (the default)."
            )
        x = x.tocsr()  # no-op when already CSR

        def _pin(a):
            # Hand out a view that keeps the CSR (and its shm bundle, in the
            # Python decode path) alive for as long as any DLPack consumer holds
            # the array — see _HandoffArray.
            v = a.view(_HandoffArray)
            v._cellstream_dlpack_keepalive = x
            return v

        return ShardCSRDLPack(
            data=_pin(x.data), indices=_pin(x.indices), indptr=_pin(x.indptr), shape=x.shape
        )

    def to_cupy_anndata(self, **kw) -> ad.AnnData:
        """Read this shard into a cupy-backed AnnData (deferred I/O)."""
        if self._decoded is not None:
            raise ValueError(
                "to_cupy_anndata() is not supported on a pre-decoded GroupShard "
                "(iter_group_shards(prefetch=...) decodes to host CSR/dense); use "
                ".to_anndata() or read without prefetch for a cupy materialization."
            )
        return self.archive[self.global_start : self.global_stop].to_cupy_anndata(**kw)


class ShardedArchive:
    """Read entry point for packed v2 (.shx shard) archives.

    Parses manifest.json on construction; schema versions 2/3/4 are handled.
    The v1 h5ad-shard reader was removed in #154, so there is no longer a
    ``schema_version == 1`` branch. CPU materializations default to
    ``output_backing="heap"`` and return ordinary heap arrays; pass
    ``output_backing="shm"`` to keep a SharedMemory bundle attached to the
    returned AnnData (released on AnnData GC). The pure-Python fallback uses
    shared memory internally regardless.

    """

    def __init__(self, path):
        self.path = Path(path)
        from cellstream.packed.reader import PackedArchive, is_packed_file

        # A path that does not exist is a typo, not a legacy archive: answering it with
        # the PACKED_ONLY_MIGRATION note sends the caller after a migration they do not
        # need. Checked first so all four archive entry points agree with read_h5ad,
        # which has always raised FileNotFoundError before dispatching.
        if not self.path.exists():
            raise FileNotFoundError(f"no such archive: {self.path}")
        if not (self.path.is_file() and is_packed_file(self.path)):
            reject_non_packed("ShardedArchive requires a single-file packed archive.")
        self._packed = PackedArchive(self.path)
        self.manifest = self._packed.manifest
        if self.manifest.get("layout") == "cell":
            from cellstream.errors import IncompatibleSchemaError

            raise IncompatibleSchemaError(
                f"{self.path}: this is a layout='cell' archive; open it with "
                f"cellstream.open(path) (a lazy per-cell store), not ShardedArchive."
            )
        self.schema_version = self.manifest["schema_version"]
        if self.schema_version == 4:
            self._matrices = self.manifest["matrices"]
            self.n_vars = int(self._matrices["x"]["n_vars"])
        else:
            self._matrices = None
            self.n_vars = self.manifest["n_vars"]
        self.n_obs = self.manifest["n_obs_total"]
        self.shape = (self.n_obs, self.n_vars)
        self.n_shards = self.manifest["n_shards"]
        self.row_offsets = self.manifest["row_offsets"]
        self._groups_cache: list[dict] | None = None
        self._label_map_cache: dict[str, dict] | None = None

    @property
    def obs(self):
        return self._packed.obs()

    @property
    def var(self):
        return self._packed.var()

    def _read_v2_source_kwargs(self):
        """Return the packed manifest + shard locator kwargs for reader_cpu."""
        return {"manifest": self.manifest, "shard_locator": self._packed.shard_location}

    @property
    def matrices(self) -> list[tuple[str, str]]:
        """List of (key, role). Single-matrix v2/v3 archives report [("x","x")]."""
        if self._matrices is None:
            return [("x", "x")]
        return [(k, v["role"]) for k, v in self._matrices.items()]

    def layer(self, name: str) -> ShardedArchiveView:
        """Scoped view whose X is layer ``name`` (parallels the Plan-2 .modality()).
        Uses the View ``matrix=`` support added in Task 8."""
        if self._matrices is None:
            raise ValueError("layer() is only valid on a multi-matrix (v4) archive.")
        if f"layer:{name}" not in self._matrices:
            raise KeyError(
                f"layer {name!r} not in archive layers "
                f"{[k[6:] for k in self._matrices if k.startswith('layer:')]}"
            )
        return ShardedArchiveView(self, slice(None), matrix=f"layer:{name}")

    @property
    def modalities(self) -> list[tuple[str, str]]:
        """List of (key, role) for modality:* families ([] if none)."""
        if self._matrices is None:
            return []
        return [
            (k, v["role"])
            for k, v in self._matrices.items()
            if str(v["role"]).startswith("modality:")
        ]

    def modality(self, name: str) -> ShardedArchiveView:
        """Scoped view whose X is modality ``name`` (parallels .layer())."""
        if self._matrices is None:
            raise ValueError("modality() is only valid on a multi-matrix (v4) archive.")
        key = f"modality:{name}"
        # Require a genuine modality ANCHOR: a modality-layer key
        # ("modality:<m>:layer:<n>") also starts with "modality:" and is a real
        # key, but its role is "layer:<n>" (!= key), so reject it here rather than
        # silently returning a layer sub-family view.
        if key not in self._matrices or self._matrices[key].get("role") != key:
            raise KeyError(
                f"modality {name!r} not in archive modalities "
                f"{[k[9:] for k in self._matrices if k.startswith('modality:')]}"
            )
        return ShardedArchiveView(self, slice(None), matrix=key)

    def _matrix_submanifest(self, key: str) -> dict:
        """A v2/v3-shaped manifest specialised to one family (shared partition
        + that matrix's per-family fields), for reader_cpu / resolve_plan."""
        md = self._matrices[key]
        sub = {
            "schema_version": 3 if self.manifest.get("group_by") is not None else 2,
            "n_shards": self.n_shards,
            "n_obs_total": self.n_obs,
            "n_vars": md["n_vars"],
            "row_offsets": self.row_offsets,
            "nnz_per_shard": md["nnz_per_shard"],
            "total_nnz": md["total_nnz"],
            "x_data_dtype_on_disk": md["x_data_dtype_on_disk"],
            "x_indices_dtype_on_disk": md["x_indices_dtype_on_disk"],
            "x_data_dtype_min": md["x_data_dtype_min"],
            "x_indices_dtype_min": md["x_indices_dtype_min"],
            "codec": md["codec"],
            "chunk_size_elements": md["chunk_size_elements"],
            "shard_filename_template": md["shard_filename_template"],
        }
        if self.manifest.get("group_by") is not None:
            sub["group_by"] = self.manifest["group_by"]
            sub["reference_shard"] = self.manifest.get("reference_shard")
        return sub

    def _matrix_shard_locator(self, key: str):
        return lambda idx: self._packed.shard_location(idx, matrix=key)

    def _matrix_var(self, key: str):
        """var DataFrame for a family: own-var families read their sidecar;
        layer:* resolve to their owner's var; the primary comes from the header."""
        md = self._matrices[key]
        if md["role"].startswith("layer:"):
            return self._matrix_var(md["owner"])
        if key == "x":
            return self.var  # header var mirrors var/x.parquet
        import io

        import pandas as pd

        return pd.read_parquet(io.BytesIO(self._packed.var_bytes(key)))

    def _matrix_meta(self, key: str):
        """The metadata AnnData (obsm/varm/uns) for a family, from its meta
        sidecar; None if the family has no sidecar (metadata is the global header)."""
        md = self._matrices[key]
        mf = md.get("meta_filename")
        if not mf:
            return None
        import io

        return ad.read_h5ad(io.BytesIO(self._packed.meta_bytes(key)))

    def _read_matrix_X(
        self,
        key: str,
        *,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        n_workers=16,
        blosc_nthreads=4,
        output_backing="heap",
    ):
        """Full-archive read of ONE family. Returns (X, bundle).

        The materialization knobs (``container``/``data_dtype``/``index_dtype``/
        ``allow_lossy``/``cast_back``) forward to ``resolve_plan`` exactly as the
        single-matrix path does. ``output_backing`` selects heap vs a /dev/shm
        bundle (threaded to ``read_v2_to_host``); on ``"shm"`` the returned X is a
        shm-backed view and ``bundle`` is a real bundle (both engines)."""
        from cellstream.v2.materialize import resolve_plan

        sub = self._matrix_submanifest(key)
        plan = resolve_plan(
            sub,
            container=container,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
            cast_back=cast_back,
            target="cpu",
        )
        bundle = None
        try:
            result = reader_cpu.read_v2_to_host(
                self.path,
                plan=plan,
                n_workers=n_workers,
                omp_nthreads=blosc_nthreads,
                output_backing=output_backing,
                _return_shm_bundle=True,
                manifest=sub,
                shard_locator=self._matrix_shard_locator(key),
            )
            bundle = result[-1]  # capture BEFORE _x_from_result can raise (shm-leak safety)
            X, _ = self._x_from_result(result, plan, n_obs_out=self.n_obs, n_vars=sub["n_vars"])
            return X, bundle
        except BaseException:
            if bundle is not None:
                bundle.close_and_unlink()
            raise

    def _reload_packed(self):
        from cellstream.packed.reader import PackedArchive

        self._packed = PackedArchive(self.path)
        self.manifest = self._packed.manifest

    def update_obs(self, new_obs, *, unsafe_no_lock=False):
        from cellstream.packed.writer import update_obs_packed

        update_obs_packed(self.path, new_obs, unsafe_no_lock=unsafe_no_lock)
        self._reload_packed()

    def update_var(self, new_var, *, unsafe_no_lock=False):
        from cellstream.packed.writer import update_var_packed

        update_var_packed(self.path, new_var, unsafe_no_lock=unsafe_no_lock)
        self._reload_packed()

    def to_anndata(
        self,
        *,
        layer=None,
        cast_back=True,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        n_workers=16,
        blosc_nthreads=4,
        output_backing="heap",
    ):
        """Read the whole archive into an AnnData.

        Parameters
        ----------
        layer : str | None, default None
            Multi-matrix (v4) archives only. ``None`` rebuilds the full AnnData
            (``X`` + every *primary* ``.layers[...]`` + ``.raw``; modality
            families are excluded — use ``.modality()`` / ``to_mudata()``). A layer name
            promotes that layer to ``X`` (it also remains in ``.layers``). Passing
            a non-``None`` value on a single-matrix (v2/v3) archive raises
            ``ValueError``.
        container : {"csr", "dense"}, default "csr"
            Output X container. ``"dense"`` materializes a numpy ndarray X directly
            in the decode workers (no CSR intermediate) — RAM ≈
            ``n_obs * n_vars * itemsize``. ``"csr"`` returns a scipy CSR.
        data_dtype : str | np.dtype | None, default None
            Target ``X.data`` dtype (e.g. ``"float64"``, ``"float16"``, ``"int32"``,
            ``"uint16"``). ``None`` uses the ``cast_back`` default.
        index_dtype : str | np.dtype | None, default None
            Target column-index dtype for CSR (``"uint16"``/``"uint32"``/``"int32"``/
            ``"int64"``/…); ignored for ``container="dense"``. ``None`` uses the
            policy default (matching int32/int64 by nnz).
        allow_lossy : bool, default False
            Permit narrowing casts (e.g. float32→float16, or float→int on
            non-integral values). Default is fail-loud: a lossy cast raises
            :class:`~cellstream.errors.LossyCastError`.
        cast_back : bool, default True
            Back-compat alias. ``True`` → float32 data (from uint16) + matching
            int32/int64 indices; ``False`` → native on-disk dtypes. An explicit
            ``data_dtype``/``index_dtype`` overrides the corresponding default.
        output_backing : {"heap", "shm"}, default "heap"
            Where the decoded X is allocated on the Rust read path. ``"heap"``
            (default) uses plain numpy arrays (no ``/dev/shm``). ``"shm"`` uses a
            ``/dev/shm``-backed bundle for cross-process / GPU-staging handoff.
            ``"pinned"`` is reserved (GPU H2D, #40). Ignored on the pure-Python
            fallback (which always uses ``/dev/shm``). Multi-matrix (v4) archives
            only support ``"heap"`` for now (Plan 2 adds shm-backed multi reads).

        Notes
        -----
        On a multi-matrix (v4) archive, the
        per-family materialization knobs (``container``/``data_dtype``/
        ``index_dtype``/``allow_lossy``/``cast_back``) ARE honored on the heap
        path — each primary-group family is read with the same plan. Only a
        non-``"heap"`` ``output_backing`` raises ``NotImplementedError`` (shm-backed
        multi-matrix reads are Plan 2 chunk D).
        """
        if self._matrices is None:
            if layer is not None:
                raise ValueError(
                    "layer= is only valid on a multi-matrix (v4) archive; this archive "
                    "has a single matrix."
                )
            return self._to_anndata_single(
                cast_back=cast_back,
                container=container,
                data_dtype=data_dtype,
                index_dtype=index_dtype,
                allow_lossy=allow_lossy,
                n_workers=n_workers,
                blosc_nthreads=blosc_nthreads,
                output_backing=output_backing,
            )
        # --- v4 rebuild: read ONLY the primary family group (x + x-owned layers
        # + raw), skipping modality families (filtered via is_primary_family),
        # then reassemble X + .layers + .raw from the specs. ---
        from cellstream import matrices as _mx

        # output_backing validation is delegated to read_v2_to_host (accepts
        # "heap"/"shm"; "pinned" -> NotImplementedError). The shm-backed v4 rebuild
        # (Plan 2 chunk D) keeps each family's decode in its /dev/shm bundle and
        # pins them all to the returned AnnData (two-level pin via
        # shm.attach_buffer_cleanup_multi) instead of copying every family to heap.
        shm_backed = output_backing != "heap"
        if layer is not None and f"layer:{layer}" not in self._matrices:
            raise KeyError(
                f"layer {layer!r} not in archive layers "
                f"{[k for k in self._matrices if k.startswith('layer:')]}"
            )
        header_adata = self._packed.header_adata()
        specs, var_by_key = [], {}
        # Read ONLY the primary family group (x, raw, layers owned by x): a
        # modality:* family (or a layer owned by one) is neither part of the
        # returned AnnData nor cheap to skip after the fact, so it must never be
        # read here. reassemble_anndata (pre-existing) filters layers by
        # role.startswith("layer:") with NO owner check, so passing it a
        # modality-owned layer spec would silently leak it into out.layers.
        primary_keys = [
            k
            for k, md in self._matrices.items()
            if _mx.is_primary_family(md["role"], md.get("owner"))
        ]
        # (role, bundle) per family read into shm; pinned to the AnnData after
        # reassembly. Any bundle in this list is closed if a later step raises.
        role_bundles = []
        try:
            for key in primary_keys:
                md = self._matrices[key]
                X, bundle = self._read_matrix_X(
                    key,
                    container=container,
                    data_dtype=data_dtype,
                    index_dtype=index_dtype,
                    allow_lossy=allow_lossy,
                    cast_back=cast_back,
                    n_workers=n_workers,
                    blosc_nthreads=blosc_nthreads,
                    output_backing=output_backing,
                )
                # Track every allocated bundle IMMEDIATELY so a failure before it is
                # pinned (shm) or released (heap) still hits the except cleanup
                # below. The pure-Python heap copy can raise (e.g. MemoryError on a
                # large family), which would otherwise leak an untracked bundle.
                if bundle is not None:
                    role_bundles.append((md["role"], bundle))
                if not shm_backed and bundle is not None:
                    # heap: the Rust path returns owned arrays (bundle None); the
                    # pure-Python fallback returns arrays that are VIEWS into the
                    # bundle's SharedMemory/mmap. reassemble_anndata keeps each
                    # family's X as out.X / out.layers[...] / out.raw.X, so detach
                    # by copying to owned heap, then release the bundle immediately
                    # (accessing a view after close_and_unlink is a UAF segfault).
                    # X.copy() runs while the bundle is still tracked, so a copy
                    # failure is cleaned up by the except block.
                    X = X.copy()
                    role_bundles.pop()  # this family's bundle (appended just above)
                    bundle.close_and_unlink()
                specs.append(
                    _mx.MatrixSpec(
                        key=key,
                        role=md["role"],
                        owner=md.get("owner"),
                        n_vars=md["n_vars"],
                        X=X,
                        var=None,
                    )
                )
                if not md["role"].startswith("layer:"):
                    var_by_key[key] = self._matrix_var(key)
            for s in specs:  # layers resolve to owner var
                if s.role.startswith("layer:"):
                    var_by_key[s.key] = var_by_key[s.owner]
            # obsm/varm/uns: prefer x's own meta sidecar (set when the source was a
            # MuData whose primary modality carried its own obsm/varm/uns); fall
            # back to the shared header when x has no sidecar (non-MuData case).
            xmeta = self._matrix_meta("x")
            _src = xmeta if xmeta is not None else header_adata
            out = _mx.reassemble_anndata(
                specs,
                obs=header_adata.obs,
                var_by_key=var_by_key,
                uns=_src.uns,
                obsm=_src.obsm,
                varm=_src.varm,
                obsp=_src.obsp,
                varp=_src.varp,
            )
        except BaseException:
            for _role, _b in role_bundles:
                _b.close_and_unlink()
            raise
        if shm_backed and role_bundles:
            from cellstream import shm as _shm

            # Map each family to its stored array in `out` (which keeps each X by
            # reference), then two-level pin the bundles to the AnnData.
            pairs = []
            for role, bundle in role_bundles:
                if role == "x":
                    arr = out.X
                elif role == "raw":
                    arr = out.raw.X if out.raw is not None else None
                elif role.startswith("layer:"):
                    arr = out.layers.get(role[len("layer:") :])
                else:
                    arr = None
                if arr is not None:
                    pairs.append((arr, bundle))
                else:
                    bundle.close_and_unlink()  # unreachable family -> don't leak
            _shm.attach_buffer_cleanup_multi(out, pairs)
        if layer is not None:  # promote a layer to X (keep it in .layers too)
            out.X = out.layers[layer]
        return out

    def _to_anndata_single(
        self,
        *,
        cast_back=True,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        n_workers=16,
        blosc_nthreads=4,
        output_backing="heap",
    ):
        """Single-matrix (v2/v3) whole-archive read. See ``to_anndata`` for the
        parameter docs; this is the pre-multi-matrix implementation, unchanged."""
        from cellstream.v2.materialize import resolve_plan

        plan = resolve_plan(
            self.manifest,
            container=container,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
            cast_back=cast_back,
            target="cpu",
        )
        # bundle initialised before the try so that an exception (incl. Ctrl+C)
        # during read_v2_to_host itself doesn't leak — close_and_unlink is
        # idempotent so the `bundle is not None` guard is the only thing
        # protecting against the read_v2_to_host-raises case.
        bundle = None
        try:
            result = reader_cpu.read_v2_to_host(
                self.path,
                plan=plan,
                n_workers=n_workers,
                omp_nthreads=blosc_nthreads,
                output_backing=output_backing,
                _return_shm_bundle=True,
                **self._read_v2_source_kwargs(),
            )
            X, bundle = self._x_from_result(result, plan, n_obs_out=self.n_obs)
            return self._build_anndata_v2(X, bundle, n_obs_out=self.n_obs, obs_selector=None)
        except BaseException:
            if bundle is not None:
                bundle.close_and_unlink()
            raise

    def to_mudata(
        self,
        *,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        n_workers=16,
        blosc_nthreads=4,
        output_backing="heap",
    ):
        """Reconstruct a MuData across modality:* families (requires 'mudata').

        The primary family becomes ``mod[<x.modality_name>]`` via ``to_anndata()``
        (X + x-owned layers + ``.raw`` + the primary's own obsm/varm/uns from its
        meta sidecar — ``to_anndata()`` already restricts itself to the primary
        family group, so it is reused as-is rather than re-implemented here);
        each ``modality:*`` family becomes ``mod[<name>]``. MuData global
        obs/obsm/uns come from the shared header. Raises if the archive has no
        modalities (use ``to_anndata()``).

        The materialization knobs (``container``/``data_dtype``/``index_dtype``/
        ``allow_lossy``/``cast_back``) mirror ``to_anndata()`` and apply to EVERY
        modality (one value for all). Default is float32 CSR, unchanged.
        ``output_backing="shm"`` (default ``"heap"``) keeps every modality's
        families in /dev/shm bundles pinned to their AnnData; ``mdata.
        close_shared_memory()`` then fans out to all modalities. A lossy request
        without ``allow_lossy=True`` raises
        :class:`~cellstream.errors.LossyCastError`."""
        from cellstream import matrices as _mx
        from cellstream import shm as _shm
        from cellstream.mudata_adapter import build_mudata

        if self._matrices is None or not self.modalities:
            raise ValueError(
                "this archive has no modalities; use to_anndata() (or to_anndata(layer=))."
            )
        shm_backed = output_backing != "heap"
        header_adata = self._packed.header_adata()

        def _family_anndata(anchor_key, modality_name):
            # (tag, bundle) per family read into shm ("__anchor__" for X, else the
            # layer name). Tracked immediately so a copy failure still hits the
            # except cleanup; closed there before this modality is attached.
            fam_bundles = []
            try:
                X, bundle = self._read_matrix_X(
                    anchor_key,
                    container=container,
                    data_dtype=data_dtype,
                    index_dtype=index_dtype,
                    allow_lossy=allow_lossy,
                    cast_back=cast_back,
                    n_workers=n_workers,
                    blosc_nthreads=blosc_nthreads,
                    output_backing=output_backing,
                )
                if bundle is not None:
                    fam_bundles.append(("__anchor__", bundle))
                if not shm_backed and bundle is not None:
                    X = X.copy()
                    fam_bundles.pop()  # anchor bundle (appended just above)
                    bundle.close_and_unlink()
                # this modality's layers
                layer_items = []
                for k, md in self._matrices.items():
                    if md["role"].startswith("layer:") and md.get("owner") == anchor_key:
                        lname = md["role"][len("layer:") :]
                        lx, lb = self._read_matrix_X(
                            k,
                            container=container,
                            data_dtype=data_dtype,
                            index_dtype=index_dtype,
                            allow_lossy=allow_lossy,
                            cast_back=cast_back,
                            n_workers=n_workers,
                            blosc_nthreads=blosc_nthreads,
                            output_backing=output_backing,
                        )
                        if lb is not None:
                            fam_bundles.append((lname, lb))
                        if not shm_backed and lb is not None:
                            lx = lx.copy()
                            fam_bundles.pop()  # this layer's bundle (appended just above)
                            lb.close_and_unlink()
                        layer_items.append((lname, lx))
            except BaseException:
                for _tag, _b in fam_bundles:
                    _b.close_and_unlink()
                raise
            meta = self._matrix_meta(anchor_key)
            _src = meta if meta is not None else header_adata
            # Per-modality obs: use the meta sidecar's obs when it carries columns
            # (Bucket 3); fall back to the global header obs for old archives whose
            # meta sidecars are index-only (pre-Bucket-3).
            mod_obs = (
                meta.obs if (meta is not None and len(meta.obs.columns) > 0) else header_adata.obs
            )
            madata = _mx.reassemble_modality_anndata(
                anchor_X=X,
                var=self._matrix_var(anchor_key),
                obs=mod_obs,
                layer_items=layer_items,
                obsm=_src.obsm,
                varm=_src.varm,
                uns=_src.uns,
                obsp=_src.obsp,
                varp=_src.varp,
            )
            if shm_backed and fam_bundles:
                pairs = [
                    (madata.X if tag == "__anchor__" else madata.layers[tag], b)
                    for tag, b in fam_bundles
                ]
                _shm.attach_buffer_cleanup_multi(madata, pairs)
            return madata

        mod = {}
        built = []  # attached adatas to close on a mid-build failure (shm)
        try:
            primary_name = self._matrices["x"]["modality_name"]
            # Primary via to_anndata() (not _family_anndata): to_anndata() already
            # rebuilds X + every x-owned layer + `.raw` (the "raw" family, owned by
            # no one, is invisible to _family_anndata's owner-scoped layer scan and
            # has no analogue there) + the primary's own obsm/varm/uns via its meta
            # sidecar (to_anndata() prefers x's own sidecar over the shared header —
            # see its docstring). _family_anndata is reserved for modality:* families
            # below, which never carry a `.raw` of their own (write-side pre-flight).
            mod[primary_name] = self.to_anndata(
                container=container,
                data_dtype=data_dtype,
                index_dtype=index_dtype,
                allow_lossy=allow_lossy,
                cast_back=cast_back,
                n_workers=n_workers,
                blosc_nthreads=blosc_nthreads,
                output_backing=output_backing,
            )
            built.append(mod[primary_name])
            # to_anndata() returns the primary with the GLOBAL header obs; re-apply
            # the primary modality's OWN obs from its meta sidecar when present
            # (Bucket 3; old index-only sidecars fall back to the global obs already
            # in place). Reassigning .obs does not touch X, so shm pins survive.
            _xmeta = self._matrix_meta("x")
            if _xmeta is not None and len(_xmeta.obs.columns) > 0:
                mod[primary_name].obs = _xmeta.obs.copy()
            for key, md in self._matrices.items():
                if md["role"].startswith("modality:"):
                    madata = _family_anndata(key, md["modality_name"])
                    mod[md["modality_name"]] = madata
                    built.append(madata)
        except BaseException:
            if shm_backed:
                for a in built:
                    if hasattr(a, "close_shared_memory"):
                        a.close_shared_memory()
            raise
        mdata = build_mudata(
            mod,
            obs=header_adata.obs,
            obsm=dict(header_adata.obsm),
            uns=dict(header_adata.uns),
            obsp=dict(header_adata.obsp),
            varp=dict(header_adata.varp),
        )
        if shm_backed:
            _shm.attach_mudata_close(mdata, list(mod.values()))
        return mdata

    def to_torch(
        self,
        *,
        cast_back=True,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        n_streams=4,
        n_workers=16,
    ):
        """Read into a torch tensor on the device via DLPack.

        container="csr" -> torch.sparse_csr_tensor; container="dense" -> a dense
        torch tensor. PyTorch lacks a uint16 data dtype, so the narrow path
        (cast_back=False / data_dtype="uint16") is rejected — use to_cupy_anndata
        for that.
        """
        if (not cast_back and data_dtype is None) or (
            data_dtype is not None and np.dtype(data_dtype) == np.dtype(np.uint16)
        ):
            raise ValueError(
                "to_torch() does not support a uint16 data dtype (PyTorch lacks it). "
                "Use the default float path (cast_back=True) or pass a float data_dtype "
                "(e.g. 'float32'/'float16'); for the narrow-dtype GPU path use "
                "to_cupy_anndata(cast_back=False)."
            )
        gpu_adata = self.to_cupy_anndata(
            cast_back=cast_back,
            container=container,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
            n_streams=n_streams,
            n_workers=n_workers,
        )
        if container == "dense":
            import torch

            # Guaranteed zero-copy cupy -> torch hand-off (torch.as_tensor is not
            # zero-copy for cupy in all torch versions).
            return torch.from_dlpack(gpu_adata.X.toDlpack())
        return _cupy_csr_to_torch(gpu_adata.X, total_nnz=self.manifest["total_nnz"])

    def to_cupy_anndata(
        self,
        *,
        cast_back=True,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        n_streams=4,
        n_workers=16,
        _gpu_threshold_bytes=5_000_000_000,
    ):
        """Read into a cupy X on the device.

        Hybrid dispatch by projected output size:
        - small archives go via CPU decode (honoring the materialization knobs) +
          cupy upload (dense -> dense cupy ndarray; csr -> cupyx CSR).
        - large v2 archives decode on the GPU (v2.reader_gpu.read_v2_to_device;
          nvCOMP-RAW zstd + cupy byte-filter inverse), CSR only.

        A device CSR requires float32/float64 data (cupyx/scipy sparse do not
        support narrow uint16 / float16), so ``container='csr'`` with
        ``cast_back=False`` or a narrow/float16 ``data_dtype`` raises; use
        ``container='dense'`` for those.
        """
        # Resolve the materialization plan once — it drives both the float-CSR guard
        # and the size threshold (Codex review: size from the resolved dtypes, not a
        # cast_back heuristic; a dense read is sized as n_obs*n_vars, not CSR bytes).
        from cellstream.v2.materialize import resolve_plan
        from cellstream.v2.reader_gpu import require_csr_float

        plan = resolve_plan(
            self.manifest,
            container=container,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
            cast_back=cast_back,
            target="cupy",
        )
        if container == "csr":
            # A device CSR requires float32/float64 data (cupyx/scipy constraint);
            # fail early + clearly for both the small (upload) and large (decode) paths.
            require_csr_float(plan.data_dtype)
            bytes_output = (
                self.manifest["total_nnz"] * (plan.data_dtype.itemsize + plan.index_dtype.itemsize)
                + (self.n_obs + 1) * plan.indptr_dtype.itemsize
            )
        else:  # dense
            bytes_output = self.n_obs * self.n_vars * plan.data_dtype.itemsize
        if bytes_output < _gpu_threshold_bytes:
            adata = self.to_anndata(
                cast_back=cast_back,
                container=container,
                data_dtype=data_dtype,
                index_dtype=index_dtype,
                allow_lossy=allow_lossy,
                n_workers=n_workers,
            )
            return _upload_anndata_to_device(adata)
        # Large v2 archive: on-GPU decode (nvCOMP-RAW zstd + cupy byte-filter
        # inverse), one shard at a time, preflight-gated (#40).
        if container != "csr":
            raise NotImplementedError(
                "GPU device decode of a large archive supports container='csr' only "
                "(dense would allocate n_obs*n_vars and usually OOM). Use container='csr', "
                "a subset, or the CPU reader (sh.to_anndata())."
            )
        from cellstream.v2.reader_gpu import read_v2_to_device

        gpu_csr = read_v2_to_device(
            self.path,
            cast_back=cast_back,
            n_streams=n_streams,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
        )
        return _make_anndata_from_device_csr(self, gpu_csr, self.n_obs, obs_selector=None)

    # ------------------------------------------------------------------
    # Target-aware reader API (Phase C)
    # ------------------------------------------------------------------

    def _require_grouped(self) -> None:
        """Raise ValueError if this archive was not written with group_by.

        This guard covers all three grouped-read methods so callers on an archive
        written without group_by get a clear hint to re-write with group_by=...

        Deliberately says nothing about schema_version: the ungrouped case is schema 2
        for a single-matrix source but schema **4** for a multi-matrix one (build_v4_manifest
        omits group_by rather than lowering the version), and this guard fires on both.
        Verified by writing a layered archive with no group_by: schema_version 4,
        group_by None, guard fires.
        """
        if self.manifest.get("group_by") is None:
            raise ValueError(
                "This archive was not written with a group_by key. "
                "To use read_reference(), iter_group_shards(), or read_group(), "
                "re-write the archive with write_sharded(..., group_by=<column>)."
            )

    def _load_groups(self) -> list[dict]:
        """Lazily load groups.json and cache the result on this archive."""
        if self._groups_cache is None:
            import json

            if not self._packed.has_member("groups.json"):
                group_col = self.manifest.get("group_by", "<unknown>")
                raise ValueError(
                    f"groups.json is missing from packed archive at {self.path!r} even "
                    f"though the manifest declares group_by={group_col!r}. "
                    f"The archive may be corrupt or written by an older version. "
                    f"Re-write the archive with write_sharded(..., group_by={group_col!r})."
                )
            data = json.loads(self._packed.member_bytes("groups.json").decode("utf-8"))
            if not isinstance(data, list):
                raise ValueError(
                    f"groups.json in {self.path!r} must contain a JSON array, "
                    f"got {type(data).__name__!r}."
                )
            self._groups_cache = data
            # Validate records once, on first load (ultrareview L1): a corrupt or
            # hand-edited groups.json must fail loud here rather than producing a
            # KeyError / IndexError or a silently out-of-bounds row slice in
            # iter_group_shards / read_group.
            validate_group_records(self._groups_cache, n_shards=self.n_shards, n_obs=self.n_obs)
        return self._groups_cache

    def read_reference(self, **kw) -> ad.AnnData | None:
        """Read the reference shard as an AnnData.

        Returns ``None`` if the archive was written without a reference.
        Uses the manifest ``reference_shard`` field only — does not load
        groups.json.

        Raises
        ------
        ValueError
            If the archive was not written with ``group_by``.
        """
        self._require_grouped()
        ref_shard = self.manifest.get("reference_shard")
        if ref_shard is None:
            return None
        row_offsets = self.row_offsets
        start = row_offsets[ref_shard]
        stop = row_offsets[ref_shard + 1]
        return self[start:stop].to_anndata(**kw)

    def iter_group_shards(
        self,
        prefetch: int = 0,
        n_workers: int = 16,
        *,
        cast_back: bool = True,
        container: str = "csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy: bool = False,
        blosc_nthreads: int = 4,
        pinned: bool = False,
        matrix: str = "x",
    ) -> Iterator[GroupShard]:
        """Iterate over non-reference shards as :class:`GroupShard` objects.

        With ``prefetch=0`` (default) this is the original lazy iterator: it
        yields one ``GroupShard`` per non-reference shard in shard order with
        I/O deferred until ``.to_anndata()`` is called on the returned object.

        With ``prefetch>=1`` it decodes ahead: a background thread decodes
        batches of ``n_workers`` consecutive shards in parallel and the yielded
        ``GroupShard`` carries its already-decoded AnnData (``.to_anndata()``
        returns it instantly). The decode knobs (``cast_back`` … ``blosc_nthreads``)
        apply to that pre-decode and are only valid when ``prefetch>=1``.

        With ``pinned=True`` (``prefetch>=1``, CSR only) the producer page-locks
        each decoded batch's CSR arrays so ``GroupShard.dlpack()`` yields
        async-H2D-capable buffers for a GPU consumer (#160). cupy is optional —
        absent or on registration failure it falls back to pageable memory with
        a one-time warning. ``pinned=False`` (default) is byte-identical to today.

        Resident host RAM is roughly ``(prefetch + n_workers)`` shards: both
        throughput (``n_workers``-way parallel decode) and footprint scale with
        ``n_workers``. Set ``prefetch >= n_workers`` for full decode/compute
        overlap. The reference shard (if any) is excluded.

        ``matrix`` (default ``"x"``) selects one family of a multi-matrix (v4)
        archive — e.g. ``matrix="layer:counts"``. The partition is shared across
        families, so shard *i* is the same cells regardless of ``matrix``:
        zipping two ``iter_group_shards(matrix=...)`` calls by ``shard_index``
        aligns row-for-row. On a single-matrix (v2/v3) archive, ``matrix``
        must stay ``"x"``.

        Yields
        ------
        GroupShard
            Holds the shard's global row range and a mapping of label →
            shard-local slice. Each yielded ``GroupShard`` exposes
            ``.to_anndata()`` and ``.x()`` (the raw scipy CSR / dense matrix
            without an AnnData wrapper); the per-group row slices are on
            ``.groups``.

        Raises
        ------
        ValueError
            If the archive was not written with ``group_by``; if ``prefetch`` is
            negative; or if a non-default decode knob is passed with
            ``prefetch=0``.
        """
        if prefetch < 0:
            raise ValueError(f"prefetch must be >= 0, got {prefetch}")
        if pinned and container != "csr":
            raise ValueError(
                f"pinned=True is CSR-only; got container={container!r}. Use "
                "container='csr' (the default) for the pinned DLPack handoff."
            )
        self._require_grouped()
        all_records = self._load_groups()
        row_offsets = self.row_offsets

        # Gather non-reference records grouped by shard, preserving shard order.
        by_shard: dict[int, list[dict]] = defaultdict(list)
        for rec in all_records:
            if rec.get("role") != "reference":
                by_shard[rec["shard"]].append(rec)

        plans = []
        for shard_idx in sorted(by_shard.keys()):
            off = row_offsets[shard_idx]
            local_groups = {
                rec["label"]: slice(rec["row_start"] - off, rec["row_stop"] - off)
                for rec in by_shard[shard_idx]
            }
            plans.append((shard_idx, off, row_offsets[shard_idx + 1], local_groups))

        if prefetch == 0:
            nondefault = [
                name
                for name, val, default in (
                    ("n_workers", n_workers, 16),
                    ("cast_back", cast_back, True),
                    ("container", container, "csr"),
                    ("data_dtype", data_dtype, None),
                    ("index_dtype", index_dtype, None),
                    ("allow_lossy", allow_lossy, False),
                    ("blosc_nthreads", blosc_nthreads, 4),
                    ("pinned", pinned, False),
                )
                if val != default
            ]
            if nondefault:
                raise ValueError(
                    f"decode knobs {nondefault} only take effect with prefetch>=1; "
                    "with prefetch=0 (lazy mode) pass them to the per-shard "
                    "GroupShard.to_anndata() call instead."
                )
            for shard_idx, gstart, gstop, local_groups in plans:
                yield GroupShard(
                    archive=self,
                    shard_index=shard_idx,
                    global_start=gstart,
                    global_stop=gstop,
                    groups=local_groups,
                    matrix=matrix,
                )
            return

        from cellstream.prefetch import _ShardPlan, iter_prefetched

        shard_plans = [
            _ShardPlan(shard_idx, gstart, gstop, local_groups)
            for shard_idx, gstart, gstop, local_groups in plans
        ]
        decode_kwargs = dict(
            cast_back=cast_back,
            container=container,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
            blosc_nthreads=blosc_nthreads,
        )
        # Only add matrix to decode_kwargs when non-default, so an existing
        # single-matrix caller (matrix="x", the default) threads byte-for-byte
        # the same decode_kwargs dict through iter_prefetched as before Task 8.
        if matrix != "x":
            decode_kwargs["matrix"] = matrix
        for plan, decoded in iter_prefetched(
            self,
            shard_plans,
            prefetch=prefetch,
            n_workers=n_workers,
            decode_kwargs=decode_kwargs,
            pinned=pinned,
        ):
            yield GroupShard(
                archive=self,
                shard_index=plan.shard_index,
                global_start=plan.global_start,
                global_stop=plan.global_stop,
                groups=plan.local_groups,
                _decoded=decoded,
                matrix=matrix,
            )

    def read_group(self, label: str, **kw) -> ad.AnnData:
        """Read all cells belonging to *label* as an AnnData.

        Looks up *label* in groups.json and reads exactly the covering
        rows.  Raises :class:`KeyError` (with near-match hints) if the
        label is not present.

        Raises
        ------
        KeyError
            If *label* is not found in the grouped archive's group index.
        ValueError
            If the archive was not written with ``group_by``.
        """
        self._require_grouped()

        if self._label_map_cache is None:
            all_records = self._load_groups()
            # Build a label→record map once and cache it for future calls.
            # Include ALL labels (group and reference) so that read_group on
            # the reference label returns its cells.  Non-reference records
            # win on the rare label collision (e.g. a gene also named "NTC").
            lm: dict[str, dict] = {}
            for rec in all_records:
                if rec.get("role") == "reference":
                    lm.setdefault(rec["label"], rec)
                else:
                    lm[rec["label"]] = rec
            self._label_map_cache = lm

        label_map = self._label_map_cache

        if label not in label_map:
            # Use difflib for bounded, sensible near-match suggestions.
            # get_close_matches handles empty/short labels gracefully
            # (returns [] instead of the full key list like substring matching).
            near = difflib.get_close_matches(label, list(label_map), n=5)
            hint = f". Near matches: {near!r}" if near else ""
            raise KeyError(f"Group label {label!r} not found in archive{hint}.")

        rec = label_map[label]
        return self[rec["row_start"] : rec["row_stop"]].to_anndata(**kw)

    def __getitem__(self, key):
        return ShardedArchiveView(self, key)

    def _x_from_result(self, result, plan, *, n_obs_out, n_vars=None):
        """Turn read_v2_to_host's (container-dependent) return into ``(X, bundle)``.

        - dense container: result is ``(dense_2d, bundle)`` -> X is the ndarray.
        - csr container: result is ``(data, indices, indptr, bundle)`` -> assemble
          a csr_matrix from the three shm-backed arrays.

        ``n_vars`` defaults to ``self.n_vars`` (the single/primary matrix); a v4
        family read (``_read_matrix_X``) passes that family's own ``n_vars`` so a
        layer/raw with a different column count assembles correctly.
        """
        if n_vars is None:
            n_vars = self.n_vars
        if plan.container == "dense":
            dense, bundle = result
            if bundle is None:
                # output_backing="heap": plain heap ndarray, no shm to keep mapped.
                return dense, None
            # Re-wrap the SAME shm segment as the BUFFER-OWNING _ShmDenseArray
            # carrying the bundle pin (issue #182). numpy preserves the buffer-owner
            # in the .base chain of every view (slicing, np.asarray, AnnData
            # wrapping), so the backing SharedMemory survives a temporary AnnData's
            # GC. Pinning on an OUTER .view() is stripped by np.asarray -> UAF segfault.
            # Forward dense.strides so the wrapper mirrors the decoded array's layout.
            return _shm.dense_array_for_bundle(
                bundle, dense.shape, dense.dtype, strides=dense.strides
            ), bundle
        data, indices, indptr, bundle = result
        if indices.size:
            # Unsigned index dtypes can't be negative; skip the min() scan (mirrors
            # reader_cpu._validate_decoded_shard). Catches an out-of-range column
            # before scipy mis-assembles / segfaults (ultrareview M4, engine-agnostic).
            if indices.dtype.kind != "u":
                cmin = int(indices.min())
                if cmin < 0:
                    raise ValueError(f"column index out of range [0, {n_vars}) (min={cmin})")
            cmax = int(indices.max())
            if cmax >= n_vars:
                raise ValueError(f"column index out of range [0, {n_vars}) (max={cmax})")
        # CSR-from-shm assembly consistency (ultrareview L4): a corrupt/short shm
        # buffer would otherwise let scipy mis-assemble or segfault on a later op.
        # Checked unconditionally (incl. the empty case) before handing to scipy.
        if len(indptr) != n_obs_out + 1:
            raise ValueError(
                f"corrupt CSR: len(indptr)={len(indptr)} != n_obs_out+1={n_obs_out + 1}"
            )
        if int(indptr[0]) != 0:
            raise ValueError(f"corrupt CSR: indptr[0]={int(indptr[0])} must be 0")
        if not (int(indptr[-1]) == len(data) == len(indices)):
            raise ValueError(
                f"corrupt CSR: indptr[-1]={int(indptr[-1])} must equal "
                f"len(data)={len(data)} == len(indices)={len(indices)}"
            )
        # Non-decreasing indptr (mirrors reader_cpu._validate_indptr from M1): a
        # decreasing pair => negative per-row nnz, which segfaults scipy/C++ ops.
        # arr[1:] >= arr[:-1] avoids a np.diff temporary (repo perf guideline).
        if not np.all(indptr[1:] >= indptr[:-1]):
            raise ValueError("corrupt CSR: indptr must be non-decreasing")
        X = sp.csr_matrix((n_obs_out, n_vars), dtype=data.dtype)
        X.data = data
        X.indices = indices
        X.indptr = indptr
        return X, bundle

    def _build_anndata_v2(self, X, bundle, *, n_obs_out, obs_selector=None, var_override=None):
        """Wrap an already-formed X (CSR or dense ndarray) into an AnnData.
        Attaches the shm bundle so it survives as long as the AnnData (and its X
        view) is alive.

        obs_selector: None for a full read (take all obs rows); otherwise a slice,
        bool mask, or int array selecting the rows of obs / obsm that match X. Must
        be the SAME selector used to fetch X so obs and X stay aligned.

        var_override: None uses the header's var (unchanged pre-Task-8 behavior);
        otherwise (a family read on a multi-matrix archive whose var differs from
        the primary matrix's) this var DataFrame is used in its place.
        """
        header_adata = self._packed.header_adata()
        if obs_selector is None:
            obs_sel = header_adata.obs
            obsm_sel = header_adata.obsm
        else:
            # Apply the same selector to obs and every obsm key. A slice, bool
            # mask, or int array all index obs.iloc / the obsm arrays identically.
            obs_sel = header_adata.obs.iloc[obs_selector]
            obsm_sel = {k: v[obs_selector] for k, v in header_adata.obsm.items()}
        out = ad.AnnData(
            X=X,
            obs=obs_sel.copy(),
            var=var_override if var_override is not None else header_adata.var,
            uns=header_adata.uns,
            obsm=obsm_sel,
            varm=header_adata.varm,
            # obsp/varp (Bucket 3) are whole-dataset graphs in header partition
            # order — attach only for a full read; a subset view (obs_selector
            # set) does not carry them (an induced subgraph is out of scope).
            obsp=dict(header_adata.obsp) if obs_selector is None else {},
            varp=dict(header_adata.varp) if obs_selector is None else {},
        )
        if bundle is not None:
            _shm.attach_shm_cleanup(out, bundle)
        return out


class ShardedArchiveView:
    """A subset view; defers I/O until .to_anndata()."""

    def __init__(self, parent, key, *, matrix="x"):
        self.parent = parent
        self.key = key
        # Family selector (Task 8): the default family this view reads when
        # .x()/.to_anndata() are called without an explicit matrix=/layer=.
        # "x" (the primary matrix) preserves single-matrix (v2/v3) behavior.
        self.matrix = matrix

    def _resolve_matrix(self, matrix, layer):
        """Resolve matrix=/layer= into a single family key.

        At most one of matrix=/layer= may be set (raise ValueError otherwise).
        layer=<n> is sugar for matrix=f"layer:{n}"; if neither is given, falls
        back to this view's own default family (self.matrix).

        A view scoped to a modality anchor (``self.matrix`` starts with
        ``"modality:"``, e.g. produced by ``ShardedArchive.modality(name)``)
        resolves layer= RELATIVE to that modality: ``layer="tfidf"`` on
        ``archive.modality("atac")`` means "atac's own tfidf layer"
        (``modality:atac:layer:tfidf``), matching that family's manifest key
        (``matrices.decompose``'s modality-layer keying), not the primary's
        global ``layer:tfidf``.
        """
        if layer is not None:
            if matrix is not None:
                raise ValueError("pass at most one of matrix= / layer=.")
            base = self.matrix
            if isinstance(base, str) and base.startswith("modality:"):
                key = f"{base}:layer:{layer}"
            else:
                key = f"layer:{layer}"
        else:
            key = matrix if matrix is not None else self.matrix

        if self.parent._matrices is None:
            if key != "x":
                raise ValueError(
                    f"selecting a non-primary matrix (matrix={key!r}) is only supported "
                    f"on multi-matrix (v4) archives; this archive has a single matrix."
                )
        elif key not in self.parent._matrices:
            # Validate here so a bogus matrix=/layer= raises a clear error instead
            # of a raw KeyError bubbling from _matrix_submanifest (self._matrices[key]).
            raise KeyError(
                f"matrix key {key!r} not found in archive; available matrices: "
                f"{list(self.parent._matrices)}"
            )
        return key

    def to_anndata(
        self,
        *,
        matrix=None,
        layer=None,
        cast_back=True,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        n_workers=16,
        blosc_nthreads=4,
        output_backing="heap",
    ):
        """Read this view into an AnnData.

        ``matrix``/``layer`` (Task 8) select a single family of a multi-matrix
        (v4) archive — at most one of them may be set; ``layer=<n>`` is sugar
        for ``matrix=f"layer:{n}"``. Neither is v1-supported. ``None``/``None``
        falls back to this view's own default family (``self.matrix``, "x" for
        a view produced by ``archive[...]``/``archive.layer(...)``).
        """
        key = self._resolve_matrix(matrix, layer)
        X, bundle, n_obs_out, obs_selector = self._read_x_v2(
            cast_back=cast_back,
            container=container,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
            n_workers=n_workers,
            blosc_nthreads=blosc_nthreads,
            output_backing=output_backing,
            matrix=key,
        )
        # _read_x_v2 already cleaned up the bundle on a read failure; guard the
        # AnnData build the same way so any failure here can't leak the bundle.
        # var_override lookup is INSIDE this try so a _matrix_var failure (e.g. a
        # corrupt per-family var sidecar) still reaches the except and closes the
        # bundle, rather than leaking it (Gemini-flagged bundle-cleanup gap).
        try:
            var_override = (
                self.parent._matrix_var(key) if self.parent._matrices is not None else None
            )
            return self.parent._build_anndata_v2(
                X, bundle, n_obs_out=n_obs_out, obs_selector=obs_selector, var_override=var_override
            )
        except BaseException:
            if bundle is not None:
                bundle.close_and_unlink()
            raise

    def _read_x_v2(
        self,
        *,
        cast_back,
        container,
        data_dtype,
        index_dtype,
        allow_lossy,
        n_workers,
        blosc_nthreads,
        output_backing,
        matrix="x",
    ):
        """Read this view's v2 X. Returns ``(X, bundle, n_obs_out, obs_selector)``.

        Shared front half of ``to_anndata()`` and ``x()``: resolve the plan, read
        into host buffers, assemble the X matrix. Cleans up the shm bundle and
        re-raises on any failure before returning.

        ``matrix`` (Task 8) selects a family on a multi-matrix (v4) archive: the
        per-family submanifest (``_matrix_submanifest``) + shard locator
        (``_matrix_shard_locator``) stand in for the parent's manifest / packed
        locator, and that family's own ``n_vars`` is used to assemble X. On a
        single-matrix (v2/v3) archive (``self.parent._matrices is None``) this
        is the unchanged pre-Task-8 path: the parent's manifest + source kwargs +
        ``n_vars``, regardless of what ``matrix`` is passed.
        """
        from cellstream.v2.materialize import resolve_plan

        if self.parent._matrices is not None:
            sub = self.parent._matrix_submanifest(matrix)
            source_kwargs = {
                "manifest": sub,
                "shard_locator": self.parent._matrix_shard_locator(matrix),
            }
            n_vars = sub["n_vars"]
        else:
            sub = self.parent.manifest
            source_kwargs = self.parent._read_v2_source_kwargs()
            n_vars = self.parent.n_vars
        plan = resolve_plan(
            sub,
            container=container,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
            cast_back=cast_back,
            target="cpu",
        )
        row_start, row_stop, row_mask, row_indices = _normalize_key(self.key, self.parent.n_obs)
        # bundle initialised before the try so that an exception during
        # read_v2_to_host doesn't leak.
        bundle = None
        try:
            result = reader_cpu.read_v2_to_host(
                self.parent.path,
                plan=plan,
                n_workers=n_workers,
                omp_nthreads=blosc_nthreads,
                row_start=row_start,
                row_stop=row_stop,
                row_mask=row_mask,
                row_indices=row_indices,
                output_backing=output_backing,
                _return_shm_bundle=True,
                **source_kwargs,
            )
            # Extract the bundle BEFORE the obs_selector prep / _x_from_result so
            # that an exception in either (e.g. _x_from_result's CSR validation)
            # still reaches the except below and cleans it up — otherwise bundle
            # stays None and the shm segment leaks. read_v2_to_host is called with
            # _return_shm_bundle=True, so the bundle is always result[-1] for both
            # the CSR (data, indices, indptr, bundle) and dense (dense_2d, bundle)
            # shapes. Whether it is None depends on the ENGINE, not just output_backing:
            # the Rust path honors output_backing="heap" literally and returns None for
            # both CSR and dense, while the pure-Python multiprocess path always returns
            # a real bundle for both (its workers must write into shared memory, and it
            # skips the copy-out whenever _return_shm_bundle is set). That is what
            # to_anndata's docstring means by "Ignored on the pure-Python fallback".
            # Either way the caller is safe: bundle-backed CSR arrays each pin their own
            # segment (#274), and a bundle-backed dense array is re-wrapped by
            # _x_from_result via dense_array_for_bundle (#182).
            bundle = result[-1]
            if row_start is not None:
                obs_selector = slice(row_start, row_stop)
                n_obs_out = row_stop - row_start
            elif row_mask is not None:
                obs_selector = np.asarray(row_mask)
                n_obs_out = int(obs_selector.sum())
            else:
                # row_indices: reader sorts internally, so obs_selector
                # must use the SAME sorted order to align with the returned data.
                obs_selector = np.sort(np.asarray(row_indices, dtype=np.int64))
                n_obs_out = len(obs_selector)
            X, _ = self.parent._x_from_result(result, plan, n_obs_out=n_obs_out, n_vars=n_vars)
            return X, bundle, n_obs_out, obs_selector
        except BaseException:
            if bundle is not None:
                bundle.close_and_unlink()
            raise

    def x(
        self,
        *,
        matrix=None,
        layer=None,
        cast_back=True,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        n_workers=16,
        blosc_nthreads=4,
        output_backing="heap",
    ):
        """Read this view's raw X matrix (scipy CSR or dense ndarray), no AnnData.

        Skips the per-call header read + AnnData construction of ``to_anndata()``.
        The shm bundle (if any) is pinned to the returned matrix so it stays valid
        after this view is dropped (mirrors ``attach_buffer_cleanup`` on X).

        ``matrix``/``layer`` (Task 8) select a single family of a multi-matrix
        (v4) archive — see ``to_anndata()`` for the shared semantics.
        """
        key = self._resolve_matrix(matrix, layer)
        X, bundle, _n_obs_out, _obs_selector = self._read_x_v2(
            cast_back=cast_back,
            container=container,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
            n_workers=n_workers,
            blosc_nthreads=blosc_nthreads,
            output_backing=output_backing,
            matrix=key,
        )
        if bundle is not None:
            try:
                _shm.attach_buffer_cleanup_matrix(X, bundle)
            except BaseException:
                bundle.close_and_unlink()
                raise
        return X

    def x_cupy(self, *, cast_back=True, data_dtype=None, index_dtype=None, allow_lossy=False):
        """Return this view's raw X as a device-resident cupy CSR (no AnnData).

        Device sibling of ``x()`` (v2 byte-filter only). Decodes on the GPU
        (nvCOMP-RAW zstd + cupy byte-filter inverse). The view must be a
        shard-aligned contiguous range (e.g. a whole shard / GroupShard); a
        sub-shard or non-contiguous view raises ``NotImplementedError`` (#150).
        Small arbitrary subsets should use ``to_cupy_anndata`` (CPU-upload path).
        """
        from cellstream.v2.reader_gpu import read_v2_subset_to_device

        row_start, row_stop, row_mask, row_indices = _normalize_key(self.key, self.parent.n_obs)
        return read_v2_subset_to_device(
            self.parent.path,
            cast_back=cast_back,
            row_start=row_start,
            row_stop=row_stop,
            row_mask=row_mask,
            row_indices=row_indices,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
        )

    def to_torch(
        self,
        *,
        cast_back=True,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        n_streams=4,
        n_workers=16,
    ):
        """Subset view → torch tensor on device via DLPack.

        v2 subset GPU reads raise NotImplementedError (#40); the knobs do not lift
        that. PyTorch lacks uint16, so a uint16 data dtype is rejected.
        """
        if (not cast_back and data_dtype is None) or (
            data_dtype is not None and np.dtype(data_dtype) == np.dtype(np.uint16)
        ):
            raise ValueError(
                "to_torch() does not support a uint16 data dtype (PyTorch lacks it). "
                "Use the default float path (cast_back=True) or pass a float data_dtype; "
                "for the narrow-dtype GPU path use to_cupy_anndata(cast_back=False)."
            )
        gpu_adata = self.to_cupy_anndata(
            cast_back=cast_back,
            container=container,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
            n_streams=n_streams,
            n_workers=n_workers,
        )
        if container == "dense":
            import torch

            return torch.from_dlpack(gpu_adata.X.toDlpack())
        return _cupy_csr_to_torch(gpu_adata.X, total_nnz=self.parent.manifest["total_nnz"])

    def to_cupy_anndata(
        self,
        *,
        cast_back=True,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        n_streams=4,
        n_workers=16,
    ):
        """Subset view → cupy X on device.

        Dispatches to v2.reader_gpu.read_v2_subset_to_device,
        which raises NotImplementedError (#40, on-GPU subset decode). The
        materialization knobs are accepted for API parity but do NOT lift that.
        """
        # v2 subset: hybrid by projected output size (mirrors the full-archive path).
        from cellstream.v2.materialize import resolve_plan
        from cellstream.v2.reader_gpu import read_v2_subset_to_device

        row_start, row_stop, row_mask, row_indices = _normalize_key(self.key, self.parent.n_obs)
        if row_start is not None:
            n_sel = row_stop - row_start
        elif row_mask is not None:
            n_sel = int(np.asarray(row_mask).sum())
        else:
            n_sel = len(row_indices)
        plan = resolve_plan(
            self.parent.manifest,
            container=container,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
            cast_back=cast_back,
            target="cupy",
        )
        total_nnz = self.parent.manifest["total_nnz"]
        frac = n_sel / max(self.parent.n_obs, 1)
        if container == "csr":
            from cellstream.v2.reader_gpu import require_csr_float

            require_csr_float(plan.data_dtype)
            approx_bytes = (
                int(frac * total_nnz) * (plan.data_dtype.itemsize + plan.index_dtype.itemsize)
                + (n_sel + 1) * plan.indptr_dtype.itemsize
            )
        else:  # dense: sized as n_sel * n_vars (mirror the full-read path)
            approx_bytes = n_sel * self.parent.n_vars * plan.data_dtype.itemsize
        # Small subset -> CPU decode + device upload (no nvCOMP needed; honors dense).
        if approx_bytes < 5_000_000_000:
            adata_host = self.to_anndata(
                cast_back=cast_back,
                container=container,
                data_dtype=data_dtype,
                index_dtype=index_dtype,
                allow_lossy=allow_lossy,
                n_workers=n_workers,
            )
            return _upload_anndata_to_device(adata_host)
        # Large subset: on-GPU decode of the affected shards (CSR only; must be
        # shard-aligned contiguous, else read_v2_subset_to_device raises #150).
        if container != "csr":
            raise NotImplementedError(
                "GPU device decode of a large subset supports container='csr' only. "
                "Use container='csr' or the CPU reader (sh[...].to_anndata())."
            )
        gpu_csr = read_v2_subset_to_device(
            self.parent.path,
            cast_back=cast_back,
            n_streams=n_streams,
            row_start=row_start,
            row_stop=row_stop,
            row_mask=row_mask,
            row_indices=row_indices,
            data_dtype=data_dtype,
            index_dtype=index_dtype,
            allow_lossy=allow_lossy,
        )
        # Resolve obs_selector from the same key so obs and X stay aligned.
        if row_start is not None:
            obs_selector = slice(row_start, row_stop)
        elif row_mask is not None:
            obs_selector = np.asarray(row_mask)
        else:
            obs_selector = np.sort(np.asarray(row_indices, dtype=np.int64))
        return _make_anndata_from_device_csr(
            self.parent, gpu_csr, n_obs=gpu_csr.shape[0], obs_selector=obs_selector
        )


def _normalize_key(key, n_obs):
    """Resolve a __getitem__ key into (row_start, row_stop, row_mask, row_indices) — exactly one non-None.

    Mirrors v1's ``ShardedH5ad.__getitem__`` routing so v2/v3 indexing matches
    the v1 reference:
    - a scalar int selects a single row (wrapped to a length-1 selection).
    - int arrays must be 1-D with all values in [0, n_obs); negatives and
      out-of-range indices both raise IndexError (same bounds-check as v1's
      ``ShardedH5adView.from_int_array``). This is v1's contract, NOT numpy
      fancy-indexing: negatives are rejected (not wrapped to n-1) and rows are
      returned in sorted shard order, not input order.
    - bool masks must be 1-D with length == n_obs (else ValueError).
    - any other dtype (e.g. float) raises TypeError rather than silently
      truncating 1.9 -> 1, matching numpy and v1's rejection of non-integer
      ndarrays.
    """
    if isinstance(key, slice):
        start, stop, step = key.indices(n_obs)
        if step != 1:
            raise NotImplementedError("strided subset not supported")
        return start, stop, None, None
    # Scalar int -> single-row selection (matches v1's __getitem__). Exclude
    # bool: Python's bool subclasses int, so True/False would otherwise be
    # silently read as integer row indices 1/0 instead of being rejected.
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        key = np.array([int(key)], dtype=np.int64)
    arr = np.asarray(key)
    if arr.dtype == bool:
        if arr.ndim != 1 or arr.shape[0] != n_obs:
            raise ValueError(
                f"boolean mask must be 1-D with length n_obs={n_obs}, got shape {arr.shape}"
            )
        return None, None, arr, None
    # Integer-array path. Reject non-integer dtypes FIRST (e.g. float), so they
    # always raise TypeError regardless of ndim rather than silently truncating
    # 1.9 -> 1 — matching numpy + v1's rejection of non-integer ndarrays. An
    # empty array is exempt: np.asarray([]) is float64 but a zero-length
    # selection is always valid.
    if arr.size > 0 and not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"integer index array must have an integer dtype, got {arr.dtype}")
    # Shape error uses ValueError to match v1's ShardedH5adView.from_int_array.
    if arr.ndim != 1:
        raise ValueError(f"integer index array must be 1-D, got shape {arr.shape}")
    if arr.size > 0:
        lo, hi = int(arr.min()), int(arr.max())
        if lo < 0 or hi >= n_obs:
            raise IndexError(f"integer index out of range [0, {n_obs}); got [{lo}, {hi}]")
    return None, None, None, arr.astype(np.int64)


def _upload_csr_to_device(adata):
    """Upload a host AnnData's CSR X to the device via the three-array path.

    cupy.asarray does NOT accept scipy.sparse.csr_matrix; upload data/indices/
    indptr separately and re-construct a cupyx.scipy.sparse.csr_matrix.
    """
    import anndata as _ad
    import cupy as cp
    import cupyx.scipy.sparse as cusp

    from cellstream.v2.reader_gpu import require_csr_float

    csr = adata.X.tocsr()
    # Centralized float-CSR guard: covers v1 + v2 small-archive upload paths (cupyx
    # rejects non-float CSR data with a cryptic error otherwise). Codex review.
    require_csr_float(csr.data.dtype)
    d_data = cp.asarray(csr.data)
    d_idx = cp.asarray(csr.indices)
    d_indptr = cp.asarray(csr.indptr)
    X_dev = cusp.csr_matrix((d_data, d_idx, d_indptr), shape=csr.shape)
    out = _ad.AnnData(
        X=X_dev, obs=adata.obs, var=adata.var, uns=adata.uns, obsm=adata.obsm, varm=adata.varm
    )
    return out


def _upload_anndata_to_device(adata):
    """Upload a host AnnData's X to the device, handling dense or CSR X.

    Dense X (a numpy ndarray, incl. the _ShmDenseArray subclass) -> a dense cupy
    ndarray via a single H2D memcpy. Sparse X -> the three-array cupyx CSR path.
    """
    import anndata as _ad
    import cupy as cp

    X = adata.X
    if isinstance(X, np.ndarray):
        return _ad.AnnData(
            X=cp.asarray(X),
            obs=adata.obs,
            var=adata.var,
            uns=adata.uns,
            obsm=adata.obsm,
            varm=adata.varm,
        )
    return _upload_csr_to_device(adata)


def _make_anndata_from_device_csr(archive, gpu_csr, n_obs, obs_selector=None):
    """Wrap a cupy CSR into an AnnData with host obs/var/uns/etc.

    ``archive`` is the ``ShardedArchive`` (not a path) so the packed member
    locator remains available while loading the header.

    obs_selector: None for a full read (take first n_obs rows); otherwise a
    slice, bool mask, or int array selecting the rows of obs / obsm that match
    the data. Must be the SAME selector used to produce gpu_csr so obs and X
    stay aligned. This mirrors the obs_selector pattern from _build_anndata_v2.
    """
    header_adata = archive._packed.header_adata()
    if obs_selector is None:
        obs_sel = header_adata.obs.iloc[:n_obs]
        obsm_sel = header_adata.obsm
    elif isinstance(obs_selector, slice):
        obs_sel = header_adata.obs.iloc[obs_selector]
        obsm_sel = {k: v[obs_selector] for k, v in header_adata.obsm.items()}
    else:
        # bool mask or int array — applies the same way to both
        obs_sel = header_adata.obs.iloc[obs_selector]
        obsm_sel = {k: v[obs_selector] for k, v in header_adata.obsm.items()}
    return ad.AnnData(
        X=gpu_csr,
        obs=obs_sel.copy(),
        var=header_adata.var,
        uns=header_adata.uns,
        obsm=obsm_sel,
        varm=header_adata.varm,
    )


def _cupy_csr_to_torch(cupy_csr, *, total_nnz):
    """Zero-copy DLPack hand-off from cupy.sparse.csr_matrix to torch.sparse_csr_tensor.

    Enforces the matching-dtype rule by re-casting indices/indptr via
    _index_dtypes_for(target='torch') if needed.
    """
    import torch

    from cellstream.v2.dtypes import _index_dtypes_for

    target_idx, target_indptr = _index_dtypes_for(
        total_nnz=total_nnz, cast_back=True, target="torch"
    )
    # Preserve the cupy data dtype (it already reflects the requested data_dtype,
    # e.g. float16/float32/float64) instead of hard-casting to float32 — the old
    # hardcode silently discarded a requested dtype.
    data = cupy_csr.data
    indices = cupy_csr.indices.astype(target_idx, copy=False)
    indptr = cupy_csr.indptr.astype(target_indptr, copy=False)
    t_data = torch.from_dlpack(data.toDlpack())
    t_idx = torch.from_dlpack(indices.toDlpack())
    t_indptr = torch.from_dlpack(indptr.toDlpack())
    return torch.sparse_csr_tensor(t_indptr, t_idx, t_data, size=cupy_csr.shape)


def open(path):  # noqa: A001
    """Open a layout='cell' archive as a lazy CellStore (footer + offsets/indptr + mmap).

    Distinct from read_h5ad (which materializes). Raises a clear error if ``path`` is
    not a layout='cell' archive.
    """
    from cellstream.cell.reader import open_cell

    return open_cell(path)


# Top-level functional API preserved from v0.1.
def read_h5ad(
    path,
    *,
    cast_back=True,
    n_workers=None,
    blosc_nthreads=4,
    spill_dir=None,
    backing="shm",
    container="csr",
    data_dtype=None,
    index_dtype=None,
    allow_lossy=False,
    output_backing="heap",
):
    """Dispatch to the right reader based on the input.

    - packed (SHPK) file → ShardedArchive, or the cell reader for layout="cell"
    - .h5ad file with uint16 X.data on disk → read_narrow_h5ad
    - other .h5ad file → stock anndata.read_h5ad

    Delegates the file-dispatch logic to h5ad_io.read_h5ad for single-file
    paths. Directories are recognized and rejected with ``PackedOnlyError``.

    For archives, an explicit ``n_workers`` is forwarded to the v2 CPU reader.
    ``None`` (the default) selects the public archive-reader default of **16**
    — ``ShardedArchive.to_anndata``'s own default, reached by omitting the
    argument (packed ``layout="cell"`` passes 16 explicitly). The low-level
    ``reader_cpu.read_v2_to_host`` default of 8 is not what this path uses.
    There is no allocation-aware sizing here; that was v1-only (#51). For a
    single ``.h5ad`` file there is nothing to fan out over, so passing
    ``n_workers`` raises ``NotImplementedError`` rather than being silently
    dropped (#154; the ignored-knob class of #261).

    ``backing`` / ``spill_dir`` (#59) selected the output backing of the v1
    parallel read. **That reader was removed in #154, so they no longer apply
    to anything**; they are kept in the signature only to reject an explicit
    request loudly instead of ignoring it. An invalid ``backing`` raises
    ``ValueError`` up front, before any dispatch. An out-of-shm request
    (``backing != "shm"`` or ``spill_dir`` set) then raises
    ``NotImplementedError`` for packed archives and single ``.h5ad`` files.
    A directory raises ``PackedOnlyError`` before the backing check is reached.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if backing not in ("shm", "mmap", "auto"):
        raise ValueError(f"backing must be 'shm', 'mmap', or 'auto'; got {backing!r}")
    # Materialization knobs forwarded to the v2 to_anndata path; rejected for
    # single .h5ad files (v2-only).
    _mat = dict(
        container=container, data_dtype=data_dtype, index_dtype=index_dtype, allow_lossy=allow_lossy
    )
    if p.is_dir():
        reject_non_packed(f"{p} is a directory; cellstream reads only single-file packed archives.")
    # A single-file packed archive is a v2 archive (just bundled into one file),
    # so detect it BEFORE the single-file out-of-shm rejection (#106 Gap B):
    # out-of-shm backing was a v1-only capability (#59) and v1 is gone, so a packed
    # archive must report that ("v1-only"), not the single-file message below (#106 Gap B).
    from cellstream.packed.reader import PackedArchive, is_packed_file

    if is_packed_file(p):
        if backing != "shm" or spill_dir is not None:
            raise NotImplementedError(
                "out-of-shm backing (spill_dir / backing!='shm') is v1-only; "
                "v2 archives are not supported yet"
            )
        _man = PackedArchive(p).manifest
        if _man.get("layout") == "cell":
            from cellstream.cell.reader import read_cell_bulk

            return read_cell_bulk(
                p,
                cast_back=cast_back,
                container=container,
                data_dtype=data_dtype,
                index_dtype=index_dtype,
                allow_lossy=allow_lossy,
                n_workers=(16 if n_workers is None else n_workers),
                output_backing=output_backing,
            )
        sh = ShardedArchive(p)
        v2_kwargs = {} if n_workers is None else {"n_workers": n_workers}
        return sh.to_anndata(
            cast_back=cast_back,
            blosc_nthreads=blosc_nthreads,
            output_backing=output_backing,
            **v2_kwargs,
            **_mat,
        )
    # Single .h5ad file (narrow / stock): doesn't shard and never over-allocates a
    # single shm/mmap buffer, so out-of-shm backing doesn't apply — reject an
    # explicit request rather than silently ignoring it.
    if backing != "shm" or spill_dir is not None:
        raise NotImplementedError(
            "out-of-shm backing (spill_dir / backing!='shm') was supported only by "
            "the removed v1 sharded archive reader; it is unavailable for single "
            ".h5ad files"
        )
    if n_workers is not None:
        # A single .h5ad is read serially -- there is nothing to fan out over. This used
        # to be forwarded to the v1 parallel reader; with v1 gone (#154) it would be
        # silently dropped, which is the ignored-knob class #261 fixed. Reject instead.
        raise NotImplementedError(
            "n_workers applies to sharded archives, not single .h5ad files; "
            "drop it or point at an archive"
        )
    if _materialization_kwargs_set(container, data_dtype, index_dtype):
        raise NotImplementedError(_V1_MATERIALIZATION_MSG)
    return h5ad_io.read_h5ad(p, cast_back=cast_back)


def read_obs(path):
    from cellstream.packed.reader import PackedArchive, is_packed_file

    p = Path(path)
    if not p.exists():  # a typo must not get the migration note (see ShardedArchive)
        raise FileNotFoundError(f"no such archive: {p}")
    if is_packed_file(p):  # packed: read obs from the packed member
        return PackedArchive(p).obs()
    reject_non_packed("read_obs on a non-packed archive.")


def read_var(path):
    from cellstream.packed.reader import PackedArchive, is_packed_file

    p = Path(path)
    if not p.exists():  # a typo must not get the migration note (see ShardedArchive)
        raise FileNotFoundError(f"no such archive: {p}")
    if is_packed_file(p):
        return PackedArchive(p).var()
    reject_non_packed("read_var on a non-packed archive.")


read_narrow_h5ad = h5ad_io.read_narrow_h5ad


__all__ = [
    "GroupShard",
    "ShardedArchive",
    "open",
    "read_h5ad",
    "read_narrow_h5ad",
    "read_obs",
    "read_var",
]
