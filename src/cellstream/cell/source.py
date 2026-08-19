"""Block-streaming sources for layout=cell writes (#219).

`write_cell_archive` used to materialise the whole CSR (`X = adata.X.tocsr()`), so a ~40 GB
archive needed a 300-700 GB allocation. The ENCODER already streams -- the INPUT was the ceiling.

A `CellBlockSource` is the weakest contract that removes it: contiguous input-order row blocks,
plus per-row nnz from INDPTR ALONE (never data/indices). The storage permutation is paid later,
on the encoded frames (spec §4.1), so a source never needs random row access.

`obs` / `var` / `uns` are PROPERTIES, not methods: cellstream.grouping duck-types on `.obs`.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellstream.matrices import real_layer_names

from . import format as fmt
from .format import UNSUPPORTED_FIELDS as _UNSUPPORTED_FIELDS


class InMemoryAnnDataSource:
    """An already-resident AnnData. Implements the protocol so `_resolve_storage_order` has ONE
    uniform way to interrogate a source -- but `is_in_memory=True` routes `write_cell_archive`
    to its unchanged single-shot path. It does NOT stream."""

    is_in_memory = True

    def __init__(self, adata):
        self._a = adata
        X = adata.X
        # X is USUALLY already sparse (the streaming split only wraps AnnData directly for the
        # unchanged single-shot path); the dense branch exists for an in-memory AnnData whose .X
        # is a plain dense ndarray (a dense .h5ad now streams via DenseH5adSource, #225) --
        # ndarray has no .tocsr(). Mirrors the sp.issparse(...) else sp.csr_matrix(...) pattern
        # already used for the same purpose in v2/writer.py.
        self._X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        self._X.sort_indices()
        self.n_obs, self.n_vars = adata.shape

    @property
    def X(self):
        return self._X

    @property
    def obs(self):
        return self._a.obs

    @property
    def var(self):
        return self._a.var

    @property
    def uns(self):
        return dict(self._a.uns)

    @property
    def raw(self):
        return self._a.raw

    @property
    def obsm(self):
        return dict(self._a.obsm)

    @property
    def varm(self):
        return dict(self._a.varm)

    @property
    def obsp(self):
        return dict(self._a.obsp)

    @property
    def varp(self):
        return dict(self._a.varp)

    def dropped_fields(self):
        return [
            f
            for f in _UNSUPPORTED_FIELDS
            if len(real_layer_names(self._a) if f == "layers" else getattr(self._a, f, ())) > 0
        ]

    def row_nnz(self):
        return np.diff(self._X.indptr).astype(np.int64)

    def read_block(self, lo, hi):
        return self._X[lo:hi]

    def data_dtype(self):
        return self._X.data.dtype

    def index_dtype(self):
        return self._X.indices.dtype

    def close(self):
        pass


class BackedH5adSource:
    """A backed .h5ad with a CSR-encoded X. Blocks are single contiguous hyperslabs."""

    is_in_memory = False

    def __init__(self, path):
        self._path = Path(path)
        self._a = ad.read_h5ad(str(path), backed="r")
        self.n_obs, self.n_vars = self._a.shape
        self._row_nnz = None
        self._index_dtype = None

    @property
    def obs(self):
        return self._a.obs

    @property
    def var(self):
        return self._a.var

    @property
    def uns(self):
        return dict(self._a.uns)

    @property
    def raw(self):
        # MUST be the real .raw, not None. write_cell_archive warns when it drops .raw, and
        # hard-coding None here would make a backed h5ad drop it SILENTLY while the in-memory
        # path warns -- a behaviour regression invisible to every in-memory test.
        return self._a.raw

    @property
    def obsm(self):
        return dict(self._a.obsm)

    @property
    def varm(self):
        return dict(self._a.varm)

    @property
    def obsp(self):
        return dict(self._a.obsp)

    @property
    def varp(self):
        return dict(self._a.varp)

    def dropped_fields(self):
        return [
            f
            for f in _UNSUPPORTED_FIELDS
            if len(real_layer_names(self._a) if f == "layers" else getattr(self._a, f, ())) > 0
        ]

    def row_nnz(self):
        if self._row_nnz is None:
            import h5py

            # indptr ALONE: n_obs+1 int64 = 44 MB even at CCL_2 scale (5.5M cells). The old
            # writer materialised the WHOLE CSR just to compute this (#219 site 1).
            with h5py.File(str(self._path), "r") as f:
                self._row_nnz = np.diff(f["X/indptr"][:]).astype(np.int64)
        return self._row_nnz

    def read_block(self, lo, hi):
        blk = sp.csr_matrix(self._a.X[lo:hi])
        blk.sort_indices()
        return blk

    def data_dtype(self):
        return np.dtype(self._a.X.dtype)

    def index_dtype(self):
        if self._index_dtype is None:
            import h5py

            with h5py.File(str(self._path), "r") as f:
                self._index_dtype = np.dtype(f["X/indices"].dtype)
        return self._index_dtype

    def close(self):
        if getattr(self._a, "file", None) is not None:
            self._a.file.close()


class DenseH5adSource:
    """A backed .h5ad whose X is DENSE -- an HDF5 2-D Dataset rather than a Group (#225).
    open_cell_source routes ANY Dataset X here (a sparse X is always a Group), including older
    files that lack encoding-type=="array". Blocks are single contiguous hyperslabs read from the
    dataset and wrapped as CSR -- the sibling of BackedH5adSource for dense inputs. Metadata comes from a backed AnnData; the matrix comes from
    a persistent raw h5py handle (version-independent, and read_block is called once per encode
    block, so the handle is held rather than reopened per call).

    A dense X has NO indptr, so row_nnz() reads the data (two passes over X: the nnz scan, then
    encode). Peak RAM stays O(block) in both passes."""

    is_in_memory = False

    def __init__(self, path):
        import h5py

        self._path = Path(path)
        self._a = ad.read_h5ad(str(path), backed="r")  # obs/var/uns/raw/shape -- X NOT materialised
        self._h5 = None
        try:
            self._h5 = h5py.File(str(path), "r")
            self._X = self._h5["X"]  # the 2-D dense Dataset
            self.n_obs, self.n_vars = self._a.shape
            self._row_nnz = None
        except Exception:
            self.close()  # a later init step failed -- don't leak the backed AnnData / h5py handle
            raise

    @property
    def obs(self):
        return self._a.obs

    @property
    def var(self):
        return self._a.var

    @property
    def uns(self):
        return dict(self._a.uns)

    @property
    def raw(self):
        # MUST be the real .raw (mirrors BackedH5adSource): write_cell_archive warns when it drops
        # .raw; returning None here would drop it SILENTLY relative to the in-memory path.
        return self._a.raw

    @property
    def obsm(self):
        return dict(self._a.obsm)

    @property
    def varm(self):
        return dict(self._a.varm)

    @property
    def obsp(self):
        return dict(self._a.obsp)

    @property
    def varp(self):
        return dict(self._a.varp)

    def dropped_fields(self):
        return [
            f
            for f in _UNSUPPORTED_FIELDS
            if len(real_layer_names(self._a) if f == "layers" else getattr(self._a, f, ())) > 0
        ]

    def _dense_chunk_rows(self):
        # Rows whose DENSE hyperslab fits the ~2 GiB budget. Bounds BOTH the row_nnz scan and
        # read_block's dense materialisation, so peak RAM is O(block) no matter how many rows the
        # writer's (nnz-cost-based) block planner grouped for a sparse-but-dense matrix -- reading
        # such a tall block as one dense hyperslab would blow the budget and OOM (Gemini + Copilot
        # #231 review). fmt._BLOCK_BYTES is read at call time so a test can monkeypatch it. Dense
        # rows are uniform width, so a fixed ROW COUNT is the right model here.
        itemsize = self._X.dtype.itemsize
        return max(1, fmt._BLOCK_BYTES // max(1, self.n_vars * itemsize))

    def row_nnz(self):
        # A dense X has NO indptr -> per-row nnz must be read from the data, in bounded row chunks
        # (two passes over X: this nnz scan, then encode). Cached: the writer calls this twice
        # (block planning + final indptr). Peak = one chunk -> O(block).
        if self._row_nnz is None:
            rows = self._dense_chunk_rows()
            parts = [
                np.count_nonzero(self._X[lo : min(lo + rows, self.n_obs)], axis=1)
                for lo in range(0, self.n_obs, rows)
            ]
            self._row_nnz = (
                np.concatenate(parts).astype(np.int64) if parts else np.empty(0, dtype=np.int64)
            )
        return self._row_nnz

    def read_block(self, lo, hi):
        # Chunk the DENSE read so peak RAM stays O(block): the writer's planner sizes (lo,hi) by CSR
        # nnz cost, which for a sparse-but-dense matrix can group very many rows -- reading that
        # whole span as one dense hyperslab would OOM (Gemini + Copilot #231). Read <= chunk_rows at
        # a time, convert each to CSR, and vstack; byte-identical to sp.csr_matrix(self._X[lo:hi]).
        chunk_rows = self._dense_chunk_rows()
        if hi - lo <= chunk_rows:
            blk = sp.csr_matrix(self._X[lo:hi])
        else:
            parts = [
                sp.csr_matrix(self._X[c : min(c + chunk_rows, hi)])
                for c in range(lo, hi, chunk_rows)
            ]
            blk = sp.vstack(parts, format="csr")
        # sp.vstack (and, defensively, csr_matrix) can widen indices/indptr to int64 on some scipy
        # versions/platforms; index_dtype() hardcodes int32, so force int32 to keep read_block's
        # output matching the source contract (Gemini #231 review). Values always fit: n_vars and a
        # per-block nnz are both < 2**31.
        if blk.indices.dtype != np.int32:
            blk.indices = blk.indices.astype(np.int32, copy=False)
        if blk.indptr.dtype != np.int32:
            blk.indptr = blk.indptr.astype(np.int32, copy=False)
        blk.sort_indices()
        return blk

    def data_dtype(self):
        return np.dtype(self._X.dtype)

    def index_dtype(self):
        # int32: sp.csr_matrix(dense) builds SIGNED indices (verified int32 even at n_vars=70000),
        # which is what read_block hands the encoder -- the source contract. NOT
        # choose_index_dtype(n_vars): that is the ON-DISK width, computed by the writer itself.
        # Mirrors CellArchiveSource.index_dtype().
        return np.dtype(np.int32)

    def close(self):
        if getattr(self, "_h5", None) is not None:
            self._h5.close()
            self._h5 = None
        if getattr(self._a, "file", None) is not None:
            self._a.file.close()


class CellArchiveSource:
    """An existing layout=cell .shad -- the re-encode / codec-migration path, the one #219's OOM
    was measured on. Contiguous row ranges gather sequentially from the mmap'd payload."""

    is_in_memory = False

    def __init__(self, store, *, owns_store: bool = True):
        self._s = store
        self._owns_store = owns_store
        self.n_obs = store.n_obs
        self.n_vars = store.n_vars
        self._row_nnz = None
        # Re-encode at the archive's OWN on-disk value width (#277). gather_rows' default
        # output is float32 for every non-float64 archive, which ROUNDS integer counts above
        # 2**24, and data_dtype() then reported that float32 as the SOURCE dtype -- so the
        # writer re-encoded the rounded values and still stamped the result uint32. The frame
        # decoders already produce the on-disk width; only the output cast was lossy.
        self._src_dtype = np.dtype(store.manifest["value_dtype_on_disk"])

    @property
    def obs(self):
        return self._s.obs

    @property
    def var(self):
        return self._s.var

    @property
    def uns(self):
        return self._s.uns

    @property
    def raw(self):
        # a cell archive can't have .raw (it stores X + obs/var/uns/obsm/varm/obsp/varp, not .raw)
        return None

    @property
    def obsm(self):
        return dict(self._s.obsm)

    @property
    def varm(self):
        return dict(self._s.varm)

    @property
    def obsp(self):
        return dict(self._s.obsp)

    @property
    def varp(self):
        return dict(self._s.varp)

    def dropped_fields(self):
        # a cell archive stores obsm/varm/obsp/varp (#229) and never has layers -> nothing dropped
        return []

    def row_nnz(self):
        # Cached: the streaming writer calls this multiple times (block planning + final indptr),
        # and self._s.indptr is O(n_obs); recomputing np.diff each call is avoidable (matches
        # BackedH5adSource). Copilot #221 review.
        if self._row_nnz is None:
            self._row_nnz = np.diff(self._s.indptr).astype(np.int64)
        return self._row_nnz

    def read_block(self, lo, hi):
        blk = self._s.gather_rows(np.arange(lo, hi, dtype=np.int64), out_dtype=self._src_dtype)
        blk.sort_indices()
        return blk

    def data_dtype(self):
        # The archive's OWN on-disk value dtype -- what read_block's blocks now contain, and
        # therefore what use_rust_for_cell_encode and Rust's match arm see. All four on-disk
        # widths (uint16/uint32/float32/float64) are in _RUST_SRC_DATA. The optimistic-encode
        # retry (#216) still probes integrality per value during encode.
        return self._src_dtype

    def index_dtype(self):
        return np.dtype(np.int32)  # gather_rows builds a scipy CSR -> signed indices

    def close(self):
        # Only close a store THIS adapter opened (open_cell_source(path)). A caller-supplied
        # CellStore (open_cell_source(store) / write_cell_archive(store, ...)) is not owned by
        # the adapter -- closing it out from under the caller is the #219 final-review Important
        # finding: write_cell_archive(store, out) used to kill the caller's store as a side
        # effect of the write completing.
        if self._owns_store:
            self._s.close()


def open_cell_source(src):
    """Dispatch a write source to its CellBlockSource adapter.

    AnnData          -> InMemoryAnnDataSource (is_in_memory=True; the unchanged single-shot path)
    .h5ad path       -> BackedH5adSource (CSR X) / DenseH5adSource (dense Dataset X, #225) /
                        InMemoryAnnDataSource (CSC / unknown encoding -- fail-safe fallback)
    cell .shad path  -> CellArchiveSource (owns the CellStore it opens -> close() closes it)
    CellStore        -> CellArchiveSource (does NOT own the caller's store -> close() leaves it
                        open; the caller opened it and the caller closes it)
    """
    from .reader import CellStore, open_cell

    if isinstance(src, CellStore):
        return CellArchiveSource(src, owns_store=False)
    if isinstance(src, ad.AnnData):
        return InMemoryAnnDataSource(src)

    p = Path(src)
    with open(p, "rb") as f:
        magic = f.read(4)

    if magic == b"SHPK":  # a packed cellstream archive
        from cellstream.packed.reader import PackedArchive

        man = PackedArchive(p).manifest
        if man.get("layout") != fmt.LAYOUT_CELL:
            raise ValueError(
                f"{p}: this is a shard-layout archive (layout={man.get('layout')!r}); "
                f"streaming a shard-layout archive into a cell archive is not supported. "
                f"Read it into memory first, then pass the AnnData."
            )
        return CellArchiveSource(open_cell(p))  # opened here -> owns_store=True (default)

    # .h5ad -- route by X's on-disk FORM. A sparse X (CSR/CSC) is ALWAYS an HDF5 Group; a dense X
    # is a 2-D Dataset. So a Dataset X -> DenseH5adSource (#225, O(block)) -- which also streams
    # older/third-party dense files that lack encoding-type=="array" rather than materialising them
    # (Gemini #231 review); a CSR Group -> BackedH5adSource; a CSC Group / other / missing X ->
    # the in-memory fallback.
    import h5py

    with h5py.File(str(p), "r") as f:
        if "X" in f:
            x_is_dataset = isinstance(f["X"], h5py.Dataset)
            enc = f["X"].attrs.get("encoding-type")
        else:
            x_is_dataset = False
            enc = None
    if hasattr(enc, "decode"):
        enc = enc.decode()  # h5py may return a fixed-length attr as bytes/np.bytes_ (Copilot #231)
    if x_is_dataset:
        return DenseH5adSource(p)
    if enc == "csr_matrix":
        return BackedH5adSource(p)
    return InMemoryAnnDataSource(ad.read_h5ad(str(p)))
