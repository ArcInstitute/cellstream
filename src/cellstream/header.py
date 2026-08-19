"""header.h5ad: full obs/var/uns/obsm/varm with no X.

The header is authoritative for cross-shard metadata. Per-shard /obs
is frozen at write time; if the header is updated, per-shard /obs
becomes stale by design (shards still readable as standalone h5ad).

obs.parquet is a convenience cache of header.obs in parquet format,
for tool-agnostic reads (polars, duckdb, pandas).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem


def permute_cell_axis_fields(obsm: dict, obsp: dict, perm) -> tuple[dict, dict]:
    """Select/reorder cell-axis annotation matrices by a row index array.

    ``perm`` may be a full permutation (a grouped write's storage order) OR a subset of rows (an
    annotated subset read) -- both are fancy-indexing. ``obsm`` entries are row-indexed (per-type
    dispatch: DataFrame/Series -> ``.iloc``; ndarray -> fancy-index; sparse -> explicit 2D
    ``v[perm, :]``; else cast-then-index); ``obsp`` (cell x cell) is indexed on BOTH axes
    (``obsp[perm, :][:, perm]`` -- the induced subgraph when ``perm`` is a subset). ``perm=None``
    returns shallow copies unchanged. Feature-axis fields (``varm``/``varp``) and ``uns`` are never
    permuted -- callers pass them through untouched."""
    import numpy as np
    import scipy.sparse as sp

    if perm is None:
        return dict(obsm), dict(obsp)

    def _perm_rows(v):
        if hasattr(v, "iloc"):  # pandas DataFrame/Series
            return v.iloc[perm]
        if sp.issparse(v):
            return v[perm, :]  # explicit 2D row-index (unambiguous across scipy spmatrix/sparray)
        if isinstance(v, np.ndarray):
            return v[perm]
        return np.asarray(v)[perm]

    return (
        {k: _perm_rows(v) for k, v in obsm.items()},
        {k: v[perm, :][:, perm] for k, v in obsp.items()},
    )


def _container_kind(v):
    """Which of ``permute_cell_axis_fields``' three dispatch arms a value belongs to. pandas is
    tested FIRST, matching ``_perm_rows``' order, so the two functions can never disagree about
    what a value is."""
    if hasattr(v, "iloc"):  # pandas DataFrame/Series
        return "pandas"
    if sp.issparse(v):
        return "sparse"
    return "ndarray"


def _concat_hint(field, key, detail):
    return (
        f"concat: {field}[{key!r}] {detail}; align it across parts or pass "
        f'annotations="drop" to merge X/obs/var/uns only.'
    )


def _require_same_keys(dicts, field):
    keys = set(dicts[0])
    for i, d in enumerate(dicts[1:], start=1):
        if set(d) != keys:
            raise ValueError(
                f"concat: {field} keys differ across parts (part 0: {sorted(keys)}, "
                f'part {i}: {sorted(d)}); align them across parts or pass annotations="drop".'
            )
    return sorted(keys)


def _require_one_kind(blocks, field, key):
    kinds = sorted({_container_kind(v) for v in blocks})
    if len(kinds) != 1:
        raise ValueError(
            _concat_hint(field, key, f"has mixed container kinds across parts {kinds}")
        )
    return kinds[0]


def _concat_obsm_key(blocks, key):
    kind = _require_one_kind(blocks, "obsm", key)
    if kind == "pandas":
        ref = blocks[0]
        for i, v in enumerate(blocks[1:], start=1):
            if hasattr(ref, "columns") != hasattr(v, "columns"):
                raise ValueError(
                    _concat_hint(
                        "obsm",
                        key,
                        f"is a Series in one part and a DataFrame in another (part {i})",
                    )
                )
            same = ref.columns.equals(v.columns) if hasattr(ref, "columns") else ref.name == v.name
            if not same:
                raise ValueError(
                    _concat_hint(
                        "obsm",
                        key,
                        f"has different columns in part {i}; pd.concat would silently NaN-fill "
                        f"the union rather than fail",
                    )
                )
        return pd.concat(blocks)
    if kind == "sparse":
        ref = blocks[0]
        for i, v in enumerate(blocks[1:], start=1):
            if v.shape[1] != ref.shape[1]:
                raise ValueError(
                    _concat_hint(
                        "obsm", key, f"has width {v.shape[1]} in part {i}, {ref.shape[1]} in part 0"
                    )
                )
        return sp.vstack(blocks, format="csr")
    arrs = [np.asarray(v) for v in blocks]
    ref = arrs[0]
    for i, v in enumerate(arrs[1:], start=1):
        if v.shape[1:] != ref.shape[1:]:
            raise ValueError(
                _concat_hint(
                    "obsm",
                    key,
                    f"has trailing shape {v.shape[1:]} in part {i}, {ref.shape[1:]} in part 0",
                )
            )
    return np.concatenate(arrs, axis=0)


def _block_diag_obsp_key(blocks, key):
    kind = _require_one_kind(blocks, "obsp", key)
    if kind == "pandas":
        raise ValueError(
            _concat_hint(
                "obsp", key, "is a pandas object; obsp must be a sparse or dense cell x cell matrix"
            )
        )
    if kind == "ndarray":
        # A key stored dense in EVERY part comes back dense, so concat(parts) equals a single write
        # of the same data (write_cell_archive round-trips a dense obsp). scipy.linalg's block_diag
        # allocates the (sum n_i)^2 output once and copies each block in; routing dense blocks
        # through sp.block_diag(...).toarray() instead measured 11.8x slower at 2.1x the peak memory
        # on 3x(2000x2000) f32, for byte-identical output (Gemini, PR #243). Imported lazily: a
        # dense obsp is rare and header.py is on the package import path.
        from scipy.linalg import block_diag as _dense_block_diag

        return _dense_block_diag(*blocks)
    return sp.block_diag(blocks, format="csr")


def concat_cell_axis_fields(obsm_dicts, obsp_dicts) -> tuple[dict, dict]:
    """Assemble N parts' cell-axis annotation matrices into one pair, in INPUT ROW ORDER (#237).

    ``obsm`` entries are row-concatenated with the same per-type dispatch
    ``permute_cell_axis_fields`` uses (pandas -> ``pd.concat``; sparse -> ``sp.vstack``; else
    ``np.concatenate``). ``obsp`` (cell x cell) is assembled **block-diagonal** -- the semantics of
    ``anndata.concat(..., pairwise=True)``: graphs computed independently per part carry no
    cross-part edges. Sparse blocks go through ``sp.block_diag``, dense ones through
    ``scipy.linalg.block_diag``; both rebase each block's row/column indices by its own offset, so
    no manual cell-offset arithmetic is needed.

    Every part must carry the SAME keys, the same container kind per key, and matching widths;
    anything else raises ``ValueError`` naming the field, key and part (require-equal, never a
    silent drop or NaN-fill). Value dtypes MAY differ and take numpy/pandas natural promotion,
    mirroring ``_widest`` for X's on-disk dtypes.

    The result is UNPERMUTED: the caller passes it to ``permute_cell_axis_fields`` to reach storage
    order. Feature-axis fields (``varm``/``varp``) are not concatenated -- all parts share ``var``,
    so the caller inherits them from part 0 after a require-equal check.
    """
    obsm_keys = _require_same_keys(obsm_dicts, "obsm") if obsm_dicts else []
    obsp_keys = _require_same_keys(obsp_dicts, "obsp") if obsp_dicts else []
    return (
        {k: _concat_obsm_key([d[k] for d in obsm_dicts], k) for k in obsm_keys},
        {k: _block_diag_obsp_key([d[k] for d in obsp_dicts], k) for k in obsp_keys},
    )


def write_header(adata: ad.AnnData, path: str | Path) -> None:
    """Write a header.h5ad containing obs, var, uns, obsm, varm, obsp, varp; X is empty CSR.

    ``obsp``/``varp`` (pairwise graphs) are serialized when non-empty (Bucket 3).
    A no-graph header is byte-identical to the pre-Bucket-3 output — anndata writes
    no ``obsp``/``varp`` groups for empty ``OverloadedDict``s. The cell-axis ``obsp``
    is expected to already be in partition order: the caller permutes it (in-memory
    grouped writes via ``adata[perm].copy()``; grouped path writes via
    ``_write_permuted_metadata``). ``layers``/``.raw`` are still written separately
    by the multi-matrix writer, not here.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    empty_X = sp.csr_matrix((adata.n_obs, adata.n_vars), dtype="float32")
    h = ad.AnnData(
        X=empty_X,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        uns=dict(adata.uns),
        obsm=dict(adata.obsm),
        varm=dict(adata.varm),
    )
    # Attach obsp/varp only when non-empty, so a no-graph header stays
    # byte-identical to the pre-Bucket-3 output on every anndata version (passing
    # an empty dict to the constructor can still materialize an empty group).
    for k, v in adata.obsp.items():
        h.obsp[k] = v
    for k, v in adata.varp.items():
        h.varp[k] = v
    h.write_h5ad(path)


def read_header_obs(path: str | Path) -> pd.DataFrame:
    import h5py

    with h5py.File(str(path), "r") as f:
        return read_elem(f["obs"])


def read_header_var(path: str | Path) -> pd.DataFrame:
    import h5py

    with h5py.File(str(path), "r") as f:
        return read_elem(f["var"])


def read_header_full(path: str | Path) -> ad.AnnData:
    """Return the full header AnnData (X empty)."""
    return ad.read_h5ad(str(path))


def write_obs_parquet(obs: pd.DataFrame, path: str | Path) -> None:
    """Atomic write of obs as zstd parquet."""
    import os

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    obs.to_parquet(tmp, compression="zstd", index=True)
    os.replace(tmp, path)


def read_obs_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_var_parquet(var: pd.DataFrame, path: str | Path) -> None:
    """Atomic write of a matrix family's var as zstd parquet (index preserved)."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    var.to_parquet(tmp, compression="zstd", index=True)
    os.replace(tmp, path)


def read_var_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_modality_meta(adata: ad.AnnData, path: str | Path, *, perm=None) -> None:
    """Atomic empty-X h5ad holding a modality's full obs + obsm/varm/uns + obsp/varp.

    The modality's X lives in its shard family and its var in the parquet sidecar,
    so those are not duplicated here; everything else on the cell/feature axes is.
    Written next to that family's shards as its metadata sidecar.

    ``perm``, when given, is the cell-axis row permutation the archive's shared
    partition applies to every family (e.g. from a ``group_by`` write). Cell-axis
    fields are reordered so sidecar row i lines up with the family's on-disk shard
    row i: ``obs`` and ``obsm`` are row-permuted (per-type dispatch: DataFrame/
    Series -> ``.iloc``, ndarray/sparse -> fancy-index, else cast-then-index) and
    ``obsp`` (cell×cell) is permuted on BOTH axes (``obsp[perm][:, perm]``).
    ``varm``/``varp`` (feature axis) and ``uns`` (global) are never permuted.
    ``perm=None`` (default) applies no permutation.

    Backstops the writer's per-modality pre-flight: rejects a modality's own
    ``.raw`` here too (obsp/varp ARE stored — Bucket 3), so it is never silently
    dropped even if a caller skips the pre-flight (mirrors ``write_header``).
    """
    _reject_unsupported_modality_fields(adata)
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    obs = adata.obs.iloc[perm].copy() if perm is not None else adata.obs.copy()
    obsm, obsp = permute_cell_axis_fields(dict(adata.obsm), dict(adata.obsp), perm)
    meta = ad.AnnData(
        X=sp.csr_matrix((adata.n_obs, adata.n_vars), dtype="float32"),
        obs=obs,
        var=pd.DataFrame(index=adata.var_names),
        uns=dict(adata.uns),
        obsm=obsm,
        varm=dict(adata.varm),
    )
    # Attach obsp/varp only when non-empty (no empty groups on any anndata version).
    for k, v in obsp.items():
        meta.obsp[k] = v
    for k, v in adata.varp.items():
        meta.varp[k] = v
    tmp = path.with_suffix(path.suffix + ".tmp")
    meta.write_h5ad(tmp)
    os.replace(tmp, path)


def read_modality_meta(path: str | Path) -> ad.AnnData:
    return ad.read_h5ad(str(path))


def _reject_unsupported_modality_fields(adata: ad.AnnData) -> None:
    """Raise if a modality carries a field the archive cannot store: its own
    ``.raw`` (deferred). ``obsp``/``varp`` ARE stored (Bucket 3)."""
    if adata.raw is not None:
        raise ValueError(
            "a modality's own .raw is not supported (deferred); clear it "
            "(del modality.raw) before writing or file a feature request."
        )
