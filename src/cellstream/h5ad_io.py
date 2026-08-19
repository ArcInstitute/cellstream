"""Single-file ``.h5ad`` read helpers.

Extracted from the former ``_v1.py`` when the dormant v1 *archive*
surface (``ShardedH5ad`` / ``ShardedH5adView``) was removed in #154. Only the
h5ad utilities survived: they are live public API reached through
``cellstream.read`` (``read_narrow_h5ad``, ``read_h5ad``, ``read_obs``,
``read_var``) and by the packed reader.

``read_h5ad`` here handles **single files only**. Directory archives are
packed-only since v0.5 and are routed to ``ShardedArchive`` by
``cellstream.read.read_h5ad`` before this module is reached.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import hdf5plugin  # noqa: F401  (register Blosc filter)
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem

from cellstream import header as _header


def _target_indices_dtype(total_nnz: int) -> np.dtype:
    """Pick int32/int64 for X.indices based on scipy CSR's dtype unification.

    scipy's csr_matrix unifies the integer dtype of `indices` and `indptr`
    to one that can hold the max of either. Since `indptr[-1]` equals
    `total_nnz`, when `total_nnz >= 2**31` both arrays must be int64:
    the value 2**31 cannot be represented as a signed int32 (max is
    2**31 - 1 = 2_147_483_647).  Below that threshold int32 fits both.
    """
    return np.dtype("int64") if total_nnz >= 2**31 else np.dtype("int32")


def _read_obs_from_path(out_dir: Path) -> pd.DataFrame:
    """Read obs preferring obs.parquet; fall back to header.h5ad."""
    parquet_path = out_dir / "obs.parquet"
    if parquet_path.exists():
        return _header.read_obs_parquet(parquet_path)
    return _header.read_header_obs(out_dir / "header.h5ad")


def _read_anndata_extras(f) -> dict:
    """Read uns, obsm, varm, layers, obsp, varp from an open h5py.File. Returns kwargs dict."""
    extras = {}
    for key in ("uns", "obsm", "varm", "layers", "obsp", "varp"):
        if key in f:
            extras[key] = read_elem(f[key])
    return extras


def read_narrow_h5ad(
    path: str | Path,
    *,
    cast_back: bool = True,
) -> ad.AnnData:
    """Read a narrow-dtype .h5ad with one fused HDF5 read-and-cast pass.

    Uses h5py.Dataset.read_direct into a pre-allocated destination buffer
    in the target dtype, so HDF5 reads disk uint16, decompresses, AND
    converts to the destination dtype in a single pass -- no intermediate
    uint16 array materialized in Python.

    Parameters
    ----------
    path : path-like
        .h5ad file with X.data and X.indices stored as uint16 on disk.
    cast_back : bool, default True
        If True, return X.data as float32, X.indices as int32/int64.
        If False, X.data and X.indices stay uint16.
    """
    path = Path(path)
    with h5py.File(str(path), "r") as f:
        # X is assembled below by DIRECT attribute assignment, which deliberately bypasses
        # the scipy constructor to preserve narrow dtypes -- and therefore bypasses all of
        # its validation too. On a CSC file the same three arrays mean the transpose (indptr
        # runs over COLUMNS, indices holds ROW ids), so the result would be silently wrong,
        # and an indptr of length n_vars+1 on a matrix declared (n_obs, n_vars) reads out of
        # bounds downstream. write_one_h5ad already refuses non-CSR X with TypeError; this
        # makes the read path symmetric with the write path (#251).
        #
        # A MISSING marker is rejected too, rather than assumed CSR. anndata has written
        # encoding-type since 0.7.1; genuinely old 0.6.x files used h5sparse_format /
        # h5sparse_shape, which this function cannot read ANYWAY because it requires the
        # modern `shape` attribute two lines below. So a markerless X carrying a modern
        # `shape` is non-standard, and guessing CSR for it is precisely the silent misread
        # this guard exists to prevent. cellstream's own write_one_h5ad emits
        # encoding-type="csr_matrix" (verified), so nothing we produce is affected.
        # (codex review of #251.)
        enc = f["X"].attrs.get("encoding-type")
        if isinstance(enc, bytes):  # np.bytes_ is a bytes subclass, so this covers it too
            enc = enc.decode()
        if enc != "csr_matrix":
            raise ValueError(
                f"read_narrow_h5ad requires a CSR-encoded X; {path} has "
                f"encoding-type={enc!r}. Convert it first -- "
                "scipy.sparse.csr_matrix(adata.X) handles dense and CSC alike, whereas "
                ".tocsr() exists only on a sparse matrix -- then rewrite the file."
            )
        shape = tuple(f["X"].attrs["shape"])
        nnz = f["X/data"].shape[0]

        if cast_back:
            data_dtype = np.float32
            indices_dtype = _target_indices_dtype(nnz)
        else:
            data_dtype = f["X/data"].dtype
            indices_dtype = f["X/indices"].dtype

        data_buf = np.empty(nnz, dtype=data_dtype)
        f["X/data"].read_direct(data_buf)

        indices_buf = np.empty(nnz, dtype=indices_dtype)
        f["X/indices"].read_direct(indices_buf)

        indptr = f["X/indptr"][...]
        if cast_back:
            # scipy CSR matrices want indices and indptr to share a dtype; match indices.
            indptr = indptr.astype(indices_buf.dtype, copy=False)
        # else: keep on-disk int64 indptr — cumulative nnz can exceed the narrow indices range

        obs = read_elem(f["obs"])
        var = read_elem(f["var"])

        extras = _read_anndata_extras(f)

    # Build CSR matrix via manual attribute assignment to preserve narrow dtypes.
    # The csr_matrix((data, indices, indptr), ...) constructor normalises indices
    # to int32 regardless of input dtype.
    X = sp.csr_matrix(shape, dtype=data_buf.dtype)
    X.data = data_buf
    X.indices = indices_buf
    X.indptr = indptr
    return ad.AnnData(X=X, obs=obs, var=var, **extras)


def read_obs(out_dir: str | Path) -> pd.DataFrame:
    """Top-level convenience: read obs from an archive directory's sidecars."""
    return _read_obs_from_path(Path(out_dir))


def read_var(out_dir: str | Path) -> pd.DataFrame:
    """Top-level convenience: read var from an archive directory's header."""
    return _header.read_header_var(Path(out_dir) / "header.h5ad")


def read_h5ad(path: str | Path, *, cast_back: bool = True) -> ad.AnnData:
    """Read a **single** ``.h5ad`` file, narrow or stock.

    - ``.h5ad`` with ``uint16`` ``X.data`` on disk → :func:`read_narrow_h5ad`
    - any other ``.h5ad`` → stock ``anndata.read_h5ad``

    Directory archives never reach here: ``cellstream.read.read_h5ad`` routes a
    directory to ``ShardedArchive`` (packed-only since v0.5) and only calls
    this for single files. The out-of-shm knobs (``spill_dir`` / ``backing``)
    and ``n_workers`` belonged to the removed v1 directory path and are
    rejected by ``cellstream.read.read_h5ad`` before it gets here (#154).

    Parameters
    ----------
    path
        Path to a single ``.h5ad`` file.
    cast_back
        If True (default), ``X.data`` is float32 and ``X.indices`` are
        int32/int64. If False, on-disk narrow dtypes are preserved.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    IsADirectoryError
        If handed a directory — open packed archives with ``ShardedArchive``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir():
        raise IsADirectoryError(
            f"{p} is a directory; these archives have been packed single files since v0.5 — "
            f"open one with cellstream.ShardedArchive(path) or cellstream.open(path)."
        )
    # File: check on-disk X.data dtype.
    on_disk_data_dtype = None
    with h5py.File(str(p), "r") as f:
        if "X" in f and isinstance(f["X"], h5py.Group) and "data" in f["X"]:
            on_disk_data_dtype = f["X/data"].dtype

    if on_disk_data_dtype == np.uint16:
        return read_narrow_h5ad(p, cast_back=cast_back)
    return ad.read_h5ad(str(p))
