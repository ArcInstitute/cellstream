"""Per-shard / single-file h5ad write+read primitives.

The write path is: cast X.data and X.indices to narrow dtypes (lossless-
verified), write uncompressed via anndata, then recompress numeric
datasets via codecs._recompress_numeric_datasets. This dance avoids the
SIGFPE that anndata.write_h5ad(compression=...) hits on vlen strings.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import hdf5plugin  # noqa: F401  (register Blosc filter on import)
import scipy.sparse as sp

from cellstream import codecs
from cellstream.matrices import real_layer_names
from cellstream.narrow import lossless_cast


def write_one_h5ad(
    adata: ad.AnnData,
    path: str | Path,
    *,
    codec: str,
    where: str = "X",
) -> None:
    """Write ``adata`` as a single .h5ad with narrow X dtypes per the codec preset.

    Parameters
    ----------
    adata : AnnData
        Must be in-memory (not backed). X must be a CSR sparse matrix.
    path : path-like
        Destination .h5ad file.
    codec : str
        Codec preset name (e.g. 'blosc-zstd-1-u16u16').
    where : str
        Diagnostic prefix in LossyCastError messages.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not sp.issparse(adata.X) or adata.X.format != "csr":
        raise TypeError(
            f"adata.X must be a CSR sparse matrix; got "
            f"{type(adata.X).__name__}"
            + (f" (format={adata.X.format!r})" if sp.issparse(adata.X) else "")
        )

    # The lossless casts MUST happen before any disk write so we fail loud early.
    data_dtype = codecs.get_data_dtype(codec)
    indices_dtype = codecs.get_indices_dtype(codec)

    # We copy on cast to avoid mutating the caller's AnnData.
    X = adata.X
    new_data = lossless_cast(X.data, data_dtype, where=f"{where}.data")
    new_indices = lossless_cast(X.indices, indices_dtype, where=f"{where}.indices")

    # Build via manual attribute assignment to preserve narrow dtypes.
    # The csr_matrix((data, indices, indptr), ...) constructor normalises indices
    # to int32 regardless of input dtype, so we assign the arrays directly.
    X_narrow = sp.csr_matrix(X.shape, dtype=new_data.dtype)
    X_narrow.data = new_data
    X_narrow.indices = new_indices
    X_narrow.indptr = X.indptr.astype("int64")
    a_narrow = ad.AnnData(
        X=X_narrow,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        uns=dict(adata.uns),
        obsm=dict(adata.obsm),
        varm=dict(adata.varm),
        layers={k: adata.layers[k] for k in real_layer_names(adata)},
        obsp=dict(adata.obsp),
        varp=dict(adata.varp),
    )
    a_narrow.write_h5ad(path)

    # Now apply Blosc to numeric chunked datasets only.
    codecs._recompress_numeric_datasets(
        path,
        codecs.get_filter_kwargs(codec),
        data_filter_kwargs=codecs.get_data_filter_kwargs(codec),
    )
