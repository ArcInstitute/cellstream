"""cellstream: sharded h5ad with narrow-dtype compression."""

from __future__ import annotations

try:
    from cellstream._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

from cellstream import errors  # noqa: F401  (namespace re-export)
from cellstream.cell.reader import CellStore, GroupSpan
from cellstream.errors import (
    ArchiveExistsError,
    ArchiveLockedError,
    CellstreamError,
    GpuOutOfMemoryError,
    IncompatibleSchemaError,
    LossyCastError,
    PackedOnlyError,
)
from cellstream.read import (
    ShardedArchive,
    open,  # noqa: A004
    read_h5ad,
    read_narrow_h5ad,
    read_obs,
    read_var,
)
from cellstream.runtime import rust_available, rust_status
from cellstream.write import write_h5ad_narrow, write_sharded

__all__ = [
    "__version__",
    "errors",
    "ArchiveExistsError",
    "ArchiveLockedError",
    "CellStore",
    "GpuOutOfMemoryError",
    "GroupSpan",
    "IncompatibleSchemaError",
    "LossyCastError",
    "PackedOnlyError",
    "rust_available",
    "rust_status",
    "CellstreamError",
    "write_h5ad_narrow",
    "write_sharded",
    "open",
    "read_h5ad",
    "read_narrow_h5ad",
    "ShardedArchive",
    "read_obs",
    "read_var",
]

try:
    from cellstream import anndata_backend as _anndata_backend

    anndata_backend_registered = _anndata_backend.register()
except Exception:
    anndata_backend_registered = False

__all__.append("anndata_backend_registered")
