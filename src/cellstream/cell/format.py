"""On-disk format constants, dtype selection, and manifest fields for layout=cell.

The cell layout stores per-cell pfordelta frames in a single SHPK packed file
(``.shad``) reusing cellstream's ``packed/`` container: three immutable members
(``x_payload`` mmap'd, ``x_offsets`` and ``x_indptr`` int64[n_obs+1]) plus the
standard metadata tail (obs.parquet + header.h5ad + manifest.json + groups.json).
``schema_version`` is a monotonic compatibility gate; pfordelta cell archives are
version 5, zstd cell archives (integer or float) are version 6.
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = 5  # pfordelta cell archives
SCHEMA_VERSION_ZSTD = 6  # zstd cell archives (integer or float)
LAYOUT_CELL = "cell"
CODEC_PFORDELTA = "pfordelta"
CODEC_ZSTD = "zstd"
SUPPORTED_CODECS = frozenset({CODEC_PFORDELTA, CODEC_ZSTD})
PYFASTPFOR_CODEC = "simdfastpfor256"

# Fields layout='cell' does not store (multi-matrix, deferred). The writer warns and drops them;
# a streaming source reports them via dropped_fields() so a backed h5ad warns identically.
# obsm/varm/obsp/varp ARE stored in header.h5ad (#229); .raw is warned separately by the writer.
UNSUPPORTED_FIELDS = ("layers",)

# Packed member names.
PAYLOAD = "x_payload"  # concatenated per-cell frames, storage order; 4 KB-aligned, mmap'd
OFFSETS = "x_offsets"  # int64[n_obs+1] byte offsets into PAYLOAD (offsets[0]==0)
INDPTR = "x_indptr"  # int64[n_obs+1] CSR indptr (cumulative nnz), storage order
GROUPS = "groups.json"  # per-group row-ranges (present iff grouped)
OBS = "obs.parquet"
HEADER = "header.h5ad"
MANIFEST = "manifest.json"

# New packed member roles for the immutable per-cell data region.
ROLE_PAYLOAD = "payload"
ROLE_OFFSETS = "offsets"
ROLE_INDPTR = "indptr"

_DTYPES = {
    "uint16": np.dtype(np.uint16),
    "uint32": np.dtype(np.uint32),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}


def np_dtype(name: str) -> np.dtype:
    return _DTYPES[name]


def choose_index_dtype(n_vars: int) -> str:
    return "uint16" if n_vars < 65536 else "uint32"


def choose_value_dtype(max_value: int) -> str:
    return "uint16" if max_value < 65536 else "uint32"


# Byte budget for an auto-planned block's in-RAM CSR. 2 GiB keeps peak RAM bounded while
# amortising per-block overhead. An explicit block_size (a ROW COUNT) overrides it. Lives here
# (not writer.py) alongside _plan_blocks, which the streaming cell writer imports -- writer.py ->
# source.py is already a function-local import to avoid a cycle (commit ef6ad38), so keeping the
# block planner in format.py sidesteps a module-level source.py -> writer.py import the other way.
_BLOCK_BYTES = 2 << 30


def _plan_blocks(row_nnz, block_size, itemsize):
    """Contiguous [lo, hi) input-order row ranges.

    block_size=None -> NNZ-BALANCED variable ranges under _BLOCK_BYTES. A fixed CELL COUNT is a
    bad default: nnz/cell varies by orders of magnitude across datasets (CCL_2 is 6510 nnz/cell
    ~ 52 KB/cell through gather_rows, so a naive 200K-cell block is 10.4 GB, while a sparse
    dataset makes the same block trivially small). Always >= 1 row, so one huge cell still forms
    a block.
    block_size=N -> fixed N-row ranges.
    """
    n = len(row_nnz)
    if n == 0:
        return []
    if block_size is not None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1; got {block_size!r}.")
        return [(lo, min(lo + block_size, n)) for lo in range(0, n, block_size)]

    out, lo, acc = [], 0, 0
    for i in range(n):
        cost = int(row_nnz[i]) * itemsize
        if acc + cost > _BLOCK_BYTES and i > lo:
            out.append((lo, i))
            lo, acc = i, cost
        else:
            acc += cost
    out.append((lo, n))
    return out
