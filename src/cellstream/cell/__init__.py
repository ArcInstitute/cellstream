"""Flat, shardless per-cell store — cellstream's ``layout="cell"`` access mode.

A single packed ``.shad`` file of per-cell pfordelta frames + an int64 offset
index. Delivers O(1) scattered random access, exact per-group subset reads, and
h5ad-class bulk throughput. Experimental (schema_version 5).
"""

from __future__ import annotations

from .archive_writer import CellArchiveWriter, concat_cell_archives  # noqa: E402
from .codec import PforDeltaCodec  # noqa: E402
from .reader import CellStore, GroupSpan, open_cell  # noqa: E402
from .view import AnnDataView  # noqa: E402
from .writer import write_cell_archive  # noqa: E402

__all__ = [
    "AnnDataView",
    "CellArchiveWriter",
    "CellStore",
    "GroupSpan",
    "PforDeltaCodec",
    "concat_cell_archives",
    "open_cell",
    "write_cell_archive",
]
