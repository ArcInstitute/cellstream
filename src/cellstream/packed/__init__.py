"""Single-file packed archive container (issue #91).

Additive on-disk layout that stores a whole v2 directory archive in one
file with an offset-index footer. Shard bytes are byte-identical to the
directory layout; only how members are located on disk changes.
"""

from __future__ import annotations

from cellstream.packed.reader import PackedArchive, is_packed_file  # noqa: E402

__all__ = ["PackedArchive", "is_packed_file"]
