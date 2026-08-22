"""cellstream exception hierarchy.

Every exception class defined in this module inherits from CellstreamError, so a
caller can catch the package-specific failures uniformly. Ordinary argument
validation elsewhere still raises the built-ins (ValueError, TypeError,
FileNotFoundError); CellstreamError is not a catch-all for everything the package
can raise.
"""

from __future__ import annotations

from typing import NoReturn

#: The one true statement about a legacy v1/directory archive, shared by every message that has
#: to make it -- :data:`PACKED_ONLY_MIGRATION` here and ``write.py``'s ``format`` guard.
# NOTE it names NO install command, deliberately (#311). The pre-#311 text told the reader to
# pip-install a git URL for the pre-rename repository at tag v0.4.0 -- an instruction that cannot
# work for the audience reading it, because that repository is not public and v0.4.0 was never
# uploaded to PyPI either. `pip install 'cellstream<0.5'` is equally impossible: nothing was ever
# published under this name that early. R1 ("recognize and reject" rather than degrade to
# FileNotFoundError) was chosen so a caller holding a legacy archive gets something ACTIONABLE --
# so name the remedy that exists, re-encoding from the source, rather than a URL that 404s. The
# seed gate's `non-public ArcInstitute repo url` SANITIZE arm (#312) refuses to seed the public
# repo if one comes back -- and `tests/test_packed_only.py` fails on every push, which is the arm
# that runs without anyone remembering to: CI does not invoke the seed gate.
LEGACY_V1_REMEDY = (
    "no installable release can read a legacy v1/directory archive -- the last one that could, "
    "v0.4.0, predates this project's rename and was never published to PyPI. Re-encode from the "
    "original source data with the current writer instead."
)

#: Appended to every PackedOnlyError message by :func:`reject_non_packed`.
PACKED_ONLY_MIGRATION = (
    " Since v0.5 only the single-file packed container is supported (SHPK head magic, "
    "conventionally .shad); " + LEGACY_V1_REMEDY
)


def reject_non_packed(reason: str) -> NoReturn:
    """Reject a non-packed archive with the legacy migration guidance."""
    raise PackedOnlyError(reason + PACKED_ONLY_MIGRATION)


class CellstreamError(Exception):
    """Base for the package-specific exception classes defined in this module."""


class ArchiveLockedError(CellstreamError):
    """Another process holds the write lock on the archive."""


class IncompatibleSchemaError(CellstreamError):
    """The manifest.json schema_version is not supported by this cellstream version."""


class LossyCastError(CellstreamError):
    """A dtype cast (e.g. float32 → uint16) would lose information."""


class ArchiveExistsError(CellstreamError):
    """The write target already exists and ``overwrite=False``.

    A packed ``.shad`` file for the public writers; a directory holding a
    ``manifest.json`` for the internal staging writer.
    """


class GpuOutOfMemoryError(CellstreamError):
    """Pre-flight projected memory exceeds the device's effective free
    memory. Raised before any cupy/CUDA allocation.

    The message is expected to include:
      - projected total bytes
      - effective free bytes (physical free + pool unused)
      - a suggestion: use to_anndata() (CPU) or sh[a:b].to_cupy_anndata().
    """


class InsufficientShmError(CellstreamError):
    """Projected CSR buffers exceed the backing store's available space.

    Raised by the to_anndata_shm preflight BEFORE any worker spawns (so the
    failure is an actionable error, not a worker SIGBUS / BrokenProcessPool).
    The message names the path checked (/dev/shm or spill_dir), the required
    and available byte counts, and the remedies (grow the store, pass
    spill_dir with backing='mmap'/'auto', or use the serial to_anndata).
    """


class PackedArchiveError(CellstreamError):
    """A packed (.shad) container is malformed: bad head/trailer magic,
    footer CRC mismatch, truncated file, or a missing required member."""


class PackedOnlyError(CellstreamError):
    """A non-packed archive (v1 h5ad-shard or v2 directory) was passed to a reader.

    Since v0.5 -- a release made under the project's former name -- only the
    single-file packed (SHPK / .shad) container is supported. What to do with a
    legacy archive instead is in :data:`PACKED_ONLY_MIGRATION`, which is the single
    place it is spelled out. There is deliberately no install command to offer: see
    :data:`LEGACY_V1_REMEDY` (#311).

    Read-side only since #154 part B3b: the write side no longer has a container
    to reject, so ``write_sharded(container=...)`` is an ordinary ``TypeError``.
    Raised via :func:`reject_non_packed`, which appends the migration note.
    """


class NonIntegerCountsError(CellstreamError, ValueError):
    """Float values are not non-negative integers, so integer storage is impossible.

    THE ONLY condition on which write_cell_archive retries with zstd float storage. It
    subclasses ValueError so existing callers that catch ValueError keep working.

    Deliberately NOT raised for a value that merely exceeds uint32: zstd-int cannot store
    that either, so it is a genuine error and must not trigger a silent switch to float.
    """
