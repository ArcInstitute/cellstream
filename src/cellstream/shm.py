"""SharedMemory allocation and lifecycle for the parallel read path.

Workers ATTACH and close() shm; only the parent unlinks. Lifecycle:
  parent allocates → fork → workers attach and write → workers close →
  parent constructs AnnData with views → weakref finalizer on AnnData
  unlinks the /dev/shm names (mappings remain valid) → SharedMemory
  python objects naturally GC when no view references them, which then
  munmaps.

Explicit `adata.close_shared_memory()` releases both the /dev/shm name
and the local mapping immediately. Idempotent.
"""

from __future__ import annotations

import mmap
import os
import shutil
import tempfile
import warnings
import weakref
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from cellstream.errors import InsufficientShmError

SHM_DIR = "/dev/shm"
_PAGE = mmap.PAGESIZE  # == os.sysconf("SC_PAGE_SIZE")


def _free_bytes(path) -> int:
    """Free bytes on the filesystem backing ``path`` (statvfs f_bavail —
    blocks available to an unprivileged process). Monkeypatched in tests."""
    return shutil.disk_usage(str(path)).free


def _ceil_page(n: int) -> int:
    """Round n up to a whole memory page (tmpfs charges + mmap faults by page)."""
    return (-(-n // _PAGE)) * _PAGE if n else 0


def _required_bytes(*, total_nnz: int, n_indptr: int, data_dtype, indices_dtype) -> int:
    """Page-rounded byte total of the three CSR buffers (the preflight target)."""
    return (
        _ceil_page(total_nnz * data_dtype.itemsize)
        + _ceil_page(total_nnz * indices_dtype.itemsize)
        + _ceil_page(n_indptr * 8)
    )


def _insufficient_msg(path, required: int, avail: int) -> str:
    return (
        f"out-of-shm preflight: need {required:,} bytes but {path} has {avail:,} "
        f"available. Options: (1) free/grow {path}; (2) pass spill_dir=<real-disk "
        f"directory> with backing='mmap' or 'auto'; (3) use the serial to_anndata() "
        f"(streams shards, no single large buffer)."
    )


def _release_segment(seg) -> None:
    """Best-effort close()+unlink() of ONE SharedMemory. Idempotent.

    close() and unlink() are guarded SEPARATELY and in that order: a BufferError out of
    close() (a live numpy view still exporting segment.buf) must NOT skip the unlink, or the
    /dev/shm name outlives the process. A second call is a no-op plus a swallowed
    FileNotFoundError. Shared by the bundle close_and_unlink methods and the allocate_*
    cleanup arms, so the rule is stated once. (#287)

    The NESTED finally is deliberate. A flat `except Exception` around close() would skip the
    unlink when a KeyboardInterrupt arrives inside it; an `except BaseException` would swallow
    the interrupt itself. This form swallows the ordinary cleanup failures, still unlinks when
    a BaseException passes through, and lets that BaseException propagate -- unless unlink()
    itself raises something other than FileNotFoundError/OSError, in which case that error
    wins and the original survives as __context__.

    Because it CAN re-raise, multi-segment callers must go through _release_segments rather
    than looping over this directly.
    """
    try:
        try:
            seg.close()
        except Exception:
            pass
    finally:
        try:
            seg.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _release_segments(segs) -> None:
    """Release EVERY segment in segs (skipping None), then re-raise the first escaping error.

    A bare `for s in segs: _release_segment(s)` abandons the remaining segments the moment one
    of them lets a BaseException through -- e.g. a KeyboardInterrupt inside the first
    segment's close(). Deferring that error until every handle has been attempted keeps both
    properties: nothing is stranded, and the interrupt is not swallowed. (#287, codex
    checkpoint 3.)
    """
    first = None
    for s in segs:
        if s is None:
            continue
        try:
            _release_segment(s)
        except BaseException as e:
            if first is None:
                first = e
    if first is not None:
        raise first


@dataclass
class ShmBundle:
    data: shared_memory.SharedMemory
    indices: shared_memory.SharedMemory
    indptr: shared_memory.SharedMemory

    def unlink_segments(self) -> None:
        """Remove /dev/shm entries; do NOT close (mappings stay valid). Idempotent."""
        for s in (self.data, self.indices, self.indptr):
            try:
                s.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                # Best-effort: an unexpected OSError is unlikely on Linux but
                # don't tear the world down on close.
                pass

    def close_and_unlink(self) -> None:
        """Full cleanup: munmap THIS process's mapping AND unlink the segment. Idempotent."""
        _release_segments((self.data, self.indices, self.indptr))


@dataclass
class DenseShmBundle:
    """Single-segment shm backing for a dense materialized X (sibling of ShmBundle).

    Duck-types to attach_buffer_cleanup (exposes unlink_segments / close_and_unlink)."""

    array: shared_memory.SharedMemory

    def unlink_segments(self) -> None:
        try:
            self.array.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def close_and_unlink(self) -> None:
        _release_segment(self.array)


@dataclass
class MmapBundle:
    """File-mmap backing (sibling of ShmBundle) for out-of-shm reads.

    Holds the TYPED np.memmap objects (these ARE adata.X.data/.indices/.indptr),
    so close_and_unlink can force-close the mapping on explicit release — parity
    with ShmBundle. The three .bin files live in a unique temp subdir under
    spill_dir so cleanup is a single rmdir after unlinking the files.
    """

    dir: Path
    data: np.memmap
    indices: np.memmap
    indptr: np.memmap

    def unlink_segments(self) -> None:
        """Unlink the 3 files + rmdir the (now-empty) dir; mappings stay valid
        (open mmaps hold the inode). Idempotent. Used by the GC finalizer."""
        for m in (self.data, self.indices, self.indptr):
            try:
                os.unlink(m.filename)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        try:
            os.rmdir(self.dir)
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def close_and_unlink(self) -> None:
        """Full cleanup: munmap THIS process's mappings, then unlink + rmdir.

        WARNING: after this, the arrays (and any adata.X views) are invalid —
        accessing them segfaults. This is the explicit early-release path.
        Idempotent.
        """
        for m in (self.data, self.indices, self.indptr):
            try:
                m._mmap.close()
            except Exception:
                pass
        self.unlink_segments()


def allocate_shm_buffers(*, data_nbytes: int, indices_nbytes: int, indptr_nbytes: int) -> ShmBundle:
    """Allocate three SharedMemory segments atomically — clean up any earlier
    successes if a later allocation fails, for ANY exception type.

    BaseException, not Exception: a KeyboardInterrupt or SystemExit landing between the three
    constructors used to escape the cleanup arms entirely. It unwinds BEFORE the ShmBundle is
    built, so the caller receives nothing and has no handle to unlink — and unlike a dropped
    mapping, a named /dev/shm entry SURVIVES process death, holding tmpfs pages until someone
    removes it by hand. (#287)

    The `return` is INSIDE the try: with it outside, a failure constructing the bundle (or an
    interrupt at that bytecode boundary) stranded all three segments with no owner at all.

    Not fixable here: an interrupt landing INSIDE SharedMemory.__init__, between shm_open and
    that constructor's own `except OSError: self.unlink()` arm, still orphans. That window is
    CPython's and is strictly narrower than the one closed here.
    """
    data = indices = indptr = None
    try:
        data = shared_memory.SharedMemory(create=True, size=data_nbytes)
        indices = shared_memory.SharedMemory(create=True, size=indices_nbytes)
        indptr = shared_memory.SharedMemory(create=True, size=indptr_nbytes)
        return ShmBundle(data=data, indices=indices, indptr=indptr)
    except BaseException:
        # Reverse of creation order, and best-effort PER HANDLE: the old arms chained four bare
        # calls, so a raising data.close() skipped data.unlink() and leaked the name regardless.
        _release_segments((indptr, indices, data))
        raise


def _dense_insufficient_msg(path, required: int, avail: int) -> str:
    return (
        f"out-of-shm preflight (dense): need {required:,} bytes but {path} has "
        f"{avail:,} available. A full-archive dense read is n_obs x n_var x itemsize. "
        f"Options: (1) read a subset/group (sh[a:b]/read_group); (2) use a smaller "
        f"data_dtype (e.g. float16); (3) free/grow {path} or use a bigger node."
    )


def allocate_dense_shm_buffer(*, nbytes: int) -> DenseShmBundle:
    """Allocate a single zero-filled (demand-zero) shm segment for a dense X.

    Runs a page-rounded /dev/shm preflight first, raising InsufficientShmError
    (actionable) instead of a worker SIGBUS deep in the pool."""
    required = _ceil_page(nbytes)
    avail = _free_bytes(SHM_DIR)
    if required > avail:
        raise InsufficientShmError(_dense_insufficient_msg(SHM_DIR, required, avail))
    # Same protected-return shape as allocate_shm_buffers: constructing the bundle outside the
    # region left the segment with no owner if DenseShmBundle raised. (#287, codex checkpoint 2.)
    seg = None
    try:
        seg = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
        return DenseShmBundle(array=seg)
    except BaseException:
        if seg is not None:
            _release_segment(seg)
        raise


def allocate_mmap_buffers(
    *, spill_dir, total_nnz: int, n_indptr: int, data_dtype, indices_dtype
) -> MmapBundle:
    """Allocate three file-backed memmaps in a unique temp subdir under spill_dir.

    np.memmap(mode='w+') ftruncates to full length (sparse — no zero-fill), so
    this does not pre-charge disk; the preflight guards the eventual fill.
    Atomic: rmtree the subdir if any allocation fails.
    """
    d = Path(tempfile.mkdtemp(prefix=f"cellstream-mmap-{os.getpid()}-", dir=str(spill_dir)))
    try:
        data = np.memmap(d / "data.bin", mode="w+", dtype=data_dtype, shape=(total_nnz,))
        indices = np.memmap(d / "indices.bin", mode="w+", dtype=indices_dtype, shape=(total_nnz,))
        indptr = np.memmap(d / "indptr.bin", mode="w+", dtype="int64", shape=(n_indptr,))
        # Inside the try: built outside it, a failure constructing MmapBundle left the spill
        # dir and its three .bin files on disk. (#287, codex checkpoint 2.)
        return MmapBundle(dir=d, data=data, indices=indices, indptr=indptr)
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)
        raise


def select_and_allocate(
    *, backing: str, spill_dir, total_nnz: int, n_indptr: int, data_dtype, indices_dtype
):
    """Validate args, run the page-rounded preflight, allocate the chosen backing.

    Returns ``(bundle, chosen)`` where ``chosen`` is "shm" or "mmap". Raises
    ``ValueError`` for bad backing / missing spill_dir, and
    ``InsufficientShmError`` when the selected store can't hold the projected
    buffers. For backing="auto", tries /dev/shm first and warns before spilling
    to spill_dir.
    """
    if backing not in ("shm", "mmap", "auto"):
        raise ValueError(f"backing must be 'shm', 'mmap', or 'auto'; got {backing!r}")
    if backing in ("mmap", "auto") and spill_dir is None:
        raise ValueError(f"backing={backing!r} requires spill_dir to be set")

    required = _required_bytes(
        total_nnz=total_nnz, n_indptr=n_indptr, data_dtype=data_dtype, indices_dtype=indices_dtype
    )

    def _alloc_shm():
        return allocate_shm_buffers(
            data_nbytes=total_nnz * data_dtype.itemsize,
            indices_nbytes=total_nnz * indices_dtype.itemsize,
            indptr_nbytes=n_indptr * 8,
        )

    def _alloc_mmap():
        return allocate_mmap_buffers(
            spill_dir=spill_dir,
            total_nnz=total_nnz,
            n_indptr=n_indptr,
            data_dtype=data_dtype,
            indices_dtype=indices_dtype,
        )

    if backing == "shm":
        avail = _free_bytes(SHM_DIR)
        if required > avail:
            raise InsufficientShmError(_insufficient_msg(SHM_DIR, required, avail))
        return _alloc_shm(), "shm"

    if backing == "mmap":
        avail = _free_bytes(spill_dir)
        if required > avail:
            raise InsufficientShmError(_insufficient_msg(spill_dir, required, avail))
        return _alloc_mmap(), "mmap"

    # auto: prefer shm, fall back to mmap with a warning.
    shm_avail = _free_bytes(SHM_DIR)
    if required <= shm_avail:
        return _alloc_shm(), "shm"
    warnings.warn(
        f"cellstream: {SHM_DIR} has {shm_avail:,} bytes < required {required:,}; "
        f"spilling to {spill_dir} (disk-backed mmap; slower).",
        stacklevel=2,
    )
    spill_avail = _free_bytes(spill_dir)
    if required > spill_avail:
        raise InsufficientShmError(_insufficient_msg(spill_dir, required, spill_avail))
    return _alloc_mmap(), "mmap"


class _ShmBackedArray(np.ndarray):
    """A trivial np.ndarray subclass that accepts a shm-lifetime attribute.

    A plain ndarray rejects arbitrary attributes, so a backing bundle or segment
    cannot be pinned to one. Constructed DIRECTLY over a segment's buffer, an
    instance of this subclass becomes the buffer owner numpy keeps in the ``.base``
    chain of every derived view — which is what makes the pin outlive the container.
    See ``array_for_shared_memory`` (CSR, #274) and ``dense_array_for_bundle``
    (dense, #182)."""


class _ShmDenseArray(_ShmBackedArray):
    """The dense-X flavour, handed to callers AS the subclass (issue #182).

    A distinct class rather than an alias, so its ``__name__``/repr are unchanged.
    It accepts the ``_cellstream_shm_bundle`` attribute (set by
    ``attach_buffer_cleanup``), tying the bundle's lifetime to the array — so the
    data stays mapped even if the caller extracts ``adata.X`` and drops the AnnData.
    AnnData preserves the subclass and the attribute through construction (verified).
    The CSR path hands out plain-ndarray views instead."""


def dense_array_for_bundle(bundle, shape, dtype, strides=None):
    """Wrap a DenseShmBundle's segment as the BUFFER-OWNING dense array (issue #182).

    Returns an ``_ShmDenseArray`` constructed directly over ``bundle.array.buf``
    with the bundle pinned as ``_cellstream_shm_bundle``. The pin lives on the
    buffer-owning array — the one numpy keeps in the ``.base`` chain of EVERY
    derived view (slicing, ``np.asarray``, AnnData wrapping) — so the backing
    ``SharedMemory`` stays mapped as long as any view is alive and munmaps only
    when the last view drops. Pinning on an OUTER ``.view(_ShmDenseArray)`` is
    stripped by ``np.asarray``, leaving the segment to be munmap'd under a live
    view (use-after-free segfault).

    ``strides`` is forwarded so the wrapper faithfully mirrors the source array's
    layout (the reader's dense X is C-contiguous today, so ``None`` is the common
    case, but passing the source strides is correct for any layout).
    """
    arr = _ShmDenseArray(shape, dtype=dtype, buffer=bundle.array.buf, strides=strides)
    # arr is ALWAYS this _ShmDenseArray subclass (constructed just above), which
    # accepts arbitrary attributes, so the pin cannot fail. It is deliberately NOT
    # wrapped in try/except: a silent guard would mask a failed pin — exactly the
    # use-after-free this helper exists to prevent. Fail loud instead.
    arr._cellstream_shm_bundle = bundle
    return arr


def array_for_shared_memory(segment: shared_memory.SharedMemory, shape, dtype) -> np.ndarray:
    """Wrap ONE SharedMemory segment as a plain ndarray that keeps it mapped (#274).

    CSR sibling of ``dense_array_for_bundle``. Returns a plain ``np.ndarray`` — so
    scipy, AnnData, pyo3 and ``cudaHostRegister`` see exactly the type they see today
    — that is a view of a buffer-owning ``_ShmBackedArray`` holding ``segment``.

    Why the pin is needed: ``np.ndarray(shape, buffer=shm.buf)`` records the underlying
    ``mmap`` in ``.base``, never the ``SharedMemory``, so nothing keeps the segment
    object alive; once its last reference dies, ``SharedMemory.__del__`` → ``close()``
    → ``mmap.close()`` munmaps under any live array (numpy has released its buffer
    export by then, so no ``BufferError`` blocks it).

    Why it works: numpy's base-collapse walks up only while the base is itself an
    ndarray. The owner's base is an ``mmap``, so the collapse stops AT the owner. Every
    derived array — ``np.asarray``, ``.view()``, a basic slice, scipy's CSR attribute
    assignment, AnnData wrapping, the per-shard views ``prefetch._split_batch`` carves —
    therefore keeps the owner and its segment alive.

    The pin is PER-SEGMENT. Pinning a whole ``ShmBundle`` here would let a retained
    800-byte ``indptr`` keep a multi-gigabyte ``data`` segment (and its tmpfs pages)
    resident after the names are unlinked, and would buy nothing: a live container
    already holds the bundle, and a dropped one leaves no route to reach the other
    arrays anyway.

    ``ShmBundle.unlink_segments`` (the GC finalizer) only removes ``/dev/shm`` names,
    which never invalidates a live mapping on Linux. Explicit
    ``bundle.close_and_unlink()`` (i.e. ``X.close_shared_memory()``) still munmaps
    immediately and invalidates every array derived from the segment — that is the
    documented early-release path, unchanged.

    For ``MmapBundle`` this helper is neither needed nor applicable: its ``np.memmap``
    objects already own their mappings.
    """
    # The owner MUST be built directly over segment.buf. Building it over an
    # intermediate ndarray lets numpy collapse PAST it and silently drops the pin --
    # a real prototype of this fix did exactly that and still segfaulted.
    owner = _ShmBackedArray(shape, dtype=dtype, buffer=segment.buf)
    # Deliberately unguarded: _ShmBackedArray always accepts the attribute, and a
    # silent try/except would mask a failed pin -- precisely the use-after-free this
    # helper exists to prevent. Fail loud (mirrors dense_array_for_bundle).
    owner._cellstream_shm_segment = segment
    view = owner.view(np.ndarray)
    if view.base is not owner:
        # numpy collapsed past the owner: the pin is gone and every consumer silently
        # regains the #274 use-after-free. A real raise (not an `assert`) so a -O build
        # cannot drop the check; RuntimeError rather than AssertionError so it is not
        # conflated with a debug-time assertion.
        raise RuntimeError(
            "cellstream: shm pin lost -- numpy collapsed the .base chain past the buffer "
            f"owner (got {type(view.base).__name__}). array_for_shared_memory must "
            "build the owner directly over segment.buf."
        )
    return view


def attach_buffer_cleanup(adata, bundle) -> None:
    """Wire a backing bundle's cleanup to AnnData GC + expose explicit close.

    Works for either ShmBundle or MmapBundle (both expose unlink_segments /
    close_and_unlink). On adata GC: unlink_segments (drop the /dev/shm names or
    the spill files+dir; mappings stay valid for any X views the user kept). On
    explicit adata.close_shared_memory(): full close_and_unlink (after which X is
    invalid — see the bundle docstrings).
    """
    # Strong refs so the bundle (and its mmaps / SharedMemory objects) stay
    # alive as long as the AnnData OR the X matrix is alive.
    adata._cellstream_shm_bundle = bundle
    if hasattr(adata, "X") and adata.X is not None:
        try:
            adata.X._cellstream_shm_bundle = bundle
        except (AttributeError, TypeError):
            # Safety net: a plain numpy ndarray X rejects arbitrary attributes.
            # Shm-backed dense X is wrapped in _ShmDenseArray (which accepts it), so
            # this normally succeeds; the AnnData-level ref + finalizer below cover
            # any other non-attributable X. Without the X-level ref a dense array
            # extracted via `adata.X` (then `del adata`) would lose its backing.
            pass

    adata.close_shared_memory = bundle.close_and_unlink

    weakref.finalize(adata, bundle.unlink_segments)


def attach_buffer_cleanup_matrix(X, bundle) -> None:
    """Pin a backing bundle's lifetime to a bare matrix object (no AnnData).

    Sibling of attach_buffer_cleanup for the per-shard accessor path
    (GroupShard.x() in lazy mode): the returned matrix is a scipy csr_matrix or
    the _ShmDenseArray dense subclass, both of which accept arbitrary attributes.
    On GC of X: unlink_segments ONLY — it removes the /dev/shm names (or the spill
    files); unlinking alone invalidates no live mapping. On explicit
    X.close_shared_memory(): full close_and_unlink, which DOES munmap — after that X
    and everything derived from it are invalid.

    This function pins the bundle to the CONTAINER only. Whether an array EXTRACTED
    from X (``X.data``, or a dense X itself) outlives X depends on how that array was
    built, not on this function:

    - shm-backed CSR arrays come from ``array_for_shared_memory``, which keeps each
      array's own segment alive through its ``.base`` chain (#274) — safe;
    - a shm-backed dense X comes from ``dense_array_for_bundle`` (#182) — safe;
    - ``MmapBundle`` memmaps own their mappings — safe;
    - an ordinary heap array owns its own buffer and is unaffected either way — safe;
    - only a NON-owning view over a bundle segment built WITHOUT those helpers would
      dangle. No current high-level reader object passed to this helper is one; do not
      introduce one. (The LOW-level ``read_v2_to_host`` does still hand back a bare
      dense view, but that never reaches this function — see its own docstring.)
    """
    X._cellstream_shm_bundle = bundle
    X.close_shared_memory = bundle.close_and_unlink
    weakref.finalize(X, bundle.unlink_segments)


def attach_buffer_cleanup_multi(container, pairs) -> None:
    """Pin MULTIPLE backing bundles to one container (multi-matrix shm reads).

    ``pairs`` is an iterable of ``(array, bundle)`` — one per family read into a
    /dev/shm (or mmap) bundle. Two levels of ownership:

    1. Per-matrix pin — each bundle is pinned to its family array exactly as
       ``attach_buffer_cleanup_matrix`` does, so an extracted array (e.g.
       ``adata.layers["counts"]``) keeps its backing mapped after the container
       is dropped (parity with single-matrix shm). A pin that fails because the
       array rejects attributes (a plain ndarray produced by an AnnData copy) is
       tolerated — the container list below still owns the bundle.
    2. Container-level ownership — ``container._cellstream_shm_bundles`` holds strong
       refs to every bundle, ``container.close_shared_memory()`` closes ALL of
       them (idempotent; each ``close_and_unlink`` is idempotent), and a GC
       finalizer unlinks ALL segments.

    Used by ``to_anndata()``/``to_mudata()``'s v4 shm-backed rebuild.
    """
    pairs = list(pairs)
    bundles = [b for _, b in pairs]
    for arr, bundle in pairs:
        try:
            arr._cellstream_shm_bundle = bundle
            arr.close_shared_memory = bundle.close_and_unlink
            weakref.finalize(arr, bundle.unlink_segments)
        except (AttributeError, TypeError):
            # A plain ndarray (AnnData coerced/copied this family) rejects
            # arbitrary attributes; the container-level list still owns the
            # bundle, so nothing leaks or dangles.
            pass
    container._cellstream_shm_bundles = bundles

    def _close_all(_bundles=tuple(bundles)):
        for b in _bundles:
            b.close_and_unlink()

    container.close_shared_memory = _close_all
    weakref.finalize(container, lambda _bs=tuple(bundles): [b.unlink_segments() for b in _bs])


def attach_mudata_close(mdata, adatas) -> None:
    """Give a MuData a single ``close_shared_memory()`` fanning out to modalities.

    MuData holds no shm bundles of its own (its global obs/obsm/uns/obsp/varp are
    heap from the header); its families' bundles live on the per-modality AnnData,
    each already ``attach_buffer_cleanup_multi``-attached and held BY REFERENCE in
    ``mdata.mod``. GC already frees them (each AnnData carries its own finalizer);
    this only adds an explicit combined close for parity with
    ``adata.close_shared_memory()``.
    """
    closers = [a.close_shared_memory for a in adatas if hasattr(a, "close_shared_memory")]

    def _close_all(_closers=tuple(closers)):
        for c in _closers:
            c()

    mdata.close_shared_memory = _close_all


# Back-compat alias (kept: internal callers + any external importers).
attach_shm_cleanup = attach_buffer_cleanup
