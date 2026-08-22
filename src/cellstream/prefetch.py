"""Decode-ahead (prefetch) streaming over group shards (#159).

A single background producer thread decodes batches of consecutive non-reference
shards via the existing parallel reader and hands per-shard AnnData views to the
consumer through a bounded queue. This module must NOT import cellstream.read (no
import cycle): it takes the archive + plain plan tuples and yields
(plan, _DecodedShard) — a carrier holding the raw matrix with deferred AnnData.
"""

from __future__ import annotations

import queue
import threading
import weakref
from typing import NamedTuple

import anndata as ad
import scipy.sparse as sp


class _ShardPlan(NamedTuple):
    shard_index: int
    global_start: int
    global_stop: int
    local_groups: dict  # label -> shard-local slice


class _DecodedShard:
    """A pre-decoded shard whose AnnData construction is deferred.

    Holds the raw per-shard X matrix (``.x``) plus the references needed to build
    the per-shard AnnData lazily. Consumers that only need the matrix
    (``GroupShard.x()``) pay no AnnData construction cost; ``.to_anndata()``
    builds — and caches — the wrapper on first use, byte-identically to a
    per-shard read. The matrix and the lazily-built AnnData both pin the parent
    batch (``_cellstream_batch_ref``) so the batch's shm/heap buffers outlive them.
    """

    __slots__ = ("x", "_batch_adata", "_lo", "_hi", "_anndata")

    def __init__(self, x, batch_adata, lo, hi):
        self.x = x
        self._batch_adata = batch_adata
        self._lo = lo
        self._hi = hi
        self._anndata = None

    def to_anndata(self):
        if self._anndata is None:
            b = self._batch_adata
            sub_ad = ad.AnnData(
                X=self.x,
                obs=b.obs.iloc[self._lo : self._hi],
                var=b.var,
                uns=b.uns,
                obsm={k: v[self._lo : self._hi] for k, v in b.obsm.items()},
                varm=b.varm,
            )
            sub_ad._cellstream_batch_ref = b
            self._anndata = sub_ad
        return self._anndata


def _form_batches(plans: list[_ShardPlan], n_workers: int) -> list[list[_ShardPlan]]:
    """Split ascending-shard-index plans into batches of consecutive shard
    indices, each at most ``n_workers`` long. A non-consecutive index (an
    interior reference shard gap) starts a new batch so the contiguous-range
    decode never spans the gap."""
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    batches: list[list[_ShardPlan]] = []
    run: list[_ShardPlan] = []
    for plan in plans:
        if run and plan.shard_index == run[-1].shard_index + 1 and len(run) < n_workers:
            run.append(plan)
        else:
            if run:
                batches.append(run)
            run = [plan]
    if run:
        batches.append(run)
    return batches


def _split_batch(batch_adata, batch_plans, batch_global_start, *, pinned=False):
    """Slice a decoded batch (covering global rows from ``batch_global_start``)
    into one ``_DecodedShard`` per shard, byte-identically to a per-shard read.

    The split is **zero-copy and dtype-preserving**: each per-shard CSR is built
    from numpy *views* into the batch's ``data``/``indices`` arrays (so the native
    on-disk dtypes — e.g. uint16 indices for a narrow ``cast_back=False`` read —
    survive; scipy's own row-slicing would canonicalize them to int32). AnnData
    construction is **deferred** to ``_DecodedShard.to_anndata()``; the parent
    ``batch_adata`` is pinned to each per-shard matrix (``_cellstream_batch_ref``) so
    its shm-backed buffers stay alive for as long as the matrix is held.
    """
    Xb = batch_adata.X
    is_sparse = sp.issparse(Xb)
    if is_sparse:
        Xb = Xb.tocsr()  # no-op (returns self) when already CSR
        if pinned:
            # Approach 1 (#160): page-lock the batch's three contiguous CSR arrays
            # once; every per-shard view built below inherits the pinned backing.
            # Tie the idempotent unregister to each BASE array via weakref.finalize:
            # numpy clears weakrefs BEFORE freeing the buffer, and a held view /
            # DLPack tensor keeps the base array alive, so unregister is deferred
            # until no view remains (the #182-class unregister-while-viewed hazard).
            from cellstream.pin import pin_array_inplace

            for _arr in (Xb.data, Xb.indices, Xb.indptr):
                _closer = pin_array_inplace(_arr)
                if _closer is not None:
                    weakref.finalize(_arr, _closer)
    parts = []
    for plan in batch_plans:
        lo = plan.global_start - batch_global_start
        hi = plan.global_stop - batch_global_start
        if is_sparse:
            p0, p1 = int(Xb.indptr[lo]), int(Xb.indptr[hi])
            # Assign the sliced arrays as attributes rather than via the
            # sp.csr_matrix((data, indices, indptr), ...) constructor: the
            # constructor runs get_index_dtype, which canonicalizes indices to
            # int32 and indptr to int32 — silently widening a narrow read's
            # native uint16 indices / int64 indptr and breaking byte-identity.
            # This mirrors read._x_from_result, which assigns for the same reason.
            sub = sp.csr_matrix((hi - lo, Xb.shape[1]), dtype=Xb.dtype)
            sub.data = Xb.data[p0:p1]  # view — preserves data dtype
            sub.indices = Xb.indices[p0:p1]  # view — preserves index dtype
            sub.indptr = Xb.indptr[lo : hi + 1] - Xb.indptr[lo]  # rebased to 0
            # Pin the batch (→ shm bundle) to the matrix object so extracting it
            # (x = shard.x()) and discarding the GroupShard still keeps the shm
            # buffers alive (mirrors shm.attach_buffer_cleanup, which pins to both
            # the AnnData and its .X). For the dense branch the view's .base chain
            # already holds the shm-backed parent array alive.
            sub._cellstream_batch_ref = batch_adata
            x_sub = sub
        else:
            # ndarray row-slice view. Pin the batch to the view too (it is always a
            # _ShmDenseArray, which accepts attributes): the .base chain keeps the
            # parent BUFFER alive but NOT the DenseShmBundle that owns the shm
            # mapping (output_backing="shm" / the pure-Python fallback). Without
            # this pin, dropping the GroupShard lets batch_adata — and its bundle —
            # GC, whose SharedMemory.__del__ munmaps the buffer x_sub still views
            # (use-after-free / segfault). Mirrors the sparse branch above.
            x_sub = Xb[lo:hi]
            try:
                x_sub._cellstream_batch_ref = batch_adata
            except (AttributeError, TypeError):
                # A plain numpy.ndarray (e.g. a heap-backed batch, or a custom/mock
                # AnnData in tests) rejects arbitrary attributes — and needs no shm
                # pin: its .base chain keeps the heap buffer alive. Only the
                # _ShmDenseArray (shm-backed) case both accepts the attr and needs
                # it. Mirrors shm.attach_buffer_cleanup's guarded X assignment.
                pass
        parts.append((plan, _DecodedShard(x_sub, batch_adata, lo, hi)))
    return parts


def iter_prefetched(archive, plans, *, prefetch, n_workers, decode_kwargs, pinned=False):
    """Yield ``(plan, _DecodedShard)`` per non-reference shard in shard order,
    decoding ahead. One producer thread decodes batches of up to ``n_workers``
    consecutive shards via ``archive[start:stop].to_anndata(n_workers=…, **decode_kwargs)``
    and pushes per-shard carriers onto a bounded queue (``maxsize=prefetch``)."""
    batches = _form_batches(plans, n_workers)
    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
    stop = threading.Event()
    err: list[BaseException] = []
    _DONE = object()

    def _produce():
        try:
            for batch in batches:
                if stop.is_set():
                    return
                bstart, bstop = batch[0].global_start, batch[-1].global_stop
                batch_adata = archive[bstart:bstop].to_anndata(n_workers=n_workers, **decode_kwargs)
                for item in _split_batch(batch_adata, batch, bstart, pinned=pinned):
                    # Block for space but wake periodically to honor `stop`.
                    while not stop.is_set():
                        try:
                            q.put(item, timeout=0.25)
                            break
                        except queue.Full:
                            continue
                    if stop.is_set():
                        return
        except BaseException as e:  # surface on the consumer thread
            err.append(e)
        finally:
            # Reliably deliver the end-of-stream sentinel so a slow but normally
            # draining consumer always terminates: retry until the queue has room.
            # Bail only if the consumer aborted early (stop set). Do NOT set stop
            # here — that is the consumer-finally's job; setting it would skip the
            # sentinel and hang the consumer on q.get() (Gemini #176 HIGH).
            while not stop.is_set():
                try:
                    q.put(_DONE, timeout=0.25)
                    break
                except queue.Full:
                    continue

    producer = threading.Thread(target=_produce, name="cellstream-prefetch", daemon=True)
    producer.start()
    try:
        while True:
            item = q.get()
            if item is _DONE:
                break
            yield item
    finally:
        # Signal stop and join with a timeout. The producer's q.put has a 0.25s
        # timeout that rechecks `stop`, so a producer parked on a full queue
        # exits within ~0.25s of stop.set() (join returns as soon as it dies).
        # The timeout only caps the rare case where the producer is mid-decode
        # inside to_anndata (uninterruptible); the daemon thread then finishes
        # in the background rather than blocking the consumer's early break.
        stop.set()
        producer.join(timeout=2.0)
    if err:
        raise err[0]
