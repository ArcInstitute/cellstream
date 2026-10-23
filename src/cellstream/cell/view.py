"""Thin AnnData-like view exposing the .X[rows] -> csr contract (state3's backend).

`.X` accepts everything :meth:`CellStore.gather_rows` accepts -- a 1-D integer array or
array-like, an integer scalar, a 0-D integer array, and a 1-D bool mask of length ``n_obs`` --
plus a ``slice``, which is expanded here because `gather_rows` has no slice handling. Selector
validation and its error types belong to `gather_rows`; this proxy adds the slice case and
otherwise stays out of the way.

Alongside `.X` the view carries what such a consumer reaches for: `obs`, `var`, `var_names`,
`uns`, the annotation matrices, `shape`/`n_obs`/`n_vars`, `isbacked`, and a `file` handle whose
`close()` calls the store's -- enough for a consumer that keeps a bounded cache of open datasets
and evicts through ``adata.file.close()``.

This is deliberately the narrow row-read contract a training dataloader uses, not the whole of
backed AnnData. ``obs_names`` is not provided (``.obs.index`` serves it), and `.X` takes a row
selector only -- there is no two-axis ``X[rows, cols]``.
"""

from __future__ import annotations

import errno

import numpy as np


class _XProxy:
    """``.X[key] -> csr_matrix``.

    Only ``slice`` is handled here -- `gather_rows` has no slice case. Every other selector is
    passed through UNTOUCHED, which is load-bearing rather than lazy: `_normalize_row_ids`
    already inspects the un-promoted array so that a bool mask is recognised as a mask before
    any integer coercion can turn it into ``n_obs`` row ids of value 0/1, promotes a 0-D integer
    array via ``atleast_1d``, and raises ``ValueError``/``TypeError`` with selector-specific
    messages. Re-deriving any of that here would only make this proxy disagree with a direct
    ``gather_rows`` call.

    A slice is materialised into explicit row ids, so ``X[:]`` on a 40 M-row store allocates a
    40 M-element int64 selector (~305 MiB) before `gather_rows` allocates anything of its own.
    Everything here scales with the number of rows SELECTED, not with ``n_obs`` -- a full slice
    is simply the case where those are the same number, and it is the whole-file read that is
    expensive, not slice handling as such. Read in batches (``X[i : i + n]``) when the matrix
    does not fit. Not guarded against: a full row slice is valid, and rejecting it would break
    the contract.
    """

    def __init__(self, store, n_threads: int = 1):
        self._store = store
        # Lower bound only -- `gather_rows` also caps at the selected row count, which is a
        # per-call quantity and cannot be known here. What this buys is that 0 and negatives
        # behave identically to the default, including taking the one-argument call form below,
        # which a bare `int()` would not.
        self._n_threads = max(1, int(n_threads))

    def __getitem__(self, key):
        if isinstance(key, slice):
            # slice.indices clips out-of-range bounds, raises on a zero step, and yields
            # descending ids for a negative step -- all of which gather_rows preserves.
            key = np.arange(*key.indices(self._store.n_obs), dtype=np.int64)
        if self._n_threads == 1:
            # Byte-identical to the pre-n_threads call. `AnnDataView` is public and takes any
            # object with a `gather_rows`, so a duck-typed store written against the old
            # one-argument signature keeps working at the default; only a caller that actually
            # asks for threads has to accept the keyword.
            return self._store.gather_rows(key)
        return self._store.gather_rows(key, n_threads=self._n_threads)


#: The only ``OSError`` errnos a *correct* close can raise: the fd is already closed, or the
#: file is gone. Anything else -- EIO, ENOSPC on a flush -- is a real failure and propagates.
_BENIGN_CLOSE_ERRNOS = frozenset({errno.EBADF, errno.ENOENT})


class _FileShim:
    """``.file.close()`` -- the handle a backed-AnnData consumer closes to release the store.

    Two failures are swallowed, and only because they are reachable on a *correct* close and
    leave the caller nothing to do: `BufferError`, when an in-flight gather still holds an
    exported view of the mmap, and an `OSError` carrying `EBADF` or `ENOENT`. Every other
    `OSError` -- and every other exception, including a missing `close` on a duck-typed store --
    propagates. An eviction path is a bad place to swallow an unexpected failure: a store that
    cannot be released leaks a file handle per eviction, silently and without bound.
    """

    def __init__(self, store):
        self._store = store

    def close(self) -> None:
        try:
            self._store.close()
        except BufferError:
            pass
        except OSError as exc:
            if exc.errno not in _BENIGN_CLOSE_ERRNOS:
                raise


def _format_repr(kind, n_obs, n_vars, path, sections, x_line=None) -> str:
    """anndata's backed ``__repr__`` layout, shared by :class:`CellStore` and :class:`AnnDataView`.

    ``sections`` is an ordered iterable of ``(name, keys)``; an empty ``keys`` prints nothing,
    as anndata skips an empty slot. Keys are rendered with anndata's own idiom,
    ``str(list(keys))[1:-1]`` -- quoted and comma-separated, and it does not fail on a
    non-string label the way ``", ".join`` does. ``x_line``, when given, becomes a trailing
    ``X:`` line; anndata has no equivalent, so it is kept out of the header (#336).
    """
    n_obs = "?" if n_obs is None else n_obs
    n_vars = "?" if n_vars is None else n_vars
    head = f"{kind} with n_obs × n_vars = {n_obs} × {n_vars}"
    if path is not None:
        head += f" backed at {str(path)!r}"
    lines = [head]
    for name, keys in sections:
        keys = list(keys)
        if keys:
            lines.append(f"    {name}: {str(keys)[1:-1]}")
    if x_line is not None:
        lines.append(f"    X: {x_line}")
    return "\n".join(lines)


class AnnDataView:
    def __init__(self, store, n_threads: int = 1):
        self._store = store
        self.X = _XProxy(store, n_threads)
        # Read back rather than normalise a second time: two `int()` calls on the same input
        # cannot disagree today, but two normalisation sites is how they start to.
        self._n_threads = self.X._n_threads
        # One shim for the life of the view, so `view.file is view.file` holds -- anndata's own
        # `.file` is a persistent object, and an identity-sensitive caller should not see a new
        # one on every attribute access.
        self._file = _FileShim(store)

    def __repr__(self):
        # Not `repr(self._store)` with the class name swapped: the store is duck-typed (anything
        # with `gather_rows`), and a foreign store's repr is neither in this layout nor known to
        # be cheap. A CellStore hands over the metadata-only parts it prints itself; any other
        # store shows what it exposes, and nothing is read from it (#336).
        parts = getattr(self._store, "_repr_parts", None)
        if callable(parts):
            n_obs, n_vars, path, sections, x_line = parts()
        else:
            n_obs = getattr(self._store, "n_obs", None)
            n_vars = getattr(self._store, "n_vars", None)
            path, sections, x_line = getattr(self._store, "path", None), (), None
        return _format_repr("AnnDataView", n_obs, n_vars, path, sections, x_line)

    @property
    def obs(self):
        return self._store.obs

    @property
    def var(self):
        return self._store.var

    @property
    def var_names(self):
        return self._store.var.index

    @property
    def uns(self):
        return self._store.uns

    @property
    def obsm(self):
        return self._store.obsm

    @property
    def varm(self):
        return self._store.varm

    @property
    def obsp(self):
        return self._store.obsp

    @property
    def varp(self):
        return self._store.varp

    @property
    def shape(self):
        return self._store.shape

    @property
    def n_obs(self):
        return self._store.n_obs

    @property
    def n_vars(self):
        return self._store.n_vars

    @property
    def isbacked(self) -> bool:
        """Always True: the payload is mmap'd and rows are decoded on demand. There is no eager
        mode for this view -- :meth:`CellStore.to_anndata` rejects ``backed=False``."""
        return True

    @property
    def file(self):
        return self._file
