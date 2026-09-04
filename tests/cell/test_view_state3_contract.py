"""AnnDataView's backed-AnnData contract: slice keys, var_names, file, isbacked, n_threads.

`tests/cell/test_reader.py::test_view_contract` covers the pre-existing surface (an ndarray key,
`obs`/`var`, `shape`/`n_obs`/`n_vars`). This file covers the rest, and in particular pins that
`.X` DELEGATES selector handling to `gather_rows` rather than re-deriving it -- a proxy that
validated selectors itself would disagree with a direct `gather_rows` call on 0-D arrays, on
error types, and on bool masks.
"""

from __future__ import annotations

import errno
import os

import numpy as np
import pytest

import cellstream
from cellstream.cell.reader import CellStore
from cellstream.cell.view import AnnDataView
from cellstream.cell.writer import write_cell_archive
from tests.cell.conftest import make_cell_adata


@pytest.fixture
def view_and_ref(tmp_path):
    a = make_cell_adata()
    out = tmp_path / "v.shad"
    write_cell_archive(a, out)
    ref = a.X.tocsr()
    ref.sort_indices()
    return cellstream.open(out).to_anndata(backed=True), ref, a


def _dense(m):
    return np.asarray(m.todense())


# --- slices: the one selector gather_rows does not take -------------------------------------


def test_slice_forward(view_and_ref):
    view, ref, a = view_and_ref
    np.testing.assert_array_equal(_dense(view.X[10:20]), _dense(ref[10:20]))
    np.testing.assert_array_equal(_dense(view.X[:4]), _dense(ref[:4]))
    np.testing.assert_array_equal(_dense(view.X[::7]), _dense(ref[np.arange(0, a.n_obs, 7)]))


def test_slice_negative_bounds(view_and_ref):
    view, ref, _a = view_and_ref
    np.testing.assert_array_equal(_dense(view.X[-3:]), _dense(ref[-3:]))
    np.testing.assert_array_equal(_dense(view.X[-10:-5]), _dense(ref[-10:-5]))


def test_slice_negative_step_preserves_descending_order(view_and_ref):
    """slice.indices yields descending ids for a negative step and gather_rows preserves request
    order, so a reversed slice must come back reversed -- not silently sorted or emptied."""
    view, ref, a = view_and_ref
    np.testing.assert_array_equal(_dense(view.X[::-1]), _dense(ref[np.arange(a.n_obs)[::-1]]))
    np.testing.assert_array_equal(_dense(view.X[20:10:-2]), _dense(ref[np.arange(20, 10, -2)]))


def test_slice_out_of_range_is_clipped(view_and_ref):
    view, ref, a = view_and_ref
    np.testing.assert_array_equal(_dense(view.X[: a.n_obs + 1000]), _dense(ref[:]))
    assert view.X[a.n_obs + 5 : a.n_obs + 9].shape == (0, a.n_vars)


def test_slice_empty(view_and_ref):
    view, _ref, a = view_and_ref
    assert view.X[5:5].shape == (0, a.n_vars)


def test_slice_zero_step_raises(view_and_ref):
    view, _ref, _a = view_and_ref
    with pytest.raises(ValueError):
        view.X[::0]


# --- everything else must behave EXACTLY as a direct gather_rows call -----------------------


@pytest.mark.parametrize(
    "key",
    [
        np.array([5, 2, 2, 9]),  # 1-D int array, unsorted + duplicated
        [5, 2, 9],  # list
        (5, 2, 9),  # tuple
        9,  # Python int scalar
        np.int64(9),  # numpy integer scalar
        # 0-D integer array: atleast_1d promotes it. A proxy-side ndim check rejected this,
        # which is the regression that made the first version of _XProxy non-additive.
        np.array(9),
    ],
)
def test_selector_matches_direct_gather(view_and_ref, key):
    view, _ref, _a = view_and_ref
    np.testing.assert_array_equal(_dense(view.X[key]), _dense(view._store.gather_rows(key)))


def test_bool_mask_is_a_mask_not_row_ids(view_and_ref):
    """A length-n_obs mask must be recognised as a mask. Coercing it to int64 would select
    n_obs rows of id 0/1 -- a silently wrong read rather than an error."""
    view, ref, a = view_and_ref
    mask = np.zeros(a.n_obs, dtype=bool)
    mask[[2, 40, 100]] = True
    got = _dense(view.X[mask])
    assert got.shape[0] == 3, "a mask must select its True positions, not n_obs rows"
    np.testing.assert_array_equal(got, _dense(ref[mask]))


@pytest.mark.parametrize(
    ("key", "exc"),
    [
        (np.array([[1, 2], [3, 4]]), ValueError),  # 2-D integer array
        (np.zeros((2, 3), dtype=bool), ValueError),  # 2-D bool -> malformed mask
        (np.array([True, False]), ValueError),  # wrong-length mask
        (True, ValueError),  # bool scalar is a malformed mask, not row 1
        (np.array([1.5, 2.5]), TypeError),  # non-integer dtype
    ],
)
def test_bad_selectors_raise_the_same_error_as_gather_rows(view_and_ref, key, exc):
    """The proxy must not substitute its own exception type for gather_rows'."""
    view, _ref, _a = view_and_ref
    with pytest.raises(exc):
        view.X[key]
    with pytest.raises(exc):
        view._store.gather_rows(key)


# --- the rest of the contract ----------------------------------------------------------------


def test_var_names(view_and_ref):
    view, _ref, a = view_and_ref
    assert list(view.var_names) == list(a.var_names)


def test_isbacked(view_and_ref):
    view, _ref, _a = view_and_ref
    assert view.isbacked is True


class _SpyStore:
    """Minimal stand-in recording what the proxy forwards."""

    n_obs = 10
    n_vars = 3

    def __init__(self):
        self.calls = []
        self.closed = 0

    def gather_rows(self, row_ids, n_threads=1, out_dtype=None):
        self.calls.append((row_ids, n_threads))
        return "csr"

    def close(self):
        self.closed += 1


def test_file_close_actually_closes_the_store():
    """A blanket-suppressing shim would pass a no-exception-escaped test even if close() were
    missing entirely, so assert the underlying call happened."""
    spy = _SpyStore()
    AnnDataView(spy).file.close()
    assert spy.closed == 1


def test_file_close_is_repeatable():
    spy = _SpyStore()
    v = AnnDataView(spy)
    v.file.close()
    v.file.close()
    assert spy.closed == 2


def test_file_suppresses_a_close_racing_an_exported_mmap():
    class _Busy(_SpyStore):
        def close(self):
            raise BufferError("cannot close exported pointers exist")

    AnnDataView(_Busy()).file.close()  # must not raise


@pytest.mark.parametrize("code", [errno.EBADF, errno.ENOENT])
def test_file_suppresses_a_close_of_an_already_gone_file(code):
    """The other half of the narrowed suppression. An eviction path closing a store whose fd is
    already closed, or whose file has been unlinked, must not take the consumer down with it."""

    class _Gone(_SpyStore):
        def close(self):
            raise OSError(code, os.strerror(code))

    AnnDataView(_Gone()).file.close()  # must not raise


@pytest.mark.parametrize("code", [errno.EIO, errno.ENOSPC, None])
def test_file_does_not_swallow_an_oserror_that_means_something(code):
    """`suppress(OSError)` would hide these. EIO or ENOSPC on close is a real failure, and an
    OSError with NO errno carries no evidence that it is benign -- neither should vanish just
    because they share a base class with EBADF."""

    class _Broken(_SpyStore):
        def close(self):
            raise OSError(code, "boom") if code is not None else OSError("boom")

    with pytest.raises(OSError):
        AnnDataView(_Broken()).file.close()


def test_file_does_not_swallow_an_unexpected_error():
    class _Broken(_SpyStore):
        def close(self):
            raise RuntimeError("something genuinely wrong")

    with pytest.raises(RuntimeError):
        AnnDataView(_Broken()).file.close()


def test_file_is_a_stable_object(view_and_ref):
    """anndata's `.file` is persistent; an identity-sensitive caller must not get a new shim on
    every attribute access."""
    view, _ref, _a = view_and_ref
    assert view.file is view.file


def test_n_threads_is_actually_forwarded():
    """Assert the value reaches gather_rows. Comparing results would not: they are identical
    whether or not threading is honoured, so a proxy that stored 4 and always passed 1 would
    pass a results-only test.

    The default row is weaker on purpose and says so: at n_threads=1 the proxy omits the
    keyword entirely (see the legacy-store test), so the 1 recorded here is the STORE's own
    default, not a value the view sent. What it pins is that the effective thread count at the
    default is 1 either way."""
    spy = _SpyStore()
    AnnDataView(spy, n_threads=4).X[np.array([1, 2])]
    assert spy.calls[-1][1] == 4
    default = _SpyStore()
    AnnDataView(default).X[np.array([1, 2])]
    assert default.calls[-1][1] == 1


def test_to_anndata_actually_forwards_n_threads():
    """`to_anndata` is the only public way to set this, and a results-only test cannot check it:
    a serial and a threaded gather return the same matrix, so `AnnDataView(self)` -- dropping
    n_threads on the floor -- would pass one. Call to_anndata unbound on the spy instead and
    read the value off the gather the view issues."""
    spy = _SpyStore()
    CellStore.to_anndata(spy, backed=True, n_threads=4).X[np.array([1, 2])]
    assert spy.calls[-1][1] == 4


def test_n_threads_forwarded_through_to_anndata(tmp_path):
    """Parity only: threading must not change the answer. See the spy test above for whether
    the value is forwarded at all."""
    a = make_cell_adata()
    out = tmp_path / "t.shad"
    write_cell_archive(a, out)
    rows = np.arange(0, a.n_obs, 3, dtype=np.int64)
    one = cellstream.open(out).to_anndata(backed=True)
    many = cellstream.open(out).to_anndata(backed=True, n_threads=4)
    np.testing.assert_array_equal(_dense(one.X[rows]), _dense(many.X[rows]))


@pytest.mark.parametrize("bad", [0, -1, -8])
def test_a_nonsensical_thread_count_behaves_exactly_like_the_default(bad):
    """`gather_rows` clamps with `max(1, min(n_threads, n_rows))`, so 0 and negatives are
    already 1 there. The view clamps the same way rather than passing them through, which also
    keeps them on the one-argument call form -- a bare `int()` would have sent `n_threads=0` as
    a keyword and broken a legacy store on an input that means nothing."""
    spy = _SpyStore()
    view = AnnDataView(spy, n_threads=bad)
    view.X[np.array([1, 2])]
    assert spy.calls[-1][1] == 1
    assert view._n_threads == 1, "the attribute must report what is actually used"


def test_the_view_and_its_proxy_cannot_disagree_about_n_threads():
    for n in (1, 2, 4, 0, -3):
        view = AnnDataView(_SpyStore(), n_threads=n)
        assert view._n_threads == view.X._n_threads


def test_a_legacy_one_argument_store_still_works_at_the_default():
    """`AnnDataView` is publicly exported and duck-typed, so a store written against the
    pre-n_threads `gather_rows(row_ids)` signature must keep working. The default path calls
    the store exactly as it did before; only n_threads != 1 sends the keyword."""

    class _LegacyStore:
        n_obs, n_vars = 10, 3

        def __init__(self):
            self.seen = []

        def gather_rows(self, row_ids):  # no n_threads, as before this change
            self.seen.append(row_ids)
            return "csr"

    legacy = _LegacyStore()
    view = AnnDataView(legacy)
    assert view.X[np.array([1, 2])] == "csr"
    assert view.X[2:5] == "csr", "the slice path must also use the one-argument call"
    assert len(legacy.seen) == 2

    with pytest.raises(TypeError):
        AnnDataView(legacy, n_threads=4).X[np.array([1, 2])]


def test_defaults_are_unchanged(view_and_ref):
    """A caller written against the old signature gets exactly the old behaviour."""
    view, ref, _a = view_and_ref
    rows = np.array([1, 7, 3])
    np.testing.assert_array_equal(_dense(view.X[rows]), _dense(ref[rows]))
    assert view.X[rows].dtype == np.float32
