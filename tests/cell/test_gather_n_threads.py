"""``n_threads`` plumbing for the cell gather path (issue #244 ask #4).

Parallel decode MUST be byte-identical to the serial default, the kwarg must reach the Rust
decoder bounded by the selected-row count, and the group/adata wrappers must forward it.
``n_threads`` only affects the Rust ``decode_cells`` path, so these are Rust-gated.
"""

from __future__ import annotations

import numpy as np
import pytest

import cellstream
from cellstream.cell._rust_dispatch import rust_available
from cellstream.cell.writer import write_cell_archive
from tests.cell.conftest import make_cell_adata

pytestmark = pytest.mark.skipif(
    not rust_available(), reason="n_threads only parallelises the Rust decode path"
)


@pytest.fixture
def ungrouped(tmp_path):
    out = tmp_path / "u.shad"
    write_cell_archive(make_cell_adata(n_obs=512, n_vars=60), out)
    return out


@pytest.fixture
def grouped(tmp_path):
    out = tmp_path / "g.shad"
    write_cell_archive(
        make_cell_adata(n_obs=512, n_vars=60), out, group_by="target_gene", reference=["t00"]
    )
    return out


def test_gather_rows_parallel_matches_serial(ungrouped):
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 512, size=300).tolist()
    store = cellstream.open(ungrouped)
    try:
        serial = store.gather_rows(rows, n_threads=1)
        for nt in (2, 4, 8):
            par = store.gather_rows(rows, n_threads=nt)
            np.testing.assert_array_equal(par.indptr, serial.indptr)
            np.testing.assert_array_equal(par.indices, serial.indices)
            np.testing.assert_array_equal(par.data, serial.data)
    finally:
        store.close()


def test_gather_rows_default_is_serial(ungrouped):
    store = cellstream.open(ungrouped)
    try:
        rows = [0, 5, 5, 100, 511]
        default = store.gather_rows(rows)  # no n_threads -> default 1
        explicit = store.gather_rows(rows, n_threads=1)
        np.testing.assert_array_equal(default.toarray(), explicit.toarray())
    finally:
        store.close()


def test_gather_rows_n_threads_over_rowcount_is_safe(ungrouped):
    store = cellstream.open(ungrouped)
    try:
        rows = [3, 3, 7]  # far fewer rows than requested threads -> nw clamps to rowcount
        got = store.gather_rows(rows, n_threads=64)
        ref = store.gather_rows(rows, n_threads=1)
        np.testing.assert_array_equal(got.toarray(), ref.toarray())
        empty = store.gather_rows([], n_threads=16)  # empty gather + many threads: no crash
        assert empty.shape[0] == 0
    finally:
        store.close()


def test_read_group_forwards_n_threads(grouped):
    store = cellstream.open(grouped)
    try:
        serial = store.read_group("t03")
        par = store.read_group("t03", n_threads=4)
        np.testing.assert_array_equal(par.toarray(), serial.toarray())
    finally:
        store.close()


def test_gather_rows_adata_forwards_n_threads(ungrouped):
    store = cellstream.open(ungrouped)
    try:
        rows = [1, 2, 3, 4, 5]
        par = store.gather_rows_adata(rows, n_threads=4)
        serial = store.gather_rows_adata(rows)
        np.testing.assert_array_equal(par.X.toarray(), serial.X.toarray())
    finally:
        store.close()


def test_gather_rows_promotes_scalar_rejects_2d(ungrouped):
    # scalar promotes to a 1-row gather (matches gather_rows_adata's atleast_1d contract);
    # >1-D raises a clean ValueError instead of a confusing Rust TypeError.
    store = cellstream.open(ungrouped)
    try:
        assert store.gather_rows(5).shape == (1, store.n_vars)
        with pytest.raises(ValueError, match="1-dimensional"):
            store.gather_rows(np.array([[0, 1], [2, 3]]))
    finally:
        store.close()


def test_n_threads_reaches_decoder_through_all_wrappers(grouped, monkeypatch):
    # Prove forwarding (not just equal output): capture the n_threads decode_cells actually
    # receives, for every public entry point. gather_rows re-imports decode_cells from the
    # module each call, so patching the module attribute is picked up.
    import cellstream.cell._rust_dispatch as rd

    real = rd.decode_cells
    seen = []

    def spy(*a, **k):
        seen.append(k["n_threads"])
        return real(*a, **k)

    monkeypatch.setattr(rd, "decode_cells", spy)
    store = cellstream.open(grouped)
    try:
        grp = len(store._group_rows("t03"))
        seen.clear()
        store.gather_rows([0, 1, 2, 3], n_threads=3)
        assert seen == [3]
        seen.clear()
        store.gather_rows_adata([0, 1, 2, 3, 4], n_threads=4)
        assert seen == [4]
        seen.clear()
        store.read_group("t03", n_threads=99)
        assert seen == [min(99, grp)]
        seen.clear()
        store.read_group_adata("t03", n_threads=2)
        assert seen == [min(2, grp)]
        seen.clear()
        store.read_reference(n_threads=99)
        assert seen and seen[0] >= 2
    finally:
        store.close()
