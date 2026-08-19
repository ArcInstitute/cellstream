"""CellStore gather parity vs a scipy reference + the state3 view contract."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import cellstream
from cellstream.cell.writer import write_cell_archive
from cellstream.errors import IncompatibleSchemaError
from tests.cell.conftest import make_cell_adata


@pytest.fixture
def written(tmp_path):
    a = make_cell_adata()
    out = tmp_path / "u.shad"
    write_cell_archive(a, out)
    ref = a.X.tocsr()
    ref.sort_indices()
    return out, ref, a


def _assert_rows_equal(got, ref_csr, row_ids):
    exp = ref_csr[np.asarray(row_ids)]
    got = got.tocsr()
    got.sort_indices()
    exp = exp.tocsr()
    exp.sort_indices()
    np.testing.assert_array_equal(got.indptr, exp.indptr)
    np.testing.assert_array_equal(got.indices, exp.indices)
    np.testing.assert_array_equal(got.data, exp.data.astype(np.float32))


def test_gather_contiguous(written):
    out, ref, a = written
    store = cellstream.open(out)
    rows = [0, 1, 2, 3, 4]
    got = store.gather_rows(rows)
    assert isinstance(got, sp.csr_matrix)
    assert got.dtype == np.float32
    assert got.indices.dtype == np.int32
    assert got.shape == (5, a.n_vars)
    _assert_rows_equal(got, ref, rows)


def test_gather_scattered_dup_unsorted(written):
    out, ref, a = written
    store = cellstream.open(out)
    rng = np.random.default_rng(1)
    rows = rng.integers(0, a.n_obs, size=40).tolist()
    rows += [rows[0], rows[0]]  # dups
    got = store.gather_rows(rows)
    _assert_rows_equal(got, ref, rows)


def test_gather_rows_rust_matches_python(tmp_path, cell_adata, monkeypatch):
    import numpy as np

    import cellstream
    from cellstream.cell._rust_dispatch import rust_available

    if not rust_available():
        import pytest

        pytest.skip("Rust core not built/enabled")
    p = tmp_path / "cell.shad"
    cellstream.write_sharded(cell_adata, p, layout="cell")
    rows = np.array([3, 3, 0, 239, 17], dtype=np.int64)

    store = cellstream.open(p)
    got = store.gather_rows(rows).toarray()
    store.close()
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    store2 = cellstream.open(p)
    want = store2.gather_rows(rows).toarray()
    store2.close()
    np.testing.assert_array_equal(got, want)


def test_view_contract(written):
    out, ref, a = written
    view = cellstream.open(out).to_anndata(backed=True)
    rows = np.array([10, 3, 3, 200 % a.n_obs])
    got = view.X[rows]
    _assert_rows_equal(got, ref, rows)
    assert view.n_obs == a.n_obs and view.n_vars == a.n_vars
    assert view.shape == (a.n_obs, a.n_vars)
    assert "target_gene" in view.obs.columns
    assert list(view.var.index) == list(a.var_names)


def test_out_of_range_row_raises(written):
    out, ref, a = written
    store = cellstream.open(out)
    with pytest.raises(IndexError):
        store.gather_rows([a.n_obs])


def test_open_rejects_non_cell_archive(sample_adata_path):
    with pytest.raises(IncompatibleSchemaError):
        cellstream.open(sample_adata_path)


def test_shardedarchive_rejects_cell(written):
    from cellstream import ShardedArchive

    out, ref, a = written
    with pytest.raises(IncompatibleSchemaError, match="cellstream.open"):
        ShardedArchive(out)


@pytest.fixture
def sample_adata_path(tmp_path):
    from cellstream.write import write_sharded

    a = make_cell_adata()
    p = tmp_path / "shard.shad"
    write_sharded(a, p, format="v2", n_shards=2)
    return p


def test_gather_noncontiguous_row_ids(written):
    # X[rows] with a non-contiguous index array (a reversed/strided ndarray view) must work and
    # match the reference under BOTH engines. Regression for "row_ids must be contiguous" — the
    # Rust decode_cells reads row_ids via as_slice() (needs C-contiguous); every prior gather test
    # passed a Python list or fresh np.array (contiguous), so the non-contiguous ndarray path (as
    # produced by the SLURM bench's dup+unsorted draw) was uncovered.
    out, ref, a = written
    store = cellstream.open(out)
    rows = np.arange(a.n_obs, dtype=np.int64)[::-1]  # reversed view -> NOT C-contiguous
    assert not rows.flags["C_CONTIGUOUS"]
    got = store.gather_rows(rows)
    _assert_rows_equal(got, ref, rows)
