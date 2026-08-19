"""read_group / read_reference / group_labels on a grouped cell archive."""

from __future__ import annotations

import numpy as np
import pytest

import cellstream
from cellstream.cell.writer import write_cell_archive
from tests.cell.conftest import make_cell_adata


@pytest.fixture
def grouped(tmp_path):
    a = make_cell_adata()
    out = tmp_path / "g.shad"
    write_cell_archive(a, out, group_by="target_gene", reference=["t00"])
    ref = a.X.tocsr()
    ref.sort_indices()
    return out, ref, a


def test_group_labels(grouped):
    out, ref, a = grouped
    store = cellstream.open(out)
    labels = set(store.group_labels())
    assert labels == set(np.unique(a.obs["target_gene"]))


def test_read_group_matches_membership(grouped):
    out, ref, a = grouped
    store = cellstream.open(out)
    obs = store.obs
    got = store.read_group("t03")
    # exactly the storage rows whose obs label == t03
    want_rows = np.flatnonzero(obs["target_gene"].to_numpy() == "t03")
    assert got.shape[0] == want_rows.size
    exp = store.gather_rows(want_rows).tocsr()
    got = got.tocsr()
    got.sort_indices()
    exp.sort_indices()
    np.testing.assert_array_equal(got.indices, exp.indices)
    np.testing.assert_array_equal(got.data, exp.data)


def test_read_reference_is_leading_block(grouped):
    out, ref, a = grouped
    store = cellstream.open(out)
    obs = store.obs
    ctrl = store.read_reference()
    n_ref = int((obs["target_gene"].to_numpy() == "t00").sum())
    assert ctrl is not None and ctrl.shape[0] == n_ref


def test_read_reference_none_when_ungrouped(tmp_path):
    a = make_cell_adata()
    out = tmp_path / "u.shad"
    write_cell_archive(a, out)
    assert cellstream.open(out).read_reference() is None


def test_read_reference_multi_label_full_block(tmp_path):
    import cellstream
    from cellstream.cell.writer import write_cell_archive
    from tests.cell.conftest import make_cell_adata

    a = make_cell_adata()
    out = tmp_path / "g2.shad"
    write_cell_archive(a, out, group_by="target_gene", reference=["t00", "t01"])
    store = cellstream.open(out)
    ctrl = store.read_reference()
    labels = a.obs["target_gene"].to_numpy()
    n_ref = int(((labels == "t00") | (labels == "t01")).sum())
    assert ctrl is not None and ctrl.shape[0] == n_ref


def test_read_group_unknown_label_raises(grouped):
    out, ref, a = grouped
    with pytest.raises(KeyError):
        cellstream.open(out).read_group("does-not-exist")
