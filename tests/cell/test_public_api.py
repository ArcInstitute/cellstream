"""write_sharded(layout='cell') end-to-end + guardrails."""

from __future__ import annotations

import numpy as np
import pytest

import cellstream
from cellstream.write import write_sharded
from tests.cell.conftest import make_cell_adata


def test_write_sharded_cell_end_to_end(tmp_path):
    a = make_cell_adata()
    out = tmp_path / "cell.shad"
    with pytest.warns(UserWarning, match="experimental"):
        write_sharded(a, out, layout="cell")  # ungrouped: storage order == input order
    store = cellstream.open(out)
    ref = a.X.tocsr()
    ref.sort_indices()
    rows = np.array([0, 7, 7, a.n_obs - 1])
    got = store.to_anndata(backed=True).X[rows].tocsr()
    exp = ref[rows].tocsr()
    got.sort_indices()
    exp.sort_indices()
    np.testing.assert_array_equal(got.indices, exp.indices)
    np.testing.assert_array_equal(got.data, exp.data.astype(np.float32))


def test_write_sharded_cell_grouped_passes_through(tmp_path):
    # group_by/reference must reach write_cell_archive: the store is grouped and has a
    # reference block. (Grouped X[rows] is storage-order by design, so this checks the
    # param pass-through rather than input-order addressing.)
    a = make_cell_adata()
    out = tmp_path / "cellg.shad"
    with pytest.warns(UserWarning, match="experimental"):
        write_sharded(a, out, layout="cell", group_by="target_gene", reference=["t00"])
    store = cellstream.open(out)
    assert set(store.group_labels()) == set(np.unique(a.obs["target_gene"].to_numpy()))
    assert store.read_reference() is not None


def test_write_sharded_default_layout_unchanged(tmp_path, recwarn):
    a = make_cell_adata()
    out = tmp_path / "shard.shad"
    write_sharded(a, out, format="v2", n_shards=2)
    # a normal shard archive still opens via ShardedArchive
    assert cellstream.ShardedArchive(out).schema_version in (2, 3)


@pytest.mark.parametrize(
    "kw",
    [
        {"n_shards": 2},
        {"target_shard_bytes": 8192},
        {"write_nthreads": 4},
    ],
)
def test_cell_rejects_shard_only_params(tmp_path, kw):
    a = make_cell_adata()
    with pytest.raises(ValueError, match="layout='cell'"):
        write_sharded(a, tmp_path / "x.shad", layout="cell", **kw)


def test_invalid_layout_rejected(tmp_path):
    a = make_cell_adata()
    with pytest.raises(ValueError, match="layout"):
        write_sharded(a, tmp_path / "x.shad", layout="bogus")


def test_cell_flags_pass_through(tmp_path):
    from cellstream.packed.reader import PackedArchive

    out = tmp_path / "c.shad"
    with pytest.warns(UserWarning, match="experimental"):
        write_sharded(make_cell_adata(), out, layout="cell", payload_checksum=True)
    man = PackedArchive(out).manifest
    assert man["payload_sha256"] is not None
    assert man["indices_canonical"] is True


@pytest.mark.parametrize("flag", ["payload_checksum", "require_unique_indices"])
def test_cell_flags_rejected_for_shard(tmp_path, flag):
    a = make_cell_adata()
    with pytest.raises(ValueError, match="cell-only"):
        write_sharded(a, tmp_path / "s.shad", format="v2", n_shards=2, **{flag: True})


def test_write_sharded_streams_a_path_source(tmp_path):
    import cellstream
    from cellstream.cell.writer import write_cell_archive

    from .conftest import assert_archives_equivalent, make_cell_adata

    a = make_cell_adata()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    mat, stream = tmp_path / "m.shad", tmp_path / "s.shad"
    write_cell_archive(a, mat, group_by="target_gene")
    with pytest.warns(UserWarning, match="experimental"):
        cellstream.write_sharded(src, stream, layout="cell", group_by="target_gene", block_size=11)
    assert_archives_equivalent(mat, stream)


def test_block_size_rejected_for_shard_layout(tmp_path):
    import cellstream

    from .conftest import make_cell_adata

    a = make_cell_adata()
    with pytest.raises(ValueError, match="cell-only"):
        cellstream.write_sharded(a, tmp_path / "o.shad", n_shards=2, block_size=100)


def test_cell_exports_writer_and_concat():
    import cellstream.cell as c

    assert hasattr(c, "CellArchiveWriter")
    assert hasattr(c, "concat_cell_archives")
    assert "CellArchiveWriter" in c.__all__
    assert "concat_cell_archives" in c.__all__
