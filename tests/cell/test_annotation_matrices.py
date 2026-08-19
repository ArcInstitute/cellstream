"""#229 — obsm/varm/obsp/varp in layout=cell."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

import cellstream
from cellstream.cell.reader import open_cell
from cellstream.cell.source import open_cell_source
from cellstream.cell.writer import write_cell_archive

from .conftest import assert_archives_equivalent, make_cell_adata


def test_sources_expose_annotation_matrices(tmp_path):
    a = make_cell_adata(annotations=True)
    # in-memory
    src = open_cell_source(a)
    assert set(src.obsm) == {"X_pca"} and set(src.varm) == {"PCs"}
    assert set(src.obsp) == {"connectivities", "distances"} and set(src.varp) == {"corr"}
    np.testing.assert_array_equal(src.obsm["X_pca"], a.obsm["X_pca"])
    d = src.obsm  # dict() is a KEY-level copy (values shared, like uns): adding a key must not leak
    d["new_key"] = np.ones((a.n_obs, 5), dtype=np.float32)
    assert "new_key" not in a.obsm
    src.close()
    # backed CSR .h5ad
    p = tmp_path / "csr.h5ad"
    a.write_h5ad(p)
    src = open_cell_source(p)
    np.testing.assert_array_equal(
        src.obsp["connectivities"].toarray(), a.obsp["connectivities"].toarray()
    )
    src.close()
    # dense .h5ad (#225 DenseH5adSource)
    ad_dense = make_cell_adata(annotations=True)
    ad_dense.X = ad_dense.X.toarray().astype(np.float32)
    pd_ = tmp_path / "dense.h5ad"
    ad_dense.write_h5ad(pd_)
    src = open_cell_source(pd_)
    np.testing.assert_array_equal(src.varm["PCs"], ad_dense.varm["PCs"])
    src.close()


def test_writer_stores_permuted_annotations_grouped(tmp_path):
    from cellstream.packed.reader import PackedArchive

    a = make_cell_adata(annotations=True)
    out = tmp_path / "grp.shad"
    write_cell_archive(a, out, group_by="target_gene")
    hdr = PackedArchive(out).header_adata()  # storage order
    expected = a[list(hdr.obs_names)]  # anndata reindex permutes obsm rows + obsp both axes
    np.testing.assert_array_equal(hdr.obsm["X_pca"], expected.obsm["X_pca"])
    np.testing.assert_array_equal(
        hdr.obsp["connectivities"].toarray(), expected.obsp["connectivities"].toarray()
    )
    np.testing.assert_array_equal(hdr.varm["PCs"], a.varm["PCs"])  # feature axis: unpermuted
    np.testing.assert_array_equal(hdr.varp["corr"].toarray(), a.varp["corr"].toarray())


def test_dropped_fields_lists_only_layers(tmp_path):
    a = make_cell_adata(annotations=True)
    a.layers["counts"] = a.X.copy()
    p = tmp_path / "rich.h5ad"
    a.write_h5ad(p)
    src = open_cell_source(p)
    assert set(src.dropped_fields()) == {"layers"}  # obsm/varp no longer dropped
    src.close()


def test_full_read_roundtrips_annotations_ungrouped(tmp_path, engine_env):
    a = make_cell_adata(annotations=True)
    out = tmp_path / "u.shad"
    write_cell_archive(a, out)
    r = cellstream.read_h5ad(str(out))
    np.testing.assert_array_equal(r.obsm["X_pca"], a.obsm["X_pca"])
    np.testing.assert_array_equal(
        r.obsp["connectivities"].toarray(), a.obsp["connectivities"].toarray()
    )
    np.testing.assert_array_equal(r.varm["PCs"], a.varm["PCs"])
    np.testing.assert_array_equal(r.varp["corr"].toarray(), a.varp["corr"].toarray())


def test_full_read_roundtrips_annotations_grouped(tmp_path, engine_env):
    a = make_cell_adata(annotations=True)
    out = tmp_path / "g.shad"
    write_cell_archive(a, out, group_by="target_gene")
    r = cellstream.read_h5ad(str(out))
    expected = a[list(r.obs_names)]  # ground truth in the read-back (storage) order
    np.testing.assert_array_equal(r.obsm["X_pca"], expected.obsm["X_pca"])
    np.testing.assert_array_equal(
        r.obsp["distances"].toarray(), expected.obsp["distances"].toarray()
    )
    np.testing.assert_array_equal(r.varm["PCs"], a.varm["PCs"])


def test_dataframe_obsm_roundtrips(tmp_path, engine_env):
    # a DataFrame obsm entry (a supported type) must survive the .iloc permute + round-trip with
    # its column/index labels intact -- exercises the DataFrame dispatch in permute_cell_axis_fields.
    a = make_cell_adata()
    a.obsm["meta"] = pd.DataFrame({"score": np.arange(a.n_obs, dtype=float)}, index=a.obs_names)
    out = tmp_path / "df.shad"
    write_cell_archive(a, out, group_by="target_gene")
    r = cellstream.read_h5ad(str(out))
    expected = a[list(r.obs_names)].obsm["meta"]  # reindex to storage order
    # anndata slicing (a[names]) returns a DataFrameView subclass; cellstream returns a plain
    # DataFrame. Compare the data/labels/dtypes, not the pandas subclass identity (verified:
    # values, index, columns, dtypes are all byte-identical).
    pd.testing.assert_frame_equal(r.obsm["meta"], expected, check_frame_type=False)


def test_cross_path_equivalence_annotated(tmp_path, engine_env):
    a = make_cell_adata(annotations=True)
    # write the h5ad FIRST: AnnData.write_h5ad() mutates a repeated-string obs column (target_gene)
    # to Categorical IN PLACE (anndata's own strings_to_categoricals side effect) -- writing the
    # in-memory reference (p_mem) from `a` AFTER this keeps its obs dtype consistent with what the
    # backed/dense re-reads see, instead of comparing object vs category for identical values.
    # Matches the established convention already used by every other cross-path test in
    # test_streaming_write.py (e.g. test_streamed_h5ad_equals_materialised writes the h5ad first).
    h = tmp_path / "a.h5ad"
    a.write_h5ad(h)
    p_mem = tmp_path / "mem.shad"
    write_cell_archive(a, p_mem, group_by="target_gene")
    p_csr = tmp_path / "csr.shad"
    write_cell_archive(str(h), p_csr, group_by="target_gene")
    assert_archives_equivalent(p_mem, p_csr)  # now also compares obsm/varm/obsp/varp
    a_dense = a.copy()
    a_dense.X = a_dense.X.toarray().astype(np.float32)
    hd = tmp_path / "dense.h5ad"
    a_dense.write_h5ad(hd)
    p_dense = tmp_path / "dense.shad"
    write_cell_archive(str(hd), p_dense, group_by="target_gene")
    assert_archives_equivalent(p_mem, p_dense)


def test_reencode_preserves_annotations(tmp_path, engine_env):
    a = make_cell_adata(annotations=True)
    p1 = tmp_path / "one.shad"
    write_cell_archive(a, p1, group_by="target_gene")
    p2 = tmp_path / "two.shad"
    write_cell_archive(p1, p2)  # input is a .shad -> CellArchiveSource re-encode
    r1 = cellstream.read_h5ad(str(p1))
    r2 = cellstream.read_h5ad(str(p2))
    np.testing.assert_array_equal(r2.obsm["X_pca"], r1.obsm["X_pca"])  # storage order preserved
    np.testing.assert_array_equal(
        r2.obsp["connectivities"].toarray(), r1.obsp["connectivities"].toarray()
    )
    np.testing.assert_array_equal(r2.varp["corr"].toarray(), r1.varp["corr"].toarray())


def test_read_group_adata_slices_obsm_and_induced_obsp(tmp_path, engine_env):
    a = make_cell_adata(annotations=True)
    out = tmp_path / "g.shad"
    write_cell_archive(a, out, group_by="target_gene")
    store = open_cell(out)
    try:
        label = store.group_labels()[0]
        sub = store.read_group_adata(label)
        full = cellstream.read_h5ad(str(out))
        expected = full[list(sub.obs_names)]  # induced subgraph = full-read then slice
        np.testing.assert_array_equal(sub.obsm["X_pca"], expected.obsm["X_pca"])
        np.testing.assert_array_equal(
            sub.obsp["connectivities"].toarray(), expected.obsp["connectivities"].toarray()
        )
        np.testing.assert_array_equal(sub.varm["PCs"], a.varm["PCs"])  # feature axis: whole
        assert sp.issparse(store.read_group(label))  # bare accessor unchanged
    finally:
        store.close()


def test_gather_rows_adata_induced_subgraph(tmp_path, engine_env):
    a = make_cell_adata(annotations=True)
    out = tmp_path / "u.shad"
    write_cell_archive(a, out)
    store = open_cell(out)
    try:
        rows = np.array([5, 0, 9, 3])
        sub = store.gather_rows_adata(rows)
        np.testing.assert_array_equal(sub.obsm["X_pca"], a.obsm["X_pca"][rows])
        np.testing.assert_array_equal(
            sub.obsp["distances"].toarray(), a.obsp["distances"][rows][:, rows].toarray()
        )
        assert sp.issparse(store.gather_rows(rows))  # bare accessor unchanged
    finally:
        store.close()


def test_gather_rows_adata_accepts_scalar_row_id(tmp_path, engine_env):
    # a scalar row id must be accepted (atleast_1d, not asarray -> a 0-D array): returns a 1-row
    # AnnData carrying that row's obsm, not a 0-D-array indexing crash (Gemini #232).
    a = make_cell_adata(annotations=True)
    out = tmp_path / "scalar.shad"
    write_cell_archive(a, out)  # ungrouped -> storage order == input order, so row 3 == input row 3
    store = open_cell(out)
    try:
        sub = store.gather_rows_adata(3)  # scalar int, not [3]
        assert sub.n_obs == 1
        np.testing.assert_array_equal(sub.obsm["X_pca"], a.obsm["X_pca"][[3]])
    finally:
        store.close()


def test_ungrouped_write_skips_obsm_obsp_permute(tmp_path, engine_env, monkeypatch):
    # #240: an ungrouped (group_by=None) write resolves to an identity perm. The obsm/obsp
    # permute must be short-circuited by passing perm=None, not the arange(n) identity array,
    # so permute_cell_axis_fields does no both-axes reindex. Spy on the helper the writer calls.
    from cellstream.cell import writer as cell_writer

    a = make_cell_adata(annotations=True)
    assert a.obsm and a.obsp  # precondition: the wasteful path only bites non-empty annotations

    real = cell_writer.header.permute_cell_axis_fields
    seen = []

    def spy(obsm, obsp, perm):
        seen.append(perm)
        return real(obsm, obsp, perm)

    monkeypatch.setattr(cell_writer.header, "permute_cell_axis_fields", spy)

    out = tmp_path / "u.shad"
    write_cell_archive(a, out)  # ungrouped
    # exactly one permute call (writer.py:1224), and it short-circuited with perm=None. Assert
    # BEFORE reading back: the reader also calls permute_cell_axis_fields (would append to seen).
    assert len(seen) == 1 and seen[0] is None, seen

    # round-trip fidelity is preserved (obsm/obsp read back equal to the input)
    r = cellstream.read_h5ad(str(out))
    np.testing.assert_array_equal(r.obsm["X_pca"], a.obsm["X_pca"])
    np.testing.assert_array_equal(
        r.obsp["connectivities"].toarray(), a.obsp["connectivities"].toarray()
    )
