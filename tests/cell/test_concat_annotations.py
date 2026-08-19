"""#237 -- concat_cell_archives preserves obsm/varm/obsp/varp."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.cell import CellArchiveWriter, concat_cell_archives, open_cell
from cellstream.cell.writer import write_cell_archive

from .conftest import assert_archives_equivalent, make_cell_adata
from .test_concat import _write_slices


def _block_bounds(n, n_parts):
    """The contiguous row ranges _write_slices cuts -- kept in sync with it by construction."""
    return [(k * n) // n_parts for k in range(n_parts + 1)]


def _block_diag_mask(n, bounds):
    """Explicit dense block mask, deliberately NOT block_diag: the expectation must be independent
    of the implementation it checks, or the test is a tautology."""
    m = np.zeros((n, n), dtype=bool)
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        m[lo:hi, lo:hi] = True
    return m


def _expected_merge(a, n_parts):
    """The AnnData that a concat of `a`'s n_parts contiguous slices must reproduce.

    obs/obsm/varm/varp/uns are identical to `a` (the slices are contiguous and in input order),
    and obsp is `a`'s graph restricted to the block diagonal -- each part's graph was computed
    (here: sliced) independently, so cross-part edges do not exist in any part and cannot be
    invented by the merge.
    """
    merged = a.copy()
    mask = _block_diag_mask(a.n_obs, _block_bounds(a.n_obs, n_parts))
    for k, v in a.obsp.items():
        dense = v.toarray() if sp.issparse(v) else np.asarray(v)
        blocked = dense * mask
        merged.obsp[k] = sp.csr_matrix(blocked) if sp.issparse(v) else blocked
    return merged


def _halves(a):
    n = a.n_obs
    return a[: n // 2].copy(), a[n // 2 :].copy()


def _write_pair(tmp_path, a1, a2, names=("ap0.shad", "ap1.shad")):
    """Write two independently-built AnnDatas as ungrouped parts (they must already share var)."""
    paths = []
    for a, name in zip((a1, a2), names, strict=True):
        p = tmp_path / name
        write_cell_archive(a, p, group_by=None)
        paths.append(p)
    return paths


def test_concat_preserves_annotations_ungrouped(tmp_path, engine_env):
    a = make_cell_adata(annotations=True)
    parts = _write_slices(a, tmp_path, 3)
    expected = tmp_path / "expected.shad"
    write_cell_archive(_expected_merge(a, 3), expected, group_by=None)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out)
    assert_archives_equivalent(expected, out)


def test_concat_preserves_annotations_grouped(tmp_path, engine_env):
    """Grouped concat must permute obsm rows AND both obsp axes to storage order."""
    a = make_cell_adata(annotations=True)
    parts = _write_slices(a, tmp_path, 4)
    expected = tmp_path / "expected.shad"
    write_cell_archive(_expected_merge(a, 4), expected, group_by="target_gene", reference=["t00"])
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out, group_by="target_gene", reference=["t00"])
    assert_archives_equivalent(expected, out)


def test_concat_annotations_drop(tmp_path, engine_env):
    """annotations="drop" merges X/obs/var/uns only -- and says nothing: the caller asked."""
    a = make_cell_adata(annotations=True)
    stripped = a.copy()
    for field in ("obsm", "varm", "obsp", "varp"):
        d = getattr(stripped, field)
        for k in list(d):
            del d[k]
    expected = tmp_path / "expected.shad"
    write_cell_archive(stripped, expected, group_by=None)
    parts = _write_slices(a, tmp_path, 3)
    out = tmp_path / "out.shad"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        concat_cell_archives(parts, out, annotations="drop")
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert_archives_equivalent(expected, out)
    src = open_cell(parts[0])  # the parts themselves are untouched
    try:
        assert src.obsm and src.obsp
    finally:
        src.close()


def test_concat_rejects_invalid_annotations_value(tmp_path):
    a = make_cell_adata()
    parts = _write_slices(a, tmp_path, 2)
    with pytest.raises(ValueError, match="annotations must be"):
        concat_cell_archives(parts, tmp_path / "out.shad", annotations="nope")


def test_concat_without_annotations_unchanged(tmp_path, engine_env):
    """Byte-identity guard: annotation-free concat must be exactly what it was before #237."""
    a = make_cell_adata()  # no annotations
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by="target_gene", reference=["t00"])
    parts = _write_slices(a, tmp_path, 3)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out, group_by="target_gene", reference=["t00"])
    assert_archives_equivalent(ref, out)
    s = open_cell(out)
    try:
        assert not s.obsm and not s.varm and not s.obsp and not s.varp
    finally:
        s.close()


def test_concat_obsp_is_block_diagonal(tmp_path, engine_env):
    """The merged graph has no cross-part edges -- and DOES keep the within-part ones (without the
    second assert, an implementation that zeroed obsp entirely would pass)."""
    a = make_cell_adata(annotations=True)
    parts = _write_slices(a, tmp_path, 3)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out)  # ungrouped -> identity perm -> rows stay in part order
    s = open_cell(out)
    try:
        got = s.obsp["connectivities"].toarray()
        mask = _block_diag_mask(a.n_obs, _block_bounds(a.n_obs, 3))
        assert not got[~mask].any(), "cross-part obsp entries must be zero"
        assert got[mask].any(), "within-part obsp entries must survive"
    finally:
        s.close()


def test_concat_dataframe_obsm_roundtrips(tmp_path, engine_env):
    """The pd.concat arm, including index labels."""
    a = make_cell_adata(annotations=True)
    a.obsm["meta"] = pd.DataFrame({"score": np.arange(a.n_obs, dtype=float)}, index=a.obs_names)
    parts = _write_slices(a, tmp_path, 3)
    expected = tmp_path / "expected.shad"
    write_cell_archive(_expected_merge(a, 3), expected, group_by=None)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out)
    assert_archives_equivalent(expected, out)


def test_concat_dense_obsp_stays_dense(tmp_path, engine_env):
    """Container type survives the merge: dense in every part -> dense out, sparse -> CSR."""
    n = 60
    a = make_cell_adata(n_obs=n, annotations=True)
    a.obsp["dense_graph"] = np.arange(n * n, dtype=np.float32).reshape(n, n)
    parts = _write_slices(a, tmp_path, 3)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out)
    s = open_cell(out)
    try:
        assert isinstance(s.obsp["dense_graph"], np.ndarray)
        assert sp.issparse(s.obsp["connectivities"])
        mask = _block_diag_mask(n, _block_bounds(n, 3))
        np.testing.assert_array_equal(s.obsp["dense_graph"], a.obsp["dense_graph"] * mask)
    finally:
        s.close()


def test_concat_promotes_obsm_dtype(tmp_path, engine_env):
    """Value dtypes may differ across parts and promote, mirroring _widest for X."""
    a = make_cell_adata(annotations=True)
    a1, a2 = _halves(a)
    a1.obsm["X_pca"] = a1.obsm["X_pca"].astype(np.float32)
    a2.obsm["X_pca"] = a2.obsm["X_pca"].astype(np.float64)
    parts = _write_pair(tmp_path, a1, a2)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out)
    s = open_cell(out)
    try:
        assert s.obsm["X_pca"].dtype == np.float64
        assert s.obsm["X_pca"].shape == (a.n_obs, 10)
    finally:
        s.close()


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda a2: a2.obsm.__setitem__("extra", np.zeros((a2.n_obs, 2), dtype=np.float32)),
            "obsm keys differ",
        ),
        (lambda a2: a2.obsm.__setitem__("X_pca", a2.obsm["X_pca"][:, :5]), "trailing shape"),
        (
            lambda a2: a2.obsm.__setitem__(
                "X_pca",
                # str column names: anndata refuses to write an obsm DataFrame with integer columns
                pd.DataFrame(
                    a2.obsm["X_pca"],
                    index=a2.obs_names,
                    columns=[f"pc{i}" for i in range(a2.obsm["X_pca"].shape[1])],
                ),
            ),
            "mixed container kinds",
        ),
        (
            lambda a2: a2.obsp.__setitem__("connectivities", a2.obsp["connectivities"].toarray()),
            "mixed container kinds",
        ),
        (lambda a2: a2.varm.__setitem__("PCs", a2.varm["PCs"] + 1.0), "different varm"),
        (lambda a2: a2.varp.__setitem__("corr", a2.varp["corr"] * 2.0), "different varp"),
    ],
    ids=["obsm_keys", "obsm_width", "obsm_container", "obsp_dense_vs_sparse", "varm", "varp"],
)
def test_concat_rejects_annotation_mismatch(tmp_path, mutate, match):
    """Every cross-part annotation disagreement fails loud through the public API."""
    a = make_cell_adata(annotations=True)
    a1, a2 = _halves(a)
    mutate(a2)
    parts = _write_pair(tmp_path, a1, a2)
    with pytest.raises(ValueError, match=match):
        concat_cell_archives(parts, tmp_path / "out.shad")


def test_concat_annotation_mismatch_leaves_no_output(tmp_path):
    """The pre-flight rejects BEFORE the byte-copy, so nothing is written at `out`."""
    a = make_cell_adata(annotations=True)
    a1, a2 = _halves(a)
    a2.obsm["extra"] = np.zeros((a2.n_obs, 2), dtype=np.float32)
    parts = _write_pair(tmp_path, a1, a2)
    out = tmp_path / "out.shad"
    with pytest.raises(ValueError):
        concat_cell_archives(parts, out)
    assert not out.exists()


def test_concat_annotation_mismatch_ignored_when_dropping(tmp_path, engine_env):
    """annotations="drop" does not read the parts' annotations, so a mismatch cannot fail it."""
    a = make_cell_adata(annotations=True)
    a1, a2 = _halves(a)
    a2.obsm["extra"] = np.zeros((a2.n_obs, 2), dtype=np.float32)
    parts = _write_pair(tmp_path, a1, a2)
    out = tmp_path / "out.shad"
    concat_cell_archives(parts, out, annotations="drop")
    s = open_cell(out)
    try:
        assert not s.obsm and not s.varm and not s.obsp and not s.varp
        assert s.n_obs == a.n_obs
    finally:
        s.close()


def test_cellarchivewriter_finalize_unchanged(tmp_path):
    """Byte-identity guard: _reorder_and_finalize gained four parameters; the push writer passes
    none of them, and its output must not move. Grouped, so the non-identity permute path runs
    with empty annotations."""
    a = make_cell_adata()
    ref = tmp_path / "ref.shad"
    write_cell_archive(a, ref, group_by="target_gene", reference=["t00"])
    out = tmp_path / "pushed.shad"
    w = CellArchiveWriter(out, n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16")
    w.add_block(sp.csr_matrix(a.X))
    w.finalize(obs=a.obs, var=a.var, group_by="target_gene", reference=["t00"])
    assert_archives_equivalent(ref, out)
    s = open_cell(out)
    try:
        assert not s.obsm and not s.varm and not s.obsp and not s.varp
    finally:
        s.close()
