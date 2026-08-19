"""Option-parity footgun fixes (spec 2026-06-03), plus the shard/cell sort-knob parity.

The spec's subject was v1-vs-v2 parity: which knobs each format accepted, and which
sentinel each resolved. #154 removed the v1 writer, so the v1 half of every pair is gone
(``blosc_nthreads`` is no longer a ``write_sharded`` parameter at all, and
``target_shard_bytes`` has no format that rejects it). What survives is the v2 side of the
sentinel checks and the layout='shard'-vs-layout='cell' sort-parameter parity, which is a
different axis and untouched by the removal.
"""

import numpy as np
import pytest
import scipy.sparse as sp


def _make_adata(n_obs=120, n_var=20, seed=7):
    import anndata as ad

    rng = np.random.default_rng(seed)
    nnz = int(n_obs * n_var * 0.1)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_var, size=nnz)
    data = rng.integers(0, 200, size=nnz).astype(np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_var))
    a = ad.AnnData(X=X)
    a.var["gene_id"] = [f"g{i}" for i in range(n_var)]
    return a


def _write_v2(tmp_path, name="v2arch", **kw):
    from cellstream.write import write_sharded

    a = _make_adata()
    out = tmp_path / name
    write_sharded(a, out, format="v2", n_workers=2, **kw)
    return a, out


def test_v2_write_default_target_shard_bytes_ok(tmp_path):
    # v2 resolves target_shard_bytes None -> 2 GiB internally and writes successfully.
    a, out = _write_v2(tmp_path)
    assert out.exists()  # packed: one file


def _sortable_adata():
    """Numeric group labels, so natural (2 < 10) and alphabetical ("10" < "2") DIVERGE. With
    string labels the two coincide and an invalid mode looks harmless."""
    import anndata as ad
    import pandas as pd

    n_obs, n_var = 6, 4
    X = sp.csr_matrix(np.arange(n_obs * n_var, dtype=np.int32).reshape(n_obs, n_var))
    obs = pd.DataFrame({"plate": [2, 10, 2, 10, 3, 3]}, index=[f"c{i}" for i in range(n_obs)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_var)]))


def _raises_message(tmp_path, layout, **kw):
    """Return str(exc) from write_sharded, or None if it did not raise."""
    import warnings

    from cellstream.write import write_sharded

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            write_sharded(_sortable_adata(), tmp_path / f"{layout}.out", layout=layout, **kw)
    except ValueError as e:
        return str(e)
    return None


@pytest.mark.parametrize(
    "kw",
    [
        {"group_by": "plate", "sort_order": "alphabetial"},
        {"sort_order": "alphabetical"},
        {"sort_within_group": "plate"},
    ],
    ids=["invalid_mode", "sort_order_no_group_by", "sort_within_group_no_group_by"],
)
def test_sort_params_raise_identically_on_both_layouts(tmp_path, kw):
    """Pre-fix layout='cell' ACCEPTED all three: it stored natural order and stamped the
    invalid string into the manifest. Compare the MESSAGE, not a substring -- a substring
    match would pass even if the two layouts disagreed on the wording."""
    shard = _raises_message(tmp_path / "s", "shard", **kw)
    cell = _raises_message(tmp_path / "c", "cell", **kw)
    assert shard is not None, "layout='shard' regressed: it no longer rejects this"
    assert cell == shard, f"cell={cell!r} shard={shard!r}"


@pytest.mark.parametrize("sort_order", ["natural", "alphabetical"])
def test_valid_sort_order_still_applies_on_cell(tmp_path, sort_order):
    """The guard must not break the working modes -- and the manifest must record the mode that
    was actually applied."""
    import warnings

    import cellstream
    from cellstream.write import write_sharded

    p = tmp_path / f"cell_{sort_order}.shad"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        write_sharded(_sortable_adata(), p, layout="cell", group_by="plate", sort_order=sort_order)
    store = cellstream.open(p)
    expected = [2, 2, 3, 3, 10, 10] if sort_order == "natural" else [10, 10, 2, 2, 3, 3]
    assert list(store.obs["plate"]) == expected
    assert store.manifest["sort_order"] == sort_order


def test_direct_cell_writers_also_reject_an_invalid_sort_order(tmp_path):
    """The fix lands in _resolve_storage_order, so the public cellstream.cell writers get it too --
    not just the write_sharded facade. CellArchiveWriter.finalize is the one the production push
    pipelines call."""
    from cellstream.cell import CellArchiveWriter, write_cell_archive

    a = _sortable_adata()
    with pytest.raises(ValueError, match="sort_order must be 'natural' or 'alphabetical'"):
        write_cell_archive(a, tmp_path / "wca.shad", group_by="plate", sort_order="alphabetial")

    w = CellArchiveWriter(tmp_path / "push.shad", n_vars=a.n_vars, value_dtype="int32")
    w.add_block(a.X)
    with pytest.raises(ValueError, match="sort_order must be 'natural' or 'alphabetical'"):
        w.finalize(obs=a.obs, var=a.var, group_by="plate", sort_order="alphabetial")


def test_concat_cell_archives_also_rejects_an_invalid_sort_order(tmp_path):
    """The third direct writer. concat reaches _resolve_storage_order through the same
    _reorder_and_finalize tail as CellArchiveWriter.finalize, so one call site covers it -- but
    the spec claims all three, so all three are asserted.

    The parts are written WITH group_by='plate' so their manifests agree on the grouping; concat
    resolves group/sort defaults from the manifests and requires agreement, and a mismatch there
    would fail the test for an unrelated reason. Each part gets a distinct obs index so the
    concatenated obs has unique names.

    VERIFIED PRE-FIX on 7b1ca4a with exactly this construction: concat ACCEPTED
    sort_order='alphabetial', stored [2,2,2,2,3,3,3,3,10,10,10,10], and wrote manifest
    sort_order='alphabetial'. So this fails before the fix for the right reason."""
    import warnings

    from cellstream.cell import concat_cell_archives, write_cell_archive

    parts = []
    for i in range(2):
        a = _sortable_adata()
        a.obs.index = [f"c{i}_{j}" for j in range(a.n_obs)]
        p = tmp_path / f"part{i}.shad"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            write_cell_archive(a, p, group_by="plate")
        parts.append(p)

    with pytest.raises(ValueError, match="sort_order must be 'natural' or 'alphabetical'"):
        concat_cell_archives(
            parts, tmp_path / "cat.shad", group_by="plate", sort_order="alphabetial"
        )
