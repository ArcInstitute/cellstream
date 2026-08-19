"""Coverage for the single-file `.h5ad` helpers extracted to ``cellstream.h5ad_io`` (#154).

These behaviours were previously exercised only through ``tests/test_read_sharded.py`` (deleted
with the v1 surface — most of its tests built a v1 archive) and a v1-only test in
``tests/test_out_of_shm.py`` (trimmed, not deleted). The handful of tests in that file which
did NOT depend on v1 were restored rather than dropped: the missing-path and dense-stock cases
here, and the large-nnz indptr regression in ``tests/packed/test_read_parity.py``.
(``blosc_nthreads_in_env``'s save/restore contract was restored alongside them in
``tests/test_workers_omp_env.py``, then removed in part B — the v1 writer was its last
caller.) The code under test survives — it is live public API reached through
``cellstream.read_narrow_h5ad`` / ``cellstream.read_h5ad`` / ``read_obs`` / ``read_var`` — so the
coverage is restored here against the module that now owns it, rather than lost with the
v1 tests.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cellstream
from cellstream import h5ad_io, header


def _tiny(n_obs=12, n_var=5):
    X = sp.random(n_obs, n_var, density=0.5, format="csr", dtype="float32", random_state=3)
    X.data = np.ceil(X.data * 20).astype("float32")
    a = ad.AnnData(X=X)
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.var_names = [f"g{i}" for i in range(n_var)]
    a.obs["sample"] = [f"s{i % 3}" for i in range(n_obs)]
    a.var["gene_id"] = [f"ENSG{i:03d}" for i in range(n_var)]
    return a


def _tiny_dense(n_obs=10, n_var=4):
    """A DENSE-X file: exercises read_h5ad's stock fallback, where the `X` group check
    fails outright rather than finding a non-uint16 sparse group."""
    rng = np.random.default_rng(5)
    a = ad.AnnData(X=rng.random((n_obs, n_var), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.var_names = [f"g{i}" for i in range(n_var)]
    return a


# --- _read_obs_from_path: obs.parquet preferred, header.h5ad the fallback ---


def test_read_obs_prefers_obs_parquet(tmp_path):
    a = _tiny()
    d = tmp_path / "arch"
    d.mkdir()
    header.write_header(a, d / "header.h5ad")
    # A parquet that deliberately differs from the header, so we can tell which one was read.
    marked = a.obs.copy()
    marked["sample"] = ["from_parquet"] * len(marked)
    header.write_obs_parquet(marked, d / "obs.parquet")

    got = h5ad_io._read_obs_from_path(d)
    assert list(got["sample"]) == ["from_parquet"] * len(marked)


def test_read_obs_falls_back_to_header_when_no_parquet(tmp_path):
    a = _tiny()
    d = tmp_path / "arch"
    d.mkdir()
    header.write_header(a, d / "header.h5ad")
    assert not (d / "obs.parquet").exists()

    got = h5ad_io._read_obs_from_path(d)
    # The recovered VALUES matter, not just the index -- a fallback that returned an empty
    # frame with the right index would pass an index-only assertion. Compared column-wise
    # rather than with assert_frame_equal because the header.h5ad round-trip legitimately
    # converts a string column to a pandas Categorical (anndata's writer does that), so
    # dtype identity is not the property under test.
    pd.testing.assert_index_equal(got.index, a.obs.index)
    assert list(got["sample"]) == list(a.obs["sample"])


def test_read_var_reads_the_header(tmp_path):
    a = _tiny()
    d = tmp_path / "arch"
    d.mkdir()
    header.write_header(a, d / "header.h5ad")
    # Full frame, so the var COLUMNS are covered and not just the index.
    pd.testing.assert_frame_equal(h5ad_io.read_var(d), a.var)


# --- read_h5ad: single-file dispatch (narrow vs stock) ---


def test_read_h5ad_autodetects_a_narrow_file(tmp_path):
    """A `.h5ad` with uint16 X.data on disk must route to read_narrow_h5ad.

    The discriminator is **cast_back=True widening**, NOT narrow dtypes. Verified in this
    environment: stock ``anndata.read_h5ad`` on a narrow file returns uint16 data *and*
    uint16 indices, so asserting either dtype under ``cast_back=False`` proves nothing about
    which reader ran. Only read_narrow_h5ad widens to float32 data + int32/int64 indices.
    """
    a = _tiny()
    f = tmp_path / "narrow.h5ad"
    cellstream.write_h5ad_narrow(a, f)

    widened = cellstream.read_h5ad(f)  # public entry point, cast_back=True default
    assert widened.X.data.dtype == np.float32
    assert widened.X.indices.dtype in (np.int32, np.int64)
    assert widened.X.indptr.dtype == widened.X.indices.dtype  # scipy dtype unification
    assert (widened.X != a.X).nnz == 0

    # And cast_back=False keeps the on-disk widths (the documented RAM-saving mode).
    narrow = cellstream.read_h5ad(f, cast_back=False)
    assert narrow.X.data.dtype == np.uint16
    assert narrow.shape == a.shape


def test_read_h5ad_reads_a_stock_sparse_file(tmp_path):
    a = _tiny()
    f = tmp_path / "stock.h5ad"
    a.write_h5ad(f)

    got = cellstream.read_h5ad(f)
    assert got.shape == a.shape
    assert (got.X != a.X).nnz == 0


def test_read_h5ad_reads_a_stock_dense_file(tmp_path):
    """Dense X takes the other stock branch: `X` is a Dataset, not a Group, so the
    on-disk dtype probe finds nothing and falls through to anndata."""
    a = _tiny_dense()
    f = tmp_path / "dense.h5ad"
    a.write_h5ad(f)

    got = cellstream.read_h5ad(f)
    assert got.shape == a.shape
    assert not sp.issparse(got.X)
    np.testing.assert_allclose(np.asarray(got.X), np.asarray(a.X))


def test_read_h5ad_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cellstream.read_h5ad(tmp_path / "nope.h5ad")
    with pytest.raises(FileNotFoundError):
        h5ad_io.read_h5ad(tmp_path / "nope.h5ad")


def test_read_h5ad_directory_raises(tmp_path):
    """Directories are archives; they never reach this single-file reader (#154)."""
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(IsADirectoryError):
        h5ad_io.read_h5ad(d)


# --- the public wrapper rejects the knobs that belonged to the removed v1 reader ---


def test_public_read_h5ad_rejects_n_workers_for_a_single_file(tmp_path):
    """#154: n_workers drove the v1 parallel reader. On a single file it would now be
    silently dropped -- the ignored-knob class #261 fixed -- so it must raise."""
    a = _tiny()
    f = tmp_path / "stock.h5ad"
    a.write_h5ad(f)

    with pytest.raises(NotImplementedError, match="n_workers"):
        cellstream.read_h5ad(f, n_workers=2)

    assert cellstream.read_h5ad(f).shape == a.shape  # unaffected without the knob


@pytest.mark.parametrize(
    "kw", [{"backing": "mmap", "spill_dir": "."}, {"backing": "auto", "spill_dir": "."}]
)
def test_public_read_h5ad_rejects_out_of_shm_backing_for_a_single_file(tmp_path, kw):
    a = _tiny()
    f = tmp_path / "stock.h5ad"
    a.write_h5ad(f)
    if "spill_dir" in kw:
        kw = {**kw, "spill_dir": str(tmp_path)}

    with pytest.raises(NotImplementedError):
        cellstream.read_h5ad(f, **kw)
