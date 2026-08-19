"""read_narrow_h5ad: streaming-cast read via h5py.read_direct."""

import numpy as np
import scipy.sparse as sp


def _make_and_write(tmp_path, codec="blosc-zstd-1-u16u16"):
    import anndata as ad

    from cellstream.shard_io import write_one_h5ad

    rng = np.random.default_rng(7)
    n_obs, n_var = 150, 40
    nnz = int(n_obs * n_var * 0.1)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_var, size=nnz)
    data = rng.integers(0, 500, size=nnz).astype(np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_var))
    a = ad.AnnData(X=X)
    a.obs["sample"] = ["s"] * n_obs
    a.var["gene_id"] = [f"g{i}" for i in range(n_var)]
    out = tmp_path / "n.h5ad"
    write_one_h5ad(a, out, codec=codec)
    return a, out


def test_cast_back_true_returns_float32_indices_int32(tmp_path):
    from cellstream.read import read_narrow_h5ad

    a_orig, out = _make_and_write(tmp_path)
    a = read_narrow_h5ad(out, cast_back=True)
    assert a.X.data.dtype == np.float32
    # nnz < 2**31, so indices should be int32
    assert a.X.indices.dtype == np.int32
    assert (a.X != a_orig.X).nnz == 0


def test_cast_back_false_keeps_narrow(tmp_path):
    from cellstream.read import read_narrow_h5ad

    a_orig, out = _make_and_write(tmp_path)
    a = read_narrow_h5ad(out, cast_back=False)
    assert a.X.data.dtype == np.uint16
    assert a.X.indices.dtype == np.uint16
    # values round-trip
    a_orig_u16 = a_orig.X.astype(np.float32)  # convert original to float32 first
    assert np.array_equal(a.X.data.astype(np.float32), a_orig_u16.data.astype(np.float32))


def test_top_level_write_h5ad_narrow_round_trip(tmp_path):
    """write_h5ad_narrow + read_narrow_h5ad as the public surface."""
    import cellstream

    a_orig, _ = _make_and_write(tmp_path)  # we ignore the second value
    out = tmp_path / "via_top_level.h5ad"
    cellstream.write_h5ad_narrow(a_orig, out)
    a = cellstream.read_narrow_h5ad(out)
    assert (a.X != a_orig.X).nnz == 0


def test_full_anndata_round_trip(tmp_path):
    """write_h5ad_narrow + read_narrow_h5ad preserves uns, obsm, varm, layers, obsp, varp."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    import cellstream

    rng = np.random.default_rng(11)
    n_obs, n_var = 50, 20
    nnz = int(n_obs * n_var * 0.1)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_var, size=nnz)
    data = rng.integers(0, 200, size=nnz).astype(np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_var))
    a = ad.AnnData(X=X)
    a.obs["sample"] = ["s_a"] * 25 + ["s_b"] * 25
    a.var["gene_id"] = [f"g{i}" for i in range(n_var)]
    a.uns["pipeline_version"] = "v1.0"
    a.obsm["pca"] = rng.standard_normal((n_obs, 3))
    a.varm["coef"] = rng.standard_normal((n_var, 2))
    a.layers["counts"] = sp.csr_matrix(X)  # same shape
    a.obsp["distances"] = sp.csr_matrix((n_obs, n_obs), dtype="float32")
    a.varp["correlation"] = sp.csr_matrix((n_var, n_var), dtype="float32")

    out = tmp_path / "full.h5ad"
    cellstream.write_h5ad_narrow(a, out)
    a2 = cellstream.read_narrow_h5ad(out)
    assert (a2.X != a.X).nnz == 0
    assert a2.uns["pipeline_version"] == "v1.0"
    np.testing.assert_allclose(a2.obsm["pca"], a.obsm["pca"])
    np.testing.assert_allclose(a2.varm["coef"], a.varm["coef"])
    assert "counts" in a2.layers
    assert "distances" in a2.obsp
    assert "correlation" in a2.varp


def test_read_narrow_indptr_no_overflow_at_large_nnz(tmp_path):
    """Regression: read_narrow_h5ad indptr must not overflow at total_nnz > 65535."""
    import anndata as ad

    import cellstream

    rng = np.random.default_rng(43)
    # Use a fully-dense 700×100 matrix (70 000 nnz) so deduplication is moot
    # and the nnz count is guaranteed > 65535.
    n_obs, n_var = 700, 100
    data = rng.integers(1, 200, size=(n_obs, n_var)).astype(np.float32)
    X = sp.csr_matrix(data)
    assert X.nnz > 65535
    a = ad.AnnData(X=X)
    a.obs["s"] = ["a"] * n_obs
    a.var["g"] = [f"g{i}" for i in range(n_var)]
    out = tmp_path / "single.h5ad"
    cellstream.write_h5ad_narrow(a, out)
    a2 = cellstream.read_narrow_h5ad(out, cast_back=False)
    assert a2.X.indptr.dtype == np.int64
    assert int(a2.X.indptr[-1]) > 65535
    a_orig_arr = a.X.toarray()
    a2_arr = a2.X.astype(np.float32).toarray()
    assert np.array_equal(a_orig_arr, a2_arr)


def test_write_one_h5ad_rejects_non_csr(tmp_path):
    """write_one_h5ad raises TypeError for dense or non-CSR sparse X."""
    import anndata as ad
    import numpy as np
    import pytest
    import scipy.sparse as sp

    from cellstream.shard_io import write_one_h5ad

    # Dense X
    a_dense = ad.AnnData(X=np.zeros((10, 5), dtype="float32"))
    with pytest.raises(TypeError):
        write_one_h5ad(a_dense, tmp_path / "dense.h5ad", codec="blosc-zstd-1-u16u16")

    # CSC X
    a_csc = ad.AnnData(X=sp.csc_matrix((10, 5), dtype="float32"))
    with pytest.raises(TypeError):
        write_one_h5ad(a_csc, tmp_path / "csc.h5ad", codec="blosc-zstd-1-u16u16")


def test_read_narrow_h5ad_rejects_csc_encoding(tmp_path):
    """#251: a CSC-encoded .h5ad must be REJECTED, not silently misread as CSR.

    read_narrow_h5ad builds the CSR by direct attribute assignment (deliberately, to keep
    narrow dtypes the constructor would normalise) -- which also means none of scipy's
    validation runs. For a CSC file the same three arrays mean the transpose: indptr runs
    over COLUMNS and indices holds ROW ids, so the result is silently wrong and, with an
    indptr of length n_vars+1 on a matrix declared (n_obs, n_vars), reads out of bounds
    downstream.

    Note write_one_h5ad already rejects CSC with TypeError -- the read path was simply not
    symmetric with the write path.
    """
    import anndata as ad
    import numpy as np
    import pytest
    import scipy.sparse as sp

    from cellstream.read import read_narrow_h5ad

    dense = (np.arange(20, dtype=np.uint16).reshape(4, 5) % 7).astype(np.uint16)
    a = ad.AnnData(X=sp.csc_matrix(dense))
    p = tmp_path / "csc.h5ad"
    a.write_h5ad(p)

    # message names the required encoding AND the one actually found
    with pytest.raises(ValueError, match="requires a CSR-encoded X") as exc:
        read_narrow_h5ad(p)
    assert "csc_matrix" in str(exc.value)


def test_read_narrow_h5ad_still_accepts_csr(tmp_path):
    """#251 guard must not reject the format it exists to read."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    from cellstream.read import read_narrow_h5ad

    dense = (np.arange(20, dtype=np.uint16).reshape(4, 5) % 7).astype(np.uint16)
    a = ad.AnnData(X=sp.csr_matrix(dense))
    p = tmp_path / "csr.h5ad"
    a.write_h5ad(p)

    out = read_narrow_h5ad(p, cast_back=False)
    np.testing.assert_array_equal(out.X.toarray(), dense)


def test_read_narrow_h5ad_rejects_markerless_and_dense_x(tmp_path):
    """#251 (codex review): a MISSING encoding-type is rejected, not assumed CSR.

    anndata has written encoding-type since 0.7.1, and genuinely old 0.6.x sparse files used
    h5sparse_format/h5sparse_shape -- which this function cannot read anyway, since it requires
    the modern `shape` attribute. So a markerless X with a modern `shape` is non-standard, and
    guessing CSR for it is the silent misread the guard exists to prevent.
    """
    import anndata as ad
    import h5py
    import numpy as np
    import pytest
    import scipy.sparse as sp

    from cellstream.read import read_narrow_h5ad

    dense = (np.arange(20, dtype=np.uint16).reshape(4, 5) % 7).astype(np.uint16)

    # (a) CSR file with the marker stripped -> rejected, not guessed.
    p_strip = tmp_path / "markerless.h5ad"
    ad.AnnData(X=sp.csr_matrix(dense)).write_h5ad(p_strip)
    with h5py.File(p_strip, "r+") as f:
        del f["X"].attrs["encoding-type"]
    with pytest.raises(ValueError, match="requires a CSR-encoded X"):
        read_narrow_h5ad(p_strip)

    # (b) dense X -> a clear message instead of the old KeyError on the missing X/data group.
    p_dense = tmp_path / "dense.h5ad"
    ad.AnnData(X=dense.astype(np.float32)).write_h5ad(p_dense)
    with pytest.raises(ValueError, match="requires a CSR-encoded X") as exc:
        read_narrow_h5ad(p_dense)
    assert "array" in str(exc.value)
    # The hint must be actionable for a DENSE X too: np.ndarray has no .tocsr()
    # (Copilot #259), so the message must not recommend only that.
    assert "csr_matrix(" in str(exc.value)


def test_read_narrow_h5ad_accepts_bytes_encoding_attr(tmp_path):
    """#251 (codex review): a fixed-width/bytes encoding-type attr must still be accepted."""
    import anndata as ad
    import h5py
    import numpy as np
    import scipy.sparse as sp

    from cellstream.read import read_narrow_h5ad

    dense = (np.arange(20, dtype=np.uint16).reshape(4, 5) % 7).astype(np.uint16)
    p = tmp_path / "bytes_attr.h5ad"
    ad.AnnData(X=sp.csr_matrix(dense)).write_h5ad(p)
    with h5py.File(p, "r+") as f:
        del f["X"].attrs["encoding-type"]
        f["X"].attrs.create("encoding-type", np.bytes_(b"csr_matrix"))
    out = read_narrow_h5ad(p, cast_back=False)
    np.testing.assert_array_equal(out.X.toarray(), dense)
