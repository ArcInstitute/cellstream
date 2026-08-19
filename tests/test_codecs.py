"""codecs.py: Blosc preset registry + numeric-dataset recompression."""

import numpy as np
import pytest


def test_filter_kwargs_for_zstd_u16u16():
    from cellstream.codecs import get_data_dtype, get_filter_kwargs, get_indices_dtype

    fk = get_filter_kwargs("blosc-zstd-1-u16u16")
    # hdf5plugin.Blosc instance behaves like a dict of h5py create_dataset kwargs
    as_dict = dict(fk)
    assert "compression" in as_dict
    assert as_dict["compression"] == 32001  # libblosc filter ID
    assert get_data_dtype("blosc-zstd-1-u16u16") == "uint16"
    assert get_indices_dtype("blosc-zstd-1-u16u16") == "uint16"


def test_unknown_codec_raises():
    from cellstream.codecs import get_filter_kwargs

    with pytest.raises(KeyError):
        get_filter_kwargs("not-a-codec")


def test_recompress_numeric_datasets_round_trip(tmp_path):
    """Write a small h5ad uncompressed, recompress, verify content round-trips."""
    import anndata as ad
    import h5py
    import scipy.sparse as sp

    from cellstream.codecs import _recompress_numeric_datasets, get_filter_kwargs

    rng = np.random.default_rng(0)
    n_obs, n_var, density = 100, 50, 0.1
    X = sp.random(n_obs, n_var, density=density, format="csr", random_state=rng)
    X.data = (X.data * 1000).astype(np.float32)
    a = ad.AnnData(X=X)
    a.obs["sample"] = ["s_a"] * 60 + ["s_b"] * 40
    a.var["gene_id"] = [f"g{i}" for i in range(n_var)]

    src = tmp_path / "a.h5ad"
    a.write_h5ad(src)

    _recompress_numeric_datasets(src, get_filter_kwargs("blosc-zstd-1-u16u16"))

    # The recompressed file is still a valid h5ad
    a2 = ad.read_h5ad(src)
    assert (a2.X != a.X).nnz == 0
    assert list(a2.obs["sample"]) == list(a.obs["sample"])
    # X/data should now be Blosc-compressed (hdf5plugin filters show as "unknown" in h5py)
    with h5py.File(src, "r") as f:
        assert f["X/data"].compression is not None


def test_recompress_zero_nnz(tmp_path):
    """Empty sparse matrices (0 nnz) recompress without crashing."""
    import anndata as ad
    import scipy.sparse as sp

    from cellstream.codecs import _recompress_numeric_datasets, get_filter_kwargs

    X = sp.csr_matrix((10, 20), dtype=np.float32)  # 0 nnz
    a = ad.AnnData(X=X)
    a.obs["s"] = ["a"] * 10
    a.var["g"] = [f"g{i}" for i in range(20)]
    src = tmp_path / "empty.h5ad"
    a.write_h5ad(src)
    _recompress_numeric_datasets(src, get_filter_kwargs("blosc-zstd-1-u16u16"))
    a2 = ad.read_h5ad(src)
    assert a2.X.nnz == 0


def test_datashuf_preset_registered():
    from cellstream.codecs import (
        get_data_dtype,
        get_data_filter_kwargs,
        get_indices_dtype,
        list_codecs,
    )

    name = "blosc-zstd-1-u16u16-datashuf"
    assert name in list_codecs()
    # data override present for the new preset, absent for the base preset
    assert get_data_filter_kwargs("blosc-zstd-1-u16u16") is None
    fk = get_data_filter_kwargs(name)
    assert fk is not None
    assert dict(fk)["compression"] == 32026  # Blosc2 filter id
    # narrowing identical to the base preset
    assert get_data_dtype(name) == "uint16"
    assert get_indices_dtype(name) == "uint16"


def _filter_ids(ds):
    """HDF5 filter ids on a dataset's creation property list."""
    dcpl = ds.id.get_create_plist()
    return [dcpl.get_filter(i)[0] for i in range(dcpl.get_nfilters())]


def _write_csr_h5ad(tmp_path, n_obs=2000, n_var=2000, density=0.05):
    """CSR h5ad large enough that data & indices exceed the 64KB recompress floor."""
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    X = sp.random(n_obs, n_var, density=density, format="csr", random_state=rng)
    X.data = (X.data * 100).astype(np.float32)
    a = ad.AnnData(X=X)
    src = tmp_path / "a.h5ad"
    a.write_h5ad(src)
    return src


def test_datashuf_applies_blosc2_to_csr_data_only(tmp_path):
    import h5py

    from cellstream.codecs import (
        _recompress_numeric_datasets,
        get_data_filter_kwargs,
        get_filter_kwargs,
    )

    name = "blosc-zstd-1-u16u16-datashuf"
    src = _write_csr_h5ad(tmp_path)
    _recompress_numeric_datasets(
        src,
        get_filter_kwargs(name),
        data_filter_kwargs=get_data_filter_kwargs(name),
    )
    with h5py.File(src, "r") as f:
        assert 32026 in _filter_ids(f["X/data"])  # Blosc2 SHUFFLE on data
        assert 32001 in _filter_ids(f["X/indices"])  # Blosc1 bitshuffle on indices


def test_base_preset_data_filter_unchanged(tmp_path):
    import h5py

    from cellstream.codecs import _recompress_numeric_datasets, get_filter_kwargs

    src = _write_csr_h5ad(tmp_path)
    _recompress_numeric_datasets(src, get_filter_kwargs("blosc-zstd-1-u16u16"))
    with h5py.File(src, "r") as f:
        assert 32001 in _filter_ids(f["X/data"])  # base path untouched -> Blosc1
        assert 32026 not in _filter_ids(f["X/data"])


def test_datashuf_targeting_guard_non_csr_data(tmp_path):
    """A dataset literally named `data` whose parent is NOT a sparse group keeps
    the default filter (not Blosc2-SHUFFLE)."""
    import anndata as ad
    import h5py
    import scipy.sparse as sp

    from cellstream.codecs import (
        _recompress_numeric_datasets,
        get_data_filter_kwargs,
        get_filter_kwargs,
    )

    rng = np.random.default_rng(1)
    X = sp.random(2000, 2000, density=0.05, format="csr", random_state=rng)
    X.data = (X.data * 100).astype(np.float32)
    a = ad.AnnData(X=X)
    # obsm member named "data", big enough to be compressed; parent encoding-type != csr/csc
    a.obsm["data"] = rng.standard_normal((2000, 40)).astype(np.float32)
    src = tmp_path / "g.h5ad"
    a.write_h5ad(src)

    name = "blosc-zstd-1-u16u16-datashuf"
    _recompress_numeric_datasets(
        src, get_filter_kwargs(name), data_filter_kwargs=get_data_filter_kwargs(name)
    )
    with h5py.File(src, "r") as f:
        assert 32026 in _filter_ids(f["X/data"])  # real CSR data -> Blosc2
        assert 32001 in _filter_ids(f["obsm/data"])  # non-CSR "data" -> default Blosc1
        assert 32026 not in _filter_ids(f["obsm/data"])


def test_write_one_h5ad_datashuf_roundtrip(tmp_path):
    import anndata as ad
    import h5py
    import hdf5plugin  # noqa: F401  (register filters for read-back)
    import scipy.sparse as sp

    from cellstream.shard_io import write_one_h5ad

    rng = np.random.default_rng(2)
    X = sp.random(2000, 2000, density=0.05, format="csr", random_state=rng)
    X.data = np.rint(X.data * 50).astype(np.float32)  # integer counts -> lossless u16
    a = ad.AnnData(X=X)

    out = tmp_path / "shard.h5ad"
    write_one_h5ad(a, out, codec="blosc-zstd-1-u16u16-datashuf", where="X")

    # data dataset uses Blosc2-SHUFFLE
    with h5py.File(out, "r") as f:
        assert 32026 in _filter_ids(f["X/data"])

    # lossless round-trip through a normal read
    a2 = ad.read_h5ad(out)
    assert (a2.X != a.X).nnz == 0


def test_default_codecs_per_entry_point():
    """Public write defaults as of v0.3.0.

    write_sharded defaults to format='v2' with a ``codec=None`` sentinel that resolves to
    the v2 byte-filter codec. (Until #154 removed the v1 writer, an explicit
    ``format='v1'`` resolved the SAME sentinel to the datashuf preset, per PR #70's
    read-side win; there is only one arm now.) write_h5ad_narrow (single-file narrow
    writer) is unaffected by the sharded default flip and still defaults to the datashuf
    preset.
    """
    import inspect

    from cellstream.write import write_h5ad_narrow, write_sharded

    sig = inspect.signature(write_sharded)
    # write_sharded: format default flipped to v2; codec is a None sentinel.
    assert sig.parameters["format"].default == "v2"
    assert sig.parameters["codec"].default is None
    # write_h5ad_narrow keeps the datashuf preset default.
    expected_narrow = "blosc-zstd-1-u16u16-datashuf"
    assert inspect.signature(write_h5ad_narrow).parameters["codec"].default == expected_narrow


def test_default_write_h5ad_narrow_uses_datashuf(tmp_path):
    """A default write (no explicit codec=) applies Blosc2-SHUFFLE to CSR data,
    Blosc1-bitshuffle to indices, and round-trips losslessly."""
    import anndata as ad
    import h5py
    import hdf5plugin  # noqa: F401  (register filters for read-back)
    import scipy.sparse as sp

    from cellstream.write import write_h5ad_narrow

    rng = np.random.default_rng(3)
    X = sp.random(2000, 2000, density=0.05, format="csr", random_state=rng)
    X.data = np.rint(X.data * 50).astype(np.float32)  # integer counts -> lossless u16
    a = ad.AnnData(X=X)

    out = tmp_path / "default.h5ad"
    write_h5ad_narrow(a, out)  # no codec= -> uses the promoted default

    with h5py.File(out, "r") as f:
        assert 32026 in _filter_ids(f["X/data"])  # Blosc2-SHUFFLE on data
        assert 32001 in _filter_ids(f["X/indices"])  # Blosc1-bitshuffle on indices

    a2 = ad.read_h5ad(out)
    assert (a2.X != a.X).nnz == 0
