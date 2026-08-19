# tests/test_read_modality.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.read import ShardedArchive
from cellstream.write import write_sharded


def _gex_atac(n_obs=30):
    def _csr(nv, s):
        m = sp.random(n_obs, nv, density=0.3, format="csr", random_state=s)
        return sp.csr_matrix(
            (
                (m.data * 20 + 1).astype(np.uint16),
                m.indices.astype(np.uint16),
                m.indptr.astype(np.int64),
            ),
            shape=(n_obs, nv),
        )

    obs = pd.DataFrame({"ct": ["a", "b"] * (n_obs // 2)}, index=[f"c{i}" for i in range(n_obs)])
    gex = ad.AnnData(
        X=_csr(5, 1), obs=obs.copy(), var=pd.DataFrame(index=[f"g{i}" for i in range(5)])
    )
    gex.layers["norm"] = _csr(5, 4)
    atac = ad.AnnData(
        X=_csr(7, 2), obs=obs.copy(), var=pd.DataFrame(index=[f"p{i}" for i in range(7)])
    )
    atac.layers["tfidf"] = _csr(7, 3)
    return gex, atac


def test_to_anndata_returns_primary_only(tmp_path):
    gex, atac = _gex_atac()
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac})
    arch = ShardedArchive(f)
    out = arch.to_anndata()
    np.testing.assert_array_equal(out.X.toarray(), gex.X.toarray())
    # anndata 0.13 exposes X as layers[None]; compare only real layer names.
    assert set(out.layers.keys()) - {None} == {"norm"}  # x layer only; NOT tfidf
    np.testing.assert_array_equal(out.layers["norm"].toarray(), gex.layers["norm"].toarray())
    assert list(out.var.index) == list(gex.var.index)  # primary var, NOT peaks


def test_modalities_and_modality_view(tmp_path):
    gex, atac = _gex_atac()
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac})
    arch = ShardedArchive(f)
    assert arch.modalities == [("modality:atac", "modality:atac")]
    Xa = arch.modality("atac").x()
    np.testing.assert_array_equal(Xa.toarray(), atac.X.toarray())
    Xl = arch.modality("atac").x(layer="tfidf")
    np.testing.assert_array_equal(Xl.toarray(), atac.layers["tfidf"].toarray())


def test_modality_rejects_non_anchor_key(tmp_path):
    # a modality-LAYER key ("modality:atac:layer:tfidf") starts with "modality:"
    # and is a real key, but must NOT be accepted as a modality name.
    gex, atac = _gex_atac()
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac})
    arch = ShardedArchive(f)
    with pytest.raises(KeyError):
        arch.modality("atac:layer:tfidf")


def test_to_mudata_data_dtype_uint16(tmp_path):
    # #198: per-family read dtype -> u16 CSR for integer counts (the read-RAM lever).
    pytest.importorskip("mudata")
    gex, atac = _gex_atac()
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac})
    mdata = ShardedArchive(f).to_mudata(data_dtype="uint16")
    for k in mdata.mod:
        assert mdata.mod[k].X.dtype == np.uint16
    np.testing.assert_array_equal(mdata.mod["atac"].X.toarray(), atac.X.toarray())


def test_to_mudata_default_is_float32(tmp_path):
    # default is unchanged: float32 CSR per modality.
    pytest.importorskip("mudata")
    gex, atac = _gex_atac()
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac})
    mdata = ShardedArchive(f).to_mudata()
    for k in mdata.mod:
        assert mdata.mod[k].X.dtype == np.float32


def test_to_mudata_lossy_without_allow_raises(tmp_path):
    # a lossy request without allow_lossy must fail loud.
    pytest.importorskip("mudata")
    from cellstream.errors import LossyCastError

    n = 6
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    gex = ad.AnnData(  # values > uint8 max -> uint8 cast is lossy
        X=sp.csr_matrix(np.full((n, 3), 300, dtype=np.uint16)),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    atac = ad.AnnData(
        X=sp.csr_matrix(np.full((n, 3), 300, dtype=np.uint16)),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"p{i}" for i in range(3)]),
    )
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac})
    with pytest.raises(LossyCastError):
        ShardedArchive(f).to_mudata(data_dtype="uint8")


def test_to_mudata_dense_container(tmp_path):
    # container="dense" -> dense X per modality.
    pytest.importorskip("mudata")
    gex, atac = _gex_atac()
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac})
    mdata = ShardedArchive(f).to_mudata(container="dense")
    for k in mdata.mod:
        assert not sp.issparse(mdata.mod[k].X)
    np.testing.assert_array_equal(np.asarray(mdata.mod["atac"].X), atac.X.toarray())


def test_to_mudata_cast_back_false_native_dtype(tmp_path):
    # cast_back=False -> native on-disk dtype (integer data is stored uint32).
    pytest.importorskip("mudata")
    gex, atac = _gex_atac()
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac})
    arch = ShardedArchive(f)
    mdata = arch.to_mudata(cast_back=False)
    expected = np.dtype(arch.manifest["matrices"]["modality:atac"]["x_data_dtype_on_disk"])
    assert mdata.mod["atac"].X.dtype == expected
