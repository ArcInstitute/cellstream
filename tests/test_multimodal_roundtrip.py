# tests/test_multimodal_roundtrip.py
"""Task 9 (multiomics A+B — modalities + mudata): end-to-end native round-trip,
byte-identity guard, and write-determinism. Native (mudata-free) coverage
complements tests/test_mudata_adapter.py's MuData-input round-trips."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.read import ShardedArchive
from cellstream.write import write_sharded


def _gex_atac(n_obs=40):
    def _csr(nv, s):
        m = sp.random(n_obs, nv, density=0.3, format="csr", random_state=s)
        return sp.csr_matrix(
            (
                (m.data * 25 + 1).astype(np.uint16),
                m.indices.astype(np.uint16),
                m.indptr.astype(np.int64),
            ),
            shape=(n_obs, nv),
        )

    obs = pd.DataFrame({"g": ["x", "y"] * (n_obs // 2)}, index=[f"c{i}" for i in range(n_obs)])
    gex = ad.AnnData(
        X=_csr(6, 1), obs=obs.copy(), var=pd.DataFrame(index=[f"g{i}" for i in range(6)])
    )
    gex.layers["norm"] = _csr(6, 5)
    gex.raw = ad.AnnData(X=_csr(6, 6), var=gex.var.copy())
    atac = ad.AnnData(
        X=_csr(9, 2), obs=obs.copy(), var=pd.DataFrame(index=[f"p{i}" for i in range(9)])
    )
    atac.layers["tfidf"] = _csr(9, 3)
    return gex, atac


def test_native_modality_round_trip_csr(tmp_path):
    gex, atac = _gex_atac()
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac})
    arch = ShardedArchive(f)
    out = arch.to_anndata()
    np.testing.assert_array_equal(out.X.toarray(), gex.X.toarray())
    np.testing.assert_array_equal(out.layers["norm"].toarray(), gex.layers["norm"].toarray())
    np.testing.assert_array_equal(out.raw.X.toarray(), gex.raw.X.toarray())
    a = arch.modality("atac")
    np.testing.assert_array_equal(a.x().toarray(), atac.X.toarray())
    np.testing.assert_array_equal(a.x(layer="tfidf").toarray(), atac.layers["tfidf"].toarray())


def test_native_modality_round_trip_dense(tmp_path):
    gex, atac = _gex_atac()
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac})
    out = ShardedArchive(f).to_anndata(container="csr")  # v4 rebuild is CSR per family
    np.testing.assert_array_equal(np.asarray(out.X.todense()), gex.X.toarray())


def test_no_modality_write_omits_meta_and_fields(tmp_path):
    # byte-identity guard: X+layers+raw stays Plan-1-shaped (no meta dir / new fields)
    gex, _ = _gex_atac()
    f = tmp_path / "plain.shad"
    write_sharded(gex, f)
    arch = ShardedArchive(f)
    m = arch.manifest
    assert m["schema_version"] == 4
    for md in m["matrices"].values():
        assert "modality_name" not in md and "meta_filename" not in md
    # Stronger than the field check above: no "meta/*" member was ever packed
    # into the archive at all (the packed analog of the directory-format
    # `assert not (d / "meta").exists()` guard in
    # tests/v2/test_multimatrix_modality_writer.py).
    assert not any(mem["name"].startswith("meta/") for mem in arch._packed.members)


def test_write_determinism(tmp_path):
    gex, atac = _gex_atac()
    f1, f2 = tmp_path / "a.shad", tmp_path / "b.shad"
    write_sharded(gex, f1, modalities={"atac": atac})
    write_sharded(gex, f2, modalities={"atac": atac})
    # shard families are byte-identical across two writes of the same input
    a1, a2 = ShardedArchive(f1), ShardedArchive(f2)
    for key in ("x", "modality:atac", "modality:atac:layer:tfidf"):
        b1 = a1._packed.member_bytes(a1._matrices[key]["shard_filename_template"].format(0))
        b2 = a2._packed.member_bytes(a2._matrices[key]["shard_filename_template"].format(0))
        assert b1 == b2


def test_native_modalities_to_mudata_roundtrip(tmp_path):
    # T8c: a native `modalities=` write (no MuData input) reconstructs a MuData
    # via to_mudata(). Primary modality name defaults to "rna".
    pytest.importorskip("mudata")
    gex, atac = _gex_atac()
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac})
    mdata = ShardedArchive(f).to_mudata()
    assert set(mdata.mod) == {"rna", "atac"}
    np.testing.assert_array_equal(mdata.mod["atac"].X.toarray(), atac.X.toarray())
    assert list(mdata.mod["atac"].var.index) == list(atac.var.index)


def test_path_source_primary_plus_modalities_roundtrip(tmp_path):
    # T8c: a path-source primary (.h5ad on disk) + in-memory modalities= writes
    # and round-trips through to_mudata().
    pytest.importorskip("mudata")
    gex, atac = _gex_atac()
    gex_path = tmp_path / "gex.h5ad"
    gex.write_h5ad(gex_path)
    f = tmp_path / "mm.shad"
    write_sharded(str(gex_path), f, modalities={"atac": atac})
    mdata = ShardedArchive(f).to_mudata()
    assert "atac" in mdata.mod
    np.testing.assert_array_equal(mdata.mod["atac"].X.toarray(), atac.X.toarray())
