import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellstream import write_sharded
from cellstream.read import ShardedArchive

mudata = pytest.importorskip("mudata")


def _mod(n, v, seed):
    a = ad.AnnData(X=sp.random(n, v, density=0.5, format="csr", dtype="float32", random_state=seed))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"m{seed}_{j}" for j in range(v)]
    return a


def test_to_mudata_roundtrips_graphs_and_per_modality_obs(tmp_path):
    n = 10
    rna, atac = _mod(n, 5, 1), _mod(n, 7, 2)
    rna.obsp["rna_conn"] = sp.random(
        n, n, density=0.3, format="csr", dtype="float32", random_state=5
    )  # PRIMARY modality's own graph
    atac.obs["nucleosome_signal"] = np.arange(n, dtype="float64")
    atac.obsp["atac_conn"] = sp.random(
        n, n, density=0.3, format="csr", dtype="float32", random_state=3
    )
    md = mudata.MuData({"rna": rna, "atac": atac})
    md.obs["ct"] = ["A"] * n
    md.obsp["wnn"] = sp.random(n, n, density=0.3, format="csr", dtype="float32", random_state=4)

    dst = tmp_path / "mm.shad"
    write_sharded(md, dst, format="v2", primary_modality="rna")
    back = ShardedArchive(dst).to_mudata()
    assert "wnn" in back.obsp  # global graph
    assert "nucleosome_signal" in back["atac"].obs.columns  # per-modality obs
    assert "atac_conn" in back["atac"].obsp  # per-modality graph
    assert (back["atac"].obsp["atac_conn"] != atac.obsp["atac_conn"]).nnz == 0
    # PRIMARY modality's own obsp round-trips too (via meta/x.h5ad)
    assert "rna_conn" in back["rna"].obsp
    assert (back["rna"].obsp["rna_conn"] != rna.obsp["rna_conn"]).nnz == 0


def _mod_with_layer(n, v, seed):
    a = _mod(n, v, seed)
    lc = sp.random(n, v, density=0.5, format="csr", dtype="float32", random_state=seed + 100)
    a.layers["norm"] = lc
    return a


def test_to_mudata_shm_value_parity_and_backed(tmp_path):
    n = 12
    rna, atac = _mod_with_layer(n, 5, 1), _mod_with_layer(n, 7, 2)
    md = mudata.MuData({"rna": rna, "atac": atac})
    md.obs["ct"] = ["A"] * n
    dst = tmp_path / "mm.shad"
    write_sharded(md, dst, format="v2", primary_modality="rna")
    arch = ShardedArchive(dst)

    heap = arch.to_mudata(output_backing="heap")
    out = arch.to_mudata(output_backing="shm")
    for name in ("rna", "atac"):
        np.testing.assert_array_equal(out[name].X.toarray(), heap[name].X.toarray())
        np.testing.assert_array_equal(
            out[name].layers["norm"].toarray(), heap[name].layers["norm"].toarray()
        )
        # each modality AnnData is shm-backed (anchor X + its "norm" layer)
        assert hasattr(out[name], "_cellstream_shm_bundles")
        assert hasattr(out[name].X, "_cellstream_shm_bundle")
        assert hasattr(out[name].layers["norm"], "_cellstream_shm_bundle")
    # MuData-level fanout close is present and idempotent
    out.close_shared_memory()
    out.close_shared_memory()


def test_to_mudata_shm_no_leak_on_failure(tmp_path, monkeypatch):
    import gc
    import os

    n = 12
    rna, atac = _mod_with_layer(n, 5, 1), _mod_with_layer(n, 7, 2)
    md = mudata.MuData({"rna": rna, "atac": atac})
    dst = tmp_path / "mm.shad"
    write_sharded(md, dst, format="v2", primary_modality="rna")
    arch = ShardedArchive(dst)

    before = set(os.listdir("/dev/shm"))
    orig = ShardedArchive._read_matrix_X
    calls = {"n": 0}

    def boom(self, key, **kw):
        calls["n"] += 1
        if calls["n"] >= 3:  # fail after the primary modality's families are read
            raise RuntimeError("boom")
        return orig(self, key, **kw)

    monkeypatch.setattr(ShardedArchive, "_read_matrix_X", boom)
    with pytest.raises(RuntimeError, match="boom"):
        arch.to_mudata(output_backing="shm")
    gc.collect()
    after = set(os.listdir("/dev/shm"))
    assert not (after - before)
