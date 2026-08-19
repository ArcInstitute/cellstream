import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.read import ShardedArchive
from cellstream.v2.writer import write_v2_archive
from tests._archive_helpers import pack_directory


def _adata_dense(D):
    n = D.shape[0]
    return ad.AnnData(
        X=D,
        obs=pd.DataFrame(
            {"target_gene": [f"g{i % 3}" for i in range(n)]},
            index=[f"c{i}" for i in range(n)],
        ),
    )


def test_dense_float_input_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    D = rng.standard_normal((40, 8)).astype(np.float32)
    D[D < 0.5] = 0.0
    out = tmp_path / "arc"
    write_v2_archive(_adata_dense(D), out, group_by="target_gene", codec="v2-byte-filter")
    # route D: the DENSE-input encoder is the subject, so the write stays internal and the
    # public read goes through a packed copy (_pack_directory is a pure byte concat).
    a = ShardedArchive(pack_directory(out, tmp_path / "arc.shad")).to_anndata(cast_back=False)
    Xb = (a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)).tocsr()
    assert Xb.data.dtype == np.float32
    assert int(Xb.nnz) == int((D != 0).sum())
    assert np.allclose(np.sort(Xb.data), np.sort(D[D != 0]))


def test_dense_u16_matches_scipy_csr_bytes(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    rng = np.random.default_rng(1)
    D = (rng.integers(0, 40, size=(40, 8))).astype(np.float32)
    D[rng.random((40, 8)) < 0.5] = 0.0
    out_dense = tmp_path / "dense"
    out_csr = tmp_path / "csr"
    write_v2_archive(_adata_dense(D), out_dense, group_by="target_gene", codec="v2-byte-filter")
    write_v2_archive(
        _adata_dense(sp.csr_matrix(D)), out_csr, group_by="target_gene", codec="v2-byte-filter"
    )
    shards = sorted(p.name for p in out_csr.glob("shard_*.shx"))
    assert shards
    for shx in shards:
        assert (out_dense / shx).read_bytes() == (out_csr / shx).read_bytes()
