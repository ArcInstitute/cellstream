import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellstream import ShardedArchive, write_sharded
from cellstream.v2 import _rust_dispatch
from tests._archive_helpers import archive_manifest

pytestmark = pytest.mark.skipif(
    not _rust_dispatch.rust_available(), reason="Rust core not built/enabled"
)


def _roundtrip_dense(tmp_path, adata, name, **kw):
    # Packed write: _write_v2_inmem / _write_v2_grouped still do the encoding, with the
    # packed assembly step on top (#154 part B removes the directory container).
    out = tmp_path / name
    write_sharded(adata, out, n_workers=2, **kw)
    back = ShardedArchive(out).to_anndata(cast_back=True, n_workers=2)
    X = back.X
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def test_plain_sparse_decide_once_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    dense = ((rng.random((40, 12)) < 0.3) * rng.integers(1, 50, size=(40, 12))).astype(np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(dense))
    back = _roundtrip_dense(tmp_path, adata, "arch")
    assert np.array_equal(back, dense)


def test_plain_dense_decide_once_roundtrips(tmp_path):
    rng = np.random.default_rng(1)
    dense = ((rng.random((30, 9)) < 0.4) * rng.integers(1, 20, size=(30, 9))).astype(np.float32)
    adata = ad.AnnData(X=dense)  # dense X input
    back = _roundtrip_dense(tmp_path, adata, "darch")
    assert np.array_equal(back, dense)


def test_grouped_inmem_decide_once_roundtrips(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(2)
    dense = ((rng.random((24, 10)) < 0.35) * rng.integers(1, 30, size=(24, 10))).astype(np.float32)
    X = sp.csr_matrix(dense)
    obs = pd.DataFrame({"grp": (["a"] * 8 + ["b"] * 8 + ["c"] * 8)})
    adata = ad.AnnData(X=X, obs=obs)
    out = tmp_path / "garch"
    write_sharded(adata, out, n_workers=2, group_by="grp", target_shard_bytes=4096)
    # round-trip equality is checked by the existing grouped read tests; here we
    # assert the manifest reports the uint32 on-disk dtypes + the recorded minimums.

    man = archive_manifest(out)
    assert man["x_data_dtype_on_disk"] == "uint32"
    assert man["x_indices_dtype_on_disk"] == "uint32"
    assert man["x_data_dtype_min"] == "uint8"  # values 1..29 -> uint8
    assert man["x_indices_dtype_min"] == "uint16"  # n_vars=10 <= 65536
