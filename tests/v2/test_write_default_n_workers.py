"""The public write default for n_workers is 16 (fastest in the rw-bench sweep
for both read and write; matches the reader default). Recorded in the manifest's
writer_fingerprint, so a default-args write must stamp 16."""

import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellstream import ShardedArchive, write_sharded


def test_write_sharded_default_n_workers_is_16(tmp_path):
    rng = np.random.default_rng(0)
    X = sp.random(200, 50, density=0.1, format="csr", dtype=np.float32, random_state=rng)
    X.data = np.ceil(X.data * 5).astype(np.float32)  # integer-valued counts
    out = tmp_path / "a.shad"
    write_sharded(ad.AnnData(X=X), str(out))  # no n_workers -> the default
    arch = ShardedArchive(str(out))
    assert arch.manifest["writer_fingerprint"]["n_workers_at_write"] == 16
