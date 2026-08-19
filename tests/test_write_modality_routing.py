# tests/test_write_modality_routing.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellstream.packed.reader import PackedArchive
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

    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    gex = ad.AnnData(
        X=_csr(5, 1), obs=obs.copy(), var=pd.DataFrame(index=[f"g{i}" for i in range(5)])
    )
    atac = ad.AnnData(
        X=_csr(7, 2), obs=obs.copy(), var=pd.DataFrame(index=[f"p{i}" for i in range(7)])
    )
    return gex, atac


def test_write_sharded_routes_modalities_to_v4_packed(tmp_path):
    gex, atac = _gex_atac()
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac})
    pa = PackedArchive(f)
    m = pa.manifest
    assert m["schema_version"] == 4
    assert "modality:atac" in m["matrices"]
    assert m["matrices"]["x"]["modality_name"] == "rna"  # native default label
