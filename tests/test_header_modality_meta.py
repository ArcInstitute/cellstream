# tests/test_header_modality_meta.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream import header


def _atac():
    a = ad.AnnData(
        X=sp.csr_matrix(np.ones((3, 4), dtype=np.uint16)),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["p0", "p1", "p2", "p3"]),
    )
    a.obsm["X_lsi"] = np.arange(6, dtype="float32").reshape(3, 2)
    a.varm["loadings"] = np.arange(8, dtype="float32").reshape(4, 2)
    a.uns["method"] = "lsi"
    return a


def test_modality_meta_round_trip(tmp_path):
    a = _atac()
    p = tmp_path / "meta" / "modality_atac.h5ad"
    header.write_modality_meta(a, p)
    back = header.read_modality_meta(p)
    np.testing.assert_array_equal(back.obsm["X_lsi"], a.obsm["X_lsi"])
    np.testing.assert_array_equal(back.varm["loadings"], a.varm["loadings"])
    assert back.uns["method"] == "lsi"
    assert back.n_obs == 3 and back.n_vars == 4


def test_reject_modality_raw():
    a = _atac()
    a.raw = ad.AnnData(X=sp.csr_matrix(np.ones((3, 4), dtype=np.uint16)))
    with pytest.raises(ValueError, match="raw"):
        header._reject_unsupported_modality_fields(a)


def test_modality_obsp_accepted():
    # obsp/varp on a modality are now stored (Bucket 3), no longer rejected.
    a = _atac()
    a.obsp["conn"] = sp.csr_matrix((3, 3), dtype="float32")
    a.varp["net"] = sp.csr_matrix((4, 4), dtype="float32")
    header._reject_unsupported_modality_fields(a)  # must NOT raise


def test_write_modality_meta_backstops_raw(tmp_path):
    # write_modality_meta must self-backstop (never silently drop a modality's
    # own .raw), mirroring write_header's own .raw guard.
    a = _atac()
    a.raw = ad.AnnData(X=sp.csr_matrix(np.ones((3, 4), dtype=np.uint16)))
    with pytest.raises(ValueError, match="raw"):
        header.write_modality_meta(a, tmp_path / "meta" / "m.h5ad")
