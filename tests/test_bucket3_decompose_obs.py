import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellstream import matrices as mx


def _ad(n, v, seed):
    a = ad.AnnData(X=sp.random(n, v, density=0.5, format="csr", dtype="float32", random_state=seed))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"m{seed}_{j}" for j in range(v)]
    return a


def test_decompose_accepts_modality_own_obs_column():
    n = 6
    primary, atac = _ad(n, 4, 1), _ad(n, 5, 2)
    atac.obs["nucleosome_signal"] = np.arange(n, dtype="float64")  # modality-only obs
    specs = mx.decompose(primary, {"atac": atac})  # must NOT raise
    assert any(s.role == "modality:atac" for s in specs)


def test_decompose_still_rejects_unaligned():
    primary = _ad(6, 4, 1)
    atac = _ad(5, 5, 2)  # different n_obs
    with pytest.raises(ValueError, match="not aligned"):
        mx.decompose(primary, {"atac": atac})
