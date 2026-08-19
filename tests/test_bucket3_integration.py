import anndata as ad
import numpy as np
import scipy.sparse as sp

from cellstream import header, write_sharded
from cellstream.read import ShardedArchive


def _proc(n=12, v=6, seed=1):
    # Integer-valued X so the (pre-existing) backed-grouped-path uint narrowing is
    # satisfied — this test exercises the obsp/varp round-trip, not X dtype. The
    # graphs stay float (stored as-is in the sidecar, never narrowed).
    Xd = (np.arange(n * v) % 5).astype("float32").reshape(n, v)
    a = ad.AnnData(X=sp.csr_matrix(Xd))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{j}" for j in range(v)]
    a.obsp["connectivities"] = sp.random(
        n, n, density=0.3, format="csr", dtype="float32", random_state=seed + 1
    )
    a.varp["vp"] = sp.random(
        v, v, density=0.4, format="csr", dtype="float32", random_state=seed + 2
    )
    return a


def test_backed_path_grouped_roundtrips_graphs(tmp_path):
    # The would-be fail-loud corner: v2 grouped write from a BACKED PATH source.
    a = _proc()
    a.obs["ct"] = (["A"] * 6) + (["B"] * 6)
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    dst = tmp_path / "g.shad"
    write_sharded(str(src), dst, format="v2", group_by="ct")
    back = ShardedArchive(dst).to_anndata()
    orig = np.asarray(back.obs["_orig_row"])
    exp = a.obsp["connectivities"][orig][:, orig]
    assert (back.obsp["connectivities"] != exp).nnz == 0


def test_no_graph_header_has_no_empty_graph_groups(tmp_path):
    # Byte-identity guard: an archive WITHOUT graphs writes no obsp/varp groups.
    a = _proc()
    del a.obsp["connectivities"]
    del a.varp["vp"]
    dst = tmp_path / "a.shad"
    write_sharded(a, dst, format="v2")
    h = ShardedArchive(dst)._packed.header_adata()
    assert len(h.obsp) == 0 and len(h.varp) == 0


def test_old_style_meta_is_column_less_obs(tmp_path):
    # Back-compat: a modality meta sidecar with a column-less obs stays column-less
    # on read (the to_mudata fallback then supplies the global obs).
    m = ad.AnnData(X=sp.csr_matrix((4, 3), dtype="float32"))
    m.obs_names = [f"c{i}" for i in range(4)]
    p = tmp_path / "meta.h5ad"
    header.write_modality_meta(m, p, perm=None)
    back = ad.read_h5ad(p)
    assert len(back.obs.columns) == 0
