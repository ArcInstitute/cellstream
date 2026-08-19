from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.read import ShardedArchive
from cellstream.write import write_sharded


def _adata(n_obs=50, n_vars=7):
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=5)
    X = sp.csr_matrix(
        (
            (X.data * 25 + 1).astype(np.uint16),
            X.indices.astype(np.uint16),
            X.indptr.astype(np.int64),
        ),
        shape=(n_obs, n_vars),
    )
    counts = X.copy()
    counts.data = (counts.data * 3).astype(np.uint16)
    raw = sp.random(n_obs, n_vars + 2, density=0.3, format="csr", random_state=6)
    raw = sp.csr_matrix(
        (
            (raw.data * 15 + 1).astype(np.uint16),
            raw.indices.astype(np.uint16),
            raw.indptr.astype(np.int64),
        ),
        shape=(n_obs, n_vars + 2),
    )
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    a.layers["counts"] = counts
    a.raw = ad.AnnData(
        X=raw, obs=a.obs.copy(), var=pd.DataFrame(index=[f"r{i}" for i in range(n_vars + 2)])
    )
    return a


def _write(tmp_path, a):
    f = tmp_path / "m.shad"
    write_sharded(a, f, format="v2", n_shards=3)
    return ShardedArchive(f)


def test_matrices_property(tmp_path):
    arch = _write(tmp_path, _adata())
    keys = {k for k, _ in arch.matrices}
    assert keys == {"x", "layer:counts", "raw"}


def test_to_anndata_rebuilds_layers_and_raw(tmp_path):
    a = _adata()
    arch = _write(tmp_path, a)
    out = arch.to_anndata()
    np.testing.assert_array_equal(out.X.tocsr().toarray(), a.X.toarray())
    assert "counts" in out.layers
    np.testing.assert_array_equal(
        out.layers["counts"].tocsr().toarray(), a.layers["counts"].toarray()
    )
    assert out.raw is not None
    np.testing.assert_array_equal(out.raw.X.tocsr().toarray(), a.raw.X.toarray())
    assert list(out.raw.var.index) == list(a.raw.var.index)


def test_to_anndata_layer_promotes_to_X(tmp_path):
    a = _adata()
    arch = _write(tmp_path, a)
    out = arch.to_anndata(layer="counts")
    np.testing.assert_array_equal(out.X.tocsr().toarray(), a.layers["counts"].toarray())
    assert "counts" in out.layers  # remaining layers kept


def test_v4_to_anndata_heap_materialization(tmp_path):
    # #198: per-family heap materialization knobs are now honored on v4
    # (previously raised NotImplementedError). Values must be preserved.
    a = _adata()
    arch = _write(tmp_path, a)
    # data_dtype: lossless upcast to float64
    out = arch.to_anndata(data_dtype="float64")
    assert out.X.dtype == np.float64
    np.testing.assert_array_equal(out.X.toarray(), a.X.toarray())
    # cast_back=False: native on-disk dtype (integer data is stored uint32)
    outn = arch.to_anndata(cast_back=False)
    expected = np.dtype(arch.manifest["matrices"]["x"]["x_data_dtype_on_disk"])
    assert outn.X.dtype == expected
    np.testing.assert_array_equal(
        outn.X.toarray().astype(np.float64), a.X.toarray().astype(np.float64)
    )
    # dense container
    outd = arch.to_anndata(container="dense")
    assert not sp.issparse(outd.X)
    np.testing.assert_array_equal(np.asarray(outd.X), a.X.toarray())


def test_v4_to_anndata_shm_value_parity_and_backed(tmp_path):
    a = _adata()
    arch = _write(tmp_path, a)
    heap = arch.to_anndata(output_backing="heap")
    out = arch.to_anndata(output_backing="shm")
    # value parity: X, layer, raw
    np.testing.assert_array_equal(out.X.toarray(), heap.X.toarray())
    np.testing.assert_array_equal(out.layers["counts"].toarray(), heap.layers["counts"].toarray())
    np.testing.assert_array_equal(out.raw.X.toarray(), heap.raw.X.toarray())
    # shm-backed: each family carries a bundle; the container lists all three
    assert hasattr(out.X, "_cellstream_shm_bundle")
    assert hasattr(out.layers["counts"], "_cellstream_shm_bundle")
    assert hasattr(out.raw.X, "_cellstream_shm_bundle")
    assert len(out._cellstream_shm_bundles) == 3  # x, layer:counts, raw
    # explicit close is idempotent
    out.close_shared_memory()
    out.close_shared_memory()


def test_v4_to_anndata_layer_promote_shm(tmp_path):
    # layer-promote + shm: bundles are pinned BEFORE the promote, so out.X (the
    # promoted layer) is shm-backed and every family bundle stays tracked and is
    # released on close, even the x-family one that out.X no longer references.
    import gc
    import os

    a = _adata()
    arch = _write(tmp_path, a)
    before = set(os.listdir("/dev/shm"))
    out = arch.to_anndata(layer="counts", output_backing="shm")
    np.testing.assert_array_equal(out.X.toarray(), a.layers["counts"].toarray())
    assert out.X is out.layers["counts"]
    assert hasattr(out.X, "_cellstream_shm_bundle")
    assert len(out._cellstream_shm_bundles) == 3  # x, layer:counts, raw
    out.close_shared_memory()
    del out
    gc.collect()
    after = set(os.listdir("/dev/shm"))
    assert not (after - before)


def test_v4_to_anndata_shm_extract_and_drop(tmp_path):
    import gc

    arch = _write(tmp_path, _adata())
    out = arch.to_anndata(output_backing="shm")
    x = out.X
    expected = x.toarray().copy()
    del out
    gc.collect()
    # per-matrix pin keeps x's bundle mapped after the AnnData is dropped
    np.testing.assert_array_equal(x.toarray(), expected)


def test_v4_to_anndata_shm_dense(tmp_path):
    a = _adata()
    arch = _write(tmp_path, a)
    out = arch.to_anndata(container="dense", output_backing="shm")
    assert not sp.issparse(out.X)
    np.testing.assert_array_equal(np.asarray(out.X), a.X.toarray())
    assert hasattr(out.X, "_cellstream_shm_bundle")


def test_v4_to_anndata_shm_no_leak_on_failure(tmp_path, monkeypatch):
    import gc
    import os

    arch = _write(tmp_path, _adata())
    before = set(os.listdir("/dev/shm"))
    orig = ShardedArchive._read_matrix_X
    calls = {"n": 0}

    def boom(self, key, **kw):
        calls["n"] += 1
        if calls["n"] >= 2:  # fail AFTER the first family's bundle is collected
            raise RuntimeError("boom")
        return orig(self, key, **kw)

    monkeypatch.setattr(ShardedArchive, "_read_matrix_X", boom)
    with pytest.raises(RuntimeError, match="boom"):
        arch.to_anndata(output_backing="shm")
    gc.collect()
    after = set(os.listdir("/dev/shm"))
    assert not (after - before)  # the first family's bundle was closed on the failure path


def test_v4_to_anndata_heap_copy_failure_no_leak(tmp_path, monkeypatch):
    # HEAP path regression guard (Gemini review of PR #203): the pure-Python
    # fallback returns a bundle-backed VIEW that gets copied to owned heap; if
    # that copy raises, the (tracked) bundle must still be closed. Force the
    # pure-Python engine so the heap read returns a real bundle regardless of
    # how the suite is run.
    import gc
    import os

    arch = _write(tmp_path, _adata())  # write with the default engine
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")  # force bundle-returning heap read
    before = set(os.listdir("/dev/shm"))
    orig = ShardedArchive._read_matrix_X

    class _BadCopy:
        def copy(self):
            raise RuntimeError("copyboom")

    def wrap(self, key, **kw):
        _X, bundle = orig(self, key, **kw)
        # bundle is real on the pure-Python heap path; swap X for one whose
        # .copy() raises, keeping the real bundle so the leak path is exercised.
        return _BadCopy(), bundle

    monkeypatch.setattr(ShardedArchive, "_read_matrix_X", wrap)
    with pytest.raises(RuntimeError, match="copyboom"):
        arch.to_anndata(output_backing="heap")
    gc.collect()
    after = set(os.listdir("/dev/shm"))
    assert not (after - before)  # the tracked bundle was closed despite the copy failure


def test_layer_kwarg_rejected_on_v2(tmp_path):
    single = ad.AnnData(X=sp.csr_matrix(np.eye(4, dtype=np.uint16)))
    f = tmp_path / "s.shad"
    write_sharded(single, f, format="v2", n_shards=2)
    with pytest.raises(ValueError, match="layer.*multi-matrix"):
        ShardedArchive(f).to_anndata(layer="counts")


def test_view_matrix_selector_unknown_family_raises_keyerror(tmp_path):
    """On a v4 archive, a bogus matrix=/layer= raises a clear KeyError listing
    the available matrices, not a raw KeyError from _matrix_submanifest."""
    arch = _write(tmp_path, _adata())
    with pytest.raises(KeyError, match="not found in archive"):
        arch[:].x(matrix="nonexistent")
    with pytest.raises(KeyError, match="not found in archive"):
        arch[:].to_anndata(layer="nope")


def test_view_matrix_selector_rejected_on_single_matrix(tmp_path):
    """A view-based read (.x()/.to_anndata()) requesting a non-primary matrix
    on a single-matrix (v2/v3) archive must fail loud, not silently fall back
    to the primary X (regression: it used to silently return X)."""
    single = ad.AnnData(X=sp.csr_matrix(np.eye(4, dtype=np.uint16)))
    f = tmp_path / "s.shad"
    write_sharded(single, f, format="v2", n_shards=2)
    arch = ShardedArchive(f)
    with pytest.raises(ValueError, match="non-primary matrix.*single matrix"):
        arch[:].x(matrix="layer:counts")
    with pytest.raises(ValueError, match="non-primary matrix.*single matrix"):
        arch[:].to_anndata(layer="counts")
    # The default primary matrix still reads fine.
    np.testing.assert_array_equal(arch[:].x().tocsr().toarray(), single.X.toarray())


def _mod_with_obsm(n_vars, seed, prefix, n_obs=8):
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    a = ad.AnnData(
        X=sp.csr_matrix(
            ((np.arange(n_obs * n_vars) % 7) + 1).astype(np.uint16).reshape(n_obs, n_vars)
        ),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"{prefix}{i}" for i in range(n_vars)]),
    )
    a.obsm[f"X_{prefix}"] = np.arange(n_obs * 2, dtype=np.float32).reshape(n_obs, 2) + seed
    return a


def test_packed_meta_members_deduped_and_unique(tmp_path):
    # T4: two modalities each carry obsm -> each gets its own meta/<dir>.h5ad
    # sidecar. Pin that the packed writer's seen_mfs dedup lists every meta
    # member exactly once (a duplicate member name in the footer is corruption).
    gex = _mod_with_obsm(4, 0, "g")
    atac = _mod_with_obsm(5, 100, "p")
    adt = _mod_with_obsm(3, 200, "a")
    f = tmp_path / "mm.shad"
    write_sharded(gex, f, modalities={"atac": atac, "adt": adt})
    arch = ShardedArchive(f)
    assert arch._packed is not None  # .shad -> packed
    meta_names = [m["name"] for m in arch._packed.members if m["name"].startswith("meta/")]
    assert len(meta_names) == len(set(meta_names)), f"duplicate meta members: {meta_names}"
    assert len(meta_names) >= 2  # atac + adt both carry obsm


def test_update_obs_refuses_archive_with_unpreservable_members(tmp_path):
    """#254: rewriting the tail would PERMANENTLY drop the v4 per-family sidecars.

    _update_metadata_packed rebuilds the tail from role=="shard" plus a fixed list of four
    metadata blobs. A v4 multi-matrix archive also carries role=="var" (var/<dir>.parquet)
    and role=="meta" (meta/<dir>.h5ad) members, which are in neither list -- so they are
    dropped when the tail is truncated, and the .tailbak that could have restored them is
    unlinked on the success path. Silent, unrecoverable data loss from a public method.
    """
    import pytest

    from cellstream.packed.reader import PackedArchive

    a = _adata()
    f = tmp_path / "m.shad"
    write_sharded(a, f, format="v2", n_shards=3)

    # The archive really does carry members the rewrite cannot preserve.
    roles = {m.get("role") for m in PackedArchive(f).members}
    assert roles & {"var", "meta"}, f"fixture no longer has sidecars; roles={roles}"

    arch = ShardedArchive(f)
    new_obs = arch.obs.copy()
    new_obs["annotation"] = 1

    with pytest.raises(NotImplementedError, match="does not preserve"):
        arch.update_obs(new_obs)

    # And the archive is still readable -- the requested update did not rewrite the tail,
    # so the sidecars survive.
    again = ShardedArchive(f)
    assert {m.get("role") for m in PackedArchive(f).members} == roles
    assert again.to_anndata().n_obs == a.n_obs
