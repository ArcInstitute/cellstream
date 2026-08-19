"""End-to-end: write_sharded + verify file layout + recompute manifest."""

import numpy as np
import pytest
import scipy.sparse as sp

from tests._archive_helpers import archive_manifest


def _make_adata(n_obs=400, n_var=80, seed=2):
    import anndata as ad

    rng = np.random.default_rng(seed)
    nnz = int(n_obs * n_var * 0.1)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_var, size=nnz)
    data = rng.integers(0, 200, size=nnz).astype(np.float32)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_obs, n_var))
    a = ad.AnnData(X=X)
    a.obs["sample"] = (["a"] * (n_obs // 2)) + (["b"] * (n_obs - n_obs // 2))
    a.obs["sample"] = a.obs["sample"].astype("category")
    a.var["gene_id"] = [f"g{i}" for i in range(n_var)]
    a.uns["pipeline"] = "test"
    a.obsm["pca"] = rng.standard_normal((n_obs, 4))
    return a


def test_write_sharded_refuses_existing(tmp_path):
    from cellstream.errors import ArchiveExistsError
    from cellstream.write import write_sharded

    a = _make_adata()
    out = tmp_path / "arch"
    write_sharded(a, out, n_shards=2, n_workers=2)
    with pytest.raises(ArchiveExistsError):
        write_sharded(a, out, n_shards=2, n_workers=2)


def test_write_sharded_overwrite_replaces(tmp_path):
    from cellstream.write import write_sharded

    a = _make_adata()
    out = tmp_path / "arch"
    write_sharded(a, out, n_shards=2, n_workers=2)
    # Now write a different shape with overwrite=True
    a2 = _make_adata(n_obs=200, n_var=80, seed=3)
    write_sharded(a2, out, n_shards=2, n_workers=2, overwrite=True)
    # #154 part B3a, route I: overwrite is generic public-writer behaviour, not a
    # directory contract, so this drops container= and reads the manifest through the
    # container-agnostic accessor. Overwriting a packed archive replaces a FILE where
    # the directory arm replaced a tree -- a strictly harder case for the writer.
    m = archive_manifest(out)
    assert m["n_obs_total"] == 200


def _worker_writer(lock_path, started, release):
    from cellstream.lock import acquire_write_lock

    # Hold the lock manually so we control when it's released. #154 part B3a: takes the
    # LOCK PATH rather than an archive directory, because packed and directory output do
    # not lock the same file -- a directory archive locks `out/.cellstream.lock` INSIDE
    # itself, while packed locks the sibling `out.lock` (packed/writer.py:201, via
    # dst_file.with_suffix(dst_file.suffix + ".lock")). Holding the old path against a
    # packed write would contend on nothing and the test would pass vacuously.
    with acquire_write_lock(lock_path, op="write"):
        started.set()
        release.wait(timeout=60)


def test_concurrent_write_fails_loud(tmp_path):
    import multiprocessing as mp

    from cellstream.errors import ArchiveLockedError
    from cellstream.write import write_sharded

    out = tmp_path / "arch"
    # No out.mkdir(): packed output IS the file at `out`, so a pre-existing directory there
    # would make the write fail for the wrong reason.
    lock_path = out.with_suffix(out.suffix + ".lock")  # exactly what write_packed derives
    started = mp.Event()
    release = mp.Event()
    p = mp.get_context("fork").Process(
        target=_worker_writer, args=(str(lock_path), started, release)
    )
    p.start()
    try:
        assert started.wait(timeout=5)
        a = _make_adata()
        with pytest.raises(ArchiveLockedError):
            write_sharded(a, out, n_shards=2, n_workers=2)
    finally:
        release.set()
        p.join(timeout=5)


def test_write_sharded_path_input(tmp_path):
    """write_sharded(src_path, out_dir) accepts a source h5ad path and round-trips.

    It does NOT stream: this write has no group_by, and write_sharded's source-resolution
    block (the ``source = ad.read_h5ad(adata)`` arm, reached when adata is not an AnnData and
    group_by is None) loads it before calling the writer -- only a GROUPED path source is
    handed through as a path and read backed (#76). The old "streams from a source h5ad"
    wording described v1's _write_sharded_from_path, which really did stream and which #154
    removed; it had been wrong since v2 became the default. Measured, not assumed: spying on
    write_v2_archive shows an AnnData for the ungrouped case and a str for the grouped one.
    """
    import cellstream

    a = _make_adata()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    out = tmp_path / "arch"
    cellstream.write_sharded(str(src), out, n_shards=4, n_workers=2)

    sh = cellstream.ShardedArchive(out)
    assert sh.shape == (a.n_obs, a.n_vars)
    a2 = sh.to_anndata()
    assert (a2.X != a.X).nnz == 0
    assert list(a2.obs["sample"]) == list(a.obs["sample"])
    assert "pipeline" in a2.uns or a2.uns == dict(a.uns)


def test_default_format_is_v2_byte_filter(tmp_path):
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    from cellstream import ShardedArchive, write_sharded

    a = ad.AnnData(X=sp.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.float32)))
    out = tmp_path / "arch"
    write_sharded(a, str(out), n_shards=1)  # NO format / codec args -> default v2 + packed
    m = ShardedArchive(str(out)).manifest  # auto-detects packed (the new default)
    assert m["schema_version"] == 2  # v2 archive
    assert m["codec"] == "v2-byte-filter"  # byte-filter default


def test_explicit_codec_overrides_the_default(tmp_path):
    """An explicit codec= overrides the resolved default.

    Was test_explicit_codec_overrides_default_per_format: it asserted the override on
    BOTH formats, and the v1 half (an explicit blosc-zstd-1-u16u16 landing in a schema-1
    manifest) went with the v1 writer in #154. The v2 half is unchanged.
    """
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    from cellstream import write_sharded

    a = ad.AnnData(X=sp.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.float32)))
    out2 = tmp_path / "v2bit"
    write_sharded(
        a,
        str(out2),
        format="v2",
        codec="zstd-1-u16u16-bitshuffle",
        n_shards=1,
    )
    m2 = archive_manifest(out2)
    assert m2["schema_version"] == 2 and m2["codec"] != "v2-byte-filter"


def test_shard_write_with_a_literal_nan_and_missing_adds_no_new_exception(tmp_path):
    """layout='shard' must gain NO collision exception from the shared planner (#294 non-goal).

    Direct v2 DIRECTORY writers delete the prior archive before planning, which is what would
    make a planner-side raise destructive; the packed path plans into a fresh temp dir. This
    test pins the contract that matters either way: the write completes and reads back.

    The cell writer DOES reject this column -- that check is a cell-writer preflight precisely
    so the shard writers never see it.
    """
    import pandas as pd

    import cellstream

    a = _make_adata(n_obs=24, n_var=10)
    a.obs["grp"] = pd.array(["nan"] * 8 + ["g1"] * 8 + [pd.NA] * 8, dtype="string")
    out = tmp_path / "s.shad"
    # n_shards is REJECTED alongside group_by (write.py:348) -- shard count comes from
    # target_shard_bytes.
    cellstream.write_sharded(
        a, out, format="v2", group_by="grp", target_shard_bytes=4096, n_workers=1
    )
    cellstream.write_sharded(
        a, out, format="v2", group_by="grp", target_shard_bytes=4096, n_workers=1, overwrite=True
    )
    assert cellstream.read_h5ad(out).n_obs == 24


@pytest.mark.parametrize("sort_order", ["natural", "alphabetical"])
def test_shard_reference_on_a_label_that_collides_with_a_missing_value(tmp_path, sort_order):
    """#294 regression: resolve_reference and the shard writer's own key must agree.

    On pandas 2 a nullable-string column holding a literal "<NA>" beside pd.NA stringifies BOTH
    to '<NA>'. Once resolve_reference started normalizing (pd.NA -> 'nan') while the writer's
    alphabetical key still used a raw .astype(str), the mask selected only the 8 literal rows
    while the key named all 16 of them '<NA>' -- and the planner raised
    'reference_mask splits label', a new exception on the shard path that #294 must not add.

    The test above uses natural ordering and no reference, so it does not reach this.
    """
    import pandas as pd

    import cellstream

    a = _make_adata(n_obs=24, n_var=10)
    a.obs["grp"] = pd.array(["<NA>"] * 8 + ["b"] * 8 + [pd.NA] * 8, dtype="string")
    out = tmp_path / "s.shad"
    cellstream.write_sharded(
        a,
        out,
        format="v2",
        group_by="grp",
        sort_order=sort_order,
        reference=["<NA>"],
        target_shard_bytes=4096,
        n_workers=1,
    )
    assert cellstream.read_h5ad(out).n_obs == 24
