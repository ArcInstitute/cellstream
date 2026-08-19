import numpy as np
import pytest

from cellstream.write import write_sharded
from tests._archive_helpers import archive_obs, read_v2_host, write_directory


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"n_shards": 3}, id="plain"),
        pytest.param({"group_by": "target_gene", "target_shard_bytes": 8192}, id="grouped"),
    ],
)
def test_write_packed_members_identical_to_the_directory_writer(sample_adata, tmp_path, kw):
    """The PACKED writer and the DIRECTORY writer, run independently on the same input,
    produce the same members.

    Widened from shards-only to every member in #154 part B3a, taking over from
    ``test_write_packed_reads_match_directory``.

    ⚠️ Equal member bytes do **not** by themselves imply equal reads -- codex rejected that claim,
    correctly. A footer whose ``shard_idx`` values are permuted keeps every member name and byte
    intact while ``PackedArchive.shard_location`` feeds the reader the wrong shards. So this test
    also walks every ``shard_location(i)`` and compares the bytes it actually reaches, and the read
    itself is covered by ``test_write_packed_read_matches_the_directory_writers_output`` below.

    The grouped arm is what carries ``groups.json``, which the shards-only version never touched.

    ``manifest.json`` is compared as a DICT minus ``created_at`` rather than as bytes, and that
    is not a weakening -- it is a flake fix. The two writers stamp their own wall-clock time, so
    a byte comparison passes only when both writes land in the same second. (Measured: with
    ``created_at`` excluded the two manifests are otherwise equal on both arms.)
    """
    import json

    from cellstream.manifest import shard_filename
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _METADATA_ORDER

    d = tmp_path / "dir"
    f = tmp_path / "packed.shardpack"
    write_directory(sample_adata, d, **kw)
    write_sharded(sample_adata, f, format="v2", **kw)

    pa = PackedArchive(f)
    man = pa.manifest
    names = [shard_filename(man, i) for i in range(int(man["n_shards"]))] + [
        name for name in _METADATA_ORDER if (d / name).exists()
    ]
    assert sorted(mem["name"] for mem in pa.members) == sorted(names)
    if "group_by" in kw:
        assert "groups.json" in names  # or the grouped arm proves nothing extra

    for name in names:
        if name == "manifest.json":
            got = {k: v for k, v in man.items() if k != "created_at"}
            want = {
                k: v for k, v in json.loads((d / name).read_text()).items() if k != "created_at"
            }
            assert got == want
        else:
            assert pa.member_bytes(name) == (d / name).read_bytes(), name

    # by INDEX, not just by name -- see the shard_idx-permutation note in the docstring. Every shard
    # of a packed archive lives in the SAME file, so the handle is opened once (Gemini).
    with open(f, "rb") as fh:
        for i in range(int(man["n_shards"])):
            path, offset, length = pa.shard_location(i)
            assert path == str(f)  # the single-file assumption this loop rests on
            fh.seek(offset)
            assert fh.read(length) == (d / shard_filename(man, i)).read_bytes(), f"shard {i}"


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({"n_shards": 3}, id="plain"),
        pytest.param({"group_by": "target_gene", "target_shard_bytes": 8192}, id="grouped"),
    ],
)
def test_write_packed_read_matches_the_directory_writers_output(sample_adata, tmp_path, kw):
    """The packed writer's archive READS the same as the directory writer's, for both shapes.

    This is what ``test_write_packed_reads_match_directory`` asserted, minus the public directory
    read that B3b removes: the reference is now the INTERNAL reader over the directory
    (``read_v2_host``, whose locator is on the preserve-list) plus ``archive_obs``. Kept as its own
    test because the member comparison above cannot stand in for it -- see its docstring.

    ⚠️ Compares the CSR arrays EXACTLY. The deleted test's ``_same`` helper did too; codex caught
    the first version of this replacement comparing through ``toarray()``, which normalizes away
    index ordering, duplicate entries and explicit zeros -- i.e. the same weakening the replacement
    existed to avoid.
    """
    import scipy.sparse as sp

    from cellstream.read import ShardedArchive

    d = tmp_path / "dir"
    f = tmp_path / "packed.shardpack"
    write_directory(sample_adata, d, **kw)
    write_sharded(sample_adata, f, format="v2", **kw)

    want_d, want_i, want_p = read_v2_host(d)
    got = ShardedArchive(f).to_anndata()
    want = sp.csr_matrix((want_d, want_i, want_p), shape=(got.n_obs, got.n_vars))
    assert want.nnz == sample_adata.X.tocsr().nnz  # non-vacuous reference
    g = got.X.tocsr()
    assert g.shape == want.shape and g.dtype == want.dtype
    np.testing.assert_array_equal(g.indptr, want.indptr)
    np.testing.assert_array_equal(g.indices, want.indices)
    np.testing.assert_array_equal(g.data, want.data)
    # the same grouping input gives the same permutation, so obs order must agree too
    assert list(got.obs_names) == [str(x) for x in archive_obs(d).index]


# test_packed_requires_v2 lived here: packed was v2-only, so container="packed" with
# format="v1" raised "container='packed' requires format='v2'". #154 removed the v1 writer,
# so no format can reach that cross-check and write.py's guard for it went too. This was
# also the one container="packed" call site part B1 deliberately left in place, precisely
# because it was paired with format="v1".


def test_packed_overwrite(sample_adata, tmp_path):
    from cellstream.errors import ArchiveExistsError
    from cellstream.packed.reader import is_packed_file

    f = tmp_path / "packed.shardpack"
    write_sharded(sample_adata, f, format="v2", n_shards=2)
    assert is_packed_file(f)  # a packed file, not a directory
    with pytest.raises(ArchiveExistsError):
        write_sharded(sample_adata, f, format="v2", n_shards=2)
    write_sharded(sample_adata, f, format="v2", n_shards=2, overwrite=True)
    assert is_packed_file(f)
