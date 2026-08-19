"""Packing fidelity: a packed archive holds the directory's members BYTE-FOR-BYTE.

This file used to hold five directory-vs-packed READ-parity tests (full read, subset read,
grouped read, cast_back=False, basic attrs). #154 part B3b removes the public directory
reader, so those comparisons could only survive by packing both sides -- which makes every
assertion ``x == x``, the exact tautology B2 deleted ``test_perf_cpu.py`` for.

Two tests replace them.

``test_pack_preserves_every_member_byte_for_byte`` asserts the property they were really
protecting: ``_pack_directory`` is a pure byte concat (``packed/writer.py:111``), so every member
of the packed archive equals the loose file it came from.

⚠️ That is **not on its own stronger** than read parity, and an earlier version of this docstring
claimed it was. codex was right to reject the claim: ``ShardedArchive`` takes *different branches*
for packed and directory (``_read_v2_source_kwargs``, ``_load_groups``, ``obs``, ``var``) and its
matrix reads resolve through ``PackedArchive.shard_location``, so equal member bytes do not by
themselves prove the packed reader returns the right data. Concretely, swapping two ``shard_idx``
values in the footer leaves every member name and every member byte untouched and still feeds the
reader the wrong shards.

So ``test_packed_read_matches_the_internal_directory_read`` covers the read side, with the
INTERNAL directory read as its reference rather than a public one. That keeps it out of the
tautology (packed vs packed) *and* out of B3b's way -- ``read_v2_to_host``'s directory locator is
on the preserve-list, while the public directory reader is not.

``tests/packed/test_reader.py`` already checks two members individually
(``test_member_bytes_match_files``) plus the manifest / header / obs equivalences; the first test
here covers the COMPLETE member set, for a plain and a grouped fixture.
"""

import numpy as np

from cellstream.manifest import shard_filename
from cellstream.packed.reader import PackedArchive
from cellstream.packed.writer import _METADATA_ORDER
from cellstream.packed.writer import _pack_directory as pack
from cellstream.read import ShardedArchive
from tests._archive_helpers import (
    archive_groups,
    archive_manifest,
    archive_obs,
    archive_var,
    read_v2_host,
)


def test_pack_preserves_every_member_byte_for_byte(plain_dir, grouped_dir, tmp_path):
    for src in (plain_dir, grouped_dir):
        dst = tmp_path / f"{src.name}.shad"
        pack(src, dst)
        pa = PackedArchive(dst)

        # The expected member set comes from the WRITER'S OWN canonical list, never from a
        # hand-written glob: such a pattern omitted groups.json once already during #154 part
        # B3a, and that is exactly the member the grouped fixture needs covered.
        man = pa.manifest
        expected = [shard_filename(man, i) for i in range(int(man["n_shards"]))] + [
            name for name in _METADATA_ORDER if (src / name).exists()
        ]
        assert sorted(mem["name"] for mem in pa.members) == sorted(expected)

        # ...and nothing loose was skipped. The one file in the directory that is NOT a member
        # is the writer's lock sidecar -- v2/writer.py:443 creates out/.cellstream.lock and
        # lock.py never removes it (the diagnostic body is left for post-mortem). So listing
        # the directory is a real cross-check rather than a restatement of the line above: a
        # member the enumerator forgot would surface here.
        assert sorted(f.name for f in src.iterdir()) == sorted([*expected, ".cellstream.lock"])

        for name in expected:
            assert pa.member_bytes(name) == (src / name).read_bytes(), f"{src.name}/{name}"

    # the grouped fixture must genuinely carry the grouped tail, or the loop proves nothing
    # about groups.json
    assert (grouped_dir / "groups.json").exists()


def test_shard_location_resolves_to_the_matching_loose_shard(plain_dir, grouped_dir, tmp_path):
    """Every ``shard_location(i)`` must reach shard *i*'s bytes, not merely *a* shard's.

    The member-bytes test above compares BY NAME, so it cannot see a footer whose ``shard_idx``
    assignments are permuted: names and bytes are all still present and correct, while the reader
    -- which resolves matrix reads through ``shard_location(i)`` -- gets the wrong shard. This is
    the gap codex identified in the original replacement.
    """
    for src in (plain_dir, grouped_dir):
        dst = tmp_path / f"{src.name}_loc.shad"
        pack(src, dst)
        pa = PackedArchive(dst)
        man = pa.manifest
        assert int(man["n_shards"]) >= 2  # a 1-shard fixture could not detect a permutation
        for i in range(int(man["n_shards"])):
            path, offset, length = pa.shard_location(i)
            with open(path, "rb") as fh:
                fh.seek(offset)
                got = fh.read(length)
            assert got == (src / shard_filename(man, i)).read_bytes(), f"{src.name} shard {i}"


def _assert_same_csr(got, want, what):
    """Compare CSR arrays EXACTLY -- never through ``toarray()``.

    ``toarray()`` normalizes away index ordering, duplicate entries and explicit zeros, so two
    structurally different reads compare equal. The five deleted parity tests used a helper that
    compared ``indptr`` / ``indices`` / ``data`` directly; codex caught the first replacement here
    regressing to ``toarray()`` -- the same weakening class it was written to repair.
    """
    g, w = got.tocsr(), want.tocsr()
    assert g.shape == w.shape, f"{what}: shape {g.shape} != {w.shape}"
    assert g.dtype == w.dtype, f"{what}: dtype {g.dtype} != {w.dtype}"
    np.testing.assert_array_equal(g.indptr, w.indptr, err_msg=f"{what}: indptr")
    np.testing.assert_array_equal(g.indices, w.indices, err_msg=f"{what}: indices")
    np.testing.assert_array_equal(g.data, w.data, err_msg=f"{what}: data")


def _internal_csr(src, n_vars, **kw):
    """The reference: an INTERNAL directory read, assembled as CSR."""
    import scipy.sparse as sp

    d, i, ptr = read_v2_host(src, **kw)
    return sp.csr_matrix((d, i, ptr), shape=(len(ptr) - 1, n_vars))


def _assert_same_read(got, src, n_vars, what, *, rows=None, **kw):
    """Compare a public packed read against the internal directory read -- X **and** the axis names.

    The deleted ``_assert_same_anndata`` checked ``obs_names`` and ``var_names`` as well as the CSR
    arrays, and codex round 3 caught the restored version comparing X only. That gap is not
    hypothetical for the selectors: a regression that sorts X but leaves obs in the caller's input
    order produces the right SHAPE and the right data, and only the names disagree -- most visibly
    for the ``[5, 200, 7, 7, 199]`` case, whose correct order is ``[5, 7, 7, 199, 200]`` with the
    duplicate kept.

    ``rows``: a positional indexer into the directory's obs index giving the expected row order.
    ``None`` means "all rows in storage order".
    """
    _assert_same_csr(got.X, _internal_csr(src, n_vars, **kw), what)
    want_obs = archive_obs(src).index
    want_obs = list(want_obs) if rows is None else list(want_obs[rows])
    assert list(got.obs_names) == [str(x) for x in want_obs], f"{what}: obs_names"
    assert list(got.var_names) == [str(x) for x in archive_var(src).index], f"{what}: var_names"


def test_packed_read_matches_the_internal_directory_read(plain_dir, grouped_dir, tmp_path):
    """Pack a directory, then READ the packed result through the public reader.

    Byte-preserved members do not imply a correct read: the public reader has separate packed and
    directory branches for source kwargs, groups, obs and var. The reference here is the INTERNAL
    directory read (``read_v2_host`` on the directory, whose locator B3b preserves), never a public
    directory read -- so this is neither the packed-vs-packed tautology nor a dependency on the
    surface being removed.

    Covers the same ground as the five deleted parity tests: full read, the three subset selectors,
    ``cast_back=False``, and the basic attrs.
    """
    for src in (plain_dir, grouped_dir):
        dst = tmp_path / f"{src.name}_read.shad"
        pack(src, dst)
        arc = ShardedArchive(dst)
        # Expectations come from the DIRECTORY's manifest, so the packed side is checked against an
        # independent source. Reading them out of PackedArchive(dst) -- which is where
        # ShardedArchive(dst) gets them too -- would have been no oracle at all (codex round 3).
        src_man = archive_manifest(src)
        n_vars = int(src_man["n_vars"])
        n_obs = int(src_man["n_obs_total"])

        # basic attrs (was test_packed_basic_attrs, without the public directory read)
        assert arc.shape == (n_obs, n_vars)
        assert arc.n_shards == int(src_man["n_shards"])

        # full read (was test_full_read_parity)
        _assert_same_read(arc.to_anndata(), src, n_vars, f"{src.name} full")

        # cast_back=False -- the on-disk dtype, not the cast-back one (was
        # test_cast_back_false_parity). Asserted explicitly so a silently-changed default
        # cannot make the comparison trivial.
        native = arc.to_anndata(cast_back=False)
        assert native.X.dtype == np.uint32
        _assert_same_read(native, src, n_vars, f"{src.name} native", cast_back=False)

        # the three subset selectors (was test_subset_read_parity)
        _assert_same_read(
            arc[10:120].to_anndata(),
            src,
            n_vars,
            f"{src.name} slice",
            rows=slice(10, 120),
            row_start=10,
            row_stop=120,
        )
        mask = np.zeros(n_obs, dtype=bool)
        mask[::3] = True
        _assert_same_read(
            arc[mask].to_anndata(), src, n_vars, f"{src.name} mask", rows=mask, row_mask=mask
        )
        # sorted, duplicate RETAINED -- v2/reader_cpu._resolve_row_selector sorts row_indices and
        # does not de-duplicate, and both the public and internal paths go through that one function
        idx = np.array([5, 200, 7, 7, 199], dtype=np.int64)
        _assert_same_read(
            arc[idx].to_anndata(),
            src,
            n_vars,
            f"{src.name} indices",
            rows=np.sort(idx),
            row_indices=idx,
        )

    # grouped APIs -- the three read_v2_to_host cannot serve, and the only reason
    # test_grouped_read_parity existed. Compared by CONTENT against an internal read of the
    # record's own row range, not by row count: an equal-sized but wrong range would pass a
    # count check.
    dst = tmp_path / "grouped_api.shad"
    pack(grouped_dir, dst)
    arc = ShardedArchive(dst)
    n_vars = int(PackedArchive(dst).manifest["n_vars"])
    recs = archive_groups(grouped_dir)
    non_ref = [r for r in recs if r.get("role") != "reference"]
    assert non_ref, "the grouped fixture must have at least one non-reference group"
    rec = non_ref[0]
    _assert_same_read(
        arc.read_group(rec["label"]),
        grouped_dir,
        n_vars,
        f"read_group({rec['label']!r})",
        rows=slice(int(rec["row_start"]), int(rec["row_stop"])),
        row_start=int(rec["row_start"]),
        row_stop=int(rec["row_stop"]),
    )
    assert [s.shard_index for s in arc.iter_group_shards()] == sorted(
        {int(r["shard"]) for r in non_ref}
    )
    ref = [r for r in recs if r.get("role") == "reference"]
    assert len(ref) == 1, "the fixture is written with reference=['g00'], i.e. one contiguous span"
    _assert_same_read(
        arc.read_reference(),
        grouped_dir,
        n_vars,
        "read_reference",
        rows=slice(int(ref[0]["row_start"]), int(ref[0]["row_stop"])),
        row_start=int(ref[0]["row_start"]),
        row_stop=int(ref[0]["row_stop"]),
    )


def test_to_anndata_indptr_no_overflow_at_large_nnz(tmp_path):
    """Regression: indptr must not overflow for archives with total_nnz > 65535.

    Restored in #154. It lived in tests/test_read_sharded.py and was deleted with that
    file, but it never tested v1 -- ``write_sharded`` here takes the DEFAULT packed v2 path
    and only the read alias (``ShardedH5ad``) was v1-era. Re-pointed at ``ShardedArchive``.
    """
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    from cellstream.read import ShardedArchive
    from cellstream.write import write_sharded

    rng = np.random.default_rng(42)
    # A fully-dense 700x100 matrix (70 000 nnz) so dedup is moot and nnz > 65535.
    n_obs, n_var = 700, 100
    X = sp.csr_matrix(rng.integers(1, 200, size=(n_obs, n_var)).astype(np.float32))
    assert X.nnz > 65535

    a = ad.AnnData(X=X)
    a.obs["s"] = ["a"] * n_obs
    a.var["g"] = [f"g{i}" for i in range(n_var)]
    out = tmp_path / "big_nnz.shad"
    write_sharded(a, out, n_shards=4, n_workers=2)

    # cast_back=False with > 65k nnz: indptr must stay int64 and its last entry must
    # actually exceed the uint16 range, or the regression would pass vacuously.
    a2 = ShardedArchive(out).to_anndata(cast_back=False)
    assert a2.X.indptr.dtype == np.int64
    assert int(a2.X.indptr[-1]) > 65535
    assert np.array_equal(a.X.toarray(), a2.X.astype(np.float32).toarray())
