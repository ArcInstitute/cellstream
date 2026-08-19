"""Phase C tests: target-aware reader API.

Task C1: GroupShard + ShardedArchive.read_reference
Task C2: ShardedArchive.iter_group_shards
Task C3: ShardedArchive.read_group + legacy / error handling
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from tests._archive_helpers import (
    archive_groups,
    pack_directory,
    write_archive,
    write_directory,
)

# ---------------------------------------------------------------------------
# Helpers (shared with Phase B test — kept minimal here)
# ---------------------------------------------------------------------------


def _toy_adata(labels, n_vars=50, seed=42):
    """Build a small AnnData with an obs['gene'] column.

    X is a CSR float32 matrix (~20% density). Input is UNSORTED so the
    planner permutation is genuinely non-identity.
    """
    n = len(labels)
    rng = np.random.default_rng(seed)
    nnz = max(1, int(n * n_vars * 0.2))
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, 50, size=nnz, dtype=np.uint16)
    coo = sp.coo_matrix((vals, (rows, cols)), shape=(n, n_vars))
    csr = coo.tocsr().astype("float32")
    csr.indices = csr.indices.astype(np.int32)
    csr.indptr = csr.indptr.astype(np.int64)
    return ad.AnnData(X=csr, obs={"gene": labels})


def _write_grouped(tmp_path, labels, reference=None, target_shard_bytes=5_000):
    """Write a group_by='gene' PACKED archive. Returns (out_path, original_adata).

    Packed (#154 part B3a, route A): every test through this helper reads with the public
    ShardedArchive and asserts on group spans, references and row counts -- none inspects a
    loose member, so the container is incidental.
    """
    a = _toy_adata(labels)
    out = tmp_path / "arc"
    write_archive(
        a,
        out,
        group_by="gene",
        reference=reference,
        target_shard_bytes=target_shard_bytes,
    )
    return out, a


def _write_grouped_without_groups_json(tmp_path, labels, reference=None, target_shard_bytes=5_000):
    """A grouped PACKED archive whose ``groups.json`` member is ABSENT. Returns (path, adata).

    Route E, tamper-then-pack -- and it needs **no** low-level packed-corruption helper, which
    is where an earlier revision of the #154 B3a plan had this wrong. ``groups.json`` is
    *optional* in ``packed/writer.py``'s ``_METADATA_ORDER``, and ``_enumerate_members`` simply
    skips a member that is not on disk. So staging a directory, unlinking the file and packing
    yields exactly the state under test: a manifest that declares ``group_by`` over a container
    with no groups member.

    This is strictly better than staying on a directory, because the branch it reaches is the one
    that SHIPS: ``read.py:1074`` (the packed arm) rather than ``read.py:1093`` (the directory arm
    B3b deletes). The two arms raise near-identical messages, both naming ``group_by``.
    """
    a = _toy_adata(labels)
    staging = tmp_path / "arc_dir"
    write_directory(
        a,
        staging,
        group_by="gene",
        reference=reference,
        target_shard_bytes=target_shard_bytes,
    )
    (staging / "groups.json").unlink()
    return pack_directory(staging, tmp_path / "arc_no_groups.shad"), a


def _write_legacy_v2(tmp_path, n=60, n_shards=2):
    """Write a plain v2 archive (no group_by). Returns out_dir."""

    labels = ["X"] * n
    a = _toy_adata(labels)
    out = tmp_path / "legacy"
    write_archive(a, out, n_shards=n_shards)  # route A: read publicly, no loose access
    return out


# ---------------------------------------------------------------------------
# Task C1: read_reference + GroupShard basics
# ---------------------------------------------------------------------------


def test_read_reference_returns_correct_cells(tmp_path):
    """read_reference() returns exactly the NTC cells."""
    labels = np.array(["GENEB"] * 30 + ["NTC"] * 40 + ["GENEA"] * 30)  # unsorted
    out, orig = _write_grouped(tmp_path, labels, reference=["NTC"])

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)
    ref = arc.read_reference()

    assert ref is not None
    assert ref.n_obs == 40
    assert set(ref.obs["gene"].tolist()) == {"NTC"}


def test_read_reference_none_when_no_reference(tmp_path):
    """read_reference() returns None when written without reference."""
    labels = np.array(["GENEA"] * 30 + ["GENEB"] * 30)
    out, _ = _write_grouped(tmp_path, labels, reference=None)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)
    assert arc.read_reference() is None


def test_group_shard_attributes(tmp_path):
    """GroupShard has .labels, .groups (local slices), shard_index, etc."""
    labels = np.array(["GENEB"] * 30 + ["NTC"] * 40 + ["GENEA"] * 30)
    out, _ = _write_grouped(tmp_path, labels, reference=["NTC"])

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    shards = list(arc.iter_group_shards())
    assert len(shards) > 0

    gs = shards[0]
    # .labels returns the group labels in this shard
    assert len(gs.labels) > 0
    # .groups is a dict mapping label -> local slice
    for lab, sl in gs.groups.items():
        assert isinstance(sl, slice)
        assert lab in gs.labels
    # shard_index is accessible
    assert hasattr(gs, "shard_index")
    # global range is consistent: stop - start == number of rows
    n_rows = gs.global_stop - gs.global_start
    assert n_rows > 0


def test_group_shard_to_anndata_deferred(tmp_path):
    """GroupShard.to_anndata() returns the correct shard data (deferred I/O)."""
    labels = np.array(["GENEB"] * 30 + ["NTC"] * 40 + ["GENEA"] * 30)
    out, _ = _write_grouped(tmp_path, labels, reference=["NTC"])

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    for gs in arc.iter_group_shards():
        adata = gs.to_anndata(cast_back=True, n_workers=1)
        # All cells in the shard are from genes in gs.labels
        cell_genes = set(adata.obs["gene"].tolist())
        assert cell_genes <= set(gs.labels), (
            f"shard {gs.shard_index}: unexpected genes {cell_genes - set(gs.labels)}"
        )
        # Row count matches global range
        assert adata.n_obs == gs.global_stop - gs.global_start


# ---------------------------------------------------------------------------
# Task C2: iter_group_shards
# ---------------------------------------------------------------------------


def test_iter_group_shards_covers_all_non_reference_genes(tmp_path):
    """iter_group_shards yields every non-reference gene exactly once."""
    labels = np.array(["GENEB"] * 30 + ["NTC"] * 40 + ["GENEA"] * 30)
    out, _ = _write_grouped(tmp_path, labels, reference=["NTC"])

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    seen_labels = []
    for gs in arc.iter_group_shards():
        seen_labels.extend(gs.labels)

    assert sorted(seen_labels) == ["GENEA", "GENEB"]


def test_iter_group_shards_local_slices_are_correct(tmp_path):
    """GroupShard.groups local slices select the right per-gene cells."""
    labels = np.array(["GENEB"] * 30 + ["NTC"] * 40 + ["GENEA"] * 30)
    out, _ = _write_grouped(tmp_path, labels, reference=["NTC"])

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    for gs in arc.iter_group_shards():
        adata = gs.to_anndata(cast_back=True, n_workers=1)
        for lab, sl in gs.groups.items():
            # Local slice selects only cells of that gene
            subset = adata[sl]
            cell_genes = set(subset.obs["gene"].tolist())
            assert cell_genes == {lab}, f"local slice for {lab!r} contains genes {cell_genes}"
            # Slice is within bounds of the shard
            n_shard = gs.global_stop - gs.global_start
            assert 0 <= sl.start
            assert sl.stop <= n_shard


def test_iter_group_shards_no_reference_role(tmp_path):
    """iter_group_shards never yields the reference shard."""
    labels = np.array(["GENEB"] * 30 + ["NTC"] * 40 + ["GENEA"] * 30)
    out, _ = _write_grouped(tmp_path, labels, reference=["NTC"])

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    # archive_groups, not load_groups(out / "groups.json"): B1 shipped this container-agnostic
    # accessor precisely so a member read survives the packed migration -- in a packed archive
    # there is no loose groups.json path at all.
    all_groups = archive_groups(out)
    ref_shard_indices = {g["shard"] for g in all_groups if g["role"] == "reference"}

    for gs in arc.iter_group_shards():
        assert gs.shard_index not in ref_shard_indices, (
            f"iter_group_shards yielded reference shard {gs.shard_index}"
        )


def test_iter_group_shards_groups_json_cached(tmp_path):
    """groups.json is loaded lazily and cached (same object on second call)."""
    labels = np.array(["GENEA"] * 30 + ["GENEB"] * 30)
    out, _ = _write_grouped(tmp_path, labels, reference=None)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    # First iteration loads groups.json
    list(arc.iter_group_shards())
    cached = arc._groups_cache

    # Second iteration should use the same cached object
    list(arc.iter_group_shards())
    assert arc._groups_cache is cached


def test_iter_group_shards_no_reference_no_exclusion(tmp_path):
    """iter_group_shards without reference covers all genes."""
    labels = np.array(["GENEA"] * 30 + ["GENEB"] * 40 + ["GENEC"] * 20)
    out, _ = _write_grouped(tmp_path, labels, reference=None)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    seen = []
    for gs in arc.iter_group_shards():
        seen.extend(gs.labels)

    assert sorted(seen) == ["GENEA", "GENEB", "GENEC"]


# ---------------------------------------------------------------------------
# Task C3: read_group + legacy/error handling
# ---------------------------------------------------------------------------


def test_read_group_returns_correct_cells(tmp_path):
    """read_group(label) returns exactly the cells for that gene."""
    # Unsorted input: GENEB first, then NTC, then GENEA
    labels = np.array(["GENEB"] * 30 + ["NTC"] * 40 + ["GENEA"] * 30)
    out, orig = _write_grouped(tmp_path, labels, reference=["NTC"])

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    for gene in ["GENEA", "GENEB"]:
        result = arc.read_group(gene)
        cell_genes = set(result.obs["gene"].tolist())
        assert cell_genes == {gene}, f"read_group({gene!r}) returned genes {cell_genes}"
        # Count matches expectation (30 each)
        assert result.n_obs == 30


def test_read_group_values_match_original(tmp_path):
    """read_group recovers the original X values via _orig_row."""
    # Deliberately unsorted so permutation is non-identity
    n_gene = 25
    labels = np.array(["GENEB"] * n_gene + ["NTC"] * 20 + ["GENEA"] * n_gene)
    out, orig = _write_grouped(tmp_path, labels, reference=["NTC"], target_shard_bytes=2_000)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)
    orig_X = orig.X.toarray()
    orig_genes = orig.obs["gene"].to_numpy()

    for gene in ["GENEA", "GENEB"]:
        result = arc.read_group(gene, cast_back=True, n_workers=1)
        orig_rows = result.obs["_orig_row"].to_numpy()
        # Verify that the original rows all belong to the requested gene
        assert all(orig_genes[r] == gene for r in orig_rows)
        # Verify X values match original
        reconstructed = result.X.toarray()
        expected = orig_X[orig_rows]
        np.testing.assert_array_equal(reconstructed, expected)


def test_read_group_unknown_label_raises_key_error(tmp_path):
    """read_group raises KeyError for an unknown label."""
    labels = np.array(["GENEA"] * 30 + ["GENEB"] * 30)
    out, _ = _write_grouped(tmp_path, labels, reference=None)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    with pytest.raises(KeyError):
        arc.read_group("NONEXISTENT_GENE")


def test_read_group_unknown_label_error_mentions_near_matches(tmp_path):
    """KeyError for unknown label should mention near matches (e.g. GENEA vs GENE)."""
    labels = np.array(["GENEA"] * 30 + ["GENEB"] * 30)
    out, _ = _write_grouped(tmp_path, labels, reference=None)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    with pytest.raises(KeyError, match="GENE"):
        arc.read_group("GENE")


# ---------------------------------------------------------------------------
# Task C3 continued: legacy archive guard
# ---------------------------------------------------------------------------


def test_read_reference_legacy_raises_value_error(tmp_path):
    """read_reference() on a legacy v2 archive (no group_by) → ValueError."""
    out = _write_legacy_v2(tmp_path)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    with pytest.raises(ValueError, match="group_by"):
        arc.read_reference()


def test_iter_group_shards_legacy_raises_value_error(tmp_path):
    """iter_group_shards() on a legacy v2 archive → ValueError."""
    out = _write_legacy_v2(tmp_path)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    with pytest.raises(ValueError, match="group_by"):
        list(arc.iter_group_shards())


def test_read_group_legacy_raises_value_error(tmp_path):
    """read_group() on a legacy v2 archive → ValueError."""
    out = _write_legacy_v2(tmp_path)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    with pytest.raises(ValueError, match="group_by"):
        arc.read_group("anything")


def test_legacy_error_hints_rewrite(tmp_path):
    """The ValueError hint message contains 'group_by' and a re-write suggestion."""
    out = _write_legacy_v2(tmp_path)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    for method_call in [
        lambda: arc.read_reference(),
        lambda: list(arc.iter_group_shards()),
        lambda: arc.read_group("x"),
    ]:
        with pytest.raises(ValueError) as exc_info:
            method_call()
        msg = str(exc_info.value)
        assert "group_by" in msg, f"hint not in error: {msg!r}"


# ---------------------------------------------------------------------------
# Finding 5: missing groups.json raises clear ValueError (not FileNotFoundError)
# ---------------------------------------------------------------------------


def test_missing_groups_json_raises_value_error_not_fnf(tmp_path):
    """iter_group_shards on a grouped archive with groups.json absent → ValueError."""
    labels = np.array(["GENEA"] * 30 + ["NTC"] * 20)
    out, _ = _write_grouped_without_groups_json(tmp_path, labels, reference=["NTC"])

    from cellstream.packed.reader import PackedArchive
    from cellstream.read import ShardedArchive

    # The fixture is only meaningful if the member really is gone AND the manifest still
    # declares the archive grouped -- otherwise the ValueError below would be the ordinary
    # "not written with a group_by key" error from a different branch.
    pa = PackedArchive(out)
    assert not pa.has_member("groups.json")
    assert pa.manifest["group_by"] == "gene"

    arc = ShardedArchive(out)

    # Must raise ValueError (not FileNotFoundError) and mention re-writing.
    with pytest.raises(ValueError, match="groups.json is missing.*group_by"):
        list(arc.iter_group_shards())


def test_missing_groups_json_read_group_raises_value_error(tmp_path):
    """read_group() on an archive with groups.json absent → ValueError with rewrite hint."""
    labels = np.array(["GENEA"] * 30 + ["NTC"] * 20)
    out, _ = _write_grouped_without_groups_json(tmp_path, labels, reference=["NTC"])

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    with pytest.raises(ValueError, match="groups.json is missing.*group_by"):
        arc.read_group("GENEA")
