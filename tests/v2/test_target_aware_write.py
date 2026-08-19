"""Phase B tests: target-aware write-side integration.

Task B1: thread new kwargs + validation only.
Task B2: permute + group-aligned offsets + groups.json + v3 manifest.
Also covers n_shards=None byte-driven even split (group_by=None).
"""

from __future__ import annotations

import math

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from tests._archive_helpers import (
    archive_groups,
    archive_has_member,
    archive_manifest,
    archive_obs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_adata(labels, n_vars=50, seed=0):
    """Build a small AnnData with an obs['gene'] column.

    X is a CSR uint16 matrix (density ~0.2) so per-row nnz varies slightly.
    """
    n = len(labels)
    rng = np.random.default_rng(seed)
    # Build a random CSR with ~20% density.
    nnz = max(1, int(n * n_vars * 0.2))
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, 50, size=nnz, dtype=np.uint16)
    coo = sp.coo_matrix((vals, (rows, cols)), shape=(n, n_vars))
    csr = coo.tocsr()
    csr.data = csr.data.astype(np.uint16)
    csr.indices = csr.indices.astype(np.int32)
    csr.indptr = csr.indptr.astype(np.int64)
    return ad.AnnData(X=csr.astype("float32"), obs={"gene": labels})


def _read_obs_parquet(out_dir):
    """Return the obs DataFrame, for either container.

    Named for the sidecar it used to read directly; it now goes through archive_obs,
    which prefers obs.parquet but falls back to header.h5ad, so a successful call does
    NOT prove the parquet exists -- use archive_has_member for that.
    """
    return archive_obs(out_dir)


# ---------------------------------------------------------------------------
# Task B1: validation errors (signature + validation only)
# ---------------------------------------------------------------------------


def test_n_shards_with_group_by_errors(tmp_path):
    """n_shards and group_by together → ValueError with a guidance message."""
    a = _toy_adata(["NTC"] * 5 + ["GA"] * 5)
    from cellstream.write import write_sharded

    with pytest.raises(ValueError, match="floored by the number of groups"):
        write_sharded(a, tmp_path / "arc", format="v2", group_by="gene", n_shards=4)


def test_group_by_column_missing_errors(tmp_path):
    """group_by column not in obs → ValueError."""
    a = _toy_adata(["NTC"] * 5 + ["GA"] * 5)
    from cellstream.write import write_sharded

    with pytest.raises(ValueError, match="group_by"):
        write_sharded(a, tmp_path / "arc", format="v2", group_by="NONEXISTENT")


def test_reference_unknown_label_errors(tmp_path):
    """reference contains a label not present in obs[group_by] → ValueError."""
    a = _toy_adata(["NTC"] * 5 + ["GA"] * 5)
    from cellstream.write import write_sharded

    with pytest.raises(ValueError, match="[Rr]eference"):
        write_sharded(
            a,
            tmp_path / "arc",
            format="v2",
            group_by="gene",
            reference=["NOPE"],
        )


# ---------------------------------------------------------------------------
# Task B2: permute + group-aligned offsets + groups.json + v3 manifest
# ---------------------------------------------------------------------------


def test_group_by_layout(tmp_path):
    """Full B2 layout test: manifest v3, groups.json, _orig_row, lexicographic order."""
    from cellstream.write import write_sharded

    labels = np.array(["NTC"] * 40 + ["GENEB"] * 30 + ["GENEA"] * 30)
    a = _toy_adata(labels)
    out = tmp_path / "arc"
    write_sharded(
        a,
        out,
        format="v2",
        group_by="gene",
        reference=["NTC"],
        target_shard_bytes=10_000,
    )

    # --- manifest v3 fields ---
    m = archive_manifest(out)
    assert m["schema_version"] == 3
    assert m["group_by"] == "gene"
    assert m["reference_shard"] == 0

    # --- groups.json contents ---
    groups = archive_groups(out)
    ref_labels = [g["label"] for g in groups if g["role"] == "reference"]
    assert ref_labels == ["NTC"]
    nonref_labels = [g["label"] for g in groups if g["role"] == "group"]
    assert nonref_labels == ["GENEA", "GENEB"]  # lexicographic

    # --- _orig_row is a permutation of range(n) ---
    obs = _read_obs_parquet(out)
    assert sorted(obs["_orig_row"].tolist()) == list(range(len(labels)))

    # --- no gene straddles a shard ---
    row_offsets = m["row_offsets"]
    for g in groups:
        s = g["shard"]
        assert row_offsets[s] <= g["row_start"]
        assert g["row_stop"] <= row_offsets[s + 1]

    # --- groups.json was written BEFORE manifest (manifest implies complete groups.json) ---
    # Member PRESENCE, not a filesystem path: a packed archive has no `out / "x"` path,
    # so a bare .exists() here is False and the assertion stops testing anything (#154).
    assert archive_has_member(out, "groups.json")
    assert archive_has_member(out, "manifest.json")


def test_group_by_layout_bool_reference(tmp_path):
    """reference=bool_array (not list) is also accepted."""
    from cellstream.write import write_sharded

    labels = np.array(["NTC"] * 20 + ["GA"] * 20 + ["GB"] * 20)
    a = _toy_adata(labels)
    ref_mask = np.array([True] * 20 + [False] * 40)

    out = tmp_path / "arc"
    write_sharded(
        a,
        out,
        format="v2",
        group_by="gene",
        reference=ref_mask,
        target_shard_bytes=10_000,
    )

    m = archive_manifest(out)
    assert m["schema_version"] == 3
    assert m["reference_shard"] == 0

    groups = archive_groups(out)
    ref_labels = [g["label"] for g in groups if g["role"] == "reference"]
    assert ref_labels == ["NTC"]


def test_group_by_no_reference(tmp_path):
    """group_by set but reference=None → reference_shard is None, no reference role."""
    from cellstream.write import write_sharded

    labels = np.array(["GA"] * 30 + ["GB"] * 30)
    a = _toy_adata(labels)

    out = tmp_path / "arc"
    write_sharded(a, out, format="v2", group_by="gene", target_shard_bytes=10_000)

    m = archive_manifest(out)
    assert m["schema_version"] == 3
    assert m["reference_shard"] is None

    groups = archive_groups(out)
    assert all(g["role"] == "group" for g in groups)
    assert [g["label"] for g in groups] == ["GA", "GB"]


def test_n_shards_none_byte_driven_no_group_by(tmp_path):
    """n_shards=None + group_by=None → byte-driven even split (>= 1 shard)."""
    from cellstream.write import write_sharded

    # 200 cells, ~20% density over 50 vars → ~2000 nnz → ~8000 bytes
    labels = ["X"] * 200
    a = _toy_adata(labels)
    total_nnz = int(a.X.indptr[-1])

    out = tmp_path / "arc"
    # target_shard_bytes=1000 → ceil(total_nnz * 4 / 1000) >= 1 shards
    write_sharded(a, out, format="v2", target_shard_bytes=1_000)

    m = archive_manifest(out)
    expected_n = max(1, math.ceil(total_nnz * 4 / 1_000))
    assert m["n_shards"] == expected_n
    assert m["schema_version"] == 2  # no group_by → still v2


def test_n_shards_explicit_int_no_group_by(tmp_path):
    """Explicit n_shards (no group_by) still works as before."""
    from cellstream.write import write_sharded

    labels = ["X"] * 100
    a = _toy_adata(labels)

    out = tmp_path / "arc"
    write_sharded(a, out, format="v2", n_shards=4)

    m = archive_manifest(out)
    assert m["n_shards"] == 4
    assert m["schema_version"] == 2


def test_group_by_round_trip_values(tmp_path):
    """Data written with group_by reads back correctly (values preserved, obs realigns).

    Uses deliberately unsorted input so the planner permutation is non-identity.
    This guards against a silent pass when the permutation happens to be the
    identity (e.g. input already in sorted order).
    """
    from cellstream import ShardedArchive, write_sharded

    # Deliberately unsorted: GB first, then NTC (reference), then GA.
    n = 60
    labels = np.array(["GB"] * 20 + ["NTC"] * 20 + ["GA"] * 20)
    a = _toy_adata(labels)
    # Add a distinguishing obs column so we can verify obs realignment, not just X.
    a.obs["orig_idx"] = np.arange(n)
    orig_X = a.X.toarray().copy()
    orig_idx_col = a.obs["orig_idx"].to_numpy().copy()

    out = tmp_path / "arc"
    write_sharded(
        a,
        out,
        format="v2",
        group_by="gene",
        reference=["NTC"],
        target_shard_bytes=500,
    )

    arc = ShardedArchive(out)
    back = arc.to_anndata(cast_back=True, n_workers=1)

    orig_row = back.obs["_orig_row"].to_numpy()

    # The permutation must be genuinely non-identity (input was unsorted).
    assert not np.array_equal(np.argsort(orig_row), np.arange(n)), (
        "_orig_row permutation is identity — input was not reordered; test is vacuous"
    )

    inv = np.argsort(orig_row)

    # X round-trip.
    reconstructed_X = back.X.toarray()[inv]
    np.testing.assert_array_equal(reconstructed_X, orig_X)

    # obs round-trip: orig_idx column must also realign correctly.
    reconstructed_idx = back.obs["orig_idx"].to_numpy()[inv]
    np.testing.assert_array_equal(reconstructed_idx, orig_idx_col)


def test_reference_bool_mask_wrong_length_errors(tmp_path):
    """reference bool mask with wrong length → clear ValueError naming 'reference'."""
    from cellstream.write import write_sharded

    labels = np.array(["NTC"] * 10 + ["GA"] * 10)
    a = _toy_adata(labels)
    # Mask is too short (5 instead of 20).
    bad_mask = np.array([True] * 5, dtype=bool)

    with pytest.raises(ValueError, match="reference"):
        write_sharded(
            a,
            tmp_path / "arc",
            format="v2",
            group_by="gene",
            reference=bad_mask,
        )


# ---------------------------------------------------------------------------
# Finding 2: target_shard_bytes=0 → ValueError; empty AnnData → no crash
# ---------------------------------------------------------------------------


def test_write_v2_target_shard_bytes_zero_raises(tmp_path):
    """target_shard_bytes=0 → ValueError (divide-by-zero guard)."""
    a = _toy_adata(["X"] * 10)
    from cellstream.write import write_sharded

    with pytest.raises(ValueError, match="target_shard_bytes"):
        write_sharded(a, tmp_path / "arc", format="v2", target_shard_bytes=0)


def test_write_v2_target_shard_bytes_negative_raises(tmp_path):
    """target_shard_bytes < 0 → ValueError."""
    a = _toy_adata(["X"] * 10)
    from cellstream.write import write_sharded

    with pytest.raises(ValueError, match="target_shard_bytes"):
        write_sharded(a, tmp_path / "arc", format="v2", target_shard_bytes=-1)


def test_write_v2_empty_anndata_no_crash(tmp_path):
    """write_sharded with an empty AnnData (n_obs=0) produces a valid archive."""
    import scipy.sparse as _sp

    from cellstream import ShardedArchive, write_sharded

    # Build a minimal empty AnnData.
    empty_X = _sp.csr_matrix((0, 20), dtype="float32")
    empty_X.indptr = empty_X.indptr.astype(np.int64)
    a = ad.AnnData(X=empty_X)
    out = tmp_path / "empty_arc"
    # Must not raise.
    write_sharded(a, out, format="v2", target_shard_bytes=1_000)

    m = archive_manifest(out)
    assert m["n_obs_total"] == 0
    assert m["n_shards"] == 1
    assert m["row_offsets"] == [0, 0]

    # Reading back also must not crash.
    arc = ShardedArchive(out)
    back = arc.to_anndata(cast_back=True, n_workers=1)
    assert back.n_obs == 0


# ---------------------------------------------------------------------------
# Codex round-4 robustness fixes
# ---------------------------------------------------------------------------


def test_write_v2_archive_reference_without_group_by_raises(tmp_path):
    """write_v2_archive(reference=mask, group_by=None) -> ValueError (mirrored guard)."""
    from cellstream.v2.writer import write_v2_archive

    a = _toy_adata(["NTC"] * 10 + ["GA"] * 10)
    mask = np.array([True] * 10 + [False] * 10, dtype=bool)

    with pytest.raises(ValueError, match="reference requires group_by"):
        write_v2_archive(a, tmp_path / "arc", reference=mask, group_by=None)


def test_write_v2_archive_2d_bool_reference_raises(tmp_path):
    """write_v2_archive with a 2-D bool mask -> clear ValueError naming reference."""
    from cellstream.v2.writer import write_v2_archive

    labels = np.array(["NTC"] * 10 + ["GA"] * 10)
    a = _toy_adata(labels)
    # Shape (n_obs, 1) - wrong dimensionality.
    bad_mask = np.array([True] * 10 + [False] * 10, dtype=bool).reshape(-1, 1)

    with pytest.raises(ValueError, match="reference"):
        write_v2_archive(a, tmp_path / "arc", group_by="gene", reference=bad_mask)


def test_write_v2_empty_anndata_with_group_by_no_crash(tmp_path):
    """write_sharded with an empty AnnData + group_by → 1 shard, empty groups.json, reads back.

    Finding 3 (Codex round-2): grouped empty path used to produce row_offsets=[0]
    (n_shards=0), which was inconsistent with the non-grouped empty path.  The fix
    makes plan_group_shards return row_offsets=[0,0] so both paths produce exactly
    one empty shard.
    """

    import pandas as pd
    import scipy.sparse as _sp

    from cellstream import ShardedArchive, write_sharded

    empty_X = _sp.csr_matrix((0, 20), dtype="float32")
    empty_X.indptr = empty_X.indptr.astype(np.int64)
    # obs must have the group_by column (even with 0 rows).
    a = ad.AnnData(X=empty_X, obs=pd.DataFrame({"gene": pd.Series([], dtype=str)}))
    out = tmp_path / "empty_grouped"

    # Must not raise.
    write_sharded(a, out, format="v2", group_by="gene", target_shard_bytes=1_000)

    m = archive_manifest(out)
    assert m["n_obs_total"] == 0, f"expected 0 obs, got {m['n_obs_total']}"
    assert m["n_shards"] == 1, f"expected 1 shard, got {m['n_shards']}"
    assert m["row_offsets"] == [0, 0], f"expected [0, 0], got {m['row_offsets']}"

    # groups.json must exist and be an empty list.
    assert archive_has_member(out, "groups.json"), "groups.json missing"
    assert archive_groups(out) == [], "groups.json must be []"

    # Reading back must not crash and must return 0 observations.
    arc = ShardedArchive(out)
    back = arc.to_anndata(cast_back=True, n_workers=1)
    assert back.n_obs == 0


# ---------------------------------------------------------------------------
# #179: write-API round-trip quirks
# ---------------------------------------------------------------------------


def test_grouped_write_replaces_orig_row_with_warning(tmp_path):
    """An incoming reserved _orig_row is replaced (not rejected), with a warning."""
    from cellstream.write import write_sharded

    labels = ["NTC"] * 6 + ["GA"] * 6
    a = _toy_adata(labels)
    n = a.n_obs
    # A clearly-invalid sentinel so we can prove it was replaced by a real permutation.
    a.obs["_orig_row"] = np.full(n, -999, dtype="int64")

    out = tmp_path / "regrouped"
    with pytest.warns(UserWarning, match="_orig_row"):
        write_sharded(
            a,
            out,
            format="v2",
            group_by="gene",
            target_shard_bytes=10_000,
        )

    obs = _read_obs_parquet(out)
    # Replaced with the NEW permutation (a permutation of range(n)), sentinel gone.
    assert sorted(obs["_orig_row"].tolist()) == list(range(n))
    assert (obs["_orig_row"].to_numpy() != -999).all()


def test_roundtrip_regroup_succeeds(tmp_path):
    """archive -> to_anndata() -> write_sharded(group_by=) round-trips without error."""
    from cellstream import ShardedArchive
    from cellstream.write import write_sharded

    labels = ["NTC"] * 8 + ["GA"] * 4 + ["GB"] * 4
    a = _toy_adata(labels)
    n = a.n_obs

    out1 = tmp_path / "arc1"
    write_sharded(
        a,
        out1,
        format="v2",
        group_by="gene",
        reference=["NTC"],
        target_shard_bytes=10_000,
    )

    back = ShardedArchive(out1).to_anndata()
    assert "_orig_row" in back.obs.columns  # the read-side permutation metadata

    out2 = tmp_path / "arc2"
    with pytest.warns(UserWarning, match="_orig_row"):
        write_sharded(
            back,
            out2,
            format="v2",
            group_by="gene",
            reference=["NTC"],
            target_shard_bytes=10_000,
        )

    back2 = ShardedArchive(out2).to_anndata()
    assert sorted(back2.obs["_orig_row"].tolist()) == list(range(n))


def test_grouped_write_logs_resolved_layout(tmp_path, caplog):
    """A grouped write logs its resolved (n_shards, target_shard_bytes) at INFO."""
    import logging

    from cellstream.write import write_sharded

    a = _toy_adata(["NTC"] * 6 + ["GA"] * 6)
    out = tmp_path / "arc"
    with caplog.at_level(logging.INFO):
        write_sharded(
            a,
            out,
            format="v2",
            group_by="gene",
            target_shard_bytes=10_000,
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert any("target_shard_bytes=10000" in m and "shard" in m for m in msgs)
