"""Tests for cellstream.grouping — pure planner (Phase A)."""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Task A1: shard_size_bounds
# ---------------------------------------------------------------------------


def test_shard_size_bounds_defaults():
    from cellstream.grouping import shard_size_bounds

    b = shard_size_bounds(target_shard_bytes=2 * 2**30)
    assert b.target == 2 * 2**30
    assert b.min_bytes <= b.target <= b.max_bytes
    assert b.min_bytes >= 256 * 2**20  # >= a few codec chunks (1,048,576 * 4 B ~ 4 MB) + headroom
    assert b.max_bytes <= 8 * 2**30


def test_shard_size_bounds_small_target_no_raise():
    """A very small target must not raise; lo = min(512*2^20, target), hi = max(4*2^30, target)."""
    from cellstream.grouping import shard_size_bounds

    b = shard_size_bounds(target_shard_bytes=100)
    assert b.target == 100
    assert b.min_bytes <= b.target
    assert b.max_bytes >= b.target


def test_shard_size_bounds_explicit_overrides():
    """Explicit min_bytes / max_bytes are accepted when consistent."""
    from cellstream.grouping import shard_size_bounds

    b = shard_size_bounds(target_shard_bytes=1000, min_bytes=500, max_bytes=2000)
    assert b.min_bytes == 500
    assert b.max_bytes == 2000
    assert b.target == 1000


def test_shard_size_bounds_inconsistent_explicit_raises():
    """Explicit overrides that violate lo <= target <= hi must raise."""
    from cellstream.grouping import shard_size_bounds

    with pytest.raises(ValueError):
        # target below explicit min
        shard_size_bounds(target_shard_bytes=100, min_bytes=200, max_bytes=300)


# ---------------------------------------------------------------------------
# Task A2: plan_group_shards
# ---------------------------------------------------------------------------


def _toy(n_groups=6, per=1000, nnz_per_cell=10, with_ref=True):
    """group labels g0..g(n-1); optional reference 'NTC' as the first block."""
    labels, nnz = [], []
    if with_ref:
        labels += ["NTC"] * 500
        nnz += [nnz_per_cell] * 500
    for g in range(n_groups):
        labels += [f"GENE{g:03d}"] * per
        nnz += [nnz_per_cell] * per
    return np.array(labels), np.array(nnz, dtype=np.int64)


def test_plan_no_group_straddles_and_ref_isolated():
    from cellstream.grouping import plan_group_shards

    labels, nnz = _toy()
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=nnz,
        reference_mask=(labels == "NTC"),
        target_shard_bytes=200_000,
        bytes_per_nnz=4,
        min_bytes=1,
        max_bytes=10**9,
    )
    # permutation is a valid bijection
    assert sorted(plan.permutation.tolist()) == list(range(len(labels)))
    # reference is exactly one shard, shard 0, and holds only NTC
    assert plan.reference_shard == 0
    ref_recs = [g for g in plan.groups if g["role"] == "reference"]
    assert [g["label"] for g in ref_recs] == ["NTC"]
    assert ref_recs[0]["shard"] == 0
    # every group lives in exactly one shard; non-ref groups are lexicographic
    assert [g["label"] for g in plan.groups if g["role"] == "group"] == sorted(
        ll for ll in set(labels) if ll != "NTC"
    )
    # row_offsets cover [0, n] and every group's [start,stop) sits within one shard
    assert plan.row_offsets[0] == 0 and plan.row_offsets[-1] == len(labels)
    for g in plan.groups:
        s = g["shard"]
        assert plan.row_offsets[s] <= g["row_start"] < g["row_stop"] <= plan.row_offsets[s + 1]


def test_plan_giant_group_gets_own_shard():
    from cellstream.grouping import plan_group_shards

    labels = np.array(["NTC"] * 10 + ["BIG"] * 10_000 + ["small"] * 10)
    nnz = np.ones(len(labels), dtype=np.int64)
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=nnz,
        reference_mask=(labels == "NTC"),
        target_shard_bytes=100,
        bytes_per_nnz=4,
        min_bytes=1,
        max_bytes=10**9,
    )
    big = [g for g in plan.groups if g["label"] == "BIG"][0]
    # BIG exceeds target; it occupies a shard with no other group
    same_shard = [g for g in plan.groups if g["shard"] == big["shard"]]
    assert same_shard == [big]


def test_plan_no_reference():
    from cellstream.grouping import plan_group_shards

    labels = np.array(["GENEA"] * 100 + ["GENEB"] * 100)
    nnz = np.ones(len(labels), dtype=np.int64)
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=nnz,
        reference_mask=None,
        target_shard_bytes=10**9,
        bytes_per_nnz=4,
        min_bytes=1,
        max_bytes=10**9,
    )
    assert plan.reference_shard is None
    assert all(g["role"] == "group" for g in plan.groups)


def test_plan_reference_never_straddles_shards():
    """Reference block must stay in exactly one shard even when it exceeds target."""
    from cellstream.grouping import plan_group_shards

    # 5000 NTC cells with nnz=10 → 200,000 bytes; target=100 bytes → would normally split
    labels = np.array(["NTC"] * 5000 + ["GA"] * 100)
    nnz = np.full(len(labels), 10, dtype=np.int64)
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=nnz,
        reference_mask=(labels == "NTC"),
        target_shard_bytes=100,
        bytes_per_nnz=4,
        min_bytes=1,
        max_bytes=10**18,  # huge max so oversized doesn't raise
    )
    # All NTC must be in shard 0 only
    ref_recs = [g for g in plan.groups if g["role"] == "reference"]
    assert len(ref_recs) == 1
    assert ref_recs[0]["shard"] == plan.reference_shard == 0
    # No non-reference group may be in shard 0
    non_ref_in_shard0 = [g for g in plan.groups if g["role"] != "reference" and g["shard"] == 0]
    assert non_ref_in_shard0 == []


def test_plan_multiple_reference_labels_one_shard():
    """Multiple distinct reference labels still land in exactly ONE shard (shard 0)."""
    from cellstream.grouping import plan_group_shards

    labels = np.array(["NTC"] * 300 + ["CTRL"] * 300 + ["GA"] * 100 + ["GB"] * 100)
    ref_mask = np.isin(labels, ["NTC", "CTRL"])
    nnz = np.full(len(labels), 10, dtype=np.int64)
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=nnz,
        reference_mask=ref_mask,
        target_shard_bytes=100,  # tiny target would split if reference weren't special-cased
        bytes_per_nnz=4,
        min_bytes=1,
        max_bytes=10**18,
    )
    ref_recs = [g for g in plan.groups if g["role"] == "reference"]
    # both NTC and CTRL appear as reference records...
    assert {g["label"] for g in ref_recs} == {"NTC", "CTRL"}
    # ...but all reference cells occupy exactly one shard (shard 0)
    assert {g["shard"] for g in ref_recs} == {0}
    assert plan.reference_shard == 0
    # no non-reference group shares shard 0
    assert [g for g in plan.groups if g["role"] != "reference" and g["shard"] == 0] == []


def test_plan_row_offsets_length():
    """row_offsets has length n_shards + 1."""
    from cellstream.grouping import plan_group_shards

    labels, nnz = _toy(n_groups=3, per=100)
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=nnz,
        reference_mask=(labels == "NTC"),
        target_shard_bytes=500,
        bytes_per_nnz=4,
        min_bytes=1,
        max_bytes=10**9,
    )
    n_shards = max(g["shard"] for g in plan.groups) + 1
    assert len(plan.row_offsets) == n_shards + 1


# ---------------------------------------------------------------------------
# Task A3: manifest v3 + groups.json helpers
# ---------------------------------------------------------------------------


def test_manifest_v3_in_supported_versions():
    from cellstream import manifest

    assert 3 in manifest.SUPPORTED_SCHEMA_VERSIONS


def test_manifest_v3_and_groups_roundtrip(tmp_path):
    from cellstream import manifest

    assert 3 in manifest.SUPPORTED_SCHEMA_VERSIONS
    groups = [
        {"label": "NTC", "shard": 0, "row_start": 0, "row_stop": 5, "role": "reference"},
        {"label": "GENEA", "shard": 1, "row_start": 5, "row_stop": 9, "role": "group"},
    ]
    manifest.dump_groups(tmp_path / "groups.json", groups)
    assert manifest.load_groups(tmp_path / "groups.json") == groups
    # A real v3 manifest always carries the structural fields (load_manifest now
    # validates the row_offsets invariants); keep them consistent with the groups
    # above (2 shards: rows [0,5) and [5,9), so n_obs_total=9).
    manifest.dump_manifest(
        tmp_path / "manifest.json",
        {
            "schema_version": 3,
            "group_by": "gene",
            "reference_shard": 0,
            "n_shards": 2,
            "n_obs_total": 9,
            "row_offsets": [0, 5, 9],
        },
    )
    assert manifest.load_manifest(tmp_path / "manifest.json")["group_by"] == "gene"


def test_groups_path_helper(tmp_path):
    from cellstream.manifest import groups_path

    p = groups_path(tmp_path)
    assert p == tmp_path / "groups.json"


def test_plan_empty_input_no_exception():
    """plan_group_shards with n == 0 must return a ShardPlan with one empty shard."""
    from cellstream.grouping import plan_group_shards

    plan = plan_group_shards(
        group_labels=np.array([], dtype=str),
        per_row_nnz=np.array([], dtype=np.int64),
        reference_mask=None,
        target_shard_bytes=100,
        bytes_per_nnz=4,
        min_bytes=1,
        max_bytes=10**9,
    )
    assert len(plan.permutation) == 0
    assert plan.row_offsets == [0, 0]  # one empty shard (consistent with non-grouped empty path)
    assert plan.groups == []
    assert plan.reference_shard is None


def test_plan_zero_nnz_ref_isolates_reference():
    """Reference cells with per_row_nnz == 0 must still be isolated in shard 0.

    When the reference block has 0 bytes, the old ``cur_bytes > 0`` seal guard
    fails to seal before the first non-reference group, so that group leaks into
    shard 0 alongside the reference cells.
    """
    from cellstream.grouping import plan_group_shards

    labels = np.array(["NTC"] * 10 + ["GENE"] * 100)
    nnz = np.array([0] * 10 + [5] * 100, dtype=np.int64)
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=nnz,
        reference_mask=(labels == "NTC"),
        target_shard_bytes=10_000,
        bytes_per_nnz=4,
        min_bytes=1,
        max_bytes=10**9,
    )
    # Reference shard is shard 0
    assert plan.reference_shard == 0
    # No non-reference group may share shard 0
    non_ref_in_shard0 = [g for g in plan.groups if g["role"] != "reference" and g["shard"] == 0]
    assert non_ref_in_shard0 == [], f"Found non-reference groups in shard 0: {non_ref_in_shard0}"


def test_dump_groups_is_atomic(tmp_path):
    """dump_groups uses tmp+replace (atomic write) — verify the output is valid JSON."""
    import json

    from cellstream.manifest import dump_groups

    groups = [{"label": "X", "shard": 0, "row_start": 0, "row_stop": 10, "role": "group"}]
    p = tmp_path / "groups.json"
    dump_groups(p, groups)
    loaded = json.loads(p.read_text())
    assert loaded == groups


# ---------------------------------------------------------------------------
# Finding 1: bool reference_mask must align to whole group labels
# ---------------------------------------------------------------------------


def test_plan_bool_ref_mask_splitting_label_raises():
    """A bool mask that marks SOME cells of label 'A' as reference → ValueError."""
    from cellstream.grouping import plan_group_shards

    # Label "A" has 10 rows; first 5 marked reference, last 5 not → split.
    labels = np.array(["A"] * 10 + ["B"] * 10)
    nnz = np.ones(len(labels), dtype=np.int64)
    bad_mask = np.array([True] * 5 + [False] * 5 + [False] * 10, dtype=bool)

    with pytest.raises(ValueError, match="A"):
        plan_group_shards(
            group_labels=labels,
            per_row_nnz=nnz,
            reference_mask=bad_mask,
            target_shard_bytes=10_000,
            bytes_per_nnz=4,
            min_bytes=1,
            max_bytes=10**9,
        )


def test_plan_bool_ref_mask_aligned_whole_group_ok():
    """A bool mask that marks ALL cells of a label as reference → no error."""
    from cellstream.grouping import plan_group_shards

    labels = np.array(["NTC"] * 10 + ["GA"] * 10)
    nnz = np.ones(len(labels), dtype=np.int64)
    # All NTC cells are reference; all GA cells are not.
    aligned_mask = np.array([True] * 10 + [False] * 10, dtype=bool)

    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=nnz,
        reference_mask=aligned_mask,
        target_shard_bytes=10_000,
        bytes_per_nnz=4,
        min_bytes=1,
        max_bytes=10**9,
    )
    assert plan.reference_shard == 0
    ref_recs = [g for g in plan.groups if g["role"] == "reference"]
    assert [g["label"] for g in ref_recs] == ["NTC"]


# Finding 1 (Codex round-2): int 0/1 masks must also be validated
# ---------------------------------------------------------------------------


def test_plan_int_ref_mask_splitting_label_raises():
    """An int 0/1 mask (coerced to bool) that splits a label → ValueError.

    This is the core regression test: before the fix the validation was gated
    on the *original* dtype being exactly bool, so int masks slipped through.
    """
    from cellstream.grouping import plan_group_shards

    # Label "A" has 10 rows; first 5 are reference (1), last 5 are not (0).
    labels = np.array(["A"] * 10 + ["B"] * 10)
    nnz = np.ones(len(labels), dtype=np.int64)
    bad_int_mask = np.array([1] * 5 + [0] * 5 + [0] * 10, dtype=np.int64)

    with pytest.raises(ValueError, match="A"):
        plan_group_shards(
            group_labels=labels,
            per_row_nnz=nnz,
            reference_mask=bad_int_mask,
            target_shard_bytes=10_000,
            bytes_per_nnz=4,
            min_bytes=1,
            max_bytes=10**9,
        )


def test_plan_int_ref_mask_wrong_length_raises():
    """An int mask of the wrong length → clear ValueError."""
    from cellstream.grouping import plan_group_shards

    labels = np.array(["A"] * 10 + ["B"] * 10)
    nnz = np.ones(len(labels), dtype=np.int64)
    # Length is 5, but n_obs is 20.
    short_mask = np.array([1] * 5, dtype=np.int64)

    with pytest.raises(ValueError, match="length"):
        plan_group_shards(
            group_labels=labels,
            per_row_nnz=nnz,
            reference_mask=short_mask,
            target_shard_bytes=10_000,
            bytes_per_nnz=4,
            min_bytes=1,
            max_bytes=10**9,
        )


def test_plan_int_ref_mask_whole_label_aligned_ok():
    """An int 0/1 mask that is whole-label-aligned → works without error."""
    from cellstream.grouping import plan_group_shards

    # All "NTC" rows = 1 (reference), all "GA" rows = 0 (non-reference).
    labels = np.array(["NTC"] * 10 + ["GA"] * 10)
    nnz = np.ones(len(labels), dtype=np.int64)
    int_mask = np.array([1] * 10 + [0] * 10, dtype=np.int64)

    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=nnz,
        reference_mask=int_mask,
        target_shard_bytes=10_000,
        bytes_per_nnz=4,
        min_bytes=1,
        max_bytes=10**9,
    )
    assert plan.reference_shard == 0
    ref_recs = [g for g in plan.groups if g["role"] == "reference"]
    assert [g["label"] for g in ref_recs] == ["NTC"]


# ---------------------------------------------------------------------------
# Finding 4: oversized-shard warnings
# ---------------------------------------------------------------------------


def test_plan_oversized_reference_warns():
    """A reference block exceeding max_bytes emits a UserWarning about the reference shard."""
    from cellstream.grouping import plan_group_shards

    # 1000 NTC rows * 100 nnz * 4 bytes = 400_000 bytes > max_bytes=1000
    labels = np.array(["NTC"] * 1000 + ["GA"] * 10)
    nnz = np.full(len(labels), 100, dtype=np.int64)

    with pytest.warns(UserWarning, match="reference shard"):
        plan_group_shards(
            group_labels=labels,
            per_row_nnz=nnz,
            reference_mask=(labels == "NTC"),
            target_shard_bytes=100,
            bytes_per_nnz=4,
            min_bytes=1,
            max_bytes=1000,  # tiny max → reference shard far exceeds it
        )


def test_plan_oversized_group_warns():
    """A non-reference group exceeding max_bytes emits a UserWarning naming the group."""
    from cellstream.grouping import plan_group_shards

    # "BIG" group: 1000 rows * 100 nnz * 4 bytes = 400_000 bytes > max_bytes=1000
    labels = np.array(["NTC"] * 10 + ["BIG"] * 1000 + ["small"] * 10)
    nnz = np.full(len(labels), 100, dtype=np.int64)

    with pytest.warns(UserWarning, match="BIG"):
        plan_group_shards(
            group_labels=labels,
            per_row_nnz=nnz,
            reference_mask=(labels == "NTC"),
            target_shard_bytes=100,
            bytes_per_nnz=4,
            min_bytes=1,
            max_bytes=1000,  # tiny max → BIG shard far exceeds it
        )


# Finding 2 (Codex round-2): combined reference-shard oversized warning
# ---------------------------------------------------------------------------


def test_plan_combined_reference_shard_oversized_warns_once():
    """Two reference labels each under max_bytes but combined over max_bytes → exactly one warning.

    The warning must be about the TOTAL reference shard bytes, not just individual
    per-label bytes.  This is the regression the Codex review identified: when each
    label is individually below max_bytes, the old per-label check fires zero times
    and the combined oversized shard goes unnoticed.
    """
    import warnings

    from cellstream.grouping import plan_group_shards

    # Each ref label: 50 rows * 10 nnz * 4 bytes = 2_000 bytes (under max_bytes=3000).
    # Combined: 4_000 bytes > max_bytes=3000 → should emit exactly ONE warning.
    labels = np.array(["NTC"] * 50 + ["CTRL"] * 50 + ["GA"] * 10)
    nnz_ref = np.full(50, 10, dtype=np.int64)
    nnz_ga = np.full(10, 10, dtype=np.int64)
    nnz = np.concatenate([nnz_ref, nnz_ref, nnz_ga])

    ref_mask = np.isin(labels, ["NTC", "CTRL"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan_group_shards(
            group_labels=labels,
            per_row_nnz=nnz,
            reference_mask=ref_mask,
            target_shard_bytes=100,
            bytes_per_nnz=4,
            min_bytes=1,
            max_bytes=3_000,  # each label under limit but combined over it
        )

    ref_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(ref_warnings) == 1, (
        f"expected exactly 1 warning about the combined reference shard, got {len(ref_warnings)}: "
        f"{[str(w.message) for w in ref_warnings]}"
    )
    assert "reference" in str(ref_warnings[0].message).lower(), (
        f"warning text should mention 'reference': {ref_warnings[0].message}"
    )


def test_plan_single_reference_label_over_max_bytes_warns_once():
    """A single reference label over max_bytes still emits exactly one warning (no regression)."""
    import warnings

    from cellstream.grouping import plan_group_shards

    # 1000 NTC rows * 10 nnz * 4 bytes = 40_000 bytes > max_bytes=1000.
    labels = np.array(["NTC"] * 1000 + ["GA"] * 10)
    nnz = np.full(len(labels), 10, dtype=np.int64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan_group_shards(
            group_labels=labels,
            per_row_nnz=nnz,
            reference_mask=(labels == "NTC"),
            target_shard_bytes=100,
            bytes_per_nnz=4,
            min_bytes=1,
            max_bytes=1_000,
        )

    ref_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(ref_warnings) == 1, (
        f"expected exactly 1 reference-shard warning, got {len(ref_warnings)}"
    )


# Finding 3 (Codex round-2): empty plan_group_shards → one empty shard
# ---------------------------------------------------------------------------


def test_plan_empty_input_produces_one_empty_shard():
    """plan_group_shards with n == 0 must produce row_offsets=[0,0] (one empty shard).

    The old behavior returned row_offsets=[0] (zero shards), which was inconsistent
    with the non-grouped empty path (which writes one empty shard).
    """
    from cellstream.grouping import plan_group_shards

    plan = plan_group_shards(
        group_labels=np.array([], dtype=str),
        per_row_nnz=np.array([], dtype=np.int64),
        reference_mask=None,
        target_shard_bytes=100,
        bytes_per_nnz=4,
        min_bytes=1,
        max_bytes=10**9,
    )
    assert len(plan.permutation) == 0
    assert plan.row_offsets == [0, 0], f"expected [0, 0], got {plan.row_offsets}"
    assert plan.groups == []
    assert plan.reference_shard is None


# ---------------------------------------------------------------------------
# resolve_reference / validate_reference_labels (moved from v2/writer.py)
# ---------------------------------------------------------------------------


def _adata_with_labels(labels):
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    n = len(labels)
    X = sp.csr_matrix(np.ones((n, 3), dtype="float32"))
    return ad.AnnData(X=X, obs={"gene": list(labels)})


def test_resolve_reference_none_is_all_false():
    from cellstream import grouping

    a = _adata_with_labels(["A", "A", "B"])
    mask = grouping.resolve_reference(None, a, "gene")
    assert mask.dtype == bool
    assert mask.tolist() == [False, False, False]


def test_resolve_reference_label_list():
    from cellstream import grouping

    a = _adata_with_labels(["NTC", "NTC", "GA", "GB"])
    mask = grouping.resolve_reference(["NTC"], a, "gene")
    assert mask.tolist() == [True, True, False, False]


def test_resolve_reference_bool_mask_passthrough():
    import numpy as np

    from cellstream import grouping

    a = _adata_with_labels(["A", "B", "C"])
    src = np.array([True, False, True])
    mask = grouping.resolve_reference(src, a, "gene")
    assert mask.tolist() == [True, False, True]


def test_resolve_reference_unknown_label_raises():
    import pytest

    from cellstream import grouping

    a = _adata_with_labels(["A", "B"])
    with pytest.raises(ValueError, match="[Rr]eference"):
        grouping.resolve_reference(["NOPE"], a, "gene")


def test_validate_reference_labels_ok_and_bad():
    import numpy as np
    import pytest

    from cellstream import grouping

    a = _adata_with_labels(["A", "B"])
    grouping.validate_reference_labels(a, "gene", ["A"])  # no raise
    with pytest.raises(ValueError, match="reference"):
        grouping.validate_reference_labels(a, "gene", np.array([True], dtype=bool))  # wrong length


# ---------------------------------------------------------------------------
# #87: multi-level secondary sort key
# ---------------------------------------------------------------------------


def test_plan_group_shards_multi_level_secondary_key():
    """secondary_key accepts a list of arrays → lexicographic within-group sort
    (reference first, then label, then key0, then key1)."""
    from cellstream.grouping import plan_group_shards

    # 6 cells, 2 genes, no reference. Within each gene, sort by (guide, umi).
    labels = np.array(["G1", "G1", "G1", "G2", "G2", "G2"])
    guide = np.array(["b", "a", "a", "z", "y", "y"])
    umi = np.array([1, 9, 2, 5, 8, 3])
    nnz = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
    ref = np.zeros(6, bool)

    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=nnz,
        reference_mask=ref,
        target_shard_bytes=10**9,
        bytes_per_nnz=4,
        secondary_key=[guide, umi],
    )
    perm = plan.permutation
    # G1 rows {0,1,2} ordered by (guide, umi): a/2 (row2), a/9 (row1), b/1 (row0)
    assert perm[:3].tolist() == [2, 1, 0]
    # G2 rows {3,4,5} ordered by (guide, umi): y/3 (row5), y/8 (row4), z/5 (row3)
    assert perm[3:].tolist() == [5, 4, 3]


def test_plan_group_shards_single_secondary_key_unchanged():
    """A single array secondary_key still works (back-compat)."""
    from cellstream.grouping import plan_group_shards

    labels = np.array(["G", "G", "G"])
    guide = np.array(["c", "a", "b"])
    nnz = np.array([1, 1, 1], dtype=np.int64)
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=nnz,
        reference_mask=np.zeros(3, bool),
        target_shard_bytes=10**9,
        bytes_per_nnz=4,
        secondary_key=guide,
    )
    assert plan.permutation.tolist() == [1, 2, 0]  # a, b, c


def test_plan_group_shards_single_list_secondary_key_is_one_key():
    """A single Python list of length n_obs is ONE key (not n_obs 1-element keys)
    — the list/tuple ambiguity Gemini flagged on the plan (PR #97)."""
    from cellstream.grouping import plan_group_shards

    labels = ["G", "G", "G"]
    guide = ["c", "a", "b"]  # a plain list of length n_obs, NOT a list of arrays
    plan = plan_group_shards(
        group_labels=np.asarray(labels),
        per_row_nnz=np.array([1, 1, 1], dtype=np.int64),
        reference_mask=np.zeros(3, bool),
        target_shard_bytes=10**9,
        bytes_per_nnz=4,
        secondary_key=guide,
    )
    assert plan.permutation.tolist() == [1, 2, 0]  # a, b, c


def test_plan_group_shards_empty_secondary_key_is_no_sort():
    """An empty list/tuple secondary_key means 'no secondary sort' — must not
    crash on a length mismatch (Gemini high-priority, PR #97)."""
    from cellstream.grouping import plan_group_shards

    labels = np.array(["G", "G", "G"])
    nnz = np.array([1, 1, 1], dtype=np.int64)
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=nnz,
        reference_mask=np.zeros(3, bool),
        target_shard_bytes=10**9,
        bytes_per_nnz=4,
        secondary_key=[],
    )
    # group order preserved (stable), no crash
    assert plan.permutation.tolist() == [0, 1, 2]


def test_plan_group_shards_multi_key_pandas_arraylike():
    """Multi-key elements may be ANY length-n_obs array-like (pd.Index /
    Categorical), not just ndarray/Series — duck-typed detection (Gemini PR#97)."""
    import pandas as pd

    from cellstream.grouping import plan_group_shards

    labels = np.array(["G", "G", "G", "G"])
    k0 = pd.Index(["b", "a", "a", "b"])
    k1 = pd.Index(["y", "z", "x", "x"])
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=np.ones(4, dtype=np.int64),
        reference_mask=np.zeros(4, bool),
        target_shard_bytes=10**9,
        bytes_per_nnz=4,
        secondary_key=[k0, k1],
    )
    # lexsort (k0, k1): a/x(2), a/z(1), b/x(3), b/y(0)
    assert plan.permutation.tolist() == [2, 1, 3, 0]


# ---------------------------------------------------------------------------
# #87: sort_order — preserve categorical/numeric dtype through the sort
# ---------------------------------------------------------------------------


def test_plan_group_shards_categorical_primary_respects_category_order():
    """A categorical group_labels sorts by CATEGORY order (not alphabetical),
    while groups[*].label stay the original strings."""
    import pandas as pd

    from cellstream.grouping import plan_group_shards

    # categories B before A (non-alphabetical); rows interleaved
    labels = pd.Categorical(
        ["GENE_A", "GENE_B", "GENE_A", "GENE_B"], categories=["GENE_B", "GENE_A"]
    )
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=np.ones(4, dtype=np.int64),
        reference_mask=np.zeros(4, bool),
        target_shard_bytes=10**9,
        bytes_per_nnz=4,
    )
    # category order: GENE_B rows (1, 3) first, then GENE_A rows (0, 2)
    assert plan.permutation.tolist() == [1, 3, 0, 2]
    # group labels recorded as ORIGINAL strings, in category order
    assert [g["label"] for g in plan.groups] == ["GENE_B", "GENE_A"]


def test_plan_group_shards_numeric_primary_sorts_numerically():
    """A numeric group_labels sorts numerically (1 < 2 < 10), and labels are the
    numbers' string form."""
    from cellstream.grouping import plan_group_shards

    labels = np.array([10, 2, 1])
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=np.ones(3, dtype=np.int64),
        reference_mask=np.zeros(3, bool),
        target_shard_bytes=10**9,
        bytes_per_nnz=4,
    )
    assert plan.permutation.tolist() == [2, 1, 0]  # 1, 2, 10
    assert [g["label"] for g in plan.groups] == ["1", "2", "10"]


def test_plan_group_shards_categorical_secondary_respects_category_order():
    """A categorical secondary_key orders WITHIN a group by category order."""
    import pandas as pd

    from cellstream.grouping import plan_group_shards

    labels = np.array(["G", "G", "G"])
    guide = pd.Categorical(["x", "y", "z"], categories=["z", "y", "x"])  # z<y<x
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=np.ones(3, dtype=np.int64),
        reference_mask=np.zeros(3, bool),
        target_shard_bytes=10**9,
        bytes_per_nnz=4,
        secondary_key=guide,
    )
    # category order z (row2), y (row1), x (row0)
    assert plan.permutation.tolist() == [2, 1, 0]
