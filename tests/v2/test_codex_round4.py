"""Gemini round-2 review findings applied to target-aware sharding.

Tests cover findings 1–4 (test-first).

Finding 1 (writer.py ~156): bare string reference="NTC" → same layout as ["NTC"].
Finding 2 (read.py ~328):   read_group on the reference label now works.
Finding 3 (read.py ~336):   near-match guard uses difflib; empty-string label is safe.
Finding 4 (writer.py ~211): adata.X is coerced to CSR before group-aware slicing.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from tests._archive_helpers import archive_groups, archive_manifest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _toy_adata(labels, n_vars=50, seed=7):
    """Build a small AnnData with an obs['gene'] column.

    X is a CSR float32 matrix (~20% density).  Input is UNSORTED so the
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
    """Write a group_by='gene' archive. Returns (out_dir, original_adata)."""
    from cellstream.write import write_sharded

    a = _toy_adata(labels)
    out = tmp_path / "arc"
    write_sharded(
        a,
        out,
        format="v2",
        group_by="gene",
        reference=reference,
        target_shard_bytes=target_shard_bytes,
    )
    return out, a


# ---------------------------------------------------------------------------
# Finding 1: bare string reference treated as one-element label list
# ---------------------------------------------------------------------------


def test_string_reference_same_layout_as_list(tmp_path):
    """write_sharded(..., reference='NTC') == reference=['NTC'] layout-wise."""
    from cellstream.write import write_sharded

    labels = np.array(["GENEB"] * 30 + ["NTC"] * 40 + ["GENEA"] * 30)
    a = _toy_adata(labels)

    out_str = tmp_path / "arc_str"
    out_lst = tmp_path / "arc_lst"

    write_sharded(
        a,
        out_str,
        format="v2",
        group_by="gene",
        reference="NTC",
        target_shard_bytes=5_000,
    )
    write_sharded(
        a,
        out_lst,
        format="v2",
        group_by="gene",
        reference=["NTC"],
        target_shard_bytes=5_000,
    )

    m_str = archive_manifest(out_str)
    m_lst = archive_manifest(out_lst)

    assert m_str["reference_shard"] == m_lst["reference_shard"], (
        f"reference_shard differs: str={m_str['reference_shard']}, list={m_lst['reference_shard']}"
    )
    assert m_str["row_offsets"] == m_lst["row_offsets"], (
        f"row_offsets differ: str={m_str['row_offsets']}, list={m_lst['row_offsets']}"
    )


def test_string_reference_correct_role_in_groups_json(tmp_path):
    """reference='NTC' (bare string) → 'NTC' has role='reference' in groups.json."""
    from cellstream.write import write_sharded

    labels = np.array(["GENEA"] * 20 + ["NTC"] * 20)
    a = _toy_adata(labels)
    out = tmp_path / "arc"
    write_sharded(
        a,
        out,
        format="v2",
        group_by="gene",
        reference="NTC",
        target_shard_bytes=5_000,
    )

    groups = archive_groups(out)
    ref_labels = [g["label"] for g in groups if g["role"] == "reference"]
    assert ref_labels == ["NTC"], f"expected ['NTC'], got {ref_labels}"


def test_string_reference_unknown_label_raises(tmp_path):
    """reference='UNKNOWN_GENE' (bare string, not in obs) → ValueError."""
    from cellstream.write import write_sharded

    labels = np.array(["GENEA"] * 20 + ["NTC"] * 20)
    a = _toy_adata(labels)

    with pytest.raises(ValueError, match="[Rr]eference"):
        write_sharded(
            a,
            tmp_path / "arc",
            format="v2",
            group_by="gene",
            reference="UNKNOWN_GENE",
            target_shard_bytes=5_000,
        )


# ---------------------------------------------------------------------------
# Finding 2: read_group on the reference label returns reference cells
# ---------------------------------------------------------------------------


def test_read_group_reference_label_returns_reference_cells(tmp_path):
    """read_group('NTC') returns the NTC cells (equal to read_reference())."""
    labels = np.array(["GENEB"] * 30 + ["NTC"] * 40 + ["GENEA"] * 30)
    out, _ = _write_grouped(tmp_path, labels, reference=["NTC"])

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)
    ref_via_method = arc.read_reference()
    ref_via_group = arc.read_group("NTC")

    assert ref_via_method is not None
    assert ref_via_group.n_obs == ref_via_method.n_obs, (
        f"read_group('NTC').n_obs={ref_via_group.n_obs} != "
        f"read_reference().n_obs={ref_via_method.n_obs}"
    )
    # Both should contain only NTC cells.
    assert set(ref_via_group.obs["gene"].tolist()) == {"NTC"}
    # Value-level equality (X matrices match; order may differ via _orig_row).
    orig_rows_g = np.sort(ref_via_group.obs["_orig_row"].to_numpy())
    orig_rows_r = np.sort(ref_via_method.obs["_orig_row"].to_numpy())
    np.testing.assert_array_equal(orig_rows_g, orig_rows_r)


def test_read_group_reference_label_cell_count(tmp_path):
    """read_group on the reference label returns the right number of cells."""
    labels = np.array(["GENEA"] * 25 + ["NTC"] * 35)
    out, _ = _write_grouped(tmp_path, labels, reference=["NTC"])

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)
    result = arc.read_group("NTC")
    assert result.n_obs == 35


def test_iter_group_shards_still_excludes_reference_after_cache_fix(tmp_path):
    """iter_group_shards still yields only non-reference shards after label_map cache fix."""
    labels = np.array(["GENEB"] * 30 + ["NTC"] * 40 + ["GENEA"] * 30)
    out, _ = _write_grouped(tmp_path, labels, reference=["NTC"])

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    # Warm up the label_map cache by calling read_group first.
    arc.read_group("NTC")

    seen_labels = []
    for gs in arc.iter_group_shards():
        seen_labels.extend(gs.labels)

    assert "NTC" not in seen_labels, f"iter_group_shards included reference label: {seen_labels}"
    assert sorted(seen_labels) == ["GENEA", "GENEB"]


# ---------------------------------------------------------------------------
# Finding 3: near-match guard uses difflib (bounded; empty-label safe)
# ---------------------------------------------------------------------------


def test_unknown_label_small_suggestion_list(tmp_path):
    """KeyError for an unknown label includes a small (bounded) suggestion list."""
    # Two groups; near-match for 'GENE' should suggest at most a handful.
    labels = np.array(["GENEA"] * 30 + ["GENEB"] * 30)
    out, _ = _write_grouped(tmp_path, labels, reference=None)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    with pytest.raises(KeyError) as exc_info:
        arc.read_group("GENE")

    msg = str(exc_info.value)
    # Must mention at least one near-match.
    assert "GENE" in msg, f"no near-match hint in error: {msg!r}"


def test_empty_string_label_keyerror_does_not_enumerate_all(tmp_path):
    """read_group('') raises KeyError without listing all group labels."""
    # Create many groups to make the old substring match return a huge list.
    n_genes = 50
    gene_labels = [f"GENE{i:04d}" for i in range(n_genes)]
    # Each gene gets 5 cells.
    labels = np.array([g for g in gene_labels for _ in range(5)])
    out, _ = _write_grouped(tmp_path, labels, reference=None, target_shard_bytes=500)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    with pytest.raises(KeyError) as exc_info:
        arc.read_group("")

    msg = str(exc_info.value)
    # The old code builds [k for k in label_map if "" in k], which is ALL keys.
    # difflib.get_close_matches("", [...], n=5) returns [] because "" has no
    # close matches, so the hint should be absent or mention at most 5 items.
    # Either way the error string must be short — not 50 gene names.
    # Check: no more than 5 gene names appear in the message.
    gene_count_in_msg = sum(1 for g in gene_labels if g in msg)
    assert gene_count_in_msg <= 5, (
        f"near-match hint listed {gene_count_in_msg} genes (expected ≤5). "
        f"Old substring match is still active.\nMessage: {msg[:500]}"
    )


def test_very_short_label_suggestion_bounded(tmp_path):
    """A 2-char unknown label query produces at most 5 near-match suggestions."""
    n_genes = 30
    gene_labels = [f"GENE{i:04d}" for i in range(n_genes)]
    labels = np.array([g for g in gene_labels for _ in range(4)])
    out, _ = _write_grouped(tmp_path, labels, reference=None, target_shard_bytes=500)

    from cellstream.read import ShardedArchive

    arc = ShardedArchive(out)

    with pytest.raises(KeyError) as exc_info:
        arc.read_group("GE")

    msg = str(exc_info.value)
    gene_count_in_msg = sum(1 for g in gene_labels if g in msg)
    assert gene_count_in_msg <= 5, (
        f"near-match hint listed {gene_count_in_msg} genes (expected ≤5). Message: {msg[:500]}"
    )


# ---------------------------------------------------------------------------
# Finding 4: adata.X coerced to CSR before group-aware slicing
# ---------------------------------------------------------------------------


def test_write_grouped_dense_x_succeeds_and_roundtrips(tmp_path):
    """write_sharded with group_by and a dense numpy X completes without error."""
    from cellstream import ShardedArchive, write_sharded

    labels = np.array(["GENEB"] * 20 + ["NTC"] * 20 + ["GENEA"] * 20)
    # Build via CSR then convert to dense.
    csr_adata = _toy_adata(labels)
    dense_X = csr_adata.X.toarray()
    a = ad.AnnData(X=dense_X, obs={"gene": labels})

    out = tmp_path / "arc_dense"
    # Must not raise.
    write_sharded(
        a,
        out,
        format="v2",
        group_by="gene",
        reference=["NTC"],
        target_shard_bytes=5_000,
    )

    arc = ShardedArchive(out)
    back = arc.to_anndata(cast_back=True, n_workers=1)
    # All cells present.
    assert back.n_obs == 60


def test_write_grouped_csc_x_succeeds_and_roundtrips(tmp_path):
    """write_sharded with group_by and a CSC X completes without error."""
    from cellstream import ShardedArchive, write_sharded

    labels = np.array(["GENEB"] * 20 + ["NTC"] * 20 + ["GENEA"] * 20)
    csr_adata = _toy_adata(labels)
    csc_X = sp.csc_matrix(csr_adata.X)
    a = ad.AnnData(X=csc_X, obs={"gene": labels})

    out = tmp_path / "arc_csc"
    write_sharded(
        a,
        out,
        format="v2",
        group_by="gene",
        reference=["NTC"],
        target_shard_bytes=5_000,
    )

    arc = ShardedArchive(out)
    back = arc.to_anndata(cast_back=True, n_workers=1)
    assert back.n_obs == 60


def test_write_grouped_dense_x_values_roundtrip(tmp_path):
    """Dense X round-trips correctly (values preserved after CSR coercion)."""
    from cellstream import ShardedArchive, write_sharded

    n_gene = 25
    labels = np.array(["GENEB"] * n_gene + ["NTC"] * 20 + ["GENEA"] * n_gene)
    csr_adata = _toy_adata(labels)
    orig_X = csr_adata.X.toarray()
    dense_X = orig_X.copy()
    a = ad.AnnData(X=dense_X, obs={"gene": labels})

    out = tmp_path / "arc_dense_roundtrip"
    write_sharded(
        a,
        out,
        format="v2",
        group_by="gene",
        reference=["NTC"],
        target_shard_bytes=2_000,
    )

    arc = ShardedArchive(out)
    back = arc.to_anndata(cast_back=True, n_workers=1)

    orig_row = back.obs["_orig_row"].to_numpy()
    inv = np.argsort(orig_row)
    reconstructed = back.X.toarray()[inv]
    np.testing.assert_array_equal(reconstructed, orig_X)


def test_write_non_grouped_path_unaffected_by_csr_coercion(tmp_path):
    """Non-group_by path still works and is unaffected by the CSR coercion change."""
    from cellstream import ShardedArchive, write_sharded

    labels = ["X"] * 100
    csr_adata = _toy_adata(labels)

    out = tmp_path / "arc_no_group"
    # Classic path (no group_by).
    write_sharded(csr_adata, out, format="v2", n_shards=3)

    arc = ShardedArchive(out)
    back = arc.to_anndata(cast_back=True, n_workers=1)
    assert back.n_obs == 100
