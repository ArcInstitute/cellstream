"""Phase D: end-to-end integration test for target-aware sharding.

Task D1: Exercises the full write→read round-trip with a synthetic AnnData
that has >=4 genes + NTC reference, deliberately unsorted input order, and
a target_shard_bytes small enough that non-reference genes land in MULTIPLE
separate shards.  Value-level comparisons recover original rows via
obs["_orig_row"].

Assertions:
(a) read_reference() returns exactly the NTC cells (X values + obs column).
(b) read_group(g) equals the original a[a.obs.gene == g] (X values + obs).
(c) iter_group_shards() yields every non-NTC gene exactly once, and the
    union of yielded cells reconstructs all non-reference cells.
(d) Multi-shard local-slice check: verify gs.to_anndata()[gs.groups[label]]
    equals that gene's cells for a gene NOT in shard 1.
(e) Shards respect the size band except single oversized genes.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from tests._archive_helpers import archive_groups

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_adata_unsorted(seed: int = 7) -> tuple[ad.AnnData, dict[str, np.ndarray]]:
    """Build a synthetic AnnData with 4 genes + NTC, deliberately unsorted.

    Gene populations (counts chosen so byte-driven packing splits into >=3
    non-reference shards at target_shard_bytes=400):
      GENEB  - 18 cells, density ~0.7  (high nnz)
      NTC    - 25 cells, density ~0.3  (reference)
      GENEA  - 20 cells, density ~0.5
      GENEC  - 15 cells, density ~0.6
      GENED  - 12 cells, density ~0.8  (high nnz, last in lex order)

    Input row order: GENEB | NTC | GENEA | GENEC | GENED  (NOT sorted).
    """
    rng = np.random.default_rng(seed)
    n_vars = 30

    def _block(n: int, density: float, gene_name: str) -> tuple[sp.csr_matrix, np.ndarray]:
        nnz = max(1, int(n * n_vars * density))
        rows = rng.integers(0, n, size=nnz)
        cols = rng.integers(0, n_vars, size=nnz)
        vals = rng.integers(1, 200, size=nnz, dtype=np.uint16)
        coo = sp.coo_matrix((vals, (rows, cols)), shape=(n, n_vars))
        csr = coo.tocsr().astype("float32")
        csr.indices = csr.indices.astype(np.int32)
        csr.indptr = csr.indptr.astype(np.int64)
        labels = np.array([gene_name] * n)
        return csr, labels

    blocks = [
        _block(18, 0.70, "GENEB"),
        _block(25, 0.30, "NTC"),
        _block(20, 0.50, "GENEA"),
        _block(15, 0.60, "GENEC"),
        _block(12, 0.80, "GENED"),
    ]

    X = sp.vstack([b[0] for b in blocks]).tocsr()
    X.indices = X.indices.astype(np.int32)
    X.indptr = X.indptr.astype(np.int64)
    labels = np.concatenate([b[1] for b in blocks])

    n_total = X.shape[0]
    orig_idx = np.arange(n_total)

    adata = ad.AnnData(
        X=X,
        obs={"gene": labels, "orig_idx": orig_idx},
    )

    # Pre-compute per-gene index arrays on the ORIGINAL (unsorted) adata.
    per_gene = {g: np.where(labels == g)[0] for g in np.unique(labels)}
    return adata, per_gene


# target_shard_bytes chosen so non-reference genes split into >=3 shards.
# GENEA(20 cells, ~0.5 density, 30 vars) → ~150 nnz * 4 bytes ~ 600 bytes
# GENEB(18, 0.70) → ~126 nnz * 4 ~ 504 bytes
# GENEC(15, 0.60) → ~90 nnz * 4 ~ 360 bytes
# GENED(12, 0.80) → ~96 nnz * 4 ~ 384 bytes
# At target=400 bytes, GENEA and GENEB each exceed the target → own shards;
# GENEC and GENED are each at/below target → may share or be separate.
# Either way we get >=3 non-reference shards.
TARGET_SHARD_BYTES = 400


# ---------------------------------------------------------------------------
# Shared module-scoped fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grouped_arc(tmp_path_factory):
    """Write the grouped archive once; return (arc, orig_adata, per_gene_masks)."""
    from cellstream import ShardedArchive
    from cellstream.write import write_sharded

    orig, per_gene = _make_adata_unsorted()
    out = tmp_path_factory.mktemp("arc") / "arc"
    write_sharded(
        orig,
        out,
        format="v2",
        group_by="gene",
        reference=["NTC"],
        target_shard_bytes=TARGET_SHARD_BYTES,
    )
    arc = ShardedArchive(out)
    return arc, orig, per_gene


# ---------------------------------------------------------------------------
# (a) read_reference — X values + obs column
# ---------------------------------------------------------------------------


def test_read_reference_values(grouped_arc):
    """(a) read_reference() returns exactly the NTC cells with matching X + obs."""
    arc, orig, per_gene = grouped_arc

    ref = arc.read_reference(cast_back=True, n_workers=1)
    assert ref is not None

    ntc_orig_rows = per_gene["NTC"]
    assert ref.n_obs == len(ntc_orig_rows), (
        f"expected {len(ntc_orig_rows)} NTC cells, got {ref.n_obs}"
    )

    # All returned cells must be labeled NTC.
    assert set(ref.obs["gene"].tolist()) == {"NTC"}

    # Recover original rows and compare X values.
    orig_rows = ref.obs["_orig_row"].to_numpy()
    expected_X = orig.X.toarray()[orig_rows]
    np.testing.assert_array_equal(ref.X.toarray(), expected_X)

    # Also verify the orig_idx obs column round-trips correctly.
    expected_orig_idx = orig.obs["orig_idx"].to_numpy()[orig_rows]
    np.testing.assert_array_equal(ref.obs["orig_idx"].to_numpy(), expected_orig_idx)


# ---------------------------------------------------------------------------
# (b) read_group — X values + obs for every non-reference gene
# ---------------------------------------------------------------------------


def test_read_group_values_all_genes(grouped_arc):
    """(b) read_group(g) equals original a[a.obs.gene == g] for every gene."""
    arc, orig, per_gene = grouped_arc

    orig_X = orig.X.toarray()
    orig_labels = orig.obs["gene"].to_numpy()

    for gene in ["GENEA", "GENEB", "GENEC", "GENED"]:
        result = arc.read_group(gene, cast_back=True, n_workers=1)

        # Cell count matches.
        expected_count = len(per_gene[gene])
        assert result.n_obs == expected_count, (
            f"{gene}: expected {expected_count} cells, got {result.n_obs}"
        )

        # All returned cells labeled correctly.
        assert set(result.obs["gene"].tolist()) == {gene}, (
            f"{gene}: unexpected gene labels in result"
        )

        # X values match via _orig_row.
        orig_rows = result.obs["_orig_row"].to_numpy()
        assert all(orig_labels[r] == gene for r in orig_rows), (
            f"{gene}: _orig_row points at non-{gene} rows"
        )
        reconstructed = result.X.toarray()
        expected = orig_X[orig_rows]
        np.testing.assert_array_equal(reconstructed, expected, err_msg=f"{gene}: X mismatch")

        # obs column round-trip.
        expected_orig_idx = orig.obs["orig_idx"].to_numpy()[orig_rows]
        np.testing.assert_array_equal(
            result.obs["orig_idx"].to_numpy(),
            expected_orig_idx,
            err_msg=f"{gene}: orig_idx mismatch",
        )


# ---------------------------------------------------------------------------
# (c) iter_group_shards — coverage + union
# ---------------------------------------------------------------------------


def test_iter_group_shards_coverage_and_union(grouped_arc):
    """(c) iter_group_shards yields every non-NTC gene exactly once; union covers all."""
    arc, orig, per_gene = grouped_arc

    n_nonref = sum(len(v) for k, v in per_gene.items() if k != "NTC")

    seen_labels: list[str] = []
    total_cells_via_range = 0
    for gs in arc.iter_group_shards():
        seen_labels.extend(gs.labels)
        total_cells_via_range += gs.global_stop - gs.global_start

    # Every non-reference gene appears exactly once.
    assert sorted(seen_labels) == sorted(["GENEA", "GENEB", "GENEC", "GENED"]), (
        f"unexpected labels from iter_group_shards: {sorted(seen_labels)}"
    )
    assert len(seen_labels) == len(set(seen_labels)), "duplicate labels from iter_group_shards"

    # Sum of global ranges == total non-reference cells.
    assert total_cells_via_range == n_nonref, (
        f"global ranges sum to {total_cells_via_range}, expected {n_nonref}"
    )

    # Concatenating per-gene local slices recovers each gene's data.
    orig_X = orig.X.toarray()
    orig_labels = orig.obs["gene"].to_numpy()
    for gs in arc.iter_group_shards():
        adata = gs.to_anndata(cast_back=True, n_workers=1)
        for label, sl in gs.groups.items():
            subset = adata[sl]
            assert set(subset.obs["gene"].tolist()) == {label}, (
                f"local slice for {label!r} contains unexpected genes"
            )
            orig_rows = subset.obs["_orig_row"].to_numpy()
            assert all(orig_labels[r] == label for r in orig_rows)
            np.testing.assert_array_equal(
                subset.X.toarray(),
                orig_X[orig_rows],
                err_msg=f"local slice X mismatch for {label!r}",
            )


# ---------------------------------------------------------------------------
# (d) Multi-shard local-slice check
# ---------------------------------------------------------------------------


def test_multi_shard_local_slice(grouped_arc):
    """(d) Local slice for a gene in a non-first non-ref shard returns correct data."""
    arc, orig, per_gene = grouped_arc

    row_offsets = arc.row_offsets
    orig_X = orig.X.toarray()

    # Find a GroupShard that is NOT the immediate first non-reference shard.
    # Shard 0 is always the reference; shard 1 is the first non-ref shard.
    # We need a shard where row_offsets[shard_index] > row_offsets[1].
    multi_shard_gs = None
    for gs in arc.iter_group_shards():
        if row_offsets[gs.shard_index] > row_offsets[1]:
            multi_shard_gs = gs
            break

    assert multi_shard_gs is not None, (
        "No GroupShard found beyond shard 1 — TARGET_SHARD_BYTES may be too large; "
        "the test requires genes to land in >=3 non-reference shards. "
        f"row_offsets={row_offsets}"
    )

    # Read the shard and check every label's local slice.
    gs = multi_shard_gs
    adata = gs.to_anndata(cast_back=True, n_workers=1)
    for label, sl in gs.groups.items():
        subset = adata[sl]
        orig_rows = subset.obs["_orig_row"].to_numpy()
        np.testing.assert_array_equal(
            subset.X.toarray(),
            orig_X[orig_rows],
            err_msg=(f"multi-shard local slice mismatch for {label!r} in shard {gs.shard_index}"),
        )


# ---------------------------------------------------------------------------
# (e) Shard size sanity check
# ---------------------------------------------------------------------------


def test_shard_size_band(grouped_arc):
    """(e) Non-reference shards are within the byte band or are single-gene overflows."""
    arc, orig, per_gene = grouped_arc

    all_groups = archive_groups(arc.path)
    ref_shard = arc.manifest.get("reference_shard")
    row_offsets = arc.row_offsets

    # Read the full (permuted) archive to obtain the indptr.
    full = arc.to_anndata(cast_back=True, n_workers=1)
    indptr = full.X.indptr

    bytes_per_nnz = 4  # float32 data
    target = TARGET_SHARD_BYTES

    for s_idx in range(arc.n_shards):
        if s_idx == ref_shard:
            continue
        start = row_offsets[s_idx]
        stop = row_offsets[s_idx + 1]
        shard_nnz = int(indptr[stop] - indptr[start])
        shard_bytes = shard_nnz * bytes_per_nnz

        # Every non-reference shard must be either:
        # 1. Within the target (fits in band), OR
        # 2. A single-gene shard (allowed overflow for oversized groups).
        shard_records = [g for g in all_groups if g["shard"] == s_idx]
        is_single_gene = len(shard_records) == 1

        assert shard_bytes <= target or is_single_gene, (
            f"shard {s_idx}: {shard_bytes} bytes > target {target} "
            f"but contains {len(shard_records)} genes — "
            "expected only single-gene shards to exceed the target"
        )
