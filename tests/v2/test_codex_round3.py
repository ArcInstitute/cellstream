"""Codex round-3 review findings applied to target-aware sharding.

Tests cover findings 1–4 (test-first; initially expected to FAIL
until the source fixes are applied).

Finding 1: Centralize reference resolution; accept booleany int masks.
Finding 2: reference without group_by → ValueError. (It also covered "validate
    group_by/reference with format != 'v2'"; that half went with the v1 writer in #154 —
    there is no non-v2 format left to validate against, so only the reference-without-group_by
    tests remain, under the Finding 2b banner below.)
Finding 3: _orig_row collision → ValueError (reserved column).
Finding 4: _require_grouped truthiness — detect grouped via `is not None`.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from tests._archive_helpers import archive_manifest, archive_obs, pack_directory

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _toy_adata(labels, n_vars=50, seed=0):
    """Build a small AnnData with an obs['gene'] column."""
    n = len(labels)
    rng = np.random.default_rng(seed)
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


# ---------------------------------------------------------------------------
# Finding 1: Booleany int mask (dtype != bool, values in {0,1}) treated as mask
# ---------------------------------------------------------------------------


def test_int_mask_same_layout_as_bool_mask(tmp_path):
    """An int 0/1 mask of length n_obs is treated as a row mask, same layout as bool mask."""
    from cellstream.write import write_sharded

    labels = np.array(["NTC"] * 20 + ["GA"] * 20 + ["GB"] * 20)
    a = _toy_adata(labels)

    bool_mask = np.array([True] * 20 + [False] * 40)
    int_mask = bool_mask.astype(np.int32)  # dtype int, values 0/1

    out_bool = tmp_path / "arc_bool"
    out_int = tmp_path / "arc_int"

    write_sharded(
        a,
        out_bool,
        format="v2",
        group_by="gene",
        reference=bool_mask,
        target_shard_bytes=5_000,
    )
    write_sharded(
        a,
        out_int,
        format="v2",
        group_by="gene",
        reference=int_mask,
        target_shard_bytes=5_000,
    )

    m_bool = archive_manifest(out_bool)
    m_int = archive_manifest(out_int)

    # Both runs must produce the same reference_shard and row layout.
    assert m_bool["reference_shard"] == m_int["reference_shard"], (
        f"reference_shard differs: bool={m_bool['reference_shard']}, int={m_int['reference_shard']}"
    )
    assert m_bool["row_offsets"] == m_int["row_offsets"], (
        f"row_offsets differ: bool={m_bool['row_offsets']}, int={m_int['row_offsets']}"
    )


def test_int_mask_same_layout_as_label_list(tmp_path):
    """An int 0/1 mask should produce the same layout as the equivalent label list."""
    from cellstream.write import write_sharded

    labels = np.array(["NTC"] * 20 + ["GA"] * 20 + ["GB"] * 20)
    a = _toy_adata(labels)

    int_mask = np.array([1] * 20 + [0] * 40, dtype=np.int64)
    label_list = ["NTC"]

    out_int = tmp_path / "arc_int"
    out_lbl = tmp_path / "arc_lbl"

    write_sharded(
        a,
        out_int,
        format="v2",
        group_by="gene",
        reference=int_mask,
        target_shard_bytes=5_000,
    )
    write_sharded(
        a,
        out_lbl,
        format="v2",
        group_by="gene",
        reference=label_list,
        target_shard_bytes=5_000,
    )

    m_int = archive_manifest(out_int)
    m_lbl = archive_manifest(out_lbl)

    assert m_int["reference_shard"] == m_lbl["reference_shard"], (
        f"reference_shard: int={m_int['reference_shard']}, label={m_lbl['reference_shard']}"
    )
    assert m_int["row_offsets"] == m_lbl["row_offsets"], (
        f"row_offsets: int={m_int['row_offsets']}, label={m_lbl['row_offsets']}"
    )


def test_int_mask_label_collision_not_misrouted(tmp_path):
    """A 0/1 int mask is NOT sent to the label-list path even if labels '0' and '1' exist."""
    from cellstream.write import write_sharded

    # Labels are literally "0" and "1".
    labels = np.array(["1"] * 20 + ["0"] * 20)
    a = _toy_adata(labels)

    # Int mask: first 20 rows = reference (labeled "1"), next 20 = non-reference ("0").
    int_mask = np.array([1] * 20 + [0] * 20, dtype=np.int32)

    out = tmp_path / "arc_collision"
    # Should not raise — mask path treats 1→True (reference) not as label "1".
    write_sharded(
        a,
        out,
        format="v2",
        group_by="gene",
        reference=int_mask,
        target_shard_bytes=5_000,
    )
    m = archive_manifest(out)
    # reference_shard should be 0 (reference cells first).
    assert m["reference_shard"] == 0


def test_label_list_still_works(tmp_path):
    """Passing a list of label values still routes correctly to the label path."""
    from cellstream.write import write_sharded

    labels = np.array(["NTC"] * 20 + ["GA"] * 20)
    a = _toy_adata(labels)

    out = tmp_path / "arc_lbl"
    write_sharded(
        a,
        out,
        format="v2",
        group_by="gene",
        reference=["NTC"],
        target_shard_bytes=5_000,
    )
    m = archive_manifest(out)
    assert m["reference_shard"] == 0


# ---------------------------------------------------------------------------
# Finding 2a: group_by / reference once raised "requires format='v2'", then stopped when
# v1 gained in-memory group-aware sharding (spec 2026-06-04), and the two tests here
# pinned that they wrote a grouped SCHEMA-1 archive. #154 removed the v1 writer, so both
# are gone; the v2 grouped-write coverage they mirrored is the rest of this file.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Finding 2b: reference without group_by → ValueError
# ---------------------------------------------------------------------------


def test_reference_without_group_by_raises(tmp_path):
    """reference=... without group_by → ValueError ('reference requires group_by')."""
    from cellstream.write import write_sharded

    a = _toy_adata(["NTC"] * 5 + ["GA"] * 5)
    ref_mask = np.array([True] * 5 + [False] * 5)
    with pytest.raises(ValueError, match="group_by"):
        write_sharded(a, tmp_path / "arc", format="v2", reference=ref_mask)


def test_reference_label_list_without_group_by_raises(tmp_path):
    """reference=['NTC'] without group_by=... → ValueError."""
    from cellstream.write import write_sharded

    a = _toy_adata(["NTC"] * 5 + ["GA"] * 5)
    with pytest.raises(ValueError, match="group_by"):
        write_sharded(a, tmp_path / "arc", format="v2", reference=["NTC"])


# ---------------------------------------------------------------------------
# Finding 3: _orig_row collision → ValueError (reserved column)
# ---------------------------------------------------------------------------


def test_orig_row_column_collision_warns_and_replaces(tmp_path):
    """A pre-existing _orig_row → warn + replace (not raise) (#179).

    Changed in #179: the reserved column is overwritten with the new grouping
    permutation (with a UserWarning) instead of raising, so an
    archive → to_anndata() → re-shard round-trip works.
    """

    from cellstream.write import write_sharded

    labels = np.array(["NTC"] * 10 + ["GA"] * 10)
    a = _toy_adata(labels)
    n = len(labels)
    # Pre-populate the reserved column with a clearly-invalid sentinel.
    a.obs["_orig_row"] = np.full(n, -1, dtype="int64")

    out = tmp_path / "arc"
    with pytest.warns(UserWarning, match="_orig_row"):
        write_sharded(
            a,
            out,
            format="v2",
            group_by="gene",
            target_shard_bytes=5_000,
        )

    obs = archive_obs(out)
    # Sentinel gone — replaced with the new permutation (a permutation of range(n)).
    assert sorted(obs["_orig_row"].tolist()) == list(range(n))


# ---------------------------------------------------------------------------
# Finding 4: _require_grouped truthiness — group_by="" treated as grouped-absent
# (degenerate case; main fix is the is-not-None check in read.py)
# ---------------------------------------------------------------------------


def test_require_grouped_empty_string_still_raises(tmp_path):
    """An archive with manifest group_by='' should still raise ValueError on grouped methods.

    This tests the is-not-None (not truthiness) fix: an empty string is falsy
    but IS not None.  Under the buggy 'if not group_by' check an empty string
    would raise; under the correct 'if group_by is None' check it would ALSO
    raise (since '' is not None, grouped-read proceeds, but groups.json is
    absent → should raise a different ValueError).

    The simpler behavioral test: a manifest with group_by="" must not be
    silently misread as "no group_by" — accessing grouped methods raises
    ValueError (either from _require_grouped or from missing groups.json).
    """
    import json

    from cellstream.read import ShardedArchive

    # Hand-craft a minimal manifest that has group_by="" (degenerate).
    arc_dir = tmp_path / "degenerate"
    arc_dir.mkdir()
    manifest_data = {
        "schema_version": 3,
        "shard_filename_template": "shard_{:04d}.shx",
        "n_shards": 1,
        "n_obs_total": 0,
        "n_vars": 10,
        "row_offsets": [0, 0],
        "nnz_per_shard": [0],
        "total_nnz": 0,
        "x_data_dtype_on_disk": "uint16",
        "x_indices_dtype_on_disk": "uint16",
        "codec": "zstd-1-u16u16-bitshuffle",
        "chunk_size_elements": 1048576,
        "group_by": "",  # degenerate: present but empty
        "reference_shard": None,
        "writer_fingerprint": {"cellstream_version": "test"},
        "created_at": "2026-01-01T00:00:00Z",
    }
    (arc_dir / "manifest.json").write_text(json.dumps(manifest_data))
    # Create a stub shard file so the missing-shard existence check at
    # ShardedArchive.__init__ passes (item 5 hardening).  The shard content
    # is irrelevant because the test only calls grouped-read methods that
    # raise before any I/O.
    (arc_dir / "shard_0000.shx").touch()

    # #154 part B3a, route E: the hand-built LOOSE manifest is the fixture, but the assertion
    # is through the public grouped API, so pack it and read the packed copy. schema_version 3
    # is in SUPPORTED_SCHEMA_VERSIONS, so _pack_directory's load_manifest accepts it, and
    # groups.json is optional in _METADATA_ORDER -- so "grouped manifest, absent groups.json"
    # survives the concat, which is exactly the state under test.
    arc = ShardedArchive(pack_directory(arc_dir, tmp_path / "degenerate.shad"))
    # group_by="" is NOT None, so the manifest says "this is a grouped archive".
    # Grouped reads should NOT silently fall through to the ungrouped path.
    # The behavior is: _require_grouped passes (group_by is not None → grouped),
    # then groups.json is missing → ValueError with a clear message.
    #
    # match= added with the #154 part B3a migration, and it is not cosmetic: a bare
    # pytest.raises(ValueError) cannot tell "grouped, but groups.json is absent" apart from
    # "this archive was not written with a group_by key", and distinguishing those two IS
    # finding 4. Probed both ways on the packed fixture -- group_by="" gives the message below,
    # while group_by=None gives "was not written with a group_by key" -- so the assertion now
    # fails if the is-not-None check ever regresses to a truthiness check.
    with pytest.raises(ValueError, match="groups.json"):
        list(arc.iter_group_shards())
