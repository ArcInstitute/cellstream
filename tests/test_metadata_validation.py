"""Integration tests for groups.json record validation (ultrareview L1).

A corrupt/tampered groups.json must fail loud at the load boundary
(ShardedArchive._load_groups) rather than producing a KeyError / TypeError or a
silently out-of-bounds row slice in read_group / iter_group_shards.

#154 part B3a, route E ("tamper-then-pack"): the tamper needs a LOOSE groups.json on disk, but
the assertion is through the PUBLIC grouped API, so the directory is staged and then packed.
That works because ``_pack_directory`` validates only ``manifest.json`` -- ``_enumerate_members``
runs ``load_manifest`` on it (``packed/writer.py:35``) and byte-copies every other member, with
``groups.json`` optional in ``_METADATA_ORDER``. So a structurally corrupt groups.json survives
packing intact and reaches the reader's load boundary exactly as it did loose. Deleting a member,
or tampering the manifest itself, does NOT have that property -- see the two route-E residuals in
tests/v2/test_target_aware_read.py.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellstream.manifest import groups_path
from cellstream.packed.reader import PackedArchive
from cellstream.read import ShardedArchive
from tests._archive_helpers import pack_directory, write_directory


def _grouped_archive(tmp_path):
    rng = np.random.default_rng(3)
    n_obs, n_vars = 120, 20
    nnz = 1500
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, 50, size=nnz, dtype=np.uint16)
    csr = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    x = sp.csr_matrix(csr.shape, dtype=np.uint16)
    x.data = csr.data.astype(np.uint16)
    x.indices = csr.indices.astype(np.uint16)
    x.indptr = csr.indptr.astype(np.int64)
    labels = rng.choice([f"g{i}" for i in range(6)], size=n_obs)
    obs = {"target_gene": labels}
    out = tmp_path / "grouped"
    write_directory(
        ad.AnnData(X=x, obs=obs),
        out,
        group_by="target_gene",
        target_shard_bytes=4096,
    )
    return out


def _corrupt_groups(out, mutate):
    gp = groups_path(out)
    recs = json.loads(gp.read_text())
    mutate(recs)
    gp.write_text(json.dumps(recs))


def _packed_with_corrupt_groups(out, mutate):
    """Tamper the loose groups.json, then pack -- returns the packed archive to read.

    Asserts the tamper SURVIVED the concat, which the tests below cannot afford to assume.
    ``_pack_directory`` byte-copies ``groups.json``, but if it ever stopped -- or if the member
    were dropped -- every test here would still raise ``ValueError`` (from a missing or malformed
    member) while testing nothing about record validation. Each test pins its own message for the
    same reason; codex flagged the bare ``pytest.raises(ValueError)`` these had after the packing
    step was added.

    The comparison is on RAW BYTES, not decoded records. Comparing decoded JSON would pass a
    re-serializing packer, which is precisely the change that would invalidate every "the tamper
    reached the reader unchanged" claim built on top of this helper. (codex's round-2 point: an
    earlier version compared ``json.loads`` on both sides and the plan then described it as
    byte-for-byte.)
    """
    _corrupt_groups(out, mutate)
    tampered_bytes = groups_path(out).read_bytes()
    packed = pack_directory(out, out.parent / f"{out.name}.shad")
    assert PackedArchive(packed).member_bytes("groups.json") == tampered_bytes, (
        "the pack did not carry the tampered groups.json through byte-for-byte"
    )
    return packed


def test_read_group_rejects_row_stop_past_n_obs(tmp_path):
    out = _grouped_archive(tmp_path)

    def mutate(recs):
        # Push a non-reference record's row_stop past n_obs.
        for r in recs:
            if r.get("role") != "reference":
                r["row_stop"] = 10**6
                break

    ar = ShardedArchive(_packed_with_corrupt_groups(out, mutate))
    label = next(r["label"] for r in json.loads(groups_path(out).read_text()))
    with pytest.raises(ValueError, match=r"row range .* violates"):
        ar.read_group(label)


def test_iter_group_shards_rejects_bad_shard_index(tmp_path):
    out = _grouped_archive(tmp_path)

    def mutate(recs):
        for r in recs:
            if r.get("role") != "reference":
                r["shard"] = 999  # out of [0, n_shards)
                break

    ar = ShardedArchive(_packed_with_corrupt_groups(out, mutate))
    with pytest.raises(ValueError, match=r"shard=999 out of range"):
        list(ar.iter_group_shards())


def test_read_group_rejects_missing_key(tmp_path):
    out = _grouped_archive(tmp_path)

    def mutate(recs):
        for r in recs:
            if r.get("role") != "reference":
                del r["row_start"]
                break

    ar = ShardedArchive(_packed_with_corrupt_groups(out, mutate))
    label = next(r["label"] for r in json.loads(groups_path(out).read_text()))
    with pytest.raises(ValueError, match=r"missing required key 'row_start'"):
        ar.read_group(label)
