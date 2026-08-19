# tests/packed/test_multimatrix_footer.py
from __future__ import annotations

import pandas as pd
import pytest

from cellstream.packed.reader import PackedArchive, _index_shard_members, _is_safe_member_name
from cellstream.packed.writer import _enumerate_members
from cellstream.packed.writer import _pack_directory as pack


def test_safe_name_accepts_two_segment_family_paths():
    assert _is_safe_member_name("x/shard_0000.shx")
    assert _is_safe_member_name("var/raw.parquet")
    assert _is_safe_member_name("manifest.json")  # still accepts bare names


def test_safe_name_still_rejects_traversal():
    assert not _is_safe_member_name("../x/shard_0000.shx")
    assert not _is_safe_member_name("/abs/shard_0000.shx")
    assert not _is_safe_member_name("a/b/c.shx")  # 3 segments
    assert not _is_safe_member_name("a\\b.shx")
    assert not _is_safe_member_name("x/..")


def test_index_shard_members_v4_groups_by_matrix():
    members = [
        {"name": "x/shard_0000.shx", "role": "shard", "matrix": "x", "shard_idx": 0},
        {"name": "x/shard_0001.shx", "role": "shard", "matrix": "x", "shard_idx": 1},
        {"name": "raw/shard_0000.shx", "role": "shard", "matrix": "raw", "shard_idx": 0},
        {"name": "raw/shard_0001.shx", "role": "shard", "matrix": "raw", "shard_idx": 1},
        {"name": "manifest.json", "role": "manifest"},
    ]
    manifest = {"schema_version": 4, "n_shards": 2, "matrices": {"x": {}, "raw": {}}}
    idx = _index_shard_members(members, manifest)
    assert set(idx) == {"x", "raw"}
    assert [m["shard_idx"] for m in idx["x"]] == [0, 1]


def test_index_shard_members_v4_rejects_gap():
    members = [
        {"name": "x/shard_0000.shx", "role": "shard", "matrix": "x", "shard_idx": 0},
        {"name": "x/shard_0002.shx", "role": "shard", "matrix": "x", "shard_idx": 2},
    ]
    manifest = {"schema_version": 4, "n_shards": 2, "matrices": {"x": {}}}
    from cellstream.errors import PackedArchiveError

    with pytest.raises(PackedArchiveError, match="not exactly"):
        _index_shard_members(members, manifest)


def _hand_build_v4_dir(tmp_path):
    """A minimal on-disk v4 directory: dummy shard bytes + real var sidecars +
    a valid header/obs + a v4 manifest. PackedArchive does not parse .shx at
    open, so dummy shard payloads are fine (offsets/alignment come from pack)."""
    import anndata as ad
    import scipy.sparse as sp

    from cellstream import header as _h
    from cellstream.manifest import build_v4_manifest, dump_manifest

    d = tmp_path / "v4dir"
    (d / "x").mkdir(parents=True)
    (d / "raw").mkdir()
    (d / "var").mkdir()
    for i in range(2):
        (d / "x" / f"shard_{i:04d}.shx").write_bytes(b"SHX2xshard" + bytes(i))
        (d / "raw" / f"shard_{i:04d}.shx").write_bytes(b"SHX2rawsh" + bytes(i))
    pd.DataFrame(index=["g0", "g1", "g2"]).to_parquet(d / "var" / "x.parquet")
    pd.DataFrame(index=["g0", "g1", "g2", "g3"]).to_parquet(d / "var" / "raw.parquet")
    obs = pd.DataFrame(index=[f"c{i}" for i in range(4)])
    hh = ad.AnnData(
        X=sp.csr_matrix((4, 3), dtype="float32"),
        obs=obs,
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    _h.write_header(hh, d / "header.h5ad")
    _h.write_obs_parquet(obs, d / "obs.parquet")

    def _mx(n_vars, dir_, role, owner=None, has_var=True):
        e = {
            "role": role,
            "owner": owner,
            "n_vars": n_vars,
            "nnz_per_shard": [1, 1],
            "total_nnz": 2,
            "x_data_dtype_on_disk": "uint32",
            "x_indices_dtype_on_disk": "uint32",
            "x_data_dtype_min": "uint16",
            "x_indices_dtype_min": "uint16",
            "codec": "v2-byte-filter",
            "chunk_size_elements": 1048576,
            "shard_filename_template": f"{dir_}/shard_{{:04d}}.shx",
        }
        if has_var:
            e["var_filename"] = f"var/{dir_}.parquet"
        return e

    m = build_v4_manifest(
        n_obs_total=4,
        row_offsets=[0, 2, 4],
        matrices={"x": _mx(3, "x", "x"), "raw": _mx(4, "raw", "raw")},
        writer_fingerprint={"cellstream_version": "test"},
        created_at="2026-07-04T00:00:00Z",
    )
    dump_manifest(d / "manifest.json", m)
    return d


def test_enumerate_members_dedups_shared_var_filename(tmp_path):
    """If two matrices point at the same var_filename, _enumerate_members must
    emit that var member only ONCE (a duplicate footer member name would corrupt
    the packed archive). Today x/raw own distinct sidecars; this guards the
    invariant against a future shared-var extension."""
    import json

    d = _hand_build_v4_dir(tmp_path)
    man = json.loads((d / "manifest.json").read_text())
    # Point raw's var at x's sidecar so both matrices share var/x.parquet.
    man["matrices"]["raw"]["var_filename"] = "var/x.parquet"
    (d / "manifest.json").write_text(json.dumps(man))

    members = _enumerate_members(d)
    var_names = [rel for (rel, role, *_rest) in members if role == "var"]
    assert var_names.count("var/x.parquet") == 1  # deduped, not appended twice
    assert len(var_names) == len(set(var_names))  # no duplicate var members at all


def test_packed_v4_shard_location_and_var(tmp_path):
    d = _hand_build_v4_dir(tmp_path)
    f = tmp_path / "a.shad"
    pack(d, f)
    pa = PackedArchive(f)
    fpath, base, length = pa.shard_location(1, matrix="raw")
    assert base % 4096 == 0 and length > 0
    with open(fpath, "rb") as fh:
        fh.seek(base)
        assert fh.read(length).startswith(b"SHX2rawsh")
    raw_var = pd.read_parquet(__import__("io").BytesIO(pa.var_bytes("raw")))
    assert list(raw_var.index) == ["g0", "g1", "g2", "g3"]
    from cellstream.errors import PackedArchiveError

    with pytest.raises(PackedArchiveError):
        pa.shard_location(0)  # v4 requires matrix=


def test_unpreservable_members_is_name_based_not_role_based():
    """#254/#259: the tail rewrite re-emits fixed NAMES, so a role allowlist is not enough.

    Copilot's point on #259: a member carrying an ALLOWED role but an unexpected name
    (role="manifest", name="manifest2.json") passes a role check and is still silently dropped
    by the rewrite. Not reachable from today's _enumerate_members, whose role map is 1:1 with
    the emitted names -- but the guard exists to fail closed on members it was not written for.
    """
    from cellstream.packed.writer import _unpreservable_members

    shards_and_metadata = [
        {"name": "x/shard_0000.shx", "role": "shard", "matrix": "x", "shard_idx": 0},
        {"name": "header.h5ad", "role": "header"},
        {"name": "obs.parquet", "role": "obs"},
        {"name": "manifest.json", "role": "manifest"},
    ]
    # Everything the rewrite re-emits -> nothing at risk (a v1/v2 archive must still update).
    assert _unpreservable_members(shards_and_metadata) == []
    # groups.json is optional and also re-emitted.
    assert (
        _unpreservable_members(shards_and_metadata + [{"name": "groups.json", "role": "groups"}])
        == []
    )

    # v4 sidecars -> refused (the original #254 case).
    assert _unpreservable_members(
        shards_and_metadata + [{"name": "var/x.parquet", "role": "var"}]
    ) == ["var/x.parquet"]

    # ALLOWED role, unexpected name -> still refused. A role-only check would MISS this.
    assert _unpreservable_members(
        shards_and_metadata + [{"name": "manifest2.json", "role": "manifest"}]
    ) == ["manifest2.json"]

    # Unknown role entirely -> refused (fails closed on future member kinds).
    assert _unpreservable_members(
        shards_and_metadata + [{"name": "future.bin", "role": "something-new"}]
    ) == ["future.bin"]
