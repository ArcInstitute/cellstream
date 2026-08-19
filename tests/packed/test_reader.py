import anndata as ad
import pandas as pd
import pytest

from cellstream.errors import PackedArchiveError
from cellstream.manifest import load_manifest
from cellstream.packed.reader import PackedArchive, is_packed_file
from cellstream.packed.writer import _pack_directory as pack


def test_is_packed_file(grouped_dir, tmp_path):
    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    assert is_packed_file(dst)
    assert not is_packed_file(grouped_dir)  # a directory
    assert not is_packed_file(grouped_dir / "manifest.json")  # JSON, not SHPK


def test_manifest_matches_directory(grouped_dir, tmp_path):
    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    pa = PackedArchive(dst)
    assert pa.manifest == load_manifest(grouped_dir / "manifest.json")
    assert pa.n_shards == pa.manifest["n_shards"]


def test_member_bytes_match_files(grouped_dir, tmp_path):
    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    pa = PackedArchive(dst)
    assert pa.member_bytes("manifest.json") == (grouped_dir / "manifest.json").read_bytes()
    assert pa.member_bytes("shard_0000.shx") == (grouped_dir / "shard_0000.shx").read_bytes()


def test_shard_location_offsets(grouped_dir, tmp_path):
    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    pa = PackedArchive(dst)
    for i in range(pa.n_shards):
        f, base, length = pa.shard_location(i)
        assert f == str(dst)
        assert length == (grouped_dir / f"shard_{i:04d}.shx").stat().st_size


def test_bad_crc_raises(grouped_dir, tmp_path):
    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    # Corrupt one footer byte (footer sits just before the 64-byte trailer).
    data = bytearray(dst.read_bytes())
    data[-80] ^= 0xFF
    dst.write_bytes(bytes(data))
    with pytest.raises(PackedArchiveError):
        PackedArchive(dst)


def test_header_adata_matches_directory(grouped_dir, tmp_path):
    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    pa = PackedArchive(dst)
    got = pa.header_adata()
    want = ad.read_h5ad(grouped_dir / "header.h5ad")
    pd.testing.assert_frame_equal(got.var, want.var)
    assert got.n_vars == want.n_vars


def test_obs_matches_directory(grouped_dir, tmp_path):
    from cellstream import h5ad_io

    dst = tmp_path / "a.shardpack"
    pack(grouped_dir, dst)
    pa = PackedArchive(dst)
    pd.testing.assert_frame_equal(pa.obs(), h5ad_io.read_obs(grouped_dir))
