import numpy as np
import pytest

from cellstream.packed.reader import PackedArchive
from cellstream.read import ShardedArchive
from cellstream.write import write_sharded


def test_update_obs_reflects_and_keeps_shards(sample_adata, tmp_path):
    adata = sample_adata
    f = tmp_path / "a.shardpack"
    write_sharded(adata, f, format="v2", n_shards=3)
    before = {
        i: PackedArchive(f).member_bytes(f"shard_{i:04d}.shx")
        for i in range(PackedArchive(f).n_shards)
    }
    before_offsets = {
        m["name"]: m["offset"] for m in PackedArchive(f).members if m["role"] == "shard"
    }

    sh = ShardedArchive(f)
    new_obs = sh.obs.copy()
    new_obs["qc_pass"] = np.arange(len(new_obs)) % 2 == 0
    sh.update_obs(new_obs)

    sh2 = ShardedArchive(f)
    assert "qc_pass" in sh2.obs.columns
    pa = PackedArchive(f)
    for i, b in before.items():
        assert pa.member_bytes(f"shard_{i:04d}.shx") == b  # shard bytes unchanged
    after_offsets = {m["name"]: m["offset"] for m in pa.members if m["role"] == "shard"}
    assert after_offsets == before_offsets  # shard offsets unchanged


def test_update_obs_length_mismatch_raises(sample_adata, tmp_path):
    f = tmp_path / "a.shardpack"
    write_sharded(sample_adata, f, format="v2", n_shards=2)
    sh = ShardedArchive(f)
    with pytest.raises(ValueError):
        sh.update_obs(sh.obs.iloc[:-1].copy())


def test_crash_rollback(sample_adata, tmp_path, monkeypatch):
    import cellstream.packed.writer as W

    f = tmp_path / "a.shardpack"
    write_sharded(sample_adata, f, format="v2", n_shards=2)
    orig_obs = ShardedArchive(f).obs.copy()

    # Inject a failure AFTER truncation/backup but DURING the tail rewrite.
    boom = RuntimeError("boom mid-edit")

    real_finalize = W._finalize_tail

    def _exploding_finalize(*a, **k):
        raise boom

    monkeypatch.setattr(W, "_finalize_tail", _exploding_finalize)
    new_obs = orig_obs.copy()
    new_obs["x"] = 1
    with pytest.raises(RuntimeError):
        ShardedArchive(f).update_obs(new_obs)

    # The .tailbak exists; opening the archive auto-rolls-back.
    monkeypatch.setattr(W, "_finalize_tail", real_finalize)
    sh = ShardedArchive(f)
    assert "x" not in sh.obs.columns
    assert list(sh.obs.index) == list(orig_obs.index)
    _ = sh.to_anndata()  # still readable


def test_update_obs_index_mismatch_raises(sample_adata, tmp_path):
    """update_obs must reject a frame whose index doesn't match (no reordering)."""
    f = tmp_path / "a.shardpack"
    write_sharded(sample_adata, f, format="v2", n_shards=2)
    sh = ShardedArchive(f)
    bad = sh.obs.copy()
    bad.index = [f"renamed_{i}" for i in range(len(bad))]  # same length, different index
    with pytest.raises(ValueError, match="index does not match"):
        sh.update_obs(bad)


def test_recovery_skips_when_lock_held(sample_adata, tmp_path):
    """A reader must NOT roll back the tail while a writer holds the lock."""
    from cellstream import lock as _lock

    f = tmp_path / "a.shardpack"
    write_sharded(sample_adata, f, format="v2", n_shards=2)
    # Simulate an in-progress edit: a .tailbak exists AND a writer holds the lock.
    # (The trailer is still valid here; recovery must not touch the file because
    # the lock is held — it must defer to the live writer.)
    tailbak = f.with_suffix(f.suffix + ".tailbak")
    tailbak.write_bytes(b"\x00" * 16)  # bogus backup; recovery must NOT apply it
    lock_path = f.with_suffix(f.suffix + ".lock")
    size_before = f.stat().st_size
    with _lock.acquire_write_lock(lock_path, op="update_obs"):
        # Opening under a held lock: recovery is skipped, archive still readable.
        sh = ShardedArchive(f)
        assert sh.to_anndata().n_obs == sample_adata.n_obs
        assert tailbak.exists()  # recovery did not run (lock held by "writer")
        assert f.stat().st_size == size_before  # file untouched


def test_update_obs_fast_fails_when_locked(sample_adata, tmp_path):
    """A metadata edit acquires the lock BEFORE parsing → ArchiveLockedError under
    contention, not a confusing torn-tail error."""
    from cellstream import lock as _lock
    from cellstream.errors import ArchiveLockedError

    f = tmp_path / "a.shardpack"
    write_sharded(sample_adata, f, format="v2", n_shards=2)
    sh = ShardedArchive(f)
    new_obs = sh.obs.copy()
    new_obs["k"] = 1
    lock_path = f.with_suffix(f.suffix + ".lock")
    with _lock.acquire_write_lock(lock_path, op="update_obs"):
        with pytest.raises(ArchiveLockedError):
            ShardedArchive(f).update_obs(new_obs)
