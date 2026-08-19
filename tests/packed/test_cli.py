from cellstream.cli import main
from cellstream.read import ShardedArchive, read_h5ad
from cellstream.write import write_sharded


def test_cli_info(sample_adata, tmp_path, capsys):
    """`cellstream info` prints the member table. The `pack`/`unpack` subcommands went with
    the directory container in #154, so the packed file is written directly."""
    f = tmp_path / "a.shad"
    write_sharded(sample_adata, f, format="v2", n_shards=3)
    assert main(["info", str(f)]) == 0
    out = capsys.readouterr().out
    assert "shard_0000.shx" in out and "header.h5ad" in out


def test_read_h5ad_dispatches_packed(sample_adata, tmp_path):
    f = tmp_path / "a.shardpack"
    write_sharded(sample_adata, f, format="v2", n_shards=2)
    a = read_h5ad(f)
    assert a.shape == ShardedArchive(f).shape
