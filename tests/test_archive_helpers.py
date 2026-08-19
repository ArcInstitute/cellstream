"""The shared builders themselves need tests: every migrated fixture inherits their
behaviour, so a silent defect here (wrong codec, wrong worker count, wrong shard sizing)
would make ~160 call sites test something other than what they claim (#154 part B).
"""

import json

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellstream.packed.reader import PackedArchive, is_packed_file
from cellstream.read import ShardedArchive
from tests._archive_helpers import (
    DEFAULT_TARGET_SHARD_BYTES,
    PUBLIC_V2_CODEC,
    REMOVED_KWARGS,
    UNMIRRORED_VALIDATIONS,
    UNSUPPORTED_KWARGS,
    archive_groups,
    archive_has_member,
    archive_manifest,
    archive_member_bytes,
    archive_obs,
    archive_var,
    pack_directory,
    read_v2_host,
    shard_member_bytes,
    shard_paths_and_bases,
    write_archive,
    write_directory,
)


def _toy(n_obs=60, n_var=20, seed=1):
    X = sp.random(n_obs, n_var, density=0.4, format="csr", dtype="float32", random_state=seed)
    X.data = np.ceil(X.data * 50).astype("float32")
    a = ad.AnnData(X=X)
    third = n_obs // 3
    a.obs["gene"] = ["NTC"] * third + ["GA"] * third + ["GB"] * (n_obs - 2 * third)
    return a


def test_write_archive_produces_a_packed_file_at_an_extensionless_path(tmp_path):
    """Migrated fixtures keep their existing paths, so the packed writer must accept a
    path with no .shad suffix and still be detected as packed on read."""
    out = write_archive(_toy(), tmp_path / "arc")
    assert out.is_file()
    assert is_packed_file(out)
    assert ShardedArchive(str(out)).to_anndata().shape == (60, 20)


def test_write_directory_uses_the_SAME_codec_as_the_public_writer(tmp_path):
    """The trap this helper exists to close: write_v2_archive's own default is
    bitshuffle, while write_sharded resolves None -> v2-byte-filter. A directory
    fixture must not silently test a different codec than the packed one."""
    d = write_directory(_toy(), tmp_path / "d")
    packed = write_archive(_toy(), tmp_path / "p")
    assert archive_manifest(d)["codec"] == archive_manifest(packed)["codec"]
    assert archive_manifest(d)["codec"] == PUBLIC_V2_CODEC


def test_write_directory_uses_the_PUBLIC_n_workers_default(tmp_path):
    """Second instance of the same trap as the codec: write_sharded defaults n_workers
    to 16, write_v2_archive to 8 -- and the value is RECORDED in the manifest, so a
    directory fixture built with the internal default differs observably from its
    packed twin."""
    d = write_directory(_toy(), tmp_path / "d")
    p = write_archive(_toy(), tmp_path / "p")
    assert archive_manifest(d)["writer_fingerprint"]["n_workers_at_write"] == 16
    assert (
        archive_manifest(d)["writer_fingerprint"]["n_workers_at_write"]
        == archive_manifest(p)["writer_fingerprint"]["n_workers_at_write"]
    )


def test_write_directory_normalizes_an_explicit_codec_None(tmp_path):
    """Callers migrating a fixture may pass codec=None explicitly; forwarding it raw
    reaches write_v2_archive as an unknown codec rather than the public default."""
    d = write_directory(_toy(), tmp_path / "d", codec=None)
    assert archive_manifest(d)["codec"] == PUBLIC_V2_CODEC


def test_accessors_are_container_agnostic(tmp_path):
    kw = dict(group_by="gene", reference=["NTC"], target_shard_bytes=8192)
    d = write_directory(_toy(), tmp_path / "d", **kw)
    p = write_archive(_toy(), tmp_path / "p", **kw)
    assert archive_manifest(d)["group_by"] == archive_manifest(p)["group_by"] == "gene"
    assert [g["label"] for g in archive_groups(d)] == [g["label"] for g in archive_groups(p)]
    assert archive_member_bytes(d, "manifest.json")[:1] == b"{"
    assert archive_member_bytes(p, "manifest.json")[:1] == b"{"
    assert len(shard_member_bytes(d, 0)) > 0
    assert len(shard_member_bytes(p, 0)) > 0


def test_read_v2_host_is_container_agnostic(tmp_path):
    """The internal reader takes a directory positionally but needs manifest= AND
    shard_locator= for a packed archive, so the two arms take DIFFERENT code paths and
    could silently disagree -- the class of defect that made archive_obs's header
    fallback a bug. Pin them equal on the same source matrix."""
    d = write_directory(_toy(), tmp_path / "d", n_shards=3)
    p = write_archive(_toy(), tmp_path / "p", n_shards=3)
    dd, di, dp = read_v2_host(d)
    pd_, pi, pp = read_v2_host(p)
    np.testing.assert_array_equal(dd, pd_)
    np.testing.assert_array_equal(di, pi)
    np.testing.assert_array_equal(dp, pp)
    assert dd.size == _toy().X.nnz


@pytest.mark.parametrize("container", ["packed", "directory"])
def test_read_v2_host_forwards_kwargs_to_the_internal_reader(tmp_path, container):
    """A row selector must reach read_v2_to_host through BOTH arms -- a helper that
    swallowed **kw would make every migrated subset test read the whole archive and
    still pass its equality assertion against a full-read reference.

    Parameterized over both containers because the two arms call read_v2_to_host with DIFFERENT
    positional/keyword shapes, so forwarding could break in one and not the other. codex pointed out
    the docstring said "BOTH arms" while the test only ever built a packed archive."""
    build = write_archive if container == "packed" else write_directory
    arc = build(_toy(), tmp_path / container[0], n_shards=3)
    full = read_v2_host(arc)[2]
    part = read_v2_host(arc, row_start=0, row_stop=10)[2]
    assert len(full) == _toy().n_obs + 1
    assert len(part) == 11


def test_the_readers_reject_a_multimatrix_archive_the_SAME_WAY(tmp_path):
    """Both arms must refuse a v4 archive, and refuse it identically.

    The packed arm would refuse on its own (shard_location needs matrix=); the directory arm would
    NOT, because a v4 manifest carries the primary family's shard_filename_template at top level, so
    it can silently treat a v4 archive as primary-X-only. Pinning both arms keeps that back.

    ⚠️ The fixture is checked before it is used, on codex's point that the test otherwise cannot
    prove it exercised BOTH arms: if the public multi-matrix route ever produced a directory, both
    iterations would take the directory branch and pass. And schema v4 does not require more than one
    matrix -- an X-only matrices map is valid -- so a writer that silently dropped ``layer:counts``
    while keeping a v4 manifest would pass too.

    ⚠️⚠️ And the counts layer is deliberately DIFFERENT from X. codex round 4: while it was
    ``a.X.copy()``, a locator that aliased ``layer:counts`` to the primary X shard -- or a writer that
    stored X a second time under the counts key -- satisfied both the manifest-key assertion and an
    ``SHX_MAGIC``-only reachability check. That is the plausible-but-wrong-locator defect rounds 2 and
    3 already repaired twice, so the counts family's decoded CONTENT is compared here too."""
    from cellstream.v2.writer import write_v2_multi

    a = _toy()
    # observably different from X, and the control for that is asserted below
    counts = a.X.copy()
    counts.data = counts.data * 3.0 + 1.0
    a.layers["counts"] = counts  # .layers routes the public writer to the v4 packed path
    assert not np.array_equal(counts.data, a.X.tocsr().data), (
        "the counts layer must differ from X, or aliasing it to the X shard would pass"
    )
    # Snapshot the expectations BEFORE either writer runs, and COPY them. AnnData keeps the
    # assigned CSR object and csr.tocsr() defaults to copy=False, so an expectation taken after
    # the write would alias the very array the writer had access to -- an in-place writer mutation
    # would redefine the expected result and pass. (codex round 5.)
    want_counts = counts.tocsr().copy()
    want_x = a.X.tocsr().copy()

    pk = write_archive(a, tmp_path / "pk4")
    dr = tmp_path / "dr4"
    write_v2_multi(a, dr, n_shards=2)

    # the two fixtures really are the two containers
    assert is_packed_file(pk)
    assert dr.is_dir()

    for arc in (pk, dr):
        man = archive_manifest(arc)
        assert man["schema_version"] == 4
        assert {"x", "layer:counts"} <= set(man["matrices"])  # not silently X-only

    # ...and the counts family's shards are really reachable THROUGH EACH CONTAINER'S OWN locator,
    # so "v4" is not just a manifest field. codex round 3: the first version of this check read the
    # template out of the PACKED manifest and joined it onto the directory, which both contradicted
    # the "each container's own locator" claim and would have missed a bad directory template (or
    # rejected two valid writers that chose different ones).
    from cellstream.v2.format import SHX_MAGIC

    packed_path, packed_off, packed_len = PackedArchive(pk).shard_location(0, matrix="layer:counts")
    assert packed_len > 0
    with open(packed_path, "rb") as fh:
        fh.seek(packed_off)
        assert (
            fh.read(4) == SHX_MAGIC
        )  # a real shard at the packed offset, not just a nonzero length

    dr_tmpl = archive_manifest(dr)["matrices"]["layer:counts"]["shard_filename_template"]
    dir_shard = dr / dr_tmpl.format(0)
    with open(dir_shard, "rb") as fh:
        assert fh.read(4) == SHX_MAGIC  # ...and a real shard at the directory's own path

    # ...and the family really holds COUNTS, not a second copy of X. Both containers are read
    # through the public layer accessor -- the directory via pack_directory, since a directory has
    # no public reader. This is what makes "wrote X only" impossible to pass.
    #
    # BOTH families are compared against their own pre-write snapshot. Asserting only
    # "counts != X" would reject a total alias but accept partial corruption of either family
    # (2 shards here, so half of one could be wrong and the inequality would still hold).
    for arc in (pk, pack_directory(dr, tmp_path / "dr4.shad")):
        for got, want, what in (
            (ShardedArchive(arc).layer("counts").to_anndata().X.tocsr(), want_counts, "counts"),
            (ShardedArchive(arc).to_anndata().X.tocsr(), want_x, "x"),
        ):
            assert got.shape == want.shape, f"{arc.name}/{what}: shape"
            # assert_array_equal compares VALUES, not dtypes -- pin the cast-back dtype too
            assert got.dtype == np.dtype(np.float32), f"{arc.name}/{what}: dtype {got.dtype}"
            np.testing.assert_array_equal(got.indptr, want.indptr, err_msg=f"{what}: indptr")
            np.testing.assert_array_equal(got.indices, want.indices, err_msg=f"{what}: indices")
            np.testing.assert_array_equal(got.data, want.data, err_msg=f"{what}: data")

    # identical refusal from both arms -- compared as messages with the path removed, not merely
    # matched on a shared substring
    seen = {}
    for arc in (pk, dr):
        for fn in (read_v2_host, shard_paths_and_bases):
            with pytest.raises(ValueError, match="single-matrix only") as ei:
                fn(arc)
            seen.setdefault(fn.__name__, set()).add(str(ei.value).replace(str(arc), "<ARCHIVE>"))
    for name, msgs in seen.items():
        assert len(msgs) == 1, f"{name} refuses the two containers differently: {msgs}"
        # the message must stay ACTIONABLE, not merely present. codex has now corrected this hint
        # three times (it once pointed at [key], which selects rows; then at .modality(name), which
        # returns a view), so the working advice is pinned here rather than left to review.
        msg = next(iter(msgs))
        assert ".layer(name).to_anndata()" in msg
        assert ".modality(name).to_anndata()" in msg
        assert "selects ROWS" in msg


def test_shard_paths_and_bases_matches_the_container(tmp_path):
    """The two containers give structurally DIFFERENT lists, and that is the point: a
    directory is one file per shard at offset 0, a packed archive is one file n times at
    each shard's aligned offset. Production builds them the same way
    (v2/reader_cpu.py:385-386)."""
    d = write_directory(_toy(), tmp_path / "d", n_shards=3)
    p = write_archive(_toy(), tmp_path / "p", n_shards=3)

    dpaths, dbases = shard_paths_and_bases(d)
    assert len(dpaths) == len(set(dpaths)) == 3
    assert dbases == [0, 0, 0]

    ppaths, pbases = shard_paths_and_bases(p)
    assert len(ppaths) == 3 and set(ppaths) == {str(p)}
    # 4 KB-aligned, strictly increasing, and past the packed head block
    assert pbases == sorted(pbases) and len(set(pbases)) == 3
    assert all(b > 0 and b % 4096 == 0 for b in pbases)


def test_shard_paths_and_bases_actually_locates_the_shard_bytes(tmp_path):
    """(path, base) is only useful if the bytes AT that offset are the shard: the packed
    arm hands a base into the middle of one big file, so an off-by-one member lookup
    would still return a plausible-looking path/offset pair. Check the SHX magic."""
    from cellstream.v2 import format as v2fmt

    for arc in (
        write_directory(_toy(), tmp_path / "d", n_shards=3),
        write_archive(_toy(), tmp_path / "p", n_shards=3),
    ):
        paths, bases = shard_paths_and_bases(arc)
        for path, base in zip(paths, bases, strict=True):
            with open(path, "rb") as fh:
                fh.seek(base)
                assert fh.read(4) == v2fmt.SHX_MAGIC, f"{arc.name}: no shard at {base}"


def test_member_presence_and_obs_var_are_container_agnostic(tmp_path):
    """A packed archive has no `out / "header.h5ad"` path, so a bare .exists() check
    silently returns False and the assertion stops testing anything. Same for reading
    obs.parquet directly. These accessors are what migrated fixtures use instead."""
    d = write_directory(_toy(), tmp_path / "d")
    p = write_archive(_toy(), tmp_path / "p")
    for arc in (d, p):
        assert archive_has_member(arc, "header.h5ad")
        assert archive_has_member(arc, "obs.parquet")
        assert not archive_has_member(arc, "nope.json")
        assert list(archive_obs(arc)["gene"]) == list(_toy().obs["gene"])
        assert len(archive_var(arc)) == 20
    # the naive check the accessor replaces: False on packed, so it must not be used
    assert not (p / "header.h5ad").exists()


def test_archive_obs_falls_back_to_the_header_in_BOTH_containers(tmp_path):
    """Regression pin for the second real bug codex found: the directory arm read
    obs.parquet unconditionally, while PackedArchive.obs() (and the public directory
    reader) fall back to header.h5ad when the sidecar is absent -- so the two containers
    disagreed on any archive written without it.

    The test above cannot catch this: every archive the builders produce HAS obs.parquet,
    so the fallback is never reached. Removing the sidecar is the only way in.
    """
    d = write_directory(_toy(), tmp_path / "d")
    (d / "obs.parquet").unlink()  # control: the sidecar was there to remove
    p = pack_directory(d, tmp_path / "p")  # _enumerate_members skips absent metadata

    assert not archive_has_member(d, "obs.parquet")
    assert not archive_has_member(p, "obs.parquet")
    for arc in (d, p):
        assert list(archive_obs(arc)["gene"]) == list(_toy().obs["gene"])
    assert list(archive_obs(d)["gene"]) == list(archive_obs(p)["gene"])


def test_tamper_then_pack_is_visible_through_the_public_reader(tmp_path):
    """Lets the manifest-tamper suites keep public-reader coverage under option (b)."""
    d = write_directory(_toy(), tmp_path / "d")
    m = json.loads((d / "manifest.json").read_text())
    assert "x_data_dtype_min" in m  # control: the key is there before we remove it
    m.pop("x_data_dtype_min")
    (d / "manifest.json").write_text(json.dumps(m))

    f = pack_directory(d, tmp_path / "arc")
    assert "x_data_dtype_min" not in ShardedArchive(str(f)).manifest


def _spy_on_the_internal_writer(monkeypatch):
    """Capture the kwargs reaching write_v2_archive, for BOTH builders.

    write_sharded's packed branch goes through packed.writer.write_packed, which imports
    write_v2_archive inside the function body -- so patching the module attribute catches
    the public path as well as write_directory's direct call.
    """
    seen = []
    import cellstream.v2.writer as W

    real = W.write_v2_archive
    monkeypatch.setattr(
        W, "write_v2_archive", lambda *a, **k: (seen.append((a, k)), real(*a, **k))[1]
    )
    return seen


def test_write_directory_normalizes_target_shard_bytes_like_write_sharded(tmp_path, monkeypatch):
    """write_sharded turns None into 2 GiB before calling write_v2_archive. A helper
    that forwarded the None raw would not resize anything -- it would crash in the
    writer's `target_shard_bytes <= 0` check -- so the failure this oracle has to catch
    is a wrong-but-POSITIVE normalization.

    Asserts the VALUE reaching the writer, not a shard count. An unnormalized None does
    fail loudly (the internal writer dies on `None <= 0`), but a wrong-but-positive
    normalization does not: a toy archive is ~2 KB, so 1 GiB and 2 GiB both yield exactly
    one shard and a count comparison could never tell them apart.

    And it asserts PARITY, not a duplicated constant: both builders run under one spy and
    their kwargs are compared to each other, so if write_sharded's normalization changed
    while the helper stayed at 2 GiB this goes red instead of agreeing with itself.
    """
    seen = _spy_on_the_internal_writer(monkeypatch)

    write_directory(_toy(), tmp_path / "d", target_shard_bytes=None)
    write_archive(_toy(), tmp_path / "p", target_shard_bytes=None)
    assert len(seen) == 2, "the packed builder must also reach write_v2_archive"
    internal, public = seen[0][1], seen[1][1]

    for knob in ("codec", "n_workers", "target_shard_bytes"):
        assert internal[knob] == public[knob], f"{knob} differs between the two builders"
    # ...and the values are the public ones, not the internal writer's own defaults.
    assert public["target_shard_bytes"] == DEFAULT_TARGET_SHARD_BYTES == 2 * 2**30
    assert public["codec"] == PUBLIC_V2_CODEC
    assert public["n_workers"] == 16


def test_write_directory_refuses_multi_matrix_inputs(tmp_path):
    """write_sharded routes .layers / .raw / MuData to the multi-matrix packed writer.
    Accepting them here would silently write X only."""
    a = _toy()
    a.layers["counts"] = a.X.copy()
    with pytest.raises(TypeError, match="multi-matrix"):
        write_directory(a, tmp_path / "d")


def test_write_directory_refuses_a_multi_matrix_PATH_source(tmp_path):
    """Regression pin for the real bug codex found: the guard used to look only at an
    in-memory AnnData, so a path source carrying layers/raw sailed past it and was
    written X only. The in-memory case above would NOT have caught that -- only a path
    source does, which is why this is a separate test.
    """
    a = _toy()
    a.layers["counts"] = a.X.copy()
    src = tmp_path / "multi.h5ad"
    a.write_h5ad(src)

    out = tmp_path / "d"
    with pytest.raises(TypeError, match="multi-matrix"):
        write_directory(str(src), out)
    assert not out.exists(), "must reject BEFORE writing anything"


def test_write_directory_refuses_a_modalities_source(tmp_path):
    """modalities= is the third public multi-matrix shape. Without it forwarded to the
    detector the call still fails, but on write_v2_archive's 'unexpected keyword
    argument' rather than the deliberate diagnostic."""
    with pytest.raises(TypeError, match="multi-matrix"):
        write_directory(_toy(), tmp_path / "d", modalities={"other": _toy()})


def test_path_source_resolution_matches_write_sharded(tmp_path, monkeypatch):
    """Non-grouped path sources are loaded first; grouped ones are handed to the writer
    to read backed. That second half is a PUBLIC promise -- write_sharded's docstring
    says a v2 path source with group_by "is read lazily (backed) and each shard's rows
    are gathered on demand, the full matrix is never resident" -- so pinning only the
    helper's mirror of it is half a test: write_sharded could start eagerly loading a
    grouped path source and every round-trip test would still pass. Both builders run
    under one spy and are compared to each other.
    """
    src = tmp_path / "src.h5ad"
    _toy().write_h5ad(src)
    seen = _spy_on_the_internal_writer(monkeypatch)

    write_directory(str(src), tmp_path / "d1", n_shards=2)
    write_archive(str(src), tmp_path / "p1", n_shards=2)
    write_directory(str(src), tmp_path / "d2", group_by="gene", target_shard_bytes=8192)
    write_archive(str(src), tmp_path / "p2", group_by="gene", target_shard_bytes=8192)
    assert len(seen) == 4, "both builders must reach write_v2_archive, twice each"

    # A shard-count assertion would pass either way; the argument TYPE is what
    # distinguishes an eagerly-loaded source from one handed through as a path.
    loaded = [isinstance(args[0], ad.AnnData) for args, _kw in seen]
    assert loaded == [True, True, False, False], (
        "non-grouped path sources must be loaded first and grouped ones must stay a "
        f"path, for BOTH builders; got {loaded}"
    )


def test_write_archive_refuses_arguments_that_would_yield_a_directory(tmp_path):
    """write_archive's contract is "packed". The guard dates from when the suite ran with
    packed-only enforcement disabled and container="directory" would have handed back a
    directory from the builder whose whole purpose is to avoid one; #154 part B3b removed
    the parameter, so write_sharded now refuses it first.

    The format arm is now belt-and-braces rather than load-bearing: since #154 part B
    removed the v1 writer, write_sharded rejects format="v1" itself. What this still pins
    is WHOSE error you get -- write_archive's own, naming write_archive -- and that the
    guard fires for any non-v2 value, not just the one that used to build a directory.
    """
    with pytest.raises(ValueError, match="only.*v2|v2.*only"):
        write_archive(_toy(), tmp_path / "v1", format="v1")
    with pytest.raises(ValueError, match="only.*v2|v2.*only"):
        write_archive(_toy(), tmp_path / "v3", format="v3")
    with pytest.raises(ValueError, match="container"):
        write_archive(_toy(), tmp_path / "dir", container="directory")
    # and the redundant-but-harmless spelling is refused too, since the parameter goes away
    with pytest.raises(ValueError, match="container"):
        write_archive(_toy(), tmp_path / "pk", container="packed")


def test_the_two_keyword_lists_match_the_real_signatures():
    """Now that these are tuples rather than comma-separated strings, the classification
    they document can be CHECKED instead of trusted. Without this, a new write_v2_archive
    parameter silently moves a keyword from one category to the other and both lists rot.
    """
    import inspect

    from cellstream.v2.writer import write_v2_archive
    from cellstream.write import write_sharded

    public = set(inspect.signature(write_sharded).parameters)
    internal = set(inspect.signature(write_v2_archive).parameters)

    for name in UNSUPPORTED_KWARGS:
        assert name in public, f"{name} is not a write_sharded parameter"
        assert name not in internal, (
            f"{name} IS accepted by write_v2_archive -- it belongs in UNMIRRORED_VALIDATIONS"
        )
    for name in REMOVED_KWARGS:
        assert name not in public, f"removed write_sharded parameter {name} returned"
    for name in UNMIRRORED_VALIDATIONS:
        assert name in public, f"{name} is not a write_sharded parameter"
        assert name in internal, (
            f"{name} is NOT accepted by write_v2_archive -- it belongs in UNSUPPORTED_KWARGS"
        )
    assert not set(UNSUPPORTED_KWARGS) & set(UNMIRRORED_VALIDATIONS)
