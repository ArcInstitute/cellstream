"""Lean target-aware (grouped) v2 write — gather workers + orchestrator (#76)."""

from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.errors import LossyCastError
from tests._archive_helpers import (
    archive_has_member,
    pack_directory,
    write_archive,
    write_directory,
)


def _csr_adata(seed=5, n_obs=120, n_vars=40, density=0.2):
    rng = np.random.default_rng(seed)
    raw = sp.random(n_obs, n_vars, density=density, format="csr", random_state=rng)
    csr = sp.csr_matrix(raw.shape, dtype=np.float32)
    csr.data = np.ceil(raw.data * 60).astype(np.float32)
    csr.indices = raw.indices.astype(np.int32)
    csr.indptr = raw.indptr.astype(np.int64)
    return ad.AnnData(X=csr)


def _make_grouped_adata():
    # Interleaved labels via the fixed rng(123) shuffle -> a scattered permutation, which
    # is what makes the grouped writer actually gather rather than copy contiguous runs.
    # (Was named _make_golden_adata and had to match tests/v2/_make_grouped_golden.py;
    # #154 part B3 removed both the goldens and that generator, so it stands alone now.)
    rng = np.random.default_rng(7)
    base = np.array(["non-targeting"] * 40 + ["GENE_A"] * 90 + ["GENE_B"] * 60 + ["GENE_C"] * 50)
    labels = base[np.random.default_rng(123).permutation(240)].tolist()
    raw = sp.random(240, 60, density=0.15, format="csr", random_state=rng)
    csr = sp.csr_matrix(raw.shape, dtype=np.float32)
    csr.data = np.ceil(raw.data * 80).astype(np.float32)
    csr.indices = raw.indices.astype(np.int32)
    csr.indptr = raw.indptr.astype(np.int64)
    return ad.AnnData(X=csr, obs={"target_gene": labels})


def test_lean_grouped_inmem_roundtrip(tmp_path):
    from cellstream import ShardedArchive

    a = _make_grouped_adata()
    out = tmp_path / "arc"
    write_directory(
        a,
        out,
        group_by="target_gene",
        reference="non-targeting",
        n_workers=4,
        target_shard_bytes=64 * 1024,
    )
    # #154 part B3a: this file's writer stays write_directory throughout -- several tests
    # compare loose shard BYTES between two directory archives (route B), and the backed
    # grouped write path is the subject (route D). The four tests that also read through the
    # PUBLIC reader pack the directory first; _pack_directory is a pure byte concat
    # (packed/writer.py:111), so the shards and manifest asserted on above are unchanged.
    arc = ShardedArchive(pack_directory(out, tmp_path / "arc.shad"))
    full = arc.to_anndata()
    orig_row = full.obs["_orig_row"].to_numpy()
    assert (full.X.toarray() == a.X.toarray()[orig_row]).all()

    import json

    m = json.loads((out / "manifest.json").read_text())
    assert m["reference_shard"] == 0


def test_gather_workers_byte_identical_local_vs_backed(tmp_path):
    """The local (in-RAM) and backed gather workers produce byte-identical .shx
    for the same scattered row list."""
    from cellstream.v2.writer import (
        _encode_one_shard_gather_backed,
        _encode_one_shard_gather_local,
    )

    a = _csr_adata()
    oi = np.array([90, 1, 5, 5, 119, 0, 33, 12, 77], dtype=np.int64)
    n_vars = a.n_vars

    d_local = tmp_path / "local"
    d_local.mkdir()
    _encode_one_shard_gather_local(
        (
            d_local,
            0,
            a.X.data,
            a.X.indices,
            a.X.indptr,
            oi,
            n_vars,
            "bitshuffle",
            "uint32",  # on-disk integer width is now uint32 (matches the backed worker)
            "uint32",
        )
    )
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    d_backed = tmp_path / "backed"
    d_backed.mkdir()
    _encode_one_shard_gather_backed((d_backed, 0, str(src), oi, n_vars, "bitshuffle", "uint32"))

    assert (d_local / "shard_0000.shx").read_bytes() == (d_backed / "shard_0000.shx").read_bytes()


def test_gather_shm_matches_local(tmp_path):
    if sys.platform != "linux":
        pytest.skip("SharedMemory path is Linux-only")
    from cellstream import shm as _shm
    from cellstream.v2.writer import (
        _encode_one_shard_gather_local,
        _encode_one_shard_gather_shm,
    )

    a = _csr_adata(seed=6)
    oi = np.array([10, 2, 100, 0, 55], dtype=np.int64)
    n_vars = a.n_vars
    total_nnz = int(a.X.data.size)
    n_indptr = a.n_obs + 1

    d_local = tmp_path / "local"
    d_local.mkdir()
    _encode_one_shard_gather_local(
        (
            d_local,
            0,
            a.X.data,
            a.X.indices,
            a.X.indptr,
            oi,
            n_vars,
            "v2-byte-filter",
            "uint16",
            "uint16",
        )
    )

    bundle = _shm.allocate_shm_buffers(
        data_nbytes=max(1, total_nnz * 2),
        indices_nbytes=max(1, total_nnz * 2),
        indptr_nbytes=max(1, n_indptr * 8),
    )
    try:
        dv = np.ndarray((total_nnz,), dtype=np.uint16, buffer=bundle.data.buf)
        iv = np.ndarray((total_nnz,), dtype=np.uint16, buffer=bundle.indices.buf)
        pv = np.ndarray((n_indptr,), dtype=np.int64, buffer=bundle.indptr.buf)
        dv[:] = a.X.data
        iv[:] = a.X.indices
        pv[:] = a.X.indptr
        d_shm = tmp_path / "shm"
        d_shm.mkdir()
        _encode_one_shard_gather_shm(
            (
                d_shm,
                0,
                bundle.data.name,
                bundle.indices.name,
                bundle.indptr.name,
                total_nnz,
                n_indptr,
                oi,
                n_vars,
                "v2-byte-filter",
                "uint16",
                "uint16",
            )
        )
    finally:
        bundle.close_and_unlink()

    assert (d_local / "shard_0000.shx").read_bytes() == (d_shm / "shard_0000.shx").read_bytes()


def test_backed_worker_accepts_counts_over_uint16(tmp_path):
    """M3: on-disk integer width is uint32 (#158), so the backed worker must ACCEPT
    counts > 65535 when given a uint32 target (was a hardcoded uint16 cast). The
    archive-wide recorded min is now set by the upstream pre-scan (see the end-to-end
    ``test_backed_grouped_write_accepts_large_counts``), so the worker no longer reports
    a per-shard min."""
    from cellstream.v2.writer import _encode_one_shard_gather_backed

    a = _csr_adata(n_obs=30, n_vars=10)
    a.X.data[0] = 70000.0  # exceeds uint16, fits uint32
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    out = tmp_path / "out"
    out.mkdir()
    res = _encode_one_shard_gather_backed(
        (out, 0, str(src), np.arange(30, dtype=np.int64), 10, "bitshuffle", "uint32")
    )
    assert (out / "shard_0000.shx").exists()
    assert res["nnz"] > 0


def test_backed_worker_negative_index_guard(tmp_path):
    """L12: when n_vars > 65536 the backed worker mirrors the in-mem guard and fails
    loud on a negative column index (which would silently wrap to a huge uint32)."""
    from cellstream.v2.writer import _encode_one_shard_gather_backed

    n_vars = 70000  # > 65536
    X = sp.csr_matrix(
        (
            np.array([1, 2, 3], dtype=np.float32),
            np.array([5, 10, 20], dtype=np.int32),
            np.array([0, 1, 2, 3], dtype=np.int64),
        ),
        shape=(3, n_vars),
    )
    X.indices[0] = -1  # corrupt: negative column index
    a = ad.AnnData(X=X)
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(LossyCastError):
        _encode_one_shard_gather_backed(
            (out, 0, str(src), np.array([0], dtype=np.int64), n_vars, "bitshuffle", "uint32")
        )


def test_backed_grouped_write_accepts_large_counts(tmp_path):
    """M3 end-to-end: a backed grouped write of counts > 65535 succeeds, round-trips,
    and records x_data_dtype_min == 'uint32'."""
    from cellstream import ShardedArchive
    from cellstream.manifest import load_manifest

    a = _csr_adata(n_obs=120, n_vars=40)
    a.X.data[0] = 70000.0  # > uint16
    a.obs["target_gene"] = [f"g{i % 6}" for i in range(120)]
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    out = tmp_path / "out"
    write_directory(
        str(src),
        out,
        group_by="target_gene",
        target_shard_bytes=2048,
    )
    assert load_manifest(out / "manifest.json")["x_data_dtype_min"] == "uint32"
    back = ShardedArchive(pack_directory(out, tmp_path / "out.shad")).to_anndata(cast_back=True)
    assert int(back.X.tocsr().nnz) == int(a.X.nnz)
    assert float(back.X.tocsr().data.max()) == 70000.0


def test_backed_grouped_write_small_counts_records_uint8(tmp_path):
    """A backed grouped write whose counts all fit uint8 records the narrowest-uint min
    (uint8), matching the in-memory path. (Pre-fix, the backed path floored the recorded
    min at uint16; parity now records uint8, a strictly-safe read-time cast hint.)"""
    from cellstream.manifest import load_manifest

    a = _csr_adata(n_obs=120, n_vars=40)  # values <= ~60
    a.obs["target_gene"] = [f"g{i % 6}" for i in range(120)]
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    out = tmp_path / "out"
    write_directory(
        str(src),
        out,
        group_by="target_gene",
        target_shard_bytes=2048,
    )
    assert load_manifest(out / "manifest.json")["x_data_dtype_min"] == "uint8"


def test_resolve_metadata_inmem_and_backed_match(tmp_path):
    from cellstream.v2.writer import _resolve_grouped_metadata

    a = _csr_adata(n_obs=40, n_vars=10)
    a.obs["target_gene"] = ["NTC"] * 10 + ["G1"] * 15 + ["G2"] * 15
    a.obsm["X_pca"] = np.arange(40 * 3, dtype=np.float64).reshape(40, 3)
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)

    m_inmem = _resolve_grouped_metadata(a)
    m_backed = _resolve_grouped_metadata(str(src))
    assert m_inmem.n_obs == m_backed.n_obs == 40
    assert m_inmem.n_vars == m_backed.n_vars == 10
    assert (m_inmem.indptr == m_backed.indptr).all()
    assert list(m_inmem.obs["target_gene"]) == list(m_backed.obs["target_gene"])
    assert (m_inmem.obsm["X_pca"] == m_backed.obsm["X_pca"]).all()


def test_write_permuted_metadata_orders_obs_and_obsm(tmp_path):
    import pandas as pd

    from cellstream.header import read_header_full
    from cellstream.v2.writer import _resolve_grouped_metadata, _write_permuted_metadata

    a = _csr_adata(n_obs=12, n_vars=6)
    a.obs["target_gene"] = ["B"] * 6 + ["A"] * 6
    a.obs.index = [f"cell{i}" for i in range(12)]
    a.obsm["emb"] = np.arange(12 * 2, dtype=np.float64).reshape(12, 2)
    m = _resolve_grouped_metadata(a)
    perm = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=np.int64)  # A-first
    out = tmp_path / "arc"
    out.mkdir()
    _write_permuted_metadata(out, m, perm)

    h = read_header_full(out / "header.h5ad")
    assert list(h.obs.index) == [f"cell{i}" for i in perm]
    assert h.obs["_orig_row"].tolist() == perm.tolist()
    assert (h.obsm["emb"] == m.obsm["emb"][perm]).all()
    pq = pd.read_parquet(out / "obs.parquet")
    assert pq["_orig_row"].tolist() == perm.tolist()


# test_write_nthreads_v1_raises lived here: write_nthreads was the v2 encode-thread knob
# and raised for format="v1" (which used blosc_nthreads instead). Both sides of that pair
# went in #154 -- the v1 writer AND the write-side blosc_nthreads parameter -- so
# write_nthreads has only the positivity check below left to fail.


def test_write_nthreads_must_be_positive(tmp_path):
    from cellstream import write_sharded

    a = _make_grouped_adata()
    with pytest.raises(ValueError, match="positive integer"):
        write_sharded(
            a,
            tmp_path / "arc",
            group_by="target_gene",
            reference="non-targeting",
            write_nthreads=0,
            target_shard_bytes=64 * 1024,
        )


def test_write_nthreads_v2_honored(tmp_path, monkeypatch):
    """Should succeed and produce a valid archive with an explicit thread count.

    Stays on the PUBLIC writer (#154 part B): write_nthreads is in the helper's
    UNMIRRORED_VALIDATIONS list, and "honored" means write_sharded FORWARDS it -- which
    routing this at the internal writer would stop testing. And "produced an archive" on
    its own does not prove forwarding, so the value reaching write_v2_archive is
    asserted too; write_packed imports it inside the function body, so patching the
    module attribute catches the packed path.
    """
    import cellstream.v2.writer as W

    seen = {}
    real = W.write_v2_archive
    monkeypatch.setattr(W, "write_v2_archive", lambda *a, **k: (seen.update(k), real(*a, **k))[1])

    a = _make_grouped_adata()
    out = tmp_path / "arc"
    write_archive(
        a,
        out,
        group_by="target_gene",
        reference="non-targeting",
        write_nthreads=2,
        n_workers=2,
        target_shard_bytes=64 * 1024,
    )
    assert seen["write_nthreads"] == 2, "write_sharded must forward write_nthreads"
    # a bare (out / "manifest.json").exists() is always False on a packed archive
    assert archive_has_member(out, "manifest.json")


@pytest.mark.parametrize("codec", ["v2-byte-filter", "bitshuffle"])
def test_grouped_path_source_byte_identical_to_inmem(tmp_path, codec):

    a = _make_grouped_adata()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)

    out_inmem = tmp_path / "inmem"
    write_directory(
        a,
        out_inmem,
        group_by="target_gene",
        reference="non-targeting",
        codec=codec,
        n_workers=4,
        target_shard_bytes=64 * 1024,
    )

    out_path = tmp_path / "path"
    write_directory(
        src,
        out_path,
        group_by="target_gene",
        reference="non-targeting",
        codec=codec,
        n_workers=4,
        target_shard_bytes=64 * 1024,
    )

    inmem_shards = sorted(out_inmem.glob("shard_*.shx"))
    path_shards = sorted(out_path.glob("shard_*.shx"))
    assert [p.name for p in inmem_shards] == [p.name for p in path_shards]
    for inmem, path in zip(inmem_shards, path_shards, strict=True):
        assert inmem.read_bytes() == path.read_bytes(), f"{path.name} differs from in-memory"


def _corrupt_index_adata(n_obs=200, n_vars=70000, density=0.001, seed=0):
    """Integer-valued float32 CSR with n_vars > 65536 and a corrupt negative column
    index — a genuinely un-storable structure (a negative index would wrap to a huge
    uint32), so the backed grouped write fails loud mid-encode."""
    rng = np.random.default_rng(seed)
    raw = sp.random(n_obs, n_vars, density=density, format="csr", random_state=rng)
    X = sp.csr_matrix(
        (
            np.ceil(raw.data * 10).astype(np.float32),
            raw.indices.astype(np.int32),
            raw.indptr.astype(np.int64),
        ),
        shape=(n_obs, n_vars),
    )
    X.indices[0] = -1  # corrupt column index
    return ad.AnnData(X=X, obs={"target_gene": [f"g{i % 5}" for i in range(n_obs)]})


def test_grouped_path_fail_loud_leaves_no_manifest(tmp_path):
    from cellstream import write_sharded

    # A negative COLUMN INDEX with n_vars > 65536 is genuinely un-storable -> fail loud.
    # (Negative DATA values are now stored as native float, matching the in-mem path, so
    # they no longer trigger the lossy path.)
    a = _corrupt_index_adata()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)

    out = tmp_path / "arc"
    with pytest.raises((LossyCastError, RuntimeError)):
        write_sharded(
            src,
            out,
            group_by="target_gene",
            n_workers=4,
            target_shard_bytes=64 * 1024,
        )
    # Packed: the archive is one file, so its ABSENCE is the assertion. The old
    # per-member check is False for a packed path no matter what the writer did.
    assert not out.exists()

    # A clean source (no corrupt index) writes a manifest.
    n_obs, n_vars = 200, 70000
    rng = np.random.default_rng(1)
    raw = sp.random(n_obs, n_vars, density=0.001, format="csr", random_state=rng)
    clean = ad.AnnData(
        X=sp.csr_matrix(
            (
                np.ceil(raw.data * 10).astype(np.float32),
                raw.indices.astype(np.int32),
                raw.indptr.astype(np.int64),
            ),
            shape=(n_obs, n_vars),
        ),
        obs={"target_gene": [f"g{i % 5}" for i in range(n_obs)]},
    )
    clean.write_h5ad(src)
    write_directory(
        src,
        out,
        group_by="target_gene",
        n_workers=4,
        target_shard_bytes=64 * 1024,
        overwrite=True,
    )
    assert (out / "manifest.json").exists()


def test_inmem_grouped_has_no_eager_permuted_copy():
    """Static guard: the lean grouped in-mem path must NOT reintroduce the eager
    full-CSR `adata[perm].copy()` idiom. Checks the executable body only
    (docstring stripped) so a docstring that *mentions* the legacy idiom doesn't
    trip the guard."""
    import ast
    import inspect
    import textwrap

    import cellstream.v2.writer as w

    def _executable_body(fn):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        func = tree.body[0]
        body = func.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]  # drop the docstring
        return "\n".join(ast.unparse(node) for node in body)

    text = _executable_body(w._write_v2_grouped) + "\n" + _executable_body(w._grouped_encode_inmem)
    assert "].copy()" not in text, "eager permuted-copy idiom must not reappear in the lean path"


def test_grouped_obsm_sparse_preserved_and_permuted(tmp_path):
    """A sparse obsm must survive the grouped write without being densified, and
    must be permuted consistently with _orig_row (guards the obsm-permute fix)."""
    from cellstream.header import read_header_full

    a = _make_grouped_adata()
    a.obsm["sp_emb"] = sp.random(a.n_obs, 5, density=0.4, format="csr", random_state=1)
    out = tmp_path / "arc"
    write_directory(
        a,
        out,
        group_by="target_gene",
        reference="non-targeting",
        target_shard_bytes=64 * 1024,
    )
    h = read_header_full(out / "header.h5ad")
    assert sp.issparse(h.obsm["sp_emb"]), "sparse obsm must not be densified by the writer"
    orig_row = h.obs["_orig_row"].to_numpy()
    assert (h.obsm["sp_emb"].toarray() == a.obsm["sp_emb"].toarray()[orig_row]).all()


# ---------------------------------------------------------------------------
# #87: sort_within_group
# ---------------------------------------------------------------------------


# test_sort_within_group_v1_raises lived here (#154): the knob was v2-only and raised for
# format="v1". grouping.validate_sort_params lost that arm with the v1 writer; its
# requires-group_by and invalid-value arms are the tests below.


def test_sort_within_group_requires_group_by(tmp_path):
    from cellstream import write_sharded

    a = _make_grouped_adata()
    with pytest.raises(ValueError, match="group_by"):
        write_sharded(
            a,
            tmp_path / "arc",
            sort_within_group="target_gene",
            target_shard_bytes=64 * 1024,
        )


def test_sort_within_group_missing_column_raises(tmp_path):
    from cellstream import write_sharded

    a = _make_grouped_adata()
    with pytest.raises(ValueError, match="nope"):
        write_sharded(
            a,
            tmp_path / "arc",
            group_by="target_gene",
            reference="non-targeting",
            sort_within_group="nope",
            target_shard_bytes=64 * 1024,
        )


def _sorted_adata():
    # 90 cells, 3 genes, a guide column with several guides per gene.
    rng = np.random.default_rng(11)
    genes = np.array(["GENE_A"] * 30 + ["GENE_B"] * 30 + ["GENE_C"] * 30)
    guides = np.array([f"g{(i % 3)}" for i in range(90)])  # g0/g1/g2 within each gene
    raw = sp.random(90, 24, density=0.25, format="csr", random_state=rng)
    csr = sp.csr_matrix(raw.shape, dtype=np.float32)
    csr.data = np.ceil(raw.data * 40).astype(np.float32)
    csr.indices = raw.indices.astype(np.int32)
    csr.indptr = raw.indptr.astype(np.int64)
    return ad.AnnData(X=csr, obs={"target_gene": genes, "guide": guides})


@pytest.mark.parametrize("keys", ["guide", ["guide"]])
def test_sort_within_group_roundtrip_and_order(tmp_path, keys):
    """A sorted grouped write round-trips, and within each group the rows are in
    the secondary-key order of the source."""
    from cellstream import ShardedArchive

    a = _sorted_adata()
    out = tmp_path / "arc"
    # PUBLIC writer (#154 part B): these four sort-behaviour tests read the archive
    # back through ShardedArchive and never touch loose members, and their subject is
    # that write_sharded FORWARDS the sort knobs -- which sort_within_group and
    # sort_order being in UNMIRRORED_VALIDATIONS says it must prove on the public API.
    write_archive(
        a,
        out,
        group_by="target_gene",
        sort_within_group=keys,
        n_workers=4,
        target_shard_bytes=64 * 1024,
    )
    arc = ShardedArchive(out)
    full = arc.to_anndata()
    orig = full.obs["_orig_row"].to_numpy()
    # round-trip: X matches the source gathered in the written order
    assert (full.X.toarray() == a.X.toarray()[orig]).all()
    # within each gene, rows ordered by guide (stable)
    genes = a.obs["target_gene"].to_numpy()[orig]
    guides = a.obs["guide"].to_numpy()[orig]
    for g in ("GENE_A", "GENE_B", "GENE_C"):
        gg = guides[genes == g]
        assert list(gg) == sorted(gg)


def test_sort_within_group_inmem_matches_path(tmp_path):
    """Sorted output is byte-identical between the in-mem and path providers."""

    a = _sorted_adata()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    out_mem = tmp_path / "mem"
    out_path = tmp_path / "path"
    for tgt, source in ((out_mem, a), (out_path, str(src))):
        write_directory(
            source,
            tgt,
            group_by="target_gene",
            sort_within_group=["guide"],
            n_workers=4,
            target_shard_bytes=64 * 1024,
        )
    for n in sorted(out_mem.glob("shard_*.shx")):
        assert n.read_bytes() == (out_path / n.name).read_bytes(), f"{n.name} differs"


# test_sort_order_v1_raises lived here (#154) -- same story as its sort_within_group twin
# above: a v2-only knob whose v1 arm went with the v1 writer.


def test_sort_order_requires_group_by(tmp_path):
    from cellstream import write_sharded

    a = _make_grouped_adata()
    with pytest.raises(ValueError, match="group_by"):
        write_sharded(
            a,
            tmp_path / "arc",
            sort_order="natural",
            target_shard_bytes=64 * 1024,
        )


def test_sort_order_invalid_value_raises(tmp_path):
    from cellstream import write_sharded

    a = _make_grouped_adata()
    with pytest.raises(ValueError, match="natural"):
        write_sharded(
            a,
            tmp_path / "arc",
            group_by="target_gene",
            sort_order="bogus",
            target_shard_bytes=64 * 1024,
        )


def _cat_order_adata():
    # 60 cells, 2 genes with a NON-alphabetical category order (B before A) and a
    # guide column with a non-alphabetical category order (g2 < g1 < g0).
    rng = np.random.default_rng(7)
    genes = pd.Categorical(["GENE_A"] * 30 + ["GENE_B"] * 30, categories=["GENE_B", "GENE_A"])
    guides = pd.Categorical([f"g{i % 3}" for i in range(60)], categories=["g2", "g1", "g0"])
    raw = sp.random(60, 24, density=0.25, format="csr", random_state=rng)
    csr = sp.csr_matrix(raw.shape, dtype=np.float32)
    csr.data = np.ceil(raw.data * 40).astype(np.float32)
    csr.indices = raw.indices.astype(np.int32)
    csr.indptr = raw.indptr.astype(np.int64)
    return ad.AnnData(X=csr, obs={"target_gene": genes, "guide": guides})


def _written_group_order(out_dir):
    from cellstream import ShardedArchive

    written = ShardedArchive(out_dir).to_anndata().obs["target_gene"].astype(str).to_numpy()
    return list(dict.fromkeys(written))  # distinct groups in written order


def test_sort_order_natural_respects_category_order(tmp_path):
    """Default sort_order ('natural') orders a categorical group_by by CATEGORY
    order, not alphabetically."""

    a = _cat_order_adata()
    out = tmp_path / "arc"
    # PUBLIC writer -- see test_sort_within_group_roundtrip_and_order (#154 part B).
    write_archive(
        a,
        out,
        group_by="target_gene",
        n_workers=4,
        target_shard_bytes=64 * 1024,
    )
    assert _written_group_order(out) == ["GENE_B", "GENE_A"]


def test_sort_order_alphabetical_overrides_category_order(tmp_path):
    """sort_order='alphabetical' forces alphabetical group order."""

    a = _cat_order_adata()
    out = tmp_path / "arc"
    # PUBLIC writer -- see test_sort_within_group_roundtrip_and_order (#154 part B).
    write_archive(
        a,
        out,
        group_by="target_gene",
        sort_order="alphabetical",
        n_workers=4,
        target_shard_bytes=64 * 1024,
    )
    assert _written_group_order(out) == ["GENE_A", "GENE_B"]


def test_sort_order_natural_secondary_respects_category_order(tmp_path):
    """A categorical sort_within_group is ordered by category order under the
    natural default (and differs from alphabetical)."""
    from cellstream import ShardedArchive

    a = _cat_order_adata()
    out = tmp_path / "arc"
    # PUBLIC writer -- see test_sort_within_group_roundtrip_and_order (#154 part B).
    write_archive(
        a,
        out,
        group_by="target_gene",
        sort_within_group="guide",
        n_workers=4,
        target_shard_bytes=64 * 1024,
    )
    full = ShardedArchive(out).to_anndata()
    orig = full.obs["_orig_row"].to_numpy()
    genes = a.obs["target_gene"].astype(str).to_numpy()[orig]
    guides = a.obs["guide"].astype(str).to_numpy()[orig]
    cat_rank = {"g2": 0, "g1": 1, "g0": 2}
    for g in ("GENE_A", "GENE_B"):
        gg = list(guides[genes == g])
        assert gg == sorted(gg, key=lambda x: cat_rank[x])  # category order
        assert gg != sorted(gg)  # differs from alphabetical (g0<g1<g2)


def test_sort_order_inmem_matches_path_categorical(tmp_path):
    """in-mem == path byte-identity for a custom-order categorical group_by +
    categorical sort_within_group (anndata preserves the category order on h5ad)."""

    a = _cat_order_adata()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    out_mem = tmp_path / "mem"
    out_path = tmp_path / "path"
    for tgt, source in ((out_mem, a), (out_path, str(src))):
        write_directory(
            source,
            tgt,
            group_by="target_gene",
            sort_within_group="guide",
            n_workers=4,
            target_shard_bytes=64 * 1024,
        )
    shards = sorted(out_mem.glob("shard_*.shx"))
    assert shards  # non-empty
    for n in shards:
        assert n.read_bytes() == (out_path / n.name).read_bytes(), f"{n.name} differs"


@pytest.mark.parametrize(
    "dtype, vals",
    [
        (np.float32, [1.5, 2.25, 0.75]),  # fractional -> (float32, float32)
        (np.float32, [3.0, 100.0, 7.0]),  # integer-valued float -> (uint32, uint8)
        (np.int32, [-1, 5, 3]),  # negative -> (float64, float64)
        (np.uint64, [5, 10, 5_000_000_000]),  # > uint32 -> (float64, float64)
        (np.uint16, [3, 250, 7]),  # small counts -> (uint32, uint8)
    ],
)
def test_decide_backed_matches_inmem(tmp_path, dtype, vals):
    from cellstream import narrow
    from cellstream.v2.writer import _decide_backed_on_disk_data_dtype

    n_obs, n_var = 8, 4
    dense = np.zeros((n_obs, n_var), dtype=dtype)
    flat = dense.reshape(-1)
    for i, v in enumerate(vals):
        flat[i * 2] = v
    X = sp.csr_matrix(dense.astype(dtype))
    a = ad.AnnData(X=X)
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)

    inmem_on_disk, _, inmem_min, _ = narrow.decide_on_disk_dtypes(X.data, X.indices, n_var)
    backed_on_disk, backed_min = _decide_backed_on_disk_data_dtype(str(src))
    assert backed_on_disk == inmem_on_disk
    assert backed_min == inmem_min


def _fractional_adata(seed=11, n_obs=120, n_vars=40, density=0.2):
    rng = np.random.default_rng(seed)
    raw = sp.random(n_obs, n_vars, density=density, format="csr", random_state=rng)
    csr = sp.csr_matrix(raw.shape, dtype=np.float32)
    csr.data = (raw.data * 3.0 + 0.25).astype(np.float32)  # fractional
    csr.indices = raw.indices.astype(np.int32)
    csr.indptr = raw.indptr.astype(np.int64)
    a = ad.AnnData(X=csr)
    a.obs["grp"] = [f"g{i % 6}" for i in range(n_obs)]
    return a


def test_backed_grouped_float_roundtrip(tmp_path):
    """REGRESSION: a backed grouped write of fractional float X succeeds and round-trips
    (previously raised LossyCastError)."""
    from cellstream import ShardedArchive
    from cellstream.manifest import load_manifest

    a = _fractional_adata()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    out = tmp_path / "out"
    write_directory(str(src), out, group_by="grp", target_shard_bytes=2048)
    m = load_manifest(out / "manifest.json")
    assert m["x_data_dtype_on_disk"] == "float32"
    assert m["x_data_dtype_min"] == "float32"
    back = (
        ShardedArchive(pack_directory(out, tmp_path / "out.shad"))
        .to_anndata(cast_back=True)
        .X.tocsr()
    )
    exp = a.X.tocsr()
    assert back.nnz == exp.nnz
    np.testing.assert_allclose(np.sort(back.data), np.sort(exp.data), rtol=0, atol=0)


def test_backed_grouped_float_byte_identical_to_inmem(tmp_path):
    """A backed grouped float write is byte-identical (shards + dtype manifest fields) to
    the in-memory grouped write on the same data."""
    from cellstream.manifest import load_manifest

    a = _fractional_adata()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    d_in = tmp_path / "inmem"
    d_bk = tmp_path / "backed"
    write_directory(a, d_in, group_by="grp", target_shard_bytes=2048)
    write_directory(str(src), d_bk, group_by="grp", target_shard_bytes=2048)
    shards = sorted(p.name for p in d_in.glob("shard_*.shx"))
    assert shards and shards == sorted(p.name for p in d_bk.glob("shard_*.shx"))
    for name in shards:
        assert (d_in / name).read_bytes() == (d_bk / name).read_bytes(), name
    mi, mb = load_manifest(d_in / "manifest.json"), load_manifest(d_bk / "manifest.json")
    for k in (
        "x_data_dtype_on_disk",
        "x_data_dtype_min",
        "x_indices_dtype_on_disk",
        "x_indices_dtype_min",
    ):
        assert mi[k] == mb[k], k


def test_backed_grouped_integer_float_still_uint32(tmp_path):
    """Integer-valued float32 through the backed path still lands uint32 on-disk (no
    regression of the compression win)."""
    from cellstream.manifest import load_manifest

    a = _csr_adata(n_obs=120, n_vars=40)  # np.ceil(...) -> integer-valued float32
    a.obs["grp"] = [f"g{i % 6}" for i in range(120)]
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    out = tmp_path / "out"
    write_directory(str(src), out, group_by="grp", target_shard_bytes=2048)
    assert load_manifest(out / "manifest.json")["x_data_dtype_on_disk"] == "uint32"


def test_backed_grouped_negative_stores_float(tmp_path):
    """BEHAVIOR CHANGE (matches in-mem): a negative value through the backed path stores
    as native float (float32 here, the source's width) and round-trips, instead of
    raising LossyCastError. (An int source with negatives widens to float64; see
    test_decide_backed_matches_inmem.)"""
    from cellstream import ShardedArchive
    from cellstream.manifest import load_manifest

    a = _csr_adata(n_obs=120, n_vars=40)  # float32 source
    a.X.data[0] = -5.0
    a.obs["grp"] = [f"g{i % 6}" for i in range(120)]
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    out = tmp_path / "out"
    write_directory(str(src), out, group_by="grp", target_shard_bytes=2048)
    assert load_manifest(out / "manifest.json")["x_data_dtype_on_disk"] == "float32"
    back = (
        ShardedArchive(pack_directory(out, tmp_path / "out.shad"))
        .to_anndata(cast_back=True)
        .X.tocsr()
    )
    assert float(back.data.min()) == -5.0
