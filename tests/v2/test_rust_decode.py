"""Tests for the Rust v2 decode core (cellstream.v2._rust.decode_v2_csr et al.)."""

from __future__ import annotations

import numpy as np
import pytest

from tests._archive_helpers import (
    archive_manifest,
    read_v2_host,
    shard_paths_and_bases,
    write_archive,
)


def _write_small_uint16_archive(tmp_path):
    """Small integral float32 AnnData -> narrows to uint16 on disk (PACKED).

    #154 part B3a, route A: the subject of this file is the Rust decode core, never the
    container. The public reads below work unchanged on packed, and the low-level
    decode_v2_csr callers get their per-shard (file, offset) pairs from
    ``shard_paths_and_bases`` -- which for packed is the SAME pair production builds
    (v2/reader_cpu.py:385-386), so the container swap makes those tests stricter rather
    than weaker.
    """
    import anndata as ad
    import scipy.sparse as sp

    X = sp.random(40, 30, density=0.3, format="csr", random_state=7).astype(np.float32)
    X.data = np.floor(X.data * 50).astype(np.float32) + 1.0  # integral, in-range
    adata = ad.AnnData(X=X)
    out = tmp_path / "u16"
    write_archive(adata, out, n_shards=4)
    return out, adata


def test_use_rust_for_read_accepts_rawdata_marker():
    """#144: the Rust read gate must engage for BOTH byte-filter markers (the
    old shuffled and the new rawdata), so float archives use the fast Rust path.
    bitshuffle still falls back to Python."""
    pytest.importorskip("cellstream.v2._rust")
    from cellstream.v2 import codec
    from cellstream.v2._rust_dispatch import rust_available, use_rust_for_read
    from cellstream.v2.materialize import MaterializePlan

    if not rust_available():
        pytest.skip("Rust core disabled; gate is only meaningful with Rust enabled")

    plan = MaterializePlan(
        container="csr",
        data_dtype=np.dtype("float32"),
        index_dtype=np.dtype("uint16"),
        indptr_dtype=np.dtype("int64"),
        allow_lossy=False,
        n_vars=100,
    )
    assert use_rust_for_read(codec.BYTE_FILTER_SHUFFLE_NAME, plan) is True
    assert use_rust_for_read(codec.BYTE_FILTER_RAWDATA_SHUFFLE_NAME, plan) is True
    assert use_rust_for_read("bitshuffle", plan) is False
    # dense container too (data-dtype-only gate)
    dense_plan = MaterializePlan(
        container="dense",
        data_dtype=np.dtype("float32"),
        index_dtype=np.dtype("int64"),
        indptr_dtype=np.dtype("int64"),
        allow_lossy=False,
        n_vars=100,
    )
    assert use_rust_for_read(codec.BYTE_FILTER_RAWDATA_SHUFFLE_NAME, dense_plan) is True


def test_rust_rawdata_decode_matches_python(tmp_path, monkeypatch):
    """#144 end-to-end: a float (rawdata-marker) archive decoded by the Rust core
    equals the Python decode and is lossless. Confirms the Rust read path actually
    handles the rawdata marker (not a silent fallback)."""
    pytest.importorskip("cellstream.v2._rust")
    import anndata as ad
    import scipy.sparse as sp

    from cellstream import ShardedArchive
    from cellstream.v2._rust_dispatch import rust_available
    from cellstream.v2.format import parse_shx_header

    rng = np.random.default_rng(3)
    X = sp.random(60, 24, density=0.3, format="csr", random_state=3).astype(np.float32)
    X.data = np.log1p(rng.integers(1, 40, size=X.nnz)).astype(np.float32)  # non-integral -> f32
    adata = ad.AnnData(X=X)
    out = tmp_path / "f32"
    write_archive(adata, out, n_shards=4, codec="v2-byte-filter")

    # archive carries the rawdata marker (float data on disk). This used to glob
    # shard_*.shx in a directory; packed keeps the same check by seeking to each shard's
    # base -- parse_shx_header reads from the handle's CURRENT position (v2/format.py:329),
    # and _pack_directory is a pure byte concat, so the shard header bytes are unchanged.
    paths, bases = shard_paths_and_bases(out)
    assert len(paths) == 4
    for path, base in zip(paths, bases, strict=True):
        with open(path, "rb") as fh:
            fh.seek(base)
            assert parse_shx_header(fh).shuffle == "byteshuffle-bytedelta-rawdata"

    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    if not rust_available():
        pytest.skip("Rust core disabled; nothing to compare against the Python path")
    ru = ShardedArchive(out).to_anndata(data_dtype="float32")  # Rust path (gate engages)
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    py = ShardedArchive(out).to_anndata(data_dtype="float32")  # Python path

    np.testing.assert_array_equal(ru.X.toarray(), py.X.toarray())
    np.testing.assert_array_equal(ru.X.toarray(), adata.X.toarray())  # no grouping -> source order


def test_decode_v2_csr_smoke(tmp_path):
    rust = pytest.importorskip("cellstream.v2._rust")

    out, adata = _write_small_uint16_archive(tmp_path)
    m = archive_manifest(out)
    n_shards = m["n_shards"]
    paths, bases = shard_paths_and_bases(out)
    nnz_per_shard = list(map(int, m["nnz_per_shard"]))  # realized nnz per shard (full read)
    row_offsets = m["row_offsets"]
    row_lens = [row_offsets[i + 1] - row_offsets[i] for i in range(n_shards)]
    total_nnz = int(m["total_nnz"])
    n_obs = int(m["n_obs_total"])

    out_data = np.empty(total_nnz, dtype=np.float32)
    out_idx = np.empty(total_nnz, dtype=np.int32)
    out_indptr = np.zeros(n_obs + 1, dtype=np.int64)
    rust.decode_v2_csr(
        paths,
        bases,
        nnz_per_shard,  # out_lens (full read: realized nnz == full nnz)
        row_lens,  # out rows per shard
        row_lens,  # shard_rows (full n_rows per shard)
        nnz_per_shard,  # shard_nnz (full nnz per shard)
        [0] * n_shards,  # sel_starts
        list(row_lens),  # sel_stops (whole shard)
        np.empty(0, dtype=np.int64),  # row_idx_flat (range mode)
        out_data,
        out_idx,
        out_indptr,
        n_vars=int(m["n_vars"]),
        row_gather=False,
        n_threads=2,
    )
    import scipy.sparse as sp

    X = sp.csr_matrix((out_data, out_idx, out_indptr), shape=(n_obs, m["n_vars"]))
    np.testing.assert_array_equal(X.toarray(), adata.X.toarray().astype(np.float32))


def _read_both_modes(out, monkeypatch_modes, tmp_path):
    """Read out via to_anndata with Rust enabled and disabled; return both X."""
    import subprocess
    import sys

    script = (
        "import numpy as np, scipy.sparse as sp;"
        "from cellstream import ShardedArchive;"
        f"ad=ShardedArchive(r'{out}').to_anndata();"
        "x=ad.X.tocsr() if sp.issparse(ad.X) else ad.X;"
        "import numpy as np;"
        "np.save(r'%s', x.toarray() if sp.issparse(x) else np.asarray(x))"
    )
    results = {}
    for mode, disable in (("rust", "0"), ("py", "1")):
        npy = tmp_path / f"{mode}.npy"
        import os

        e = dict(os.environ)
        if disable == "1":
            e["CELLSTREAM_DISABLE_RUST"] = "1"
        else:
            e.pop("CELLSTREAM_DISABLE_RUST", None)
        subprocess.run([sys.executable, "-c", script % str(npy)], check=True, env=e)
        results[mode] = np.load(npy)
    return results["rust"], results["py"]


def test_rust_vs_python_full_read_identical(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    out, adata = _write_small_uint16_archive(tmp_path)
    rust_X, py_X = _read_both_modes(out, None, tmp_path)
    np.testing.assert_array_equal(rust_X, py_X)
    np.testing.assert_array_equal(rust_X, adata.X.toarray().astype(np.float32))


def test_to_anndata_heap_no_bundle(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream.v2 import _rust_dispatch

    # The heap (bundle=None) path is Rust-only; with the Rust path disabled the
    # Python reader returns a shm bundle, so this assertion does not apply.
    if not _rust_dispatch.rust_available():
        pytest.skip("Rust read path disabled (CELLSTREAM_DISABLE_RUST); heap path is Rust-only")
    from cellstream import ShardedArchive

    out, adata = _write_small_uint16_archive(tmp_path)
    got = ShardedArchive(out).to_anndata()  # default output_backing="heap"
    np.testing.assert_array_equal(got.X.toarray(), adata.X.toarray().astype(np.float32))
    # heap arrays are plain numpy — no shm bundle attribute attached.
    assert not hasattr(got, "_cellstream_shm_bundle") or got._cellstream_shm_bundle is None


def test_output_backing_validation(tmp_path):
    from cellstream import ShardedArchive

    out, _ = _write_small_uint16_archive(tmp_path)
    with pytest.raises(ValueError, match="output_backing"):
        ShardedArchive(out).to_anndata(output_backing="bogus")


def test_output_backing_heap_default(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream import ShardedArchive

    out, adata = _write_small_uint16_archive(tmp_path)
    got = ShardedArchive(out).to_anndata(output_backing="heap")
    np.testing.assert_array_equal(got.X.toarray(), adata.X.toarray().astype(np.float32))


def test_output_backing_shm_matches_heap(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream.v2.materialize import resolve_plan

    out, _ = _write_small_uint16_archive(tmp_path)
    plan = resolve_plan(
        archive_manifest(out),
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    d_heap, i_heap, p_heap = read_v2_host(out, plan=plan, output_backing="heap")
    res = read_v2_host(out, plan=plan, output_backing="shm", _return_shm_bundle=True)
    d_shm, i_shm, p_shm, bundle = res
    try:
        np.testing.assert_array_equal(d_heap, d_shm)
        np.testing.assert_array_equal(i_heap, i_shm)
        np.testing.assert_array_equal(p_heap, p_shm)
    finally:
        bundle.close_and_unlink()


def test_rust_vs_python_dense_identical(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream import ShardedArchive

    out, adata = _write_small_uint16_archive(tmp_path)
    dense = ShardedArchive(out).to_anndata(container="dense").X
    np.testing.assert_array_equal(np.asarray(dense), adata.X.toarray().astype(np.float32))


def test_rust_vs_python_contiguous_subset(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream import ShardedArchive

    out, adata = _write_small_uint16_archive(tmp_path)
    ar = ShardedArchive(out)
    sub = ar[5:23].to_anndata()  # spans shard boundaries (10 rows/shard)
    expected = adata.X.toarray().astype(np.float32)[5:23]
    np.testing.assert_array_equal(sub.X.toarray(), expected)


def test_rust_vs_python_row_mask(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    import scipy.sparse as sp

    from cellstream import ShardedArchive
    from cellstream.v2.materialize import resolve_plan

    out, adata = _write_small_uint16_archive(tmp_path)
    ar = ShardedArchive(out)
    plan = resolve_plan(
        ar.manifest,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    mask = np.zeros(ar.manifest["n_obs_total"], dtype=bool)
    mask[::3] = True
    d, i, p = read_v2_host(out, plan=plan, row_mask=mask)
    X = sp.csr_matrix((d, i, p), shape=(int(mask.sum()), ar.manifest["n_vars"]))
    np.testing.assert_array_equal(X.toarray(), adata.X.toarray().astype(np.float32)[mask])


def test_rust_vs_python_row_indices(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    import scipy.sparse as sp

    from cellstream import ShardedArchive
    from cellstream.v2.materialize import resolve_plan

    out, adata = _write_small_uint16_archive(tmp_path)
    ar = ShardedArchive(out)
    plan = resolve_plan(
        ar.manifest,
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    sel = np.array([2, 30, 5, 11], dtype=np.int64)  # unsorted; reader sorts internally
    d, i, p = read_v2_host(out, plan=plan, row_indices=sel)
    X = sp.csr_matrix((d, i, p), shape=(len(sel), ar.manifest["n_vars"]))
    expected = adata.X.toarray().astype(np.float32)[np.sort(sel)]
    np.testing.assert_array_equal(X.toarray(), expected)


def test_rust_vs_python_dense_contiguous_subset(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream import ShardedArchive

    out, adata = _write_small_uint16_archive(tmp_path)
    ar = ShardedArchive(out)
    sub = ar[5:23].to_anndata(container="dense").X  # spans shard boundaries
    expected = adata.X.toarray().astype(np.float32)[5:23]
    np.testing.assert_array_equal(np.asarray(sub), expected)


def test_rust_vs_python_dense_row_mask(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream import ShardedArchive
    from cellstream.v2.materialize import resolve_plan

    out, adata = _write_small_uint16_archive(tmp_path)
    ar = ShardedArchive(out)
    plan = resolve_plan(
        ar.manifest,
        container="dense",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    mask = np.zeros(ar.manifest["n_obs_total"], dtype=bool)
    mask[::4] = True
    dense = read_v2_host(out, plan=plan, row_mask=mask)
    np.testing.assert_array_equal(np.asarray(dense), adata.X.toarray().astype(np.float32)[mask])


def test_rust_vs_python_read_group(tmp_path):
    """read_group(label) resolves to a contiguous [row_start, row_stop) view, so it
    routes through the same contiguous-subset Rust path as a slice read. Confirms the
    Rust core and the Python fallback decode that group identically.

    Builds its own grouped archive. It used to copy tests/v2/golden/grouped_v2bf, which
    #154 part B3 removed -- and those frozen bytes were never this test's subject: it
    compares TWO READS OF THE SAME archive, so any grouped archive serves equally. Packed
    now, because nothing here inspects loose members; the label is known by construction
    rather than parsed back out of a loose groups.json.
    """
    pytest.importorskip("cellstream.v2._rust")
    import os
    import subprocess
    import sys

    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    from cellstream import ShardedArchive
    from tests._archive_helpers import write_archive

    label = "geneA"
    labels = [label] * 40 + ["geneB"] * 40 + ["non-targeting"] * 40
    X = sp.random(len(labels), 30, density=0.3, format="csr", random_state=13).astype(np.float32)
    X.data = np.floor(X.data * 50).astype(np.float32) + 1.0  # integral, in-range
    obs = pd.DataFrame({"gene": labels}, index=[f"c{i}" for i in range(len(labels))])
    dst = tmp_path / "grouped"
    write_archive(
        ad.AnnData(X=X, obs=obs),
        dst,
        group_by="gene",
        reference="non-targeting",
        target_shard_bytes=5_000,
    )

    # Rust (default) read of the group.
    x_rust = ShardedArchive(dst).read_group(label).X
    g_rust = x_rust.toarray() if sp.issparse(x_rust) else np.asarray(x_rust)

    # Python-fallback read of the same group, in a subprocess with Rust disabled.
    npy = tmp_path / "py.npy"
    script = (
        "import numpy as np, scipy.sparse as sp;"
        "from cellstream import ShardedArchive;"
        f"X=ShardedArchive(r'{dst}').read_group(r'{label}').X;"
        "arr=X.toarray() if sp.issparse(X) else np.asarray(X);"
        f"np.save(r'{npy}', arr)"
    )
    env = dict(os.environ)
    env["CELLSTREAM_DISABLE_RUST"] = "1"
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
    g_py = np.load(npy)
    np.testing.assert_array_equal(g_rust, g_py)
    # Guard against both sides trivially agreeing on the WHOLE archive: read_group must
    # return only its own 40 rows, not all 120.
    assert g_rust.shape[0] == 40, f"read_group returned {g_rust.shape[0]} rows, expected 40"


def _to_anndata_subprocess(out, kw, tmp_path, *, disable_rust):
    """Read out.to_anndata(**kw) in a subprocess (optionally Rust-disabled) and
    return X.toarray() (CSR) or the dense ndarray."""
    import json
    import os
    import subprocess
    import sys

    npy = tmp_path / f"sub_{'py' if disable_rust else 'rust'}.npy"
    kw_json = json.dumps(kw)
    script = (
        "import json, numpy as np, scipy.sparse as sp;"
        "from cellstream import ShardedArchive;"
        f"kw=json.loads(r'''{kw_json}''');"
        f"X=ShardedArchive(r'{out}').to_anndata(**kw).X;"
        "arr=X.toarray() if sp.issparse(X) else np.asarray(X);"
        f"np.save(r'{npy}', arr)"
    )
    e = dict(os.environ)
    if disable_rust:
        e["CELLSTREAM_DISABLE_RUST"] = "1"
    else:
        e.pop("CELLSTREAM_DISABLE_RUST", None)
    subprocess.run([sys.executable, "-c", script], check=True, env=e)
    return np.load(npy)


@pytest.mark.parametrize("data_dtype", ["float64", "float32", "int32", "uint16"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64", "uint16", "uint32"])
def test_rust_vs_python_explicit_dtypes(tmp_path, data_dtype, index_dtype):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream import ShardedArchive

    out, _ = _write_small_uint16_archive(tmp_path)
    kw = dict(data_dtype=data_dtype, index_dtype=index_dtype)
    rust_X = ShardedArchive(out).to_anndata(**kw).X
    py_X = _to_anndata_subprocess(out, kw, tmp_path, disable_rust=True)
    np.testing.assert_array_equal(rust_X.toarray(), py_X)
    assert rust_X.data.dtype == np.dtype(data_dtype)
    assert rust_X.indices.dtype == np.dtype(index_dtype)


def _write_lossy_source_archive(tmp_path):
    """float64 source with NON-integral values -> stays float64 on disk; casting to a
    narrower/int dtype is lossy."""
    import anndata as ad
    import scipy.sparse as sp

    X = sp.random(30, 20, density=0.4, format="csr", random_state=3).astype(np.float64)
    X.data = X.data + 0.5  # non-integral -> no u16 narrow; float64 on disk
    out = tmp_path / "f64"
    write_archive(ad.AnnData(X=X), out, n_shards=3)
    return out, X


def test_lossy_default_raises_both_modes(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream import ShardedArchive
    from cellstream.errors import LossyCastError

    out, _ = _write_lossy_source_archive(tmp_path)
    with pytest.raises(LossyCastError):
        ShardedArchive(out).to_anndata(data_dtype="int32")  # float64->int32 lossy


def test_lossy_allow_matches_numpy(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    from cellstream import ShardedArchive

    out, X = _write_lossy_source_archive(tmp_path)
    got = ShardedArchive(out).to_anndata(data_dtype="float32", allow_lossy=True)
    np.testing.assert_array_equal(got.X.toarray(), X.toarray().astype(np.float32))


def _csr_decode_args(out):
    """Build the positional decode_v2_csr args for a full read of `out` (either container).

    ``bases`` is what makes this container-agnostic: 0 for a directory, the shard's
    4 KB-aligned offset for packed. The two corruption tests below therefore tamper AT
    ``bases[0]``, not at file offset 0 -- on packed, offset 0 is the SHPK head block, so a
    hardcoded seek(0) would clobber the container header instead of the shard magic and
    the test would pass for the wrong reason.
    """
    m = archive_manifest(out)
    n_shards = m["n_shards"]
    paths, bases = shard_paths_and_bases(out)
    nnz_per_shard = list(map(int, m["nnz_per_shard"]))
    row_offsets = m["row_offsets"]
    row_lens = [row_offsets[i + 1] - row_offsets[i] for i in range(n_shards)]
    total_nnz = int(m["total_nnz"])
    n_obs = int(m["n_obs_total"])
    out_data = np.empty(total_nnz, dtype=np.float32)
    out_idx = np.empty(total_nnz, dtype=np.int32)
    out_indptr = np.zeros(n_obs + 1, dtype=np.int64)
    return dict(
        paths=paths,
        bases=bases,
        out_lens=nnz_per_shard,
        row_lens=row_lens,
        shard_rows=row_lens,
        shard_nnz=nnz_per_shard,
        sel_starts=[0] * n_shards,
        sel_stops=list(row_lens),
        row_idx_flat=np.empty(0, dtype=np.int64),
        out_data=out_data,
        out_indices=out_idx,
        out_indptr=out_indptr,
        n_vars=int(m["n_vars"]),
    )


def test_corrupt_bad_magic_raises_value_error_not_lossy(tmp_path):
    """A shard with a clobbered SHX magic -> clean ValueError (#133): not a
    pyo3_runtime.PanicException, and not remapped to LossyCastError."""
    pytest.importorskip("cellstream.v2._rust")
    from cellstream.errors import LossyCastError
    from cellstream.v2 import _rust_dispatch

    out, _ = _write_small_uint16_archive(tmp_path)
    a = _csr_decode_args(out)
    with open(a["paths"][0], "r+b") as fh:  # clobber the 4-byte magic of shard 0
        fh.seek(a["bases"][0])
        fh.write(b"XXXX")
    with pytest.raises(ValueError) as ei:  # ValueError, never PanicException
        _rust_dispatch.decode_v2_csr(**a, n_threads=2)
    assert not isinstance(ei.value, LossyCastError)  # Corrupt must NOT remap to lossy


def test_truncated_shard_raises_os_error(tmp_path):
    """A truncated shard file -> OSError family (read_exact EOF), never a
    pyo3_runtime.PanicException (#133)."""
    rust = pytest.importorskip("cellstream.v2._rust")

    out, _ = _write_small_uint16_archive(tmp_path)
    a = _csr_decode_args(out)
    with open(a["paths"][0], "r+b") as fh:  # truncate shard 0 below its 10-byte prefix
        fh.truncate(a["bases"][0] + 4)
    with pytest.raises(OSError):  # OSError, never PanicException
        rust.decode_v2_csr(
            a["paths"],
            a["bases"],
            a["out_lens"],
            a["row_lens"],
            a["shard_rows"],
            a["shard_nnz"],
            a["sel_starts"],
            a["sel_stops"],
            a["row_idx_flat"],
            a["out_data"],
            a["out_indices"],
            a["out_indptr"],
            n_vars=a["n_vars"],
            row_gather=False,
            n_threads=2,
        )
