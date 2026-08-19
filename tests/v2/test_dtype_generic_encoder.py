import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.read import ShardedArchive
from cellstream.v2.writer import write_v2_archive
from tests._archive_helpers import pack_directory

# #154 part B3a, route D: write_v2_archive is the SUBJECT here -- the whole file is about what
# the dtype-generic encoder puts on disk -- so these calls must NOT become write_archive. The
# five round-trip tests pack_directory() the writer's output and read the packed copy through
# the public reader; _pack_directory is a pure byte concat (packed/writer.py:111), so the
# encoded streams under test are unchanged. The two manifest-only tests need no public read at
# all and stay entirely on the directory.


def _adata(X):
    n = X.shape[0]
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"target_gene": [f"g{i % 3}" for i in range(n)]},
            index=[f"c{i}" for i in range(n)],
        ),
    )


def test_manifest_records_float_dtype(tmp_path):
    X = sp.random(40, 9, density=0.5, format="csr", random_state=5).astype(np.float64)
    X.data = X.data - 0.5
    out = tmp_path / "arc"
    write_v2_archive(_adata(X), out, group_by="target_gene", codec="v2-byte-filter")
    m = json.loads((out / "manifest.json").read_text())
    assert m["x_data_dtype_on_disk"] == "float64"
    assert m["x_indices_dtype_on_disk"] == "uint32"
    assert m["x_data_dtype_min"] == "float64"


def test_manifest_integer_data_is_uint32_with_min(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    X = sp.random(40, 9, density=0.5, format="csr", random_state=6)
    X.data = np.ceil(X.data * 30)  # integer-valued, max ~30 -> min uint8
    out = tmp_path / "arc"
    write_v2_archive(_adata(X), out, group_by="target_gene", codec="v2-byte-filter")
    m = json.loads((out / "manifest.json").read_text())
    assert m["x_data_dtype_on_disk"] == "uint32"
    assert m["x_indices_dtype_on_disk"] == "uint32"
    assert m["x_data_dtype_min"] == "uint8"
    assert m["x_indices_dtype_min"] == "uint16"


def test_integer_roundtrip_cast_back_false_is_uint32(tmp_path):
    pytest.importorskip("cellstream.v2._rust")
    X = sp.random(50, 9, density=0.6, format="csr", random_state=12)
    X.data = np.ceil(X.data * 30)
    out = tmp_path / "arc"
    write_v2_archive(_adata(X), out, group_by="target_gene", codec="v2-byte-filter")
    # cast_back=False returns the on-disk dtype, now uint32 (was uint16).
    a = ShardedArchive(pack_directory(out, tmp_path / "arc.shad")).to_anndata(cast_back=False)
    Xb = (a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)).tocsr()
    assert Xb.data.dtype == np.uint32
    assert int(Xb.nnz) == int(X.nnz)


def test_float64_roundtrip_castback_false(tmp_path):
    X = sp.random(50, 9, density=0.6, format="csr", random_state=11).astype(np.float64)
    X.data = X.data - 0.5
    out = tmp_path / "arc"
    write_v2_archive(_adata(X), out, group_by="target_gene", codec="v2-byte-filter")
    a = ShardedArchive(pack_directory(out, tmp_path / "arc.shad")).to_anndata(cast_back=False)
    Xb = (a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)).tocsr()
    assert Xb.data.dtype == np.float64
    assert np.isclose(np.sort(Xb.data), np.sort(X.data)).all()


def test_float64_roundtrip_default_castback_no_downcast(tmp_path):
    X = sp.random(50, 9, density=0.6, format="csr", random_state=13).astype(np.float64)
    X.data = X.data - 0.5
    out = tmp_path / "arc"
    write_v2_archive(_adata(X), out, group_by="target_gene", codec="v2-byte-filter")
    a = ShardedArchive(pack_directory(out, tmp_path / "arc.shad")).to_anndata()
    Xb = (a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)).tocsr()
    assert Xb.data.dtype == np.float64
    assert np.isclose(np.sort(Xb.data), np.sort(X.data)).all()


def test_float32_roundtrip(tmp_path):
    X = sp.random(50, 9, density=0.6, format="csr", random_state=14).astype(np.float32)
    X.data = X.data + 0.25
    out = tmp_path / "arc"
    write_v2_archive(_adata(X), out, group_by="target_gene", codec="v2-byte-filter")
    a = ShardedArchive(pack_directory(out, tmp_path / "arc.shad")).to_anndata(cast_back=False)
    Xb = (a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)).tocsr()
    assert Xb.data.dtype == np.float32
    assert np.allclose(np.sort(Xb.data), np.sort(X.data))


def test_fallback_float64_roundtrips_disabled_rust(tmp_path, monkeypatch):
    # Explicitly exercise the Python fallback writing float64 (independent of the
    # outer CELLSTREAM_DISABLE_RUST env) - must round-trip via the dtype-aware reader.
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    import importlib

    from cellstream.v2 import _rust_dispatch

    importlib.reload(_rust_dispatch)
    try:
        X = sp.random(40, 8, density=0.5, format="csr", random_state=20).astype(np.float64)
        X.data = X.data - 0.5
        out = tmp_path / "arc"
        write_v2_archive(_adata(X), out, group_by="target_gene", codec="v2-byte-filter")
        from cellstream.read import ShardedArchive

        a = ShardedArchive(pack_directory(out, tmp_path / "arc.shad")).to_anndata(cast_back=False)
        Xb = (a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)).tocsr()
        assert Xb.data.dtype == np.float64
        assert np.isclose(np.sort(Xb.data), np.sort(X.data)).all()
    finally:
        monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
        importlib.reload(_rust_dispatch)
