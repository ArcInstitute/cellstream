"""Fixtures for cell-layout tests."""

from __future__ import annotations

import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.cell import format as fmt
from cellstream.cell.reader import open_cell
from cellstream.packed.reader import PackedArchive


def pytest_collection_modifyitems(config, items):
    """Skip cell-layout tests when the optional 'cell' extra (pyfastpfor) is absent.

    Mirrors the tests/v2/gpu/conftest.py pattern for optional dependencies: CI
    installs ``.[test,cell]`` so these run, but a base ``.[test]`` install skips
    them cleanly instead of erroring on the writer/reader import guard.
    """
    try:
        import pyfastpfor  # noqa: F401
    except ImportError as e:
        skip = pytest.mark.skip(reason=f"cell extra not installed (pyfastpfor): {e}")
        for item in items:
            path = str(item.fspath)
            name = os.path.basename(path)
            # test_pyfastpfor_optional.py is EXEMPT: it exists to prove the cell layout works
            # WITHOUT the extra, so skipping it exactly when the extra is missing means a base
            # install never exercises the configuration it was written for. Its own
            # `requires_rust` marker handles the case where neither engine can do the work.
            # (codex, checkpoint 2.)
            # Exact basename, not a substring: "test_pyfastpfor_optional" not in path would also
            # exempt a future test_pyfastpfor_optional_extra.py, silently un-skipping a module
            # nobody wrote that exemption for. (Copilot, suppressed comment.)
            if "tests/cell" in path and name != "test_pyfastpfor_optional.py":
                item.add_marker(skip)


def make_cell_adata(seed=7, n_obs=240, n_vars=60, annotations=False):
    """Small uint16 CSR AnnData with a target_gene grouping col and n_umi."""
    rng = np.random.default_rng(seed)
    nnz = 4000
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = rng.integers(1, 50, size=nnz, dtype=np.uint16)
    csr = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    csr.sum_duplicates()
    x = sp.csr_matrix(csr.shape, dtype=np.uint16)
    x.data = csr.data.astype(np.uint16)
    x.indices = csr.indices.astype(np.int32)
    x.indptr = csr.indptr.astype(np.int64)
    genes = [f"g{i:03d}" for i in range(n_vars)]
    labels = rng.choice([f"t{i:02d}" for i in range(8)], size=n_obs)
    obs = {"target_gene": labels, "n_umi": rng.integers(1, 1000, size=n_obs)}
    a = ad.AnnData(X=x, obs=obs)
    a.var_names = genes
    if annotations:
        a.obs_names = [f"cell{i}" for i in range(n_obs)]
        a.obsm["X_pca"] = rng.standard_normal((n_obs, 10)).astype(np.float32)
        a.varm["PCs"] = rng.standard_normal((n_vars, 5)).astype(np.float32)
        a.obsp["connectivities"] = sp.random(
            n_obs, n_obs, density=0.05, random_state=seed + 1, format="csr", dtype=np.float32
        )
        a.obsp["distances"] = sp.random(
            n_obs, n_obs, density=0.05, random_state=seed + 2, format="csr", dtype=np.float32
        )
        a.varp["corr"] = sp.random(
            n_vars, n_vars, density=0.1, random_state=seed + 3, format="csr", dtype=np.float32
        )
        a.uns["method"] = "pca"
    return a


@pytest.fixture
def cell_adata():
    return make_cell_adata()


@pytest.fixture(params=[None, "1"])
def engine_env(request, monkeypatch):
    """Run a test under both engines: default (Rust, when available) and
    CELLSTREAM_DISABLE_RUST=1 (pure-Python fallback). Declaring `engine_env` as a test
    parameter is enough -- the fixture itself sets the env var; the test body never
    reads the value. Mirrors the ad hoc `@pytest.mark.parametrize("engine_env", [None,
    "1"])` + `monkeypatch.setenv(...)` pattern already used in test_errors.py, as a
    reusable fixture (#229)."""
    if request.param:
        monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", request.param)
    else:
        # actively CLEAR it on the default variant, so this param exercises the Rust path even when
        # the ambient env already has CELLSTREAM_DISABLE_RUST=1 (else "both engines" would silently be
        # python twice). rust_available() reads the env at call time (Copilot #232).
        monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
        # ...but on a checkout with NO extension built, clearing it lands this variant in
        # check_cell_python_fallback's `extension_missing` arm, which RAISES (#244). The suite
        # is testing the Python engine there, not asking for a fast read -- so opt in explicitly.
        from cellstream.cell._rust_dispatch import _HAVE

        if not _HAVE:
            monkeypatch.setenv("CELLSTREAM_ALLOW_PYTHON_FALLBACK", "1")
    return request.param


def _assert_matrix_equal(x, y, name):
    if isinstance(x, pd.DataFrame) or isinstance(y, pd.DataFrame):
        # DataFrame obsm: compare labels too (np.asarray would drop column/index names)
        pd.testing.assert_frame_equal(x, y, obj=name)
        return
    xa = x.toarray() if sp.issparse(x) else np.asarray(x)
    ya = y.toarray() if sp.issparse(y) else np.asarray(y)
    np.testing.assert_array_equal(xa, ya, err_msg=f"{name} differs")


def assert_archives_equivalent(a, b):
    """Assert two cell .shad archives are byte-identical in every way that matters.

    NOT a whole-file compare: manifest.json embeds `created_at` (writer.py), so two writes of
    the same input that straddle a second boundary produce different FILES with identical
    PAYLOADS. A whole-file filecmp would fail spuriously -- and if written as skip-on-mismatch,
    would silently stop testing anything (Gemini #220 r2).

    Compares: manifest.json minus `created_at`; x_offsets; x_indptr; x_payload bytes; obs/var
    (pandas equality); uns (dict equality); groups.json when present on either side (dict
    equality -- present on only one side is itself a failure). obs/var/uns/groups.json are
    compared via their LOADED VALUES, not raw bytes -- that IS the contract that matters, not
    incidental parquet/hdf5 encoding.

    obs/var/uns/groups.json used to be entirely unchecked here -- this gate never read
    obs.parquet or header.h5ad at all, which is why CellArchiveSource.uns unconditionally
    returning {} (PR4 review finding 1) sailed through every existing test undetected
    (PR4 review finding 2).
    """
    ma = {k: v for k, v in PackedArchive(a).manifest.items() if k != "created_at"}
    mb = {k: v for k, v in PackedArchive(b).manifest.items() if k != "created_at"}
    assert ma == mb, (
        f"manifest differs:\n{json.dumps(ma, indent=2, sort_keys=True)}\n"
        f"!=\n{json.dumps(mb, indent=2, sort_keys=True)}"
    )

    sa, sb = open_cell(a), open_cell(b)
    pa = pb = None
    try:
        np.testing.assert_array_equal(sa.offsets, sb.offsets, err_msg="x_offsets differ")
        np.testing.assert_array_equal(sa.indptr, sb.indptr, err_msg="x_indptr differ")

        pd.testing.assert_frame_equal(sa.obs, sb.obs)
        pd.testing.assert_frame_equal(sa.var, sb.var)
        assert sa.uns == sb.uns, f"uns differs:\n{sa.uns!r}\n!=\n{sb.uns!r}"

        for attr in ("obsm", "varm", "obsp", "varp"):
            da, db = getattr(sa, attr), getattr(sb, attr)
            assert set(da) == set(db), f"{attr} keys differ: {set(da)} != {set(db)}"
            for k in da:
                _assert_matrix_equal(da[k], db[k], f"{attr}[{k!r}]")

        has_ga = sa._packed.has_member(fmt.GROUPS)
        has_gb = sb._packed.has_member(fmt.GROUPS)
        assert has_ga == has_gb, (
            f"groups.json present on one archive but not the other (a={has_ga}, b={has_gb})"
        )
        if has_ga:
            ga = json.loads(sa._packed.member_bytes(fmt.GROUPS))
            gb = json.loads(sb._packed.member_bytes(fmt.GROUPS))
            assert ga == gb, (
                f"groups.json differs:\n{json.dumps(ga, indent=2, sort_keys=True)}\n"
                f"!=\n{json.dumps(gb, indent=2, sort_keys=True)}"
            )

        # sa._mm IS the payload member (not the whole file); offsets are 0-based into it.
        pa = np.frombuffer(sa._mm, dtype=np.uint8)[sa.offsets[0] : sa.offsets[-1]]
        pb = np.frombuffer(sb._mm, dtype=np.uint8)[sb.offsets[0] : sb.offsets[-1]]
        equal = np.array_equal(pa, pb)
    finally:
        # A live frombuffer view PINS the mmap -> close() raises BufferError("cannot close
        # exported pointers exist"), which would MASK a real assertion failure escaping the try.
        # Drop the views in `finally` BEFORE closing. (PR2 fixed this in _rust_dispatch; Gemini
        # re-found it in a test on #220.)
        del pa, pb
        sa.close()
        sb.close()
    assert equal, "x_payload bytes differ"
