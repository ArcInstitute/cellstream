"""pyfastpfor (the ``cell`` extra) must be optional whenever the Rust engine does the work.

``pyfastpfor`` ships NO linux wheel -- pip gets an sdist and compiles C++ -- so requiring it
for the default ``layout='cell'`` write makes a plain ``pip install cellstream`` unable to write
the format the package exists for. Measured before this fix, in a pyfastpfor-free venv with the
Rust extension present and working: ``codec='pfordelta'`` and ``codec=None`` both raised
ImportError at WRITE, and an archive written elsewhere raised at ``cellstream.open``.

It is not needed on that path. Probed with a stub module whose codec object raises on any use:
with Rust active, write + gather + bulk read all completed bit-exact and the stub was never
touched; under ``CELLSTREAM_DISABLE_RUST=1`` it raised on ``encodeArray``/``decodeArray``
immediately. So the import is eager, not necessary.

The two halves below are equally load-bearing. The Rust half asserts the extra is optional; the
pure-Python half asserts the actionable error survives for the case where it genuinely IS
required -- an "optional" that silently degrades to a ModuleNotFoundError deep in a decode
would be worse than the eager guard it replaced.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.sparse as sp

import cellstream
from cellstream.cell._rust_dispatch import rust_available
from cellstream.write import write_sharded

from .conftest import make_cell_adata

requires_rust = pytest.mark.skipif(
    not rust_available(), reason="the extra is genuinely required without the Rust engine"
)


def _have_pyfastpfor() -> bool:
    try:
        import pyfastpfor  # noqa: F401
    except ImportError:
        return False
    return True


# The module is exempt from conftest.py's directory-wide pyfastpfor skip, so a base install runs
# it -- which is the point. But two tests below introspect the PURE-PYTHON decoder itself, so they
# genuinely need the package and must carry their own marker; without it the exemption made them
# fail in exactly the configuration it was added to support. (codex, checkpoint 2 round 2.)
requires_pyfastpfor = pytest.mark.skipif(
    not _have_pyfastpfor(), reason="introspects the pure-Python pfordelta decoder"
)


@pytest.fixture
def no_pyfastpfor(monkeypatch):
    """Make ``import pyfastpfor`` raise ImportError for the duration of the test.

    ``sys.modules[name] = None`` is the documented way to force an import failure without
    touching the filesystem; the import machinery raises ImportError on the None entry. The
    guards under test catch ImportError, so this reproduces a slim install exactly.
    """
    monkeypatch.setitem(sys.modules, "pyfastpfor", None)
    with pytest.raises(ImportError):
        import pyfastpfor  # noqa: F401
    return None


def _counts(n_obs=120, n_vars=60):
    a = make_cell_adata(n_obs=n_obs, n_vars=n_vars)
    return a


@requires_rust
def test_write_pfordelta_without_pyfastpfor(tmp_path, no_pyfastpfor):
    a = _counts()
    p = tmp_path / "explicit.csad"
    write_sharded(a, p, layout="cell", codec="pfordelta", overwrite=True)
    store = cellstream.open(p)
    try:
        assert store.manifest["codec"] == "pfordelta"
        assert store.n_obs == a.n_obs
    finally:
        store.close()


@requires_rust
def test_default_codec_write_without_pyfastpfor(tmp_path, no_pyfastpfor):
    """codec=None on integer counts plans pfordelta first -- the out-of-the-box path."""
    a = _counts()
    p = tmp_path / "default.csad"
    write_sharded(a, p, layout="cell", overwrite=True)
    store = cellstream.open(p)
    try:
        assert store.manifest["codec"] == "pfordelta"
    finally:
        store.close()


@requires_rust
def test_open_and_gather_pfordelta_without_pyfastpfor(tmp_path, monkeypatch):
    """Write with the extra present, then read it back with the extra gone."""
    a = _counts()
    p = tmp_path / "read.csad"
    write_sharded(a, p, layout="cell", codec="pfordelta", overwrite=True)
    want = cellstream.open(p)
    try:
        expected = want.gather_rows(np.arange(want.n_obs))
    finally:
        want.close()

    monkeypatch.setitem(sys.modules, "pyfastpfor", None)
    store = cellstream.open(p)
    try:
        got = store.gather_rows(np.arange(store.n_obs))
    finally:
        store.close()
    assert abs(got - expected).nnz == 0


@requires_rust
def test_bulk_read_pfordelta_without_pyfastpfor(tmp_path, monkeypatch):
    a = _counts()
    p = tmp_path / "bulk.csad"
    write_sharded(a, p, layout="cell", codec="pfordelta", overwrite=True)
    expected = cellstream.read_h5ad(p).X
    monkeypatch.setitem(sys.modules, "pyfastpfor", None)
    got = cellstream.read_h5ad(p).X
    assert abs(sp.csr_matrix(got) - sp.csr_matrix(expected)).nnz == 0


@requires_rust
def test_push_writer_pfordelta_without_pyfastpfor(tmp_path, no_pyfastpfor):
    from cellstream.cell.archive_writer import CellArchiveWriter

    a = _counts()
    p = tmp_path / "push.csad"
    w = CellArchiveWriter(p, n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16")
    w.add_block(sp.csr_matrix(a.X[:60]))
    w.add_block(sp.csr_matrix(a.X[60:]))
    w.finalize(obs=a.obs, var=a.var)
    store = cellstream.open(p)
    try:
        assert store.n_obs == a.n_obs
        assert store.manifest["codec"] == "pfordelta"
    finally:
        store.close()


def test_zstd_never_needed_pyfastpfor(tmp_path, no_pyfastpfor):
    """Regression guard: zstd was already independent of the extra, on either engine."""
    a = _counts()
    p = tmp_path / "zstd.csad"
    write_sharded(a, p, layout="cell", codec="zstd", overwrite=True)
    store = cellstream.open(p)
    try:
        assert store.manifest["codec"] == "zstd"
        assert store.gather_rows(np.arange(store.n_obs)).nnz > 0
    finally:
        store.close()


# --------------------------------------------------------------- the error must survive
# Without Rust the pure-Python codec really does call pyfastpfor, so the actionable message
# must still be what the user sees. `match=` pins both halves of it: what is missing, and how
# to install it. A bare ModuleNotFoundError from deep inside a decode would pass a weaker test.

_MSG = r"requires the 'pyfastpfor' package.*cellstream\[cell\]"


def test_write_without_rust_or_pyfastpfor_says_how_to_fix(tmp_path, monkeypatch, no_pyfastpfor):
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    with pytest.raises(ImportError, match=_MSG):
        write_sharded(_counts(), tmp_path / "x.csad", layout="cell", codec="pfordelta")


def test_gather_without_rust_or_pyfastpfor_says_how_to_fix(tmp_path, monkeypatch):
    a = _counts()
    p = tmp_path / "y.csad"
    write_sharded(a, p, layout="cell", codec="pfordelta", overwrite=True)
    monkeypatch.setitem(sys.modules, "pyfastpfor", None)
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    store = cellstream.open(p)  # opening must not need the extra either
    try:
        with pytest.raises(ImportError, match=_MSG):
            store.gather_rows(np.arange(4))
    finally:
        store.close()


def test_push_writer_without_rust_or_pyfastpfor_says_how_to_fix(
    tmp_path, monkeypatch, no_pyfastpfor
):
    from cellstream.cell.archive_writer import CellArchiveWriter

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    a = _counts()
    with pytest.raises(ImportError, match=_MSG):
        w = CellArchiveWriter(
            tmp_path / "z.csad", n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16"
        )
        w.add_block(sp.csr_matrix(a.X[:60]))
        w.finalize(obs=a.obs[:60], var=a.var)


# ------------------------------------------------- the lazy build must be observationally eager
# Deferring construction is the change; changing what the decoder is constructed FROM is not.
# Without the open-time n_vars snapshot, a store whose n_vars is reassigned before the first
# gather builds a decoder bound to the NEW value, which moves an out-of-range column's rejection
# from the CSR assembly's bounds scan to the pfordelta frame check -- a different error for a
# tampered archive. test_fallback_equivalence.py::test_fallback_rejects_out_of_range_column is
# what caught it; these pin the property directly rather than through that symptom.


@requires_pyfastpfor
def test_decoder_is_not_built_at_open_and_is_cached(tmp_path):
    a = _counts()
    p = tmp_path / "lazy.csad"
    write_sharded(a, p, layout="cell", codec="pfordelta", overwrite=True)
    store = cellstream.open(p)
    try:
        assert store._decoder_cache is None, "opening must not build the pure-Python decoder"
        first = store._decoder
        assert store._decoder_cache is first
        assert store._decoder is first, "the decoder owns reused scratch; it must be cached"
    finally:
        store.close()


@requires_pyfastpfor
def test_decoder_binds_open_time_n_vars(tmp_path):
    a = _counts()
    p = tmp_path / "snap.csad"
    write_sharded(a, p, layout="cell", codec="pfordelta", overwrite=True)
    store = cellstream.open(p)
    try:
        real = store.n_vars
        store.n_vars = 2  # a caller mutates it after open (only tests do)
        assert store._decoder._n_vars == real, (
            "the lazily built decoder must use n_vars as of open, not the mutated value"
        )
    finally:
        store.close()
