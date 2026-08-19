"""Public Rust-runtime visibility (#244 ask #2)."""

import numpy as np
import pytest

import cellstream
from cellstream import runtime as rt

STATUS_KEYS = {
    "version",
    "available",
    "extension_importable",
    "disabled_by_env",
    "allow_python_fallback",
    "shard_ready",
    "cell_ready",
    "reason",
}


def test_exports_are_public():
    assert "rust_available" in cellstream.__all__
    assert "rust_status" in cellstream.__all__
    assert cellstream.rust_available is rt.rust_available
    assert cellstream.rust_status is rt.rust_status


def test_status_shape():
    s = cellstream.rust_status()
    assert set(s) == STATUS_KEYS
    assert isinstance(s["available"], bool)
    assert s["version"] == cellstream.__version__
    assert s["available"] == cellstream.rust_available()


def test_available_is_conjunction_of_both_gates():
    s = cellstream.rust_status()
    assert s["available"] == (s["shard_ready"] and s["cell_ready"])


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    s = cellstream.rust_status()
    assert s["disabled_by_env"] is True
    assert s["available"] is False
    assert "CELLSTREAM_DISABLE_RUST" in s["reason"]


def test_allow_python_fallback_reported(monkeypatch):
    monkeypatch.setenv("CELLSTREAM_ALLOW_PYTHON_FALLBACK", "1")
    assert cellstream.rust_status()["allow_python_fallback"] is True
    monkeypatch.delenv("CELLSTREAM_ALLOW_PYTHON_FALLBACK")
    assert cellstream.rust_status()["allow_python_fallback"] is False


def test_reason_none_when_available(monkeypatch):
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    s = cellstream.rust_status()
    if s["available"]:
        assert s["reason"] is None
    else:
        assert isinstance(s["reason"], str) and s["reason"]


def test_reason_names_missing_extension(monkeypatch):
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    monkeypatch.setattr("cellstream.v2._rust_dispatch._HAVE_RUST", False)
    s = cellstream.rust_status()
    assert s["extension_importable"] is False
    assert s["available"] is False
    assert "not importable" in s["reason"]


def test_reason_names_incomplete_extension(monkeypatch):
    """Extension imports but predates the cell entry points."""
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    monkeypatch.setattr("cellstream.v2._rust_dispatch._HAVE_RUST", True)
    monkeypatch.setattr("cellstream.cell._rust_dispatch._HAVE", False)
    s = cellstream.rust_status()
    if s["shard_ready"]:
        assert s["cell_ready"] is False
        assert s["available"] is False
        assert "cell" in s["reason"]


def test_cli_runtime_line(capsys, monkeypatch):
    from cellstream.cli import main

    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    assert main(["runtime"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    line = out[0]
    assert line.startswith(f"cellstream {cellstream.__version__} | rust=")
    assert " | rust=yes | " in line or " | rust=no | " in line


def test_cli_runtime_reports_disabled(capsys, monkeypatch):
    from cellstream.cli import main

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    assert main(["runtime"]) == 0
    line = capsys.readouterr().out.strip()
    assert " | rust=no | " in line
    assert "CELLSTREAM_DISABLE_RUST" in line


@pytest.fixture
def fresh_warnings():
    from cellstream.cell import _rust_dispatch as rd

    rd._reset_fallback_warnings()
    yield rd
    rd._reset_fallback_warnings()


def test_disabled_by_env_allows_and_warns(monkeypatch, fresh_warnings):
    rd = fresh_warnings
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    with pytest.warns(UserWarning, match="pure-Python"):
        rd.check_cell_python_fallback("t")


def test_disabled_by_env_wins_over_missing_extension(monkeypatch, fresh_warnings):
    """CELLSTREAM_DISABLE_RUST=1 must never raise, even with no extension built -- it drives a
    whole CI job and the engine_env fixture."""
    rd = fresh_warnings
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    monkeypatch.setattr(rd, "_HAVE", False)
    with pytest.warns(UserWarning):
        rd.check_cell_python_fallback("t")


def test_missing_extension_raises(monkeypatch, fresh_warnings):
    rd = fresh_warnings
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    monkeypatch.delenv("CELLSTREAM_ALLOW_PYTHON_FALLBACK", raising=False)
    monkeypatch.setattr(rd, "_HAVE", False)
    with pytest.raises(RuntimeError) as e:
        rd.check_cell_python_fallback("CellStore.gather_rows")
    msg = str(e.value)
    assert "CELLSTREAM_ALLOW_PYTHON_FALLBACK=1" in msg
    assert "cellstream.v2._rust" in msg
    assert "CellStore.gather_rows" in msg
    # _HAVE is `hasattr(decode_cells) and hasattr(encode_cells)`, so it is ALSO false for an
    # extension that imports fine but predates the cell work. The message must not send that
    # user hunting for a build problem they do not have. (Copilot #247 checkpoint 2.)
    assert "decode_cells" in msg and "encode_cells" in msg
    assert "rust_status" in msg


def test_missing_extension_opt_in_allows(monkeypatch, fresh_warnings):
    rd = fresh_warnings
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    monkeypatch.setenv("CELLSTREAM_ALLOW_PYTHON_FALLBACK", "1")
    monkeypatch.setattr(rd, "_HAVE", False)
    with pytest.warns(UserWarning):
        rd.check_cell_python_fallback("t")


def test_gate_declined_never_raises(monkeypatch, fresh_warnings):
    """Extension present but the dtype/codec combo is one Rust cannot produce. Installing a
    wheel would not help, so this must warn, never raise."""
    rd = fresh_warnings
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    monkeypatch.delenv("CELLSTREAM_ALLOW_PYTHON_FALLBACK", raising=False)
    monkeypatch.setattr(rd, "_HAVE", True)
    with pytest.warns(UserWarning):
        rd.check_cell_python_fallback("t")


def test_warns_only_once_per_reason(monkeypatch, fresh_warnings, recwarn):
    rd = fresh_warnings
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    rd.check_cell_python_fallback("t")
    rd.check_cell_python_fallback("t")
    rd.check_cell_python_fallback("t")
    assert len([w for w in recwarn if issubclass(w.category, UserWarning)]) == 1


def _tiny_cell_archive(tmp_path):
    import anndata as ad
    import scipy.sparse as sp

    import cellstream

    X = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 5, 0]], dtype=np.float32))
    a = ad.AnnData(X=X)
    out = tmp_path / "t.shad"
    cellstream.write_sharded(a, out, layout="cell", overwrite=True)
    return out


def test_gather_rows_raises_without_extension(tmp_path, monkeypatch, fresh_warnings):
    from cellstream.cell import _rust_dispatch as rd

    path = _tiny_cell_archive(tmp_path)
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    monkeypatch.delenv("CELLSTREAM_ALLOW_PYTHON_FALLBACK", raising=False)
    monkeypatch.setattr(rd, "_HAVE", False)
    store = cellstream.open(path)
    try:
        with pytest.raises(RuntimeError, match="CELLSTREAM_ALLOW_PYTHON_FALLBACK"):
            store.gather_rows(np.array([0, 2], dtype=np.int64))
    finally:
        store.close()


def test_gather_rows_allowed_with_opt_in(tmp_path, monkeypatch, fresh_warnings):
    from cellstream.cell import _rust_dispatch as rd

    path = _tiny_cell_archive(tmp_path)
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    monkeypatch.setenv("CELLSTREAM_ALLOW_PYTHON_FALLBACK", "1")
    monkeypatch.setattr(rd, "_HAVE", False)
    store = cellstream.open(path)
    try:
        with pytest.warns(UserWarning):
            X = store.gather_rows(np.array([0, 2], dtype=np.int64))
        np.testing.assert_array_equal(X.toarray(), [[1, 0, 2], [4, 5, 0]])
    finally:
        store.close()


def test_gather_rows_disabled_env_does_not_raise(tmp_path, monkeypatch, fresh_warnings):
    path = _tiny_cell_archive(tmp_path)
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    store = cellstream.open(path)
    try:
        X = store.gather_rows(np.array([0, 1, 2], dtype=np.int64))
        np.testing.assert_array_equal(X.toarray(), [[1, 0, 2], [0, 3, 0], [4, 5, 0]])
    finally:
        store.close()


def test_read_cell_bulk_raises_without_extension(tmp_path, monkeypatch, fresh_warnings):
    """The policy is wired into read_cell_bulk too, and nothing else covered that: deleting the
    call there left every other test green. (codex checkpoint-2 review.)"""
    from cellstream.cell import _rust_dispatch as rd
    from cellstream.cell.reader import read_cell_bulk

    path = _tiny_cell_archive(tmp_path)
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    monkeypatch.delenv("CELLSTREAM_ALLOW_PYTHON_FALLBACK", raising=False)
    monkeypatch.setattr(rd, "_HAVE", False)
    with pytest.raises(RuntimeError, match="CELLSTREAM_ALLOW_PYTHON_FALLBACK"):
        read_cell_bulk(path, n_workers=1)


def test_read_cell_bulk_closes_store_when_policy_raises(tmp_path, monkeypatch, fresh_warnings):
    """read_cell_bulk holds an OPEN store at the policy check, so the raise must not leak it.
    Spy on CellStore.close to prove the except-branch actually runs. (codex checkpoint-2.)"""
    from cellstream.cell import _rust_dispatch as rd
    from cellstream.cell import reader as rdr

    path = _tiny_cell_archive(tmp_path)
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    monkeypatch.delenv("CELLSTREAM_ALLOW_PYTHON_FALLBACK", raising=False)
    monkeypatch.setattr(rd, "_HAVE", False)

    calls = []
    real_close = rdr.CellStore.close

    def spy(self):
        calls.append(1)
        return real_close(self)

    monkeypatch.setattr(rdr.CellStore, "close", spy)
    with pytest.raises(RuntimeError):
        rdr.read_cell_bulk(path, n_workers=1)
    assert calls, "read_cell_bulk leaked the open store when the fallback policy raised"


def test_read_cell_bulk_allowed_with_opt_in(tmp_path, monkeypatch, fresh_warnings):
    from cellstream.cell import _rust_dispatch as rd
    from cellstream.cell.reader import read_cell_bulk

    path = _tiny_cell_archive(tmp_path)
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    monkeypatch.setenv("CELLSTREAM_ALLOW_PYTHON_FALLBACK", "1")
    monkeypatch.setattr(rd, "_HAVE", False)
    with pytest.warns(UserWarning):
        a = read_cell_bulk(path, n_workers=1)
    np.testing.assert_array_equal(a.X.toarray(), [[1, 0, 2], [0, 3, 0], [4, 5, 0]])
