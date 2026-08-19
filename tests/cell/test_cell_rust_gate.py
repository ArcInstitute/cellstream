"""The cell Rust gate must require every entry point the module calls (#282 item 4).

_HAVE checked only decode_cells/encode_cells, but the module also calls
_rust.encode_cells_block (:309) and _rust.reorder_frames (:343), both added in #219.
An extension built in between reported rust_available() == True and then
AttributeError'd -- worst for reorder_frames, whose `if not rust_available()` Python
fallback could never fire.
"""

from __future__ import annotations

import importlib
import types

import pytest

import cellstream
import cellstream.v2
from cellstream.cell import _rust_dispatch as crd

_ALL = ("decode_cells", "encode_cells", "encode_cells_block", "reorder_frames")


def _stub_ext(*missing):
    """A stub extension carrying every cell entry point EXCEPT ``missing``."""
    stub = types.SimpleNamespace()
    for name in _ALL:
        if name not in missing:
            setattr(stub, name, lambda *a, **k: None)
    return stub


def _install(monkeypatch, stub):
    """Make BOTH gates see ``stub``.

    Patching only cellstream.v2._rust is not enough for rust_status(), which reads
    _HAVE_RUST and _rust from cellstream.v2._rust_dispatch: with no extension built it
    would take the 'not importable' branch, and with a complete one it would inspect
    the real module -- passing for the wrong reason either way.
    """
    from cellstream.v2 import _rust_dispatch as vrd

    monkeypatch.setattr(cellstream.v2, "_rust", stub, raising=False)
    monkeypatch.setattr(vrd, "_rust", stub, raising=False)
    monkeypatch.setattr(vrd, "_HAVE_RUST", True, raising=False)
    return importlib.reload(crd)


def _install_partial(monkeypatch):
    """An extension that predates #219: the two original entry points only."""
    return _install(monkeypatch, _stub_ext("encode_cells_block", "reorder_frames"))


def test_entry_point_set_is_the_four_the_module_calls():
    assert crd._CELL_ENTRY_POINTS == _ALL


@pytest.mark.parametrize("missing", _ALL)
def test_dropping_any_single_entry_point_closes_the_gate(monkeypatch, missing):
    """One symbol at a time.

    Dropping BOTH new names together (as the #219-era stub does) cannot tell a gate
    that requires all four from one that requires only three: a gate that added
    encode_cells_block but forgot reorder_frames would pass that stub. Each name has
    to be able to close the gate on its own. (codex, checkpoint 2.)
    """
    reloaded = _install(monkeypatch, _stub_ext(missing))
    try:
        assert reloaded._HAVE is False, f"the gate stayed open with {missing!r} absent"
        assert not reloaded.rust_available()
    finally:
        monkeypatch.undo()
        importlib.reload(crd)


def test_a_complete_stub_leaves_the_gate_open(monkeypatch):
    """The control for the parametrisation above.

    Without it, a gate hard-wired to False -- or one requiring a name that does not
    exist -- would satisfy all four cases and look correct.
    """
    reloaded = _install(monkeypatch, _stub_ext())
    try:
        assert reloaded._HAVE is True
    finally:
        monkeypatch.undo()
        importlib.reload(crd)


def test_partial_extension_reads_as_unavailable(monkeypatch):
    reloaded = _install_partial(monkeypatch)
    try:
        assert reloaded._HAVE is False
        assert not reloaded.rust_available()
    finally:
        monkeypatch.undo()
        importlib.reload(crd)


def test_partial_extension_raises_the_rebuild_error(monkeypatch):
    """check_cell_python_fallback must RAISE, not merely carry a message constant."""
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    monkeypatch.delenv("CELLSTREAM_ALLOW_PYTHON_FALLBACK", raising=False)
    reloaded = _install_partial(monkeypatch)
    try:
        reloaded._reset_fallback_warnings()
        with pytest.raises(RuntimeError) as ei:
            reloaded.check_cell_python_fallback("test")
        for name in ("encode_cells_block", "reorder_frames"):
            assert name in str(ei.value)
    finally:
        monkeypatch.undo()
        importlib.reload(crd)


def test_rust_status_reason_names_the_missing_entry_points(monkeypatch):
    """rust_status() is the diagnostic every message tells the user to run, so its
    reason string must not still say 'decode_cells/encode_cells'."""
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    _install_partial(monkeypatch)
    try:
        status = cellstream.rust_status()
        assert status["extension_importable"] is True
        assert status["cell_ready"] is False
        reason = status["reason"]
        assert reason is not None
        for name in ("encode_cells_block", "reorder_frames"):
            assert name in reason
        assert "decode_cells/encode_cells)" not in reason
    finally:
        monkeypatch.undo()
        importlib.reload(crd)


@pytest.mark.parametrize("missing", ("encode_cells_block", "reorder_frames"))
def test_rust_status_reason_names_exactly_the_one_that_is_missing(monkeypatch, missing):
    """The reason string is derived from the live gate, not a hard-coded list -- so a
    single missing name must be named alone, not alongside the other three."""
    monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
    _install(monkeypatch, _stub_ext(missing))
    try:
        reason = cellstream.rust_status()["reason"]
        assert reason is not None
        # Parse the parenthesised list rather than substring-testing it: "encode_cells"
        # is a substring of "encode_cells_block", so a naive `name in reason` reports the
        # present names as missing too.
        named = reason.rsplit("(", 1)[1].rstrip(")").split("/")
        assert named == [missing], reason
    finally:
        monkeypatch.undo()
        importlib.reload(crd)
