"""Public view of the Rust acceleration runtime (#244 ask #2).

There are TWO independent Rust gates and they can disagree: v2._rust_dispatch._HAVE_RUST is
only "did `cellstream.v2._rust` import", while cell._rust_dispatch._HAVE additionally requires
every name in cell._rust_dispatch._CELL_ENTRY_POINTS to exist on it. An extension built
before the cell work gives shard=True, cell=False. rust_available() is the conservative conjunction, so any partial
state reads as False; rust_status() is how you find out WHICH part is missing.

The gate booleans are read through the modules (not imported by value) so a test that
monkeypatches `_HAVE_RUST` / `_HAVE` is observed here.
"""

from __future__ import annotations

import os


def _gates():
    from cellstream.cell import _rust_dispatch as _cell
    from cellstream.v2 import _rust_dispatch as _shard

    return _shard, _cell


def rust_available() -> bool:
    """True iff the Rust extension is importable, complete, and not disabled by env.

    Conservative on purpose: this answers "am I on the fast path?", so a partially usable
    extension is False. Use :func:`rust_status` to see which gate failed.
    """
    shard, cell = _gates()
    return bool(shard.rust_available() and cell.rust_available())


def rust_status() -> dict:
    """Diagnostic detail behind :func:`rust_available`.

    Keys: version, available, extension_importable, disabled_by_env, allow_python_fallback,
    shard_ready, cell_ready, reason (a human string when unavailable, else None).
    """
    from cellstream import __version__

    shard, cell = _gates()
    disabled = os.environ.get("CELLSTREAM_DISABLE_RUST") == "1"
    importable = bool(shard._HAVE_RUST)
    shard_ready = bool(shard.rust_available())
    cell_ready = bool(cell.rust_available())
    available = shard_ready and cell_ready
    if available:
        reason = None
    elif disabled:
        reason = "disabled by CELLSTREAM_DISABLE_RUST=1"
    elif not importable:
        reason = "extension not importable (built without the Rust toolchain)"
    elif not cell_ready:
        # Read cell._rust, NOT shard._rust: cell._rust_dispatch binds its own reference via
        # `from cellstream.v2 import _rust`, and that is the object its _HAVE was computed from,
        # so it is the one whose missing names explain the gate. getattr(..., None) plus the
        # `or` fallback keeps the expression total -- this branch is only reached when the
        # extension IS importable, so both defences are belt-and-braces. (#282 item 4)
        _cell_rust = getattr(cell, "_rust", None)
        missing = [n for n in cell._CELL_ENTRY_POINTS if not hasattr(_cell_rust, n)]
        reason = (
            "extension present but missing the cell entry points "
            f"({'/'.join(missing or cell._CELL_ENTRY_POINTS)})"
        )
    else:
        reason = "extension present but the shard entry points are unavailable"
    return {
        "version": __version__,
        "available": available,
        "extension_importable": importable,
        "disabled_by_env": disabled,
        "allow_python_fallback": os.environ.get("CELLSTREAM_ALLOW_PYTHON_FALLBACK") == "1",
        "shard_ready": shard_ready,
        "cell_ready": cell_ready,
        "reason": reason,
    }
