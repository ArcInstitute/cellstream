"""``bitshuffle`` is an OPTIONAL dependency since #305: only the legacy codec imports it, lazily.

Five properties, each named with the failure it guards against:

1. Importing the package -- and every module that used to drag ``bitshuffle`` in -- leaves it
   out of ``sys.modules``. Run in a SUBPROCESS on purpose, because in-process the answer
   depends on what earlier tests did: with ``bitshuffle`` already in ``sys.modules`` the check
   fails for the wrong reason, and with the cellstream modules already cached a re-introduced
   module-scope import would not even execute, so the check would PASS for the wrong reason. A
   fresh interpreter has neither problem.
2. The codecs the per-cell layout uses (byte-filter, zstd-only) round-trip with the package
   unavailable. This is the user-facing property: a default install reads and writes
   ``layout="cell"`` archives without the extra.
3. With the package unavailable, the legacy codec's encode and decode raise ``ImportError``
   carrying the full install instruction, chained from the real import failure, and BEFORE
   any zstd work -- proven for decode by handing it bytes that are not a zstd frame (an import
   placed after decompression would surface zstd's error instead), and for encode by replacing
   the compressor with one that raises if reached.
4. The same through the reader's self-describing stream dispatch (``_decode_stream``), which
   is where a legacy archive's decode reaches the codec.
5. A bitshuffle that is installed but BROKEN -- the package is found, and executing it raises:
   an extension that fails to load, or a module missing INSIDE the package -- is not misreported
   as absent: the very same exception object propagates, because "install the extra" would be
   wrong advice there.

"Unavailable" is simulated with ``sys.modules["bitshuffle"] = None``: the import system then
raises ``ModuleNotFoundError(name="bitshuffle")`` for ``import bitshuffle`` exactly as it does
for a package that is not installed, and it works here even though the package IS installed
(the ``test`` extra pulls it in). The control for that mechanism is built into (3) and (4): the
same encode SUCCEEDS before the poison is applied, producing the blob the decode half then
fails on.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import mmap
import os
import subprocess
import sys

import numpy as np
import pytest

# The complete instruction, not just the extra's name: the message is what a user acts on.
_INSTRUCTION = r"pip install 'cellstream\[bitshuffle\]'"


def _assert_actionable(excinfo):
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)  # chained, not replaced
    assert excinfo.value.__cause__.name == "bitshuffle"


def test_importing_cellstream_does_not_import_bitshuffle():
    code = (
        "import sys\n"
        "import cellstream, cellstream.cell, cellstream.v2.codec, cellstream.v2.reader_cpu, "
        "cellstream.v2.writer, cellstream.v2._rust_dispatch\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] == 'bitshuffle')\n"
        "assert not leaked, leaked\n"
        "print('ok')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy()
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


@pytest.fixture
def no_bitshuffle(monkeypatch):
    """Make ``import bitshuffle`` raise ImportError for the duration of a test."""
    monkeypatch.setitem(sys.modules, "bitshuffle", None)


def test_cell_layout_codecs_do_not_need_bitshuffle(no_bitshuffle):
    from cellstream.v2 import codec

    rng = np.random.default_rng(305)
    data = rng.integers(0, 5000, size=4096, dtype=np.uint32)
    indices = np.sort(rng.integers(0, 20000, size=4096, dtype=np.uint32))
    floats = rng.random(4096, dtype=np.float32)

    blob = codec.byte_filter_encode(data, role="data")
    np.testing.assert_array_equal(
        codec.byte_filter_decode(blob, dtype=np.uint32, n_elements=data.size, role="data"), data
    )
    blob = codec.byte_filter_encode(indices, role="indices")
    np.testing.assert_array_equal(
        codec.byte_filter_decode(blob, dtype=np.uint32, n_elements=indices.size, role="indices"),
        indices,
    )
    blob = codec.zstd_only_encode(floats)
    np.testing.assert_array_equal(
        codec.zstd_only_decode(blob, dtype=np.float32, n_elements=floats.size), floats
    )


def test_legacy_encode_without_bitshuffle_raises_actionable_error(no_bitshuffle):
    from cellstream.v2 import codec

    arr = np.arange(64, dtype=np.uint16)
    with pytest.raises(ImportError, match=_INSTRUCTION) as ei:
        codec.bitshuffle_zstd_encode(arr, level=1)
    _assert_actionable(ei)
    # Import BEFORE any zstd work: with the compressor replaced by a bomb, reaching it would raise
    # RuntimeError, so the ImportError proves the import comes first.
    with monkeypatch_compressor_bomb(codec):
        with pytest.raises(ImportError, match=_INSTRUCTION):
            codec.bitshuffle_zstd_encode(arr, level=1)
    # An empty stream is a bare zstd frame: it never invokes bitshuffle, so it needs no package.
    assert codec.bitshuffle_zstd_encode(np.empty(0, dtype=np.uint16), level=1)


class monkeypatch_compressor_bomb:
    """Replace ``codec.zstd.ZstdCompressor`` with a constructor that raises, for one block."""

    def __init__(self, codec_module):
        self._mod = codec_module.zstd

    def __enter__(self):
        self._orig = self._mod.ZstdCompressor

        def bomb(*a, **k):
            raise RuntimeError("zstd reached before the bitshuffle import")

        self._mod.ZstdCompressor = bomb
        return self

    def __exit__(self, *exc):
        self._mod.ZstdCompressor = self._orig
        return False


def test_legacy_decode_without_bitshuffle_raises_actionable_error(monkeypatch):
    from cellstream.v2 import codec

    arr = np.arange(64, dtype=np.uint16)
    blob = codec.bitshuffle_zstd_encode(arr, level=1)  # control: works with the package present
    monkeypatch.setitem(sys.modules, "bitshuffle", None)
    with pytest.raises(ImportError, match=_INSTRUCTION) as ei:
        codec.bitshuffle_zstd_decode(blob, dtype=np.uint16, n_elements=arr.size)
    _assert_actionable(ei)
    # Import BEFORE decompression: these bytes are not a zstd frame, so an import placed after
    # the zstd step would surface zstd's error here instead of the install instruction.
    with pytest.raises(ImportError, match=_INSTRUCTION):
        codec.bitshuffle_zstd_decode(b"not a zstd frame", dtype=np.uint16, n_elements=arr.size)
    # The zero-length early return does not need the package either.
    assert codec.bitshuffle_zstd_decode(blob, dtype=np.uint16, n_elements=0).size == 0


def test_decode_stream_legacy_archive_without_bitshuffle_raises_actionable_error(
    tmp_path, monkeypatch
):
    from cellstream.v2 import codec
    from cellstream.v2.format import StreamSpec
    from cellstream.v2.reader_cpu import _decode_stream

    arr = np.arange(2048, dtype=np.uint16)
    blob = codec.bitshuffle_zstd_encode(arr, level=1)  # control, as above
    p = tmp_path / "bs.bin"
    p.write_bytes(blob)
    spec = StreamSpec.from_dict(
        {
            "dtype": "uint16",
            "n_chunks": 1,
            "chunks": [
                {"offset": 0, "compressed_size": len(blob), "decompressed_size": arr.nbytes}
            ],
        }
    )
    with p.open("rb") as fh, mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ) as mm:
        np.testing.assert_array_equal(
            _decode_stream(mm, spec, np.uint16, shuffle="bitshuffle", role="data"), arr
        )
        monkeypatch.setitem(sys.modules, "bitshuffle", None)
        with pytest.raises(ImportError, match=_INSTRUCTION) as ei:
            _decode_stream(mm, spec, np.uint16, shuffle="bitshuffle", role="data")
        _assert_actionable(ei)


class _LoaderThatRaises(importlib.abc.Loader):
    """The package is FOUND; executing it raises -- what a broken install looks like."""

    def __init__(self, exc):
        self.exc = exc

    def create_module(self, spec):
        return None  # default module object

    def exec_module(self, module):
        raise self.exc


class _FoundButBrokenFinder(importlib.abc.MetaPathFinder):
    def __init__(self, exc):
        self.exc = exc

    def find_spec(self, name, path=None, target=None):
        if name == "bitshuffle":
            return importlib.machinery.ModuleSpec("bitshuffle", _LoaderThatRaises(self.exc))
        return None


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("libgomp.so.1: cannot open shared object file"),
        ModuleNotFoundError("No module named 'bitshuffle._native'", name="bitshuffle._native"),
    ],
    ids=["extension-load-failure", "module-missing-inside-the-package"],
)
def test_broken_bitshuffle_install_is_not_reported_as_absent(monkeypatch, exc):
    from cellstream.v2 import codec

    monkeypatch.delitem(sys.modules, "bitshuffle", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_FoundButBrokenFinder(exc), *sys.meta_path])
    with pytest.raises(ImportError) as ei:
        codec.bitshuffle_zstd_encode(np.arange(8, dtype=np.uint16), level=1)
    # The SAME exception object, untouched: no install instruction, nothing chained on top. The
    # second case is a ModuleNotFoundError whose `name` is a module INSIDE the package -- the
    # branch that must not be mistaken for "bitshuffle is not installed".
    assert ei.value is exc
    assert "cellstream[bitshuffle]" not in str(ei.value)
    assert ei.value.__cause__ is None
