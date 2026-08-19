"""#174: the pure-Python narrow-target read must reject a tampered/corrupt recorded
x_*_dtype_min that would otherwise silently truncate via a blind astype.

Python mirror of the Rust facet in tests/v2/test_low_plane_min_tamper.py (#170). The
fix lives in the Python decoder's cast_data seam, so these run only under the Python
engine; under Rust the same tamper raises a plain ValueError from the #170 high-plane
check, which that test covers.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cellstream.errors import LossyCastError
from cellstream.v2._rust_dispatch import rust_available
from tests._archive_helpers import archive_manifest

# Gate on the ENGINE's own dispatch predicate, not a raw bool(env): the Python
# decode path is active iff the Rust core is not (extension absent OR disabled by
# CELLSTREAM_DISABLE_RUST=1). This also runs the tests in a pure-Python install (no
# Rust extension) and correctly skips them under CELLSTREAM_DISABLE_RUST=0 (Rust active).
pytestmark = pytest.mark.skipif(
    rust_available(),
    reason="Python full-width decode + cast_data seam runs only when the Rust decode path is inactive (Rust facet: test_low_plane_min_tamper.py / #170)",
)


def _srcdir(out):
    """Scaffolding directory beside the archive under test.

    A manifest tamper rewrites a member in place, which changes its byte length and
    breaks a packed footer -- so the tamper happens on a directory, which is then packed
    into ``out``. The archive the tests read is packed, so they keep exercising the
    PUBLIC reader (#154 part B).
    """
    return out.parent / (out.name + "_src")


def _write_always_uint32(out, data, indices, indptr, shape):
    import anndata as ad
    import scipy.sparse as sp

    from tests._archive_helpers import pack_directory, write_directory

    X = sp.csr_matrix((data, indices, indptr), shape=shape)
    write_directory(ad.AnnData(X=X), _srcdir(out), n_shards=1)
    pack_directory(_srcdir(out), out)


def _tamper_min(out, key, value):
    from tests._archive_helpers import pack_directory

    mpath = _srcdir(out) / "manifest.json"
    m = json.loads(mpath.read_text())
    m[key] = value
    mpath.write_text(json.dumps(m))
    pack_directory(_srcdir(out), out, overwrite=True)  # republish with the tamper
    return m


def test_python_narrow_data_read_rejects_tampered_min(tmp_path):
    from cellstream.read import ShardedArchive

    # always-uint32 archive holding 70000 (> 65535) -> true x_data_dtype_min = uint32.
    out = tmp_path / "tamper_data"
    _write_always_uint32(
        out,
        np.array([70000, 5, 9], dtype=np.uint32),
        np.array([0, 2, 1], dtype=np.int32),
        np.array([0, 1, 2, 3], dtype=np.int64),
        (3, 5),
    )
    m = _tamper_min(out, "x_data_dtype_min", "uint16")
    assert m["x_data_dtype_on_disk"] == "uint32"  # sanity: stored full-width

    # Explicit narrow uint16 data target: safe=True (tampered min), integer narrowing.
    with pytest.raises(LossyCastError, match="lossy"):
        ShardedArchive(out).to_anndata(data_dtype="uint16", n_workers=1)


def test_python_narrow_index_read_rejects_tampered_min(tmp_path):
    from cellstream.read import ShardedArchive

    # WIDE archive (n_vars > 65536) with a column index 69999 (> 65535) -> true
    # x_indices_dtype_min = uint32. Tamper it to uint16 and request a uint16 index.
    out = tmp_path / "tamper_idx"
    _write_always_uint32(
        out,
        np.array([1, 2, 3], dtype=np.uint32),
        np.array([0, 69999, 3], dtype=np.int32),
        np.array([0, 1, 2, 3], dtype=np.int64),
        (3, 70000),
    )
    m = _tamper_min(out, "x_indices_dtype_min", "uint16")
    assert m["x_indices_dtype_on_disk"] == "uint32"
    with pytest.raises(LossyCastError, match="lossy"):
        ShardedArchive(out).to_anndata(index_dtype="uint16", n_workers=1)


def test_python_honest_narrow_data_read_roundtrips(tmp_path):
    # No false positive: honest always-uint32 archive whose values fit uint16 -> the
    # lossless_cast scan passes and the narrow read succeeds.
    from cellstream.read import ShardedArchive

    out = tmp_path / "honest"
    _write_always_uint32(
        out,
        np.array([1, 2, 65535], dtype=np.uint32),
        np.array([0, 2, 1], dtype=np.int32),
        np.array([0, 1, 2, 3], dtype=np.int64),
        (3, 5),
    )
    assert archive_manifest(out)["x_data_dtype_on_disk"] == "uint32"
    a = ShardedArchive(out).to_anndata(data_dtype="uint16", n_workers=1)
    assert a.X.data.dtype == np.uint16
    assert int(a.X.data.max()) == 65535


def test_python_tampered_min_default_float32_read_unaffected(tmp_path):
    # No hot-path regression: the DEFAULT read (float32 data) of the tampered archive is
    # unaffected -- full-width upcast, no truncation, no false positive.
    from cellstream.read import ShardedArchive

    out = tmp_path / "tamper_default"
    _write_always_uint32(
        out,
        np.array([70000, 5, 9], dtype=np.uint32),
        np.array([0, 2, 1], dtype=np.int32),
        np.array([0, 1, 2, 3], dtype=np.int64),
        (3, 5),
    )
    _tamper_min(out, "x_data_dtype_min", "uint16")
    a = ShardedArchive(out).to_anndata(n_workers=1)  # default cast_back=True -> float32
    assert int(a.X.data.max()) == 70000  # upcast, no truncation


def test_cast_data_seam_unit():
    # Direct unit coverage of the cast_data branch (engine-agnostic; no archive).
    from cellstream.v2.materialize import cast_data

    arr = np.array([70000, 5, 9], dtype=np.uint32)  # a decoded full-width buffer
    # safe=True + integer narrowing + not allow_lossy -> verify -> raise on the lossy value.
    with pytest.raises(LossyCastError, match="lossy"):
        cast_data(arr, np.uint16, allow_lossy=False, where="X.data", safe=True)
    # honest values fit -> passes with the scan.
    ok = cast_data(
        np.array([1, 65535], dtype=np.uint32),
        np.uint16,
        allow_lossy=False,
        where="X.data",
        safe=True,
    )
    assert ok.dtype == np.uint16 and ok.tolist() == [1, 65535]
    # float target -> excluded (fast upcast, no raise even with a wrong safe).
    f = cast_data(arr, np.float32, allow_lossy=False, where="X.data", safe=True)
    assert f.dtype == np.float32 and float(f.max()) == 70000.0
    # allow_lossy=True -> caller opted into truncation (unchanged).
    t = cast_data(arr, np.uint16, allow_lossy=True, where="X.data", safe=True)
    assert int(t[0]) == 70000 & 0xFFFF  # 4464
