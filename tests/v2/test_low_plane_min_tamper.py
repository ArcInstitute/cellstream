"""L5: the Rust low-plane narrowing read path must reject a tampered/corrupt
recorded-min that would otherwise silently truncate the discarded high planes.

A value > 65535 stored as always-uint32 has a nonzero high byte-plane. If the
manifest's recorded min is tampered to claim uint16, the #161 low-plane gather
reads only the low 2 planes and (without the high-plane check) silently truncates.
The fix verifies the discarded high planes are all zero on the narrowing path.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from cellstream.v2._rust_dispatch import rust_available

# The low-plane high-plane check lives in the Rust decoder, so these only exercise
# it when the Rust read path is actually active (skip under CELLSTREAM_DISABLE_RUST,
# where read_v2_to_host falls back to the full-width Python decoder).
pytestmark = pytest.mark.skipif(
    not rust_available() or bool(os.environ.get("CELLSTREAM_DISABLE_RUST")),
    reason="Rust low-plane decode path required (not exercised in Python fallback)",
)


def test_rust_read_rejects_tampered_min_narrowing(tmp_path):
    import anndata as ad
    import scipy.sparse as sp

    from cellstream.v2.reader_cpu import read_v2_to_host
    from tests._archive_helpers import write_directory

    # always-uint32 archive containing 70000 (> 65535) -> true x_data_dtype_min = uint32.
    data = np.array([70000, 5, 9], dtype=np.uint32)
    indices = np.array([0, 2, 1], dtype=np.int32)
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    X = sp.csr_matrix((data, indices, indptr), shape=(3, 5))
    out = tmp_path / "tamper"
    write_directory(ad.AnnData(X=X), out, n_shards=1)

    mpath = out / "manifest.json"
    m = json.loads(mpath.read_text())
    assert m["x_data_dtype_min"] == "uint32"  # sanity: 70000 needs uint32
    m["x_data_dtype_min"] = "uint16"  # tamper: falsely claim the high planes are zero
    mpath.write_text(json.dumps(m))

    # The narrowing gather (selected by the tampered min) would read only the low
    # 2 planes of 70000 and silently yield 4464; the high-plane check rejects it.
    with pytest.raises(ValueError, match="high byte-plane|min dtype"):
        read_v2_to_host(out, cast_back=False, n_workers=1)


def test_rust_read_honest_min_uint32_value_roundtrips(tmp_path):
    # Control: with the TRUE min (uint32) the value is read full-width and is not
    # truncated -- the high-plane check must NOT fire on an honest archive.
    import anndata as ad
    import scipy.sparse as sp

    from cellstream.v2.reader_cpu import read_v2_to_host
    from tests._archive_helpers import write_directory

    data = np.array([70000, 5, 9], dtype=np.uint32)
    indices = np.array([0, 2, 1], dtype=np.int32)
    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    X = sp.csr_matrix((data, indices, indptr), shape=(3, 5))
    out = tmp_path / "honest"
    write_directory(ad.AnnData(X=X), out, n_shards=1)

    d, _, _ = read_v2_to_host(out, cast_back=False, n_workers=1)
    assert int(d.max()) == 70000  # full-width, no truncation
