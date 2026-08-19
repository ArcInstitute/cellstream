"""Golden-frame freeze (guards PR2 Rust byte-identity) + cross-codec + schema gate."""

from __future__ import annotations

import numpy as np
import pytest

from .test_zstd_writer import _dense_skewed_adata

# Frozen on-disk frames from the pure-Python codec. If a change here is intentional,
# regenerate; otherwise a diff means the on-disk bytes moved (PR2's Rust port must match).
GOLDEN = {
    "uint32": b"\x14\x00\x00\x00(\xb5/\xfd \x10]\x00\x00(\x00\x02\x03\x04\x00\x01\x00\x06\xd0\x02(\xb5/\xfd \x10]\x00\x00(\x03\x01\x04\x01\x00\x01\x00\x06\xd0\x02",
    "float32": b"\x14\x00\x00\x00(\xb5/\xfd \x10]\x00\x00(\x00\x02\x03\x04\x00\x01\x00\x06\xd0\x02(\xb5/\xfd \x10\x81\x00\x00\x00\x00\xc0?\x00\x00\x10@\x00\x00@\xc0\x00\x00\x00\x00",
    "float64": b"\x14\x00\x00\x00(\xb5/\xfd \x10]\x00\x00(\x00\x02\x03\x04\x00\x01\x00\x06\xd0\x02(\xb5/\xfd  \xed\x00\x00\xb0\x00\x00\x00\x00\x00\x00\xf8?\x00\x02@\x00\x08\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00@\xc6\x05\x8e",
}


@pytest.mark.parametrize("value_dtype", ["uint32", "float32", "float64"])
def test_golden_frame_bytes_frozen(value_dtype):
    from cellstream.cell.codec import ByteZstdCodec

    idx = np.array([0, 2, 5, 9], np.uint32)
    vals = {
        "uint32": np.array([3, 1, 4, 1], np.uint32),
        "float32": np.array([1.5, 2.25, -3.0, 0.0], np.float32),
        "float64": np.array([1.5, 2.25, -3.0, 0.0], np.float64),
    }[value_dtype]
    frame = ByteZstdCodec("uint32", value_dtype).new_encoder().encode(idx, vals)
    assert frame == GOLDEN[value_dtype]


def test_cross_codec_identical_csr_and_zstd_smaller(tmp_path):
    from cellstream.cell.reader import read_cell_bulk
    from cellstream.cell.writer import write_cell_archive

    a = _dense_skewed_adata()  # dense skewed cells: the regime where zstd-int wins
    p = tmp_path / "p.shad"
    z = tmp_path / "z.shad"
    write_cell_archive(a, p, codec="pfordelta", overwrite=True)
    write_cell_archive(a, z, codec="zstd", overwrite=True)
    xp = read_cell_bulk(p).X.tocsr()
    xz = read_cell_bulk(z).X.tocsr()
    np.testing.assert_array_equal(xp.toarray(), xz.toarray())
    assert z.stat().st_size < p.stat().st_size


def test_unsupported_schema_version_rejected():
    # The schema gate: a reader rejects a version it does not support. After this PR,
    # v6 is supported and v99 stands in for "a newer archive an older reader can't read"
    # (the same mechanism by which a <=v5 reader rejects a v6 zstd archive).
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import validate_manifest

    m = {
        "schema_version": 99,
        "layout": "cell",
        "codec": "zstd",
        "n_obs_total": 1,
        "n_vars": 1,
        "total_nnz": 0,
        "value_dtype_on_disk": "float64",
        "index_dtype_on_disk": "uint32",
        "x_data_dtype_min": "float64",
        "x_indices_dtype_min": "uint16",
        "pyfastpfor_codec": None,
    }
    with pytest.raises(IncompatibleSchemaError):
        validate_manifest(m)
