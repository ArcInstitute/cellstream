"""#277 defect 4: verify_payload() must not report an intact EMPTY archive as corrupt."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellstream.cell import format as fmt
from cellstream.cell.reader import CellStore, read_cell_bulk
from cellstream.cell.writer import write_cell_archive


def _empty(tmp_path, name="e.shad"):
    a = ad.AnnData(
        X=sp.csr_matrix((0, 4), dtype=np.uint16),
        obs=pd.DataFrame(index=pd.Index([], dtype=str)),
        var=pd.DataFrame(index=[f"g{j}" for j in range(4)]),
    )
    p = tmp_path / name
    write_cell_archive(a, p, payload_checksum=True)
    return p


def _control(tmp_path, name="c.shad"):
    vals = np.array([[1, 2, 0, 0], [0, 3, 4, 0], [0, 0, 5, 6], [7, 0, 0, 8]], dtype=np.uint16)
    a = ad.AnnData(
        X=sp.csr_matrix(vals, dtype=np.uint16),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(4)]),
    )
    p = tmp_path / name
    write_cell_archive(a, p, payload_checksum=True)
    return p


def test_empty_archive_maps_an_empty_window(tmp_path):
    s = CellStore(_empty(tmp_path))
    try:
        assert s.n_obs == 0
        assert s.has_payload_checksum
        assert len(s._mm) == 0, "a 0-byte payload must not map the whole container"
    finally:
        s.close()


def test_empty_archive_verifies_intact(tmp_path):
    s = CellStore(_empty(tmp_path))
    try:
        assert s.verify_payload() is True
    finally:
        s.close()


def test_control_archive_still_verifies(tmp_path):
    s = CellStore(_control(tmp_path))
    try:
        assert len(s._mm) > 0
        assert s.verify_payload() is True
    finally:
        s.close()


def test_tampered_control_still_fails(tmp_path):
    """Negative control: the fix must not make verify_payload vacuously true."""
    p = _control(tmp_path, "t.shad")
    s = CellStore(p)
    base = s._packed.member_location(fmt.PAYLOAD)[1]
    s.close()
    with open(p, "r+b") as fh:
        fh.seek(base)
        orig = fh.read(1)
        fh.seek(base)
        fh.write(bytes([orig[0] ^ 0xFF]))
    s = CellStore(p)
    try:
        assert s.verify_payload() is False
    finally:
        s.close()


def test_empty_archive_still_reads(tmp_path):
    got = read_cell_bulk(_empty(tmp_path), data_dtype="uint16")
    assert got.shape == (0, 4)


def test_empty_archive_gather_rows_is_empty(tmp_path):
    s = CellStore(_empty(tmp_path))
    try:
        assert s.gather_rows(np.array([], dtype=np.int64)).shape == (0, 4)
    finally:
        s.close()
