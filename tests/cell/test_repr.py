"""#336 -- ``repr()`` of CellStore and AnnDataView: anndata's backed layout, from metadata only.

Two things are pinned here beyond the text itself. (1) Parity with anndata: the slot lines are
compared against ``anndata.read_h5ad(..., backed="r")`` of the same AnnData, so a drift in either
format shows up as a failure rather than a silent divergence. (2) Cost: a repr runs on every
implicit notebook display, so it must not read obs.parquet or materialise the header AnnData --
the spies below make either read an error, and the caches are checked to be still empty.
"""

from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

from cellstream.cell import AnnDataView, open_cell, write_cell_archive
from cellstream.packed.reader import PackedArchive, _MmapWindow

from .conftest import make_cell_adata

SLOTS = ("obs", "var", "uns", "obsm", "varm", "obsp", "varp")


def _slot_lines(text):
    return [ln.strip() for ln in text.splitlines()[1:] if ln.strip().split(":")[0] in SLOTS]


def _write(tmp_path, name, a, **kw):
    p = tmp_path / name
    write_cell_archive(a, p, **kw)
    return p


def test_plain_store_exact_text(tmp_path):
    p = _write(tmp_path, "plain.csad", make_cell_adata())
    st = open_cell(p)
    assert repr(st) == (
        f"CellStore with n_obs × n_vars = 240 × 60 backed at {str(p)!r}\n"
        "    obs: 'target_gene', 'n_umi'\n"
        "    X: uint16 on disk, reads as float32"
    )
    # The view prints the same thing under its own name -- via the store's parts, not its text.
    assert repr(AnnDataView(st)) == repr(st).replace("CellStore", "AnnDataView", 1)
    assert repr(st.to_anndata()) == repr(AnnDataView(st))
    st.close()


def test_annotated_store_exact_text_and_anndata_parity(tmp_path):
    a = make_cell_adata(annotations=True)
    a.var["gene_name"] = list(a.var_names)
    p = _write(tmp_path, "ann.csad", a)
    st = open_cell(p)
    assert repr(st) == (
        f"CellStore with n_obs × n_vars = 240 × 60 backed at {str(p)!r}\n"
        "    obs: 'target_gene', 'n_umi'\n"
        "    var: 'gene_name'\n"
        "    uns: 'method'\n"
        "    obsm: 'X_pca'\n"
        "    varm: 'PCs'\n"
        "    obsp: 'connectivities', 'distances'\n"
        "    varp: 'corr'\n"
        "    X: uint16 on disk, reads as float32"
    )
    # Parity with anndata's backed repr of the same object: header and every slot line.
    h5 = tmp_path / "ann.h5ad"
    a.write_h5ad(h5)
    b = ad.read_h5ad(h5, backed="r")
    try:
        ours, theirs = repr(st), repr(b)
    finally:
        b.file.close()
    assert ours.splitlines()[0] == theirs.splitlines()[0].replace(
        "AnnData object", "CellStore"
    ).replace(repr(str(h5)), repr(str(p)))
    assert _slot_lines(ours) == _slot_lines(theirs)
    st.close()


def test_pairwise_only_slots_are_listed(tmp_path):
    # The first version gated obsp/varp on the obsm/varm lengths and labelled all four "obsm".
    a = make_cell_adata()
    a.obsp["connectivities"] = sp.random(a.n_obs, a.n_obs, density=0.05, format="csr")
    a.varp["corr"] = sp.random(a.n_vars, a.n_vars, density=0.1, format="csr")
    st = open_cell(_write(tmp_path, "pairs.csad", a))
    lines = _slot_lines(repr(st))
    assert lines == ["obs: 'target_gene', 'n_umi'", "obsp: 'connectivities'", "varp: 'corr'"]
    st.close()


def test_repr_reads_metadata_only(tmp_path, monkeypatch):
    p = _write(tmp_path, "ann.csad", make_cell_adata(annotations=True))
    st = open_cell(p)
    expected = repr(st)  # unspied first, as the reference text
    st.close()

    st = open_cell(p)
    view = AnnDataView(st)

    def boom(*args, **kwargs):
        raise AssertionError("repr must not read obs.parquet or materialise header.h5ad")

    # Whole-member reads: member_bytes feeds `obs` (read_parquet) and header_adata feeds
    # var/uns/obsm/varm/obsp/varp. Neither may run for a repr.
    monkeypatch.setattr(PackedArchive, "member_bytes", boom)
    monkeypatch.setattr(PackedArchive, "header_adata", boom)
    monkeypatch.setattr(pd, "read_parquet", boom)
    # Nor may the metadata readers themselves touch data: no HDF5 dataset read (a 60-element
    # var index costs nothing at this size and would pass the byte bounds below -- codex,
    # checkpoint 2 round 2) and no parquet row group. Group listings and attributes stay legal.
    monkeypatch.setattr(h5py.Dataset, "__getitem__", boom)
    monkeypatch.setattr(h5py.Dataset, "read_direct", boom)
    for name in ("read", "read_row_group", "read_row_groups", "iter_batches"):
        monkeypatch.setattr(pq.ParquetFile, name, boom)
    monkeypatch.setattr(pq, "read_table", boom)
    assert repr(st) == expected
    assert repr(view) == expected.replace("CellStore", "AnnDataView", 1)
    for attr in ("_obs", "_var", "_uns", "_obsm", "_varm", "_obsp", "_varp", "_hdr"):
        assert getattr(st, attr) is None, attr
    st.close()


def test_repr_bytes_read_do_not_scale_with_n_obs(tmp_path, monkeypatch):
    """The spies above prove the OLD loaders are not called; this pins the NEW ones' cost directly.

    Everything the repr reads from disk goes through `_MmapWindow.read` (`readinto` delegates
    to it), so counting there is counting the repr's I/O. Measured: pyarrow's 64 KiB footer
    probe plus a few dozen HDF5 object-header reads, ~74 KiB, identical at 20 k and 200 k cells,
    while the two members themselves grow from ~130 KiB + ~2 MiB to ~850 KiB + ~20 MiB. A
    regression that read a row group or a dataset would scale with n_obs and blow both bounds.
    The small archive must exceed that 64 KiB probe, or the probe reads the whole (smaller)
    file and the flat-in-n_obs comparison is meaningless -- 2 k cells read 26 KiB. (codex,
    checkpoint 2.)
    """
    counted = {"bytes": 0}
    real_read = _MmapWindow.read

    def read(self, n=-1):
        out = real_read(self, n)
        counted["bytes"] += len(out)
        return out

    monkeypatch.setattr(_MmapWindow, "read", read)

    def bytes_for(n_obs):
        a = make_cell_adata(n_obs=n_obs)
        a.obs["barcode"] = [
            f"CELL{i:07d}" for i in range(n_obs)
        ]  # a column that does not compress away
        st = open_cell(_write(tmp_path, f"n{n_obs}.csad", a))
        members = {m["name"]: m["length"] for m in st._packed.members}
        counted["bytes"] = 0
        repr(st)
        st.close()
        return counted["bytes"], members["obs.parquet"] + members["header.h5ad"]

    small, _ = bytes_for(20_000)
    big, big_members = bytes_for(200_000)
    assert 0 < big < 256 * 1024 and big < big_members / 50, (big, big_members)
    assert big <= small + 4096, (small, big)


def test_cached_frames_agree_with_metadata_and_follow_mutation(tmp_path):
    p = _write(tmp_path, "ann.csad", make_cell_adata(annotations=True))
    fresh = open_cell(p)
    warm = open_cell(p)
    for attr in SLOTS:
        getattr(warm, attr)  # populate every cache the ordinary way
    assert warm._obs is not None and warm._hdr is not None
    assert repr(warm) == repr(fresh)
    # A cached dict is the live object anndata-style mutation lands in; the repr follows it,
    # exactly as an AnnData's would.
    warm.uns["extra"] = 1
    assert "    uns: 'method', 'extra'" in repr(warm).splitlines()
    fresh.close()
    warm.close()


def test_x_line_reports_on_disk_and_read_dtypes(tmp_path):
    a = make_cell_adata()
    a.X = sp.csr_matrix(a.X, dtype=np.float64)
    a.X.data += 0.25  # genuine floats, stored as float64 by the zstd codec
    st = open_cell(_write(tmp_path, "f64.csad", a, codec="zstd", data_dtype="float64"))
    assert repr(st).splitlines()[-1] == "    X: float64 on disk, reads as float64"
    assert st.gather_rows(np.arange(3)).dtype == np.float64
    st.close()


class _DuckStore:
    n_obs = 3
    n_vars = 4

    def gather_rows(self, rows):  # pragma: no cover - never called by repr
        raise NotImplementedError


class _BareStore:
    def gather_rows(self, rows):  # pragma: no cover - never called by repr
        raise NotImplementedError


@pytest.mark.parametrize(
    ("store", "expected"),
    [
        (_DuckStore(), "AnnDataView with n_obs × n_vars = 3 × 4"),
        (_BareStore(), "AnnDataView with n_obs × n_vars = ? × ?"),
    ],
)
def test_view_repr_over_a_duck_typed_store_reads_nothing(store, expected):
    assert repr(AnnDataView(store)) == expected
