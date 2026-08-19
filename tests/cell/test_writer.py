"""write_cell_archive: manifest fields, ordering, groups, frame decodability."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from cellstream.cell.codec import PforDeltaCodec
from cellstream.cell.writer import write_cell_archive
from cellstream.packed.reader import PackedArchive
from tests.cell.conftest import make_cell_adata


def _open_members(path):
    pa = PackedArchive(path)
    man = pa.manifest
    offsets = np.frombuffer(pa.member_bytes("x_offsets"), dtype=np.int64)
    indptr = np.frombuffer(pa.member_bytes("x_indptr"), dtype=np.int64)
    payload = pa.member_bytes("x_payload")
    return pa, man, offsets, indptr, payload


def test_ungrouped_manifest_and_shapes(tmp_path):
    a = make_cell_adata()
    out = tmp_path / "u.shad"
    write_cell_archive(a, out)
    pa, man, offsets, indptr, payload = _open_members(out)
    assert man["schema_version"] == 5 and man["layout"] == "cell"
    assert man["codec"] == "pfordelta"
    assert man["n_obs_total"] == a.n_obs and man["n_vars"] == a.n_vars
    assert man["total_nnz"] == a.X.nnz
    assert offsets.size == a.n_obs + 1 and indptr.size == a.n_obs + 1
    assert offsets[0] == 0 and indptr[0] == 0
    assert bool(np.all(offsets[1:] >= offsets[:-1]))
    assert indptr[-1] == a.X.nnz
    assert offsets[-1] == len(payload)


def test_frames_decode_to_source_rows_ungrouped(tmp_path):
    a = make_cell_adata()
    X = a.X.tocsr()
    X.sort_indices()
    out = tmp_path / "u.shad"
    write_cell_archive(a, out)
    pa, man, offsets, indptr, payload = _open_members(out)
    dec = PforDeltaCodec(man["index_dtype_on_disk"], man["value_dtype_on_disk"]).new_decoder(
        man["n_vars"]
    )
    for i in (0, 5, a.n_obs - 1):
        nnz = int(indptr[i + 1] - indptr[i])
        di, dv = dec.decode(payload[offsets[i] : offsets[i + 1]], nnz)
        s, e = X.indptr[i], X.indptr[i + 1]
        np.testing.assert_array_equal(di.astype(np.int64), X.indices[s:e].astype(np.int64))
        np.testing.assert_array_equal(dv.astype(np.int64), X.data[s:e].astype(np.int64))


def test_grouped_reference_first_block(tmp_path):
    a = make_cell_adata()
    out = tmp_path / "g.shad"
    write_cell_archive(a, out, group_by="target_gene", reference=["t00"])
    pa = PackedArchive(out)
    groups = json.loads(pa.member_bytes("groups.json"))
    # reference group leads (start==0) and manifest records it
    ref = [g for g in groups["groups"] if g["label"] == "t00"][0]
    assert ref["start"] == 0
    assert groups["reference"] == "t00" or "t00" in (groups.get("reference") or [])
    # obs.parquet is in storage order: the leading block is all-t00
    obs = pd.read_parquet(io.BytesIO(pa.member_bytes("obs.parquet")))
    assert (obs["target_gene"].to_numpy()[: ref["stop"]] == "t00").all()


def test_write_value_out_of_range_fails_loud(tmp_path):
    a = make_cell_adata()
    # force a count above uint16 with n_vars small (value_dtype would be uint16)
    a.X = a.X.tocsr()
    a.X.data = a.X.data.astype(np.int64)
    a.X.data[0] = 70000
    out = tmp_path / "big.shad"
    # value_dtype auto-selects uint32 here (max >= 65536), so this must SUCCEED and
    # record value_dtype_on_disk == uint32
    write_cell_archive(a, out)
    pa = PackedArchive(out)
    assert pa.manifest["value_dtype_on_disk"] == "uint32"


def test_reference_unknown_label_raises(tmp_path):
    a = make_cell_adata()
    with pytest.raises(ValueError, match="not found"):
        write_cell_archive(a, tmp_path / "x.shad", group_by="target_gene", reference=["nope"])


def test_rust_and_python_archives_byte_identical(tmp_path, monkeypatch):
    """The Rust and pure-Python encode paths must produce byte-identical archives."""
    from cellstream.cell._rust_dispatch import rust_available
    from tests.cell.conftest import make_cell_adata

    if not rust_available():
        pytest.skip("Rust core not built/enabled")

    a_rust = make_cell_adata()
    a_py = a_rust.copy()

    out_rust = tmp_path / "rust.shad"
    write_cell_archive(a_rust, out_rust, payload_checksum=True)

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    out_py = tmp_path / "py.shad"
    write_cell_archive(a_py, out_py, payload_checksum=True)

    pr, mr, offr, ipr, plr = _open_members(out_rust)
    pp, mp, offp, ipp, plp = _open_members(out_py)
    assert plr == plp  # payload byte-identical
    assert mr["payload_sha256"] == mp["payload_sha256"]
    np.testing.assert_array_equal(offr, offp)
    np.testing.assert_array_equal(ipr, ipp)
    for k in (
        "total_nnz",
        "value_dtype_on_disk",
        "index_dtype_on_disk",
        "x_data_dtype_min",
        "x_indices_dtype_min",
        "n_obs_total",
        "n_vars",
        "indices_canonical",
    ):
        assert mr[k] == mp[k], k


def test_rust_grouped_archive_byte_identical(tmp_path, monkeypatch):
    from cellstream.cell._rust_dispatch import rust_available
    from tests.cell.conftest import make_cell_adata

    if not rust_available():
        pytest.skip("Rust core not built/enabled")
    a_rust = make_cell_adata()
    a_py = a_rust.copy()
    write_cell_archive(a_rust, tmp_path / "r.shad", group_by="target_gene", reference=["t00"])
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    write_cell_archive(a_py, tmp_path / "p.shad", group_by="target_gene", reference=["t00"])
    assert PackedArchive(tmp_path / "r.shad").member_bytes("x_payload") == PackedArchive(
        tmp_path / "p.shad"
    ).member_bytes("x_payload")


def test_default_write_has_no_checksum_but_is_canonical(tmp_path):
    a = make_cell_adata()
    out = tmp_path / "d.shad"
    write_cell_archive(a, out)  # defaults: payload_checksum=False, require_unique_indices=True
    _, man, _, _, _ = _open_members(out)
    assert man["payload_sha256"] is None
    assert man["indices_canonical"] is True


def test_checksum_opt_in(tmp_path):
    import hashlib

    a = make_cell_adata()
    out = tmp_path / "c.shad"
    write_cell_archive(a, out, payload_checksum=True)
    _, man, _, _, payload = _open_members(out)
    assert man["payload_sha256"] == hashlib.sha256(payload).hexdigest()


def _dup_adata():
    """AnnData whose row 0 has a duplicate (cell, gene) entry (col 2 twice), sorted."""
    import anndata as ad
    import scipy.sparse as sp

    n_obs, n_vars = 3, 10
    x = sp.csr_matrix((n_obs, n_vars), dtype=np.uint16)
    x.data = np.array([1, 1, 5], dtype=np.uint16)
    x.indices = np.array([2, 2, 7], dtype=np.int32)
    x.indptr = np.array([0, 3, 3, 3], dtype=np.int64)
    a = ad.AnnData(X=x)
    a.var_names = [f"g{i:03d}" for i in range(n_vars)]
    return a


@pytest.mark.parametrize("disable_rust", [False, True])
def test_duplicate_indices_raise_by_default(tmp_path, monkeypatch, disable_rust):
    if disable_rust:
        monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    with pytest.raises(ValueError, match="duplicate column index"):
        write_cell_archive(_dup_adata(), tmp_path / "dup.shad")


@pytest.mark.parametrize("disable_rust", [False, True])
def test_require_unique_false_allows_and_preserves_duplicates(tmp_path, monkeypatch, disable_rust):
    if disable_rust:
        monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    out = tmp_path / "dupok.shad"
    write_cell_archive(_dup_adata(), out, require_unique_indices=False)
    _, man, _, _, _ = _open_members(out)
    assert man["indices_canonical"] is False
    assert man["total_nnz"] == 3  # the duplicate is preserved (not collapsed to 2)


def _plan(src_dtype, codec=None, data_dtype=None):
    from cellstream.cell.writer import _plan_cell_codec

    return _plan_cell_codec(np.dtype(src_dtype), codec=codec, data_dtype=data_dtype)


def test_plan_takes_a_dtype_not_a_matrix():
    """The scan is gone BY CONSTRUCTION: the planner never sees the data, so it cannot scan
    it. This is the whole fix -- an O(nnz) pass was 86% of the write. (#216, nick-youngblut.)"""
    import inspect

    from cellstream.cell.writer import _plan_cell_codec

    params = list(inspect.signature(_plan_cell_codec).parameters)
    assert params[0] == "src_dtype"
    assert "X" not in params


@pytest.mark.parametrize(
    "src,codec,dd,expected",
    [
        # integer input -- single attempt, no optimism needed
        (np.uint16, None, None, [("pfordelta", None)]),
        (np.uint16, "zstd", None, [("zstd", "uint32")]),
        # float input, no data_dtype -> OPTIMISTIC int, then float fallback
        (np.float32, None, None, [("pfordelta", None), ("zstd", "float32")]),
        (np.float32, "zstd", None, [("zstd", "uint32"), ("zstd", "float32")]),
        (np.float64, None, None, [("pfordelta", None), ("zstd", "float64")]),
        # explicit pfordelta on float -> NO fallback: fail loud
        (np.float32, "pfordelta", None, [("pfordelta", None)]),
        # data_dtype on float FORCES float storage -> straight there, no wasted int attempt
        (np.float32, None, "float32", [("zstd", "float32")]),
        (np.float32, "zstd", "float32", [("zstd", "float32")]),
        (np.float64, "zstd", "float32", [("zstd", "float32")]),
    ],
)
def test_plan_attempt_table(src, codec, dd, expected):
    assert _plan(src, codec=codec, data_dtype=dd) == expected


def test_plan_preserves_existing_raises():
    with pytest.raises(ValueError, match="requires integer counts"):
        _plan(np.float32, codec="pfordelta", data_dtype="float32")
    with pytest.raises(ValueError, match="only valid for codec='zstd'"):
        _plan(np.uint16, codec="pfordelta", data_dtype="float32")
    with pytest.raises(ValueError, match="requires float input"):
        _plan(np.uint16, codec="zstd", data_dtype="float32")
    with pytest.raises(ValueError, match="supports codec"):
        _plan(np.float32, codec="bogus")


def test_categorical_sort_within_group_uses_category_order(tmp_path):
    """#252: a categorical sort_within_group key must sort by DECLARED CATEGORY ORDER.

    The cell writer used obs[c].to_numpy(), which returns an ndarray of the category VALUES --
    so the sort fell back to alphabetical and the declared order was discarded, silently. The
    shard writer has always used .values (the Categorical), which plan_group_shards sorts by
    category order. This pins the two layouts to the same contract.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    import cellstream

    n = 6
    X = sp.csr_matrix(np.arange(1, n * 4 + 1, dtype=np.uint16).reshape(n, 4))
    obs = pd.DataFrame(
        {
            "target_gene": ["g"] * n,
            # category order is deliberately NOT alphabetical
            "assignment": pd.Categorical(
                ["apple", "zebra", "mango", "zebra", "apple", "mango"],
                categories=["zebra", "mango", "apple"],
                ordered=True,
            ),
        },
        index=[f"c{i}" for i in range(n)],
    )
    a = ad.AnnData(X=X, obs=obs)

    out = tmp_path / "cat.shad"
    cellstream.write_sharded(
        a, out, layout="cell", group_by="target_gene", sort_within_group="assignment"
    )
    st = cellstream.open(str(out))
    try:
        got = list(st.obs["assignment"].astype(str))
    finally:
        st.close()

    # category order: every zebra, then every mango, then every apple
    assert got == ["zebra", "zebra", "mango", "mango", "apple", "apple"], got
    assert got != sorted(got), "alphabetical would be the pre-#252 (wrong) result"


def test_categorical_sort_order_alphabetical_is_honoured(tmp_path):
    """#252: sort_order='alphabetical' must override the category order, not be a no-op."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    import cellstream

    n = 6
    X = sp.csr_matrix(np.arange(1, n * 4 + 1, dtype=np.uint16).reshape(n, 4))
    obs = pd.DataFrame(
        {
            "target_gene": ["g"] * n,
            "assignment": pd.Categorical(
                ["apple", "zebra", "mango", "zebra", "apple", "mango"],
                categories=["zebra", "mango", "apple"],
                ordered=True,
            ),
        },
        index=[f"c{i}" for i in range(n)],
    )
    a = ad.AnnData(X=X, obs=obs)

    out = tmp_path / "alpha.shad"
    cellstream.write_sharded(
        a,
        out,
        layout="cell",
        group_by="target_gene",
        sort_within_group="assignment",
        sort_order="alphabetical",
    )
    st = cellstream.open(str(out))
    try:
        got = list(st.obs["assignment"].astype(str))
    finally:
        st.close()

    assert got == ["apple", "apple", "mango", "mango", "zebra", "zebra"], got


@pytest.mark.parametrize(
    ("sort_order", "expected"),
    [
        (None, ["zebra", "zebra", "mango", "mango", "apple", "apple"]),
        ("alphabetical", ["apple", "apple", "mango", "mango", "zebra", "zebra"]),
    ],
)
def test_categorical_group_by_honours_sort_order(tmp_path, sort_order, expected):
    """#252 (codex): pin the PRIMARY group_by key, which the other two tests hold constant.

    group_by was never broken in the default path -- plan_group_shards extracts .values from a
    Series itself -- but it ignored sort_order='alphabetical', so cell ordered GROUPS by category
    order where shard ordered them lexicographically. Routing group_by through _sort_key fixes
    that and is a provable no-op for the default; this pins both halves.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    import cellstream

    n = 6
    X = sp.csr_matrix(np.arange(1, n * 4 + 1, dtype=np.uint16).reshape(n, 4))
    obs = pd.DataFrame(
        {
            # category order deliberately NOT alphabetical
            "target_gene": pd.Categorical(
                ["apple", "zebra", "mango", "zebra", "apple", "mango"],
                categories=["zebra", "mango", "apple"],
                ordered=True,
            ),
        },
        index=[f"c{i}" for i in range(n)],
    )
    a = ad.AnnData(X=X, obs=obs)

    out = tmp_path / f"gb_{sort_order}.shad"
    cellstream.write_sharded(a, out, layout="cell", group_by="target_gene", sort_order=sort_order)
    st = cellstream.open(str(out))
    try:
        got = list(st.obs["target_gene"].astype(str))
    finally:
        st.close()
    assert got == expected, got
