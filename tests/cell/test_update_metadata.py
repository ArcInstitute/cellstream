"""In-place obs / var edits on a layout='cell' archive (#338).

``update_obs_packed`` / ``update_var_packed`` truncate the archive at ``shard_region_end`` and
re-append the metadata tail (docs/format.md section 2.4). A cell archive keeps its matrix in three
members BELOW that boundary -- ``x_payload``, ``x_offsets``, ``x_indptr`` -- so the truncation never
touches their bytes, and an edit has only the tail to rewrite. Every cell archive was refused
instead: the rewrite carried only ``role == "shard"`` members across.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cellstream
from cellstream.cell.reader import open_cell
from cellstream.packed.writer import update_obs_packed, update_var_packed
from tests.cell.conftest import _assert_matrix_equal, make_cell_adata

_DATA_MEMBERS = ("x_payload", "x_offsets", "x_indptr")
_ANNOTATION_SLOTS = ("obsm", "varm", "obsp", "varp")
_GROUPED = {"group_by": "target_gene", "reference": "t00"}
_CASES = {
    "pfordelta": {"codec": "pfordelta"},
    "pfordelta-grouped": {"codec": "pfordelta", **_GROUPED},
    "zstd-grouped": {"codec": "zstd", **_GROUPED},
}


def _write(tmp_path, case):
    a = make_cell_adata(annotations=True)
    # A per-sample QC column, the kind the issue drops so that parts share one var for
    # concat_cell_archives.
    a.var["total_counts"] = np.asarray(a.X.sum(axis=0)).ravel().astype(np.int64)
    p = tmp_path / "a.csad"
    cellstream.write_sharded(a, p, layout="cell", payload_checksum=True, **_CASES[case])
    return p


def _state(p):
    """Everything an obs / var edit must leave alone, plus obs and var themselves.

    Opened through CellStore first, so a pending ``.tailbak`` is recovered by the cell reader's
    own open, and every member is read through that store's footer.
    """
    store = open_cell(p)
    pa = store._packed
    try:
        return {
            "entries": [m for m in pa.members if m["name"] in _DATA_MEMBERS],
            "data": {n: pa.member_bytes(n) for n in _DATA_MEMBERS},
            "manifest": pa.member_bytes("manifest.json"),
            "groups": pa.member_bytes("groups.json") if pa.has_member("groups.json") else None,
            "X": store.gather_rows(np.arange(store.n_obs)),
            "payload_verified": store.verify_payload(),
            "obs": store.obs.copy(),
            "var": store.var.copy(),
            "uns": dict(store.uns),
            **{slot: dict(getattr(store, slot)) for slot in _ANNOTATION_SLOTS},
        }
    finally:
        store.close()


def _assert_only_edited_changed(p, before, after, edited):
    # The data region is untouched: same bytes, and the same footer entries -- so the same
    # offsets -- for all three members.
    assert after["entries"] == before["entries"]
    assert after["data"] == before["data"]
    assert before["payload_verified"] and after["payload_verified"]
    _assert_matrix_equal(after["X"], before["X"], "X")
    # The rest of the tail is carried over.
    assert after["manifest"] == before["manifest"]
    assert after["groups"] == before["groups"]
    other = "var" if edited == "obs" else "obs"
    pd.testing.assert_frame_equal(after[other], before[other])
    assert after["uns"] == before["uns"]
    for slot in _ANNOTATION_SLOTS:
        assert set(after[slot]) == set(before[slot]), f"{slot} keys"
        for k in before[slot]:
            _assert_matrix_equal(after[slot][k], before[slot][k], f"{slot}[{k!r}]")
    # The edit completed: its crash-rollback backup is gone.
    assert not p.with_suffix(p.suffix + ".tailbak").exists()


@pytest.mark.parametrize("case", sorted(_CASES))
def test_update_var_packed_on_a_cell_archive(tmp_path, case):
    """The issue's repro: drop a var column in place."""
    p = _write(tmp_path, case)
    before = _state(p)
    new_var = before["var"].drop(columns=["total_counts"])

    update_var_packed(p, new_var)

    after = _state(p)
    pd.testing.assert_frame_equal(after["var"], new_var)
    _assert_only_edited_changed(p, before, after, "var")


@pytest.mark.parametrize("case", sorted(_CASES))
def test_update_obs_packed_on_a_cell_archive(tmp_path, case):
    p = _write(tmp_path, case)
    before = _state(p)
    new_obs = before["obs"].copy()
    new_obs["qc_pass"] = np.arange(len(new_obs)) % 2 == 0

    update_obs_packed(p, new_obs)

    after = _state(p)
    pd.testing.assert_frame_equal(after["obs"], new_obs)
    _assert_only_edited_changed(p, before, after, "obs")


def test_update_var_packed_on_an_empty_cell_archive(tmp_path):
    """n_obs == 0: x_payload is 0 bytes long and shares its offset with x_offsets."""
    var = pd.DataFrame({"total_counts": np.arange(30)}, index=[f"g{j}" for j in range(30)])
    p = tmp_path / "empty.csad"
    cellstream.write_sharded(
        ad.AnnData(X=sp.csr_matrix((0, 30), dtype="uint16"), obs=pd.DataFrame(index=[]), var=var),
        p,
        layout="cell",
        codec="pfordelta",
    )
    before = _state(p)
    payload, offsets = before["entries"][:2]
    assert payload["length"] == 0 and payload["offset"] == offsets["offset"]
    new_var = before["var"].drop(columns=["total_counts"])

    update_var_packed(p, new_var)

    after = _state(p)
    pd.testing.assert_frame_equal(after["var"], new_var)
    _assert_only_edited_changed(p, before, after, "var")


def test_update_obs_packed_on_a_push_archive_with_an_integer_obs_index(tmp_path):
    """The push writer stores the obs it is handed: given a default integer index, obs.parquet
    keeps it as int64 while header.h5ad holds anndata's string obs_names. The edit compared the
    two indexes as they were, so it refused ``store.obs`` plus a column as a reordering.
    """
    from cellstream.cell.archive_writer import CellArchiveWriter

    a = make_cell_adata()
    p = tmp_path / "push.csad"
    w = CellArchiveWriter(p, n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16")
    w.add_block(a.X[:100])
    w.add_block(a.X[100:])
    w.finalize(obs=a.obs.reset_index(drop=True), var=a.var)
    before = _state(p)
    assert before["obs"].index.dtype == np.int64  # the case: not anndata's string names
    new_obs = before["obs"].copy()
    new_obs["qc_pass"] = np.arange(len(new_obs)) % 2 == 0

    update_obs_packed(p, new_obs)

    after = _state(p)
    pd.testing.assert_frame_equal(after["obs"], new_obs)
    _assert_only_edited_changed(p, before, after, "obs")


def test_update_obs_packed_refuses_a_reorder_of_labels_equal_as_strings(tmp_path):
    """The index check is exact: cast to strings on pandas 2, the label "nan" and a missing label
    are the same, so a check on string casts let a swap of those two rows through -- obs
    annotations moved onto each other's X rows. Both members keep the distinction, so both
    comparisons can be exact.
    """
    from cellstream.cell.archive_writer import CellArchiveWriter

    a = make_cell_adata(n_obs=6, n_vars=5)
    obs = a.obs.set_axis(pd.CategoricalIndex(["nan", None, "c2", "c3", "c4", "c5"]))
    p = tmp_path / "push.csad"
    w = CellArchiveWriter(p, n_vars=a.n_vars, codec="pfordelta", value_dtype="uint16")
    w.add_block(a.X)
    w.finalize(obs=obs, var=a.var)
    stored = _state(p)["obs"]
    swapped = stored.iloc[[1, 0, 2, 3, 4, 5]]
    # The case: the swap exchanges a missing label and the label "nan". (Not asserted through
    # astype(str), which keeps the missing value on pandas 3. codex.)
    assert pd.isna(swapped.index[0]) and swapped.index[1] == "nan"
    before = p.read_bytes()

    with pytest.raises(ValueError, match="index does not match"):
        update_obs_packed(p, swapped)

    assert p.read_bytes() == before


def test_interrupted_cell_edit_rolls_back_on_open(tmp_path, monkeypatch):
    """A crash mid-rewrite leaves the archive cut at shard_region_end plus its ``.tailbak``; the
    next open restores the old tail (section 2.4). That path could not run on a cell archive
    before #338 -- every edit was refused before the backup was taken.
    """
    import cellstream.packed.writer as W

    p = _write(tmp_path, "pfordelta-grouped")
    before = _state(p)

    def _crash(*args, **kwargs):
        raise RuntimeError("crash mid-rewrite")

    monkeypatch.setattr(W, "_finalize_tail", _crash)
    with pytest.raises(RuntimeError, match="crash mid-rewrite"):
        update_var_packed(p, before["var"].drop(columns=["total_counts"]))
    monkeypatch.undo()
    assert p.with_suffix(p.suffix + ".tailbak").exists()

    after = _state(p)  # CellStore's open recovers the tail
    pd.testing.assert_frame_equal(after["var"], before["var"])
    _assert_only_edited_changed(p, before, after, "var")


# ------------------------------------------------- the group_by column (#339)
#
# groups.json -- the spans read_group() / group_spans() serve -- is carried over byte-for-byte, so
# an obs edit must keep the labels those spans were planned from. They are compared the way the
# writer names groups, every missing flavour being one label (#294) -- kept apart from the literal
# "nan" the writer also maps there.


def _rename_a_group(col):
    """The issue's repro: every row of one group relabelled."""
    return col.replace({"t01": "t01_renamed"})


def _move_a_row(col):
    """One row into another existing group: no new label appears, yet the span the row sits in no
    longer holds only its own label."""
    col = col.copy()
    col.iloc[int(np.flatnonzero(col.to_numpy() == "t01")[0])] = "t02"
    return col


def _mark_a_row_missing(col):
    col = col.copy()
    col.iloc[int(np.flatnonzero(col.to_numpy() == "t01")[0])] = None
    return col


@pytest.mark.parametrize(
    "relabel", [_rename_a_group, _move_a_row, _mark_a_row_missing], ids=lambda f: f.__name__
)
def test_update_obs_packed_refuses_relabelling_the_group_by_column(tmp_path, relabel):
    p = _write(tmp_path, "pfordelta-grouped")
    obs = _state(p)["obs"]
    new_obs = obs.assign(target_gene=relabel(obs["target_gene"]))
    before = p.read_bytes()

    with pytest.raises(ValueError, match="changes the group label of"):
        update_obs_packed(p, new_obs)

    assert p.read_bytes() == before


def test_update_obs_packed_refuses_dropping_the_group_by_column(tmp_path):
    p = _write(tmp_path, "pfordelta-grouped")
    obs = _state(p)["obs"]
    before = p.read_bytes()

    with pytest.raises(ValueError, match="has no 'target_gene' column"):
        update_obs_packed(p, obs.drop(columns=["target_gene"]))

    assert p.read_bytes() == before


def test_update_obs_packed_accepts_the_group_by_column_in_another_dtype(tmp_path):
    """Labels are compared, not stored values: a categorical holding the same labels leaves every
    group as it was."""
    p = _write(tmp_path, "pfordelta-grouped")
    before = _state(p)
    new_obs = before["obs"].assign(target_gene=before["obs"]["target_gene"].astype("category"))

    update_obs_packed(p, new_obs)

    after = _state(p)
    pd.testing.assert_frame_equal(after["obs"], new_obs)
    _assert_only_edited_changed(p, before, after, "obs")


def _write_with_t03_as(tmp_path, value):
    """A grouped cell archive whose t03 rows carry ``value`` -- a missing one, or a literal."""
    a = make_cell_adata()
    col = a.obs["target_gene"].astype(object)
    col[col == "t03"] = value
    a.obs["target_gene"] = col
    p = tmp_path / "m.csad"
    cellstream.write_sharded(a, p, layout="cell", codec="pfordelta", **_GROUPED)
    return p


@pytest.mark.parametrize("flavour", [None, np.nan, pd.NA], ids=["None", "NaN", "NA"])
def test_update_obs_packed_accepts_another_missing_flavour_in_the_group_by_column(
    tmp_path, flavour
):
    """#294: the writer names every missing value, whatever its flavour, as the one group "nan",
    so writing a missing label as another flavour moves no row out of that group."""
    p = _write_with_t03_as(tmp_path, None)
    before = _state(p)
    stored = before["obs"]["target_gene"]
    missing = stored.isna().to_numpy()
    assert missing.any()  # the case: the archive has a "nan" group
    new = stored.astype(object)
    new[missing] = flavour

    update_obs_packed(p, before["obs"].assign(target_gene=new))

    _assert_only_edited_changed(p, before, _state(p), "obs")


@pytest.mark.parametrize(
    ("stored_as", "edited_to"), [(None, "nan"), ("nan", None)], ids=["missing-to-literal", "back"]
)
def test_update_obs_packed_refuses_trading_a_missing_label_for_the_literal_nan(
    tmp_path, stored_as, edited_to
):
    """One group to this writer, but not to one before #294, which named missing rows 'None' or
    '<NA>': the literal "nan" would then be a label no record carries. A partial swap would also
    leave a column holding both, which no writer accepts (validate_group_label_collision)."""
    p = _write_with_t03_as(tmp_path, stored_as)
    obs = _state(p)["obs"]
    col = obs["target_gene"]
    rows = (col.isna() if stored_as is None else col == "nan").to_numpy()
    assert rows.any()  # the case: the t03 rows carry stored_as
    new = col.astype(object)
    new[rows] = edited_to
    before = p.read_bytes()

    with pytest.raises(ValueError, match="changes the group label of"):
        update_obs_packed(p, obs.assign(target_gene=new))

    assert p.read_bytes() == before


def _write_int_categorical(tmp_path):
    """Grouped by an integer categorical with a missing value -- on pandas 2 header.h5ad keeps it
    categorical ('1') while obs.parquet reads it back as float64 ('1.0')."""
    a = make_cell_adata()
    codes = (np.arange(a.n_obs) % 3).astype(np.int8)
    codes[::7] = -1
    a.obs["batch"] = pd.Categorical.from_codes(codes, categories=pd.Index([0, 1, 2], dtype="int64"))
    p = tmp_path / "b.csad"
    cellstream.write_sharded(a, p, layout="cell", codec="pfordelta", group_by="batch")
    return p


@pytest.mark.parametrize("copy", ["obs.parquet", "header.h5ad"])
def test_update_obs_packed_accepts_the_column_from_either_stored_copy(tmp_path, copy):
    """The archive stores obs twice, and the copies can name the same groups differently: on
    pandas 2 an integer categorical with a missing value stays categorical in header.h5ad ('1')
    but reads back from obs.parquet as float64 ('1.0'). An unchanged column taken from either --
    the obs a reader returns is obs.parquet's -- keeps every group as written, so both pass.
    (On pandas 3 the two copies agree, and this is the plain unchanged-column case.)"""
    from cellstream.packed.reader import PackedArchive

    p = _write_int_categorical(tmp_path)
    before = _state(p)
    obs = before["obs"] if copy == "obs.parquet" else PackedArchive(p).header_adata().obs

    update_obs_packed(p, obs.assign(qc_pass=True))

    _assert_only_edited_changed(p, before, _state(p), "obs")


def test_update_obs_packed_reports_a_relabel_against_the_closer_copy(tmp_path):
    """One row relabelled in obs.parquet's copy: the refusal counts that one row and names it.
    Measured against the header's copy instead, every row would differ ('0' against '0.0') on
    pandas 2, and the first one reported would be a row the edit never touched. (codex.)"""
    p = _write_int_categorical(tmp_path)
    obs = _state(p)["obs"]
    i = int(np.flatnonzero(obs["batch"].to_numpy() == 0)[3])
    new = obs["batch"].copy()
    new.iloc[i] = 1.0
    before = p.read_bytes()

    with pytest.raises(ValueError) as ei:
        update_obs_packed(p, obs.assign(batch=new))

    msg = str(ei.value)
    assert f"changes the group label of 1 of {len(obs)} rows" in msg
    assert f"(first: row {i}, {obs.index[i]!r}," in msg
    assert p.read_bytes() == before


def test_update_obs_packed_refuses_a_match_with_a_copy_that_merges_groups(tmp_path, monkeypatch):
    """obs.parquet's copy counts only while it groups the rows as the header's does. A read-back
    lossy enough to merge two groups -- on pandas 2, integer categories 2**53 and 2**53 + 1 come
    back as one float64 -- would otherwise let an edit write that merge into both copies while
    groups.json still holds the two groups. Simulated here by merging t02 into t01 on read."""
    import cellstream.packed.writer as W

    p = _write(tmp_path, "pfordelta-grouped")
    obs = _state(p)["obs"]
    merged = obs["target_gene"].replace({"t02": "t01"})
    real = W._parquet_obs_column
    monkeypatch.setattr(
        W, "_parquet_obs_column", lambda pa, name: real(pa, name).replace({"t02": "t01"})
    )
    before = p.read_bytes()

    with pytest.raises(ValueError, match="against the stored column in header.h5ad"):
        update_obs_packed(p, obs.assign(target_gene=merged))

    assert p.read_bytes() == before


def test_update_obs_packed_on_an_ungrouped_archive_may_drop_any_column(tmp_path):
    """No group_by, no groups.json: nothing in the archive depends on target_gene."""
    p = _write(tmp_path, "pfordelta")
    before = _state(p)
    new_obs = before["obs"].drop(columns=["target_gene"])

    update_obs_packed(p, new_obs)

    after = _state(p)
    pd.testing.assert_frame_equal(after["obs"], new_obs)
    _assert_only_edited_changed(p, before, after, "obs")


def test_update_obs_packed_refuses_an_archive_whose_obs_already_lost_the_column(
    tmp_path, monkeypatch
):
    """An edit made before this check could drop the group_by column. The labels groups.json was
    planned from are then gone, so no new column can be checked against them: refused, even one
    that restores the column."""
    import cellstream.packed.writer as W

    p = _write(tmp_path, "pfordelta-grouped")
    obs = _state(p)["obs"]
    monkeypatch.setattr(W, "_refuse_group_by_change", lambda *a, **k: None)
    update_obs_packed(p, obs.drop(columns=["target_gene"]))  # what an edit before #339 did
    monkeypatch.undo()
    before = p.read_bytes()

    with pytest.raises(ValueError, match="stored obs has no 'target_gene' column"):
        update_obs_packed(p, obs)

    assert p.read_bytes() == before


# ------------------------------------------------- CellStore.update_obs / update_var


def test_cellstore_update_var_refreshes_the_store(tmp_path):
    """The store that made the edit shows it at once, its cached var included, and X keeps
    decoding through the store's existing mmap: nothing below shard_region_end moved."""
    p = _write(tmp_path, "pfordelta-grouped")
    before = _state(p)
    store = open_cell(p)
    try:
        assert "total_counts" in store.var.columns  # cached before the edit
        spans = store.group_spans()

        store.update_var(store.var.drop(columns=["total_counts"]))

        assert "total_counts" not in store.var.columns
        _assert_matrix_equal(store.gather_rows(np.arange(store.n_obs)), before["X"], "X")
        assert store.group_spans() == spans
        pd.testing.assert_frame_equal(store.obs, before["obs"])
    finally:
        store.close()
    after = _state(p)
    pd.testing.assert_frame_equal(after["var"], before["var"].drop(columns=["total_counts"]))
    _assert_only_edited_changed(p, before, after, "var")


def test_cellstore_update_obs_shows_through_its_view(tmp_path):
    """An AnnDataView reads obs through its store, so the store's edit is the view's."""
    p = _write(tmp_path, "pfordelta")
    store = open_cell(p)
    try:
        view = store.to_anndata(backed=True)
        new_obs = view.obs.copy()
        new_obs["qc_pass"] = True

        store.update_obs(new_obs)

        pd.testing.assert_frame_equal(view.obs, new_obs)
    finally:
        store.close()


def test_cellstore_refused_edit_changes_nothing(tmp_path):
    p = _write(tmp_path, "pfordelta")
    store = open_cell(p)
    try:
        obs = store.obs.copy()
        before = p.read_bytes()
        with pytest.raises(ValueError, match="index does not match"):
            store.update_obs(obs.iloc[::-1])
        assert p.read_bytes() == before
        pd.testing.assert_frame_equal(store.obs, obs)
    finally:
        store.close()


def test_cellstore_update_obs_refuses_relabelling_its_groups(tmp_path):
    """#339 through the store: refused before the tail is touched, so the store keeps serving the
    groups and the obs it had."""
    p = _write(tmp_path, "pfordelta-grouped")
    store = open_cell(p)
    try:
        obs = store.obs.copy()
        spans = store.group_spans()
        before = p.read_bytes()
        with pytest.raises(ValueError, match="changes the group label of"):
            store.update_obs(obs.assign(target_gene=_rename_a_group(obs["target_gene"])))
        assert p.read_bytes() == before
        assert store.group_spans() == spans
        pd.testing.assert_frame_equal(store.obs, obs)
    finally:
        store.close()


def test_cellstore_edit_interrupted_mid_rewrite_is_rolled_back(tmp_path, monkeypatch):
    """A crash after the truncation: the store re-reads its footer on the way out, which applies
    the ``.tailbak`` -- so the store keeps serving the old metadata, not a torn tail."""
    import cellstream.packed.writer as W

    p = _write(tmp_path, "pfordelta-grouped")
    before = _state(p)

    def _crash(*args, **kwargs):
        raise RuntimeError("crash mid-rewrite")

    store = open_cell(p)
    try:
        monkeypatch.setattr(W, "_finalize_tail", _crash)
        with pytest.raises(RuntimeError, match="crash mid-rewrite"):
            store.update_var(store.var.drop(columns=["total_counts"]))
        monkeypatch.undo()

        assert not p.with_suffix(p.suffix + ".tailbak").exists()
        pd.testing.assert_frame_equal(store.var, before["var"])
        _assert_matrix_equal(store.gather_rows(np.arange(store.n_obs)), before["X"], "X")
    finally:
        store.close()
    _assert_only_edited_changed(p, before, _state(p), "var")
