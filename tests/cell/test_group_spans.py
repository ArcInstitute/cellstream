"""group_spans() / reference_row_count() — the public accessors from #248."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cellstream
from cellstream.cell import GroupSpan
from cellstream.cell import format as fmt
from cellstream.cell.writer import write_cell_archive
from tests.cell.conftest import make_cell_adata

# Source label sets for the CRAFTED colliding archives (see `_crafted_duplicate`). These are
# ordinary, non-colliding columns: the duplicate label is introduced afterwards by rewriting
# groups.json, not by coaxing the writer into producing one.
#
# Until #294 these fixtures were built from a literal "<NA>" beside pd.NA, relying on both
# stringifying to '<NA>' on pandas 2. Since #294 every missing flavour normalizes to one 'nan'
# and the literal keeps its own name, so that no longer collides -- and a literal "nan" beside
# a missing value is now rejected at write time. For the common obs dtypes the writer can no
# longer MAKE a duplicate label through the public API at all. The read-side guards below still
# have to hold, because archives written by earlier versions can carry duplicate labels and
# nothing migrates them.
LABELS_A_B_C = pd.array(["a"] * 8 + ["b"] * 8 + ["c"] * 8, dtype="string")
LABELS_R_A = pd.array(["r"] * 8 + ["a"] * 8, dtype="string")
LABELS_R_A_B = pd.array(["r"] * 8 + ["a"] * 8 + ["b"] * 8, dtype="string")


def _write(tmp_path, labels, name="g.shad", **kw):
    """Write a cell archive whose obs['grp'] is exactly `labels` (dtype preserved)."""
    a = make_cell_adata(n_obs=len(labels))
    a.obs["grp"] = (
        labels
        if isinstance(labels, pd.api.extensions.ExtensionArray)
        else np.asarray(labels, dtype=object)
    )
    out = tmp_path / name
    write_cell_archive(a, out, **kw)
    return out


def _write_empty(tmp_path, name="zero.shad", **kw):
    """A grouped archive with n_obs == 0 — reachable, and the one case where `groups` is []."""
    import anndata as ad

    a = ad.AnnData(
        X=sp.csr_matrix((0, 20), dtype=np.uint16),
        obs=pd.DataFrame({"grp": pd.Series([], dtype=object)}, index=pd.Index([], dtype=str)),
    )
    a.var_names = [f"g{i:03d}" for i in range(20)]
    out = tmp_path / name
    write_cell_archive(a, out, **kw)
    return out


def _record(store):
    return json.loads(store._packed.member_bytes(fmt.GROUPS))


def _duplicated_label(store) -> str:
    """The label carried by more than one group record, asserting one exists.

    The colliding archives are CRAFTED (see ``_crafted_duplicate``): ordinary groups are
    written, then one record in ``groups.json`` is relabelled to match another. Spans are
    untouched, so the records still tile ``[0, n_obs)`` -- the archive is exactly what an
    older writer could have produced, and the collision shape is fixed rather than
    pandas-version-dependent.

    Asserting here rather than returning ``None`` means a fixture that stopped colliding fails
    loudly instead of letting a test pass vacuously -- which is how the previous, writer-produced
    fixtures were caught when #294 stopped them colliding.
    """
    labels = [g["label"] for g in _record(store)["groups"]]
    dupes = sorted({lab for lab in labels if labels.count(lab) > 1})
    assert dupes, f"FIXTURE NO LONGER COLLIDES: every label is unique ({labels})"
    return dupes[0]


def _assert_csr_identical(got, exp):
    """Full sparse equality — indptr and dtype included. Comparing only indices/data would pass
    for two matrices with different ROW partitioning."""
    got, exp = got.tocsr(), exp.tocsr()
    got.sort_indices()
    exp.sort_indices()
    assert got.shape == exp.shape
    assert got.dtype == exp.dtype
    np.testing.assert_array_equal(got.indptr, exp.indptr)
    np.testing.assert_array_equal(got.indices, exp.indices)
    np.testing.assert_array_equal(got.data, exp.data)


# ---------------------------------------------------------------------------------------
# Task 1 — GroupSpan + group_spans()
# ---------------------------------------------------------------------------------------
def test_group_spans_matches_groups_json_in_order(tmp_path, engine_env):
    out = _write(tmp_path, [f"t{i % 4:02d}" for i in range(40)], group_by="grp")
    store = cellstream.open(out)
    try:
        want = [(g["label"], g["start"], g["stop"]) for g in _record(store)["groups"]]
        assert [(s.label, s.start, s.stop) for s in store.group_spans()] == want
    finally:
        store.close()


def test_group_spans_preserves_duplicate_labels(tmp_path, engine_env):
    """The whole reason the return type is a LIST: a dict would drop all but one of these.

    Crafted archive, records a / b / a -- the two "a" records are NON-adjacent, so a
    label-keyed mapping would silently drop one of them and its rows."""
    out = _crafted_duplicate(tmp_path, LABELS_A_B_C, dup_at=2, match_at=0, group_by="grp")
    store = cellstream.open(out)
    try:
        dup = _duplicated_label(store)
        spans = store.group_spans()
        # exactly groups.json, in order, nothing collapsed
        assert [(s.label, s.start, s.stop) for s in spans] == [
            (g["label"], g["start"], g["stop"]) for g in _record(store)["groups"]
        ]
        carrying = [s for s in spans if s.label == dup]
        assert len(carrying) > 1
        # a label-keyed mapping would have kept ONE of these and lost the other rows
        assert sum(s.n_rows for s in carrying) > max(s.n_rows for s in carrying)
        assert len({s.label for s in spans}) < len(spans)
    finally:
        store.close()


def test_group_spans_n_rows(tmp_path, engine_env):
    out = _write(tmp_path, [f"t{i % 4:02d}" for i in range(40)], group_by="grp")
    store = cellstream.open(out)
    try:
        for s in store.group_spans():
            assert s.n_rows == s.stop - s.start
            assert store.read_group(s.label).shape[0] == s.n_rows
    finally:
        store.close()


def test_group_spans_empty_when_ungrouped(tmp_path, engine_env):
    out = _write(tmp_path, [f"t{i % 4:02d}" for i in range(40)])  # no group_by
    store = cellstream.open(out)
    try:
        assert not store._packed.has_member(fmt.GROUPS)
        assert not store.is_grouped
        assert store.group_spans() == []
    finally:
        store.close()


def test_group_spans_on_a_zero_row_grouped_archive(tmp_path, engine_env):
    """n_obs == 0 with group_by set is legal and writes {"reference": null, "groups": []} —
    the ONE archive where an empty `groups` list is correct."""
    out = _write_empty(tmp_path, group_by="grp")
    store = cellstream.open(out)
    try:
        assert store.n_obs == 0
        assert store.is_grouped
        assert store._packed.has_member(fmt.GROUPS)
        assert store.group_spans() == []
    finally:
        store.close()


def test_group_spans_is_a_fresh_immutable_list(tmp_path, engine_env):
    """Mutating what a caller got back must not corrupt the store's cached record."""
    out = _write(tmp_path, [f"t{i % 4:02d}" for i in range(40)], group_by="grp")
    store = cellstream.open(out)
    try:
        first = store.group_spans()
        first.append(GroupSpan("injected", 0, 40))
        del first[0]
        assert [s.label for s in store.group_spans()] == ["t00", "t01", "t02", "t03"]
        with pytest.raises(AttributeError):
            store.group_spans()[0].start = 99  # NamedTuple is immutable
    finally:
        store.close()


def test_group_span_is_a_plain_tuple(tmp_path, engine_env):
    out = _write(tmp_path, [f"t{i % 4:02d}" for i in range(40)], group_by="grp")
    store = cellstream.open(out)
    try:
        label, start, stop = store.group_spans()[0]
        assert (label, start, stop) == ("t00", 0, 10)
    finally:
        store.close()


def test_group_spans_exported():
    import cellstream.cell

    assert "GroupSpan" in cellstream.cell.__all__
    assert cellstream.GroupSpan is cellstream.cell.GroupSpan


# ---------------------------------------------------------------------------------------
# Task 2 — groups.json / manifest validation
# ---------------------------------------------------------------------------------------
def _repack(
    tmp_path,
    src,
    mutate_groups=None,
    drop_groups=False,
    raw_groups=None,
    mutate_manifest=None,
    name="bad.shad",
):
    """Rebuild an archive from its own members, optionally rewriting, replacing with raw bytes,
    or dropping groups.json, and optionally rewriting the manifest.

    Repacked rather than byte-patched, so every other member is exactly what the writer wrote.
    """
    from cellstream.manifest import dump_manifest
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    stage = tmp_path / f"stage_{name}"
    stage.mkdir()
    arc = PackedArchive(src)
    manifest = dict(arc.manifest)
    if mutate_manifest is not None:
        manifest = mutate_manifest(manifest)
    for member in arc.members:
        mname = member["name"]
        if mname == fmt.MANIFEST:
            continue
        if mname == fmt.GROUPS:
            if drop_groups:
                continue
            data = (
                raw_groups
                if raw_groups is not None
                else json.dumps(mutate_groups(json.loads(arc.member_bytes(mname)))).encode()
            )
        else:
            data = arc.member_bytes(mname)
        (stage / mname).write_bytes(data)
    dump_manifest(stage / fmt.MANIFEST, manifest)
    dst = tmp_path / name
    pack(stage, dst, overwrite=True, _delete_sources=True)
    return dst


def _crafted_duplicate(tmp_path, labels, *, dup_at, match_at=0, name="dup.shad", **kw):
    """An archive whose groups.json carries TWO records with the same label.

    Since #294 the writer normalizes every missing flavour to one 'nan' and rejects a literal
    'nan' beside missing values, so it can no longer MAKE a duplicate label for the common obs
    dtypes. These read-side guards still have to hold for archives written by earlier versions,
    so the fixture is crafted: write ordinary groups, then relabel record `dup_at` to match
    record `match_at`. Spans are untouched, so the records still tile [0, n_obs).

    `match_at` is not always 0: test_reference_row_count_with_a_split_colliding_label needs the
    LEADING record to stay the reference "r" while two NON-reference records collide.
    """
    src = _write(tmp_path, labels, name=f"src_{name}", **kw)

    def mutate(rec):
        rec["groups"][dup_at]["label"] = rec["groups"][match_at]["label"]
        return rec

    return _repack(tmp_path, src, mutate_groups=mutate, name=name)


def _last(rec, **kw):
    rec["groups"][-1].update(kw)
    return rec


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: _last(r, stop=10_000), "outside"),  # stop past n_obs
        (lambda r: _last(r, start=-1), "outside"),  # negative start
        (lambda r: _last(r, start=35, stop=30), "outside"),  # start > stop
        (lambda r: _last(r, start=40, stop=40), "empty"),  # zero-row record
        (lambda r: _last(r, stop="40"), "non-integer"),  # string stop
        (lambda r: _last(r, stop=True), "non-integer"),  # bool is not an int here
        (lambda r: _last(r, label=7), "string 'label'"),  # non-string label
        (lambda r: {"reference": None, "groups": {}}, "'groups' list"),  # groups not a list
        (lambda r: _last(r, start=31), "tile"),  # gap / overlap
        (lambda r: {**r, "groups": r["groups"][::-1]}, "tile"),  # out of storage order
        (lambda r: {**r, "groups": r["groups"][:-1]}, "tile"),  # does not reach n_obs
        (lambda r: {**r, "reference": "ghost"}, "no group record"),  # ref label with no record
        (lambda r: {**r, "reference": ["t00", "ghost"]}, "no group record"),  # PARTIALLY missing
        (lambda r: {**r, "reference": 7}, "must be null"),  # ref not str/list/None
        (lambda r: {**r, "reference": []}, "must be null"),  # empty list, never written
        (lambda r: {**r, "reference": ["t00", 7]}, "must be null"),  # list with a non-string
        (lambda r: {**r, "reference": None}, "disagrees"),  # deleted vs the manifest
        # invented vs the manifest. It must be a set that PASSES membership and prefix, or the
        # earlier checks fire first and this stops testing the cross-check: "t00","t01" are
        # records 0 and 1, so they are present and leading — and still disagree with the
        # manifest's "t00".
        (lambda r: {**r, "reference": ["t00", "t01"]}, "disagrees"),
    ],
)
def test_corrupt_groups_json_is_rejected(tmp_path, engine_env, mutate, match):
    from cellstream.errors import IncompatibleSchemaError

    src = _write(
        tmp_path,
        [f"t{i % 4:02d}" for i in range(40)],
        name="ok.shad",
        group_by="grp",
        reference=["t00"],
    )
    bad = _repack(tmp_path, src, mutate)
    store = cellstream.open(bad)
    try:
        with pytest.raises(IncompatibleSchemaError, match=match):
            store.group_spans()
    finally:
        store.close()


@pytest.mark.parametrize(
    ("mutate_manifest", "match"),
    [
        (lambda m: {**m, "group_by": 7}, "'group_by' must be null"),
        (lambda m: {**m, "reference": 7}, "'reference' must be null"),
        (lambda m: {**m, "reference": []}, "'reference' must be null"),
        # a dict is the hazard the docstring names: sorted({"t00": 1}) == ["t00"], so without
        # this check it would normalize to the SAME tuple as the string "t00" and pass.
        (lambda m: {**m, "reference": {"t00": 1}}, "'reference' must be null"),
        (lambda m: {**m, "reference": ["t00", 7]}, "'reference' must be null"),
        (lambda m: {**m, "group_by": None, "reference": "t00"}, "names reference"),
    ],
)
def test_corrupt_manifest_grouping_fields_are_rejected(
    tmp_path, engine_env, mutate_manifest, match
):
    """The cross-check reads the manifest, so the manifest's own grouping fields must be typed
    before `_normalize_reference` touches them — an int would otherwise raise a bare TypeError,
    and a dict would normalize to the same tuple as a string."""
    from cellstream.errors import IncompatibleSchemaError

    src = _write(
        tmp_path,
        [f"t{i % 4:02d}" for i in range(40)],
        name="okm.shad",
        group_by="grp",
        reference=["t00"],
    )
    bad = _repack(
        tmp_path, src, lambda r: r, mutate_manifest=mutate_manifest, name="badmanifest.shad"
    )
    store = cellstream.open(bad)
    try:
        with pytest.raises(IncompatibleSchemaError, match=match):
            store.group_spans()
    finally:
        store.close()


def test_reference_that_is_not_the_leading_block_is_rejected(tmp_path, engine_env):
    """Swap two record LABELS so 't00' (the manifest's reference) lands in the middle. The
    reference VALUE still agrees with the manifest — only the prefix invariant is violated."""
    from cellstream.errors import IncompatibleSchemaError

    def swap(rec):
        rec["groups"][0]["label"], rec["groups"][2]["label"] = (
            rec["groups"][2]["label"],
            rec["groups"][0]["label"],
        )
        return rec

    src = _write(
        tmp_path,
        [f"t{i % 4:02d}" for i in range(40)],
        name="ok_lead.shad",
        group_by="grp",
        reference=["t00"],
    )
    bad = _repack(tmp_path, src, swap, name="notleading.shad")
    store = cellstream.open(bad)
    try:
        with pytest.raises(IncompatibleSchemaError, match="leading"):
            store.group_spans()
    finally:
        store.close()


def test_unparseable_groups_json_is_rejected(tmp_path, engine_env):
    """The error contract is IncompatibleSchemaError — not a raw JSONDecodeError."""
    from cellstream.errors import IncompatibleSchemaError

    src = _write(
        tmp_path, [f"t{i % 4:02d}" for i in range(40)], name="ok_json.shad", group_by="grp"
    )
    bad = _repack(tmp_path, src, raw_groups=b"{not json", name="badjson.shad")
    store = cellstream.open(bad)
    try:
        with pytest.raises(IncompatibleSchemaError, match="groups.json"):
            store.group_spans()
    finally:
        store.close()


def test_missing_groups_json_on_a_grouped_archive_is_rejected(tmp_path, engine_env):
    """Every writer emits groups.json whenever group_by is set, so a missing member means a
    damaged archive — not an ungrouped one. Silently reporting 'no groups, no reference' for an
    archive that has both is exactly the silent-wrong-answer class this PR exists to close."""
    from cellstream.errors import IncompatibleSchemaError

    src = _write(
        tmp_path,
        [f"t{i % 4:02d}" for i in range(40)],
        name="ok2.shad",
        group_by="grp",
        reference=["t00"],
    )
    bad = _repack(tmp_path, src, drop_groups=True, name="nogroups.shad")
    store = cellstream.open(bad)
    try:
        assert store.is_grouped  # manifest still says grouped
        with pytest.raises(IncompatibleSchemaError, match="no groups.json"):
            store.group_spans()
    finally:
        store.close()


def test_groups_json_on_an_ungrouped_manifest_is_rejected(tmp_path, engine_env):
    """The other direction of "present iff group_by": a groups.json member on a manifest that
    declares no grouping. Both manifest fields must be cleared, or the manifest validator's
    "no group_by but names reference" arm fires first and proves something else."""
    from cellstream.errors import IncompatibleSchemaError

    src = _write(
        tmp_path,
        [f"t{i % 4:02d}" for i in range(40)],
        name="okg.shad",
        group_by="grp",
        reference=["t00"],
    )
    bad = _repack(
        tmp_path,
        src,
        lambda r: r,
        mutate_manifest=lambda m: {**m, "group_by": None, "reference": None},
        name="ungrouped_with_groups.shad",
    )
    store = cellstream.open(bad)
    try:
        assert not store.is_grouped
        with pytest.raises(IncompatibleSchemaError, match="declares no group_by"):
            store.group_spans()
    finally:
        store.close()


def test_valid_groups_json_still_loads(tmp_path, engine_env):
    """The repack itself must not be what breaks: an unmodified round-trip still reads."""
    src = _write(
        tmp_path,
        [f"t{i % 4:02d}" for i in range(40)],
        name="ok3.shad",
        group_by="grp",
        reference=["t00"],
    )
    same = _repack(tmp_path, src, lambda r: r, name="same.shad")
    store = cellstream.open(same)
    try:
        spans = store.group_spans()
        assert spans[0].start == 0 and spans[-1].stop == store.n_obs
        assert store.read_group("t00").shape[0] == spans[0].n_rows
    finally:
        store.close()


def test_colliding_labels_still_validate(tmp_path, engine_env):
    """Duplicate labels are LEGAL — the validator must not reject them."""
    out = _crafted_duplicate(
        tmp_path, LABELS_A_B_C, dup_at=2, match_at=0, group_by="grp", name="collide_ok.shad"
    )
    store = cellstream.open(out)
    try:
        _duplicated_label(store)  # precondition: the archive really does collide
        spans = store.group_spans()  # must not raise
        assert spans[0].start == 0 and spans[-1].stop == store.n_obs
    finally:
        store.close()


def test_a_duplicated_reference_label_in_the_prefix_validates(tmp_path, engine_env):
    """Crafted: two records that repeat a REFERENCE label sit in the LEADING block. The prefix
    check must accept that.

    Records are r / r -- record 1 is relabelled from "a" to "r", so both carry the reference
    label. The precondition below pins that the duplicate really IS a reference label; without
    it the test would pass just as happily on an archive whose duplicate sits outside the
    prefix, which is a different (and already covered) case."""
    out = _crafted_duplicate(
        tmp_path,
        LABELS_R_A,
        dup_at=1,
        match_at=0,
        group_by="grp",
        reference=["r"],
        name="dupref.shad",
    )
    store = cellstream.open(out)
    try:
        ref = _record(store)["reference"]
        ref_labels = set([ref] if isinstance(ref, str) else ref)
        dup = _duplicated_label(store)
        assert dup is not None and dup in ref_labels, "the duplicate must BE a reference label"
        spans = store.group_spans()  # validation ran and ACCEPTED the archive
        hit = [i for i, s in enumerate(spans) if s.label in ref_labels]
        assert hit == list(range(len(hit))), "reference records must be the leading block"
        # the reference block is [0, R), so R is the last reference record's stop
        assert store.reference_row_count() == spans[hit[-1]].stop
        assert store.read_reference().shape[0] == store.reference_row_count()
    finally:
        store.close()


# ---------------------------------------------------------------------------------------
# Task 3 — reference_row_count() and the read_reference delegation
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kw", "expect"),
    [
        ({}, None),  # ungrouped
        ({"group_by": "grp"}, None),  # grouped, no reference
        ({"group_by": "grp", "reference": ["t00"]}, 10),  # single label
        ({"group_by": "grp", "reference": ["t00", "t01"]}, 20),  # multi label
        ({"group_by": "grp", "reference": ["t03"]}, 10),  # ref label sorts LAST
    ],
)
def test_reference_row_count_matches_read_reference(tmp_path, engine_env, kw, expect):
    out = _write(tmp_path, [f"t{i % 4:02d}" for i in range(40)], **kw)
    store = cellstream.open(out)
    try:
        got = store.reference_row_count()
        ref = store.read_reference()
        assert got == expect
        assert got == (None if ref is None else ref.shape[0])
    finally:
        store.close()


def test_reference_row_count_none_on_a_zero_row_grouped_archive(tmp_path, engine_env):
    out = _write_empty(tmp_path, name="zero2.shad", group_by="grp")
    store = cellstream.open(out)
    try:
        assert store.reference_row_count() is None
        assert store.read_reference() is None
    finally:
        store.close()


def test_reference_row_count_is_the_leading_block(tmp_path, engine_env):
    """The count must be the reference block's stop even when its label sorts last."""
    out = _write(tmp_path, [f"t{i % 4:02d}" for i in range(40)], group_by="grp", reference=["t03"])
    store = cellstream.open(out)
    try:
        spans = store.group_spans()
        assert (spans[0].label, spans[0].start, spans[0].stop) == ("t03", 0, 10)
        assert store.reference_row_count() == spans[0].stop
    finally:
        store.close()


def test_reference_row_count_with_a_split_colliding_label(tmp_path, engine_env):
    """The premise of the behaviour-preservation argument: a colliding label produces TWO
    records while a reference exists. The count must be the REFERENCE block, not the last
    colliding span's stop.

    Crafted records r / a / a -- record 0 stays the reference, and the duplicate is NOT a
    reference label, so a max() over the wrong records overshoots 8 by the whole archive."""
    out = _crafted_duplicate(
        tmp_path, LABELS_R_A_B, dup_at=2, match_at=1, group_by="grp", reference=["r"]
    )
    store = cellstream.open(out)
    try:
        spans = store.group_spans()
        dup = _duplicated_label(store)
        # the colliding label really is split into several records, and they are NOT the
        # reference -- so a naive max() over the wrong records would overshoot badly
        carrying = [s for s in spans if s.label == dup]
        assert len(carrying) > 1
        assert max(s.stop for s in carrying) > 8
        # ...yet the count is exactly the leading "r" block
        assert (spans[0].label, spans[0].start, spans[0].stop) == ("r", 0, 8)
        assert store.reference_row_count() == 8
        assert store.read_reference().shape[0] == 8
    finally:
        store.close()


def test_read_reference_delegates_to_reference_row_count(tmp_path, engine_env, monkeypatch):
    """STRUCTURAL, not behavioural. Every equality test above would also pass if
    read_reference kept its own independent derivation — and the two could then drift. This
    pins that there is exactly one derivation."""
    out = _write(tmp_path, [f"t{i % 4:02d}" for i in range(40)], group_by="grp", reference=["t00"])
    store = cellstream.open(out)
    try:
        calls = []
        real = type(store).reference_row_count

        def spy(self):
            calls.append(True)
            return real(self)

        monkeypatch.setattr(type(store), "reference_row_count", spy)
        assert store.read_reference().shape[0] == 10
        assert len(calls) == 1
    finally:
        store.close()


def test_writer_rejects_a_reference_mask_that_splits_a_label(tmp_path):
    """Why read_reference cannot serve a partially-referenced label: the whole-label-alignment
    guard rejects a bool reference mask that covers only part of one label, so such an archive
    never exists to be read.

    Deliberately uses a plain string column rather than a colliding one -- the guard groups on
    the stringified view, whose treatment of missing values is pandas-version-dependent (see
    `_duplicated_label`), and this invariant is not."""
    labels = ["a"] * 12 + ["b"] * 12
    a = make_cell_adata(n_obs=len(labels))
    a.obs["grp"] = np.asarray(labels, dtype=object)
    mask = np.zeros(len(labels), dtype=bool)
    mask[:6] = True  # only HALF of the 'a' rows
    with pytest.raises(ValueError, match="splits label"):
        write_cell_archive(a, tmp_path / "split.shad", group_by="grp", reference=mask)


# ---------------------------------------------------------------------------------------
# Task 4 — read_group fails loud on an ambiguous label
# ---------------------------------------------------------------------------------------
def test_read_group_raises_on_an_ambiguous_label(tmp_path, engine_env):
    """Several records carry one label; before the fix read_group silently returned the rows
    of only the LAST of them. Crafted records a / b / a."""
    out = _crafted_duplicate(tmp_path, LABELS_A_B_C, dup_at=2, match_at=0, group_by="grp")
    store = cellstream.open(out)
    try:
        dup = _duplicated_label(store)
        spans = [s for s in store.group_spans() if s.label == dup]
        assert len(spans) > 1
        # what the old dict-collapse would have returned, vs. what the label really covers
        assert spans[-1].n_rows < sum(s.n_rows for s in spans)
        with pytest.raises(ValueError, match="ambiguous") as exc:
            store.read_group(dup)
        msg = str(exc.value)
        assert "group_spans()" in msg
        for s in spans:  # every colliding span is named, so the caller can address them
            assert f"[{s.start}, {s.stop})" in msg
        with pytest.raises(ValueError, match="ambiguous"):
            store.read_group_adata(dup)
    finally:
        store.close()


def test_read_group_unique_label_unchanged_on_a_colliding_archive(tmp_path, engine_env):
    """Only the ambiguous label is affected — the UNIQUE label still reads exactly its own span.

    Crafted records a / b / a, so "b" is the unique one. The precondition pins that the archive
    really is colliding; without it this passes on any ordinary archive and tests nothing."""
    out = _crafted_duplicate(tmp_path, LABELS_A_B_C, dup_at=2, match_at=0, group_by="grp")
    store = cellstream.open(out)
    try:
        assert _duplicated_label(store) is not None
        span = next(s for s in store.group_spans() if s.label == "b")
        _assert_csr_identical(
            store.read_group("b"),
            store.gather_rows(np.arange(span.start, span.stop, dtype=np.int64)),
        )
    finally:
        store.close()


def test_read_group_unknown_label_still_raises_keyerror_with_hint(tmp_path, engine_env):
    out = _write(tmp_path, [f"t{i % 4:02d}" for i in range(40)], group_by="grp")
    store = cellstream.open(out)
    try:
        with pytest.raises(KeyError, match="not found") as exc:
            store.read_group("t0")
        assert "Near matches" in str(exc.value)
    finally:
        store.close()
