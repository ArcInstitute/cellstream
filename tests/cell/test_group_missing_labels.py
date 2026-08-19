"""#294 + #282 item 5, layout='cell' end to end: a missing group label names ONE group."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import cellstream
from cellstream.cell.writer import write_cell_archive

try:
    import anndata as ad
except ImportError:  # pragma: no cover
    pytest.skip("anndata required", allow_module_level=True)


def _adata(col, n=24):
    X = sp.random(n, 10, density=0.3, format="csr", random_state=0)
    X.data = np.abs(X.data * 100).astype(np.float32) + 1
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    # .values: assigning a Series would align on index and silently NaN the whole column.
    obs["grp"] = col.values if isinstance(col, pd.Series) else col
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(10)]))


def _spans(tmp_path, col, sort_order="natural", n=24):
    out = tmp_path / f"a_{sort_order}.shad"
    write_cell_archive(
        _adata(col, n), out, group_by="grp", sort_order=sort_order, data_dtype="float32"
    )
    return [(s.label, s.start, s.stop) for s in cellstream.open(str(out)).group_spans()]


MISSING_COLS = [
    pytest.param(
        pd.Series(pd.Categorical(["g0"] * 8 + ["g1"] * 8 + [None] * 8)), id="categorical-NaN"
    ),
    pytest.param(
        pd.Series(pd.array(["g0"] * 8 + ["g1"] * 8 + [pd.NA] * 8, dtype="string")), id="pd-NA"
    ),
    pytest.param(pd.Series(["g0"] * 8 + ["g1"] * 8 + [np.nan] * 8, dtype=object), id="object-nan"),
]


@pytest.mark.parametrize("col", MISSING_COLS)
@pytest.mark.parametrize("sort_order", ["natural", "alphabetical"])
def test_missing_rows_form_exactly_one_trailing_group(tmp_path, col, sort_order):
    """Exact spans, not just a count -- a count passes for any layout that happens to have one
    'nan' record, including a wrong one."""
    assert _spans(tmp_path, col, sort_order) == [("g0", 0, 8), ("g1", 8, 16), ("nan", 16, 24)]


@pytest.mark.parametrize("sort_order", ["natural", "alphabetical"])
def test_mixed_none_and_nan_form_one_group(tmp_path, sort_order):
    """Measured pre-fix, sort_order='alphabetical':
    [('None',0,4), ('g0',4,12), ('g1',12,20), ('nan',20,24)] -- the two NA runs are NOT
    adjacent. _sort_key has ALREADY stringified them by the time the planner sees them, so
    normalizing the label view alone leaves them two records with DIFFERENT names; only
    normalizing the sort key merges them into one run.
    """
    col = pd.Series(["g0"] * 8 + ["g1"] * 8 + [None] * 4 + [np.nan] * 4, dtype=object)
    assert _spans(tmp_path, col, sort_order) == [("g0", 0, 8), ("g1", 8, 16), ("nan", 16, 24)]


@pytest.mark.parametrize("sort_order", ["natural", "alphabetical"])
def test_read_group_nan_returns_exactly_the_missing_rows(tmp_path, sort_order):
    """Identity, not just a count: gather the same rows by position and compare."""
    col = pd.Series(pd.array(["g0"] * 8 + ["g1"] * 8 + [pd.NA] * 8, dtype="string"))
    out = tmp_path / "a.shad"
    write_cell_archive(
        _adata(col), out, group_by="grp", sort_order=sort_order, data_dtype="float32"
    )
    store = cellstream.open(str(out))
    span = next(s for s in store.group_spans() if s.label == "nan")
    by_label = store.read_group("nan")
    by_position = store.gather_rows(np.arange(span.start, span.stop, dtype=np.int64))
    assert by_label.shape == (8, 10)
    assert (by_label != by_position).nnz == 0
    # and they really are the rows whose obs label is missing
    assert store.obs["grp"].isna().to_numpy()[span.start : span.stop].all()


def test_missing_rows_as_reference_lead_the_archive(tmp_path):
    """The reader-side interaction: one merged 'nan' record must still satisfy the
    leading-reference-prefix invariant #248 validates on load."""
    col = pd.Series(pd.array(["g0"] * 8 + ["g1"] * 8 + [pd.NA] * 8, dtype="string"))
    out = tmp_path / "a.shad"
    write_cell_archive(_adata(col), out, group_by="grp", reference=["nan"], data_dtype="float32")
    store = cellstream.open(str(out))
    assert store.group_spans()[0].label == "nan"
    assert store.reference_row_count() == 8
    assert store.read_reference().shape[0] == 8


@pytest.mark.parametrize("sort_order", ["natural", "alphabetical"])
def test_literal_nan_plus_missing_raises_at_write(tmp_path, sort_order):
    col = pd.Series(pd.array(["nan"] * 8 + ["g1"] * 8 + [pd.NA] * 8, dtype="string"))
    with pytest.raises(ValueError, match=r"obs\['grp'\] holds both a literal 'nan' label"):
        write_cell_archive(
            _adata(col),
            tmp_path / "a.shad",
            group_by="grp",
            sort_order=sort_order,
            data_dtype="float32",
        )


def test_the_collision_is_caught_before_the_output_is_touched(tmp_path):
    """The preflight must run before any destructive step, or overwrite=True would delete a
    good archive and then fail."""
    good = pd.Series(pd.array(["g0"] * 12 + ["g1"] * 12, dtype="string"))
    out = tmp_path / "a.shad"
    write_cell_archive(_adata(good), out, group_by="grp", data_dtype="float32")
    before = out.read_bytes()
    bad = pd.Series(pd.array(["nan"] * 8 + ["g1"] * 8 + [pd.NA] * 8, dtype="string"))
    with pytest.raises(ValueError, match=r"both a literal 'nan' label"):
        write_cell_archive(_adata(bad), out, group_by="grp", overwrite=True, data_dtype="float32")
    assert out.read_bytes() == before, "the prior archive was damaged by a failed write"


def test_literal_nan_without_missing_still_writes(tmp_path):
    col = pd.Series(pd.array(["nan"] * 12 + ["g1"] * 12, dtype="string"))
    spans = _spans(tmp_path, col)
    assert sorted(lbl for lbl, _, _ in spans) == ["g1", "nan"]


def _legacize(tmp_path, src, name, new_label):
    """Rewrite the archive's missing-label 'nan' to `new_label` in groups.json AND the manifest.

    What a pre-#294 writer produced: the missing label was named by dtype, so a nullable-string
    column gave '<NA>' and an object+None column gave 'None'. Repacked, so offsets and the
    footer CRC are recomputed.
    """
    import json

    from cellstream.cell import format as fmt
    from cellstream.manifest import dump_manifest
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    stage = tmp_path / f"stage_{name}"
    stage.mkdir()

    def _relabel_reference(ref):
        # The writer stores a SINGLE reference label as a string and several as a LIST
        # (verified: reference=['r'] -> 'r', reference=['<NA>','a'] -> ['<NA>', 'a']). A
        # string-only rewrite silently skips the list form, which would leave the fixture
        # non-legacy and let its test pass while exercising nothing.
        if isinstance(ref, str):
            return new_label if ref == "nan" else ref
        if isinstance(ref, list):
            return [new_label if x == "nan" else x for x in ref]
        return ref

    arc = PackedArchive(src)
    manifest = dict(arc.manifest)
    manifest["reference"] = _relabel_reference(manifest.get("reference"))
    for member in arc.members:
        mname = member["name"]
        if mname == fmt.MANIFEST:
            continue
        data = arc.member_bytes(mname)
        if mname == fmt.GROUPS:
            rec = json.loads(data)
            for g in rec["groups"]:
                if g["label"] == "nan":
                    g["label"] = new_label
            rec["reference"] = _relabel_reference(rec.get("reference"))
            data = json.dumps(rec).encode()
        (stage / mname).write_bytes(data)
    dump_manifest(stage / fmt.MANIFEST, manifest)
    dst = tmp_path / name
    pack(stage, dst, overwrite=True, _delete_sources=True)
    return dst


@pytest.mark.parametrize("legacy_label", ["<NA>", "None"])
def test_concat_of_legacy_archives_keeps_a_missing_value_reference(tmp_path, legacy_label):
    """#294 regression: concat USED TO inherit the reference LABEL and re-resolve it.

    A pre-#294 archive recorded its missing label by dtype, and the new normalization no longer
    produces those names -- so re-resolving an inherited '<NA>' / 'None' matched nothing and
    concat failed on archives that concatenated fine before. Concat now preserves the validated
    positional prefix from each source instead, so the label's spelling no longer matters.
    """
    from cellstream.cell.archive_writer import concat_cell_archives

    col = pd.Series(pd.array([pd.NA] * 8 + ["b"] * 8, dtype="string"))
    src = tmp_path / "src.shad"
    write_cell_archive(
        _adata(col, n=16), src, group_by="grp", reference=["nan"], data_dtype="float32"
    )
    parts = [_legacize(tmp_path, src, f"legacy{i}.shad", legacy_label) for i in (1, 2)]
    out = tmp_path / "cat.shad"
    concat_cell_archives(parts, out)
    store = cellstream.open(str(out))
    # the missing rows are still the reference, and the output uses today's single sentinel
    assert [(s.label, s.start, s.stop) for s in store.group_spans()] == [
        ("nan", 0, 16),
        ("b", 16, 32),
    ]
    assert store.reference_row_count() == 16


def test_concat_of_a_legacy_collision_keeps_every_reference_row(tmp_path):
    """The SILENT half. A legacy archive could hold a literal '<NA>' AND real pd.NA under one
    reference label '<NA>'. That label still resolves -- to the literal rows only -- so
    re-resolving it dropped the missing rows from the reference set with no error at all.
    Measured before the positional fix: 32 reference rows where 48 were recorded.
    """
    from cellstream.cell.archive_writer import concat_cell_archives

    col = pd.Series(pd.array(["<NA>"] * 8 + ["a"] * 8 + [pd.NA] * 8, dtype="string"))
    src = tmp_path / "src.shad"
    write_cell_archive(
        _adata(col, n=24), src, group_by="grp", reference=["<NA>", "a"], data_dtype="float32"
    )
    # legacy naming: pd.NA was ALSO '<NA>', so all 24 rows are the reference prefix
    parts = [_legacize(tmp_path, src, f"coll{i}.shad", "<NA>") for i in (1, 2)]
    assert cellstream.open(str(parts[0])).reference_row_count() == 24
    out = tmp_path / "cat.shad"
    concat_cell_archives(parts, out)
    assert cellstream.open(str(out)).reference_row_count() == 48


def test_concat_group_by_override_does_not_inherit_the_old_positional_reference(tmp_path):
    """Positional preservation must be scoped to the SAME grouping axis.

    A caller can override group_by while leaving reference to inherit, and the sources'
    membership then means nothing on the new axis. Sources grouped by old=[r,r,x,x] with
    reference 'r' carry the mask [T,T,F,F]; on group_by='new' where new=[a,a,r,r] the reference
    must be 'r', not 'a'. Measured before the guard: the output's leading group was 'a'.
    """
    import scipy.sparse as sp_mod

    from cellstream.cell.archive_writer import concat_cell_archives

    X = sp_mod.random(4, 10, density=0.5, format="csr", random_state=0)
    X.data = np.abs(X.data * 100).astype(np.float32) + 1
    obs = pd.DataFrame(index=[f"c{i}" for i in range(4)])
    obs["old"] = ["r", "r", "x", "x"]
    obs["new"] = ["a", "a", "r", "r"]
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(10)]))
    src = tmp_path / "src.shad"
    write_cell_archive(a, src, group_by="old", reference=["r"], data_dtype="float32")

    out = tmp_path / "override.shad"
    concat_cell_archives([src, src], out, group_by="new")
    spans = [(s.label, s.start, s.stop) for s in cellstream.open(str(out)).group_spans()]
    assert spans[0][0] == "r", f"the new axis's 'r' must lead, got {spans}"

    # ...and on the SAME axis, positional preservation still applies
    same = tmp_path / "same.shad"
    concat_cell_archives([src, src], same)
    store = cellstream.open(str(same))
    assert store.group_spans()[0].label == "r"
    assert store.reference_row_count() == 4


def test_concat_propagates_a_corrupt_source_group_record(tmp_path):
    """The fail-closed gate. `_inherited_reference_mask` must NOT swallow the read-side
    validation error and quietly fall back to label resolution -- a source whose group record
    does not validate would then be republished whenever its manifest label happened to resolve.

    Discriminating by construction: the label 'r' DOES resolve against this obs, so restoring
    the bare `except Exception: return reference` makes CONCAT succeed silently -- and this
    test fail, which is the point. Verified by doing exactly that in place.
    """
    import json

    from cellstream.cell import format as fmt
    from cellstream.cell.archive_writer import concat_cell_archives
    from cellstream.errors import IncompatibleSchemaError
    from cellstream.manifest import dump_manifest
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    col = pd.Series(pd.array(["r"] * 8 + ["x"] * 8, dtype="string"))
    src = tmp_path / "src.shad"
    write_cell_archive(
        _adata(col, n=16), src, group_by="grp", reference=["r"], data_dtype="float32"
    )

    def _corrupt(name):
        stage = tmp_path / f"stage_{name}"
        stage.mkdir()
        arc = PackedArchive(src)
        for member in arc.members:
            mname = member["name"]
            if mname == fmt.MANIFEST:
                continue
            data = arc.member_bytes(mname)
            if mname == fmt.GROUPS:
                rec = json.loads(data)
                rec["groups"][0]["stop"] = "not-an-int"  # rejected by _validate_groups_record
                data = json.dumps(rec).encode()
            (stage / mname).write_bytes(data)
        dump_manifest(stage / fmt.MANIFEST, dict(arc.manifest))
        dst = tmp_path / name
        pack(stage, dst, overwrite=True, _delete_sources=True)
        return dst

    parts = [_corrupt(f"bad{i}.shad") for i in (1, 2)]
    with pytest.raises(IncompatibleSchemaError):
        concat_cell_archives(parts, tmp_path / "cat.shad")


def test_concat_rejects_compensating_per_part_obs_length_errors(tmp_path):
    """The per-part obs length check, distinguished from the aggregate one.

    The inherited reference mask is built from each part's declared n_obs, so two malformed
    parts whose declared and actual obs lengths differ in COMPENSATING directions (8/8 declared,
    7/9 actual) sum correctly and sail past an aggregate-only check -- while the mask and the
    payload keep 8/8 boundaries against obs's 7/9. Only a per-part check catches it.
    """

    import pyarrow.parquet as pq

    from cellstream.cell import format as fmt
    from cellstream.cell.archive_writer import concat_cell_archives
    from cellstream.manifest import dump_manifest
    from cellstream.packed.reader import PackedArchive
    from cellstream.packed.writer import _pack_directory as pack

    col = pd.Series(pd.array(["r"] * 4 + ["x"] * 4, dtype="string"))
    src = tmp_path / "src.shad"
    write_cell_archive(_adata(col, n=8), src, group_by="grp", reference=["r"], data_dtype="float32")

    def _resize_obs(name, n_rows):
        """Keep the manifest's n_obs at 8 but give obs.parquet a different row count."""
        stage = tmp_path / f"stage_{name}"
        stage.mkdir()
        arc = PackedArchive(src)
        for member in arc.members:
            mname = member["name"]
            if mname == fmt.MANIFEST:
                continue
            data = arc.member_bytes(mname)
            if mname == fmt.OBS:
                import io

                frame = pq.read_table(io.BytesIO(data)).to_pandas()
                if n_rows <= len(frame):
                    frame = frame.iloc[:n_rows]
                else:
                    frame = pd.concat([frame, frame.iloc[: n_rows - len(frame)]])
                buf = io.BytesIO()
                frame.to_parquet(buf)
                data = buf.getvalue()
            (stage / mname).write_bytes(data)
        dump_manifest(stage / fmt.MANIFEST, dict(arc.manifest))  # n_obs stays 8
        dst = tmp_path / name
        pack(stage, dst, overwrite=True, _delete_sources=True)
        return dst

    parts = [_resize_obs("short.shad", 7), _resize_obs("long.shad", 9)]  # 7 + 9 == 8 + 8
    with pytest.raises(ValueError, match=r"declares n_obs=8 but its obs has"):
        concat_cell_archives(parts, tmp_path / "cat.shad")


def test_concat_still_rejects_a_caller_supplied_unknown_reference(tmp_path):
    """Positional preservation applies ONLY to the inherited reference. A reference the caller
    passes explicitly is still resolved by label, so a bogus one must still fail loud."""
    from cellstream.cell.archive_writer import concat_cell_archives

    col = pd.Series(pd.array([pd.NA] * 8 + ["b"] * 8, dtype="string"))
    src = tmp_path / "src.shad"
    write_cell_archive(
        _adata(col, n=16), src, group_by="grp", reference=["nan"], data_dtype="float32"
    )
    parts = [_legacize(tmp_path, src, f"p{i}.shad", "<NA>") for i in (1, 2)]
    with pytest.raises(ValueError, match="Reference labels not found"):
        concat_cell_archives(parts, tmp_path / "cat.shad", group_by="grp", reference=["zzz"])


def test_sort_within_group_with_missing_values_does_not_raise(tmp_path):
    """A secondary sort key is not a label -- the collision check must not fire on it."""
    grp = pd.Series(pd.array(["g0"] * 12 + ["g1"] * 12, dtype="string"))
    sub = pd.Series(pd.array(["nan"] * 6 + [pd.NA] * 6 + ["x"] * 12, dtype="string"))
    a = _adata(grp)
    a.obs["sub"] = sub.values
    out = tmp_path / "a.shad"
    write_cell_archive(
        a,
        out,
        group_by="grp",
        sort_within_group="sub",
        sort_order="alphabetical",
        data_dtype="float32",
    )
    assert sorted(s.label for s in cellstream.open(str(out)).group_spans()) == ["g0", "g1"]
