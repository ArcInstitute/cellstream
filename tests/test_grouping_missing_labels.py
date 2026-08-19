"""#294 + #282 item 5: one sentinel for a missing group label."""

import types

import numpy as np
import pandas as pd
import pytest

from cellstream.grouping import (
    MISSING_GROUP_LABEL,
    plan_group_shards,
    resolve_reference,
    stringify_group_labels,
    validate_group_label_collision,
)


def test_sentinel_is_nan():
    assert MISSING_GROUP_LABEL == "nan"


@pytest.mark.parametrize(
    "col",
    [
        pytest.param(pd.Series(["a", "b", "c"], dtype=object), id="object-str"),
        pytest.param(pd.Series(pd.array(["a", "b", "c"], dtype="string")), id="nullable-string"),
        pytest.param(pd.Series(pd.Categorical(["a", "b", "c"])), id="categorical"),
        pytest.param(pd.Series(pd.Categorical([3, 1, 2])), id="categorical-int"),
        pytest.param(pd.Series([1.0, 2.5, 3.0]), id="float"),
        pytest.param(pd.Series([1, 2, 3]), id="int"),
        pytest.param(pd.Series([True, False, True]), id="bool"),
        # The three that caught the .astype(object) round-trip: it renders a datetime as
        # '2020-01-01 00:00:00' where .astype(str) gives '2020-01-01'. Grouping on a date
        # column is ordinary, so that would rename real groups in real archives.
        pytest.param(pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"])), id="datetime"),
        # And this one caught `.values`, which strips the timezone.
        pytest.param(
            pd.Series(pd.to_datetime(["2020-01-01 00:00:00-08:00", "2020-01-02 00:00:00-08:00"])),
            id="datetime-tz",
        ),
        pytest.param(pd.Series(pd.to_timedelta([1, 2], unit="D")), id="timedelta"),
        pytest.param(
            pd.Series(pd.Categorical(pd.to_datetime(["2020-01-01", "2020-01-02"]))),
            id="categorical-datetime",
        ),
    ],
)
def test_non_missing_labels_are_byte_identical_to_astype_str(col):
    """The whole change must be invisible when nothing is missing.

    A divergence here renames groups in ordinary archives -- a format change, not a fix.
    """
    got = stringify_group_labels(col)
    want = col.astype(str).to_numpy()
    assert list(got) == list(want), f"{list(got)} != {list(want)}"


def test_nat_is_a_missing_value_not_the_string_NaT():
    col = pd.Series(pd.to_datetime(["2020-01-01", None]))
    assert list(stringify_group_labels(col)) == ["2020-01-01", MISSING_GROUP_LABEL]


@pytest.mark.parametrize(
    "col,flavour",
    [
        pytest.param(pd.Series(["a", None], dtype=object), "object-None", id="object-None"),
        pytest.param(pd.Series(["a", np.nan], dtype=object), "object-nan", id="object-nan"),
        pytest.param(pd.Series(pd.array(["a", pd.NA], dtype="string")), "pd.NA", id="pd-NA"),
        pytest.param(pd.Series(pd.Categorical(["a", None])), "cat-NaN", id="categorical-NaN"),
    ],
)
def test_every_missing_flavour_becomes_the_one_sentinel(col, flavour):
    got = stringify_group_labels(col)
    assert list(got) == ["a", MISSING_GROUP_LABEL], f"{flavour}: {list(got)}"


def test_mixed_none_and_nan_in_one_object_column_collapse():  # #282 item 5
    col = pd.Series(["g0", None, np.nan, "g0"], dtype=object)
    assert list(stringify_group_labels(col)) == ["g0", "nan", "nan", "g0"]


def test_stringify_adds_no_collision_exception():
    """The normalizer must introduce no collision-validation exception: the shard writers
    reach it through plan_group_shards, and for a DIRECT v2 directory write that happens after
    overwrite=True has already deleted the prior archive."""
    col = pd.Series(pd.array(["nan", "g1", pd.NA], dtype="string"))
    assert list(stringify_group_labels(col)) == ["nan", "g1", "nan"]


def test_validate_raises_on_literal_nan_with_missing():
    col = pd.Series(pd.array(["nan", "g1", pd.NA], dtype="string"))
    with pytest.raises(ValueError, match=r"both a literal 'nan' label and missing values"):
        validate_group_label_collision(col, where="obs['grp']")


def test_validate_error_names_the_column():
    col = pd.Series(pd.array(["nan", pd.NA], dtype="string"))
    with pytest.raises(ValueError, match=r"obs\['grp'\]"):
        validate_group_label_collision(col, where="obs['grp']")


def test_validate_is_silent_without_missing():
    col = pd.Series(pd.array(["nan", "g1"], dtype="string"))
    assert validate_group_label_collision(col, where="obs['grp']") is None
    assert list(stringify_group_labels(col)) == ["nan", "g1"]


def test_validate_is_silent_without_a_literal_nan():
    col = pd.Series(pd.array(["g0", pd.NA], dtype="string"))
    assert validate_group_label_collision(col, where="obs['grp']") is None


def test_validate_catches_an_object_column_with_a_literal_nan_beside_np_nan():
    """A float column cannot hold a literal 'nan' string, but an OBJECT one can -- which is
    why this fixture is object dtype and not float. The neighbouring
    test_float_nan_value_in_a_numeric_column_is_missing_not_a_label covers the float case."""
    col = pd.Series(["nan", np.nan], dtype=object)
    with pytest.raises(ValueError, match=r"both a literal 'nan' label and missing values"):
        validate_group_label_collision(col, where="obs['grp']")


@pytest.mark.parametrize(
    "col",
    [
        pytest.param(pd.Series(pd.Categorical(["nan", "g1", None])), id="categorical"),
        pytest.param(
            pd.Series(pd.array(["nan", "g1", pd.NA], dtype="string")), id="nullable-string"
        ),
        pytest.param(pd.Series(["nan", "g1", np.nan], dtype=object), id="object"),
    ],
)
def test_the_dtype_fast_path_still_catches_every_real_collision(col):
    """The gate for the fast path in validate_group_label_collision.

    It skips the O(n) stringification for a dtype that cannot hold a non-missing value
    rendering as 'nan', and for a Categorical whose categories contain none. Each of those
    short-circuits must be unreachable when a REAL collision is present -- a Categorical whose
    categories DO include the literal 'nan' is the case the categories check could wrongly skip.
    """
    with pytest.raises(ValueError, match=r"both a literal 'nan' label and missing values"):
        validate_group_label_collision(col, where="obs['grp']")


@pytest.mark.parametrize(
    "col",
    [
        pytest.param(pd.Series(pd.Categorical(["g0", "g1", None])), id="categorical-no-nan"),
        pytest.param(pd.Series([1.0, 2.0, np.nan]), id="float"),
        pytest.param(pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02", None])), id="datetime"),
        pytest.param(pd.Series(pd.to_timedelta([1, 2, None], unit="D")), id="timedelta"),
    ],
)
def test_the_dtype_fast_path_is_silent_when_no_collision_is_possible(col):
    """The other side: these all HAVE missing values, so the isna() short-circuit does not fire
    and the dtype gate is what spares them the stringification."""
    assert validate_group_label_collision(col, where="obs['grp']") is None


def test_float_nan_value_in_a_numeric_column_is_missing_not_a_label():
    """A float column's NaN is a missing value, not the number 'nan'."""
    col = pd.Series([1.0, np.nan])
    assert list(stringify_group_labels(col)) == ["1.0", "nan"]


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(pd.Index([f"cell{i}" for i in range(4)]), id="obs_names-like"),
        pytest.param(pd.Index(["x"] * 4), id="duplicate"),
        pytest.param(pd.Index([7, 3, 99, 1]), id="unsorted-int"),
        pytest.param(pd.RangeIndex(3, -1, -1), id="reversed"),
        pytest.param(pd.MultiIndex.from_tuples([("a", i) for i in range(4)]), id="multi"),
    ],
)
def test_a_non_default_index_does_not_reorder_or_realign(index):
    """Both helpers are POSITIONAL and deliberately do not reset the index.

    Pinned so nobody re-adds `.reset_index(drop=True)` 'defensively' -- it is redundant here
    (verified byte-identical over 19 dtypes x 7 index shapes on pandas 2.3.3 and 3.0.5) and on
    pandas 2 it copies the whole column's data buffer -- and so nobody introduces an
    index-ALIGNED step, which would silently reorder rows. Row order here is archive row order.
    """
    col = pd.Series(["g0", None, "g1", np.nan], dtype=object)
    col.index = index
    assert list(stringify_group_labels(col)) == ["g0", "nan", "g1", "nan"]
    # the collision preflight is positional for the same reason
    col2 = pd.Series(["nan", None, "g1", "g1"], dtype=object)
    col2.index = index
    with pytest.raises(ValueError, match=r"both a literal 'nan' label and missing values"):
        validate_group_label_collision(col2, where="obs['grp']")


def test_accepts_a_bare_ndarray_and_an_empty_input():
    assert list(stringify_group_labels(np.array(["a", "b"], dtype=object))) == ["a", "b"]
    assert list(stringify_group_labels(pd.Series([], dtype=object))) == []


def test_all_missing_column():
    col = pd.Series(pd.array([pd.NA, pd.NA], dtype="string"))
    assert list(stringify_group_labels(col)) == ["nan", "nan"]


# ---------------------------------------------------------------------------------------
# Task 2 -- plan_group_shards' label view
# ---------------------------------------------------------------------------------------


def _records(labels, reference_mask=None):
    plan = plan_group_shards(
        group_labels=labels,
        per_row_nnz=np.ones(len(labels), np.int64),
        reference_mask=reference_mask,
        target_shard_bytes=1 << 20,
        bytes_per_nnz=4,
    )
    return [(g["label"], g["row_start"], g["row_stop"]) for g in plan.groups]


def test_missing_rows_form_ONE_record_not_one_per_row():
    """#294: on pandas 3 this returned 10 records for 8 missing rows."""
    labels = pd.Series(pd.array(["g0"] * 8 + ["g1"] * 8 + [pd.NA] * 8, dtype="string"))
    assert _records(labels) == [("g0", 0, 8), ("g1", 8, 16), ("nan", 16, 24)]


def test_categorical_with_nan_forms_one_record():
    labels = pd.Series(pd.Categorical(["g0"] * 8 + ["g1"] * 8 + [None] * 8))
    assert _records(labels) == [("g0", 0, 8), ("g1", 8, 16), ("nan", 16, 24)]


def test_none_and_nan_do_not_split_into_two_records():  # #282 item 5
    labels = pd.Series(["g0"] * 8 + ["g1"] * 8 + [None] * 4 + [np.nan] * 4, dtype=object)
    assert _records(labels) == [("g0", 0, 8), ("g1", 8, 16), ("nan", 16, 24)]


def test_reference_mask_splitting_the_missing_group_still_raises():
    """The alignment guard uses groupby(labels), which DROPS NaN keys -- so on pandas 3 the
    missing rows were silently exempt from the check that exists to stop a split mask."""
    labels = pd.Series(pd.array(["g0"] * 2 + ["g1"] * 2 + [pd.NA] * 4, dtype="string"))
    ref = np.array([True] * 2 + [False] * 2 + [True, False, True, False])
    with pytest.raises(ValueError, match=r"reference_mask splits label 'nan'"):
        _records(labels, reference_mask=ref)


def test_whole_missing_group_as_reference_is_accepted():
    labels = pd.Series(pd.array(["g0"] * 4 + [pd.NA] * 4, dtype="string"))
    ref = np.array([False] * 4 + [True] * 4)
    assert _records(labels, reference_mask=ref) == [("nan", 0, 4), ("g0", 4, 8)]


# ---------------------------------------------------------------------------------------
# Task 3 -- resolve_reference
# ---------------------------------------------------------------------------------------


def _adata_like(col):
    obs = pd.DataFrame(index=[f"c{i}" for i in range(len(col))])
    # .values: assigning a Series would align on index and silently NaN the column.
    obs["grp"] = col.values if isinstance(col, pd.Series) else col
    return types.SimpleNamespace(obs=obs, n_obs=len(obs))


def test_reference_nan_matches_the_missing_rows():
    col = pd.Series(pd.array(["g0"] * 4 + [pd.NA] * 4, dtype="string"))
    mask = resolve_reference(["nan"], _adata_like(col), "grp")
    assert list(mask) == [False] * 4 + [True] * 4


def test_reference_unknown_label_still_raises():
    col = pd.Series(pd.array(["g0"] * 4 + [pd.NA] * 4, dtype="string"))
    with pytest.raises(ValueError, match="Reference labels not found"):
        resolve_reference(["nope"], _adata_like(col), "grp")


@pytest.mark.parametrize("selector", [[None], [np.nan], [pd.NA]], ids=["None", "np.nan", "pd.NA"])
@pytest.mark.parametrize(
    "col",
    [
        pytest.param(pd.Series(pd.array(["g0"] * 4 + [pd.NA] * 4, dtype="string")), id="pd-NA"),
        pytest.param(pd.Series(pd.Categorical(["g0"] * 4 + [None] * 4)), id="categorical-NaN"),
    ],
)
def test_a_missing_SELECTOR_value_also_resolves_to_the_sentinel(col, selector):
    """`reference` is documented as label values present in obs[group_by]. The column now
    offers 'nan', so `str(lbl)` on the selector ('None' / '<NA>') would find nothing.

    Parametrized over the COLUMN dtype as well, because pre-fix each selector could match by
    coincidence for exactly one dtype: on pandas 2 a nullable-string column named its missing
    values '<NA>', which is also `str(pd.NA)`, and a Categorical named them 'nan', which is also
    `str(np.nan)`. One dtype alone therefore leaves one selector passing pre-fix and testing
    nothing. Across the 2x3 matrix every selector is discriminating for at least one dtype.
    """
    mask = resolve_reference(selector, _adata_like(col), "grp")
    assert list(mask) == [False] * 4 + [True] * 4


def test_resolve_reference_does_not_raise_on_the_collision():
    """The collision is the cell writer's preflight, not this function's job -- resolve_reference
    is also reached from the shard writers."""
    col = pd.Series(pd.array(["nan"] * 4 + [pd.NA] * 4, dtype="string"))
    mask = resolve_reference(["nan"], _adata_like(col), "grp")
    assert list(mask) == [True] * 8
