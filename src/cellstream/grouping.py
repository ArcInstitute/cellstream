"""Pure planner for target-aware sharding.

No I/O — all helpers operate on in-memory numpy/pandas data.  The planner
produces a permutation + group-aligned row_offsets + per-group records that
the writer uses to layout shards and the reader uses to serve group/reference
reads.

Public API:
    validate_sort_params(*, group_by, sort_within_group, sort_order) → None
    ShardSizeBounds — frozen dataclass with target/min/max byte counts.
    shard_size_bounds(target_shard_bytes, ...) → ShardSizeBounds
    ShardPlan — frozen dataclass: permutation, row_offsets, groups, reference_shard.
    plan_group_shards(group_labels, per_row_nnz, reference_mask, ...) → ShardPlan
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


def validate_sort_params(*, group_by, sort_within_group, sort_order):
    """Validate the grouped-write sort knobs (``sort_within_group``, ``sort_order``).

    THE single implementation of these rules, so the two writer families cannot drift again.

    Called from write.py (the shard path, IN PLACE -- moving it would reorder unrelated
    diagnostics) and from cell/writer.py::_resolve_storage_order, whose two callers cover every
    cell writer: write_cell_archive, write_sharded(layout='cell'), CellArchiveWriter.finalize
    and concat_cell_archives. Before this, layout='cell' skipped the checks entirely: an
    invalid mode fell through to _sort_key's ``else`` branch, storing NATURAL order while
    stamping the invalid string into the manifest (#261).

    Took a ``format`` argument until #154: both knobs were v2-only and raised for
    ``format="v1"``. The cell callers always passed nothing (the cell layout has no format),
    and the shard caller now has no other format to pass, so the parameter is gone with the
    two arms. Dropping it rather than leaving it inert also keeps the shard and cell paths
    reaching this function identically -- the property
    tests/test_option_parity.py::test_sort_params_raise_identically_on_both_layouts asserts
    by comparing the two layouts' MESSAGES.
    """
    # Collapsed rather than nested (its v1 sibling was the second arm); the sort_order block
    # below stays nested because it still has two.
    if sort_within_group is not None and group_by is None:
        raise ValueError(
            "sort_within_group requires group_by (it orders cells WITHIN each "
            "group). Pass group_by=<obs column> too, or drop sort_within_group."
        )
    if sort_order is not None:
        if group_by is None:
            raise ValueError(
                "sort_order requires group_by (it controls how groups, and any "
                "sort_within_group keys, are ordered). Pass group_by=<obs column>."
            )
        if sort_order not in ("natural", "alphabetical"):
            raise ValueError(f"sort_order must be 'natural' or 'alphabetical', got {sort_order!r}.")


# ---------------------------------------------------------------------------
# ShardSizeBounds + shard_size_bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardSizeBounds:
    """Byte band [min_bytes, target, max_bytes] for working-set sizing."""

    target: int
    min_bytes: int
    max_bytes: int


def shard_size_bounds(
    *,
    target_shard_bytes: int,
    min_bytes: int | None = None,
    max_bytes: int | None = None,
) -> ShardSizeBounds:
    """Resolve the [min, target, max] working-set-byte band.

    Default lo = min(512*2^20, target) so a small target (e.g. in unit tests)
    never raises.  Default hi = max(4*2^30, target) similarly.

    Explicit *min_bytes* / *max_bytes* are used as-is; if they violate
    ``lo <= target <= hi`` a ValueError is raised.
    """
    if target_shard_bytes < 0:
        raise ValueError(f"target_shard_bytes must be >= 0, got {target_shard_bytes}.")
    target = int(target_shard_bytes)
    lo = int(min_bytes) if min_bytes is not None else min(512 * 2**20, target)
    hi = int(max_bytes) if max_bytes is not None else max(4 * 2**30, target)
    if not (lo <= target <= hi):
        raise ValueError(
            f"target {target} not within [min {lo}, max {hi}]; "
            "adjust min_bytes / max_bytes or target_shard_bytes."
        )
    return ShardSizeBounds(target=target, min_bytes=lo, max_bytes=hi)


MISSING_GROUP_LABEL = "nan"


def _as_label_series(values) -> pd.Series:
    # Keep the Series -- `values.values` STRIPS a tz-aware datetime's timezone, renaming
    # '2020-01-01 00:00:00-08:00' to '2020-01-01 08:00:00' (and a UTC column to '2020-01-01').
    # NO .reset_index(drop=True): every downstream step is positional (.astype(str).to_numpy(),
    # .isna().to_numpy(), then ndarray masking), so no index alignment can occur -- and on
    # pandas 2 the reset copies the whole column's data buffer for nothing.
    # test_a_non_default_index_does_not_reorder_or_realign pins this both ways.
    return values if isinstance(values, pd.Series) else pd.Series(values)


def stringify_group_labels(values) -> np.ndarray:
    """Stringified view of a group column, every missing flavour mapped to ONE sentinel.

    The single place a group label's name is decided. A bare ``.astype(str)`` is not usable:

    * on pandas 2 it names a missing value ``'None'`` / ``'nan'`` / ``'<NA>'`` depending on the
      column's dtype, so one logical "missing" becomes up to three groups, and none of those
      names survives an obs round-trip (parquet and h5ad normalize all three back to one NA);
    * on pandas 3 it PRESERVES the missing value, so the boundary scan in ``plan_group_shards``
      sees ``nan != nan`` at every adjacent pair and emits one record per missing row (#294).

    Stringify FIRST, then overwrite the missing positions. Normalizing via
    ``.astype(object).where(...)`` instead renames non-missing labels: a ``datetime64`` renders
    as ``'2020-01-01 00:00:00'`` where ``.astype(str)`` gives ``'2020-01-01'``, and a
    ``timedelta64`` as ``'1 days 00:00:00'`` rather than ``'1 days'``. Grouping on a date column
    is ordinary, so that would silently rename real groups.

    Pure: it normalizes and validates nothing, so it adds no error of its own. (Not "never
    raises" -- ``.astype(str)`` can still propagate a conversion error from an exotic object
    value, which is pre-existing and unchanged here.) Adding no NEW failure mode is the property
    the shard writers depend on: they reach this through ``plan_group_shards``, and for a DIRECT
    v2 directory write that happens after ``overwrite=True`` has already deleted the prior
    archive, so raising here would be destructive. The collision check is
    ``validate_group_label_collision``, a cell-writer preflight.
    """
    s = _as_label_series(values)
    # copy=True is deliberate here and NOT a waste, unlike in validate_group_label_collision:
    # this function MUTATES `out` below. Dropping the copy would return -- and then write into
    # -- whatever buffer `.astype(str)` handed back, and whether that is ever a view of the
    # caller's obs column is a pandas-version and dtype detail, not a contract. Measured, the
    # copy costs ~150ms at 10M rows; an aliasing bug that silently rewrites obs costs more.
    out = s.astype(str).to_numpy(dtype=object, copy=True)
    out[s.isna().to_numpy()] = MISSING_GROUP_LABEL
    return out


def validate_group_label_collision(values, *, where: str) -> None:
    """Reject a group column holding BOTH missing values and a literal 'nan' label.

    The one case where normalizing fuses two distinct populations under one name. Measured, the
    usual outcome is a SILENT fusion -- the literal-'nan' rows and the missing rows sort
    adjacently and coalesce into one record, so ``read_group('nan')`` succeeds and returns both
    populations. (Two records sharing the label, which #248's ambiguity error would catch,
    arise only when a third label sorts between the runs.) Raising at write time leaves the
    user somewhere they can still fix the input, rather than shipping a wrong read.

    Deliberately NOT called from ``plan_group_shards`` -- see ``stringify_group_labels``.
    """
    s = _as_label_series(values)
    na = s.isna().to_numpy()
    if not na.any():
        return
    # Skip the O(n) stringification when the dtype provably cannot hold a NON-missing value
    # that renders as 'nan'. Numeric / bool / datetime64 / timedelta64 / period / interval all
    # qualify: the only float rendering as 'nan' IS NaN, which `na` above already covers.
    # Measured at 10M rows: 282ms -> ~0 for a Categorical, the usual obs grouping dtype.
    dtype = s.dtype
    is_cat = isinstance(dtype, pd.CategoricalDtype)
    if not (is_cat or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)):
        return
    if is_cat:
        # A Categorical's categories are its only non-missing values (NaN is never a category),
        # so if none of them stringifies to the sentinel there is nothing to collide.
        if not any(str(c) == MISSING_GROUP_LABEL for c in dtype.categories):
            return
    elif not pd.api.types.is_object_dtype(dtype):
        # A `string` dtype holds only real strings, so equality IS the stringified comparison.
        # Deliberately NOT done for object dtype: there `==` would miss a non-missing value
        # whose str() is 'nan' without being equal to it, narrowing the check for a speedup.
        if not (s == MISSING_GROUP_LABEL).any():
            return
    # copy=False: unlike stringify_group_labels this NEVER mutates `out` -- it is a read-only
    # collision scan -- so the extra object array is pure waste. Measured at 10M rows: the copy
    # alone cost 154ms of 805ms for a Categorical and 126ms of 292ms for an object column.
    out = s.astype(str).to_numpy(dtype=object, copy=False)
    if (out[~na] == MISSING_GROUP_LABEL).any():
        raise ValueError(
            f"{where} holds both a literal 'nan' label and missing values. Both would be "
            f"named {MISSING_GROUP_LABEL!r}, so they would be merged into one group and "
            f"read_group({MISSING_GROUP_LABEL!r}) would silently return both populations. "
            f"Rename the literal label, or fill the missing values, before writing."
        )


# ---------------------------------------------------------------------------
# ShardPlan + plan_group_shards
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardPlan:
    """Output of the group-shard planner.

    Attributes
    ----------
    permutation:
        int64 array of length n_obs.  ``permutation[new_row] = original_row``.
    row_offsets:
        List of length n_shards + 1.  Shard *s* covers rows
        ``[row_offsets[s], row_offsets[s+1])``.
    groups:
        List of dicts with keys ``label``, ``shard``, ``row_start``,
        ``row_stop``, ``role`` (``"reference"`` or ``"group"``).
        Rows are **global** (into the permuted order).
    reference_shard:
        Index of the reference shard, or ``None`` if no reference was given.
    """

    permutation: np.ndarray  # int64[n_obs]
    row_offsets: list[int]  # len n_shards + 1
    groups: list[dict]  # {label, shard, row_start, row_stop, role}
    reference_shard: int | None


def plan_group_shards(
    *,
    group_labels,
    per_row_nnz,
    reference_mask,
    target_shard_bytes: int,
    bytes_per_nnz: int,
    min_bytes: int | None = None,
    max_bytes: int | None = None,
    secondary_key=None,
) -> ShardPlan:
    """Compute a permutation + group-aligned shard boundaries.

    Algorithm
    ---------
    1. Stable-sort rows: reference first (``~ref_mask`` sorts False<True),
       then lexicographic by label, then optionally by *secondary_key*.
    2. Derive per-label blocks from label-change boundaries.
    3. Bin-pack into shards, cutting only at group edges:
       - Reference block → its own shard (shard 0), never mixed with groups,
         never split across shards even if it exceeds *target*.
       - Non-reference groups → greedy accumulation until adding the next
         would exceed *target*; then seal.  A single over-target group gets
         its own shard.

    Parameters
    ----------
    group_labels:
        Array-like of string labels, length n_obs.
    per_row_nnz:
        int64-compatible array of per-row non-zero counts, length n_obs.
    reference_mask:
        Boolean array-like of length n_obs (True = reference cell), or
        ``None`` (no dedicated reference shard).
    target_shard_bytes:
        Passed to ``shard_size_bounds`` for resolving the byte band.
    bytes_per_nnz:
        Bytes per non-zero (e.g. 4 for uint16 data+indices); sizes each group.
    min_bytes, max_bytes:
        Optional explicit overrides for the byte band (see ``shard_size_bounds``).
    secondary_key:
        Optional secondary sort key(s) to order rows WITHIN each group: one
        array-like (e.g. a guide label), or a list of array-likes for a
        multi-level (lexicographic) sort. Each array is length n_obs. A pandas
        Categorical key is sorted by its category order; numeric numerically.
    """
    # Preserve the sort-column dtype: a pandas Categorical sorts by its CATEGORY
    # order, a numeric column numerically, a string column alphabetically — pandas
    # sort_values honors each natively. Group NAMES and block boundaries are always
    # derived from the string form below, so a categorical group_by orders groups
    # by category while still naming them by their string value. Plain arrays/lists
    # (v1's .astype(str) callers, existing tests) become an object/numeric Series →
    # identical behavior to the previous np.asarray path.
    _lab = pd.Series(
        group_labels.values if isinstance(group_labels, pd.Series) else group_labels
    ).reset_index(drop=True)
    # Every missing flavour -> one sentinel BEFORE anything derives from this view. Names,
    # block boundaries and the ref-mask alignment guard all read `labels`; a bare .astype(str)
    # fragments them on pandas 3 and splits them by dtype on pandas 2 (#294, #282 item 5).
    labels = stringify_group_labels(_lab)
    nnz = np.asarray(per_row_nnz, dtype=np.int64)
    bounds = shard_size_bounds(
        target_shard_bytes=target_shard_bytes,
        min_bytes=min_bytes,
        max_bytes=max_bytes,
    )
    n = len(labels)
    if n == 0:
        return ShardPlan(
            permutation=np.empty(0, dtype=np.int64),
            row_offsets=[0, 0],
            groups=[],
            reference_shard=None,
        )

    # ------------------------------------------------------------------
    # Coerce reference_mask to bool and run length + whole-label-alignment
    # validation BEFORE anything else.
    #
    # The dtype guard was previously gated on the *original* dtype being
    # exactly bool, which meant int 0/1 masks were coerced silently and
    # skipped validation — a split-label mask could slip through and
    # re-expose the Critical bug.  Now we validate for every non-None mask
    # regardless of the original dtype.
    # ------------------------------------------------------------------
    if reference_mask is not None:
        ref_mask_raw = np.asarray(reference_mask)
        if len(ref_mask_raw) != n:
            raise ValueError(
                f"reference_mask has length {len(ref_mask_raw)} but group_labels has "
                f"length {n}; lengths must match."
            )
        ref_mask = ref_mask_raw.astype(bool)
        # Whole-label-alignment check: every label must be either entirely
        # reference or entirely non-reference.  O(N) via groupby.nunique.
        nun = pd.Series(ref_mask).groupby(labels).nunique()
        split = nun[nun > 1].index.tolist()
        if split:
            lbl = split[0]
            raise ValueError(
                f"reference_mask splits label {lbl!r}: some rows are marked "
                f"reference and others are not. Reference must apply to whole "
                f"groups. Either mark all rows of {lbl!r} as reference or none."
            )
    else:
        ref_mask = np.zeros(n, bool)

    # ------------------------------------------------------------------
    # Step 1 — stable sort: reference first, then label, then secondary.
    # ~ref_mask: True for non-reference (False < True → reference first).
    # ------------------------------------------------------------------
    sort_df = pd.DataFrame({"ref": ~ref_mask})
    sort_df["lab"] = _lab.values  # Categorical/numeric/object preserved → native sort order
    cols = ["ref", "lab"]
    if secondary_key is not None and len(secondary_key) > 0:
        # secondary_key is one array-like (back-compat) OR a list/tuple of
        # array-likes (multi-level). Disambiguate by the FIRST element: a
        # list/tuple whose [0] is itself a length-n_obs array → multiple keys.
        # Otherwise treat the whole thing as one key — including a plain list of
        # n_obs scalars, which isinstance(list) alone would mis-split (#87).
        # An empty list/tuple (len 0) means "no secondary sort" — skip (#97).
        first = (
            secondary_key[0]
            if isinstance(secondary_key, (list, tuple)) and len(secondary_key)
            else None
        )
        # Duck-typed: `first` is a per-row key array if it's a non-string
        # sized sequence of length n_obs (ndarray, Series, pd.Index/Categorical,
        # list, ...). A scalar (incl. str) first element → one key (#97).
        is_multi = (
            not isinstance(first, (str, bytes))
            and hasattr(first, "__len__")
            and len(first) == len(labels)
        )
        keys = secondary_key if is_multi else [secondary_key]
        for i, k in enumerate(keys):
            # pd.Series(k).values (not np.asarray): keeps a Categorical's category
            # order and drops any index; numeric/object keys are unaffected.
            sort_df[f"sub{i}"] = pd.Series(k).values
            cols.append(f"sub{i}")
    perm_all = sort_df.sort_values(cols, kind="stable").index.to_numpy(dtype=np.int64)

    s_lab = labels[perm_all]
    s_ref = ref_mask[perm_all]

    # ------------------------------------------------------------------
    # Step 2 — derive per-label blocks from label-change boundaries.
    # ------------------------------------------------------------------
    # Group boundaries: positions where the label changes.
    starts = np.flatnonzero(np.r_[True, s_lab[1:] != s_lab[:-1]])
    stops = np.r_[starts[1:], n]
    # Each entry: (label, role, rows_into_perm_all)
    ordered = [
        (str(s_lab[a]), "reference" if s_ref[a] else "group", perm_all[a:b])
        for a, b in zip(starts, stops, strict=True)
    ]

    # ------------------------------------------------------------------
    # Step 3 — bin-pack into shards.
    # ------------------------------------------------------------------
    def _bytes(rows: np.ndarray) -> int:
        return int(nnz[rows].sum()) * bytes_per_nnz

    perm: list[int] = []
    row_offsets: list[int] = [0]
    groups: list[dict] = []
    cur_bytes: int = 0
    cur_shard: int = 0
    last_role: str | None = None

    def _seal() -> None:
        nonlocal cur_bytes
        row_offsets.append(len(perm))
        cur_bytes = 0

    total_ref_bytes: int = 0  # accumulate reference-shard bytes for the combined warning

    for lab, role, rows in ordered:
        gb = _bytes(rows)

        # Decide whether to start a new shard before appending this group.
        if role == "reference":
            # Reference must be isolated in its own shard(s) and never mixed
            # with non-reference groups.  If we already have content from a
            # different role (shouldn't happen with the sort, but be safe),
            # seal first.  Also seal if there is already some reference
            # content — each reference label still gets a boundary check,
            # but since reference is a single block after sorting, in
            # practice there is only one reference label block.
            if len(perm) > 0 and last_role != "reference":
                _seal()
                cur_shard += 1
            total_ref_bytes += gb
        else:
            # Non-reference: seal on role switch or when target would be exceeded.
            boundary = last_role == "reference" or (
                cur_bytes + gb > bounds.target and len(perm) > 0
            )
            if boundary and len(perm) > 0:
                _seal()
                cur_shard += 1
            # Warn if a single non-reference group exceeds max_bytes.
            if gb > bounds.max_bytes:
                warnings.warn(
                    f"Group {lab!r} occupies {gb} bytes, which exceeds "
                    f"max_bytes={bounds.max_bytes}. This group will be in an oversized shard.",
                    stacklevel=2,
                )

        start = len(perm)
        perm.extend(rows.tolist())
        stop = len(perm)
        groups.append(
            {
                "label": lab,
                "shard": cur_shard,
                "row_start": start,
                "row_stop": stop,
                "role": role,
            }
        )
        cur_bytes += gb
        last_role = role

    # Seal the final shard.
    _seal()

    # Warn once on the TOTAL reference-shard bytes when they exceed max_bytes.
    # Two labels each individually under max_bytes but combined over it would
    # produce a silent oversized reference shard under a per-label check.
    if total_ref_bytes > bounds.max_bytes:
        warnings.warn(
            f"The combined reference shard occupies {total_ref_bytes} bytes, which exceeds "
            f"max_bytes={bounds.max_bytes}. The reference shard will be oversized.",
            stacklevel=2,
        )

    ref_shard = next((g["shard"] for g in groups if g["role"] == "reference"), None)
    return ShardPlan(
        permutation=np.asarray(perm, dtype=np.int64),
        row_offsets=row_offsets,
        groups=groups,
        reference_shard=ref_shard,
    )


# ---------------------------------------------------------------------------
# Reference-mask resolution (shared by the v2 shard writer and the cell writer)
# ---------------------------------------------------------------------------


def _is_booleany_mask(arr: np.ndarray, n_obs: int) -> bool:
    """Return True iff *arr* should be treated as a row-mask rather than a label list.

    A value is a mask iff it is 1-D, length == n_obs, and every element is in
    {0, 1, True, False} (i.e. booleany).  This intentionally catches int arrays
    of 0s and 1s that have the same length as obs, which are almost certainly
    masks rather than lists of label values "0"/"1".
    """
    if arr.ndim != 1 or len(arr) != n_obs:
        return False
    # Vectorized fast path for numeric arrays: a length-n_obs 0/1 int mask can be
    # huge, and .tolist()+set materializes O(n) Python objects. Fall back to the
    # set check only for non-numeric / object dtypes (e.g. a label array).
    if np.issubdtype(arr.dtype, np.number):
        return bool(np.all((arr == 0) | (arr == 1)))
    try:
        unique_vals = set(arr.ravel().tolist())
        return unique_vals <= {0, 1}
    except TypeError:
        return False


def resolve_reference(reference, adata, group_by: str) -> np.ndarray:
    """Resolve *reference* to a boolean mask of length n_obs.

    Resolution rules:
    - None → all-False mask (no reference cells).
    - dtype==bool (always treated as a mask) → validate length == n_obs, then return.
    - array-like that is 1-D, length == n_obs, values ⊆ {0,1,True,False}
      → treat as a row-mask; coerce to bool.
    - otherwise → treat as a list of label values in obs[group_by]; convert to
      a row-mask via np.isin.  Raises ValueError for unknown labels.

    The returned mask always has dtype=bool and length n_obs.
    """
    n_obs = adata.n_obs
    if reference is None:
        return np.zeros(n_obs, dtype=bool)

    # A bare string is a single-element label list, not a character array.
    if isinstance(reference, str):
        reference = [reference]

    ref_arr = np.asarray(reference)

    # Bool-dtype arrays are always masks — enforce shape constraints up front
    # so that a wrong-shape bool array gets a clear "reference" error rather
    # than silently falling through to the label-list path.
    if ref_arr.dtype == bool:
        if ref_arr.ndim != 1:
            raise ValueError(
                f"reference bool mask must be 1-D, got shape {ref_arr.shape}. "
                "Reshape to a 1-D array of length n_obs before passing as reference."
            )
        if len(ref_arr) != n_obs:
            raise ValueError(
                f"reference bool mask has length {len(ref_arr)} but adata has "
                f"n_obs={n_obs}; lengths must match."
            )
        return ref_arr.astype(bool)

    if _is_booleany_mask(ref_arr, n_obs):
        return ref_arr.astype(bool)

    # Label-list path: validate each label.
    # Same view plan_group_shards derives names from -- otherwise reference=['nan'] matches
    # nothing on pandas 3 and matches only np.nan-flavoured columns on pandas 2 (#294).
    labels_col = stringify_group_labels(adata.obs[group_by])
    obs_labels = set(labels_col)
    # BOTH sides: a caller may legitimately pass the missing value itself (reference=[None] /
    # [pd.NA]), which str() would render 'None' / '<NA>' -- names the column no longer offers.
    # Materialize ONCE: a generator would be exhausted by the normalization and then zip
    # strict=True would raise a spurious length error.
    ref_given = list(reference)
    ref_labels = list(stringify_group_labels(pd.Series(ref_given, dtype=object)))
    unknown = [
        orig for orig, norm in zip(ref_given, ref_labels, strict=True) if norm not in obs_labels
    ]
    if unknown:
        raise ValueError(
            f"Reference labels not found in obs[{group_by!r}]: {unknown}. "
            f"Available labels: {sorted(obs_labels)}"
        )
    return np.isin(labels_col, ref_labels)


def validate_reference_labels(adata, group_by, reference):
    """Raise ValueError if reference is invalid (unknown labels or wrong-length mask).

    Delegates to resolve_reference which centralises the routing logic; this
    wrapper exists for early-validation calls (before any disk I/O) and simply
    discards the resolved mask.
    """
    resolve_reference(reference, adata, group_by)
