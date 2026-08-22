# tests/test_matrices.py
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream import matrices as mx


def _adata_with_layers_and_raw():
    X = sp.csr_matrix(np.array([[0, 1, 2], [3, 0, 0]], dtype=np.uint16))
    counts = sp.csr_matrix(np.array([[0, 10, 20], [30, 0, 0]], dtype=np.uint16))
    var = pd.DataFrame(index=["g0", "g1", "g2"])
    obs = pd.DataFrame({"ct": ["a", "b"]}, index=["c0", "c1"])
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.layers["counts"] = counts
    raw_X = sp.csr_matrix(np.array([[0, 1, 2, 0], [3, 0, 0, 9]], dtype=np.uint16))
    raw_var = pd.DataFrame(index=["g0", "g1", "g2", "g3"])
    a.raw = ad.AnnData(X=raw_X, obs=obs, var=raw_var)
    return a


def test_matrix_dir_sanitises_keys():
    assert mx.matrix_dir("x") == "x"
    assert mx.matrix_dir("raw") == "raw"
    assert mx.matrix_dir("layer:counts") == "layer_counts"
    assert mx.matrix_dir("layer:log 1p/norm") == "layer_log_1p_norm"


def test_var_member_name():
    assert mx.var_member_name("x") == "var/x.parquet"
    assert mx.var_member_name("raw") == "var/raw.parquet"


def test_decompose_order_and_roles():
    a = _adata_with_layers_and_raw()
    specs = mx.decompose_anndata(a)
    assert [s.key for s in specs] == ["x", "layer:counts", "raw"]
    x, layer, raw = specs
    assert x.role == "x" and x.owner is None and x.n_vars == 3 and x.var is not None
    assert layer.role == "layer:counts" and layer.owner == "x" and layer.var is None
    assert raw.role == "raw" and raw.owner is None and raw.n_vars == 4 and raw.var is not None
    np.testing.assert_array_equal(x.X.toarray(), a.X.toarray())
    np.testing.assert_array_equal(layer.X.toarray(), a.layers["counts"].toarray())
    np.testing.assert_array_equal(raw.X.toarray(), a.raw.X.toarray())


def test_decompose_single_matrix_is_just_x():
    a = ad.AnnData(X=sp.csr_matrix(np.eye(3, dtype=np.uint16)))
    specs = mx.decompose_anndata(a)
    assert [s.key for s in specs] == ["x"]


def test_reassemble_round_trip():
    a = _adata_with_layers_and_raw()
    specs = mx.decompose_anndata(a)
    var_by_key = {"x": a.var, "raw": a.raw.var}
    rebuilt = mx.reassemble_anndata(
        specs, obs=a.obs, var_by_key=var_by_key, uns={}, obsm={}, varm={}
    )
    np.testing.assert_array_equal(rebuilt.X.toarray(), a.X.toarray())
    np.testing.assert_array_equal(rebuilt.layers["counts"].toarray(), a.layers["counts"].toarray())
    assert list(rebuilt.var.index) == list(a.var.index)
    np.testing.assert_array_equal(rebuilt.raw.X.toarray(), a.raw.X.toarray())
    assert list(rebuilt.raw.var.index) == list(a.raw.var.index)


def test_reassemble_rejects_missing_primary():
    layer = mx.MatrixSpec(key="layer:c", role="layer:c", owner="x", n_vars=2, X=None, var=None)
    with pytest.raises(ValueError, match="primary 'x'"):
        mx.reassemble_anndata(
            [layer], obs=pd.DataFrame(index=["c0"]), var_by_key={}, uns={}, obsm={}, varm={}
        )


def test_decompose_rejects_sanitize_collision():
    a = ad.AnnData(X=sp.csr_matrix(np.eye(2, dtype=np.uint16)))
    a.layers["a/b"] = sp.csr_matrix(np.eye(2, dtype=np.uint16))
    a.layers["a_b"] = sp.csr_matrix(np.eye(2, dtype=np.uint16))
    with pytest.raises(ValueError, match="on-disk directory"):
        mx.decompose_anndata(a)


def test_matrixspec_eq_returns_a_bool_with_array_valued_fields():
    """CPython 3.13 generates a PAIRWISE dataclass ``__eq__`` instead of comparing the two field
    tuples, which loses ``PyObject_RichCompareBool``'s identity shortcut and its bool coercion --
    so with ``X`` holding a sparse matrix the generated ``__eq__`` returned the elementwise MATRIX
    and any ``bool()`` of it raised "the truth value of an array ... is ambiguous".

    Asserts ``is True`` / ``is False`` rather than truthiness on purpose: the wrong return is a
    sparse matrix, which is neither truthy nor falsy but RAISES, so a plain ``assert a == b`` is
    what let this reach a released version unnoticed on 3.13 while passing on 3.11 and 3.12.
    """
    m = sp.csr_matrix(np.ones((3, 3), dtype=np.uint16))
    a = mx.MatrixSpec("k", "x", None, 3, m, None)
    b = mx.MatrixSpec("k", "x", None, 3, m, None)
    assert (a == b) is True
    assert (a != b) is False
    assert (a == mx.MatrixSpec("other", "x", None, 3, m, None)) is False
    assert (a == 42) is False


def test_matrixspec_eq_raises_on_distinct_but_equal_arrays():
    """The deliberate limit of the fix, pinned so a future change to it is a decision, not a drift.

    Two DIFFERENT arrays in the same field raise, because the field's own ``==`` returns an
    elementwise result that has no truth value. That is what the pre-3.13 tuple comparison did too,
    so this is not a regression and the fix restores it faithfully rather than papering over it --
    only the SAME-object case, which used to short-circuit on identity, was broken on 3.13.

    Gemini (PR #319) proposed making this raise-free by special-casing sparse matrices, arrays and
    DataFrames. That is new semantics rather than a restoration, and it wants a shared helper across
    four dataclasses, so it is tracked in #318 instead of decided here.
    """
    m1 = sp.csr_matrix(np.ones((3, 3), dtype=np.uint16))
    m2 = sp.csr_matrix(np.ones((3, 3), dtype=np.uint16))
    assert m1 is not m2
    a = mx.MatrixSpec("k", "x", None, 3, m1, None)
    b = mx.MatrixSpec("k", "x", None, 3, m2, None)
    with pytest.raises(ValueError, match="truth value"):
        a == b  # noqa: B015 -- the comparison itself is the thing under test


def test_matrixspec_eq_with_a_dataframe_field():
    """``var`` holds a DataFrame, whose ``==`` also returns a frame rather than a bool -- the same
    trap on the other field, and the one a `decompose` result actually carries."""
    var = pd.DataFrame(index=["g0", "g1"])
    a = mx.MatrixSpec("x", "x", None, 2, None, var)
    b = mx.MatrixSpec("x", "x", None, 2, None, var)
    assert (a == b) is True
    # Differing on an EARLIER field, so the tuple comparison short-circuits before `var`. Two
    # genuinely different frames in the same field still raise, on every version -- that is
    # pre-existing behaviour this fix deliberately does not change.
    assert (a == mx.MatrixSpec("other", "x", None, 2, None, var)) is False
