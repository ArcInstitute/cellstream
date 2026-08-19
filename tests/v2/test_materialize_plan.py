from __future__ import annotations

import numpy as np
import pytest

from cellstream.errors import LossyCastError
from cellstream.v2.materialize import MaterializePlan, cast_data, resolve_plan


def _manifest(*, on_disk_data="uint16", on_disk_idx="uint16", total_nnz=100, n_vars=4):
    return {
        "x_data_dtype_on_disk": on_disk_data,
        "x_indices_dtype_on_disk": on_disk_idx,
        "total_nnz": total_nnz,
        "n_vars": n_vars,
    }


def test_default_plan_matches_cast_back_true():
    p = resolve_plan(
        _manifest(),
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    assert isinstance(p, MaterializePlan)
    assert p.container == "csr"
    assert p.data_dtype == np.dtype(np.float32)
    assert p.n_vars == 4


def test_cast_back_false_keeps_native_data():
    p = resolve_plan(
        _manifest(on_disk_data="uint16"),
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=False,
        target="cpu",
    )
    assert p.data_dtype == np.dtype(np.uint16)


def test_float64_on_disk_default_stays_float64():
    p = resolve_plan(
        _manifest(on_disk_data="float64"),
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    assert p.data_dtype == np.dtype(np.float64)


def test_explicit_data_dtype_overrides():
    p = resolve_plan(
        _manifest(on_disk_data="uint16"),
        container="csr",
        data_dtype="float64",
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    assert p.data_dtype == np.dtype(np.float64)


def test_bad_container_raises():
    with pytest.raises(ValueError, match="container"):
        resolve_plan(
            _manifest(),
            container="csc",
            data_dtype=None,
            index_dtype=None,
            allow_lossy=False,
            cast_back=True,
            target="cpu",
        )


def test_non_numeric_data_dtype_raises():
    with pytest.raises(ValueError, match="data_dtype"):
        resolve_plan(
            _manifest(),
            container="csr",
            data_dtype="object",
            index_dtype=None,
            allow_lossy=False,
            cast_back=True,
            target="cpu",
        )


def test_dense_container_roundtrips():
    p = resolve_plan(
        _manifest(),
        container="dense",
        data_dtype="float32",
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    assert p.container == "dense"
    assert p.data_dtype == np.dtype(np.float32)
    assert p.n_vars == 4


def test_float32_on_disk_cast_back_stays_float32():
    p = resolve_plan(
        _manifest(on_disk_data="float32"),
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    assert p.data_dtype == np.dtype(np.float32)


def test_invalid_target_raises():
    with pytest.raises(ValueError, match="target"):
        resolve_plan(
            _manifest(),
            container="csr",
            data_dtype=None,
            index_dtype=None,
            allow_lossy=False,
            cast_back=True,
            target="gpu",
        )


def test_indptr_autosizes_to_int64_for_huge_nnz():
    p = resolve_plan(
        _manifest(total_nnz=2**31 + 5),
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    assert p.indptr_dtype == np.dtype(np.int64)


def test_cast_back_false_indptr_is_int64():
    p = resolve_plan(
        _manifest(total_nnz=100),
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=False,
        target="cpu",
    )
    assert p.indptr_dtype == np.dtype(np.int64)


def test_explicit_index_dtype_overrides():
    p = resolve_plan(
        _manifest(),
        container="csr",
        data_dtype=None,
        index_dtype="uint32",
        allow_lossy=False,
        cast_back=True,
        target="cpu",
    )
    assert p.index_dtype == np.dtype(np.uint32)


def test_torch_requires_int_index_dtype():
    with pytest.raises(ValueError, match="torch"):
        resolve_plan(
            _manifest(),
            container="csr",
            data_dtype="float32",
            index_dtype="uint32",
            allow_lossy=False,
            cast_back=True,
            target="torch",
        )


def test_torch_indptr_matches_index_dtype():
    p = resolve_plan(
        _manifest(total_nnz=100),
        container="csr",
        data_dtype="float32",
        index_dtype="int64",
        allow_lossy=False,
        cast_back=True,
        target="torch",
    )
    assert p.indptr_dtype == np.dtype(np.int64) and p.index_dtype == np.dtype(np.int64)


def test_torch_int32_index_with_huge_nnz_raises():
    with pytest.raises(ValueError, match="nnz|int64"):
        resolve_plan(
            _manifest(total_nnz=2**31 + 5),
            container="csr",
            data_dtype="float32",
            index_dtype="int32",
            allow_lossy=False,
            cast_back=True,
            target="torch",
        )


def test_cast_data_safe_pair_no_check():
    out = cast_data(
        np.array([1, 2, 3], dtype=np.uint16),
        np.dtype(np.float64),
        allow_lossy=False,
        where="X.data",
    )
    assert out.dtype == np.dtype(np.float64)
    assert np.array_equal(out, [1.0, 2.0, 3.0])


def test_cast_data_lossy_raises_without_flag():
    with pytest.raises(LossyCastError):
        cast_data(
            np.array([1.5, 2.5], dtype=np.float32),
            np.dtype(np.int32),
            allow_lossy=False,
            where="X.data",
        )


def test_cast_data_lossy_allowed_with_flag():
    out = cast_data(
        np.array([1.5, 2.5], dtype=np.float32), np.dtype(np.int32), allow_lossy=True, where="X.data"
    )
    assert out.dtype == np.dtype(np.int32)
    assert np.array_equal(out, [1, 2])  # truncation accepted


def test_cast_data_value_level_lossless_passes():
    out = cast_data(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.dtype(np.int32),
        allow_lossy=False,
        where="X.data",
    )
    assert np.array_equal(out, [1, 2, 3])


def test_torch_default_index_dtype_matches_small_nnz():
    # index_dtype=None + torch + small nnz -> _index_dtypes_for gives matching int32/int32.
    p = resolve_plan(
        _manifest(total_nnz=100),
        container="csr",
        data_dtype="float32",
        index_dtype=None,
        allow_lossy=False,
        cast_back=True,
        target="torch",
    )
    assert p.index_dtype == np.dtype(np.int32)
    assert p.indptr_dtype == np.dtype(np.int32)


def test_cast_data_identity_returns_same_object():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    assert cast_data(arr, np.float32, allow_lossy=False, where="X.data") is arr


def test_narrow_read_index_dtype_is_nvars_aware():
    # H1 (ultrareview): the cast_back=False default must mirror the write side
    # (narrow.py: uint16 iff n_vars <= 65536, else uint32). A wide archive read
    # narrow with the uint16 default would raise LossyCastError (or truncate).
    wide = resolve_plan(
        _manifest(n_vars=70000),
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=False,
        target="cpu",
    )
    assert wide.index_dtype == np.dtype(np.uint32)

    narrow = resolve_plan(
        _manifest(n_vars=4),
        container="csr",
        data_dtype=None,
        index_dtype=None,
        allow_lossy=False,
        cast_back=False,
        target="cpu",
    )
    assert narrow.index_dtype == np.dtype(np.uint16)


def test_cast_back_true_index_dtype_unaffected_by_wide_nvars():
    # cast_back=True already returns int32/int64, which hold wide column indices;
    # the H1 fix must not perturb it at either n_vars.
    for nv in (4, 70000):
        p = resolve_plan(
            _manifest(n_vars=nv, total_nnz=100),
            container="csr",
            data_dtype=None,
            index_dtype=None,
            allow_lossy=False,
            cast_back=True,
            target="cpu",
        )
        assert p.index_dtype == np.dtype(np.int32)
