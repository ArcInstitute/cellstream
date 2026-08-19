"""Regression tests for anndata 0.13's ``layers[None]`` alias for ``X``.

On anndata 0.13, ``ad.AnnData(X=X).layers.keys() == [None]`` (was ``[]`` on
<=0.12). cellstream must skip that ``None`` key everywhere it enumerates layers,
else an X-only AnnData is mis-detected as multi-matrix and single-matrix writes
get mis-routed into the v2 multi-matrix path. These tests assert the correct
single-matrix behavior, so they pass trivially on anndata <=0.12 and guard the
regression on >=0.13.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellstream import matrices as mx
from cellstream.read import ShardedArchive
from cellstream.shard_io import write_one_h5ad
from cellstream.write import _source_is_multimatrix, write_sharded


def _xonly(n_obs=6, n_var=5):
    """X-only AnnData with small integer-valued float32 X (narrows to u16)."""
    vals = (np.arange(1, n_obs * n_var + 1) % 7).reshape(n_obs, n_var).astype(np.float32)
    return ad.AnnData(X=sp.csr_matrix(vals))


# --- Task 1: the helper -------------------------------------------------------


def test_real_layer_names_xonly_is_empty():
    assert mx.real_layer_names(_xonly()) == []


def test_real_layer_names_keeps_real_layers_only():
    a = _xonly()
    a.layers["spliced"] = a.X.copy()
    assert mx.real_layer_names(a) == ["spliced"]


# --- Task 2: decompose --------------------------------------------------------


def test_decompose_xonly_is_just_x():
    assert [s.key for s in mx.decompose_anndata(_xonly())] == ["x"]


def test_decompose_keeps_real_layer_skips_none():
    a = _xonly()
    a.layers["spliced"] = a.X.copy()
    assert [s.key for s in mx.decompose_anndata(a)] == ["x", "layer:spliced"]


# --- Task 3: multi-matrix detection ------------------------------------------


def test_source_is_multimatrix_false_for_xonly():
    assert _source_is_multimatrix(_xonly()) is False


def test_source_is_multimatrix_true_with_real_layer():
    a = _xonly()
    a.layers["counts"] = a.X.copy()
    assert _source_is_multimatrix(a) is True


# --- Task 4: shard_io narrow rewrite -----------------------------------------


def test_shard_io_narrow_rewrite_xonly_roundtrips(tmp_path):
    # write_one_h5ad builds ad.AnnData(X=X_narrow, layers=dict(adata.layers)); on
    # 0.13 that is layers={None: adata.X}, and anndata raises "layers[None] and X
    # must be identical" because X_narrow (uint16) != adata.X (float32).
    a = _xonly()
    p = tmp_path / "narrow.h5ad"
    write_one_h5ad(a, p, codec="blosc-zstd-1-u16u16")
    out = ad.read_h5ad(p)
    assert mx.real_layer_names(out) == []
    np.testing.assert_array_equal(out.X.toarray(), a.X.toarray())


# --- Task 5: end-to-end write_sharded round-trip, both engines ---------------


@pytest.mark.parametrize("disable_rust", [False, True])
def test_write_sharded_xonly_single_matrix_roundtrip(tmp_path, monkeypatch, disable_rust):
    if disable_rust:
        monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    a = _xonly(n_obs=40, n_var=8)
    out = tmp_path / "arch.shad"
    write_sharded(a, out)  # default format (v2 packed)

    arch = ShardedArchive(str(out))
    # An X-only archive must carry the flat single-matrix dtype keys, not a
    # multi-matrix layout (which lacks x_data_dtype_on_disk).
    assert "x_data_dtype_on_disk" in arch.manifest

    back = arch.to_anndata()
    assert mx.real_layer_names(back) == []
    np.testing.assert_array_equal(back.X.toarray(), a.X.toarray())
