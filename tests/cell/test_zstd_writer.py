"""Writer codec-resolution + zstd-int / zstd-float archive tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from .conftest import make_cell_adata


def _write(tmp_path, adata, **kw):
    from cellstream.cell.writer import write_cell_archive

    out = tmp_path / "a.shad"
    write_cell_archive(adata, out, overwrite=True, **kw)
    return out


def test_zstd_int_roundtrip_and_manifest(tmp_path):
    from cellstream.cell.reader import open_cell

    a = make_cell_adata()
    out = _write(tmp_path, a, codec="zstd")
    store = open_cell(out)
    assert store.manifest["schema_version"] == 6
    assert store.manifest["codec"] == "zstd"
    assert store.manifest["value_dtype_on_disk"] == "uint32"
    assert store.manifest["index_dtype_on_disk"] == "uint32"
    assert store.manifest["pyfastpfor_codec"] is None
    X = store.gather_rows(np.arange(a.n_obs))
    expected = a.X.tocsr()
    np.testing.assert_array_equal(X.toarray(), expected.toarray())
    store.close()


def _dense_skewed_adata(n_obs=300, n_vars=5000, nnz=600_000, seed=7):
    """Dense cells (~high nnz/cell) with a skewed count distribution (mostly 1/2/3),
    like real scRNA UMI counts. This is the regime where zstd's entropy coding beats
    FastPFor's bit floor: per-cell zstd framing overhead (~2 frames/cell) amortizes at
    high nnz/cell, and skewed counts compress below FastPFor's magnitude bit-width. On
    sparse or uniform data zstd-int is NOT smaller (see PR #216 size measurement)."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n_obs, size=nnz)
    cols = rng.integers(0, n_vars, size=nnz)
    vals = np.minimum(rng.geometric(0.55, size=nnz), 60000).astype(np.uint32)
    csr = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    csr.sum_duplicates()
    x = sp.csr_matrix(csr.shape, dtype=np.uint32)
    x.data = csr.data.astype(np.uint32)
    x.indices = csr.indices.astype(np.int32)
    x.indptr = csr.indptr.astype(np.int64)
    a = ad.AnnData(X=x)
    a.var_names = [f"g{i}" for i in range(n_vars)]
    return a


def test_zstd_int_smaller_than_pfordelta(tmp_path):
    # zstd entropy-codes the skewed count distribution below FastPFor's bit floor.
    # The win is density-dependent (see the module docstring for _dense_skewed_adata):
    # demonstrated here on dense skewed cells; definitive number on real CCL data comes
    # from the Phase 6 SLURM validation.
    from cellstream.cell.writer import write_cell_archive

    a = _dense_skewed_adata()
    p = tmp_path / "p.shad"
    z = tmp_path / "z.shad"
    write_cell_archive(a, p, codec="pfordelta", overwrite=True)
    write_cell_archive(a, z, codec="zstd", overwrite=True)
    assert z.stat().st_size < p.stat().st_size


def test_pfordelta_default_unchanged(tmp_path):
    from cellstream.cell.reader import open_cell

    a = make_cell_adata()
    out = _write(tmp_path, a)  # codec=None -> auto -> pfordelta for integer counts
    store = open_cell(out)
    assert store.manifest["schema_version"] == 5
    assert store.manifest["codec"] == "pfordelta"
    store.close()


def test_pfordelta_on_float_data_raises(tmp_path):
    a = make_cell_adata()
    x = a.X.tocsr().astype(np.float32)
    x.data[0] = 1.5  # genuine float
    a.X = x
    with pytest.raises(ValueError, match="requires integer counts"):
        _write(tmp_path, a, codec="pfordelta")


def test_data_dtype_rejected_on_integer_data(tmp_path):
    a = make_cell_adata()
    with pytest.raises(ValueError, match="data_dtype"):
        _write(tmp_path, a, codec="zstd", data_dtype="float32")


def test_zstd_int_rejects_negative(tmp_path):
    # A signed-int input with negatives + codec="zstd" must fail loud, not silently
    # wrap on astype(uint32). (Gemini PR #216, HIGH.)
    a = make_cell_adata()
    x = a.X.tocsr().astype(np.int32)
    x.data[0] = -5
    a.X = x
    with pytest.raises(ValueError, match="non-negative"):
        _write(tmp_path, a, codec="zstd")


def test_public_write_layout_cell_zstd(tmp_path):
    # end-to-end through the public write_sharded(layout="cell") entry.
    import cellstream
    from cellstream.cell.reader import open_cell

    a = make_cell_adata()
    out = tmp_path / "pub.shad"
    cellstream.write_sharded(a, out, layout="cell", codec="zstd", overwrite=True)
    store = open_cell(out)
    assert store.manifest["codec"] == "zstd"
    store.close()


# --- zstd float storage (float32 / float64) ---


def _float_adata(dtype, with_nonfinite=False):
    import anndata as ad

    rng = np.random.default_rng(5)
    n_obs, n_vars = 120, 40
    x = sp.random(n_obs, n_vars, density=0.2, random_state=rng, dtype=np.float64)
    x = x.tocsr()
    x.data = (x.data * 10.0 + 0.5).astype(dtype)  # genuine (non-integer) floats
    x.data[0] = np.asarray(0.5, dtype)  # force >=1 fractional value (deterministic genuine-float)
    if with_nonfinite and x.data.size >= 3:
        x.data[0], x.data[1], x.data[2] = np.nan, np.inf, -np.inf
    a = ad.AnnData(X=x)
    a.var_names = [f"g{i:03d}" for i in range(n_vars)]
    return a


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zstd_float_preserve_width_default(tmp_path, dtype):
    from cellstream.cell.reader import open_cell, read_cell_bulk

    a = _float_adata(dtype)
    out = _write(tmp_path, a)  # codec=None auto -> genuine float -> zstd
    store = open_cell(out)
    assert store.manifest["codec"] == "zstd"
    assert store.manifest["schema_version"] == 6
    expected_disk = "float64" if dtype is np.float64 else "float32"
    assert store.manifest["value_dtype_on_disk"] == expected_disk
    store.close()
    adata = read_cell_bulk(out)
    assert adata.X.dtype == np.dtype(expected_disk)
    np.testing.assert_array_equal(adata.X.toarray(), a.X.toarray().astype(expected_disk))


def test_zstd_float64_nonfinite_bit_exact(tmp_path):
    from cellstream.cell.reader import read_cell_bulk

    a = _float_adata(np.float64, with_nonfinite=True)
    out = _write(tmp_path, a)
    adata = read_cell_bulk(out)
    got = adata.X.toarray()
    exp = a.X.toarray()
    assert np.array_equal(got, exp, equal_nan=True)


def test_data_dtype_float32_downcasts_f64(tmp_path):
    from cellstream.cell.reader import open_cell

    a = _float_adata(np.float64)
    out = _write(tmp_path, a, codec="zstd", data_dtype="float32", allow_lossy=True)
    store = open_cell(out)
    assert store.manifest["value_dtype_on_disk"] == "float32"
    store.close()


def test_data_dtype_float32_lossless_checked_raises(tmp_path):
    from cellstream.errors import LossyCastError

    a = _float_adata(np.float64)
    # ensure at least one value is NOT float32-representable
    x = a.X.tocsr()
    x.data[0] = np.float64(1) / np.float64(3)  # 0.3333... not f32-exact
    a.X = x
    with pytest.raises(LossyCastError):
        _write(tmp_path, a, codec="zstd", data_dtype="float32")  # allow_lossy defaults False


def test_data_dtype_float64_upcasts_f32(tmp_path):
    from cellstream.cell.reader import open_cell

    a = _float_adata(np.float32)
    out = _write(tmp_path, a, codec="zstd", data_dtype="float64")
    store = open_cell(out)
    assert store.manifest["value_dtype_on_disk"] == "float64"
    store.close()


def test_invalid_data_dtype_raises(tmp_path):
    a = _float_adata(np.float64)
    with pytest.raises(ValueError, match="data_dtype must be"):
        _write(tmp_path, a, codec="zstd", data_dtype="uint16")


def _empty_float_adata(dtype, n_obs=4, n_vars=6):
    import anndata as ad

    a = ad.AnnData(X=sp.csr_matrix((n_obs, n_vars), dtype=dtype))
    a.var_names = [f"g{i}" for i in range(n_vars)]
    return a


def _intvalued_float_adata(dtype, n_obs=8, n_vars=10):
    import anndata as ad

    rng = np.random.default_rng(3)
    x = sp.random(n_obs, n_vars, density=0.4, random_state=rng, dtype=np.float64).tocsr()
    x.data = np.rint(x.data * 5 + 1).astype(dtype)  # whole numbers stored as float
    a = ad.AnnData(X=x)
    a.var_names = [f"g{i}" for i in range(n_vars)]
    return a


def test_empty_float_with_explicit_data_dtype_stores_float(tmp_path):
    # Regression (Gemini #216 HIGH): an empty float array + explicit data_dtype must
    # store float, not raise or misroute to pfordelta.
    from cellstream.cell.reader import open_cell

    a = _empty_float_adata(np.float32)
    out = _write(tmp_path, a, codec="zstd", data_dtype="float32")
    store = open_cell(out)
    assert store.manifest["codec"] == "zstd"
    assert store.manifest["value_dtype_on_disk"] == "float32"
    store.close()


def test_intvalued_float_explicit_data_dtype_stores_float(tmp_path):
    # Integer-valued float + explicit data_dtype -> float storage (honors the request).
    from cellstream.cell.reader import open_cell, read_cell_bulk

    a = _intvalued_float_adata(np.float64)
    out = _write(tmp_path, a, codec="zstd", data_dtype="float64")
    store = open_cell(out)
    assert store.manifest["value_dtype_on_disk"] == "float64"
    store.close()
    got = read_cell_bulk(out).X.toarray()
    np.testing.assert_array_equal(got, a.X.toarray())


def test_intvalued_float_zstd_no_data_dtype_is_size_win_uint32(tmp_path):
    # codec="zstd" on integer-valued float WITHOUT data_dtype stays the zstd-int
    # size-win path (spec 5.1) — uint32 on disk, NOT float.
    from cellstream.cell.reader import open_cell

    a = _intvalued_float_adata(np.float32)
    out = _write(tmp_path, a, codec="zstd")
    store = open_cell(out)
    assert store.manifest["value_dtype_on_disk"] == "uint32"
    store.close()


def test_empty_float_auto_routes_pfordelta(tmp_path):
    # Empty float, no codec/data_dtype: no crash; defaults to pfordelta (empty archive).
    from cellstream.cell.reader import open_cell

    a = _empty_float_adata(np.float64)
    out = _write(tmp_path, a)
    store = open_cell(out)
    assert store.manifest["codec"] == "pfordelta"
    store.close()


# --- optimistic encode + warn + retry (replaces the O(nnz) codec scan; #216 review) ---


def test_fractional_float_warns_and_falls_back_to_zstd(tmp_path):
    """codec=None on FRACTIONAL float data: the pfordelta attempt is rejected by the encoder,
    we warn, discard it, and re-encode as zstd float. Same outcome the old scan produced --
    without scanning."""
    import anndata as ad

    from cellstream.cell.reader import open_cell, read_cell_bulk
    from cellstream.cell.writer import write_cell_archive

    X = sp.csr_matrix(np.array([[1.0, 2.5], [0.0, 3.25]], dtype=np.float32))
    a = ad.AnnData(X=X)
    a.var_names = ["g0", "g1"]
    out = tmp_path / "auto.shad"
    with pytest.warns(UserWarning, match="re-encoding with codec='zstd'"):
        write_cell_archive(a, out, overwrite=True)

    s = open_cell(str(out))
    try:
        assert s.manifest["codec"] == "zstd"
        assert s.manifest["value_dtype_on_disk"] == "float32"
    finally:
        s.close()
    np.testing.assert_array_equal(read_cell_bulk(out).X.toarray(), X.toarray())


def test_fallback_archive_is_byte_identical_to_explicit_zstd(tmp_path):
    """THE GATE. The auto-fallback must produce the SAME payload as asking for zstd directly.
    Compare payload_sha256 -- NEVER filecmp (manifest.json embeds created_at)."""
    import anndata as ad

    from cellstream.cell.reader import open_cell
    from cellstream.cell.writer import write_cell_archive

    rng = np.random.default_rng(3)
    dense = rng.random((60, 25), dtype=np.float32)
    dense[dense < 0.7] = 0.0
    X = sp.csr_matrix(dense)
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(25)]

    auto, explicit = tmp_path / "auto.shad", tmp_path / "explicit.shad"
    with pytest.warns(UserWarning):
        write_cell_archive(a, auto, overwrite=True, payload_checksum=True)
    write_cell_archive(
        a, explicit, codec="zstd", data_dtype="float32", overwrite=True, payload_checksum=True
    )

    def sha(p):
        s = open_cell(str(p))
        try:
            return s.manifest["payload_sha256"]
        finally:
            s.close()

    assert sha(auto) is not None
    assert sha(auto) == sha(explicit)


def test_integral_float_takes_pfordelta_with_no_warning(tmp_path):
    """The COMMON case: float32 holding integral counts. Optimistic pfordelta SUCCEEDS, so
    there is no retry, no warning -- and, crucially, no scan."""
    import warnings

    import anndata as ad

    from cellstream.cell.reader import open_cell
    from cellstream.cell.writer import write_cell_archive

    X = sp.csr_matrix(np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32))
    a = ad.AnnData(X=X)
    a.var_names = ["g0", "g1"]
    out = tmp_path / "int.shad"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # ANY warning fails the test
        write_cell_archive(a, out, overwrite=True)
    s = open_cell(str(out))
    try:
        assert s.manifest["codec"] == "pfordelta"
    finally:
        s.close()


def test_explicit_pfordelta_on_fractional_still_raises(tmp_path):
    """Explicit codec='pfordelta' gets NO fallback -- we must not silently switch codecs
    behind the user's back. The actionable message is preserved."""
    import anndata as ad

    from cellstream.cell.writer import write_cell_archive

    X = sp.csr_matrix(np.array([[1.0, 2.5]], dtype=np.float32))
    a = ad.AnnData(X=X)
    a.var_names = ["g0", "g1"]
    with pytest.raises(ValueError, match="requires integer counts"):
        write_cell_archive(a, tmp_path / "x.shad", codec="pfordelta", overwrite=True)


def test_clear_cell_staging_sweeps_rust_part_files(tmp_path):
    """_clear_cell_staging must ALSO glob `x_payload.part*`.

    The Rust encoder writes one staging file per worker (`{payload}.part{N}`, pfor.rs) and
    removes them only on its SUCCESS path -- a rejected value returns Err before that cleanup,
    so they survive the failed attempt. The worker count is not knowable at the call site, so
    they cannot be enumerated by name; they must be globbed. The sweep must also be surgical:
    a non-staging member sharing the dir survives.
    """
    from cellstream.cell import format as fmt
    from cellstream.cell.writer import _clear_cell_staging

    for name in (fmt.PAYLOAD, fmt.OFFSETS, fmt.INDPTR):
        (tmp_path / name).write_bytes(b"stale")
    for w in range(5):
        (tmp_path / f"{fmt.PAYLOAD}.part{w}").write_bytes(b"leaked")
    (tmp_path / fmt.MANIFEST).write_bytes(b"{}")  # not staging -> must survive

    _clear_cell_staging(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [fmt.MANIFEST]

    _clear_cell_staging(tmp_path)  # idempotent: a no-op on an already-clean dir
    assert sorted(p.name for p in tmp_path.iterdir()) == [fmt.MANIFEST]


def test_fallback_sweeps_staging_before_retry(tmp_path, monkeypatch):
    """The retry must start from a CLEAN staging dir -- including the Rust encoder's
    `x_payload.part*` worker files (see test_clear_cell_staging_sweeps_rust_part_files). At
    6.5B nnz a leaked set is GBs of dead staging held for the whole re-encode.

    Spies BOTH sides of the clear: what the staging dir held when the sweep was entered, and
    what the retry attempt actually saw.
    """
    import anndata as ad

    from cellstream.cell import writer as W
    from cellstream.cell._rust_dispatch import use_rust_for_cell_encode

    seen_before_clear = []
    attempt_saw = []
    real_clear, real_encode = W._clear_cell_staging, W._encode_cell_payload

    def spy_clear(tmpd):
        seen_before_clear.append(sorted(p.name for p in Path(tmpd).iterdir()))
        return real_clear(tmpd)

    def spy_encode(X, perm, n_obs, n_vars, tmpd, **kw):
        attempt_saw.append(sorted(p.name for p in Path(tmpd).iterdir()))
        return real_encode(X, perm, n_obs, n_vars, tmpd, **kw)

    monkeypatch.setattr(W, "_clear_cell_staging", spy_clear)
    monkeypatch.setattr(W, "_encode_cell_payload", spy_encode)

    # Enough cells that the Rust encoder splits across several workers -> several .part files.
    rng = np.random.default_rng(11)
    dense = rng.random((256, 40), dtype=np.float32)
    dense[dense < 0.6] = 0.0
    dense[0, 0] = 2.5  # a fractional value -> the optimistic pfordelta attempt is rejected
    X = sp.csr_matrix(dense)
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(40)]

    with pytest.warns(UserWarning, match="re-encoding with codec='zstd'"):
        W.write_cell_archive(a, tmp_path / "a.shad", overwrite=True)

    assert len(attempt_saw) == 2, "expected the optimistic attempt + exactly one retry"
    assert len(seen_before_clear) == 1

    retry_saw = attempt_saw[1]
    assert not [n for n in retry_saw if n.startswith(W.fmt.PAYLOAD)], retry_saw
    assert "x_offsets" not in retry_saw and "x_indptr" not in retry_saw

    # On the Rust engine, prove the sweep is not vacuous: the rejected attempt really did
    # leave per-worker .part files behind for it to remove. (Under CELLSTREAM_DISABLE_RUST=1 the
    # Python arm raises before opening any file, so there is nothing to leak.)
    if use_rust_for_cell_encode("pfordelta", X.data.dtype, X.indices.dtype, "uint32"):
        assert any(".part" in n for n in seen_before_clear[0]), seen_before_clear[0]
