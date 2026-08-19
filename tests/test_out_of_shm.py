"""Out-of-shm materialization (#59/#54): preflight + file-mmap backing."""

import numpy as np
import pytest


def test_insufficient_shm_error_is_cellstream_error():
    from cellstream.errors import CellstreamError, InsufficientShmError

    assert issubclass(InsufficientShmError, CellstreamError)
    with pytest.raises(CellstreamError):
        raise InsufficientShmError("boom")


def test_allocate_mmap_buffers_sparse_typed_and_cleanup(tmp_path):
    import os

    from cellstream import shm as _shm

    # Large logical size so the sparseness check is robust to block rounding
    # (a tiny file's 1-byte tail-write rounds up to a full block that can
    # EXCEED the logical size; use ~40 MB instead).
    n = 10_000_000
    b = _shm.allocate_mmap_buffers(
        spill_dir=tmp_path,
        total_nnz=n,
        n_indptr=51,
        data_dtype=np.dtype("float32"),
        indices_dtype=np.dtype("int32"),
    )
    # Typed memmaps of the requested shape/dtype.
    assert b.data.dtype == np.float32 and b.data.shape == (n,)
    assert b.indices.dtype == np.int32 and b.indices.shape == (n,)
    assert b.indptr.dtype == np.int64 and b.indptr.shape == (51,)
    # Sparse: logical size is full (40 MB), physical allocation is ~a page.
    st = os.stat(b.data.filename)
    assert st.st_size == n * 4
    assert st.st_blocks * 512 < st.st_size  # robust at 40 MB logical
    bundle_dir = b.dir
    assert bundle_dir.is_dir()

    # Writes are readable back through the same memmap.
    b.data[:3] = [1.0, 2.0, 3.0]
    assert list(b.data[:3]) == [1.0, 2.0, 3.0]

    # unlink_segments removes files AND the (now-empty) dir; mapping stays valid.
    b.unlink_segments()
    assert not bundle_dir.exists()
    assert list(b.data[:3]) == [1.0, 2.0, 3.0]  # mapping still valid post-unlink

    # Idempotent.
    b.unlink_segments()
    b.close_and_unlink()


def test_allocate_mmap_buffers_atomic_on_failure(tmp_path, monkeypatch):
    from cellstream import shm as _shm

    calls = {"n": 0}
    real_memmap = np.memmap

    def flaky_memmap(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:  # fail on the third (indptr) allocation
            raise OSError("simulated allocation failure")
        return real_memmap(*a, **k)

    monkeypatch.setattr(_shm.np, "memmap", flaky_memmap)
    with pytest.raises(OSError):
        _shm.allocate_mmap_buffers(
            spill_dir=tmp_path,
            total_nnz=10,
            n_indptr=5,
            data_dtype=np.dtype("float32"),
            indices_dtype=np.dtype("int32"),
        )
    # No leaked temp subdir.
    assert list(tmp_path.glob("cellstream-mmap-*")) == []


_F32 = np.dtype("float32")
_I32 = np.dtype("int32")


def _alloc_kwargs(tmp_path, backing, total_nnz=1000):
    return dict(
        backing=backing,
        spill_dir=tmp_path,
        total_nnz=total_nnz,
        n_indptr=51,
        data_dtype=_F32,
        indices_dtype=_I32,
    )


def test_select_and_allocate_validation(tmp_path):
    from cellstream import shm as _shm

    with pytest.raises(ValueError, match="backing must be"):
        _shm.select_and_allocate(**{**_alloc_kwargs(tmp_path, "bogus")})
    with pytest.raises(ValueError, match="requires spill_dir"):
        _shm.select_and_allocate(
            backing="mmap",
            spill_dir=None,
            total_nnz=10,
            n_indptr=5,
            data_dtype=_F32,
            indices_dtype=_I32,
        )
    with pytest.raises(ValueError, match="requires spill_dir"):
        _shm.select_and_allocate(
            backing="auto",
            spill_dir=None,
            total_nnz=10,
            n_indptr=5,
            data_dtype=_F32,
            indices_dtype=_I32,
        )


def test_select_and_allocate_shm_insufficient(tmp_path, monkeypatch):
    from cellstream import shm as _shm
    from cellstream.errors import InsufficientShmError

    monkeypatch.setattr(_shm, "_free_bytes", lambda path: 1024)  # tiny
    with pytest.raises(InsufficientShmError, match=r"/dev/shm"):
        _shm.select_and_allocate(**_alloc_kwargs(tmp_path, "shm"))


def test_select_and_allocate_mmap_insufficient(tmp_path, monkeypatch):
    from cellstream import shm as _shm
    from cellstream.errors import InsufficientShmError

    monkeypatch.setattr(_shm, "_free_bytes", lambda path: 1024)
    with pytest.raises(InsufficientShmError, match=str(tmp_path)):
        _shm.select_and_allocate(**_alloc_kwargs(tmp_path, "mmap"))


def test_select_and_allocate_auto_falls_back_to_mmap(tmp_path, monkeypatch):
    from cellstream import shm as _shm

    # shm too small, spill_dir huge.
    def fake_free(path):
        return 1024 if str(path) == _shm.SHM_DIR else 10**12

    monkeypatch.setattr(_shm, "_free_bytes", fake_free)
    with pytest.warns(UserWarning, match="spilling to"):
        bundle, chosen = _shm.select_and_allocate(**_alloc_kwargs(tmp_path, "auto"))
    assert chosen == "mmap"
    assert isinstance(bundle, _shm.MmapBundle)
    bundle.close_and_unlink()


def test_select_and_allocate_auto_uses_shm_when_fits(tmp_path, monkeypatch):
    from cellstream import shm as _shm

    monkeypatch.setattr(_shm, "_free_bytes", lambda path: 10**12)
    bundle, chosen = _shm.select_and_allocate(**_alloc_kwargs(tmp_path, "auto"))
    assert chosen == "shm"
    assert isinstance(bundle, _shm.ShmBundle)
    bundle.close_and_unlink()


def test_required_bytes_page_rounded(tmp_path):
    from cellstream import shm as _shm

    # 1 element each → each buffer rounds up to one page.
    req = _shm._required_bytes(total_nnz=1, n_indptr=1, data_dtype=_F32, indices_dtype=_I32)
    assert req == 3 * _shm._PAGE


def test_attach_buffer_cleanup_mmap(tmp_path):
    import gc

    import anndata as ad
    import scipy.sparse as sp

    from cellstream import shm as _shm

    bundle = _shm.allocate_mmap_buffers(
        spill_dir=tmp_path, total_nnz=0, n_indptr=1, data_dtype=_F32, indices_dtype=_I32
    )
    X = sp.csr_matrix((2, 3), dtype=_F32)
    adata = ad.AnnData(X=X)
    _shm.attach_buffer_cleanup(adata, bundle)
    assert hasattr(adata, "close_shared_memory")
    bundle_dir = bundle.dir
    assert bundle_dir.is_dir()

    # GC path: weakref finalizer runs unlink_segments → dir removed.
    del adata
    gc.collect()
    assert not bundle_dir.exists()


def test_attach_shm_cleanup_alias_exists():
    from cellstream import shm as _shm

    assert _shm.attach_shm_cleanup is _shm.attach_buffer_cleanup


def test_read_h5ad_single_file_backing_contract(tmp_path):
    import anndata as ad
    import scipy.sparse as sp

    import cellstream

    a = ad.AnnData(X=sp.csr_matrix(np.eye(4, dtype=np.float32)))
    f = tmp_path / "single.h5ad"
    a.write_h5ad(f)

    # Bogus backing → ValueError at dispatcher entry (before the path branch).
    with pytest.raises(ValueError, match="backing must be"):
        cellstream.read_h5ad(f, backing="bogus")
    # Explicit out-of-shm on a single file → NotImplementedError (not silent no-op).
    with pytest.raises(NotImplementedError, match="sharded archive"):
        cellstream.read_h5ad(f, backing="mmap", spill_dir=tmp_path)
    with pytest.raises(NotImplementedError, match="sharded archive"):
        cellstream.read_h5ad(f, spill_dir=tmp_path)
    # Default path still reads the file.
    assert cellstream.read_h5ad(f).shape == (4, 4)


def test_read_h5ad_v2_packed_rejects_out_of_shm(tmp_path):
    """A single-file packed archive is a v2 archive, so out-of-shm backing must
    raise the v2 'v1-only' message (#106 Gap B) — not the 'single .h5ad files'
    message that the single-file branch would give before packed is detected."""
    import anndata as ad
    import scipy.sparse as sp

    import cellstream
    from cellstream.packed.reader import is_packed_file
    from cellstream.write import write_sharded

    rng = np.random.default_rng(8)
    n_obs, n_var = 200, 40
    nnz = int(n_obs * n_var * 0.1)
    X = sp.csr_matrix(
        (
            rng.integers(0, 100, size=nnz).astype(np.float32),
            (rng.integers(0, n_obs, size=nnz), rng.integers(0, n_var, size=nnz)),
        ),
        shape=(n_obs, n_var),
    )
    a = ad.AnnData(X=X)
    out = tmp_path / "arch.shardpack"
    write_sharded(a, out, format="v2", n_shards=4, n_workers=2)
    assert is_packed_file(out)
    spill = tmp_path / "spill"
    spill.mkdir()

    with pytest.raises(NotImplementedError, match="v1-only"):
        cellstream.read_h5ad(out, backing="mmap", spill_dir=spill)
    with pytest.raises(NotImplementedError, match="v1-only"):
        cellstream.read_h5ad(out, spill_dir=spill)
    # shm (default) backing still reads the packed archive fine.
    back = cellstream.read_h5ad(out, n_workers=2)
    assert back.X.nnz == a.X.nnz


def _packed_for_backend(tmp_path):
    """A packed v2 archive plus a registered anndata hook, or skip."""
    import anndata as ad
    import scipy.sparse as sp

    import cellstream  # noqa: F401  (import triggers monkey-patch registration)
    from cellstream import anndata_backend
    from cellstream.write import write_sharded

    anndata_backend.register()
    if not getattr(ad.read_h5ad, "_cellstream_wrapped", False):
        pytest.skip("anndata monkey-patch not active in this environment")

    X = sp.random(40, 8, density=0.3, format="csr", dtype="float32", random_state=1)
    X.data = np.ceil(X.data * 10).astype("float32")
    out = tmp_path / "packed.shad"
    write_sharded(ad.AnnData(X=X), out, n_shards=2, n_workers=2)
    return ad, out


@pytest.mark.parametrize("knob", ["backing", "spill_dir"])
def test_anndata_backend_forwards_each_out_of_shm_knob_independently(tmp_path, knob):
    """The ``anndata.read_h5ad`` monkey-patch must forward ``backing`` AND ``spill_dir``,
    not swallow either.

    Before #154 this was proved by handing the hook a v1 archive and asserting the result was
    memmap-backed. Out-of-shm backing was v1-only, so with v1 gone no archive accepts it — but
    the property is the same and still observable: a forwarded request must surface the
    reader's rejection instead of silently falling back to the default /dev/shm path, which is
    exactly the bypass this hook exists to prevent.

    Each knob is passed ALONE and parametrized. Passing both together would pass even if one
    of the two ``kwargs.pop`` calls in anndata_backend were deleted, since the reader rejects
    when *either* is non-default — that is the hole this shape closes.
    """
    ad, out = _packed_for_backend(tmp_path)
    spill = tmp_path / "spill"
    spill.mkdir()
    kwargs = {"backing": "mmap"} if knob == "backing" else {"spill_dir": spill}

    with pytest.raises(NotImplementedError, match="out-of-shm backing"):
        ad.read_h5ad(out, **kwargs)


def test_anndata_backend_default_read_still_works(tmp_path):
    """Control for the two rejection tests above: without the knobs the same archive reads
    fine, so those raises are the forwarded request being rejected -- not an unreadable
    archive or a broken hook."""
    ad, out = _packed_for_backend(tmp_path)
    assert ad.read_h5ad(out).shape == (40, 8)
