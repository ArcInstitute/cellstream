"""shm.py: allocate, attach_shm_cleanup, close_shared_memory idempotency."""

import gc
from pathlib import Path


def test_allocate_and_cleanup(tmp_path):
    from cellstream.shm import allocate_shm_buffers

    shms = allocate_shm_buffers(data_nbytes=1024, indices_nbytes=2048, indptr_nbytes=512)
    try:
        assert shms.data.size >= 1024
        assert shms.indices.size >= 2048
        assert shms.indptr.size >= 512
        # /dev/shm files exist
        assert (Path("/dev/shm") / shms.data.name).exists()
    finally:
        shms.close_and_unlink()


def test_attach_cleanup_on_anndata_gc():
    import anndata as ad
    import scipy.sparse as sp

    from cellstream.shm import allocate_shm_buffers, attach_shm_cleanup

    shms = allocate_shm_buffers(data_nbytes=64, indices_nbytes=64, indptr_nbytes=64)
    a = ad.AnnData(X=sp.csr_matrix((1, 1), dtype="float32"))
    attach_shm_cleanup(a, shms)
    name = shms.data.name
    del a
    gc.collect()
    # /dev/shm/<name> should be gone now
    assert not Path(f"/dev/shm/{name}").exists()


def test_close_shared_memory_idempotent():
    import anndata as ad
    import scipy.sparse as sp

    from cellstream.shm import allocate_shm_buffers, attach_shm_cleanup

    shms = allocate_shm_buffers(data_nbytes=64, indices_nbytes=64, indptr_nbytes=64)
    a = ad.AnnData(X=sp.csr_matrix((1, 1), dtype="float32"))
    attach_shm_cleanup(a, shms)
    a.close_shared_memory()
    # Second call should not raise.
    a.close_shared_memory()


class _FakeBundle:
    def __init__(self):
        self.closed = 0
        self.unlinked = 0

    def close_and_unlink(self):
        self.closed += 1

    def unlink_segments(self):
        self.unlinked += 1


class _Holder:  # stands in for AnnData/MuData (accepts arbitrary attributes)
    pass


def test_attach_buffer_cleanup_multi_pins_and_closes_all():
    import scipy.sparse as sp

    from cellstream import shm

    arrs = [sp.csr_matrix((2, 2)), sp.csr_matrix((2, 2)), sp.csr_matrix((2, 2))]
    bundles = [_FakeBundle(), _FakeBundle(), _FakeBundle()]
    cont = _Holder()
    shm.attach_buffer_cleanup_multi(cont, zip(arrs, bundles, strict=True))
    # container owns the full list
    assert cont._cellstream_shm_bundles == bundles
    # each array is pinned to its own bundle
    for a, b in zip(arrs, bundles, strict=True):
        assert a._cellstream_shm_bundle is b
    # combined close fans out to every bundle; second call does not error
    cont.close_shared_memory()
    assert all(b.closed >= 1 for b in bundles)
    cont.close_shared_memory()


def test_attach_buffer_cleanup_multi_tolerates_unpinnable_array():
    import numpy as np

    from cellstream import shm

    plain = np.zeros(3)  # a plain ndarray rejects arbitrary attributes
    b = _FakeBundle()
    cont = _Holder()
    shm.attach_buffer_cleanup_multi(cont, [(plain, b)])  # must NOT raise
    assert cont._cellstream_shm_bundles == [b]
    cont.close_shared_memory()
    assert b.closed == 1


def test_attach_mudata_close_fans_out():
    from cellstream import shm

    a1, a2, a3 = _Holder(), _Holder(), _Holder()
    calls = []
    a1.close_shared_memory = lambda: calls.append("a1")
    a2.close_shared_memory = lambda: calls.append("a2")
    # a3 has NO close_shared_memory (heap modality) -> skipped, no error
    md = _Holder()
    shm.attach_mudata_close(md, [a1, a2, a3])
    md.close_shared_memory()
    assert calls == ["a1", "a2"]


def _drop_segment_names(*names):
    """Unlink /dev/shm entries by name (the owning SharedMemory objects are gone)."""
    from multiprocessing import shared_memory

    for nm in names:
        try:
            t = shared_memory.SharedMemory(name=nm)
            t.close()
            t.unlink()
        except FileNotFoundError:
            pass


def test_array_for_shared_memory_returns_plain_ndarray():
    """scipy/AnnData/pyo3 must see exactly np.ndarray, not the pinning subclass."""
    import numpy as np

    from cellstream.shm import allocate_shm_buffers, array_for_shared_memory

    bundle = allocate_shm_buffers(data_nbytes=400, indices_nbytes=400, indptr_nbytes=44)
    try:
        arr = array_for_shared_memory(bundle.data, (100,), np.dtype("float32"))
        assert type(arr) is np.ndarray, type(arr)
        assert arr.flags["C_CONTIGUOUS"] and arr.flags["WRITEABLE"]
    finally:
        bundle.close_and_unlink()


def test_array_for_shared_memory_pins_owner_in_base_chain():
    """The buffer-owning subclass holding the segment must survive numpy's base-collapse."""
    import numpy as np

    from cellstream.shm import _ShmBackedArray, allocate_shm_buffers, array_for_shared_memory

    bundle = allocate_shm_buffers(data_nbytes=400, indices_nbytes=400, indptr_nbytes=44)
    try:
        arr = array_for_shared_memory(bundle.data, (100,), np.dtype("float32"))
        assert isinstance(arr.base, _ShmBackedArray), type(arr.base)
        assert arr.base._cellstream_shm_segment is bundle.data
    finally:
        bundle.close_and_unlink()


def test_array_for_shared_memory_keeps_its_segment_alive():
    """The array alone must keep its SharedMemory object alive (issue #274).

    Ownership is asserted with a weakref FIRST. That ordering is what makes the read
    below safe in-process: by the time we dereference, the segment is already proven
    live, so a missing pin fails the assertion rather than segfaulting the test runner.
    The end-to-end dereference after a real container GC lives in the subprocess tests.
    """
    import weakref

    import numpy as np

    from cellstream.shm import allocate_shm_buffers, array_for_shared_memory

    bundle = allocate_shm_buffers(data_nbytes=400, indices_nbytes=400, indptr_nbytes=44)
    names = (bundle.data.name, bundle.indices.name, bundle.indptr.name)
    try:
        arr = array_for_shared_memory(bundle.data, (100,), np.dtype("float32"))
        arr[:] = 1.5
        wr = weakref.ref(bundle.data)
        # Drop the bundle and hence every direct reference to all three segments.
        del bundle
        gc.collect()
        gc.collect()
        assert wr() is not None, "segment was collected -- the array does not pin it"
        assert float(arr.sum()) == 150.0
        del arr
        gc.collect()
        gc.collect()
        assert wr() is None, "segment outlived its only array -- the pin leaks"
    finally:
        _drop_segment_names(*names)


def test_array_for_shared_memory_pin_is_per_segment_not_per_bundle():
    """Holding one array must NOT keep the OTHER segments mapped (per-segment pin).

    Whole-bundle pinning would let a retained indptr keep a huge data segment resident.
    """
    import weakref

    import numpy as np

    from cellstream.shm import allocate_shm_buffers, array_for_shared_memory

    bundle = allocate_shm_buffers(data_nbytes=400, indices_nbytes=400, indptr_nbytes=44)
    names = (bundle.data.name, bundle.indices.name, bundle.indptr.name)
    try:
        indptr = array_for_shared_memory(bundle.indptr, (11,), np.dtype("int32"))
        wr_data = weakref.ref(bundle.data)
        wr_indptr = weakref.ref(bundle.indptr)
        del bundle
        gc.collect()
        gc.collect()
        assert wr_indptr() is not None, "the held array's own segment was released"
        assert wr_data() is None, "an unheld segment stayed alive -- pin is per-bundle"
        assert indptr.shape == (11,)
    finally:
        _drop_segment_names(*names)


def test_array_for_shared_memory_pin_survives_derived_views():
    """np.asarray / .view() / a basic slice all keep the owner in the .base chain."""
    import weakref

    import numpy as np

    from cellstream.shm import allocate_shm_buffers, array_for_shared_memory

    for make in (
        lambda a: np.asarray(a),
        lambda a: a.view(np.float32),
        lambda a: a[10:20],
    ):
        bundle = allocate_shm_buffers(data_nbytes=400, indices_nbytes=400, indptr_nbytes=44)
        names = (bundle.data.name, bundle.indices.name, bundle.indptr.name)
        try:
            arr = array_for_shared_memory(bundle.data, (100,), np.dtype("float32"))
            arr[:] = np.arange(100, dtype="float32")
            derived = make(arr)
            wr = weakref.ref(bundle.data)
            del bundle, arr
            gc.collect()
            gc.collect()
            assert wr() is not None, f"{make} lost the pin"
            assert float(derived.sum()) > 0.0
        finally:
            _drop_segment_names(*names)


def test_array_for_shared_memory_pin_survives_scipy_csr_assembly():
    """A csr_matrix built by attribute assignment keeps each segment alive via its array."""
    import weakref

    import numpy as np
    import scipy.sparse as sp

    from cellstream.shm import allocate_shm_buffers, array_for_shared_memory

    bundle = allocate_shm_buffers(data_nbytes=400, indices_nbytes=400, indptr_nbytes=44)
    names = (bundle.data.name, bundle.indices.name, bundle.indptr.name)
    try:
        data = array_for_shared_memory(bundle.data, (100,), np.dtype("float32"))
        indices = array_for_shared_memory(bundle.indices, (100,), np.dtype("int32"))
        indptr = array_for_shared_memory(bundle.indptr, (11,), np.dtype("int32"))
        data[:] = 1.5
        indices[:] = np.tile(np.arange(10, dtype="int32"), 10)
        indptr[:] = np.arange(11, dtype="int32") * 10

        X = sp.csr_matrix((10, 10), dtype="float32")
        X.data, X.indices, X.indptr = data, indices, indptr
        extracted = X.data
        wr = weakref.ref(bundle.data)
        del bundle, X, data, indices, indptr
        gc.collect()
        gc.collect()
        assert wr() is not None, "extracting X.data lost the segment pin"
        assert float(extracted.sum()) == 150.0
    finally:
        _drop_segment_names(*names)
