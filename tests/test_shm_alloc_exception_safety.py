"""#287 defect 2: the shm.allocate_* family must not strand what it has already created.

A separate module from tests/test_shm.py so the whole #287 oracle -- positive control,
negative controls, injections -- reads in one place.

The oracle is NAME-based, never count-based: a global /dev/shm inventory is machine-wide and
races any concurrent test run (see [[concurrent-pytest-breaks-shm-leak-tests]]). The injected
constructor records the names it really created and we ask whether THOSE paths survive.
"""

from __future__ import annotations

import os
from multiprocessing import shared_memory

import numpy as np
import pytest

import cellstream.shm as shm

SHM_DIR = "/dev/shm"

pytestmark = pytest.mark.skipif(not os.path.isdir(SHM_DIR), reason="no /dev/shm on this platform")


def _inject(monkeypatch, exc_type, fail_on, break_close=None, close_exc=BufferError):
    """Make the fail_on-th SharedMemory(create=True) raise exc_type.

    Returns the live list of segments the fake actually created. Pass fail_on larger than 3 to
    let every allocation succeed. `break_close`, when given a list, shadows close() on every
    created instance with a raiser that appends to that list first -- so a test can assert
    close() was really ATTEMPTED, not merely that the name went away.
    """
    real = shared_memory.SharedMemory
    created = []

    def fake(*a, **kw):
        fake.n += 1
        if fake.n == fail_on:
            raise exc_type("INJECTED")
        seg = real(*a, **kw)
        if break_close is not None:

            def _boom(_name=seg.name):
                break_close.append(_name)
                raise close_exc("INJECTED-close")

            seg.close = _boom  # instance attribute shadows the bound method
        created.append(seg)
        return seg

    fake.n = 0
    monkeypatch.setattr(shared_memory, "SharedMemory", fake)
    return created


def _release(seg):
    """Unconditional cleanup for one segment, ignoring any shadow a test installed."""
    seg.__dict__.pop("close", None)  # so __del__ cannot re-raise the shadow either
    seg.__dict__.pop("unlink", None)
    try:
        seg.close()
    except Exception:
        pass
    try:
        seg.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _cleanup(created):
    """Never orphan anything ourselves, whatever the assertions did."""
    for seg in created:
        _release(seg)


def _alloc():
    return shm.allocate_shm_buffers(data_nbytes=4096, indices_nbytes=4096, indptr_nbytes=512)


def _leaked(created):
    return [s.name for s in created if os.path.exists(f"{SHM_DIR}/{s.name}")]


class _Interrupted(BaseException):
    """A BaseException that is NOT an Exception -- i.e. the KeyboardInterrupt/SystemExit class.

    Custom rather than KeyboardInterrupt itself so nothing downstream can treat it specially.
    Every constructor-failure test below is parametrized over an Exception AND this, so a
    mutant narrowing any `except BaseException` back to `except Exception` cannot survive.
    (codex, checkpoint 2.)
    """


def _raiser_for(exc):
    def _raise(**kw):
        raise exc("INJECTED")

    return _raise


def test_detector_can_see_a_live_segment():
    """POSITIVE CONTROL. Without it, every '_leaked() == []' below is indistinguishable from a
    probe that can never fire -- the exact failure mode that let three vacuous oracles through
    on PR #290."""
    seg = shared_memory.SharedMemory(create=True, size=64)
    path = f"{SHM_DIR}/{seg.name}"
    try:
        assert os.path.exists(path), "detector is blind: a live segment is not visible"
    finally:
        seg.close()
        seg.unlink()
    assert not os.path.exists(path), "detector is stuck: an unlinked segment still looks live"


@pytest.mark.parametrize("fail_on", [2, 3])
@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_baseexception_mid_allocation_orphans_nothing(monkeypatch, exc_type, fail_on):
    """The defect: `except Exception` let KeyboardInterrupt/SystemExit through, so the segments
    already created were never unlinked and survived the process."""
    created = _inject(monkeypatch, exc_type, fail_on)
    try:
        with pytest.raises(exc_type):
            _alloc()
        assert len(created) == fail_on - 1, (
            f"control: injection landed wrong -- created {len(created)}, wanted {fail_on - 1}"
        )
        assert not _leaked(created), f"orphaned /dev/shm segments: {_leaked(created)}"
    finally:
        _cleanup(created)


@pytest.mark.parametrize("fail_on", [2, 3])
@pytest.mark.parametrize("exc_type", [RuntimeError, MemoryError])
def test_ordinary_exception_still_orphans_nothing(monkeypatch, exc_type, fail_on):
    """NEGATIVE CONTROL: these are Exception subclasses the old arms already handled. They must
    stay clean, i.e. this test passes on the UNFIXED tree too -- it guards the behaviour being
    preserved, not the behaviour being added."""
    created = _inject(monkeypatch, exc_type, fail_on)
    try:
        with pytest.raises(exc_type):
            _alloc()
        assert len(created) == fail_on - 1, (
            f"control: injection landed wrong -- created {len(created)}, wanted {fail_on - 1}"
        )
        assert not _leaked(created), f"orphaned /dev/shm segments: {_leaked(created)}"
    finally:
        _cleanup(created)


@pytest.mark.parametrize("exc_type", [RuntimeError, _Interrupted])
def test_bundle_construction_failure_orphans_nothing(monkeypatch, exc_type):
    """`return ShmBundle(...)` used to sit OUTSIDE the protected region, so a failure building
    the bundle stranded all THREE segments with no owner at all."""
    created = _inject(monkeypatch, RuntimeError, fail_on=99)  # never fires; all three succeed
    monkeypatch.setattr(shm, "ShmBundle", _raiser_for(exc_type))
    try:
        with pytest.raises(exc_type, match="INJECTED"):
            _alloc()
        assert len(created) == 3, f"control: expected 3 segments, got {len(created)}"
        assert not _leaked(created), f"orphaned /dev/shm segments: {_leaked(created)}"
    finally:
        _cleanup(created)


@pytest.mark.parametrize("exc_type", [RuntimeError, _Interrupted])
def test_dense_bundle_construction_failure_orphans_nothing(monkeypatch, exc_type):
    """Same shape in allocate_dense_shm_buffer: the segment is created, then DenseShmBundle is
    constructed with no protection at all."""
    created = _inject(monkeypatch, RuntimeError, fail_on=99)
    monkeypatch.setattr(shm, "DenseShmBundle", _raiser_for(exc_type))
    try:
        with pytest.raises(exc_type, match="INJECTED"):
            shm.allocate_dense_shm_buffer(nbytes=4096)
        assert len(created) == 1, f"control: expected 1 segment, got {len(created)}"
        assert not _leaked(created), f"orphaned /dev/shm segments: {_leaked(created)}"
    finally:
        _cleanup(created)


@pytest.mark.parametrize("exc_type", [RuntimeError, _Interrupted])
def test_mmap_bundle_construction_failure_removes_the_spill_dir(monkeypatch, tmp_path, exc_type):
    """allocate_mmap_buffers has an `except BaseException: rmtree` arm, but built MmapBundle
    OUTSIDE it -- so a failure there left the spill dir and its three .bin files behind.

    The successful call up front is the POSITIVE CONTROL: it proves an allocation really does
    leave exactly one subdir here, so 'the directory is empty' is a signal rather than the
    default state."""
    kw = dict(
        spill_dir=tmp_path,
        total_nnz=8,
        n_indptr=3,
        data_dtype=np.dtype("float32"),
        indices_dtype=np.dtype("int32"),
    )
    ok = shm.allocate_mmap_buffers(**kw)
    assert len(list(tmp_path.iterdir())) == 1, "control: a successful allocation leaves a subdir"
    ok.close_and_unlink()
    assert list(tmp_path.iterdir()) == [], "control: close_and_unlink did not clear the spill dir"

    monkeypatch.setattr(shm, "MmapBundle", _raiser_for(exc_type))
    with pytest.raises(exc_type, match="INJECTED"):
        shm.allocate_mmap_buffers(**kw)
    assert list(tmp_path.iterdir()) == [], (
        f"the spill dir survived a bundle-construction failure: {list(tmp_path.iterdir())}"
    )


def test_failing_close_does_not_skip_unlink(monkeypatch):
    """The cleanup calls used to be chained bare (`data.close(); data.unlink()`), so a
    BufferError out of close() skipped the unlink and stranded the name anyway.

    The close_calls assertion is what stops this passing for the wrong reason: an
    implementation that never calls close() at all would satisfy every other assertion here."""
    close_calls = []
    created = _inject(monkeypatch, RuntimeError, fail_on=2, break_close=close_calls)
    try:
        with pytest.raises(RuntimeError, match="INJECTED"):
            _alloc()
        assert len(created) == 1, f"control: expected 1 segment, got {len(created)}"
        assert len(close_calls) == 1, (
            f"cleanup must ATTEMPT close() exactly once (got {len(close_calls)}); an "
            "implementation that only unlinks would otherwise pass this test"
        )
        assert not _leaked(created), (
            f"a failing close() stranded the /dev/shm name: {_leaked(created)}"
        )
    finally:
        _cleanup(created)


def test_release_segment_unlinks_and_propagates_a_baseexception_from_close():
    """_release_segment's nested `finally` has to do two things at once.

    A KeyboardInterrupt out of close() must (a) still leave the name unlinked, and (b) still
    PROPAGATE -- a flat `except Exception` skips the unlink, and an `except BaseException`
    would swallow the interrupt. Only the nested form satisfies both."""
    seg = shared_memory.SharedMemory(create=True, size=64)
    path = f"{SHM_DIR}/{seg.name}"
    calls = []

    def _boom():
        calls.append(1)
        raise KeyboardInterrupt("INJECTED")

    seg.close = _boom
    try:
        assert os.path.exists(path), "control: the segment is not live to begin with"
        with pytest.raises(KeyboardInterrupt):
            shm._release_segment(seg)
        assert calls == [1], "control: close() was never attempted"
        assert not os.path.exists(path), "the interrupt skipped the unlink"
    finally:
        _release(seg)


def test_close_and_unlink_releases_every_segment_when_the_first_close_interrupts():
    """_release_segment is per-segment safe but CAN re-raise, and a bare
    `for s in segs: _release_segment(s)` would then abandon the rest.

    Shadow the FIRST segment's close() with a KeyboardInterrupt and require that all three
    names are gone anyway, with the interrupt still propagating. The single-segment test above
    cannot see this."""
    bundle = shm.allocate_shm_buffers(data_nbytes=4096, indices_nbytes=4096, indptr_nbytes=512)
    segs = [bundle.data, bundle.indices, bundle.indptr]
    names = [s.name for s in segs]
    calls = []

    def _boom():
        calls.append(1)
        raise KeyboardInterrupt("INJECTED")

    bundle.data.close = _boom  # the FIRST one close_and_unlink walks
    try:
        assert all(os.path.exists(f"{SHM_DIR}/{n}") for n in names), (
            f"control: not every segment is live to begin with ({names})"
        )
        with pytest.raises(KeyboardInterrupt):
            bundle.close_and_unlink()
        assert calls == [1], "control: the shadowed close() was never attempted"
        alive = [n for n in names if os.path.exists(f"{SHM_DIR}/{n}")]
        assert not alive, f"an interrupt on the first close() abandoned: {alive}"
    finally:
        for s in segs:
            _release(s)


def test_release_segments_reports_the_first_escaping_error():
    """_release_segments must keep the FIRST escaping error, not the last.

    Changing `if first is None: first = e` to an unconditional `first = e` survives every other
    test in this module, because they all inject a single escaping error. Two distinguishable
    ones pin the choice -- and the attempt ORDER and the no-abandonment property with it.
    (codex, checkpoint 2.)"""
    bundle = shm.allocate_shm_buffers(data_nbytes=4096, indices_nbytes=4096, indptr_nbytes=512)
    segs = [bundle.data, bundle.indices, bundle.indptr]
    names = [s.name for s in segs]
    calls = []

    def _mk(tag):
        def _boom():
            calls.append(tag)
            raise _Interrupted(tag)

        return _boom

    bundle.data.close = _mk("FIRST")
    bundle.indices.close = _mk("SECOND")
    try:
        with pytest.raises(_Interrupted) as excinfo:
            bundle.close_and_unlink()
        assert calls == ["FIRST", "SECOND"], f"control: closes attempted in order {calls}"
        assert str(excinfo.value) == "FIRST", (
            f"the LAST escaping error won, not the first ({excinfo.value})"
        )
        alive = [n for n in names if os.path.exists(f"{SHM_DIR}/{n}")]
        assert not alive, f"segments abandoned despite two escaping errors: {alive}"
    finally:
        for s in segs:
            _release(s)


def test_release_segment_propagates_a_baseexception_from_unlink():
    """The unlink() arm must stay narrow: FileNotFoundError/OSError only.

    Every other helper test injects its escaping error from close(), so widening unlink()'s
    guard to `except BaseException` passes all of them while SWALLOWING an interrupt -- exactly
    what _release_segment's docstring says it must not do. (codex, checkpoint 2 round 2.)"""
    seg = shared_memory.SharedMemory(create=True, size=64)
    calls = []

    def _boom():
        calls.append(1)
        raise _Interrupted("INJECTED-UNLINK")

    seg.unlink = _boom
    try:
        with pytest.raises(_Interrupted, match="INJECTED-UNLINK"):
            shm._release_segment(seg)
        assert calls == [1], "control: unlink() was never attempted"
    finally:
        _release(seg)
