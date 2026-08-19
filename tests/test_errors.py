"""Errors form a single hierarchy under CellstreamError."""


def test_error_hierarchy():
    from cellstream.errors import (
        ArchiveExistsError,
        ArchiveLockedError,
        CellstreamError,
        IncompatibleSchemaError,
        LossyCastError,
        PackedOnlyError,
    )

    for cls in (
        ArchiveLockedError,
        IncompatibleSchemaError,
        LossyCastError,
        ArchiveExistsError,
        # PackedOnlyError's only hierarchy pin used to live in the deleted
        # test_guard_raises_when_enforced (#154 B3b); it is still public and still raised,
        # and #301 is about whether it stays exported, so it belongs here.
        PackedOnlyError,
    ):
        assert issubclass(cls, CellstreamError)
    assert issubclass(CellstreamError, Exception)


def test_errors_carry_messages():
    from cellstream.errors import LossyCastError

    e = LossyCastError("X.data lossy: max=72341")
    assert "72341" in str(e)


def test_errors_reexported_from_package():
    """Errors are accessible directly on the cellstream namespace."""
    import cellstream

    assert hasattr(cellstream, "CellstreamError")
    assert cellstream.CellstreamError is cellstream.errors.CellstreamError
    assert hasattr(cellstream, "LossyCastError")
    assert cellstream.LossyCastError is cellstream.errors.LossyCastError
    assert hasattr(cellstream, "PackedOnlyError")
    assert cellstream.PackedOnlyError is cellstream.errors.PackedOnlyError


def test_removed_error_classes_are_gone():
    """#154: these classes lost every raiser and were removed with them.

    IncompatibleAppendError came from append_to_archive and, through
    _require_v1_for_mutation, from update_obs_in_archive / update_var_in_archive;
    IncompatibleConversionError came from the v1->v2 converter. Catching either is
    now unreachable."""
    import cellstream
    import cellstream.errors

    for name in ("IncompatibleAppendError", "IncompatibleConversionError", "MissingShardError"):
        assert not hasattr(cellstream.errors, name), f"cellstream.errors.{name} should be gone"
        assert not hasattr(cellstream, name), f"cellstream.{name} should be gone"
        assert name not in cellstream.__all__, f"cellstream.__all__ still exports {name}"


def test_gpu_oom_error_is_cellstream_error():
    from cellstream.errors import CellstreamError, GpuOutOfMemoryError

    assert issubclass(GpuOutOfMemoryError, CellstreamError)


def test_new_error_classes_are_top_level_importable():
    import cellstream

    assert hasattr(cellstream, "GpuOutOfMemoryError")
    assert hasattr(cellstream.errors, "GpuOutOfMemoryError")
