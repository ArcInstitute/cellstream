"""Smoke tests: package imports and reports a version."""


def test_import():
    import cellstream  # noqa: F401


def test_has_version():
    import cellstream

    assert isinstance(cellstream.__version__, str)
    assert len(cellstream.__version__) > 0
