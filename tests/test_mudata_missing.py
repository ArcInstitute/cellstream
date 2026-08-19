# tests/test_mudata_missing.py
from __future__ import annotations

import builtins

import pytest


def test_to_native_import_error(monkeypatch):
    from cellstream import mudata_adapter

    real_import = builtins.__import__

    def _no_mudata(name, *a, **k):
        if name == "mudata" or name.startswith("mudata."):
            raise ImportError("no mudata")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_mudata)
    with pytest.raises(ImportError, match="cellstream\\[mudata\\]"):
        mudata_adapter.build_mudata({}, obs=None, obsm={}, uns={})
