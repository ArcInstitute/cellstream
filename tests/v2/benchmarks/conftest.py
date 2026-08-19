"""Benchmarks: opt-in only — skipped unless CELLSTREAM_BENCH=1."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CELLSTREAM_BENCH") != "1":
        skip = pytest.mark.skip(reason="set CELLSTREAM_BENCH=1 to run benchmarks")
        for item in items:
            if "tests/v2/benchmarks" in str(item.fspath):
                item.add_marker(skip)
