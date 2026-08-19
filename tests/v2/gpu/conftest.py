"""GPU tests: skipped unless cupy import + visible CUDA GPU present."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA devices visible")
    except Exception as e:
        skip = pytest.mark.skip(reason=f"no GPU: {e}")
        for item in items:
            if "tests/v2/gpu" in str(item.fspath) and "test_gpu_unsupported" not in str(
                item.fspath
            ):
                item.add_marker(skip)
