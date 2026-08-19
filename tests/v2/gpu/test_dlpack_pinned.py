"""GPU-gated (#160): pinned DLPack handoff is_pinned + async H2D correctness.

Skipped in CI (no GPU). Run on a GPU node with torch (cu12x) + cupy installed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cupy")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import cellstream  # noqa: E402
from tests.v2.test_prefetch_stream import _write_grouped  # noqa: E402


def test_pinned_dlpack_is_pinned_and_async_correct(tmp_path):
    out, _ = _write_grouped(tmp_path, [f"g{i}" for i in range(40)])
    arc = cellstream.ShardedArchive(out)
    seen = 0
    for gs in arc.iter_group_shards(prefetch=2, n_workers=2, pinned=True):
        x = gs.x()
        caps = gs.dlpack()
        t = torch.from_dlpack(caps.data)
        assert t.is_pinned(), "registered batch array should be pinned"
        d = t.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        np.testing.assert_array_equal(d.cpu().numpy(), x.data)
        seen += 1
    assert seen > 0
