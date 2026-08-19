"""row-gather kernel loader (#150).

Caches compiled cupy.RawKernel objects per output-dtype pair so the JIT cost is
paid once per pair per process. Uses Python-side string substitution for type
parameterization (NOT C++ templates) -- avoids name mangling at JIT time.

The production v2 GPU decode path (#40) is pure-cupy (cellstream.v2.codec_gpu); the
only kernel here is ``row_gather`` for the deferred on-GPU non-contiguous subset
scatter (#150). The old bit-plane ``bitshuffle_inverse`` kernel was removed (it
was wrong for the v2-byte-filter codec and never wired up).
"""

from __future__ import annotations

import functools
import importlib.resources

import numpy as np

_C_TYPE_FOR_NP = {
    np.uint16: ("uint16_t", 2),
    np.uint32: ("uint32_t", 4),
    np.int32: ("int32_t", 4),
    np.int64: ("int64_t", 8),
    np.float32: ("float", 4),
    np.float64: ("double", 8),
}


@functools.lru_cache(maxsize=8)
def row_gather_kernel(*, output_data_dtype, output_indices_dtype):
    """Compile (and cache) the row-gather kernel for a given output dtype pair.

    The kernel gathers selected rows from a per-shard intermediate buffer
    (always uint16) into the typed output buffers. Parameterized by string
    substitution (OUTPUT_DATA_T, OUTPUT_INDICES_T) to avoid C++ name mangling
    at JIT time. Cached by dtype pair so compile cost is paid once per process.

    v0.2 note: the kernel is compiled here and ready for v0.3 to wire up the
    full on-GPU non-contiguous subset path. In v0.2 the non-contig path falls
    back to CPU read + GPU upload.
    """
    import cupy as cp

    src = importlib.resources.files("cellstream.v2.kernels").joinpath("row_gather.cu").read_text()
    out_data_ct, _ = _C_TYPE_FOR_NP[output_data_dtype]
    out_idx_ct, _ = _C_TYPE_FOR_NP[output_indices_dtype]
    parameterized = src.replace("OUTPUT_DATA_T", out_data_ct).replace(
        "OUTPUT_INDICES_T", out_idx_ct
    )
    return cp.RawKernel(parameterized, "row_gather", options=("-std=c++14",))
