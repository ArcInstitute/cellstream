"""Index dtype policy: on-disk indices are always uint32 (no scan); the recorded
MIN index dtype is derived scan-free from n_vars (uint16 if n_vars <= 65536 else
uint32). (Supersedes the #128 uint16-vs-uint32 on-disk index scan, now removed.)"""

import numpy as np

from cellstream import narrow


def test_on_disk_index_always_uint32():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([0, 5, 12], dtype=np.int32)
    assert narrow.decide_on_disk_dtypes(data, indices, n_vars=18533)[1] == np.dtype(np.uint32)
    assert narrow.decide_on_disk_dtypes(data, indices, n_vars=100000)[1] == np.dtype(np.uint32)


def test_min_index_dtype_derived_from_n_vars_no_scan():
    data = np.array([1.0, 2.0], dtype=np.float32)
    # n_vars <= 65536 -> min uint16 (no index scan).
    min_i_small = narrow.decide_on_disk_dtypes(
        data, np.array([0, 5], dtype=np.int64), n_vars=18533
    )[3]
    assert min_i_small == np.dtype(np.uint16)
    # n_vars > 65536 -> min uint32, a scan-free upper bound: it is uint32 even though
    # the actual max index (65535) would fit uint16 (indices are no longer scanned).
    min_i_big = narrow.decide_on_disk_dtypes(
        data, np.array([0, 65535], dtype=np.int64), n_vars=100000
    )[3]
    assert min_i_big == np.dtype(np.uint32)
