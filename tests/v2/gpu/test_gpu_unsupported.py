"""v2 GPU reads must fail LOUD and actionably (not a cryptic error). #40.

Byte-filter GPU decode now works, so these cover the two genuine fail-loud
contracts, both runnable WITHOUT a GPU (this module is exempt from the
tests/v2/gpu no-GPU skip in conftest.py):
  1. an unsupported codec (bitshuffle) -> NotImplementedError, before any cupy import;
  2. missing GPU deps (no cupy) -> a clear ImportError with a cellstream[gpu] hint.
"""

import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream import write_sharded
from cellstream.errors import PackedOnlyError
from cellstream.read import ShardedArchive
from cellstream.v2 import reader_gpu
from cellstream.v2.reader_gpu import read_v2_to_device


def _cupy_available():
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def _write(tmp_path, codec):
    X = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32))
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=["c0", "c1", "c2"]))
    out = tmp_path / "arch"
    write_sharded(a, str(out), format="v2", codec=codec, overwrite=True)
    return ShardedArchive(str(out))


def test_check_codec_rejects_non_byte_filter():
    """_check_codec fails fast on any non-byte-filter shuffle (import-free)."""
    fake = types.SimpleNamespace(shuffle="bitshuffle")
    with pytest.raises(NotImplementedError, match="v2-byte-filter only"):
        reader_gpu._check_codec(fake)


def test_archive_locator_rejects_a_directory(tmp_path):
    directory = tmp_path / "archive"
    directory.mkdir()
    with pytest.raises(PackedOnlyError):
        reader_gpu._archive_locator(directory)


def test_unsupported_codec_archive_raises_before_import(tmp_path):
    """A bitshuffle v2 archive raises NotImplementedError (not ImportError), even
    on a host without cupy — the codec gate runs before _import_gpu (#40 scope)."""
    arch = _write(tmp_path, codec="bitshuffle")
    with pytest.raises(NotImplementedError, match="v2-byte-filter only"):
        read_v2_to_device(arch.path)
    with pytest.raises(NotImplementedError, match="v2-byte-filter only"):
        arch[:].x_cupy()


@pytest.mark.skipif(_cupy_available(), reason="runs only where GPU deps are absent")
def test_byte_filter_gpu_read_without_cupy_raises_importerror(tmp_path):
    """A supported (byte-filter) archive on a host WITHOUT cupy raises a clear
    ImportError with the cellstream[gpu] install hint (not a cryptic nvcomp error)."""
    arch = _write(tmp_path, codec="v2-byte-filter")
    with pytest.raises(ImportError, match=r"cellstream\[gpu\]"):
        read_v2_to_device(arch.path)
