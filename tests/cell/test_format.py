"""cell/format.py constant + dtype-helper tests."""

from __future__ import annotations

import numpy as np


def test_constants():
    from cellstream.cell import format as fmt

    assert fmt.SCHEMA_VERSION == 5
    assert fmt.LAYOUT_CELL == "cell"
    assert fmt.CODEC_PFORDELTA == "pfordelta"
    assert fmt.PYFASTPFOR_CODEC == "simdfastpfor256"
    assert fmt.CODEC_PFORDELTA in fmt.SUPPORTED_CODECS


def test_np_dtype():
    from cellstream.cell import format as fmt

    assert fmt.np_dtype("uint16") == np.dtype(np.uint16)
    assert fmt.np_dtype("uint32") == np.dtype(np.uint32)


def test_choose_index_dtype():
    from cellstream.cell import format as fmt

    assert fmt.choose_index_dtype(60) == "uint16"
    assert fmt.choose_index_dtype(65535) == "uint16"
    assert fmt.choose_index_dtype(65536) == "uint32"
    assert fmt.choose_index_dtype(200000) == "uint32"


def test_choose_value_dtype():
    from cellstream.cell import format as fmt

    assert fmt.choose_value_dtype(0) == "uint16"
    assert fmt.choose_value_dtype(65535) == "uint16"
    assert fmt.choose_value_dtype(65536) == "uint32"


def test_zstd_codec_constant_and_supported():
    from cellstream.cell import format as fmt

    assert fmt.CODEC_ZSTD == "zstd"
    assert fmt.SUPPORTED_CODECS == frozenset({fmt.CODEC_PFORDELTA, fmt.CODEC_ZSTD})
    # pfordelta schema unchanged; zstd is a new, separate version
    assert fmt.SCHEMA_VERSION == 5
    assert fmt.SCHEMA_VERSION_ZSTD == 6


def test_float_dtypes_registered():
    import numpy as np

    from cellstream.cell import format as fmt

    assert fmt.np_dtype("float32") == np.dtype(np.float32)
    assert fmt.np_dtype("float64") == np.dtype(np.float64)
