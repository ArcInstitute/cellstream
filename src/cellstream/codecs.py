"""Blosc codec presets + numeric-dataset recompression.

Two presets ship: 'blosc-zstd-1-u16u16' (pure Blosc1-bitshuffle) and
'blosc-zstd-1-u16u16-datashuf' (Blosc2-SHUFFLE on the CSR data array,
bitshuffle elsewhere) — the latter is the ``write_h5ad_narrow`` default. It was the schema-v1
*archive* write default too, until #154 removed that writer; the shard/packed writers resolve
``codec=None`` to the v2 byte-filter codec instead. Both presets are still accepted there and
map to bitshuffle.

The recompression helper exists because anndata.write_h5ad(compression=...)
SIGFPEs when it propagates the filter to vlen string datasets. We write
uncompressed first, then streaming-copy with Blosc applied only to
numeric chunked datasets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np

# 1 MB elements per chunk (HANDOFF.md, validated).
HDF5_CHUNK_SIZE = 1_048_576

_BITSHUFFLE = hdf5plugin.Blosc.BITSHUFFLE


_PRESETS: dict[str, dict[str, Any]] = {
    "blosc-zstd-1-u16u16": {
        "filter_kwargs": hdf5plugin.Blosc(cname="zstd", clevel=1, shuffle=_BITSHUFFLE),
        "data_dtype": "uint16",
        "indices_dtype": "uint16",
    },
    # Opt-in: Blosc2-SHUFFLE on the CSR `data` array only (+18% on real
    # counts; decode faster). indices/indptr keep Blosc1-bitshuffle (SHUFFLE
    # regresses indices). See docs/superpowers/specs/2026-06-01-v1-data-shuffle-codec-design.md
    "blosc-zstd-1-u16u16-datashuf": {
        "filter_kwargs": hdf5plugin.Blosc(cname="zstd", clevel=1, shuffle=_BITSHUFFLE),
        "data_filter_kwargs": hdf5plugin.Blosc2(
            cname="zstd", clevel=1, filters=hdf5plugin.Blosc2.SHUFFLE
        ),
        "data_dtype": "uint16",
        "indices_dtype": "uint16",
    },
}


def list_codecs() -> list[str]:
    return list(_PRESETS)


def get_filter_kwargs(name: str):
    return _PRESETS[name]["filter_kwargs"]


def get_data_filter_kwargs(name: str):
    """Per-array override filter for the CSR ``data`` dataset, or None if the
    preset applies its single ``filter_kwargs`` uniformly."""
    return _PRESETS[name].get("data_filter_kwargs")


def get_data_dtype(name: str) -> str:
    return _PRESETS[name]["data_dtype"]


def get_indices_dtype(name: str) -> str:
    return _PRESETS[name]["indices_dtype"]


def _choose_filter(
    obj: h5py.Dataset,
    name: str,
    filter_kwargs: Any,
    data_filter_kwargs: Any | None,
) -> Any:
    """Blosc2-SHUFFLE override only for the genuine CSR/CSC ``data`` member."""
    if data_filter_kwargs is not None and (name == "data" or name.endswith("/data")):
        enc = obj.parent.attrs.get("encoding-type")
        if hasattr(enc, "decode"):
            enc = enc.decode()
        if isinstance(enc, str) and enc in ("csr_matrix", "csc_matrix"):
            return data_filter_kwargs
    return filter_kwargs


def _recompress_numeric_datasets(
    path: str | Path,
    filter_kwargs,
    *,
    data_filter_kwargs=None,
    hdf5_chunk_size: int = HDF5_CHUNK_SIZE,
) -> None:
    """Rewrite ``path`` so numeric chunked datasets use the given Blosc filter.

    HDF5 doesn't reclaim space on in-place delete+recreate, so we stream
    everything into a temp file then atomically rename. vlen / object dtype
    datasets are copied verbatim — Blosc can't handle them.
    """
    src_path = str(path)
    tmp_path = src_path + ".tmp"
    MIN_BYTES = 64 * 1024

    def pick_chunks(shape, itemsize):
        n_per_chunk = hdf5_chunk_size
        if len(shape) == 1:
            return (min(shape[0], n_per_chunk),)
        leading = max(1, n_per_chunk // max(1, int(np.prod(shape[1:]))))
        return (min(shape[0], leading), *shape[1:])

    try:
        with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            def copy_item(name, obj):
                if isinstance(obj, h5py.Group):
                    grp = dst.require_group(name)
                    for k, v in obj.attrs.items():
                        grp.attrs[k] = v
                    return
                if not isinstance(obj, h5py.Dataset):
                    return
                if obj.dtype.kind in ("O", "S", "U"):
                    src.copy(name, dst, name=name)
                    return
                if obj.size == 0:
                    # h5py rejects chunks=(0,); copy zero-size numeric datasets verbatim.
                    src.copy(name, dst, name=name)
                    return
                n_bytes = obj.size * obj.dtype.itemsize
                use_blosc = obj.dtype.kind in ("f", "i", "u", "b") and (
                    obj.chunks is not None or n_bytes >= MIN_BYTES
                )
                if use_blosc:
                    chunks = pick_chunks(obj.shape, obj.dtype.itemsize)
                    chosen = _choose_filter(obj, name, filter_kwargs, data_filter_kwargs)
                    new_ds = dst.create_dataset(
                        name,
                        shape=obj.shape,
                        dtype=obj.dtype,
                        chunks=chunks,
                        **dict(chosen),
                    )
                    step = chunks[0]
                    for i in range(0, obj.shape[0], step):
                        new_ds[i : i + step] = obj[i : i + step]
                else:
                    src.copy(name, dst, name=name)
                    return
                for k, v in obj.attrs.items():
                    new_ds.attrs[k] = v

            src.visititems(copy_item)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    os.replace(tmp_path, src_path)
