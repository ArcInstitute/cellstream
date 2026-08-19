"""Best-effort registration with anndata's IO registry.

When `import cellstream` runs, this attempts to monkey-patch
`anndata.read_h5ad` so that calling it with a directory containing a
cellstream manifest produces cellstream's packed-only migration error. Packed
archive files dispatch into cellstream's reader; other files fall through to
the original anndata implementation.

To opt out, set the environment variable `CELLSTREAM_NO_MONKEY_PATCH=1`
before importing cellstream.

To unregister manually:
    import anndata as ad
    if getattr(ad.read_h5ad, "_cellstream_wrapped", False):
        ad.read_h5ad = ad.read_h5ad._cellstream_original
"""

from __future__ import annotations

import functools
import os
from pathlib import Path


def register() -> bool:
    """Best-effort registration of `ad.read_h5ad(dir_or_file)` dispatch.

    Returns True if hooks are in place after this call (whether installed
    now or previously). Returns False on any failure or if opted out.
    """
    if os.environ.get("CELLSTREAM_NO_MONKEY_PATCH"):
        return False
    try:
        import anndata as ad
    except Exception:
        return False

    if getattr(ad.read_h5ad, "_cellstream_wrapped", False):
        return True

    _original = ad.read_h5ad

    @functools.wraps(_original)
    def read_h5ad_with_cellstream(path, *args, **kwargs):
        # Retain recognition of a legacy directory solely so cellstream can produce
        # its packed-only migration error. Packed archives route to cellstream too.
        if isinstance(path, (str, Path)):
            from cellstream.packed.reader import is_packed_file

            p = Path(path)
            if (p.is_dir() and (p / "manifest.json").exists()) or is_packed_file(p):
                from cellstream.read import read_h5ad as _cellstream_read_h5ad

                # cellstream's reader is keyword-only; anndata's positional `backed=`
                # (and any other positional) has no cellstream equivalent. Reject it
                # rather than silently dropping it and returning an in-memory read
                # (ultrareview L7).
                if args:
                    raise TypeError(
                        "anndata.read_h5ad on a cellstream archive accepts keyword "
                        f"arguments only; got positional {args!r} (anndata's "
                        "positional `backed=` is not supported for cellstream archives)."
                    )
                # Forward every knob the sharded reader understands, INCLUDING the
                # materialization knobs (container/data_dtype/index_dtype/
                # allow_lossy/output_backing) — dropping them silently returned a
                # CSR even when the caller asked for container='dense' (L7).
                # Unknown kwargs are still silently dropped (documented behavior).
                cast_back = kwargs.pop("cast_back", True)
                # Default None so the monkey-patched anndata.read_h5ad(path) route resolves
                # n_workers exactly the way a direct cellstream.read_h5ad call does (#51) --
                # forward the sentinel rather than substituting a number here. The
                # "None -> sched_getaffinity for v1" half of this note went with the v1
                # reader in #154 part A; read_h5ad now owns the whole resolution.
                n_workers = kwargs.pop("n_workers", None)
                # Forward the out-of-shm knobs too (#59) — otherwise a hook
                # caller asking for backing="mmap" would silently get the
                # default /dev/shm path, the very bypass this hook must avoid.
                spill_dir = kwargs.pop("spill_dir", None)
                backing = kwargs.pop("backing", "shm")
                blosc_nthreads = kwargs.pop("blosc_nthreads", 4)
                return _cellstream_read_h5ad(
                    p,
                    cast_back=cast_back,
                    n_workers=n_workers,
                    spill_dir=spill_dir,
                    backing=backing,
                    blosc_nthreads=blosc_nthreads,
                    container=kwargs.pop("container", "csr"),
                    data_dtype=kwargs.pop("data_dtype", None),
                    index_dtype=kwargs.pop("index_dtype", None),
                    allow_lossy=kwargs.pop("allow_lossy", False),
                    output_backing=kwargs.pop("output_backing", "heap"),
                )
        # Fall through to the original implementation for everything else.
        return _original(path, *args, **kwargs)

    read_h5ad_with_cellstream._cellstream_wrapped = True
    read_h5ad_with_cellstream._cellstream_original = _original
    try:
        ad.read_h5ad = read_h5ad_with_cellstream
        return True
    except Exception:
        return False
