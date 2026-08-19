"""Optional MuData <-> native (primary AnnData + modalities dict) adapter.

`mudata` is an OPTIONAL dependency (`pip install cellstream[mudata]`); it is
imported lazily here and NOWHERE else, so cellstream has no hard dependency on it.
"""

from __future__ import annotations

import anndata as ad
import pandas as pd
import scipy.sparse as sp

from cellstream.matrices import real_layer_names


def _require_mudata():
    try:
        import mudata  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "MuData support requires the optional 'mudata' package. "
            "Install it with: pip install 'cellstream[mudata]'"
        ) from e
    return mudata


def mudata_to_native(mdata, primary_modality=None):
    """Convert a MuData into (primary_adata, modalities, primary_meta, primary_name).

    - The primary modality (default: first in ``mdata.mod``; else the named key)
      becomes ``x`` — its X/var/layers/raw ride the primary path; the shared
      header gets MuData's GLOBAL obs/obsm/uns + GLOBAL obsp/varp (e.g. the WNN
      joint graph); ``primary_meta`` carries the primary modality's OWN full obs
      + obsm/varm/uns + its own obsp/varp for the ``meta/x.h5ad`` sidecar.
    - Every other modality is returned as-is in ``modalities``.
    - A genuine non-empty global ``varm`` is rejected (deferred).
    """
    _require_mudata()
    # MuData.update() (run on construction and on most .mod/.obs/.var mutations)
    # auto-populates BOTH .obsm and .varm with a per-modality boolean membership
    # mask keyed by modality name (mudata._core.mudata.MuData._update_attr(_legacy):
    # `attrm[mod] = mapping > 0`) — pure structural bookkeeping, not user data;
    # build_mudata()'s own `MuData(dict(mod))` re-derives the identical masks on
    # read. `len(mdata.varm) > 0` is therefore true for EVERY multi-modality
    # MuData regardless of user input, so only a varm key that is NOT a modality
    # name is a genuine global annotation (it lives on the concatenated
    # global-var axis, which cellstream has no per-family analogue for) and gets
    # rejected here.
    #
    # LIMITATION (T8b): this guard is by KEY NAME, not content. A genuine global
    # varm annotation keyed with a modality name (e.g. a user setting
    # `mdata.varm["rna"]` themselves rather than the auto-mask) is
    # indistinguishable from mudata's auto-created membership masks and would pass
    # silently. Content-based detection is not feasible. Callers must not store a
    # global varm annotation under a modality-name key.
    _extra_varm = [k for k in (getattr(mdata, "varm", None) or {}) if k not in mdata.mod.keys()]
    if _extra_varm:
        raise ValueError(
            f"MuData global .varm key(s) {_extra_varm} are not supported (deferred); "
            f"they live on the concatenated global-var axis. Clear them before writing."
        )
    mod_keys = list(mdata.mod.keys())
    if not mod_keys:
        raise ValueError("MuData has no modalities to write.")
    if len(mod_keys) < 2:
        raise ValueError(
            f"MuData has a single modality {mod_keys!r}; the multi-matrix format needs "
            f">=2 modalities to store the global-vs-per-modality metadata split. Write its "
            f"one modality as a plain AnnData instead: write_sharded(mdata.mod['{mod_keys[0]}'], ...)."
        )
    primary_name = primary_modality if primary_modality is not None else mod_keys[0]
    if primary_name not in mdata.mod:
        raise ValueError(
            f"primary_modality {primary_name!r} is not a modality of this MuData (have {mod_keys})."
        )
    pmod = mdata.mod[primary_name]
    # The primary modality's cells must align to the GLOBAL obs: we pair pmod.X
    # with mdata.obs below. AnnData catches an n_obs mismatch but silently pairs
    # positionally on an ORDER mismatch — so check obs_names explicitly (this is
    # the primary's counterpart of decompose()'s per-modality alignment check).
    if pmod.n_obs != mdata.n_obs or not pmod.obs_names.equals(mdata.obs_names):
        raise ValueError(
            f"primary modality {primary_name!r} is not aligned to the MuData global "
            f"obs: the same cells in the same order are required (pairing pmod.X with "
            f"the global obs would otherwise mislabel cells)."
        )
    # primary_adata: primary modality's matrices + GLOBAL cell-axis metadata
    # (obs/obsm/uns + the MuData GLOBAL obsp/varp, e.g. the WNN joint graph —
    # these ride the shared header). The primary modality's OWN obs/obsp/varp
    # are stored separately in primary_meta below (Bucket 3).
    primary_adata = ad.AnnData(
        X=pmod.X,
        obs=mdata.obs.copy(),
        var=pmod.var.copy(),
        uns=dict(mdata.uns),
        obsm=dict(mdata.obsm),
        obsp=dict(getattr(mdata, "obsp", {}) or {}),
        varp=dict(getattr(mdata, "varp", {}) or {}),
    )
    for ln in real_layer_names(pmod):
        primary_adata.layers[ln] = pmod.layers[ln]
    if pmod.raw is not None:
        # AnnData.raw's setter only accepts an AnnData (or None) — NOT a Raw
        # object (`primary_adata.raw = pmod.raw` raises "Can only init raw
        # attribute with an AnnData object"). Raw.to_adata() converts it.
        primary_adata.raw = pmod.raw.to_adata()
    # primary_meta = the primary modality's OWN metadata carrier: its full obs +
    # obsm/varm/uns + its own obsp/varp (Bucket 3). NOT `pmod` itself, because
    # `pmod` may carry a `.raw` (already captured above as the `raw` family via
    # `primary_adata.raw`) and write_modality_meta's backstop rejects a carrier
    # with a non-empty `.raw`; this clean carrier omits `.raw` while preserving
    # everything else on the cell/feature axes.
    primary_meta = ad.AnnData(
        X=sp.csr_matrix((pmod.n_obs, pmod.n_vars), dtype="float32"),
        obs=pmod.obs.copy(),
        var=pd.DataFrame(index=pmod.var_names),
        obsm=dict(pmod.obsm),
        varm=dict(pmod.varm),
        uns=dict(pmod.uns),
        obsp=dict(pmod.obsp),
        varp=dict(pmod.varp),
    )
    modalities = {k: mdata.mod[k] for k in mod_keys if k != primary_name}
    return primary_adata, modalities, primary_meta, primary_name


def build_mudata(mod, obs, obsm, uns, obsp=None, varp=None):
    """Assemble a MuData from a modality dict + global cell-axis metadata.

    ``obsp``/``varp`` are the MuData GLOBAL pairwise graphs (e.g. the WNN joint
    graph); set after construction so they survive ``MuData.update()``'s obsm/varm
    membership-mask bookkeeping."""
    mudata = _require_mudata()
    md = mudata.MuData(dict(mod))
    if obs is not None:
        md.obs = obs
    for k, v in (obsm or {}).items():
        md.obsm[k] = v
    for k, v in (uns or {}).items():
        md.uns[k] = v
    for k, v in (obsp or {}).items():
        md.obsp[k] = v
    for k, v in (varp or {}).items():
        md.varp[k] = v
    return md
