# src/cellstream/matrices.py
"""Pure AnnData <-> ordered matrix-spec model for multi-matrix (v4) archives.

No I/O: decompose an AnnData (X, .layers, .raw) into an ordered list of
MatrixSpec, and reassemble the reverse. Matrix keys ARE role strings:
"x" (primary/anchor), "layer:<name>", "raw". The primary is always "x".
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, fields
from typing import Any

import anndata as ad

PRIMARY_KEY = "x"
RAW_KEY = "raw"
_LAYER_PREFIX = "layer:"
MODALITY_PREFIX = "modality:"


def real_layer_names(obj: ad.AnnData) -> list[str]:
    """``obj.layers`` keys excluding anndata 0.13's ``None`` alias for ``X``.

    anndata 0.13 exposes ``.X`` as ``layers[None]``; ``.layers.keys()`` therefore
    yields a spurious ``None`` for an X-only object (anndata <=0.12 yields ``[]``).
    Every place cellstream enumerates layers must skip it, else an X-only AnnData is
    mis-detected as multi-matrix. A no-op on anndata <=0.12.
    """
    return [k for k in obj.layers.keys() if k is not None]


def matrix_dir(key: str) -> str:
    """Filesystem-safe family directory for a matrix key (write-time only;
    readers use the manifest's stored paths).

    ``x``/``raw`` unchanged (preserves Plan-1 byte-identity); ``layer:<n>``
    (primary) -> ``layer_<safe n>``; ``modality:<m>`` -> ``modality_<safe m>``;
    ``modality:<m>:layer:<n>`` -> ``modality_<safe m>__layer_<safe n>``.
    Every char outside ``[0-9A-Za-z._-]`` becomes ``_``.
    """

    def _safe(s: str) -> str:
        return re.sub(r"[^0-9A-Za-z._-]", "_", s)

    if key == PRIMARY_KEY:
        return "x"
    if key == RAW_KEY:
        return "raw"
    if key.startswith(MODALITY_PREFIX):
        rest = key[len(MODALITY_PREFIX) :]
        marker = f":{_LAYER_PREFIX}"  # ":layer:"
        if marker in rest:
            mod, lyr = rest.rsplit(marker, 1)
            return f"modality_{_safe(mod)}__layer_{_safe(lyr)}"
        return f"modality_{_safe(rest)}"
    if key.startswith(_LAYER_PREFIX):
        return f"layer_{_safe(key[len(_LAYER_PREFIX) :])}"
    raise ValueError(f"unknown matrix key {key!r}")


def var_member_name(key: str) -> str:
    """Member/sidecar name for a family's var parquet: ``var/<dir>.parquet``."""
    return f"var/{matrix_dir(key)}.parquet"


def meta_member_name(key: str) -> str:
    """Member/sidecar name for a family's per-modality metadata: ``meta/<dir>.h5ad``."""
    return f"meta/{matrix_dir(key)}.h5ad"


@dataclass(frozen=True)
class MatrixSpec:
    key: str
    role: str
    owner: str | None
    n_vars: int
    X: Any = None
    var: Any = None

    def __eq__(self, other):
        """Compare field TUPLES -- what dataclasses generated before CPython 3.13.

        3.13 generates a PAIRWISE field comparison instead of building the two tuples, which loses
        `PyObject_RichCompareBool`'s identity shortcut AND its bool coercion. With `X` holding a
        sparse matrix (or `var` a DataFrame) the generated `__eq__` then RETURNS THE ELEMENTWISE
        MATRIX instead of `True`, and any `bool()` of that raises "The truth value of an array with
        more than one element is ambiguous". Measured with identical scipy/anndata/pandas:

            MatrixSpec("k", "x", None, 3, m, None) == MatrixSpec("k", "x", None, 3, m, None)
              3.11 / 3.12 -> True          3.13 -> a 3x3 bool sparse matrix

        Comparing the tuples restores byte-for-byte identical behaviour on every version, including
        the pre-existing one where two DIFFERENT arrays still raise -- that is not a change here.
        A class-defined `__eq__` wins over the generated one (`dataclasses._set_new_attribute` does
        not overwrite), and `__hash__` is still generated because `has_explicit_hash` stays False.

        ⚠️ `read.ShardCSRDLPack`, `grouping.ShardPlan` and `v2.writer._GroupedMeta` have array- or
        DataFrame-valued compared fields and the same latent problem; none has an observed failure
        or a test, so they are filed as #318 rather than fixed by a blanket `__eq__` change
        across four modules.
        """
        if other.__class__ is not self.__class__:
            return NotImplemented
        return _spec_fields(self) == _spec_fields(other)


_spec_fields = operator.attrgetter(*(f.name for f in fields(MatrixSpec)))


def decompose_anndata(adata: ad.AnnData) -> list[MatrixSpec]:
    """Ordered ``[x, *layers, raw?]`` specs. ``x``/``raw`` carry their var;
    ``layer:*`` carry ``var=None`` (they share the primary's var)."""
    specs = [
        MatrixSpec(
            key=PRIMARY_KEY,
            role="x",
            owner=None,
            n_vars=int(adata.n_vars),
            X=adata.X,
            var=adata.var,
        )
    ]
    for name in real_layer_names(adata):
        specs.append(
            MatrixSpec(
                key=f"{_LAYER_PREFIX}{name}",
                role=f"{_LAYER_PREFIX}{name}",
                owner=PRIMARY_KEY,
                n_vars=int(adata.n_vars),
                X=adata.layers[name],
                var=None,
            )
        )
    if adata.raw is not None:
        specs.append(
            MatrixSpec(
                key=RAW_KEY,
                role="raw",
                owner=None,
                n_vars=int(adata.raw.n_vars),
                X=adata.raw.X,
                var=adata.raw.var,
            )
        )
    # Fail loud if two matrix keys sanitise to the same on-disk directory
    # (e.g. layers "a/b" and "a_b" both -> "layer_a_b") — otherwise their shards
    # would silently overwrite each other during the write.
    seen: dict[str, str] = {}
    for s in specs:
        d = matrix_dir(s.key)
        if d in seen:
            raise ValueError(
                f"matrix keys {seen[d]!r} and {s.key!r} both map to the on-disk "
                f"directory {d!r}; rename a layer to avoid the collision."
            )
        seen[d] = s.key
    return specs


def reassemble_anndata(
    matrices, obs, var_by_key, uns, obsm, varm, obsp=None, varp=None
) -> ad.AnnData:
    """Rebuild an AnnData from matrix specs (each carrying its X) + shared axes.

    ``obsp``/``varp`` (pairwise graphs) are attached when provided (Bucket 3);
    they are already in the archive's on-disk partition order."""
    by_key = {s.key: s for s in matrices}
    if PRIMARY_KEY not in by_key:
        raise ValueError("cannot reassemble: primary 'x' matrix is missing")
    x = by_key[PRIMARY_KEY]
    out = ad.AnnData(
        X=x.X,
        obs=obs.copy(),
        var=var_by_key[PRIMARY_KEY],
        uns=dict(uns),
        obsm=dict(obsm),
        varm=dict(varm),
        obsp=dict(obsp or {}),
        varp=dict(varp or {}),
    )
    for s in matrices:
        if s.role.startswith(_LAYER_PREFIX):
            out.layers[s.role[len(_LAYER_PREFIX) :]] = s.X
    if RAW_KEY in by_key:
        raw = by_key[RAW_KEY]
        out.raw = ad.AnnData(X=raw.X, obs=obs.copy(), var=var_by_key[RAW_KEY])
    return out


def is_primary_family(role: str, owner: str | None) -> bool:
    """True if a family belongs to the primary AnnData group (the group
    ``to_anndata()`` returns): ``x``, ``raw``, or a layer owned by ``x``."""
    if role in ("x", "raw"):
        return True
    return role.startswith(_LAYER_PREFIX) and owner == PRIMARY_KEY


def _check_modality_aligned(name: str, madata: ad.AnnData, primary: ad.AnnData) -> None:
    if madata.n_obs != primary.n_obs or not madata.obs_names.equals(primary.obs_names):
        raise ValueError(
            f"modality {name!r} is not aligned to the primary matrix: its cells "
            f"(n_obs={madata.n_obs}) must match the primary's obs_names in the same "
            f"order (n_obs={primary.n_obs}). Unaligned modalities are out of scope."
        )
    # Per-modality obs columns ARE stored (Bucket 3) — in the modality's meta
    # sidecar — so there is no obs-column subset check. Alignment (same cells,
    # same order) is still required.


def decompose(
    primary: ad.AnnData, modalities: dict[str, ad.AnnData] | None = None
) -> list[MatrixSpec]:
    """Ordered specs for the whole multi-matrix source:
    ``[x, *x-layers, raw?, *(modality anchor + its layers) ...]``.

    Each modality is validated as aligned to ``primary`` (same obs_names/order);
    its own obs columns are stored in its meta sidecar (Bucket 3). Modality
    anchors carry their own var (owner None); modality layers carry ``var=None``
    and owner ``"modality:<name>"``. A full on-disk-dir collision check fails loud.
    """
    specs = decompose_anndata(primary)  # x, x-layers, raw (+ its own collision check)
    for name, madata in (modalities or {}).items():
        if f":{_LAYER_PREFIX}" in name:
            raise ValueError(
                f"modality name {name!r} may not contain ':{_LAYER_PREFIX}' (reserved marker "
                f"used to encode modality layers in on-disk directory names)."
            )
        _check_modality_aligned(name, madata, primary)
        mkey = f"{MODALITY_PREFIX}{name}"
        specs.append(
            MatrixSpec(
                key=mkey,
                role=mkey,
                owner=None,
                n_vars=int(madata.n_vars),
                X=madata.X,
                var=madata.var,
            )
        )
        for ln in real_layer_names(madata):
            specs.append(
                MatrixSpec(
                    key=f"{mkey}:{_LAYER_PREFIX}{ln}",
                    role=f"{_LAYER_PREFIX}{ln}",
                    owner=mkey,
                    n_vars=int(madata.n_vars),
                    X=madata.layers[ln],
                    var=None,
                )
            )
    seen: dict[str, str] = {}
    for s in specs:
        d = matrix_dir(s.key)
        if d in seen:
            raise ValueError(
                f"matrix keys {seen[d]!r} and {s.key!r} both map to the on-disk "
                f"directory {d!r}; rename a layer/modality to avoid the collision."
            )
        seen[d] = s.key
    return specs


def reassemble_modality_anndata(
    anchor_X,
    var,
    obs,
    layer_items: list[tuple[str, Any]],
    obsm: dict,
    varm: dict,
    uns: dict,
    obsp=None,
    varp=None,
) -> ad.AnnData:
    """Build ONE modality's AnnData from its anchor X + var + its obs + its
    layers + its own obsm/varm/uns + its own obsp/varp (used by ``to_mudata()``).
    ``obs`` is the modality's own obs (Bucket 3); ``obsp``/``varp`` are attached
    when provided (already in the archive's on-disk partition order)."""
    out = ad.AnnData(
        X=anchor_X,
        obs=obs.copy(),
        var=var,
        uns=dict(uns),
        obsm=dict(obsm),
        varm=dict(varm),
        obsp=dict(obsp or {}),
        varp=dict(varp or {}),
    )
    for ln, lx in layer_items:
        out.layers[ln] = lx
    return out
