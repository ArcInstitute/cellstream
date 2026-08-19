"""manifest.json: load, dump, schema check, and row-offset math.

Schema v1 fields:
  schema_version, n_shards, n_obs_total, n_vars, row_offsets,
  nnz_per_shard, total_nnz, codec, x_data_dtype_on_disk,
  x_indices_dtype_on_disk, writer_fingerprint, created_at.

v0.2 added schema_version=2 (SHX2 archives): adds `shard_filename_template`
and `chunk_size_elements` fields. See the v0.2 design spec for the full
v2 manifest schema. `load_manifest` accepts v1, v2, and v3; readers
dispatch on schema_version. Schema v3 adds group_by + reference_shard fields and a groups.json sidecar.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

from cellstream.errors import IncompatibleSchemaError

SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3, 4, 5, 6)

# Deprecated single-value alias: this is ALWAYS 1 and does NOT track the highest
# supported schema. An external check like `version <= SUPPORTED_SCHEMA_VERSION`
# would wrongly reject v2/v3 archives — use SUPPORTED_SCHEMA_VERSIONS (the tuple
# above) instead. Kept only so existing external importers don't break on import.
SUPPORTED_SCHEMA_VERSION = 1


def compute_row_offsets(n_obs: int, n_shards: int) -> list[int]:
    """Even-ish row partition; remainder distributed to first shards.

    Returns ``[0, …, n_obs]`` of length ``n_shards + 1``.
    """
    if n_shards <= 0:
        raise ValueError(f"n_shards must be > 0, got {n_shards}")
    if n_shards > n_obs:
        raise ValueError(f"n_shards={n_shards} > n_obs={n_obs}")
    base, extra = divmod(n_obs, n_shards)
    offsets = [0]
    for i in range(n_shards):
        offsets.append(offsets[-1] + base + (1 if i < extra else 0))
    return offsets


def shard_for_row(row_offsets: list[int], row: int) -> int:
    """Return the shard index that contains the global row ``row``."""
    # bisect_right finds insertion point; subtract 1 to get containing shard.
    return bisect.bisect_right(row_offsets, row) - 1


def shards_for_range(row_offsets: list[int], start: int, stop: int) -> list[int]:
    """Return shard indices intersecting [start, stop)."""
    if stop <= start:
        return []
    n_shards = len(row_offsets) - 1
    first = shard_for_row(row_offsets, start)
    last = shard_for_row(row_offsets, max(stop - 1, 0))
    return list(range(max(first, 0), min(last + 1, n_shards)))


def shard_filename(manifest: dict, shard_idx: int) -> str:
    """Resolve a shard's filename via the manifest's template.

    v1 manifests lack `shard_filename_template`; default to the implicit
    v0.1 convention `shard_{:04d}.h5ad`. v2 manifests carry the template
    explicitly.
    """
    template = manifest.get("shard_filename_template", "shard_{:04d}.h5ad")
    return template.format(shard_idx)


def dump_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Atomic write: tmp + os.replace."""
    import os

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, path)


def build_v4_manifest(
    *,
    n_obs_total: int,
    row_offsets: list[int],
    matrices: dict[str, dict],
    group_by: str | None = None,
    reference_shard: int | None = None,
    writer_fingerprint: dict,
    created_at: str,
) -> dict[str, Any]:
    """Assemble a schema-4 manifest: shared-partition block + per-matrix map.

    Keeps top-level ``n_shards``/``n_obs_total``/``row_offsets`` so the existing
    ``validate_manifest`` shape checks run unchanged; ``n_vars`` and
    ``shard_filename_template`` mirror the primary ``x`` for tool convenience.
    """
    if "x" not in matrices:
        raise ValueError("build_v4_manifest requires a primary 'x' matrix")
    primary = matrices["x"]
    m: dict[str, Any] = {
        "schema_version": 4,
        "shard_filename_template": primary["shard_filename_template"],
        "n_shards": len(row_offsets) - 1,
        "n_obs_total": int(n_obs_total),
        "n_vars": int(primary["n_vars"]),
        "row_offsets": list(row_offsets),
        "total_nnz": int(primary["total_nnz"]),
        "matrices": matrices,
        "writer_fingerprint": writer_fingerprint,
        "created_at": created_at,
    }
    if group_by is not None:
        m["group_by"] = group_by
        m["reference_shard"] = reference_shard
    return m


def validate_manifest(m: dict[str, Any]) -> None:
    """Validate the ``row_offsets`` shape invariants of a loaded manifest.

    A structurally valid manifest must satisfy: ``row_offsets`` is a list of
    plain ``int`` (``bool`` rejected) of length ``n_shards + 1``, starts at 0,
    ends at ``n_obs_total``, and is non-decreasing (grouped writes can emit empty
    shards, so adjacent offsets may be equal). A corrupt/tampered manifest that
    violates any of these would otherwise silently mis-route reads
    (``shard_for_row``/``shards_for_range`` assume the invariants hold).

    Raises :class:`IncompatibleSchemaError` — the same manifest-family error
    ``load_manifest`` and the packed reader raise for a bad ``schema_version``,
    keeping the error type consistent across directory and packed archives.
    """
    if m.get("layout") == "cell":
        _validate_cell(m)
        return

    def _bad(why: str):
        return IncompatibleSchemaError(f"manifest is malformed: {why}")

    n_shards = m.get("n_shards")
    n_obs = m.get("n_obs_total")
    offsets = m.get("row_offsets")
    # bool is an int subclass; reject it explicitly for the integer fields.
    # Non-negative too: a negative n_shards with an empty row_offsets would slip
    # past the length check (n_shards+1 == 0 == len([])) into an IndexError.
    if not isinstance(n_shards, int) or isinstance(n_shards, bool) or n_shards < 0:
        raise _bad(f"n_shards must be a non-negative int, got {n_shards!r}")
    if not isinstance(n_obs, int) or isinstance(n_obs, bool) or n_obs < 0:
        raise _bad(f"n_obs_total must be a non-negative int, got {n_obs!r}")
    if not isinstance(offsets, list):
        raise _bad(f"row_offsets must be a list, got {type(offsets).__name__!r}")
    if len(offsets) != n_shards + 1:
        raise _bad(f"len(row_offsets)={len(offsets)} but expected n_shards+1={n_shards + 1}")
    if any(not isinstance(o, int) or isinstance(o, bool) for o in offsets):
        raise _bad(f"row_offsets must be all ints, got {offsets!r}")
    if offsets[0] != 0:
        raise _bad(f"row_offsets[0]={offsets[0]} must be 0")
    if offsets[-1] != n_obs:
        raise _bad(f"row_offsets[-1]={offsets[-1]} must equal n_obs_total={n_obs}")
    if any(offsets[i] > offsets[i + 1] for i in range(len(offsets) - 1)):
        raise _bad(f"row_offsets must be non-decreasing, got {offsets!r}")

    if m.get("schema_version") == 4:
        _validate_v4_matrices(m, n_shards=n_shards)


def _validate_cell(m: dict) -> None:
    """Validate a layout=cell manifest — schema 5 (pfordelta) or 6 (zstd).

    Cell archives are flat and shardless — no n_shards / row_offsets — so they skip
    the shard invariants and get their own required-field gate. v5 = pfordelta
    (value/index uint16|uint32, string pyfastpfor_codec). v6 = zstd (codec 'zstd',
    index uint32, value uint32|float32|float64, null-or-string pyfastpfor_codec).
    Raises IncompatibleSchemaError on any missing/invalid field.
    """

    def _req_nonneg_int(key):
        v = m.get(key, _MISSING)
        if v is _MISSING or isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise IncompatibleSchemaError(
                f"cell manifest field {key!r} must be a non-negative int; got {v!r}."
            )

    sv = m.get("schema_version")
    if sv not in (5, 6):
        raise IncompatibleSchemaError(f"cell manifest requires schema_version 5 or 6; got {sv!r}.")
    codec = m.get("codec")
    if sv == 5:
        if codec != "pfordelta":
            raise IncompatibleSchemaError(
                f"cell manifest schema_version 5 requires codec 'pfordelta'; got {codec!r}."
            )
        allowed_value = ("uint16", "uint32")
        allowed_index = ("uint16", "uint32")
    else:  # sv == 6
        if codec != "zstd":
            raise IncompatibleSchemaError(
                f"cell manifest schema_version 6 requires codec 'zstd'; got {codec!r}."
            )
        allowed_value = ("uint32", "float32", "float64")
        allowed_index = ("uint32",)

    for key in ("n_obs_total", "n_vars", "total_nnz"):
        _req_nonneg_int(key)
    if m.get("value_dtype_on_disk") not in allowed_value:
        raise IncompatibleSchemaError(
            f"cell manifest field 'value_dtype_on_disk' must be one of {allowed_value}; "
            f"got {m.get('value_dtype_on_disk')!r}."
        )
    if m.get("index_dtype_on_disk") not in allowed_index:
        raise IncompatibleSchemaError(
            f"cell manifest field 'index_dtype_on_disk' must be one of {allowed_index}; "
            f"got {m.get('index_dtype_on_disk')!r}."
        )
    for key in ("x_data_dtype_min", "x_indices_dtype_min"):
        if not isinstance(m.get(key), str):
            raise IncompatibleSchemaError(
                f"cell manifest field {key!r} must be a string; got {m.get(key)!r}."
            )
    # pyfastpfor_codec: a string for pfordelta (v5); null for zstd (v6, no pyfastpfor).
    pf = m.get("pyfastpfor_codec", _MISSING)
    if sv == 5:
        if not isinstance(pf, str):
            raise IncompatibleSchemaError(
                f"cell manifest field 'pyfastpfor_codec' must be a string; got {pf!r}."
            )
    else:
        if pf is not None and not isinstance(pf, str):
            raise IncompatibleSchemaError(
                f"cell manifest field 'pyfastpfor_codec' must be null or a string; got {pf!r}."
            )
    # payload_sha256 is nullable (payload_checksum=False -> None); indices_canonical is optional
    # (absent on pre-feature archives). Validate them only when present + non-null.
    sha = m.get("payload_sha256", _MISSING)
    if sha is not _MISSING and sha is not None:
        is_hex64 = (
            isinstance(sha, str)
            and len(sha) == 64
            and all(c in "0123456789abcdefABCDEF" for c in sha)
        )
        if not is_hex64:
            raise IncompatibleSchemaError(
                f"cell manifest field 'payload_sha256' must be null or a 64-char hex string; got {sha!r}."
            )
    canon = m.get("indices_canonical", _MISSING)
    if canon is not _MISSING and not isinstance(canon, bool):
        raise IncompatibleSchemaError(
            f"cell manifest field 'indices_canonical' must be a bool; got {canon!r}."
        )


def _validate_v4_matrices(m: dict[str, Any], *, n_shards: int) -> None:
    """Validate the schema-4 ``matrices`` map against the shared partition.

    Each matrix's ``nnz_per_shard`` must have length ``n_shards``; exactly one
    matrix must have ``role == "x"``; every matrix's ``owner`` (if not None)
    must resolve to another key in ``matrices``; ``n_vars`` must be a
    non-negative int.

    Modality rules: a ``layer:*`` matrix's owner must be an own-var family
    (owner's role is ``"x"`` or ``"modality:*"``); a ``modality:*`` family
    must have ``owner is None`` and carry its own ``var_filename``; and if
    any ``modality:*`` family is present, the ``"x"`` entry must carry
    ``modality_name`` (so ``to_mudata()`` can name the primary modality).
    """

    def _bad(why: str):
        return IncompatibleSchemaError(f"v4 manifest is malformed: {why}")

    matrices = m.get("matrices")
    if not isinstance(matrices, dict) or not matrices:
        raise _bad(f"'matrices' must be a non-empty object, got {type(matrices).__name__!r}")
    primaries = [k for k, v in matrices.items() if isinstance(v, dict) and v.get("role") == "x"]
    if len(primaries) != 1:
        raise _bad(f"expected exactly one matrix with role 'x', found {primaries!r}")
    for key, md in matrices.items():
        if not isinstance(md, dict):
            raise _bad(f"matrix {key!r} entry is not an object")
        nps = md.get("nnz_per_shard")
        if not isinstance(nps, list) or len(nps) != n_shards:
            raise _bad(
                f"matrix {key!r} nnz_per_shard must be a list of length n_shards={n_shards}, "
                f"got {nps!r}"
            )
        owner = md.get("owner")
        if owner is not None and owner not in matrices:
            raise _bad(f"matrix {key!r} owner {owner!r} is not a matrix key")
        n_vars = md.get("n_vars")
        if not isinstance(n_vars, int) or isinstance(n_vars, bool) or n_vars < 0:
            raise _bad(f"matrix {key!r} n_vars must be a non-negative int, got {n_vars!r}")

    has_modality = any(
        isinstance(v, dict) and str(v.get("role", "")).startswith("modality:")
        for v in matrices.values()
    )
    for key, md in matrices.items():
        role = str(md.get("role", ""))
        owner = md.get("owner")
        if role.startswith("layer:"):
            owner_md = matrices.get(owner) if owner is not None else None
            owner_role = str(owner_md.get("role", "")) if isinstance(owner_md, dict) else ""
            if not (owner_role == "x" or owner_role.startswith("modality:")):
                raise _bad(
                    f"layer matrix {key!r} owner {owner!r} must be an own-var family "
                    f"(role 'x' or 'modality:*'), got role {owner_role!r}"
                )
        if role.startswith("modality:"):
            if owner is not None:
                raise _bad(f"modality matrix {key!r} must have owner=None, got {owner!r}")
            if not md.get("var_filename"):
                raise _bad(f"modality matrix {key!r} must have its own var_filename")
    if has_modality:
        x_md = next((v for v in matrices.values() if v.get("role") == "x"), None)
        if not (isinstance(x_md, dict) and x_md.get("modality_name")):
            raise _bad(
                "a multi-modality archive requires the 'x' matrix to carry "
                "'modality_name' (so to_mudata() can name the primary modality)"
            )


def validate_group_records(records: list[Any], *, n_shards: int, n_obs: int) -> None:
    """Validate groups.json records against the archive's shape (ultrareview L1).

    Each record must be a dict carrying ``shard``/``row_start``/``row_stop``
    (plain ``int``; ``bool`` rejected) and ``label`` (``str``), with
    ``0 <= shard < n_shards`` and ``0 <= row_start <= row_stop <= n_obs``.
    Without this, a corrupt/hand-edited groups.json yields a ``KeyError`` /
    ``TypeError`` or a silently out-of-bounds slice that reads the wrong rows.

    Raises :class:`ValueError` describing the first offending record.
    """
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"groups.json record {i} is not an object: {rec!r}")
        for key in ("shard", "row_start", "row_stop"):
            v = rec.get(key, _MISSING)
            if v is _MISSING:
                raise ValueError(f"groups.json record {i} is missing required key {key!r}")
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"groups.json record {i} key {key!r} must be an int, got {v!r}")
        label = rec.get("label", _MISSING)
        if label is _MISSING:
            raise ValueError(f"groups.json record {i} is missing required key 'label'")
        if not isinstance(label, str):
            raise ValueError(f"groups.json record {i} 'label' must be a str, got {label!r}")
        shard, start, stop = rec["shard"], rec["row_start"], rec["row_stop"]
        if not (0 <= shard < n_shards):
            raise ValueError(f"groups.json record {i} shard={shard} out of range [0, {n_shards})")
        if not (0 <= start <= stop <= n_obs):
            raise ValueError(
                f"groups.json record {i} row range [{start}, {stop}) violates "
                f"0 <= row_start <= row_stop <= n_obs ({n_obs})"
            )


# Sentinel for "key absent" so a present-but-None value is still rejected by the
# type checks above (rather than mistaken for missing).
_MISSING = object()


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate schema_version + row_offsets invariants."""
    path = Path(path)
    m = json.loads(path.read_text())
    v = m.get("schema_version")
    if v not in SUPPORTED_SCHEMA_VERSIONS:
        raise IncompatibleSchemaError(
            f"manifest schema_version {v!r} is not supported "
            f"(this cellstream supports {SUPPORTED_SCHEMA_VERSIONS}). "
            f"Either upgrade cellstream or regenerate the archive."
        )
    validate_manifest(m)
    return m


# ---------------------------------------------------------------------------
# Schema v3: groups.json helpers
# ---------------------------------------------------------------------------


def groups_path(archive_dir: str | Path) -> Path:
    """Return the canonical path to the groups.json sidecar."""
    return Path(archive_dir) / "groups.json"


def dump_groups(path: str | Path, groups: list[dict[str, Any]]) -> None:
    """Atomic write of groups.json: tmp + os.replace (mirrors dump_manifest)."""
    import os

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(groups, indent=2))
    os.replace(tmp, path)


def load_groups(path: str | Path) -> list[dict[str, Any]]:
    """Load groups.json and return the list of group records."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"groups.json must contain a JSON array, got {type(data).__name__!r}.")
    return data
