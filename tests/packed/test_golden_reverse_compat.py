"""Reverse-compat guard: future cellstream MUST keep reading these committed packed
.shardpack archives (plain + grouped). Reads the COMMITTED bytes (does not
regenerate) and asserts decoded content matches the frozen expected.json. Robust
to zstd-library version drift (zstd decode is frame-stable).

Since #154 part B3a this is the ONLY reverse-compat guard in the shard layout, and it is
self-contained. It used to say it mirrored tests/v2/test_golden_reverse_compat.py; that file and its
directory fixtures are deleted (Alex: "we do not need goldens"). The two corpora were not
byte-identical -- 0.2.1.dev54 there, 0.3.1.dev9 here -- but they did assert the same decoded content
(both expected.json files carry data_sha256 1d15bdb4...caf6221), so what died was the reference, not
the equivalence. The packed container is the one that ships, which is why this half stayed."""

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from cellstream.packed.reader import is_packed_file
from cellstream.read import ShardedArchive

GOLDEN = Path(__file__).parent / "golden"
EXPECTED = json.loads((GOLDEN / "expected.json").read_text())


def _hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def test_plain_packed_golden_reads():
    pack = GOLDEN / "plain.shardpack"
    assert is_packed_file(pack)  # a single-file packed container, not a directory
    arch = ShardedArchive(str(pack))
    assert arch.manifest["schema_version"] == 2
    assert arch.manifest["codec"] == "v2-byte-filter"
    a = arch.to_anndata()
    assert a.shape == (EXPECTED["n_obs"], EXPECTED["n_vars"])
    X = a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)
    assert int(X.nnz) == EXPECTED["nnz"]
    assert _hash(X.data.astype(np.uint16)) == EXPECTED["data_sha256"]
    if hasattr(a, "close_shared_memory"):
        a.close_shared_memory()


def test_grouped_packed_golden_reads():
    pack = GOLDEN / "grouped.shardpack"
    assert is_packed_file(pack)
    arch = ShardedArchive(str(pack))
    assert arch.manifest["schema_version"] == 3
    assert arch.manifest["codec"] == "v2-byte-filter"
    assert arch.manifest["group_by"] == "target_gene"
    g = arch.read_group("GENE_A")
    assert g.shape[0] == EXPECTED["group_GENE_A_n"]
    assert (g.obs["target_gene"].astype(str) == "GENE_A").all()
    ref = arch.read_reference()
    assert ref is not None and ref.shape[0] == EXPECTED["reference_n"]
    for x in (g, ref):
        if hasattr(x, "close_shared_memory"):
            x.close_shared_memory()
