"""Generate the committed golden PACKED archives + expected.json. Run ONCE; commit
the outputs. The reverse-compat test reads the COMMITTED bytes (never regenerates),
so editing this generator does NOT change the frozen fixtures. Re-run only to
intentionally refresh the packed golden set.

Self-contained since #154 part B3a. It used to say it mirrored
tests/v2/golden/generate_golden.py, which is deleted (Alex: "we do not need goldens").

The two golden sets were never byte-identical -- the deleted directory manifest recorded
0.2.1.dev54 while these packed archives record 0.3.1.dev9 (both under the pre-rename
`shardad_version` key, which is what these fixtures still carry on disk), and the grouped
pair came from different writers (v2-native vs v2-native-lean) -- but they DID intentionally
represent the same source data and expectations: both expected.json files carry the same
data_sha256, 1d15bdb4...caf6221. Only the archive bytes and manifests differed. The
deterministic AnnData below is now the sole definition of this fixture's content.

Run: .venv/bin/python tests/packed/golden/generate_packed_golden.py
"""

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cellstream import write_sharded

HERE = Path(__file__).parent


def _adata():
    # The fixture's content, defined here and nowhere else. It was described as "identical to
    # tests/v2/golden/generate_golden.py::_adata so packed and directory goldens hold the same
    # content (same data_sha256)" -- that generator is deleted, but the claim was TRUE: the two
    # expected.json files share data_sha256 1d15bdb4...caf6221. Only the reference is dead.
    n, n_vars = 40, 12
    X = sp.random(n, n_vars, density=0.4, format="csr", dtype=np.float32, random_state=0)
    X.data = np.ceil(X.data * 9).astype(np.float32)
    labels = ["GENE_A"] * 10 + ["GENE_B"] * 10 + ["GENE_C"] * 8 + ["non-targeting"] * 12
    obs = pd.DataFrame(
        {
            "target_gene": pd.Categorical(labels),
            "assignment": pd.Categorical([f"g{i % 5}" for i in range(n)]),
        },
        index=[f"cell{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"g{j}" for j in range(n_vars)])
    return ad.AnnData(X=X, obs=obs, var=var)


def _hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main():
    # Re-running rewrites the .shardpack files; non-content fields (created_at,
    # writer_fingerprint in the metadata tail/footer) legitimately change and are
    # NOT asserted by the reverse-compat test. sp.random(random_state=0) pins the
    # sparsity pattern; any intentional refresh that switches to the scipy `rng=`
    # keyword changes the bytes and MUST regenerate fixtures + update expected.json.
    a = _adata()
    plain = HERE / "plain.shardpack"
    grouped = HERE / "grouped.shardpack"
    write_sharded(
        a,
        str(plain),
        format="v2",
        codec="v2-byte-filter",
        n_shards=3,
        overwrite=True,
    )
    write_sharded(
        a,
        str(grouped),
        format="v2",
        codec="v2-byte-filter",
        group_by="target_gene",
        reference=["non-targeting"],
        target_shard_bytes=512,
        overwrite=True,
    )
    expected = {
        "n_obs": int(a.shape[0]),
        "n_vars": int(a.shape[1]),
        "nnz": int(a.X.nnz),
        "data_sha256": _hash(a.X.data.astype(np.uint16)),
        "group_GENE_A_n": int((a.obs["target_gene"].astype(str) == "GENE_A").sum()),
        "reference_n": int((a.obs["target_gene"].astype(str) == "non-targeting").sum()),
    }
    (HERE / "expected.json").write_text(json.dumps(expected, indent=2))
    print("wrote packed golden fixtures + expected.json")


if __name__ == "__main__":
    main()
