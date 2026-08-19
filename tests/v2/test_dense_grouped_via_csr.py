import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream.v2.writer import write_v2_archive

pytest.importorskip("cellstream.v2._rust")


def _adata_dense(D):
    n = D.shape[0]
    return ad.AnnData(
        X=D,
        obs=pd.DataFrame(
            {"target_gene": [f"g{i % 4}" for i in range(n)]},
            index=[f"c{i}" for i in range(n)],
        ),
    )


def _shard_bytes(d):
    return {p.name: p.read_bytes() for p in sorted(d.glob("shard_*.shx"))}


@pytest.mark.parametrize("maker", ["u16safe", "wide_float"])
def test_dense_via_csr_default_matches_scipy_csr(tmp_path, maker):
    rng = np.random.default_rng(0)
    if maker == "u16safe":
        D = rng.integers(0, 50, size=(60, 9)).astype(np.float32)
    else:  # values that do NOT narrow to u16 -> on-disk stays float32
        D = (rng.integers(0, 50, size=(60, 9)).astype(np.float32)) + 0.5
    D[rng.random(D.shape) < 0.5] = 0.0

    out_dense = tmp_path / "dense"
    out_csr = tmp_path / "csr"
    write_v2_archive(_adata_dense(D), out_dense, group_by="target_gene", codec="v2-byte-filter")
    write_v2_archive(
        _adata_dense(sp.csr_matrix(D)), out_csr, group_by="target_gene", codec="v2-byte-filter"
    )
    db, cb = _shard_bytes(out_dense), _shard_bytes(out_csr)
    assert db and db == cb


from cellstream.write import write_sharded  # noqa: E402
from tests._archive_helpers import write_archive  # noqa: E402


def test_via_csr_matches_direct(tmp_path):
    """The via-CSR (True) and direct (False/default) paths must produce
    byte-identical output — the #145/#146 invariant this opt-in flip preserves.
    (Replaces test_knob_false_matches_default, which compared default-vs-False;
    after the flip the default IS False, so that comparison went trivial.)"""
    rng = np.random.default_rng(7)
    D = rng.integers(0, 50, size=(60, 9)).astype(np.float32)
    D[rng.random(D.shape) < 0.5] = 0.0
    a = _adata_dense(D)
    out_via = tmp_path / "via_csr"
    out_direct = tmp_path / "direct"
    write_v2_archive(
        a,
        out_via,
        group_by="target_gene",
        codec="v2-byte-filter",
        dense_grouped_via_csr=True,
    )
    write_v2_archive(
        a,
        out_direct,
        group_by="target_gene",
        codec="v2-byte-filter",
        dense_grouped_via_csr=False,
    )
    assert _shard_bytes(out_via) == _shard_bytes(out_direct)
    # And the omitted-knob default must equal the direct path (default is now False).
    out_default = tmp_path / "default"
    write_v2_archive(a, out_default, group_by="target_gene", codec="v2-byte-filter")
    assert _shard_bytes(out_default) == _shard_bytes(out_direct)


# test_knob_raises_on_v1 lived here: dense_grouped_via_csr=True was v2-only and raised
# "v2-only" for format="v1". Gone with the v1 writer (#154), along with
# test_false_on_v1_does_not_raise_knob_guard below -- the ultrareview L13 pin that an
# explicit False did NOT trip that guard on a v1 write. The False-is-a-noop half of L13
# survives as test_false_without_group_by_is_noop.


def test_knob_raises_without_group_by(tmp_path):
    # Only an actual via-CSR opt-in (=True) requires group_by; False/None are no-ops.
    a = _adata_dense(np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="requires group_by"):
        write_sharded(a, tmp_path / "ng", dense_grouped_via_csr=True)


def test_false_without_group_by_is_noop(tmp_path):
    """dense_grouped_via_csr=False == the documented default; it must NOT raise
    'requires group_by' when group_by is omitted (ultrareview L13)."""
    a = _adata_dense(np.zeros((4, 3), dtype=np.float32))
    write_sharded(a, tmp_path / "ok", dense_grouped_via_csr=False)  # must not raise


def _spy_dispatch(monkeypatch):
    """Wrap both dense dispatch fns to count calls while still producing output.
    Returns the live counter dict {"direct", "via_csr"}."""
    import cellstream.v2._rust_dispatch as rd

    if not rd.rust_available():
        pytest.skip(
            "dense dispatch is Rust-only; under CELLSTREAM_DISABLE_RUST the dense "
            "branch takes the scipy→CSR path and calls neither dispatch fn"
        )
    calls = {"direct": 0, "via_csr": 0}
    real_direct = rd.encode_inmem_dense
    real_via = rd.encode_inmem_dense_via_csr

    def spy_direct(*a, **k):
        calls["direct"] += 1
        return real_direct(*a, **k)

    def spy_via(*a, **k):
        calls["via_csr"] += 1
        return real_via(*a, **k)

    monkeypatch.setattr(rd, "encode_inmem_dense", spy_direct)
    monkeypatch.setattr(rd, "encode_inmem_dense_via_csr", spy_via)
    return calls


def _dense_for_path_test():
    rng = np.random.default_rng(11)
    D = rng.integers(0, 50, size=(60, 9)).astype(np.float32)
    D[rng.random(D.shape) < 0.5] = 0.0
    return _adata_dense(D)


def test_default_uses_direct_path(tmp_path, monkeypatch):
    # Knob OMITTED -> default is now the direct encoder (opt-in via-CSR after the
    # 2026-06-18 flip). Byte-equality tests can't see this (both paths are
    # byte-identical), so pin the dispatch explicitly.
    #
    # Stays on the PUBLIC writer (#154 part B): the subject is that write_sharded
    # resolves and FORWARDS dense_grouped_via_csr, so routing this at the internal
    # writer would leave a dropped forwarding argument green. Neither of these two
    # tests reads the output, only the spy's counts, so packed is fine.
    calls = _spy_dispatch(monkeypatch)
    write_archive(
        _dense_for_path_test(),
        tmp_path / "def",
        codec="v2-byte-filter",
        group_by="target_gene",
    )
    assert calls["direct"] == 1 and calls["via_csr"] == 0


def test_true_uses_via_csr_path(tmp_path, monkeypatch):
    calls = _spy_dispatch(monkeypatch)
    write_archive(
        _dense_for_path_test(),
        tmp_path / "via",
        codec="v2-byte-filter",
        group_by="target_gene",
        dense_grouped_via_csr=True,
    )
    assert calls["via_csr"] == 1 and calls["direct"] == 0


def test_knob_noop_for_csr_source(tmp_path):
    rng = np.random.default_rng(3)
    D = rng.integers(0, 50, size=(60, 9)).astype(np.float32)
    D[rng.random(D.shape) < 0.5] = 0.0
    a = _adata_dense(sp.csr_matrix(D))
    out = tmp_path / "csr_src"
    # Accepted (v2 + group_by) and simply ignored for a CSR source.
    write_v2_archive(
        a,
        out,
        group_by="target_gene",
        codec="v2-byte-filter",
        dense_grouped_via_csr=False,
    )
    assert _shard_bytes(out)
