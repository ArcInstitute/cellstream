"""PR4 streaming cell write (#219): the byte-identity gate.

THE GATE (spec §3): a streamed archive and a materialised archive of the same input must be
equivalent -- identical payload bytes, offsets, indptr, and manifest (minus `created_at`).
NOT a whole-file compare: manifest.json embeds created_at, so two writes of the same input
straddling a second boundary yield different FILES with identical PAYLOADS (Gemini #220 r2).
"""

from __future__ import annotations

import itertools
import shutil

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from cellstream.cell._rust_dispatch import use_rust_for_cell_encode
from cellstream.cell.source import DenseH5adSource, open_cell_source
from cellstream.cell.writer import _plan_cell_codec, write_cell_archive
from cellstream.errors import LossyCastError
from cellstream.packed.reader import PackedArchive

from .conftest import assert_archives_equivalent, make_cell_adata


def test_gate_helper_accepts_two_identical_writes(tmp_path):
    """Two materialised writes of the same input differ only in created_at -> equivalent."""
    a = make_cell_adata()
    p, q = tmp_path / "p.shad", tmp_path / "q.shad"
    write_cell_archive(a, p)
    write_cell_archive(a, q)
    assert_archives_equivalent(p, q)


def test_gate_helper_rejects_a_different_storage_order(tmp_path):
    """`b` is a row-reversed copy of `a`; both write group_by=None -> identical manifest (obs
    isn't in it, and n_obs/n_vars/total_nnz/codec/dtypes are permutation-invariant) but a
    genuinely different storage order, so the AssertionError must come from inside the try
    block (x_offsets). A grouped-vs-ungrouped pair can't test this: group_by itself differs, so
    the manifest compare fires first and the offsets/indptr/payload comparisons never run.
    """
    a = make_cell_adata()
    b = a[np.arange(a.n_obs)[::-1], :].copy()
    p, q = tmp_path / "p.shad", tmp_path / "q.shad"
    write_cell_archive(a, p, group_by=None)
    write_cell_archive(b, q, group_by=None)
    with pytest.raises(AssertionError, match="x_offsets differ"):
        assert_archives_equivalent(p, q)


def test_gate_helper_rejects_a_payload_byte_flip(tmp_path):
    """`q` starts as a raw byte-copy of `p` (identical manifest AND x_offsets AND x_indptr),
    then one x_payload byte is flipped -- so the AssertionError can only come from the payload
    compare itself. A grouped-vs-ungrouped pair can't reach this comparison: its manifest
    differs first. This also guards the mmap del-before-close discipline in
    ``assert_archives_equivalent``: were that broken, a BufferError would surface here instead
    of a clean AssertionError.
    """
    a = make_cell_adata()
    p, q = tmp_path / "p.shad", tmp_path / "q.shad"
    write_cell_archive(a, p)
    shutil.copyfile(p, q)
    _, base, length = PackedArchive(q).member_location("x_payload")
    assert length > 0
    with open(q, "r+b") as f:
        f.seek(base)
        flipped = f.read(1)[0] ^ 0xFF
        f.seek(base)
        f.write(bytes([flipped]))
    with pytest.raises(AssertionError, match="x_payload bytes differ"):
        assert_archives_equivalent(p, q)


def test_resolve_storage_order_never_touches_matrix_data(cell_adata):
    """#219 site 1: the planner needs per-row nnz, which comes from INDPTR ALONE. Hand it a
    source whose read_block explodes -- the perm must still come out correct.

    NOT sufficient on its own: InMemoryAnnDataSource.X is eagerly pre-built (see its
    __init__), so the OLD `adata.X.tocsr()` body never reached read_block either and passed
    this test unchanged. The test that actually pins the contract is
    test_resolve_storage_order_works_on_a_backed_source_without_X below.
    """
    from cellstream.cell.source import InMemoryAnnDataSource
    from cellstream.cell.writer import _resolve_storage_order

    class _Exploding(InMemoryAnnDataSource):
        def read_block(self, lo, hi):
            raise AssertionError("_resolve_storage_order must not read matrix data")

    src = _Exploding(cell_adata)
    perm, groups = _resolve_storage_order(src.obs, src.row_nnz(), "target_gene", None, None, None)
    assert sorted(perm.tolist()) == list(range(cell_adata.n_obs))
    assert groups is not None and len(groups["groups"]) > 0

    perm_u, groups_u = _resolve_storage_order(src.obs, None, None, None, None, None)
    np.testing.assert_array_equal(perm_u, np.arange(cell_adata.n_obs))
    assert groups_u is None


def test_resolve_storage_order_works_on_a_backed_source_without_X(tmp_path, cell_adata):
    """BackedH5adSource has NO .X property -- the old `adata.X.tocsr()` cannot even RUN against
    it. This is what actually proves _resolve_storage_order no longer needs the matrix. The
    exploding-read_block test above cannot prove it: InMemoryAnnDataSource.X is eagerly
    pre-built, so the old code never reached read_block either and passed that test too.
    (Task 3 review, Important.)
    """
    from cellstream.cell.source import open_cell_source
    from cellstream.cell.writer import _resolve_storage_order

    p = tmp_path / "s.h5ad"
    cell_adata.write_h5ad(p)
    src = open_cell_source(p)
    try:
        perm, groups = _resolve_storage_order(
            src.obs, src.row_nnz(), "target_gene", None, None, None
        )
        assert sorted(perm.tolist()) == list(range(cell_adata.n_obs))
        assert groups is not None and len(groups["groups"]) > 0
    finally:
        src.close()


def test_resolve_storage_order_rejects_mismatched_row_nnz_length(cell_adata):
    """The grouped branch feeds row_nnz into plan_group_shards -- a length mismatch would
    otherwise silently corrupt the plan rather than raise (Copilot #222 review). Fail loud
    instead, and confirm the ungrouped row_nnz=None path (which never touches row_nnz) still
    works unchanged.
    """
    from cellstream.cell.writer import _resolve_storage_order

    bad_row_nnz = np.ones(cell_adata.n_obs + 1, dtype=np.int64)
    with pytest.raises(ValueError, match="row_nnz length"):
        _resolve_storage_order(cell_adata.obs, bad_row_nnz, "target_gene", None, None, None)

    perm_u, groups_u = _resolve_storage_order(cell_adata.obs, None, None, None, None, None)
    np.testing.assert_array_equal(perm_u, np.arange(cell_adata.n_obs))
    assert groups_u is None


def test_gate_cross_product_never_lets_rust_raise():
    """THE PR3 LESSON (Gemini #212 r2; Codex found it in the gate itself on #220): if a planned
    (codec, value_dtype) attempt is a combination the Rust encoder REJECTS while
    use_rust_for_cell_encode returns True, the writer RAISES instead of falling back to the Python
    encoder. Re-verify the two accept-sets agree for EVERY attempt _plan_cell_codec can emit.

    The invariant Rust relies on: FLOAT ON-DISK => FLOAT SOURCE.
    """
    for src, codec, ddt in itertools.product(
        ["uint16", "uint32", "int32", "int64", "float32", "float64"],
        [None, "pfordelta", "zstd"],
        [None, "float32", "float64"],
    ):
        try:
            attempts = _plan_cell_codec(np.dtype(src), codec=codec, data_dtype=ddt)
        except ValueError:
            continue  # rejected by the planner before Rust ever sees it -- fine
        for resolved, value_dtype in attempts:
            is_float_storage = value_dtype in ("float32", "float64")
            if is_float_storage:
                assert np.dtype(src).kind == "f", (
                    f"is_float_storage with integer source {src!r}: Rust's encode_cells rejects "
                    f"this, so the gate returns True and RAISEs instead of falling back to Python"
                )
            for idx in ("int32", "int64"):
                if use_rust_for_cell_encode(resolved, src, idx, value_dtype or "uint32"):
                    # whatever the gate accepts, Rust must accept too
                    assert not (is_float_storage and np.dtype(src).kind != "f")


@pytest.mark.parametrize("codec", ["pfordelta", "zstd"])
@pytest.mark.parametrize("group_by", [None, "target_gene"])
@pytest.mark.parametrize("block_size", [1, 7, 10_000])
def test_streamed_h5ad_equals_materialised(tmp_path, codec, group_by, block_size):
    """THE GATE (spec §3): streaming must be byte-identical to materialising.
    group_by='target_gene' exercises reorder_frames; group_by=None skips it."""
    a = make_cell_adata()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)

    mat, stream = tmp_path / "m.shad", tmp_path / "s.shad"
    write_cell_archive(a, mat, codec=codec, group_by=group_by)  # in-memory
    write_cell_archive(src, stream, codec=codec, group_by=group_by, block_size=block_size)
    assert_archives_equivalent(mat, stream)


@pytest.mark.parametrize("codec", ["pfordelta", "zstd"])
@pytest.mark.parametrize("group_by", [None, "target_gene"])
def test_streamed_cell_archive_equals_materialised(tmp_path, codec, group_by):
    """The re-encode path -- the one #219's OOM was measured on."""
    a = make_cell_adata()
    src = tmp_path / "src.shad"
    write_cell_archive(a, src)

    # Materialise from the SAME logical input the stream sees: gather_rows widens to float32,
    # so the in-memory reference must be built from gathered blocks, not from `a` directly.
    # var/uns must ALSO come from the source rather than AnnData's auto-generated defaults --
    # assert_archives_equivalent now compares both (PR4 review findings 1/2), and an
    # auto-generated var (numeric column names, not the real gene names) or a dropped uns
    # would make this "materialised" reference wrong in exactly the way the streamed side
    # used to be, which is precisely how this bug stayed invisible.
    s = open_cell_source(src)
    a_gathered = ad.AnnData(
        X=s.read_block(0, s.n_obs), obs=s.obs.copy(), var=s.var.copy(), uns=s.uns
    )
    s.close()

    mat, stream = tmp_path / "m.shad", tmp_path / "s.shad"
    write_cell_archive(a_gathered, mat, codec=codec, group_by=group_by)
    write_cell_archive(src, stream, codec=codec, group_by=group_by, block_size=13)
    assert_archives_equivalent(mat, stream)


def test_streamed_reencode_preserves_uns_and_var(tmp_path):
    """PR4 review finding 1: CellArchiveSource.uns unconditionally returned {}, so
    write_cell_archive(some.shad, out.shad, ...) -- the codec-migration / re-encode workflow
    PR4 exists for -- silently dropped uns on every re-encode, no warning, no error. Finding
    2: the gate never read obs.parquet/header.h5ad, so this sailed through every existing
    test. Must fail before the CellArchiveSource.uns fix and pass after -- pins both the fix
    and a non-trivial var column through the SAME re-encode. var is compared by VALUE
    (list(...)), not pandas dtype-strict equality: anndata's h5ad writer silently promotes a
    repeated-string column like this one to Categorical, which is irrelevant to what this
    test is pinning.
    """
    from cellstream.cell.reader import open_cell

    a = make_cell_adata()
    a.uns["important_metadata"] = {"description": "pinned by PR4 review finding 1", "value": 7}
    a.var["chromosome"] = [f"chr{i % 5}" for i in range(a.n_vars)]

    src = tmp_path / "src.shad"
    write_cell_archive(a, src)

    out = tmp_path / "out.shad"
    write_cell_archive(src, out, block_size=13)  # the streaming re-encode path

    reopened = open_cell(out)
    try:
        assert reopened.uns == dict(a.uns), f"uns was dropped on re-encode: {reopened.uns!r}"
        assert list(reopened.var["chromosome"]) == list(a.var["chromosome"]), (
            "var column 'chromosome' was not preserved on re-encode"
        )
    finally:
        reopened.close()


def test_block_size_larger_than_n_obs_degenerates(tmp_path):
    a = make_cell_adata()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    mat, stream = tmp_path / "m.shad", tmp_path / "s.shad"
    write_cell_archive(a, mat, group_by="target_gene")
    write_cell_archive(src, stream, group_by="target_gene", block_size=10**9)
    assert_archives_equivalent(mat, stream)


def test_empty_cells_and_an_all_empty_block(tmp_path):
    """Rows with nnz=0, including a whole block of them (block_size=5 over rows 10-19)."""
    a = make_cell_adata()
    X = a.X.tocsr().tolil()
    for i in range(10, 20):
        X[i, :] = 0
    a.X = X.tocsr().astype(np.uint16)
    a.X.eliminate_zeros()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    mat, stream = tmp_path / "m.shad", tmp_path / "s.shad"
    write_cell_archive(a, mat, group_by="target_gene")
    write_cell_archive(src, stream, group_by="target_gene", block_size=5)
    assert_archives_equivalent(mat, stream)


@pytest.mark.parametrize("group_by", [None, "target_gene"])
def test_streamed_float_with_nan(tmp_path, group_by):
    """NaN is FIRST-CLASS in a float cell archive (PR3's lossy gate uses equal_nan, so NaN -> NaN
    is lossless). The pure-Python streaming encoder must not choke computing block_max:
    `int(np.max(data))` raises ValueError("cannot convert float NaN to integer"). Float storage
    never consumes max_value anyway. (Gemini #221 r2.)

    MUST be exercised on BOTH engines -- only the pure-Python one has the int() call.
    """
    rng = np.random.default_rng(11)
    X = sp.random(60, 40, density=0.3, format="csr", random_state=rng, dtype=np.float32)
    X.data[::7] = np.nan
    X.data[1::11] = np.inf
    X.sort_indices()
    a = ad.AnnData(X=X)
    a.obs["target_gene"] = [f"t{i % 3}" for i in range(60)]
    src = tmp_path / "nan.h5ad"
    a.write_h5ad(src)

    mat, stream = tmp_path / "m.shad", tmp_path / "s.shad"
    write_cell_archive(a, mat, codec="zstd", group_by=group_by)
    write_cell_archive(src, stream, codec="zstd", group_by=group_by, block_size=7)
    assert_archives_equivalent(mat, stream)


def test_payload_checksum_matches_across_paths(tmp_path):
    a = make_cell_adata()
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    mat, stream = tmp_path / "m.shad", tmp_path / "s.shad"
    write_cell_archive(a, mat, group_by="target_gene", payload_checksum=True)
    write_cell_archive(src, stream, group_by="target_gene", payload_checksum=True, block_size=9)
    assert_archives_equivalent(mat, stream)  # the manifest compare covers payload_sha256


def test_empty_archive_streams(tmp_path):
    """n_obs=0 -> _plan_blocks returns [] -> encode_cells_block is NEVER CALLED. Without the
    up-front truncate, tmpd/x_payload would not exist and pack() would fail on a missing member."""
    a = ad.AnnData(X=sp.csr_matrix((0, 12), dtype=np.uint16))
    src = tmp_path / "e.h5ad"
    a.write_h5ad(src)
    mat, stream = tmp_path / "m.shad", tmp_path / "s.shad"
    write_cell_archive(a, mat)
    write_cell_archive(src, stream, block_size=4)
    assert_archives_equivalent(mat, stream)


def test_wide_n_vars_uint32_indices(tmp_path):
    """n_vars > 65536 -> uint32 on-disk indices, on the streaming path too."""
    a = make_cell_adata(n_obs=40, n_vars=70_000)
    src = tmp_path / "src.h5ad"
    a.write_h5ad(src)
    mat, stream = tmp_path / "m.shad", tmp_path / "s.shad"
    write_cell_archive(a, mat, group_by="target_gene")
    write_cell_archive(src, stream, group_by="target_gene", block_size=6)
    assert_archives_equivalent(mat, stream)


def test_tmp_dir_defaults_next_to_out_path(tmp_path, monkeypatch):
    """§7.4: tmp_dir defaults to out_path.parent, NOT TMPDIR -- the output filesystem must hold
    the archive anyway, whereas /tmp typically cannot hold even 1x payload."""
    import tempfile as _tf

    seen = []
    real = _tf.TemporaryDirectory

    class _Spy(real):
        def __init__(self, *args, **kw):
            seen.append(kw.get("dir"))
            super().__init__(*args, **kw)

    monkeypatch.setattr("cellstream.cell.writer.tempfile.TemporaryDirectory", _Spy)
    a = make_cell_adata()
    out = tmp_path / "sub" / "o.shad"
    out.parent.mkdir()
    write_cell_archive(a, out)
    assert seen and str(seen[0]) == str(out.parent)


def test_streamed_negative_int_fails_loud(tmp_path):
    """A negative count in a STREAMED (path) source must fail loud, exactly like the in-memory
    path already does (test_zstd_int_rejects_negative: "must fail loud, not silently wrap on
    astype(uint32)", Gemini PR #216 HIGH)."""
    a = make_cell_adata()
    x = a.X.tocsr().astype(np.int32)
    x.data[0] = -5
    a.X = x
    src = tmp_path / "neg.h5ad"
    a.write_h5ad(src)
    with pytest.raises(ValueError, match="got non-integer, negative, or non-finite values"):
        write_cell_archive(src, tmp_path / "out.shad", block_size=7)


def test_streamed_zstd_int_rejects_negative(tmp_path):
    """Same as above, through the zstd (not pfordelta) streaming branch of
    _stream_encode_block_python -- same function, different on_disk_value, must not drift."""
    a = make_cell_adata()
    x = a.X.tocsr().astype(np.int32)
    x.data[0] = -5
    a.X = x
    src = tmp_path / "neg.h5ad"
    a.write_h5ad(src)
    with pytest.raises(ValueError, match="got non-integer, negative, or non-finite values"):
        write_cell_archive(src, tmp_path / "out.shad", codec="zstd", block_size=7)


def test_streamed_int_overflow_fails_loud(tmp_path):
    """A uint32-overflowing count (2**32, the reviewer's exact repro) in a STREAMED (path)
    source must raise Rust's OTHER CountU32 message -- distinct from the negative/non-integer
    one above -- naming the offending value, not the generic fused-fold message the streaming
    Python fallback used to raise for both failure modes alike (PR4 review finding 3)."""
    a = make_cell_adata()
    x = a.X.tocsr().astype(np.int64)
    x.data[0] = 2**32
    a.X = x
    src = tmp_path / "big.h5ad"
    a.write_h5ad(src)
    with pytest.raises(ValueError, match=r"exceeds the uint32 maximum") as excinfo:
        write_cell_archive(src, tmp_path / "out.shad", block_size=7)
    assert "4294967296" in str(excinfo.value)


def test_streamed_zstd_int_overflow_fails_loud(tmp_path):
    """Same as above through the zstd streaming branch -- same function, different
    on_disk_value, must not drift."""
    a = make_cell_adata()
    x = a.X.tocsr().astype(np.int64)
    x.data[0] = 2**32
    a.X = x
    src = tmp_path / "big.h5ad"
    a.write_h5ad(src)
    with pytest.raises(ValueError, match=r"exceeds the uint32 maximum") as excinfo:
        write_cell_archive(src, tmp_path / "out.shad", codec="zstd", block_size=7)
    assert "4294967296" in str(excinfo.value)


def test_streamed_float64_to_float32_lossy_raises_without_allow_lossy(tmp_path):
    """The f64->f32 lossy gate (LossyCastError) must still fire on a STREAMED pure-Python float
    encode -- _stream_encode_block_python must route through cast_data, not a raw .astype() that
    would silently truncate precision (mirrors the in-memory
    test_data_dtype_float32_lossless_checked_raises)."""
    rng = np.random.default_rng(3)
    X = sp.random(60, 40, density=0.3, format="csr", random_state=rng, dtype=np.float64)
    X.data[:] = np.abs(X.data) + 1.0
    X.data[0] = 1.0 + 2**-40  # not exactly representable in float32 -> a lossy downcast
    a = ad.AnnData(X=X)
    src = tmp_path / "f64.h5ad"
    a.write_h5ad(src)
    with pytest.raises(LossyCastError):
        write_cell_archive(
            src, tmp_path / "out.shad", codec="zstd", data_dtype="float32", block_size=7
        )


def test_streamed_float64_to_float32_lossless_matches_materialised(tmp_path):
    """Happy-path complement: a float64->float32 narrow that DOES round-trip exactly must still
    succeed via cast_data on the streaming pure-Python path, and be byte-identical to the
    materialised write -- not merely "does not raise", genuinely correct bytes."""
    rng = np.random.default_rng(5)
    X = sp.random(60, 40, density=0.3, format="csr", random_state=rng, dtype=np.float64)
    X.data[:] = np.round(np.abs(X.data) * 1000).astype(np.float64)  # exact in float32 too
    a = ad.AnnData(X=X)
    src = tmp_path / "f64ok.h5ad"
    a.write_h5ad(src)
    mat, stream = tmp_path / "m.shad", tmp_path / "s.shad"
    write_cell_archive(a, mat, codec="zstd", data_dtype="float32")
    write_cell_archive(src, stream, codec="zstd", data_dtype="float32", block_size=7)
    assert_archives_equivalent(mat, stream)


@pytest.mark.parametrize(
    "dtype,codec",
    [(np.uint16, "pfordelta"), (np.float32, "zstd"), (np.float64, "zstd")],
)
@pytest.mark.parametrize("group_by", [None, "target_gene"])
@pytest.mark.parametrize("block_size", [None, 7])
def test_dense_h5ad_equals_materialised_and_csr(tmp_path, dtype, codec, group_by, block_size):
    """THE #225 GATE: a dense-X h5ad streams (via DenseH5adSource) to an archive byte-identical to
    (a) the in-memory path over the same dense data and (b) the CSR-equivalent h5ad. Covers
    pfordelta(int) + zstd(float32/float64), grouped + ungrouped, auto + fixed block_size."""
    a = make_cell_adata()
    a.X = a.X.toarray().astype(dtype)  # dense-X, in-memory AnnData

    dense_p = tmp_path / "dense.h5ad"
    a.write_h5ad(dense_p)  # dense on disk -> encoding-type "array"
    a_csr = a.copy()
    a_csr.X = sp.csr_matrix(a.X)  # CSR of the SAME dense values -> identical nonzero structure
    csr_p = tmp_path / "csr.h5ad"
    a_csr.write_h5ad(csr_p)

    # Guard: the dense h5ad must actually take the streaming DenseH5adSource path, not silently
    # fall back to the in-memory read (which would make this gate vacuous).
    probe = open_cell_source(dense_p)
    assert isinstance(probe, DenseH5adSource)
    probe.close()

    mat, dfrom, cfrom = tmp_path / "m.shad", tmp_path / "d.shad", tmp_path / "c.shad"
    write_cell_archive(a, mat, codec=codec, group_by=group_by)  # in-memory reference
    write_cell_archive(dense_p, dfrom, codec=codec, group_by=group_by, block_size=block_size)
    write_cell_archive(csr_p, cfrom, codec=codec, group_by=group_by, block_size=block_size)

    assert_archives_equivalent(mat, dfrom)
    assert_archives_equivalent(dfrom, cfrom)
