"""The optimised pure-Python cell fallback must be BIT-IDENTICAL to the old
concatenate-based one -- for valid archives and corrupt ones alike (#244 ask #3)."""

import numpy as np
import pytest

from cellstream.cell.codec import codec_for

# #250 made new_decoder(n_vars) required so the pfordelta decoder can bound the wide
# cumsum before narrowing. These equivalence tests are about decode_into == decode, not
# about bounds, so they use a ceiling comfortably above the largest column they build
# (40000); the bounds behaviour itself is covered by tests/cell/test_column_bounds.py.
_N_VARS = 1 << 20


def _roundtrip_frame(codec_name, index_dtype, value_dtype, indices, values):
    c = codec_for(codec_name, index_dtype, value_dtype)
    return c, c.new_encoder().encode(indices, values)


# Both MIXED pfordelta widths are included deliberately: index and value on-disk dtypes are
# chosen independently by the writer, and _copy_via's narrowing rule is keyed on each one
# separately -- so (uint16, uint32) and (uint32, uint16) exercise arms the matched pairs do not.
# (codex checkpoint-2 review.)
CODECS = [
    ("pfordelta", "uint16", "uint16"),
    ("pfordelta", "uint32", "uint32"),
    ("pfordelta", "uint16", "uint32"),
    ("pfordelta", "uint32", "uint16"),
    ("zstd", "uint32", "uint32"),
    ("zstd", "uint32", "float32"),
    ("zstd", "uint32", "float64"),
]


@pytest.mark.parametrize("codec_name,idt,vdt", CODECS)
@pytest.mark.parametrize("out_dtype", [np.float32, np.float64])
def test_decode_into_matches_decode(codec_name, idt, vdt, out_dtype):
    idx = np.array([0, 3, 7, 11, 40000], dtype=np.uint32)
    val = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    c, frame = _roundtrip_frame(codec_name, idt, vdt, idx, val)
    dec = c.new_decoder(_N_VARS)

    ref_i, ref_v = dec.decode(frame, idx.size)
    expected_i = ref_i.astype(np.int32)
    expected_v = ref_v.astype(out_dtype)

    dec2 = c.new_decoder(_N_VARS)
    out_i = np.empty(idx.size, dtype=np.int32)
    out_v = np.empty(idx.size, dtype=out_dtype)
    dec2.decode_into(frame, idx.size, out_i, out_v)

    np.testing.assert_array_equal(out_i, expected_i)
    np.testing.assert_array_equal(out_v, expected_v)
    assert out_i.dtype == np.int32 and out_v.dtype == out_dtype


@pytest.mark.parametrize("codec_name,idt,vdt", CODECS)
def test_decode_into_empty_row(codec_name, idt, vdt):
    idx = np.array([], dtype=np.uint32)
    val = np.array([], dtype=np.uint32)
    c, frame = _roundtrip_frame(codec_name, idt, vdt, idx, val)
    out_i = np.empty(0, dtype=np.int32)
    out_v = np.empty(0, dtype=np.float32)
    c.new_decoder(_N_VARS).decode_into(frame, 0, out_i, out_v)


@pytest.mark.parametrize("codec_name,idt,vdt", CODECS)
def test_decode_into_int64_output_indices(codec_name, idt, vdt):
    """gather_rows picks int64 indices once maxval/total exceed INT32_MAX; decode_into must
    fill that width identically. Every other decode_into test uses int32. (codex checkpoint-2.)"""
    idx = np.array([0, 3, 7, 11, 40000], dtype=np.uint32)
    val = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    c, frame = _roundtrip_frame(codec_name, idt, vdt, idx, val)
    ref_i, _ = c.new_decoder(_N_VARS).decode(frame, idx.size)
    out_i = np.empty(idx.size, dtype=np.int64)
    out_v = np.empty(idx.size, dtype=np.float32)
    c.new_decoder(_N_VARS).decode_into(frame, idx.size, out_i, out_v)
    np.testing.assert_array_equal(out_i, ref_i.astype(np.int64))
    assert out_i.dtype == np.int64


@pytest.mark.parametrize("codec_name,idt,vdt", CODECS)
def test_decode_into_row_larger_than_initial_scratch(codec_name, idt, vdt):
    """_PforDecoder's scratch starts at 4096; every other test stays under it, so _grow was
    never exercised. Decode a row well past the boundary, then a SMALLER row through the SAME
    decoder to confirm the grown buffers are still used correctly. (codex checkpoint-2.)"""
    n = 9000
    idx = np.arange(n, dtype=np.uint32) * 3
    val = (np.arange(n, dtype=np.uint32) % 4000) + 1
    c = codec_for(codec_name, idt, vdt)
    enc = c.new_encoder()
    big = enc.encode(idx, val)
    small = enc.encode(idx[:10], val[:10])
    dec = c.new_decoder(_N_VARS)

    ref_i, ref_v = c.new_decoder(_N_VARS).decode(big, n)
    out_i = np.empty(n, dtype=np.int32)
    out_v = np.empty(n, dtype=np.float32)
    dec.decode_into(big, n, out_i, out_v)
    np.testing.assert_array_equal(out_i, ref_i.astype(np.int32))
    np.testing.assert_array_equal(out_v, ref_v.astype(np.float32))

    # reuse the SAME (now grown) decoder on a short row
    ref2_i, ref2_v = c.new_decoder(_N_VARS).decode(small, 10)
    o2_i = np.empty(10, dtype=np.int32)
    o2_v = np.empty(10, dtype=np.float32)
    dec.decode_into(small, 10, o2_i, o2_v)
    np.testing.assert_array_equal(o2_i, ref2_i.astype(np.int32))
    np.testing.assert_array_equal(o2_v, ref2_v.astype(np.float32))


def test_decode_into_writes_only_its_slice():
    idx = np.array([1, 2], dtype=np.uint32)
    val = np.array([9, 8], dtype=np.uint32)
    c, frame = _roundtrip_frame("pfordelta", "uint32", "uint32", idx, val)
    buf_i = np.full(6, -1, dtype=np.int32)
    buf_v = np.full(6, -1.0, dtype=np.float32)
    c.new_decoder(_N_VARS).decode_into(frame, 2, buf_i[2:4], buf_v[2:4])
    np.testing.assert_array_equal(buf_i, [-1, -1, 1, 2, -1, -1])
    np.testing.assert_array_equal(buf_v, [-1, -1, 9, 8, -1, -1])


def test_decode_into_reuses_buffers_across_rows():
    """The pfordelta scratch is reused; the cumsum must be copied out BEFORE the value
    _unpack overwrites it."""
    c = codec_for("pfordelta", "uint32", "uint32")
    enc = c.new_encoder()
    frames = [
        enc.encode(np.array([0, 5], dtype=np.uint32), np.array([7, 7], dtype=np.uint32)),
        enc.encode(np.array([2, 9], dtype=np.uint32), np.array([1, 4], dtype=np.uint32)),
    ]
    dec = c.new_decoder(_N_VARS)
    out_i = np.empty(4, dtype=np.int32)
    out_v = np.empty(4, dtype=np.float32)
    dec.decode_into(frames[0], 2, out_i[0:2], out_v[0:2])
    dec.decode_into(frames[1], 2, out_i[2:4], out_v[2:4])
    np.testing.assert_array_equal(out_i, [0, 5, 2, 9])
    np.testing.assert_array_equal(out_v, [7, 7, 1, 4])


def test_decode_into_preserves_narrowing_truncation():
    """On-disk uint16 narrows: a value above 65535 must truncate EXACTLY as the old
    two-step (uint32 -> uint16 -> float32) path did, not pass through widened."""
    c = codec_for("pfordelta", "uint16", "uint16")
    frame = c.new_encoder().encode(
        np.array([0, 1], dtype=np.uint32), np.array([5, 70000], dtype=np.uint32)
    )
    ref_i, ref_v = c.new_decoder(_N_VARS).decode(frame, 2)
    expected_v = ref_v.astype(np.float32)
    out_i = np.empty(2, dtype=np.int32)
    out_v = np.empty(2, dtype=np.float32)
    c.new_decoder(_N_VARS).decode_into(frame, 2, out_i, out_v)
    np.testing.assert_array_equal(out_v, expected_v)
    # NOT np.uint16(70000): on numpy 2.x that raises OverflowError. Truncate via an array.
    assert out_v[1] == np.array([70000], dtype=np.uint32).astype(np.uint16)[0]
    assert out_v[1] == np.float32(4464)


@pytest.mark.parametrize("codec_name,idt,vdt", CODECS)
def test_decode_into_validates_corrupt_empty_frame(codec_name, idt, vdt):
    """decode() validates the framing even at nnz==0, so decode_into must too -- and the
    CALLER must not skip it for empty rows."""
    c = codec_for(codec_name, idt, vdt)
    out_i = np.empty(0, dtype=np.int32)
    out_v = np.empty(0, dtype=np.float32)
    with pytest.raises(ValueError, match="shorter than 4-byte length prefix"):
        c.new_decoder(_N_VARS).decode_into(b"\x00", 0, out_i, out_v)
    with pytest.raises(ValueError, match="idx blob length exceeds frame"):
        c.new_decoder(_N_VARS).decode_into(b"\xff\xff\xff\xff", 0, out_i, out_v)


def _reference_gather(store, row_ids):
    """The PRE-#244 concatenate-based fallback, verbatim, as the equivalence oracle."""
    import scipy.sparse as sp

    row_ids = np.atleast_1d(np.asarray(row_ids)).astype(np.int64, copy=False)
    dec = store._codec.new_decoder(store.n_vars)
    parts_i, parts_v = [], []
    counts = np.empty(row_ids.size, dtype=np.int64)
    for k, i in enumerate(row_ids):
        nnz_i = int(store.indptr[i + 1] - store.indptr[i])
        frame = store._mm[store.offsets[i] : store.offsets[i + 1]]
        di, dv = dec.decode(frame, nnz_i)
        parts_i.append(di.astype(np.int32))
        parts_v.append(dv)
        counts[k] = nnz_i
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    indices = np.concatenate(parts_i) if parts_i else np.empty(0, np.int32)
    data = (np.concatenate(parts_v) if parts_v else np.empty(0, dtype=store._x_out_dtype)).astype(
        store._x_out_dtype, copy=False
    )
    return sp.csr_matrix((data, indices, indptr), shape=(row_ids.size, store.n_vars))


# The five configurations the spec requires. Integer values alone only reach pfordelta uint16
# and zstd integer, so this deliberately varies n_vars (to move the on-disk INDEX dtype) and
# the value range/dtype (to move the on-disk VALUE dtype).
#   label            codec        n_vars   values
ARCHIVE_CONFIGS = [
    ("pfor_narrow", "pfordelta", 120, "small_int"),
    ("pfor_wide", "pfordelta", 70000, "big_int"),
    ("zstd_int", "zstd", 120, "small_int"),
    ("zstd_f32", "zstd", 120, "float32"),
    ("zstd_f64", "zstd", 120, "float64"),
]


def _build(tmp_path, label, codec, n_vars, values, n_obs=40, seed=0):
    import anndata as ad
    import scipy.sparse as sp

    import cellstream

    rng = np.random.default_rng(seed)
    mask = rng.random((n_obs, n_vars)) < (0.02 if n_vars > 1000 else 0.15)
    if values == "small_int":
        dense = (mask * rng.integers(1, 50, (n_obs, n_vars))).astype(np.float32)
    elif values == "big_int":
        dense = (mask * rng.integers(70000, 200000, (n_obs, n_vars))).astype(np.float32)
    elif values == "float32":
        dense = (mask * rng.random((n_obs, n_vars))).astype(np.float32)
    else:
        dense = (mask * rng.random((n_obs, n_vars))).astype(np.float64)
    dense[3, :] = 0  # an all-zero row
    a = ad.AnnData(X=sp.csr_matrix(dense))
    out = tmp_path / f"{label}.shad"
    cellstream.write_sharded(a, out, layout="cell", codec=codec, overwrite=True)
    return out, dense


def test_configs_cover_five_distinct_on_disk_combos(tmp_path):
    """Guard the coverage claim itself: if the writer's dtype rules differ from what these
    builders assume, this fails loudly instead of silently testing the same combo five times."""
    import cellstream

    seen = {}
    for label, codec, n_vars, values in ARCHIVE_CONFIGS:
        path, _ = _build(tmp_path, label, codec, n_vars, values)
        store = cellstream.open(path)
        try:
            m = store.manifest
            seen[label] = (m["codec"], m["index_dtype_on_disk"], m["value_dtype_on_disk"])
        finally:
            store.close()
    assert len(set(seen.values())) == 5, f"configs collapsed onto fewer combos: {seen}"


ROW_SELECTIONS = [
    "all",
    "empty",
    "single",
    "reversed",
    "duplicated",
    "zero_row_only",
]


def _rows(kind, n_obs):
    if kind == "all":
        return np.arange(n_obs, dtype=np.int64)
    if kind == "empty":
        return np.empty(0, dtype=np.int64)
    if kind == "single":
        return np.array([7], dtype=np.int64)
    if kind == "reversed":
        return np.arange(n_obs - 1, -1, -1, dtype=np.int64)
    if kind == "duplicated":
        return np.array([2, 2, 5, 2], dtype=np.int64)
    return np.array([3], dtype=np.int64)  # the all-zero row


def _assert_csr_identical(got, want, ctx):
    np.testing.assert_array_equal(got.data, want.data, err_msg=f"{ctx}: data")
    np.testing.assert_array_equal(got.indices, want.indices, err_msg=f"{ctx}: indices")
    np.testing.assert_array_equal(got.indptr, want.indptr, err_msg=f"{ctx}: indptr")
    assert got.data.dtype == want.data.dtype, f"{ctx}: data dtype"
    assert got.indices.dtype == want.indices.dtype, f"{ctx}: indices dtype"
    assert got.indptr.dtype == want.indptr.dtype, f"{ctx}: indptr dtype"
    assert got.shape == want.shape, f"{ctx}: shape"
    # The CACHED canonical flag, read through scipy's private attribute so the comparison does
    # not itself trigger the lazy O(nnz) canonicalisation scan (that would set it on BOTH
    # matrices and make this check vacuous). gather must not pre-cache what the scipy ctor left
    # unset -- see test_gather_does_not_cache_canonical_flag. has_sorted_indices is deliberately
    # NOT compared: caching it is documented deviation (c). (codex checkpoint-2, CRITICAL.)
    # Default False, not None: on scipy 1.17.1 the attribute is ABSENT until something caches
    # it (verified with hasattr), and a future scipy could instead store False. With a None
    # default those two spellings of "not cached" would compare unequal and fail spuriously;
    # False collapses them while still catching a cached True. (Gemini #247 r2.)
    assert getattr(got, "_has_canonical_format", False) == getattr(
        want, "_has_canonical_format", False
    ), f"{ctx}: has_canonical_format cached differently"


@pytest.mark.parametrize("label,codec,n_vars,values", ARCHIVE_CONFIGS)
@pytest.mark.parametrize("kind", ROW_SELECTIONS)
def test_fallback_matches_reference(tmp_path, monkeypatch, label, codec, n_vars, values, kind):
    import cellstream

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    path, _ = _build(tmp_path, label, codec, n_vars, values)
    store = cellstream.open(path)
    try:
        rows = _rows(kind, store.n_obs)
        got = store.gather_rows(rows)
        want = _reference_gather(store, rows)
        _assert_csr_identical(got, want, f"{label}/{kind}")
    finally:
        store.close()


@pytest.mark.parametrize("label,codec,n_vars,values", ARCHIVE_CONFIGS)
@pytest.mark.parametrize("kind", ROW_SELECTIONS)
def test_fallback_matches_rust(tmp_path, monkeypatch, label, codec, n_vars, values, kind):
    import cellstream
    from cellstream.cell._rust_dispatch import rust_available

    # Skip BEFORE building: this is 30 parametrizations, and each _build writes a real archive.
    # Deciding after the write burned all 30 writes on a run that then skipped. (codex r2 NIT.)
    if not rust_available():
        pytest.skip("Rust extension unavailable")
    path, _ = _build(tmp_path, label, codec, n_vars, values)
    store = cellstream.open(path)
    try:
        rows = _rows(kind, store.n_obs)
        rust = store.gather_rows(rows)
    finally:
        store.close()
    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    store = cellstream.open(path)
    try:
        py = store.gather_rows(rows)
    finally:
        store.close()
    # Raw arrays AND dtypes, not just toarray() -- an indptr dtype drift is invisible to a
    # dense comparison and is exactly the regression the scipy ctor's recast would cause.
    _assert_csr_identical(py, rust, f"{label}/{kind} py-vs-rust")


def test_fallback_scalar_row_id_still_promotes(tmp_path, monkeypatch):
    import cellstream

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    path, dense = _build(tmp_path, "pfor_narrow", "pfordelta", 120, "small_int")
    store = cellstream.open(path)
    try:
        X = store.gather_rows(5)
        assert X.shape == (1, store.n_vars)
        np.testing.assert_array_equal(X.toarray()[0], dense[5])
    finally:
        store.close()


def test_fallback_rejects_out_of_range_column(tmp_path, monkeypatch):
    """NEW hardening, not preservation: the scipy triple-form ctor only runs
    check_format(full_check=False), which does NOT bounds-check columns -- verified, a column
    index of 99 in a 5-column matrix passes it silently. check_indices=True closes a gap where
    the Python path was less safe than Rust, which validates every column inline."""
    import cellstream

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    path, _ = _build(tmp_path, "pfor_narrow", "pfordelta", 120, "small_int")
    store = cellstream.open(path)
    try:
        store.n_vars = 2  # pretend the archive claims far fewer columns
        with pytest.raises(ValueError, match="column index out of range"):
            store.gather_rows(np.arange(store.n_obs, dtype=np.int64))
    finally:
        store.close()


def test_fallback_still_decodes_empty_rows(tmp_path, monkeypatch):
    """Row 3 is all-zero. The resulting CSR must have an empty row 3."""
    import cellstream

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    path, dense = _build(tmp_path, "pfor_narrow", "pfordelta", 120, "small_int")
    store = cellstream.open(path)
    try:
        X = store.gather_rows(np.array([2, 3, 4], dtype=np.int64))
        assert X.indptr[2] == X.indptr[1]  # row 3 contributes no entries
        np.testing.assert_array_equal(X.toarray()[1], 0)
    finally:
        store.close()


def _corrupt_empty_row_frame(path, row):
    """Overwrite an EMPTY row's frame header with a bogus idx_len, in place and same-length.

    0xffffffff makes `4 + idx_len > len(frame)` true for any real frame, so both decoders
    raise "idx blob length exceeds frame". Same-length so no offset shifts.
    """
    import cellstream
    from cellstream.cell import format as fmt

    store = cellstream.open(path)
    try:
        assert int(store.indptr[row + 1] - store.indptr[row]) == 0, "row must be empty"
        p, base, _ = store._packed.member_location(fmt.PAYLOAD)
        off = int(store.offsets[row])
        span = int(store.offsets[row + 1]) - off
        assert span >= 4, f"empty frame too short to corrupt: {span}"
    finally:
        store.close()
    with open(p, "r+b") as fh:
        fh.seek(base + off)
        fh.write(b"\xff\xff\xff\xff")


@pytest.mark.parametrize(
    "label,codec,n_vars,values", [ARCHIVE_CONFIGS[0], ARCHIVE_CONFIGS[2]], ids=["pfor", "zstd"]
)
def test_corrupt_empty_frame_raises_in_gather(tmp_path, monkeypatch, label, codec, n_vars, values):
    """THE test that pins 'decode_into is called for every row'. A caller that skips empty rows
    passes test_fallback_still_decodes_empty_rows but fails here."""
    import cellstream

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    path, _ = _build(tmp_path, label, codec, n_vars, values)
    _corrupt_empty_row_frame(path, 3)
    store = cellstream.open(path)
    try:
        with pytest.raises(ValueError, match="idx blob length exceeds frame"):
            store.gather_rows(np.array([2, 3, 4], dtype=np.int64))
    finally:
        store.close()


@pytest.mark.parametrize(
    "label,codec,n_vars,values", [ARCHIVE_CONFIGS[0], ARCHIVE_CONFIGS[2]], ids=["pfor", "zstd"]
)
def test_corrupt_empty_frame_raises_in_bulk(tmp_path, monkeypatch, label, codec, n_vars, values):
    """Same guarantee for the bulk fallback worker. _dispatch_workers calls fut.result(), which
    re-raises the worker's original ValueError, and read_cell_bulk does not wrap it -- so assert
    the exact type. If nothing propagates, the worker is skipping empty rows and the fix belongs
    in _decode_cell_range_into_shm."""
    from cellstream.cell.reader import read_cell_bulk

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    path, _ = _build(tmp_path, label, codec, n_vars, values)
    _corrupt_empty_row_frame(path, 3)
    with pytest.raises(ValueError, match="idx blob length exceeds frame"):
        read_cell_bulk(path, n_workers=2)


def test_gather_rejects_n_vars_above_int32(tmp_path, monkeypatch):
    """The n_vars > INT32_MAX guard sits BEFORE the engine branch, so it must fire on both.

    An empty gather reaches the guard without allocating anything large, and doubles as scipy's
    `0 in shape` corner. The boundary value itself must stay allowed: at n_vars == _I32MAX the
    largest legal column is _I32MAX - 1, which still fits int32."""
    import cellstream
    from cellstream.cell._rust_dispatch import _HAVE
    from cellstream.cell.reader import _I32MAX

    path, _ = _build(tmp_path, "pfor_narrow", "pfordelta", 120, "small_int")
    empty = np.empty(0, dtype=np.int64)
    for engine in ("python", "rust"):
        if engine == "python":
            monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
        elif not _HAVE:
            # On a checkout with NO extension built, clearing CELLSTREAM_DISABLE_RUST lands the
            # boundary-value gather in check_cell_python_fallback's `extension_missing` arm,
            # which RAISES -- unrelated to the guard under test. The python iteration above
            # already covers the assertion. (codex checkpoint-2 review.)
            continue
        else:
            monkeypatch.delenv("CELLSTREAM_DISABLE_RUST", raising=False)
        store = cellstream.open(path)
        try:
            store.n_vars = _I32MAX + 1
            with pytest.raises(ValueError, match="does not support n_vars"):
                store.gather_rows(empty)
            store.n_vars = _I32MAX  # boundary stays allowed on both engines
            assert store.gather_rows(empty).shape == (0, _I32MAX)
        finally:
            store.close()


def test_gather_does_not_cache_canonical_flag(tmp_path, monkeypatch):
    """gather_rows must not cache scipy's has_canonical_format (codex checkpoint-2, CRITICAL).

    ``indices_canonical`` is a MANIFEST claim, and a tampered archive controls it. Caching
    ``has_canonical_format=True`` on that claim makes ``sum_duplicates()`` return early, so
    duplicate columns survive silently and every subsequent op sees a wrong matrix. The scipy
    constructor this rewrite replaced never cached it, so caching it would be a fourth
    corrupt-input deviation beyond the three the rewrite is allowed.
    """
    import cellstream

    monkeypatch.setenv("CELLSTREAM_DISABLE_RUST", "1")
    path, _ = _build(tmp_path, "pfor_narrow", "pfordelta", 120, "small_int")
    store = cellstream.open(path)
    try:
        # If the writer stopped setting this the test would pass vacuously -- assert it first.
        assert store._indices_canonical is True
        X = store.gather_rows(np.arange(store.n_obs, dtype=np.int64))
        # scipy internal: None == "not cached, scan lazily". Anything truthy here is the bug.
        # default False, matching _assert_csr_identical. Cosmetic HERE -- this is `not
        # getattr(...)` and both None and False are falsy, so the default cannot change the
        # outcome; it is load-bearing only at the comparison site above. One convention.
        assert not getattr(X, "_has_canonical_format", False)
        # ...while has_sorted_indices IS still cached -- documented deviation (c).
        assert getattr(X, "_has_sorted_indices", None) is True
    finally:
        store.close()


def test_cached_canonical_flag_would_hide_duplicates():
    """Pins WHY the flag matters, independently of any archive: with it cached, a CSR holding
    duplicate columns silently skips sum_duplicates(). Guards the reasoning in
    test_gather_does_not_cache_canonical_flag against a future scipy that ignores the flag."""
    import scipy.sparse as sp

    from cellstream.cell.reader import _csr_direct_assign

    def build(canonical):
        return _csr_direct_assign(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1, 1, 2], dtype=np.int32),  # column 1 twice in row 0
            np.array([0, 2, 3], dtype=np.int32),
            n_obs=2,
            n_vars=5,
            canonical=canonical,
        )

    cached = build(True)
    cached.sum_duplicates()
    assert cached.nnz == 3  # the no-op: duplicates survive

    honest = build(False)
    honest.sum_duplicates()
    assert honest.nnz == 2  # collapsed, matching the scipy ctor

    ctor = sp.csr_matrix(
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1, 1, 2], dtype=np.int32),
            np.array([0, 2, 3], dtype=np.int32),
        ),
        shape=(2, 5),
    )
    ctor.sum_duplicates()
    assert ctor.nnz == 2


@pytest.mark.parametrize("codec_name,idt,vdt", CODECS)
def test_decode_into_rejects_mis_sized_out_buffers(codec_name, idt, vdt):
    """np.copyto BROADCASTS: without a guard, decode_into(nnz=1) into a 5-long buffer silently
    fills all five slots instead of raising. Mismatches above 1 already raise inside numpy, so
    nnz==1 is the case that actually needs the explicit check. (Copilot #247 r4.)"""
    c = codec_for(codec_name, idt, vdt)
    frame = c.new_encoder().encode(np.array([4], dtype=np.uint32), np.array([9], dtype=np.uint32))
    big_i = np.full(5, -1, dtype=np.int32)
    big_v = np.full(5, -1.0, dtype=np.float32)
    with pytest.raises(ValueError, match="must be exactly nnz"):
        c.new_decoder(_N_VARS).decode_into(frame, 1, big_i, big_v)
    # the buffers must be untouched -- the guard runs before any write
    np.testing.assert_array_equal(big_i, -1)
    np.testing.assert_array_equal(big_v, -1.0)
    # and the correctly-sized call still works
    ok_i = np.empty(1, dtype=np.int32)
    ok_v = np.empty(1, dtype=np.float32)
    c.new_decoder(_N_VARS).decode_into(frame, 1, ok_i, ok_v)
    assert ok_i[0] == 4 and ok_v[0] == 9.0


@pytest.mark.parametrize("codec_name,idt,vdt", CODECS)
def test_decode_into_rejects_mis_sized_out_on_empty_row(codec_name, idt, vdt):
    """The guard also covers nnz==0, where a non-empty out buffer would be left untouched and
    the caller would silently read stale data."""
    c = codec_for(codec_name, idt, vdt)
    frame = c.new_encoder().encode(np.array([], dtype=np.uint32), np.array([], dtype=np.uint32))
    with pytest.raises(ValueError, match="must be exactly nnz"):
        c.new_decoder(_N_VARS).decode_into(
            frame, 0, np.empty(2, dtype=np.int32), np.empty(2, dtype=np.float32)
        )
