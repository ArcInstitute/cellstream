"""The public API is permanently packed-only; legacy directories fail with guidance."""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import cellstream
from cellstream import ShardedArchive, read_h5ad, read_obs, read_var, write_sharded
from cellstream.errors import LEGACY_V1_REMEDY, PACKED_ONLY_MIGRATION, PackedOnlyError
from tests._archive_helpers import write_directory

# errors.LEGACY_V1_REMEDY, verbatim and as a LITERAL. Error text is API here, so the expected
# value has to live in the test: `LEGACY_V1_REMEDY in message` alone only proves the sites carry
# whatever the constant currently says, so keeping the anchor and swapping the advice for
# `uv add`/`conda install` left every assertion green (codex). #311 also replaced an older anchor
# ("shardad.git@v0.4.0"); leaving THAT one in place would have made the negative arm vacuous,
# since it asserts absence and so passes for free once the string exists nowhere.
_EXPECTED_REMEDY = (
    "no installable release can read a legacy v1/directory archive -- the last one that could, "
    "v0.4.0, predates this project's rename and was never published to PyPI. Re-encode from the "
    "original source data with the current writer instead."
)

# errors.PACKED_ONLY_MIGRATION, verbatim: the remedy with the packed-container sentence in front.
_EXPECTED_MIGRATION = (
    " Since v0.5 only the single-file packed container is supported (SHPK head magic, "
    "conventionally .shad); " + _EXPECTED_REMEDY
)

# Substrings no legacy-archive message may contain, in any case. #311's URL was lowercase, but
# GitHub hosts and paths are case-insensitive, so a `"github.com" not in message` check would let
# `GitHub.com/...` straight through -- and so would the seed gate's arm if IT were case-sensitive
# (it is not; see `_NonPublicArcRepo` there). `githubusercontent.com` is the raw-content host, a
# second spelling of the same leak. (codex.)
_FORBIDDEN_IN_MESSAGE = ("github.com", "githubusercontent.com", "pip install", "git+")


def test_the_shared_remedy_text_is_unchanged():
    """The one assertion that pins the CONTENT rather than the plumbing (#311).

    Separate from the message arms on purpose: they prove each site carries the remedy, this
    proves the remedy still says what it is supposed to say. Change the wording and this is the
    test that fails, which is the signal you are editing user-facing API text.
    """
    assert LEGACY_V1_REMEDY == _EXPECTED_REMEDY
    assert PACKED_ONLY_MIGRATION == _EXPECTED_MIGRATION


def _assert_names_no_install(message: str) -> None:
    """The remedy must END the message, in full, with no reachable-looking install anywhere."""
    assert message.endswith(_EXPECTED_REMEDY), (
        "the message must carry the whole remedy verbatim, as its last words"
    )
    lowered = message.lower()
    for forbidden in _FORBIDDEN_IN_MESSAGE:
        assert forbidden not in lowered, f"{forbidden!r} is back in a legacy-archive message"


def _tiny():
    X = sp.random(40, 8, density=0.3, format="csr", dtype="float32")
    X.data = np.ceil(X.data * 10).astype("float32")
    return ad.AnnData(X=X)


def _packed(tmp_path):
    out = tmp_path / "p.shardpack"
    write_sharded(_tiny(), out)
    return out


def _directory(tmp_path, name="dir"):
    d = tmp_path / name
    write_directory(_tiny(), d)
    return d


@pytest.mark.parametrize("fmt", ["v1", "v3", "", "V2"])
def test_only_format_v2_is_accepted(tmp_path, fmt):
    """The v1 writer is gone; ``format`` now accepts exactly ``"v2"`` (#154)."""
    with pytest.raises(ValueError, match="format") as e:
        write_sharded(_tiny(), tmp_path / f"a_{fmt or 'empty'}", format=fmt)
    assert not isinstance(e.value, PackedOnlyError), (
        "a bad format must be an ordinary invalid-argument ValueError now, not the "
        "packed-only archive error"
    )


def test_format_v2_is_accepted(tmp_path):
    """Control for the invalid-format cases: the supported format must still write."""
    out = tmp_path / "ok"
    write_sharded(_tiny(), out, format="v2")
    assert out.is_file()


def test_blosc_nthreads_is_not_a_write_parameter(tmp_path):
    """The removed v1-only write knob is a plain Python signature error."""
    with pytest.raises(TypeError, match="blosc_nthreads"):
        write_sharded(_tiny(), tmp_path / "b1", blosc_nthreads=4)


def test_read_side_blosc_nthreads_is_still_accepted(tmp_path):
    """The distinct, live read-side knob remains accepted."""
    p = _packed(tmp_path)
    assert ShardedArchive(p).to_anndata(blosc_nthreads=2).n_obs == 40
    assert read_h5ad(p, blosc_nthreads=2).n_obs == 40


def test_write_directory_raises(tmp_path):
    with pytest.raises(TypeError, match="container"):
        write_sharded(_tiny(), tmp_path / "b", container="directory")


def test_write_packed_ok(tmp_path):
    out = tmp_path / "c.shardpack"
    write_sharded(_tiny(), out)
    assert out.is_file()


def test_read_packed_ok(tmp_path):
    p = _packed(tmp_path)
    a = ShardedArchive(p).to_anndata()
    assert a.n_obs == 40


def test_read_obs_var_packed_ok(tmp_path):
    p = _packed(tmp_path)
    assert len(read_obs(p)) == 40
    assert len(read_var(p)) == 8


@pytest.mark.parametrize("entry", ["shardedarchive", "read_h5ad", "read_obs", "read_var"])
def test_a_missing_path_raises_FileNotFoundError_not_the_migration_note(tmp_path, entry):
    """A typo is not a legacy archive, and must not be answered with migration advice.

    NOT a B3b regression pin: with enforcement ON (i.e. in production) the pre-B3b
    constructor also raised PackedOnlyError here, because guard_packed_only fired before
    load_manifest could raise FileNotFoundError -- the FileNotFoundError was reachable only
    with the escape hatch off, which is to say only inside this test suite. So this fixes a
    v0.5-era behaviour rather than restoring one. ``read_h5ad`` is the control: it has always
    checked ``p.exists()`` first, and is what the other three now agree with.

    The anndata hook is excluded on purpose: for a path that does not exist it never
    recognizes an archive and falls through to stock anndata, whose error is anndata's to
    define.
    """
    missing = tmp_path / "definitely_not_here.shad"
    calls = {
        "shardedarchive": lambda: ShardedArchive(missing),
        "read_h5ad": lambda: read_h5ad(missing),
        "read_obs": lambda: read_obs(missing),
        "read_var": lambda: read_var(missing),
    }
    with pytest.raises(FileNotFoundError) as exc_info:
        calls[entry]()
    # The distinguishing assertion: FileNotFoundError is not a CellstreamError, so a leftover
    # PackedOnlyError could not satisfy pytest.raises above -- but a future change could make
    # a subclass do both, and the point is that the caller is NOT handed the legacy-archive
    # migration guidance. Asserted as the WHOLE message rather than the absence of a fragment,
    # because absence-only passes for any guidance that happens to be worded differently: the
    # message must say where the file is and nothing else. (codex.)
    expected = str(missing) if entry == "read_h5ad" else f"no such archive: {missing}"
    assert str(exc_info.value) == expected


@pytest.mark.parametrize(
    "entry",
    ["shardedarchive", "read_h5ad", "read_obs", "read_var", "anndata_hook"],
)
def test_public_read_entry_points_reject_a_directory(tmp_path, entry):
    d = _directory(tmp_path)
    if entry == "anndata_hook":
        # Registration is best-effort and opt-out-able (CELLSTREAM_NO_MONKEY_PATCH=1), so this
        # arm SKIPS rather than asserts -- an unconditional assert failed the suite under a
        # supported configuration. Every other backend test skips the same way.
        if not cellstream.anndata_backend_registered:
            pytest.skip("anndata monkey-patch not registered (opt-out or unavailable)")
        assert ad.read_h5ad._cellstream_wrapped
    calls = {
        "shardedarchive": lambda: ShardedArchive(d),
        "read_h5ad": lambda: read_h5ad(d),
        "read_obs": lambda: read_obs(d),
        "read_var": lambda: read_var(d),
        "anndata_hook": lambda: ad.read_h5ad(d),
    }
    fragments = {
        "shardedarchive": "ShardedArchive requires",
        "read_h5ad": "is a directory",
        "read_obs": "read_obs on a non-packed archive",
        "read_var": "read_var on a non-packed archive",
    }

    with pytest.raises(PackedOnlyError) as exc_info:
        calls[entry]()
    message = str(exc_info.value)
    _assert_names_no_install(message)
    fragment = fragments.get(entry)
    if fragment is not None:
        assert fragment in message


def test_no_legacy_archive_message_names_an_inaccessible_install(tmp_path):
    """#311: neither legacy-archive message may send a reader to a URL that 404s for them.

    Both SHIPPED sites are exercised rather than the constants they share: the
    ``PackedOnlyError`` from the packed-only reject path, and the ``ValueError`` from
    ``write_sharded``'s ``format`` guard. The remedy's own wording is pinned separately, by
    ``test_the_shared_remedy_text_is_unchanged`` -- asserting ``LEGACY_V1_REMEDY in message``
    here would only prove the site carries whatever that constant currently says.

    The seed gate's ``non-public ArcInstitute repo url`` SANITIZE arm (#312) covers the same text,
    but only when someone seeds the public repo, whereas this runs on every push. The
    ``endswith(_EXPECTED_REMEDY)`` assertion is what stops the absence checks going vacuous:
    without it they would all pass on a message that had lost the guidance entirely.
    """
    with pytest.raises(PackedOnlyError) as packed:
        ShardedArchive(_directory(tmp_path))
    with pytest.raises(ValueError) as bad_format:
        write_sharded(_tiny(), tmp_path / "v1.shad", format="v1")

    # EXACT, not `endswith`. A suffix pin stays green for `"Run uv add shardad==0.4.0. " +
    # LEGACY_V1_REMEDY`: the message still ends with the remedy and still contains none of the
    # forbidden substrings, so an unreachable instruction can be smuggled in FRONT of it. Error
    # text is API here, so the whole of both shipped messages is pinned. (codex.)
    assert str(packed.value) == (
        "ShardedArchive requires a single-file packed archive." + _EXPECTED_MIGRATION
    )
    assert str(bad_format.value) == (
        "format must be 'v2', got 'v1'. The legacy v1 (h5ad-shard) format was removed in #154 "
        "and has been unsupported since v0.5; " + _EXPECTED_REMEDY
    )
    for message in (str(packed.value), str(bad_format.value)):
        _assert_names_no_install(message)
