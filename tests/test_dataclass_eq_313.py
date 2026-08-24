# tests/test_dataclass_eq_313.py
"""#318 -- dataclasses whose compared fields hold arrays must still return a bool.

WHAT MAKES THESE TESTS NON-VACUOUS ON 3.11 AND 3.12. The defect only reproduces on CPython
3.13+, so `test_equal_instances_compare_true` is a tautology on the older interpreters -- the
GENERATED `__eq__` passes it there too. That is exactly the "green that tested nothing" shape
this repo keeps getting burned by, so every behavioural test below is paired with a STRUCTURAL
one: `test_eq_was_installed_by_the_decorator` fails on every version the moment a class loses
`@field_tuple_eq`. Two of the three CI jobs would otherwise police nothing here.

The structural arm asserts a PROJECT-OWNED marker, not `__eq__.__code__.co_filename !=
"<string>"`. That would work on 3.11 and 3.13 today, but it reads a CPython code-generation
detail no version promises: an interpreter that compiled generated methods with a real filename
would make the test silently stop detecting anything. (codex round 1.)

Negative control, run by restoring the unfixed sources from the base commit: see the PR.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellstream._dataclass_eq import EQ_MARKER, field_tuple_eq
from cellstream.grouping import ShardPlan
from cellstream.matrices import MatrixSpec
from cellstream.read import ShardCSRDLPack
from cellstream.shm import MmapBundle
from cellstream.v2.writer import _GroupedMeta

# Every dataclass in the package whose compared fields can hold an array-like. #318 named the
# first three; `MatrixSpec` was fixed earlier and moved onto the same decorator here; `MmapBundle`
# was missed by the issue and found while reviewing this branch.
ARRAY_DATACLASSES = [ShardCSRDLPack, ShardPlan, _GroupedMeta, MatrixSpec, MmapBundle]

# Per class: a non-array compared field to perturb, and one array-valued compared field whose
# object-sharing the perturbation test asserts. Keeping both here means a field rename breaks the
# test loudly instead of quietly weakening it.
_SCALAR_FIELD = {
    ShardCSRDLPack: ("shape", (9, 9)),
    ShardPlan: ("reference_shard", 7),
    _GroupedMeta: ("n_vars", 99),
    MatrixSpec: ("n_vars", 99),
    MmapBundle: ("dir", Path("/nowhere")),
}
_ARRAY_FIELD = {
    ShardCSRDLPack: "data",
    ShardPlan: "permutation",
    _GroupedMeta: "indptr",
    MatrixSpec: "X",
    MmapBundle: "data",
}


def _pair(cls):
    """Two DISTINCT instances of `cls` whose compared fields hold the SAME objects.

    EVERY compared field is shared, dicts included -- not just the array-valued ones. Sharing is
    the whole point: that is the case `PyObject_RichCompareBool`'s identity shortcut used to
    settle without ever calling the array's `__eq__`, and the case 3.13's pairwise comparison
    broke. Two *different* arrays holding equal values are a separate question, pinned by
    `test_distinct_multi_element_arrays_still_raise`.
    """
    if cls is ShardCSRDLPack:
        d, i, p, shape = np.arange(4), np.arange(4), np.array([0, 2, 4]), (2, 3)
        return cls(d, i, p, shape), cls(d, i, p, shape)
    if cls is ShardPlan:
        perm, offs, groups = np.arange(4), [0, 2, 4], [{"label": "g", "shard": 0}]
        return cls(perm, offs, groups, None), cls(perm, offs, groups, None)
    if cls is _GroupedMeta:
        obs, var = pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [3]})
        uns, obsm, varm, obsp, varp = {}, {}, {}, {}, {}
        ip = np.array([0, 1, 2])
        args = (obs, var, uns, obsm, varm, obsp, varp, ip, 2, 1)
        return cls(*args), cls(*args)
    if cls is MatrixSpec:
        m, v = sp.random(3, 3, density=0.5, format="csr"), pd.DataFrame({"g": [1, 2, 3]})
        return cls("k", "x", None, 3, m, v), cls("k", "x", None, 3, m, v)
    if cls is MmapBundle:
        # np.memmap does not override __eq__ -- `np.memmap.__eq__ is np.ndarray.__eq__` is True,
        # asserted in test_memmap_equality_is_ndarray_equality -- so a plain ndarray exercises
        # this class's comparison identically and costs no temp files.
        dd, d, i, p = Path("/tmp/does-not-need-to-exist"), np.arange(4), np.arange(4), np.arange(3)
        return cls(dd, d, i, p), cls(dd, d, i, p)
    raise AssertionError(f"no pair builder for {cls!r}")


@pytest.mark.parametrize("cls", ARRAY_DATACLASSES, ids=lambda c: c.__name__)
def test_equal_instances_compare_true(cls):
    """Real on 3.13, tautological on 3.11/3.12 -- paired with the structural test below."""
    a, b = _pair(cls)
    result = a == b
    assert type(result) is bool, f"{cls.__name__}.__eq__ returned {type(result).__name__}"
    assert result is True


@pytest.mark.parametrize("cls", ARRAY_DATACLASSES, ids=lambda c: c.__name__)
def test_eq_was_installed_by_the_decorator(cls):
    """THE version-independent arm: the class's OWN `__eq__` is the decorator's.

    The marker is read off the function in `cls.__dict__["__eq__"]`, not off the class. A class
    attribute would survive someone reassigning `C.__eq__` afterwards, so this test would keep
    passing while the method it vouches for was gone. (codex round 2.)
    """
    eq = cls.__dict__.get("__eq__")
    assert eq is not None, f"{cls.__name__} has no __eq__ of its own"
    names = getattr(eq, EQ_MARKER, None)
    assert names is not None, (
        f"{cls.__name__} lost @field_tuple_eq and is back on the dataclass-generated __eq__, "
        f"which does not return a bool for its array-valued fields on CPython 3.13 (#318)"
    )
    assert names == tuple(f.name for f in dataclasses.fields(cls) if f.compare)


@pytest.mark.parametrize("cls", ARRAY_DATACLASSES, ids=lambda c: c.__name__)
def test_unequal_scalar_field_compares_false(cls):
    """A differing NON-array field must still give a plain `False`, not raise.

    The array-sharing assertion is the load-bearing line, not decoration. Without it this test
    can pass for the wrong reason: for `MatrixSpec` the perturbed `n_vars` is compared BEFORE
    `X`, so the comparison returns False before it ever reaches an array and would pass even
    with unshared arrays. Asserting the sharing directly makes every class's arm mean the same
    thing regardless of field order.
    """
    a, b = _pair(cls)
    name, value = _SCALAR_FIELD[cls]
    array_field = _ARRAY_FIELD[cls]
    assert getattr(a, array_field) is getattr(b, array_field), "_pair stopped sharing objects"

    object.__setattr__(b, name, value)
    result = a == b
    assert type(result) is bool
    assert result is False
    assert (a != b) is True


@pytest.mark.parametrize("cls", ARRAY_DATACLASSES, ids=lambda c: c.__name__)
def test_foreign_type_returns_notimplemented(cls):
    """Our side must return `NotImplemented` -- asserted directly, because the `a == "x"` form
    only shows Python's *fallback* answer and would also pass for an `__eq__` that returned
    False outright."""
    a, _ = _pair(cls)
    assert cls.__eq__(a, "not a dataclass") is NotImplemented
    assert cls.__eq__(a, 17) is NotImplemented
    assert (a == "not a dataclass") is False
    assert (a == 17) is False


@pytest.mark.parametrize("cls", ARRAY_DATACLASSES, ids=lambda c: c.__name__)
def test_hash_method_generation_is_unchanged(cls):
    """`@field_tuple_eq` replaces `__eq__` AFTER `@dataclass` has run, so it cannot disturb the
    generated `__hash__`: frozen `eq=True` classes keep theirs, non-frozen `eq=True` classes keep
    `None`. Measured identical before and after the fix on 3.11.13 and 3.13.4.

    This asserts METHOD GENERATION, not that instances hash: hashing a real instance of the
    frozen three still raises, because ndarray / sparse / DataFrame are themselves unhashable.
    """
    frozen = cls.__dataclass_params__.frozen
    assert (cls.__hash__ is not None) is frozen


def test_memmap_equality_is_ndarray_equality():
    """Pins the substitution `_pair` makes for `MmapBundle`: measured, not assumed."""
    assert np.memmap.__eq__ is np.ndarray.__eq__


# scipy warns that `==` on two sparse matrices is inefficient. Here that comparison is the point.
@pytest.mark.filterwarnings("ignore:Comparing two sparse matrices:")
@pytest.mark.parametrize("cls", ARRAY_DATACLASSES, ids=lambda c: c.__name__)
def test_distinct_multi_element_arrays_still_raise(cls):
    """SCOPE GUARD, not a wish. The fix restores 3.11/3.12 behaviour and nothing more: the tuple
    comparison short-circuits on IDENTITY only, so two EQUAL BUT DISTINCT array-likes still reach
    the array's own `__eq__` and, being multi-element, still raise. That was true before 3.13 and
    is true after this fix. If this goes red someone widened the fix -- read the scope note in
    `cellstream._dataclass_eq` before deciding that is an improvement.
    """
    a, _ = _pair(cls)
    name = _ARRAY_FIELD[cls]
    # EQUAL BUT DISTINCT, precisely: one field replaced by a copy of itself, every other field
    # still shared. A second `_pair()` build would not do -- MatrixSpec's `X` is a fresh
    # `sp.random` each call, so its arm would have raised over an UNEQUAL matrix and proved
    # something weaker than the docstring claims. (codex round 2.)
    b = dataclasses.replace(a, **{name: getattr(a, name).copy()})
    assert getattr(a, name) is not getattr(b, name)
    with pytest.raises(ValueError, match="[Tt]ruth value"):
        a == b  # noqa: B015 -- the comparison itself is the assertion


def test_distinct_one_element_arrays_do_not_raise():
    """The other half of the scope, and the reason the test above says MULTI-ELEMENT.

    A one-element comparison result has an unambiguous truth value, so distinct one-element
    numpy arrays compare cleanly and answer True. Stating the rule as "distinct arrays always
    raise" would be wrong. It is numpy-shaped, not universal: a 1x1 sparse matrix also compares
    cleanly, while a 1x1 pandas DataFrame still raises -- all three measured. (codex rounds 1-2.)
    """

    @field_tuple_eq
    @dataclass(frozen=True)
    class Tiny:
        a: object
        b: int

    result = Tiny(np.array([1]), 0) == Tiny(np.array([1]), 0)
    assert type(result) is bool
    assert result is True


def test_subclass_sees_only_the_base_compared_fields():
    """The decorator binds the DEFINING class's fields once, which is what makes "identical to
    3.11" literally true.

    A subclass declared `eq=False` inherits this `__eq__`. The 3.11-generated base method
    compared only the base's fields, so it answered True here. Discovering fields per call from
    `fields(obj)` -- the first version of this helper -- answered False instead, because it saw
    the subclass's extra field. (codex round 1.)
    """

    @dataclass(frozen=True, eq=False)
    class Child(ShardPlan):
        extra: int = 0

    perm, offs, groups = np.arange(4), [0, 2, 4], [{"label": "g"}]
    a = Child(perm, offs, groups, None, extra=1)
    b = Child(perm, offs, groups, None, extra=2)
    assert (a == b) is True
    assert "__eq__" not in Child.__dict__, (
        "Child must INHERIT the base __eq__, not define its own, for this to mean anything"
    )


# ------------------------------------------------------- the decorator's own two edges
#
# Neither is reachable through the five real classes -- all have >= 2 compared fields and none
# sets compare=False. Without these the case for not spelling the helper `operator.attrgetter`
# would be an untested claim.


def test_single_compared_field_still_compares_a_tuple():
    """`operator.attrgetter("x")` returns the VALUE; the decorator must build a 1-tuple.

    Otherwise a class trimmed to a single array-valued compared field would go straight back to
    comparing that array directly -- #318 reintroduced by a refactor that looks unrelated. The
    proof is behavioural: under the attrgetter spelling this comparison would RETURN an ndarray
    (and the `type(result) is bool` assertion below would fail); here it returns True.
    """

    @field_tuple_eq
    @dataclass(frozen=True)
    class OneField:
        only: object

    assert getattr(OneField.__dict__["__eq__"], EQ_MARKER) == ("only",)
    arr = np.arange(4)
    result = OneField(arr) == OneField(arr)
    assert type(result) is bool
    assert result is True


def test_zero_compared_fields_still_compares_a_tuple():
    """The third arm of the getter dispatch, which no real class reaches.

    `operator.attrgetter()` with no names raises, so this edge has its own spelling -- and an
    arm with no synthetic case is an arm that can quietly stop working. Every instance being
    equal is what the generated `__eq__` does too (`() == ()`).
    """

    @field_tuple_eq
    @dataclass
    class Empty:
        ignored: object = field(default=None, compare=False)

    assert getattr(Empty.__dict__["__eq__"], EQ_MARKER) == ()
    result = Empty(np.arange(4)) == Empty(np.arange(9))
    assert type(result) is bool
    assert result is True


def test_compare_false_field_is_ignored():
    """A `field(compare=False)` must stay out, the way the generated `__eq__` keeps it out."""

    @field_tuple_eq
    @dataclass
    class Mixed:
        counted: int
        ignored: object = field(default=None, compare=False)

    assert getattr(Mixed.__dict__["__eq__"], EQ_MARKER) == ("counted",)
    # An array in the ignored field would raise if it were compared; it must not be.
    assert (Mixed(1, np.arange(4)) == Mixed(1, np.arange(4))) is True
    assert (Mixed(1) == Mixed(2)) is False


def test_eq_false_dataclass_is_refused():
    """An ordinary `@dataclass(eq=False)` keeps identity hashing, so value equality on top would
    leave equal instances hashing differently -- measured on 3.11.13 and 3.13.4.

    The refusal is deliberately CONSERVATIVE, not precise: `eq=False, unsafe_hash=True` and
    `eq=False` with an explicit `__hash__ = None` are both safe and are refused anyway. That is
    the documented scope, not an oversight. (codex rounds 2-3.)
    """

    with pytest.raises(TypeError, match="eq=True"):

        @field_tuple_eq
        @dataclass(eq=False)
        class NoEq:
            v: int


def test_non_dataclass_is_refused():
    """A plain class has no compared fields to bind."""

    with pytest.raises(TypeError, match="__dataclass_params__"):

        @field_tuple_eq
        class Plain:
            pass


def test_subclass_with_only_inherited_params_is_refused():
    """THE guard that matters, and the first version got it wrong by using `getattr`.

    A subclass INHERITS `__dataclass_params__` while `@dataclass` never settled its own equality
    or hash state, so `is_dataclass()` says True and `getattr` waves it through. Measured on both
    interpreters.

    Two shapes, because they cost different amounts and the first does NOT show the hazard: a
    plain subclass of a default `eq=True` dataclass inherits `__hash__ = None` and is simply
    unhashable, so decorating it would be wrong but harmless. The damaging one has identity
    semantics back -- there value equality would sit next to `object.__hash__` and equal
    instances would hash differently. An earlier version of this test claimed the hazard while
    only building the harmless shape. (codex rounds 3-4.)
    """

    @dataclass
    class Base:
        x: int

    class PlainChild(Base):
        pass

    assert dataclasses.is_dataclass(PlainChild), "the trap is that this is True"
    assert "__dataclass_params__" not in PlainChild.__dict__
    assert PlainChild.__hash__ is None, "harmless shape: unhashable, not identity-hashing"

    with pytest.raises(TypeError, match="__dataclass_params__"):
        field_tuple_eq(PlainChild)

    class Identity:
        __eq__ = object.__eq__
        __hash__ = object.__hash__

    class IdentityChild(Identity, Base):
        pass

    assert dataclasses.is_dataclass(IdentityChild)
    assert "__dataclass_params__" not in IdentityChild.__dict__
    assert IdentityChild.__hash__ is object.__hash__, "the damaging shape"
    assert (IdentityChild(1) == IdentityChild(1)) is False, "identity equality, before decorating"

    with pytest.raises(TypeError, match="__dataclass_params__"):
        field_tuple_eq(IdentityChild)

    # ... while a subclass @dataclass actually ran on is accepted. Not frozen: a frozen
    # dataclass cannot inherit from a non-frozen one, which is a dataclasses rule, not ours.
    @field_tuple_eq
    @dataclass
    class ProperChild(Base):
        y: int = 0

    assert getattr(ProperChild.__dict__["__eq__"], EQ_MARKER) == ("x", "y")


def test_installed_eq_wears_the_owning_module():
    """`__module__` must name the owning class, not this helper: pickling the UNBOUND
    `C.__eq__` resolves it by module + qualname and fails otherwise. (codex round 2.)"""
    import pickle

    for cls in ARRAY_DATACLASSES:
        eq = cls.__dict__["__eq__"]
        assert eq.__module__ == cls.__module__
        assert eq.__qualname__ == f"{cls.__qualname__}.__eq__"
        assert pickle.loads(pickle.dumps(cls.__eq__)) is cls.__eq__
