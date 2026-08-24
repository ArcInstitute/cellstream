"""One field-tuple ``__eq__`` for the dataclasses whose compared fields hold arrays (#318).

WHAT 3.13 CHANGED. CPython 3.13 generates a PAIRWISE field comparison for a dataclass ``__eq__``
instead of building and comparing the two field tuples: the body became, in effect,
``self.a == other.a and self.b == other.b and ...``. Of the ways the two spellings differ, TWO
decide the comparison's result, and both matter here:

* an ``and`` chain TRUTH-TESTS each SYNTACTICALLY NON-FINAL result it reaches. An ambiguous
  result raises there -- "The truth value of an array with more than one element is ambiguous" --
  and a falsey one short-circuits, returning that result OBJECT unchanged. If every non-final
  result is truthy, the final one is returned unchanged too. Either way the chain hands back
  whatever the deciding field's ``__eq__`` returned, bool or not.
* tuple comparison calls ``PyObject_RichCompareBool`` for each pair it reaches. That tries
  IDENTITY FIRST, so a field holding the *same* object on both sides never reaches that object's
  ``__eq__`` at all; failing that it truth-tests, raising on an ambiguous result exactly as the
  chain does. The overall answer is always a bool.

So 3.13 lost the per-field identity shortcut AND the guarantee that the returned value is a bool.
(A third difference is not about the result: building a tuple reads every compared field eagerly,
while the chain reads them pairwise and may stop early. Nothing in #318 turns on it.)

The two losses show up separately. DISTINCT one-element arrays isolate the lost BOOL, since
identity cannot be the explanation there::

    Two(np.array([1]), np.array([1])) == Two(np.array([1]), np.array([1]))
      3.11.13 -> True (bool)            3.13.4 -> array([ True])   (ndarray)
    Two(np.array([1]), np.array([9])) == Two(np.array([2]), np.array([9]))
      3.11.13 -> False (bool)           3.13.4 -> array([False])   (ndarray, short-circuited)

Equal, SHARED field objects isolate the lost SHORTCUT::

    ShardCSRDLPack(d, i, p, shape)         3.11 -> True    3.13 -> ValueError (array, non-final)
    _GroupedMeta(obs, var, ..., ip, 2, 1)  3.11 -> True    3.13 -> ValueError (DataFrame, non-final)

⚠️ Two categorical statements that look right and are not. Issue #318 says the sparse-``X``
``MatrixSpec`` case RETURNS a 3x3 bool matrix on 3.13; it RAISES, because ``X`` is penultimate
and truthy-testing a multi-element sparse comparison is ambiguous. And an earlier draft of this
module said only a FINAL array-like can return a non-bool; a falsey non-final one is returned
unchanged too, as the second measurement above shows. Which symptom a class exhibits depends on
field order and on the values, so do not reason from the symptom back to the cause.

SCOPE, stated exactly, because it is narrower than "array dataclasses now compare safely".
Comparing the tuples restores behaviour identical to 3.11/3.12 for instances of the decorated
class, and nothing more. Two DIFFERENT objects are still compared by the field's own ``__eq__``,
so a multi-element numpy array or a pandas DataFrame still raises there, exactly as it did before
3.13. It is not even a clean "arrays raise" rule: a ONE-element numpy array and a 1x1 sparse
matrix both compare cleanly, because a one-element result has an unambiguous truth value, while
a 1x1 DataFrame still raises. All three measured. This module makes 3.13 behave like its
predecessors; it does not make array-valued dataclasses safe to compare in general.

WHY A DECORATOR RATHER THAN A CLASS-BODY ``__eq__``. Binding the compared-field names ONCE, at
class creation, is what makes the "identical to 3.11" claim literally true. Discovering them per
call from ``fields(obj)`` instead is subtly different: a subclass declared ``eq=False`` inherits
this ``__eq__``, and a per-call lookup would then compare the subclass's extra fields where the
3.11-generated base method ignored them. Measured -- with the per-call spelling,
``Child(perm, offs, groups, None, extra=1) == Child(..., extra=2)`` answered ``False`` where the
generated base ``__eq__`` answers ``True``. (codex round 1.)

It also keeps hashing out of the argument. ``@dataclass`` runs first and settles the equality and
hash state together; this decorator then replaces ``__eq__`` alone. Frozen ``eq=True`` classes
keep the ``__hash__`` dataclasses generated for them, and non-frozen ``eq=True`` classes keep the
``__hash__ = None`` it installed. Nothing here reasons about ``has_explicit_hash``, because no
``__eq__`` is ever present in the class body.

PRECONDITIONS, enforced:

* the class must carry its OWN ``__dataclass_params__``, i.e. ``@dataclass`` must have run on
  THIS class. ``getattr`` is the wrong probe and was the first version's bug: a subclass INHERITS
  the metadata while its own equality and hash state were never settled, so the guard passed for
  a class ``@dataclass`` had never touched. ``dataclasses.is_dataclass()`` has the same
  inheritance behaviour and is equally unsuitable. A subclass that is itself decorated with
  ``@dataclass`` does carry its own params and is accepted.

  What the hole actually costs depends on the subclass. A PLAIN subclass of a default
  ``eq=True`` dataclass inherits ``__hash__ = None`` and is merely unhashable, so decorating it
  would be harmless-but-wrong. The damaging shape is one that has identity semantics back --
  e.g. an earlier MRO base restoring ``object.__eq__``/``object.__hash__``: there
  ``@field_tuple_eq`` would install VALUE equality next to ``object.__hash__`` and leave equal
  instances hashing differently. Both shapes are refused, and both are pinned by tests.
  (codex rounds 3-4.)
* ``eq=True``, which is the default. This is a deliberately CONSERVATIVE refusal rather than a
  precise one: the ordinary ``eq=False`` dataclass keeps identity hashing, so value equality on
  top of it would break the hash invariant -- but ``eq=False, unsafe_hash=True`` and ``eq=False``
  with an explicit ``__hash__ = None`` are both safe, and are refused anyway. Supporting every
  safe explicit-or-inherited hash configuration is out of scope; repairing the unsafe one by
  writing ``__hash__`` here would be worse.

NOT enforced, because after ``@dataclass`` has run there is no robust way to tell a generated
``__eq__`` from a hand-written one: this decorator overwrites whatever ``__eq__`` the class ended
up with, so do not apply it to a class that means to define its own comparison, and note that a
custom ``__ne__`` would survive and could then disagree.

Apply it ABOVE ``@dataclass`` -- decorators run bottom-up, so ``@dataclass`` must have finished::

    @field_tuple_eq
    @dataclass(frozen=True)
    class ShardCSRDLPack:
        ...
"""

from __future__ import annotations

import operator
from dataclasses import fields

#: Set on the ``__eq__`` FUNCTION this decorator installs, holding the compared-field names it
#: bound. It is the project-owned marker the test suite asserts on.
#:
#: On the function rather than the class, deliberately: a class attribute would survive someone
#: later reassigning ``C.__eq__``, so a structural test reading it would keep passing while the
#: method it vouches for was gone. Reading it off ``cls.__dict__["__eq__"]`` instead means
#: ORDINARY REASSIGNMENT cannot leave a stale marker behind, and an undecorated subclass fails
#: the check for the natural reason -- it has no ``__eq__`` of its own. Not airtight: a
#: replacement wrapped in ``functools.wraps(old_eq)`` copies the wrapped function's ``__dict__``
#: and would carry the marker across. (codex rounds 2-3.)
#:
#: The obvious alternative -- checking ``__eq__.__code__.co_filename != "<string>"`` -- reads a
#: CPython code-generation detail that no version promises, so a future interpreter compiling
#: generated methods with a real filename would make that test silently stop detecting anything.
#: (codex round 1.)
EQ_MARKER = "_cellstream_field_tuple_eq_fields"


def compared_field_names(cls) -> tuple[str, ...]:
    """The names ``@dataclass`` would have compared, in order.

    ``f.compare`` is honoured, so a ``field(compare=False)`` stays out of the comparison the way
    the generated ``__eq__`` keeps it out.
    """
    return tuple(f.name for f in fields(cls) if f.compare)


def field_tuple_eq(cls):
    """Replace ``cls.__eq__`` with one that compares field TUPLES. See the module docstring.

    The result is ALWAYS a tuple, including for a one-field dataclass. That is the property the
    whole module rests on, and it is why the getter is built by an explicit three-way dispatch
    rather than by ``operator.attrgetter(*names)`` alone: attrgetter returns the BARE VALUE for a
    single name and rejects zero names, so trimming a class to one compared field would silently
    reintroduce exactly the defect this module exists to remove. Both edges are pinned by tests.
    """
    # `cls.__dict__`, never `getattr`: see the PRECONDITIONS note in the module docstring.
    params = cls.__dict__.get("__dataclass_params__")
    if params is None:
        raise TypeError(
            f"@field_tuple_eq expects @dataclass to have run on {cls!r} itself; it carries no "
            f"__dataclass_params__ of its own (an inherited one does not count -- such a "
            f"subclass never had its equality or hash state settled)"
        )
    if not params.eq:
        raise TypeError(
            f"@field_tuple_eq requires @dataclass(eq=True); {cls.__name__} sets eq=False. This "
            f"is a CONSERVATIVE refusal, not a diagnosis: an ordinary eq=False dataclass keeps "
            f"identity hashing, and value equality on top of that would leave equal instances "
            f"hashing differently -- but eq=False with unsafe_hash=True, or with an explicit "
            f"__hash__ = None, is hash-safe and is refused here too. Supporting every safe "
            f"configuration is out of scope; see cellstream._dataclass_eq for the reasoning."
        )

    names = compared_field_names(cls)

    # Built ONCE, at decoration time. `attrgetter` is C-speed: measured 6.4x faster than a
    # per-call `tuple(getattr(obj, n) for n in names)` at four fields and 4.7x at ten, over
    # 200k comparisons (0.92us -> 0.14us). Small in absolute terms, since none of these classes
    # is compared in a loop today -- but it costs one branch at import and removes the
    # "not a hot path" hand-wave from the argument. (gemini.)
    if not names:

        def _field_tuple(_obj):
            return ()

    elif len(names) == 1:
        (_only,) = names

        def _field_tuple(obj):
            return (getattr(obj, _only),)

    else:
        _field_tuple = operator.attrgetter(*names)

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return _field_tuple(self) == _field_tuple(other)

    # Wear the owning class's identity, the way a generated method does. `__module__` is not
    # cosmetic: pickling the UNBOUND `C.__eq__` looks the function up by module + qualname, and
    # fails if `__module__` still points here. (codex round 2.)
    __eq__.__module__ = cls.__module__
    __eq__.__qualname__ = f"{cls.__qualname__}.__eq__"
    __eq__.__doc__ = (
        f"Compare the field tuple {names} -- what dataclasses generated before CPython 3.13. "
        f"Installed by cellstream._dataclass_eq.field_tuple_eq (#318)."
    )
    setattr(__eq__, EQ_MARKER, names)
    cls.__eq__ = __eq__
    return cls
