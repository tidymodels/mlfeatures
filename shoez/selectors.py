"""Role-, type-, name-, and provenance-aware column selectors.

Import as ``from shoez import Selectors as s`` and pass the result of a
factory call directly to any step:

```python
StepDropCorr(s.numeric_features())
StepPCA(s.starts_with("x") & s.features())
StepPCA(s.has_source(step_type=StepDummy))
```

Selectors are resolved to a list of column names **at fit time**, when the
frame schema (dtypes), the recipe roles, and the column-lineage DAG are all
available. They are the recommended, fully composable selector surface for
shoez. ``shoez`` also still accepts raw polars / narwhals selectors anywhere a
selector is expected (see ``shoez._frame.select``), but those compose cleanly
only *within* their own family — see ``design/selectors.md`` and `Selectors.wrap`.

See ``design/selectors.md`` for the full design.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

if TYPE_CHECKING:
    from shoez.schema import Schema
    from shoez.step import Step
    from shoez.types import Frame, Roles


class ShoezSelector:
    """Base class for all shoez-native column selectors.

    Subclasses implement `resolve`. Instances compose with ``&`` (intersection),
    ``|`` (union), ``~`` (complement), and ``-`` (difference). A shoez-native
    selector must **lead** any cross-family expression — ``shoez_sel & cs.numeric()``
    works (the right operand is coerced via `wrap`), but ``cs.numeric() & shoez_sel``
    does not: polars raises ``TypeError`` and narwhals silently produces a broken
    ``Expr``, neither of which shoez can intercept. Use `Selectors.wrap` to put a
    polars / narwhals selector first.
    """

    def resolve(
        self,
        frame: Frame,
        roles: Roles,
        schema: Schema | None = None,
        steps: dict[str, Step] | None = None,
    ) -> list[str]:
        """Resolve to concrete column names against ``frame`` (and ``roles``).

        ``schema`` and ``steps`` are the live lineage DAG and id→step map, only
        needed by provenance selectors; they are ``None`` outside a recipe fit.
        """
        raise NotImplementedError

    def __and__(self, other: object) -> ShoezSelector:
        return _CompositeSelector("and", [self, _coerce(other)])

    def __or__(self, other: object) -> ShoezSelector:
        return _CompositeSelector("or", [self, _coerce(other)])

    def __sub__(self, other: object) -> ShoezSelector:
        return _CompositeSelector("and", [self, _Complement(_coerce(other))])

    def __invert__(self) -> ShoezSelector:
        return _Complement(self)


def _coerce(value: object) -> ShoezSelector:
    """Lift a selector-like value into a `ShoezSelector` leaf.

    Accepts an existing `ShoezSelector` (returned as-is), a column name, a
    sequence of names, or a polars / narwhals selector. This is what lets a
    shoez selector lead a cross-family expression.
    """
    if isinstance(value, ShoezSelector):
        return value
    return _ForeignSelector(value)


class _ForeignSelector(ShoezSelector):
    """Adapts a raw name / list / polars / narwhals selector to `ShoezSelector`.

    Resolution defers to ``shoez._frame.select``'s existing type dispatch, so a
    polars selector still expands via the schema and a narwhals selector via
    narwhals — for any backend.
    """

    def __init__(self, value: object) -> None:
        self._value = value

    def __repr__(self) -> str:
        return repr(self._value)

    def resolve(
        self,
        frame: Frame,
        roles: Roles,
        schema: Schema | None = None,
        steps: dict[str, Step] | None = None,
    ) -> list[str]:
        from shoez import _frame

        # ``self._value`` is one of the non-ShoezSelector Selector members
        # (name, sequence of names, polars selector, or narwhals selector),
        # all handled by ``_frame.select``'s type dispatch.
        return _frame.select(frame, self._value, roles).columns  # type: ignore[arg-type]


class _CompositeSelector(ShoezSelector):
    """Set-op over child selectors, preserving frame column order."""

    def __init__(self, op: str, children: Sequence[ShoezSelector]) -> None:
        self._op = op
        self._children = list(children)

    def __repr__(self) -> str:
        joiner = " & " if self._op == "and" else " | "
        return joiner.join(repr(child) for child in self._children)

    def resolve(
        self,
        frame: Frame,
        roles: Roles,
        schema: Schema | None = None,
        steps: dict[str, Step] | None = None,
    ) -> list[str]:
        from shoez import _frame

        resolved: list[set[str]] = [
            set(child.resolve(frame, roles, schema, steps)) for child in self._children
        ]
        keep: set[str] = set()
        if resolved:
            keep = resolved[0]
            for other in resolved[1:]:
                keep = keep & other if self._op == "and" else keep | other
        # Restore deterministic frame order.
        return [c for c in _frame.columns(frame) if c in keep]


class _Complement(ShoezSelector):
    """Everything in the frame *except* the wrapped selector's columns."""

    def __init__(self, inner: ShoezSelector) -> None:
        self._inner = inner

    def __repr__(self) -> str:
        return f"~{self._inner!r}"

    def resolve(
        self,
        frame: Frame,
        roles: Roles,
        schema: Schema | None = None,
        steps: dict[str, Step] | None = None,
    ) -> list[str]:
        from shoez import _frame

        excluded = set(self._inner.resolve(frame, roles, schema, steps))
        return [c for c in _frame.columns(frame) if c not in excluded]


class _RoleTypeSelector(ShoezSelector):
    """Columns filtered by role (optional) and narwhals dtype (optional)."""

    def __init__(
        self,
        role: str | None,
        dtype: type | tuple[type, ...] | None,
    ) -> None:
        self._role = role
        self._dtype = dtype

    def __repr__(self) -> str:
        if self._role is None and self._dtype is None:
            return "everything()"
        parts: list[str] = []
        if self._role is not None:
            parts.append(f"role={self._role!r}")
        if self._dtype is not None:
            if isinstance(self._dtype, tuple):
                names = ", ".join(t.__name__ for t in self._dtype)
                parts.append(f"type=({names})")
            else:
                parts.append(f"type={self._dtype.__name__}")
        return f"select({', '.join(parts)})"

    def resolve(
        self,
        frame: Frame,
        roles: Roles,
        schema: Schema | None = None,
        steps: dict[str, Step] | None = None,
    ) -> list[str]:
        from shoez import _frame

        if self._role is not None:
            candidates = [
                c for c in roles.columns_with_role(self._role) if c in frame.columns
            ]
        else:
            candidates = _frame.columns(frame)

        if self._dtype is None:
            return candidates
        return [c for c in candidates if issubclass(type(frame.schema[c]), self._dtype)]


class _NameSelector(ShoezSelector):
    """Columns whose names match a serializable name rule, in frame order.

    The match rule is stored as a ``kind`` string plus its ``params`` (never a
    closure), so the selector serializes to plain data -- the no-closure
    guarantee `shoez.serialization` relies on. ``resolve`` rebuilds the
    predicate from ``kind``/``params`` on the fly.
    """

    def __init__(self, kind: str, params: "Sequence[object]", label: str) -> None:
        self._kind = kind
        self._params = list(params)
        self._label = label

    def __repr__(self) -> str:
        return self._label

    def _predicate(self) -> "Callable[[str], bool]":
        kind, params = self._kind, self._params
        if kind == "col":
            wanted = set(params)
            return lambda c: c in wanted
        if kind == "starts_with":
            prefix = params[0]
            assert isinstance(prefix, str)
            return lambda c: c.startswith(prefix)
        if kind == "ends_with":
            suffix = params[0]
            assert isinstance(suffix, str)
            return lambda c: c.endswith(suffix)
        if kind == "contains":
            substring = params[0]
            assert isinstance(substring, str)
            return lambda c: substring in c
        if kind == "matches":
            pattern = params[0]
            assert isinstance(pattern, str)
            compiled = re.compile(pattern)
            return lambda c: compiled.search(c) is not None
        if kind == "num_range":
            wanted = set(params)
            return lambda c: c in wanted
        raise ValueError(f"unknown name-selector kind {kind!r}")

    def resolve(
        self,
        frame: Frame,
        roles: Roles,
        schema: Schema | None = None,
        steps: dict[str, Step] | None = None,
    ) -> list[str]:
        from shoez import _frame

        predicate = self._predicate()
        return [c for c in _frame.columns(frame) if predicate(c)]


def _available(what: str, names: list[str]) -> str:
    """Tail of a provenance error: what was on offer, or that nothing was."""
    if not names:
        return "this is the recipe's first step"
    return f"earlier {what} {names}"


class _SourceSelector(ShoezSelector):
    """Columns selected by which step produced them, via the lineage DAG.

    ``ancestral=False`` matches a column's immediate producer (``has_source``);
    ``ancestral=True`` matches any step in the column's ancestry (``has_ancestor``).
    """

    def __init__(
        self,
        step: str | None,
        step_type: "type[Step] | None",
        ancestral: bool,
    ) -> None:
        if (step is None) == (step_type is None):
            raise ValueError("pass exactly one of 'step' or 'step_type='")
        self._step = step
        self._step_type = step_type
        self._ancestral = ancestral

    def __repr__(self) -> str:
        name = "has_ancestor" if self._ancestral else "has_source"
        if self._step is not None:
            arg = repr(self._step)
        else:
            assert self._step_type is not None
            arg = f"step_type={self._step_type.__name__}"
        return f"{name}({arg})"

    def _matching_ids(self, schema: Schema, steps: dict[str, Step] | None) -> set[str]:
        if steps is None:
            # No id→step map (e.g. resolved outside a recipe fit): no matches,
            # and nothing to validate the id or type against.
            return {self._step} if self._step is not None else set()
        # A step id or type either names a step in the recipe or it is a typo,
        # and that is knowable from the recipe alone. So unlike a dtype or role
        # selector, matching nothing here is an error rather than the identity
        # transform. Only steps fitted before this one are visible: a later
        # step has produced no columns yet.
        if self._step is not None:
            if self._step != "original" and self._step not in steps:
                raise ValueError(
                    f"no step with id {self._step!r} before this one; "
                    + _available("step id(s)", sorted(steps))
                )
            return {self._step}
        assert self._step_type is not None
        matched = {sid for sid, s in steps.items() if isinstance(s, self._step_type)}
        if not matched:
            seen = sorted({type(s).__name__ for s in steps.values()})
            raise ValueError(
                f"no step of type {self._step_type.__name__!r} before this one; "
                + _available("step type(s)", seen)
            )
        return matched

    def resolve(
        self,
        frame: Frame,
        roles: Roles,
        schema: Schema | None = None,
        steps: dict[str, Step] | None = None,
    ) -> list[str]:
        from shoez import _frame

        if schema is None:
            # No DAG available (e.g. resolved outside a recipe fit): no matches.
            return []
        ids = self._matching_ids(schema, steps)
        if not ids:
            return []

        live = set(_frame.columns(frame))
        selected: set[str]
        if not self._ancestral:
            selected = {
                name
                for sid in ids
                for name in schema.columns_from_source(sid)
                if name in live
            }
        else:
            selected = set()
            for name in live:
                node_sources = {schema[name].source} | {
                    schema[a].source for a in schema.ancestors(name) if a in schema
                }
                if node_sources & ids:
                    selected.add(name)
        return [c for c in _frame.columns(frame) if c in selected]


class Selectors:
    """Namespace of factory methods for built-in shoez column selectors.

    All factories return a `ShoezSelector` that composes with ``&`` / ``|`` /
    ``~`` / ``-``. See ``design/selectors.md``.

    Examples
    --------
    A selector stands in for an explicit column list and is resolved against the
    frame at fit time. Here ``starts_with("x")`` picks up ``x1`` and ``x2`` but
    leaves ``y`` untouched:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepNormalize
    from shoez import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame(
        {"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "y": [7.0, 8.0, 9.0]}
    )
    recipe = Recipe().add(StepNormalize(s.starts_with("x")))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    # --- By name -----------------------------------------------------------

    @classmethod
    def col(cls, *names: str) -> ShoezSelector:
        """The named column(s), if present.

        Names that aren't in the frame are silently skipped (tolerant, like R's
        ``any_of``). This differs from passing a plain ``list`` of names as a
        selector operand, which is strict and raises on a missing column (like
        ``all_of``); use `col` when absence should be a no-op.

        ```python
        s.col("x1")
        s.col("x1", "x2")
        ```
        """
        label = f"col({', '.join(repr(n) for n in names)})"
        return _NameSelector("col", list(names), label)

    @classmethod
    def starts_with(cls, prefix: str) -> ShoezSelector:
        """Columns whose name starts with ``prefix``."""
        return _NameSelector("starts_with", [prefix], f"starts_with({prefix!r})")

    @classmethod
    def ends_with(cls, suffix: str) -> ShoezSelector:
        """Columns whose name ends with ``suffix``."""
        return _NameSelector("ends_with", [suffix], f"ends_with({suffix!r})")

    @classmethod
    def contains(cls, substring: str) -> ShoezSelector:
        """Columns whose name contains ``substring``."""
        return _NameSelector("contains", [substring], f"contains({substring!r})")

    @classmethod
    def matches(cls, pattern: str) -> ShoezSelector:
        """Columns whose name matches the regular expression ``pattern``.

        Uses ``re.search`` (anchor explicitly with ``^``/``$`` if desired).
        """
        return _NameSelector("matches", [pattern], f"matches({pattern!r})")

    @classmethod
    def num_range(cls, prefix: str, rng: Iterable[int]) -> ShoezSelector:
        """Columns named ``f"{prefix}{i}"`` for ``i`` in ``rng`` (e.g. x1..x5).

        ```python
        s.num_range("x", range(1, 6))  # x1, x2, x3, x4, x5
        ```
        """
        rng = list(rng)
        wanted = [f"{prefix}{i}" for i in rng]
        return _NameSelector("num_range", wanted, f"num_range({prefix!r}, {rng!r})")

    @classmethod
    def everything(cls) -> ShoezSelector:
        """All columns."""
        return _RoleTypeSelector(role=None, dtype=None)

    # --- By role -----------------------------------------------------------

    @classmethod
    def has_role(cls, role: str) -> ShoezSelector:
        """All columns carrying ``role``."""
        return _RoleTypeSelector(role=role, dtype=None)

    @classmethod
    def features(cls) -> ShoezSelector:
        """All feature columns regardless of type."""
        return _RoleTypeSelector(role="feature", dtype=None)

    @classmethod
    def targets(cls) -> ShoezSelector:
        """All target columns regardless of type."""
        return _RoleTypeSelector(role="target", dtype=None)

    # --- By type (bare) ----------------------------------------------------

    @classmethod
    def numeric(cls) -> ShoezSelector:
        """All numeric (int + float) columns."""
        return _RoleTypeSelector(role=None, dtype=nw.dtypes.NumericType)

    @classmethod
    def float(cls) -> ShoezSelector:
        """All floating-point columns."""
        return _RoleTypeSelector(role=None, dtype=nw.dtypes.FloatType)

    #: Alias of `float`, matching R's ``all_double``.
    double = float

    @classmethod
    def integer(cls) -> ShoezSelector:
        """All integer columns (signed or unsigned)."""
        return _RoleTypeSelector(role=None, dtype=nw.dtypes.IntegerType)

    @classmethod
    def string(cls) -> ShoezSelector:
        """All string columns."""
        return _RoleTypeSelector(role=None, dtype=nw.String)

    @classmethod
    def categorical(cls) -> ShoezSelector:
        """All categorical/enum columns (ordered or unordered)."""
        return _RoleTypeSelector(role=None, dtype=(nw.Categorical, nw.Enum))

    @classmethod
    def ordered(cls) -> ShoezSelector:
        """All ordered-categorical columns (narwhals ``Enum``)."""
        return _RoleTypeSelector(role=None, dtype=nw.Enum)

    @classmethod
    def unordered(cls) -> ShoezSelector:
        """All unordered-categorical columns (narwhals ``Categorical``)."""
        return _RoleTypeSelector(role=None, dtype=nw.Categorical)

    @classmethod
    def nominal(cls) -> ShoezSelector:
        """All nominal columns (string, categorical, or enum)."""
        return _RoleTypeSelector(role=None, dtype=(nw.String, nw.Categorical, nw.Enum))

    @classmethod
    def boolean(cls) -> ShoezSelector:
        """All boolean columns."""
        return _RoleTypeSelector(role=None, dtype=nw.Boolean)

    @classmethod
    def date(cls) -> ShoezSelector:
        """All Date columns."""
        return _RoleTypeSelector(role=None, dtype=nw.Date)

    @classmethod
    def datetime(cls) -> ShoezSelector:
        """All Datetime columns."""
        return _RoleTypeSelector(role=None, dtype=nw.Datetime)

    # --- Compound feature selectors (preferred in step args) -------------

    @classmethod
    def numeric_features(cls) -> ShoezSelector:
        """All feature columns that are numeric."""
        return _RoleTypeSelector(role="feature", dtype=nw.dtypes.NumericType)

    @classmethod
    def categorical_features(cls) -> ShoezSelector:
        """All feature columns that are categorical."""
        return _RoleTypeSelector(role="feature", dtype=(nw.Categorical, nw.Enum))

    @classmethod
    def nominal_features(cls) -> ShoezSelector:
        """All feature columns that are nominal (categorical, enum, or string)."""
        return _RoleTypeSelector(
            role="feature", dtype=(nw.String, nw.Categorical, nw.Enum)
        )

    @classmethod
    def string_features(cls) -> ShoezSelector:
        """All feature columns that are string-typed."""
        return _RoleTypeSelector(role="feature", dtype=nw.String)

    @classmethod
    def integer_features(cls) -> ShoezSelector:
        """All feature columns that are integer-typed."""
        return _RoleTypeSelector(role="feature", dtype=nw.dtypes.IntegerType)

    @classmethod
    def double_features(cls) -> ShoezSelector:
        """All feature columns that are floating-point typed."""
        return _RoleTypeSelector(role="feature", dtype=nw.dtypes.FloatType)

    @classmethod
    def date_features(cls) -> ShoezSelector:
        """All feature columns that are Date-typed."""
        return _RoleTypeSelector(role="feature", dtype=nw.Date)

    @classmethod
    def datetime_features(cls) -> ShoezSelector:
        """All feature columns that are Datetime-typed."""
        return _RoleTypeSelector(role="feature", dtype=nw.Datetime)

    @classmethod
    def boolean_features(cls) -> ShoezSelector:
        """All feature columns that are boolean-typed."""
        return _RoleTypeSelector(role="feature", dtype=nw.Boolean)

    # --- By source (lineage DAG) -------------------------------------------

    @classmethod
    def has_source(
        cls,
        step: str | None = None,
        *,
        step_type: "type[Step] | None" = None,
    ) -> ShoezSelector:
        """Columns whose **immediate** producer is the given step.

        Pass either a step id or ``step_type=`` (a `Step` subclass). Resolves
        against the lineage DAG at fit time. For columns that trace back to a
        step *anywhere* in their ancestry, use `has_ancestor`.

        A step's default id is its type's short name, and a second step of the
        same type in one recipe is given a suffixed id (``"dummy_2"``), so an
        id here names exactly one step.

        Unlike a dtype or role selector, matching nothing is an error, not the
        identity transform: raises `ValueError` if no step earlier in the recipe
        has that id (or is an instance of that type). Pass ``"original"`` as the
        id for the input columns that survive to this point.

        ```python
        s.has_source("dummy")
        s.has_source(step_type=StepDummy)
        ```
        """
        return _SourceSelector(step=step, step_type=step_type, ancestral=False)

    @classmethod
    def has_ancestor(
        cls,
        step: str | None = None,
        *,
        step_type: "type[Step] | None" = None,
    ) -> ShoezSelector:
        """Columns with the given step **anywhere in their ancestry**.

        The ancestral counterpart of `has_source`: matches columns even after
        they were further transformed (e.g. dummies that were then normalized).
        """
        return _SourceSelector(step=step, step_type=step_type, ancestral=True)

    # --- Escape hatch ------------------------------------------------------

    @classmethod
    def wrap(cls, selector: object) -> ShoezSelector:
        """Lift a raw name / list / polars / narwhals selector into shoez-land.

        Use this to let a polars or narwhals selector *lead* a composed
        expression, since ``cs.numeric() & shoez_sel`` cannot work directly:

        ```python
        import polars.selectors as cs
        from shoez import Selectors as s

        s.wrap(cs.starts_with("x_")) & s.features()
        ```
        """
        return _coerce(selector)
