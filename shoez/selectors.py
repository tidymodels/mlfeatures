"""Role-, type-, name-, and provenance-aware column selectors.

Import as ``from shoez import Selectors as cs`` and pass the result of a
factory call directly to any step:

```python
StepDropCorr(Selectors.numeric_predictors())
StepPCA(Selectors.starts_with("x") & Selectors.predictors())
StepPCA(Selectors.has_source(step_type=StepDummy))
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
    """Columns whose names match a predicate, in frame order."""

    def __init__(
        self, predicate: "Callable[[str], bool]", label: str = "name_selector()"
    ) -> None:
        self._predicate = predicate
        self._label = label

    def __repr__(self) -> str:
        return self._label

    def resolve(
        self,
        frame: Frame,
        roles: Roles,
        schema: Schema | None = None,
        steps: dict[str, Step] | None = None,
    ) -> list[str]:
        from shoez import _frame

        return [c for c in _frame.columns(frame) if self._predicate(c)]


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
        if self._step is not None:
            return {self._step}
        if steps is None:
            return set()
        return {
            sid
            for sid, s in steps.items()
            if self._step_type is not None and isinstance(s, self._step_type)
        }

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
    from shoez import Recipe, StepNormalize, Selectors
    ```

    ```{python}
    df = pd.DataFrame(
        {"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "y": [7.0, 8.0, 9.0]}
    )
    step = StepNormalize(Selectors.starts_with("x"))
    Recipe().add(step).fit(df).transform(df)
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
        Selectors.col("x1")
        Selectors.col("x1", "x2")
        ```
        """
        wanted = set(names)
        label = f"col({', '.join(repr(n) for n in names)})"
        return _NameSelector(lambda c: c in wanted, label)

    @classmethod
    def starts_with(cls, prefix: str) -> ShoezSelector:
        """Columns whose name starts with ``prefix``."""
        return _NameSelector(lambda c: c.startswith(prefix), f"starts_with({prefix!r})")

    @classmethod
    def ends_with(cls, suffix: str) -> ShoezSelector:
        """Columns whose name ends with ``suffix``."""
        return _NameSelector(lambda c: c.endswith(suffix), f"ends_with({suffix!r})")

    @classmethod
    def contains(cls, substring: str) -> ShoezSelector:
        """Columns whose name contains ``substring``."""
        return _NameSelector(lambda c: substring in c, f"contains({substring!r})")

    @classmethod
    def matches(cls, pattern: str) -> ShoezSelector:
        """Columns whose name matches the regular expression ``pattern``.

        Uses ``re.search`` (anchor explicitly with ``^``/``$`` if desired).
        """
        compiled = re.compile(pattern)
        return _NameSelector(
            lambda c: compiled.search(c) is not None, f"matches({pattern!r})"
        )

    @classmethod
    def num_range(cls, prefix: str, rng: Iterable[int]) -> ShoezSelector:
        """Columns named ``f"{prefix}{i}"`` for ``i`` in ``rng`` (e.g. x1..x5).

        ```python
        Selectors.num_range("x", range(1, 6))  # x1, x2, x3, x4, x5
        ```
        """
        rng = list(rng)
        wanted = {f"{prefix}{i}" for i in rng}
        return _NameSelector(lambda c: c in wanted, f"num_range({prefix!r}, {rng!r})")

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
    def predictors(cls) -> ShoezSelector:
        """All predictor columns regardless of type."""
        return _RoleTypeSelector(role="predictor", dtype=None)

    @classmethod
    def outcomes(cls) -> ShoezSelector:
        """All outcome columns regardless of type."""
        return _RoleTypeSelector(role="outcome", dtype=None)

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

    # --- Compound predictor selectors (preferred in step args) -------------

    @classmethod
    def numeric_predictors(cls) -> ShoezSelector:
        """All predictor columns that are numeric."""
        return _RoleTypeSelector(role="predictor", dtype=nw.dtypes.NumericType)

    @classmethod
    def categorical_predictors(cls) -> ShoezSelector:
        """All predictor columns that are categorical."""
        return _RoleTypeSelector(role="predictor", dtype=(nw.Categorical, nw.Enum))

    @classmethod
    def nominal_predictors(cls) -> ShoezSelector:
        """All predictor columns that are nominal (categorical, enum, or string)."""
        return _RoleTypeSelector(
            role="predictor", dtype=(nw.String, nw.Categorical, nw.Enum)
        )

    @classmethod
    def string_predictors(cls) -> ShoezSelector:
        """All predictor columns that are string-typed."""
        return _RoleTypeSelector(role="predictor", dtype=nw.String)

    @classmethod
    def integer_predictors(cls) -> ShoezSelector:
        """All predictor columns that are integer-typed."""
        return _RoleTypeSelector(role="predictor", dtype=nw.dtypes.IntegerType)

    @classmethod
    def double_predictors(cls) -> ShoezSelector:
        """All predictor columns that are floating-point typed."""
        return _RoleTypeSelector(role="predictor", dtype=nw.dtypes.FloatType)

    @classmethod
    def date_predictors(cls) -> ShoezSelector:
        """All predictor columns that are Date-typed."""
        return _RoleTypeSelector(role="predictor", dtype=nw.Date)

    @classmethod
    def datetime_predictors(cls) -> ShoezSelector:
        """All predictor columns that are Datetime-typed."""
        return _RoleTypeSelector(role="predictor", dtype=nw.Datetime)

    @classmethod
    def boolean_predictors(cls) -> ShoezSelector:
        """All predictor columns that are boolean-typed."""
        return _RoleTypeSelector(role="predictor", dtype=nw.Boolean)

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

        ```python
        Selectors.has_source("dummy_abc123")
        Selectors.has_source(step_type=StepDummy)
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
        Selectors.wrap(cs.starts_with("x_")) & Selectors.predictors()
        ```
        """
        return _coerce(selector)
