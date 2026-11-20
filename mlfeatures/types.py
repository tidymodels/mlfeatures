"""Shared type aliases, sentinels, and small value types for the step contract.

These are the names a step is written and annotated against: the `Frame` and
`Selector` aliases, the step-facing `ColSchema` and `SchemaInfo`, `Roles`, the
`Mode` and `KeepOriginals` enums, and the dtype sets returned by
``Step.accepts_dtypes()``. The recipe-side column-lineage DAG lives in
`mlfeatures.lineage` and the training-data snapshot in `mlfeatures.schema`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import narwhals.stable.v2 as nw
from narwhals.selectors import Selector as NwSelector

from ._databackend import AbstractBackend

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from polars import Expr as PlExpr
    from polars.selectors import Selector as PlSelector

    from mlfeatures.lineage import Lineage
    from mlfeatures.selectors import FeaturesSelector
    from mlfeatures.step import Step

    #: A native data frame at the package boundary, as accepted by
    #: ``Recipe.fit()`` and returned by ``FittedRecipe.transform()``. The
    #: annotation names pandas and polars, the two backends tested; any other
    #: backend narwhals supports is not covered by that guarantee. Steps do
    #: **not** see this type: a data frame is converted to a `Frame` on entry
    #: to the recipe, and ``transform()`` converts back to the type that went
    #: in (and the sklearn adapter to an array or pandas data frame).
    DataFrame = Union["pd.DataFrame", "pl.DataFrame"]
else:
    DataFrame = Any

    class PlSelector(AbstractBackend):
        # ``isinstance(x, PlSelector)`` is True for a ``polars.selectors.Selector``
        # without importing polars unless it is already loaded (see _databackend).
        # That public class only exists from polars 1.32, so runtime detection
        # goes through ``isinstance(x, PlExpr)`` (a selector is an ``Expr`` on
        # every version); this class is the type-level name for annotations and
        # the ``Selector`` union.
        _backends = [("polars.selectors", "Selector")]

    class PlExpr(AbstractBackend):
        # ``isinstance(x, PlExpr)`` is True for a polars expression, selectors
        # included, without importing polars unless it is already loaded. It is
        # accepted as a selector when it is a plain column selection such as
        # ``polars.col("a")``.
        _backends = [("polars", "Expr")]


#: The internal data frame type that flows between steps. A backend-agnostic narwhals
#: ``DataFrame`` wrapping the user's native data frame; steps write their data ops
#: against this via `mlfeatures._frame`.
Frame: TypeAlias = nw.DataFrame[Any]

#: The type of every step's ``columns`` parameter: a single column name, an
#: explicit list of names, a mlfeatures selector such as
#: ``Selectors.numeric_features()`` (see `Selectors` for the full set), or a
#: narwhals or polars selector (``narwhals.selectors.numeric()``,
#: ``polars.selectors.numeric()``). A polars column selection such as
#: ``polars.col("a")`` is accepted too. Any selector is resolved against the
#: data frame's schema at fit time, so the same selector works on any backend.
Selector: TypeAlias = Union[
    str, Sequence[str], NwSelector, "PlSelector", "PlExpr", "FeaturesSelector"
]

# Listed so the selector aliases (``NwSelector`` is a renamed import; ``PlSelector``
# and ``PlExpr`` are conditional imports under TYPE_CHECKING) count as explicit
# re-exports for strict type-checkers; the other names are module-level
# definitions and re-exportable regardless.
__all__ = [
    "ANY_DTYPE",
    "NUMERIC_DTYPES",
    "AcceptedDtypes",
    "ColSchema",
    "DataFrame",
    "Frame",
    "Mode",
    "NwSelector",
    "PlExpr",
    "PlSelector",
    "SchemaInfo",
    "Selector",
]


class ColSchema:
    """Column schema: the names and dtypes a step reports for a column set.

    Carries the column names and their dtype names: enough for a step to
    describe what it consumed (``schema_in``) and produced (``schema_out``).
    This is the lightweight step-facing report; the recipe-side column-lineage
    DAG that tracks roles, source, sparsity, and provenance is
    `Lineage`. Not to be confused with `SchemaInfo` (what a step is handed at
    fit time) or `Schema` (the snapshot of the training data that
    ``transform()`` validates new data against).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import ColSchema
    ```

    ```{python}
    schema = ColSchema(columns=["age", "city"], dtypes={"age": "Int64", "city": "String"})
    schema.columns, schema.dtypes
    ```
    """

    #: Column names, in order.
    columns: list[str]
    #: Dtype name of each column, keyed by column name.
    dtypes: dict[str, str]

    def __init__(
        self,
        columns: list[str] | None = None,
        dtypes: dict[str, str] | None = None,
    ):
        self.columns = list(columns) if columns is not None else []
        self.dtypes = dict(dtypes) if dtypes is not None else {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"ColSchema(columns={self.columns!r})"


class SchemaInfo:
    """Schema-derived context handed to `Step.fit()`.

    Carries the current column names, their dtype names, and the live `Roles`.
    In practice a step reads only ``roles`` (for selector resolution). The
    recipe-side column-lineage DAG (`mlfeatures.lineage.Lineage`)
    is maintained separately by the recipe and is not handed to steps for their
    own logic; ``lineage`` and ``producing_steps`` are populated by the recipe
    purely so provenance selectors (``has_source`` / ``has_ancestor``) can
    resolve against the DAG at fit time. Both are ``None`` outside a recipe fit.
    Not to be confused with `ColSchema` (what a step reports back) or `Schema`
    (the snapshot of the training data used for boundary checks).

    Examples
    --------
    Build a snapshot from a data frame with `from_frame`; steps read ``columns``,
    ``dtypes``, and ``roles`` off it during ``fit()``:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import SchemaInfo, _frame
    ```

    ```{python}
    frame = _frame.from_native(pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}))
    info = SchemaInfo.from_frame(frame)
    info.columns, info.dtypes
    ```
    """

    #: Current column names.
    columns: list[str]
    #: Dtype name of each column, keyed by column name.
    dtypes: dict[str, str]
    #: The live roles of the columns.
    roles: Roles
    #: The live column-lineage DAG, set by the recipe during fit (else None).
    lineage: "Lineage | None"
    #: Map of step id → fitted step, set by the recipe during fit (else None).
    #: Lets ``has_source(step_type=...)`` map ids back to step instances.
    producing_steps: "dict[str, Step] | None"

    def __init__(
        self,
        columns: list[str] | None = None,
        dtypes: dict[str, str] | None = None,
        roles: Roles | None = None,
        lineage: "Lineage | None" = None,
        producing_steps: "dict[str, Step] | None" = None,
    ):
        self.columns = list(columns) if columns is not None else []
        self.dtypes = dict(dtypes) if dtypes is not None else {}
        self.roles = roles if roles is not None else Roles()
        self.lineage = lineage
        self.producing_steps = producing_steps

    @classmethod
    def from_frame(cls, frame: "Frame") -> "SchemaInfo":
        """Build a `SchemaInfo` snapshot from an internal `Frame`.

        Only ``columns`` and ``dtypes`` are filled in. ``roles`` is empty and
        ``lineage`` and ``producing_steps`` are ``None``; the recipe sets all
        three itself during a fit.
        """
        from mlfeatures import _frame

        dtypes = {name: str(dtype) for name, dtype in _frame.dtypes(frame).items()}
        return cls(columns=_frame.columns(frame), dtypes=dtypes)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SchemaInfo(columns={self.columns!r})"


class Roles:
    """Column → set of role names.

    Every column carries **zero or more** roles stored as a ``set[str]``. The
    standard roles are ``"target"``, ``"feature"``, and ``"case_weights"``;
    users may register arbitrary custom roles. A column may hold several roles
    at once, but nonsensical combinations are rejected at assignment time:
    ``"target"`` and ``"feature"`` cannot share a column.

    A column that carries no role is not selected by role selectors. Note that
    when a recipe declares no ``"feature"`` columns at all, fitting assigns the
    role to every column that is not a target or a case-weights column, custom
    roles included; declare features explicitly to keep, say, an ID column
    out of them.

    Examples
    --------
    Construct from a column → roles mapping, then query it:

    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Roles
    ```

    ```{python}
    roles = Roles(
        {"price": {"target"}, "age": {"feature"}, "city": {"feature"}}
    )
    roles.columns_with_role("feature"), roles.of("price")
    ```
    """

    #: Role pairs that may not co-occur on a single column.
    _INCOMPATIBLE: tuple[frozenset[str], ...] = (frozenset({"target", "feature"}),)

    #: The roles held by each column, keyed by column name. A column with no
    #: roles has no entry.
    _roles: dict[str, set[str]]

    #: Whether ``add()`` and ``remove()`` raise. Set on the read-only view a
    #: recipe hands out; ``copy()`` always returns a mutable instance.
    _frozen: bool

    def __init__(self, mapping: dict[str, set[str]] | None = None):
        self._roles = {}
        self._frozen = False
        if mapping is not None:
            for column, roles in mapping.items():
                for role in roles:
                    self.add(column, role)

    def _check_mutable(self) -> None:
        if self._frozen:
            raise TypeError(
                "these roles are a read-only copy of the recipe's roles; "
                "use set_roles() or the role operations on the recipe, "
                "or copy() for a mutable copy"
            )

    def add(self, column: str, role: str) -> None:
        """Assign ``role`` to ``column``, rejecting incompatible combinations."""
        self._check_mutable()
        # Validate before touching ``_roles`` so a rejected add cannot register
        # the column at all. Today the check can only fire for a column that is
        # already present, but registering first would leave an empty role set
        # behind the moment ``_INCOMPATIBLE`` grows a single-role entry.
        candidate = self._roles.get(column, set()) | {role}
        for pair in self._INCOMPATIBLE:
            if pair <= candidate:
                conflicting = ", ".join(sorted(pair))
                raise ValueError(
                    f"column {column!r} cannot hold incompatible roles: {conflicting}"
                )
        self._roles.setdefault(column, set()).add(role)

    def remove(self, column: str, role: str) -> None:
        """Remove ``role`` from ``column``. No-op if the column or role is absent."""
        self._check_mutable()
        if column in self._roles:
            self._roles[column].discard(role)
            if not self._roles[column]:
                del self._roles[column]

    def of(self, column: str) -> set[str]:
        """Roles assigned to ``column`` (empty set if none)."""
        return set(self._roles.get(column, set()))

    def columns_with_role(self, role: str) -> list[str]:
        """Columns carrying ``role``, in insertion order."""
        return [col for col, roles in self._roles.items() if role in roles]

    def role_names(self) -> list[str]:
        """Every role name currently in use, sorted. Empty if no roles assigned."""
        return sorted({role for roles in self._roles.values() for role in roles})

    def all_columns(self) -> list[str]:
        """Every column carrying at least one role, in insertion order."""
        return list(self._roles)

    def copy(self, *, read_only: bool = False) -> "Roles":
        """A deep copy, safe to mutate independently.

        With ``read_only=True`` the copy's ``add()`` and ``remove()`` raise.
        """
        new = Roles({col: set(roles) for col, roles in self._roles.items()})
        new._frozen = read_only
        return new

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Roles):
            return NotImplemented
        return self._roles == other._roles

    def __repr__(self) -> str:  # pragma: no cover
        return f"Roles({self._roles!r})"


def _apply_role_assignments(  # pyright: ignore[reportUnusedFunction]
    roles: Roles,
    assignments: dict[str, str | list[str] | None],
    *,
    check_col: Callable[[str], None] | None = None,
) -> None:
    """Apply a role→columns mapping to *roles*, with optional column validation."""
    for role, cols in assignments.items():
        if cols is None:
            continue
        for column in [cols] if isinstance(cols, str) else cols:
            if check_col is not None:
                check_col(column)
            roles.add(column, role)


class _AnyDtype:
    """Sentinel type for `ANY_DTYPE` (a singleton)."""

    _instance: "_AnyDtype | None" = None

    def __new__(cls) -> "_AnyDtype":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover
        return "ANY_DTYPE"

    def __contains__(self, _item: object) -> bool:
        # Every dtype is a member: ANY_DTYPE skips dtype validation entirely.
        return True


#: Sentinel returned by ``accepts_dtypes()`` to skip dtype validation entirely.
#: The correct choice for type-agnostic steps such as `StepDropZeroVariance` or
#: `StepDrop`.
ANY_DTYPE = _AnyDtype()

#: Return type of ``Step.accepts_dtypes()``: an explicit set of narwhals dtype
#: classes, or the `ANY_DTYPE` sentinel. Using narwhals' own dtype types
#: rather than a custom string taxonomy keeps the framework aligned with the one
#: backend-agnostic type system it already depends on. A column is accepted
#: when its dtype's *exact* class is in the set, so list concrete classes such
#: as ``nw.Int64``; an abstract base like ``nw.dtypes.NumericType`` or
#: ``nw.dtypes.IntegerType`` matches nothing. Use `NUMERIC_DTYPES` for the
#: numeric family.
AcceptedDtypes = Union[set[type[nw.dtypes.DType]], _AnyDtype]

#: The concrete narwhals numeric dtypes (integers, floats, and ``Decimal``): a
#: convenience set for steps that only accept numeric columns (e.g.
#: `StepDropCorr`, `StepNormalize`). It is a plain module-level set, so copy it
#: (``set(NUMERIC_DTYPES)``) before adding to it. ``Decimal`` only occurs on
#: backends that have the type, such as polars; pandas stores decimals as
#: ``object``.
NUMERIC_DTYPES: set[type[nw.dtypes.DType]] = {
    nw.Int8,
    nw.Int16,
    nw.Int32,
    nw.Int64,
    nw.Int128,
    nw.UInt8,
    nw.UInt16,
    nw.UInt32,
    nw.UInt64,
    nw.UInt128,
    nw.Float32,
    nw.Float64,
    nw.Decimal,
}

#: Dtypes valid for a step whose logic is level-based (an imputed mode, a
#: recoded "unknown"/"other" level) rather than numeric or general discrete.
#: Shared by ``step_impute_mode``, ``step_unknown``, and ``step_other``, which
#: previously each redefined this set verbatim.
_CATEGORICAL_DTYPES: set[type[nw.dtypes.DType]] = {nw.Categorical, nw.Enum, nw.String}


class Mode(Enum):
    """Shape of a step's input/output relationship.

    Every concrete step class declares one ``mode``. The framework reads it for
    two things: how the step's columns are recorded in the column lineage, and
    whether ``keep_original_cols`` is fixed or configurable (see
    `KeepOriginals`).

    The members, with the value ``keep_original_cols`` takes for each:

    - ``ONE_TO_ONE``: each input column produces exactly one output column,
      e.g. `StepNormalize`. Fixed at ``False``.
    - ``ONE_TO_MANY``: each input column produces one or more output columns,
      e.g. `StepDummy`. Configurable, default ``False``.
    - ``MANY_TO_MANY``: a set of input columns produces a new set of output
      columns, e.g. `StepPca`. Configurable, default ``False``.
    - ``REMOVING``: input columns are dropped and none are created, e.g.
      `StepDropZeroVariance`. Fixed at ``False``.
    - ``ADDING``: new columns appear with no corresponding input, e.g.
      `StepIntercept`. Fixed at ``True``.
    - ``ROW_OP``: no change to the columns, e.g. `StepFilter`. Fixed at
      ``True``. Row steps only act while the recipe is fitted and are skipped
      by ``transform()`` on new data.
    - ``UNCONSTRAINED``: the relationship cannot be declared in advance, e.g.
      `StepMutate`. Fixed at ``True``.

    Examples
    --------
    Each step class declares its mode; for example `StepDummy` expands one
    column into many:

    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Mode, StepDummy
    ```

    ```{python}
    StepDummy.mode, [m.value for m in Mode]
    ```
    """

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"
    REMOVING = "removing"
    ADDING = "adding"
    ROW_OP = "row_op"
    UNCONSTRAINED = "unconstrained"


class KeepOriginals(Enum):
    """Whether a step's ``keep_original_cols`` is fixed or up to the caller.

    A step does not declare this: it follows from `Mode`, via
    `KeepOriginals.for_mode`. A step that turns its input columns into derived
    ones (``ONE_TO_MANY``, ``MANY_TO_MANY``) is ``CONFIGURABLE``, because keeping
    the sources beside the derived columns is a legitimate thing to want; every
    other mode has only one sensible answer, so ``Step.__init__`` fixes the value
    and ``keep_original_cols`` cannot be set to anything else. The attribute is a read-only
    property on every step, configurable or not: its value is settled when the
    step is constructed.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import KeepOriginals, Mode, StepDummy
    ```

    ```{python}
    KeepOriginals.for_mode(StepDummy.mode), KeepOriginals.for_mode(Mode.ROW_OP)
    ```
    """

    #: The step always retains the columns it already had (``keep_original_cols``
    #: is ``True``).
    KEEP = "keep"
    #: The step always replaces or drops its input columns
    #: (``keep_original_cols`` is ``False``).
    DISCARD = "discard"
    #: The caller chooses, through a ``keep_original_cols`` constructor argument.
    CONFIGURABLE = "configurable"

    @classmethod
    def for_mode(cls, mode: Mode) -> "KeepOriginals":
        """The policy that a step of ``mode`` must follow."""
        return _KEEP_POLICY[mode]

    def fixed_value(self) -> bool | None:
        """The mandated ``keep_original_cols``, or ``None`` if configurable."""
        if self is KeepOriginals.CONFIGURABLE:
            return None
        return self is KeepOriginals.KEEP


_KEEP_POLICY: dict[Mode, KeepOriginals] = {
    Mode.ONE_TO_ONE: KeepOriginals.DISCARD,
    Mode.REMOVING: KeepOriginals.DISCARD,
    Mode.ADDING: KeepOriginals.KEEP,
    Mode.ROW_OP: KeepOriginals.KEEP,
    Mode.UNCONSTRAINED: KeepOriginals.KEEP,
    Mode.ONE_TO_MANY: KeepOriginals.CONFIGURABLE,
    Mode.MANY_TO_MANY: KeepOriginals.CONFIGURABLE,
}
