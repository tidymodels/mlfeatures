"""Shared type aliases and sentinels for the step contract.

Several of these are placeholders for types owned by other foundation issues
(notably the schema subsystem — see ``design/schema.md``). They exist so that
the `shoez.step` contract is importable and type-checkable today. The
concrete implementations will replace the placeholder bodies here without
changing the names the rest of the team codes against.
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
    from polars.selectors import Selector as PlSelector

    from shoez.schema import Schema
    from shoez.selectors import ShoezSelector
    from shoez.step import Step

    #: A native tabular structure at the package boundary — concretely a pandas or
    #: polars ``DataFrame``. Steps do **not** see this type; it is converted to a
    #: `Frame` on entry to the recipe and back at the sklearn boundary.
    DataFrame = Union["pd.DataFrame", "pl.DataFrame"]
else:
    DataFrame = Any

    class PlSelector(AbstractBackend):
        # ``isinstance(x, PlSelector)`` is True for a ``polars.selectors.Selector``
        # without importing polars unless it is already loaded (see _databackend).
        _backends = [("polars.selectors", "Selector")]


#: The internal frame type that flows between steps. A backend-agnostic narwhals
#: ``DataFrame`` wrapping the user's native frame; steps write their data ops
#: against this via `shoez._frame`. See ``design/ecosystem.md``.
Frame: TypeAlias = nw.DataFrame[Any]

#: The type of every step's ``columns`` parameter: a single column name, an
#: explicit list of names, a shoez selector such as
#: ``Selectors.numeric_features()`` (see `Selectors` for the full set), or a
#: narwhals or polars selector (``narwhals.selectors.numeric()``,
#: ``polars.selectors.numeric()``). Any selector is resolved against the frame's
#: schema at fit time, so the same selector works on any backend.
Selector: TypeAlias = Union[
    str, Sequence[str], NwSelector, "PlSelector", "ShoezSelector"
]

# Listed so the selector aliases (``NwSelector`` is a renamed import; ``PlSelector``
# is a conditional import under TYPE_CHECKING) count as explicit re-exports for
# strict type-checkers; the other names are module-level definitions and
# re-exportable regardless.
__all__ = [
    "ANY_DTYPE",
    "NUMERIC_DTYPES",
    "AcceptedDtypes",
    "ColSchema",
    "DataFrame",
    "Frame",
    "Mode",
    "NwSelector",
    "PlSelector",
    "SchemaInfo",
    "Selector",
]


class ColSchema:
    """Column schema: the names and dtypes a step reports for a column set.

    Carries the column names and their dtype names — enough for a step to
    describe what it consumed (``schema_in``) and produced (``schema_out``).
    This is the lightweight step-facing report; the recipe-side column-lineage
    DAG that tracks roles, source, sparsity, and provenance is
    `shoez.schema.Schema` (``design/schema.md``).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    from shoez import ColSchema
    ```

    ```{python}
    schema = ColSchema(columns=["age", "city"], dtypes={"age": "Int64", "city": "String"})
    schema.columns, schema.dtypes
    ```
    """

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
    recipe-side column-lineage DAG (`shoez.schema.Schema`, ``design/schema.md``)
    is maintained separately by the recipe and is not handed to steps for their
    own logic; ``schema`` and ``producing_steps`` are populated by the recipe
    purely so provenance selectors (``has_source`` / ``has_ancestor``) can
    resolve against the DAG at fit time. Both are ``None`` outside a recipe fit.

    Examples
    --------
    Build a snapshot from a frame with `from_frame`; steps read ``columns``,
    ``dtypes``, and ``roles`` off it during ``fit()``:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import SchemaInfo, _frame
    ```

    ```{python}
    frame = _frame.from_native(pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}))
    info = SchemaInfo.from_frame(frame)
    info.columns, info.dtypes
    ```
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        dtypes: dict[str, str] | None = None,
        roles: Roles | None = None,
        schema: "Schema | None" = None,
        producing_steps: "dict[str, Step] | None" = None,
    ):
        self.columns = list(columns) if columns is not None else []
        self.dtypes = dict(dtypes) if dtypes is not None else {}
        self.roles: Roles = roles if roles is not None else Roles()
        #: The live column-lineage DAG, set by the recipe during fit (else None).
        self.schema: "Schema | None" = schema
        #: Map of step id → fitted step, set by the recipe during fit (else None).
        #: Lets ``has_source(step_type=...)`` map ids back to step instances.
        self.producing_steps: "dict[str, Step] | None" = producing_steps

    @classmethod
    def from_frame(cls, frame: "Frame") -> "SchemaInfo":
        """Build a `SchemaInfo` snapshot from an internal `Frame`."""
        from shoez import _frame

        dtypes = {name: str(dtype) for name, dtype in _frame.dtypes(frame).items()}
        return cls(columns=_frame.columns(frame), dtypes=dtypes)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SchemaInfo(columns={self.columns!r})"


class Roles:
    """Column → set of role names. See ``design/roles.md``.

    Every column carries **zero or more** roles stored as a ``set[str]``. The
    standard roles are ``"target"``, ``"feature"``, and ``"case_weights"``;
    users may register arbitrary custom roles. A column may hold several roles
    at once, but nonsensical combinations are rejected at assignment time.

    Examples
    --------
    Construct from a column → roles mapping, then query it:

    ```{python}
    #| echo: false
    #| output: false
    from shoez import Roles
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

    def __init__(self, mapping: dict[str, set[str]] | None = None):
        self._roles: dict[str, set[str]] = {}
        if mapping is not None:
            for column, roles in mapping.items():
                for role in roles:
                    self.add(column, role)

    def add(self, column: str, role: str) -> None:
        """Assign ``role`` to ``column``, rejecting incompatible combinations."""
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

    def copy(self) -> "Roles":
        """A deep copy safe to mutate independently."""
        return Roles({col: set(roles) for col, roles in self._roles.items()})

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
#: The correct choice for type-agnostic steps such as ``step_zv`` or ``step_rm``.
ANY_DTYPE = _AnyDtype()

#: Return type of ``Step.accepts_dtypes()``: an explicit set of narwhals dtype
#: classes, or the `ANY_DTYPE` sentinel. Using narwhals' own dtype types
#: rather than a custom string taxonomy keeps the framework aligned with the one
#: backend-agnostic type system it already depends on (see ``design/schema.md``).
AcceptedDtypes = Union[set[type[nw.dtypes.DType]], _AnyDtype]

#: The narwhals numeric dtypes — a convenience set for steps that only accept
#: numeric columns (e.g. `StepDropCorr`, ``step_normalize``).
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


class Mode(Enum):
    """Shape of a step's input/output relationship.

    Informs how the framework records lineage. See ``design/step-contract.md``.

    Examples
    --------
    Each step class declares its mode; for example `StepDummy` expands one
    column into many:

    ```{python}
    #| echo: false
    #| output: false
    from shoez import Mode, StepDummy
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
    and ``keep_original_cols`` is read-only. See ``design/step-contract.md``.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    from shoez import KeepOriginals, Mode, StepDummy
    ```

    ```{python}
    KeepOriginals.for_mode(StepDummy.mode), KeepOriginals.for_mode(Mode.ROW_OP)
    ```
    """

    #: The step always retains its input columns.
    KEEP = "keep"
    #: The step always drops or consumes its input columns.
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
