"""Shared type aliases and sentinels for the step contract.

Several of these are placeholders for types owned by other foundation issues
(notably the schema subsystem — see ``design/schema.md``). They exist so that
the :mod:`shoez.step` contract is importable and type-checkable today. The
concrete implementations will replace the placeholder bodies here without
changing the names the rest of the team codes against.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import narwhals.stable.v2 as nw

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    #: A native tabular structure at the package boundary — concretely a pandas or
    #: polars ``DataFrame``. Steps do **not** see this type; it is converted to a
    #: :data:`Frame` on entry to the recipe and back at the sklearn boundary.
    DataFrame = Union["pd.DataFrame", "pl.DataFrame"]
else:
    DataFrame = Any

#: The internal frame type that flows between steps. A backend-agnostic narwhals
#: ``DataFrame`` wrapping the user's native frame; steps write their data ops
#: against this via :mod:`shoez._frame`. See ``design/ecosystem.md``.
Frame: TypeAlias = nw.DataFrame[Any]


class ColSchema:
    """Column schema: names and dtypes for a column set.

    Minimal stand-in for the full type defined by the schema subsystem
    (``design/schema.md``) — it carries only the column names and their dtype
    names, enough for a step to describe what it consumed (``schema_in``) and
    produced (``schema_out``). Roles, source, and sparsity land with the real
    schema subsystem.
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
    """Schema-derived context handed to :meth:`Step.fit`.

    Minimal stand-in for the full type defined by the schema subsystem
    (``design/schema.md``). For now it carries the current column names and
    dtype names; roles and the lineage DAG land with the real schema subsystem.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        dtypes: dict[str, str] | None = None,
    ):
        self.columns = list(columns) if columns is not None else []
        self.dtypes = dict(dtypes) if dtypes is not None else {}

    @classmethod
    def from_frame(cls, frame: "Frame") -> "SchemaInfo":
        """Build a :class:`SchemaInfo` snapshot from an internal :data:`Frame`."""
        from shoez import _frame

        dtypes = {name: str(dtype) for name, dtype in _frame.dtypes(frame).items()}
        return cls(columns=_frame.columns(frame), dtypes=dtypes)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SchemaInfo(columns={self.columns!r})"


class Roles:
    """Column → set of role labels. See ``design/roles.md``.

    Every column carries **zero or more** roles stored as a ``set[str]``. The
    standard roles are ``"outcome"``, ``"predictor"``, and ``"case_weights"``;
    users may register arbitrary custom roles. A column may hold several roles
    at once, but nonsensical combinations are rejected at assignment time.
    """

    #: Role pairs that may not co-occur on a single column.
    _INCOMPATIBLE: tuple[frozenset[str], ...] = (frozenset({"outcome", "predictor"}),)

    def __init__(self, mapping: dict[str, set[str]] | None = None):
        self._roles: dict[str, set[str]] = {}
        if mapping is not None:
            for column, roles in mapping.items():
                for role in roles:
                    self.add(column, role)

    def add(self, column: str, role: str) -> None:
        """Assign ``role`` to ``column``, rejecting incompatible combinations."""
        current = self._roles.setdefault(column, set())
        candidate = current | {role}
        for pair in self._INCOMPATIBLE:
            if pair <= candidate:
                conflicting = ", ".join(sorted(pair))
                raise ValueError(
                    f"column {column!r} cannot hold incompatible roles: {conflicting}"
                )
        current.add(role)

    def of(self, column: str) -> set[str]:
        """Roles assigned to ``column`` (empty set if none)."""
        return set(self._roles.get(column, set()))

    def columns_with_role(self, role: str) -> list[str]:
        """Columns carrying ``role``, in insertion order."""
        return [col for col, roles in self._roles.items() if role in roles]

    def copy(self) -> "Roles":
        """A deep copy safe to mutate independently."""
        return Roles({col: set(roles) for col, roles in self._roles.items()})

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Roles):
            return NotImplemented
        return self._roles == other._roles

    def __repr__(self) -> str:  # pragma: no cover
        return f"Roles({self._roles!r})"


class _AnyDtype:
    """Sentinel type for :data:`ANY_DTYPE` (a singleton)."""

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
#: classes, or the :data:`ANY_DTYPE` sentinel. Using narwhals' own dtype types
#: rather than a custom string taxonomy keeps the framework aligned with the one
#: backend-agnostic type system it already depends on (see ``design/schema.md``).
AcceptedDtypes = Union[set[type[nw.dtypes.DType]], _AnyDtype]

#: The narwhals numeric dtypes — a convenience set for steps that only accept
#: numeric columns (e.g. :class:`~shoez.step_drop_corr.StepDropCorr`, ``step_normalize``).
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
    """

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"
    REMOVING = "removing"
    ADDING = "adding"
    ROW_OP = "row_op"
    UNCONSTRAINED = "unconstrained"
