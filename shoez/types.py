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

#: Return type of ``Step.accepts_dtypes()``: an explicit set of dtype names, or
#: the :data:`ANY_DTYPE` sentinel.
AcceptedDtypes = Union[set[str], _AnyDtype]


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
