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

import narwhals.stable.v1 as nw

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
    """Column schema: names, dtypes, roles, source, and sparsity for a column set.

    Placeholder for the type defined by the schema subsystem
    (``design/schema.md``). Concrete fields land with that issue.
    """


class SchemaInfo:
    """Schema-derived context handed to :meth:`Step.fit`.

    Placeholder for the type defined by the schema subsystem
    (``design/schema.md``).
    """


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
