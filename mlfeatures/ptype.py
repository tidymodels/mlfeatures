"""The ptype snapshot and boundary validation. See ``design/lifecycle.md``.

A **ptype** (prototype) is a zero-row structural snapshot of the training data:
column names, dtypes, and role assignments. It is captured at ``Recipe.fit()``
and stored on the resulting ``FittedRecipe``; both ``fit()`` and
``transform()`` validate the incoming frame against it before any step runs.
This catches the two most common boundary failures: the wrong dataframe passed
in, or a column silently changing dtype between training and serving.

The name→dtype half reuses narwhals' own :class:`narwhals.Schema` (an ordered
name→dtype mapping that is backend-agnostic across pandas and polars). Roles and
the per-role *requirements* (which columns must be present at ``transform()``
time) are mlfeatures concepts narwhals knows nothing about, so they live here.
"""

from __future__ import annotations

from dataclasses import dataclass

import narwhals.stable.v2 as nw

from mlfeatures._frame import dtypes as frame_dtypes
from mlfeatures.types import Frame, Roles

#: Roles whose transform-time requirement is fixed and cannot be configured via
#: ``update_role_requirements``: a ``feature`` is always required at transform,
#: a ``target`` never is (it is unknown at serving time).
_NON_CONFIGURABLE_ROLES = ("feature", "target")

#: Default transform-time requirement for configurable roles. Any role not
#: listed (including custom roles and unrolled columns) defaults to ``True``.
_DEFAULT_REQUIRED = {"case_weights": False}


def column_transform_required(
    column: str,
    roles: Roles,
    overrides: dict[str, bool],
) -> bool:
    """Whether ``column`` must be present in new data at ``transform()`` time.

    Hardcoded roles win first: a ``feature`` is always required, a
    ``target`` never is. Otherwise an explicit ``overrides`` entry applies; with
    no override the requirement is the most-demanding role default (a column is
    required if *any* of its roles defaults to required). Unrolled columns
    default to required.
    """
    column_roles = roles.of(column)
    if "feature" in column_roles:
        return True
    if "target" in column_roles:
        return False
    if column in overrides:
        return overrides[column]
    if not column_roles:
        return True
    return any(_DEFAULT_REQUIRED.get(role, True) for role in column_roles)


@dataclass(frozen=True)
class Ptype:
    """A structural snapshot of the training data used for boundary checks.

    Captured at ``Recipe.fit()`` and exposed as ``FittedRecipe.ptype``. Holds the
    training :class:`narwhals.Schema` (column names and dtypes), the role
    assignments, and a precomputed per-column flag for whether each column is
    required at ``transform()`` time.

    Attributes
    ----------
    schema
        The training frame's narwhals schema: an ordered name→dtype mapping.
    roles
        Role assignments captured at fit (see ``design/roles.md``).
    transform_required
        Per-column flag: ``True`` if the column must be present in new data at
        ``transform()`` time. Computed once at fit from role requirements and
        any ``update_role_requirements`` overrides.

    Examples
    --------
    The ptype is built by the recipe; read it back from a fitted recipe:

    ```{python}
    import pandas as pd
    from mlfeatures import Recipe, StepNormalize
    from mlfeatures import Selectors as s

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
    rec = Recipe(target="y", features="x").add(
        StepNormalize(s.numeric_features())
    )
    rec = rec.fit(df)
    rec.ptype.schema.names()
    ```
    """

    schema: nw.Schema
    roles: Roles
    transform_required: dict[str, bool]

    @classmethod
    def from_frame(
        cls,
        frame: Frame,
        roles: Roles,
        overrides: dict[str, bool],
    ) -> "Ptype":
        """Snapshot ``frame`` into a ptype given role assignments and overrides.

        The per-column transform-time requirement is precomputed here from the
        role defaults and any ``update_role_requirements`` ``overrides``.
        """
        schema = nw.Schema(frame_dtypes(frame))
        return cls(
            schema=schema,
            roles=roles.copy(),
            transform_required={
                col: column_transform_required(col, roles, overrides)
                for col in schema.names()
            },
        )

    def check(self, frame: Frame, *, context: str) -> None:
        """Validate ``frame`` against this snapshot.

        ``context`` is ``"fit"`` or ``"transform"``. At ``"fit"`` every ptype
        column is required; at ``"transform"`` only columns flagged in
        ``transform_required`` are. In both contexts, columns that *are* present
        must match the snapshot dtype. Extra columns beyond the snapshot are
        ignored.

        Raises
        ------
        ValueError
            If a required column is missing, or a present column's dtype differs
            from the snapshot.
        """
        present = frame_dtypes(frame)

        if context == "fit":
            required = list(self.schema.names())
        else:
            required = [c for c in self.schema.names() if self.transform_required[c]]

        missing = [c for c in required if c not in present]
        if missing:
            cols = ", ".join(repr(c) for c in missing)
            raise ValueError(f"missing required column(s) at {context} time: {cols}")

        mismatched = [
            f"{name!r} (expected {self.schema[name]}, got {present[name]})"
            for name in self.schema.names()
            if name in present and present[name] != self.schema[name]
        ]
        if mismatched:
            raise ValueError(
                "dtype mismatch against the training ptype: " + "; ".join(mismatched)
            )
