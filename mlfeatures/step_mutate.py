"""Arbitrary column mutation: the ``UNCONSTRAINED`` escape hatch.

`step_mutate` is the ``UNCONSTRAINED`` prototype,
modelled on ``step_mutate`` from the R ``recipes`` package. It applies caller-supplied expressions to the frame, so its
input/output relationship cannot be declared in advance: a mutation may add new
columns, overwrite existing ones in place, or change a column's dtype. Because
of that, `StepMutate` reads the concrete effect off the training frame at
``fit()`` time rather than declaring a fixed shape up front.

The *apply* half lives in `apply_mutate()` as a plain function so the mutation
logic can be exercised on its own; `StepMutate` wraps it in the `Step` lifecycle
so a `Recipe` can fit and apply it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from mlfeatures import _frame
from mlfeatures.step import Step
from mlfeatures.types import (
    ANY_DTYPE,
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    Roles,
    SchemaInfo,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

__all__ = ["apply_mutate", "StepMutate"]

#: A mutation maps the internal `Frame` to the value of one output column: a
#: narwhals expression or series named by the mutation's key.
Mutation = Callable[[Frame], Any]


def apply_mutate(selected: Frame, mutations: Mapping[str, Mutation]) -> Frame:
    """Apply each named mutation to ``selected``.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals). Native frames are wrapped via
        `mlfeatures._frame.from_native()` at the recipe boundary.
    mutations
        Mapping from output column name to a callable taking ``selected`` and
        returning that column's value (a narwhals expression or series). A name
        that already exists overwrites the column in place; a new name appends a
        column. Applied together via a single ``with_columns`` so the mutations
        all see the original frame, matching ``dplyr::mutate`` /
        ``pl.DataFrame.with_columns`` semantics.

    Returns
    -------
    Frame
        ``selected`` with the mutations applied. Backend (pandas/polars) is
        preserved end-to-end.
    """
    return selected.with_columns(
        **{name: fn(selected) for name, fn in mutations.items()}
    )


class StepMutate(Step):
    """Apply arbitrary column expressions: the escape hatch.

    Each supplied mutation either adds a new column or overwrites an existing
    one, so the exact column change is only known after the mutations run on
    the training frame. `fit()` applies them once to read off which columns
    were *added* (and their dtypes) versus modified in place, then re-applies
    the same expressions at transform time.

    New columns are attributed to no declared parents (``output_parents`` maps
    each to ``[]``): the expressions are opaque callables, so mlfeatures does not try
    to infer which inputs a mutation read. This is the deliberate trade-off of
    an escape hatch: full flexibility, coarse lineage.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    mutations
        Mapping from output column name to a callable taking the frame and
        returning that column's value (a narwhals expression or series). Must be
        a non-empty ``dict`` of ``str`` names to callables.
    role
        Role assigned to any *new* columns the mutations create (see `Roles`).
        Columns overwritten in place keep their existing role. Default
        ``"feature"``.
    id
        Stable step identifier. Defaults to ``"mutate"``; a second step of
        this type in the same recipe becomes ``"mutate_2"``, and passing an
        ``id`` already used in the recipe is an error.

    Examples
    --------
    ``x_sq`` is added from an expression over ``x``; ``x`` itself is overwritten
    in place:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepMutate
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(
        StepMutate({"x_sq": lambda df: df["x"] ** 2, "x": lambda df: df["x"] + 1})
    )
    recipe
    ```

    `fit()` runs the mutations once on the training frame to read off which
    columns they add and with what dtypes:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` applies them together, so every mutation sees the original
    frame; ``x_sq`` squares the incoming ``x``, not the overwritten one:

    ```{python}
    fitted.transform(df)
    ```
    """

    mode = Mode.UNCONSTRAINED

    def __init__(
        self,
        mutations: Mapping[str, Mutation],
        *,
        role: str | None = "feature",
        id: str | None = None,
    ) -> None:
        if not isinstance(mutations, dict) or not mutations:
            raise TypeError(
                "'mutations' must be a non-empty mapping of name to callable"
            )
        for name, fn in mutations.items():
            if not callable(fn):
                raise TypeError(
                    f"mutation {name!r} must be callable; got {type(fn).__name__!r}"
                )
        self.mutations = dict(mutations)
        super().__init__(id, "mutate", output_role=role)

    def to_state(self) -> dict[str, object]:
        """Not supported: ``step_mutate`` carries user callables it needs at transform.

        Unlike `StepFilter`, whose predicate is inert once fitted, ``StepMutate``
        applies its callables to *new* data at ``transform`` time, so they are
        load-bearing code that JSON cannot represent. Serializing would silently drop
        them and produce a recipe that no longer computes the mutations, so it fails
        loudly instead. Use ``pickle`` for recipes containing ``step_mutate``, which
        requires the mutations to be picklable: define them as module-level functions,
        since ``pickle`` cannot handle a ``lambda``.
        """
        raise TypeError(
            "step_mutate carries user-supplied callables and cannot be "
            "serialized to JSON; pickle the recipe instead"
        )

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Apply the mutations once to read off which columns are new."""
        in_cols = _frame.columns(training_data)
        in_dtypes = {name: str(dt) for name, dt in _frame.dtypes(training_data).items()}
        self._schema_in = ColSchema(columns=in_cols, dtypes=in_dtypes)
        produced = apply_mutate(training_data, self.mutations)
        out_dtypes = {name: str(dt) for name, dt in _frame.dtypes(produced).items()}
        self._schema_out = ColSchema(
            columns=_frame.columns(produced), dtypes=out_dtypes
        )
        # Added columns (not present before) are "produced"; overwritten columns
        # are in-place edits, not new columns. Escape-hatch expressions are
        # opaque, so a produced column declares no parents.
        in_set = set(in_cols)
        self._output_parents = {
            name: [] for name in _frame.columns(produced) if name not in in_set
        }
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Re-apply the stored mutations. Never re-estimates."""
        if self._schema_out is None:
            raise RuntimeError("StepMutate.transform() called before fit()")
        return apply_mutate(new_data, self.mutations)

    def schema_in(self) -> ColSchema:
        """Schema of the frame this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepMutate.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the frame this step produced. Known after ``fit()``."""
        if self._schema_out is None:
            raise RuntimeError("StepMutate.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """New columns descend from no declared parents (opaque expressions)."""
        if self._output_parents is None:
            raise RuntimeError("StepMutate.output_parents() called before fit()")
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """An escape hatch operates on whatever columns the expressions read."""
        return ANY_DTYPE

    def update_roles(self, roles: Roles) -> None:
        if self._output_parents and self.output_role is not None:
            for col in self._output_parents:
                roles.add(col, self.output_role)

    @property
    def produces_sparse(self) -> bool:
        """``step_mutate`` never produces sparse output."""
        return False
