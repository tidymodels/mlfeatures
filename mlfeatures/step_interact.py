"""Pairwise interaction columns: fit + apply.

`step_interact` is a **pairwise-only** port of R recipes' `step_interact`: it
does not build 3-way-or-higher interaction terms (`A:B:C`). For every
unordered pair among the selected columns it adds one column holding their
elementwise product, preserving readable names (`{a}{sep}{b}`) rather than
the anonymous, name-dropping expansion `sklearn.preprocessing.PolynomialFeatures
(interaction_only=True)` produces. A caller who needs an n-way term should
write it explicitly with `mlfeatures.step_mutate.apply_mutate` / `StepMutate`.

The *fit* half (`fit_interact()`) only needs the selected columns' names to
enumerate the pairs; there is nothing to estimate from the data. The *apply*
half (`apply_interact()`) computes each pairwise product. `StepInteract`
wraps both into the `Step` lifecycle so a `Recipe` can fit and apply them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

from mlfeatures import _frame
from mlfeatures.step import Step
from mlfeatures.types import (
    NUMERIC_DTYPES,
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    Roles,
    SchemaInfo,
    Selector,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = [
    "InteractFit",
    "fit_interact",
    "apply_interact",
    "StepInteract",
]


@dataclass(frozen=True)
class InteractFit:
    """The fitted result returned by `fit_interact()`.

    Attributes
    ----------
    columns
        Selected column names, in fit-time order.
    sep
        Separator joining each pair's two column names into the output name.
    pairs
        Every unordered pair of `columns`, in `itertools.combinations` order.
        Empty when fewer than 2 columns were selected.
    """

    columns: list[str]
    sep: str
    pairs: list[tuple[str, str]]


def fit_interact(selected: Frame, *, sep: str = "_x_") -> InteractFit:
    """Enumerate the pairwise interactions among ``selected``'s columns.

    There is no per-column statistic to estimate: interacting two columns
    only needs their names, so this just records the column order and the
    pairs `apply_interact()` will compute. Fewer than 2 columns yields an
    ``InteractFit`` with no pairs; `apply_interact()` passes the data frame
    through unchanged and warns.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the columns to interact. Native
        data frames are wrapped via `mlfeatures._frame.from_native()` at the recipe
        boundary.
    sep
        Separator joining each pair's two column names into the output
        column name.

    Returns
    -------
    InteractFit
        The column order and pair list `apply_interact()` needs.
    """
    columns = list(selected.columns)
    pairs = list(combinations(columns, 2))
    return InteractFit(columns=columns, sep=sep, pairs=pairs)


def apply_interact(
    data: Frame,
    fit_result: InteractFit,
    *,
    keep_original_cols: bool = False,
) -> Frame:
    """Add one product column per pair recorded in ``fit_result``.

    For every ``(a, b)`` in ``fit_result.pairs``, adds a column named
    ``f"{a}{fit_result.sep}{b}"`` holding the elementwise product ``a * b``.
    Nulls propagate through the multiplication: a null in either column
    produces a null product for that row. The source columns are dropped
    unless ``keep_original_cols``; every other column of ``data`` passes
    through unchanged.

    Parameters
    ----------
    data
        Internal `Frame` to transform.
    fit_result
        Output of `fit_interact()`.
    keep_original_cols
        When ``True``, the source columns are kept alongside the interaction
        columns instead of being dropped.

    Returns
    -------
    Frame
        ``data`` with one new column per pair. Backend (pandas/polars) is
        preserved end-to-end.

    Warns
    -----
    UserWarning
        When ``fit_result.pairs`` is empty (fewer than 2 columns were
        selected), no interaction columns can be produced and the data frame is
        returned unchanged.
    """
    if not fit_result.pairs:
        warnings.warn(
            "fewer than 2 columns were selected; no interaction columns were created",
            UserWarning,
            stacklevel=2,
        )
        return data

    data = data.with_columns(
        (nw.col(a) * nw.col(b)).alias(f"{a}{fit_result.sep}{b}")
        for a, b in fit_result.pairs
    )
    if not keep_original_cols:
        data = data.drop(fit_result.columns)
    return data


class StepInteract(Step):
    """Add the pairwise product of every pair of selected columns.

    For every unordered pair among the selected columns, adds one column holding their
    elementwise product, named by joining the two source names with ``sep``.
    ``n`` selected columns produce ``n * (n - 1) / 2`` interaction columns.

    This step is **pairwise only**: it does not build 3-way-or-higher
    interaction terms. A user who needs one can write it explicitly with
    `StepMutate`.

    See [Arithmetic](https://feaz-book.com/numeric-arithmetic) in *Feature
    Engineering A to Z* for background on products of columns (feature
    crosses).

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to interact. Accepts a single column name, a list of names, a
        mlfeatures selector such as ``Selectors.numeric_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the data frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only numeric columns
        are meaningful here.
    sep
        Separator joining each pair's two column names into the output column
        name. Default ``"_x_"``.
    keep_original_cols
        If ``True``, the source columns are kept alongside the interaction
        columns instead of being dropped. Default ``False``.
    role
        Role assigned to the new interaction columns (see `Roles`). Default
        ``"feature"``.
    id
        Stable step identifier. Defaults to ``"interact"``; a second step of
        this type in the same recipe becomes ``"interact_2"``, and passing an
        ``id`` already used in the recipe is an error.

    Examples
    --------
    Every pair of the three numeric columns gets its own product column:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepInteract
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepInteract(s.numeric_features()))
    recipe
    ```

    `fit()` records the pairs to interact:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` adds one product column per pair and drops the sources:

    ```{python}
    fitted.transform(df)
    ```

    A custom ``sep`` changes the generated names:

    ```{python}
    recipe = Recipe().add(StepInteract(s.numeric_features(), sep="_times_"))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps the source columns alongside the products:

    ```{python}
    recipe = Recipe().add(
        StepInteract(s.numeric_features(), keep_original_cols=True)
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.MANY_TO_MANY

    #: Populated by ``fit()``: the resolved columns and the column pairs to multiply.
    _estimates: InteractFit | None

    def __init__(
        self,
        columns: Selector,
        *,
        sep: str = "_x_",
        keep_original_cols: bool = False,
        role: str | None = "feature",
        id: str | None = None,
    ) -> None:
        self.columns = columns
        self.sep = sep
        super().__init__(
            id, "interact", keep_original_cols=keep_original_cols, output_role=role
        )

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Record the pairwise interactions among the selected columns."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.lineage, info.producing_steps
        )
        self._estimates = fit_interact(selected, sep=self.sep)
        dtypes_in = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes_in)
        out_names = [f"{a}{self.sep}{b}" for a, b in self._estimates.pairs]
        # Read the products' actual output dtype off the training data rather
        # than assuming Float64: two integer columns multiply to an integer
        # dtype on both backends. Aliased to positional temp names, never the
        # (possibly colliding) real output names: two different pairs can
        # legitimately produce the same `out_name` (see `produced_columns()`),
        # and aliasing two columns identically within one `with_columns()` call
        # raises a backend-specific error on its own, before the recipe's
        # centralized collision check gets a chance to reject it cleanly.
        if out_names:
            products = selected.with_columns(
                (nw.col(a) * nw.col(b)).alias(f"_interact_tmp_{i}")
                for i, (a, b) in enumerate(self._estimates.pairs)
            )
            product_dtypes = _frame.dtypes(products)
            out_dtypes = {
                name: str(product_dtypes[f"_interact_tmp_{i}"])
                for i, name in enumerate(out_names)
            }
        else:
            out_dtypes = {}
        self._schema_out = ColSchema(columns=out_names, dtypes=out_dtypes)
        self._output_parents = {
            f"{a}{self.sep}{b}": [a, b] for a, b in self._estimates.pairs
        }
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Add each pair's product column. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepInteract.transform() called before fit()")
        return apply_interact(
            new_data, self._estimates, keep_original_cols=self.keep_original_cols
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepInteract.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the interaction columns this step produced."""
        if self._schema_out is None:
            raise RuntimeError("StepInteract.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each interaction column descends from its pair of source columns."""
        if self._output_parents is None:
            raise RuntimeError("StepInteract.output_parents() called before fit()")
        return self._output_parents

    def produced_columns(self) -> list[str]:
        """Interaction column names in order, preserving duplicates (see ``Step``).

        Two different pairs can produce the same ``f"{a}{sep}{b}"`` name for an
        adversarial choice of column names and ``sep`` (e.g. ``sep="_x_"`` with
        the pairs ``("a_x", "b")`` and ``("a", "x_b")``). ``output_parents()`` is
        a dict, so such a collision would otherwise vanish there; this overrides
        the base-class default (which derives from ``output_parents()``'s keys)
        to preserve the duplicate so the recipe's centralized name-collision
        check can reject it.
        """
        return list(self.schema_out().columns)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Pairwise products are only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    def update_roles(self, roles: Roles) -> None:
        if self._output_parents is None:
            return
        if self.output_role is not None:
            for col in self._output_parents:
                roles.add(col, self.output_role)
        if self.keep_original_cols:
            return
        interacted = {
            parent for parents in self._output_parents.values() for parent in parents
        }
        for col in interacted:
            for role in list(roles.of(col)):
                roles.remove(col, role)

    @property
    def produces_sparse(self) -> bool:
        """Pairwise products are dense by construction."""
        return False
