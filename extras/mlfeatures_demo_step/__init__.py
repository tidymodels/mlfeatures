"""An out-of-tree step, written the way a plugin author would write one.

This package is not shipped and is not part of mlfeatures. It exists to hold the
public API to its promise: a third party must be able to write a complete,
strictly-typed step using only the names in ``mlfeatures.__all__``.

Two things enforce that, and neither works if this step lives under ``tests/``:

* ``tests/`` is excluded from both mypy and pyright, so a fixture there proves
  nothing about strict-mode typability. This directory is added to both
  checkers' scope instead.
* ``tests/test_out_of_tree_step.py`` scans this module's imports and fails if
  any of them reach into a private module (``mlfeatures.types``, ``mlfeatures._frame``).

So the awkward-looking constraint below is the point of the file: everything
comes from the top-level ``mlfeatures`` namespace, and nothing else does.

See ``design/extensibility.md`` and issue #197.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

from mlfeatures import (
    NUMERIC_DTYPES,
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    SchemaInfo,
    Selector,
    Step,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["StepDemoMaxAbs"]


class StepDemoMaxAbs(Step):
    """Scale numeric columns to ``[-1, 1]`` by dividing by their maximum absolute value.

    A deliberately small transform: the point of this step is its imports and
    its type annotations, not its statistics.

    Parameters
    ----------
    columns
        Columns to scale. Accepts a single column name, a list of names, a
        mlfeatures selector such as ``Selectors.numeric_features()``, or a narwhals
        or polars selector. Only numeric columns are meaningful here.
    id
        Stable step identifier. Defaults to ``"demo_max_abs"``.
    """

    mode = Mode.ONE_TO_ONE

    def __init__(self, columns: Selector, *, id: str | None = None) -> None:
        self.columns = columns
        super().__init__(id, "demo_max_abs")
        self._estimates: dict[str, float] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> Self:
        """Estimate each selected column's maximum absolute value.

        This is the sanctioned selector-resolution route: ``resolve_selection``
        gives the column names and narwhals' own ``schema`` gives the dtypes,
        so no private mlfeatures module is needed.
        """
        names = self.resolve_selection(training_data, info)
        selected = training_data.select(names)
        dtypes = {name: str(dt) for name, dt in selected.schema.items()}

        estimates: dict[str, float] = {}
        for col in selected.columns:
            scale = float(selected[col].abs().max())
            # A constant-zero column has nothing to scale by; leave it alone
            # rather than dividing by zero.
            if scale != 0.0:
                estimates[col] = scale

        self._estimates = estimates
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Divide by the stored maxima. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepDemoMaxAbs.transform() called before fit()")
        exprs = [
            (nw.col(col) / scale).alias(col) for col, scale in self._estimates.items()
        ]
        return new_data.with_columns(*exprs)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepDemoMaxAbs.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Same columns; every scaled one comes out float."""
        if self._schema_in is None or self._estimates is None:
            raise RuntimeError("StepDemoMaxAbs.schema_out() called before fit()")
        scaled = set(self._estimates)
        dtypes = {
            col: ("Float64" if col in scaled else self._schema_in.dtypes[col])
            for col in self._schema_in.columns
        }
        return ColSchema(columns=self._schema_in.columns, dtypes=dtypes)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Scaling is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """Dividing by a constant preserves zeros, but output stays dense."""
        return False
