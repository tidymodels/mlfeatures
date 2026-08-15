"""A stand-in for a third-party step, defined outside the ``shoez`` package.

Nothing here is a step shoez ships. ``StepScaleShift`` exists only so
``test_extensibility.py`` has something to push through every integration
surface a plugin would use, in shoez's own CI, without depending on a package
that lives in another repository.

The point is the *constraint*, not the arithmetic: this module may import from
``shoez`` and nothing else. `tests.test_extensibility.TestPublicApiOnly` parses
this file and fails if it reaches into a private module, so the day a step can
no longer be written against the public API, a test says so. The current escapes
are allowlisted there with the issue that tracks each.

Kept underscore-prefixed so pytest does not collect it as a test module.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw

from shoez import NUMERIC_DTYPES, ColSchema, Frame, SchemaInfo, Step

# Neither alias is re-exported from `shoez`, so a step cannot annotate the
# methods the ABC requires without reaching in here. Tracked by #197; the
# allowlist in test_extensibility.py is what keeps this from growing.
from shoez.types import AcceptedDtypes, Selector

__all__ = ["StepScaleShift"]


class StepScaleShift(Step):
    """Center each selected column, then add a constant.

    Each selected column ``x`` becomes ``(x - mean) + shift``. Deliberately
    trivial: the mean is the only thing estimated at fit time, which is enough
    to exercise the fit/transform split, and ``shift`` gives the step a
    hyperparameter to carry through serialization.
    """

    def __init__(
        self,
        columns: Selector,
        shift: float = 0.0,
        id: str | None = None,
    ):
        self.columns = columns
        self.shift = shift
        self.id = id if id is not None else "scale_shift"
        self.trained = False
        self.keep_original_cols = False
        self.output_role = None
        self._estimates: dict[str, float] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "StepScaleShift":
        """Estimate each selected column's mean.

        Resolves the selection through the public ``resolve_selection()`` rather
        than ``shoez._frame.select()``, and reads dtypes off narwhals' own
        ``frame.schema``. Both are the routes available to a plugin.
        """
        names = self.resolve_selection(training_data, info)
        features = training_data.select(names)
        self._estimates = {col: float(features[col].mean()) for col in features.columns}
        self._schema_in = ColSchema(
            columns=features.columns,
            dtypes={name: str(dt) for name, dt in features.schema.items()},
        )
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Apply the stored means and shift. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepScaleShift.transform() called before fit()")
        return new_data.with_columns(
            *(
                ((nw.col(col) - mean) + self.shift).alias(col)
                for col, mean in self._estimates.items()
            )
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepScaleShift.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Same columns, now float."""
        if self._schema_in is None:
            raise RuntimeError("StepScaleShift.schema_out() called before fit()")
        return ColSchema(
            columns=self._schema_in.columns,
            dtypes=dict.fromkeys(self._schema_in.columns, "Float64"),
        )

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Centering is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """Centering shifts zeros off zero, but the output stays dense."""
        return False
