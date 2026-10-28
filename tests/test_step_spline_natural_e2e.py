"""End-to-end lifecycle tests for StepSplineNatural: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepSplineNatural + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic. Structured to parallel ``tests/test_step_polynomial_e2e.py``.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures import Mode, Recipe, SchemaInfo, StepNormalize, StepSplineNatural

X_1_TO_10 = list(range(1, 11))


def _data() -> dict[str, list[float]]:
    return {
        "x": [float(v) for v in X_1_TO_10],
        "const": [7.0] * 10,
        "b": [5.0, 1.0, 4.0, 1.0, 9.0, 2.0, 6.0, 3.0, 8.0, 0.0],
    }


# ---------------------------------------------------------------------------
# Recipe-level end-to-end behaviour
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_pandas_expands_in_place(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepSplineNatural(["x"], deg_free=3)])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["x_1", "x_2", "x_3", "const", "b"]

    def test_polars_expands_in_place(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepSplineNatural(["x"], deg_free=2))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["x_1", "x_2", "const", "b"]

    def test_natural_values_match_r_splines2(self) -> None:
        # step_spline_natural(x, deg_free = 4): the same ground truth used in
        # the unit tests.
        expected = np.array(
            [
                [0.31138545953360763, 0.0146319158664838, 0.0],
                [0.49108367626886135, 0.1170553269318701, 0.0],
            ]
        )
        df = pd.DataFrame({"x": [float(v) for v in X_1_TO_10]})
        out = Recipe(steps=[StepSplineNatural(["x"], deg_free=4)]).fit(df).transform(df)
        np.testing.assert_allclose(
            out[["x_1", "x_2", "x_3"]].to_numpy()[1:3], expected, atol=1e-9
        )

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepSplineNatural(["x"], deg_free=2)]).fit(df).transform(df)
        assert out["b"].tolist() == _data()["b"]
        assert out["const"].tolist() == [7.0] * 10

    def test_constant_column_warns_and_is_left_unchanged(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepSplineNatural(["x", "const"], deg_free=2)])
        with pytest.warns(UserWarning) as record:
            fitted = recipe.fit(df)
        assert str(record[0].message) == snapshot
        with pytest.warns(UserWarning) as record:
            out = fitted.transform(df)
        assert str(record[0].message) == snapshot
        assert out["const"].tolist() == [7.0] * 10
        assert list(out.columns) == ["x_1", "x_2", "const", "b"]

    def test_transform_pins_knots_from_fit_not_new_data(self) -> None:
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        fitted = Recipe(steps=[StepSplineNatural(["x"], deg_free=2)]).fit(train)
        new = pd.DataFrame({"x": [100.0, 200.0, 300.0, 400.0, 500.0]})
        out = fitted.transform(new)
        # Values far outside the fit-time boundary extrapolate linearly rather
        # than sitting inside [0, 1] the way in-range natural-spline values do.
        assert list(out.columns) == ["x_1", "x_2"]
        assert out["x_1"].abs().max() > 1.0 or out["x_2"].abs().max() > 1.0

    def test_keep_original_cols_retains_sources(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepSplineNatural(["x"], deg_free=2, keep_original_cols=True))
            .fit(df)
            .transform(df)
        )
        assert list(out.columns) == ["x", "x_1", "x_2", "const", "b"]
        assert out["x"].tolist() == [float(v) for v in X_1_TO_10]

    def test_keep_original_cols_retains_source_roles(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(
            steps=[StepSplineNatural(["x"], deg_free=2, keep_original_cols=True)]
        ).fit(df)
        assert "feature" in fitted.roles.of("x")
        assert "feature" in fitted.roles.of("x_1")

    def test_source_loses_roles_when_not_kept(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepSplineNatural(["x"], deg_free=2)]).fit(df)
        assert fitted.roles.of("x") == set()

    def test_output_parents_and_schema_out(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepSplineNatural(["x"], deg_free=2)]).fit(df)
        step = fitted.steps[0]
        assert step.schema_out().columns == ["x_1", "x_2"]
        assert step.output_parents() == {"x_1": ["x"], "x_2": ["x"]}

    def test_retained_sources_are_selectable_downstream(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepSplineNatural(["x"], deg_free=2, keep_original_cols=True))
            .add(StepNormalize(["x"]))
            .fit(df)
            .transform(df)
        )
        assert out["x"].mean() == pytest.approx(0.0, abs=1e-12)

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepSplineNatural(["x"], deg_free=2)]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_integer_in_float_out_through_recipe(self) -> None:
        df = pd.DataFrame({"x": X_1_TO_10})
        out = Recipe(steps=[StepSplineNatural(["x"], deg_free=2)]).fit(df).transform(df)
        assert out["x_1"].dtype == float
        assert out["x_2"].dtype == float

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepSplineNatural(["x"], deg_free=2)]).fit(
            pd.DataFrame(_data())
        )
        assert all(step.trained for step in fitted.steps)

    def test_explicit_knots_through_recipe(self) -> None:
        df = pd.DataFrame({"x": [float(v) for v in X_1_TO_10]})
        recipe = Recipe(
            steps=[StepSplineNatural(["x"], complete_set=False, knots=[4.0, 7.0])]
        )
        out = recipe.fit(df).transform(df)
        assert list(out.columns) == ["x_1", "x_2", "x_3"]

    def test_non_default_pandas_index_does_not_introduce_nan(self) -> None:
        df = pd.DataFrame(
            {"x": [float(v) for v in X_1_TO_10], "b": list(range(10))},
            index=[500, 100, 800, 200, 700, 1, 2, 3, 4, 5],
        )
        out = Recipe(steps=[StepSplineNatural(["x"], deg_free=2)]).fit(df).transform(df)
        assert out.isna().sum().sum() == 0
        assert list(out.index) == list(df.index)
        assert out["b"].tolist() == list(range(10))


# ---------------------------------------------------------------------------
# StepSplineNatural: unit-level invariants
# ---------------------------------------------------------------------------


class TestStepSplineNaturalUnit:
    def test_mode_is_one_to_many(self) -> None:
        assert StepSplineNatural(["x"]).mode is Mode.ONE_TO_MANY

    def test_default_id(self) -> None:
        assert StepSplineNatural(["x"]).id == "spline_natural"

    def test_explicit_id(self) -> None:
        assert StepSplineNatural(["x"], id="my_spline").id == "my_spline"

    def test_default_deg_free_is_ten(self) -> None:
        assert StepSplineNatural(["x"]).deg_free == 10

    def test_default_complete_set_is_false(self) -> None:
        assert StepSplineNatural(["x"]).complete_set is False

    def test_default_role_is_feature(self) -> None:
        assert StepSplineNatural(["x"]).output_role == "feature"

    def test_keep_original_cols_default_false(self) -> None:
        assert StepSplineNatural(["x"]).keep_original_cols is False

    def test_keep_original_cols_is_settable(self) -> None:
        step = StepSplineNatural(["x"], keep_original_cols=True)
        assert step.keep_original_cols is True

    def test_fit_returns_self(self) -> None:
        step = StepSplineNatural(["x"], deg_free=2)
        frame = nw.from_native(pd.DataFrame(_data()))
        fitted = step.fit(frame, SchemaInfo.from_frame(frame))
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepSplineNatural(["x"])
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepSplineNatural(["x"]).schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepSplineNatural(["x"]).schema_out()

    def test_output_parents_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepSplineNatural(["x"]).output_parents()

    def test_accepts_numeric_dtypes(self) -> None:
        accepted = StepSplineNatural(["x"]).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        assert nw.String not in accepted

    def test_not_sparse(self) -> None:
        assert StepSplineNatural(["x"]).produces_sparse is False

    def test_required_pkgs_includes_scipy(self) -> None:
        assert "scipy" in StepSplineNatural(["x"]).required_pkgs()

    def test_invalid_deg_free_raises_at_fit(self, snapshot: SnapshotAssertion) -> None:
        df = pd.DataFrame(_data())
        step = StepSplineNatural(["x"], deg_free=1)
        frame = nw.from_native(df)
        with pytest.raises(ValueError) as excinfo:
            step.fit(frame, SchemaInfo.from_frame(frame))
        assert str(excinfo.value) == snapshot
