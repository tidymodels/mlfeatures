"""End-to-end lifecycle tests for StepPolynomial: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepPolynomial + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic. Structured to parallel ``tests/test_step_normalize_e2e.py``.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez import Mode, Recipe, SchemaInfo, StepNormalize, StepPolynomial


def _data() -> dict[str, list[float]]:
    # ``const`` is constant (degree-0 column); ``a`` varies broadly; ``b``
    # is left out of the polynomial expansion entirely so we can check it
    # passes through unchanged.
    return {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "const": [7.0, 7.0, 7.0, 7.0, 7.0],
        "b": [5.0, 1.0, 4.0, 1.0, 9.0],
    }


# ---------------------------------------------------------------------------
# Recipe-level end-to-end behaviour
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_pandas_expands_in_place(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepPolynomial(["a"], degree=3)])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        # "a" replaced in its slot by three polynomial columns; the other
        # columns pass through in order.
        assert list(out.columns) == ["a_1", "a_2", "a_3", "const", "b"]

    def test_polars_expands_in_place(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepPolynomial(["a"], degree=2))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["a_1", "a_2", "const", "b"]

    def test_polynomial_values_match_r_poly(self) -> None:
        # poly([1, 2, 3], 2) — the same ground truth used in the unit tests.
        expected = np.array(
            [
                [-7.071067812e-01, 0.4082482905],
                [-9.681035012e-17, -0.8164965809],
                [7.071067812e-01, 0.4082482905],
            ]
        )
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        out = Recipe(steps=[StepPolynomial(["x"], degree=2)]).fit(df).transform(df)
        np.testing.assert_allclose(out[["x_1", "x_2"]].to_numpy(), expected, atol=1e-9)

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        # Only expand "a"; "const" and "b" pass through untouched.
        out = Recipe(steps=[StepPolynomial(["a"], degree=2)]).fit(df).transform(df)
        assert out["b"].tolist() == [5.0, 1.0, 4.0, 1.0, 9.0]
        assert out["const"].tolist() == [7.0] * 5

    def test_constant_column_warns_and_is_left_unchanged(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepPolynomial(["a", "const"], degree=2)])
        # Warn at fit (during the inline transform in Recipe.fit), and warn
        # again on the explicit transform — both come from apply_polynomial.
        with pytest.warns(UserWarning) as record:
            fitted = recipe.fit(df)
        assert str(record[0].message) == snapshot
        with pytest.warns(UserWarning) as record:
            out = fitted.transform(df)
        assert str(record[0].message) == snapshot
        # "const" stays put with its original value.
        assert out["const"].tolist() == [7.0] * 5
        assert list(out.columns) == ["a_1", "a_2", "const", "b"]

    def test_transform_pins_degree_from_fit_not_new_data(self) -> None:
        # The effective degree is chosen at fit time and reused at transform.
        # Here train has only 3 unique values -> effective degree min(5, 2) = 2.
        # Even though `new` has 5 unique values (which would support degree 4
        # at a fresh fit), the transform still produces only 2 columns.
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        fitted = Recipe(steps=[StepPolynomial(["a"], degree=5)]).fit(train)
        new = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = fitted.transform(new)
        assert list(out.columns) == ["a_1", "a_2"]

    def test_keep_original_cols_retains_sources(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepPolynomial(["a"], degree=2, keep_original_cols=True))
            .fit(df)
            .transform(df)
        )
        assert list(out.columns) == ["a", "a_1", "a_2", "const", "b"]
        assert out["a"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_keep_original_cols_retains_source_roles(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(
            steps=[StepPolynomial(["a"], degree=2, keep_original_cols=True)]
        ).fit(df)
        assert "predictor" in fitted.roles.of("a")
        assert "predictor" in fitted.roles.of("a_1")

    def test_source_loses_roles_when_not_kept(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepPolynomial(["a"], degree=2)]).fit(df)
        assert fitted.roles.of("a") == set()

    def test_keep_original_cols_leaves_produced_columns_alone(self) -> None:
        # The flag governs the source columns only; what the step *produces* is
        # unchanged.
        df = pd.DataFrame(_data())
        fitted = Recipe(
            steps=[StepPolynomial(["a"], degree=2, keep_original_cols=True)]
        ).fit(df)
        step = fitted.steps[0]
        assert step.schema_out().columns == ["a_1", "a_2"]
        assert step.output_parents() == {"a_1": ["a"], "a_2": ["a"]}

    def test_retained_sources_are_selectable_downstream(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepPolynomial(["a"], degree=2, keep_original_cols=True))
            .add(StepNormalize(["a"]))
            .fit(df)
            .transform(df)
        )
        assert out["a"].mean() == pytest.approx(0.0, abs=1e-12)

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepPolynomial(["a"], degree=2)]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_integer_in_float_out_through_recipe(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4]})  # integer column
        out = Recipe(steps=[StepPolynomial(["a"], degree=2)]).fit(df).transform(df)
        assert out["a_1"].dtype == float  # polynomial basis is real-valued
        assert out["a_2"].dtype == float

    def test_degree_capped_by_unique_values(self) -> None:
        # 3 unique values, degree=5 -> effective degree 2.
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]})
        out = Recipe(steps=[StepPolynomial(["a"], degree=5)]).fit(df).transform(df)
        # Capped at 2 columns, not the requested 5.
        assert list(out.columns) == ["a_1", "a_2"]

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepPolynomial(["a"], degree=2)]).fit(
            pd.DataFrame(_data())
        )
        assert all(step.trained for step in fitted.steps)

    def test_basis_is_pinned_at_fit_not_recomputed_at_transform(self) -> None:
        # The orthogonal basis (centering mean + QR transform matrix) must be
        # learned at fit time on the training data and reused unchanged at
        # transform time. Concretely: if we transform the same rows twice —
        # once embedded in their original training context, once together with
        # held-out rows that would shift the recomputed mean — the matching
        # rows must produce identical polynomial values.
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        fitted = Recipe(steps=[StepPolynomial(["a"], degree=3)]).fit(train)
        train_out = fitted.transform(train)
        # Adding far-away points would shift the mean if the basis were
        # re-estimated; with a pinned basis, the first five rows must match
        # train_out exactly.
        mixed = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0]})
        mixed_out = fitted.transform(mixed)
        for col in ("a_1", "a_2", "a_3"):
            np.testing.assert_allclose(
                mixed_out[col].to_numpy()[:5],
                train_out[col].to_numpy(),
                atol=1e-12,
                err_msg=f"basis drifted between transforms on column {col}",
            )

    def test_fit_stores_training_mean_on_step(self) -> None:
        # Whitebox check: the StepPolynomial actually carries the training
        # mean on its fitted result, so we know the centering can't drift.
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        fitted = Recipe(steps=[StepPolynomial(["a"], degree=2)]).fit(train)
        # ``_estimates`` is private but exists post-fit; using it for the
        # invariant check is the most direct way to assert pinning.
        fit_result = fitted.steps[0]._estimates
        assert fit_result is not None
        assert fit_result["a"].mean == pytest.approx(3.0)

    def test_non_default_pandas_index_does_not_introduce_nan(self) -> None:
        # Regression: when a pandas frame has a shuffled / non-default index
        # (e.g. coming out of sklearn.train_test_split), an earlier version
        # used `nw.concat(how="horizontal")` to assemble the output, which
        # is index-aligned on pandas — producing all-NaN rows wherever the
        # source index didn't match the polynomial frame's RangeIndex.
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]},
            index=[500, 100, 800, 200, 700],
        )
        out = Recipe(steps=[StepPolynomial(["a"], degree=3)]).fit(df).transform(df)
        # All-finite expansions, original index preserved, pass-through intact.
        assert out.isna().sum().sum() == 0
        assert list(out.index) == [500, 100, 800, 200, 700]
        assert out["b"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]


# ---------------------------------------------------------------------------
# StepPolynomial — unit-level invariants
# ---------------------------------------------------------------------------


class TestStepPolynomialUnit:
    def test_mode_is_one_to_many(self) -> None:
        assert StepPolynomial(["a"]).mode is Mode.ONE_TO_MANY

    def test_default_id(self) -> None:
        assert StepPolynomial(["a"]).id == "polynomial"

    def test_explicit_id(self) -> None:
        assert StepPolynomial(["a"], id="my_poly").id == "my_poly"

    def test_default_degree_is_two(self) -> None:
        assert StepPolynomial(["a"]).degree == 2

    def test_default_role_is_predictor(self) -> None:
        assert StepPolynomial(["a"]).output_role == "predictor"

    def test_keep_original_cols_default_false(self) -> None:
        assert StepPolynomial(["a"]).keep_original_cols is False

    def test_keep_original_cols_is_settable(self) -> None:
        step = StepPolynomial(["a"], keep_original_cols=True)
        assert step.keep_original_cols is True

    def test_fit_returns_self(self) -> None:
        step = StepPolynomial(["a"], degree=2)
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepPolynomial(["a"])
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepPolynomial(["a"]).schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepPolynomial(["a"]).schema_out()

    def test_accepts_numeric_dtypes(self) -> None:
        accepted = StepPolynomial(["a"]).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        assert nw.String not in accepted

    def test_schema_in_lists_selected_columns(self) -> None:
        step = StepPolynomial(["a"], degree=2)
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert step.schema_in().columns == ["a"]

    def test_schema_out_lists_polynomial_columns(self) -> None:
        step = StepPolynomial(["a"], degree=3)
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert step.schema_out().columns == ["a_1", "a_2", "a_3"]
        assert step.schema_out().dtypes == {
            "a_1": "Float64",
            "a_2": "Float64",
            "a_3": "Float64",
        }

    def test_schema_out_keeps_degree_zero_column_with_original_dtype(self) -> None:
        # "const" is constant -> effective degree 0; it stays in schema_out
        # under its original name with its original dtype.
        df = pd.DataFrame(_data())
        step = StepPolynomial(["a", "const"], degree=2)
        step.fit(nw.from_native(df), SchemaInfo())
        out = step.schema_out()
        assert out.columns == ["a_1", "a_2", "const"]
        assert out.dtypes["a_1"] == "Float64"
        assert out.dtypes["a_2"] == "Float64"
        # Original dtype preserved (Float64 here because pandas reads as float).
        assert out.dtypes["const"] == "Float64"

    def test_not_sparse(self) -> None:
        assert StepPolynomial(["a"]).produces_sparse is False

    def test_invalid_degree_raises_at_fit(self, snapshot: SnapshotAssertion) -> None:
        df = pd.DataFrame(_data())
        step = StepPolynomial(["a"], degree=0)
        with pytest.raises(ValueError) as excinfo:
            step.fit(nw.from_native(df), SchemaInfo())
        assert str(excinfo.value) == snapshot
