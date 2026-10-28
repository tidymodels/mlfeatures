"""End-to-end lifecycle tests for StepInteract: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepInteract + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic. Structured to parallel ``tests/test_step_polynomial_e2e.py``.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures import Mode, Recipe, SchemaInfo, StepInteract, StepNormalize


def _data() -> dict[str, list[float]]:
    return {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [5.0, 6.0, 7.0, 8.0],
        "c": [9.0, 10.0, 11.0, 12.0],
        "d": [0.0, 1.0, 0.0, 1.0],
    }


# ---------------------------------------------------------------------------
# Recipe-level end-to-end behaviour
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_pandas_produces_pairwise_columns(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepInteract(["a", "b", "c"])]).fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["d", "a_x_b", "a_x_c", "b_x_c"]

    def test_three_columns_product_values_are_correct(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepInteract(["a", "b", "c"])]).fit(df).transform(df)
        # a = [1, 2, 3, 4], b = [5, 6, 7, 8], c = [9, 10, 11, 12]
        assert out["a_x_b"].tolist() == [5.0, 12.0, 21.0, 32.0]
        assert out["a_x_c"].tolist() == [9.0, 20.0, 33.0, 48.0]
        assert out["b_x_c"].tolist() == [45.0, 60.0, 77.0, 96.0]

    def test_polars_produces_pairwise_columns(self) -> None:
        df = pl.DataFrame(_data())
        out = Recipe().add(StepInteract(["a", "b"])).fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["c", "d", "a_x_b"]

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepInteract(["a", "b"])]).fit(df).transform(df)
        assert out["c"].tolist() == [9.0, 10.0, 11.0, 12.0]
        assert out["d"].tolist() == [0.0, 1.0, 0.0, 1.0]

    def test_product_values_are_correct(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepInteract(["a", "b"])]).fit(df).transform(df)
        assert out["a_x_b"].tolist() == [5.0, 12.0, 21.0, 32.0]

    def test_custom_sep(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe(steps=[StepInteract(["a", "b"], sep="_times_")])
            .fit(df)
            .transform(df)
        )
        assert "a_times_b" in out.columns

    def test_keep_original_cols_retains_sources(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepInteract(["a", "b"], keep_original_cols=True))
            .fit(df)
            .transform(df)
        )
        assert list(out.columns) == ["a", "b", "c", "d", "a_x_b"]

    def test_keep_original_cols_retains_source_roles(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepInteract(["a", "b"], keep_original_cols=True)]).fit(
            df
        )
        assert "feature" in fitted.roles.of("a")
        assert "feature" in fitted.roles.of("a_x_b")

    def test_source_loses_roles_when_not_kept(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepInteract(["a", "b"])]).fit(df)
        assert fitted.roles.of("a") == set()
        assert fitted.roles.of("b") == set()

    def test_fewer_than_two_columns_warns_and_is_noop(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepInteract(["a"])])
        with pytest.warns(UserWarning) as record:
            fitted = recipe.fit(df)
        assert str(record[0].message) == snapshot
        with pytest.warns(UserWarning):
            out = fitted.transform(df)
        assert list(out.columns) == list(df.columns)

    def test_retained_sources_are_selectable_downstream(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepInteract(["a", "b"], keep_original_cols=True))
            .add(StepNormalize(["a"]))
            .fit(df)
            .transform(df)
        )
        assert out["a"].mean() == pytest.approx(0.0, abs=1e-12)

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepInteract(["a", "b"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepInteract(["a", "b"])]).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)


# ---------------------------------------------------------------------------
# Name collisions
# ---------------------------------------------------------------------------


class TestNameCollisions:
    """Generated interaction names are validated centrally by the recipe."""

    @pytest.mark.parametrize("ctor", [pd.DataFrame, pl.DataFrame], ids=["pd", "pl"])
    def test_two_pairs_producing_the_same_name_raise(
        self, ctor: type, snapshot: SnapshotAssertion
    ) -> None:
        # With sep="_x_", the pairs ("a_x", "b") and ("a", "x_b") both name
        # their product "a_x_x_b". `produced_columns()` must preserve this
        # duplicate (rather than silently losing it the way a dict keyed by
        # name would) so `Lineage.check_additions` rejects it up front.
        data = {"a_x": [1.0, 2.0], "b": [3.0, 4.0], "a": [5.0, 6.0], "x_b": [7.0, 8.0]}
        df = ctor(data)
        recipe = Recipe(steps=[StepInteract(["a_x", "b", "a", "x_b"])])
        with pytest.raises(ValueError) as excinfo:
            recipe.fit(df)
        assert str(excinfo.value) == snapshot


# ---------------------------------------------------------------------------
# StepInteract: unit-level invariants
# ---------------------------------------------------------------------------


class TestStepInteractUnit:
    def test_mode_is_many_to_many(self) -> None:
        assert StepInteract(["a", "b"]).mode is Mode.MANY_TO_MANY

    def test_default_id(self) -> None:
        assert StepInteract(["a", "b"]).id == "interact"

    def test_explicit_id(self) -> None:
        assert StepInteract(["a", "b"], id="my_interact").id == "my_interact"

    def test_default_sep(self) -> None:
        assert StepInteract(["a", "b"]).sep == "_x_"

    def test_default_role_is_feature(self) -> None:
        assert StepInteract(["a", "b"]).output_role == "feature"

    def test_keep_original_cols_default_false(self) -> None:
        assert StepInteract(["a", "b"]).keep_original_cols is False

    def test_keep_original_cols_is_settable(self) -> None:
        step = StepInteract(["a", "b"], keep_original_cols=True)
        assert step.keep_original_cols is True

    def test_fit_returns_self(self) -> None:
        step = StepInteract(["a", "b"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepInteract(["a", "b"])
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepInteract(["a", "b"]).schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepInteract(["a", "b"]).schema_out()

    def test_output_parents_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepInteract(["a", "b"]).output_parents()

    def test_accepts_numeric_dtypes(self) -> None:
        accepted = StepInteract(["a", "b"]).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        assert nw.String not in accepted

    def test_schema_in_lists_selected_columns(self) -> None:
        step = StepInteract(["a", "b"])
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert step.schema_in().columns == ["a", "b"]

    def test_schema_out_lists_interaction_columns(self) -> None:
        step = StepInteract(["a", "b", "c"])
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert step.schema_out().columns == ["a_x_b", "a_x_c", "b_x_c"]

    def test_output_parents_maps_each_column_to_its_pair(self) -> None:
        step = StepInteract(["a", "b"])
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert step.output_parents() == {"a_x_b": ["a", "b"]}

    def test_not_sparse(self) -> None:
        assert StepInteract(["a", "b"]).produces_sparse is False
