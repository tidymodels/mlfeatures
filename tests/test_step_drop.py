"""Unit tests for StepDrop's own surface: fit-time resolution and guards.

StepDrop estimates nothing, so there is no ``fit_*`` logic function to test on
its own; what there is to check is that ``fit()`` records the resolved selection
and that ``transform()`` replays exactly that decision.
"""

from __future__ import annotations

import narwhals as nw
import pandas as pd
import pytest

from mlfeatures import Mode, Recipe, SchemaInfo, StepDrop
from mlfeatures import Selectors as s


def _data() -> dict[str, list[object]]:
    return {
        "a": [1.0, 2.0, 3.0],
        "b": [4.0, 5.0, 6.0],
        "note": ["x", "y", "z"],
    }


def _frame() -> nw.DataFrame[object]:
    return nw.from_native(pd.DataFrame(_data()), eager_only=True)


class TestFit:
    def test_records_resolved_names(self) -> None:
        step = StepDrop(["a", "note"]).fit(_frame(), SchemaInfo())
        assert step.schema_in().columns == ["a", "note"]

    def test_records_dtypes_of_consumed_columns(self) -> None:
        step = StepDrop(["a", "note"]).fit(_frame(), SchemaInfo())
        assert set(step.schema_in().dtypes) == {"a", "note"}

    def test_produces_nothing(self) -> None:
        step = StepDrop(["a"]).fit(_frame(), SchemaInfo())
        assert step.schema_out().columns == []
        assert step.produced_columns() == []

    def test_fit_returns_self(self) -> None:
        step = StepDrop(["a"])
        assert step.fit(_frame(), SchemaInfo()) is step


class TestTransform:
    def test_drops_the_selected_columns(self) -> None:
        step = StepDrop(["a", "note"]).fit(_frame(), SchemaInfo())
        out = step.transform(_frame())
        assert out.columns == ["b"]

    def test_replays_the_fit_time_decision(self) -> None:
        # The selector is resolved once; new data with an extra string column
        # does not widen the selection.
        step = StepDrop(s.string()).fit(_frame(), SchemaInfo())
        new = nw.from_native(
            pd.DataFrame({"a": [1.0], "note": ["x"], "extra": ["q"]}), eager_only=True
        )
        assert step.transform(new).columns == ["a", "extra"]

    def test_tolerates_already_absent_columns(self) -> None:
        step = StepDrop(["a", "note"]).fit(_frame(), SchemaInfo())
        new = nw.from_native(pd.DataFrame({"b": [1.0]}), eager_only=True)
        assert step.transform(new).columns == ["b"]

    def test_dropping_everything_leaves_no_columns(self) -> None:
        step = StepDrop(["a", "b", "note"]).fit(_frame(), SchemaInfo())
        assert step.transform(_frame()).columns == []


class TestRoles:
    def test_dropped_columns_lose_their_roles(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(data=df).add(StepDrop(["note"])).fit(df)
        assert fitted.roles.of("note") == set()
        assert "feature" in fitted.roles.of("a")

    def test_target_is_dropped_like_any_other_column(self) -> None:
        # No role guard: the selection is taken at face value.
        df = pd.DataFrame(_data())
        fitted = Recipe(data=df).set_roles(target="b").add(StepDrop(["b"])).fit(df)
        assert "b" not in fitted.transform(df).columns
        assert fitted.roles.of("b") == set()


class TestGuards:
    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepDrop(["a"]).transform(_frame())

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepDrop(["a"]).schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepDrop(["a"]).schema_out()


class TestStepSurface:
    def test_mode_is_removing(self) -> None:
        assert StepDrop(["a"]).mode is Mode.REMOVING

    def test_default_id(self) -> None:
        assert StepDrop(["a"]).id == "drop"

    def test_explicit_id(self) -> None:
        assert StepDrop(["a"], id="my_drop").id == "my_drop"

    def test_accepts_any_dtype(self) -> None:
        accepted = StepDrop(["a"]).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.String in accepted

    def test_not_sparse(self) -> None:
        assert StepDrop(["a"]).produces_sparse is False
