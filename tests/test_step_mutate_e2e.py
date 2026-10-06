"""End-to-end lifecycle tests for StepMutate: Recipe -> fit -> transform.

Exercises the ``UNCONSTRAINED`` escape hatch through the full vertical slice on
both pandas and polars: added columns are attributed to the mutate step, and
in-place overwrites keep the frame's shape.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures import Mode, Recipe, StepMutate


def _data() -> dict[str, list[float]]:
    return {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}


class TestEndToEnd:
    def test_pandas_adds_and_overwrites(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepMutate({"x_sq": lambda d: d["x"] ** 2, "x": lambda d: d["x"] + 1}))
            .fit(df)
            .transform(df)
        )
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["x", "y", "x_sq"]
        assert out["x"].tolist() == [2.0, 3.0, 4.0]
        assert out["x_sq"].tolist() == [1.0, 4.0, 9.0]

    def test_polars_adds_column(self) -> None:
        df = pl.DataFrame(_data())
        out = (
            Recipe()
            .add(StepMutate({"z": lambda d: d["x"] + d["y"]}))
            .fit(df)
            .transform(df)
        )
        assert isinstance(out, pl.DataFrame)
        assert out["z"].to_list() == [5.0, 7.0, 9.0]

    def test_added_column_attributed_to_step(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepMutate({"z": lambda d: d["x"] * 2})]).fit(df)
        node = fitted.schema.summary()
        # The new column carries the step id as its source in the lineage DAG.
        z_source = node.filter(pl.col("name") == "z")["source"].to_list()
        assert z_source == ["mutate"]

    def test_new_column_gets_output_role(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepMutate({"z": lambda d: d["x"] * 2})]).fit(df)
        assert "feature" in fitted.roles.of("z")

    def test_clash_with_new_and_overwrite_distinguished(self) -> None:
        # Overwriting an existing column is allowed (it is not a "produced"
        # column); only genuinely new names go through the collision guard.
        df = pd.DataFrame(_data())
        out = (
            Recipe(steps=[StepMutate({"x": lambda d: d["x"] * 0})])
            .fit(df)
            .transform(df)
        )
        assert out["x"].tolist() == [0.0, 0.0, 0.0]

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepMutate({"z": lambda d: d["x"] * 2})]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))


class TestStepMutate:
    def test_mode_is_unconstrained(self) -> None:
        assert StepMutate({"z": lambda d: d["x"]}).mode is Mode.UNCONSTRAINED

    def test_transform_before_fit_raises(self) -> None:
        import narwhals as nw

        step = StepMutate({"z": lambda d: d["x"]})
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_empty_mutations_snapshot(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(TypeError) as excinfo:
            StepMutate({})
        assert str(excinfo.value) == snapshot
