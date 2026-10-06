"""Tests for fit_other and apply_other."""

from __future__ import annotations

import warnings
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures import StepOther
from mlfeatures.step_other import apply_other, fit_other


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


# ---------------------------------------------------------------------------
# fit_other
# ---------------------------------------------------------------------------


class TestFitOther:
    def test_threshold_keeps_frequent_levels(self) -> None:
        # "a" is 3/5, "b" is 1/5, "c" is 1/5.
        df = _df(("g", ["a", "a", "a", "b", "c"]))
        assert fit_other(df, threshold=0.25) == {"g": ["a"]}

    def test_threshold_boundary_level_is_kept(self) -> None:
        # "b" is exactly 1/4 == the threshold; ties are kept.
        df = _df(("g", ["a", "a", "a", "b"]))
        assert fit_other(df, threshold=0.25) == {"g": ["a", "b"]}

    def test_min_count_keeps_frequent_levels(self) -> None:
        df = _df(("g", ["a", "a", "b", "c", "c", "c"]))
        assert fit_other(df, min_count=2) == {"g": ["a", "c"]}

    def test_min_count_boundary_level_is_kept(self) -> None:
        df = _df(("g", ["a", "a", "b"]))
        assert fit_other(df, min_count=1) == {"g": ["a", "b"]}

    def test_default_is_five_percent(self) -> None:
        # 1 of 20 rows == 0.05, so the rare level sits exactly on the default
        # boundary and is kept; 1 of 21 falls below it.
        kept = fit_other(_df(("g", ["a"] * 19 + ["b"])))
        assert kept == {"g": ["a", "b"]}
        kept = fit_other(_df(("g", ["a"] * 20 + ["b"])))
        assert kept == {"g": ["a"]}

    def test_every_column_present_in_estimate(self) -> None:
        # A column with nothing to collapse still gets an entry: the kept set is
        # what pools levels unseen at fit time.
        df = _df(("b", ["x", "y"]), ("a", ["p", "q"]))
        assert fit_other(df, threshold=0.25) == {"b": ["x", "y"], "a": ["p", "q"]}

    def test_kept_levels_are_sorted(self) -> None:
        df = _df(("g", ["c", "b", "a"]))
        assert fit_other(df, min_count=1)["g"] == ["a", "b", "c"]

    def test_nulls_excluded_from_denominator(self) -> None:
        # 4 non-null rows: "b" is 1/4, above a 0.25 threshold only because the
        # nulls do not inflate the denominator.
        df = _df(("g", ["a", "a", "a", "b", None, None]))
        assert fit_other(df, threshold=0.25) == {"g": ["a", "b"]}

    def test_all_null_column_keeps_nothing(self) -> None:
        df = _df(("g", [None, None]))
        assert fit_other(df, threshold=0.25) == {"g": []}

    def test_empty_selection(self) -> None:
        assert fit_other(_df(), threshold=0.25) == {}

    def test_collision_warns(self, snapshot: SnapshotAssertion) -> None:
        df = _df(("g", ["a", "a", "other"]))
        with pytest.warns(UserWarning) as record:
            fit_other(df, threshold=0.25)
        assert str(record[0].message) == snapshot

    def test_no_collision_no_warning(self) -> None:
        df = _df(("g", ["a", "a", "b"]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fit_other(df, threshold=0.25) == {"g": ["a", "b"]}

    def test_custom_level_collision(self, snapshot: SnapshotAssertion) -> None:
        df = _df(("g", ["a", "a", "RARE"]))
        with pytest.warns(UserWarning) as record:
            fit_other(df, threshold=0.25, other_level="RARE")
        assert str(record[0].message) == snapshot


# ---------------------------------------------------------------------------
# apply_other
# ---------------------------------------------------------------------------


class TestApplyOther:
    def test_infrequent_levels_pooled(self) -> None:
        df = _df(("g", ["a", "b", "c"]))
        out = apply_other(df, {"g": ["a"]})
        assert out["g"].to_list() == ["a", "other", "other"]

    def test_unseen_level_pooled(self) -> None:
        # "z" never appeared at fit time; it is not in the kept set, so it is
        # pooled rather than passed through.
        df = _df(("g", ["a", "z"]))
        out = apply_other(df, {"g": ["a", "b"]})
        assert out["g"].to_list() == ["a", "other"]

    def test_nulls_stay_null(self) -> None:
        df = _df(("g", ["a", "b", None]))
        out = apply_other(df, {"g": ["a"]})
        assert out["g"].is_null().to_list() == [False, False, True]
        assert out["g"].to_list()[:2] == ["a", "other"]

    def test_custom_level(self) -> None:
        df = _df(("g", ["a", "b"]))
        out = apply_other(df, {"g": ["a"]}, other_level="RARE")
        assert out["g"].to_list() == ["a", "RARE"]

    def test_nothing_pooled_when_all_levels_kept(self) -> None:
        df = _df(("g", ["a", "b"]))
        out = apply_other(df, {"g": ["a", "b"]})
        assert out["g"].to_list() == ["a", "b"]

    def test_empty_kept_set_pools_everything(self) -> None:
        df = _df(("g", ["a", "b"]))
        out = apply_other(df, {"g": []})
        assert out["g"].to_list() == ["other", "other"]

    def test_string_input_recoded_to_string(self) -> None:
        df = _df(("g", ["a", "b"]))
        out = apply_other(df, {"g": ["a"]})
        assert out["g"].dtype == nw.String
        assert out["g"].to_list() == ["a", "other"]

    def test_categorical_input_recoded_to_string(self) -> None:
        df = _df(("g", pd.Categorical(["a", "b", "a"])))
        out = apply_other(df, {"g": ["a"]})
        assert out["g"].dtype == nw.String
        assert out["g"].to_list() == ["a", "other", "a"]

    def test_enum_input_recoded_to_string(self) -> None:
        pl = pytest.importorskip("polars")
        series = pl.Series(["a", "b", "a"], dtype=pl.Enum(["a", "b"]))
        df = nw.from_native(pl.DataFrame({"g": series}), eager_only=True)
        out = apply_other(df, {"g": ["a"]})
        assert out["g"].dtype == nw.String
        assert out["g"].to_list() == ["a", "other", "a"]

    def test_columns_not_listed_pass_through(self) -> None:
        df = _df(("g", ["a", "b"]), ("h", ["p", "q"]))
        out = apply_other(df, {"g": ["a"]})
        assert out["g"].to_list() == ["a", "other"]
        assert out["h"].to_list() == ["p", "q"]

    def test_preserves_columns_and_order(self) -> None:
        df = _df(("b", ["x", "y"]), ("a", ["p", "q"]))
        out = apply_other(df, {"b": ["x"], "a": ["p"]})
        assert out.columns == ["b", "a"]
        assert out.shape == df.shape


# ---------------------------------------------------------------------------
# StepOther argument validation
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_both_criteria_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            StepOther(["g"], threshold=0.1, min_count=5)
        assert str(excinfo.value) == snapshot

    def test_threshold_of_one_raises(self, snapshot: SnapshotAssertion) -> None:
        # The likeliest units mix-up: recipes would read 1 as a count.
        with pytest.raises(ValueError) as excinfo:
            StepOther(["g"], threshold=1.0)
        assert str(excinfo.value) == snapshot

    @pytest.mark.parametrize("threshold", [0.0, -0.1, 5.0])
    def test_threshold_out_of_range_raises(self, threshold: float) -> None:
        with pytest.raises(ValueError):
            StepOther(["g"], threshold=threshold)

    def test_min_count_below_one_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            StepOther(["g"], min_count=0)
        assert str(excinfo.value) == snapshot

    def test_fractional_min_count_raises(self, snapshot: SnapshotAssertion) -> None:
        # A count of rows has to be a whole number; 2.5 would otherwise be
        # accepted and act as 3, silently.
        with pytest.raises(ValueError) as excinfo:
            StepOther(["g"], min_count=2.5)  # type: ignore[arg-type]
        assert str(excinfo.value) == snapshot

    def test_integral_float_min_count_allowed(self) -> None:
        # 3.0 is a whole number of rows, so it is accepted and behaves as 3.
        df = _df(("g", ["a", "a", "a", "b", "b"]))
        step = StepOther(["g"], min_count=3.0)  # type: ignore[arg-type]
        assert step.min_count == 3
        assert fit_other(df, min_count=step.min_count) == {"g": ["a"]}

    def test_neither_criterion_is_allowed(self) -> None:
        step = StepOther(["g"])
        assert step.threshold is None
        assert step.min_count is None

    def test_arguments_stored_unchanged(self) -> None:
        step = StepOther(["g"], min_count=3, other_level="RARE")
        assert step.min_count == 3
        assert step.threshold is None
        assert step.other_level == "RARE"
