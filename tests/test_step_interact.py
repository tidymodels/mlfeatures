"""Tests for :mod:`shoez.step_interact`.

The functions under test are written against the internal :data:`~shoez.types.Frame`
type (a narwhals ``DataFrame``), so these tests build inputs via
:func:`narwhals.stable.v2.from_native` and inspect outputs as narwhals frames.
"""

from __future__ import annotations

from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez.step_interact import InteractFit, apply_interact, fit_interact


def _df(data: dict[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(data))


# ---------------------------------------------------------------------------
# fit_interact — pair enumeration
# ---------------------------------------------------------------------------


class TestFitPairs:
    def test_two_columns_gives_one_pair(self) -> None:
        result = fit_interact(_df({"a": [1.0], "b": [2.0]}))
        assert result.pairs == [("a", "b")]

    def test_three_columns_gives_three_pairs(self) -> None:
        result = fit_interact(_df({"a": [1.0], "b": [2.0], "c": [3.0]}))
        assert result.pairs == [("a", "b"), ("a", "c"), ("b", "c")]

    def test_pairs_follow_input_column_order(self) -> None:
        result = fit_interact(_df({"c": [1.0], "a": [2.0], "b": [3.0]}))
        assert result.pairs == [("c", "a"), ("c", "b"), ("a", "b")]

    def test_single_column_gives_no_pairs(self) -> None:
        result = fit_interact(_df({"a": [1.0]}))
        assert result.pairs == []

    def test_zero_columns_gives_no_pairs(self) -> None:
        result = fit_interact(_df({}))
        assert result.pairs == []

    def test_default_sep(self) -> None:
        result = fit_interact(_df({"a": [1.0], "b": [2.0]}))
        assert result.sep == "_x_"

    def test_custom_sep_is_recorded(self) -> None:
        result = fit_interact(_df({"a": [1.0], "b": [2.0]}), sep="_times_")
        assert result.sep == "_times_"

    def test_returns_interact_fit(self) -> None:
        assert isinstance(fit_interact(_df({"a": [1.0], "b": [2.0]})), InteractFit)


# ---------------------------------------------------------------------------
# apply_interact — pairwise products
# ---------------------------------------------------------------------------


class TestApplyProducts:
    def test_product_values_are_correct(self) -> None:
        frame = _df({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        out = apply_interact(frame, fit_interact(frame))
        assert out["a_x_b"].to_list() == [4.0, 10.0, 18.0]

    def test_column_name_uses_sep(self) -> None:
        frame = _df({"a": [1.0], "b": [2.0]})
        fit = fit_interact(frame, sep="_times_")
        out = apply_interact(frame, fit)
        assert out.columns == ["a_times_b"]

    def test_sources_dropped_by_default(self) -> None:
        frame = _df({"a": [1.0], "b": [2.0]})
        out = apply_interact(frame, fit_interact(frame))
        assert out.columns == ["a_x_b"]

    def test_keep_original_cols_retains_sources(self) -> None:
        frame = _df({"a": [1.0], "b": [2.0]})
        out = apply_interact(frame, fit_interact(frame), keep_original_cols=True)
        assert out.columns == ["a", "b", "a_x_b"]

    def test_three_columns_produces_three_interaction_columns(self) -> None:
        frame = _df({"a": [1.0], "b": [2.0], "c": [3.0]})
        out = apply_interact(frame, fit_interact(frame))
        assert out.columns == ["a_x_b", "a_x_c", "b_x_c"]

    def test_three_columns_product_values_are_correct(self) -> None:
        frame = _df({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        out = apply_interact(frame, fit_interact(frame))
        assert out["a_x_b"].to_list() == [3.0, 8.0]
        assert out["a_x_c"].to_list() == [5.0, 12.0]
        assert out["b_x_c"].to_list() == [15.0, 24.0]

    def test_null_in_either_column_produces_null_product(self) -> None:
        frame = _df({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})
        out = apply_interact(frame, fit_interact(frame))
        values = out["a_x_b"].to_list()
        assert values[0] == 4.0
        assert values[1] is None or values[1] != values[1]
        assert values[2] is None or values[2] != values[2]

    def test_no_pairs_warns_and_returns_unchanged(
        self, snapshot: SnapshotAssertion
    ) -> None:
        frame = _df({"a": [1.0, 2.0, 3.0]})
        with pytest.warns(UserWarning) as record:
            out = apply_interact(frame, fit_interact(frame))
        assert str(record[0].message) == snapshot
        assert out.columns == ["a"]
        assert out["a"].to_list() == [1.0, 2.0, 3.0]

    def test_zero_selected_columns_warns_and_returns_unchanged(self) -> None:
        """A selector matching nothing yields a 0-column fit; applying it to
        the full frame (unrelated columns included) is a no-op, same as the
        1-column case above but exercised with a genuinely empty selection."""
        selected = _df({})
        frame = _df({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        with pytest.warns(UserWarning):
            out = apply_interact(frame, fit_interact(selected))
        assert out.columns == ["a", "b"]
        assert out["a"].to_list() == [1.0, 2.0, 3.0]
        assert out["b"].to_list() == [4.0, 5.0, 6.0]

    def test_returns_narwhals_dataframe(self) -> None:
        frame = _df({"a": [1.0], "b": [2.0]})
        out = apply_interact(frame, fit_interact(frame))
        assert isinstance(out, nw.DataFrame)
