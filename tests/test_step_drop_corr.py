"""Tests for fit_drop_corr."""

from __future__ import annotations

import warnings

import narwhals as nw
import pandas as pd
import polars as pl
import polars.selectors as cs
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez import SchemaInfo
from shoez.step_drop_corr import StepDropCorr, fit_drop_corr


def _df(*cols: tuple[str, list[float | None]]) -> nw.DataFrame:
    return nw.from_native(pd.DataFrame(dict(cols)))


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_empty_result_is_list(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [4.0, 5.0, 6.0]))
        result = fit_drop_corr(df, threshold=0.9)
        assert isinstance(result, list)

    def test_nonempty_result_is_list_of_str(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [1.0, 2.0, 3.0]))
        result = fit_drop_corr(df, threshold=0.5)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    """``threshold``/``method`` are validated at construction, not at fit time.

    ``fit_drop_corr`` itself performs no validation, matching every other
    step's ``fit_*`` free function (e.g. ``fit_log``): hyperparameter checks
    live solely on the step's ``__init__``.
    """

    def test_threshold_above_1_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            StepDropCorr(["a", "b"], threshold=1.1)
        assert str(excinfo.value) == snapshot

    def test_threshold_below_0_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            StepDropCorr(["a", "b"], threshold=-0.1)
        assert str(excinfo.value) == snapshot

    def test_invalid_method_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            StepDropCorr(["a", "b"], method="kendall")  # type: ignore[arg-type]
        assert str(excinfo.value) == snapshot


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_single_column_returns_empty(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]))
        assert len(fit_drop_corr(df)) == 0

    def test_zero_row_returns_empty(self) -> None:
        df = _df(("a", []), ("b", []))
        assert len(fit_drop_corr(df)) == 0

    def test_no_pair_above_threshold_returns_empty(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [3.0, 1.0, 2.0]))
        assert len(fit_drop_corr(df, threshold=0.9)) == 0

    def test_threshold_exactly_equal_not_dropped(self) -> None:
        # threshold is strict (>), so r == threshold should not trigger removal
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [1.0, 2.0, 3.0]))
        assert len(fit_drop_corr(df, threshold=1.0)) == 0


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


class TestCoreAlgorithm:
    def test_perfectly_correlated_pair_drops_row_variable(self) -> None:
        # a and b have equal mean |corr|, so the tie-break decides: it goes to
        # the row variable (lower column index) → "a". Asserting the exact
        # column catches a `>`→`>=` regression in the rank comparison, which
        # would instead drop "b".
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [1.0, 2.0, 3.0]))
        result = fit_drop_corr(df, threshold=0.9)
        assert result == ["a"]

    def test_integer_columns_drop_row_variable(self) -> None:
        # drop_corr accepts every numeric dtype, not just float — a perfectly
        # correlated *integer* pair must behave the same as a float pair,
        # dropping the row variable "a" on the tie.
        df = _df(("a", [1, 2, 3, 4]), ("b", [2, 4, 6, 8]))
        result = fit_drop_corr(df, threshold=0.9)
        assert result == ["a"]

    def test_negatively_correlated_pair_dropped(self) -> None:
        # abs(r) is what matters — strong negative correlation should also
        # trigger; the tie-break still drops the row variable "a".
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [-1.0, -2.0, -3.0]))
        result = fit_drop_corr(df, threshold=0.9)
        assert result == ["a"]

    def test_drops_column_with_higher_mean_abs_corr(self) -> None:
        # c is correlated with both a and b; a and b are roughly uncorrelated.
        # c should have the highest average |corr| and be removed.
        import numpy as np

        rng = np.random.default_rng(42)
        a = rng.standard_normal(200)
        b = rng.standard_normal(200)
        c = a + b
        df = _df(("a", a.tolist()), ("b", b.tolist()), ("c", c.tolist()))
        result = fit_drop_corr(df, threshold=0.5)
        assert "c" in result

    def test_exact_removals_fixed_seed(self) -> None:
        # Mirrors R's "high filter" test: duplicate column and a near-negative
        # duplicate should both be flagged at a low threshold.
        import numpy as np

        rng = np.random.default_rng(424)
        data = {f"V{i}": rng.standard_normal(100).tolist() for i in range(1, 6)}
        data["duplicate"] = data["V1"]
        data["V6"] = [-v + rng.uniform(0, 0.2) for v in data["V2"]]
        df = nw.from_native(pd.DataFrame(data))
        result = fit_drop_corr(df, threshold=0.5)
        # Deterministic for this fixed seed: the row variable of each correlated
        # pair is dropped — V1 (from the V1/duplicate pair) and V6 (from the
        # V2/V6 pair). Asserting exact removals makes the test falsifiable.
        assert sorted(result) == ["V1", "V6"]


# ---------------------------------------------------------------------------
# NaN / missing value handling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    def test_zero_variance_column_is_dropped(self) -> None:
        df = _df(
            ("a", [1.0, 2.0, 3.0]),
            ("b", [1.0, 2.0, 3.0]),
            ("const", [5.0, 5.0, 5.0]),
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = fit_drop_corr(df, threshold=0.9)
        assert "const" in result

    def test_zero_variance_column_warns(self, snapshot: SnapshotAssertion) -> None:
        df = _df(
            ("a", [1.0, 2.0, 3.0]),
            ("b", [4.0, 5.0, 6.0]),
            ("const", [1.0, 1.0, 1.0]),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit_drop_corr(df, threshold=0.9)
        msg = next(str(x.message) for x in w if "removed" in str(x.message).lower())
        assert msg == snapshot

    def test_too_many_na_columns_warns_and_returns_na_drops(
        self, snapshot: SnapshotAssertion
    ) -> None:
        # Two all-null columns out of three → fewer than 2 non-NA remain
        df = _df(
            ("a", [1.0, 2.0, 3.0]),
            ("b", [None, None, None]),
            ("c", [None, None, None]),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fit_drop_corr(df, threshold=0.9)
        msg = next(str(x.message) for x in w if "skipping" in str(x.message).lower())
        assert msg == snapshot
        # b and c should still be returned as drops
        assert "b" in result
        assert "c" in result

    def test_sporadic_nan_warns_and_drops_expected(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = _df(
            ("a", [1.0, None, 3.0]),
            ("b", [1.0, 2.0, None]),
            ("c", [2.0, 4.0, 6.0]),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fit_drop_corr(df, threshold=0.9)
        # The pandas correlation matrix has sporadic NaNs from pairwise-complete
        # samples; the step warns and still returns a deterministic drop.
        msg = next(str(x.message) for x in w if "sporadic" in str(x.message).lower())
        assert msg == snapshot
        assert result == ["c"]


# ---------------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------------


class TestMethod:
    def test_spearman_does_not_raise(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [1.0, 2.0, 3.0]))
        result = fit_drop_corr(df, threshold=0.5, method="spearman")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Column selection via a narwhals selector
# ---------------------------------------------------------------------------


class TestSelectorColumns:
    """``StepDropCorr.columns`` may be a narwhals selector, resolved at fit time."""

    def _mixed_df(self) -> nw.DataFrame:
        # a/b are a perfectly-correlated numeric pair; c is uncorrelated numeric;
        # label is non-numeric and must be excluded by ``selectors.numeric()``.
        return nw.from_native(
            pd.DataFrame(
                {
                    "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "b": [2.0, 4.0, 6.0, 8.0, 10.0],
                    "c": [5.0, 1.0, 4.0, 1.0, 9.0],
                    "label": ["p", "q", "r", "s", "t"],
                }
            )
        )

    def test_numeric_selector_resolves_to_numeric_columns(self) -> None:
        step = StepDropCorr(nw.selectors.numeric()).fit(self._mixed_df(), SchemaInfo())
        # Only the numeric columns are considered; the selector drops ``label``.
        assert step.schema_in().columns == ["a", "b", "c"]

    def test_numeric_selector_drops_one_of_correlated_pair(self) -> None:
        step = StepDropCorr(nw.selectors.numeric(), threshold=0.9).fit(
            self._mixed_df(), SchemaInfo()
        )
        out = nw.from_native(step.transform(self._mixed_df()))
        # One of the correlated pair (a/b) is dropped; c and the non-numeric
        # label both pass through untouched.
        assert "c" in out.columns
        assert "label" in out.columns
        assert ("a" in out.columns) ^ ("b" in out.columns)
        assert len(out.columns) == 3

    def test_string_name_selector_selects_single_column(self) -> None:
        # A bare column-name string is also a valid selection and is preserved.
        step = StepDropCorr("a").fit(self._mixed_df(), SchemaInfo())
        assert step.schema_in().columns == ["a"]


class TestPolarsSelectorColumns:
    """``columns`` may be a polars selector when the frame is polars-backed."""

    def _mixed_pl(self) -> nw.DataFrame:
        # Same shape as the narwhals case, but polars-backed so a polars selector
        # can be resolved against the native frame.
        return nw.from_native(
            pl.DataFrame(
                {
                    "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "b": [2.0, 4.0, 6.0, 8.0, 10.0],
                    "c": [5.0, 1.0, 4.0, 1.0, 9.0],
                    "label": ["p", "q", "r", "s", "t"],
                }
            )
        )

    def test_polars_selector_resolves_to_selected_columns(self) -> None:
        step = StepDropCorr(cs.numeric()).fit(self._mixed_pl(), SchemaInfo())
        # The polars selector resolves to the numeric columns, dropping label.
        assert step.schema_in().columns == ["a", "b", "c"]

    def test_polars_selector_drops_one_of_correlated_pair(self) -> None:
        step = StepDropCorr(cs.numeric(), threshold=0.9).fit(
            self._mixed_pl(), SchemaInfo()
        )
        out = nw.from_native(step.transform(self._mixed_pl()))
        assert "c" in out.columns
        assert "label" in out.columns
        assert ("a" in out.columns) ^ ("b" in out.columns)
        assert len(out.columns) == 3

    def test_polars_selector_resolves_against_pandas_frame(self) -> None:
        # Resolution is pure schema name-resolution (Schema.to_polars), so a
        # polars selector works against a pandas-backed frame too.
        df = nw.from_native(
            pd.DataFrame(
                {
                    "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "b": [2.0, 4.0, 6.0, 8.0, 10.0],
                    "c": [5.0, 1.0, 4.0, 1.0, 9.0],
                    "label": ["p", "q", "r", "s", "t"],
                }
            )
        )
        step = StepDropCorr(cs.numeric()).fit(df, SchemaInfo())
        assert step.schema_in().columns == ["a", "b", "c"]
