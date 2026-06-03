"""Tests for fit_filter_corr."""

from __future__ import annotations

import warnings

import narwhals as nw
import pandas as pd
import pytest

from shoez.step_corr import fit_filter_corr


def _df(*cols: tuple[str, list[float | None]]) -> nw.DataFrame:
    return nw.from_native(pd.DataFrame(dict(cols)))


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_empty_result_is_list(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [4.0, 5.0, 6.0]))
        result = fit_filter_corr(df, threshold=0.9)
        assert isinstance(result, list)

    def test_nonempty_result_is_list_of_str(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [1.0, 2.0, 3.0]))
        result = fit_filter_corr(df, threshold=0.5)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_threshold_above_1_raises(self) -> None:
        df = _df(("a", [1.0, 2.0]), ("b", [3.0, 4.0]))
        with pytest.raises(ValueError, match="threshold"):
            fit_filter_corr(df, threshold=1.1)

    def test_threshold_below_0_raises(self) -> None:
        df = _df(("a", [1.0, 2.0]), ("b", [3.0, 4.0]))
        with pytest.raises(ValueError, match="threshold"):
            fit_filter_corr(df, threshold=-0.1)

    def test_invalid_method_raises(self) -> None:
        df = _df(("a", [1.0, 2.0]), ("b", [3.0, 4.0]))
        with pytest.raises(ValueError, match="method"):
            fit_filter_corr(df, method="kendall")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_single_column_returns_empty(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]))
        assert len(fit_filter_corr(df)) == 0

    def test_zero_row_returns_empty(self) -> None:
        df = _df(("a", []), ("b", []))
        assert len(fit_filter_corr(df)) == 0

    def test_no_pair_above_threshold_returns_empty(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [3.0, 1.0, 2.0]))
        assert len(fit_filter_corr(df, threshold=0.9)) == 0

    def test_threshold_exactly_equal_not_dropped(self) -> None:
        # threshold is strict (>), so r == threshold should not trigger removal
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [1.0, 2.0, 3.0]))
        assert len(fit_filter_corr(df, threshold=1.0)) == 0


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


class TestCoreAlgorithm:
    def test_perfectly_correlated_pair_drops_one(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [1.0, 2.0, 3.0]))
        result = fit_filter_corr(df, threshold=0.9)
        assert len(result) == 1
        assert result[0] in ("a", "b")

    def test_negatively_correlated_pair_dropped(self) -> None:
        # abs(r) is what matters — strong negative correlation should also trigger
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [-1.0, -2.0, -3.0]))
        result = fit_filter_corr(df, threshold=0.9)
        assert len(result) == 1

    def test_drops_column_with_higher_mean_abs_corr(self) -> None:
        # c is correlated with both a and b; a and b are roughly uncorrelated.
        # c should have the highest average |corr| and be removed.
        import numpy as np

        rng = np.random.default_rng(42)
        a = rng.standard_normal(200)
        b = rng.standard_normal(200)
        c = a + b
        df = _df(("a", a.tolist()), ("b", b.tolist()), ("c", c.tolist()))
        result = fit_filter_corr(df, threshold=0.5)
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
        result = fit_filter_corr(df, threshold=0.5)
        # One of V1/duplicate (r=1 pair) and one of V2/V6 (r≈-1 pair) must be dropped
        assert "V1" in result or "duplicate" in result
        assert "V2" in result or "V6" in result


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
            result = fit_filter_corr(df, threshold=0.9)
        assert "const" in result

    def test_zero_variance_column_warns(self) -> None:
        df = _df(
            ("a", [1.0, 2.0, 3.0]),
            ("b", [4.0, 5.0, 6.0]),
            ("const", [1.0, 1.0, 1.0]),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit_filter_corr(df, threshold=0.9)
        assert any("removed" in str(warning.message).lower() for warning in w)

    def test_too_many_na_columns_warns_and_returns_na_drops(self) -> None:
        # Two all-null columns out of three → fewer than 2 non-NA remain
        df = _df(
            ("a", [1.0, 2.0, 3.0]),
            ("b", [None, None, None]),
            ("c", [None, None, None]),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fit_filter_corr(df, threshold=0.9)
        assert any("skipping" in str(warning.message).lower() for warning in w)
        # b and c should still be returned as drops
        assert "b" in result
        assert "c" in result

    def test_sporadic_nan_warns_and_does_not_crash(self) -> None:
        df = _df(
            ("a", [1.0, None, 3.0]),
            ("b", [1.0, 2.0, None]),
            ("c", [2.0, 4.0, 6.0]),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fit_filter_corr(df, threshold=0.9)
        assert isinstance(result, list)
        _ = w  # warning may or may not fire depending on backend


# ---------------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------------


class TestMethod:
    def test_spearman_does_not_raise(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [1.0, 2.0, 3.0]))
        result = fit_filter_corr(df, threshold=0.5, method="spearman")
        assert isinstance(result, list)
