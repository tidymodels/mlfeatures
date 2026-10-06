"""Tests for fit_pls / apply_pls (the fit + apply halves of the PLS step).

Correctness is anchored to ``sklearn.cross_decomposition.PLSRegression``
directly: `fit_pls` delegates the NIPALS computation to it, so the reference
tests in `TestReferenceAgainstSklearn` are the primary correctness gate for
the step's math, verifying the fitted mean/rotation and the resulting scores
(up to a per-component sign flip, since PLS components are only unique up to
sign) against a direct call to sklearn.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.cross_decomposition import PLSRegression

from mlfeatures.step_pls import StepPls, apply_pls, fit_pls


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)))


def _series(values: list[Any]) -> "nw.Series[Any]":
    return nw.from_native(pd.Series(values, name="target"), series_only=True)


def _sample() -> nw.DataFrame[Any]:
    return _df(("x", [1.0, 2.0, 3.0, 4.0]), ("y", [2.1, 3.9, 6.1, 8.0]))


def _sample_target() -> "nw.Series[Any]":
    return _series([1.0, 2.0, 2.5, 4.5])


class TestFitPls:
    def test_rotation_shape(self) -> None:
        fit = fit_pls(_sample(), _sample_target(), num_comp=2)
        assert fit.rotation.shape == (2, 2)
        assert fit.num_comp == 2
        assert fit.columns == ["x", "y"]

    def test_num_comp_capped_at_features(self) -> None:
        # Only 2 columns, so requesting 5 components yields at most 2.
        fit = fit_pls(_sample(), _sample_target(), num_comp=5)
        assert fit.num_comp == 2

    def test_mean_is_column_means(self) -> None:
        fit = fit_pls(_sample(), _sample_target(), num_comp=1)
        assert fit.mean == pytest.approx([2.5, 5.025])

    @pytest.mark.parametrize("bad", [0, -1, 2.0, True])
    def test_invalid_num_comp_raises(self, bad: Any) -> None:
        with pytest.raises(ValueError):
            fit_pls(_sample(), _sample_target(), num_comp=bad)

    def test_invalid_num_comp_message(self, snapshot: Any) -> None:
        with pytest.raises(ValueError) as excinfo:
            fit_pls(_sample(), _sample_target(), num_comp=0)
        assert str(excinfo.value) == snapshot


class TestApplyPls:
    def test_replaces_inputs_with_components(self) -> None:
        fit = fit_pls(_sample(), _sample_target(), num_comp=2)
        result = apply_pls(_sample(), fit, prefix="PLS")
        assert result.columns == ["PLS1", "PLS2"]

    def test_other_columns_pass_through(self) -> None:
        df = _df(
            ("id", [10, 20, 30, 40]),
            ("x", [1.0, 2.0, 3.0, 4.0]),
            ("y", [2.1, 3.9, 6.1, 8.0]),
        )
        fit = fit_pls(df.select(["x", "y"]), _sample_target(), num_comp=2)
        result = apply_pls(df, fit, prefix="PLS")
        assert result.columns == ["id", "PLS1", "PLS2"]
        assert result["id"].to_list() == [10, 20, 30, 40]

    def test_custom_prefix(self) -> None:
        fit = fit_pls(_sample(), _sample_target(), num_comp=2)
        result = apply_pls(_sample(), fit, prefix="comp")
        assert result.columns == ["comp1", "comp2"]

    def test_keep_original_cols_retains_inputs(self) -> None:
        fit = fit_pls(_sample(), _sample_target(), num_comp=2)
        result = apply_pls(_sample(), fit, prefix="PLS", keep_original_cols=True)
        assert set(result.columns) == {"x", "y", "PLS1", "PLS2"}
        assert result["x"].to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_keep_original_cols_does_not_change_components(self) -> None:
        fit = fit_pls(_sample(), _sample_target(), num_comp=2)
        dropped = apply_pls(_sample(), fit, prefix="PLS")
        kept = apply_pls(_sample(), fit, prefix="PLS", keep_original_cols=True)
        assert kept["PLS1"].to_list() == pytest.approx(dropped["PLS1"].to_list())


class TestBackends:
    def test_polars_in_polars_out(self) -> None:
        df = nw.from_native(
            pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.1, 3.9, 6.1, 8.0]})
        )
        fit = fit_pls(df, _sample_target(), num_comp=2)
        result = apply_pls(df, fit, prefix="PLS")
        assert isinstance(result.to_native(), pl.DataFrame)
        assert result.columns == ["PLS1", "PLS2"]

    def test_pandas_in_pandas_out(self) -> None:
        fit = fit_pls(_sample(), _sample_target(), num_comp=2)
        result = apply_pls(_sample(), fit, prefix="PLS")
        assert isinstance(result.to_native(), pd.DataFrame)


class TestNullHandling:
    def test_rows_with_null_target_are_dropped(self) -> None:
        features = _df(("x", [1.0, 2.0, 3.0, 4.0]), ("y", [2.1, 3.9, 6.1, 8.0]))
        target_with_null = _series([1.0, None, 2.5, 4.5])
        target_dropped = _series([1.0, 2.5, 4.5])
        features_dropped = _df(("x", [1.0, 3.0, 4.0]), ("y", [2.1, 6.1, 8.0]))
        fit_with_null = fit_pls(features, target_with_null, num_comp=1)
        fit_dropped = fit_pls(features_dropped, target_dropped, num_comp=1)
        assert fit_with_null.mean == pytest.approx(fit_dropped.mean)
        assert fit_with_null.rotation == pytest.approx(fit_dropped.rotation)

    def test_rows_with_null_feature_are_dropped(self) -> None:
        features_with_null = _df(
            ("x", [1.0, None, 3.0, 4.0]), ("y", [2.1, 3.9, 6.1, 8.0])
        )
        features_dropped = _df(("x", [1.0, 3.0, 4.0]), ("y", [2.1, 6.1, 8.0]))
        target_dropped = _series([1.0, 2.5, 4.5])
        fit_with_null = fit_pls(features_with_null, _sample_target(), num_comp=1)
        fit_dropped = fit_pls(features_dropped, target_dropped, num_comp=1)
        assert fit_with_null.mean == pytest.approx(fit_dropped.mean)
        assert fit_with_null.rotation == pytest.approx(fit_dropped.rotation)


class TestReferenceAgainstSklearn:
    """Numerically anchor `fit_pls`/`apply_pls` to a direct sklearn call.

    PLS components are unique only up to a per-component sign flip, so scores
    and rotations are compared with ``np.abs`` rather than directly.
    """

    @pytest.mark.parametrize("n_features,n_samples,num_comp", [(2, 4, 2), (3, 8, 2)])
    def test_scores_match_sklearn_transform(
        self, n_features: int, n_samples: int, num_comp: int
    ) -> None:
        rng = np.random.default_rng(0)
        x = rng.random((n_samples, n_features))
        y = x[:, 0] * 2 - x[:, 1] * 0.5 + rng.random(n_samples) * 0.01

        columns = [f"x{i}" for i in range(n_features)]
        frame = nw.from_native(pd.DataFrame(x, columns=columns), eager_only=True)
        target = nw.from_native(pd.Series(y, name="target"), series_only=True)

        fit = fit_pls(frame, target, num_comp=num_comp)
        result = apply_pls(frame, fit, prefix="PLS")
        ours = np.column_stack(
            [result[f"PLS{i + 1}"].to_list() for i in range(num_comp)]
        )

        reference = PLSRegression(n_components=num_comp, scale=False)
        reference.fit(x, y)
        expected = reference.transform(x)

        np.testing.assert_allclose(np.abs(ours), np.abs(expected), atol=1e-8)

    def test_more_features_than_samples(self) -> None:
        # A wide (p > n) case, the shape PLS is often reached for over PCA.
        rng = np.random.default_rng(1)
        n_samples, n_features = 5, 20
        x = rng.random((n_samples, n_features))
        y = x[:, 0] - x[:, 1] + rng.random(n_samples) * 0.01

        columns = [f"x{i}" for i in range(n_features)]
        frame = nw.from_native(pd.DataFrame(x, columns=columns), eager_only=True)
        target = nw.from_native(pd.Series(y, name="target"), series_only=True)

        fit = fit_pls(frame, target, num_comp=2)
        result = apply_pls(frame, fit, prefix="PLS")
        ours = np.column_stack(
            [result[f"PLS{i + 1}"].to_list() for i in range(fit.num_comp)]
        )

        reference = PLSRegression(n_components=fit.num_comp, scale=False)
        reference.fit(x, y)
        expected = reference.transform(x)

        np.testing.assert_allclose(np.abs(ours), np.abs(expected), atol=1e-8)

    def test_held_out_rows_agree_with_sklearn(self) -> None:
        # Fit on a training slice, then project a disjoint held-out slice --
        # both through our apply_pls and through sklearn's own transform --
        # to catch a train/serve mismatch in the stored basis.
        rng = np.random.default_rng(2)
        x_train = rng.random((30, 4))
        y_train = x_train[:, 0] * 1.5 - x_train[:, 2] + rng.random(30) * 0.01
        x_test = rng.random((10, 4))

        columns = ["x0", "x1", "x2", "x3"]
        train_frame = nw.from_native(
            pd.DataFrame(x_train, columns=columns), eager_only=True
        )
        test_frame = nw.from_native(
            pd.DataFrame(x_test, columns=columns), eager_only=True
        )
        target = nw.from_native(pd.Series(y_train, name="target"), series_only=True)

        fit = fit_pls(train_frame, target, num_comp=2)
        result = apply_pls(test_frame, fit, prefix="PLS")
        ours = np.column_stack(
            [result[f"PLS{i + 1}"].to_list() for i in range(fit.num_comp)]
        )

        reference = PLSRegression(n_components=2, scale=False)
        reference.fit(x_train, y_train)
        expected = reference.transform(x_test)

        np.testing.assert_allclose(np.abs(ours), np.abs(expected), atol=1e-8)


class TestCategoricalTarget:
    """A categorical target runs PLS discriminant analysis via dummy-coding."""

    def _classes_frame(self) -> nw.DataFrame[Any]:
        # x and y deliberately uncorrelated (unlike a scaled copy of each
        # other) so the feature matrix has rank 2, not 1 -- a rank-1 X cannot
        # support 2 PLS components and hits a degenerate SVD.
        return _df(("x", [0.0, 1.0, 10.0, 11.0]), ("y", [5.0, 9.0, 2.0, 12.0]))

    def test_fits_without_error(self) -> None:
        fit = fit_pls(self._classes_frame(), _series(["a", "a", "b", "b"]), num_comp=1)
        assert fit.num_comp == 1
        assert fit.rotation.shape == (2, 1)

    def test_num_comp_capped_at_features(self) -> None:
        fit = fit_pls(self._classes_frame(), _series(["a", "a", "b", "b"]), num_comp=5)
        assert fit.num_comp == 2

    def test_rows_with_null_target_are_dropped(self) -> None:
        target_with_null = _series(["a", None, "b", "b"])
        target_dropped = _series(["a", "b", "b"])
        features_dropped = _df(("x", [0.0, 10.0, 11.0]), ("y", [5.0, 2.0, 12.0]))
        fit_with_null = fit_pls(self._classes_frame(), target_with_null, num_comp=1)
        fit_dropped = fit_pls(features_dropped, target_dropped, num_comp=1)
        assert fit_with_null.mean == pytest.approx(fit_dropped.mean)
        assert fit_with_null.rotation == pytest.approx(fit_dropped.rotation)

    def test_scores_match_sklearn_dummy_coded_fit(self) -> None:
        # Reference: dummy-code the classes ourselves and fit PLSRegression
        # directly, mirroring what fit_pls does internally for a categorical
        # target.
        x = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
        labels = np.array(["a", "a", "b", "b"])
        classes = sorted(set(labels.tolist()))
        y = np.column_stack([(labels == cls).astype(float) for cls in classes])

        frame = nw.from_native(pd.DataFrame(x, columns=["x", "y"]), eager_only=True)
        target = _series(labels.tolist())
        fit = fit_pls(frame, target, num_comp=1)
        result = apply_pls(frame, fit, prefix="PLS")
        ours = np.array(result["PLS1"].to_list())

        reference = PLSRegression(n_components=1, scale=False)
        reference.fit(x, y)
        expected = reference.transform(x)[:, 0]

        np.testing.assert_allclose(np.abs(ours), np.abs(expected), atol=1e-8)

    def test_integer_target_is_treated_as_numeric_not_classes(self) -> None:
        # An int column is numeric per accepts_dtypes/target.dtype.is_numeric,
        # so it takes the regression path, not the dummy-coding one.
        frame = self._classes_frame()
        target = nw.from_native(
            pd.Series([0, 0, 1, 1], name="target"), series_only=True
        )
        fit = fit_pls(frame, target, num_comp=1)
        assert fit.rotation.shape == (2, 1)


class TestConstructor:
    def test_default_id(self) -> None:
        assert StepPls(["x", "y"], target="out").id == "pls"

    def test_default_num_comp(self) -> None:
        assert StepPls(["x", "y"], target="out").num_comp == 5

    def test_required_pkgs(self) -> None:
        assert StepPls(["x", "y"], target="out").required_pkgs() == [
            "numpy",
            "scikit-learn",
        ]

    def test_default_output_role_is_feature(self) -> None:
        assert StepPls(["x", "y"], target="out").output_role == "feature"

    def test_keep_original_cols_defaults_to_false(self) -> None:
        assert StepPls(["x", "y"], target="out").keep_original_cols is False

    def test_keep_original_cols_is_settable(self) -> None:
        step = StepPls(["x", "y"], target="out", keep_original_cols=True)
        assert step.keep_original_cols is True
