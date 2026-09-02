"""The executable step contract: `StepConformanceTest`.

``design/step-contract.md`` defines the properties every step must satisfy. This
module makes them runnable against *any* step, first-party or third-party, by
parameterizing them over two hooks a subclass supplies: `make_step` and
`training_data`.

Ships as part of the installed package (unlike ``tests/``) so a plugin author
can run the same contract shoez's own steps are held to::

    from shoez.testing import StepConformanceTest

    class TestMyStepConformance(StepConformanceTest):
        def make_step(self):
            return StepMyThing(["a"])

        training_data = {"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]}

pytest collects the inherited ``test_*`` methods from the subclass. The base
class is named ``Step...`` rather than ``Test...`` precisely so pytest does not
collect it on its own.

Steps for which a check is meaningless opt out through the hooks rather than by
omitting a test: `make_empty_step` returning ``None`` skips the empty-selection
check with an explicit reason, which keeps "considered and not applicable"
distinguishable from "forgotten".
"""

from __future__ import annotations

import inspect
import math
import pickle
from typing import TYPE_CHECKING, Any, Callable, ClassVar, cast

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from shoez import _frame
from shoez.recipe import FittedRecipe, Recipe
from shoez.types import DataFrame, KeepOriginals, SchemaInfo

if TYPE_CHECKING:
    from collections.abc import Mapping

    from shoez.step import Step

__all__ = [
    "StepConformanceTest",
    "assert_backend_parity",
    "fit_isolated",
    "fit_transform_isolated",
]


def _is_missing(value: Any) -> bool:
    """Treat polars ``None`` and pandas ``NaN`` as the same missing marker."""
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _assert_columns_equal(
    name: str, pd_values: list[Any], pl_values: list[Any]
) -> None:
    """Assert one column's values match across backends, missing-aware."""
    assert len(pd_values) == len(pl_values), (
        f"column {name!r} length divergence: pandas {len(pd_values)} != "
        f"polars {len(pl_values)}"
    )
    for i, (pd_v, pl_v) in enumerate(zip(pd_values, pl_values)):
        pd_missing, pl_missing = _is_missing(pd_v), _is_missing(pl_v)
        if pd_missing or pl_missing:
            assert pd_missing and pl_missing, (
                f"column {name!r} row {i} missing-value divergence: "
                f"pandas {pd_v!r} != polars {pl_v!r}"
            )
            continue
        if isinstance(pd_v, float) or isinstance(pl_v, float):
            assert pd_v == pytest.approx(pl_v), (
                f"column {name!r} row {i} value divergence: "
                f"pandas {pd_v!r} != polars {pl_v!r}"
            )
        else:
            assert pd_v == pl_v, (
                f"column {name!r} row {i} value divergence: "
                f"pandas {pd_v!r} != polars {pl_v!r}"
            )


def _cast_columns(
    df: pd.DataFrame | pl.DataFrame,
    dtype_casts: Mapping[str, type[nw.dtypes.DType]],
) -> Any:
    """Cast ``dtype_casts``' columns to their declared narwhals dtype.

    A plain Python list can't materialize some dtypes identically on both
    backends -- e.g. a list of ``datetime.date`` becomes pandas ``Object``
    (pandas can only represent narwhals' ``Date`` through a pyarrow-backed
    column) but polars ``Date`` natively. Casting after construction, on both
    backends alike, is what lets a Date-only step's fixture materialize as the
    same dtype either way.
    """
    if not dtype_casts:
        return df
    frame = nw.from_native(df, eager_only=True)
    frame = frame.with_columns(
        # ``Expr.cast()``'s overloads only cover its own concrete dtype
        # classes; `dtype_casts` is typed as the general
        # `type[nw.dtypes.DType]` so a test can name any of them, so the cast
        # here is wider than any single overload matches.
        nw.col(name).cast(cast("Any", dtype))
        for name, dtype in dtype_casts.items()
    )
    return frame.to_native()


def assert_backend_parity(
    make_recipe: Callable[[], Recipe],
    data: Mapping[str, list[Any]],
    *,
    dtype_casts: Mapping[str, type[nw.dtypes.DType]] | None = None,
) -> None:
    """Assert a step produces identical output on pandas and polars.

    Fits and transforms ``make_recipe()`` on the *same* ``data`` materialized as
    both a pandas and a polars frame, then asserts the two narwhals results are
    identical: same column names in the same order, same dtypes, *and* the same
    values (missing-aware, with float columns compared approximately). This
    catches both output-dtype divergence between backends -- the class of bug
    PR #74 fixed for ``StepDummy``, where narwhals' ``lit(None)`` branch unified
    pandas indicator columns to ``Object`` while polars stayed numeric -- and
    value/estimate divergence, where a step computes different numbers depending
    on which backend it fitted and transformed on.

    ``make_recipe`` must return a *fresh, unfitted* recipe on each call so the
    two backends do not share fitted estimates. ``dtype_casts`` names columns
    to cast to a specific narwhals dtype on both backends before fitting --
    see `_cast_columns`.
    """
    casts = dtype_casts or {}
    pd_df = _cast_columns(pd.DataFrame(data), casts)
    pl_df = _cast_columns(pl.DataFrame(data), casts)
    pd_out = make_recipe().fit(pd_df).transform(pd_df)
    pl_out = make_recipe().fit(pl_df).transform(pl_df)
    pd_nw = nw.from_native(pd_out, eager_only=True)
    pl_nw = nw.from_native(pl_out, eager_only=True)

    pd_schema, pl_schema = pd_nw.schema, pl_nw.schema
    assert dict(pd_schema) == dict(pl_schema), (
        f"backend dtype divergence: pandas {dict(pd_schema)} != polars {dict(pl_schema)}"
    )

    pd_cols = pd_nw.to_dict(as_series=False)
    pl_cols = pl_nw.to_dict(as_series=False)
    for name in pd_schema:
        _assert_columns_equal(name, pd_cols[name], pl_cols[name])


def fit_isolated(step: Step, data: DataFrame) -> Step:
    """Fit a single step against a stubbed ``SchemaInfo``, returning the fitted step.

    Exercises the step contract with no ``Recipe`` in the loop: the info is a
    bare ``SchemaInfo.from_frame`` snapshot (no lineage DAG, no role overrides),
    which is all a step reads for its own logic.
    """
    frame = _frame.from_native(data)
    return step.fit(frame, SchemaInfo.from_frame(frame))


def fit_transform_isolated(step: Step, data: DataFrame) -> DataFrame:
    """Fit and transform a single step against a stubbed ``SchemaInfo``.

    Returns the native output frame so callers can compare its columns against
    the recipe-driven path. See `fit_isolated` for the no-``Recipe`` rationale.
    """
    frame = _frame.from_native(data)
    return _frame.to_native(fit_isolated(step, data).transform(frame))


class StepConformanceTest:
    """The step contract, runnable against any step.

    Subclass it, implement `make_step`, set `training_data`, and pytest runs the
    whole contract. Every check is a generic property from
    ``design/step-contract.md``, never step-specific behaviour; a step's own
    logic belongs in its unit tests.
    """

    #: Columns the step is fitted and transformed on. Must contain whatever the
    #: step's selector needs, and is materialized as both pandas and polars.
    training_data: ClassVar[Mapping[str, list[Any]]] = {}

    #: Columns of `training_data` to cast to a specific narwhals dtype on both
    #: backends before fitting. Needed for a dtype a plain Python list can't
    #: materialize identically on both backends -- e.g. ``{"d": nw.Date}`` for
    #: a step that only accepts Date columns (see `_cast_columns`). Empty for
    #: every step whose `training_data` values already produce the same dtype
    #: on both backends.
    dtype_casts: ClassVar[Mapping[str, type[nw.dtypes.DType]]] = {}

    def make_step(self) -> Step:
        """Return a *fresh, unfitted* step. Called repeatedly, never reused."""
        raise NotImplementedError("conformance subclasses must implement make_step()")

    # -- optional hooks ----------------------------------------------------

    def make_empty_step(self) -> Step | None:
        """A step whose selection resolves to nothing, or ``None`` if not applicable.

        The default takes `make_step` and empties its ``columns`` selector,
        which is right for every selector-taking step. Steps with no input
        selector (``StepIntercept``) have no empty-selection case at all, so the
        default returns ``None`` for them and the check skips with a reason
        rather than silently vanishing.
        """
        step = self.make_step()
        if not hasattr(step, "columns"):
            return None
        step.columns = []
        return step

    #: Attribute holding fit-time estimates, checked to be ``None`` before fit
    #: and populated after. Set to ``None`` for steps that estimate nothing.
    estimates_attr: ClassVar[str | None] = "_estimates"

    #: Whether the step can round-trip through JSON. Set ``False`` for a step
    #: that carries user callables it needs at transform time (``StepMutate``);
    #: the JSON check then asserts it *rejects* serialization loudly rather than
    #: silently dropping them. Either way the behaviour is pinned by a test.
    supports_json: ClassVar[bool] = True

    # -- fixtures ----------------------------------------------------------

    def _frame(self) -> pd.DataFrame:
        return cast(
            "pd.DataFrame",
            _cast_columns(pd.DataFrame(dict(self.training_data)), self.dtype_casts),
        )

    def _fitted(self) -> tuple[FittedRecipe, pd.DataFrame]:
        train = self._frame()
        return Recipe(steps=[self.make_step()]).fit(train), train

    # -- the contract ------------------------------------------------------

    def test_optional_arguments_are_keyword_only(self) -> None:
        """Only required arguments are positional; every option is keyword-only.

        Phrased as "no parameter with a default may be positional" so it needs no
        per-step allowlist: required arguments (``columns``, ``target``,
        ``mutations``, ``predicate``) have no default and stay positional, while
        everything else, ``id`` included, must sit after the ``*``.
        """
        params = inspect.signature(type(self.make_step()).__init__).parameters
        positional = [
            name
            for name, p in params.items()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            and p.default is not inspect.Parameter.empty
        ]
        assert not positional, (
            f"optional parameter(s) {positional} can be passed positionally; "
            "add a '*' after the required argument(s) so every option is "
            "keyword-only"
        )

    def test_constructor_delegates_to_step_init(self) -> None:
        """``__init__`` hands the framework-owned attributes to ``Step.__init__``.

        ``id``, ``trained``, ``keep_original_cols``, and ``output_role`` are
        annotated on ``Step`` without defaults, so a step that never calls
        ``super().__init__()`` fails this with an ``AttributeError`` rather than
        surfacing later as a missing attribute mid-fit.
        """
        step = self.make_step()
        assert isinstance(step.id, str) and step.id
        assert step.trained is False
        assert isinstance(step.keep_original_cols, bool)
        assert step.output_role is None or isinstance(step.output_role, str)

    def test_keep_original_cols_follows_the_mode(self) -> None:
        """The step's `Mode` decides whether ``keep_original_cols`` can vary.

        ``ONE_TO_MANY`` and ``MANY_TO_MANY`` steps expose it as a constructor
        argument defaulting to ``False``; every other mode fixes it (``True`` for
        the modes that keep their inputs, ``False`` for the ones that consume
        them) and must not offer the argument at all. This is the rule from
        ``design/step-contract.md``, checked rather than merely written down.
        """
        step = self.make_step()
        policy = KeepOriginals.for_mode(type(step).mode)
        params = inspect.signature(type(step).__init__).parameters
        param = params.get("keep_original_cols")
        fixed = policy.fixed_value()
        if fixed is None:
            assert param is not None, (
                f"a step with mode {type(step).mode.value!r} must expose "
                "'keep_original_cols' as a constructor argument"
            )
            assert param.default is False
        else:
            assert param is None, (
                f"a step with mode {type(step).mode.value!r} must not expose "
                f"'keep_original_cols'; it is fixed at {fixed!r}"
            )
            assert step.keep_original_cols is fixed

    def test_keep_original_cols_is_read_only(self) -> None:
        """The value settles at construction; reassigning it would break the step."""
        step = self.make_step()
        with pytest.raises(AttributeError):
            step.keep_original_cols = True  # type: ignore[misc]

    def test_empty_selection_is_noop(self) -> None:
        """An empty column selection is a no-op: the frame passes through unchanged."""
        step = self.make_empty_step()
        if step is None:
            pytest.skip("step takes no input selector; empty selection cannot occur")
        df = self._frame()
        out = Recipe(steps=[step]).fit(df).transform(df)
        assert list(out.columns) == list(df.columns)
        pd.testing.assert_frame_equal(out, df)

    def test_zero_row_transform(self) -> None:
        """transform() on a 0-row frame yields schema-correct, empty output."""
        fitted, train = self._fitted()
        expected = list(fitted.transform(train).columns)
        out = fitted.transform(train.head(0))
        assert list(out.columns) == expected
        assert out.shape[0] == 0

    def test_one_row_transform(self) -> None:
        """transform() on a 1-row frame yields schema-correct output."""
        fitted, train = self._fitted()
        expected = list(fitted.transform(train).columns)
        out = fitted.transform(train.head(1))
        assert list(out.columns) == expected
        assert out.shape[0] == 1

    def test_pickle_preserves_transform_output(self) -> None:
        """A pickled fitted step reproduces the same transform() output."""
        fitted, train = self._fitted()
        before = fitted.transform(train)
        restored = pickle.loads(pickle.dumps(fitted))
        pd.testing.assert_frame_equal(restored.transform(train), before)

    def test_json_preserves_transform_output(self) -> None:
        """A fitted step round-trips through JSON to identical transform() output.

        Doubles as the no-closure guarantee (``design/serialization.md``): a
        closure cannot be JSON-encoded, so a passing round-trip proves the step
        and its selectors captured only plain data.
        """
        fitted, train = self._fitted()
        if not self.supports_json:
            with pytest.raises(TypeError):
                fitted.to_json()
            return
        before = fitted.transform(train)
        restored = fitted.from_json(fitted.to_json())
        pd.testing.assert_frame_equal(restored.transform(train), before)

    def test_fit_is_deterministic(self) -> None:
        """Two independent fits on the same data yield identical transform output."""
        first, train = self._fitted()
        second, _ = self._fitted()
        pd.testing.assert_frame_equal(first.transform(train), second.transform(train))

    def test_id_is_stable(self) -> None:
        """The same step type with the same hyperparameters yields the same id."""
        first, _ = self._fitted()
        second, _ = self._fitted()
        assert first.steps[0].id == second.steps[0].id

    def test_estimates_populated_after_fit(self) -> None:
        """Fit-time estimates are None on a fresh step and populated by fit()."""
        attr = self.estimates_attr
        step = self.make_step()
        if attr is None or not hasattr(step, attr):
            pytest.skip("step declares no fit-time estimates")
        assert getattr(step, attr) is None
        assert getattr(fit_isolated(step, self._frame()), attr) is not None

    def test_output_parents_matches_added_columns(self) -> None:
        """output_parents() declares exactly the added columns, each with valid parents."""
        fitted, train = self._fitted()
        step = fitted.steps[0]
        pre = set(train.columns)
        added = set(fitted.transform(train).columns) - pre
        parents = step.output_parents()
        assert set(parents) == added
        assert all(
            parent in pre for parents_of in parents.values() for parent in parents_of
        )

    def test_pandas_and_polars_output_match(self) -> None:
        """fit + transform yields identical output (schema and values) on both backends."""
        assert_backend_parity(
            lambda: Recipe(steps=[self.make_step()]),
            dict(self.training_data),
            dtype_casts=self.dtype_casts,
        )

    def test_isolated_fit_matches_recipe(self) -> None:
        """A step fits and transforms against a stubbed SchemaInfo, with no Recipe."""
        fitted, train = self._fitted()
        expected = list(fitted.transform(train).columns)
        out = fit_transform_isolated(self.make_step(), train)
        assert list(out.columns) == expected

    def test_transform_does_not_reestimate(self) -> None:
        """transform() applies stored state and never re-estimates from new data.

        The headline invariant on `Step`: whatever a step learns is computed once
        at fit time, so ``transform()`` must be row-wise with respect to that
        stored state. Transforming a *subset* of the training frame is therefore
        the same as transforming the whole frame and then taking the subset. A
        step that recomputes its statistics from whatever it is handed diverges
        here, because the subset has different statistics.

        Every other check in this suite transforms the training frame itself,
        where re-estimation is invisible: the recomputed statistics come out the
        same. This is the one that sees it.
        """
        fitted, train = self._fitted()
        from_subset = cast("pd.DataFrame", fitted.transform(train.head(2)))
        from_whole = cast("pd.DataFrame", fitted.transform(train)).head(
            len(from_subset)
        )
        pd.testing.assert_frame_equal(
            from_subset.reset_index(drop=True), from_whole.reset_index(drop=True)
        )

    def test_transform_is_repeatable(self) -> None:
        """transform() is pure: re-transforming the same frame yields identical output."""
        fitted, train = self._fitted()
        pd.testing.assert_frame_equal(fitted.transform(train), fitted.transform(train))

    def test_accepts_dtypes_is_well_formed(self) -> None:
        """accepts_dtypes() returns ANY_DTYPE or a set of narwhals dtype classes.

        The recipe enforces the declaration before ``fit()``, so a set holding
        anything other than dtype *classes* (a dtype instance, a string name)
        would silently reject every column instead.
        """
        from shoez.types import ANY_DTYPE

        accepted = self.make_step().accepts_dtypes()
        if accepted is ANY_DTYPE:
            return
        assert isinstance(accepted, set)
        assert accepted
        for dtype in accepted:
            assert isinstance(dtype, type)
            assert issubclass(dtype, nw.dtypes.DType)

    def test_produces_sparse_is_bool(self) -> None:
        """produces_sparse is a plain bool."""
        fitted, _ = self._fitted()
        assert isinstance(fitted.steps[0].produces_sparse, bool)
