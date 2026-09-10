"""Tests for :mod:`shoez.step_holiday`.

Built as polars frames throughout; ``StepHoliday`` itself is backend-agnostic
(see `test_step_holiday_infrastructure.py`, which exercises the pandas side
via pyarrow and `shoez.testing`'s `dtype_casts` hook).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import narwhals.stable.v2 as nw
import polars as pl
import pytest
from dateutil.rrule import SA, SU, WEEKLY, rrule
from syrupy.assertion import SnapshotAssertion

from shoez import Mode, SchemaInfo
from shoez._dependencies import MissingDependencyError
from shoez.step_holiday import (
    HolidayFit,
    StepHoliday,
    apply_holiday,
    fit_holiday,
)


def _dates() -> list[date]:
    return [date(2023, 12, 24), date(2023, 12, 25), date(2023, 1, 1), date(2023, 3, 1)]


def _data() -> dict[str, list[Any]]:
    return {"d": _dates(), "n": [1.0, 2.0, 3.0, 4.0]}


def _df(data: dict[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pl.DataFrame(data))


def _weekend_rule(dtstart: date = date(2023, 1, 1)) -> rrule:
    return rrule(WEEKLY, byweekday=(SA, SU), dtstart=dtstart)


# ---------------------------------------------------------------------------
# fit_holiday / apply_holiday
# ---------------------------------------------------------------------------


class TestFitHoliday:
    def test_records_columns_and_holidays(self) -> None:
        frame = _df({"d": _dates()})
        result = fit_holiday(frame, ("Christmas Day",), "US")
        assert result == HolidayFit(
            columns=["d"],
            holidays=("Christmas Day",),
            country="US",
            subdiv=None,
            rules={},
            dtypes={"d": nw.Date()},
        )

    def test_unknown_holiday_name_raises(self, snapshot: SnapshotAssertion) -> None:
        frame = _df({"d": _dates()})
        with pytest.raises(ValueError) as excinfo:
            fit_holiday(frame, ("Not A Real Holiday",), "US")
        assert str(excinfo.value) == snapshot

    def test_all_null_column_skips_validation(self) -> None:
        frame = _df({"d": pl.Series("d", [None, None], dtype=pl.Date)})
        result = fit_holiday(frame, ("Not A Real Holiday",), "US")
        assert result.holidays == ("Not A Real Holiday",)

    def test_rules_default_to_empty_dict(self) -> None:
        frame = _df({"d": _dates()})
        result = fit_holiday(frame, (), "US")
        assert result.rules == {}

    def test_records_rules(self) -> None:
        frame = _df({"d": _dates()})
        rule = _weekend_rule()
        result = fit_holiday(frame, (), "US", rules={"weekend": rule})
        assert result.rules == {"weekend": rule}

    def test_non_rule_value_raises(self, snapshot: SnapshotAssertion) -> None:
        frame = _df({"d": _dates()})
        with pytest.raises(TypeError) as excinfo:
            fit_holiday(frame, (), "US", rules={"weekend": "not a rule"})  # type: ignore[dict-item]
        assert str(excinfo.value) == snapshot

    def test_duplicate_slug_between_holiday_and_rule_raises(
        self, snapshot: SnapshotAssertion
    ) -> None:
        frame = _df({"d": _dates()})
        with pytest.raises(ValueError) as excinfo:
            fit_holiday(
                frame,
                ("Christmas Day",),
                "US",
                rules={"Christmas Day": _weekend_rule()},
            )
        assert str(excinfo.value) == snapshot

    def test_duplicate_slug_between_two_rules_raises(
        self, snapshot: SnapshotAssertion
    ) -> None:
        frame = _df({"d": _dates()})
        with pytest.raises(ValueError) as excinfo:
            fit_holiday(
                frame,
                (),
                "US",
                rules={"Pay Day": _weekend_rule(), "pay day": _weekend_rule()},
            )
        assert str(excinfo.value) == snapshot

    def test_rules_only_skips_holidays_package_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_import(module_name: str, *, purpose: str | None = None) -> Any:
            raise AssertionError("holidays package should not be imported")

        monkeypatch.setattr("shoez.step_holiday.import_optional", fail_import)
        frame = _df({"d": _dates()})
        result = fit_holiday(frame, (), "US", rules={"weekend": _weekend_rule()})
        assert result.holidays == ()


class TestApplyHoliday:
    def test_flags_the_requested_holiday(self) -> None:
        frame = _df({"d": _dates()})
        fit_result = fit_holiday(frame, ("Christmas Day",), "US")
        out = apply_holiday(frame, fit_result).to_native()
        assert out["d_christmas_day"].to_list() == [0, 1, 0, 0]

    def test_multiple_holidays_produce_multiple_columns(self) -> None:
        frame = _df({"d": _dates()})
        fit_result = fit_holiday(frame, ("Christmas Day", "New Year's Day"), "US")
        out = apply_holiday(frame, fit_result).to_native()
        assert out["d_christmas_day"].to_list() == [0, 1, 0, 0]
        assert out["d_new_years_day"].to_list() == [0, 0, 1, 0]

    def test_drops_source_column_by_default(self) -> None:
        frame = _df(_data())
        fit_result = fit_holiday(_df({"d": _dates()}), ("Christmas Day",), "US")
        out = apply_holiday(frame, fit_result).to_native()
        assert "d" not in out.columns
        assert list(out.columns) == ["d_christmas_day", "n"]

    def test_keep_original_cols_retains_source(self) -> None:
        frame = _df(_data())
        fit_result = fit_holiday(_df({"d": _dates()}), ("Christmas Day",), "US")
        out = apply_holiday(frame, fit_result, keep_original_cols=True).to_native()
        assert list(out.columns) == ["d", "d_christmas_day", "n"]

    def test_output_dtype_is_int32(self) -> None:
        frame = _df({"d": _dates()})
        fit_result = fit_holiday(frame, ("Christmas Day",), "US")
        out = apply_holiday(frame, fit_result)
        assert str(out.schema["d_christmas_day"]) == "Int32"

    def test_datetime_column_matched_by_date_part(self) -> None:
        frame = _df({"d": [datetime(2023, 12, 25, 13, 30), datetime(2023, 3, 1, 0, 0)]})
        fit_result = fit_holiday(frame, ("Christmas Day",), "US")
        out = apply_holiday(frame, fit_result).to_native()
        assert out["d_christmas_day"].to_list() == [1, 0]

    def test_country_changes_which_dates_match(self) -> None:
        # Canada Day (July 1) is not a US holiday.
        frame = _df({"d": [date(2023, 7, 1), date(2023, 7, 2)]})
        fit_result = fit_holiday(frame, ("Canada Day",), "CA")
        out = apply_holiday(frame, fit_result).to_native()
        assert out["d_canada_day"].to_list() == [1, 0]

    def test_new_data_outside_fit_year_range_is_still_classified(self) -> None:
        fit_result = fit_holiday(
            _df({"d": [date(2020, 1, 1)]}), ("Christmas Day",), "US"
        )
        new_frame = _df({"d": [date(2030, 12, 25), date(2030, 12, 26)]})
        out = apply_holiday(new_frame, fit_result).to_native()
        assert out["d_christmas_day"].to_list() == [1, 0]

    def test_rule_only_produces_indicator_column(self) -> None:
        frame = _df({"d": _dates()})
        fit_result = fit_holiday(frame, (), "US", rules={"weekend": _weekend_rule()})
        out = apply_holiday(frame, fit_result).to_native()
        assert out["d_weekend"].to_list() == [1, 0, 1, 0]

    def test_combined_holidays_and_rules_columns_in_order(self) -> None:
        frame = _df({"d": _dates()})
        fit_result = fit_holiday(
            frame, ("Christmas Day",), "US", rules={"weekend": _weekend_rule()}
        )
        out = apply_holiday(frame, fit_result).to_native()
        assert list(out.columns) == ["d_christmas_day", "d_weekend"]

    def test_timezone_aware_rule_does_not_crash(self) -> None:
        rule = rrule(
            WEEKLY,
            byweekday=(SA, SU),
            dtstart=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        frame = _df({"d": _dates()})
        fit_result = fit_holiday(frame, (), "US", rules={"weekend": rule})
        out = apply_holiday(frame, fit_result).to_native()
        assert out["d_weekend"].to_list() == [1, 0, 1, 0]

    def test_new_data_outside_fit_year_range_is_still_classified_by_rule(self) -> None:
        fit_result = fit_holiday(
            _df({"d": [date(2020, 1, 1)]}),
            (),
            "US",
            rules={"weekend": _weekend_rule(date(2020, 1, 1))},
        )
        new_frame = _df({"d": [date(2030, 1, 5), date(2030, 1, 7)]})
        out = apply_holiday(new_frame, fit_result).to_native()
        assert out["d_weekend"].to_list() == [1, 0]


# ---------------------------------------------------------------------------
# StepHoliday — unit-level invariants
# ---------------------------------------------------------------------------


class TestStepHolidayUnit:
    def test_mode_is_one_to_many(self) -> None:
        assert StepHoliday(["d"]).mode is Mode.ONE_TO_MANY

    def test_default_id(self) -> None:
        assert StepHoliday(["d"]).id == "holiday"

    def test_explicit_id(self) -> None:
        assert StepHoliday(["d"], id="my_holiday").id == "my_holiday"

    def test_default_holidays(self) -> None:
        assert StepHoliday(["d"]).holidays == (
            "New Year's Day",
            "Labor Day",
            "Christmas Day",
        )

    def test_default_country_is_us(self) -> None:
        assert StepHoliday(["d"]).country == "US"

    def test_default_subdiv_is_none(self) -> None:
        assert StepHoliday(["d"]).subdiv is None

    def test_default_rules_is_empty_dict(self) -> None:
        assert StepHoliday(["d"]).rules == {}

    def test_default_rules_not_shared_between_instances(self) -> None:
        first = StepHoliday(["d"])
        first.rules["weekend"] = _weekend_rule()
        assert StepHoliday(["d"]).rules == {}

    def test_default_role_is_feature(self) -> None:
        assert StepHoliday(["d"]).output_role == "feature"

    def test_keep_original_cols_default_false(self) -> None:
        assert StepHoliday(["d"]).keep_original_cols is False

    def test_fit_returns_self(self) -> None:
        step = StepHoliday(["d"], holidays=["Christmas Day"])
        fitted = step.fit(_df(_data()), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepHoliday(["d"])
        with pytest.raises(RuntimeError):
            step.transform(_df(_data()))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepHoliday(["d"]).schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepHoliday(["d"]).schema_out()

    def test_output_parents_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepHoliday(["d"]).output_parents()

    def test_accepts_date_and_datetime_dtypes_only(self) -> None:
        accepted = StepHoliday(["d"]).accepts_dtypes()
        assert nw.Date in accepted
        assert nw.Datetime in accepted
        assert nw.Float64 not in accepted

    def test_schema_in_lists_selected_columns(self) -> None:
        step = StepHoliday(["d"], holidays=["Christmas Day"])
        step.fit(_df(_data()), SchemaInfo())
        assert step.schema_in().columns == ["d"]

    def test_schema_out_lists_holiday_indicator_columns(self) -> None:
        step = StepHoliday(["d"], holidays=["Christmas Day", "New Year's Day"])
        step.fit(_df(_data()), SchemaInfo())
        assert step.schema_out().columns == ["d_christmas_day", "d_new_years_day"]
        assert step.schema_out().dtypes == {
            "d_christmas_day": "Int32",
            "d_new_years_day": "Int32",
        }

    def test_update_roles_with_no_output_role(self) -> None:
        from shoez.types import Roles

        step = StepHoliday(["d"], holidays=["Christmas Day"], role=None)
        step.fit(_df(_data()), SchemaInfo())
        roles = Roles()
        roles.add("d", "feature")
        step.update_roles(roles)
        assert roles.of("d_christmas_day") == set()
        assert roles.of("d") == set()

    def test_output_parents_maps_each_column_to_its_source(self) -> None:
        step = StepHoliday(["d"], holidays=["Christmas Day", "New Year's Day"])
        step.fit(_df(_data()), SchemaInfo())
        assert step.output_parents() == {
            "d_christmas_day": ["d"],
            "d_new_years_day": ["d"],
        }

    def test_schema_out_lists_rule_indicator_columns(self) -> None:
        step = StepHoliday(
            ["d"], holidays=["Christmas Day"], rules={"weekend": _weekend_rule()}
        )
        step.fit(_df(_data()), SchemaInfo())
        assert step.schema_out().columns == ["d_christmas_day", "d_weekend"]
        assert step.schema_out().dtypes == {
            "d_christmas_day": "Int32",
            "d_weekend": "Int32",
        }

    def test_produced_columns_matches_schema_out(self) -> None:
        step = StepHoliday(
            ["d"], holidays=["Christmas Day"], rules={"weekend": _weekend_rule()}
        )
        step.fit(_df(_data()), SchemaInfo())
        assert step.produced_columns() == ["d_christmas_day", "d_weekend"]

    def test_output_parents_includes_rule_columns(self) -> None:
        step = StepHoliday(["d"], holidays=[], rules={"weekend": _weekend_rule()})
        step.fit(_df(_data()), SchemaInfo())
        assert step.output_parents() == {"d_weekend": ["d"]}

    def test_not_sparse(self) -> None:
        assert StepHoliday(["d"]).produces_sparse is False

    def test_required_pkgs_names_holidays(self) -> None:
        assert StepHoliday(["d"]).required_pkgs() == ["holidays"]

    def test_required_pkgs_empty_when_no_holidays(self) -> None:
        step = StepHoliday(["d"], holidays=[], rules={"weekend": _weekend_rule()})
        assert step.required_pkgs() == []

    def test_missing_holidays_package_raises_friendly_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_import_optional(
            module_name: str, *, purpose: str | None = None
        ) -> Any:
            raise MissingDependencyError(
                f"`{module_name}` is required but not installed"
            )

        monkeypatch.setattr("shoez.step_holiday.import_optional", fake_import_optional)
        step = StepHoliday(["d"], holidays=["Christmas Day"])
        with pytest.raises(MissingDependencyError):
            step.fit(_df(_data()), SchemaInfo())
