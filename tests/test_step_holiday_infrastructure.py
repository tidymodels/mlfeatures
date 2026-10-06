"""Conformance tests for StepHoliday.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepHoliday belongs in ``test_step_holiday.py`` and ``test_step_holiday_e2e.py``.

See ``test_step_date_features_infrastructure.py`` for why the Date column is
cast via ``dtype_casts`` (pandas cannot represent narwhals' ``Date`` dtype
from a plain list of ``datetime.date`` objects without pyarrow).
"""

from __future__ import annotations

from datetime import date

import narwhals.stable.v2 as nw

from mlfeatures import StepHoliday
from mlfeatures.testing import StepConformanceTest


class TestHolidayConformance(StepConformanceTest):
    training_data = {
        "d": [date(2023, 12, 24), date(2023, 12, 25), date(2023, 1, 1)],
        "n": [1.0, 2.0, 3.0],
    }
    dtype_casts = {"d": nw.Date}

    def make_step(self) -> StepHoliday:
        return StepHoliday(["d"], holidays=["Christmas Day"])
