"""Shared level-collision warning for `StepUnknown` and `StepOther`.

Both steps recode categorical values to a caller-chosen sentinel level and
warn if that sentinel already exists among a column's real values, since
recoding would otherwise merge genuine samples into the sentinel bucket. The
warning differs only in the noun describing what gets merged in (recoded
nulls vs. pooled levels); this module holds the shared message.

Not part of the public API: only `shoez.step_unknown` and `shoez.step_other`
import from here.
"""

from __future__ import annotations

import warnings


def warn_level_collision(level: str, collisions: list[str], *, merges: str) -> None:
    """Warn that ``level`` already exists as a value in ``collisions``' columns.

    A no-op when ``collisions`` is empty. ``merges`` names what will be
    merged into the existing level (e.g. ``"recoded nulls"``,
    ``"pooled levels"``).
    """
    if not collisions:
        return
    warnings.warn(
        f"the level {level!r} already exists in {len(collisions)} column(s); "
        f"{merges} will be merged with it: {collisions!r}",
        stacklevel=3,
    )
