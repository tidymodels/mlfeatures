"""Role- and type-aware column selectors for use in step declarations.

Import as ``from shoez import Selectors as cs`` and pass the result of a
factory call directly to any step::

    StepDropCorr(cs.predictors_numeric())

Selectors are resolved at fit time, when both the frame schema (dtypes) and
the recipe roles are available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from shoez.types import Frame, Roles


class ShoezSelector:
    """Base class for all shoez-native column selectors."""

    def resolve(self, frame: Frame, roles: Roles) -> list[str]:
        raise NotImplementedError


class _RoleTypeSelector(ShoezSelector):
    def __init__(
        self,
        role: str | None,
        dtype: Literal[
            "numeric",
            "categorical",
            "nominal",
            "string",
            "integer",
            "double",
            "date",
            "datetime",
            "boolean",
        ]
        | None,
    ) -> None:
        self._role = role
        self._dtype = dtype

    def resolve(self, frame: Frame, roles: Roles) -> list[str]:
        from shoez import _frame

        if self._role is not None:
            candidates = [
                c for c in roles.columns_with_role(self._role) if c in frame.columns
            ]
        else:
            candidates = _frame.columns(frame)

        if self._dtype == "numeric":
            return [c for c in candidates if _frame.is_numeric(frame, c)]
        if self._dtype == "categorical":
            return [c for c in candidates if _frame.is_categorical(frame, c)]
        if self._dtype == "nominal":
            return [c for c in candidates if _frame.is_nominal(frame, c)]
        if self._dtype == "string":
            return [c for c in candidates if _frame.is_string(frame, c)]
        if self._dtype == "integer":
            return [c for c in candidates if _frame.is_integer(frame, c)]
        if self._dtype == "double":
            return [c for c in candidates if _frame.is_double(frame, c)]
        if self._dtype == "date":
            return [c for c in candidates if _frame.is_date(frame, c)]
        if self._dtype == "datetime":
            return [c for c in candidates if _frame.is_datetime(frame, c)]
        if self._dtype == "boolean":
            return [c for c in candidates if _frame.is_boolean(frame, c)]
        return candidates


class Selectors:
    """Namespace of factory methods for built-in shoez column selectors."""

    @classmethod
    def predictors_numeric(cls) -> ShoezSelector:
        """All predictor columns that are numeric."""
        return _RoleTypeSelector(role="predictor", dtype="numeric")

    @classmethod
    def predictors_categorical(cls) -> ShoezSelector:
        """All predictor columns that are categorical."""
        return _RoleTypeSelector(role="predictor", dtype="categorical")

    @classmethod
    def predictors(cls) -> ShoezSelector:
        """All predictor columns regardless of type."""
        return _RoleTypeSelector(role="predictor", dtype=None)

    @classmethod
    def predictors_nominal(cls) -> ShoezSelector:
        """All predictor columns that are nominal (categorical, enum, or string)."""
        return _RoleTypeSelector(role="predictor", dtype="nominal")

    @classmethod
    def predictors_string(cls) -> ShoezSelector:
        """All predictor columns that are string-typed."""
        return _RoleTypeSelector(role="predictor", dtype="string")

    @classmethod
    def predictors_integer(cls) -> ShoezSelector:
        """All predictor columns that are integer-typed."""
        return _RoleTypeSelector(role="predictor", dtype="integer")

    @classmethod
    def predictors_double(cls) -> ShoezSelector:
        """All predictor columns that are floating-point typed."""
        return _RoleTypeSelector(role="predictor", dtype="double")

    @classmethod
    def predictors_date(cls) -> ShoezSelector:
        """All predictor columns that are Date-typed."""
        return _RoleTypeSelector(role="predictor", dtype="date")

    @classmethod
    def predictors_datetime(cls) -> ShoezSelector:
        """All predictor columns that are Datetime-typed."""
        return _RoleTypeSelector(role="predictor", dtype="datetime")

    @classmethod
    def predictors_boolean(cls) -> ShoezSelector:
        """All predictor columns that are boolean-typed."""
        return _RoleTypeSelector(role="predictor", dtype="boolean")

    @classmethod
    def outcomes(cls) -> ShoezSelector:
        """All outcome columns regardless of type."""
        return _RoleTypeSelector(role="outcome", dtype=None)
