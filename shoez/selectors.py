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
        dtype: Literal["numeric", "categorical"] | None,
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
    def all_predictors(cls) -> ShoezSelector:
        """All predictor columns regardless of type."""
        return _RoleTypeSelector(role="predictor", dtype=None)

    @classmethod
    def all_numeric(cls) -> ShoezSelector:
        """All numeric columns regardless of role."""
        return _RoleTypeSelector(role=None, dtype="numeric")
