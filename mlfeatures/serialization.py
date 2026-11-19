"""Diffable JSON/YAML serialization for fitted recipes.

A `FittedRecipe` is serialized to a plain, human-readable ``dict`` (and from
there to JSON or YAML) that captures every step's **hyperparameters and learned
statistics** plus the recipe-level role, schema, and lineage state. The format is
stable and diffable so a trained recipe can be reviewed in a pull request, saved
alongside a model, and reloaded in another process for inference.

Nothing in the payload is a closure, environment, or pickled object: step state
is plain fields, selectors are serializable expression objects, and the step
class is looked up by qualified name at load time. That property is what the
no-closure conformance test relies on -- a closure cannot be JSON-encoded, so a
passing round-trip *is* the guarantee.

Loading is restricted to what the payload may *name*. While `from_dict` (and so
`from_json` and `from_yaml`) decodes a payload, the class names in it (a step's
``type``, a fit dataclass, a selector's ``step_type``) are resolved only against
``mlfeatures``, the packages that register a step under the ``mlfeatures.steps``
entry-point group, and the packages of any classes the caller passes as
``trusted``. They must resolve to a `Step` subclass or a dataclass respectively.
A payload naming anything else (``builtins.print``, ``subprocess.call``) raises
``ValueError`` before that module is imported, so loading a payload never runs
code the payload picks. The check also looks at where the resolved class is
defined, so a trusted module that re-exports a third-party class does not make
it loadable. The trust unit is a top-level package. It does not validate the
values inside a payload, and a step from a registered plugin runs its own
author's code when the recipe transforms. Pickle, by contrast, executes
arbitrary code by design and should only be used on files you trust.

The allow-list applies to payloads, not to state produced in this process:
`Step.from_state` on a step's own `Step.to_state` (the ``minimize`` idiom) is
unrestricted, because nothing outside the process chose the names. Saving warns
when a recipe names a class that `from_dict` would refuse by default, so the
mismatch shows up when the file is written, not when it is first loaded.

The public surface is a small module API -- `to_dict`, `from_dict`, `to_json`,
`from_json`, `to_yaml`, `from_yaml`, and `minimize` -- mirrored by thin
delegating methods on `FittedRecipe` (``fitted.to_json()``,
``FittedRecipe.from_json(text)``, ...).

JSON and YAML are two thin encodings over the same JSON-safe ``dict`` format, so
both round-trip identically. YAML support needs the optional ``pyyaml`` package;
`to_yaml` / `from_yaml` raise a clear ``ImportError`` when it is missing.

Known limitations
-----------------
- ``Enum`` categories do not round-trip: an ``Enum`` column's schema/lineage
  dtype comes back as an empty-category ``Enum`` after a JSON round-trip. This
  is enough for the lineage/schema checks, which only compare dtype *identity*
  by base type, but it does mean ``ColSchema``, which already stores dtypes as
  ``str(dt)``, is a more faithful record of an ``Enum``'s categories than a
  deserialized ``Schema``/``Lineage`` is.
- ``Datetime`` (time-unit, time zone) and ``Duration`` (time-unit) *do*
  round-trip their parameters (see `_encode_dtype_instance` /
  `_decode_dtype_instance`) -- `StepDatetimeFeatures`'s conformance suite
  depends on it: a pandas data frame's nanosecond-precision ``Datetime`` must still
  match the schema snapshot after a JSON round-trip.
- Raw polars / narwhals selectors are not serializable. Only mlfeatures `Selectors`
  and plain name lists are; a native selector raises a clear ``TypeError``.
- ``dateutil.rrule.rrule`` / ``rruleset`` instances are not serializable
  (`StepHoliday`'s ``rules``); a step storing one raises a clear ``TypeError``
  from `to_json` / `to_yaml` rather than silently dropping the rule.
"""

# This module is the one place that deliberately reaches into the internal
# representation of steps, selectors, and the lineage and juggles the
# resulting plain runtime data (``Any`` scalars, lists, dicts, ndarrays) -- that
# is what a serializer does. Pyright strict flags every private-field access
# (reportPrivateUsage) and every ``Any``-typed value the codec touches
# (reportUnknown*); those add no safety a serializer can act on, so they are
# disabled here (and only here). mypy strict still fully type-checks this file,
# so the relaxation is scoped and covered.
# pyright: reportPrivateUsage=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import importlib
import json
import math
import os
import sys
import warnings
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw
import numpy as np
from dateutil.rrule import rrule, rruleset

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator

    from mlfeatures.recipe import FittedRecipe
    from mlfeatures.step import Step

__all__ = [
    "to_dict",
    "from_dict",
    "to_json",
    "from_json",
    "to_yaml",
    "from_yaml",
    "minimize",
]

#: Bumped when the on-disk layout changes incompatibly within a major version.
FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# narwhals dtype codec
# --------------------------------------------------------------------------- #

# The narwhals dtype classes mlfeatures supports. Encoding uses the class name (the
# same base representation ``ColSchema`` stores via ``str(dt)``); decoding maps
# the name back to the class. Parametrized dtypes round-trip by base name only.
_DTYPE_CLASSES: tuple[type[nw.dtypes.DType], ...] = (
    nw.Int8,
    nw.Int16,
    nw.Int32,
    nw.Int64,
    nw.Int128,
    nw.UInt8,
    nw.UInt16,
    nw.UInt32,
    nw.UInt64,
    nw.UInt128,
    nw.Float32,
    nw.Float64,
    nw.Decimal,
    nw.String,
    nw.Boolean,
    nw.Categorical,
    nw.Enum,
    nw.Date,
    nw.Datetime,
    nw.Duration,
    nw.Binary,
    nw.Object,
)

# The abstract dtype *categories* narwhals groups concrete dtypes under. Steps
# never store these, but ``Selectors.numeric()`` and friends hold one as their
# ``dtype``, so decoding a selector has to resolve them back to a class. They are
# never instantiated, which is why ``_dtype_instance`` only handles the concrete
# classes above.
_DTYPE_CATEGORY_CLASSES: tuple[type[nw.dtypes.DType], ...] = (
    nw.dtypes.NumericType,
    nw.dtypes.IntegerType,
    nw.dtypes.SignedIntegerType,
    nw.dtypes.UnsignedIntegerType,
    nw.dtypes.FloatType,
    nw.dtypes.TemporalType,
)
_NAME_TO_DTYPE: dict[str, type[nw.dtypes.DType]] = {
    cls.__name__: cls for cls in (*_DTYPE_CLASSES, *_DTYPE_CATEGORY_CLASSES)
}


def _dtype_name(dtype: type[nw.dtypes.DType] | nw.dtypes.DType) -> str:
    """The base class name of a narwhals dtype class or instance."""
    cls = dtype if isinstance(dtype, type) else type(dtype)
    return cls.__name__


def _dtype_class(name: str) -> type[nw.dtypes.DType]:
    """Resolve a base dtype name back to its narwhals class."""
    try:
        return _NAME_TO_DTYPE[name]
    except KeyError:
        raise ValueError(f"unknown narwhals dtype {name!r}") from None


def _dtype_instance(name: str) -> nw.dtypes.DType:
    """Instantiate a dtype from its base name, using default parameters.

    ``Enum`` needs a category list it no longer has after a by-name round-trip;
    it falls back to an empty-category enum, which is enough for the lineage/schema
    checks that only compare dtype *identity* by base type.
    """
    cls = _dtype_class(name)
    if cls is nw.Enum:
        return nw.Enum([])
    return cls()


def _encode_dtype_instance(dtype: nw.dtypes.DType) -> str | dict[str, Any]:
    """Encode a concrete dtype instance for the schema/lineage-node codecs.

    Unlike `_dtype_name` (used where only the base type matters, e.g. a
    selector's stored dtype), this preserves ``Datetime``'s time-unit and time
    zone, and ``Duration``'s time-unit, since two ``Datetime`` columns of
    different precision are not the same schema -- `Schema.check` would
    otherwise accept a nanosecond-precision new-data column against a
    microsecond-precision training snapshot that a round-trip silently
    downgraded to. Every other dtype still round-trips by base name only (see
    the module docstring's ``Enum`` caveat).
    """
    name = _dtype_name(dtype)
    if isinstance(dtype, nw.Datetime):
        return {
            "name": name,
            "time_unit": dtype.time_unit,
            "time_zone": dtype.time_zone,
        }
    if isinstance(dtype, nw.Duration):
        return {"name": name, "time_unit": dtype.time_unit}
    return name


def _decode_dtype_instance(payload: str | dict[str, Any]) -> nw.dtypes.DType:
    """Inverse of `_encode_dtype_instance`."""
    if isinstance(payload, dict):
        cls = _dtype_class(payload["name"])
        if cls is nw.Datetime:
            return nw.Datetime(
                time_unit=payload["time_unit"], time_zone=payload["time_zone"]
            )
        if cls is nw.Duration:
            return nw.Duration(time_unit=payload["time_unit"])
        raise ValueError(f"unexpected parametrized dtype payload: {payload!r}")
    return _dtype_instance(payload)


# --------------------------------------------------------------------------- #
# Recursive value codec
# --------------------------------------------------------------------------- #
#
# Anything JSON cannot represent natively is wrapped in a single-key tagged
# ``dict`` (``{"__tuple__": ...}``). ``_decode`` dispatches on that tag. Plain
# ``dict`` values with string keys are passed through recursively, so a payload
# stays as readable as the data allows.


def _encode(value: Any) -> Any:
    """Encode an arbitrary step/recipe field into a JSON-safe structure."""
    # bool must precede int (bool is an int subclass) and is JSON-native anyway.
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        # NaN / +-inf are not valid JSON tokens; tag them explicitly.
        return {"__float__": repr(value)}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return {"__set__": [_encode(v) for v in _sorted_safe(value)]}
    if isinstance(value, np.ndarray):
        data = value.tolist()
        # A float array can hold NaN / +-inf, which ``tolist`` surfaces as raw
        # Python floats. Those are not valid JSON tokens, so route the data back
        # through ``_encode`` (which tags them) whenever the array is not all
        # finite -- mirroring the scalar-float path above. The common finite
        # case keeps the flat ``tolist`` payload untouched.
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            data = _encode(data)
        return {
            "__ndarray__": {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": data,
            }
        }
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _encode(value.item())
    if isinstance(value, nw.dtypes.DType) or (
        isinstance(value, type) and issubclass(value, nw.dtypes.DType)
    ):
        return {"__dtype__": _dtype_name(value)}
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: _encode(v) for k, v in value.items()}
        # Non-string keys (rare) round-trip as an ordered list of pairs.
        return {"__dict__": [[_encode(k), _encode(v)] for k, v in value.items()]}
    # ColSchema, FeaturesSelector, and frozen fit dataclasses.
    from mlfeatures.selectors import FeaturesSelector
    from mlfeatures.types import ColSchema

    if isinstance(value, ColSchema):
        return {
            "__colschema__": {
                "columns": list(value.columns),
                "dtypes": dict(value.dtypes),
            }
        }
    if isinstance(value, FeaturesSelector):
        return {"__selector__": _encode_selector(value)}
    if isinstance(value, (rrule, rruleset)):
        raise TypeError(
            "cannot serialize a 'dateutil.rrule.rrule'/'rruleset' instance; a "
            "step storing one (e.g. StepHoliday's 'rules') does not support "
            "JSON serialization"
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        return {
            "__dataclass__": _qualified_name(cls),
            "fields": {
                f.name: _encode(getattr(value, f.name))
                for f in dataclasses.fields(value)
            },
        }
    raise TypeError(
        f"cannot serialize value of type {type(value).__name__!r}; "
        "step state must be plain data (no closures, environments, or native "
        "polars/narwhals selectors)"
    )


def _decode(obj: Any) -> Any:
    """Inverse of `_encode`."""
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    if isinstance(obj, dict):
        if "__dataclass__" in obj:
            return _decode_dataclass(obj)
        if len(obj) == 1:
            (tag,) = obj
            payload = obj[tag]
            if tag == "__float__":
                return float(payload)
            if tag == "__tuple__":
                return tuple(_decode(v) for v in payload)
            if tag == "__set__":
                return {_decode(v) for v in payload}
            if tag == "__ndarray__":
                # ``data`` is a plain nested list in the finite case and a codec-
                # tagged structure when the array held NaN / +-inf; ``_decode``
                # is the identity on the former and untags the latter.
                arr = np.array(_decode(payload["data"]), dtype=payload["dtype"])
                return arr.reshape(payload["shape"])
            if tag == "__dtype__":
                return _dtype_instance(payload)
            if tag == "__dict__":
                return {_decode(k): _decode(v) for k, v in payload}
            if tag == "__colschema__":
                from mlfeatures.types import ColSchema

                return ColSchema(
                    columns=list(payload["columns"]), dtypes=dict(payload["dtypes"])
                )
            if tag == "__selector__":
                return _decode_selector(payload)
        return {k: _decode(v) for k, v in obj.items()}
    return obj


def _decode_dataclass(obj: dict[str, Any]) -> Any:
    qualname = obj["__dataclass__"]
    cls = _load_qualname(qualname)
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise ValueError(f"{qualname!r} is not a dataclass; cannot deserialize it")
    fields = obj.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(
            f"expected the 'fields' of {qualname!r} to be a mapping; "
            f"got {type(fields).__name__!r}"
        )
    try:
        return cls(**{name: _decode(v) for name, v in fields.items()})
    except TypeError as exc:
        raise ValueError(
            f"cannot construct {qualname!r} from the payload's fields; {exc}"
        ) from exc


def _sorted_safe(values: Any) -> list[Any]:
    """Sort a set for a deterministic (diffable) payload, tolerating mixed types."""
    items = list(values)
    try:
        return sorted(items)
    except TypeError:
        return sorted(items, key=repr)


# --------------------------------------------------------------------------- #
# Selector codec
# --------------------------------------------------------------------------- #


def _encode_selector(sel: Any) -> dict[str, Any]:
    from mlfeatures.selectors import (
        _Complement,
        _CompositeSelector,
        _ForeignSelector,
        _NameSelector,
        _RoleTypeSelector,
        _SourceSelector,
    )

    if isinstance(sel, _CompositeSelector):
        return {
            "cls": "composite",
            "op": sel._op,
            "children": [_encode(c) for c in sel._children],
        }
    if isinstance(sel, _Complement):
        return {"cls": "complement", "inner": _encode(sel._inner)}
    if isinstance(sel, _NameSelector):
        return {
            "cls": "name",
            "kind": sel._kind,
            "params": list(sel._params),
            "label": sel._label,
        }
    if isinstance(sel, _RoleTypeSelector):
        return {
            "cls": "roletype",
            "role": sel._role,
            "dtype": _encode_dtype_spec(sel._dtype),
        }
    if isinstance(sel, _SourceSelector):
        step_type = None
        if sel._step_type is not None:
            step_type = _qualified_name(sel._step_type)
        return {
            "cls": "source",
            "step": sel._step,
            "step_type": step_type,
            "ancestral": sel._ancestral,
        }
    if isinstance(sel, _ForeignSelector):
        value = sel._value
        if isinstance(value, str) or (
            isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)
        ):
            return {"cls": "foreign", "value": _encode(value)}
        raise TypeError(
            "cannot serialize a native polars/narwhals selector; use mlfeatures "
            "'Selectors' (or a plain list of column names) instead"
        )
    raise TypeError(f"cannot serialize selector of type {type(sel).__name__!r}")


def _decode_selector(state: dict[str, Any]) -> Any:
    from mlfeatures.selectors import (
        _Complement,
        _CompositeSelector,
        _ForeignSelector,
        _NameSelector,
        _RoleTypeSelector,
        _SourceSelector,
    )

    kind = state["cls"]
    if kind == "composite":
        return _CompositeSelector(state["op"], [_decode(c) for c in state["children"]])
    if kind == "complement":
        return _Complement(_decode(state["inner"]))
    if kind == "name":
        return _NameSelector(state["kind"], list(state["params"]), state["label"])
    if kind == "roletype":
        return _RoleTypeSelector(state["role"], _decode_dtype_spec(state["dtype"]))
    if kind == "source":
        step_type = (
            _load_step_class(state["step_type"])
            if state["step_type"] is not None
            else None
        )
        return _SourceSelector(
            step=state["step"], step_type=step_type, ancestral=state["ancestral"]
        )
    if kind == "foreign":
        return _ForeignSelector(_decode(state["value"]))
    raise ValueError(f"unknown selector kind {kind!r}")


def _encode_dtype_spec(
    dtype: type | tuple[type, ...] | None,
) -> None | str | list[str]:
    """Encode a `_RoleTypeSelector` dtype (class, tuple of classes, or None)."""
    if dtype is None:
        return None
    if isinstance(dtype, tuple):
        return [_dtype_name(d) for d in dtype]
    return _dtype_name(dtype)


def _decode_dtype_spec(spec: None | str | list[str]) -> type | tuple[type, ...] | None:
    if spec is None:
        return None
    if isinstance(spec, list):
        return tuple(_dtype_class(name) for name in spec)
    return _dtype_class(spec)


# --------------------------------------------------------------------------- #
# Qualified-name step/class lookup
# --------------------------------------------------------------------------- #


def _root(module_name: str) -> str:
    return module_name.partition(".")[0]


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _entry_point_parts(ep: Any) -> tuple[str, str] | None:
    """The ``(module, attr)`` an entry point names, or ``None`` if it is malformed."""
    try:
        return ep.module, ep.attr or ""
    except AttributeError:
        return None


_UNTRUSTED = (
    "only classes from 'mlfeatures', from a package that registers a step under "
    "the 'mlfeatures.steps' entry-point group, or passed as 'trusted' are loaded"
)


class _Trust:
    """Which classes a payload may name, resolved lazily and at most once per load.

    A payload is data from outside the process, so the class names in it are a
    request, not an instruction: importing and calling whatever it names would let
    a crafted file run ``subprocess.call``. Only ``mlfeatures`` itself, the
    packages that register a step under the ``mlfeatures.steps`` entry-point group
    (an install-time declaration, not payload data), and the packages of the
    classes passed as ``trusted`` are honored. The trust unit is a top-level
    package, and the check runs twice: on the name before anything is imported
    (importing runs code), and on the module that defines the resolved class, so
    a trusted module re-exporting a third-party class does not make it loadable.

    ``enforce=False`` trusts every package. That is the mode for state produced in
    this process (`Step.from_state` on a step's own `Step.to_state`), where nothing
    outside the process chose the names.
    """

    def __init__(self, trusted: Iterable[type] = (), *, enforce: bool = True) -> None:
        self._enforce = enforce
        self._extra_roots = frozenset(_root(cls.__module__) for cls in trusted)
        self._entry_points: list[tuple[Any, str, str]] | None = None
        self._registered_roots: frozenset[str] | None = None

    def _registered(self) -> list[tuple[Any, str, str]]:
        """The ``(entry point, module, attr)`` of each usable registered step.

        Read on first use rather than up front: a payload of built-in steps never
        needs it, and one malformed entry point in any installed package must not
        break loading them. Malformed entry points are skipped.
        """
        if self._entry_points is None:
            found: list[tuple[Any, str, str]] = []
            for ep in _step_entry_points():
                parts = _entry_point_parts(ep)
                if parts is not None:
                    found.append((ep, *parts))
            self._entry_points = found
        return self._entry_points

    def allows(self, module_name: str) -> bool:
        """Whether a class defined in ``module_name`` may be loaded."""
        root = _root(module_name)
        if not self._enforce or root == "mlfeatures" or root in self._extra_roots:
            return True
        if self._registered_roots is None:
            self._registered_roots = frozenset(
                _root(module) for _, module, _ in self._registered()
            )
        return root in self._registered_roots

    def load(self, qualname: object) -> Any:
        """Return the class ``module.Qualname``, or raise ``ValueError``.

        The module is imported only when its package is trusted; importing is
        itself code execution, so the check comes first. A step re-exported from
        a package other than the one that defines it is found through the
        ``mlfeatures.steps`` entry-point group, by the name it was serialized
        under.
        """
        if not isinstance(qualname, str):
            raise ValueError(
                f"expected a class name as a string; got {type(qualname).__name__!r}"
            )
        module_name, _, attr = qualname.rpartition(".")
        reason = _UNTRUSTED
        cause: Exception | None = None
        if module_name and self.allows(module_name):
            try:
                obj = getattr(importlib.import_module(module_name), attr)
            except (ImportError, AttributeError) as exc:
                reason, cause = str(exc), exc
            else:
                defined_in = getattr(obj, "__module__", None)
                if isinstance(defined_in, str) and self.allows(defined_in):
                    return obj
                reason = (
                    f"it resolves to a class defined in {defined_in!r}, "
                    "which is not a trusted package"
                )
        cls = self._registered_class(qualname, attr)
        if cls is not None:
            return cls
        raise ValueError(
            f"cannot locate {qualname!r} for deserialization; {reason}"
        ) from cause

    def _registered_class(self, qualname: str, attr: str) -> Any:
        """The registered step class serialized as ``qualname``, or ``None``.

        Entry points are matched on their attribute name before any is loaded, so
        a name no plugin registers imports nothing, and a plugin that fails to
        import cannot supply the class.
        """
        for ep, _, ep_attr in self._registered():
            if ep_attr.rpartition(".")[2] != attr:
                continue
            try:
                cls = ep.load()
            except Exception:
                continue
            if isinstance(cls, type) and _qualified_name(cls) == qualname:
                return cls
        return None

    def can_load(self, qualname: str) -> bool:
        try:
            self.load(qualname)
        except ValueError:
            return False
        return True


#: The `_Trust` for the payload being decoded; ``None`` outside `from_dict`, where
#: nothing outside the process chose the names (see `_Trust`).
_active_trust: contextvars.ContextVar[_Trust | None] = contextvars.ContextVar(
    "_active_trust", default=None
)


@contextlib.contextmanager
def _decoding(trusted: Iterable[type]) -> Generator[None]:
    """Restrict the class names a payload may use for the duration of a block."""
    token = _active_trust.set(_Trust(trusted))
    try:
        yield
    finally:
        _active_trust.reset(token)


def _load_qualname(qualname: object) -> Any:
    """`_Trust.load` under the trust of the payload being decoded."""
    trust = _active_trust.get()
    return (_Trust(enforce=False) if trust is None else trust).load(qualname)


def _is_loaded_by_default(  # pyright: ignore[reportUnusedFunction]
    cls: type,
) -> bool:
    """Whether `from_dict` loads ``cls`` without the caller listing it as trusted.

    Its only caller is ``mlfeatures.testing``, which pyright does not check, so it
    sees no use of this function in the package.
    """
    return _Trust().allows(cls.__module__)


def _load_step_class(qualname: object) -> type[Step]:
    """`_load_qualname`, requiring the result to be a `Step` subclass."""
    from mlfeatures.step import Step

    cls = _load_qualname(qualname)
    if not (isinstance(cls, type) and issubclass(cls, Step)):
        raise ValueError(f"{qualname!r} is not a step class; cannot deserialize it")
    return cls


def _step_entry_points() -> Any:
    from importlib.metadata import entry_points

    try:
        return entry_points(group="mlfeatures.steps")
    except TypeError:  # pragma: no cover - Python <3.10 signature
        return entry_points().get("mlfeatures.steps", [])


# --------------------------------------------------------------------------- #
# Recipe-level codecs (roles / schema / lineage)
# --------------------------------------------------------------------------- #


def _encode_roles(roles: Any) -> dict[str, list[str]]:
    return {col: sorted(roles.of(col)) for col in roles.all_columns()}


def _decode_roles(payload: dict[str, list[str]]) -> Any:
    from mlfeatures.types import Roles

    return Roles({col: set(names) for col, names in payload.items()})


def _encode_schema(schema: Any) -> dict[str, Any]:
    return {
        "dtypes": {
            name: _encode_dtype_instance(schema.dtypes[name])
            for name in schema.dtypes.names()
        },
        "roles": _encode_roles(schema.roles),
        "transform_required": dict(schema.transform_required),
    }


def _decode_schema(payload: dict[str, Any]) -> Any:
    from mlfeatures.schema import Schema

    dtypes = nw.Schema(
        {name: _decode_dtype_instance(dt) for name, dt in payload["dtypes"].items()}
    )
    return Schema(
        dtypes=dtypes,
        roles=_decode_roles(payload["roles"]),
        transform_required=dict(payload["transform_required"]),
    )


def _encode_node(node: Any) -> dict[str, Any]:
    return {
        "name": node.name,
        "dtype": _encode_dtype_instance(node.dtype),
        "roles": sorted(node.roles),
        "source": node.source,
        "parent_cols": list(node.parent_cols),
        "dtype_history": [_encode_dtype_instance(d) for d in node.dtype_history],
        "sparse": node.sparse,
        "removed_by": node.removed_by,
    }


def _decode_node(payload: dict[str, Any]) -> Any:
    from mlfeatures.lineage import ColumnNode

    return ColumnNode(
        name=payload["name"],
        dtype=_decode_dtype_instance(payload["dtype"]),
        roles=set(payload["roles"]),
        source=payload["source"],
        parent_cols=list(payload["parent_cols"]),
        dtype_history=[_decode_dtype_instance(d) for d in payload["dtype_history"]],
        sparse=payload["sparse"],
        removed_by=payload["removed_by"],
    )


def _encode_lineage(lineage: Any) -> dict[str, Any]:
    return {
        "live": [_encode_node(node) for node in lineage._live.values()],
        "history": [_encode_node(node) for node in lineage._history],
    }


def _decode_lineage(payload: dict[str, Any]) -> Any:
    from mlfeatures.lineage import Lineage

    lineage = Lineage()
    for node_payload in payload["live"]:
        node = _decode_node(node_payload)
        lineage._live[node.name] = node
    lineage._history = [_decode_node(n) for n in payload["history"]]
    return lineage


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _encode_step(step: Any) -> dict[str, Any]:
    return {
        "type": _qualified_name(type(step)),
        "state": step.to_state(),
    }


def _decode_step(payload: Any) -> Any:
    state = payload.get("state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(
            "expected each step payload to be a mapping with a 'type' and a "
            "'state' mapping"
        )
    cls = _load_step_class(payload.get("type"))
    return cls.from_state(state)


def _class_names(obj: Any) -> Iterator[str]:
    """The class names inside an encoded step: fit dataclasses and selector step types."""
    if isinstance(obj, list):
        for item in obj:
            yield from _class_names(item)
    elif isinstance(obj, dict):
        if isinstance(obj.get("__dataclass__"), str):
            yield obj["__dataclass__"]
        if obj.get("cls") == "source" and isinstance(obj.get("step_type"), str):
            yield obj["step_type"]
        for value in obj.values():
            yield from _class_names(value)


def _external_stacklevel() -> int:
    """The ``warnings.warn`` stack level of the first frame outside mlfeatures.

    The save functions reach the warning through several layers (the
    ``FittedRecipe`` method, ``to_json``, ``to_dict``), so a fixed level would
    point at library code for some of the entry points.
    """
    package = os.path.dirname(__file__) + os.sep
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and frame.f_code.co_filename.startswith(package):
        frame = frame.f_back
        level += 1
    return level


def _warn_if_unloadable(steps: list[dict[str, Any]]) -> None:
    """Warn when the encoded ``steps`` name classes `from_dict` refuses by default.

    Without this the file is written without complaint and only fails when it is
    first loaded, possibly in another process long after the recipe was fit.
    """
    trust = _Trust()
    names = {step["type"] for step in steps} | set(_class_names(steps))
    refused = sorted(name for name in names if not trust.can_load(name))
    if refused:
        warnings.warn(
            f"recipe names class(es) {refused!r} that 'from_dict' refuses by "
            "default; register a step from their package(s) under the "
            "'mlfeatures.steps' entry-point group, or pass a class from them as "
            "'trusted' when loading",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )


def to_dict(recipe: FittedRecipe) -> dict[str, Any]:
    """Serialize a `FittedRecipe` to a diffable, JSON-safe ``dict``.

    A fitted recipe has to outlive the process that fit it: it is saved next to
    a model, loaded elsewhere to transform new data, and reviewed in a pull
    request when it changes. So the ``dict`` captures what `transform()` needs
    (every step's hyperparameters and learned statistics, plus the roles,
    training schema, and lineage) and nothing that ties it to the training
    process: no training data, closures, or environments.

    The ``dict`` holds plain data only (strings, numbers, booleans, lists, and
    mappings), so it round-trips through `from_dict` and is stable enough to
    diff in code review. Unlike a pickle, it can be read and checked without
    running anything, and loading it only builds classes from trusted packages
    (see `from_dict`). Use `to_json` or `to_yaml` to write it to a file, and the
    ``dict`` itself when embedding the recipe in a larger payload. It has these
    keys, in this order:

    - ``mlfeatures_version``: the version of mlfeatures that wrote it.
    - ``format_version``: the layout version, which `from_dict` checks.
    - ``roles``: the role of each column.
    - ``schema``: the training data's column names and dtypes, or ``None``.
    - ``lineage``: the column lineage, or ``None``.
    - ``steps``: for each step, its class name and its state (the
      hyperparameters and the statistics learned at fit time).

    Parameters
    ----------
    recipe
        The fitted recipe to serialize.

    Returns
    -------
    dict
        The payload described above.

    Raises
    ------
    TypeError
        If a step holds something that cannot be serialized: a raw polars or
        narwhals selector, a ``dateutil`` rule, or a function (``StepMutate``).

    Warns
    -----
    UserWarning
        If a step, or a dataclass in its fit state, comes from a package that
        `from_dict` refuses by default (see `from_dict`). The payload is still
        produced, but loading it needs the step registered or passed as
        ``trusted``.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Recipe, StepNormalize, to_dict
    from mlfeatures import Selectors as s
    from mlfeatures.data import credit

    credit = credit.drop_nulls()
    ```

    ```{python}
    recipe = Recipe().add(StepNormalize(s.numeric_features()))
    fitted = recipe.fit(credit)
    payload = to_dict(fitted)
    payload["steps"][0]["type"]
    ```
    """
    from mlfeatures import __version__

    steps = [_encode_step(step) for step in recipe.steps]
    _warn_if_unloadable(steps)
    payload: dict[str, Any] = {
        "mlfeatures_version": __version__,
        "format_version": FORMAT_VERSION,
        "roles": _encode_roles(recipe.roles),
        "schema": None if recipe.schema is None else _encode_schema(recipe.schema),
        "lineage": None if recipe.lineage is None else _encode_lineage(recipe.lineage),
        "steps": steps,
    }
    return payload


def from_dict(data: dict[str, Any], *, trusted: Iterable[type] = ()) -> FittedRecipe:
    """Reconstruct a `FittedRecipe` from the ``dict`` produced by `to_dict`.

    The payload is treated as untrusted data: it can name only classes from
    ``mlfeatures``, from packages that register a step under the
    ``mlfeatures.steps`` entry-point group, and from the packages of the classes
    passed as ``trusted``. Each class must also be of the right kind (a `Step`
    subclass for a step, a dataclass for fit state). Loading never calls
    anything else the payload names, and a name is checked before its module is
    imported. The values inside the payload are not validated.

    A payload without a ``format_version`` is read as the current version. One
    with a different version is refused, because the layout changed
    incompatibly; fit the recipe again and save it again.

    Parameters
    ----------
    data
        The ``dict`` produced by `to_dict`.
    trusted
        Classes whose top-level packages the payload may also name classes from,
        for a custom step that is not registered (one defined in a script or
        notebook, or still under development). Pass only classes you wrote or
        trust: a package listed here is imported as the payload asks. Defaults
        to none.

    Returns
    -------
    FittedRecipe
        The recipe, ready to `FittedRecipe.transform` new data.

    Raises
    ------
    ValueError
        If the payload is not a mapping with ``roles`` and ``steps``, has a
        ``format_version`` this mlfeatures does not read, names a class outside
        the trusted packages, or names something that is not a step or
        dataclass.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Recipe, StepNormalize, from_dict, to_dict
    from mlfeatures import Selectors as s
    from mlfeatures.data import credit

    credit = credit.drop_nulls()
    ```

    ```{python}
    recipe = Recipe().add(StepNormalize(s.numeric_features()))
    fitted = recipe.fit(credit)
    restored = from_dict(to_dict(fitted))
    restored
    ```
    """
    from mlfeatures.recipe import FittedRecipe

    if not isinstance(data, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError(
            f"expected the payload to be a mapping, got {type(data).__name__!r}"
        )
    found = data.get("format_version", FORMAT_VERSION)
    if found != FORMAT_VERSION:
        raise ValueError(
            f"cannot load a recipe serialized with format_version {found!r}; "
            f"this mlfeatures writes and reads format_version {FORMAT_VERSION}, and "
            "the layout changed incompatibly; re-fit the recipe to re-serialize"
        )
    missing = [key for key in ("roles", "steps") if key not in data]
    if missing:
        raise ValueError(f"payload is missing required key(s) {missing!r}")
    if not isinstance(data["steps"], list):
        raise ValueError("expected 'steps' to be a list of step payloads")
    if not isinstance(data["roles"], dict):
        raise ValueError("expected 'roles' to be a mapping of column to roles")
    with _decoding(trusted):
        steps = [_decode_step(p) for p in data["steps"]]
        roles = _decode_roles(data["roles"])
        schema = None if data.get("schema") is None else _decode_schema(data["schema"])
        lineage = (
            None if data.get("lineage") is None else _decode_lineage(data["lineage"])
        )
    return FittedRecipe(steps=steps, roles=roles, lineage=lineage, schema=schema)


def to_json(recipe: FittedRecipe, *, indent: int = 2) -> str:
    """Serialize a `FittedRecipe` to a diffable JSON string.

    The string is the `to_dict` payload as strictly valid JSON (non-finite
    floats are encoded explicitly, never as a bare ``NaN``). It needs no
    optional dependency, so it is the default choice for storing a recipe next
    to a model. Warns like
    `to_dict` when the recipe names a class that `from_json` refuses by default.

    Parameters
    ----------
    recipe
        The fitted recipe to serialize.
    indent
        Spaces of indentation per level. Defaults to 2.

    Returns
    -------
    str
        The JSON text.

    Raises
    ------
    TypeError
        As for `to_dict`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Recipe, StepNormalize, to_json
    from mlfeatures import Selectors as s
    from mlfeatures.data import credit

    credit = credit.drop_nulls()
    ```

    ```{python}
    recipe = Recipe().add(StepNormalize(s.numeric_features()))
    fitted = recipe.fit(credit)
    text = to_json(fitted)
    text[:200]
    ```

    A smaller `indent` gives a more compact file.

    ```{python}
    compact = to_json(fitted, indent=0)
    compact[:200]
    ```
    """
    # allow_nan=False keeps the output strictly valid JSON. Non-finite floats are
    # already encoded explicitly by the value codec, so a stray NaN reaching here
    # is a bug to surface, not an invalid ``NaN`` token to emit.
    return json.dumps(to_dict(recipe), indent=indent, allow_nan=False)


def from_json(text: str, *, trusted: Iterable[type] = ()) -> FittedRecipe:
    """Reconstruct a `FittedRecipe` from the JSON produced by `to_json`.

    Takes the same ``trusted`` argument as `from_dict` and raises the same
    ``ValueError`` for a class outside the trusted packages. Text that is not
    valid JSON raises ``json.JSONDecodeError`` (a ``ValueError`` subclass).

    Parameters
    ----------
    text
        The JSON text produced by `to_json`.
    trusted
        Classes whose top-level packages the payload may also name classes from;
        see `from_dict`. Defaults to none.

    Returns
    -------
    FittedRecipe
        The recipe, ready to `FittedRecipe.transform` new data.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Recipe, StepNormalize, from_json, to_json
    from mlfeatures import Selectors as s
    from mlfeatures.data import credit

    credit = credit.drop_nulls()
    ```

    ```{python}
    recipe = Recipe().add(StepNormalize(s.numeric_features()))
    fitted = recipe.fit(credit)
    restored = from_json(to_json(fitted))
    restored.transform(credit).head()
    ```
    """
    return from_dict(json.loads(text), trusted=trusted)


def _require_yaml() -> Any:
    """Import ``pyyaml`` or raise a clear, actionable ``ImportError``."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "YAML serialization needs the optional 'pyyaml' package; "
            "install it with 'pip install pyyaml' (or 'pip install mlfeatures[yaml]')"
        ) from exc
    return yaml


def to_yaml(recipe: FittedRecipe) -> str:
    """Serialize a `FittedRecipe` to a diffable YAML string.

    The same `to_dict` payload as `to_json`, written as YAML with the keys in
    their stable order and a block layout that diffs cleanly in review, so the
    two formats round-trip identically. Choose it over `to_json` when people
    will read or review the file; it needs the optional ``pyyaml`` package.
    Warns like `to_dict` when the recipe names a class that `from_yaml` refuses
    by default.

    Parameters
    ----------
    recipe
        The fitted recipe to serialize.

    Returns
    -------
    str
        The YAML text.

    Raises
    ------
    ImportError
        If the optional ``pyyaml`` package is not installed
        (``pip install mlfeatures[yaml]``).
    TypeError
        As for `to_dict`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Recipe, StepNormalize, to_yaml
    from mlfeatures import Selectors as s
    from mlfeatures.data import credit

    credit = credit.drop_nulls()
    ```

    ```{python}
    recipe = Recipe().add(StepNormalize(s.numeric_features()))
    fitted = recipe.fit(credit)
    text = to_yaml(fitted)
    text[:200]
    ```
    """
    yaml = _require_yaml()
    # sort_keys=False keeps the payload's stable key order, and the block style
    # (default_flow_style=False) is the layout that diffs cleanly in review.
    return str(
        yaml.safe_dump(to_dict(recipe), sort_keys=False, default_flow_style=False)
    )


def from_yaml(text: str, *, trusted: Iterable[type] = ()) -> FittedRecipe:
    """Reconstruct a `FittedRecipe` from the YAML produced by `to_yaml`.

    Takes the same ``trusted`` argument as `from_dict` and raises the same
    ``ValueError`` for a class outside the trusted packages. The YAML is parsed
    without building arbitrary objects, and the class names inside it are
    restricted by `from_dict`, as they are for JSON.

    Parameters
    ----------
    text
        The YAML text produced by `to_yaml`.
    trusted
        Classes whose top-level packages the payload may also name classes from;
        see `from_dict`. Defaults to none.

    Returns
    -------
    FittedRecipe
        The recipe, ready to `FittedRecipe.transform` new data.

    Raises
    ------
    ImportError
        If the optional ``pyyaml`` package is not installed
        (``pip install mlfeatures[yaml]``).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Recipe, StepNormalize, from_yaml, to_yaml
    from mlfeatures import Selectors as s
    from mlfeatures.data import credit

    credit = credit.drop_nulls()
    ```

    ```{python}
    recipe = Recipe().add(StepNormalize(s.numeric_features()))
    fitted = recipe.fit(credit)
    restored = from_yaml(to_yaml(fitted))
    restored.transform(credit).head()
    ```
    """
    yaml = _require_yaml()
    return from_dict(yaml.safe_load(text), trusted=trusted)


def minimize(recipe: FittedRecipe) -> FittedRecipe:
    """Return a `FittedRecipe` that keeps only what ``transform()`` needs.

    Each step decides what it can drop: a step that keeps something only for
    inspection after fitting, or only because fitting needed it, returns a
    lighter version of itself that transforms identically. A step that has
    nothing to drop is kept as is, shared with ``recipe`` rather than copied,
    and ``recipe`` itself is never modified. The roles, schema, and lineage are kept, so the minimized recipe
    still validates new data at the boundary and reports its lineage.

    Call it before `to_dict`, `to_json`, or `to_yaml` when the recipe is going
    to be shipped with a model, so the saved file carries only what scoring
    needs; the writers save the recipe as it is and do not minimize it. No
    shipped step currently holds anything to drop, so today the result is an
    equivalent recipe.

    Parameters
    ----------
    recipe
        The fitted recipe to minimize.

    Returns
    -------
    FittedRecipe
        A new recipe with each step minimized.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Recipe, StepNormalize, minimize
    from mlfeatures import Selectors as s
    from mlfeatures.data import credit

    credit = credit.drop_nulls()
    ```

    ```{python}
    recipe = Recipe().add(StepNormalize(s.numeric_features()))
    fitted = recipe.fit(credit)
    small = minimize(fitted)
    small.transform(credit).head()
    ```
    """
    from mlfeatures.recipe import FittedRecipe

    return FittedRecipe(
        steps=[step.minimize() for step in recipe.steps],
        roles=recipe.roles,
        lineage=recipe.lineage,
        schema=recipe.schema,
    )
