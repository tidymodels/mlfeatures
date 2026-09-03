"""Diffable JSON/YAML serialization for fitted recipes (``design/serialization.md``).

A `FittedRecipe` is serialized to a plain, human-readable ``dict`` (and from
there to JSON or YAML) that captures every step's **hyperparameters and learned
statistics** plus the recipe-level role, ptype, and lineage state. The format is
stable and diffable so a trained recipe can be reviewed in a pull request, saved
alongside a model, and reloaded in another process for inference.

Nothing in the payload is a closure, environment, or pickled object: step state
is plain fields, selectors are serializable expression objects, and the step
class is looked up by qualified name at load time. That property is what the
no-closure conformance test relies on -- a closure cannot be JSON-encoded, so a
passing round-trip *is* the guarantee.

The public surface is a small module API -- `to_dict`, `from_dict`, `to_json`,
`from_json`, `to_yaml`, `from_yaml`, and `minimize` -- mirrored by thin
delegating methods on `FittedRecipe` (``fitted.to_json()``,
`FittedRecipe.from_json(text)``, ...).

JSON and YAML are two thin encodings over the same JSON-safe ``dict`` format, so
both round-trip identically. YAML support needs the optional ``pyyaml`` package;
`to_yaml` / `from_yaml` raise a clear ``ImportError`` when it is missing.

Known limitations
-----------------
- ``Enum`` categories do not round-trip: an ``Enum`` column's ptype/schema
  dtype comes back as an empty-category ``Enum`` after a JSON round-trip. This
  is enough for the schema/ptype checks, which only compare dtype *identity*
  by base type, but it does mean ``ColSchema``, which already stores dtypes as
  ``str(dt)``, is a more faithful record of an ``Enum``'s categories than a
  deserialized ``Ptype``/``Schema`` is.
- ``Datetime`` (time-unit, time zone) and ``Duration`` (time-unit) *do*
  round-trip their parameters (see `_encode_dtype_instance` /
  `_decode_dtype_instance`) -- `StepDatetimeFeatures`'s conformance suite
  depends on it: a pandas frame's nanosecond-precision ``Datetime`` must still
  match the ptype snapshot after a JSON round-trip.
- Raw polars / narwhals selectors are not serializable. Only shoez `Selectors`
  and plain name lists are; a native selector raises a clear ``TypeError``.
"""

# This module is the one place that deliberately reaches into the internal
# representation of steps, selectors, and the lineage schema and juggles the
# resulting plain runtime data (``Any`` scalars, lists, dicts, ndarrays) -- that
# is what a serializer does. Pyright strict flags every private-field access
# (reportPrivateUsage) and every ``Any``-typed value the codec touches
# (reportUnknown*); those add no safety a serializer can act on, so they are
# disabled here (and only here). mypy strict still fully type-checks this file,
# so the relaxation is scoped and covered.
# pyright: reportPrivateUsage=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import dataclasses
import importlib
import json
import math
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw
import numpy as np

if TYPE_CHECKING:
    from shoez.recipe import FittedRecipe

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

# The narwhals dtype classes shoez supports. Encoding uses the class name (the
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
    it falls back to an empty-category enum, which is enough for the schema/ptype
    checks that only compare dtype *identity* by base type.
    """
    cls = _dtype_class(name)
    if cls is nw.Enum:
        return nw.Enum([])
    return cls()


def _encode_dtype_instance(dtype: nw.dtypes.DType) -> str | dict[str, Any]:
    """Encode a concrete dtype instance for the ptype/schema-node codecs.

    Unlike `_dtype_name` (used where only the base type matters, e.g. a
    selector's stored dtype), this preserves ``Datetime``'s time-unit and time
    zone, and ``Duration``'s time-unit, since two ``Datetime`` columns of
    different precision are not the same ptype -- `Ptype.check` would
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
    # ColSchema, ShoezSelector, and frozen fit dataclasses.
    from shoez.selectors import ShoezSelector
    from shoez.types import ColSchema

    if isinstance(value, ColSchema):
        return {
            "__colschema__": {
                "columns": list(value.columns),
                "dtypes": dict(value.dtypes),
            }
        }
    if isinstance(value, ShoezSelector):
        return {"__selector__": _encode_selector(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        return {
            "__dataclass__": f"{cls.__module__}.{cls.__qualname__}",
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
            return _decode_dataclass(obj["__dataclass__"], obj["fields"])
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
                from shoez.types import ColSchema

                return ColSchema(
                    columns=list(payload["columns"]), dtypes=dict(payload["dtypes"])
                )
            if tag == "__selector__":
                return _decode_selector(payload)
        return {k: _decode(v) for k, v in obj.items()}
    return obj


def _decode_dataclass(qualname: str, fields: dict[str, Any]) -> Any:
    cls = _load_qualname(qualname)
    return cls(**{name: _decode(v) for name, v in fields.items()})


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
    from shoez.selectors import (
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
            step_type = f"{sel._step_type.__module__}.{sel._step_type.__qualname__}"
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
            "cannot serialize a native polars/narwhals selector; use shoez "
            "'Selectors' (or a plain list of column names) instead"
        )
    raise TypeError(f"cannot serialize selector of type {type(sel).__name__!r}")


def _decode_selector(state: dict[str, Any]) -> Any:
    from shoez.selectors import (
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
            _load_qualname(state["step_type"])
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


def _load_qualname(qualname: str) -> Any:
    """Import ``module.Qualname`` and return the attribute.

    Falls back to the ``shoez.steps`` entry-point group so third-party plugin
    steps deserialize by the name they registered under.
    """
    module_name, _, attr = qualname.rpartition(".")
    if module_name:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        except (ImportError, AttributeError):
            pass
    for ep in _step_entry_points():
        cls = ep.load()
        if f"{cls.__module__}.{cls.__qualname__}" == qualname:
            return cls
    raise ValueError(f"cannot locate {qualname!r} for deserialization")


def _step_entry_points() -> Any:
    from importlib.metadata import entry_points

    try:
        return entry_points(group="shoez.steps")
    except TypeError:  # pragma: no cover - Python <3.10 signature
        return entry_points().get("shoez.steps", [])


# --------------------------------------------------------------------------- #
# Recipe-level codecs (roles / ptype / schema)
# --------------------------------------------------------------------------- #


def _encode_roles(roles: Any) -> dict[str, list[str]]:
    return {col: sorted(roles.of(col)) for col in roles.all_columns()}


def _decode_roles(payload: dict[str, list[str]]) -> Any:
    from shoez.types import Roles

    return Roles({col: set(names) for col, names in payload.items()})


def _encode_ptype(ptype: Any) -> dict[str, Any]:
    return {
        "schema": {
            name: _encode_dtype_instance(ptype.schema[name])
            for name in ptype.schema.names()
        },
        "roles": _encode_roles(ptype.roles),
        "transform_required": dict(ptype.transform_required),
    }


def _decode_ptype(payload: dict[str, Any]) -> Any:
    from shoez.ptype import Ptype

    schema = nw.Schema(
        {name: _decode_dtype_instance(dt) for name, dt in payload["schema"].items()}
    )
    return Ptype(
        schema=schema,
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
    from shoez.schema import ColumnNode

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


def _encode_schema(schema: Any) -> dict[str, Any]:
    return {
        "live": [_encode_node(node) for node in schema._live.values()],
        "history": [_encode_node(node) for node in schema._history],
    }


def _decode_schema(payload: dict[str, Any]) -> Any:
    from shoez.schema import Schema

    schema = Schema()
    for node_payload in payload["live"]:
        node = _decode_node(node_payload)
        schema._live[node.name] = node
    schema._history = [_decode_node(n) for n in payload["history"]]
    return schema


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _encode_step(step: Any) -> dict[str, Any]:
    cls = type(step)
    return {
        "type": f"{cls.__module__}.{cls.__qualname__}",
        "state": step.to_state(),
    }


def _decode_step(payload: dict[str, Any]) -> Any:
    cls = _load_qualname(payload["type"])
    return cls.from_state(payload["state"])


def to_dict(recipe: FittedRecipe) -> dict[str, Any]:
    """Serialize a `FittedRecipe` to a diffable, JSON-safe ``dict``.

    Captures every step's hyperparameters and learned statistics plus the
    recipe-level roles, training ptype, and lineage schema. The result contains
    only plain data, so it round-trips through `from_dict` and is stable enough
    to diff in code review.
    """
    from shoez import __version__

    payload: dict[str, Any] = {
        "shoez_version": __version__,
        "format_version": FORMAT_VERSION,
        "roles": _encode_roles(recipe.roles),
        "ptype": None if recipe.ptype is None else _encode_ptype(recipe.ptype),
        "schema": None if recipe.schema is None else _encode_schema(recipe.schema),
        "steps": [_encode_step(step) for step in recipe.steps],
    }
    return payload


def from_dict(data: dict[str, Any]) -> FittedRecipe:
    """Reconstruct a `FittedRecipe` from the ``dict`` produced by `to_dict`."""
    from shoez.recipe import FittedRecipe

    found = data.get("format_version", FORMAT_VERSION)
    if found != FORMAT_VERSION:
        raise ValueError(
            f"cannot load a recipe serialized with format_version {found!r}; "
            f"this shoez writes and reads format_version {FORMAT_VERSION}, and "
            "the layout changed incompatibly; re-fit the recipe to re-serialize"
        )
    steps = [_decode_step(p) for p in data["steps"]]
    roles = _decode_roles(data["roles"])
    ptype = None if data.get("ptype") is None else _decode_ptype(data["ptype"])
    schema = None if data.get("schema") is None else _decode_schema(data["schema"])
    return FittedRecipe(steps=steps, roles=roles, schema=schema, ptype=ptype)


def to_json(recipe: FittedRecipe, *, indent: int = 2) -> str:
    """Serialize a `FittedRecipe` to a diffable JSON string.

    ``allow_nan=False`` keeps the output strictly valid JSON: non-finite floats
    are already encoded explicitly by the value codec, so a stray ``NaN`` here
    signals a bug rather than emitting an invalid ``NaN`` token.
    """
    return json.dumps(to_dict(recipe), indent=indent, allow_nan=False)


def from_json(text: str) -> FittedRecipe:
    """Reconstruct a `FittedRecipe` from the JSON produced by `to_json`."""
    return from_dict(json.loads(text))


def _require_yaml() -> Any:
    """Import ``pyyaml`` or raise a clear, actionable ``ImportError``."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "YAML serialization needs the optional 'pyyaml' package; "
            "install it with 'pip install pyyaml' (or 'pip install shoez[yaml]')"
        ) from exc
    return yaml


def to_yaml(recipe: FittedRecipe) -> str:
    """Serialize a `FittedRecipe` to a diffable YAML string.

    A thin encoding over the same JSON-safe ``dict`` as `to_json`, so the two
    formats round-trip identically. ``sort_keys=False`` preserves the payload's
    stable key order and ``default_flow_style=False`` keeps the block layout that
    diffs cleanly in review. Needs the optional ``pyyaml`` package.
    """
    yaml = _require_yaml()
    return str(
        yaml.safe_dump(to_dict(recipe), sort_keys=False, default_flow_style=False)
    )


def from_yaml(text: str) -> FittedRecipe:
    """Reconstruct a `FittedRecipe` from the YAML produced by `to_yaml`."""
    yaml = _require_yaml()
    return from_dict(yaml.safe_load(text))


def minimize(recipe: FittedRecipe) -> FittedRecipe:
    """Return a copy of ``recipe`` with each step's transform-unneeded artifacts stripped.

    Delegates to each step's `Step.minimize`, which by default returns the step
    unchanged. Steps that retain training artifacts only needed at fit time
    (e.g. a future KNN imputer's neighbor matrix) override ``minimize`` to drop
    them. The roles, ptype, and lineage schema are preserved so the minimized
    recipe still validates new data at the boundary and reports its schema.
    """
    from shoez.recipe import FittedRecipe

    return FittedRecipe(
        steps=[step.minimize() for step in recipe.steps],
        roles=recipe.roles,
        schema=recipe.schema,
        ptype=recipe.ptype,
    )
