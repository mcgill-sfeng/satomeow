from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent.metamodel import OutputSpec

_SCALAR_TYPES = {
    "str": str,
    "string": str,
    "int": int,
    "float": float,
    "bool": bool,
}

_STRUCTURED_FORMATS = {"json", "toml", "yaml"}
_LIST_TYPE_RE = re.compile(r"^list\[(?P<inner>[a-z]+)\]$")


@dataclass(frozen=True)
class OutputFieldSpec:
    name: str
    type_name: str


@dataclass(frozen=True)
class OutputSchemaSpec:
    format: str  # 'json' | 'toml' | 'yaml' | 'markdown' | 'string'
    fields: tuple[OutputFieldSpec, ...] = ()

    @property
    def is_structured(self) -> bool:
        return self.format in _STRUCTURED_FORMATS


def parse_output_schema(output_spec: OutputSpec | None = None) -> OutputSchemaSpec:
    """Build an OutputSchemaSpec from a metamodel OutputSpec."""
    if output_spec is None:
        return OutputSchemaSpec(format="string", fields=())

    fmt = (output_spec.format or "string").strip().lower()
    fields: list[OutputFieldSpec] = []
    for field in output_spec.fields or []:
        name = (field.name or "").strip()
        type_name = (field.type or "str").strip().lower()
        if name:
            _python_type_for_name(type_name)
            fields.append(OutputFieldSpec(name=name, type_name=type_name))

    return OutputSchemaSpec(format=fmt, fields=tuple(fields))


def describe_output_schema(schema: OutputSchemaSpec) -> str:
    if schema.format == "markdown":
        return "Return Markdown."
    if schema.format == "string":
        return "Return plain text."
    fields = ", ".join(f"{f.name}: {f.type_name}" for f in schema.fields)
    if schema.format == "toml":
        return f"Return valid TOML with exactly these keys and types: {fields}."
    if schema.format == "yaml":
        return f"Return valid YAML with exactly these keys and types: {fields}."
    # json (default structured)
    return f"Return a JSON object with exactly these fields and types: {fields}."


def coerce_structured_output(payload: Any, schema: OutputSchemaSpec) -> dict[str, Any]:
    """Validate and type-coerce a structured output payload against the schema.

    For TOML and YAML, ``payload`` is expected to be a string that is first
    parsed into a dict, then validated against the field schema.
    For JSON, ``payload`` should already be a dict (the SDK enforces this via
    ``output_type``).
    """
    if not schema.is_structured:
        raise ValueError("Structured coercion only applies to structured schemas.")

    if schema.format in ("toml", "yaml"):
        payload = _parse_text_format(payload, schema.format)

    if not isinstance(payload, dict):
        raise TypeError("Structured output must be a JSON object.")

    result = {}
    missing = [field.name for field in schema.fields if field.name not in payload]
    if missing:
        raise ValueError(f"Missing structured output field(s): {', '.join(missing)}")

    for field in schema.fields:
        result[field.name] = _coerce_value(payload[field.name], field.type_name)

    extras = [key for key in payload.keys() if key not in result]
    if extras:
        raise ValueError(f"Unexpected structured output field(s): {', '.join(extras)}")

    return result


def _parse_text_format(payload: Any, fmt: str) -> Any:
    """Parse a TOML or YAML string into a dict."""
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        raise TypeError(f"Expected a {fmt.upper()} string, got {type(payload).__name__}")
    if fmt == "toml":
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[import-not-found,no-redef]
            except ImportError as exc:
                raise ImportError("Install 'tomli' for TOML output validation on Python < 3.11") from exc
        return tomllib.loads(payload)
    if fmt == "yaml":
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install 'pyyaml' for YAML output validation") from exc
        return yaml.safe_load(payload)
    return payload


def _coerce_value(value: Any, type_name: str) -> Any:
    list_match = _LIST_TYPE_RE.match(type_name)
    if list_match:
        if not isinstance(value, list):
            raise TypeError(f"Expected list for type {type_name}, got {type(value).__name__}")
        inner_name = list_match.group("inner")
        inner_type = _SCALAR_TYPES.get(inner_name)
        if inner_type is None:
            raise ValueError(f"Unsupported output schema type: {type_name}")
        return [_coerce_scalar(item, inner_type) for item in value]
    target_type = _python_type_for_name(type_name)
    return _coerce_scalar(value, target_type)


def _coerce_scalar(value: Any, target_type: type) -> Any:
    if isinstance(value, target_type):
        return value
    if target_type is bool and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    try:
        return target_type(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Could not coerce value {value!r} to {target_type.__name__}") from exc


def _python_type_for_name(type_name: str) -> type:
    scalar = _SCALAR_TYPES.get(type_name)
    if scalar is not None:
        return scalar

    list_match = _LIST_TYPE_RE.match(type_name)
    if list_match:
        return list

    raise ValueError(f"Unsupported output schema type: {type_name}")
