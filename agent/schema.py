from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, get_args, get_origin

_SCALAR_TYPES = {
    "str": str,
    "string": str,
    "int": int,
    "float": float,
    "bool": bool,
}

_TEXT_SCHEMAS = {"string", "text", "markdown"}
_LIST_TYPE_RE = re.compile(r"^list\[(?P<inner>[a-z]+)\]$")


@dataclass(frozen=True)
class OutputFieldSpec:
    name: str
    type_name: str


@dataclass(frozen=True)
class OutputSchemaSpec:
    raw: str
    mode: str
    fields: tuple[OutputFieldSpec, ...] = ()

    @property
    def is_text(self) -> bool:
        return self.mode == "text"

    @property
    def is_structured(self) -> bool:
        return self.mode == "structured"


def parse_output_schema(schema_text: str | None) -> OutputSchemaSpec:
    raw = (schema_text or "string").strip()
    normalized = raw.lower()

    if normalized in _TEXT_SCHEMAS:
        return OutputSchemaSpec(raw=raw, mode="text")

    field_specs = []
    for item in raw.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                "Structured output schemas must use 'field: type' items " f"separated by commas. Got: {raw!r}"
            )
        name, type_name = chunk.split(":", 1)
        field_name = name.strip()
        normalized_type = type_name.strip().lower()
        if not field_name:
            raise ValueError(f"Output schema field name is empty in {raw!r}")
        _python_type_for_name(normalized_type)
        field_specs.append(OutputFieldSpec(name=field_name, type_name=normalized_type))

    if not field_specs:
        return OutputSchemaSpec(raw=raw, mode="text")

    return OutputSchemaSpec(raw=raw, mode="structured", fields=tuple(field_specs))


def describe_output_schema(schema: OutputSchemaSpec) -> str:
    if schema.is_text:
        if schema.raw.lower() == "markdown":
            return "Return Markdown."
        return "Return plain text."

    fields = ", ".join(f"{field.name}: {field.type_name}" for field in schema.fields)
    return "Return a JSON object with exactly these fields and types: " f"{fields}."


def coerce_structured_output(payload: Any, schema: OutputSchemaSpec) -> dict[str, Any]:
    if not schema.is_structured:
        raise ValueError("Structured coercion only applies to structured schemas.")
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


def build_dspy_signature_class(schema_text: str, class_name: str = "GeneratedSignature"):
    schema = parse_output_schema(schema_text)
    if schema.is_text:
        return None

    try:
        import dspy
    except ImportError as exc:
        raise RuntimeError("dspy is not installed. Install dspy to build structured signatures.") from exc

    annotations = {}
    namespace = {"__annotations__": annotations}

    for field in schema.fields:
        annotations[field.name] = _python_type_for_name(field.type_name)
        namespace[field.name] = dspy.OutputField(desc=field.name.replace("_", " "))

    return type(class_name, (dspy.Signature,), namespace)


def _coerce_value(value: Any, type_name: str) -> Any:
    target_type = _python_type_for_name(type_name)
    origin = get_origin(target_type)
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"Expected list for type {type_name}, got {type(value).__name__}")
        inner_type = get_args(target_type)[0]
        return [_coerce_scalar(item, inner_type) for item in value]
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
        inner_name = list_match.group("inner")
        inner_type = _SCALAR_TYPES.get(inner_name)
        if inner_type is None:
            raise ValueError(f"Unsupported output schema type: {type_name}")
        return list[inner_type]

    raise ValueError(f"Unsupported output schema type: {type_name}")
