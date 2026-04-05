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

_TEXT_FORMATS = {"string", "markdown"}
_STRUCTURED_FORMATS = {"json", "toml", "yaml"}
_LIST_TYPE_RE = re.compile(r"^list\[(?P<inner>[a-z]+)\]$")


@dataclass(frozen=True)
class OutputFieldSpec:
    name: str
    type_name: str


@dataclass(frozen=True)
class OutputSchemaSpec:
    """Normalised output schema derived from the IR's output_format + output_fields."""

    format: str  # 'json' | 'toml' | 'yaml' | 'markdown' | 'string'
    fields: tuple[OutputFieldSpec, ...] = ()

    @property
    def is_text(self) -> bool:
        return self.format in _TEXT_FORMATS

    @property
    def is_structured(self) -> bool:
        return self.format in _STRUCTURED_FORMATS

    # Legacy compatibility: callers that used .mode == "structured" / "text"
    @property
    def mode(self) -> str:
        return "structured" if self.is_structured else "text"

    # Legacy compatibility: .raw used in a few places
    @property
    def raw(self) -> str:
        if not self.fields:
            return self.format
        parts = ", ".join(f"{f.name}: {f.type_name}" for f in self.fields)
        return f"{self.format} {{ {parts} }}"


def parse_output_schema(
    output_format: str | None = None,
    output_fields: list[dict[str, str]] | None = None,
) -> OutputSchemaSpec:
    """Build an OutputSchemaSpec from the IR's output_format and output_fields.

    Also accepts a bare legacy string (e.g. ``"status: str, message: str"``) as
    the first argument for backwards-compatibility with tests and generated
    modules that pass the old single-string form.
    """
    fmt = (output_format or "string").strip().lower()

    # --- Legacy: single comma-separated string like "status: str, message: str" ---
    if ":" in fmt and fmt not in _STRUCTURED_FORMATS and fmt not in _TEXT_FORMATS:
        return _parse_legacy_string(output_format or "string")

    fields: list[OutputFieldSpec] = []
    for item in output_fields or []:
        name = item.get("name", "").strip()
        type_name = item.get("type", "str").strip().lower()
        if name:
            _python_type_for_name(type_name)  # validate
            fields.append(OutputFieldSpec(name=name, type_name=type_name))

    return OutputSchemaSpec(format=fmt, fields=tuple(fields))


def _parse_legacy_string(schema_text: str) -> OutputSchemaSpec:
    """Parse the old comma-separated 'field: type' string format."""
    raw = schema_text.strip()
    normalized = raw.lower()

    if normalized in _TEXT_FORMATS or normalized in _STRUCTURED_FORMATS:
        return OutputSchemaSpec(format=normalized, fields=())

    field_specs = []
    for item in raw.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                "Structured output schemas must use 'field: type' items "
                f"separated by commas. Got: {raw!r}"
            )
        name, type_name = chunk.split(":", 1)
        field_name = name.strip()
        normalized_type = type_name.strip().lower()
        if not field_name:
            raise ValueError(f"Output schema field name is empty in {raw!r}")
        _python_type_for_name(normalized_type)
        field_specs.append(OutputFieldSpec(name=field_name, type_name=normalized_type))

    if not field_specs:
        return OutputSchemaSpec(format="string", fields=())

    return OutputSchemaSpec(format="json", fields=tuple(field_specs))


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
                import tomli as tomllib  # type: ignore[no-redef]
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


def build_dspy_signature_class(schema_text: str, class_name: str = "GeneratedSignature"):
    schema = _parse_legacy_string(schema_text)
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
