from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a simple SVG bar chart.")
    parser.add_argument("--input", required=True, help="Path to list[{label, value}] JSON data")
    parser.add_argument("--output", required=True, help="Path to the SVG file to create")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    records = json.loads(input_path.read_text(encoding="utf-8"))

    validate_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_svg(records), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "artifact_path": str(output_path),
                "points": len(records),
            }
        )
    )
    return 0


def validate_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a non-empty list of records.")

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be an object.")
        if "label" not in record or "value" not in record:
            raise ValueError("Each record must contain 'label' and 'value'.")
        if not isinstance(record["label"], str):
            raise ValueError("Record 'label' must be a string.")
        if not isinstance(record["value"], (int, float)):
            raise ValueError("Record 'value' must be numeric.")


def render_svg(records) -> str:
    width = 720
    height = 420
    margin = 50
    chart_height = height - (margin * 2)
    bar_width = 80
    gap = 24
    max_value = max(record["value"] for record in records)
    if max_value <= 0:
        raise ValueError("All values are non-positive; cannot render chart.")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f5ef" />',
        '<text x="50" y="36" font-size="24" font-family="monospace" fill="#22333b">Monthly Data Visualization</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#22333b" stroke-width="2" />',
    ]

    for index, record in enumerate(records):
        scaled_height = (record["value"] / max_value) * (chart_height - 20)
        x = margin + 20 + index * (bar_width + gap)
        y = height - margin - scaled_height
        parts.append(
            f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{scaled_height:.2f}" fill="#2a9d8f" rx="6" />'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{height - margin + 20}" text-anchor="middle" font-size="12" font-family="monospace" fill="#22333b">{record["label"]}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-size="12" font-family="monospace" fill="#22333b">{record["value"]}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
