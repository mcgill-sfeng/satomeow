from __future__ import annotations

import argparse
from pathlib import Path

TEMPLATES = {
    "monthly_csv_to_json": "monthly_csv_to_json.py.tmpl",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a preprocessing script from a template.")
    parser.add_argument("--template", required=True, choices=sorted(TEMPLATES))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    template_path = templates_dir / TEMPLATES[args.template]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
