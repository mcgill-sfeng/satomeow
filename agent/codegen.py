from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from agent.ir import build_prompt_ir
from agent.parser import parse_model

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def render_agent_module_from_ir(prompt_ir: dict, *, source_model_path: str | None = None) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("generated_agent.py.j2")
    return template.render(
        spec_json=json.dumps(prompt_ir, indent=2, sort_keys=True),
        source_model_path=source_model_path,
        project_root=str(Path(__file__).resolve().parent.parent),
    )


def render_agent_module(model_path: str | Path) -> str:
    system = parse_model(model_path)
    prompt_ir = build_prompt_ir(system)
    return render_agent_module_from_ir(prompt_ir, source_model_path=str(model_path))


def generate_agent_module(model_path: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.write_text(render_agent_module(model_path), encoding="utf-8")
    return output
