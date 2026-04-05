from __future__ import annotations

import importlib.util
import importlib.metadata as importlib_metadata
import json
import os
import re
import shutil
import stat
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from agent.ir import build_prompt_ir
from agent.parser import parse_model

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
RUNTIME_FILES = ("__init__.py", "runtime.py", "schema.py")
PORTABLE_ROOT_IMPORTS = ("agents", "openai", "dotenv", "pydantic", "yaml")
_REQ_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")
_IMPORT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def render_agent_module_from_ir(prompt_ir: dict, *, source_model_path: str | None = None) -> str:
    return _render_template(
        "generated_agent.py.j2",
        prompt_ir,
        source_model_path=source_model_path,
        project_root=str(Path(__file__).resolve().parent.parent),
    )


def render_portable_agent_module_from_ir(prompt_ir: dict) -> str:
    return _render_template(
        "portable_agent.py.j2",
        prompt_ir,
        source_model_path=None,
        project_root="",
    )


def _render_template(
    template_name: str,
    prompt_ir: dict,
    *,
    source_model_path: str | None,
    project_root: str,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_name)
    return template.render(
        spec_json=json.dumps(prompt_ir, indent=2, sort_keys=True),
        source_model_path=source_model_path,
        project_root=project_root,
    )


def render_agent_module(model_path: str | Path) -> str:
    system = parse_model(model_path)
    prompt_ir = build_prompt_ir(system)
    return render_agent_module_from_ir(prompt_ir, source_model_path=str(model_path))


def generate_agent_module(model_path: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.write_text(render_agent_module(model_path), encoding="utf-8")
    return output


def generate_portable_agent_bundle(model_path: str | Path, output_dir: str | Path) -> Path:
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system = parse_model(model_path)
    prompt_ir = build_prompt_ir(system)

    (output_dir / "main.py").write_text(
        render_portable_agent_module_from_ir(prompt_ir),
        encoding="utf-8",
    )
    shutil.copy2(model_path, output_dir / "model.agent")

    compiled_sidecar = model_path.with_suffix(model_path.suffix + ".compiled.json")
    if compiled_sidecar.exists():
        shutil.copy2(compiled_sidecar, output_dir / "model.agent.compiled.json")

    lib_dir = output_dir / "lib"
    lib_dir.mkdir(exist_ok=True)
    _copy_local_runtime_package(lib_dir / "agent")
    _copy_dependency_closure(lib_dir)

    agent_sh = output_dir / "agent.sh"
    agent_sh.write_text(
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        "SCRIPT_DIR=$(CDPATH= cd -- \"$(dirname \"$0\")\" && pwd)\n"
        "PYTHON_BIN=${PYTHON:-python3}\n"
        "PYTHONPATH=\"$SCRIPT_DIR/lib${PYTHONPATH:+:$PYTHONPATH}\" exec \"$PYTHON_BIN\" "
        "\"$SCRIPT_DIR/main.py\" \"$@\"\n",
        encoding="utf-8",
    )
    agent_sh.chmod(agent_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (output_dir / "PORTABLE_TODO.md").write_text(
        "# Portable Bundle Status\n\n"
        "This bundle is experimental.\n\n"
        "Known limitation:\n"
        "- Repo-relative assets, scripts, and data referenced from `.agent` skills are not copied into the bundle.\n"
        "- Hard-coded paths such as `models/data_visualizer/scripts/visualize_data.py` will still point back to the source repository layout.\n"
        "- Real portability needs DSL-level asset declarations and/or path rewriting during bundle generation.\n",
        encoding="utf-8",
    )
    return output_dir


def _copy_local_runtime_package(target_dir: Path) -> None:
    source_dir = Path(__file__).resolve().parent
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in RUNTIME_FILES:
        shutil.copy2(source_dir / filename, target_dir / filename)


def _copy_import_target(module_name: str, destination_root: Path) -> None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise RuntimeError(f"Could not locate installed module for portable bundle: {module_name}")

    if spec.submodule_search_locations:
        source_path = Path(next(iter(spec.submodule_search_locations)))
        target_path = destination_root / source_path.name
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)
        return

    if spec.origin is None:
        raise RuntimeError(f"Could not resolve module origin for portable bundle: {module_name}")

    source_file = Path(spec.origin)
    shutil.copy2(source_file, destination_root / source_file.name)


def _copy_dependency_closure(destination_root: Path) -> None:
    import_names: set[str] = set()
    pending_distributions = list(_distribution_names_for_imports(PORTABLE_ROOT_IMPORTS))
    seen_distributions: set[str] = set()

    while pending_distributions:
        distribution_name = pending_distributions.pop()
        normalized = distribution_name.lower().replace("_", "-")
        if normalized in seen_distributions:
            continue
        seen_distributions.add(normalized)

        try:
            distribution = importlib_metadata.distribution(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            continue

        import_names.update(_top_level_imports_for_distribution(distribution))

        for requirement in distribution.requires or ():
            requirement_name = _parse_requirement_name(requirement)
            if requirement_name:
                pending_distributions.append(requirement_name)

    for import_name in sorted(import_names):
        if import_name == "agent":
            continue
        if not _IMPORT_NAME_RE.match(import_name):
            continue
        _copy_import_target(import_name, destination_root)


def _distribution_names_for_imports(import_names: tuple[str, ...]) -> set[str]:
    mapping = importlib_metadata.packages_distributions()
    distribution_names: set[str] = set()
    for import_name in import_names:
        distribution_names.update(mapping.get(import_name, [import_name]))
    return distribution_names


def _top_level_imports_for_distribution(distribution: importlib_metadata.Distribution) -> set[str]:
    top_level = distribution.read_text("top_level.txt")
    if top_level:
        return {
            line.strip()
            for line in top_level.splitlines()
            if line.strip() and _IMPORT_NAME_RE.match(line.strip())
        }

    inferred: set[str] = set()
    for file in distribution.files or ():
        parts = Path(file).parts
        if not parts:
            continue
        head = parts[0]
        if head.endswith(".dist-info") or head.endswith(".data") or head == "__pycache__":
            continue
        if head.endswith(".py"):
            candidate = head[:-3]
        else:
            candidate = head
        if _IMPORT_NAME_RE.match(candidate):
            inferred.add(candidate)
    return inferred


def _parse_requirement_name(requirement: str) -> str | None:
    if "extra ==" in requirement:
        return None
    requirement = requirement.split(";", 1)[0].strip()
    match = _REQ_NAME_RE.match(requirement)
    if match is None:
        return None
    return match.group(0)
