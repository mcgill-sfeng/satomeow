import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import cli
from agent.codegen import (
    render_agent_module,
    render_agent_module_from_system,
    render_portable_agent_module_from_system,
)
from agent.parser import parse_model


def test_render_agent_module_contains_runtime_entrypoints():
    rendered = render_agent_module("models/example_full.agent")
    assert "def build_runtime" in rendered
    assert "def run_agent" in rendered
    assert "planner.llm =" in rendered
    assert "def _build_system" in rendered


def _metamodel_import_line(rendered: str) -> str:
    for line in rendered.splitlines():
        if line.startswith("from agent.metamodel import"):
            return line
    raise AssertionError("rendered module has no 'from agent.metamodel import' line")


def test_metamodel_defines_codegen_classes():
    """The names emitted by codegen must actually exist in agent.metamodel.

    Runs fully offline (agent.metamodel has no third-party imports). This catches
    an *imported-but-undefined* class, which the substring/compile checks below
    cannot — compile() does not execute imports.
    """
    import agent.metamodel as mm

    assert all(hasattr(mm, name) for name in ("Rule", "Skill", "Example"))


@pytest.mark.parametrize(
    "model_path",
    ["models/example_full.agent", "models/agent_composer/agent_composer.agent"],
)
def test_generated_module_compiles_and_imports_metamodel_classes(model_path):
    """Render + compile() both templates and assert the metamodel import line.

    Offline net for a *used-but-unimported* name: a class referenced in generated
    code that is missing from the 'from agent.metamodel import ...' line.
    """
    system = parse_model(model_path)

    rendered = render_agent_module_from_system(system)
    compile(rendered, "<generated>", "exec")
    import_line = _metamodel_import_line(rendered)
    for name in ("Rule", "Skill", "Example"):
        assert name in import_line

    # example_full has skills + example commands, so SkillArgument/ExampleCommand
    # must be imported too (offline the exec round-trip that would catch their
    # absence is agents-gated).
    if "example_full" in model_path:
        for name in ("SkillArgument", "ExampleCommand", "ExampleCommandArgument"):
            assert name in import_line


def test_portable_module_compiles_and_imports_metamodel_classes():
    """The portable template is rendered by a distinct function — render + compile
    it too so a half-converted portable template can't escape the offline nets.
    """
    rendered = render_portable_agent_module_from_system(parse_model("models/example_full.agent"))
    compile(rendered, "<portable>", "exec")
    import_line = _metamodel_import_line(rendered)
    for name in ("Rule", "Skill", "Example", "SkillArgument", "ExampleCommand"):
        assert name in import_line
    assert "SimpleNamespace" not in rendered
    assert "_build_skill" not in rendered
    assert "_build_executor" not in rendered


@pytest.mark.parametrize(
    "model_path",
    [
        "models/example_full.agent",
        "models/agent_composer/agent_composer.agent",
        "models/example_minimal.agent",
    ],
)
def test_generated_module_round_trips_system(model_path):
    """Exec the rendered module and round-trip key fields against the parsed model.

    Gated behind the agents SDK because the generated module imports agent.runtime
    (which pulls `agents` transitively). A skip here is NOT a pass.
    """
    pytest.importorskip("agents")

    system = parse_model(model_path)
    rendered = render_agent_module_from_system(system)
    namespace: dict = {}
    exec(compile(rendered, "<generated>", "exec"), namespace)  # noqa: S102
    rebuilt = namespace["_build_system"]()

    assert rebuilt.executors[0].task.name == system.executors[0].task.name
    assert rebuilt.executors[0].task.outputSpec.format == system.executors[0].task.outputSpec.format

    if "example_full" in model_path:
        rebuilt_skills = {s.name for s in rebuilt.executors[0].task.skills}
        assert rebuilt_skills == {"webSearch", "docReader"}
        rebuilt_rules = {r.name for r in rebuilt.executors[0].rules}
        assert rebuilt_rules == {"verify_sources", "no_hallucination"}
        # Compare examples by value — parse-path and codegen-path examples are
        # different classes, so never compare by ==/is/type.
        rebuilt_example = rebuilt.executors[0].task.examples[0]
        source_example = system.executors[0].task.examples[0]
        assert rebuilt_example.input == source_example.input
        assert rebuilt_example.output == source_example.output
        assert [c.toolName for c in rebuilt_example.commands] == [
            c.toolName for c in source_example.commands
        ]

    if "agent_composer" in model_path:
        # Validator's json output exercises the non-empty OutputField path.
        validator = rebuilt.executors[1]
        assert validator.task.outputSpec.format == "json"
        assert {f.name for f in validator.task.outputSpec.fields} == {
            "valid",
            "errors",
            "suggestions",
        }
        # chatAgent reconstruction branch is unchanged by this refactor, but exec-ing
        # the module drives it; a light assertion keeps failures correctly triaged.
        assert rebuilt.chatAgent.name == "Intake"

    if "example_minimal" in model_path:
        # WITHOUT-skills/rules/output/chat model: exercises empty loops and the
        # default OutputSpec reconstruction. Benign at runtime (guarded), so do
        # not assert a crash.
        assert rebuilt.executors[0].task.outputSpec.format == "string"
        assert rebuilt.executors[0].task.outputSpec.fields == []


def test_cli_generate_writes_module(tmp_path, capsys):
    output_path = tmp_path / "generated_agent.py"
    cli.main(["generate", "models/example_full.agent", "--output", str(output_path)])
    assert output_path.exists()
    assert "run_agent" in output_path.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert str(output_path) in captured.out


def test_cli_portable_writes_bundle(tmp_path, capsys):
    output_dir = tmp_path / "portable-agent"
    cli.main(["portable", "models/example_full.agent", "--output", str(output_dir)])
    assert (output_dir / "main.py").exists()
    assert (output_dir / "agent.sh").exists()
    assert (output_dir / "model.agent").exists()
    assert (output_dir / "PORTABLE_TODO.md").exists()
    assert (output_dir / "lib" / "agent" / "runtime.py").exists()
    assert (output_dir / "lib" / "agent" / "metamodel.py").exists()
    assert (output_dir / "lib" / "agent" / "dspy_compile.py").exists()
    captured = capsys.readouterr()
    assert "portable is experimental/TODO" in captured.out
    assert str(output_dir) in captured.out


def test_cli_portable_bundle_imports_cross_platform(tmp_path):
    """Run the bundled main.py -h in a subprocess so imports actually execute.

    py_compile would not detect a missing vendored metamodel.py (it does not run
    imports). Requires the agents SDK because main.py imports agent.runtime.
    """
    pytest.importorskip("agents")
    output_dir = tmp_path / "portable-agent"
    cli.main(["portable", "models/example_full.agent", "--output", str(output_dir)])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(output_dir / "lib")
    completed = subprocess.run(
        [sys.executable, "-S", str(output_dir / "main.py"), "-h"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Run the portable agent bundle." in completed.stdout


@pytest.mark.skipif(os.name == "nt", reason="portable agent.sh execution test is POSIX-only")
def test_cli_portable_help_runs_without_site_packages(tmp_path):
    output_dir = tmp_path / "portable-agent"
    cli.main(["portable", "models/example_full.agent", "--output", str(output_dir)])

    wrapper = tmp_path / "python-no-site.sh"
    wrapper.write_text(
        "#!/usr/bin/env sh\n" f'exec {sys.executable} -S "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)

    env = {"PATH": os.environ.get("PATH", ""), "PYTHON": str(wrapper)}
    completed = subprocess.run(
        [str(output_dir / "agent.sh"), "-h"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert "Run the portable agent bundle." in completed.stdout


def test_cli_run_uses_sdk_runtime(tmp_path, monkeypatch, capsys):
    model_path = _write_safe_model(tmp_path / "runtime.agent")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(
        "agent.runtime.Runner.run_sync",
        lambda *args, **kwargs: SimpleNamespace(
            final_output="hello from runtime",
            raw_responses=[{"id": "resp_1"}],
            new_items=[],
        ),
    )
    cli.main(["run", str(model_path), "Say hello"])
    captured = capsys.readouterr()
    assert "hello from runtime" in captured.out


def test_cli_verbose_prints_composed_prompt(tmp_path, monkeypatch, capsys):
    model_path = _write_safe_model(tmp_path / "runtime.agent")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(
        "agent.runtime.Runner.run_sync",
        lambda *args, **kwargs: SimpleNamespace(
            final_output="hello from runtime",
            raw_responses=[{"id": "resp_1"}],
            new_items=[],
        ),
    )
    cli.main(["run", str(model_path), "--verbose", "Say hello"])
    captured = capsys.readouterr()
    assert "[system_prompt]" in captured.out
    assert "[user_input]" in captured.out
    assert "Say hello" in captured.out


def test_cli_prompt_dump_prints_executor_payload(capsys):
    cli.main(["prompt", "models/data_visualizer/data_visualizer.agent", "Say hello"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "executor"
    assert payload["agent_name"] == "DataVisualizer"
    assert payload["input"] == "Say hello"
    assert "system_prompt" in payload


def test_cli_prompt_dump_prints_planner_payload(capsys):
    cli.main(["prompt", "models/example_full.agent", "--planner", "Compare APIs"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "planner"
    assert payload["agent_name"] == "Planner"
    assert payload["input"] == "Compare APIs"


def test_cli_prompt_dump_requires_selector_for_multi_executor():
    try:
        cli.main(["prompt", "models/example_full.agent", "Compare APIs"])
    except SystemExit as exc:
        assert "require --executor NAME or --planner" in str(exc)
    else:
        raise AssertionError("expected SystemExit for ambiguous multi-executor prompt dump")


def _write_safe_model(path: Path) -> Path:
    path.write_text(
        """llm: "gpt-5.4-nano"
reasoning: "medium"

Hello : "shell agent" {
    input: "Greeting request"
    behavior: "print a greeting"
    output: string

    example {
        input: "Say hello"
        commands: []
        output: "hello from runtime"
    }
}
""",
        encoding="utf-8",
    )
    return path
