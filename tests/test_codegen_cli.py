from pathlib import Path
from types import SimpleNamespace

from agent import cli
from agent.codegen import render_agent_module


def test_render_agent_module_contains_runtime_entrypoints():
    rendered = render_agent_module("models/example_full.agent")
    assert "def build_runtime" in rendered
    assert "def run_agent" in rendered
    assert "SYSTEM_SPEC_JSON" in rendered


def test_cli_generate_writes_module(tmp_path, capsys):
    output_path = tmp_path / "generated_agent.py"
    cli.main(["generate", "models/example_full.agent", "--output", str(output_path)])
    assert output_path.exists()
    assert "run_agent" in output_path.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert str(output_path) in captured.out


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


def test_cli_legacy_print_ir(capsys):
    cli.main(["models/example_full.agent", "--print-ir"])
    captured = capsys.readouterr()
    assert '"planner"' in captured.out


def _write_safe_model(path: Path) -> Path:
    path.write_text(
        """llm: "gpt-5"
reasoning: "react"

Hello : "shell agent" {
    input: "Greeting request"
    behavior: "print a greeting"
    output: string

    example {
        input: "Say hello"
        commands: ["printf 'hello from runtime'"]
        output: "hello from runtime"
    }
}
""",
        encoding="utf-8",
    )
    return path
