from pathlib import Path

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


def test_cli_run_uses_example_driven_model_without_provider(tmp_path, monkeypatch, capsys):
    model_path = _write_safe_model(tmp_path / "runtime.agent")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    cli.main(["run", str(model_path), "--use-examples", "Say hello"])
    captured = capsys.readouterr()
    assert "hello from runtime" in captured.out


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
    output: "text"

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
