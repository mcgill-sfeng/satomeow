"""Tests for DSPy compile-time integration (agent/dspy_compile.py)."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.dspy_compile import (
    get_compiled_examples,
    load_compiled_sidecar,
    compile_system_spec,
)
from agent.ir import build_prompt_ir
from agent.parser import parse_model
from agent.runtime import AgentSystemRuntime, build_examples_prompt


MODEL_PATH = "models/example_full.agent"


# ---------------------------------------------------------------------------
# Sidecar path convention
# ---------------------------------------------------------------------------


def test_sidecar_path_is_next_to_model(tmp_path):
    from agent.dspy_compile import _sidecar_path
    p = tmp_path / "my_agent.agent"
    assert _sidecar_path(p) == tmp_path / "my_agent.agent.compiled.json"


# ---------------------------------------------------------------------------
# load_compiled_sidecar
# ---------------------------------------------------------------------------


def test_load_compiled_sidecar_returns_none_when_absent(tmp_path):
    result = load_compiled_sidecar(tmp_path / "nonexistent.agent")
    assert result is None


def test_load_compiled_sidecar_returns_none_on_bad_json(tmp_path):
    p = tmp_path / "bad.agent"
    p.touch()
    sidecar = tmp_path / "bad.agent.compiled.json"
    sidecar.write_text("not json", encoding="utf-8")
    assert load_compiled_sidecar(p) is None


def test_load_compiled_sidecar_loads_valid_sidecar(tmp_path):
    p = tmp_path / "my.agent"
    p.touch()
    sidecar = tmp_path / "my.agent.compiled.json"
    data = {"executors": {"Foo": {"compiled_examples": [{"input": "x", "output": "y"}]}}}
    sidecar.write_text(json.dumps(data), encoding="utf-8")
    result = load_compiled_sidecar(p)
    assert result == data


# ---------------------------------------------------------------------------
# get_compiled_examples
# ---------------------------------------------------------------------------


def test_get_compiled_examples_none_on_absent_sidecar():
    assert get_compiled_examples(None, "Foo") is None


def test_get_compiled_examples_none_on_missing_executor():
    sidecar = {"executors": {"Bar": {"compiled_examples": []}}}
    assert get_compiled_examples(sidecar, "Foo") is None


def test_get_compiled_examples_returns_list():
    examples = [{"input": "a", "output": "b"}]
    sidecar = {"executors": {"Foo": {"compiled_examples": examples}}}
    assert get_compiled_examples(sidecar, "Foo") == examples


# ---------------------------------------------------------------------------
# compile_system_spec — no dspy needed (mocked)
# ---------------------------------------------------------------------------


def test_compile_system_spec_writes_sidecar(tmp_path):
    ir = build_prompt_ir(parse_model(MODEL_PATH))

    mock_dspy = MagicMock()
    # Make BootstrapFewShot.compile return a module with named_predictors
    mock_predictor = MagicMock()
    mock_predictor.demos = [
        {"task_input": "q", "task_output": "a"},
    ]
    mock_compiled = MagicMock()
    mock_compiled.named_predictors.return_value = [("pred", mock_predictor)]
    mock_dspy.BootstrapFewShot.return_value.compile.return_value = mock_compiled
    mock_dspy.Predict.return_value = MagicMock()
    mock_dspy.Example.side_effect = lambda **kw: SimpleNamespace(**kw, with_inputs=lambda *a: SimpleNamespace(**kw))

    # Patch the Signature class creation
    mock_sig_instance = MagicMock()
    mock_dspy.Signature = MagicMock(return_value=mock_sig_instance)

    # Write a dummy source .agent file
    source = tmp_path / "test.agent"
    source.touch()

    with patch("agent.dspy_compile._require_dspy", return_value=mock_dspy):
        sidecar_path = compile_system_spec(ir, source)

    assert sidecar_path.exists()
    data = json.loads(sidecar_path.read_text())
    assert "executors" in data


def test_compile_system_spec_skips_executors_without_examples(tmp_path):
    # Parse a model whose executors have no examples
    from agent.parser import parse_model_text
    text = (
        'llm: "gpt-5.4-nano"\nreasoning: "medium"\n'
        'A : "a" { input: "x" behavior: "y" }\n'
    )
    ir = build_prompt_ir(parse_model_text(text))
    source = tmp_path / "no_examples.agent"
    source.touch()

    mock_dspy = MagicMock()
    with patch("agent.dspy_compile._require_dspy", return_value=mock_dspy):
        sidecar_path = compile_system_spec(ir, source)

    data = json.loads(sidecar_path.read_text())
    assert data["executors"] == {}


def test_compile_system_spec_fallback_on_error(tmp_path):
    ir = build_prompt_ir(parse_model(MODEL_PATH))
    source = tmp_path / "test.agent"
    source.touch()

    mock_dspy = MagicMock()
    mock_dspy.BootstrapFewShot.return_value.compile.side_effect = RuntimeError("API error")
    mock_dspy.Predict.return_value = MagicMock()
    mock_dspy.Example.side_effect = lambda **kw: SimpleNamespace(**kw, with_inputs=lambda *a: SimpleNamespace(**kw))

    with patch("agent.dspy_compile._require_dspy", return_value=mock_dspy):
        sidecar_path = compile_system_spec(ir, source)

    data = json.loads(sidecar_path.read_text())
    # Fallback: original examples preserved, error noted
    for name, entry in data["executors"].items():
        assert "compiled_examples" in entry
        assert "bootstrap_error" in entry


# ---------------------------------------------------------------------------
# build_examples_prompt with compiled_examples
# ---------------------------------------------------------------------------


def test_build_examples_prompt_uses_compiled_when_provided():
    ir = build_prompt_ir(parse_model(MODEL_PATH))
    executor = next(e for e in ir["executors"] if e["name"] == "WebResearch")

    compiled = [{"input": "compiled q", "output": "compiled a"}]
    prompt = build_examples_prompt(executor, use_dspy=True, compiled_examples=compiled)

    assert "DSPy-compiled" in prompt
    assert "compiled q" in prompt
    assert "compiled a" in prompt


def test_build_examples_prompt_ignores_compiled_when_none():
    ir = build_prompt_ir(parse_model(MODEL_PATH))
    executor = next(e for e in ir["executors"] if e["name"] == "WebResearch")

    prompt = build_examples_prompt(executor, use_dspy=False, compiled_examples=None)
    assert "DSPy-compiled" not in prompt


# ---------------------------------------------------------------------------
# Runtime loads sidecar automatically when use_dspy=True and sidecar exists
# ---------------------------------------------------------------------------


def test_runtime_loads_sidecar_on_use_dspy(tmp_path, monkeypatch):
    model_path = tmp_path / "test.agent"
    model_path.touch()
    sidecar = tmp_path / "test.agent.compiled.json"
    compiled_data = {
        "executors": {
            "WebResearch": {
                "compiled_examples": [{"input": "ci", "output": "co"}]
            }
        }
    }
    sidecar.write_text(json.dumps(compiled_data), encoding="utf-8")

    ir = build_prompt_ir(parse_model(MODEL_PATH))
    runtime = AgentSystemRuntime(ir, source_model_path=str(model_path), use_dspy=True)

    assert runtime._compiled_sidecar == compiled_data


def test_runtime_sidecar_is_none_without_use_dspy(tmp_path):
    model_path = tmp_path / "test.agent"
    model_path.touch()
    sidecar = tmp_path / "test.agent.compiled.json"
    sidecar.write_text('{"executors": {}}', encoding="utf-8")

    ir = build_prompt_ir(parse_model(MODEL_PATH))
    runtime = AgentSystemRuntime(ir, source_model_path=str(model_path), use_dspy=False)
    assert runtime._compiled_sidecar is None


# ---------------------------------------------------------------------------
# CLI compile subcommand
# ---------------------------------------------------------------------------


def test_compile_cli_registered():
    from agent.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["compile", MODEL_PATH])
    assert args.command == "compile"
    assert args.model is None


def test_compile_cli_model_flag():
    from agent.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["compile", MODEL_PATH, "--model", "openai/gpt-5.4-nano"])
    assert args.model == "openai/gpt-5.4-nano"


def test_compile_cli_no_examples_prints_message(tmp_path, capsys):
    from agent import cli
    from agent.parser import parse_model_text

    # Write a real .agent file with no examples so the CLI can parse it from disk.
    text = 'llm: "gpt-5.4-nano"\nreasoning: "medium"\nA : "a" { input: "x" behavior: "y" }\n'
    model_file = tmp_path / "no_examples.agent"
    model_file.write_text(text, encoding="utf-8")

    cli.main(["compile", str(model_file)])
    out = capsys.readouterr().out
    assert "nothing to compile" in out.lower()
