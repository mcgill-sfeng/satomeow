"""Tests for the 'agent cli chat' subcommand and ChatModeAgent pipeline."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import cli
from agent.ir import build_prompt_ir
from agent.parser import parse_model
from agent.runtime import AgentSystemRuntime, ShellToolExecutor

CHAT_MODEL = "models/example_chat.agent"
NO_CHAT_MODEL = "models/example_full.agent"


# ---------------------------------------------------------------------------
# Parsing and IR
# ---------------------------------------------------------------------------


def test_chat_model_parses():
    system = parse_model(CHAT_MODEL)
    assert system.chat_agent is not None
    assert system.chat_agent.name == "Intake"
    assert system.chat_agent.executor_ref == "Greeter"
    assert len(system.chat_agent.questions) == 2


def test_chat_model_ir_has_chat_agent():
    ir = build_prompt_ir(parse_model(CHAT_MODEL))
    ca = ir["chat_agent"]
    assert ca is not None
    assert ca["name"] == "Intake"
    assert ca["executor_ref"] == "Greeter"
    assert ca["questions"] == ["What is your name?", "Do you prefer a formal or casual greeting?"]


def test_no_chat_model_ir_has_none():
    ir = build_prompt_ir(parse_model(NO_CHAT_MODEL))
    assert ir.get("chat_agent") is None


def test_chat_model_invalid_executor_ref(tmp_path):
    bad = tmp_path / "bad.agent"
    bad.write_text(
        'llm: "gpt-5.4-nano"\nreasoning: "react"\n'
        'A : "a" { input: "x" behavior: "y" }\n'
        'chat C : "c" { goal: "g" questions: ["q?"] executor: NoSuchExecutor }\n',
        encoding="utf-8",
    )
    from textx.exceptions import TextXSemanticError
    with pytest.raises(TextXSemanticError, match="unknown executor"):
        parse_model(bad)


def test_chat_agent_requires_at_least_one_question(tmp_path):
    # The grammar (+=) rejects an empty list at parse time.
    bad = tmp_path / "bad.agent"
    bad.write_text(
        'llm: "gpt-5.4-nano"\nreasoning: "react"\n'
        'A : "a" { input: "x" behavior: "y" }\n'
        'chat C : "c" { goal: "g" questions: [] }\n',
        encoding="utf-8",
    )
    from textx.exceptions import TextXSyntaxError
    with pytest.raises(TextXSyntaxError):
        parse_model(bad)


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def test_build_chat_input_bundles_qa():
    result = AgentSystemRuntime._build_chat_input(
        goal="Greet user",
        questions=["What is your name?", "Formal or casual?"],
        answers=["Alice", "casual"],
    )
    assert "Task: Greet user" in result
    assert "Q: What is your name?" in result
    assert "A: Alice" in result
    assert "Q: Formal or casual?" in result
    assert "A: casual" in result


# ---------------------------------------------------------------------------
# CLI chat — argument parsing
# ---------------------------------------------------------------------------


def test_chat_parser_registered():
    parser = cli._build_parser()
    args = parser.parse_args(["chat", CHAT_MODEL])
    assert args.command == "chat"
    assert args.verbose is False


def test_chat_parser_verbose_flag():
    parser = cli._build_parser()
    args = parser.parse_args(["chat", CHAT_MODEL, "--verbose"])
    assert args.verbose is True


# ---------------------------------------------------------------------------
# CLI chat — model without ChatAgent fails gracefully
# ---------------------------------------------------------------------------


def test_chat_requires_chat_agent_block(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    cli.main(["chat", NO_CHAT_MODEL])
    out = capsys.readouterr().out
    assert "no 'chat' agent" in out.lower() or "error" in out.lower()


# ---------------------------------------------------------------------------
# CLI chat — Q&A loop and exit paths
# ---------------------------------------------------------------------------


def test_chat_quit_during_questions(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    inputs = iter(["quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    cli.main(["chat", CHAT_MODEL])
    # Should exit cleanly without calling LLM


def test_chat_ctrl_d_during_questions(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(EOFError()))

    cli.main(["chat", CHAT_MODEL])
    out = capsys.readouterr().out
    assert "error" not in out.lower()


def test_chat_cancel_at_confirmation(monkeypatch, capsys):
    """Answering 'n' at the confirmation prompt cancels without running the executor."""
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    call_sequence = iter(["Alice", "casual", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(call_sequence))

    def fake_run_sync(agent, user_input, *, run_config, hooks=None, max_turns=None):
        if agent.name == "ConfirmationGenerator":
            return SimpleNamespace(final_output="I'll greet Alice casually. Shall I proceed?", raw_responses=[], new_items=[])
        raise AssertionError("Executor should not be called after cancel")

    monkeypatch.setattr("agent.runtime.Runner.run_sync", fake_run_sync)

    cli.main(["chat", CHAT_MODEL])
    out = capsys.readouterr().out
    assert "cancelled" in out.lower()


def test_chat_full_flow_executes_executor(monkeypatch, capsys):
    """Full happy path: Q&A → confirmation → Y → executor runs → output printed."""
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    call_sequence = iter(["Alice", "casual", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(call_sequence))

    executor_called_with = {}

    def fake_run_sync(agent, user_input, *, run_config, hooks=None, max_turns=None):
        if agent.name == "ConfirmationGenerator":
            return SimpleNamespace(final_output="I'll greet Alice casually. Shall I proceed?", raw_responses=[], new_items=[])
        # Executor call
        executor_called_with["agent"] = agent.name
        executor_called_with["input"] = user_input
        return SimpleNamespace(final_output="Hey Alice!", raw_responses=["raw"], new_items=[])

    monkeypatch.setattr("agent.runtime.Runner.run_sync", fake_run_sync)

    cli.main(["chat", CHAT_MODEL])
    out = capsys.readouterr().out

    assert executor_called_with.get("agent") == "Greeter"
    assert "Alice" in executor_called_with.get("input", "")
    assert "Hey Alice!" in out


def test_chat_bundled_input_contains_answers(monkeypatch, capsys):
    """The input passed to the executor contains all Q&A pairs."""
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    call_sequence = iter(["Bob", "formal", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(call_sequence))

    captured_input = {}

    def fake_run_sync(agent, user_input, *, run_config, hooks=None, max_turns=None):
        if agent.name == "ConfirmationGenerator":
            return SimpleNamespace(final_output="I'll greet Bob formally. Proceed?", raw_responses=[], new_items=[])
        captured_input["value"] = user_input
        return SimpleNamespace(final_output="Good day, Bob.", raw_responses=[], new_items=[])

    monkeypatch.setattr("agent.runtime.Runner.run_sync", fake_run_sync)
    cli.main(["chat", CHAT_MODEL])

    bundled = captured_input.get("value", "")
    assert "Bob" in bundled
    assert "formal" in bundled


# ---------------------------------------------------------------------------
# AgentSystemRuntime.run() — executor_name override (regression)
# ---------------------------------------------------------------------------


def test_run_with_executor_name_bypasses_planner(monkeypatch):
    system = parse_model(NO_CHAT_MODEL)
    prompt_ir = build_prompt_ir(system)
    captured = {}

    def fake_run_sync(agent, user_input, *, run_config, hooks=None, max_turns=None):
        captured["agent_name"] = agent.name
        return SimpleNamespace(final_output="edited text", raw_responses=[], new_items=[])

    monkeypatch.setattr("agent.runtime.Runner.run_sync", fake_run_sync)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    runtime = AgentSystemRuntime(prompt_ir)
    result = runtime.run("Fix grammar", executor_name="TextEditor")

    assert captured["agent_name"] == "TextEditor"
    assert result.executor_name == "TextEditor"
    assert "--executor TextEditor" in result.planner_reason


def test_run_with_unknown_executor_name_raises(monkeypatch):
    system = parse_model(NO_CHAT_MODEL)
    prompt_ir = build_prompt_ir(system)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(
        "agent.runtime.Runner.run_sync",
        lambda *a, **kw: SimpleNamespace(final_output="x", raw_responses=[], new_items=[]),
    )
    runtime = AgentSystemRuntime(prompt_ir)
    with pytest.raises(KeyError, match="NoSuchExecutor"):
        runtime.run("hello", executor_name="NoSuchExecutor")
