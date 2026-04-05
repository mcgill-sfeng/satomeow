import pytest

from agent.parser import load_metamodel, parse_model


def test_load_metamodel():
    meta_model = load_metamodel()
    assert meta_model is not None


def test_parse_minimal_model():
    system = parse_model("models/example_minimal.agent")
    assert system is not None
    assert system.planner is not None
    assert len(system.executors) >= 1


def test_parse_full_model():
    system = parse_model("models/example_full.agent")
    assert system.planner.llm == "gpt-5.4-nano"
    assert len(system.executors) == 2
    assert system.executors[0].task.name == "WebResearch"


def test_planner_auto_configured_from_defaults():
    system = parse_model("models/example_minimal.agent")
    assert system.planner.persona == "planner"
    assert system.planner.llm == "gpt-5.4-nano"
    assert system.planner.reasoningStrategy is None


def test_executor_inherits_global_defaults():
    system = parse_model("models/example_minimal.agent")
    executor = system.executors[0]
    assert executor.llm == "gpt-5.4-nano"
    assert executor.reasoningStrategy is None


def test_executor_can_override_defaults():
    system = parse_model("models/example_full.agent")
    editor = system.executors[1]  # TextEditor overrides llm and reasoning
    assert editor.llm == "gpt-5.4-nano"
    assert editor.reasoningStrategy == "medium"


def test_invalid_reasoning_value_fails_parse(tmp_path):
    bad = tmp_path / "bad.agent"
    bad.write_text(
        'llm: "gpt-5.4-nano"\nreasoning: "react"\nA : "a" { input: "x" behavior: "y" }\n',
        encoding="utf-8",
    )
    from textx.exceptions import TextXSemanticError
    with pytest.raises(TextXSemanticError, match="must be one of"):
        parse_model(bad)


def test_reasoning_is_optional(tmp_path):
    model = tmp_path / "ok.agent"
    model.write_text(
        'llm: "gpt-5.4-nano"\nA : "a" { input: "x" behavior: "y" }\n',
        encoding="utf-8",
    )
    system = parse_model(model)
    assert system.planner.reasoningStrategy is None
    assert system.executors[0].reasoningStrategy is None
