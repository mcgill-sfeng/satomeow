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
    assert system.planner.llm == "gpt-5"
    assert len(system.executors) == 2
    assert system.executors[0].task.name == "WebResearch"


def test_planner_auto_configured_from_defaults():
    system = parse_model("models/example_minimal.agent")
    assert system.planner.persona == "planner"
    assert system.planner.llm == "gpt-5"
    assert system.planner.reasoningStrategy == "chain-of-thought"


def test_executor_inherits_global_defaults():
    system = parse_model("models/example_minimal.agent")
    executor = system.executors[0]
    assert executor.llm == "gpt-5"
    assert executor.reasoningStrategy == "chain-of-thought"


def test_executor_can_override_defaults():
    system = parse_model("models/example_full.agent")
    editor = system.executors[1]  # TextEditor overrides llm and reasoning
    assert editor.llm == "claude-sonnet-4.5"
    assert editor.reasoningStrategy == "react"
